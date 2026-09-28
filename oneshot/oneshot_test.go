package oneshot

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"testing"

	"github.com/v0lka/sp4rk/llm"
)

// fakeCaller records every request and replays queued responses.
type fakeCaller struct {
	mu        sync.Mutex
	responses []*llm.ChatResponse
	err       error // transport error returned on every call
	requests  []llm.ChatRequest
	onCall    func(n int) // invoked with the 1-based call count before responding
}

func (f *fakeCaller) Call(ctx context.Context, req llm.ChatRequest) (*llm.ChatResponse, error) {
	f.mu.Lock()
	f.requests = append(f.requests, req)
	n := len(f.requests)
	f.mu.Unlock()
	if f.onCall != nil {
		f.onCall(n)
	}
	if f.err != nil {
		return nil, f.err
	}
	if n <= len(f.responses) {
		return f.responses[n-1], nil
	}
	return &llm.ChatResponse{}, nil
}

func (f *fakeCaller) callCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.requests)
}

func (f *fakeCaller) request(n int) llm.ChatRequest {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.requests[n-1]
}

// baseRequest is the canonical [system, user] one-shot request.
func baseRequest() llm.ChatRequest {
	return llm.ChatRequest{
		Model: "test-model",
		Messages: []llm.Message{
			{Role: "system", Content: "system prompt"},
			{Role: "user", Content: "user task"},
		},
	}
}

// parseEchoOK succeeds only when the response echoes "good"; otherwise it
// fails with a format diagnostic.
func parseEchoOK(resp *llm.ChatResponse) (string, error) {
	if resp != nil && strings.TrimSpace(resp.Message.Content) == "good" {
		return "parsed:" + resp.Message.Content, nil
	}
	return "", errors.New("missing the required marker")
}

// --- Success paths ---

func TestDo_SuccessFirstAttempt(t *testing.T) {
	caller := &fakeCaller{responses: []*llm.ChatResponse{
		respWith("good", "", ""),
	}}
	got, err := Do(context.Background(), caller, baseRequest(), parseEchoOK, Options[string]{Kind: "probe"})
	if err != nil {
		t.Fatalf("Do() error = %v", err)
	}
	if got != "parsed:good" {
		t.Errorf("Do() = %q, want %q", got, "parsed:good")
	}
	if caller.callCount() != 1 {
		t.Errorf("call count = %d, want 1 (no retries on success)", caller.callCount())
	}
}

func TestDo_RetriesOnceWithNudge(t *testing.T) {
	caller := &fakeCaller{responses: []*llm.ChatResponse{
		respWith("bad output", "", ""),
		respWith("good", "", ""),
	}}
	got, err := Do(context.Background(), caller, baseRequest(), parseEchoOK, Options[string]{
		Kind:      "probe",
		RetryHint: "Answer strictly as one line: good",
	})
	if err != nil {
		t.Fatalf("Do() error = %v", err)
	}
	if got != "parsed:good" {
		t.Errorf("Do() = %q, want %q", got, "parsed:good")
	}
	if caller.callCount() != 2 {
		t.Fatalf("call count = %d, want 2", caller.callCount())
	}

	// The retried request = original messages + [assistant echo, user nudge].
	retryReq := caller.request(2)
	gotMsgs := retryReq.Messages
	if len(gotMsgs) != 4 {
		t.Fatalf("retry request has %d messages, want 4", len(gotMsgs))
	}
	if gotMsgs[0].Role != "system" || gotMsgs[1].Role != "user" {
		t.Errorf("retry request must keep the original prefix, got %q/%q", gotMsgs[0].Role, gotMsgs[1].Role)
	}
	if gotMsgs[2].Role != "assistant" || gotMsgs[2].Content != "bad output" {
		t.Errorf("retry message 2 = (%q, %q), want assistant echo of the failed output", gotMsgs[2].Role, gotMsgs[2].Content)
	}
	if gotMsgs[3].Role != "user" {
		t.Errorf("retry message 3 role = %q, want user", gotMsgs[3].Role)
	}
	if !strings.HasPrefix(gotMsgs[3].Content, "[System]") {
		t.Errorf("nudge must start with %q, got %q", "[System]", gotMsgs[3].Content)
	}
	if !strings.Contains(gotMsgs[3].Content, "Answer strictly as one line: good") {
		t.Errorf("nudge must carry the RetryHint, got %q", gotMsgs[3].Content)
	}
}

func TestDo_RetriesTwiceWithNudges(t *testing.T) {
	caller := &fakeCaller{responses: []*llm.ChatResponse{
		respWith("bad one", "", ""),
		respWith("bad two", "", ""),
		respWith("good", "", ""),
	}}
	got, err := Do(context.Background(), caller, baseRequest(), parseEchoOK, Options[string]{})
	if err != nil {
		t.Fatalf("Do() error = %v", err)
	}
	if got != "parsed:good" {
		t.Errorf("Do() = %q, want %q", got, "parsed:good")
	}
	if caller.callCount() != 3 {
		t.Fatalf("call count = %d, want 3", caller.callCount())
	}

	// Attempt 3 carries both nudge pairs in conversation order.
	msgs := caller.request(3).Messages
	if len(msgs) != 6 {
		t.Fatalf("attempt-3 request has %d messages, want 6", len(msgs))
	}
	wantRoles := []string{"system", "user", "assistant", "user", "assistant", "user"}
	for i, role := range wantRoles {
		if msgs[i].Role != role {
			t.Errorf("attempt-3 message %d role = %q, want %q", i, msgs[i].Role, role)
		}
	}
	if msgs[2].Content != "bad one" || msgs[4].Content != "bad two" {
		t.Errorf("assistant echoes = (%q, %q), want (bad one, bad two)", msgs[2].Content, msgs[4].Content)
	}
}

func TestDo_EchoFallsBackToReasoningFields(t *testing.T) {
	// When Content is empty the echo takes the first non-empty reasoning
	// candidate, mirroring the multi-candidate extraction order.
	caller := &fakeCaller{responses: []*llm.ChatResponse{
		respWith("", "reasoned but wrong", ""),
		respWith("good", "", ""),
	}}
	if _, err := Do(context.Background(), caller, baseRequest(), parseEchoOK, Options[string]{}); err != nil {
		t.Fatalf("Do() error = %v", err)
	}
	echo := caller.request(2).Messages[2]
	if echo.Role != "assistant" || echo.Content != "reasoned but wrong" {
		t.Errorf("echo = (%q, %q), want assistant echoing the reasoning field", echo.Role, echo.Content)
	}
}

// --- Final refusals and failure policies ---

func TestDo_FinalRefusal_ErrorPolicy(t *testing.T) {
	caller := &fakeCaller{responses: []*llm.ChatResponse{
		respWith("bad one", "", ""),
		respWith("bad two", "", ""),
		respWith("bad three", "", ""),
	}}
	got, err := Do(context.Background(), caller, baseRequest(), parseEchoOK, Options[string]{Kind: "probe"})
	if err == nil {
		t.Fatalf("Do() = %q, want an error after the final refusal", got)
	}
	if got != "" {
		t.Errorf("Do() value = %q, want the zero value", got)
	}
	if caller.callCount() != 3 {
		t.Errorf("call count = %d, want exactly 3 (initial + two nudges)", caller.callCount())
	}
	if !strings.Contains(err.Error(), "probe") || !strings.Contains(err.Error(), "3 attempts") {
		t.Errorf("error %q should name the kind and the attempt count", err.Error())
	}
	if !strings.Contains(err.Error(), "missing the required marker") {
		t.Errorf("error %q should wrap the last parse error", err.Error())
	}
}

func TestDo_FinalRefusal_FallbackPolicy(t *testing.T) {
	caller := &fakeCaller{responses: []*llm.ChatResponse{
		respWith("bad one", "", ""), respWith("bad two", "", ""), respWith("bad three", "", ""),
	}}
	got, err := Do(context.Background(), caller, baseRequest(), parseEchoOK, Options[string]{
		OnFailure:     OnFailureFallback,
		FallbackValue: "the original prompt",
	})
	if err != nil {
		t.Fatalf("Do() error = %v, want nil with OnFailureFallback", err)
	}
	if got != "the original prompt" {
		t.Errorf("Do() = %q, want the fallback value", got)
	}
	if caller.callCount() != 3 {
		t.Errorf("call count = %d, want 3", caller.callCount())
	}
}

// verdict models the judge-style typed fail-safe value (CONFIRM/DENY): the
// zero value is the UNSAFE choice, so the fail-safe must carry the safe one.
type verdict int

const (
	verdictAllow verdict = iota // zero value = unsafe
	verdictConfirm
)

// parseVerdictEcho is the verdict-typed parse used by the fail-safe test.
func parseVerdictEcho(resp *llm.ChatResponse) (verdict, error) {
	if resp != nil && strings.TrimSpace(resp.Message.Content) == "good" {
		return verdictAllow, nil
	}
	return 0, errors.New("missing the required marker")
}

func TestDo_FinalRefusal_FailSafePolicy(t *testing.T) {
	caller := &fakeCaller{responses: []*llm.ChatResponse{
		respWith("garbage", "", ""), respWith("garbage", "", ""), respWith("garbage", "", ""),
	}}
	got, err := Do(context.Background(), caller, baseRequest(), parseVerdictEcho, Options[verdict]{
		OnFailure:     OnFailureFailSafe,
		FallbackValue: verdictConfirm,
	})
	if err != nil {
		t.Fatalf("Do() error = %v, want nil with OnFailureFailSafe", err)
	}
	if got != verdictConfirm {
		t.Errorf("Do() = %v, want the typed fail-safe value verdictConfirm", got)
	}
}

// --- Transport errors are never retried ---

func TestDo_TransportErrorNotRetried(t *testing.T) {
	boom := errors.New("connection refused")
	caller := &fakeCaller{err: boom}
	got, err := Do(context.Background(), caller, baseRequest(), parseEchoOK, Options[string]{})
	if !errors.Is(err, boom) {
		t.Fatalf("Do() error = %v, want the transport error %v", err, boom)
	}
	if got != "" {
		t.Errorf("Do() value = %q, want the zero value", got)
	}
	if caller.callCount() != 1 {
		t.Errorf("call count = %d, want 1 (transport errors are the Router's domain)", caller.callCount())
	}
}

func TestDo_NilResponseIsParseFailure(t *testing.T) {
	// (nil, nil) is a provider contract violation but still "nothing
	// parseable": it must ride the nudge loop and the failure policies, not
	// surface as a transport error.
	caller := &fakeCaller{responses: []*llm.ChatResponse{nil, nil, nil}}
	got, err := Do(context.Background(), caller, baseRequest(), parseEchoOK, Options[string]{
		OnFailure:     OnFailureFallback,
		FallbackValue: "fallback",
	})
	if err != nil {
		t.Fatalf("Do() error = %v, want nil", err)
	}
	if got != "fallback" {
		t.Errorf("Do() = %q, want the fallback value", got)
	}
	if caller.callCount() != 3 {
		t.Errorf("call count = %d, want 3 (nil responses are retried, not propagated)", caller.callCount())
	}
}

// --- Guard rails ---

func TestDo_NilCaller(t *testing.T) {
	if _, err := Do(context.Background(), nil, baseRequest(), parseEchoOK, Options[string]{}); err == nil {
		t.Error("Do() with a nil caller must fail fast")
	}
}

func TestDo_NilParse(t *testing.T) {
	caller := &fakeCaller{responses: []*llm.ChatResponse{respWith("good", "", "")}}
	if _, err := Do[string](context.Background(), caller, baseRequest(), nil, Options[string]{}); err == nil {
		t.Error("Do() with a nil parse function must fail fast")
	}
	if caller.callCount() != 0 {
		t.Errorf("call count = %d, want 0 (misconfigured Do must not call the LLM)", caller.callCount())
	}
}

func TestDo_RequestNotMutated(t *testing.T) {
	caller := &fakeCaller{responses: []*llm.ChatResponse{
		respWith("bad", "", ""),
		respWith("good", "", ""),
	}}
	req := baseRequest()
	original := fmt.Sprint(req.Messages)
	if _, err := Do(context.Background(), caller, req, parseEchoOK, Options[string]{}); err != nil {
		t.Fatalf("Do() error = %v", err)
	}
	if got := fmt.Sprint(req.Messages); got != original {
		t.Errorf("caller's request was mutated: %s (want %s)", got, original)
	}
}

func TestDo_MetadataAgnostic(t *testing.T) {
	// Temperature / ReasoningEffort / CallPurpose come in on the request and
	// must reach every attempt untouched.
	temp := 0.2
	req := baseRequest()
	req.Temperature = &temp
	req.ReasoningEffort = "high"
	req.CallPurpose = llm.CallPurposeRouting

	caller := &fakeCaller{responses: []*llm.ChatResponse{
		respWith("bad", "", ""),
		respWith("good", "", ""),
	}}
	if _, err := Do(context.Background(), caller, req, parseEchoOK, Options[string]{}); err != nil {
		t.Fatalf("Do() error = %v", err)
	}
	retryReq := caller.request(2)
	if retryReq.Temperature == nil || *retryReq.Temperature != 0.2 {
		t.Errorf("retry request temperature = %v, want 0.2 preserved", retryReq.Temperature)
	}
	if retryReq.ReasoningEffort != "high" {
		t.Errorf("retry request reasoning effort = %q, want %q", retryReq.ReasoningEffort, "high")
	}
	if retryReq.CallPurpose != llm.CallPurposeRouting {
		t.Errorf("retry request call purpose = %q, want %q", retryReq.CallPurpose, llm.CallPurposeRouting)
	}
}

func TestDo_ContextCanceledBetweenAttempts(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	caller := &fakeCaller{
		responses: []*llm.ChatResponse{respWith("bad", "", ""), respWith("good", "", "")},
		onCall: func(n int) {
			if n == 1 {
				cancel()
			}
		},
	}
	got, err := Do(ctx, caller, baseRequest(), parseEchoOK, Options[string]{})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Do() error = %v, want context.Canceled", err)
	}
	if got != "" {
		t.Errorf("Do() value = %q, want the zero value", got)
	}
	if caller.callCount() != 1 {
		t.Errorf("call count = %d, want 1 (no extra call after cancellation)", caller.callCount())
	}
}

// --- Dump integration ---

func TestDo_DumpRecordsEveryAttempt(t *testing.T) {
	var buf bytes.Buffer
	caller := &fakeCaller{responses: []*llm.ChatResponse{
		respWith("bad", "", ""),
		respWith("good", "", ""),
	}}
	if _, err := Do(context.Background(), caller, baseRequest(), parseEchoOK, Options[string]{
		Kind: "probe",
		Dump: &buf,
	}); err != nil {
		t.Fatalf("Do() error = %v", err)
	}

	entries := make([]dumpEntry, 0, 4)
	for _, line := range strings.Split(strings.TrimSpace(buf.String()), "\n") {
		var e dumpEntry
		if err := json.Unmarshal([]byte(line), &e); err != nil {
			t.Fatalf("dump line %q is not valid JSON: %v", line, err)
		}
		entries = append(entries, e)
	}
	if len(entries) != 4 {
		t.Fatalf("got %d dump records, want 4 (request+response per attempt)", len(entries))
	}
	wantDirections := []string{"request", "response", "request", "response"}
	wantAttempts := []int{1, 1, 2, 2}
	for i, e := range entries {
		if e.Direction != wantDirections[i] {
			t.Errorf("record %d direction = %q, want %q", i, e.Direction, wantDirections[i])
		}
		if e.Attempt != wantAttempts[i] {
			t.Errorf("record %d attempt = %d, want %d", i, e.Attempt, wantAttempts[i])
		}
		if e.Timestamp == "" {
			t.Errorf("record %d carries no timestamp", i)
		}
	}
	// The second dumped request must carry the nudge pair.
	var secondReq llm.ChatRequest
	if err := json.Unmarshal(entries[2].Data, &secondReq); err != nil {
		t.Fatalf("request record is not a ChatRequest: %v", err)
	}
	if len(secondReq.Messages) != 4 {
		t.Errorf("dumped retry request has %d messages, want 4", len(secondReq.Messages))
	}
}

// --- Structured logging ---

func captureLogs(t *testing.T) (*bytes.Buffer, *slog.Logger) {
	t.Helper()
	var buf bytes.Buffer
	return &buf, slog.New(slog.NewJSONHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug}))
}

func decodeLogRecords(t *testing.T, buf *bytes.Buffer) []map[string]any {
	t.Helper()
	var out []map[string]any
	for _, line := range strings.Split(strings.TrimSpace(buf.String()), "\n") {
		if line == "" {
			continue
		}
		var rec map[string]any
		if err := json.Unmarshal([]byte(line), &rec); err != nil {
			t.Fatalf("log line %q is not valid JSON: %v", line, err)
		}
		out = append(out, rec)
	}
	return out
}

func TestDo_LoggingSuccessOutcome(t *testing.T) {
	buf, logger := captureLogs(t)
	caller := &fakeCaller{responses: []*llm.ChatResponse{respWith("good", "", "")}}
	if _, err := Do(context.Background(), caller, baseRequest(), parseEchoOK, Options[string]{
		Kind: "probe", Logger: logger,
	}); err != nil {
		t.Fatalf("Do() error = %v", err)
	}
	recs := decodeLogRecords(t, buf)
	if len(recs) != 1 {
		t.Fatalf("got %d log records, want 1", len(recs))
	}
	rec := recs[0]
	if rec["kind"] != "probe" {
		t.Errorf("log kind = %v, want probe", rec["kind"])
	}
	if rec["model"] != "test-model" {
		t.Errorf("log model = %v, want test-model", rec["model"])
	}
	if rec["outcome"] != "ok" {
		t.Errorf("log outcome = %v, want ok", rec["outcome"])
	}
	if rec["attempt"] != float64(1) {
		t.Errorf("log attempt = %v, want 1", rec["attempt"])
	}
	if _, ok := rec["duration"].(float64); !ok {
		// slog's JSON handler renders time.Duration as integer nanoseconds.
		t.Errorf("log duration = %v, want a numeric duration", rec["duration"])
	}
}

func TestDo_LoggingFailSafeOutcome(t *testing.T) {
	buf, logger := captureLogs(t)
	caller := &fakeCaller{responses: []*llm.ChatResponse{
		respWith("bad", "", ""), respWith("bad", "", ""), respWith("bad", "", ""),
	}}
	if _, err := Do(context.Background(), caller, baseRequest(), parseEchoOK, Options[string]{
		Kind: "probe", Logger: logger,
		OnFailure:     OnFailureFailSafe,
		FallbackValue: "safe",
	}); err != nil {
		t.Fatalf("Do() error = %v", err)
	}
	recs := decodeLogRecords(t, buf)
	if len(recs) == 0 {
		t.Fatal("expected at least one log record")
	}
	last := recs[len(recs)-1]
	if last["outcome"] != "fail_safe" {
		t.Errorf("final log outcome = %v, want fail_safe", last["outcome"])
	}
	if last["level"] != "WARN" {
		t.Errorf("final log level = %v, want WARN", last["level"])
	}
	if last["attempt"] != float64(3) {
		t.Errorf("final log attempt = %v, want 3", last["attempt"])
	}
}

func TestDo_LoggingTransportErrorOmitsDetails(t *testing.T) {
	// The judge's security posture: provider error text can echo sensitive
	// request content, so the log record must not carry it (the returned
	// error still does).
	buf, logger := captureLogs(t)
	caller := &fakeCaller{err: errors.New("connection refused: secrets in transit")}
	_, err := Do(context.Background(), caller, baseRequest(), parseEchoOK, Options[string]{
		Kind: "probe", Logger: logger,
	})
	if err == nil {
		t.Fatal("Do() expected the transport error to be returned")
	}
	recs := decodeLogRecords(t, buf)
	if len(recs) != 1 {
		t.Fatalf("got %d log records, want 1", len(recs))
	}
	if recs[0]["outcome"] != "transport_error" {
		t.Errorf("log outcome = %v, want transport_error", recs[0]["outcome"])
	}
	if line := buf.String(); strings.Contains(line, "secrets in transit") {
		t.Errorf("log record leaked the transport error detail: %s", line)
	}
}

// --- Default nudge wording ---

func TestNudgeMessage_DefaultAndHint(t *testing.T) {
	def := nudgeMessage("")
	if !strings.HasPrefix(def, "[System]") || !strings.Contains(def, "could not be parsed") {
		t.Errorf("default nudge = %q, want the [System] parse-failure wording", def)
	}
	withHint := nudgeMessage("Answer strictly as two lines: VERDICT: … / REASON: …")
	if !strings.Contains(withHint, "VERDICT: … / REASON: …") {
		t.Errorf("hinted nudge = %q, want the hint embedded", withHint)
	}
}

// --- Compile-time interface checks ---

func TestCallerInterfaceCompatibility(t *testing.T) {
	// llm.Router satisfies Caller structurally, so hosts pass their router
	// directly to Do.
	var _ Caller = (*llm.Router)(nil)
}

// TestDo_RetryHintFn verifies that a non-nil RetryHintFn renders the nudge
// hint from the triggering parse error (taking precedence over the static
// RetryHint), so callers whose corrective feedback depends on WHAT failed can
// restate the exact problem.
func TestDo_RetryHintFn(t *testing.T) {
	caller := &fakeCaller{responses: []*llm.ChatResponse{
		respWith("bad one", "", ""),
		respWith("bad two", "", ""),
		respWith("good", "", ""),
	}}
	hints := []string{}
	got, err := Do(context.Background(), caller, baseRequest(), parseEchoOK, Options[string]{
		RetryHint: "STATIC HINT MUST NOT APPEAR",
		RetryHintFn: func(parseErr error) string {
			hints = append(hints, parseErr.Error())
			return "dynamic hint from: " + parseErr.Error()
		},
	})
	if err != nil {
		t.Fatalf("Do() error = %v", err)
	}
	if got != "parsed:good" {
		t.Errorf("Do() = %q, want %q", got, "parsed:good")
	}
	if caller.callCount() != 3 {
		t.Fatalf("call count = %d, want 3", caller.callCount())
	}
	if len(hints) != 2 {
		t.Fatalf("RetryHintFn invocations = %d (%v), want 2 (one per nudge)", len(hints), hints)
	}
	for n, want := range []string{"missing the required marker", "missing the required marker"} {
		msg := caller.request(n + 2).Messages[3]
		if msg.Role != "user" || !strings.Contains(msg.Content, "dynamic hint from: "+want) {
			t.Errorf("nudge %d = (%q, %q), want the dynamic hint carrying the parse error", n+1, msg.Role, msg.Content)
		}
		if strings.Contains(msg.Content, "STATIC HINT MUST NOT APPEAR") {
			t.Errorf("nudge %d carries the static RetryHint although RetryHintFn is set", n+1)
		}
	}
}
