package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"syscall"
	"testing"
	"time"

	oai "github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/responses"
)

// TestTransportKind pins the diagnostic classification of connection-level
// failures. Observability only: the labels let upstream curators tell a
// timeout from a mid-body reset from a DNS miss when a failure reproduces on
// specific payloads across runs.
func TestTransportKind(t *testing.T) {
	deadlineCtx, cancel := context.WithTimeout(context.Background(), 5*time.Millisecond)
	defer cancel()
	<-deadlineCtx.Done()

	dnsErr := &net.DNSError{Err: "no such host", Name: "api.example.com", IsNotFound: true}
	sentinel := errors.New("wrapped")

	tests := []struct {
		name string
		err  error
		want string
	}{
		{"nil", nil, ""},
		{"context deadline", deadlineCtx.Err(), "timeout"},
		{"dns", dnsErr, "dns"},
		{"dns wrapped", fmt.Errorf("dial: %w", dnsErr), "dns"},
		{"conn reset", fmt.Errorf("read: %w", syscall.ECONNRESET), "reset"},
		{"broken pipe", syscall.EPIPE, "reset"},
		{"refused", syscall.ECONNREFUSED, "connect"},
		{"host unreachable", syscall.EHOSTUNREACH, "connect"},
		{"op error reset", &net.OpError{Op: "read", Err: syscall.ECONNRESET}, "reset"},
		{"op error refused", &net.OpError{Op: "dial", Err: syscall.ECONNREFUSED}, "connect"},
		{"url error inner reset", &url.Error{Op: "Post", URL: "https://x", Err: &net.OpError{Op: "read", Err: syscall.ECONNRESET}}, "reset"},
		{"url error plain", &url.Error{Op: "Post", URL: "https://x", Err: sentinel}, "connect"},
		{"eof", io.EOF, "eof"},
		{"unexpected eof", io.ErrUnexpectedEOF, "eof"},
		{"plain error", sentinel, "other"},
		{"tls text", errors.New("tls: handshake failure"), "tls"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := TransportKind(tt.err); got != tt.want {
				t.Errorf("TransportKind = %q, want %q", got, tt.want)
			}
		})
	}
}

// TestAnnotateChatTransport_ConnectionLevelOnly pins that the transport
// suffix attaches only to HTTP 0 (*Error with StatusCode 0) failures and that
// the appended text stays outside the *Error wrapper so retryability
// classification is untouched.
func TestAnnotateChatTransport_ConnectionLevelOnly(t *testing.T) {
	p := &OpenAIProvider{name: "deepseek"}

	// Connection-level failure: annotated, still retryable via the chain.
	inner := NewError("deepseek", 0, true, fmt.Errorf("openai chat completion: %w",
		&url.Error{Op: "Post", Err: &net.OpError{Op: "read", Err: syscall.ECONNRESET}}))
	got := p.annotateChatTransport(inner, oai.ChatCompletionNewParams{})
	if got == nil {
		t.Fatal("expected an error")
	}
	if !strings.Contains(got.Error(), "(transport=reset, request_bytes=") {
		t.Errorf("annotated error = %q, want transport=reset suffix with a size estimate", got.Error())
	}
	if !IsRetryable(got) {
		t.Error("annotation must not alter retryability")
	}
	var llmErr *Error
	if !errors.As(got, &llmErr) || llmErr.StatusCode != 0 {
		t.Error("*Error must stay unwrap-reachable with StatusCode 0 after annotation")
	}

	// HTTP-carrying failure: untouched.
	httpErr := NewError("deepseek", 429, true, errors.New("rate limited"))
	before := httpErr.Error()
	if after := p.annotateChatTransport(httpErr, oai.ChatCompletionNewParams{}).Error(); after != before {
		t.Errorf("HTTP-carrying failure changed: %q -> %q", before, after)
	}

	// Nil passes through.
	if err := p.annotateChatTransport(nil, oai.ChatCompletionNewParams{}); err != nil {
		t.Errorf("nil error must pass through, got %v", err)
	}

	// Non-llm error (pre-call validation): untouched.
	plain := errors.New("openai: missing media type")
	if after := p.annotateChatTransport(plain, oai.ChatCompletionNewParams{}).Error(); after != plain.Error() {
		t.Errorf("non-llm error changed: %q -> %q", plain.Error(), after)
	}
}

// TestAnnotateChatTransport_RequestSizeEstimate pins the serialized-size
// estimate: the marshaled request body length, so census triage can correlate
// reproducible failures with payload size.
func TestAnnotateChatTransport_RequestSizeEstimate(t *testing.T) {
	p := &OpenAIProvider{name: "deepseek"}
	params := oai.ChatCompletionNewParams{}
	params.Messages = append(params.Messages, oai.UserMessage(strings.Repeat("x", 1000)))
	inner := NewError("deepseek", 0, true, errors.New("post \"https://api.example.com\": connection reset by peer"))
	got := p.annotateChatTransport(inner, params)
	if got == nil {
		t.Fatal("expected an error")
	}
	if !strings.Contains(got.Error(), "request_bytes=") {
		t.Errorf("annotated error = %q, want a request_bytes estimate", got.Error())
	}
	if strings.Contains(got.Error(), "request_bytes=-1") {
		t.Errorf("marshalable params must produce a real size estimate, got %q", got.Error())
	}
}

// TestTransportKindErrno pins the errno-to-label mapping behind TransportKind's
// errors.As(syscall.Errno) branch. The POSIX sentinels classify identically on
// every platform — on Windows they are matched for parity even though real
// failures carry raw Winsock codes (see errno_windows_test.go for those).
func TestTransportKindErrno(t *testing.T) {
	tests := []struct {
		name  string
		errno syscall.Errno
		want  string
	}{
		{"ECONNRESET", syscall.ECONNRESET, "reset"},
		{"EPIPE", syscall.EPIPE, "reset"},
		{"ECONNREFUSED", syscall.ECONNREFUSED, "connect"},
		{"EHOSTUNREACH", syscall.EHOSTUNREACH, "connect"},
		{"ENETUNREACH", syscall.ENETUNREACH, "connect"},
		{"ETIMEDOUT", syscall.ETIMEDOUT, "connect"},
		{"ECONNABORTED", syscall.ECONNABORTED, "connect"},
		{"unrelated", syscall.EINVAL, ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := transportKindForErrno(tt.errno); got != tt.want {
				t.Errorf("transportKindForErrno(%s = %d) = %q, want %q", tt.name, tt.errno, got, tt.want)
			}
		})
	}
}

// TestAnnotateChatTransport_LogsRequestedModel pins that the WARN record's
// model key carries the model actually requested (params.Model) — the
// dimension census triage correlates failures by — not the provider name,
// which the provider key already carries.
func TestAnnotateChatTransport_LogsRequestedModel(t *testing.T) {
	var buf bytes.Buffer
	p := &OpenAIProvider{name: "deepseek", logger: slog.New(slog.NewTextHandler(&buf, nil))}

	inner := NewError("deepseek", 0, true, fmt.Errorf("openai chat completion: %w",
		&url.Error{Op: "Post", Err: &net.OpError{Op: "read", Err: syscall.ECONNRESET}}))
	params := oai.ChatCompletionNewParams{Model: "deepseek-chat"}
	if got := p.annotateChatTransport(inner, params); got == nil {
		t.Fatal("expected an error")
	}
	logged := buf.String()
	if !strings.Contains(logged, "model=deepseek-chat") {
		t.Errorf("WARN log must carry the requested model (deepseek-chat), got %q", logged)
	}
	if strings.Contains(logged, "model=deepseek ") || strings.Contains(logged, "model=deepseek\n") {
		t.Errorf("model key must not carry the provider name, got %q", logged)
	}
	if !strings.Contains(logged, "provider=deepseek") {
		t.Errorf("WARN log must keep the provider key, got %q", logged)
	}
}

// TestAnnotateChatTransport_CallerCancellationNotAnnotated pins that
// caller-initiated cancellation is not annotated: a cancellation is a routine
// host decision (shutdown, pause, step budget), not a connection-level
// provider failure, so it passes through with no WARN log and no
// transport=…/request_bytes=… suffix. A genuine deadline expiry must stay
// annotated as a timeout.
func TestAnnotateChatTransport_CallerCancellationNotAnnotated(t *testing.T) {
	var buf bytes.Buffer
	p := &OpenAIProvider{name: "deepseek", logger: slog.New(slog.NewTextHandler(&buf, nil))}

	// Real-shaped cancellation: HTTP 0 *Error wrapping the SDK's *url.Error
	// wrapping context.Canceled — exactly how a mid-call cancel surfaces.
	inner := NewError("deepseek", 0, true, fmt.Errorf("openai chat completion: %w",
		&url.Error{Op: "Post", URL: "https://api.example.com/v1/chat/completions", Err: context.Canceled}))
	got := p.annotateChatTransport(inner, oai.ChatCompletionNewParams{Model: "deepseek-chat"})
	if got == nil {
		t.Fatal("expected an error")
	}
	if !errors.Is(got, context.Canceled) {
		t.Error("cancellation must stay errors.Is-detectable after annotate")
	}
	if msg := got.Error(); strings.Contains(msg, "transport=") || strings.Contains(msg, "request_bytes=") {
		t.Errorf("caller cancellation must not be annotated, got %q", msg)
	}
	if buf.Len() != 0 {
		t.Errorf("caller cancellation must not WARN-log, got %q", buf.String())
	}

	// Bare and plainly wrapped cancellation pass through untouched as well.
	for _, err := range []error{
		context.Canceled,
		fmt.Errorf("openai chat completion: %w", context.Canceled),
	} {
		if got := p.annotateChatTransport(err, oai.ChatCompletionNewParams{}); got.Error() != err.Error() {
			t.Errorf("cancellation %v changed: %q", err, got.Error())
		}
	}
	if buf.Len() != 0 {
		t.Errorf("cancellation must never WARN-log, got %q", buf.String())
	}

	// A genuine deadline expiry is a transport outcome, not cancellation:
	// still annotated as a timeout.
	deadline := NewError("deepseek", 0, true, fmt.Errorf("openai chat completion: %w",
		&url.Error{Op: "Post", URL: "https://api.example.com/v1/chat/completions", Err: context.DeadlineExceeded}))
	annotated := p.annotateChatTransport(deadline, oai.ChatCompletionNewParams{Model: "deepseek-chat"})
	if !strings.Contains(annotated.Error(), "(transport=timeout, request_bytes=") {
		t.Errorf("deadline expiry must stay annotated as timeout, got %q", annotated.Error())
	}
}

// ---------------------------------------------------------------------------
// Responses API transport annotation (annotateResponsesTransport) — the
// Responses-path parity of the Chat Completions tests above. Both wire
// branches (synchronous Responses.New and streaming stream.Err()) route
// their connection-level failures through the same annotator.
// ---------------------------------------------------------------------------

// TestAnnotateResponsesTransport_ConnectionLevelOnly pins that the transport
// suffix attaches only to HTTP 0 (*Error with StatusCode 0) failures on the
// Responses path and that the appended text stays outside the *Error
// wrapper so retryability classification is untouched. The annotated error is
// shaped exactly as production hands it over: wrapResponsesError's HTTP-0
// *Error around the SDK's transport error.
func TestAnnotateResponsesTransport_ConnectionLevelOnly(t *testing.T) {
	silent := slog.New(slog.NewTextHandler(io.Discard, nil))

	// Connection-level failure: annotated, still retryable via the chain.
	inner := wrapResponsesError("p", &url.Error{Op: "Post", URL: "https://api.example.com/v1/responses",
		Err: &net.OpError{Op: "read", Err: syscall.ECONNRESET}})
	params := responses.ResponseNewParams{Model: "gpt-5.6"}
	got := annotateResponsesTransport(silent, "p", inner, params)
	if got == nil {
		t.Fatal("expected an error")
	}
	if !strings.Contains(got.Error(), "(transport=reset, request_bytes=") {
		t.Errorf("annotated error = %q, want transport=reset suffix with a size estimate", got.Error())
	}
	if !IsRetryable(got) {
		t.Error("annotation must not alter retryability")
	}
	var llmErr *Error
	if !errors.As(got, &llmErr) || llmErr.StatusCode != 0 {
		t.Error("*Error must stay unwrap-reachable with StatusCode 0 after annotation")
	}

	// HTTP-carrying failure: untouched.
	httpErr := NewError("p", 429, true, errors.New("rate limited"))
	before := httpErr.Error()
	if after := annotateResponsesTransport(silent, "p", httpErr, responses.ResponseNewParams{}).Error(); after != before {
		t.Errorf("HTTP-carrying failure changed: %q -> %q", before, after)
	}

	// Nil passes through.
	if err := annotateResponsesTransport(silent, "p", nil, responses.ResponseNewParams{}); err != nil {
		t.Errorf("nil error must pass through, got %v", err)
	}

	// Non-llm error (pre-call validation): untouched.
	plain := errors.New("p: invalid image block")
	if after := annotateResponsesTransport(silent, "p", plain, responses.ResponseNewParams{}).Error(); after != plain.Error() {
		t.Errorf("non-llm error changed: %q -> %q", plain.Error(), after)
	}
}

// TestAnnotateResponsesTransport_RequestSizeEstimate pins the serialized-size
// estimate on the Responses path: the marshaled request body length, so
// census triage can correlate reproducible failures with payload size.
func TestAnnotateResponsesTransport_RequestSizeEstimate(t *testing.T) {
	silent := slog.New(slog.NewTextHandler(io.Discard, nil))
	params := responses.ResponseNewParams{Model: "gpt-5.6"}
	params.Input = responses.ResponseNewParamsInputUnion{
		OfInputItemList: convertToResponsesInput([]Message{{Role: "user", Content: strings.Repeat("x", 1000)}}, nil, nil),
	}
	inner := wrapResponsesError("p", errors.New("post \"https://api.example.com\": connection reset by peer"))
	got := annotateResponsesTransport(silent, "p", inner, params)
	if got == nil {
		t.Fatal("expected an error")
	}
	if !strings.Contains(got.Error(), "request_bytes=") {
		t.Errorf("annotated error = %q, want a request_bytes estimate", got.Error())
	}
	if strings.Contains(got.Error(), "request_bytes=-1") {
		t.Errorf("marshalable params must produce a real size estimate, got %q", got.Error())
	}
}

// TestAnnotateResponsesTransport_WarnAttrsAreSecretFree pins the exact WARN
// attribute set — provider, model, transport, request_bytes, err — and
// nothing else. The request body (which may embed task content, image
// payloads or caller-threaded credentials) is deliberately reduced to a byte
// count, and no credential-shaped attribute may ride the record.
func TestAnnotateResponsesTransport_WarnAttrsAreSecretFree(t *testing.T) {
	var buf bytes.Buffer
	params := responses.ResponseNewParams{Model: "gpt-5.6"}
	params.Input = responses.ResponseNewParamsInputUnion{
		OfInputItemList: convertToResponsesInput([]Message{{Role: "user", Content: "hi"}}, nil, nil),
	}
	inner := wrapResponsesError("p", &url.Error{Op: "Post", URL: "https://api.example.com/v1/responses",
		Err: &net.OpError{Op: "read", Err: syscall.ECONNRESET}})
	if got := annotateResponsesTransport(slog.New(slog.NewJSONHandler(&buf, nil)), "p", inner, params); got == nil {
		t.Fatal("expected an error")
	}

	var rec map[string]any
	if err := json.Unmarshal([]byte(strings.TrimSpace(buf.String())), &rec); err != nil {
		t.Fatalf("WARN record is not a single JSON object: %v (%q)", err, buf.String())
	}
	for key, want := range map[string]any{
		"provider": "p", "model": "gpt-5.6", "transport": "reset",
	} {
		if got := rec[key]; got != want {
			t.Errorf("attr %s = %v, want %v", key, got, want)
		}
	}
	if rb, ok := rec["request_bytes"].(float64); !ok || rb <= 0 {
		t.Errorf("request_bytes = %v, want a positive size estimate", rec["request_bytes"])
	}
	if msg, _ := rec["msg"].(string); msg != "openai: connection-level responses API failure" {
		t.Errorf("msg = %q", msg)
	}
	if s, _ := rec["err"].(string); s == "" {
		t.Errorf("err attr missing or empty: %v", rec["err"])
	}
	for _, forbidden := range []string{
		"api_key", "apikey", "authorization", "token", "bearer", "secret", "credentials",
		"input", "messages", "body", "request",
	} {
		if _, present := rec[forbidden]; present {
			t.Errorf("WARN record carries forbidden attr %q: %v", forbidden, rec)
		}
	}
	allowed := map[string]bool{
		"time": true, "level": true, "msg": true, "provider": true,
		"model": true, "transport": true, "request_bytes": true, "err": true,
	}
	for k := range rec {
		if !allowed[k] {
			t.Errorf("unexpected WARN attr %q — the attribute set is pinned (time/level/msg/provider/model/transport/request_bytes/err)", k)
		}
	}
}

// TestAnnotateResponsesTransport_CallerCancellationNotAnnotated pins that
// caller-initiated cancellation is not annotated on the Responses path: a
// cancellation is a routine host decision (shutdown, pause, step budget), not
// a connection-level provider failure, so it passes through with no WARN log
// and no transport=…/request_bytes=… suffix. A genuine deadline expiry must
// stay annotated as a timeout.
func TestAnnotateResponsesTransport_CallerCancellationNotAnnotated(t *testing.T) {
	var buf bytes.Buffer
	logger := slog.New(slog.NewTextHandler(&buf, nil))

	// Real-shaped cancellation: HTTP 0 *Error wrapping the SDK's *url.Error
	// wrapping context.Canceled — exactly how a mid-call cancel surfaces.
	inner := wrapResponsesError("p", &url.Error{Op: "Post", URL: "https://api.example.com/v1/responses", Err: context.Canceled})
	got := annotateResponsesTransport(logger, "p", inner, responses.ResponseNewParams{Model: "gpt-5.6"})
	if got == nil {
		t.Fatal("expected an error")
	}
	if !errors.Is(got, context.Canceled) {
		t.Error("cancellation must stay errors.Is-detectable after annotate")
	}
	if msg := got.Error(); strings.Contains(msg, "transport=") || strings.Contains(msg, "request_bytes=") {
		t.Errorf("caller cancellation must not be annotated, got %q", msg)
	}
	if buf.Len() != 0 {
		t.Errorf("caller cancellation must not WARN-log, got %q", buf.String())
	}

	// Bare and plainly wrapped cancellation pass through untouched as well.
	for _, err := range []error{
		context.Canceled,
		fmt.Errorf("responses API: %w", context.Canceled),
	} {
		if got := annotateResponsesTransport(logger, "p", err, responses.ResponseNewParams{}); got.Error() != err.Error() {
			t.Errorf("cancellation %v changed: %q", err, got.Error())
		}
	}
	if buf.Len() != 0 {
		t.Errorf("cancellation must never WARN-log, got %q", buf.String())
	}

	// A genuine deadline expiry is a transport outcome, not cancellation:
	// still annotated as a timeout.
	deadline := wrapResponsesError("p", &url.Error{Op: "Post", URL: "https://api.example.com/v1/responses", Err: context.DeadlineExceeded})
	annotated := annotateResponsesTransport(logger, "p", deadline, responses.ResponseNewParams{Model: "gpt-5.6"})
	if !strings.Contains(annotated.Error(), "(transport=timeout, request_bytes=") {
		t.Errorf("deadline expiry must stay annotated as timeout, got %q", annotated.Error())
	}
}

// TestOpenAIProvider_ResponsesSyncTransportAnnotatedOnWire induces a genuine
// connection failure (a closed loopback port) through the real SDK stack on
// the synchronous Responses path and pins the observable outcome: one WARN
// carrying the transport class, the provider and the requested model, a
// byte-count estimate, and no credential material; the returned error text
// carries the same annotation for the caller.
func TestOpenAIProvider_ResponsesSyncTransportAnnotatedOnWire(t *testing.T) {
	var lc net.ListenConfig
	lis, err := lc.Listen(context.Background(), "tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	closedURL := "http://" + lis.Addr().String()
	_ = lis.Close() // the port now refuses every dial

	var buf bytes.Buffer
	client := oai.NewClient(
		option.WithAPIKey("sk-test-secret-value"),
		option.WithBaseURL(closedURL),
		option.WithMaxRetries(0),
	)

	req := ChatRequest{Model: "gpt-5.6", Messages: []Message{{Role: "user", Content: "hi"}}}
	_, err = responsesAPICompletion(context.Background(), &client, "p", closedURL, req, slog.New(slog.NewTextHandler(&buf, nil)))
	if err == nil {
		t.Fatal("expected the connection failure to surface")
	}
	logged := buf.String()
	if !strings.Contains(logged, "openai: connection-level responses API failure") {
		t.Errorf("expected a connection-level WARN, got %q", logged)
	}
	if !strings.Contains(logged, "provider=p") {
		t.Errorf("WARN must carry the provider, got %q", logged)
	}
	if !strings.Contains(logged, "model=gpt-5.6") {
		t.Errorf("WARN must carry the requested model, got %q", logged)
	}
	if !strings.Contains(logged, "transport=connect") {
		t.Errorf("refused loopback dial must classify as connect, got %q", logged)
	}
	if !strings.Contains(logged, "request_bytes=") || strings.Contains(logged, "request_bytes=-1") {
		t.Errorf("WARN must carry a real request size estimate, got %q", logged)
	}
	if strings.Contains(logged, "sk-test-secret-value") {
		t.Errorf("WARN leaks the API key: %q", logged)
	}
	if !strings.Contains(err.Error(), "(transport=connect, request_bytes=") {
		t.Errorf("returned error must carry the annotation, got %q", err.Error())
	}
	if strings.Contains(err.Error(), "sk-test-secret-value") {
		t.Errorf("returned error leaks the API key: %q", err.Error())
	}
}

// TestOpenAIProvider_ResponsesStreamTransportAnnotatedOnWire induces a
// mid-stream connection reset (an SSE server that accepts the request,
// delivers one event, then resets the connection with a linger-0 close) and
// pins that the stream.Err() branch annotates exactly like the synchronous
// one: WARN with a stable transport class, size estimate, and no credential
// material. The exact label for an abrupt close is OS/timing dependent
// (reset/eof/connect), so the class is checked against the stable
// TransportKind vocabulary — the exact-label pins live in the unit tests
// above.
func TestOpenAIProvider_ResponsesStreamTransportAnnotatedOnWire(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		h, ok := w.(http.Hijacker)
		if !ok {
			return
		}
		conn, rw, hErr := h.Hijack()
		if hErr != nil {
			return
		}
		// One complete chunked SSE event keeps the client's stream live; the
		// linger-0 close then resets the next body read (no terminating
		// zero-chunk, no clean EOF).
		payload := "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_x\"}}\n\n"
		_, _ = fmt.Fprintf(rw, "%x\r\n%s\r\n", len(payload), payload)
		_ = rw.Flush()
		if tc, ok := conn.(*net.TCPConn); ok {
			_ = tc.SetLinger(0)
		}
		_ = conn.Close()
	}))
	t.Cleanup(srv.Close)

	var buf bytes.Buffer
	client := oai.NewClient(
		option.WithAPIKey("sk-test-secret-value"),
		option.WithBaseURL(srv.URL),
		option.WithMaxRetries(0),
	)

	req := ChatRequest{Model: "gpt-5.6", Messages: []Message{{Role: "user", Content: "hi"}}}
	_, err := responsesAPICompletionStream(context.Background(), &client, "p", srv.URL, req, false, slog.New(slog.NewTextHandler(&buf, nil)))
	if err == nil {
		t.Fatal("expected the mid-stream failure to surface")
	}
	logged := buf.String()
	if !strings.Contains(logged, "openai: connection-level responses API failure") {
		t.Errorf("expected a connection-level WARN, got %q", logged)
	}
	if !strings.Contains(logged, "provider=p") || !strings.Contains(logged, "model=gpt-5.6") {
		t.Errorf("WARN must carry provider and requested model, got %q", logged)
	}
	idx := strings.Index(logged, "transport=")
	if idx < 0 {
		t.Fatalf("WARN must carry a transport class, got %q", logged)
	}
	label := logged[idx+len("transport="):]
	if i := strings.IndexAny(label, " \n"); i >= 0 {
		label = label[:i]
	}
	known := map[string]bool{"timeout": true, "dns": true, "connect": true, "reset": true, "tls": true, "eof": true, "other": true}
	if !known[label] {
		t.Errorf("transport label %q is not a stable TransportKind class, got %q", label, logged)
	}
	if !strings.Contains(logged, "request_bytes=") || strings.Contains(logged, "request_bytes=-1") {
		t.Errorf("WARN must carry a real request size estimate, got %q", logged)
	}
	if strings.Contains(logged, "sk-test-secret-value") {
		t.Errorf("WARN leaks the API key: %q", logged)
	}
	if !strings.Contains(err.Error(), "(transport="+label+", request_bytes=") {
		t.Errorf("returned error must carry the annotation, got %q", err.Error())
	}
	if strings.Contains(err.Error(), "sk-test-secret-value") {
		t.Errorf("returned error leaks the API key: %q", err.Error())
	}
}

// TestOpenAIProvider_ResponsesTransportSuccessStaysSilent pins that a
// successful Responses call — synchronous and streamed — emits no
// connection-level WARN: the annotation is failure-path only.
func TestOpenAIProvider_ResponsesTransportSuccessStaysSilent(t *testing.T) {
	completed := `{"id":"resp_1","object":"response","status":"completed","output":[{"type":"message","id":"msg_1","role":"assistant","content":[{"type":"output_text","text":"ok"}]}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`

	t.Run("sync", func(t *testing.T) {
		var buf bytes.Buffer
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, completed)
		}))
		t.Cleanup(srv.Close)
		p, err := NewOpenAIProvider(OpenAIProviderConfig{Name: "p", APIKey: "k", BaseURL: srv.URL, Logger: slog.New(slog.NewTextHandler(&buf, nil))})
		if err != nil {
			t.Fatalf("NewOpenAIProvider: %v", err)
		}
		if _, err := p.ChatCompletion(context.Background(), ChatRequest{
			Model:    "gpt-5.6",
			Protocol: ProtocolResponses,
			Messages: []Message{{Role: "user", Content: "hi"}},
		}); err != nil {
			t.Fatalf("ChatCompletion(responses): %v", err)
		}
		if buf.Len() != 0 {
			t.Errorf("successful sync call must not WARN-log, got %q", buf.String())
		}
	})

	t.Run("stream", func(t *testing.T) {
		var buf bytes.Buffer
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			respSSE(w, `{"type":"response.created","response":{"id":"resp_1"}}`)
			respSSE(w, fmt.Sprintf(`{"type":"response.completed","response":%s}`, completed))
		}))
		t.Cleanup(srv.Close)
		p, err := NewOpenAIProvider(OpenAIProviderConfig{Name: "p", APIKey: "k", BaseURL: srv.URL, Logger: slog.New(slog.NewTextHandler(&buf, nil))})
		if err != nil {
			t.Fatalf("NewOpenAIProvider: %v", err)
		}
		if _, err := p.ChatCompletion(context.Background(), ChatRequest{
			Model:     "gpt-5.6",
			Protocol:  ProtocolResponses,
			Messages:  []Message{{Role: "user", Content: "hi"}},
			DeltaSink: func(StreamDelta) error { return nil },
		}); err != nil {
			t.Fatalf("ChatCompletion(responses stream): %v", err)
		}
		if buf.Len() != 0 {
			t.Errorf("successful stream call must not WARN-log, got %q", buf.String())
		}
	})
}
