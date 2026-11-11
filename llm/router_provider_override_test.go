package llm

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

// TestRouter_Call_ProviderOverride_RoutesToNamedProvider verifies that a
// non-empty req.Provider routes the call to the named configured provider
// instead of the active one, and that the active provider is never attempted.
func TestRouter_Call_ProviderOverride_RoutesToNamedProvider(t *testing.T) {
	primary := &mockProvider{
		name: "primary",
		response: &ChatResponse{
			Message:    Message{Role: "assistant", Content: "from-primary"},
			StopReason: "end_turn",
		},
	}
	secondary := &mockProvider{
		name: "secondary",
		response: &ChatResponse{
			Message:    Message{Role: "assistant", Content: "from-secondary"},
			StopReason: "end_turn",
		},
	}

	router := newTestRouter(
		map[string]*mockProvider{"primary": primary, "secondary": secondary},
		"primary", "primary/model-a",
	)

	req := ChatRequest{
		Provider: "secondary",
		Model:    "model-b",
		Messages: []Message{{Role: "user", Content: "Hi"}},
	}
	resp, err := router.Call(context.Background(), req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if secondary.callCount != 1 {
		t.Errorf("secondary.callCount = %d, want 1", secondary.callCount)
	}
	if primary.callCount != 0 {
		t.Errorf("primary.callCount = %d, want 0 (override must bypass the active provider)", primary.callCount)
	}
	if resp.Message.Content != "from-secondary" {
		t.Errorf("response content = %q, want %q", resp.Message.Content, "from-secondary")
	}
	if secondary.lastReq.Model != "model-b" {
		t.Errorf("secondary received req.Model = %q, want %q", secondary.lastReq.Model, "model-b")
	}
}

// TestRouter_Call_ProviderOverride_UnknownProvider_NoRetry verifies that an
// unknown provider name fails the call immediately with the ErrUnknownProvider
// sentinel (never retried): zero attempts against any provider, so the retry
// loop never runs.
func TestRouter_Call_ProviderOverride_UnknownProvider_NoRetry(t *testing.T) {
	mock := &mockProvider{
		name: "primary",
		response: &ChatResponse{
			Message:    Message{Role: "assistant", Content: "should-never-happen"},
			StopReason: "end_turn",
		},
	}

	router := newTestRouter(
		map[string]*mockProvider{"primary": mock},
		"primary", "primary/model-a",
	)

	req := ChatRequest{
		Provider: "does-not-exist",
		Messages: []Message{{Role: "user", Content: "Hi"}},
	}
	resp, err := router.Call(context.Background(), req)
	if err == nil {
		t.Fatalf("expected an error for an unknown provider, got resp %+v", resp)
	}
	if resp != nil {
		t.Errorf("expected nil response on provider-miss error, got %+v", resp)
	}

	if !errors.Is(err, ErrUnknownProvider) {
		t.Fatalf("error %v does not match ErrUnknownProvider; hosts detect the miss via errors.Is", err)
	}
	if IsRetryable(err) {
		t.Errorf("provider-miss error must be non-retryable, got retryable=true (%v)", err)
	}
	if !strings.Contains(err.Error(), `"does-not-exist"`) {
		t.Errorf("error message must name the unknown provider, got %v", err)
	}

	// Zero retry attempts: any attempt (initial or retried) would have
	// incremented the mock's counter.
	if mock.callCount != 0 {
		t.Errorf("mock.callCount = %d, want 0 (the retry loop must never run on a provider miss)", mock.callCount)
	}
}

// TestRouter_Call_ProviderOverride_SurvivesSetModel verifies that a
// caller-pinned provider+model pair is immune to a mid-task SetModel: the
// router's active provider/model switch does not affect requests carrying an
// explicit Provider.
func TestRouter_Call_ProviderOverride_SurvivesSetModel(t *testing.T) {
	primary := &mockProvider{
		name: "primary",
		response: &ChatResponse{
			Message:    Message{Role: "assistant", Content: "from-primary"},
			StopReason: "end_turn",
		},
	}
	secondary := &mockProvider{
		name: "secondary",
		response: &ChatResponse{
			Message:    Message{Role: "assistant", Content: "from-secondary"},
			StopReason: "end_turn",
		},
	}

	router := newTestRouter(
		map[string]*mockProvider{"primary": primary, "secondary": secondary},
		"primary", "primary/model-a",
	)
	// newTestRouter leaves the reverse index empty; populate it so SetModel
	// can resolve composite IDs.
	router.modelToProvider = map[string]string{
		"primary/model-a":   "primary",
		"primary/model-a2":  "primary",
		"secondary/model-b": "secondary",
	}

	ctx := context.Background()
	req := ChatRequest{
		Provider: "secondary",
		Model:    "model-b",
		Messages: []Message{{Role: "user", Content: "Hi"}},
	}

	if _, err := router.Call(ctx, req); err != nil {
		t.Fatalf("pre-SetModel call: unexpected error: %v", err)
	}
	if secondary.callCount != 1 {
		t.Fatalf("pre-SetModel: secondary.callCount = %d, want 1", secondary.callCount)
	}

	// Mid-task switch of the router's active provider+model.
	if err := router.SetModel(ctx, "primary/model-a2"); err != nil {
		t.Fatalf("SetModel: unexpected error: %v", err)
	}

	if _, err := router.Call(ctx, req); err != nil {
		t.Fatalf("post-SetModel call: unexpected error: %v", err)
	}
	if secondary.callCount != 2 {
		t.Errorf("post-SetModel: secondary.callCount = %d, want 2 (pinned request must keep routing to secondary)", secondary.callCount)
	}
	if primary.callCount != 0 {
		t.Errorf("primary.callCount = %d, want 0 (SetModel must not affect a pinned caller)", primary.callCount)
	}
	if got := secondary.lastReq.Model; got != "model-b" {
		t.Errorf("post-SetModel: secondary received req.Model = %q, want the pinned %q", got, "model-b")
	}
}

// TestRouter_Call_ProviderOverride_DeltaSinkStreams verifies the streaming
// seam survives the provider-override path: a non-nil DeltaSink is delivered
// to the overridden provider, deltas flow through the router's retry guard,
// and the fully-assembled response is still returned.
func TestRouter_Call_ProviderOverride_DeltaSinkStreams(t *testing.T) {
	primary := &mockProvider{name: "primary"}
	secondary := &streamingMockProvider{
		name: "secondary",
		deltas: []StreamDelta{
			{Text: "Hel"},
			{Text: "lo"},
		},
		response: &ChatResponse{
			Message:    Message{Role: "assistant", Content: "Hello"},
			StopReason: "end_turn",
		},
	}

	router := newTestRouter(
		map[string]*mockProvider{"primary": primary},
		"primary", "primary/model-a",
	)
	// Register the streaming provider beyond the typed map helper: same
	// providers map, different implementation.
	router.providers["secondary"] = secondary

	var got strings.Builder
	sink := func(d StreamDelta) error {
		got.WriteString(d.Text)
		return nil
	}

	req := ChatRequest{
		Provider:  "secondary",
		Model:     "model-b",
		Messages:  []Message{{Role: "user", Content: "Hi"}},
		DeltaSink: sink,
	}
	resp, err := router.Call(context.Background(), req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.String() != "Hello" {
		t.Errorf("streamed deltas = %q, want %q", got.String(), "Hello")
	}
	if resp == nil || resp.Message.Content != "Hello" {
		t.Errorf("assembled response = %+v, want content %q", resp, "Hello")
	}
	if primary.callCount != 0 {
		t.Errorf("primary.callCount = %d, want 0", primary.callCount)
	}
}

// TestChatRequest_Provider_DumpVisibility verifies the JSON contract of the
// new field: an empty Provider is omitted entirely (existing request dumps
// stay byte-identical), while a set Provider is serialized and round-trips.
func TestChatRequest_Provider_DumpVisibility(t *testing.T) {
	legacy := ChatRequest{
		Model:    "m",
		Messages: []Message{{Role: "user", Content: "hi"}},
	}
	b, err := json.Marshal(legacy)
	if err != nil {
		t.Fatalf("marshal: unexpected error: %v", err)
	}
	if strings.Contains(string(b), "provider") {
		t.Errorf("empty Provider must be omitted from dumps (byte-identical legacy path), got %s", b)
	}

	override := legacy
	override.Provider = "openai_compatible"
	b, err = json.Marshal(override)
	if err != nil {
		t.Fatalf("marshal: unexpected error: %v", err)
	}
	if !strings.Contains(string(b), `"provider":"openai_compatible"`) {
		t.Errorf("set Provider must be dump-visible, got %s", b)
	}
	var back ChatRequest
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatalf("unmarshal: unexpected error: %v", err)
	}
	if back.Provider != "openai_compatible" {
		t.Errorf("round-tripped Provider = %q, want %q", back.Provider, "openai_compatible")
	}
}

// streamingMockProvider is a Provider that invokes the request's DeltaSink
// (if any) before returning its assembled response — the minimal streaming
// contract Router.Call relies on.
type streamingMockProvider struct {
	name     string
	deltas   []StreamDelta
	response *ChatResponse
	calls    int
}

func (m *streamingMockProvider) Name() string { return m.name }

func (m *streamingMockProvider) ChatCompletion(_ context.Context, req ChatRequest) (*ChatResponse, error) {
	m.calls++
	if req.DeltaSink != nil {
		for _, d := range m.deltas {
			if err := req.DeltaSink(d); err != nil {
				return nil, err
			}
		}
	}
	return m.response, nil
}
