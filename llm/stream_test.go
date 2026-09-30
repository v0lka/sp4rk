package llm

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"
)

// streamMockProvider is a Provider that can emit deltas through the optional
// ChatRequest.DeltaSink hook, letting tests exercise the router's streaming
// path without touching the network.
type streamMockProvider struct {
	name  string
	calls int
	// run is invoked once per ChatCompletion; it may call req.DeltaSink.
	run func(req ChatRequest) (*ChatResponse, error)
}

func (m *streamMockProvider) Name() string { return m.name }

func (m *streamMockProvider) ChatCompletion(_ context.Context, req ChatRequest) (*ChatResponse, error) {
	m.calls++
	if m.run != nil {
		return m.run(req)
	}
	return &ChatResponse{
		Message:    Message{Role: "assistant", Content: "ok"},
		StopReason: "end_turn",
	}, nil
}

func newStreamTestRouter(p Provider) *Router {
	return &Router{
		providers:          map[string]Provider{"primary": p},
		activeProvider:     p,
		activeModel:        "primary/test-model",
		activeBareModel:    "test-model",
		activeProviderName: "primary",
		maxRetries:         2,
		initialBackoff:     time.Millisecond,
		maxBackoff:         2 * time.Millisecond,
	}
}

// TestRouter_Call_ForwardsDeltas verifies that deltas emitted by the provider
// through req.DeltaSink reach the caller's sink in order, and the assembled
// response is returned unchanged.
func TestRouter_Call_ForwardsDeltas(t *testing.T) {
	p := &streamMockProvider{
		name: "streamer",
		run: func(req ChatRequest) (*ChatResponse, error) {
			if req.DeltaSink == nil {
				t.Fatal("provider did not receive a delta sink")
			}
			for _, part := range []string{"Hel", "lo ", "world"} {
				if err := req.DeltaSink(StreamDelta{Text: part}); err != nil {
					return nil, err
				}
			}
			return &ChatResponse{
				Message:    Message{Role: "assistant", Content: "Hello world"},
				StopReason: "end_turn",
			}, nil
		},
	}
	router := newStreamTestRouter(p)

	var got strings.Builder
	resp, err := router.Call(context.Background(), ChatRequest{
		Messages:  []Message{{Role: "user", Content: "hi"}},
		DeltaSink: func(d StreamDelta) error { got.WriteString(d.Text); return nil },
	})
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	if got.String() != "Hello world" {
		t.Errorf("streamed text = %q, want %q", got.String(), "Hello world")
	}
	if resp == nil || resp.Message.Content != "Hello world" {
		t.Errorf("response content = %+v, want %q", resp, "Hello world")
	}
}

// TestRouter_Call_NoRetryAfterPartialStream verifies that a retryable failure
// after at least one delta was emitted is NOT retried — replaying the stream
// would duplicate text the host already rendered.
func TestRouter_Call_NoRetryAfterPartialStream(t *testing.T) {
	p := &streamMockProvider{
		name: "streamer",
		run: func(req ChatRequest) (*ChatResponse, error) {
			_ = req.DeltaSink(StreamDelta{Text: "partial"})
			return nil, NewError("streamer", 503, true, errors.New("upstream reset"))
		},
	}
	router := newStreamTestRouter(p)

	_, err := router.Call(context.Background(), ChatRequest{
		Messages:  []Message{{Role: "user", Content: "hi"}},
		DeltaSink: func(StreamDelta) error { return nil },
	})
	if err == nil {
		t.Fatal("expected error after partial stream")
	}
	if p.calls != 1 {
		t.Errorf("provider calls = %d, want 1 (no retry after partial delivery)", p.calls)
	}
}

// TestRouter_Call_RetriesWhenNoDeltaEmitted verifies that a retryable failure
// BEFORE any delta was emitted is retried normally, and the successful attempt
// streams through.
func TestRouter_Call_RetriesWhenNoDeltaEmitted(t *testing.T) {
	p := &streamMockProvider{name: "streamer"}
	p.run = func(req ChatRequest) (*ChatResponse, error) {
		if p.calls == 1 {
			return nil, NewError("streamer", 503, true, errors.New("cold start"))
		}
		_ = req.DeltaSink(StreamDelta{Text: "recovered"})
		return &ChatResponse{
			Message:    Message{Role: "assistant", Content: "recovered"},
			StopReason: "end_turn",
		}, nil
	}
	router := newStreamTestRouter(p)

	var got strings.Builder
	_, err := router.Call(context.Background(), ChatRequest{
		Messages:  []Message{{Role: "user", Content: "hi"}},
		DeltaSink: func(d StreamDelta) error { got.WriteString(d.Text); return nil },
	})
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	if p.calls != 2 {
		t.Errorf("provider calls = %d, want 2 (retried before any delta)", p.calls)
	}
	if got.String() != "recovered" {
		t.Errorf("streamed text = %q, want %q", got.String(), "recovered")
	}
}

// TestChatRequest_DeltaSinkNotSerialized guards the json:"-" contract: a request
// carrying a delta sink must still marshal (llm_dump.go marshals ChatRequest),
// and the hook must never appear on the wire.
func TestChatRequest_DeltaSinkNotSerialized(t *testing.T) {
	raw, err := json.Marshal(ChatRequest{
		Model:     "gpt-4o",
		Messages:  []Message{{Role: "user", Content: "hi"}},
		DeltaSink: func(StreamDelta) error { return nil },
	})
	if err != nil {
		t.Fatalf("json.Marshal(ChatRequest with DeltaSink) failed: %v", err)
	}
	if strings.Contains(string(raw), "DeltaSink") || strings.Contains(string(raw), "delta_sink") {
		t.Errorf("DeltaSink leaked into JSON: %s", raw)
	}
}
