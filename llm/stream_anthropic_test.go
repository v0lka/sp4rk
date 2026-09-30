package llm

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// sseAnthropic writes one Anthropic SSE frame (event + data + blank line) and
// flushes it.
func sseAnthropic(t *testing.T, w http.ResponseWriter, event, payload string) {
	t.Helper()
	if _, err := fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, payload); err != nil {
		t.Fatalf("write SSE: %v", err)
	}
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
}

// TestAnthropicProvider_ChatCompletionStream verifies that streamed text deltas
// are forwarded to req.DeltaSink and the assembled response (content, usage,
// stop reason) matches the synchronous path.
func TestAnthropicProvider_ChatCompletionStream(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		sseAnthropic(t, w, "message_start",
			`{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-3-haiku-20240307","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":12,"output_tokens":0}}}`)
		sseAnthropic(t, w, "content_block_start",
			`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`)
		sseAnthropic(t, w, "content_block_delta",
			`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hel"}}`)
		sseAnthropic(t, w, "content_block_delta",
			`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"lo"}}`)
		sseAnthropic(t, w, "content_block_stop",
			`{"type":"content_block_stop","index":0}`)
		sseAnthropic(t, w, "message_delta",
			`{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":5}}`)
		sseAnthropic(t, w, "message_stop", `{"type":"message_stop"}`)
	}))
	t.Cleanup(srv.Close)

	p, err := NewAnthropicProvider(AnthropicProviderConfig{APIKey: "k", BaseURL: srv.URL})
	if err != nil {
		t.Fatalf("NewAnthropicProvider: %v", err)
	}

	var deltas []StreamDelta
	resp, err := p.ChatCompletion(context.Background(), ChatRequest{
		Model:     "claude-3-haiku-20240307",
		MaxTokens: 16,
		Messages:  []Message{{Role: "user", Content: "hi"}},
		DeltaSink: func(d StreamDelta) error { deltas = append(deltas, d); return nil },
	})
	if err != nil {
		t.Fatalf("ChatCompletion(stream): %v", err)
	}

	if resp.Message.Content != "Hello" {
		t.Errorf("content = %q, want %q", resp.Message.Content, "Hello")
	}
	if resp.Usage.InputTokens != 12 || resp.Usage.OutputTokens != 5 {
		t.Errorf("usage = %+v, want 12/5", resp.Usage)
	}
	if resp.StopReason != "end_turn" {
		t.Errorf("stop reason = %q, want end_turn", resp.StopReason)
	}
	if len(deltas) != 2 || deltas[0].Text != "Hel" || deltas[1].Text != "lo" {
		t.Errorf("deltas = %+v, want [Hel, lo]", deltas)
	}
}

// TestAnthropicProvider_ChatCompletionStream_ReasoningDeltas verifies that
// thinking deltas (extended thinking) are forwarded as Reasoning deltas and the
// assembled response carries the thinking block — the streaming counterpart of
// the synchronous path's reasoning parity line (ADR-009).
func TestAnthropicProvider_ChatCompletionStream_ReasoningDeltas(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		sseAnthropic(t, w, "message_start",
			`{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-3-7-sonnet-latest","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":12,"output_tokens":0}}}`)
		sseAnthropic(t, w, "content_block_start",
			`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}`)
		sseAnthropic(t, w, "content_block_delta",
			`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"step"}}`)
		sseAnthropic(t, w, "content_block_stop",
			`{"type":"content_block_stop","index":0}`)
		sseAnthropic(t, w, "content_block_start",
			`{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`)
		sseAnthropic(t, w, "content_block_delta",
			`{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"ans"}}`)
		sseAnthropic(t, w, "content_block_stop",
			`{"type":"content_block_stop","index":1}`)
		sseAnthropic(t, w, "message_delta",
			`{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":7}}`)
		sseAnthropic(t, w, "message_stop", `{"type":"message_stop"}`)
	}))
	t.Cleanup(srv.Close)

	p, err := NewAnthropicProvider(AnthropicProviderConfig{APIKey: "k", BaseURL: srv.URL})
	if err != nil {
		t.Fatalf("NewAnthropicProvider: %v", err)
	}

	var deltas []StreamDelta
	resp, err := p.ChatCompletion(context.Background(), ChatRequest{
		Model:     "claude-3-7-sonnet-latest",
		MaxTokens: 64,
		Messages:  []Message{{Role: "user", Content: "hi"}},
		DeltaSink: func(d StreamDelta) error { deltas = append(deltas, d); return nil },
	})
	if err != nil {
		t.Fatalf("ChatCompletion(stream): %v", err)
	}

	if resp.Reasoning != "step" {
		t.Errorf("Reasoning = %q, want %q", resp.Reasoning, "step")
	}
	if resp.Message.Content != "ans" {
		t.Errorf("content = %q, want %q", resp.Message.Content, "ans")
	}
	if resp.StopReason != "end_turn" {
		t.Errorf("stop reason = %q, want end_turn", resp.StopReason)
	}
	if resp.Usage.InputTokens != 12 || resp.Usage.OutputTokens != 7 {
		t.Errorf("usage = %+v, want 12/7", resp.Usage)
	}
	if len(deltas) != 2 || deltas[0].Reasoning != "step" || deltas[0].Text != "" || deltas[1].Text != "ans" {
		t.Errorf("deltas = %+v, want [Reasoning step, Text ans]", deltas)
	}
}

// TestAnthropicProvider_ChatCompletionStream_SinkErrorAborts verifies that an
// error from the delta sink aborts the stream: the provider cancels the request
// instead of reading the endpoint's output to EOF, and the sink error — not the
// resulting transport failure — is returned to the caller.
func TestAnthropicProvider_ChatCompletionStream_SinkErrorAborts(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		// write tolerates write errors: once the sink error cancels the
		// request, the server observes a broken pipe — the expected abort
		// signal — and must keep failing silently instead of the test.
		write := func(event, payload string) {
			_, _ = fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, payload)
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
		}
		w.Header().Set("Content-Type", "text/event-stream")
		write("message_start",
			`{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-3-haiku-20240307","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":12,"output_tokens":0}}}`)
		write("content_block_start",
			`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`)
		write("content_block_delta",
			`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hel"}}`)
		// Plenty of follow-up output that must never reach the (already
		// failed) sink — the stream is cancelled on the sink error.
		for i := 0; i < 64; i++ {
			write("content_block_delta",
				`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"more"}}`)
		}
		write("message_delta",
			`{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":5}}`)
		write("message_stop", `{"type":"message_stop"}`)
	}))
	t.Cleanup(srv.Close)

	p, err := NewAnthropicProvider(AnthropicProviderConfig{APIKey: "k", BaseURL: srv.URL})
	if err != nil {
		t.Fatalf("NewAnthropicProvider: %v", err)
	}

	boom := errors.New("host cancelled")
	_, err = p.ChatCompletion(context.Background(), ChatRequest{
		Model:     "claude-3-haiku-20240307",
		MaxTokens: 16,
		Messages:  []Message{{Role: "user", Content: "hi"}},
		DeltaSink: func(StreamDelta) error { return boom },
	})
	if !errors.Is(err, boom) {
		t.Fatalf("err = %v, want the sink error", err)
	}
}

// TestAnthropicProvider_ChatCompletionStream_LiveDelivery verifies that deltas
// reach the sink while the response is still open: the capturing transport must
// tee the SSE body through instead of buffering it, so a live-typing host sees
// the first chunk before the endpoint has finished generating (ADR-009).
func TestAnthropicProvider_ChatCompletionStream_LiveDelivery(t *testing.T) {
	firstDelta := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		sseAnthropic(t, w, "message_start",
			`{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-3-haiku-20240307","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":12,"output_tokens":0}}}`)
		sseAnthropic(t, w, "content_block_start",
			`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`)
		sseAnthropic(t, w, "content_block_delta",
			`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hel"}}`)
		// Hold the stream open until the client proves it received the first
		// delta. A buffered transport would only invoke the sink at EOF, so
		// this wait times out and fails the test instead of hanging.
		select {
		case <-firstDelta:
		case <-time.After(5 * time.Second):
			t.Error("first delta was not delivered while the stream was still open: the response body was buffered by the transport")
			return
		}
		sseAnthropic(t, w, "content_block_delta",
			`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"lo"}}`)
		sseAnthropic(t, w, "content_block_stop", `{"type":"content_block_stop","index":0}`)
		sseAnthropic(t, w, "message_delta",
			`{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":5}}`)
		sseAnthropic(t, w, "message_stop", `{"type":"message_stop"}`)
	}))
	t.Cleanup(srv.Close)

	p, err := NewAnthropicProvider(AnthropicProviderConfig{APIKey: "k", BaseURL: srv.URL})
	if err != nil {
		t.Fatalf("NewAnthropicProvider: %v", err)
	}

	var deltas []StreamDelta
	resp, err := p.ChatCompletion(context.Background(), ChatRequest{
		Model:     "claude-3-haiku-20240307",
		MaxTokens: 16,
		Messages:  []Message{{Role: "user", Content: "hi"}},
		DeltaSink: func(d StreamDelta) error {
			deltas = append(deltas, d)
			if len(deltas) == 1 {
				close(firstDelta)
			}
			return nil
		},
	})
	if err != nil {
		t.Fatalf("ChatCompletion(stream): %v", err)
	}
	if resp.Message.Content != "Hello" {
		t.Errorf("content = %q, want %q", resp.Message.Content, "Hello")
	}
	if len(deltas) != 2 || deltas[0].Text != "Hel" || deltas[1].Text != "lo" {
		t.Errorf("deltas = %+v, want [Hel, lo]", deltas)
	}
}
