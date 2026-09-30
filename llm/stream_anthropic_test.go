package llm

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
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
