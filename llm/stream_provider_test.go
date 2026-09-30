package llm

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
)

// sse writes one Server-Sent Event data line and flushes so the client sees it
// incrementally rather than at stream close.
func sse(t *testing.T, w http.ResponseWriter, payload string) {
	t.Helper()
	if _, err := fmt.Fprintf(w, "data: %s\n\n", payload); err != nil {
		t.Fatalf("write SSE: %v", err)
	}
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
}

// TestOpenAIProvider_ChatCompletionStream verifies that streamed text, tool-call
// fragments and usage are forwarded/assembled into the same ChatResponse the
// synchronous path would produce.
func TestOpenAIProvider_ChatCompletionStream(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		sse(t, w, `{"id":"1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"},"finish_reason":""}]}`)
		sse(t, w, `{"id":"1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":""}]}`)
		sse(t, w, `{"id":"1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call-1","type":"function","function":{"name":"search","arguments":"{\"q\":"}}]},"finish_reason":""}]}`)
		sse(t, w, `{"id":"1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"x\"}"}}]},"finish_reason":""}]}`)
		sse(t, w, `{"id":"1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`)
		sse(t, w, `{"id":"1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`)
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	t.Cleanup(srv.Close)

	p, err := NewOpenAIProvider(OpenAIProviderConfig{Name: "p", APIKey: "k", BaseURL: srv.URL})
	if err != nil {
		t.Fatalf("NewOpenAIProvider: %v", err)
	}

	var deltas []StreamDelta
	resp, err := p.ChatCompletion(context.Background(), ChatRequest{
		Model:     "gpt-4o",
		Messages:  []Message{{Role: "user", Content: "hi"}},
		DeltaSink: func(d StreamDelta) error { deltas = append(deltas, d); return nil },
	})
	if err != nil {
		t.Fatalf("ChatCompletion(stream): %v", err)
	}

	if resp.Message.Content != "Hello" {
		t.Errorf("content = %q, want %q", resp.Message.Content, "Hello")
	}
	if len(resp.Message.ToolCalls) != 1 || resp.Message.ToolCalls[0].Name != "search" || string(resp.Message.ToolCalls[0].Input) != `{"q":"x"}` {
		t.Errorf("tool calls = %+v, want one search {\"q\":\"x\"}", resp.Message.ToolCalls)
	}
	if resp.Usage.InputTokens != 10 || resp.Usage.OutputTokens != 5 {
		t.Errorf("usage = %+v, want 10/5", resp.Usage)
	}
	if resp.StopReason != "tool_use" {
		t.Errorf("stop reason = %q, want tool_use", resp.StopReason)
	}
	if len(deltas) != 2 || deltas[0].Text != "Hel" || deltas[1].Text != "lo" {
		t.Errorf("deltas = %+v, want [Hel, lo]", deltas)
	}
}

// TestOpenAIProvider_ChatCompletionStream_SinkErrorAborts verifies that an error
// from the delta sink aborts the stream and is surfaced to the caller.
func TestOpenAIProvider_ChatCompletionStream_SinkErrorAborts(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		sse(t, w, `{"id":"1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"x"},"finish_reason":""}]}`)
		sse(t, w, `{"id":"1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"y"},"finish_reason":"stop"}]}`)
	}))
	t.Cleanup(srv.Close)

	p, _ := NewOpenAIProvider(OpenAIProviderConfig{Name: "p", APIKey: "k", BaseURL: srv.URL})

	boom := errors.New("host cancelled")
	_, err := p.ChatCompletion(context.Background(), ChatRequest{
		Model:     "gpt-4o",
		Messages:  []Message{{Role: "user", Content: "hi"}},
		DeltaSink: func(StreamDelta) error { return boom },
	})
	if !errors.Is(err, boom) {
		t.Fatalf("err = %v, want the sink error", err)
	}
}
