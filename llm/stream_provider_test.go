package llm

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
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

// TestOpenAIProvider_ChatCompletionStream_ReasoningDeltas verifies that
// non-standard reasoning_content deltas (DeepSeek et al.) are forwarded as
// Reasoning deltas and assembled into the response — the streaming counterpart
// of the synchronous path's reasoning parity line (ADR-009).
func TestOpenAIProvider_ChatCompletionStream_ReasoningDeltas(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		sse(t, w, `{"id":"1","object":"chat.completion.chunk","created":1,"model":"deepseek-r1","choices":[{"index":0,"delta":{"reasoning_content":"think"},"finish_reason":""}]}`)
		sse(t, w, `{"id":"1","object":"chat.completion.chunk","created":1,"model":"deepseek-r1","choices":[{"index":0,"delta":{"content":"ans"},"finish_reason":""}]}`)
		sse(t, w, `{"id":"1","object":"chat.completion.chunk","created":1,"model":"deepseek-r1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`)
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	t.Cleanup(srv.Close)

	p, err := NewOpenAIProvider(OpenAIProviderConfig{Name: "p", APIKey: "k", BaseURL: srv.URL})
	if err != nil {
		t.Fatalf("NewOpenAIProvider: %v", err)
	}

	var deltas []StreamDelta
	resp, err := p.ChatCompletion(context.Background(), ChatRequest{
		Model:     "deepseek-r1",
		Messages:  []Message{{Role: "user", Content: "hi"}},
		DeltaSink: func(d StreamDelta) error { deltas = append(deltas, d); return nil },
	})
	if err != nil {
		t.Fatalf("ChatCompletion(stream): %v", err)
	}

	if resp.Message.ReasoningContent != "think" {
		t.Errorf("Message.ReasoningContent = %q, want %q", resp.Message.ReasoningContent, "think")
	}
	if resp.Reasoning != "think" {
		t.Errorf("Reasoning = %q, want %q", resp.Reasoning, "think")
	}
	if resp.Message.Content != "ans" {
		t.Errorf("content = %q, want %q", resp.Message.Content, "ans")
	}
	if len(deltas) != 2 || deltas[0].Reasoning != "think" || deltas[0].Text != "" || deltas[1].Text != "ans" {
		t.Errorf("deltas = %+v, want [Reasoning think, Text ans]", deltas)
	}
}

// TestOpenAIProvider_ChatCompletionStream_IgnoresNonZeroChoices verifies that a
// multi-choice stream (n > 1, non-standard gateways) contributes choice 0 only,
// so the assembled response matches what the synchronous path returns from
// Choices[0] instead of interleaving every choice into one string.
func TestOpenAIProvider_ChatCompletionStream_IgnoresNonZeroChoices(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		sse(t, w, `{"id":"1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"A"},"finish_reason":""}]}`)
		sse(t, w, `{"id":"1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":1,"delta":{"content":"X"},"finish_reason":""}]}`)
		sse(t, w, `{"id":"1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"B"},"finish_reason":"stop"}]}`)
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

	if resp.Message.Content != "AB" {
		t.Errorf("content = %q, want %q (choice 1 interleaved)", resp.Message.Content, "AB")
	}
	if len(deltas) != 2 || deltas[0].Text != "A" || deltas[1].Text != "B" {
		t.Errorf("deltas = %+v, want [A, B]", deltas)
	}
}

// TestOpenAIProvider_ChatCompletionStream_StreamOptionsUnsupported verifies the
// one-time retry without stream_options: OpenAI-compatible servers that do not
// know the field (older vLLM, certain Azure api-versions, proxies) answer it
// with an HTTP 400, and enabling streaming must stay safe on them — the second
// attempt omits stream_options and succeeds (usage then reports zeros).
func TestOpenAIProvider_ChatCompletionStream_StreamOptionsUnsupported(t *testing.T) {
	var mu sync.Mutex
	var asksUsage []bool // one entry per request: did the body carry stream_options?
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read request body: %v", err)
			return
		}
		asked := strings.Contains(string(body), "stream_options")
		mu.Lock()
		asksUsage = append(asksUsage, asked)
		mu.Unlock()
		if asked {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			_, _ = fmt.Fprint(w, `{"error":{"message":"Unknown parameter: 'stream_options'","type":"invalid_request_error","param":"stream_options","code":"unknown_parameter"}}`)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		sse(t, w, `{"id":"1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":""}]}`)
		sse(t, w, `{"id":"1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`)
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	t.Cleanup(srv.Close)

	p, err := NewOpenAIProvider(OpenAIProviderConfig{Name: "p", APIKey: "k", BaseURL: srv.URL})
	if err != nil {
		t.Fatalf("NewOpenAIProvider: %v", err)
	}

	resp, err := p.ChatCompletion(context.Background(), ChatRequest{
		Model:     "gpt-4o",
		Messages:  []Message{{Role: "user", Content: "hi"}},
		DeltaSink: func(StreamDelta) error { return nil },
	})
	if err != nil {
		t.Fatalf("ChatCompletion(stream) after stream_options fallback: %v", err)
	}
	if resp.Message.Content != "ok" {
		t.Errorf("content = %q, want %q", resp.Message.Content, "ok")
	}
	mu.Lock()
	defer mu.Unlock()
	if len(asksUsage) != 2 || !asksUsage[0] || asksUsage[1] {
		t.Errorf("request shapes = %v, want [asked stream_options, omitted stream_options]", asksUsage)
	}
}
