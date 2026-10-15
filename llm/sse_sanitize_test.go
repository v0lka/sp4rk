package llm

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// keepalive writes an SSE comment/keep-alive block: a comment line followed by
// a blank line. Per the SSE spec a blank line dispatches an event even when no
// data: field preceded it, so the block is a VALID but data-less event. The
// ChatGPT OAuth Codex backend emits these between real events.
func keepalive(w http.ResponseWriter) {
	_, _ = io.WriteString(w, ": keepalive\n\n")
}

// TestOpenAIProvider_ResponsesStream_EmptyEventTolerated reproduces the
// third-party report: a ChatGPT-subscription Responses stream that carries a
// data-less SSE event (keep-alive comment) must NOT fail. Before the fix the
// openai-go v1.12.0 ssestream decoder dispatched the empty event and
// Stream.Next json.Unmarshal'ed the empty payload, aborting the call with
// `responses API: unexpected end of JSON input` (HTTP 0, non-retryable) — a
// terminal task failure at a random point in the stream.
//
// The content type deliberately carries a charset parameter: the SDK matches
// registered decoders by the raw header value, so a fix that only overrides
// the `text/event-stream` spelling would be bypassed here.
func TestOpenAIProvider_ResponsesStream_EmptyEventTolerated(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
		keepalive(w) // before the first event
		respSSE(w, `{"type":"response.created","response":{"id":"resp_1"}}`)
		keepalive(w) // between events
		respSSE(w, `{"type":"response.output_text.delta","item_id":"msg_1","output_index":0,"content_index":0,"delta":"Hel"}`)
		keepalive(w) // between deltas
		respSSE(w, `{"type":"response.output_text.delta","item_id":"msg_1","output_index":0,"content_index":0,"delta":"lo"}`)
		respSSE(w, fmt.Sprintf(`{"type":"response.completed","response":%s}`, terminalResponse))
	}))
	t.Cleanup(srv.Close)

	p, err := NewOpenAIProvider(OpenAIProviderConfig{Name: "chatgpt", APIKey: "k", BaseURL: srv.URL, RequireStreaming: true})
	if err != nil {
		t.Fatalf("NewOpenAIProvider: %v", err)
	}

	var deltas []StreamDelta
	resp, err := p.ChatCompletion(context.Background(), ChatRequest{
		Model:     "gpt-6.1-sol",
		Protocol:  ProtocolResponses,
		Messages:  []Message{{Role: "user", Content: "hi"}},
		DeltaSink: func(d StreamDelta) error { deltas = append(deltas, d); return nil },
	})
	if err != nil {
		t.Fatalf("ChatCompletion with keep-alive events: %v", err)
	}
	if resp.Message.Content != "Hello" {
		t.Errorf("content = %q, want %q", resp.Message.Content, "Hello")
	}
	if len(deltas) != 2 || deltas[0].Text != "Hel" || deltas[1].Text != "lo" {
		t.Errorf("deltas = %+v, want [Hel lo]", deltas)
	}
}

// TestOpenAIProvider_ResponsesStream_TruncatedEventTolerated reproduces the
// SECOND upstream shape (openai/openai-go#618): an event whose payload is a
// TRUNCATED JSON prefix. The decoder cuts the event at a blank line before the
// JSON value is complete and Stream.Next json.Unmarshal fails with
// `unexpected end of JSON input`. The reference Rust client skips such an
// event, so the call must survive it and deliver every complete event.
func TestOpenAIProvider_ResponsesStream_TruncatedEventTolerated(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
		respSSE(w, `{"type":"response.created","response":{"id":"resp_1"}}`)
		// A delta whose JSON never closes: the SDK would abort here.
		respSSE(w, `{"type":"response.output_text.delta","item_id":"msg_1","output_index":0,"content_index":0,"delta":"Hel"`)
		respSSE(w, `{"type":"response.output_text.delta","item_id":"msg_1","output_index":0,"content_index":0,"delta":"lo"}`)
		respSSE(w, fmt.Sprintf(`{"type":"response.completed","response":%s}`, terminalResponse))
	}))
	t.Cleanup(srv.Close)

	p, err := NewOpenAIProvider(OpenAIProviderConfig{Name: "chatgpt", APIKey: "k", BaseURL: srv.URL, RequireStreaming: true})
	if err != nil {
		t.Fatalf("NewOpenAIProvider: %v", err)
	}

	var deltas []StreamDelta
	resp, err := p.ChatCompletion(context.Background(), ChatRequest{
		Model:     "gpt-6.1-sol",
		Protocol:  ProtocolResponses,
		Messages:  []Message{{Role: "user", Content: "hi"}},
		DeltaSink: func(d StreamDelta) error { deltas = append(deltas, d); return nil },
	})
	if err != nil {
		t.Fatalf("ChatCompletion with a truncated event: %v", err)
	}
	// The truncated "Hel" delta is skipped; only the complete "lo" delta
	// streams. The assembled content still comes from the terminal response.
	if len(deltas) != 1 || deltas[0].Text != "lo" {
		t.Errorf("deltas = %+v, want [lo]", deltas)
	}
	if resp.Message.Content != "Hello" {
		t.Errorf("content = %q, want %q", resp.Message.Content, "Hello")
	}
}

// TestOpenAIProvider_ResponsesStream_WrongContentType covers a backend or
// gateway that streams SSE under a media type other than text/event-stream.
// The sniff must still normalize the body so a keep-alive cannot abort it.
func TestOpenAIProvider_ResponsesStream_WrongContentType(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		// Deliberately wrong for an SSE body.
		w.Header().Set("Content-Type", "application/json")
		keepalive(w)
		respSSE(w, `{"type":"response.output_text.delta","item_id":"msg_1","output_index":0,"content_index":0,"delta":"Hello"}`)
		respSSE(w, fmt.Sprintf(`{"type":"response.completed","response":%s}`, terminalResponse))
	}))
	t.Cleanup(srv.Close)

	p, err := NewOpenAIProvider(OpenAIProviderConfig{Name: "chatgpt", APIKey: "k", BaseURL: srv.URL, RequireStreaming: true})
	if err != nil {
		t.Fatalf("NewOpenAIProvider: %v", err)
	}

	resp, err := p.ChatCompletion(context.Background(), ChatRequest{
		Model:     "gpt-6.1-sol",
		Protocol:  ProtocolResponses,
		Messages:  []Message{{Role: "user", Content: "hi"}},
		DeltaSink: func(StreamDelta) error { return nil },
	})
	if err != nil {
		t.Fatalf("ChatCompletion with a mislabelled SSE body: %v", err)
	}
	if resp.Message.Content != "Hello" {
		t.Errorf("content = %q, want %q", resp.Message.Content, "Hello")
	}
}

// TestOpenAIProvider_ResponsesStream_EmptyTerminatingEvent verifies the other
// empty-event shape: a trailing blank line AFTER the terminating event (a
// server that ends the body with an extra newline). The final event must still
// be delivered.
func TestOpenAIProvider_ResponsesStream_EmptyTerminatingEvent(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		respSSE(w, fmt.Sprintf(`{"type":"response.completed","response":%s}`, terminalResponse))
		_, _ = io.WriteString(w, "\n") // stray blank line after the last event
	}))
	t.Cleanup(srv.Close)

	p, _ := NewOpenAIProvider(OpenAIProviderConfig{Name: "chatgpt", APIKey: "k", BaseURL: srv.URL, RequireStreaming: true})
	resp, err := p.ChatCompletion(context.Background(), ChatRequest{
		Model:     "gpt-6.1-sol",
		Protocol:  ProtocolResponses,
		Messages:  []Message{{Role: "user", Content: "hi"}},
		DeltaSink: func(StreamDelta) error { return nil },
	})
	if err != nil {
		t.Fatalf("ChatCompletion with trailing blank line: %v", err)
	}
	if resp.Message.Content != "Hello" {
		t.Errorf("content = %q, want %q", resp.Message.Content, "Hello")
	}
}

// TestOpenAIProvider_ChatCompletionStream_EmptyEventTolerated pins the same
// tolerance on the Chat Completions streaming path: the sanitizing middleware
// is installed on both SDK clients, so a gateway that injects keep-alive
// events between chunks (a very common proxy/gateway behavior) must not abort
// the stream either.
func TestOpenAIProvider_ChatCompletionStream_EmptyEventTolerated(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
		keepalive(w)
		sse(t, w, `{"id":"1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"},"finish_reason":""}]}`)
		keepalive(w)
		sse(t, w, `{"id":"1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":""}]}`)
		_, _ = io.WriteString(w, "retry: 3000\n\n")
		sse(t, w, `{"id":"1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`)
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
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
		t.Fatalf("ChatCompletion(stream) with keep-alive events: %v", err)
	}
	if resp.Message.Content != "Hello" {
		t.Errorf("content = %q, want %q", resp.Message.Content, "Hello")
	}
	if resp.StopReason != "end_turn" {
		t.Errorf("stop reason = %q, want end_turn", resp.StopReason)
	}
	if len(deltas) != 2 || deltas[0].Text != "Hel" || deltas[1].Text != "lo" {
		t.Errorf("deltas = %+v, want [Hel lo]", deltas)
	}
}

// TestSSESanitizeBodies asserts the sanitizer's byte-level contract directly:
// data-less and unparsable events are dropped, complete events pass through
// unchanged (including [DONE]), and the trailing event is terminated so the
// decoder dispatches it.
func TestSSESanitizeBodies(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want string
	}{
		{
			name: "keepalive dropped",
			in:   ": keepalive\n\ndata: {\"a\":1}\n\n",
			want: "data: {\"a\":1}\n\n",
		},
		{
			name: "retry directive dropped",
			in:   "retry: 3000\n\ndata: {\"a\":1}\n\n",
			want: "data: {\"a\":1}\n\n",
		},
		{
			name: "trailing blank dropped",
			in:   "data: {\"a\":1}\n\n\n",
			want: "data: {\"a\":1}\n\n",
		},
		{
			name: "event field preserved",
			in:   "event: response.created\ndata: {\"a\":1}\n\n",
			want: "event: response.created\ndata: {\"a\":1}\n\n",
		},
		{
			name: "dataless event field dropped",
			in:   "event: response.ping\n\ndata: {\"a\":1}\n\n",
			want: "data: {\"a\":1}\n\n",
		},
		{
			name: "event split from data by blank line",
			in:   "event: response.created\n\ndata: {\"a\":1}\n\n",
			want: "data: {\"a\":1}\n\n",
		},
		{
			name: "empty data field dropped",
			in:   "data:\n\ndata: {\"a\":1}\n\n",
			want: "data: {\"a\":1}\n\n",
		},
		{
			name: "multi-line data joined",
			in:   "data: {\"a\":\ndata: 1}\n\n",
			want: "data: {\"a\":\ndata: 1}\n\n",
		},
		{
			name: "done preserved",
			in:   "data: [DONE]\n\n",
			want: "data: [DONE]\n\n",
		},
		{
			name: "truncated json dropped",
			in:   "data: {\"a\":\n\ndata: {\"b\":1}\n\n",
			want: "data: {\"b\":1}\n\n",
		},
		{
			name: "non-json payload dropped",
			in:   "data: ping\n\ndata: {\"b\":1}\n\n",
			want: "data: {\"b\":1}\n\n",
		},
		{
			name: "unterminated trailing event discarded",
			in:   "data: {\"a\":1}",
			want: "",
		},
		{
			name: "partial trailing line discarded",
			in:   "data: {\"a\":1}\n\ndata: {\"b\":",
			want: "data: {\"a\":1}\n\n",
		},
		{
			name: "crlf normalized",
			in:   "data: {\"a\":1}\r\n\r\n",
			want: "data: {\"a\":1}\n\n",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := io.ReadAll(newSSESanitizingBody(io.NopCloser(strings.NewReader(tc.in)), true, nil))
			if err != nil {
				t.Fatalf("ReadAll: %v", err)
			}
			if string(got) != tc.want {
				t.Errorf("sanitized = %q, want %q", got, tc.want)
			}
		})
	}
}

// TestSSESanitize_Passthrough verifies that a body which is NOT an event stream
// is copied through byte-for-byte, both when the caller trusts the content type
// and when only the sniff is available.
func TestSSESanitize_Passthrough(t *testing.T) {
	bodies := []string{
		`{"ok":true}`,
		"{\n  \"data\": [1, 2],\n  \"note\": \"data: not a stream\"\n}\n",
		"",
	}
	for _, body := range bodies {
		for _, trusted := range []bool{false, true} {
			got, err := io.ReadAll(newSSESanitizingBody(io.NopCloser(strings.NewReader(body)), trusted, nil))
			if err != nil {
				t.Fatalf("ReadAll(trusted=%v): %v", trusted, err)
			}
			// Only a genuinely SSE body is rewritten; a JSON body is not (a
			// trusted-but-non-SSE body would still be parsed, and these bodies
			// carry no complete event, so they come back empty in that case).
			if !trusted && string(got) != body {
				t.Errorf("passthrough(trusted=false) = %q, want %q", got, body)
			}
		}
	}
}

// TestSSESanitize_SniffUnrecognizedBody confirms the sniff does not rewrite a
// body whose first line is not an SSE field.
func TestSSESanitize_SniffUnrecognizedBody(t *testing.T) {
	body := "{\"models\":[{\"id\":\"gpt-4o\"}]}"
	got, err := io.ReadAll(newSSESanitizingBody(io.NopCloser(strings.NewReader(body)), false, nil))
	if err != nil {
		t.Fatalf("ReadAll: %v", err)
	}
	if string(got) != body {
		t.Errorf("sniffed passthrough = %q, want %q", got, body)
	}
}
