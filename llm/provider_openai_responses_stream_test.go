package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"regexp"
	"strings"
	"sync"
	"testing"
)

// respSSE writes one Responses API stream event and flushes so the client
// consumes it incrementally. Write errors after a client-side abort are
// ignored (the SDK closes the body when a sink error aborts the stream).
func respSSE(w http.ResponseWriter, payload string) {
	if _, err := fmt.Fprintf(w, "data: %s\n\n", payload); err != nil {
		return
	}
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
}

// terminalResponse is a complete Responses API response object shared by the
// streaming and synchronous parity tests: a reasoning item, one text message
// and one function call. It is compacted on first use because SSE data lines
// must be single-line.
var terminalResponse = func() string {
	var b bytes.Buffer
	if err := json.Compact(&b, []byte(`{
	"id": "resp_1",
	"object": "response",
	"status": "completed",
	"output": [
		{"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "thinking"}]},
		{"type": "message", "id": "msg_1", "role": "assistant", "content": [{"type": "output_text", "text": "Hello"}]},
		{"type": "function_call", "id": "fc_1", "call_id": "call-1", "name": "search", "arguments": "{\"q\":\"x\"}"}
	],
	"usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
}`)); err != nil {
		panic(err)
	}
	return b.String()
}()

// writeFullResponsesStream emits a realistic Responses API SSE sequence:
// reasoning summary deltas, output text deltas and function-call argument
// fragments, terminated by response.completed carrying terminalResponse.
func writeFullResponsesStream(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "text/event-stream")
	respSSE(w, `{"type":"response.created","response":{"id":"resp_1"}}`)
	respSSE(w, `{"type":"response.output_item.added","output_index":0,"item":{"type":"reasoning","id":"rs_1"}}`)
	respSSE(w, `{"type":"response.reasoning_summary_text.delta","item_id":"rs_1","output_index":0,"summary_index":0,"delta":"think"}`)
	respSSE(w, `{"type":"response.reasoning_summary_text.delta","item_id":"rs_1","output_index":0,"summary_index":0,"delta":"ing"}`)
	respSSE(w, `{"type":"response.output_item.added","output_index":1,"item":{"type":"message","id":"msg_1"}}`)
	respSSE(w, `{"type":"response.output_text.delta","item_id":"msg_1","output_index":1,"content_index":0,"delta":"Hel"}`)
	respSSE(w, `{"type":"response.output_text.delta","item_id":"msg_1","output_index":1,"content_index":0,"delta":"lo"}`)
	respSSE(w, `{"type":"response.output_item.added","output_index":2,"item":{"type":"function_call","id":"fc_1","call_id":"call-1","name":"search"}}`)
	respSSE(w, `{"type":"response.function_call_arguments.delta","item_id":"fc_1","output_index":2,"delta":"{\"q\":"}`)
	respSSE(w, `{"type":"response.function_call_arguments.delta","item_id":"fc_1","output_index":2,"delta":"\"x\"}"}`)
	respSSE(w, fmt.Sprintf(`{"type":"response.completed","sequence_number":11,"response":%s}`, terminalResponse))
}

// TestOpenAIProvider_ResponsesStream verifies that streamed text, reasoning
// and tool-call data assemble into the same ChatResponse the synchronous path
// would produce, with usage taken from the terminal response.completed event.
func TestOpenAIProvider_ResponsesStream(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		writeFullResponsesStream(w)
	}))
	t.Cleanup(srv.Close)

	p, err := NewOpenAIProvider(OpenAIProviderConfig{Name: "p", APIKey: "k", BaseURL: srv.URL})
	if err != nil {
		t.Fatalf("NewOpenAIProvider: %v", err)
	}

	var deltas []StreamDelta
	resp, err := p.ChatCompletion(context.Background(), ChatRequest{
		Model:     "gpt-5.6",
		Protocol:  ProtocolResponses,
		Messages:  []Message{{Role: "user", Content: "hi"}},
		DeltaSink: func(d StreamDelta) error { deltas = append(deltas, d); return nil },
	})
	if err != nil {
		t.Fatalf("ChatCompletion(responses stream): %v", err)
	}

	if resp.Message.Content != "Hello" {
		t.Errorf("content = %q, want %q", resp.Message.Content, "Hello")
	}
	if resp.Message.ReasoningContent != "thinking" || resp.Reasoning != "thinking" {
		t.Errorf("reasoning = %q/%q, want thinking", resp.Message.ReasoningContent, resp.Reasoning)
	}
	if len(resp.Message.ReasoningItems) != 1 || resp.Message.ReasoningItems[0].ID != "rs_1" {
		t.Errorf("reasoning items = %+v, want one rs_1", resp.Message.ReasoningItems)
	}
	if len(resp.Message.ToolCalls) != 1 || resp.Message.ToolCalls[0].Name != "search" || string(resp.Message.ToolCalls[0].Input) != `{"q":"x"}` {
		t.Errorf("tool calls = %+v, want one search {\"q\":\"x\"}", resp.Message.ToolCalls)
	}
	if resp.Usage.InputTokens != 10 || resp.Usage.OutputTokens != 5 {
		t.Errorf("usage = %+v, want 10/5 from response.completed", resp.Usage)
	}
	if resp.StopReason != "tool_use" {
		t.Errorf("stop reason = %q, want tool_use", resp.StopReason)
	}
	wantDeltas := []StreamDelta{
		{Reasoning: "think"},
		{Reasoning: "ing"},
		{Text: "Hel"},
		{Text: "lo"},
	}
	if !reflect.DeepEqual(deltas, wantDeltas) {
		t.Errorf("deltas = %+v, want %+v", deltas, wantDeltas)
	}
}

// TestOpenAIProvider_ResponsesStream_ParityWithSync verifies the streaming
// AC directly: the ChatResponse assembled from the terminal response.completed
// event is deeply equal to the synchronous path's conversion of the same
// response object.
func TestOpenAIProvider_ResponsesStream_ParityWithSync(t *testing.T) {
	syncSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, terminalResponse)
	}))
	t.Cleanup(syncSrv.Close)
	streamSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		writeFullResponsesStream(w)
	}))
	t.Cleanup(streamSrv.Close)

	syncP, err := NewOpenAIProvider(OpenAIProviderConfig{Name: "p", APIKey: "k", BaseURL: syncSrv.URL})
	if err != nil {
		t.Fatalf("NewOpenAIProvider(sync): %v", err)
	}
	streamP, err := NewOpenAIProvider(OpenAIProviderConfig{Name: "p", APIKey: "k", BaseURL: streamSrv.URL})
	if err != nil {
		t.Fatalf("NewOpenAIProvider(stream): %v", err)
	}

	syncResp, err := syncP.ChatCompletion(context.Background(), ChatRequest{
		Model:    "gpt-5.6",
		Protocol: ProtocolResponses,
		Messages: []Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("ChatCompletion(sync): %v", err)
	}
	streamResp, err := streamP.ChatCompletion(context.Background(), ChatRequest{
		Model:    "gpt-5.6",
		Protocol: ProtocolResponses,
		Messages: []Message{{Role: "user", Content: "hi"}},
		DeltaSink: func(StreamDelta) error {
			return nil
		},
	})
	if err != nil {
		t.Fatalf("ChatCompletion(stream): %v", err)
	}
	if !reflect.DeepEqual(syncResp, streamResp) {
		t.Errorf("streaming response drifts from the synchronous result:\nsync:   %+v\nstream: %+v", syncResp, streamResp)
	}
}

// TestOpenAIProvider_ResponsesStream_Incomplete verifies that a terminal
// response.incomplete event maps to the same stop reason as the synchronous
// path (max_tokens for the max_output_tokens reason).
func TestOpenAIProvider_ResponsesStream_Incomplete(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		respSSE(w, `{"type":"response.output_text.delta","item_id":"msg_1","output_index":0,"content_index":0,"delta":"part"}`)
		respSSE(w, `{"type":"response.incomplete","response":{"id":"resp_2","object":"response","status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"output":[{"type":"message","id":"msg_1","role":"assistant","content":[{"type":"output_text","text":"part"}]}],"usage":{"input_tokens":3,"output_tokens":4}}}`)
	}))
	t.Cleanup(srv.Close)

	p, _ := NewOpenAIProvider(OpenAIProviderConfig{Name: "p", APIKey: "k", BaseURL: srv.URL})

	var deltas []StreamDelta
	resp, err := p.ChatCompletion(context.Background(), ChatRequest{
		Model:     "gpt-5.6",
		Protocol:  ProtocolResponses,
		Messages:  []Message{{Role: "user", Content: "hi"}},
		DeltaSink: func(d StreamDelta) error { deltas = append(deltas, d); return nil },
	})
	if err != nil {
		t.Fatalf("ChatCompletion(responses stream): %v", err)
	}
	if resp.StopReason != "max_tokens" {
		t.Errorf("stop reason = %q, want max_tokens", resp.StopReason)
	}
	if resp.Message.Content != "part" || resp.Usage.InputTokens != 3 || resp.Usage.OutputTokens != 4 {
		t.Errorf("response = %+v, want content part / usage 3-4", resp)
	}
	if len(deltas) != 1 || deltas[0].Text != "part" {
		t.Errorf("deltas = %+v, want [part]", deltas)
	}
}

// TestOpenAIProvider_ResponsesStream_Failed verifies that a terminal
// response.failed event aborts with an error carrying the failure code and
// message (the streaming analogue of the synchronous path's HTTP-level
// failures) instead of returning a degenerate empty response.
func TestOpenAIProvider_ResponsesStream_Failed(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		respSSE(w, `{"type":"response.output_text.delta","item_id":"msg_1","output_index":0,"content_index":0,"delta":"par"}`)
		respSSE(w, `{"type":"response.failed","response":{"id":"resp_3","object":"response","status":"failed","error":{"code":"server_error","message":"the model overloaded"}}}`)
	}))
	t.Cleanup(srv.Close)

	p, _ := NewOpenAIProvider(OpenAIProviderConfig{Name: "p", APIKey: "k", BaseURL: srv.URL})

	_, err := p.ChatCompletion(context.Background(), ChatRequest{
		Model:     "gpt-5.6",
		Protocol:  ProtocolResponses,
		Messages:  []Message{{Role: "user", Content: "hi"}},
		DeltaSink: func(StreamDelta) error { return nil },
	})
	if err == nil {
		t.Fatal("expected an error from a response.failed stream")
	}
	var llmErr *Error
	if !errors.As(err, &llmErr) || llmErr.Provider != "p" {
		t.Errorf("err = %v, want a provider error from p", err)
	}
	if !strings.Contains(err.Error(), "server_error") || !strings.Contains(err.Error(), "the model overloaded") {
		t.Errorf("err = %v, want the failure code and message", err)
	}
}

// TestOpenAIProvider_ResponsesStream_ErrorEvent verifies that a mid-stream
// `error` event aborts with the event's code and message.
func TestOpenAIProvider_ResponsesStream_ErrorEvent(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		respSSE(w, `{"type":"response.created","response":{"id":"resp_4"}}`)
		respSSE(w, `{"type":"error","code":"rate_limit_exceeded","message":"too many requests"}`)
	}))
	t.Cleanup(srv.Close)

	p, _ := NewOpenAIProvider(OpenAIProviderConfig{Name: "p", APIKey: "k", BaseURL: srv.URL})

	_, err := p.ChatCompletion(context.Background(), ChatRequest{
		Model:     "gpt-5.6",
		Protocol:  ProtocolResponses,
		Messages:  []Message{{Role: "user", Content: "hi"}},
		DeltaSink: func(StreamDelta) error { return nil },
	})
	if err == nil {
		t.Fatal("expected an error from an error event stream")
	}
	if !strings.Contains(err.Error(), "too many requests") {
		t.Errorf("err = %v, want the event message", err)
	}
}

// TestOpenAIProvider_ResponsesStream_SinkErrorAborts verifies that an error
// from the delta sink aborts the stream and is surfaced to the caller
// unchanged (host-side cancellation).
func TestOpenAIProvider_ResponsesStream_SinkErrorAborts(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		respSSE(w, `{"type":"response.output_text.delta","item_id":"msg_1","output_index":0,"content_index":0,"delta":"x"}`)
		respSSE(w, `{"type":"response.output_text.delta","item_id":"msg_1","output_index":0,"content_index":0,"delta":"y"}`)
		respSSE(w, fmt.Sprintf(`{"type":"response.completed","response":%s}`, terminalResponse))
	}))
	t.Cleanup(srv.Close)

	p, _ := NewOpenAIProvider(OpenAIProviderConfig{Name: "p", APIKey: "k", BaseURL: srv.URL})

	boom := errors.New("host cancelled")
	_, err := p.ChatCompletion(context.Background(), ChatRequest{
		Model:     "gpt-5.6",
		Protocol:  ProtocolResponses,
		Messages:  []Message{{Role: "user", Content: "hi"}},
		DeltaSink: func(StreamDelta) error { return boom },
	})
	if !errors.Is(err, boom) {
		t.Fatalf("err = %v, want the sink error", err)
	}
}

// TestOpenAIProvider_ResponsesStream_RequireStreamingAndEncryptedRoundTrip
// verifies the RequireStreaming seam on the Responses path: every call goes
// out with stream:true, store:false and include reasoning.encrypted_content
// (even without a DeltaSink — the endpoint demands streaming on the wire),
// the encrypted reasoning payload returned by the backend lands in
// ReasoningItem.EncryptedContent, and it is sent back verbatim on the next
// request so a stateless backend can reconstruct the reasoning chain.
func TestOpenAIProvider_ResponsesStream_RequireStreamingAndEncryptedRoundTrip(t *testing.T) {
	encryptedResponse := func() string {
		var b bytes.Buffer
		if err := json.Compact(&b, []byte(`{
		"id": "resp_5",
		"object": "response",
		"status": "completed",
		"output": [
			{"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "ENC-PAYLOAD"},
			{"type": "message", "id": "msg_1", "role": "assistant", "content": [{"type": "output_text", "text": "Hi"}]}
		],
		"usage": {"input_tokens": 7, "output_tokens": 2}
	}`)); err != nil {
			panic(err)
		}
		return b.String()
	}()

	var mu sync.Mutex
	var bodies []map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read request body: %v", err)
			return
		}
		var parsed map[string]any
		if err := json.Unmarshal(body, &parsed); err != nil {
			t.Errorf("parse request body: %v", err)
			return
		}
		mu.Lock()
		bodies = append(bodies, parsed)
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		respSSE(w, `{"type":"response.output_text.delta","item_id":"msg_1","output_index":1,"content_index":0,"delta":"Hi"}`)
		respSSE(w, fmt.Sprintf(`{"type":"response.completed","response":%s}`, encryptedResponse))
	}))
	t.Cleanup(srv.Close)

	p, err := NewOpenAIProvider(OpenAIProviderConfig{Name: "p", APIKey: "k", BaseURL: srv.URL, RequireStreaming: true})
	if err != nil {
		t.Fatalf("NewOpenAIProvider: %v", err)
	}

	// First call: no DeltaSink — RequireStreaming alone selects the streaming
	// wire path and must pin store=false + reasoning.encrypted_content.
	resp, err := p.ChatCompletion(context.Background(), ChatRequest{
		Model:    "gpt-5.6",
		Protocol: ProtocolResponses,
		Messages: []Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("ChatCompletion(requireStreaming): %v", err)
	}
	if resp.Message.Content != "Hi" {
		t.Errorf("content = %q, want Hi", resp.Message.Content)
	}
	if len(resp.Message.ReasoningItems) != 1 || resp.Message.ReasoningItems[0].EncryptedContent != "ENC-PAYLOAD" {
		t.Errorf("reasoning items = %+v, want one carrying ENC-PAYLOAD", resp.Message.ReasoningItems)
	}

	mu.Lock()
	first := bodies[0]
	mu.Unlock()
	if first["stream"] != true {
		t.Errorf("stream = %v, want true", first["stream"])
	}
	if first["store"] != false {
		t.Errorf("store = %v, want false", first["store"])
	}
	include, _ := first["include"].([]any)
	foundEncrypted := false
	for _, inc := range include {
		if inc == "reasoning.encrypted_content" {
			foundEncrypted = true
		}
	}
	if !foundEncrypted {
		t.Errorf("include = %v, want reasoning.encrypted_content", first["include"])
	}

	// Second call: send the reasoning item back — the encrypted payload must
	// travel verbatim in the request input.
	_, err = p.ChatCompletion(context.Background(), ChatRequest{
		Model:    "gpt-5.6",
		Protocol: ProtocolResponses,
		Messages: []Message{
			{Role: "user", Content: "hi"},
			{Role: "assistant", ReasoningItems: resp.Message.ReasoningItems, Content: "Hi"},
		},
	})
	if err != nil {
		t.Fatalf("ChatCompletion(round-trip): %v", err)
	}

	mu.Lock()
	raw := bodies[len(bodies)-1]
	mu.Unlock()
	input, _ := raw["input"].([]any)
	foundRoundTrip := false
	for _, item := range input {
		m, _ := item.(map[string]any)
		if m != nil && m["type"] == "reasoning" && m["encrypted_content"] == "ENC-PAYLOAD" && m["id"] == "rs_1" {
			foundRoundTrip = true
		}
	}
	if !foundRoundTrip {
		t.Errorf("second request input = %v, want a reasoning item carrying encrypted_content ENC-PAYLOAD", raw["input"])
	}
}

// TestOpenAIProvider_ResponsesStream_RequireStreamingDropsOutputCap verifies
// that a requireStreaming (Codex-style) endpoint never receives
// max_output_tokens — such backends reject the parameter outright with
// HTTP 400 "Unsupported parameter" — while an ordinary endpoint still gets
// the cap when the caller sets MaxTokens.
func TestOpenAIProvider_ResponsesStream_RequireStreamingDropsOutputCap(t *testing.T) {
	recordBody := func() (*[]map[string]any, *httptest.Server) {
		var mu sync.Mutex
		var bodies []map[string]any
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, err := io.ReadAll(r.Body)
			if err != nil {
				t.Errorf("read request body: %v", err)
				return
			}
			var parsed map[string]any
			if err := json.Unmarshal(body, &parsed); err != nil {
				t.Errorf("parse request body: %v", err)
				return
			}
			mu.Lock()
			bodies = append(bodies, parsed)
			mu.Unlock()
			w.Header().Set("Content-Type", "text/event-stream")
			respSSE(w, `{"type":"response.completed","response":{"id":"resp_cap","object":"response","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1}}}`)
		}))
		t.Cleanup(srv.Close)
		return &bodies, srv
	}

	bodies, srv := recordBody()
	p, err := NewOpenAIProvider(OpenAIProviderConfig{Name: "codex", APIKey: "k", BaseURL: srv.URL, RequireStreaming: true})
	if err != nil {
		t.Fatalf("NewOpenAIProvider: %v", err)
	}
	if _, err := p.ChatCompletion(context.Background(), ChatRequest{
		Model:     "gpt-5.6",
		Protocol:  ProtocolResponses,
		MaxTokens: 30,
		Messages:  []Message{{Role: "user", Content: "hi"}},
	}); err != nil {
		t.Fatalf("ChatCompletion(requireStreaming, MaxTokens=30): %v", err)
	}
	if len(*bodies) != 1 {
		t.Fatalf("recorded %d request bodies, want 1", len(*bodies))
	}
	if v, present := (*bodies)[0]["max_output_tokens"]; present {
		t.Errorf("max_output_tokens = %v, want absent on a requireStreaming endpoint", v)
	}

	// Control: an ordinary endpoint (DeltaSink selects the same streaming
	// wire path) still receives the cap.
	controlBodies, controlSrv := recordBody()
	p2, err := NewOpenAIProvider(OpenAIProviderConfig{Name: "p", APIKey: "k", BaseURL: controlSrv.URL})
	if err != nil {
		t.Fatalf("NewOpenAIProvider(control): %v", err)
	}
	if _, err := p2.ChatCompletion(context.Background(), ChatRequest{
		Model:     "gpt-5.6",
		Protocol:  ProtocolResponses,
		MaxTokens: 30,
		Messages:  []Message{{Role: "user", Content: "hi"}},
		DeltaSink: func(StreamDelta) error { return nil },
	}); err != nil {
		t.Fatalf("ChatCompletion(control, MaxTokens=30): %v", err)
	}
	if len(*controlBodies) != 1 {
		t.Fatalf("recorded %d control bodies, want 1", len(*controlBodies))
	}
	if got := (*controlBodies)[0]["max_output_tokens"]; got != float64(30) {
		t.Errorf("max_output_tokens = %v, want 30 on an ordinary endpoint", got)
	}
}

// TestOpenAIProvider_ResponsesStream_DeltaOnlyCodexBackendAssemblesResponse
// reproduces the ChatGPT OAuth Codex shape observed in production: the
// backend streams the full completion (reasoning summary deltas, output text
// deltas, completed output items via response.output_item.done) but delivers
// response.completed with an EMPTY output list while billing the output
// tokens. Assembling from the terminal event alone then yields an empty
// Message — hosts lose one-shot results entirely. The accumulated stream
// must be merged in as a fallback: text from the deltas, reasoning summaries
// from the deltas, and reasoning items (with their EncryptedContent
// round-trip payload) plus function calls from output_item.done. No
// DeltaSink is set — this is the one-shot caller shape.
func TestOpenAIProvider_ResponsesStream_DeltaOnlyCodexBackendAssemblesResponse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		respSSE(w, `{"type":"response.created","response":{"id":"resp_codex"}}`)
		respSSE(w, `{"type":"response.output_item.added","output_index":0,"item":{"type":"reasoning","id":"rs_9"}}`)
		respSSE(w, `{"type":"response.reasoning_summary_text.delta","item_id":"rs_9","output_index":0,"summary_index":0,"delta":"thin"}`)
		respSSE(w, `{"type":"response.reasoning_summary_text.delta","item_id":"rs_9","output_index":0,"summary_index":0,"delta":"king"}`)
		respSSE(w, `{"type":"response.output_item.done","output_index":0,"item":{"type":"reasoning","id":"rs_9","summary":[{"type":"summary_text","text":"thinking"}],"encrypted_content":"ENC-CODEX"}}`)
		respSSE(w, `{"type":"response.output_item.added","output_index":1,"item":{"type":"message","id":"msg_9"}}`)
		respSSE(w, `{"type":"response.output_text.delta","item_id":"msg_9","output_index":1,"content_index":0,"delta":"{\"route\":"}`)
		respSSE(w, `{"type":"response.output_text.delta","item_id":"msg_9","output_index":1,"content_index":0,"delta":"\"chat\"}"}`)
		respSSE(w, `{"type":"response.output_item.done","output_index":1,"item":{"type":"message","id":"msg_9","role":"assistant","status":"completed","content":[{"type":"output_text","text":"{\"route\":\"chat\"}"}]}}`)
		respSSE(w, `{"type":"response.output_item.added","output_index":2,"item":{"type":"function_call","id":"fc_9","call_id":"call-9","name":"search"}}`)
		respSSE(w, `{"type":"response.function_call_arguments.delta","item_id":"fc_9","output_index":2,"delta":"{\"q\":"}`)
		respSSE(w, `{"type":"response.function_call_arguments.delta","item_id":"fc_9","output_index":2,"delta":"\"x\"}"}`)
		respSSE(w, `{"type":"response.output_item.done","output_index":2,"item":{"type":"function_call","id":"fc_9","call_id":"call-9","name":"search","arguments":"{\"q\":\"x\"}"}}`)
		respSSE(w, `{"type":"response.completed","response":{"id":"resp_codex","object":"response","status":"completed","output":[],"usage":{"input_tokens":9,"output_tokens":7}}}`)
	}))
	t.Cleanup(srv.Close)

	p, err := NewOpenAIProvider(OpenAIProviderConfig{Name: "codex", APIKey: "k", BaseURL: srv.URL, RequireStreaming: true})
	if err != nil {
		t.Fatalf("NewOpenAIProvider: %v", err)
	}

	resp, err := p.ChatCompletion(context.Background(), ChatRequest{
		Model:    "gpt-6.1-sol",
		Protocol: ProtocolResponses,
		Messages: []Message{{Role: "user", Content: "route me"}},
	})
	if err != nil {
		t.Fatalf("ChatCompletion(delta-only codex): %v", err)
	}

	if resp.Message.Content != `{"route":"chat"}` {
		t.Errorf("content = %q, want the accumulated delta text %q", resp.Message.Content, `{"route":"chat"}`)
	}
	if resp.Message.ReasoningContent != "thinking" || resp.Reasoning != "thinking" {
		t.Errorf("reasoning = %q/%q, want thinking from the summary deltas", resp.Message.ReasoningContent, resp.Reasoning)
	}
	if len(resp.Message.ReasoningItems) != 1 {
		t.Fatalf("reasoning items = %+v, want exactly one from output_item.done", resp.Message.ReasoningItems)
	}
	if ri := resp.Message.ReasoningItems[0]; ri.ID != "rs_9" || ri.EncryptedContent != "ENC-CODEX" || ri.Summary != "thinking" {
		t.Errorf("reasoning item = %+v, want rs_9 / ENC-CODEX / thinking", ri)
	}
	if len(resp.Message.ToolCalls) != 1 {
		t.Fatalf("tool calls = %+v, want exactly one from output_item.done", resp.Message.ToolCalls)
	}
	if tc := resp.Message.ToolCalls[0]; tc.ID != "call-9" || tc.Name != "search" || string(tc.Input) != `{"q":"x"}` {
		t.Errorf("tool call = %+v, want call-9 / search / {\"q\":\"x\"}", tc)
	}
	if resp.StopReason != "tool_use" {
		t.Errorf("stop reason = %q, want tool_use once a function call was merged in", resp.StopReason)
	}
	if resp.Usage.InputTokens != 9 || resp.Usage.OutputTokens != 7 {
		t.Errorf("usage = %+v, want 9/7 from the terminal event", resp.Usage)
	}
}

// TestOpenAIProvider_ResponsesStream_TerminalOutputWinsOverStreamedFallback
// verifies the merge is strictly fallback: when the terminal event carries a
// full output list (the official OpenAI shape), the accumulated deltas and
// output_item.done items must NOT override or duplicate it.
func TestOpenAIProvider_ResponsesStream_TerminalOutputWinsOverStreamedFallback(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		// Deltas and items disagree with the terminal payload on purpose.
		respSSE(w, `{"type":"response.output_text.delta","item_id":"msg_1","output_index":0,"content_index":0,"delta":"delta text"}`)
		respSSE(w, `{"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"msg_1","role":"assistant","content":[{"type":"output_text","text":"item done text"}]}}`)
		respSSE(w, `{"type":"response.output_item.done","output_index":1,"item":{"type":"function_call","id":"fc_1","call_id":"call-1","name":"search","arguments":"{\"q\":\"dup\"}"}}`)
		respSSE(w, fmt.Sprintf(`{"type":"response.completed","response":%s}`, terminalResponse))
	}))
	t.Cleanup(srv.Close)

	p, _ := NewOpenAIProvider(OpenAIProviderConfig{Name: "p", APIKey: "k", BaseURL: srv.URL})

	resp, err := p.ChatCompletion(context.Background(), ChatRequest{
		Model:     "gpt-5.6",
		Protocol:  ProtocolResponses,
		Messages:  []Message{{Role: "user", Content: "hi"}},
		DeltaSink: func(StreamDelta) error { return nil },
	})
	if err != nil {
		t.Fatalf("ChatCompletion(terminal wins): %v", err)
	}
	if resp.Message.Content != "Hello" {
		t.Errorf("content = %q, want Hello from the terminal response (not overridden by deltas/items)", resp.Message.Content)
	}
	if len(resp.Message.ToolCalls) != 1 {
		t.Errorf("tool calls = %+v, want exactly one (call-1 de-duplicated, not doubled)", resp.Message.ToolCalls)
	}
	if string(resp.Message.ToolCalls[0].Input) != `{"q":"x"}` {
		t.Errorf("tool call input = %s, want the terminal {\"q\":\"x\"} (not the item.done duplicate)", resp.Message.ToolCalls[0].Input)
	}
}

// TestOpenAIProvider_ResponsesStream_ItemDoneOnlyBackendAssemblesResponse
// covers a backend that emits NO deltas at all — only response.output_item.done
// items and a terminal response with an empty output list: the merged content
// and reasoning then come from the completed items.
func TestOpenAIProvider_ResponsesStream_ItemDoneOnlyBackendAssemblesResponse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		respSSE(w, `{"type":"response.output_item.done","output_index":0,"item":{"type":"reasoning","id":"rs_d","summary":[{"type":"summary_text","text":"quiet thinking"}]}}`)
		respSSE(w, `{"type":"response.output_item.done","output_index":1,"item":{"type":"message","id":"msg_d","role":"assistant","content":[{"type":"output_text","text":"Quiet reply"}]}}`)
		respSSE(w, `{"type":"response.completed","response":{"id":"resp_d","object":"response","status":"completed","output":[],"usage":{"input_tokens":2,"output_tokens":2}}}`)
	}))
	t.Cleanup(srv.Close)

	p, _ := NewOpenAIProvider(OpenAIProviderConfig{Name: "p", APIKey: "k", BaseURL: srv.URL, RequireStreaming: true})

	resp, err := p.ChatCompletion(context.Background(), ChatRequest{
		Model:    "gpt-5.6",
		Protocol: ProtocolResponses,
		Messages: []Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("ChatCompletion(item.done-only): %v", err)
	}
	if resp.Message.Content != "Quiet reply" {
		t.Errorf("content = %q, want Quiet reply from the output_item.done message", resp.Message.Content)
	}
	if resp.Message.ReasoningContent != "quiet thinking" || resp.Reasoning != "quiet thinking" {
		t.Errorf("reasoning = %q/%q, want quiet thinking from the output_item.done reasoning summary", resp.Message.ReasoningContent, resp.Reasoning)
	}
	if resp.StopReason != "end_turn" {
		t.Errorf("stop reason = %q, want end_turn (no function calls anywhere)", resp.StopReason)
	}
}

// responsesFunctionNameRe mirrors the backend-side validation for Responses
// API input function_call names (documented pattern plus the 64-char cap
// shared with tool definitions); the sanitizing-server test uses it to reject
// the request exactly like the production Codex backend does.
var responsesFunctionNameRe = regexp.MustCompile(`^[a-zA-Z0-9_-]+$`)

// TestOpenAIProvider_ResponsesStream_RequireStreamingClampsMinimalEffort
// verifies that a requireStreaming (Codex-style) endpoint never receives
// reasoning.effort "minimal" — such backends accept only the low..max ladder
// and answer "minimal" with HTTP 400 "'minimal' is not supported" — while an
// ordinary endpoint still receives the caller's value verbatim.
func TestOpenAIProvider_ResponsesStream_RequireStreamingClampsMinimalEffort(t *testing.T) {
	recordBody := func() (*[]map[string]any, *httptest.Server) {
		var mu sync.Mutex
		var bodies []map[string]any
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, err := io.ReadAll(r.Body)
			if err != nil {
				t.Errorf("read request body: %v", err)
				return
			}
			var parsed map[string]any
			if err := json.Unmarshal(body, &parsed); err != nil {
				t.Errorf("parse request body: %v", err)
				return
			}
			mu.Lock()
			bodies = append(bodies, parsed)
			mu.Unlock()
			w.Header().Set("Content-Type", "text/event-stream")
			respSSE(w, `{"type":"response.completed","response":{"id":"resp_eff","object":"response","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1}}}`)
		}))
		t.Cleanup(srv.Close)
		return &bodies, srv
	}

	bodies, srv := recordBody()
	p, err := NewOpenAIProvider(OpenAIProviderConfig{Name: "codex", APIKey: "k", BaseURL: srv.URL, RequireStreaming: true})
	if err != nil {
		t.Fatalf("NewOpenAIProvider: %v", err)
	}
	if _, err := p.ChatCompletion(context.Background(), ChatRequest{
		Model:           "gpt-6.1-sol",
		Protocol:        ProtocolResponses,
		ReasoningEffort: "minimal",
		Messages:        []Message{{Role: "user", Content: "hi"}},
	}); err != nil {
		t.Fatalf("ChatCompletion(requireStreaming, effort=minimal): %v", err)
	}
	if len(*bodies) != 1 {
		t.Fatalf("recorded %d request bodies, want 1", len(*bodies))
	}
	if got := reasoningEffortOf(t, (*bodies)[0]); got != "low" {
		t.Errorf("reasoning.effort = %q, want \"low\" (clamped from minimal) on a requireStreaming endpoint", got)
	}

	// Control: an ordinary endpoint receives the caller's effort verbatim.
	controlBodies, controlSrv := recordBody()
	p2, err := NewOpenAIProvider(OpenAIProviderConfig{Name: "p", APIKey: "k", BaseURL: controlSrv.URL})
	if err != nil {
		t.Fatalf("NewOpenAIProvider(control): %v", err)
	}
	if _, err := p2.ChatCompletion(context.Background(), ChatRequest{
		Model:           "gpt-5.6",
		Protocol:        ProtocolResponses,
		ReasoningEffort: "minimal",
		Messages:        []Message{{Role: "user", Content: "hi"}},
		DeltaSink:       func(StreamDelta) error { return nil },
	}); err != nil {
		t.Fatalf("ChatCompletion(control, effort=minimal): %v", err)
	}
	if len(*controlBodies) != 1 {
		t.Fatalf("recorded %d control bodies, want 1", len(*controlBodies))
	}
	if got := reasoningEffortOf(t, (*controlBodies)[0]); got != "minimal" {
		t.Errorf("reasoning.effort = %q, want \"minimal\" verbatim on an ordinary endpoint", got)
	}
}

// reasoningEffortOf extracts body.reasoning.effort from a recorded wire
// request ("" when reasoning is absent).
func reasoningEffortOf(t *testing.T, body map[string]any) string {
	t.Helper()
	reasoning, _ := body["reasoning"].(map[string]any)
	if reasoning == nil {
		return ""
	}
	effort, _ := reasoning["effort"].(string)
	return effort
}

// TestOpenAIProvider_ResponsesStream_SanitizesHallucinatedFunctionCallNames
// reproduces the production failure: the model hallucinated tool names
// ("functions.get_me") whose failed results ("tool not found") were recorded
// in history; re-sending that history as input then failed the backend's
// ^[a-zA-Z0-9_-]+$ pattern check with a retryable=false HTTP 400, poisoning
// every subsequent request of the conversation. The converter must sanitize
// the wire names (invalid runes → '_', 64-char cap) while passing
// already-valid names through unchanged and preserving the call_id pairing.
// The fake server enforces the pattern itself, so the request would be
// rejected outright without the fix.
func TestOpenAIProvider_ResponsesStream_SanitizesHallucinatedFunctionCallNames(t *testing.T) {
	var mu sync.Mutex
	var bodies []map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read request body: %v", err)
			return
		}
		var parsed map[string]any
		if err := json.Unmarshal(body, &parsed); err != nil {
			t.Errorf("parse request body: %v", err)
			return
		}
		mu.Lock()
		bodies = append(bodies, parsed)
		mu.Unlock()
		input, _ := parsed["input"].([]any)
		for _, item := range input {
			m, _ := item.(map[string]any)
			if m == nil || m["type"] != "function_call" {
				continue
			}
			name, _ := m["name"].(string)
			if !responsesFunctionNameRe.MatchString(name) || len(name) > 64 {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusBadRequest)
				_, _ = fmt.Fprintf(w, `{"error":{"message":"Invalid input function_call name %q","type":"invalid_request_error"}}`, name)
				return
			}
		}
		w.Header().Set("Content-Type", "text/event-stream")
		respSSE(w, `{"type":"response.completed","response":{"id":"resp_fn","object":"response","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1}}}`)
	}))
	t.Cleanup(srv.Close)

	p, err := NewOpenAIProvider(OpenAIProviderConfig{Name: "codex", APIKey: "k", BaseURL: srv.URL, RequireStreaming: true})
	if err != nil {
		t.Fatalf("NewOpenAIProvider: %v", err)
	}

	resp, err := p.ChatCompletion(context.Background(), ChatRequest{
		Model:    "gpt-6.1-sol",
		Protocol: ProtocolResponses,
		Messages: []Message{
			{Role: "user", Content: "list my projects"},
			{Role: "assistant", ToolCalls: []ToolCall{
				{ID: "call-bad", Name: "functions.get_me", Input: []byte(`{}`)},
				{ID: "call-good", Name: "list_projects", Input: []byte(`{}`)},
				{ID: "call-long", Name: strings.Repeat("t", 70), Input: []byte(`{}`)},
			}},
			{Role: "tool", ToolCallID: "call-bad", Content: "tool not found: functions.get_me"},
			{Role: "tool", ToolCallID: "call-good", Content: `{"projects":[]}`},
			{Role: "tool", ToolCallID: "call-long", Content: "tool not found"},
			{Role: "user", Content: "continue"},
		},
	})
	if err != nil {
		t.Fatalf("ChatCompletion(sanitized history): %v", err)
	}
	if resp == nil {
		t.Fatal("resp = nil, want an assembled response")
	}

	mu.Lock()
	body := bodies[len(bodies)-1]
	mu.Unlock()
	type fnCall struct {
		name   string
		callID string
	}
	var got []fnCall
	input, _ := body["input"].([]any)
	for _, item := range input {
		m, _ := item.(map[string]any)
		if m == nil || m["type"] != "function_call" {
			continue
		}
		name, _ := m["name"].(string)
		callID, _ := m["call_id"].(string)
		got = append(got, fnCall{name: name, callID: callID})
	}
	want := []fnCall{
		{name: "functions_get_me", callID: "call-bad"},
		{name: "list_projects", callID: "call-good"},
		{name: strings.Repeat("t", 64), callID: "call-long"},
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("function_call input items = %+v, want %+v", got, want)
	}
}
