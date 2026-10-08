package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/liushuangls/go-anthropic/v2"
)

// Tests in this file are wire-level regression tests for the Anthropic
// provider review findings: the bounded response-body capture (#28), the
// extended-thinking gate on tool-use continuations (#45), the dropped
// reasoning-only assistant message (#54), the default input_schema for
// schema-less tools (#57), and the cache-token usage summation (#59).

// anthropicOKBodyText returns a minimal, well-formed Anthropic Messages
// success body with the given first text block and usage JSON object.
func anthropicOKBodyText(text, usage string) string {
	return `{"id":"msg_test","type":"message","role":"assistant",` +
		`"model":"claude-3-haiku-20240307",` +
		`"content":[{"type":"text","text":` + quoteJSONString(text) + `}],` +
		`"stop_reason":"end_turn","stop_sequence":null,` +
		`"usage":` + usage + `}`
}

// quoteJSONString encodes s as a JSON string literal (including quotes).
func quoteJSONString(s string) string {
	b, err := json.Marshal(s)
	if err != nil {
		// json.Marshal of a string never fails; stay total for tests.
		return `""`
	}
	return string(b)
}

// newAnthropicWireProvider starts an httptest server that records the raw
// request body of every call and answers with the given response body. It
// returns a provider wired to the server and a function returning the last
// recorded request body.
func newAnthropicWireProvider(t *testing.T, responseBody string) (provider *AnthropicProvider, lastBody func() string) {
	t.Helper()
	var last []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, "failed to read request body", http.StatusInternalServerError)
			return
		}
		last = raw
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(responseBody))
	}))
	t.Cleanup(srv.Close)

	p, err := NewAnthropicProvider(AnthropicProviderConfig{
		APIKey:  "test-key",
		BaseURL: srv.URL,
	})
	if err != nil {
		t.Fatalf("NewAnthropicProvider: %v", err)
	}
	return p, func() string { return string(last) }
}

// decodeAnthropicRequest unmarshals a recorded request body into a generic map.
func decodeAnthropicRequest(t *testing.T, body string) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal([]byte(body), &m); err != nil {
		t.Fatalf("failed to decode request body as JSON: %v\nbody: %s", err, body)
	}
	return m
}

// TestCapturedBody_CappedAtMax verifies that capturedBody.write retains only
// the first maxCapturedBodyBytes bytes of the response (review finding #28:
// the capture used to be an unbounded bytes.Buffer, so a hostile or
// misbehaving endpoint streaming a very large or never-ending body grew the
// heap without bound while only a 2 KiB prefix is ever consumed by
// truncateForError).
func TestCapturedBody_CappedAtMax(t *testing.T) {
	chunk := bytes.Repeat([]byte{0x61}, 4096) // 'a' * 4096

	t.Run("capture stops growing at the cap", func(t *testing.T) {
		c := &capturedBody{}
		writes := (5 * maxCapturedBodyBytes) / len(chunk)
		for i := 0; i < writes; i++ {
			c.write(chunk)
		}
		if got := c.bytes(); len(got) != maxCapturedBodyBytes {
			t.Fatalf("captured %d bytes after %d writes, want the %d-byte cap", len(got), writes, maxCapturedBodyBytes)
		}
	})

	t.Run("straddling write is truncated to fit", func(t *testing.T) {
		full := bytes.Repeat([]byte{0x61}, maxCapturedBodyBytes)
		c := &capturedBody{}
		c.write(full[:maxCapturedBodyBytes-10])
		c.write(chunk) // only 10 more bytes fit below the cap
		if got := c.bytes(); len(got) != maxCapturedBodyBytes {
			t.Fatalf("straddling write produced %d bytes, want %d", len(got), maxCapturedBodyBytes)
		}
	})

	t.Run("post-cap writes are dropped", func(t *testing.T) {
		c := &capturedBody{}
		c.write(bytes.Repeat([]byte{0x62}, maxCapturedBodyBytes))
		c.write([]byte("tail that must be dropped"))
		got := c.bytes()
		if len(got) != maxCapturedBodyBytes {
			t.Fatalf("post-cap write changed capture size to %d, want %d", len(got), maxCapturedBodyBytes)
		}
		if bytes.Contains(got, []byte("tail")) {
			t.Error("post-cap bytes leaked into the capture")
		}
	})
}

// TestAnthropicProvider_LargeBodyCapture verifies the capture cap end to end:
// a success body larger than the cap is still fully decoded by the SDK (the
// capture is a passive tee and never truncates what the SDK reads), and a
// 200-with-error-object body larger than the cap still yields a descriptive
// error embedding the head of the body.
func TestAnthropicProvider_LargeBodyCapture(t *testing.T) {
	ctx := context.Background()

	t.Run("large success body is fully parsed", func(t *testing.T) {
		big := strings.Repeat("a", 3*maxCapturedBodyBytes)
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(anthropicOKBodyText(big, `{"input_tokens":1,"output_tokens":1}`)))
		}))
		defer srv.Close()

		provider, err := NewAnthropicProvider(AnthropicProviderConfig{APIKey: "k", BaseURL: srv.URL})
		if err != nil {
			t.Fatalf("NewAnthropicProvider: %v", err)
		}
		resp, err := provider.ChatCompletion(ctx, ChatRequest{
			Model:     "claude-3-haiku-20240307",
			MaxTokens: 64,
			Messages:  []Message{{Role: "user", Content: "hi"}},
		})
		if err != nil {
			t.Fatalf("ChatCompletion: %v", err)
		}
		if resp.Message.Content != big {
			t.Errorf("content length = %d, want %d (body larger than the capture cap must still be fully decoded)", len(resp.Message.Content), len(big))
		}
	})

	t.Run("large 200 error body reports the head", func(t *testing.T) {
		pad := strings.Repeat("P", 5*maxCapturedBodyBytes)
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"type":"error","error":{"type":"overloaded_error","message":"boom"}}` + pad))
		}))
		defer srv.Close()

		provider, err := NewAnthropicProvider(AnthropicProviderConfig{APIKey: "k", BaseURL: srv.URL})
		if err != nil {
			t.Fatalf("NewAnthropicProvider: %v", err)
		}
		_, err = provider.ChatCompletion(ctx, ChatRequest{
			Model:     "claude-3-haiku-20240307",
			MaxTokens: 64,
			Messages:  []Message{{Role: "user", Content: "hi"}},
		})
		if err == nil {
			t.Fatal("expected an error for a 200 response carrying an error object, got nil")
		}
		if !strings.Contains(err.Error(), "overloaded_error") {
			t.Errorf("error should embed the head of the oversized body, got: %v", err)
		}
	})
}

// TestAnthropicProvider_ThinkingGateOnToolUseContinuation verifies review
// finding #45: when reasoning is requested ("On") but the conversation
// re-sends an assistant turn carrying tool_use blocks, the request must NOT
// enable thinking. The Messages API requires such an assistant message to
// start with a thinking (or redacted_thinking) block — which the provider
// cannot re-emit, because parseResponse collapses it into
// ChatResponse.Reasoning and llm.Message carries no thinking block — and
// rejects the request with HTTP 400 otherwise, deterministically breaking the
// first tool call of every reasoning-enabled run. A fresh conversation (no
// prior assistant tool-use turn) and a text-only assistant history must still
// enable thinking.
func TestAnthropicProvider_ThinkingGateOnToolUseContinuation(t *testing.T) {
	provider, lastBody := newAnthropicWireProvider(t, anthropicOKBodyText("ok", `{"input_tokens":1,"output_tokens":1}`))
	ctx := context.Background()

	t.Run("tool-use continuation disables thinking", func(t *testing.T) {
		_, err := provider.ChatCompletion(ctx, ChatRequest{
			Model:           "claude-3-haiku-20240307",
			MaxTokens:       8192,
			ReasoningEffort: "On",
			Messages: []Message{
				{Role: "user", Content: "run the tool"},
				{Role: "assistant", ToolCalls: []ToolCall{
					{ID: "tc_1", Name: "echo", Input: json.RawMessage(`{}`)},
				}},
				{Role: "tool", ToolCallID: "tc_1", Content: "done"},
			},
		})
		if err != nil {
			t.Fatalf("ChatCompletion: %v", err)
		}
		m := decodeAnthropicRequest(t, lastBody())
		if th, ok := m["thinking"]; ok {
			t.Errorf("thinking must not be enabled on a tool-use continuation request, got %v in %s", th, lastBody())
		}
	})

	t.Run("fresh conversation keeps thinking enabled", func(t *testing.T) {
		_, err := provider.ChatCompletion(ctx, ChatRequest{
			Model:           "claude-3-haiku-20240307",
			MaxTokens:       8192,
			ReasoningEffort: "On",
			Messages:        []Message{{Role: "user", Content: "hello"}},
		})
		if err != nil {
			t.Fatalf("ChatCompletion: %v", err)
		}
		m := decodeAnthropicRequest(t, lastBody())
		th, ok := m["thinking"].(map[string]any)
		if !ok {
			t.Fatalf("expected thinking enabled on a fresh conversation, request was: %s", lastBody())
		}
		if th["type"] != "enabled" {
			t.Errorf("thinking.type = %v, want enabled", th["type"])
		}
		// budget = max_tokens/2 (8192/2), clamped down from the 32000 default.
		if th["budget_tokens"] != float64(4096) {
			t.Errorf("thinking.budget_tokens = %v, want 4096", th["budget_tokens"])
		}
	})

	t.Run("text-only assistant history keeps thinking enabled", func(t *testing.T) {
		_, err := provider.ChatCompletion(ctx, ChatRequest{
			Model:           "claude-3-haiku-20240307",
			MaxTokens:       8192,
			ReasoningEffort: "On",
			Messages: []Message{
				{Role: "user", Content: "hello"},
				{Role: "assistant", Content: "hi there"},
				{Role: "user", Content: "continue"},
			},
		})
		if err != nil {
			t.Fatalf("ChatCompletion: %v", err)
		}
		m := decodeAnthropicRequest(t, lastBody())
		if _, ok := m["thinking"]; !ok {
			t.Errorf("thinking must stay enabled when the re-sent assistant turn carries no tool_use blocks: %s", lastBody())
		}
	})
}

// TestAnthropicProvider_ReasoningOnlyAssistantDropped verifies review finding
// #54: an assistant message carrying only ReasoningContent has no renderable
// block — the assistant branch emits text and tool_use blocks only — so
// sending it would marshal a nil Content as "content": null and the API would
// reject the whole request with HTTP 400. The guard used to keep such
// messages alive; they must be dropped instead.
func TestAnthropicProvider_ReasoningOnlyAssistantDropped(t *testing.T) {
	provider, lastBody := newAnthropicWireProvider(t, anthropicOKBodyText("ok", `{"input_tokens":1,"output_tokens":1}`))

	_, err := provider.ChatCompletion(context.Background(), ChatRequest{
		Model:     "claude-3-haiku-20240307",
		MaxTokens: 64,
		Messages: []Message{
			{Role: "user", Content: "one"},
			{Role: "assistant", ReasoningContent: "internal reasoning only"},
			{Role: "user", Content: "two"},
		},
	})
	if err != nil {
		t.Fatalf("ChatCompletion: %v", err)
	}
	raw := lastBody()
	if strings.Contains(raw, `"content":null`) || strings.Contains(raw, `"content": null`) {
		t.Errorf("request must not contain a null content, was: %s", raw)
	}
	m := decodeAnthropicRequest(t, raw)
	msgs, ok := m["messages"].([]any)
	if !ok {
		t.Fatalf("request carries no messages array: %s", raw)
	}
	if len(msgs) != 2 {
		t.Fatalf("expected 2 messages after dropping the reasoning-only assistant turn, got %d: %s", len(msgs), raw)
	}
	for i, item := range msgs {
		msg, ok := item.(map[string]any)
		if !ok {
			t.Fatalf("message %d is not an object: %v", i, item)
		}
		if msg["role"] != "user" {
			t.Errorf("message %d role = %v, want user (assistant turn must be dropped)", i, msg["role"])
		}
	}
}

// TestAnthropicProvider_EmptyInputSchemaDefaultObject verifies review finding
// #57: a tool with a nil (or empty non-nil) InputSchema must be sent with the
// default object schema. Assigned to the SDK's any-typed InputSchema field, a
// typed-nil json.RawMessage defeats omitempty and marshals as
// "input_schema": null — the Messages API requires input_schema to be a JSON
// Schema object and rejects every request with HTTP 400.
func TestAnthropicProvider_EmptyInputSchemaDefaultObject(t *testing.T) {
	provider, lastBody := newAnthropicWireProvider(t, anthropicOKBodyText("ok", `{"input_tokens":1,"output_tokens":1}`))

	_, err := provider.ChatCompletion(context.Background(), ChatRequest{
		Model:     "claude-3-haiku-20240307",
		MaxTokens: 64,
		Messages:  []Message{{Role: "user", Content: "hi"}},
		Tools: []ToolDefinition{
			{Name: "nil_schema", Description: "parameterless tool"},
			{Name: "empty_schema", Description: "empty raw schema", InputSchema: json.RawMessage{}},
		},
	})
	if err != nil {
		t.Fatalf("ChatCompletion: %v", err)
	}
	raw := lastBody()
	if strings.Contains(raw, `"input_schema":null`) || strings.Contains(raw, `"input_schema": null`) {
		t.Errorf("request must not carry a null input_schema, was: %s", raw)
	}
	m := decodeAnthropicRequest(t, raw)
	tools, ok := m["tools"].([]any)
	if !ok {
		t.Fatalf("request carries no tools array: %s", raw)
	}
	if len(tools) != 2 {
		t.Fatalf("expected 2 tools, got %d: %s", len(tools), raw)
	}
	want := map[string]any{
		"type":                 "object",
		"properties":           map[string]any{},
		"required":             []any{},
		"additionalProperties": false,
	}
	for _, item := range tools {
		tool, ok := item.(map[string]any)
		if !ok {
			t.Fatalf("tool entry is not an object: %v", item)
		}
		schema, ok := tool["input_schema"].(map[string]any)
		if !ok {
			t.Fatalf("tool %v: input_schema missing or not an object: %v", tool["name"], tool["input_schema"])
		}
		if !reflect.DeepEqual(schema, want) {
			t.Errorf("tool %v input_schema = %v, want %v", tool["name"], schema, want)
		}
	}
}

// TestAnthropicProvider_UsageSumsCacheTokens verifies review finding #59 over
// the wire: Anthropic reports input_tokens as only the tokens after the last
// cache breakpoint, with the cached prefix delivered separately as
// cache_creation_input_tokens / cache_read_input_tokens. The provider must sum
// all three so usage tracking and the executor's near-context-limit warning
// see the true prompt size.
func TestAnthropicProvider_UsageSumsCacheTokens(t *testing.T) {
	provider, _ := newAnthropicWireProvider(t, anthropicOKBodyText("ok",
		`{"input_tokens":100,"output_tokens":7,"cache_creation_input_tokens":5000,"cache_read_input_tokens":7000}`))

	resp, err := provider.ChatCompletion(context.Background(), ChatRequest{
		Model:     "claude-3-haiku-20240307",
		MaxTokens: 64,
		Messages:  []Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("ChatCompletion: %v", err)
	}
	if got := resp.Usage.InputTokens; got != 12100 {
		t.Errorf("Usage.InputTokens = %d, want 100+5000+7000 = 12100", got)
	}
	if got := resp.Usage.OutputTokens; got != 7 {
		t.Errorf("Usage.OutputTokens = %d, want 7", got)
	}
}

// TestAnthropicProvider_ParseResponse_UsageCacheTokens pins the usage mapping
// at the unit level: cache counters are summed into InputTokens, and a
// response without cache counters is unchanged.
func TestAnthropicProvider_ParseResponse_UsageCacheTokens(t *testing.T) {
	p, err := NewAnthropicProvider(AnthropicProviderConfig{APIKey: "k"})
	if err != nil {
		t.Fatalf("NewAnthropicProvider: %v", err)
	}

	t.Run("cache counters summed into input tokens", func(t *testing.T) {
		resp := anthropic.MessagesResponse{
			Content:    []anthropic.MessageContent{{Type: anthropic.MessagesContentTypeText, Text: stringPtr("hi")}},
			StopReason: "end_turn",
			Usage: anthropic.MessagesUsage{
				InputTokens:              100,
				OutputTokens:             7,
				CacheCreationInputTokens: 5000,
				CacheReadInputTokens:     7000,
			},
		}
		out, err := p.parseResponse(resp, nil)
		if err != nil {
			t.Fatalf("parseResponse: %v", err)
		}
		if out.Usage.InputTokens != 12100 {
			t.Errorf("Usage.InputTokens = %d, want 12100", out.Usage.InputTokens)
		}
	})

	t.Run("no cache counters leaves input unchanged", func(t *testing.T) {
		resp := anthropic.MessagesResponse{
			Content:    []anthropic.MessageContent{{Type: anthropic.MessagesContentTypeText, Text: stringPtr("hi")}},
			StopReason: "end_turn",
			Usage:      anthropic.MessagesUsage{InputTokens: 10, OutputTokens: 5},
		}
		out, err := p.parseResponse(resp, nil)
		if err != nil {
			t.Fatalf("parseResponse: %v", err)
		}
		if out.Usage.InputTokens != 10 || out.Usage.OutputTokens != 5 {
			t.Errorf("Usage = %+v, want input 10 output 5", out.Usage)
		}
	})
}
