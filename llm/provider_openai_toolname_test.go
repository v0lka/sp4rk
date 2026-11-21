package llm

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"regexp"
	"testing"

	oai "github.com/openai/openai-go"
)

// wireToolNameRe is the pattern the Chat Completions wire enforces on every
// tools[].function.name and every assistant-history tool call (api.deepseek.com
// rejects the whole request with HTTP 400 when any of them violates it).
var wireToolNameRe = regexp.MustCompile(`^[a-zA-Z0-9_-]{1,64}$`)

// newChatMappingTestProvider builds a provider whose endpoint is unused (the
// param-building helpers below are pure); see the streaming test for the
// wire-level variant.
func newChatMappingTestProvider(t *testing.T) *OpenAIProvider {
	t.Helper()
	p, err := NewOpenAIProvider(OpenAIProviderConfig{Name: "p", APIKey: "k"})
	if err != nil {
		t.Fatalf("NewOpenAIProvider: %v", err)
	}
	return p
}

// TestOpenAIProvider_BuildChatParams_MapsToolNamesForWire pins the incident
// fix: an MCP tool catalog with dotted names is serialized with
// endpoint-legal spellings (ps.list → ps_list), while legal names pass
// through untouched. Before the mapping, ONE dotted name made DeepSeek (and
// every strict OpenAI-compatible endpoint) reject the whole request with
// HTTP 400 "Invalid 'tools[N].function.name': string does not match pattern".
func TestOpenAIProvider_BuildChatParams_MapsToolNamesForWire(t *testing.T) {
	p := newChatMappingTestProvider(t)
	req := ChatRequest{
		Model:    "deepseek-flash",
		Messages: []Message{{Role: "user", Content: "hi"}},
		Tools: []ToolDefinition{
			{Name: "bash_exec", Description: "legal name"},
			{Name: "ps.list", Description: "dotted MCP tool"},
			{Name: "app.launch", Description: "dotted MCP tool"},
			{Name: "power.status", Description: "dotted MCP tool"},
		},
	}
	mapping := newToolNameMapping(req.Tools, defaultToolNamePolicy, nil)
	params := p.buildChatParams(req, mapping)

	if len(params.Tools) != len(req.Tools) {
		t.Fatalf("tools = %d, want %d", len(params.Tools), len(req.Tools))
	}
	for i, tool := range params.Tools {
		got := tool.Function.Name
		if !wireToolNameRe.MatchString(got) {
			t.Errorf("tools[%d].function.name = %q, want match %v", i, got, wireToolNameRe)
		}
	}
	if got, want := params.Tools[0].Function.Name, "bash_exec"; got != want {
		t.Errorf("tools[0].function.name = %q, want %q (legal name untouched)", got, want)
	}
	if got, want := params.Tools[1].Function.Name, "ps_list"; got != want {
		t.Errorf("tools[1].function.name = %q, want %q", got, want)
	}
	if got, want := params.Tools[2].Function.Name, "app_launch"; got != want {
		t.Errorf("tools[2].function.name = %q, want %q", got, want)
	}
	if got, want := params.Tools[3].Function.Name, "power_status"; got != want {
		t.Errorf("tools[3].function.name = %q, want %q", got, want)
	}

	// The marshaled payload — the bytes that actually hit the wire — carries
	// only legal names.
	raw, err := json.Marshal(params)
	if err != nil {
		t.Fatalf("marshal params: %v", err)
	}
	var wire struct {
		Tools []struct {
			Function struct {
				Name string `json:"name"`
			} `json:"function"`
		} `json:"tools"`
	}
	if err := json.Unmarshal(raw, &wire); err != nil {
		t.Fatalf("unmarshal params: %v", err)
	}
	for i, tool := range wire.Tools {
		if !wireToolNameRe.MatchString(tool.Function.Name) {
			t.Errorf("wire tools[%d].name = %q, want match %v", i, tool.Function.Name, wireToolNameRe)
		}
	}
}

// TestOpenAIProvider_ConvertRequestMessage_MapsHistoryToolCallNames pins that
// assistant-history tool calls ride the same wire spelling as the tool
// definitions: a session's first turns already carry mapped names, so the
// model sees one consistent catalog across the whole conversation.
func TestOpenAIProvider_ConvertRequestMessage_MapsHistoryToolCallNames(t *testing.T) {
	p := newChatMappingTestProvider(t)
	req := ChatRequest{
		Model: "deepseek-flash",
		Messages: []Message{
			{Role: "user", Content: "list services"},
			{
				Role: "assistant",
				ToolCalls: []ToolCall{
					{ID: "call-1", Name: "ps.list", Input: json.RawMessage(`{}`)},
				},
			},
			{Role: "tool", ToolCallID: "call-1", Content: "ok"},
		},
		Tools: []ToolDefinition{{Name: "ps.list"}},
	}
	mapping := newToolNameMapping(req.Tools, defaultToolNamePolicy, nil)
	params := p.buildChatParams(req, mapping)

	raw, err := json.Marshal(params)
	if err != nil {
		t.Fatalf("marshal params: %v", err)
	}
	var wire struct {
		Messages []struct {
			Role      string `json:"role"`
			ToolCalls []struct {
				Function struct {
					Name string `json:"name"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(raw, &wire); err != nil {
		t.Fatalf("unmarshal params: %v", err)
	}
	var historyName string
	for _, m := range wire.Messages {
		for _, tc := range m.ToolCalls {
			if !wireToolNameRe.MatchString(tc.Function.Name) {
				t.Errorf("history %s tool call name = %q, want match %v", m.Role, tc.Function.Name, wireToolNameRe)
			}
			if tc.Function.Name != "" {
				historyName = tc.Function.Name
			}
		}
	}
	if got, want := historyName, "ps_list"; got != want {
		t.Errorf("history assistant tool call name = %q, want %q", got, want)
	}
}

// TestOpenAIProvider_ConvertChatResponseMessage_ReverseMapsToolCallNames pins
// the dispatch contract: the model answers with the wire spelling it was
// given, and the executor must receive the REGISTERED name.
func TestOpenAIProvider_ConvertChatResponseMessage_ReverseMapsToolCallNames(t *testing.T) {
	p := newChatMappingTestProvider(t)
	req := ChatRequest{
		Model: "deepseek-flash",
		Tools: []ToolDefinition{{Name: "ps.list"}, {Name: "bash_exec"}},
	}
	mapping := newToolNameMapping(req.Tools, defaultToolNamePolicy, nil)

	msg := oai.ChatCompletionMessage{
		Role: "assistant",
		ToolCalls: []oai.ChatCompletionMessageToolCall{
			{ID: "call-1", Type: "function", Function: oai.ChatCompletionMessageToolCallFunction{Name: "ps_list", Arguments: "{}"}},
			{ID: "call-2", Type: "function", Function: oai.ChatCompletionMessageToolCallFunction{Name: "bash_exec", Arguments: "{}"}},
			{ID: "call-3", Type: "function", Function: oai.ChatCompletionMessageToolCallFunction{Name: "hallucinated_tool", Arguments: "{}"}},
		},
	}
	got := p.convertChatResponseMessage(msg, mapping)
	if len(got.ToolCalls) != 3 {
		t.Fatalf("tool calls = %d, want 3", len(got.ToolCalls))
	}
	if name := got.ToolCalls[0].Name; name != "ps.list" {
		t.Errorf("tool call 0 name = %q, want %q (wire name reversed to the registered name)", name, "ps.list")
	}
	if name := got.ToolCalls[1].Name; name != "bash_exec" {
		t.Errorf("tool call 1 name = %q, want %q (identity name passes through)", name, "bash_exec")
	}
	if name := got.ToolCalls[2].Name; name != "hallucinated_tool" {
		t.Errorf("tool call 2 name = %q, want %q (unknown names pass through for the not-found result)", name, "hallucinated_tool")
	}
}

// TestOpenAIProvider_ChatCompletionStream_ReversesMappedToolNames pins the
// streaming counterpart: the stream reassembles a fragmented wire name and
// the assembled ChatResponse carries the REGISTERED name for dispatch.
func TestOpenAIProvider_ChatCompletionStream_ReversesMappedToolNames(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		sse(t, w, `{"id":"1","object":"chat.completion.chunk","created":1,"model":"deepseek-flash","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":""}]}`)
		sse(t, w, `{"id":"1","object":"chat.completion.chunk","created":1,"model":"deepseek-flash","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call-1","type":"function","function":{"name":"ps_","arguments":""}}]},"finish_reason":""}]}`)
		sse(t, w, `{"id":"1","object":"chat.completion.chunk","created":1,"model":"deepseek-flash","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"list","arguments":"{}"}}]},"finish_reason":""}]}`)
		sse(t, w, `{"id":"1","object":"chat.completion.chunk","created":1,"model":"deepseek-flash","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`)
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	t.Cleanup(srv.Close)

	p, err := NewOpenAIProvider(OpenAIProviderConfig{Name: "p", APIKey: "k", BaseURL: srv.URL})
	if err != nil {
		t.Fatalf("NewOpenAIProvider: %v", err)
	}

	req := ChatRequest{
		Model:     "deepseek-flash",
		Messages:  []Message{{Role: "user", Content: "list services"}},
		Tools:     []ToolDefinition{{Name: "ps.list"}},
		DeltaSink: func(StreamDelta) error { return nil },
	}
	resp, err := p.ChatCompletion(context.Background(), req)
	if err != nil {
		t.Fatalf("ChatCompletion(stream): %v", err)
	}
	if len(resp.Message.ToolCalls) != 1 {
		t.Fatalf("tool calls = %d, want 1", len(resp.Message.ToolCalls))
	}
	if got, want := resp.Message.ToolCalls[0].Name, "ps.list"; got != want {
		t.Errorf("streamed tool call name = %q, want %q (wire name reversed to the registered name)", got, want)
	}
}
