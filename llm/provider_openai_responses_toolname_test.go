package llm

import (
	"testing"

	"github.com/openai/openai-go/responses"
)

// TestConvertToResponsesTools_MapsToolNamesForWire pins the tool-definition
// side of the Responses mapping: an MCP catalog with dotted names is
// serialized with endpoint-legal spellings (the Responses API enforces the
// same ^[a-zA-Z0-9_-]{1,64}$ pattern the Chat Completions API does).
func TestConvertToResponsesTools_MapsToolNamesForWire(t *testing.T) {
	tools := []ToolDefinition{
		{Name: "bash_exec", Description: "legal name"},
		{Name: "ps.list", Description: "dotted MCP tool"},
		{Name: "app.launch", Description: "dotted MCP tool"},
	}
	mapping := newToolNameMapping(tools, defaultToolNamePolicy, nil)
	result := convertToResponsesTools(tools, mapping)
	if len(result) != len(tools) {
		t.Fatalf("tools = %d, want %d", len(result), len(tools))
	}
	for i, tool := range result {
		if tool.OfFunction == nil {
			t.Fatalf("tools[%d] carries no function definition", i)
		}
		if !wireToolNameRe.MatchString(tool.OfFunction.Name) {
			t.Errorf("tools[%d].name = %q, want match %v", i, tool.OfFunction.Name, wireToolNameRe)
		}
	}
	if got, want := result[0].OfFunction.Name, "bash_exec"; got != want {
		t.Errorf("tools[0].name = %q, want %q (legal name untouched)", got, want)
	}
	if got, want := result[1].OfFunction.Name, "ps_list"; got != want {
		t.Errorf("tools[1].name = %q, want %q", got, want)
	}
	if got, want := result[2].OfFunction.Name, "app_launch"; got != want {
		t.Errorf("tools[2].name = %q, want %q", got, want)
	}
}

// TestConvertToResponsesInput_MapsHistoryFunctionCallNames pins the history
// side: assistant function_call items ride the same wire spelling as the
// definitions, and a nil mapping (or an out-of-catalog name) still produces a
// pattern-legal item — the historical sanitizeResponsesFunctionName guarantee
// against history-poisoning 400s, now shared with the other protocols.
func TestConvertToResponsesInput_MapsHistoryFunctionCallNames(t *testing.T) {
	mapping := newToolNameMapping([]ToolDefinition{{Name: "ps.list"}}, defaultToolNamePolicy, nil)

	msgs := []Message{
		{Role: "user", Content: "list services"},
		{
			Role: "assistant",
			ToolCalls: []ToolCall{
				{ID: "call-1", Name: "ps.list", Input: []byte(`{}`)},
			},
		},
		{Role: "tool", ToolCallID: "call-1", Content: "ok"},
	}
	items := convertToResponsesInput(msgs, nil, mapping)
	var historyName string
	for _, item := range items {
		if item.OfFunctionCall != nil {
			if !wireToolNameRe.MatchString(item.OfFunctionCall.Name) {
				t.Errorf("history function_call name = %q, want match %v", item.OfFunctionCall.Name, wireToolNameRe)
			}
			historyName = item.OfFunctionCall.Name
		}
	}
	if got, want := historyName, "ps_list"; got != want {
		t.Errorf("history function_call name = %q, want %q", got, want)
	}

	// A nil mapping sanitizes out-of-catalog names too: a model-hallucinated
	// dotted name in stale history must not poison the request with a
	// retryable=false HTTP 400 on the next turn.
	hallucinated := []Message{
		{
			Role: "assistant",
			ToolCalls: []ToolCall{
				{ID: "call-x", Name: "functions.get_me", Input: []byte(`{}`)},
			},
		},
	}
	items = convertToResponsesInput(hallucinated, nil, nil)
	if len(items) != 1 || items[0].OfFunctionCall == nil {
		t.Fatalf("items = %+v, want one function_call item", items)
	}
	if got, want := items[0].OfFunctionCall.Name, "functions_get_me"; got != want {
		t.Errorf("hallucinated history name = %q, want %q (sanitized for the wire)", got, want)
	}
}

// TestConvertResponsesResponse_ReverseMapsFunctionCallNames pins the dispatch
// contract: the model echoes the wire spelling it was given, and the host
// receives the REGISTERED name.
func TestConvertResponsesResponse_ReverseMapsFunctionCallNames(t *testing.T) {
	mapping := newToolNameMapping([]ToolDefinition{{Name: "ps.list"}, {Name: "bash_exec"}}, defaultToolNamePolicy, nil)
	resp := &responses.Response{
		Status: responses.ResponseStatusCompleted,
		Output: []responses.ResponseOutputItemUnion{
			{Type: "function_call", CallID: "call-1", Name: "ps_list", Arguments: `{}`},
			{Type: "function_call", CallID: "call-2", Name: "bash_exec", Arguments: `{}`},
			{Type: "function_call", CallID: "call-3", Name: "hallucinated_tool", Arguments: `{}`},
		},
	}
	result, err := convertResponsesResponse(resp, mapping)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Message.ToolCalls) != 3 {
		t.Fatalf("tool calls = %d, want 3", len(result.Message.ToolCalls))
	}
	if got, want := result.Message.ToolCalls[0].Name, "ps.list"; got != want {
		t.Errorf("tool call 0 name = %q, want %q (wire name reversed to the registered name)", got, want)
	}
	if got, want := result.Message.ToolCalls[1].Name, "bash_exec"; got != want {
		t.Errorf("tool call 1 name = %q, want %q (identity name passes through)", got, want)
	}
	if got, want := result.Message.ToolCalls[2].Name, "hallucinated_tool"; got != want {
		t.Errorf("tool call 2 name = %q, want %q (unknown names pass through)", got, want)
	}
}

// TestApplyStreamedOutputFallback_ReverseMapsMergedFunctionCallNames pins the
// streaming fallback: function_call items merged from response.output_item.done
// when the terminal Response omitted them go through the same reversal.
func TestApplyStreamedOutputFallback_ReverseMapsMergedFunctionCallNames(t *testing.T) {
	mapping := newToolNameMapping([]ToolDefinition{{Name: "ps.list"}}, defaultToolNamePolicy, nil)
	converted := &ChatResponse{Message: Message{Role: "assistant"}}
	doneItems := []responses.ResponseOutputItemUnion{
		{Type: "function_call", CallID: "call-1", Name: "ps_list", Arguments: `{}`},
	}
	applyStreamedOutputFallback(converted, "", "", "", doneItems, mapping)
	if len(converted.Message.ToolCalls) != 1 {
		t.Fatalf("tool calls = %d, want 1 (merged from done items)", len(converted.Message.ToolCalls))
	}
	if got, want := converted.Message.ToolCalls[0].Name, "ps.list"; got != want {
		t.Errorf("merged tool call name = %q, want %q", got, want)
	}
	if converted.StopReason != "tool_use" {
		t.Errorf("stop_reason = %q, want %q", converted.StopReason, "tool_use")
	}
}
