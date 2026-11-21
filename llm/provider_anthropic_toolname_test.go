package llm

import (
	"encoding/json"
	"testing"

	"github.com/liushuangls/go-anthropic/v2"
)

// TestAnthropicProvider_BuildRequest_MapsToolNamesForWire pins the
// tool-definition and history sides of the Anthropic mapping: the Messages
// API documents ^[a-zA-Z0-9_-]{1,64}$ for tool names, so an MCP-provided
// dotted name must travel under its endpoint-legal spelling in both the
// tools array and the assistant-history tool_use blocks.
func TestAnthropicProvider_BuildRequest_MapsToolNamesForWire(t *testing.T) {
	p := anthropicProvider(t)
	req := ChatRequest{
		Model: "claude-sonnet-4-5",
		Messages: []Message{
			{Role: "user", Content: "list services"},
			{
				Role: "assistant",
				ToolCalls: []ToolCall{
					{ID: "toolu_1", Name: "ps.list", Input: json.RawMessage(`{}`)},
				},
			},
			{Role: "tool", ToolCallID: "toolu_1", Content: "ok"},
		},
		Tools: []ToolDefinition{
			{Name: "bash_exec", Description: "legal name"},
			{Name: "ps.list", Description: "dotted MCP tool"},
		},
	}
	mapping := newToolNameMapping(req.Tools, defaultToolNamePolicy, nil)
	anthropicReq, err := p.buildRequest(req, mapping)
	if err != nil {
		t.Fatalf("buildRequest: %v", err)
	}

	if len(anthropicReq.Tools) != 2 {
		t.Fatalf("tools = %d, want 2", len(anthropicReq.Tools))
	}
	if got, want := anthropicReq.Tools[0].Name, "bash_exec"; got != want {
		t.Errorf("tools[0].name = %q, want %q (legal name untouched)", got, want)
	}
	if got, want := anthropicReq.Tools[1].Name, "ps_list"; got != want {
		t.Errorf("tools[1].name = %q, want %q", got, want)
	}

	// The history assistant turn must carry the same wire spelling.
	found := false
	for _, msg := range anthropicReq.Messages {
		for _, block := range msg.Content {
			if block.Type == anthropic.MessagesContentTypeToolUse && block.MessageContentToolUse != nil {
				found = true
				if !wireToolNameRe.MatchString(block.Name) {
					t.Errorf("history tool_use name = %q, want match %v", block.Name, wireToolNameRe)
				}
				if got, want := block.Name, "ps_list"; got != want {
					t.Errorf("history tool_use name = %q, want %q", got, want)
				}
			}
		}
	}
	if !found {
		t.Fatal("history carries no tool_use block, want one")
	}
}

// TestAnthropicProvider_ParseResponse_ReverseMapsToolUseNames pins the
// dispatch contract: the model answers with the wire spelling it was given,
// and the executor receives the REGISTERED name.
func TestAnthropicProvider_ParseResponse_ReverseMapsToolUseNames(t *testing.T) {
	p := anthropicProvider(t)
	mapping := newToolNameMapping([]ToolDefinition{{Name: "ps.list"}, {Name: "bash_exec"}}, defaultToolNamePolicy, nil)

	resp := anthropic.MessagesResponse{
		Content: []anthropic.MessageContent{
			anthropic.NewToolUseMessageContent("toolu_1", "ps_list", json.RawMessage(`{}`)),
			anthropic.NewToolUseMessageContent("toolu_2", "bash_exec", json.RawMessage(`{}`)),
			anthropic.NewToolUseMessageContent("toolu_3", "hallucinated_tool", json.RawMessage(`{}`)),
		},
		StopReason: "tool_use",
		Usage: anthropic.MessagesUsage{
			InputTokens:  10,
			OutputTokens: 5,
		},
	}
	result, err := p.parseResponse(resp, nil, mapping)
	if err != nil {
		t.Fatalf("parseResponse: %v", err)
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
