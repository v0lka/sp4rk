package llm

import (
	"encoding/json"
	"strings"
	"testing"
)

// countCacheControl walks a decoded Anthropic wire request and counts every
// cache_control object across the tools, system, and message trees — the
// provider must never exceed the API's 4-breakpoints-per-request budget.
func countCacheControl(t *testing.T, v any) int {
	t.Helper()
	n := 0
	switch x := v.(type) {
	case map[string]any:
		if _, ok := x["cache_control"]; ok {
			n++
		}
		for _, vv := range x {
			n += countCacheControl(t, vv)
		}
	case []any:
		for _, vv := range x {
			n += countCacheControl(t, vv)
		}
	}
	return n
}

// blockCacheControl returns the "type" of the cache_control object attached to
// a decoded content block ("" when absent).
func blockCacheControl(t *testing.T, blk any) string {
	t.Helper()
	m, ok := blk.(map[string]any)
	if !ok {
		t.Fatalf("content block is not an object: %v", blk)
	}
	cc, ok := m["cache_control"].(map[string]any)
	if !ok {
		return ""
	}
	if cc["type"] != "ephemeral" {
		t.Fatalf("cache_control.type = %v, want ephemeral", cc["type"])
	}
	return "ephemeral"
}

// TestAnthropicProvider_CacheBreakpoints pins the wire shape of the three
// cache_control breakpoints the Anthropic provider arms on executor-shaped
// requests: the END OF THE STABLE SYSTEM (the CacheBreakMarker boundary),
// the LAST TOOL DEFINITION, and the STABLE HISTORY BOUNDARY (the last block
// of the last message). Exactly 3 breakpoints — within the API's
// 4-per-request budget — and nothing on the dynamic suffix, the first tool,
// or the older history turns.
func TestAnthropicProvider_CacheBreakpoints(t *testing.T) {
	provider, lastBody := newAnthropicWireProvider(t,
		anthropicOKBodyText("ok", `{"input_tokens":1,"output_tokens":1}`))

	req := ChatRequest{
		Model:     "claude-3-haiku-20240307",
		MaxTokens: 8192,
		Messages: []Message{
			// The planner's CacheBreakMarker split: stable prefix, dynamic suffix.
			{Role: "system", Content: "You are a coding agent. (stable)"},
			{Role: "system", Content: "Current task: refactor auth. (dynamic)"},
			{Role: "user", Content: "list files"},
			{Role: "assistant", ToolCalls: []ToolCall{
				{ID: "tc_1", Name: "ls", Input: json.RawMessage(`{}`)},
			}},
			{Role: "tool", ToolCallID: "tc_1", Content: "a.go b.go"},
		},
		Tools: []ToolDefinition{
			{Name: "ls", Description: "list", InputSchema: json.RawMessage(`{"type":"object"}`)},
			{Name: "read", Description: "read", InputSchema: json.RawMessage(`{"type":"object"}`)},
		},
	}

	if _, err := provider.ChatCompletion(t.Context(), req); err != nil {
		t.Fatalf("ChatCompletion: %v", err)
	}
	m := decodeAnthropicRequest(t, lastBody())

	// System: the array form with ONE breakpoint on the stable prefix (part
	// 0); the dynamic suffix carries none.
	sys, ok := m["system"].([]any)
	if !ok || len(sys) != 2 {
		t.Fatalf("system should be a 2-part array, got %v in %s", m["system"], lastBody())
	}
	if blockCacheControl(t, sys[0]) == "" {
		t.Errorf("system[0] (stable prefix) must carry the cache breakpoint: %s", lastBody())
	}
	if blockCacheControl(t, sys[1]) != "" {
		t.Errorf("system[1] (dynamic suffix) must NOT carry a cache breakpoint: %s", lastBody())
	}

	// Tools: ONE breakpoint on the LAST definition.
	tools, ok := m["tools"].([]any)
	if !ok || len(tools) != 2 {
		t.Fatalf("tools should be a 2-element array, got %v in %s", m["tools"], lastBody())
	}
	if blockCacheControl(t, tools[0]) != "" {
		t.Errorf("tools[0] must NOT carry a cache breakpoint: %s", lastBody())
	}
	if blockCacheControl(t, tools[1]) == "" {
		t.Errorf("tools[last] (tool-definitions prefix boundary) must carry the cache breakpoint: %s", lastBody())
	}

	// History: ONE breakpoint on the last block of the last message (the
	// tool result that closes this request's prefix).
	msgs, ok := m["messages"].([]any)
	if !ok || len(msgs) != 3 {
		t.Fatalf("messages should be a 3-element array, got %v in %s", m["messages"], lastBody())
	}
	for i, msg := range msgs[:len(msgs)-1] {
		msgMap, _ := msg.(map[string]any)
		content, _ := msgMap["content"].([]any)
		for _, blk := range content {
			if blockCacheControl(t, blk) != "" {
				t.Errorf("messages[%d] must NOT carry a cache breakpoint (only the last message does): %s", i, lastBody())
			}
		}
	}
	lastMsg, _ := msgs[len(msgs)-1].(map[string]any)
	lastBlocks, _ := lastMsg["content"].([]any)
	if blockCacheControl(t, lastBlocks[len(lastBlocks)-1]) == "" {
		t.Errorf("the last block of the last message must carry the stable-history-boundary breakpoint: %s", lastBody())
	}

	// Budget: strictly within the API's 4-breakpoint cap.
	if n := countCacheControl(t, m); n != 3 {
		t.Errorf("total cache_control breakpoints = %d, want exactly 3 (system + tools + history): %s", n, lastBody())
	}
}

// TestAnthropicProvider_CacheBreakpoints_SingleSystem pins the single-system
// case: the one all-stable part is its own boundary, rendered as a
// one-element system array with the breakpoint.
func TestAnthropicProvider_CacheBreakpoints_SingleSystem(t *testing.T) {
	provider, lastBody := newAnthropicWireProvider(t,
		anthropicOKBodyText("ok", `{"input_tokens":1,"output_tokens":1}`))

	req := ChatRequest{
		Model:     "claude-3-haiku-20240307",
		MaxTokens: 8192,
		Messages: []Message{
			{Role: "system", Content: "You are helpful."},
			{Role: "user", Content: "Hello"},
		},
	}
	if _, err := provider.ChatCompletion(t.Context(), req); err != nil {
		t.Fatalf("ChatCompletion: %v", err)
	}
	m := decodeAnthropicRequest(t, lastBody())

	sys, ok := m["system"].([]any)
	if !ok || len(sys) != 1 {
		t.Fatalf("system should be a 1-part array, got %v in %s", m["system"], lastBody())
	}
	if blockCacheControl(t, sys[0]) == "" {
		t.Errorf("the single all-stable system part must carry the cache breakpoint: %s", lastBody())
	}
	// And the growing-history breakpoint on the final user message.
	msgs, _ := m["messages"].([]any)
	lastMsg, _ := msgs[len(msgs)-1].(map[string]any)
	lastBlocks, _ := lastMsg["content"].([]any)
	if blockCacheControl(t, lastBlocks[len(lastBlocks)-1]) == "" {
		t.Errorf("the final user message must carry the history breakpoint: %s", lastBody())
	}
	if n := countCacheControl(t, m); n != 2 {
		t.Errorf("total cache_control breakpoints = %d, want exactly 2 (no tools): %s", n, lastBody())
	}
}

// TestAnthropicProvider_CacheBreakpoints_DeterministicPurpose pins the
// purpose gate: routing/compaction/summarization calls are one-shot requests
// — a cache write there is pure premium with no reuse — so their wire shape
// carries NO cache_control anywhere and keeps the legacy plain-string system.
func TestAnthropicProvider_CacheBreakpoints_DeterministicPurpose(t *testing.T) {
	provider, lastBody := newAnthropicWireProvider(t,
		anthropicOKBodyText(`{"verdict":"a"}`, `{"input_tokens":1,"output_tokens":1}`))

	req := ChatRequest{
		Model:       "claude-3-haiku-20240307",
		MaxTokens:   1024,
		CallPurpose: CallPurposeRouting,
		Messages: []Message{
			{Role: "system", Content: "You are a router."},
			{Role: "user", Content: "classify this"},
		},
		Tools: []ToolDefinition{
			{Name: "ls", Description: "list", InputSchema: json.RawMessage(`{"type":"object"}`)},
		},
	}
	if _, err := provider.ChatCompletion(t.Context(), req); err != nil {
		t.Fatalf("ChatCompletion: %v", err)
	}
	m := decodeAnthropicRequest(t, lastBody())

	if s, ok := m["system"].(string); !ok || s != "You are a router." {
		t.Fatalf("deterministic calls keep the legacy string system, got %v in %s", m["system"], lastBody())
	}
	if n := countCacheControl(t, m); n != 0 {
		t.Errorf("deterministic-purpose requests must carry zero cache_control breakpoints, got %d: %s", n, lastBody())
	}
}

// TestAnthropicProvider_ThinkingRoundTrip pins the full carriage loop:
// parseResponse captures thinking AND redacted_thinking blocks verbatim onto
// ChatResponse.Message, and the next buildRequest re-emits them at the START
// of the assistant turn on the wire, ahead of the text/tool_use blocks —
// with thinking still enabled for the continuation.
func TestAnthropicProvider_ThinkingRoundTrip(t *testing.T) {
	// First call: a thinking-enabled response with a thinking block, a
	// redacted_thinking block, text, and a tool_use block.
	body := `{"id":"msg_t","type":"message","role":"assistant",` +
		`"model":"claude-3-haiku-20240307",` +
		`"content":[` +
		`{"type":"thinking","thinking":"step one reasoning","signature":"SIG1"},` +
		`{"type":"redacted_thinking","data":"ENCRYPTED"},` +
		`{"type":"text","text":"running"},` +
		`{"type":"tool_use","id":"tc_1","name":"ls","input":{}}` +
		`],"stop_reason":"tool_use","stop_sequence":null,` +
		`"usage":{"input_tokens":10,"output_tokens":5}}`
	provider, lastBody := newAnthropicWireProvider(t, body)
	ctx := t.Context()

	resp, err := provider.ChatCompletion(ctx, ChatRequest{
		Model:           "claude-3-haiku-20240307",
		MaxTokens:       8192,
		ReasoningEffort: "On",
		Messages:        []Message{{Role: "user", Content: "list files"}},
		Tools: []ToolDefinition{
			{Name: "ls", Description: "list", InputSchema: json.RawMessage(`{"type":"object"}`)},
		},
	})
	if err != nil {
		t.Fatalf("first ChatCompletion: %v", err)
	}

	// parseResponse captures both blocks verbatim; Reasoning accumulates only
	// the readable thinking.
	blocks := resp.Message.ThinkingBlocks
	if len(blocks) != 2 {
		t.Fatalf("ThinkingBlocks = %+v, want 2 blocks (thinking + redacted_thinking)", blocks)
	}
	if blocks[0].Type != "thinking" || blocks[0].Thinking != "step one reasoning" || blocks[0].Signature != "SIG1" {
		t.Errorf("thinking block not captured verbatim: %+v", blocks[0])
	}
	if blocks[1].Type != "redacted_thinking" || blocks[1].Data != "ENCRYPTED" {
		t.Errorf("redacted_thinking block not captured verbatim: %+v", blocks[1])
	}
	if !strings.Contains(resp.Reasoning, "step one reasoning") || strings.Contains(resp.Reasoning, "ENCRYPTED") {
		t.Errorf("Reasoning = %q, want the readable thinking only", resp.Reasoning)
	}

	// Second call: the continuation re-sends the assistant turn (executor
	// history) plus the tool result. Thinking must stay enabled and the wire
	// must carry the captured blocks at the START of the assistant turn.
	cont := ChatRequest{
		Model:           "claude-3-haiku-20240307",
		MaxTokens:       8192,
		ReasoningEffort: "On",
		Messages: []Message{
			{Role: "user", Content: "list files"},
			resp.Message,
			{Role: "tool", ToolCallID: "tc_1", Content: "a.go"},
		},
		Tools: []ToolDefinition{
			{Name: "ls", Description: "list", InputSchema: json.RawMessage(`{"type":"object"}`)},
		},
	}
	if _, err := provider.ChatCompletion(ctx, cont); err != nil {
		t.Fatalf("continuation ChatCompletion: %v", err)
	}
	m := decodeAnthropicRequest(t, lastBody())

	if _, ok := m["thinking"]; !ok {
		t.Errorf("thinking must stay enabled on the carried continuation: %s", lastBody())
	}
	msgs, _ := m["messages"].([]any)
	var assistantContent []any
	for _, mm := range msgs {
		msg, _ := mm.(map[string]any)
		if role, _ := msg["role"].(string); role == "assistant" {
			assistantContent, _ = msg["content"].([]any)
		}
	}
	if len(assistantContent) != 4 {
		t.Fatalf("assistant turn should re-emit thinking + redacted_thinking + text + tool_use, got %d blocks: %s", len(assistantContent), lastBody())
	}
	first, _ := assistantContent[0].(map[string]any)
	if first["type"] != "thinking" || first["thinking"] != "step one reasoning" || first["signature"] != "SIG1" {
		t.Errorf("assistant turn must START with the verbatim thinking block, got %v", first)
	}
	second, _ := assistantContent[1].(map[string]any)
	if second["type"] != "redacted_thinking" || second["data"] != "ENCRYPTED" {
		t.Errorf("redacted_thinking must follow verbatim, got %v", second)
	}
	third, _ := assistantContent[2].(map[string]any)
	fourth, _ := assistantContent[3].(map[string]any)
	if third["type"] != "text" || fourth["type"] != "tool_use" {
		t.Errorf("text/tool_use blocks must follow the thinking blocks: %s", lastBody())
	}
}
