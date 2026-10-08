package memory

import (
	"testing"

	sdkagent "github.com/v0lka/sp4rk/agent"
	"github.com/v0lka/sp4rk/llm"
)

// testReasoningItems returns a minimal Responses-API reasoning chain whose
// items carry the IDs required for round-tripping the reasoning across turns.
func testReasoningItems(ids ...string) []llm.ReasoningItem {
	items := make([]llm.ReasoningItem, 0, len(ids))
	for _, id := range ids {
		items = append(items, llm.ReasoningItem{ID: id, Summary: "summary " + id, EncryptedContent: "enc:" + id})
	}
	return items
}

// TestStepsToMessages_CarriesReasoningItems verifies that the compaction
// renderer round-trips Step.ReasoningItems onto the assistant message,
// exactly like the live renderer ContextWindow.buildAssistantMsg. Responses-
// API reasoning models require the reasoning item to precede each
// function_call input; dropping it in the frozen compaction prefix broke the
// reasoning round-trip after compaction (and can outright reject the request
// with "function_call provided without its required reasoning item").
func TestStepsToMessages_CarriesReasoningItems(t *testing.T) {
	step := makeStep("Thinking", "result", 1)
	step.ReasoningContent = "chain of thought"
	step.ReasoningItems = testReasoningItems("rs_1")

	messages := stepsToMessages([]sdkagent.Step{step})

	if len(messages) != 2 {
		t.Fatalf("Expected 2 messages, got %d: %+v", len(messages), messages)
	}
	got := messages[0]
	if got.Role != "assistant" {
		t.Fatalf("Message 0 role = %s, want assistant", got.Role)
	}
	if got.Content != "Thinking" {
		t.Errorf("Message 0 content = %q, want %q", got.Content, "Thinking")
	}
	if got.ReasoningContent != "chain of thought" {
		t.Errorf("ReasoningContent = %q, want %q", got.ReasoningContent, "chain of thought")
	}
	if len(got.ReasoningItems) != 1 || got.ReasoningItems[0].ID != "rs_1" {
		t.Errorf("ReasoningItems not carried through: %+v", got.ReasoningItems)
	}
}

// TestStepsToMessages_GroupedCarriesReasoningItems verifies the grouped
// branch carries the group's leading step ReasoningContent and
// ReasoningItems, mirroring buildGroupedMessages.
func TestStepsToMessages_GroupedCarriesReasoningItems(t *testing.T) {
	step1 := makeStep("Group thought", "obs 1", 1)
	step1.ResponseGroup = 7
	step1.ReasoningContent = "group reasoning"
	step1.ReasoningItems = testReasoningItems("rs_g1")
	step2 := makeStep("", "obs 2", 2)
	step2.ResponseGroup = 7

	messages := stepsToMessages([]sdkagent.Step{step1, step2})

	// 1 assistant (merged) + 2 tool results
	if len(messages) != 3 {
		t.Fatalf("Expected 3 messages, got %d: %+v", len(messages), messages)
	}
	got := messages[0]
	if got.ReasoningContent != "group reasoning" {
		t.Errorf("grouped ReasoningContent = %q, want %q", got.ReasoningContent, "group reasoning")
	}
	if len(got.ReasoningItems) != 1 || got.ReasoningItems[0].ID != "rs_g1" {
		t.Errorf("grouped ReasoningItems not carried through: %+v", got.ReasoningItems)
	}
}

// TestStepsToMessages_NudgeStepWithReasoningKeepsAssistant verifies that a
// nudge-carrying step whose Thought is empty but which carries reasoning —
// the exact shape the executor's guard-nudge steps take on Responses-API
// reasoning models that emit reasoning + function_call only — still emits its
// assistant message after compaction. The renderer used to decide
// "nudge-only" from content/tool-calls alone and dropped the assistant
// message together with its whole reasoning chain.
func TestStepsToMessages_NudgeStepWithReasoningKeepsAssistant(t *testing.T) {
	step := sdkagent.Step{
		ReasoningContent: "guard reasoning",
		ReasoningItems:   testReasoningItems("rs_n1"),
		UserNudge:        "mutation gate rejected the call",
	}

	messages := stepsToMessages([]sdkagent.Step{step})

	// assistant(reasoning) + user(nudge)
	if len(messages) != 2 {
		t.Fatalf("Expected 2 messages, got %d: %+v", len(messages), messages)
	}
	got := messages[0]
	if got.Role != "assistant" || got.Content != "(proceeding)" {
		t.Errorf("Message 0 = (%s, %q), want (assistant, \"(proceeding)\")", got.Role, got.Content)
	}
	if got.ReasoningContent != "guard reasoning" {
		t.Errorf("ReasoningContent = %q, want %q", got.ReasoningContent, "guard reasoning")
	}
	if len(got.ReasoningItems) != 1 || got.ReasoningItems[0].ID != "rs_n1" {
		t.Errorf("ReasoningItems not carried through: %+v", got.ReasoningItems)
	}
	if messages[1].Role != "user" || messages[1].Content != "mutation gate rejected the call" {
		t.Errorf("Message 1 = (%s, %q), want the user nudge", messages[1].Role, messages[1].Content)
	}
}

// TestStepsToMessages_GroupedNudgeWithReasoningKeepsAssistant verifies the
// grouped branch of the nudge-with-reasoning case: a reasoning-bearing group
// whose only visible outputs are tool calls plus a trailing nudge keeps its
// merged assistant message.
func TestStepsToMessages_GroupedNudgeWithReasoningKeepsAssistant(t *testing.T) {
	step1 := makeStep("", "obs 1", 1)
	step1.ResponseGroup = 7
	step1.ReasoningContent = "group guard reasoning"
	step1.ReasoningItems = testReasoningItems("rs_gn1")
	step2 := makeStep("", "obs 2", 2)
	step2.ResponseGroup = 7
	step2.UserNudge = "finish guard rejected the call"

	messages := stepsToMessages([]sdkagent.Step{step1, step2})

	// 1 assistant (merged) + 2 tool results + 1 user nudge
	if len(messages) != 4 {
		t.Fatalf("Expected 4 messages, got %d: %+v", len(messages), messages)
	}
	got := messages[0]
	if got.Role != "assistant" || len(got.ToolCalls) != 2 {
		t.Fatalf("Message 0 should be merged assistant with 2 tool calls, got (%s, %d calls)", got.Role, len(got.ToolCalls))
	}
	if got.ReasoningContent != "group guard reasoning" {
		t.Errorf("grouped ReasoningContent = %q, want %q", got.ReasoningContent, "group guard reasoning")
	}
	if len(got.ReasoningItems) != 1 || got.ReasoningItems[0].ID != "rs_gn1" {
		t.Errorf("grouped ReasoningItems not carried through: %+v", got.ReasoningItems)
	}
	if messages[3].Role != "user" || messages[3].Content != "finish guard rejected the call" {
		t.Errorf("Message 3 = (%s, %q), want the user nudge", messages[3].Role, messages[3].Content)
	}
}

// TestStepsToMessages_ReasoningWithoutNudgeGetsPlaceholder verifies parity
// with ContextWindow.buildAssistantMsg: a reasoning-bearing, action-less step
// with an empty thought and no nudge renders as assistant{(proceeding),
// reasoning} — the same placeholder the live prompt path emits.
func TestStepsToMessages_ReasoningWithoutNudgeGetsPlaceholder(t *testing.T) {
	step := sdkagent.Step{
		ReasoningContent: "silent reasoning",
		ReasoningItems:   testReasoningItems("rs_p1"),
	}

	messages := stepsToMessages([]sdkagent.Step{step})

	if len(messages) != 1 {
		t.Fatalf("Expected 1 message, got %d: %+v", len(messages), messages)
	}
	got := messages[0]
	if got.Role != "assistant" || got.Content != "(proceeding)" {
		t.Errorf("Message 0 = (%s, %q), want (assistant, \"(proceeding)\")", got.Role, got.Content)
	}
	if got.ReasoningContent != "silent reasoning" || len(got.ReasoningItems) != 1 {
		t.Errorf("assistant message lost reasoning: content=%q items=%+v", got.ReasoningContent, got.ReasoningItems)
	}
}
