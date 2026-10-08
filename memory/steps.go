package memory

import (
	"strings"

	sdkagent "github.com/v0lka/sp4rk/agent"
	"github.com/v0lka/sp4rk/llm"
	"github.com/v0lka/sp4rk/strutil"
)

// assistantMsgFromStep renders a step's assistant message exactly as
// ContextWindow.buildAssistantMsg does: Thought (trailing-invisible-trimmed)
// as content, ReasoningContent and the Responses-API ReasoningItems carried
// through untouched, and a "(proceeding)" placeholder only when the message
// would otherwise be content-less AND tool-call-less (a message carrying
// tool_calls is already valid for OpenAI-compatible APIs). Sharing one
// renderer keeps the frozen compaction prefix identical to the live prompt
// path — most importantly, the reasoning chain a Responses-API backend
// requires on every subsequent request must round-trip through both.
func assistantMsgFromStep(thought, reasoningContent string, reasoningItems []llm.ReasoningItem, toolCalls []llm.ToolCall) llm.Message {
	msg := llm.Message{
		Role:             "assistant",
		Content:          strings.TrimRight(thought, strutil.InvisibleTrimSet),
		ReasoningContent: reasoningContent,
		ReasoningItems:   reasoningItems,
		ToolCalls:        toolCalls,
	}
	if msg.Content == "" && len(msg.ToolCalls) == 0 {
		msg.Content = "(proceeding)"
	}
	return msg
}

// isNudgeOnlyStep reports whether a step renders as nothing but its user
// nudge: no (trimmed) thought, no reasoning content, no Responses-API
// reasoning items, no tool call, and a visible nudge. Mirrors the nudgeOnly
// test in ContextWindow.buildStandaloneMessages, so a nudge-carrying step
// that carries reasoning still emits its assistant message after compaction
// instead of being dropped with its reasoning chain.
func isNudgeOnlyStep(thought, reasoningContent string, reasoningItems []llm.ReasoningItem, toolCalls []llm.ToolCall, nudge string) bool {
	return strings.TrimRight(thought, strutil.InvisibleTrimSet) == "" &&
		reasoningContent == "" &&
		len(reasoningItems) == 0 &&
		len(toolCalls) == 0 &&
		nudge != ""
}

// stepsToMessages converts a slice of Steps to LLM messages.
// Each standalone step (ResponseGroup == 0) produces:
// 1. An assistant message with Thought as content and Action as ToolCalls
// 2. A tool message with Observation as content
// 3. A user message with UserNudge, when present (mirrors buildNudgeMsg)
// Steps with matching ResponseGroup > 0 are merged into a single assistant message
// with multiple tool_calls, followed by individual tool result messages; the
// UserNudge of the group's last step is emitted after the tool results.
// Nudge-only steps (no thought, no reasoning content or items, no action,
// non-empty nudge) produce only the user message — no empty "(proceeding)"
// assistant placeholder.
func stepsToMessages(steps []sdkagent.Step) []llm.Message {
	var messages []llm.Message
	for i := 0; i < len(steps); {
		step := steps[i]

		if step.ResponseGroup > 0 {
			// Collect all consecutive steps with the same ResponseGroup
			groupEnd := i + 1
			for groupEnd < len(steps) && steps[groupEnd].ResponseGroup == step.ResponseGroup {
				groupEnd++
			}
			groupSteps := steps[i:groupEnd]

			// Build ONE assistant message with all tool calls
			var toolCalls []llm.ToolCall
			for _, gs := range groupSteps {
				if gs.Action.ID != "" {
					toolCalls = append(toolCalls, gs.Action)
				}
			}
			assistantMsg := assistantMsgFromStep(groupSteps[0].Thought, groupSteps[0].ReasoningContent, groupSteps[0].ReasoningItems, toolCalls)

			// UserNudge only on last step of group (mirrors buildGroupedMessages)
			nudge := strings.TrimRight(groupSteps[len(groupSteps)-1].UserNudge, strutil.InvisibleTrimSet)
			if !isNudgeOnlyStep(groupSteps[0].Thought, groupSteps[0].ReasoningContent, groupSteps[0].ReasoningItems, toolCalls, nudge) {
				messages = append(messages, assistantMsg)
			}
			// Nudge-only group: skip the empty assistant placeholder.

			// Add individual tool result messages
			for _, gs := range groupSteps {
				if gs.Action.ID != "" {
					observation := strings.TrimRight(gs.Observation, strutil.InvisibleTrimSet)
					if observation == "" {
						observation = "(no output)"
					}
					messages = append(messages, llm.Message{
						Role:       "tool",
						Content:    observation,
						ToolCallID: gs.Action.ID,
					})
				}
			}

			if nudge != "" {
				messages = append(messages, llm.Message{Role: "user", Content: nudge})
			}

			i = groupEnd
		} else {
			// Original logic for standalone steps
			var toolCalls []llm.ToolCall
			if step.Action.ID != "" {
				toolCalls = []llm.ToolCall{step.Action}
			}
			assistantMsg := assistantMsgFromStep(step.Thought, step.ReasoningContent, step.ReasoningItems, toolCalls)

			nudge := strings.TrimRight(step.UserNudge, strutil.InvisibleTrimSet)
			if !isNudgeOnlyStep(step.Thought, step.ReasoningContent, step.ReasoningItems, toolCalls, nudge) {
				messages = append(messages, assistantMsg)
			}
			// Nudge-only step: skip the empty assistant placeholder.

			if step.Action.ID != "" {
				observation := strings.TrimRight(step.Observation, strutil.InvisibleTrimSet)
				if observation == "" {
					observation = "(no output)"
				}
				messages = append(messages, llm.Message{
					Role:       "tool",
					Content:    observation,
					ToolCallID: step.Action.ID,
				})
			}

			if nudge != "" {
				messages = append(messages, llm.Message{Role: "user", Content: nudge})
			}

			i++
		}
	}
	return messages
}

// truncateToTokenBudget truncates text to fit within the token budget.
// TruncateUTF8 appends "…" when truncated, followed by the summary marker.
func truncateToTokenBudget(text string, maxTokens int) string {
	maxChars := maxTokens * 3
	if len(text) <= maxChars {
		return text
	}
	return strutil.TruncateUTF8(text, maxChars) + "\n[... truncated for summarization ...]"
}
