package agent

import (
	"context"
	"encoding/json"

	"github.com/v0lka/sp4rk/tools"
)

const toolFinishDescription = `Purpose: signal task completion and deliver the final result to the user.
Use when: all work is done and you have verified every acceptance criterion of your task with tool calls — not assumptions. Call this tool exactly once; if any criterion is unmet, keep working instead.
Inputs: answer (the complete result: findings, analysis, code summaries, or any deliverable the task requested, summarized concisely).
Outputs: signals completion; the answer string becomes the delivered result.
Example: after tests pass and each criterion is confirmed, call finish with the full deliverable.
Anti-example: not for progress updates, rhetorical messages, or intermediate notes — and never call it while a criterion is unmet or to ask the user a question (use ask_user).`

// FinishTool is a special tool that signals task completion.
type FinishTool struct{}

// NewFinishTool creates a new FinishTool.
func NewFinishTool() *FinishTool {
	return &FinishTool{}
}

// Name returns the tool name.
func (t *FinishTool) Name() string {
	return "finish"
}

// Description returns the tool description.
func (t *FinishTool) Description() string {
	return toolFinishDescription
}

// InputSchema returns the JSON schema for the tool input.
func (t *FinishTool) InputSchema() json.RawMessage {
	return json.RawMessage(`{
		"type": "object",
		"properties": {
			"answer": {
				"type": "string",
				"description": "The final answer to the user's task"
			}
		},
		"required": ["answer"]
	}`)
}

// DefaultPolicy returns PolicyAlwaysAllow because finish tool only signals completion.
func (t *FinishTool) DefaultPolicy() tools.ToolPolicy {
	return tools.PolicyAlwaysAllow
}

// IsUntrusted returns false — finish is a trusted internal tool.
func (t *FinishTool) IsUntrusted() bool { return false }

// Group returns GroupSystem — finish signals completion and touches no
// filesystem or network resources itself.
func (t *FinishTool) Group() tools.ToolGroup { return tools.GroupSystem }

// Execute parses the input and returns the answer.
func (t *FinishTool) Execute(ctx context.Context, input json.RawMessage) (tools.ToolResult, error) {
	var params struct {
		Answer string `json:"answer"`
	}
	if err := json.Unmarshal(input, &params); err != nil {
		//nolint:nilerr // error is reported in ToolResult.Content
		return tools.ToolResult{Content: "failed to parse finish input: " + err.Error(), IsError: true}, nil
	}
	return tools.ToolResult{Content: params.Answer, IsError: false}, nil
}
