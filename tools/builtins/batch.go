package builtins

import (
	"context"
	"encoding/json"
	"errors"

	"github.com/v0lka/sp4rk/tools"
)

const batchDescription = `Purpose: execute multiple independent tool calls sequentially in one round-trip.
Use when: the calls are known and none depends on another's output (e.g. read three files). Calls run in order; all execute even if some fail — per-call errors never abort the batch.
Inputs: calls — array of {"tool": name, "input": {...}}. Use the exact name from the available-tools catalog, e.g. "read_file", not "functions.read_file" (no namespace prefix). Every sub-call input must satisfy that tool's schema: required parameters present, correct types. The batch is validated BEFORE anything executes — any invalid sub-call rejects it: NOTHING runs and one aggregated error lists each offending index, its reason and the expected schema. Empty {} is valid only for tools with no required parameters. Fix the listed calls and resend.
Outputs: one result per call, in order, including errors.
Example: {"calls":[{"tool":"read_file","input":{"path":"a.go"}},{"tool":"list_directory","input":{"path":"src/"}}]}
Anti-example: not for dependent chains (later calls need earlier results); a single call needs no batch; {"calls":[{"tool":"read_file","input":{}}]} is rejected wholesale — read_file requires "path".`

// BatchTool allows the LLM to batch multiple independent tool calls.
// The actual batch logic is handled at the executor level; this tool
// exists to expose the JSON schema to the LLM.
type BatchTool struct {
	*tools.BaseTool
}

// NewBatchTool creates a new BatchTool instance.
func NewBatchTool() *BatchTool {
	return &BatchTool{BaseTool: &tools.BaseTool{
		ToolName:        "batch",
		ToolGroup:       tools.GroupSystem,
		ToolDescription: batchDescription,
		Schema: json.RawMessage(`{
			"type": "object",
			"properties": {
				"calls": {
					"type": "array",
					"items": {
						"type": "object",
						"properties": {
							"tool": {"type": "string", "description": "Exact tool name from the available-tools catalog, e.g. read_file, not functions.read_file. Do not add a namespace such as functions."},
							"input": {"type": "object", "description": "Arguments to pass to the tool"}
						},
						"required": ["tool", "input"]
					}
				}
			},
			"required": ["calls"]
		}`),
		Policy: tools.PolicyAlwaysAllow,
	}}
}

// Execute returns an error — batch is intercepted and handled at the executor level.
func (t *BatchTool) Execute(ctx context.Context, input json.RawMessage) (tools.ToolResult, error) {
	return tools.ToolResult{}, errors.New("batch is handled at the executor level and should not be called directly")
}
