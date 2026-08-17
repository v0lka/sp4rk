package builtins

import (
	"context"
	"encoding/json"
	"errors"

	"github.com/v0lka/sp4rk/tools"
)

const batchDescription = `Purpose: execute multiple independent tool calls sequentially in a single round-trip.
Use when: you already know several calls you want to make and none depends on another's output (e.g. read three files, check two paths). Calls run in order and all execute even if some fail — errors are captured per call and never abort the batch.
Inputs: calls — an array of {"tool": name, "input": {...}} objects.
Outputs: one result per call, in order, including per-call errors.
Example: calls=[read_file a.go, read_file b.go, list_directory src/] in one invocation.
Anti-example: not for dependent chains (read then edit must be separate turns — later calls cannot use earlier results); a single call does not need batch — invoke the tool directly.`

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
							"tool": {"type": "string", "description": "Name of the tool to call"},
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
