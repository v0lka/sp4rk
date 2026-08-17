package builtins

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/v0lka/sp4rk/agent"
	"github.com/v0lka/sp4rk/tools"
)

const toolReadStepOutputDescription = `Purpose: read the complete, untruncated output of one completed plan step by its ID.
Use when: the step's summary in your task description is insufficient and you need the full result of a dependency step before proceeding.
Inputs: step_id (e.g. "step_1") of a completed step.
Outputs: the step's raw output text exactly as it was produced, or an error if the ID is unknown/incomplete.
Example: step_id "step_2" after seeing its one-line summary.
Anti-example: not for the previous task's final answer (read_final_result); summaries usually suffice — pull full outputs only when the summary leaves you unable to act.`

const toolListStepOutputsDescription = `Purpose: list every available completed-step output with a short preview (up to 200 characters each).
Use when: you need to discover which step IDs exist and what they contain before fetching one with read_step_output.
Inputs: none.
Outputs: the list of available step IDs, each with a preview.
Example: call this first when unsure which steps have finished.
Anti-example: not for reading full outputs (read_step_output); not for the previous task's final result (read_final_result).`

// ReadStepOutputTool reads the full output of a completed step from StepOutputStore.
type ReadStepOutputTool struct {
	*tools.BaseTool
}

// NewReadStepOutputTool creates a new ReadStepOutputTool instance.
func NewReadStepOutputTool() *ReadStepOutputTool {
	return &ReadStepOutputTool{BaseTool: &tools.BaseTool{
		ToolName:        "read_step_output",
		ToolGroup:       tools.GroupSystem,
		ToolDescription: toolReadStepOutputDescription,
		Schema: json.RawMessage(`{
		"type": "object",
		"properties": {
			"step_id": {
				"type": "string",
				"description": "The ID of the completed step whose full output you want to read, e.g. \"step_1\""
			}
		},
		"required": ["step_id"]
	}`),
		Policy: tools.PolicyAlwaysAllow,
	}}
}

// ReadStepOutputInput represents the input parameters for read_step_output.
type ReadStepOutputInput struct {
	StepID string `json:"step_id"`
}

// Execute reads the step output from StepOutputStore.
func (t *ReadStepOutputTool) Execute(ctx context.Context, input json.RawMessage) (tools.ToolResult, error) {
	var params ReadStepOutputInput
	if err := json.Unmarshal(input, &params); err != nil {
		return tools.ParseInputError(err)
	}

	if params.StepID == "" {
		return tools.ToolResult{Content: "validation error: step_id is required", IsError: true}, nil
	}

	store := agent.StepOutputStoreFromContext(ctx)
	if store == nil {
		return tools.ErrorResult("Step output store not available"), nil
	}

	output, ok := store.GetStepOutput(params.StepID)
	if !ok {
		return tools.ErrorResult("No output found for step: %s", params.StepID), nil
	}

	return tools.ToolResult{Content: output}, nil
}

// ListStepOutputsTool lists all available step outputs with previews.
type ListStepOutputsTool struct {
	*tools.BaseTool
}

// NewListStepOutputsTool creates a new ListStepOutputsTool instance.
func NewListStepOutputsTool() *ListStepOutputsTool {
	return &ListStepOutputsTool{BaseTool: &tools.BaseTool{
		ToolName:        "list_step_outputs",
		ToolGroup:       tools.GroupSystem,
		ToolDescription: toolListStepOutputsDescription,
		Schema: json.RawMessage(`{
		"type": "object",
		"properties": {},
		"required": []
	}`),
		Policy: tools.PolicyAlwaysAllow,
	}}
}

// ListStepOutputsInput represents the input parameters for list_step_outputs.
type ListStepOutputsInput struct{}

const previewMaxLen = 200

// Execute lists all step outputs from StepOutputStore with previews.
func (t *ListStepOutputsTool) Execute(ctx context.Context, input json.RawMessage) (tools.ToolResult, error) {
	// Validate input (should be empty object)
	var params ListStepOutputsInput
	if err := json.Unmarshal(input, &params); err != nil {
		return tools.ParseInputError(err)
	}

	store := agent.StepOutputStoreFromContext(ctx)
	if store == nil {
		return tools.ErrorResult("Step output store not available"), nil
	}

	entries := store.ListStepOutputs()
	if len(entries) == 0 {
		return tools.ToolResult{Content: "No step outputs available yet"}, nil
	}

	var b strings.Builder
	for _, e := range entries {
		preview := e.FullOutput
		if len(preview) > previewMaxLen {
			preview = preview[:previewMaxLen] + "..."
		}
		preview = strings.ReplaceAll(preview, "\n", " ")
		preview = strings.TrimSpace(preview)

		fmt.Fprintf(&b, "- %s: %s\n", e.StepID, preview)
	}

	return tools.ToolResult{Content: b.String()}, nil
}

// -----------------------------------------------------------------------------
// read_final_result — read the prior task's final result from the blackboard
// -----------------------------------------------------------------------------

const toolReadFinalResultDescription = `Purpose: read the final answer of the previously completed task on this blackboard.
Use when: the prior exchange's outcome is not visible in your history — e.g. after a backend restart, or when the result was too large to inject verbatim — and it matters for the current task.
Inputs: none.
Outputs: the prior task's raw final answer exactly as produced, or an error if no final result is recorded.
Example: call right after a restart to recover the prior context before planning.
Anti-example: not for plan-step outputs (read_step_output or list_step_outputs); not for truncated tool outputs (tool_result_read).`

// ReadFinalResultTool reads the final result of a previously completed task
// from FinalResultStore (backed by the blackboard).
type ReadFinalResultTool struct {
	*tools.BaseTool
}

// NewReadFinalResultTool creates a new ReadFinalResultTool instance.
func NewReadFinalResultTool() *ReadFinalResultTool {
	return &ReadFinalResultTool{BaseTool: &tools.BaseTool{
		ToolName:        "read_final_result",
		ToolGroup:       tools.GroupSystem,
		ToolDescription: toolReadFinalResultDescription,
		Schema: json.RawMessage(`{
		"type": "object",
		"properties": {},
		"required": []
	}`),
		Policy: tools.PolicyAlwaysAllow,
	}}
}

// Execute reads the final result from FinalResultStore.
func (t *ReadFinalResultTool) Execute(ctx context.Context, input json.RawMessage) (tools.ToolResult, error) {
	// Validate input (should be empty object).
	var params struct{}
	if err := json.Unmarshal(input, &params); err != nil {
		return tools.ParseInputError(err)
	}

	store := agent.FinalResultStoreFromContext(ctx)
	if store == nil {
		return tools.ErrorResult("Final result store not available"), nil
	}

	output, ok := store.GetFinalResult()
	if !ok {
		return tools.ErrorResult("No final result is recorded on the blackboard"), nil
	}

	return tools.ToolResult{Content: output}, nil
}
