package agent

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/v0lka/sp4rk/llm"
	"github.com/v0lka/sp4rk/tools"
)

// TestExecutor_StopTool_TerminatesRun verifies the stop-tool terminator: a
// successful call to a host-designated stop tool ends the run (Finished=true,
// the call's observation as the output) instead of continuing to the next step.
// A later tool call the scripted model would have made must never execute.
func TestExecutor_StopTool_TerminatesRun(t *testing.T) {
	mockLLM := &mockLLMCaller{
		responses: []*llm.ChatResponse{
			llmResponseWithToolCall("declaring", "declare_goal_status", json.RawMessage(`{"status":"not_met"}`)),
			// Must never be reached — the stop tool ends the run above.
			llmResponseWithToolCall("more work", "bash_exec", json.RawMessage(`{"command":"echo no"}`)),
		},
	}
	mockTools := newMockToolExecutor()
	mockTools.results["declare_goal_status"] = tools.ToolResult{Content: "Verdict recorded: goal NOT YET MET."}

	exec := newExecutorDefaultHITL(mockLLM, mockTools, &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
	exec.SetStopTools("declare_goal_status")

	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "declare_goal_status", Description: "declare", Source: "core"},
		{Name: "bash_exec", Description: "exec", Source: "core"},
	}, newMockContextManager())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.Finished {
		t.Error("Finished = false, want true (a successful stop tool terminates the run)")
	}
	if result.Output != "Verdict recorded: goal NOT YET MET." {
		t.Errorf("Output = %q, want the stop tool's observation", result.Output)
	}
	if result.Summary != "declaring" {
		t.Errorf("Summary = %q, want the model's final assistant text (captured for consumers that need the turn's modeled output)", result.Summary)
	}
	if len(mockLLM.calls) != 1 {
		t.Errorf("LLM called %d times, want 1 — the run must end on the stop tool, not call the model again", len(mockLLM.calls))
	}
	for _, c := range mockTools.calls {
		if c.Name != "declare_goal_status" {
			t.Errorf("tool %q executed after the stop tool — the run must end immediately", c.Name)
		}
	}
}

// TestExecutor_StopTool_FailedCallDoesNotTerminate verifies a FAILED stop-tool
// call does not terminate the run: the error observation is returned to the
// model as usual and the loop continues.
func TestExecutor_StopTool_FailedCallDoesNotTerminate(t *testing.T) {
	mockLLM := &mockLLMCaller{
		responses: []*llm.ChatResponse{
			llmResponseWithToolCall("declaring", "declare_goal_status", json.RawMessage(`{"status":"not_met"}`)),
			llmResponseFinish("done after the failed stop call", "wrapped up"),
		},
	}
	mockTools := newMockToolExecutor()
	mockTools.results["declare_goal_status"] = tools.ToolResult{Content: "validation error: bad input", IsError: true}

	exec := newExecutorDefaultHITL(mockLLM, mockTools, &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
	exec.SetStopTools("declare_goal_status")

	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "declare_goal_status", Description: "declare", Source: "core"},
	}, newMockContextManager())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.Finished {
		t.Error("Finished = false, want true (the run continued past the failed stop tool to finish)")
	}
	if len(mockLLM.calls) != 2 {
		t.Errorf("LLM called %d times, want 2 — a failed stop-tool call must not terminate the run", len(mockLLM.calls))
	}
}

// TestExecutor_SetStopTools_Clears verifies SetStopTools with no (or only empty)
// names clears the stop set, restoring the default "every tool keeps the run
// going" behavior.
func TestExecutor_SetStopTools_Clears(t *testing.T) {
	exec := NewExecutor(&mockLLMCaller{}, newMockToolExecutor(), 10)
	exec.SetStopTools("declare_goal_status")
	if len(exec.stopTools) != 1 {
		t.Fatalf("stopTools = %v, want one entry", exec.stopTools)
	}
	exec.SetStopTools()
	if exec.stopTools != nil {
		t.Errorf("stopTools = %v, want nil after clearing", exec.stopTools)
	}
	exec.SetStopTools("", "")
	if exec.stopTools != nil {
		t.Errorf("stopTools = %v, want nil when only empty names are passed", exec.stopTools)
	}
}

// TestExecutor_StopTool_TerminatesRunInsideBatch verifies the stop-tool
// terminator also fires when the stop tool is invoked as a sub-call of the
// `batch` meta-tool: the run ends inside the batch (the trailing sub-call never
// executes) and the model is never called again. Without this a model that
// batched its declare_goal_status call would silently keep working past its
// verdict, defeating the bounded-turn guarantee the stop tool exists to provide.
func TestExecutor_StopTool_TerminatesRunInsideBatch(t *testing.T) {
	batchInput, err := json.Marshal(map[string]any{
		"calls": []map[string]any{
			{"tool": "bash_exec", "input": map[string]string{"command": "echo a"}},
			{"tool": "declare_goal_status", "input": map[string]string{"status": "not_met", "reason": "one more pass"}},
			{"tool": "bash_exec", "input": map[string]string{"command": "echo b"}},
		},
	})
	if err != nil {
		t.Fatalf("marshal batch input: %v", err)
	}

	mockLLM := &mockLLMCaller{
		responses: []*llm.ChatResponse{
			llmResponseWithToolCall("batching the verdict", tools.ToolBatch, batchInput),
			// Must never be reached — the stop sub-call ends the run above.
			llmResponseFinish("should not run", "done"),
		},
	}
	mockTools := newMockToolExecutor()
	mockTools.results["declare_goal_status"] = tools.ToolResult{Content: "Verdict recorded: goal NOT YET MET."}

	exec := newExecutorDefaultHITL(mockLLM, mockTools, &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
	exec.SetStopTools("declare_goal_status")

	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: tools.ToolBatch, Description: "batch", Source: "core"},
		{Name: "bash_exec", Description: "exec", Source: "core"},
		{Name: "declare_goal_status", Description: "declare", Source: "core"},
	}, newMockContextManager())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.Finished {
		t.Error("Finished = false, want true (a stop tool batched as a sub-call terminates the run)")
	}
	if result.Output != "Verdict recorded: goal NOT YET MET." {
		t.Errorf("Output = %q, want the stop tool's observation", result.Output)
	}
	if result.Summary != "batching the verdict" {
		t.Errorf("Summary = %q, want the model's final assistant text", result.Summary)
	}
	if len(mockLLM.calls) != 1 {
		t.Errorf("LLM called %d times, want 1 — the run must end inside the batch, not continue to another step", len(mockLLM.calls))
	}
	for _, c := range mockTools.calls {
		if c.Name == "bash_exec" && strings.Contains(string(c.Input), "echo b") {
			t.Errorf("sub-call after the stop tool executed (%s) — the run must end immediately", c.Input)
		}
	}
}

// TestExecutor_StopTool_FinishGuardRejectsTermination verifies a stop tool
// honours the finish guard: when the guard rejects (e.g. pending async
// delegations), the run injects the guard's nudge and RETRIES instead of
// terminating — a stop tool must not be a back door around the pending-async
// join gate the guard exists to enforce.
func TestExecutor_StopTool_FinishGuardRejectsTermination(t *testing.T) {
	mockLLM := &mockLLMCaller{
		responses: []*llm.ChatResponse{
			llmResponseWithToolCall("declaring", "declare_goal_status", json.RawMessage(`{"status":"not_met"}`)),
			llmResponseFinish("wrapped up after the nudge", "done"),
		},
	}
	mockTools := newMockToolExecutor()
	mockTools.results["declare_goal_status"] = tools.ToolResult{Content: "Verdict recorded: goal NOT YET MET."}

	exec := newExecutorDefaultHITL(mockLLM, mockTools, &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
	exec.SetStopTools("declare_goal_status")
	guardCalls := 0
	exec.SetFinishGuard(func(context.Context) error {
		guardCalls++
		if guardCalls == 1 {
			return errors.New("you have 1 pending async delegation(s): sub-1")
		}
		return nil
	})

	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "declare_goal_status", Description: "declare", Source: "core"},
	}, newMockContextManager())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if guardCalls == 0 {
		t.Fatal("finish guard was never consulted on the stop-tool termination")
	}
	if !result.Finished {
		t.Error("Finished = false, want true (the run continued past the guard-rejected stop tool to finish)")
	}
	if len(mockLLM.calls) != 2 {
		t.Errorf("LLM called %d times, want 2 — a guard-rejected stop tool must not terminate the run", len(mockLLM.calls))
	}
	nudged := false
	for _, s := range result.Steps {
		if strings.Contains(s.UserNudge, "pending async delegation") {
			nudged = true
		}
	}
	if !nudged {
		t.Error("expected the finish guard's message as a nudge step after the rejected stop tool")
	}
}
