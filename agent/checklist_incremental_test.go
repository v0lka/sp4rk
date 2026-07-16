package agent

import (
	"context"
	"encoding/json"
	"strings"
	"sync"
	"testing"

	"github.com/v0lka/sp4rk/llm"
	"github.com/v0lka/sp4rk/tools"
)

// diagCapture is a minimal Events implementation that records which
// ExecutorDiagnostic event names were emitted. It embeds *NoopEvents so all
// other event methods are satisfied.
type diagCapture struct {
	*NoopEvents
	mu    sync.Mutex
	found map[string]bool
}

func newDiagCapture() *diagCapture {
	return &diagCapture{NoopEvents: &NoopEvents{}, found: make(map[string]bool)}
}

func (d *diagCapture) ExecutorDiagnostic(_ int, event string, _ map[string]any) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.found[event] = true
}

func (d *diagCapture) has(event string) bool {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.found[event]
}

// lastChecklistStep returns the most recent successful update_checklist step,
// or nil if none. Used to assert on the batching suffix appended to the
// observation.
func lastChecklistStep(steps []Step) *Step {
	for i := len(steps) - 1; i >= 0; i-- {
		if steps[i].Action.Name == "update_checklist" && !steps[i].IsError {
			return &steps[i]
		}
	}
	return nil
}

// countStaleNudges counts steps whose UserNudge is the staleness nudge.
func countStaleNudges(steps []Step) int {
	n := 0
	for _, s := range steps {
		if strings.Contains(s.UserNudge, "tool calls since your last update_checklist") {
			n++
		}
	}
	return n
}

const checklistCannedResult = "Checklist updated."

// checklistInput builds an update_checklist JSON input from a raw todo_list.
func checklistInput(todo string) json.RawMessage {
	b, _ := json.Marshal(map[string]string{"todo_list": todo})
	return b
}

// ============================================================================
// Helpers — direct unit tests for the staleness/batching counting logic
// ============================================================================

func TestProductiveCallsSinceLastChecklistUpdate(t *testing.T) {
	e := &Executor{}
	cases := []struct {
		name  string
		steps []Step
		want  int
	}{
		{"no checklist", []Step{
			{Action: llm.ToolCall{Name: "read_file"}},
			{Action: llm.ToolCall{Name: "read_file"}},
		}, 2},
		{"calls after update", []Step{
			{Action: llm.ToolCall{Name: "update_checklist"}},
			{Action: llm.ToolCall{Name: "read_file"}},
			{Action: llm.ToolCall{Name: "read_file"}},
			{Action: llm.ToolCall{Name: "read_file"}},
		}, 3},
		{"update resets", []Step{
			{Action: llm.ToolCall{Name: "read_file"}},
			{Action: llm.ToolCall{Name: "update_checklist"}},
			{Action: llm.ToolCall{Name: "read_file"}},
		}, 1},
		{"excludes finish and nudges", []Step{
			{Action: llm.ToolCall{Name: "update_checklist"}},
			{Action: llm.ToolCall{Name: "read_file"}},
			{Action: llm.ToolCall{Name: ""}},       // nudge step
			{Action: llm.ToolCall{Name: "finish"}}, // finish
			{Action: llm.ToolCall{Name: "read_file"}},
		}, 2},
		{"failed update ignored as anchor", []Step{
			{Action: llm.ToolCall{Name: "update_checklist"}, IsError: true},
			{Action: llm.ToolCall{Name: "read_file"}},
			{Action: llm.ToolCall{Name: "read_file"}},
		}, 2},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			state := &runState{allSteps: tc.steps}
			if got := e.productiveCallsSinceLastChecklistUpdate(state); got != tc.want {
				t.Errorf("got %d, want %d", got, tc.want)
			}
		})
	}
}

func TestLastChecklistUpdateIndex(t *testing.T) {
	e := &Executor{}
	state := &runState{allSteps: []Step{
		{Action: llm.ToolCall{Name: "update_checklist"}},
		{Action: llm.ToolCall{Name: "read_file"}},
		{Action: llm.ToolCall{Name: "update_checklist"}},
		{Action: llm.ToolCall{Name: "read_file"}},
	}}
	if got := e.lastChecklistUpdateIndex(state); got != 2 {
		t.Errorf("expected last successful update at index 2, got %d", got)
	}
	state2 := &runState{allSteps: []Step{{Action: llm.ToolCall{Name: "read_file"}}}}
	if got := e.lastChecklistUpdateIndex(state2); got != -1 {
		t.Errorf("expected -1 when no update, got %d", got)
	}
}

// ============================================================================
// Layer 2 — batching detector
// ============================================================================

func TestChecklistBatching_FirstUpdate_NoSuffix(t *testing.T) {
	// First update is initialization — never a batch, so no suffix.
	mockLLM := &mockLLMCaller{
		responses: []*llm.ChatResponse{
			llmResponseWithToolCall("init", "update_checklist", checklistInput("- [ ] A\n- [ ] B")),
			llmResponseFinish("done", "Finished."),
		},
	}
	mockTools := newMockToolExecutor()
	mockTools.results["update_checklist"] = tools.ToolResult{Content: checklistCannedResult}
	exec := newExecutorDefaultHITL(mockLLM, mockTools, &mockTokenCounter{}, 20, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)

	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "read_file", Description: "read", Source: "core"},
		checklistToolDescriptor(),
	}, newMockContextManager())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	cs := lastChecklistStep(result.Steps)
	if cs == nil {
		t.Fatal("expected an update_checklist step")
	}
	if strings.Contains(cs.Observation, "[System]") {
		t.Errorf("first update should not carry a batching suffix; got: %q", cs.Observation)
	}
}

func TestChecklistBatching_SingleNewlyChecked_PositiveSuffix(t *testing.T) {
	// Second update marks exactly one previously-unchecked item → positive suffix.
	mockLLM := &mockLLMCaller{
		responses: []*llm.ChatResponse{
			llmResponseWithToolCall("init", "update_checklist", checklistInput("- [ ] A\n- [ ] B")),
			llmResponseWithToolCall("A done", "update_checklist", checklistInput("- [x] A\n- [ ] B")),
			llmResponseFinish("done", "Finished."),
		},
	}
	mockTools := newMockToolExecutor()
	mockTools.results["update_checklist"] = tools.ToolResult{Content: checklistCannedResult}
	exec := newExecutorDefaultHITL(mockLLM, mockTools, &mockTokenCounter{}, 20, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)

	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "read_file", Description: "read", Source: "core"},
		checklistToolDescriptor(),
	}, newMockContextManager())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	cs := lastChecklistStep(result.Steps)
	if cs == nil {
		t.Fatal("expected an update_checklist step")
	}
	if !strings.Contains(cs.Observation, checklistBatchPositiveSuffix) {
		t.Errorf("single newly-checked item should earn a positive suffix; got: %q", cs.Observation)
	}
}

func TestChecklistBatching_MultiNewlyChecked_WarningSuffixAndDiagnostic(t *testing.T) {
	// Second update marks three previously-unchecked items → warning + diagnostic.
	diag := newDiagCapture()
	mockLLM := &mockLLMCaller{
		responses: []*llm.ChatResponse{
			llmResponseWithToolCall("init", "update_checklist", checklistInput("- [ ] A\n- [ ] B\n- [ ] C")),
			llmResponseWithToolCall("all done", "update_checklist", checklistInput("- [x] A\n- [x] B\n- [x] C")),
			llmResponseFinish("done", "Finished."),
		},
	}
	mockTools := newMockToolExecutor()
	mockTools.results["update_checklist"] = tools.ToolResult{Content: checklistCannedResult}
	exec := newExecutorDefaultHITL(mockLLM, mockTools, &mockTokenCounter{}, 20, diag, false, ToolResultBudget{}, defaultCircuitBreakerConfig)

	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "read_file", Description: "read", Source: "core"},
		checklistToolDescriptor(),
	}, newMockContextManager())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	cs := lastChecklistStep(result.Steps)
	if cs == nil {
		t.Fatal("expected an update_checklist step")
	}
	if !strings.Contains(cs.Observation, "marked 3 items complete") {
		t.Errorf("multi newly-checked should produce a warning naming the count 3; got: %q", cs.Observation)
	}
	if !diag.has("checklist_batched_update") {
		t.Error("expected checklist_batched_update diagnostic to be emitted")
	}
}

func TestChecklistBatching_NewPrecheckedItem_NoWarning(t *testing.T) {
	// A genuinely new item added as already-checked must NOT count as a batch.
	mockLLM := &mockLLMCaller{
		responses: []*llm.ChatResponse{
			llmResponseWithToolCall("init", "update_checklist", checklistInput("- [ ] A\n- [ ] B")),
			llmResponseWithToolCall("added C", "update_checklist", checklistInput("- [ ] A\n- [ ] B\n- [x] C")),
			llmResponseFinish("done", "Finished."),
		},
	}
	mockTools := newMockToolExecutor()
	mockTools.results["update_checklist"] = tools.ToolResult{Content: checklistCannedResult}
	exec := newExecutorDefaultHITL(mockLLM, mockTools, &mockTokenCounter{}, 20, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)

	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "read_file", Description: "read", Source: "core"},
		checklistToolDescriptor(),
	}, newMockContextManager())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	cs := lastChecklistStep(result.Steps)
	if cs == nil {
		t.Fatal("expected an update_checklist step")
	}
	if strings.Contains(cs.Observation, "[System]") {
		t.Errorf("a new pre-checked item must not trigger a batch warning; got: %q", cs.Observation)
	}
}

func TestChecklistBatching_DisabledGate_NoSuffix(t *testing.T) {
	mockLLM := &mockLLMCaller{
		responses: []*llm.ChatResponse{
			llmResponseWithToolCall("init", "update_checklist", checklistInput("- [ ] A\n- [ ] B")),
			llmResponseWithToolCall("all done", "update_checklist", checklistInput("- [x] A\n- [x] B")),
			llmResponseFinish("done", "Finished."),
		},
	}
	mockTools := newMockToolExecutor()
	mockTools.results["update_checklist"] = tools.ToolResult{Content: checklistCannedResult}
	exec := newExecutorDefaultHITL(mockLLM, mockTools, &mockTokenCounter{}, 20, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
	exec.SetChecklistGateEnabled(false)

	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "read_file", Description: "read", Source: "core"},
		checklistToolDescriptor(),
	}, newMockContextManager())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	cs := lastChecklistStep(result.Steps)
	if cs == nil {
		t.Fatal("expected an update_checklist step")
	}
	if strings.Contains(cs.Observation, "[System]") {
		t.Errorf("disabled gate should not append a batching suffix; got: %q", cs.Observation)
	}
}

// ============================================================================
// Layer 3 — staleness nudge
// ============================================================================

// variedReadInput returns a read_file input with a path unique per index, so
// calls are not "identical" (which would trip the repeat circuit breaker at the
// 3rd call) and not flagged as fruitless. i is the 1-based call number.
func variedReadInput(i int) json.RawMessage {
	return json.RawMessage(`{"path": "/tmp/file_` + string(rune('a'+i)) + `.go"}`)
}

func TestChecklistStale_FiresAfterThreshold(t *testing.T) {
	// init update, then 3 reads → staleness nudge fires once (sinceUpdate=3).
	mockLLM := &mockLLMCaller{
		responses: []*llm.ChatResponse{
			llmResponseWithToolCall("init", "update_checklist", checklistInput("- [ ] A\n- [ ] B\n- [ ] C")),
			llmResponseWithToolCall("r1", "read_file", variedReadInput(1)),
			llmResponseWithToolCall("r2", "read_file", variedReadInput(2)),
			llmResponseWithToolCall("r3", "read_file", variedReadInput(3)),
			llmResponseFinish("done", "Finished."),
		},
	}
	mockTools := newMockToolExecutor()
	mockTools.results["read_file"] = tools.ToolResult{Content: strings.Repeat("x", 40)}
	mockTools.results["update_checklist"] = tools.ToolResult{Content: checklistCannedResult}
	exec := newExecutorDefaultHITL(mockLLM, mockTools, &mockTokenCounter{}, 20, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)

	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "read_file", Description: "read", Source: "core"},
		checklistToolDescriptor(),
	}, newMockContextManager())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := countStaleNudges(result.Steps); got != 1 {
		t.Errorf("expected exactly 1 staleness nudge after 3 productive calls, got %d", got)
	}
}

func TestChecklistStale_ReArmsAfterUpdate(t *testing.T) {
	// init, 3 reads (nudge#1), update (re-arm), 3 reads (nudge#2), finish.
	mockLLM := &mockLLMCaller{
		responses: []*llm.ChatResponse{
			llmResponseWithToolCall("init", "update_checklist", checklistInput("- [ ] A\n- [ ] B\n- [ ] C\n- [ ] D\n- [ ] E\n- [ ] F")),
			llmResponseWithToolCall("r1", "read_file", variedReadInput(1)),
			llmResponseWithToolCall("r2", "read_file", variedReadInput(2)),
			llmResponseWithToolCall("r3", "read_file", variedReadInput(3)),
			llmResponseWithToolCall("update2", "update_checklist", checklistInput("- [x] A\n- [ ] B\n- [ ] C\n- [ ] D\n- [ ] E\n- [ ] F")),
			llmResponseWithToolCall("r4", "read_file", variedReadInput(4)),
			llmResponseWithToolCall("r5", "read_file", variedReadInput(5)),
			llmResponseWithToolCall("r6", "read_file", variedReadInput(6)),
			llmResponseFinish("done", "Finished."),
		},
	}
	mockTools := newMockToolExecutor()
	mockTools.results["read_file"] = tools.ToolResult{Content: strings.Repeat("x", 40)}
	mockTools.results["update_checklist"] = tools.ToolResult{Content: checklistCannedResult}
	exec := newExecutorDefaultHITL(mockLLM, mockTools, &mockTokenCounter{}, 20, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)

	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "read_file", Description: "read", Source: "core"},
		checklistToolDescriptor(),
	}, newMockContextManager())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := countStaleNudges(result.Steps); got != 2 {
		t.Errorf("expected exactly 2 staleness nudges (one per re-arm window), got %d", got)
	}
}

func TestChecklistStale_RespectsCap(t *testing.T) {
	// init, 6 reads, finish. Nudges fire after read3 and read4, then capped.
	mockLLM := &mockLLMCaller{
		responses: []*llm.ChatResponse{
			llmResponseWithToolCall("init", "update_checklist", checklistInput("- [ ] A\n- [ ] B\n- [ ] C")),
			llmResponseWithToolCall("r1", "read_file", variedReadInput(1)),
			llmResponseWithToolCall("r2", "read_file", variedReadInput(2)),
			llmResponseWithToolCall("r3", "read_file", variedReadInput(3)),
			llmResponseWithToolCall("r4", "read_file", variedReadInput(4)),
			llmResponseWithToolCall("r5", "read_file", variedReadInput(5)),
			llmResponseWithToolCall("r6", "read_file", variedReadInput(6)),
			llmResponseFinish("done", "Finished."),
		},
	}
	mockTools := newMockToolExecutor()
	mockTools.results["read_file"] = tools.ToolResult{Content: strings.Repeat("x", 40)}
	mockTools.results["update_checklist"] = tools.ToolResult{Content: checklistCannedResult}
	exec := newExecutorDefaultHITL(mockLLM, mockTools, &mockTokenCounter{}, 20, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)

	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "read_file", Description: "read", Source: "core"},
		checklistToolDescriptor(),
	}, newMockContextManager())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := countStaleNudges(result.Steps); got != checklistStaleNudgeCap {
		t.Errorf("expected exactly %d staleness nudges (cap), got %d", checklistStaleNudgeCap, got)
	}
}

func TestChecklistStale_NoFireBeforeFirstUpdate(t *testing.T) {
	// No update_checklist at all → staleness gate must not fire (even though the
	// tool is available). The finish-time missing-checklist nudge may appear,
	// but no staleness nudge.
	mockLLM := &mockLLMCaller{
		responses: []*llm.ChatResponse{
			llmResponseWithToolCall("r1", "read_file", variedReadInput(1)),
			llmResponseWithToolCall("r2", "read_file", variedReadInput(2)),
			llmResponseWithToolCall("r3", "read_file", variedReadInput(3)),
			llmResponseWithToolCall("r4", "read_file", variedReadInput(4)),
			llmResponseFinish("done", "Finished."),
			llmResponseFinish("done2", "Finished again."),
		},
	}
	mockTools := newMockToolExecutor()
	mockTools.results["read_file"] = tools.ToolResult{Content: strings.Repeat("x", 40)}
	exec := newExecutorDefaultHITL(mockLLM, mockTools, &mockTokenCounter{}, 20, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)

	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "read_file", Description: "read", Source: "core"},
		checklistToolDescriptor(),
	}, newMockContextManager())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := countStaleNudges(result.Steps); got != 0 {
		t.Errorf("staleness nudge must not fire before the first checklist update, got %d", got)
	}
}

func TestChecklistStale_DisabledGate_NoFire(t *testing.T) {
	mockLLM := &mockLLMCaller{
		responses: []*llm.ChatResponse{
			llmResponseWithToolCall("init", "update_checklist", checklistInput("- [ ] A\n- [ ] B\n- [ ] C")),
			llmResponseWithToolCall("r1", "read_file", variedReadInput(1)),
			llmResponseWithToolCall("r2", "read_file", variedReadInput(2)),
			llmResponseWithToolCall("r3", "read_file", variedReadInput(3)),
			llmResponseFinish("done", "Finished."),
		},
	}
	mockTools := newMockToolExecutor()
	mockTools.results["read_file"] = tools.ToolResult{Content: strings.Repeat("x", 40)}
	mockTools.results["update_checklist"] = tools.ToolResult{Content: checklistCannedResult}
	exec := newExecutorDefaultHITL(mockLLM, mockTools, &mockTokenCounter{}, 20, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
	exec.SetChecklistGateEnabled(false)

	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "read_file", Description: "read", Source: "core"},
		checklistToolDescriptor(),
	}, newMockContextManager())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := countStaleNudges(result.Steps); got != 0 {
		t.Errorf("disabled gate must not fire staleness nudges, got %d", got)
	}
}
