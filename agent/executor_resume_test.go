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

// recordingTrajectoryStore is a TrajectoryStore that records every Sync call
// so tests can assert what the executor synced at each iteration.
type recordingTrajectoryStore struct {
	mu    sync.Mutex
	syncs [][]Step
}

func (r *recordingTrajectoryStore) Sync(steps []Step) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.syncs = append(r.syncs, append([]Step(nil), steps...))
}

func (r *recordingTrajectoryStore) Steps() []Step {
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.syncs) == 0 {
		return nil
	}
	return r.syncs[len(r.syncs)-1]
}

// firstSync returns the steps from the first Sync call (or nil if none).
func (r *recordingTrajectoryStore) firstSync() []Step {
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.syncs) == 0 {
		return nil
	}
	return r.syncs[0]
}

// resumeTestSteps builds two realistic prior ReAct steps for resume tests.
func resumeTestSteps() []Step {
	return []Step{
		{
			Thought: "I will read the first file",
			Action: llm.ToolCall{
				ID:    "call_resume_1",
				Name:  "read_file",
				Input: json.RawMessage(`{"path":"/tmp/a"}`),
			},
			Observation: "contents of a",
			TokensUsed:  50,
		},
		{
			Thought: "Now I will read the second file",
			Action: llm.ToolCall{
				ID:    "call_resume_2",
				Name:  "read_file",
				Input: json.RawMessage(`{"path":"/tmp/b"}`),
			},
			Observation: "contents of b",
			TokensUsed:  50,
		},
	}
}

// newResumingExecutor builds an Executor configured with the given resume
// steps, mirroring the option set used by the sp4rk Conductor.
func newResumingExecutor(llmCaller LLMCaller, toolRegistry ToolExecutor, counter llm.TokenCounter, maxSteps int, emitter Events, resumeSteps []Step) *Executor {
	opts := []Option{
		WithTokenCounter(counter),
		WithEvents(emitter),
		WithToolResultBudget(ToolResultBudget{}),
		WithCircuitBreaker(defaultCircuitBreakerConfig),
	}
	if resumeSteps != nil {
		opts = append(opts, WithResumeSteps(resumeSteps))
	}
	exec := NewExecutor(llmCaller, toolRegistry, maxSteps, opts...)
	// The checklist gate is unrelated to resume behavior; disable it so it
	// cannot nudge a resumed finish in these focused tests.
	exec.SetChecklistGateEnabled(false)
	return exec
}

// TestExecutor_Run_WithResumeSteps_ContinuesStepNumber verifies that a resumed
// executor's step counter continues from len(resumeSteps)+1 rather than 1.
func TestExecutor_Run_WithResumeSteps_ContinuesStepNumber(t *testing.T) {
	resumeSteps := resumeTestSteps()
	mockLLM := &mockLLMCaller{
		responses: []*llm.ChatResponse{
			llmResponseFinish("wrapping up", "done after resume"),
		},
	}
	events := &recordingEvents{}
	cm := newMockContextManager()
	exec := newResumingExecutor(mockLLM, newMockToolExecutor(), &mockTokenCounter{}, 10, events, resumeSteps)

	result, err := exec.Run(context.Background(), nil, cm)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.Finished {
		t.Error("expected Finished=true")
	}

	// The first StepStart event must be at len(resumeSteps)+1 = 3.
	var firstStart string
	for _, e := range events.events {
		if strings.HasPrefix(e, "StepStart:") {
			firstStart = e
			break
		}
	}
	if firstStart != "StepStart:3" {
		t.Errorf("first StepStart event = %q, want %q (resume should continue at step 3)", firstStart, "StepStart:3")
	}
}

// TestExecutor_Run_WithResumeSteps_ResultIncludesAllSteps verifies that the
// returned steps comprise the resumed steps plus the newly executed step.
func TestExecutor_Run_WithResumeSteps_ResultIncludesAllSteps(t *testing.T) {
	resumeSteps := resumeTestSteps()
	mockLLM := &mockLLMCaller{
		responses: []*llm.ChatResponse{
			llmResponseFinish("wrapping up", "resumed output"),
		},
	}
	cm := newMockContextManager()
	exec := newResumingExecutor(mockLLM, newMockToolExecutor(), &mockTokenCounter{}, 10, &recordingEvents{}, resumeSteps)

	result, err := exec.Run(context.Background(), nil, cm)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// 2 resumed steps + 1 finish step = 3.
	if want := len(resumeSteps) + 1; len(result.Steps) != want {
		t.Fatalf("len(result.Steps) = %d, want %d", len(result.Steps), want)
	}

	// The first two steps must be the resumed ones (by their thoughts).
	if got, want := result.Steps[0].Thought, resumeSteps[0].Thought; got != want {
		t.Errorf("result.Steps[0].Thought = %q, want %q", got, want)
	}
	if got, want := result.Steps[1].Thought, resumeSteps[1].Thought; got != want {
		t.Errorf("result.Steps[1].Thought = %q, want %q", got, want)
	}
	// The final step is the freshly-executed finish call.
	if result.Steps[2].Action.Name != "finish" {
		t.Errorf("result.Steps[2].Action.Name = %q, want %q", result.Steps[2].Action.Name, "finish")
	}
}

// TestExecutor_Run_WithResumeSteps_SyncsFullTrajectory verifies that the
// TrajectoryStore receives the resumed steps on the very first Sync (before
// any new step executes), so tools like reflect see the complete history. It
// also confirms that as new steps execute, the synced trajectory grows to
// include both the resumed steps and the newly executed step.
func TestExecutor_Run_WithResumeSteps_SyncsFullTrajectory(t *testing.T) {
	resumeSteps := resumeTestSteps()
	// LLM does a tool call (new step 3), then finishes (new step 4).
	toolInput := json.RawMessage(`{"path":"/tmp/c"}`)
	mockLLM := &mockLLMCaller{
		responses: []*llm.ChatResponse{
			llmResponseWithToolCall("read a third file", "read_file", toolInput),
			llmResponseFinish("wrapping up", "resumed output"),
		},
	}
	mockTools := newMockToolExecutor()
	mockTools.results["read_file"] = tools.ToolResult{Content: "contents of c"}
	store := &recordingTrajectoryStore{}
	ctx := WithTrajectoryStore(context.Background(), store)
	cm := newMockContextManager()
	exec := newResumingExecutor(mockLLM, mockTools, &mockTokenCounter{}, 10, &recordingEvents{}, resumeSteps)

	_, err := exec.Run(ctx, []tools.ToolDescriptor{
		{Name: "read_file", Description: "read a file", Source: "core"},
	}, cm)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// The first Sync (top of the resumed loop, before any new step) must
	// contain exactly the resumed steps.
	first := store.firstSync()
	if len(first) != len(resumeSteps) {
		t.Fatalf("first trajectory Sync had %d steps, want %d resumed steps", len(first), len(resumeSteps))
	}
	for i, s := range first {
		if got, want := s.Action.ID, resumeSteps[i].Action.ID; got != want {
			t.Errorf("first Sync step[%d].Action.ID = %q, want %q", i, got, want)
		}
	}

	// After the new tool step executes, the synced trajectory must include the
	// resumed steps plus the newly executed read_file step. (The final finish
	// step is appended to allSteps during processing but the loop exits before
	// the next Sync — so it is not itself synced, matching the non-resume
	// behavior. The point of resume is that the PRIOR steps are synced.)
	final := store.Steps()
	if want := len(resumeSteps) + 1; len(final) != want {
		t.Errorf("final trajectory had %d steps, want %d (resumed + new tool step)", len(final), want)
	}
	if final[len(final)-1].Action.Name != "read_file" {
		t.Errorf("last synced step action = %q, want %q", final[len(final)-1].Action.Name, "read_file")
	}
}

// TestExecutor_Run_WithoutResume_StartsAtStep1 verifies backward
// compatibility: with no resume steps, the executor still starts at step 1.
func TestExecutor_Run_WithoutResume_StartsAtStep1(t *testing.T) {
	mockLLM := &mockLLMCaller{
		responses: []*llm.ChatResponse{
			llmResponseFinish("done", "fresh-start output"),
		},
	}
	events := &recordingEvents{}
	cm := newMockContextManager()
	// Pass nil resume steps (the option is not even added).
	exec := newResumingExecutor(mockLLM, newMockToolExecutor(), &mockTokenCounter{}, 10, events, nil)

	result, err := exec.Run(context.Background(), nil, cm)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.Finished {
		t.Error("expected Finished=true")
	}
	var firstStart string
	for _, e := range events.events {
		if strings.HasPrefix(e, "StepStart:") {
			firstStart = e
			break
		}
	}
	if firstStart != "StepStart:1" {
		t.Errorf("first StepStart event = %q, want %q", firstStart, "StepStart:1")
	}
	// Only the single finish step was executed.
	if len(result.Steps) != 1 {
		t.Errorf("len(result.Steps) = %d, want 1", len(result.Steps))
	}
}

// TestWithResumeSteps_OptionSetsField is a focused unit test verifying that the
// WithResumeSteps option propagates into the Executor's resumeSteps field.
func TestWithResumeSteps_OptionSetsField(t *testing.T) {
	steps := resumeTestSteps()
	exec := NewExecutor(&mockLLMCaller{}, newMockToolExecutor(), 10, WithResumeSteps(steps))
	if len(exec.resumeSteps) != len(steps) {
		t.Errorf("exec.resumeSteps len = %d, want %d", len(exec.resumeSteps), len(steps))
	}
	if exec.resumeSteps[0].Action.ID != steps[0].Action.ID {
		t.Errorf("exec.resumeSteps[0].Action.ID = %q, want %q", exec.resumeSteps[0].Action.ID, steps[0].Action.ID)
	}

	// Omitting the option leaves resumeSteps nil (fresh-start behavior).
	exec2 := NewExecutor(&mockLLMCaller{}, newMockToolExecutor(), 10)
	if exec2.resumeSteps != nil {
		t.Errorf("exec2.resumeSteps = %v, want nil when option omitted", exec2.resumeSteps)
	}
}
