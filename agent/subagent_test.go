package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/v0lka/sp4rk/llm"
	"github.com/v0lka/sp4rk/tools"
)

func TestRunSubAgent_Success(t *testing.T) {
	mockLLM := &mockLLMCaller{
		responses: []*llm.ChatResponse{
			llmResponseFinish("done", "agent output"),
		},
	}
	cm := newMockContextManager()
	exec := newExecutorDefaultHITL(mockLLM, newMockToolExecutor(), &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)

	ch := RunSubAgent(context.Background(), "step_1", exec, cm, nil, "test task", nil, nil)
	result := <-ch
	if result.Error != nil {
		t.Fatalf("unexpected error: %v", result.Error)
	}
	if result.StepID != "step_1" {
		t.Errorf("StepID = %q, want %q", result.StepID, "step_1")
	}
	if result.Output != "agent output" {
		t.Errorf("Output = %q, want %q", result.Output, "agent output")
	}
}

func TestRunSubAgent_WithEmitter(t *testing.T) {
	mockLLM := &mockLLMCaller{
		responses: []*llm.ChatResponse{
			llmResponseFinish("done", "output"),
		},
	}
	cm := newMockContextManager()
	events := &recordingEvents{}
	exec := newExecutorDefaultHITL(mockLLM, newMockToolExecutor(), &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)

	ch := RunSubAgent(context.Background(), "step_1", exec, cm, nil, "task desc", events, nil)
	result := <-ch
	if result.Error != nil {
		t.Fatalf("unexpected error: %v", result.Error)
	}

	// Check emitter received launch and complete events
	foundLaunch := false
	foundComplete := false
	for _, e := range events.events {
		if e == "SubAgentLaunch:step_1" {
			foundLaunch = true
		}
		if e == "SubAgentComplete:step_1:true" {
			foundComplete = true
		}
	}
	if !foundLaunch {
		t.Error("expected SubAgentLaunch event")
	}
	if !foundComplete {
		t.Error("expected SubAgentComplete event with success=true")
	}
	if got := events.lastCompleteErr(); got != "" {
		t.Errorf("successful SubAgentComplete errMsg = %q, want empty", got)
	}
}

// TestRunSubAgent_CompleteCarriesFailureReason asserts the terminal
// SubAgentComplete event carries the SAME failure reason as the returned
// SubAgentResult.Error. Hosts (c0wrk) surface that reason in the subagent chat
// block, so the two must agree.
func TestRunSubAgent_CompleteCarriesFailureReason(t *testing.T) {
	t.Run("llm error", func(t *testing.T) {
		events := &recordingEvents{}
		mockLLM := &mockLLMCaller{errors: []error{errors.New("llm failed")}}
		exec := newExecutorDefaultHITL(mockLLM, newMockToolExecutor(), &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)

		ch := RunSubAgent(context.Background(), "step_1", exec, newMockContextManager(), nil, "task", events, nil)
		result := <-ch
		if result.Error == nil {
			t.Fatal("expected error")
		}
		if got := events.lastCompleteErr(); got != result.Error.Error() {
			t.Errorf("SubAgentComplete errMsg = %q, want %q", got, result.Error.Error())
		}
	})

	t.Run("max steps exhausted carries the abort reason", func(t *testing.T) {
		events := &recordingEvents{}
		mockLLM := &mockLLMCaller{
			responses: []*llm.ChatResponse{
				llmResponseWithToolCall("t1", "tool1", json.RawMessage(`{}`)),
				llmResponseWithToolCall("t2", "tool2", json.RawMessage(`{}`)),
			},
		}
		exec := newExecutorDefaultHITL(mockLLM, newMockToolExecutor(), &mockTokenCounter{}, 2, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)

		ch := RunSubAgent(context.Background(), "step_1", exec, newMockContextManager(), []tools.ToolDescriptor{
			{Name: "tool1", Description: "t", Source: "core"},
		}, "task", events, nil)
		result := <-ch
		if result.Error == nil {
			t.Fatal("expected error for max steps exhaustion")
		}
		got := events.lastCompleteErr()
		if got == "" || got != result.Error.Error() {
			t.Errorf("SubAgentComplete errMsg = %q, want non-empty %q", got, result.Error.Error())
		}
	})

	t.Run("recovered panic", func(t *testing.T) {
		events := &recordingEvents{}
		exec := newExecutorDefaultHITL(panickingLLMCaller{}, newMockToolExecutor(), &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)

		ch := RunSubAgent(context.Background(), "step_panic", exec, newMockContextManager(), nil, "task", events, nil)
		result := <-ch
		if result.Error == nil {
			t.Fatal("expected an error from the recovered panic")
		}
		got := events.lastCompleteErr()
		if got == "" || got != result.Error.Error() {
			t.Errorf("SubAgentComplete errMsg = %q, want non-empty %q", got, result.Error.Error())
		}
	})

	// The mutation gate rejects a finish whose Output is the model's ANSWER
	// (prose), not an "Aborted: …" message. The structured AbortReason must be
	// forwarded instead so the answer prose never surfaces as the failure cause.
	t.Run("mutation-gate rejection forwards the structured reason, not the answer prose", func(t *testing.T) {
		const answerMarker = "PROSE_ANSWER_MUST_NOT_SURFACE"
		events := &recordingEvents{}
		mockLLM := &mockLLMCaller{
			responses: []*llm.ChatResponse{
				llmResponseFinish("first attempt", answerMarker),
				llmResponseFinish("second attempt", answerMarker),
			},
		}
		exec := newExecutorDefaultHITL(mockLLM, newMockToolExecutor(), &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
		exec.SetMutationRequired(true)

		ch := RunSubAgent(context.Background(), "step_1", exec, newMockContextManager(), []tools.ToolDescriptor{
			{Name: "read_file", Description: "read", Source: "core"},
		}, "task", events, nil)
		result := <-ch
		if result.Error == nil {
			t.Fatal("expected error for mutation-gate rejection")
		}
		got := events.lastCompleteErr()
		if got == "" {
			t.Fatal("expected a non-empty failure reason from the mutation gate")
		}
		if strings.Contains(got, answerMarker) {
			t.Errorf("errMsg = %q must not carry the model answer prose", got)
		}
		if got != result.Error.Error() {
			t.Errorf("SubAgentComplete errMsg = %q, want %q", got, result.Error.Error())
		}
	})
}

func TestRunSubAgent_Paused(t *testing.T) {
	// A cooperative pause (PauseChecker trips at the first step boundary) is a
	// recoverable checkpoint, not a failure: the emitter must observe the
	// distinct SubAgentPaused event instead of SubAgentComplete(success=false),
	// while the result channel still carries ErrPaused so the host can resume.
	mockLLM := &mockLLMCaller{
		responses: []*llm.ChatResponse{
			llmResponseFinish("done", "should not reach"),
		},
	}
	cm := newMockContextManager()
	events := &recordingEvents{}
	exec := newExecutorDefaultHITL(mockLLM, newMockToolExecutor(), &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
	exec.SetPauseChecker(func(context.Context) bool { return true })

	ch := RunSubAgent(context.Background(), "step_1", exec, cm, nil, "pausable task", events, nil)
	result := <-ch

	if !errors.Is(result.Error, ErrPaused) {
		t.Fatalf("expected ErrPaused, got %v", result.Error)
	}
	var foundPaused, foundComplete bool
	for _, e := range events.events {
		switch e {
		case "SubAgentPaused:step_1":
			foundPaused = true
		case "SubAgentComplete:step_1:false":
			foundComplete = true
		}
	}
	if !foundPaused {
		t.Error("expected SubAgentPaused event for step_1")
	}
	if foundComplete {
		t.Error("SubAgentComplete must NOT be emitted for a paused sub-agent")
	}
}

func TestRunSubAgent_LLMError(t *testing.T) {
	mockLLM := &mockLLMCaller{
		errors: []error{errors.New("llm failed")},
	}
	cm := newMockContextManager()
	exec := newExecutorDefaultHITL(mockLLM, newMockToolExecutor(), &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)

	ch := RunSubAgent(context.Background(), "step_1", exec, cm, nil, "test", nil, nil)
	result := <-ch
	if result.Error == nil {
		t.Fatal("expected error")
	}
	if result.StepID != "step_1" {
		t.Errorf("StepID = %q, want %q", result.StepID, "step_1")
	}
}

func TestRunSubAgent_MaxStepsExhausted(t *testing.T) {
	// LLM always returns tool calls, never finishes
	toolInput := json.RawMessage(`{}`)
	mockLLM := &mockLLMCaller{
		responses: []*llm.ChatResponse{
			llmResponseWithToolCall("t1", "tool1", toolInput),
			llmResponseWithToolCall("t2", "tool2", json.RawMessage(`{"x":"1"}`)),
		},
	}
	cm := newMockContextManager()
	exec := newExecutorDefaultHITL(mockLLM, newMockToolExecutor(), &mockTokenCounter{}, 2, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)

	ch := RunSubAgent(context.Background(), "step_1", exec, cm, []tools.ToolDescriptor{
		{Name: "tool1", Description: "t", Source: "core"},
	}, "test", nil, nil)
	result := <-ch
	if result.Error == nil {
		t.Fatal("expected error for max steps exhaustion")
	}
	if result.StepID != "step_1" {
		t.Errorf("StepID = %q, want %q", result.StepID, "step_1")
	}
}

// panickingLLMCaller panics on every Call, simulating a nil-dereference bug
// (or any other panic) deep inside the executor's LLM path.
type panickingLLMCaller struct{}

func (panickingLLMCaller) Call(context.Context, llm.ChatRequest) (*llm.ChatResponse, error) {
	panic("boom")
}

func TestRunSubAgent_RecoversPanic(t *testing.T) {
	// A panic in sub-agent execution must not crash the process: RunSubAgent
	// recovers it, logs the stack, emits a failed completion, and returns an
	// error through the result channel so the host can mark the step failed and
	// continue.
	cm := newMockContextManager()
	events := &recordingEvents{}
	exec := newExecutorDefaultHITL(panickingLLMCaller{}, newMockToolExecutor(), &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)

	ch := RunSubAgent(context.Background(), "step_panic", exec, cm, nil, "panicking task", events, nil)
	result := <-ch

	if result.Error == nil {
		t.Fatal("expected an error from the recovered panic, got nil")
	}
	if result.StepID != "step_panic" {
		t.Errorf("StepID = %q, want %q", result.StepID, "step_panic")
	}

	// Hosts must observe a terminal SubAgentComplete(success=false) event.
	foundComplete := false
	for _, e := range events.events {
		if e == "SubAgentComplete:step_panic:false" {
			foundComplete = true
		}
	}
	if !foundComplete {
		t.Errorf("expected SubAgentComplete:step_panic:false event, got %v", events.events)
	}
}

func TestRunSubAgentsParallel_Empty(t *testing.T) {
	results := RunSubAgentsParallel(context.Background(), nil)
	if results != nil {
		t.Errorf("expected nil results, got %v", results)
	}

	results = RunSubAgentsParallel(context.Background(), []SubAgentTask{})
	if results != nil {
		t.Errorf("expected nil results for empty slice, got %v", results)
	}
}

func TestRunSubAgentsParallel_MultipleAgents(t *testing.T) {
	agents := make([]SubAgentTask, 3)
	for i := 0; i < 3; i++ {
		mockLLM := &mockLLMCaller{
			responses: []*llm.ChatResponse{
				llmResponseFinish("done", "output_"+string(rune('A'+i))),
			},
		}
		agents[i] = SubAgentTask{
			StepID:   "step_" + string(rune('1'+i)),
			Executor: newExecutorDefaultHITL(mockLLM, newMockToolExecutor(), &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig),
			CM:       newMockContextManager(),
			TaskDesc: "task " + string(rune('A'+i)),
		}
	}

	results := RunSubAgentsParallel(context.Background(), agents)
	if len(results) != 3 {
		t.Fatalf("expected 3 results, got %d", len(results))
	}

	// All should succeed
	for _, r := range results {
		if r.Error != nil {
			t.Errorf("unexpected error for %s: %v", r.StepID, r.Error)
		}
	}
}

func TestRunSubAgentsParallel_MixedResults(t *testing.T) {
	// One succeeds, one fails
	successLLM := &mockLLMCaller{
		responses: []*llm.ChatResponse{
			llmResponseFinish("ok", "success"),
		},
	}
	failLLM := &mockLLMCaller{
		errors: []error{errors.New("fail")},
	}

	agents := []SubAgentTask{
		{
			StepID:   "step_1",
			Executor: newExecutorDefaultHITL(successLLM, newMockToolExecutor(), &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig),
			CM:       newMockContextManager(),
			TaskDesc: "good task",
		},
		{
			StepID:   "step_2",
			Executor: newExecutorDefaultHITL(failLLM, newMockToolExecutor(), &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig),
			CM:       newMockContextManager(),
			TaskDesc: "bad task",
		},
	}

	results := RunSubAgentsParallel(context.Background(), agents)
	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}

	successCount := 0
	errorCount := 0
	for _, r := range results {
		if r.Error == nil {
			successCount++
		} else {
			errorCount++
		}
	}
	if successCount != 1 || errorCount != 1 {
		t.Errorf("expected 1 success and 1 error, got %d success, %d error", successCount, errorCount)
	}
}

// peakConcurrencyCaller is a shared LLMCaller that records the peak number of
// simultaneous calls and parks each call until `target` are in flight (or a
// grace period elapses), making the observation deterministic. Each subagent
// performs exactly one call, so peak in-flight calls == peak concurrent
// subagents (and, since a subagent only calls the LLM once it has acquired a
// running slot, it also equals the peak SubAgentLaunch event rate).
type peakConcurrencyCaller struct {
	target    int32
	active    int32
	maxActive int32
}

func (m *peakConcurrencyCaller) Call(_ context.Context, _ llm.ChatRequest) (*llm.ChatResponse, error) {
	cur := atomic.AddInt32(&m.active, 1)
	for {
		prev := atomic.LoadInt32(&m.maxActive)
		if cur <= prev || atomic.CompareAndSwapInt32(&m.maxActive, prev, cur) {
			break
		}
	}
	deadline := time.Now().Add(500 * time.Millisecond)
	for atomic.LoadInt32(&m.active) < m.target && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	time.Sleep(5 * time.Millisecond)
	atomic.AddInt32(&m.active, -1)
	return llmResponseFinish("done", "ok"), nil
}

// peakTaskSet builds n subagent tasks whose single LLM call shares one
// peakConcurrencyCaller, so the caller observes the true peak across all of
// them.
func peakTaskSet(n int, caller *peakConcurrencyCaller) []SubAgentTask {
	agents := make([]SubAgentTask, n)
	for i := range agents {
		agents[i] = SubAgentTask{
			StepID:   fmt.Sprintf("step_%d", i),
			Executor: newExecutorDefaultHITL(caller, newMockToolExecutor(), &mockTokenCounter{}, 5, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig),
			CM:       newMockContextManager(),
			TaskDesc: fmt.Sprintf("task %d", i),
		}
	}
	return agents
}

// TestRunSubAgentsParallel_MaxParallelSubagents proves WithMaxParallelSubagents
// bounds peak concurrency: no more than the limit run at once, results still
// come back complete and in input order, and — without the option — the fan-out
// stays unbounded (the historical behavior). This is the single chokepoint the
// host's delegate tool and plan-wave dispatcher both funnel through.
func TestRunSubAgentsParallel_MaxParallelSubagents(t *testing.T) {
	const tasks = 8

	t.Run("cap binds concurrency", func(t *testing.T) {
		const limit = 3
		caller := &peakConcurrencyCaller{target: limit}
		results := RunSubAgentsParallel(context.Background(), peakTaskSet(tasks, caller), WithMaxParallelSubagents(limit))

		if len(results) != tasks {
			t.Fatalf("got %d results, want %d", len(results), tasks)
		}
		for i, r := range results {
			if r.StepID != fmt.Sprintf("step_%d", i) {
				t.Errorf("results[%d].StepID = %q, want step_%d (input order preserved)", i, r.StepID, i)
			}
			if r.Error != nil {
				t.Errorf("step_%d error: %v", i, r.Error)
			}
		}
		if got := atomic.LoadInt32(&caller.maxActive); got != limit {
			t.Errorf("peak concurrent subagents = %d, want exactly %d", got, limit)
		}
	})

	t.Run("no cap is unbounded", func(t *testing.T) {
		caller := &peakConcurrencyCaller{target: tasks}
		results := RunSubAgentsParallel(context.Background(), peakTaskSet(tasks, caller))
		if len(results) != tasks {
			t.Fatalf("got %d results, want %d", len(results), tasks)
		}
		if got := atomic.LoadInt32(&caller.maxActive); got != tasks {
			t.Errorf("peak concurrent subagents = %d, want %d (unbounded fan-out)", got, tasks)
		}
	})
}

// concurrencyGaugeEvents wraps recordingEvents and tracks the peak number of
// subagents simultaneously mid-lifecycle, as observed through the lifecycle
// events themselves: SubAgentLaunch increments the gauge, SubAgentComplete /
// SubAgentPaused decrements it. Because a subagent only emits its launch after
// acquiring a running slot, the peak gauge value is exactly the peak rate of
// subagent lifecycle events — the quantity the concurrency cap is meant to
// bound.
type concurrencyGaugeEvents struct {
	recordingEvents

	mu     sync.Mutex
	active int
	peak   int
}

func (e *concurrencyGaugeEvents) SubAgentLaunch(stepID, description string) {
	e.mu.Lock()
	e.active++
	if e.active > e.peak {
		e.peak = e.active
	}
	e.mu.Unlock()
	e.recordingEvents.SubAgentLaunch(stepID, description)
}

func (e *concurrencyGaugeEvents) SubAgentComplete(stepID string, success bool, d time.Duration, errMsg string) {
	e.mu.Lock()
	if e.active > 0 {
		e.active--
	}
	e.mu.Unlock()
	e.recordingEvents.SubAgentComplete(stepID, success, d, errMsg)
}

func (e *concurrencyGaugeEvents) SubAgentPaused(stepID string, d time.Duration) {
	e.mu.Lock()
	if e.active > 0 {
		e.active--
	}
	e.mu.Unlock()
	e.recordingEvents.SubAgentPaused(stepID, d)
}

func (e *concurrencyGaugeEvents) peakValue() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.peak
}

// TestRunSubAgentsParallel_PeakEventRateBounded is the direct evidence for the
// "peak subagent event rate drops" property: the number of subagents
// simultaneously mid-lifecycle (launched but not yet completed) never exceeds
// the cap, so the burst of SubAgentLaunch/Complete events is bounded — and with
// the cap set it stays strictly below the unbounded fan-out.
func TestRunSubAgentsParallel_PeakEventRateBounded(t *testing.T) {
	const (
		tasks = 8
		limit = 3
	)

	events := &concurrencyGaugeEvents{}
	caller := &peakConcurrencyCaller{target: limit}
	agents := peakTaskSet(tasks, caller)
	for i := range agents {
		agents[i].Emitter = events
	}

	results := RunSubAgentsParallel(context.Background(), agents, WithMaxParallelSubagents(limit))
	if len(results) != tasks {
		t.Fatalf("got %d results, want %d", len(results), tasks)
	}
	if got := events.peakValue(); got != limit {
		t.Errorf("peak simultaneous subagent events = %d, want %d (event rate must be bounded by the cap)", got, limit)
	}
}
