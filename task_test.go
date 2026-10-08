package sp4rk

import (
	"context"
	"errors"
	"log/slog"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/v0lka/sp4rk/agent"
	"github.com/v0lka/sp4rk/agent/reflector"
	"github.com/v0lka/sp4rk/llm"
	"github.com/v0lka/sp4rk/orchestration"
	"github.com/v0lka/sp4rk/planner"
)

func TestTaskExecuteWithoutSystem(t *testing.T) {
	fw := testFramework(t)

	_, err := fw.TaskF(context.Background(), "do something").Execute()
	if err == nil {
		t.Fatal("expected error when no system prompt is configured")
	}
}

func TestTaskDefaults(t *testing.T) {
	fw := testFramework(t)

	b := fw.TaskF(context.Background(), "task")
	if b.maxRetries != 2 {
		t.Errorf("default maxRetries = %d, want 2", b.maxRetries)
	}
	if b.compaction != "sliding_window" {
		t.Errorf("default compaction = %q, want %q", b.compaction, "sliding_window")
	}
}

func TestTaskBuilderChaining(t *testing.T) {
	fw := testFramework(t)

	b := fw.TaskF(context.Background(), "task")
	checks := []bool{
		b.System("prompt") == b,
		b.Events(&orchestration.NoopEvents{}) == b,
		b.Plan() == b,
		b.Reflect() == b,
		b.MaxRetries(3) == b,
		b.Models("claude-sonnet-4-5", "gpt-4o") == b,
		b.Workspace("/tmp/ws") == b,
		b.Compaction("hierarchical") == b,
	}
	for i, ok := range checks {
		if !ok {
			t.Errorf("setter %d did not return the same builder", i)
		}
	}

	if !b.usePlanner {
		t.Error("Plan() should set usePlanner")
	}
	if !b.useReflector {
		t.Error("Reflect() should set useReflector")
	}
	if b.maxRetries != 3 {
		t.Errorf("maxRetries = %d, want 3", b.maxRetries)
	}
	if b.planModel != "claude-sonnet-4-5" || b.execModel != "gpt-4o" {
		t.Errorf("models = %q/%q, want claude-sonnet-4-5/gpt-4o", b.planModel, b.execModel)
	}
	if b.workspace != "/tmp/ws" {
		t.Errorf("workspace = %q, want /tmp/ws", b.workspace)
	}
	if b.compaction != "hierarchical" {
		t.Errorf("compaction = %q, want hierarchical", b.compaction)
	}
}

func TestResolvePlannerUsesDefaults(t *testing.T) {
	fw := testFramework(t)

	b := fw.TaskF(context.Background(), "task").Plan()

	pl, err := b.resolvePlanner(context.Background())
	if err != nil {
		t.Fatalf("resolvePlanner: %v", err)
	}
	if pl == nil {
		t.Fatal("resolvePlanner returned nil planner")
	}
	// The default planner should carry the fluent DefaultPromptSet.
	if pl.Cfg.Prompts.BasePrompt != defaultBasePrompt {
		t.Error("default planner does not use the fluent DefaultPromptSet")
	}
	// Model should be resolved from the active router model.
	if pl.Cfg.Model == "" {
		t.Error("default planner Model should be resolved from the active model")
	}
}

func TestResolveReflectorDisabledByDefault(t *testing.T) {
	fw := testFramework(t)

	b := fw.TaskF(context.Background(), "task")
	if rf := b.resolveReflector(); rf != nil {
		t.Error("resolveReflector should return nil when reflection is disabled")
	}
}

func TestResolveReflectorEnabled(t *testing.T) {
	fw := testFramework(t)

	b := fw.TaskF(context.Background(), "task").Reflect()
	if rf := b.resolveReflector(); rf == nil {
		t.Error("resolveReflector should return a reflector when Reflect() is set")
	}
}

// TestPlanCompletionStatus_Success — all steps completed cleanly → success.
func TestPlanCompletionStatus_Success(t *testing.T) {
	plan := &orchestration.Plan{Steps: []orchestration.PlanStep{
		{ID: "a"},
		{ID: "b", DependsOn: []string{"a"}},
	}}
	completed := map[string]orchestration.CompletedStep{
		"a": {StepID: "a", Output: "done a"},
		"b": {StepID: "b", Output: "done b"},
	}
	status, failed, err := planCompletionStatus(completed, plan, false, false)
	if status != orchestration.ExecutionStatusSuccess {
		t.Errorf("status = %q, want success", status)
	}
	if failed != 0 {
		t.Errorf("failed = %d, want 0", failed)
	}
	if err != nil {
		t.Errorf("err = %v, want nil", err)
	}
}

// TestPlanCompletionStatus_FailedPrecedenceOverPartial — a step fails and its
// dependent is never attempted (cascade). This is "failed" (something was
// attempted), NOT "partial", so the partial/cycle detection must not mask a
// genuine failure.
func TestPlanCompletionStatus_FailedPrecedenceOverPartial(t *testing.T) {
	plan := &orchestration.Plan{Steps: []orchestration.PlanStep{
		{ID: "a"},
		{ID: "b", DependsOn: []string{"a"}},
	}}
	completed := map[string]orchestration.CompletedStep{
		"a": {StepID: "a", Error: errors.New("boom")},
	}
	status, failed, err := planCompletionStatus(completed, plan, false, false)
	if status != orchestration.ExecutionStatusFailed {
		t.Errorf("status = %q, want failed (not partial)", status)
	}
	if failed != 1 {
		t.Errorf("failed = %d, want 1", failed)
	}
	if err != nil {
		t.Errorf("err = %v, want nil for failed", err)
	}
}

// TestPlanCompletionStatus_AbortedPrecedence — aborted wins over both failed
// and partial; the failed count is still reported.
func TestPlanCompletionStatus_AbortedPrecedence(t *testing.T) {
	plan := &orchestration.Plan{Steps: []orchestration.PlanStep{
		{ID: "a"},
		{ID: "b", DependsOn: []string{"a"}},
	}}
	completed := map[string]orchestration.CompletedStep{
		"a": {StepID: "a", Error: errors.New("failed then aborted")},
	}
	status, failed, err := planCompletionStatus(completed, plan, true, false)
	if status != orchestration.ExecutionStatusAborted {
		t.Errorf("status = %q, want aborted", status)
	}
	if failed != 1 {
		t.Errorf("failed = %d, want 1", failed)
	}
	if err != nil {
		t.Errorf("err = %v, want nil for aborted", err)
	}
}

// TestPlanCompletionStatus_PartialOnUnattemptedSteps — the core bug fix: a
// cyclic or dangling dependency graph leaves steps unattempted with no
// failures, which must surface as partial + ErrExecutionIncomplete instead of
// a false "success".
func TestPlanCompletionStatus_PartialOnUnattemptedSteps(t *testing.T) {
	// Cycle: neither step can ever become ready.
	plan := &orchestration.Plan{Steps: []orchestration.PlanStep{
		{ID: "a", DependsOn: []string{"b"}},
		{ID: "b", DependsOn: []string{"a"}},
	}}
	status, failed, err := planCompletionStatus(map[string]orchestration.CompletedStep{}, plan, false, false)
	if status != orchestration.ExecutionStatusPartial {
		t.Errorf("status = %q, want partial", status)
	}
	if failed != 0 {
		t.Errorf("failed = %d, want 0", failed)
	}
	if !errors.Is(err, orchestration.ErrExecutionIncomplete) {
		t.Errorf("err = %v, want a value wrapping ErrExecutionIncomplete", err)
	}
}

// TestPlanCompletionStatus_PausedPrecedence — a cooperative pause tripping
// mid-run is a recoverable checkpoint, not a failure. It must surface as
// paused (over partial, since some steps remain unattempted) and report no
// failed steps and no ErrExecutionIncomplete (a pause is intentional, not an
// incomplete plan).
func TestPlanCompletionStatus_PausedPrecedence(t *testing.T) {
	plan := &orchestration.Plan{Steps: []orchestration.PlanStep{
		{ID: "a"},
		{ID: "b", DependsOn: []string{"a"}},
	}}
	completed := map[string]orchestration.CompletedStep{
		"a": {StepID: "a", Output: "done a"},
		// "b" never attempted because the run paused after "a".
	}
	status, failed, err := planCompletionStatus(completed, plan, false, true)
	if status != orchestration.ExecutionStatusPaused {
		t.Errorf("status = %q, want paused", status)
	}
	if failed != 0 {
		t.Errorf("failed = %d, want 0 (a pause is not a failure)", failed)
	}
	if err != nil {
		t.Errorf("err = %v, want nil for paused (intentional checkpoint, not incomplete)", err)
	}
}

// TestCompletedInOrder — the helper that feeds Planner.Replan's completed-steps
// argument must return steps in plan order (not map iteration order) so the
// replan prompt observes a stable sequence, and must omit steps not present in
// the completed map (e.g. the failed step left out for re-execution).
func TestCompletedInOrder(t *testing.T) {
	plan := &orchestration.Plan{Steps: []orchestration.PlanStep{
		{ID: "a"}, {ID: "b"}, {ID: "c"}, {ID: "d"},
	}}
	// Insert in non-plan order to ensure ordering is by the plan, not the map.
	completed := map[string]orchestration.CompletedStep{
		"d": {StepID: "d", Output: "out d"},
		"b": {StepID: "b", Output: "out b"},
		"a": {StepID: "a", Output: "out a"},
		// "c" deliberately absent — the failed step is left out for re-execution.
	}
	got := completedInOrder(completed, plan)
	if len(got) != 3 {
		t.Fatalf("len = %d, want 3", len(got))
	}
	wantIDs := []string{"a", "b", "d"}
	for i, want := range wantIDs {
		if got[i].StepID != want {
			t.Errorf("got[%d].StepID = %q, want %q (plan order)", i, got[i].StepID, want)
		}
	}
}

// TestCompletedInOrder_NilPlan — with no plan to order by, every completed
// step is returned (order unspecified), none dropped.
func TestCompletedInOrder_NilPlan(t *testing.T) {
	completed := map[string]orchestration.CompletedStep{
		"x": {StepID: "x"},
		"y": {StepID: "y"},
	}
	got := completedInOrder(completed, nil)
	if len(got) != 2 {
		t.Fatalf("len = %d, want 2", len(got))
	}
}

// recordingCheckpointer is a minimal orchestration.Checkpointer that counts
// SaveCheckpoint calls (LoadCheckpoint/DeleteCheckpoint are no-op stubs).
type recordingCheckpointer struct {
	mu    sync.Mutex
	saves int
}

func (r *recordingCheckpointer) SaveCheckpoint(context.Context, string, orchestration.Blackboard) error {
	r.mu.Lock()
	r.saves++
	r.mu.Unlock()
	return nil
}
func (r *recordingCheckpointer) LoadCheckpoint(context.Context, string) (orchestration.Blackboard, error) {
	return nil, nil
}
func (r *recordingCheckpointer) DeleteCheckpoint(context.Context, string) error { return nil }

// TestFramework_NewBlackboard_WiresCheckpointerAndNotification verifies the
// shared blackboard factory honours Config.Checkpointer and
// Config.OnBlackboardChanged. This is the wiring the fluent
// TaskBuilder.Execute path previously dropped (it always built an in-memory
// MapBlackboard), so a crash-resume/notifications config set on the Framework
// was silently ignored via TaskF. Both Execute paths now route through
// newBlackboard.
func TestFramework_NewBlackboard_WiresCheckpointerAndNotification(t *testing.T) {
	cp := &recordingCheckpointer{}
	var changed []string
	fw := &Framework{
		cfg: Config{
			Checkpointer:        cp,
			OnBlackboardChanged: func(ct string) { changed = append(changed, ct) },
		},
		logger: slog.Default(),
	}

	bb, shutdown := fw.newBlackboard("test")
	defer shutdown()
	if _, ok := bb.(*orchestration.CheckpointedBlackboard); !ok {
		t.Fatalf("newBlackboard with Checkpointer = %T, want *CheckpointedBlackboard", bb)
	}

	// A mutation must fire OnBlackboardChanged synchronously and enqueue a
	// checkpoint; flushing via shutdown persists it.
	bb.SetOriginalRequest("hello-task")
	shutdown()

	if len(changed) == 0 {
		t.Error("OnBlackboardChanged never fired; the TaskF path was wiring neither Checkpointer nor OnBlackboardChanged")
	}
	if cp.saves == 0 {
		t.Error("Checkpointer.SaveCheckpoint never called; persistence is not wired through newBlackboard")
	}
}

// TestFramework_NewBlackboard_NoCheckpointer verifies the in-memory fallback:
// without a Checkpointer, newBlackboard returns a plain MapBlackboard and a
// no-op shutdown.
func TestFramework_NewBlackboard_NoCheckpointer(t *testing.T) {
	fw := &Framework{cfg: Config{}, logger: slog.Default()}
	bb, shutdown := fw.newBlackboard("test")
	shutdown() // must be a safe no-op
	if _, ok := bb.(*orchestration.MapBlackboard); !ok {
		t.Fatalf("newBlackboard without Checkpointer = %T, want *MapBlackboard", bb)
	}
}

// --- Fakes for the runPlanned replan tests ---

// scriptedCaller is an agent.LLMCaller returning canned responses in
// sequence, repeating the last one; with err set every call fails.
type scriptedCaller struct {
	mu    sync.Mutex
	resp  []*llm.ChatResponse
	calls int
	err   error
}

func (s *scriptedCaller) Call(_ context.Context, _ llm.ChatRequest) (*llm.ChatResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.calls++
	if s.err != nil {
		return nil, s.err
	}
	if len(s.resp) == 0 {
		return nil, errors.New("scriptedCaller: no canned responses")
	}
	i := s.calls - 1
	if i >= len(s.resp) {
		i = len(s.resp) - 1
	}
	return s.resp[i], nil
}

func (s *scriptedCaller) callCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.calls
}

// replanEventsRecorder counts OnReplanFailed fires, delegating everything
// else to NoopEvents.
type replanEventsRecorder struct {
	*orchestration.NoopEvents
	replanFailed atomic.Int32
}

func (e *replanEventsRecorder) OnReplanFailed(_ error) { e.replanFailed.Add(1) }

// noopContextManager is a minimal agent.ContextManager for conductor-based
// tests that never reach real prompt building.
type noopContextManager struct{}

func (noopContextManager) BuildPrompt() []llm.Message {
	return []llm.Message{{Role: "system", Content: "sys"}}
}
func (noopContextManager) AddStep(agent.Step)                              {}
func (noopContextManager) Compact(context.Context) *agent.CompactionResult { return nil }
func (noopContextManager) SetStrategy(agent.CompactionStrategy)            {}
func (noopContextManager) CheckFill() agent.FillCheck {
	return agent.FillCheck{Percent: 1, Status: "ok", Used: 1, Max: 100000}
}
func (noopContextManager) CorrectTokenCount(int)                       {}
func (noopContextManager) FillPercent() float64                        { return 1 }
func (noopContextManager) AvailableTokens() int                        { return 100000 }
func (noopContextManager) OutputLimit() int                            { return 4096 }
func (noopContextManager) VulnerableOutputs() []agent.VulnerableOutput { return nil }

// newReplanTestBuilder assembles a TaskBuilder wired to scripted planner and
// reflector callers plus a conductor whose step-execution caller always
// fails — the deterministic stand-in for a step the LLM cannot complete. It
// returns the planner and reflector callers so tests can assert call counts.
func newReplanTestBuilder(t *testing.T, b *TaskBuilder, planResp, reflectResps []string) (planCaller, reflectCaller *scriptedCaller) {
	t.Helper()

	toResponses := func(bodies []string) []*llm.ChatResponse {
		out := make([]*llm.ChatResponse, len(bodies))
		for i, body := range bodies {
			out[i] = &llm.ChatResponse{
				Message:    llm.Message{Role: "assistant", Content: body},
				StopReason: "end_turn",
			}
		}
		return out
	}

	planCaller = &scriptedCaller{resp: toResponses(planResp)}
	plCfg := planner.DefaultConfig()
	plCfg.Model = "test-model"
	pl, err := planner.NewPlanner(planCaller, plCfg)
	if err != nil {
		t.Fatalf("planner.NewPlanner: %v", err)
	}
	reflectCaller = &scriptedCaller{resp: toResponses(reflectResps)}
	rf := reflector.New(reflectCaller, reflector.Config{SystemPrompt: "analyze failures"})

	b.Planner(pl).Reflector(rf).Events(&replanEventsRecorder{NoopEvents: &orchestration.NoopEvents{}})
	return planCaller, reflectCaller
}

const replanTestPlanS1 = `{"steps":[{"id":"s1","summary":"s","description":"do s1","depends_on":[]}]}`
const replanTestPlanS1R = `{"steps":[{"id":"s1r","summary":"s","description":"do s1 again","depends_on":[]}]}`
const replanTestReflectionReplan = `{"summary":"broken","root_cause":"plan flawed","suggested_action":"replan","action_plan":"re-derive"}`
const replanTestReflectionRetry = `{"summary":"broken","root_cause":"transient","suggested_action":"retry"}`

// TestTaskReplanBudgetExhaustionStopsLoop is the regression test for the
// unbounded replan loop: a deterministically failing step whose reflector
// always answers "replan" used to cycle fail→re-plan→fail forever (each pass
// leaving the step un-completed so FindReadySteps re-selected it), hanging
// Execute and burning LLM budget. The replan budget must cap the adopted
// replans and terminate with a partial result.
func TestTaskReplanBudgetExhaustionStopsLoop(t *testing.T) {
	fw := testFramework(t)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	b := fw.TaskF(ctx, "impossible task").System("sys")
	// Plan on call 1; every Replan on calls 2+ re-derives the (renamed)
	// single-step plan. The reflector always recommends "replan".
	planCaller, reflectCaller := newReplanTestBuilder(t, b,
		[]string{replanTestPlanS1, replanTestPlanS1R},
		[]string{replanTestReflectionReplan},
	)

	// The step-execution caller always fails: every conductor.Run errors on
	// its first attempt, driving the reflect→replan cycle.
	conductor := orchestration.NewConductor(orchestration.ConductorConfig{
		LLM: &scriptedCaller{err: errors.New("exec backend down")},
		ContextFactory: func(_ string, _ llm.ModelMetadata, _ string, _ ...orchestration.PruningOverride) agent.ContextManager {
			return noopContextManager{}
		},
		SystemPrompt: func(_ context.Context, _ string, _ llm.ModelMetadata) string { return "sys" },
		MaxSteps:     5,
	})

	// Default budget: 3 adopted replans. Each adopted pass proposes one more
	// replan, so the planner is called 5 times (1 Plan + 4 Replans: 3 adopted
	// + 1 proposed after the budget was exhausted and refused), and the
	// refusal is reported via OnReplanFailed.
	res, execErr := b.runPlanned(ctx, conductor, orchestration.NewMapBlackboard(), nil)

	if !errors.Is(execErr, orchestration.ErrExecutionIncomplete) {
		t.Errorf("execErr = %v, want ErrExecutionIncomplete (replan budget must end in a partial result)", execErr)
	}
	if res == nil || res.Status != orchestration.ExecutionStatusPartial {
		t.Errorf("Status = %v, want %v", res, orchestration.ExecutionStatusPartial)
	}
	// The old unbounded loop would keep replanning until the context deadline
	// and blow far past this count.
	if got := planCaller.callCount(); got != 5 {
		t.Errorf("planner calls = %d, want 5 (1 Plan + 3 adopted Replans + 1 refused Replan)", got)
	}
	if got := reflectCaller.callCount(); got != 4 {
		t.Errorf("reflector calls = %d, want 4 (one per DAG pass)", got)
	}
	if rec, ok := b.events.(*replanEventsRecorder); !ok || rec.replanFailed.Load() != 1 {
		t.Errorf("OnReplanFailed fired %v times, want exactly 1 (budget refusal)", b.events)
	}
}

// TestTaskReplanKeepsCompletedMapNonNil is the regression test for the nil
// completed map: a replan whose new plan preserves no completed step made
// BuildCarryForward return nil, and the next runStep write to completed
// panicked ("assignment to entry in nil map"). Here the first pass fails and
// replans to a renamed step; the second pass then exhausts its retries and
// records the failure — which must be a clean failed status, not a panic.
func TestTaskReplanKeepsCompletedMapNonNil(t *testing.T) {
	fw := testFramework(t)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	b := fw.TaskF(ctx, "flaky plan").System("sys")
	// Reflect #1 recommends replan (adopting a renamed plan that preserves no
	// completed step); later reflections recommend retry so pass 2 exhausts
	// its retry budget and records the step failure.
	_, _ = newReplanTestBuilder(t, b,
		[]string{replanTestPlanS1, replanTestPlanS1R},
		[]string{replanTestReflectionReplan, replanTestReflectionRetry, replanTestReflectionRetry},
	)

	conductor := orchestration.NewConductor(orchestration.ConductorConfig{
		LLM: &scriptedCaller{err: errors.New("exec backend down")},
		ContextFactory: func(_ string, _ llm.ModelMetadata, _ string, _ ...orchestration.PruningOverride) agent.ContextManager {
			return noopContextManager{}
		},
		SystemPrompt: func(_ context.Context, _ string, _ llm.ModelMetadata) string { return "sys" },
		MaxSteps:     5,
	})

	res, execErr := b.runPlanned(ctx, conductor, orchestration.NewMapBlackboard(), nil)

	// Without the non-nil guard this write path panicked before returning.
	if res == nil || res.Status != orchestration.ExecutionStatusFailed {
		t.Errorf("Status = %v, want %v", res, orchestration.ExecutionStatusFailed)
	}
	if execErr != nil {
		t.Errorf("execErr = %v, want nil (a failed step is a status, not an error)", execErr)
	}
	if res.FailedSteps != 1 {
		t.Errorf("FailedSteps = %d, want 1", res.FailedSteps)
	}
}

// TestResolvePlannerWiresFrameworkRegistry proves the default planner built
// by TaskBuilder.resolvePlanner carries the framework tool registry, so the
// replan prompt's AVAILABLE-TOOLS section lists real tools on the default
// fluent path (TaskF(...).Plan().Reflect().Execute()) instead of shipping an
// empty listing.
func TestResolvePlannerWiresFrameworkRegistry(t *testing.T) {
	fw := testFramework(t)
	b := fw.TaskF(context.Background(), "task").System("sys")

	pl, err := b.resolvePlanner(context.Background())
	if err != nil {
		t.Fatalf("resolvePlanner: %v", err)
	}
	if pl.Cfg.ToolRegistry == nil {
		t.Fatal("expected the framework tool registry wired into the default planner")
	}
	listing := pl.Cfg.ToolRegistry.List()
	if len(listing) == 0 {
		t.Fatal("expected a non-empty tool inventory on the default planner")
	}
	found := false
	for _, d := range listing {
		if d.Name == "finish" {
			found = true
			break
		}
	}
	if !found {
		t.Error("expected the auto-registered finish tool in the planner's registry listing")
	}
	// The replan prompt embeds exactly this rendering; it must not be empty.
	if got := agent.BuildGroupedToolList(listing); got == "" {
		t.Error("expected non-empty grouped tool list for the replan AVAILABLE-TOOLS section")
	}
}
