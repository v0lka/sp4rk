package sp4rk

import (
	"context"
	"errors"
	"log/slog"
	"sync"
	"testing"

	"github.com/v0lka/sp4rk/orchestration"
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
