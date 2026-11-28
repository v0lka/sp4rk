package orchestration

import (
	"context"
	"strings"
	"testing"

	"github.com/v0lka/sp4rk/agent"
	"github.com/v0lka/sp4rk/llm"
)

// pendingRegistry is a minimal PendingDelegations with a fixed pending list.
type pendingRegistry struct{ ids []string }

func (p pendingRegistry) ListPending() []string { return p.ids }

// TestPendingDelegationsGuardError_NamesWaitTools pins the finish-join
// guard's recovery recipe: the wait tools are the primary join path, ahead of
// read_step_output, with cancel_delegation as the explicit opt-out.
func TestPendingDelegationsGuardError_NamesWaitTools(t *testing.T) {
	err := pendingDelegationsGuardError([]string{"sub-1", "sub-2"})
	if err == nil {
		t.Fatal("expected a non-nil guard error")
	}
	msg := err.Error()
	for _, want := range []string{
		"2 pending async delegation(s)",
		"sub-1, sub-2",
		"wait_for_all(delegates=[...])",
		"wait_for_one(delegate_id)",
		"read_step_output",
		"cancel_delegation",
	} {
		if !strings.Contains(msg, want) {
			t.Errorf("guard error missing %q:\n%s", want, msg)
		}
	}
	// The wait tools come BEFORE read_step_output in the recipe.
	if strings.Index(msg, "wait_for_all(delegates=[...])") > strings.Index(msg, "read_step_output") {
		t.Errorf("wait_for_all must be named before read_step_output:\n%s", msg)
	}
}

// TestConductor_Run_FinishGuardNudgeNamesWaitTools drives the real wiring: a
// finish issued while an async delegation is pending is rejected by the
// conductor's finish-join guard, the run continues, and the injected nudge
// the model sees names the wait tools.
func TestConductor_Run_FinishGuardNudgeNamesWaitTools(t *testing.T) {
	fakeCM := newCondFakeCM()
	cfg := ConductorConfig{
		LLM: &condMockLLM{responses: []*llm.ChatResponse{
			condFinishResponse("all done", "the answer"),
		}},
		Tools: condMockTools{},
		ContextFactory: func(_ string, _ llm.ModelMetadata, _ string, _ ...PruningOverride) agent.ContextManager {
			return fakeCM
		},
		SystemPrompt: func(_ context.Context, _ string, _ llm.ModelMetadata) string { return "system prompt" },
		MaxSteps:     10,
	}
	cond := NewConductor(cfg)

	ctx := WithDelegationRegistry(context.Background(), pendingRegistry{ids: []string{"sub-1"}})
	result, err := cond.Run(ctx, "do something", NewMapBlackboard(), nil, &agent.NoopEvents{}, "sliding_window")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result == nil {
		t.Fatal("expected a non-nil result")
	}
	var nudge string
	for _, s := range result.Steps {
		if strings.Contains(s.UserNudge, "pending async delegation") {
			nudge = s.UserNudge
			break
		}
	}
	if nudge == "" {
		t.Fatalf("expected the finish guard's rejection as a nudge step, steps=%+v", result.Steps)
	}
	for _, want := range []string{"If your toolset includes them", "wait_for_all(delegates=[...])", "wait_for_one(delegate_id)", "Otherwise use read_step_output", "sub-1"} {
		if !strings.Contains(nudge, want) {
			t.Errorf("finish-guard nudge missing %q:\n%s", want, nudge)
		}
	}
}
