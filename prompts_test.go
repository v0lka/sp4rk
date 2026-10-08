package sp4rk

import (
	"context"
	"strings"
	"testing"

	"github.com/v0lka/sp4rk/llm"
	"github.com/v0lka/sp4rk/orchestration"
	"github.com/v0lka/sp4rk/planner"
	"github.com/v0lka/sp4rk/tools"
)

func TestDefaultPromptSetNonEmpty(t *testing.T) {
	ps := DefaultPromptSet()

	for _, field := range []struct {
		name  string
		value string
	}{
		{"BasePrompt", ps.BasePrompt},
		{"PlanPreamble", ps.PlanPreamble},
		{"MultiStepGuidance", ps.MultiStepGuidance},
		{"SingleStepPreamble", ps.SingleStepPreamble},
		{"SingleStepGuidance", ps.SingleStepGuidance},
		{"ReplanPrompt", ps.ReplanPrompt},
		{"VerificationMandate", ps.VerificationMandate},
	} {
		if strings.TrimSpace(field.value) == "" {
			t.Errorf("DefaultPromptSet().%s is empty", field.name)
		}
	}
}

func TestDefaultPromptSetHasPlaceholders(t *testing.T) {
	ps := DefaultPromptSet()

	// The base prompt must reference the placeholders the planner substitutes.
	for _, placeholder := range []string{"AVAILABLE-TOOLS", "AVAILABLE-SKILLS", "MODE-PREAMBLE", "MAX-STEPS", "MODE-JSON-EXAMPLE"} {
		if !strings.Contains(ps.BasePrompt, placeholder) {
			t.Errorf("BasePrompt missing placeholder %q", placeholder)
		}
	}
	// The replan prompt must also carry the essential placeholders.
	for _, placeholder := range []string{"AVAILABLE-TOOLS", "MODE-PREAMBLE", "MODE-JSON-EXAMPLE"} {
		if !strings.Contains(ps.ReplanPrompt, placeholder) {
			t.Errorf("ReplanPrompt missing placeholder %q", placeholder)
		}
	}
}

// ---------------------------------------------------------------------------
// Slot coverage: every placeholder the planner registers must have a slot in
// the shipped templates (docs/planner.md, "Placeholder system"). A missing
// slot silently drops the substituted data from the prompt.
// ---------------------------------------------------------------------------

func TestDefaultPromptSetBasePromptHasAllDocumentedSlots(t *testing.T) {
	ps := DefaultPromptSet()

	for _, placeholder := range []string{
		// Plan-mode placeholders.
		"AVAILABLE-TOOLS", "AVAILABLE-SKILLS", "WORKSPACE-PATH",
		"MODE-PREAMBLE", "MODE-TOT", "MODE-GUIDANCE", "MODE-EXTRA-SECTIONS",
		"MODE-TAIL", "MODE-JSON-EXAMPLE", "MAX-STEPS",
		"DOMAIN-ASSIGNMENT", "AGENT-PROFILES", "RECENT-CONVERSATION",
		// Continuation placeholders (continuation mode uses BasePrompt).
		"ORIGINAL-REQUEST", "COMPLETED-PLAN-SUMMARY", "TERMINAL-STEPS",
	} {
		if !strings.Contains(ps.BasePrompt, placeholder) {
			t.Errorf("BasePrompt missing documented slot %q", placeholder)
		}
	}
}

func TestDefaultPromptSetReplanPromptHasAllDocumentedSlots(t *testing.T) {
	ps := DefaultPromptSet()

	for _, placeholder := range []string{
		"AVAILABLE-TOOLS", "MODE-PREAMBLE", "MODE-JSON-EXAMPLE", "MAX-STEPS",
		"ORIGINAL-PLAN", "COMPLETED-STEPS", "FAILED-STEP",
		"CURRENT-REFLECTION", "PREVIOUS-SESSION-REFLECTIONS",
		"AVAILABLE-SKILLS", "WORKSPACE-PATH",
	} {
		if !strings.Contains(ps.ReplanPrompt, placeholder) {
			t.Errorf("ReplanPrompt missing documented slot %q", placeholder)
		}
	}
}

// ---------------------------------------------------------------------------
// Built-prompt regression: the default replan/continuation/direct planning
// prompts must come out with every token substituted and the replan /
// failure / terminal-step context actually present.
// ---------------------------------------------------------------------------

// planPromptCapturer records the system prompt of the first LLM call and
// replies with a minimal valid one-step plan.
type planPromptCapturer struct {
	systemPrompt string
}

func (c *planPromptCapturer) Call(_ context.Context, req llm.ChatRequest) (*llm.ChatResponse, error) {
	if len(req.Messages) > 0 {
		c.systemPrompt = req.Messages[0].Content
	}
	return &llm.ChatResponse{
		Message: llm.Message{Role: "assistant", Content: `{"steps":[{"id":"step_1","summary":"s","description":"d"}]}`},
	}, nil
}

// plannerPromptTokens lists every placeholder the planner registers on the
// plan/continuation paths (trusted + data keys). None of them may survive
// into a built prompt.
var plannerPromptTokens = []string{
	"MODE-PREAMBLE", "MODE-TOT", "MODE-GUIDANCE", "MODE-EXTRA-SECTIONS",
	"MODE-TAIL", "MODE-JSON-EXAMPLE", "MAX-STEPS", "DOMAIN-ASSIGNMENT",
	"AGENT-PROFILES", "AVAILABLE-TOOLS", "AVAILABLE-SKILLS",
	"WORKSPACE-PATH", "RECENT-CONVERSATION", "REFLECTIONS",
	"ORIGINAL-REQUEST", "COMPLETED-PLAN-SUMMARY", "TERMINAL-STEPS",
}

// replanPromptTokens adds the replan-only data keys.
var replanPromptTokens = append(append([]string{}, plannerPromptTokens...),
	"ORIGINAL-PLAN", "COMPLETED-STEPS", "FAILED-STEP",
	"CURRENT-REFLECTION", "PREVIOUS-SESSION-REFLECTIONS")

func assertNoLiteralTokens(t *testing.T, prompt string, tokens []string) {
	t.Helper()
	for _, token := range tokens {
		if strings.Contains(prompt, token) {
			t.Errorf("built prompt leaked literal token %q", token)
		}
	}
}

func newDefaultPromptsPlanner(t *testing.T, caller *planPromptCapturer) *planner.Planner {
	t.Helper()
	cfg := planner.DefaultConfig()
	cfg.Prompts = DefaultPromptSet()
	pl, err := planner.NewPlanner(caller, cfg)
	if err != nil {
		t.Fatalf("NewPlanner returned error: %v", err)
	}
	return pl
}

func TestDefaultReplanPromptBuildSubstitutesAllTokens(t *testing.T) {
	caller := &planPromptCapturer{}
	pl := newDefaultPromptsPlanner(t, caller)

	originalPlan := &orchestration.Plan{Steps: []orchestration.PlanStep{
		{ID: "step_1", Description: "orig-plan-desc-marker"},
	}}
	completed := []orchestration.CompletedStep{{StepID: "step_0", Output: "completed-output-marker"}}
	failed := orchestration.CompletedStep{StepID: "step_1", Output: "failed-output-marker"}
	reflection := &orchestration.Reflection{
		FailureAnalysis: "analysis-marker",
		RootCause:       "cause-marker",
		ActionPlan:      "action-plan-marker",
	}

	if _, err := pl.Replan(context.Background(), originalPlan, completed, failed, reflection, nil, nil); err != nil {
		t.Fatalf("Replan returned error: %v", err)
	}

	got := caller.systemPrompt
	assertNoLiteralTokens(t, got, replanPromptTokens)

	// The replan context must actually be present, not just claimed.
	for _, marker := range []string{
		"orig-plan-desc-marker", "step_0", "completed-output-marker",
		"step_1", "failed-output-marker", "analysis-marker", "cause-marker",
		"action-plan-marker",
	} {
		if !strings.Contains(got, marker) {
			t.Errorf("built replan prompt lost context marker %q", marker)
		}
	}
	// The trusted pass must resolve the mode tokens (multi-step cap + JSON example).
	if !strings.Contains(got, "Create at most 10 steps") {
		t.Error("built replan prompt should resolve MAX-STEPS to the multi-step cap")
	}
	if !strings.Contains(got, `"depends_on"`) {
		t.Error("built replan prompt should embed the trusted MODE-JSON-EXAMPLE")
	}
}

func TestDefaultContinuationPromptBuildSubstitutesAllTokens(t *testing.T) {
	caller := &planPromptCapturer{}
	pl := newDefaultPromptsPlanner(t, caller)

	existing := &orchestration.Plan{Steps: []orchestration.PlanStep{
		{ID: "step_1", Description: "done step"},
		{ID: "step_2", Description: "last step", DependsOn: []string{"step_1"}},
	}}
	completed := []orchestration.CompletedStep{{StepID: "step_1", Output: "completed-output-marker"}}
	history := []llm.Message{{Role: "user", Content: "history-marker"}}

	if _, err := pl.PlanContinuation(context.Background(), "original-request-marker", existing, completed, "now add tests", nil, nil, false, history, true); err != nil {
		t.Fatalf("PlanContinuation returned error: %v", err)
	}

	got := caller.systemPrompt
	assertNoLiteralTokens(t, got, plannerPromptTokens)

	for _, marker := range []string{
		"original-request-marker", "[COMPLETED]", "step_1",
		"completed-output-marker", "step_2", "history-marker",
		"Continue the existing plan", "Create at most 10 steps",
	} {
		if !strings.Contains(got, marker) {
			t.Errorf("built continuation prompt lost marker %q", marker)
		}
	}
}

func TestDefaultPlanPromptBuildSubstitutesAllTokens(t *testing.T) {
	caller := &planPromptCapturer{}
	pl := newDefaultPromptsPlanner(t, caller)

	reflections := []orchestration.Reflection{{
		FailureAnalysis: "reflection-marker",
		RootCause:       "cause-marker",
		ActionPlan:      "action-plan-marker",
	}}
	history := []llm.Message{{Role: "user", Content: "history-marker"}}
	toolDescs := []tools.ToolDescriptor{{Name: "read_file", Description: "read a file"}}

	if _, err := pl.Plan(context.Background(), "plan the task", toolDescs, reflections, nil, false, history); err != nil {
		t.Fatalf("Plan returned error: %v", err)
	}

	got := caller.systemPrompt
	assertNoLiteralTokens(t, got, plannerPromptTokens)

	for _, marker := range []string{
		"reflection-marker", "history-marker", "read_file",
		"Create at most 10 steps",
	} {
		if !strings.Contains(got, marker) {
			t.Errorf("built plan prompt lost marker %q", marker)
		}
	}
	// Continuation-only slots must carry their descriptive defaults, not data.
	if !strings.Contains(got, "(n/a — no prior plan)") {
		t.Error("built plan prompt should fill continuation slots with the descriptive default")
	}
}

func TestDefaultReflectorPromptNonEmpty(t *testing.T) {
	p := DefaultReflectorPrompt()
	if strings.TrimSpace(p) == "" {
		t.Error("DefaultReflectorPrompt() is empty")
	}
	// Must mention the three suggested actions the reflector protocol expects.
	for _, action := range []string{"retry", "replan", "abort"} {
		if !strings.Contains(p, action) {
			t.Errorf("DefaultReflectorPrompt() missing action %q", action)
		}
	}
}
