package planner

import (
	"context"
	"strings"
	"testing"

	"github.com/v0lka/sp4rk/orchestration"
	"github.com/v0lka/sp4rk/tools"
)

// ---------------------------------------------------------------------------
// #49 — NewPlanner must not accept a Config whose injected context functions
// are nil: the plan paths dereference them unconditionally, so the first Plan
// call used to panic the host process.
// ---------------------------------------------------------------------------

func TestNewPlanner_ZeroConfigDefaultsContextFuncs(t *testing.T) {
	caller := &mockLLMCaller{
		resp: newPlanResponse(`{"steps":[{"id":"step_1","summary":"s","description":"d"}]}`),
	}
	p, err := NewPlanner(caller, Config{})
	if err != nil {
		t.Fatalf("NewPlanner with zero Config returned error: %v", err)
	}

	if p.Cfg.DomainFromContext == nil {
		t.Error("DomainFromContext left nil by NewPlanner")
	}
	if p.Cfg.ComplexityFromContext == nil {
		t.Error("ComplexityFromContext left nil by NewPlanner")
	}
	if p.Cfg.UserSkillsFromContext == nil {
		t.Error("UserSkillsFromContext left nil by NewPlanner")
	}
	if p.Cfg.FormatSkillList == nil {
		t.Error("FormatSkillList left nil by NewPlanner")
	}
	if p.Cfg.FormatWorkspacePath == nil {
		t.Error("FormatWorkspacePath left nil by NewPlanner")
	}
	if p.Cfg.AppendContextSections == nil {
		t.Error("AppendContextSections left nil by NewPlanner")
	}
	if p.Cfg.MaxExploreSteps != defaultMaxExploreSteps {
		t.Errorf("MaxExploreSteps = %d, want %d", p.Cfg.MaxExploreSteps, defaultMaxExploreSteps)
	}

	// Regression: the first Plan call with a hand-assembled Config used to
	// panic on the nil DomainFromContext dereference.
	plan, err := p.Plan(context.Background(), "do the thing", nil, nil, nil, false, nil)
	if err != nil {
		t.Fatalf("Plan with defaulted zero-config funcs returned error: %v", err)
	}
	if plan == nil || len(plan.Steps) != 1 {
		t.Fatalf("expected a 1-step plan, got %+v", plan)
	}
}

func TestNewPlanner_ExplicitFuncsNotOverridden(t *testing.T) {
	domain := func(context.Context) string { return "code" }
	p, err := NewPlanner(&mockLLMCaller{}, Config{DomainFromContext: domain})
	if err != nil {
		t.Fatalf("NewPlanner returned error: %v", err)
	}
	// Comparing function pointers is not possible in Go; a call must show the
	// injected implementation survived defaulting.
	if got := p.Cfg.DomainFromContext(context.Background()); got != "code" {
		t.Errorf("DomainFromContext = %q, want injected %q", got, "code")
	}
}

// ---------------------------------------------------------------------------
// #38 — the continuation contract lets depends_on reference the prior plan's
// terminal steps; validatePlanDAG must accept them (and still reject unknown
// IDs, duplicates, and cycles).
// ---------------------------------------------------------------------------

func TestPlanContinuation_DependsOnPriorTerminalStep(t *testing.T) {
	caller := &mockLLMCaller{
		resp: newPlanResponse(`{"steps":[{"id":"continuation_1","summary":"continue","description":"next","depends_on":["step_2"]}]}`),
	}
	cfg := makeTestConfig("base")
	cfg.Prompts.ContinuationPreamble = "cont-preamble"
	p := &Planner{llm: caller, Cfg: cfg}

	existing := &orchestration.Plan{Steps: []orchestration.PlanStep{
		{ID: "step_1", Description: "first"},
		{ID: "step_2", Description: "second", DependsOn: []string{"step_1"}},
	}}

	plan, err := p.PlanContinuation(context.Background(), "original", existing, nil, "continue work", nil, nil, false, nil, true)
	if err != nil {
		t.Fatalf("continuation step depending on prior plan's terminal step was rejected: %v", err)
	}
	if len(plan.Steps) != 1 {
		t.Fatalf("expected 1 step, got %d", len(plan.Steps))
	}
}

func TestPlanContinuation_UnknownDepStillRejected(t *testing.T) {
	caller := &mockLLMCaller{
		resp: newPlanResponse(`{"steps":[{"id":"continuation_1","summary":"continue","description":"next","depends_on":["nonexistent_step"]}]}`),
	}
	cfg := makeTestConfig("base")
	cfg.Prompts.ContinuationPreamble = "cont-preamble"
	p := &Planner{llm: caller, Cfg: cfg}

	existing := &orchestration.Plan{Steps: []orchestration.PlanStep{{ID: "step_1", Description: "done"}}}

	_, err := p.PlanContinuation(context.Background(), "original", existing, nil, "continue work", nil, nil, false, nil, true)
	if err == nil {
		t.Fatal("expected a depends_on entry on an unknown ID to be rejected")
	}
	if !strings.Contains(err.Error(), "nonexistent_step") {
		t.Errorf("error should name the unknown dependency, got: %v", err)
	}
}

func TestValidatePlanDAG_AcceptsPriorPlanIDs(t *testing.T) {
	prior := map[string]bool{"step_1": true, "step_2": true}

	// A prior-plan terminal dep plus an in-plan dep chain: accepted.
	ok := &orchestration.Plan{Steps: []orchestration.PlanStep{
		{ID: "c1", DependsOn: []string{"step_2"}},
		{ID: "c2", DependsOn: []string{"c1"}},
	}}
	if err := validatePlanDAG(ok, prior); err != nil {
		t.Errorf("cross-plan continuation deps should validate, got: %v", err)
	}

	// Unknown ID (neither in-plan nor prior) still rejected.
	ghost := &orchestration.Plan{Steps: []orchestration.PlanStep{
		{ID: "c1", DependsOn: []string{"ghost"}},
	}}
	if err := validatePlanDAG(ghost, prior); err == nil {
		t.Error("unknown dependency should be rejected even with prior IDs known")
	}

	// Without prior IDs the same cross-plan dep is rejected (strict in-plan
	// validation for non-continuation paths).
	if err := validatePlanDAG(ok, nil); err == nil {
		t.Error("cross-plan dependency should be rejected when no prior IDs are supplied")
	}

	// Duplicate IDs still rejected with prior IDs present.
	dup := &orchestration.Plan{Steps: []orchestration.PlanStep{
		{ID: "c1"}, {ID: "c1"},
	}}
	if err := validatePlanDAG(dup, prior); err == nil {
		t.Error("duplicate step ID should be rejected even with prior IDs known")
	}

	// In-plan cycles still detected with prior IDs present; a prior ID
	// participates in no cycle (it is an external anchor).
	cycle := &orchestration.Plan{Steps: []orchestration.PlanStep{
		{ID: "c1", DependsOn: []string{"step_1", "c2"}},
		{ID: "c2", DependsOn: []string{"c1"}},
	}}
	if err := validatePlanDAG(cycle, prior); err == nil {
		t.Error("in-plan cycle should be detected even with prior IDs known")
	}
}

// ---------------------------------------------------------------------------
// #38c — planRetryHint must classify DAG-validation errors as validation
// failures, not as "invalid JSON".
// ---------------------------------------------------------------------------

func TestPlanRetryHint_ClassifiesDAGValidation(t *testing.T) {
	p := &Planner{}

	_, dagErr := p.parsePlanResponse(`{"steps":[{"id":"c1","depends_on":["ghost"]}]}`, nil)
	if dagErr == nil {
		t.Fatal("expected DAG validation to fail for unknown dependency")
	}
	hint := p.planRetryHint(dagErr)
	if strings.Contains(hint, "invalid JSON") {
		t.Errorf("DAG-validation error mislabeled as invalid JSON in hint: %s", hint)
	}
	if !strings.Contains(hint, "failed validation") {
		t.Errorf("hint should describe a validation failure, got: %s", hint)
	}
	if !strings.Contains(hint, "ghost") {
		t.Errorf("hint must carry the underlying error verbatim, got: %s", hint)
	}

	_, jsonErr := p.parsePlanResponse("not json at all", nil)
	if jsonErr == nil {
		t.Fatal("expected parse failure for non-JSON response")
	}
	if hint := p.planRetryHint(jsonErr); !strings.Contains(hint, "invalid JSON") {
		t.Errorf("JSON-syntax error should keep the invalid-JSON hint, got: %s", hint)
	}

	zeroHint := p.planRetryHint(&planParseError{err: errPlanZeroSteps})
	if !strings.Contains(zeroHint, "zero steps") {
		t.Errorf("zero-steps hint changed, got: %s", zeroHint)
	}
}

// ---------------------------------------------------------------------------
// #39 — the replan prompt must go through the trusted substitution pass, so
// its MODE-* / MAX-STEPS / AVAILABLE-TOOLS tokens resolve, and through the
// shared data pass, so the replan context appears.
// ---------------------------------------------------------------------------

func TestBuildReplanSystemPrompt_RunsTrustedAndDataPasses(t *testing.T) {
	cfg := makeTestConfig("")
	cfg.Prompts.ReplanPrompt = "Tools: AVAILABLE-TOOLS | Preamble: MODE-PREAMBLE | Max: MAX-STEPS | " +
		"Skills: AVAILABLE-SKILLS | Plan: ORIGINAL-PLAN | Completed: COMPLETED-STEPS | Failed: FAILED-STEP | " +
		"Reflection: CURRENT-REFLECTION | Example: MODE-JSON-EXAMPLE"
	cfg.ToolRegistry = &mockToolRegistry{tools: []tools.ToolDescriptor{
		{Name: "read_file", Description: "read a file"},
	}}
	p := &Planner{llm: &mockLLMCaller{}, Cfg: cfg}

	rc := replanContext{
		originalPlan: &orchestration.Plan{Steps: []orchestration.PlanStep{
			{ID: "step_1", Description: "orig-plan-desc-marker"},
		}},
		completedSteps: []orchestration.CompletedStep{{StepID: "step_0", Output: "done-marker"}},
		failedStep:     orchestration.CompletedStep{StepID: "step_1", Output: "boom-marker"},
		reflection:     &orchestration.Reflection{FailureAnalysis: "analysis-marker", RootCause: "cause-marker", ActionPlan: "action-marker"},
	}
	got := p.buildReplanSystemPrompt(context.Background(), rc)

	for _, token := range []string{
		"AVAILABLE-TOOLS", "MODE-PREAMBLE", "MODE-JSON-EXAMPLE", "MAX-STEPS",
		"ORIGINAL-PLAN", "COMPLETED-STEPS", "FAILED-STEP", "CURRENT-REFLECTION",
		"AVAILABLE-SKILLS",
	} {
		if strings.Contains(got, token) {
			t.Errorf("replan prompt leaked literal token %q", token)
		}
	}
	if !strings.Contains(got, "read_file") {
		t.Error("replan prompt should list registry-sourced tools")
	}
	if !strings.Contains(got, replanModePreamble) {
		t.Error("replan prompt should carry the trusted MODE-PREAMBLE substitution")
	}
	if !strings.Contains(got, "Max: 10") {
		t.Errorf("replan prompt should resolve MAX-STEPS to the multi-step cap, got: %s", got)
	}
	if !strings.Contains(got, planModeJSONExample) {
		t.Error("replan prompt should carry the trusted MODE-JSON-EXAMPLE substitution")
	}
	for _, marker := range []string{"orig-plan-desc-marker", "done-marker", "boom-marker", "analysis-marker"} {
		if !strings.Contains(got, marker) {
			t.Errorf("replan prompt lost replan context marker %q", marker)
		}
	}
}

// ---------------------------------------------------------------------------
// #62 — the continuation prompt built from a base template carrying the
// documented slots must render the prior-plan context, and the direct-plan
// prompt must fill the continuation-only slots with descriptive defaults
// instead of leaking literal tokens.
// ---------------------------------------------------------------------------

func TestBuildContinuationSystemPrompt_RendersContinuationData(t *testing.T) {
	cfg := makeTestConfig("REQ: ORIGINAL-REQUEST | SUMMARY: COMPLETED-PLAN-SUMMARY | TERMINAL: TERMINAL-STEPS | CONV: RECENT-CONVERSATION")
	cfg.Prompts.ContinuationPreamble = "cont-preamble"
	p := &Planner{llm: &mockLLMCaller{}, Cfg: cfg}

	existing := &orchestration.Plan{Steps: []orchestration.PlanStep{
		{ID: "step_1", Description: "done step"},
		{ID: "step_2", Description: "last step", DependsOn: []string{"step_1"}},
	}}
	completed := []orchestration.CompletedStep{{StepID: "step_1", Output: "ok"}}

	got := p.buildContinuationSystemPrompt(context.Background(), p.continuationMultiMode(), "original request text", existing, completed, nil, nil, nil)

	if !strings.Contains(got, "original request text") {
		t.Error("continuation prompt lost ORIGINAL-REQUEST data")
	}
	if !strings.Contains(got, "[COMPLETED]") || !strings.Contains(got, "step_1") {
		t.Error("continuation prompt lost COMPLETED-PLAN-SUMMARY data")
	}
	if !strings.Contains(got, "TERMINAL: step_2") {
		t.Errorf("continuation prompt should list the terminal step ID under TERMINAL-STEPS, got: %s", got)
	}
	if !strings.Contains(got, "(no previous conversation)") {
		t.Error("continuation prompt should render the empty-conversation default for RECENT-CONVERSATION")
	}
	for _, token := range []string{"ORIGINAL-REQUEST", "COMPLETED-PLAN-SUMMARY", "TERMINAL-STEPS", "RECENT-CONVERSATION"} {
		if strings.Contains(got, token) {
			t.Errorf("continuation prompt leaked literal token %q", token)
		}
	}
}

func TestBuildPlanSystemPrompt_DefaultsContinuationSlots(t *testing.T) {
	cfg := makeTestConfig("REQ: ORIGINAL-REQUEST | SUMMARY: COMPLETED-PLAN-SUMMARY | TERMINAL: TERMINAL-STEPS | CONV: RECENT-CONVERSATION")
	p := &Planner{llm: &mockLLMCaller{}, Cfg: cfg}

	got := p.buildPlanSystemPrompt(context.Background(), makeTestMode(), nil, nil, nil, nil)

	if !strings.Contains(got, "(n/a — no prior plan)") {
		t.Errorf("direct-plan prompt should fill continuation slots with the descriptive default, got: %s", got)
	}
	if !strings.Contains(got, "(no previous conversation)") {
		t.Error("direct-plan prompt should fill RECENT-CONVERSATION with the empty default")
	}
	for _, token := range []string{"ORIGINAL-REQUEST", "COMPLETED-PLAN-SUMMARY", "TERMINAL-STEPS", "RECENT-CONVERSATION"} {
		if strings.Contains(got, token) {
			t.Errorf("direct-plan prompt leaked literal token %q", token)
		}
	}
}
