package tools

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/v0lka/sp4rk/llm"
	"github.com/v0lka/sp4rk/tools/internal/judge_prompts"
)

func loopJudgeResponse(content string) *llm.ChatResponse {
	return &llm.ChatResponse{Message: llm.Message{Content: content}}
}

func TestJudgeStepLimit_FailClosedNilProvider(t *testing.T) {
	j := NewToolJudge(nil, "test-model", 0, nil)
	v, reason, err := j.JudgeStepLimit(context.Background(), StepLimitJudgeRequest{CurrentStep: 5, MaxSteps: 5})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if v != LoopVerdictDeny {
		t.Errorf("nil provider: got %v, want LoopVerdictDeny", v)
	}
	if reason == "" {
		t.Error("expected a reasoning explaining the fail-closed deny")
	}
}

func TestJudgeStepLimit_FailClosedOnProviderError(t *testing.T) {
	m := &mockLLMProvider{err: errors.New("boom")}
	j := NewToolJudge(m, "test-model", 0, nil)
	v, _, err := j.JudgeStepLimit(context.Background(), StepLimitJudgeRequest{CurrentStep: 3, MaxSteps: 3})
	if err != nil {
		t.Fatalf("provider error must be swallowed (fail-closed), got %v", err)
	}
	if v != LoopVerdictDeny {
		t.Errorf("provider error: got %v, want LoopVerdictDeny", v)
	}
}

func TestJudgeStepLimit_FailClosedOnEmptyResponse(t *testing.T) {
	m := &mockLLMProvider{response: nil}
	j := NewToolJudge(m, "test-model", 0, nil)
	v, _, err := j.JudgeStepLimit(context.Background(), StepLimitJudgeRequest{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if v != LoopVerdictDeny {
		t.Errorf("nil response: got %v, want LoopVerdictDeny", v)
	}
}

func TestJudgeStepLimit_FailClosedOnUnparseable(t *testing.T) {
	m := &mockLLMProvider{response: loopJudgeResponse("???")}
	j := NewToolJudge(m, "test-model", 0, nil)
	v, reason, err := j.JudgeStepLimit(context.Background(), StepLimitJudgeRequest{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if v != LoopVerdictDeny {
		t.Errorf("unparseable: got %v, want LoopVerdictDeny", v)
	}
	if reason != loopJudgeUnparsedReason {
		t.Errorf("unparseable reason = %q, want %q", reason, loopJudgeUnparsedReason)
	}
}

func TestJudgeStepLimit_VerdictParsing(t *testing.T) {
	cases := []struct {
		name    string
		content string
		want    LoopVerdict
	}{
		{"allow_once", "VERDICT: ALLOW_ONCE\nREASON: one more step", LoopVerdictAllowOnce},
		{"allow_more", "VERDICT: ALLOW_MORE\nREASON: healthy progress", LoopVerdictAllowMore},
		{"allow_always", "VERDICT: ALLOW_ALWAYS\nREASON: long but clean", LoopVerdictAllowAlways},
		{"deny", "VERDICT: DENY\nREASON: stuck", LoopVerdictDeny},
		{"markdown", "- **VERDICT:** ALLOW ONCE\n- **REASON:** almost done", LoopVerdictAllowOnce},
		{"json", `{"verdict":"ALLOW_MORE","reason":"advancing"}`, LoopVerdictAllowMore},
		{"bare-token", "ALLOW_ALWAYS", LoopVerdictAllowAlways},
		{"bare-allow", "ALLOW", LoopVerdictAllowOnce},
		{"prose-deny", "I think we should DENY here.", LoopVerdictDeny},
		{"stop-synonym", "VERDICT: STOP", LoopVerdictDeny},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := &mockLLMProvider{response: loopJudgeResponse(tc.content)}
			j := NewToolJudge(m, "test-model", 0, nil)
			v, _, err := j.JudgeStepLimit(context.Background(), StepLimitJudgeRequest{})
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if v != tc.want {
				t.Errorf("content %q: got %v, want %v", tc.content, v, tc.want)
			}
		})
	}
}

func TestJudgeStepLimit_NotCached(t *testing.T) {
	m := &mockLLMProvider{response: loopJudgeResponse("VERDICT: ALLOW_ONCE\nREASON: x")}
	j := NewToolJudge(m, "test-model", 0, nil)
	req := StepLimitJudgeRequest{CurrentStep: 4, MaxSteps: 4, AbortCategory: LoopBoundaryBudget}
	for i := 0; i < 3; i++ {
		if _, _, err := j.JudgeStepLimit(context.Background(), req); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	}
	if got := len(m.snapshot()); got != 3 {
		t.Errorf("judge calls = %d, want 3 (step-limit decisions must not be cached)", got)
	}
}

func TestJudgeStepLimit_PromptCarriesTrajectoryAndMetrics(t *testing.T) {
	m := &mockLLMProvider{response: loopJudgeResponse("VERDICT: DENY\nREASON: loop")}
	j := NewToolJudge(m, "test-model", 0, nil)
	req := StepLimitJudgeRequest{
		TaskContext:   "refactor the parser",
		PlanSnapshot:  "step 2 of 4",
		CurrentStep:   12,
		MaxSteps:      12,
		AbortCategory: LoopBoundaryCircuitBreaker,
		AbortReason:   "Tool 'bash' called 3 times consecutively with identical arguments",
		RecentSteps: []StepLimitStepDigest{
			{Step: 10, Tool: "read_file", Args: `{"path":"a.go"}`, Result: "ok"},
			{Step: 11, Tool: "bash", Args: `{"command":"ls"}`, Result: "boom", IsError: true},
		},
		Metrics: StepLimitMetrics{Steps: 11, ToolCalls: 9, ToolErrors: 2, Nudges: 1, Aborts: 2, ParseErrors: 1, InvalidToolCalls: 1},
	}

	if _, _, err := j.JudgeStepLimit(context.Background(), req); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	reqs := m.snapshot()
	if len(reqs) != 1 {
		t.Fatalf("captured %d requests, want 1", len(reqs))
	}
	if len(reqs[0].Messages) != 2 {
		t.Fatalf("expected system+user messages, got %d", len(reqs[0].Messages))
	}
	if reqs[0].Messages[0].Content != judge_prompts.StepLimitSystem {
		t.Error("system message is not the embedded step-limit prompt")
	}
	user := reqs[0].Messages[1].Content
	for _, want := range []string{
		"circuit_breaker",       // boundary category
		"step: 12 of 12",        // boundary position
		"refactor the parser",   // task context
		"step 2 of 4",           // plan snapshot
		"read_file",             // trajectory tool
		"tool_errors: 2",        // metrics
		"hard_aborts: 2",        // metrics
		"invalid_tool_calls: 1", // metrics
		"<untrusted-content",    // trajectory wrapped as untrusted data
	} {
		if !strings.Contains(user, want) {
			t.Errorf("user prompt missing %q\n---\n%s", want, user)
		}
	}
}

func TestJudgeStepLimit_EmptyCategoryDefaultsToCircuitBreaker(t *testing.T) {
	m := &mockLLMProvider{response: loopJudgeResponse("VERDICT: DENY\nREASON: x")}
	j := NewToolJudge(m, "test-model", 0, nil)
	if _, _, err := j.JudgeStepLimit(context.Background(), StepLimitJudgeRequest{}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	user := m.snapshot()[0].Messages[1].Content
	if !strings.Contains(user, "trigger: circuit_breaker") {
		t.Errorf("empty category must fail safe to circuit_breaker, prompt:\n%s", user)
	}
}
