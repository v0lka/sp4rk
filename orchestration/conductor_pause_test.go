package orchestration

import (
	"context"
	"encoding/json"
	"errors"
	"sync"
	"testing"

	"github.com/v0lka/sp4rk/agent"
	"github.com/v0lka/sp4rk/llm"
	"github.com/v0lka/sp4rk/tools"
)

// --- Fakes for PauseChecker / resume-with-nudge conductor tests ---

// condInterjectionCM is a ContextManager that records BuildPrompt calls and
// implements the interjection lifecycle (InterjectionAware setter +
// InterjectionConsumer) so the conductor's resume-with-nudge wiring can be
// verified end-to-end. Its BuildPrompt appends the pending nudge as the FINAL
// user message — mirroring memory.ContextWindow — and only retires it on an
// explicit ConsumePendingUserInterjection call (never on read).
type condInterjectionCM struct {
	mu sync.Mutex

	task        string
	seededSteps []agent.Step
	addedSteps  []agent.Step

	// pendingNudge is set by SetPendingUserInterjection and cleared by
	// ConsumePendingUserInterjection. BuildPrompt appends it (if non-empty) as
	// the final user message.
	pendingNudge string

	// buildCalls snapshots the messages returned by each BuildPrompt call.
	buildCalls [][]llm.Message

	// consumeCount is the number of times ConsumePendingUserInterjection ran.
	consumeCount int

	// setNudgeArg records the argument the conductor passed to
	// SetPendingUserInterjection ("" if never called).
	setNudgeArg string
	setNudgeHit bool
}

func newCondInterjectionCM() *condInterjectionCM {
	return &condInterjectionCM{}
}

// ContextManager methods.
func (m *condInterjectionCM) BuildPrompt() []llm.Message {
	m.mu.Lock()
	defer m.mu.Unlock()
	msgs := []llm.Message{{Role: "system", Content: "sys"}}
	for _, s := range m.seededSteps {
		msgs = append(msgs, llm.Message{Role: "tool", Content: s.Observation})
	}
	for _, s := range m.addedSteps {
		msgs = append(msgs, llm.Message{Role: "tool", Content: s.Observation})
	}
	if m.pendingNudge != "" {
		msgs = append(msgs, llm.Message{Role: "user", Content: m.pendingNudge})
	}
	// Snapshot a copy so later mutations don't alter recorded history.
	snap := make([]llm.Message, len(msgs))
	copy(snap, msgs)
	m.buildCalls = append(m.buildCalls, snap)
	return msgs
}

func (m *condInterjectionCM) AddStep(step agent.Step) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.addedSteps = append(m.addedSteps, step)
}

func (m *condInterjectionCM) Compact(_ context.Context) *agent.CompactionResult { return nil }
func (m *condInterjectionCM) SetStrategy(_ agent.CompactionStrategy)            {}
func (m *condInterjectionCM) CheckFill() agent.FillCheck {
	return agent.FillCheck{Percent: 5, Status: "ok", Used: 100, Max: 100000}
}
func (m *condInterjectionCM) CorrectTokenCount(_ int)                     {}
func (m *condInterjectionCM) FillPercent() float64                        { return 5 }
func (m *condInterjectionCM) AvailableTokens() int                        { return 100000 }
func (m *condInterjectionCM) OutputLimit() int                            { return 4096 }
func (m *condInterjectionCM) VulnerableOutputs() []agent.VulnerableOutput { return nil }

// TaskAware.
func (m *condInterjectionCM) SetTask(task string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.task = task
}

// StepSeedable.
func (m *condInterjectionCM) SeedSteps(steps []agent.Step) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.seededSteps = append([]agent.Step(nil), steps...)
}

// InterjectionAware (orchestration) — the setter the Conductor invokes.
func (m *condInterjectionCM) SetPendingUserInterjection(msg string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.pendingNudge = msg
	m.setNudgeArg = msg
	m.setNudgeHit = true
}

// InterjectionConsumer (agent) — the retire call the Executor invokes after a
// successful LLM response.
func (m *condInterjectionCM) ConsumePendingUserInterjection() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.pendingNudge = ""
	m.consumeCount++
}

// condToolCallResponse builds a canned non-finish tool-call response.
func condToolCallResponse(thought, toolName string, input json.RawMessage) *llm.ChatResponse {
	return &llm.ChatResponse{
		Message: llm.Message{
			Role:    "assistant",
			Content: thought,
			ToolCalls: []llm.ToolCall{
				{ID: "call_1", Name: toolName, Input: input},
			},
		},
		StopReason: "tool_use",
		Usage:      llm.TokenUsage{InputTokens: 100, OutputTokens: 50},
	}
}

// --- Tests ---

// TestConductor_Run_PauseChecker_MapsToPausedStatus verifies that a
// cooperative PauseChecker configured on the Conductor trips at a step
// boundary inside Executor.Run and is surfaced as ExecutionStatusPaused with a
// wrapping ErrPaused, and (Fix 4a) an empty Output (not the raw sentinel
// string). The pause is a recoverable checkpoint, not a failure.
func TestConductor_Run_PauseChecker_MapsToPausedStatus(t *testing.T) {
	fakeCM := newCondFakeCM()
	cfg := ConductorConfig{
		LLM:   &condMockLLM{responses: []*llm.ChatResponse{condFinishResponse("thinking", "should not finish")}},
		Tools: condMockTools{},
		ContextFactory: func(_ string, _ llm.ModelMetadata, _ string, _ ...PruningOverride) agent.ContextManager {
			return fakeCM
		},
		SystemPrompt: func(_ context.Context, _ string, _ llm.ModelMetadata) string { return "system prompt" },
		MaxSteps:     10,
		PauseChecker: func(context.Context) bool { return true }, // trip on the first step boundary
	}
	cond := NewConductor(cfg)

	result, err := cond.Run(context.Background(), "do something", NewMapBlackboard(), nil, &agent.NoopEvents{}, "sliding_window")

	if result == nil {
		t.Fatal("expected non-nil result")
	}
	if result.Status != ExecutionStatusPaused {
		t.Errorf("Status = %q, want %q", result.Status, ExecutionStatusPaused)
	}
	if !errors.Is(err, agent.ErrPaused) {
		t.Errorf("errors.Is(err, ErrPaused) = false, want true; err = %v", err)
	}
	// Fix 4a: a paused run leaves Output empty (a clean checkpoint), not the
	// raw ErrPaused sentinel string ("executor paused at step boundary").
	if result.Output != "" {
		t.Errorf("Output = %q, want empty for a paused checkpoint", result.Output)
	}
}

// TestConductor_Run_PendingUserInterjection_LandsOnceThenConsumed verifies the
// resume-with-nudge round-trip through the Conductor: when
// PendingUserInterjection is configured, the conductor injects it via
// SetPendingUserInterjection, BuildPrompt appends it as the FINAL user message
// in the first LLM request, and the executor retires it (via
// ConsumePendingUserInterjection) after that request succeeds — so a second
// BuildPrompt carries no nudge. This covers the seam between the conductor's
// injection and the executor's decoupled consume-on-success.
func TestConductor_Run_PendingUserInterjection_LandsOnceThenConsumed(t *testing.T) {
	var cmRef *condInterjectionCM
	nudge := "please continue with variant B"
	resumeSteps := []agent.Step{
		{Thought: "prior", Action: llm.ToolCall{ID: "c1", Name: "read_file", Input: json.RawMessage(`{}`)}, Observation: "prior obs"},
	}
	cfg := ConductorConfig{
		LLM: &condMockLLM{responses: []*llm.ChatResponse{
			condToolCallResponse("first step, nudge present", "noop", json.RawMessage(`{}`)), // request 1 (non-finish → loop continues)
			condFinishResponse("second step, nudge gone", "done"),                            // request 2 (finish)
		}},
		Tools: condMockTools{},
		ContextFactory: func(_ string, _ llm.ModelMetadata, _ string, _ ...PruningOverride) agent.ContextManager {
			cm := newCondInterjectionCM()
			cmRef = cm
			return cm
		},
		SystemPrompt:            func(_ context.Context, _ string, _ llm.ModelMetadata) string { return "system prompt" },
		MaxSteps:                10,
		ResumeSteps:             resumeSteps,
		PendingUserInterjection: nudge,
	}
	cond := NewConductor(cfg)

	avail := []tools.ToolDescriptor{{Name: "noop", Description: "no-op tool", Source: "core"}}
	if _, err := cond.Run(context.Background(), "continue the task", NewMapBlackboard(), avail, &agent.NoopEvents{}, "sliding_window"); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if cmRef == nil {
		t.Fatal("context factory was not invoked")
	}
	// The conductor must have injected the nudge via InterjectionAware.
	if !cmRef.setNudgeHit {
		t.Fatal("SetPendingUserInterjection was never called by the Conductor")
	}
	if cmRef.setNudgeArg != nudge {
		t.Errorf("SetPendingUserInterjection arg = %q, want %q", cmRef.setNudgeArg, nudge)
	}
	if len(cmRef.buildCalls) < 2 {
		t.Fatalf("expected at least 2 BuildPrompt calls, got %d (need a second to prove the nudge was retired)", len(cmRef.buildCalls))
	}

	// First request: the nudge must be the FINAL message.
	first := cmRef.buildCalls[0]
	firstLast := first[len(first)-1]
	if firstLast.Role != "user" || firstLast.Content != nudge {
		t.Errorf("first BuildPrompt final message = %q (%q), want nudge %q",
			firstLast.Role, firstLast.Content, nudge)
	}

	// Second request: the nudge must be absent (retired by the executor after
	// the first successful LLM response).
	second := cmRef.buildCalls[1]
	for _, m := range second {
		if m.Role == "user" && m.Content == nudge {
			t.Errorf("nudge %q present in second BuildPrompt — should have been consumed after the first successful LLM response", nudge)
		}
	}

	// The executor must have retired the nudge at least once.
	if cmRef.consumeCount < 1 {
		t.Errorf("ConsumePendingUserInterjection called %d times, want >= 1", cmRef.consumeCount)
	}
}
