package orchestration

import (
	"context"
	"encoding/json"
	"strings"
	"sync"
	"testing"

	"github.com/v0lka/sp4rk/agent"
	"github.com/v0lka/sp4rk/llm"
	"github.com/v0lka/sp4rk/tools"
)

// --- Fakes for Conductor.Run resume tests ---

// condFakeCM is a minimal agent.ContextManager that also implements TaskAware
// and StepSeedable. It records whether SeedSteps was invoked and with what
// steps, so the conductor resume wiring can be verified.
type condFakeCM struct {
	mu           sync.Mutex
	task         string
	seededSteps  []agent.Step
	addedSteps   []agent.Step
	buildPrompt  []llm.Message
	fillCheck    agent.FillCheck
	fillPercent  float64
	availableTok int
}

func newCondFakeCM() *condFakeCM {
	return &condFakeCM{
		buildPrompt:  []llm.Message{{Role: "system", Content: "sys"}},
		fillCheck:    agent.FillCheck{Percent: 5, Status: "ok", Used: 100, Max: 100000},
		availableTok: 100000,
	}
}

// ContextManager methods.
func (m *condFakeCM) BuildPrompt() []llm.Message { return m.buildPrompt }
func (m *condFakeCM) AddStep(step agent.Step) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.addedSteps = append(m.addedSteps, step)
}
func (m *condFakeCM) Compact(_ context.Context) *agent.CompactionResult { return nil }
func (m *condFakeCM) SetStrategy(_ agent.CompactionStrategy)            {}
func (m *condFakeCM) CheckFill() agent.FillCheck                        { return m.fillCheck }
func (m *condFakeCM) CorrectTokenCount(_ int)                           {}
func (m *condFakeCM) FillPercent() float64                              { return m.fillPercent }
func (m *condFakeCM) AvailableTokens() int                              { return m.availableTok }
func (m *condFakeCM) OutputLimit() int                                  { return 4096 }
func (m *condFakeCM) VulnerableOutputs() []agent.VulnerableOutput       { return nil }

// TaskAware.
func (m *condFakeCM) SetTask(task string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.task = task
}

// StepSeedable.
func (m *condFakeCM) SeedSteps(steps []agent.Step) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.seededSteps = append([]agent.Step(nil), steps...)
}

// condMockLLM is a minimal agent.LLMCaller returning canned responses.
type condMockLLM struct {
	mu        sync.Mutex
	responses []*llm.ChatResponse
	callIdx   int
}

func (m *condMockLLM) Call(_ context.Context, _ llm.ChatRequest) (*llm.ChatResponse, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	idx := m.callIdx
	m.callIdx++
	if idx < len(m.responses) {
		return m.responses[idx], nil
	}
	return &llm.ChatResponse{
		Message:    llm.Message{Role: "assistant", Content: "default response"},
		StopReason: "end_turn",
	}, nil
}

func condFinishResponse(thought, answer string) *llm.ChatResponse {
	input, _ := json.Marshal(map[string]string{"answer": answer})
	return &llm.ChatResponse{
		Message: llm.Message{
			Role:    "assistant",
			Content: thought,
			ToolCalls: []llm.ToolCall{
				{ID: "call_finish", Name: "finish", Input: input},
			},
		},
		StopReason: "tool_use",
		Usage:      llm.TokenUsage{InputTokens: 100, OutputTokens: 50},
	}
}

// condRecordingEvents embeds agent.NoopEvents and overrides StepStart to
// capture the first step number the conductor's executor emits.
type condRecordingEvents struct {
	*agent.NoopEvents
	mu        sync.Mutex
	firstStep int
	started   bool
}

func (r *condRecordingEvents) StepStart(stepNum int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if !r.started {
		r.firstStep = stepNum
		r.started = true
	}
}

// condMockTools is a minimal agent.ToolExecutor.
type condMockTools struct{}

func (condMockTools) Execute(_ context.Context, _ string, _ json.RawMessage) (tools.ToolResult, error) {
	return tools.ToolResult{Content: "ok"}, nil
}
func (condMockTools) GetToolSource(_ string) string { return "core" }
func (condMockTools) IsToolUntrusted(_ string) bool { return false }
func (condMockTools) CacheStrategy(_ context.Context, _ string, _ json.RawMessage) tools.CacheMode {
	return tools.CacheModeDefault
}

// --- Tests ---

func newResumeConductor(t *testing.T, resumeSteps []agent.Step) *Conductor {
	t.Helper()
	fakeCM := newCondFakeCM()
	cfg := ConductorConfig{
		LLM: &condMockLLM{
			responses: []*llm.ChatResponse{condFinishResponse("resuming", "conductor resumed output")},
		},
		Tools: condMockTools{},
		ContextFactory: func(_ string, _ llm.ModelMetadata, _ string, _ ...PruningOverride) agent.ContextManager {
			return fakeCM
		},
		SystemPrompt: func(_ context.Context, _ string, _ llm.ModelMetadata) string { return "system prompt" },
		MaxSteps:     10,
	}
	if resumeSteps != nil {
		cfg.ResumeSteps = resumeSteps
	}
	return NewConductor(cfg)
}

// TestConductor_Run_ResumeSteps_SeedsContextManager verifies that when
// ResumeSteps is configured, Conductor.Run seeds the ContextManager with the
// steps via its StepSeedable capability.
func TestConductor_Run_ResumeSteps_SeedsContextManager(t *testing.T) {
	resumeSteps := []agent.Step{
		{Thought: "prior one", Action: llm.ToolCall{ID: "c1", Name: "read_file", Input: json.RawMessage(`{}`)}, Observation: "o1"},
		{Thought: "prior two", Action: llm.ToolCall{ID: "c2", Name: "read_file", Input: json.RawMessage(`{}`)}, Observation: "o2"},
	}
	// Capture the CM instance created by the factory so we can assert on it.
	var cmRef *condFakeCM
	cfg := ConductorConfig{
		LLM:   &condMockLLM{responses: []*llm.ChatResponse{condFinishResponse("resuming", "done")}},
		Tools: condMockTools{},
		ContextFactory: func(_ string, _ llm.ModelMetadata, _ string, _ ...PruningOverride) agent.ContextManager {
			cm := newCondFakeCM()
			cmRef = cm
			return cm
		},
		SystemPrompt: func(_ context.Context, _ string, _ llm.ModelMetadata) string { return "sys" },
		MaxSteps:     10,
		ResumeSteps:  resumeSteps,
	}
	cond := NewConductor(cfg)

	_, err := cond.Run(context.Background(), "continue the task", NewMapBlackboard(), nil, &agent.NoopEvents{}, "sliding_window")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cmRef == nil {
		t.Fatal("context factory was not invoked")
	}
	if len(cmRef.seededSteps) != len(resumeSteps) {
		t.Fatalf("ContextManager.SeedSteps received %d steps, want %d", len(cmRef.seededSteps), len(resumeSteps))
	}
	for i, s := range cmRef.seededSteps {
		if got, want := s.Action.ID, resumeSteps[i].Action.ID; got != want {
			t.Errorf("seeded step[%d].Action.ID = %q, want %q", i, got, want)
		}
	}
}

// TestConductor_Run_ResumeSteps_ContinuesStepNumber verifies that the executor
// built by Conductor.Run is seeded with the resumed steps so its step counter
// continues from len(steps)+1.
func TestConductor_Run_ResumeSteps_ContinuesStepNumber(t *testing.T) {
	resumeSteps := []agent.Step{
		{Thought: "prior one", Action: llm.ToolCall{ID: "c1", Name: "read_file", Input: json.RawMessage(`{}`)}, Observation: "o1"},
		{Thought: "prior two", Action: llm.ToolCall{ID: "c2", Name: "read_file", Input: json.RawMessage(`{}`)}, Observation: "o2"},
		{Thought: "prior three", Action: llm.ToolCall{ID: "c3", Name: "read_file", Input: json.RawMessage(`{}`)}, Observation: "o3"},
	}
	events := &condRecordingEvents{NoopEvents: &agent.NoopEvents{}}
	cond := newResumeConductor(t, resumeSteps)

	_, err := cond.Run(context.Background(), "continue the task", NewMapBlackboard(), nil, events, "sliding_window")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// 3 resumed steps → the first executed step should be #4.
	if !events.started {
		t.Fatal("expected at least one StepStart event")
	}
	if events.firstStep != len(resumeSteps)+1 {
		t.Errorf("first executor StepStart = %d, want %d", events.firstStep, len(resumeSteps)+1)
	}
}

// TestConductor_Run_WithoutResume_StartsAtStep1 verifies backward
// compatibility: with no ResumeSteps the executor starts at step 1.
func TestConductor_Run_WithoutResume_StartsAtStep1(t *testing.T) {
	events := &condRecordingEvents{NoopEvents: &agent.NoopEvents{}}
	cond := newResumeConductor(t, nil)

	_, err := cond.Run(context.Background(), "fresh task", NewMapBlackboard(), nil, events, "sliding_window")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !events.started {
		t.Fatal("expected at least one StepStart event")
	}
	if events.firstStep != 1 {
		t.Errorf("first executor StepStart = %d, want 1", events.firstStep)
	}
}

// condFakeCMNoSeed implements agent.ContextManager but deliberately omits the
// StepSeedable capability, to exercise the Conductor's fail-fast path when
// ResumeSteps is configured against an incapable ContextManager.
type condFakeCMNoSeed struct{}

func (condFakeCMNoSeed) BuildPrompt() []llm.Message {
	return []llm.Message{{Role: "system", Content: "sys"}}
}
func (condFakeCMNoSeed) AddStep(agent.Step)                              {}
func (condFakeCMNoSeed) Compact(context.Context) *agent.CompactionResult { return nil }
func (condFakeCMNoSeed) SetStrategy(agent.CompactionStrategy)            {}
func (condFakeCMNoSeed) CheckFill() agent.FillCheck {
	return agent.FillCheck{Percent: 5, Status: "ok", Used: 100, Max: 100000}
}
func (condFakeCMNoSeed) CorrectTokenCount(int)                       {}
func (condFakeCMNoSeed) FillPercent() float64                        { return 0 }
func (condFakeCMNoSeed) AvailableTokens() int                        { return 100000 }
func (condFakeCMNoSeed) OutputLimit() int                            { return 4096 }
func (condFakeCMNoSeed) VulnerableOutputs() []agent.VulnerableOutput { return nil }

// TestConductor_Run_ResumeSteps_FailsFastWhenNotStepSeedable verifies the
// fail-fast contract: when ResumeSteps is configured but the ContextManager
// produced by the factory does not implement StepSeedable, Run returns an
// error immediately instead of silently producing an incoherent resume
// (continued step counter with no seeded steps in the prompt).
func TestConductor_Run_ResumeSteps_FailsFastWhenNotStepSeedable(t *testing.T) {
	resumeSteps := []agent.Step{
		{Thought: "prior one", Action: llm.ToolCall{ID: "c1", Name: "read_file", Input: json.RawMessage(`{}`)}, Observation: "o1"},
	}
	cfg := ConductorConfig{
		LLM:   &condMockLLM{responses: []*llm.ChatResponse{condFinishResponse("x", "y")}},
		Tools: condMockTools{},
		ContextFactory: func(_ string, _ llm.ModelMetadata, _ string, _ ...PruningOverride) agent.ContextManager {
			return condFakeCMNoSeed{}
		},
		SystemPrompt: func(_ context.Context, _ string, _ llm.ModelMetadata) string { return "sys" },
		MaxSteps:     10,
		ResumeSteps:  resumeSteps,
	}
	cond := NewConductor(cfg)

	_, err := cond.Run(context.Background(), "continue the task", NewMapBlackboard(), nil, &agent.NoopEvents{}, "sliding_window")
	if err == nil {
		t.Fatal("expected error when ContextManager does not implement StepSeedable, got nil")
	}
	if !strings.Contains(err.Error(), "StepSeedable") {
		t.Errorf("error should mention StepSeedable, got: %v", err)
	}
}
