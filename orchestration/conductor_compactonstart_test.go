package orchestration

import (
	"context"
	"encoding/json"
	"sync"
	"testing"

	"github.com/v0lka/sp4rk/agent"
	"github.com/v0lka/sp4rk/llm"
)

// --- Fakes for Conductor.Run CompactOnStart tests ---

// seqRecorder records the relative order of Conductor-triggered operations
// (start-of-run compaction vs first LLM call) behind a mutex.
type seqRecorder struct {
	mu    sync.Mutex
	items []string
}

func (r *seqRecorder) record(item string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.items = append(r.items, item)
}

func (r *seqRecorder) snapshot() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]string(nil), r.items...)
}

// compactFakeCM is a minimal agent.ContextManager with the TaskAware and
// StepSeedable capabilities. It counts Compact invocations, returns a
// configurable result, and records call ordering into a shared recorder.
type compactFakeCM struct {
	mu            sync.Mutex
	seq           *seqRecorder
	compactResult *agent.CompactionResult
	compactCalls  int
	seededSteps   []agent.Step
}

func (m *compactFakeCM) BuildPrompt() []llm.Message {
	return []llm.Message{{Role: "system", Content: "sys"}}
}
func (m *compactFakeCM) AddStep(agent.Step) {}
func (m *compactFakeCM) Compact(_ context.Context) *agent.CompactionResult {
	m.mu.Lock()
	m.compactCalls++
	res := m.compactResult
	m.mu.Unlock()
	m.seq.record("compact")
	return res
}
func (m *compactFakeCM) SetStrategy(agent.CompactionStrategy) {}
func (m *compactFakeCM) CheckFill() agent.FillCheck {
	return agent.FillCheck{Percent: 5, Status: "ok", Used: 100, Max: 100000}
}
func (m *compactFakeCM) CorrectTokenCount(int)                       {}
func (m *compactFakeCM) FillPercent() float64                        { return 5 }
func (m *compactFakeCM) AvailableTokens() int                        { return 100000 }
func (m *compactFakeCM) OutputLimit() int                            { return 4096 }
func (m *compactFakeCM) VulnerableOutputs() []agent.VulnerableOutput { return nil }
func (m *compactFakeCM) compactCount() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.compactCalls
}

// TaskAware.
func (m *compactFakeCM) SetTask(string) {}

// StepSeedable.
func (m *compactFakeCM) SeedSteps(steps []agent.Step) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.seededSteps = append([]agent.Step(nil), steps...)
}

// seqLLM is a minimal agent.LLMCaller that records each call into the shared
// sequence recorder and returns a canned finish response so the run
// terminates after a single LLM round-trip.
type seqLLM struct {
	seq *seqRecorder
}

func (l *seqLLM) Call(_ context.Context, _ llm.ChatRequest) (*llm.ChatResponse, error) {
	l.seq.record("llm")
	return condFinishResponse("working", "compact-on-start output"), nil
}

// compactEvents embeds agent.NoopEvents and records ContextCompaction
// payloads (before/after percentages and step IDs).
type compactEvents struct {
	*agent.NoopEvents
	mu        sync.Mutex
	compacted []agent.CompactionResult
	stepIDs   []string
}

func (e *compactEvents) ContextCompaction(before, after float64, stepID string) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.compacted = append(e.compacted, agent.CompactionResult{BeforePercent: before, AfterPercent: after})
	e.stepIDs = append(e.stepIDs, stepID)
}

func (e *compactEvents) compactionCount() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return len(e.compacted)
}

// --- Tests ---

// runCompactOnStart runs a Conductor with two resume steps (as a manual
// compaction of a paused task would), the given CompactOnStart flag and
// Compact result, returning the fakes for assertion.
func runCompactOnStart(t *testing.T, compactOnStart bool, compactResult *agent.CompactionResult) (*compactFakeCM, *compactEvents, *seqRecorder) {
	t.Helper()
	seq := &seqRecorder{}
	cm := &compactFakeCM{seq: seq, compactResult: compactResult}
	ev := &compactEvents{NoopEvents: &agent.NoopEvents{}}
	resumeSteps := []agent.Step{
		{Thought: "prior one", Action: llm.ToolCall{ID: "c1", Name: "read_file", Input: json.RawMessage(`{}`)}, Observation: "o1"},
		{Thought: "prior two", Action: llm.ToolCall{ID: "c2", Name: "read_file", Input: json.RawMessage(`{}`)}, Observation: "o2"},
	}
	cfg := ConductorConfig{
		LLM:            &seqLLM{seq: seq},
		Tools:          condMockTools{},
		ContextFactory: func(_ string, _ llm.ModelMetadata, _ string, _ ...PruningOverride) agent.ContextManager { return cm },
		SystemPrompt:   func(_ context.Context, _ string, _ llm.ModelMetadata) string { return "sys" },
		MaxSteps:       10,
		ResumeSteps:    resumeSteps,
		CompactOnStart: compactOnStart,
	}
	cond := NewConductor(cfg)
	if _, err := cond.Run(context.Background(), "continue the task", NewMapBlackboard(), nil, ev, "sliding_window"); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	return cm, ev, seq
}

// TestConductor_Run_CompactOnStart_CompactsOnceBeforeFirstLLMCall verifies
// that ResumeSteps + CompactOnStart triggers exactly one compaction pass and
// that it happens before the first LLM call of the run.
func TestConductor_Run_CompactOnStart_CompactsOnceBeforeFirstLLMCall(t *testing.T) {
	cm, _, seq := runCompactOnStart(t, true, &agent.CompactionResult{BeforePercent: 90, AfterPercent: 30})

	if got := cm.compactCount(); got != 1 {
		t.Fatalf("Compact called %d times, want exactly 1", got)
	}
	items := seq.snapshot()
	if len(items) < 2 {
		t.Fatalf("expected at least 2 recorded operations (compact, llm), got %v", items)
	}
	if items[0] != "compact" || items[1] != "llm" {
		t.Errorf("expected [compact llm] ordering, got %v", items)
	}
	compacts := 0
	for _, item := range items {
		if item == "compact" {
			compacts++
		}
	}
	if compacts != 1 {
		t.Errorf("recorded %d compact operations across the whole run, want 1", compacts)
	}
}

// TestConductor_Run_WithoutCompactOnStart_NoCompaction verifies the default
// (flag off) behavior: no start-of-run compaction and no ContextCompaction
// event — the first recorded operation is the LLM call.
func TestConductor_Run_WithoutCompactOnStart_NoCompaction(t *testing.T) {
	cm, ev, seq := runCompactOnStart(t, false, &agent.CompactionResult{BeforePercent: 90, AfterPercent: 30})

	if got := cm.compactCount(); got != 0 {
		t.Errorf("Compact called %d times without CompactOnStart, want 0", got)
	}
	if got := ev.compactionCount(); got != 0 {
		t.Errorf("ContextCompaction emitted %d times without CompactOnStart, want 0", got)
	}
	items := seq.snapshot()
	if len(items) == 0 || items[0] != "llm" {
		t.Errorf("expected the first recorded operation to be an LLM call, got %v", items)
	}
}

// TestConductor_Run_CompactOnStart_EventCarriesRealBeforeAfter verifies the
// ContextCompaction event carries the CompactionResult's actual before/after
// percentages (and an empty step ID — it is not tied to an executor step).
func TestConductor_Run_CompactOnStart_EventCarriesRealBeforeAfter(t *testing.T) {
	_, ev, _ := runCompactOnStart(t, true, &agent.CompactionResult{BeforePercent: 87.5, AfterPercent: 12.25})

	if got := ev.compactionCount(); got != 1 {
		t.Fatalf("ContextCompaction emitted %d times, want exactly 1", got)
	}
	if got := ev.compacted[0]; got.BeforePercent != 87.5 || got.AfterPercent != 12.25 {
		t.Errorf("ContextCompaction payload = {before: %v, after: %v}, want {87.5, 12.25}", got.BeforePercent, got.AfterPercent)
	}
	if got := ev.stepIDs[0]; got != "" {
		t.Errorf("ContextCompaction stepID = %q, want empty string", got)
	}
}

// TestConductor_Run_CompactOnStart_NilResultEmitsNoEvent verifies the no-op
// contract: when Compact reports that nothing was compacted (nil result —
// empty steps or nil strategy in the real ContextWindow), the Conductor must
// not emit a ContextCompaction event.
func TestConductor_Run_CompactOnStart_NilResultEmitsNoEvent(t *testing.T) {
	cm, ev, _ := runCompactOnStart(t, true, nil)

	if got := cm.compactCount(); got != 1 {
		t.Fatalf("Compact called %d times, want 1 (the call itself still happens)", got)
	}
	if got := ev.compactionCount(); got != 0 {
		t.Errorf("ContextCompaction emitted %d times on nil compaction result, want 0", got)
	}
}
