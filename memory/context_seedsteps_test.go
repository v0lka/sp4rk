package memory

import (
	"encoding/json"
	"fmt"
	"testing"

	sdkagent "github.com/v0lka/sp4rk/agent"
	"github.com/v0lka/sp4rk/llm"
)

// seededStep builds a Step with a tool call for SeedSteps tests.
func seededStep(thought, observation, toolName string, toolID int) sdkagent.Step {
	return sdkagent.Step{
		Thought: thought,
		Action: llm.ToolCall{
			ID:    fmt.Sprintf("call_%d", toolID),
			Name:  toolName,
			Input: json.RawMessage(`{"arg":"value"}`),
		},
		Observation: observation,
		TokensUsed:  100,
	}
}

// TestSeedSteps_BuildPromptIncludesSeededSteps verifies that SeedSteps sets the
// step history and BuildPrompt renders the seeded steps as proper assistant +
// tool messages (mirroring AddStep's output).
func TestSeedSteps_BuildPromptIncludesSeededSteps(t *testing.T) {
	tracker := llm.NewContextTokenTracker(llm.NewSimpleTokenCounter())
	strategy := NewSlidingWindowStrategy(5, 5)
	cw := NewContextWindow(ContextWindowConfig{
		SystemPrompt: "You are helpful.",
		ModelMeta:    testModelMeta(128000),
		Tracker:      tracker,
		Thresholds:   testThresholds(),
		Strategy:     strategy,
	})
	cw.SetTask("Do something")

	steps := []sdkagent.Step{
		seededStep("I should read the file", "file contents here", "read_file", 1),
		seededStep("Now I will check the config", "config contents", "read_file", 2),
	}
	cw.SeedSteps(steps)

	messages := cw.BuildPrompt()

	// Expect: system, user(task), then 2x (assistant + tool) = 2 + 2*2 = 6.
	wantLen := 6
	if len(messages) != wantLen {
		t.Fatalf("len(messages) = %d, want %d", len(messages), wantLen)
	}
	if messages[0].Role != "system" {
		t.Errorf("messages[0].Role = %q, want %q", messages[0].Role, "system")
	}
	if messages[1].Role != "user" {
		t.Errorf("messages[1].Role = %q, want %q", messages[1].Role, "user")
	}

	// First seeded step → assistant message with a tool call.
	assistant1 := messages[2]
	if assistant1.Role != "assistant" {
		t.Errorf("messages[2].Role = %q, want %q", assistant1.Role, "assistant")
	}
	if assistant1.Content != "I should read the file" {
		t.Errorf("messages[2].Content = %q, want seeded thought", assistant1.Content)
	}
	if len(assistant1.ToolCalls) != 1 || assistant1.ToolCalls[0].Name != "read_file" {
		t.Errorf("messages[2].ToolCalls = %+v, want single read_file call", assistant1.ToolCalls)
	}
	// Followed by the tool result message.
	if messages[3].Role != "tool" {
		t.Errorf("messages[3].Role = %q, want %q", messages[3].Role, "tool")
	}
	if got := messages[3].Content; got != "file contents here" {
		t.Errorf("messages[3].Content = %q, want seeded observation", got)
	}

	// Second seeded step likewise renders as assistant + tool.
	if messages[4].Role != "assistant" || len(messages[4].ToolCalls) != 1 {
		t.Errorf("messages[4] = %+v, want assistant with one tool call", messages[4])
	}
	if messages[5].Role != "tool" {
		t.Errorf("messages[5].Role = %q, want %q", messages[5].Role, "tool")
	}
}

// TestSeedSteps_MatchesAddStepTrackerDelta verifies that SeedSteps recalculates
// the tracker delta for the batch identically to adding the same steps one at
// a time via AddStep.
func TestSeedSteps_MatchesAddStepTrackerDelta(t *testing.T) {
	steps := []sdkagent.Step{
		seededStep("think one", "obs one", "read_file", 1),
		seededStep("think two", "obs two", "read_file", 2),
		seededStep("think three", "obs three", "write_file", 3),
	}

	// Window A: AddStep one at a time.
	cwAdd := NewContextWindow(ContextWindowConfig{
		ModelMeta:  testModelMeta(128000),
		Tracker:    llm.NewContextTokenTracker(llm.NewSimpleTokenCounter()),
		Thresholds: testThresholds(),
		Strategy:   NewSlidingWindowStrategy(5, 5),
	})
	for _, s := range steps {
		cwAdd.AddStep(s)
	}

	// Window B: SeedSteps with the whole batch.
	cwSeed := NewContextWindow(ContextWindowConfig{
		ModelMeta:  testModelMeta(128000),
		Tracker:    llm.NewContextTokenTracker(llm.NewSimpleTokenCounter()),
		Thresholds: testThresholds(),
		Strategy:   NewSlidingWindowStrategy(5, 5),
	})
	cwSeed.SeedSteps(steps)

	// A freshly-seeded window must report a positive fill estimate reflecting
	// the seeded history (not zero).
	if got := cwSeed.Tracker().EstimateTotal(); got <= 0 {
		t.Errorf("seeded tracker EstimateTotal = %d, want > 0", got)
	}

	if addTotal, seedTotal := cwAdd.Tracker().EstimateTotal(), cwSeed.Tracker().EstimateTotal(); addTotal != seedTotal {
		t.Errorf("tracker delta mismatch: AddStep=%d SeedSteps=%d (want equal)", addTotal, seedTotal)
	}
}

// TestSeedSteps_ReplacesHistoryAndClearsCompaction verifies that SeedSteps
// fully replaces the step history and clears any frozen compaction prefix so
// buildStepMessages renders the freshly seeded steps rather than a stale
// compacted prefix.
func TestSeedSteps_ReplacesHistoryAndClearsCompaction(t *testing.T) {
	tracker := llm.NewContextTokenTracker(llm.NewSimpleTokenCounter())
	cw := NewContextWindow(ContextWindowConfig{
		ModelMeta:  testModelMeta(128000),
		Tracker:    tracker,
		Thresholds: testThresholds(),
		Strategy:   NewSlidingWindowStrategy(5, 5),
	})
	cw.SetTask("task")

	// Start with two steps added normally.
	cw.AddStep(seededStep("first", "obs first", "read_file", 1))
	cw.AddStep(seededStep("second", "obs second", "read_file", 2))

	// Re-seed with a single, different step.
	replacement := []sdkagent.Step{seededStep("replaced", "obs replaced", "search", 9)}
	cw.SeedSteps(replacement)

	msgs := cw.BuildPrompt()
	// Collect all observations present in tool messages.
	var observations []string
	for _, m := range msgs {
		if m.Role == "tool" {
			observations = append(observations, m.Content)
		}
	}
	if len(observations) != 1 || observations[0] != "obs replaced" {
		t.Errorf("after SeedSteps, tool observations = %v, want [obs replaced]", observations)
	}
}

// TestSeedSteps_NilClearsHistory verifies that passing nil (or an empty slice)
// clears any existing step history — BuildPrompt then contains no tool steps.
func TestSeedSteps_NilClearsHistory(t *testing.T) {
	cw := NewContextWindow(ContextWindowConfig{
		ModelMeta:  testModelMeta(128000),
		Tracker:    llm.NewContextTokenTracker(llm.NewSimpleTokenCounter()),
		Thresholds: testThresholds(),
		Strategy:   NewSlidingWindowStrategy(5, 5),
	})
	cw.SetTask("task")
	cw.AddStep(seededStep("prior", "prior obs", "read_file", 1))

	cw.SeedSteps(nil)

	msgs := cw.BuildPrompt()
	for _, m := range msgs {
		if m.Role == "tool" {
			t.Errorf("expected no tool messages after SeedSteps(nil), found: %+v", m)
		}
	}
}
