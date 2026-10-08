package orchestration

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/v0lka/sp4rk/agent"
	"github.com/v0lka/sp4rk/llm"
	sdkmemory "github.com/v0lka/sp4rk/memory"
	"github.com/v0lka/sp4rk/tools"
)

// condUsageLLM is a minimal agent.LLMCaller returning a single canned finish
// response carrying non-zero usage, so TrackingCaller's usage correction is
// observable through the Conductor's wiring.
type condUsageLLM struct {
	resp *llm.ChatResponse
}

func (m *condUsageLLM) Call(_ context.Context, _ llm.ChatRequest) (*llm.ChatResponse, error) {
	return m.resp, nil
}

// TestConductor_Run_WiresContextTrackerCorrection is the regression test for
// the two independent breaks that kept API-reported token usage from reaching
// the context window's fill accounting:
//
//   - TrackerProvider gate: memory.ContextWindow exposed the tracker as
//     Tracker(), not ContextTracker(), so the cm.(TrackerProvider) assertion
//     failed for the SDK's own context manager.
//   - Injector gate: the caller assertion demanded a WithContextTracker
//     returning agent.LLMCaller, while llm.TrackingCaller's method returns
//     *llm.TrackingCaller — no type could satisfy the assertion (fixed via
//     llm.TrackerInjector).
//
// The test drives the real types end-to-end — a *memory.ContextWindow as the
// Conductor's ContextManager and an *llm.TrackingCaller as ConductorConfig.LLM
// — and asserts that the executor's LLM response usage (7777 input tokens) is
// corrected into the window's tracker. Without the wiring the tracker stays
// delta-driven and its estimate stays orders of magnitude below the reported
// usage.
func TestConductor_Run_WiresContextTrackerCorrection(t *testing.T) {
	const apiInputTokens = 7777

	var window *sdkmemory.ContextWindow
	tracking := llm.NewTrackingCaller(&condUsageLLM{
		resp: &llm.ChatResponse{
			Message: llm.Message{
				Role:    "assistant",
				Content: "done",
				ToolCalls: []llm.ToolCall{{
					ID:    "call_finish",
					Name:  "finish",
					Input: json.RawMessage(`{"answer":"done"}`),
				}},
			},
			StopReason: "tool_use",
			Usage:      llm.TokenUsage{InputTokens: apiInputTokens, OutputTokens: 10},
		},
	}, llm.NewUsageTracker())

	cfg := ConductorConfig{
		LLM:   tracking,
		Tools: condMockTools{},
		ContextFactory: func(systemPrompt string, meta llm.ModelMetadata, _ string, _ ...PruningOverride) agent.ContextManager {
			w := sdkmemory.NewContextWindow(sdkmemory.ContextWindowConfig{
				SystemPrompt: systemPrompt,
				ModelMeta:    meta,
				Tracker:      llm.NewContextTokenTracker(llm.NewSimpleTokenCounter()),
			})
			window = w
			return w
		},
		SystemPrompt: func(_ context.Context, _ string, _ llm.ModelMetadata) string { return "system prompt" },
		MaxSteps:     10,
	}
	cond := NewConductor(cfg)

	avail := []tools.ToolDescriptor{{Name: "noop", Description: "no-op tool", Source: "core"}}
	result, err := cond.Run(context.Background(), "report usage", NewMapBlackboard(), avail, &agent.NoopEvents{}, "sliding_window")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result == nil || result.Status != ExecutionStatusSuccess {
		t.Fatalf("run did not succeed: status=%v err=%v", result, err)
	}
	if window == nil {
		t.Fatal("context factory was not invoked")
	}

	// Both gates must hold for the SDK's own types.
	var cm agent.ContextManager = window
	if _, ok := cm.(TrackerProvider); !ok {
		t.Fatal("*memory.ContextWindow does not satisfy TrackerProvider — gate 1 still broken")
	}
	var llmCaller agent.LLMCaller = tracking
	if _, ok := llmCaller.(llm.TrackerInjector); !ok {
		t.Fatal("*llm.TrackingCaller does not satisfy llm.TrackerInjector — gate 2 still broken")
	}

	// The API-reported usage must have been corrected into the window's
	// tracker: EstimateTotal is lastKnownUsed + pendingDelta, and without the
	// wiring lastKnownUsed stays 0 (the delta-driven estimate of these short
	// strings is far below the reported usage).
	if got := window.ContextTracker().EstimateTotal(); got < apiInputTokens {
		t.Errorf("ContextTracker().EstimateTotal() = %d, want >= %d (API-reported input tokens) — usage correction not wired",
			got, apiInputTokens)
	}
}
