package orchestration

import (
	"context"
	"testing"

	"github.com/v0lka/sp4rk/agent"
	"github.com/v0lka/sp4rk/llm"
)

// TestConductor_SetModel_ResolvesNewMetadata is the conductor-side half of
// the fluent Models() fix: the Conductor resolves the system-prompt and
// context-window metadata from cfg.Model on every Run, so when a host pins a
// dedicated execution model after construction (TaskBuilder.SetModel on the
// exec path), the resolved metadata must follow — window and output reserve
// sized for the model that actually serves the calls, not the construction
// -time default.
func TestConductor_SetModel_ResolvesNewMetadata(t *testing.T) {
	reg := llm.NewModelRegistry(nil)
	reg.SetRuntimeMetadata("model-default", llm.ModelMetadata{ContextWindow: 100000, OutputLimit: 8000})
	reg.SetRuntimeMetadata("model-exec", llm.ModelMetadata{ContextWindow: 200000, OutputLimit: 16000})

	var captured llm.ModelMetadata
	cfg := ConductorConfig{
		LLM:           &condMockLLM{responses: []*llm.ChatResponse{condFinishResponse("thinking", "done")}},
		Tools:         condMockTools{},
		Model:         "model-default",
		ModelRegistry: reg,
		ContextFactory: func(_ string, meta llm.ModelMetadata, _ string, _ ...PruningOverride) agent.ContextManager {
			captured = meta
			return newCondFakeCM()
		},
		SystemPrompt: func(_ context.Context, _ string, _ llm.ModelMetadata) string { return "sys" },
		MaxSteps:     5,
	}
	cond := NewConductor(cfg)

	// Control: before SetModel the construction-time default is resolved.
	if _, err := cond.Run(context.Background(), "task", NewMapBlackboard(), nil, &agent.NoopEvents{}, "sliding_window"); err != nil {
		t.Fatalf("run with default model: %v", err)
	}
	if captured.ContextWindow != 100000 {
		t.Errorf("ContextWindow = %d, want 100000 (construction-time default model metadata)", captured.ContextWindow)
	}

	// Pin the execution model: the next Run must resolve ITS metadata.
	cond.SetModel("model-exec")
	if _, err := cond.Run(context.Background(), "task", NewMapBlackboard(), nil, &agent.NoopEvents{}, "sliding_window"); err != nil {
		t.Fatalf("run after SetModel: %v", err)
	}
	if captured.ContextWindow != 200000 || captured.OutputLimit != 16000 {
		t.Errorf("after SetModel: ContextWindow/OutputLimit = %d/%d, want 200000/16000 — metadata still resolved from the pre-switch default model",
			captured.ContextWindow, captured.OutputLimit)
	}
}
