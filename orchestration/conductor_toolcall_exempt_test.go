package orchestration

import (
	"context"
	"encoding/json"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/v0lka/sp4rk/agent"
	"github.com/v0lka/sp4rk/llm"
	"github.com/v0lka/sp4rk/tools"
)

// exemptBlockingTools is a ToolExecutor that blocks the named tool until
// release is closed (or ctx is cancelled) — modelling an MCP-backed tool that
// legitimately waits on a slow server. Every other tool answers "ok" like
// condMockTools.
type exemptBlockingTools struct {
	blockName string
	started   chan struct{}
	release   chan struct{}
	once      sync.Once
	runs      atomic.Int32
}

func (t *exemptBlockingTools) Execute(ctx context.Context, name string, _ json.RawMessage) (tools.ToolResult, error) {
	if name != t.blockName {
		return tools.ToolResult{Content: "ok"}, nil
	}
	t.runs.Add(1)
	t.once.Do(func() { close(t.started) })
	select {
	case <-t.release:
		return tools.ToolResult{Content: "released"}, nil
	case <-ctx.Done():
		return tools.ToolResult{}, ctx.Err()
	}
}

func (t *exemptBlockingTools) GetToolSource(string) string { return "core" }

func (t *exemptBlockingTools) IsToolUntrusted(string) bool { return false }

func (t *exemptBlockingTools) CacheStrategy(context.Context, string, json.RawMessage) tools.CacheMode {
	return tools.CacheModeDefault
}

// TestConductor_Run_ToolCallTimeoutExemptTools_ThreadedToExecutor pins the
// ConductorConfig.ToolCallTimeoutExemptTools threading: the host-provided
// exemption set reaches the conductor's main executor, so a ceiling-bound
// slow tool named in the set runs to completion instead of failing the whole
// run at the ceiling.
func TestConductor_Run_ToolCallTimeoutExemptTools_ThreadedToExecutor(t *testing.T) {
	toolExec := &exemptBlockingTools{
		blockName: "slow_tool",
		started:   make(chan struct{}),
		release:   make(chan struct{}),
	}
	var releaseOnce sync.Once
	t.Cleanup(func() { releaseOnce.Do(func() { close(toolExec.release) }) })

	fakeCM := newCondFakeCM()
	cfg := ConductorConfig{
		LLM: &condMockLLM{responses: []*llm.ChatResponse{
			condToolCallResponse("call the slow tool", "slow_tool", json.RawMessage(`{}`)),
			condFinishResponse("done", "finished"),
		}},
		Tools: toolExec,
		ContextFactory: func(_ string, _ llm.ModelMetadata, _ string, _ ...PruningOverride) agent.ContextManager {
			return fakeCM
		},
		SystemPrompt: func(_ context.Context, _ string, _ llm.ModelMetadata) string { return "system prompt" },
		MaxSteps:     10,
		// A ceiling far below the tool's run time; only the exemption lets the
		// call survive it.
		ToolCallTimeout:            50 * time.Millisecond,
		ToolCallTimeoutExemptTools: []string{"slow_tool"},
	}
	cond := NewConductor(cfg)

	// Release the tool once it is in flight; ordering is channel-driven (no
	// sleeps), so the ceiling can only fire if the threading is broken.
	go func() {
		<-toolExec.started
		releaseOnce.Do(func() { close(toolExec.release) })
	}()

	result, err := cond.Run(context.Background(), "do something", NewMapBlackboard(), nil, &agent.NoopEvents{}, "sliding_window")
	if err != nil {
		t.Fatalf("exempt slow tool must not fail the run at the ceiling: %v", err)
	}
	if result == nil {
		t.Fatal("expected a non-nil result")
	}
	released := false
	for _, s := range result.Steps {
		if s.Observation == "released" {
			released = true
			break
		}
	}
	if !released {
		t.Error("expected the exempt slow tool's result in the trajectory")
	}
}

// TestConductor_Run_ToolCallTimeoutExemptTools_NilKeepsDefaultSet: a nil set
// keeps the executor's built-in default, so a slow tool outside it is still
// bounded — the historical behavior is unchanged for hosts that do not opt in.
func TestConductor_Run_ToolCallTimeoutExemptTools_NilKeepsDefaultSet(t *testing.T) {
	toolExec := &exemptBlockingTools{
		blockName: "slow_tool",
		started:   make(chan struct{}),
		release:   make(chan struct{}),
	}
	t.Cleanup(func() { close(toolExec.release) })

	fakeCM := newCondFakeCM()
	cfg := ConductorConfig{
		LLM: &condMockLLM{responses: []*llm.ChatResponse{
			condToolCallResponse("call the slow tool", "slow_tool", json.RawMessage(`{}`)),
			condFinishResponse("done", "finished"),
		}},
		Tools: toolExec,
		ContextFactory: func(_ string, _ llm.ModelMetadata, _ string, _ ...PruningOverride) agent.ContextManager {
			return fakeCM
		},
		SystemPrompt:    func(_ context.Context, _ string, _ llm.ModelMetadata) string { return "system prompt" },
		MaxSteps:        10,
		ToolCallTimeout: 50 * time.Millisecond,
		// ToolCallTimeoutExemptTools deliberately nil: slow_tool is not in the
		// built-in default set, so the ceiling must bound it.
	}
	cond := NewConductor(cfg)

	_, err := cond.Run(context.Background(), "do something", NewMapBlackboard(), nil, &agent.NoopEvents{}, "sliding_window")
	if !errors.Is(err, agent.ErrToolTimeout) {
		t.Fatalf("slow tool outside the default exempt set must be bounded: got %v, want ErrToolTimeout", err)
	}
}
