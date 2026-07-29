package orchestration

import (
	"context"
	"testing"

	"github.com/v0lka/sp4rk/agent"
	"github.com/v0lka/sp4rk/llm"
)

// blockFakeCM implements agent.ContextManager + BlockTaskAware + TaskAware.
type blockFakeCM struct {
	task        string
	taskBlocks  []llm.ContentBlock
	blockCalled bool
	taskCalled  bool
}

func (m *blockFakeCM) BuildPrompt() []llm.Message                        { return nil }
func (m *blockFakeCM) AddStep(step agent.Step)                           {}
func (m *blockFakeCM) Compact(_ context.Context) *agent.CompactionResult { return nil }
func (m *blockFakeCM) SetStrategy(_ agent.CompactionStrategy)            {}
func (m *blockFakeCM) CheckFill() agent.FillCheck                        { return agent.FillCheck{} }
func (m *blockFakeCM) CorrectTokenCount(_ int)                           {}
func (m *blockFakeCM) FillPercent() float64                              { return 0 }
func (m *blockFakeCM) AvailableTokens() int                              { return 100000 }
func (m *blockFakeCM) OutputLimit() int                                  { return 4096 }
func (m *blockFakeCM) VulnerableOutputs() []agent.VulnerableOutput       { return nil }

func (m *blockFakeCM) SetTask(task string) {
	m.task = task
	m.taskCalled = true
}

func (m *blockFakeCM) SetTaskWithBlocks(task string, blocks []llm.ContentBlock) {
	m.task = task
	m.taskBlocks = blocks
	m.blockCalled = true
}

// taskOnlyFakeCM implements agent.ContextManager + TaskAware (not BlockTaskAware).
type taskOnlyFakeCM struct {
	task        string
	taskCalled  bool
	blockCalled bool
}

func (m *taskOnlyFakeCM) BuildPrompt() []llm.Message                        { return nil }
func (m *taskOnlyFakeCM) AddStep(step agent.Step)                           {}
func (m *taskOnlyFakeCM) Compact(_ context.Context) *agent.CompactionResult { return nil }
func (m *taskOnlyFakeCM) SetStrategy(_ agent.CompactionStrategy)            {}
func (m *taskOnlyFakeCM) CheckFill() agent.FillCheck                        { return agent.FillCheck{} }
func (m *taskOnlyFakeCM) CorrectTokenCount(_ int)                           {}
func (m *taskOnlyFakeCM) FillPercent() float64                              { return 0 }
func (m *taskOnlyFakeCM) AvailableTokens() int                              { return 100000 }
func (m *taskOnlyFakeCM) OutputLimit() int                                  { return 4096 }
func (m *taskOnlyFakeCM) VulnerableOutputs() []agent.VulnerableOutput       { return nil }

func (m *taskOnlyFakeCM) SetTask(task string) {
	m.task = task
	m.taskCalled = true
}

// Compile-time checks that the fakes satisfy the capability interfaces.
var _ BlockTaskAware = (*blockFakeCM)(nil)
var _ TaskAware = (*blockFakeCM)(nil)
var _ TaskAware = (*taskOnlyFakeCM)(nil)

func TestSetTaskOnContextManager(t *testing.T) {
	blocks := []llm.ContentBlock{
		{Type: "text", Text: "analyze"},
		{Type: "image", MediaType: "image/png", ImageB64: "abc"},
	}

	t.Run("BlockTaskAware CM with blocks calls SetTaskWithBlocks", func(t *testing.T) {
		cm := &blockFakeCM{}
		setTaskOnContextManager(cm, "do work", blocks)
		if !cm.blockCalled {
			t.Error("expected SetTaskWithBlocks to be called, it was not")
		}
		if cm.taskCalled {
			t.Error("expected SetTask to NOT be called when BlockTaskAware succeeds, it was")
		}
		if cm.task != "do work" {
			t.Errorf("task = %q, want %q", cm.task, "do work")
		}
		if len(cm.taskBlocks) != 2 {
			t.Errorf("taskBlocks len = %d, want 2", len(cm.taskBlocks))
		}
	})

	t.Run("BlockTaskAware CM with empty blocks falls back to SetTask", func(t *testing.T) {
		cm := &blockFakeCM{}
		setTaskOnContextManager(cm, "do work", nil)
		if cm.blockCalled {
			t.Error("expected SetTaskWithBlocks to NOT be called with empty blocks, it was")
		}
		if !cm.taskCalled {
			t.Error("expected SetTask to be called as fallback, it was not")
		}
		if cm.task != "do work" {
			t.Errorf("task = %q, want %q", cm.task, "do work")
		}
	})

	t.Run("TaskAware-only CM with blocks falls back to SetTask", func(t *testing.T) {
		cm := &taskOnlyFakeCM{}
		setTaskOnContextManager(cm, "do work", blocks)
		if cm.blockCalled {
			t.Error("expected SetTaskWithBlocks to NOT be called on TaskAware-only CM, it was")
		}
		if !cm.taskCalled {
			t.Error("expected SetTask to be called, it was not")
		}
		if cm.task != "do work" {
			t.Errorf("task = %q, want %q", cm.task, "do work")
		}
	})

	t.Run("TaskAware-only CM without blocks calls SetTask", func(t *testing.T) {
		cm := &taskOnlyFakeCM{}
		setTaskOnContextManager(cm, "do work", nil)
		if !cm.taskCalled {
			t.Error("expected SetTask to be called, it was not")
		}
		if cm.task != "do work" {
			t.Errorf("task = %q, want %q", cm.task, "do work")
		}
	})
}
