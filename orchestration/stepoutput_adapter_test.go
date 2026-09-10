package orchestration

import (
	"errors"
	"testing"

	"github.com/v0lka/sp4rk/agent"
)

func TestNewStepOutputStore_GetStepOutput(t *testing.T) {
	bb := NewMapBlackboard()
	bb.SetStepResult("step_1", "full output here", nil, nil)

	store := NewStepOutputStore(bb)

	output, ok := store.GetStepOutput("step_1")
	if !ok {
		t.Fatal("expected step_1 to exist")
	}
	if output != "full output here" {
		t.Errorf("got %q, want %q", output, "full output here")
	}
}

func TestNewStepOutputStore_GetStepOutput_NotFound(t *testing.T) {
	bb := NewMapBlackboard()
	store := NewStepOutputStore(bb)

	_, ok := store.GetStepOutput("nonexistent")
	if ok {
		t.Error("expected ok=false for nonexistent step")
	}
}

func TestNewStepOutputStore_GetStepOutput_ErrorStep(t *testing.T) {
	bb := NewMapBlackboard()
	bb.SetStepResult("step_1", "partial output", errors.New("something failed"), nil)

	store := NewStepOutputStore(bb)

	_, ok := store.GetStepOutput("step_1")
	if ok {
		t.Error("expected ok=false for step with error")
	}
}

func TestNewStepOutputStore_GetStepOutput_EmptyOutput(t *testing.T) {
	bb := NewMapBlackboard()
	bb.SetStepResult("step_1", "", nil, nil)

	store := NewStepOutputStore(bb)

	_, ok := store.GetStepOutput("step_1")
	if ok {
		t.Error("expected ok=false for step with empty output")
	}
}

func TestNewStepOutputStore_ListStepOutputs(t *testing.T) {
	bb := NewMapBlackboard()
	bb.SetStepResult("step_2", "second output", nil, nil)
	bb.SetStepResult("step_1", "first output", nil, nil)
	bb.SetStepResult("step_3", "third output", errors.New("fail"), nil) // failed: excluded
	// Paused checkpoint: surfaced with an explicit marker (a hidden step
	// reads as vanished/failed to the model).
	bb.SetStepResult("step_0", "", agent.ErrPaused, nil)

	store := NewStepOutputStore(bb)

	entries := store.ListStepOutputs()
	if len(entries) != 3 {
		t.Fatalf("expected 3 entries, got %d", len(entries))
	}

	// Deterministic order: sorted by step ID
	if entries[0].StepID != "step_0" {
		t.Errorf("entries[0].StepID = %q, want %q", entries[0].StepID, "step_0")
	}
	if entries[0].FullOutput != pausedCheckpointOutput {
		t.Errorf("entries[0].FullOutput = %q, want the paused marker", entries[0].FullOutput)
	}
	if entries[1].StepID != "step_1" {
		t.Errorf("entries[1].StepID = %q, want %q", entries[1].StepID, "step_1")
	}
	if entries[2].StepID != "step_2" {
		t.Errorf("entries[2].StepID = %q, want %q", entries[2].StepID, "step_2")
	}
}

// TestNewStepOutputStore_PausedCheckpointReadable verifies read_step_output's
// backing store returns the paused marker (ok=true) for a checkpointed step —
// including the string-reconstructed sentinel form a host produces when it
// persists error text and rebuilds it via errors.New on restore.
func TestNewStepOutputStore_PausedCheckpointReadable(t *testing.T) {
	bb := NewMapBlackboard()
	bb.SetStepResult("step_p", "", agent.ErrPaused, nil)
	store := NewStepOutputStore(bb)

	out, ok := store.GetStepOutput("step_p")
	if !ok || out != pausedCheckpointOutput {
		t.Fatalf("GetStepOutput(paused) = (%q, %v), want (%q, true)", out, ok, pausedCheckpointOutput)
	}

	// String-reconstructed sentinel (persistence round-trip form).
	bb2 := NewMapBlackboard()
	bb2.SetStepResult("step_r", "", errors.New(agent.ErrPaused.Error()), nil)
	store2 := NewStepOutputStore(bb2)
	out2, ok2 := store2.GetStepOutput("step_r")
	if !ok2 || out2 != pausedCheckpointOutput {
		t.Fatalf("GetStepOutput(round-tripped paused) = (%q, %v), want (%q, true)", out2, ok2, pausedCheckpointOutput)
	}

	// A genuinely failed step stays invisible.
	bb3 := NewMapBlackboard()
	bb3.SetStepResult("step_f", "partial", errors.New("boom"), nil)
	if out, ok := NewStepOutputStore(bb3).GetStepOutput("step_f"); ok {
		t.Fatalf("GetStepOutput(failed) = (%q, true), want invisible", out)
	}
}

func TestNewStepOutputStore_ListStepOutputs_Empty(t *testing.T) {
	bb := NewMapBlackboard()
	store := NewStepOutputStore(bb)

	entries := store.ListStepOutputs()
	if len(entries) != 0 {
		t.Errorf("expected 0 entries, got %d", len(entries))
	}
}

// Compile-time check that the adapter implements the interface.
var _ agent.StepOutputStore = (*blackboardStepOutputStore)(nil)
