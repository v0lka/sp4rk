package builtins

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/v0lka/sp4rk/pathutil"
	"github.com/v0lka/sp4rk/tools"
)

// --- autoCorrectPath unit tests (containment gate) ---

func TestAutoCorrectPath_RepairsWithinRoot(t *testing.T) {
	ws := t.TempDir()
	ctx := tools.WithWorkspacePath(context.Background(), ws)

	original := filepath.Join(ws, "src", "src", "f.txt")
	repaired, ok := autoCorrectPath(ctx, original)
	if !ok {
		t.Fatalf("expected repair to succeed, got ok=false (repaired=%q)", repaired)
	}
	if want := filepath.Join(ws, "src", "f.txt"); repaired != want {
		t.Errorf("repaired = %q, want %q", repaired, want)
	}
}

func TestAutoCorrectPath_RejectsContainmentEscape(t *testing.T) {
	ws := t.TempDir()
	if err := os.MkdirAll(filepath.Join(ws, "x"), 0o755); err != nil {
		t.Fatal(err)
	}
	ctx := tools.WithWorkspacePath(context.Background(), ws)

	// The doubled "x" directory buffers one ".." climb: the original spelling
	// POSIX-resolves to <ws>/f.txt (inside the root), while collapsing the
	// duplicate removes the buffer and lets the second ".." escape the root
	// (parent-of-workspace/f.txt). The spelling is raw — no filepath.Join —
	// so the ".." segments survive to the repair attempt.
	sep := string(filepath.Separator)
	original := strings.Join([]string{ws, "x", "x", "..", "..", "f.txt"}, sep)

	// Pre-condition: the original spelling resolves INSIDE the root, so the
	// gate below rejects on the repair's escape, not on a missing root.
	if within, err := pathutil.IsWithinPath(ws, original); err != nil || !within {
		t.Fatalf("pre-condition failed: original %q within %q = %v (err=%v)", original, ws, within, err)
	}

	repaired, ok := autoCorrectPath(ctx, original)
	if ok {
		t.Errorf("expected containment-escaping repair to be rejected, got ok=true repaired=%q", repaired)
	}
}

func TestAutoCorrectPath_RefusedOutsideRoots(t *testing.T) {
	ws := t.TempDir()
	// No session roots attached: an absolute path outside every root is read
	// with user confirmation, and auto-correction would silently re-point an
	// explicitly chosen path. The repair must be refused even though a
	// duplicated spelling exists.
	ctx := context.Background()

	original := filepath.Join(ws, "src", "src", "f.txt")
	if _, ok := autoCorrectPath(ctx, original); ok {
		t.Error("expected repair to be refused for a path outside every session root")
	}
}

func TestAutoCorrectPath_UnchangedSpellingNotRepaired(t *testing.T) {
	ws := t.TempDir()
	ctx := tools.WithWorkspacePath(context.Background(), ws)

	original := filepath.Join(ws, "src", "f.txt")
	if _, ok := autoCorrectPath(ctx, original); ok {
		t.Error("expected no repair for a spelling without duplicated components")
	}
}

// --- read_file tool-level tests ---

func TestReadFileTool_AutoCorrectsDoubledPrefix(t *testing.T) {
	tool := NewReadFileTool()
	ws := t.TempDir()
	ctx := tools.WithWorkspacePath(context.Background(), ws)

	deepDir := filepath.Join(ws, "deep")
	if err := os.MkdirAll(deepDir, 0o755); err != nil {
		t.Fatal(err)
	}
	testFile := filepath.Join(deepDir, "file.txt")
	if err := os.WriteFile(testFile, []byte("autocorrect target"), 0o644); err != nil {
		t.Fatal(err)
	}

	// Doubled workspace prefix: the root component sequence appears twice in
	// a row — the classic raw concatenation of the workspace root onto an
	// already-absolute path. The doubled directory does NOT exist on disk,
	// so the spelling fails to open and the repair is the only way through.
	doubled := filepath.Join(ws, filepath.Base(ws), "deep", "file.txt")

	input, _ := json.Marshal(map[string]string{"path": doubled})
	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Fatalf("expected auto-corrected success, got error: %s", result.Content)
	}
	if !strings.Contains(result.Content, "[path auto-corrected:") {
		t.Errorf("expected result to carry the auto-correction flag, got: %s", result.Content)
	}
	if !strings.Contains(result.Content, "→") {
		t.Errorf("expected the flag to name both spellings (X → Y), got: %s", result.Content)
	}
	if !strings.Contains(result.Content, "autocorrect target") {
		t.Errorf("expected the repaired file's content, got: %s", result.Content)
	}
}

func TestReadFileTool_AutoCorrectsAdjacentDuplicate(t *testing.T) {
	tool := NewReadFileTool()
	ws := t.TempDir()
	ctx := tools.WithWorkspacePath(context.Background(), ws)

	srcDir := filepath.Join(ws, "src")
	if err := os.MkdirAll(srcDir, 0o755); err != nil {
		t.Fatal(err)
	}
	testFile := filepath.Join(srcDir, "data.txt")
	if err := os.WriteFile(testFile, []byte("adjacent duplicate target"), 0o644); err != nil {
		t.Fatal(err)
	}

	// <dir>/<dir>/file: the "src" component repeats, "src/src" does not exist.
	doubled := filepath.Join(ws, "src", "src", "data.txt")

	input, _ := json.Marshal(map[string]string{"path": doubled})
	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Fatalf("expected auto-corrected success, got error: %s", result.Content)
	}
	if !strings.Contains(result.Content, "[path auto-corrected:") {
		t.Errorf("expected result to carry the auto-correction flag, got: %s", result.Content)
	}
	if !strings.Contains(result.Content, "adjacent duplicate target") {
		t.Errorf("expected the repaired file's content, got: %s", result.Content)
	}
}

func TestReadFileTool_NoAutoCorrectForUniqueMissingPath(t *testing.T) {
	tool := NewReadFileTool()
	ws := t.TempDir()
	ctx := tools.WithWorkspacePath(context.Background(), ws)

	// A genuinely missing file with a unique spelling: no repair is possible
	// (nothing to de-duplicate), so the error must stay exactly as before.
	input, _ := json.Marshal(map[string]string{"path": filepath.Join(ws, "missing", "file.txt")})
	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Fatal("expected IsError=true for a missing file")
	}
	if !strings.Contains(result.Content, "failed to read file") {
		t.Errorf("expected the original error message, got: %s", result.Content)
	}
	if strings.Contains(result.Content, "auto-corrected") {
		t.Errorf("expected no auto-correction flag, got: %s", result.Content)
	}
}

func TestReadFileTool_AutoCorrectedRepairFailureKeepsOriginalError(t *testing.T) {
	tool := NewReadFileTool()
	ws := t.TempDir()
	ctx := tools.WithWorkspacePath(context.Background(), ws)

	// A duplicated spelling whose repair ALSO misses (no src directory at
	// all): the retry fails and the original error must surface unchanged.
	doubled := filepath.Join(ws, "src", "src", "nonexistent.txt")
	input, _ := json.Marshal(map[string]string{"path": doubled})
	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Fatal("expected IsError=true")
	}
	if !strings.Contains(result.Content, "failed to read file") {
		t.Errorf("expected the original error message, got: %s", result.Content)
	}
	if strings.Contains(result.Content, "auto-corrected") {
		t.Errorf("expected no auto-correction flag on a failed repair, got: %s", result.Content)
	}
}

// --- coherence interaction with the auto-corrected repair ---

// scriptedCoherenceChecker answers CheckRead calls with pre-scripted
// conflicts in order and records every path it was asked about, so a test can
// stage a stale snapshot for the failed spelling and a clean (or conflicting)
// state for the repaired one.
type scriptedCoherenceChecker struct {
	mu      sync.Mutex
	answers []*tools.CoherenceConflict
	asked   []string
}

func (c *scriptedCoherenceChecker) CheckRead(_ context.Context, path string) *tools.CoherenceConflict {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.asked = append(c.asked, path)
	if len(c.answers) == 0 {
		return nil
	}
	ans := c.answers[0]
	c.answers = c.answers[1:]
	return ans
}

func (c *scriptedCoherenceChecker) CheckWrite(_ context.Context, _ string) *tools.CoherenceConflict {
	return nil
}

func (c *scriptedCoherenceChecker) RecordWrite(_ context.Context, _ string)  {}
func (c *scriptedCoherenceChecker) RecordDelete(_ context.Context, _ string) {}
func (c *scriptedCoherenceChecker) PurgeSession(_ string)                    {}
func (c *scriptedCoherenceChecker) Lock(_ string)                            {}
func (c *scriptedCoherenceChecker) Unlock(_ string)                          {}

// TestReadFileTool_RepairDropsStaleCoherenceWarning pins the repair path's
// coherence handling: the pre-repair check targeted the FAILED spelling, so
// its conflict must not shadow a clean read of the repaired path — a
// successful repair resets the warning before re-checking the new path.
func TestReadFileTool_RepairDropsStaleCoherenceWarning(t *testing.T) {
	tool := NewReadFileTool()
	ws := t.TempDir()

	srcDir := filepath.Join(ws, "src")
	if err := os.MkdirAll(srcDir, 0o755); err != nil {
		t.Fatal(err)
	}
	testFile := filepath.Join(srcDir, "data.txt")
	if err := os.WriteFile(testFile, []byte("clean repaired target"), 0o644); err != nil {
		t.Fatal(err)
	}

	// First CheckRead (the failed <ws>/src/src/data.txt spelling) reports a
	// conflict — a stale snapshot left over for the non-existent spelling;
	// second CheckRead (the repaired path) is clean.
	checker := &scriptedCoherenceChecker{
		answers: []*tools.CoherenceConflict{
			{ModifiedBy: "stale snapshot session"},
			nil,
		},
	}
	ctx := tools.WithCoherence(tools.WithWorkspacePath(context.Background(), ws), checker)

	doubled := filepath.Join(ws, "src", "src", "data.txt")
	input, _ := json.Marshal(map[string]string{"path": doubled})
	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Fatalf("expected auto-corrected success, got error: %s", result.Content)
	}
	if !strings.Contains(result.Content, "[path auto-corrected:") {
		t.Errorf("expected the auto-correction flag, got: %s", result.Content)
	}
	if len(checker.asked) != 2 {
		t.Fatalf("CheckRead calls = %d (%v), want 2 (failed spelling + repaired path)", len(checker.asked), checker.asked)
	}
	if strings.Contains(result.Content, "was modified by") {
		t.Errorf("the failed spelling's stale coherence warning leaked into the repaired read: %s", result.Content)
	}
	if !strings.Contains(result.Content, "clean repaired target") {
		t.Errorf("expected the repaired file's content, got: %s", result.Content)
	}
}

// TestReadFileTool_RepairReportsRepairedPathConflict is the complement case:
// when the REPAIRED path itself has a coherence conflict, the warning must
// still surface (resetting the stale snapshot must not suppress real ones).
func TestReadFileTool_RepairReportsRepairedPathConflict(t *testing.T) {
	tool := NewReadFileTool()
	ws := t.TempDir()

	srcDir := filepath.Join(ws, "src")
	if err := os.MkdirAll(srcDir, 0o755); err != nil {
		t.Fatal(err)
	}
	testFile := filepath.Join(srcDir, "data.txt")
	if err := os.WriteFile(testFile, []byte("conflicting repaired target"), 0o644); err != nil {
		t.Fatal(err)
	}

	checker := &scriptedCoherenceChecker{
		answers: []*tools.CoherenceConflict{
			nil,
			{ModifiedBy: "other session"},
		},
	}
	ctx := tools.WithCoherence(tools.WithWorkspacePath(context.Background(), ws), checker)

	doubled := filepath.Join(ws, "src", "src", "data.txt")
	input, _ := json.Marshal(map[string]string{"path": doubled})
	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Fatalf("expected auto-corrected success, got error: %s", result.Content)
	}
	if !strings.Contains(result.Content, `[!] This file was modified by "other session"`) {
		t.Errorf("expected the repaired path's coherence warning, got: %s", result.Content)
	}
	if !strings.Contains(result.Content, "conflicting repaired target") {
		t.Errorf("expected the repaired file's content, got: %s", result.Content)
	}
}
