package builtins

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
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
