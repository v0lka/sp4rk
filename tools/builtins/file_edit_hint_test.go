package builtins

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/v0lka/sp4rk/tools"
)

// --- nearestMatchHint unit tests ---

func TestNearestMatchHint_WhitespaceDrift(t *testing.T) {
	content := "package main\n\nfunc computeTotal(price int, tax float64) float64 {\n\treturn price * tax\n}\n"
	// The anchor double-spaces before the return type: token-identical.
	oldString := "func computeTotal(price int, tax  float64) float64 {"

	hint := nearestMatchHint(content, oldString)
	if hint == "" {
		t.Fatal("expected a nearest-match hint for a whitespace-drifted old_string")
	}
	if !strings.Contains(hint, "[nearest match: line 3:") {
		t.Errorf("expected the hint to name line 3, got: %s", hint)
	}
	if !strings.Contains(hint, "computeTotal") {
		t.Errorf("expected the hint fragment to carry the matched line, got: %s", hint)
	}
}

func TestNearestMatchHint_DissimilarYieldsEmpty(t *testing.T) {
	content := "package main\n\nfunc main() {\n}\n"
	oldString := "zzqqxx completely unrelated gibberish yywwzz"

	if hint := nearestMatchHint(content, oldString); hint != "" {
		t.Errorf("expected no hint for a dissimilar old_string, got: %s", hint)
	}
}

func TestNearestMatchHint_WhitespaceOnlyOldString(t *testing.T) {
	content := "some content\n"
	oldString := "   \n\t\n"

	if hint := nearestMatchHint(content, oldString); hint != "" {
		t.Errorf("expected no hint for a whitespace-only old_string, got: %s", hint)
	}
}

func TestNearestMatchHint_BoundedFragment(t *testing.T) {
	// A single 5 KiB line: the hint fragment must stay bounded, never dump
	// the line (let alone the file).
	longLine := strings.Repeat("abcdefgh ", 600) // ~5.4 KiB
	content := longLine + "\n\nnext line\n"
	// old_string drifts near the start so the anchor tokens still match.
	oldString := longLine[:100] + "X" + longLine[101:200]

	hint := nearestMatchHint(content, oldString)
	if hint == "" {
		t.Fatal("expected a hint for a near-miss long line")
	}
	if len(hint) > 300 {
		t.Errorf("hint is %d bytes — the fragment must stay bounded, got: %.120s...", len(hint), hint)
	}
	if strings.Contains(hint, "next line") {
		t.Errorf("hint leaked content from another line: %s", hint)
	}
}

func TestNearestMatchHint_PicksBestLine(t *testing.T) {
	content := "alpha beta\n\talpha beta gamma\nalpha beta delta\n"
	// The exact whitespace-normalized match is line 2; it must win over the
	// partially-overlapping lines 1 and 3.
	oldString := "\talpha beta gamma"

	hint := nearestMatchHint(content, oldString)
	if hint == "" {
		t.Fatal("expected a hint")
	}
	if !strings.Contains(hint, "[nearest match: line 2:") {
		t.Errorf("expected line 2 (the tab-indented match) to win, got: %s", hint)
	}
}

// TestNearestMatchHint_WeakOverlapYieldsNoHint pins the hintMinScore
// threshold: an old_string with a small-but-NONZERO token overlap must not
// produce a hint — the feature targets near misses, not noise. (The
// dissimilar case with zero overlap is covered by
// TestNearestMatchHint_DissimilarYieldsEmpty, which passed even before the
// threshold was enforced.)
func TestNearestMatchHint_WeakOverlapYieldsNoHint(t *testing.T) {
	content := "the quick brown fox jumps over the lazy dog\n"
	// The anchor shares exactly one token ("the") with the file line:
	// score 1/11 ≈ 0.09, far below the 0.5 threshold, yet nonzero.
	oldString := "the config file"

	if hint := nearestMatchHint(content, oldString); hint != "" {
		t.Errorf("expected no hint for a weak-overlap old_string, got: %s", hint)
	}
}

// TestNearestMatchHint_OverlapAboveThresholdProducesHint is the complement of
// the weak-overlap case: a partial match reaching the 0.5 threshold (3 of 5
// union tokens here) still produces the hint.
func TestNearestMatchHint_OverlapAboveThresholdProducesHint(t *testing.T) {
	content := "alpha beta gamma epsilon\n"
	oldString := "alpha beta gamma delta"

	hint := nearestMatchHint(content, oldString)
	if hint == "" {
		t.Fatal("expected a hint for a 0.6-overlap old_string")
	}
	if !strings.Contains(hint, "[nearest match: line 1:") {
		t.Errorf("expected the hint to name line 1, got: %s", hint)
	}
}

// --- edit_file tool-level tests ---

func TestEditFileTool_NotFoundHintIncludesNearestFragment(t *testing.T) {
	tool := NewEditFileTool()
	ctx := context.Background()
	tmpDir := t.TempDir()

	testFile := filepath.Join(tmpDir, "edit.txt")
	content := "first line\nsecond line\nthird   line with spaces\nlast line\n"
	if err := os.WriteFile(testFile, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	// Same tokens as line 3, drifted whitespace: near miss.
	input, _ := json.Marshal(EditFileInput{
		Path:      testFile,
		OldString: "third line with spaces",
		NewString: "replaced",
	})
	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Fatal("expected IsError=true")
	}
	// A stale/hallucinated old_string is the model's own argument mistake —
	// the same class as the non-unique case, never an execution failure.
	if result.ErrorKind != tools.ToolErrorInvalidCall {
		t.Errorf("ErrorKind = %q, want invalid_call (a wrong old_string is an argument contract failure)", result.ErrorKind)
	}
	if !strings.Contains(result.Content, "old_string not found in file") {
		t.Errorf("expected the original error first, got: %s", result.Content)
	}
	if !strings.Contains(result.Content, "[nearest match: line 3:") {
		t.Errorf("expected a nearest-match fragment hint naming line 3, got: %s", result.Content)
	}

	// The file must be untouched by a failed edit.
	data, err := os.ReadFile(testFile)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != content {
		t.Errorf("failed edit must not modify the file, got: %q", string(data))
	}
}

func TestEditFileTool_NotFoundNoHintForDissimilar(t *testing.T) {
	tool := NewEditFileTool()
	ctx := context.Background()
	tmpDir := t.TempDir()

	testFile := filepath.Join(tmpDir, "edit.txt")
	if err := os.WriteFile(testFile, []byte("alpha\nbeta\ngamma\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	input, _ := json.Marshal(EditFileInput{
		Path:      testFile,
		OldString: "zzqqxx unrelated gibberish yywwzz",
		NewString: "replaced",
	})
	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Fatal("expected IsError=true")
	}
	if result.ErrorKind != tools.ToolErrorInvalidCall {
		t.Errorf("ErrorKind = %q, want invalid_call", result.ErrorKind)
	}
	if result.Content != "old_string not found in file" {
		t.Errorf("error message must stay exactly as before when no fragment is similar enough, got: %q", result.Content)
	}
}

func TestEditFileTool_NotFoundHintBounded(t *testing.T) {
	tool := NewEditFileTool()
	ctx := context.Background()
	tmpDir := t.TempDir()

	testFile := filepath.Join(tmpDir, "edit.txt")
	longLine := strings.Repeat("0123456789 ", 700) // ~7.7 KiB single line
	if err := os.WriteFile(testFile, []byte(longLine+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	drifted := longLine[:50] + "\t" + longLine[51:150]
	input, _ := json.Marshal(EditFileInput{
		Path:      testFile,
		OldString: drifted,
		NewString: "replaced",
	})
	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Fatal("expected IsError=true")
	}
	if !strings.Contains(result.Content, "[nearest match: line 1:") {
		t.Errorf("expected a bounded nearest-match hint for line 1, got: %s", result.Content)
	}
	if len(result.Content) > 400 {
		t.Errorf("error output is %d bytes — the hint must never dump the file, got: %.200s...", len(result.Content), result.Content)
	}
}
