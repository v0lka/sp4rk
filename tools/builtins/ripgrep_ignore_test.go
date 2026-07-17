package builtins

import (
	"context"
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"

	"github.com/v0lka/sp4rk/tools"
)

// TestRipgrepTool_HonorsRootAiignoreWithChecker verifies that, when an
// IgnoreChecker is plumbed through the context, ripgrep honours a root-level
// .aiignore via --ignore-file: files listed in .aiignore are excluded from
// search results while non-ignored files are still searched. (.aiignore is a
// c0wrk-specific name that ripgrep does NOT honour by default.)
//
// Note: ripgrep only consults the checker for PRESENCE (non-nil); it never
// queries it per-entry. rg respects .gitignore natively, and the checker gate
// additionally enables the root .aiignore via --ignore-file.
func TestRipgrepTool_HonorsRootAiignoreWithChecker(t *testing.T) {
	base := t.TempDir()
	ignoreWriteFile(t, filepath.Join(base, ".aiignore"), "secret.go\n")
	ignoreWriteFile(t, filepath.Join(base, "main.go"), "package main\n\n// needle here\n")
	ignoreWriteFile(t, filepath.Join(base, "secret.go"), "package secret\n\n// needle here\n")

	ctx := tools.WithIgnoreChecker(context.Background(), ignoreCheckerForDir(t, base))

	tool := NewRipgrepTool()
	input, _ := json.Marshal(RipgrepInput{Pattern: "needle", Path: base})
	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Fatalf("unexpected error result: %s", result.Content)
	}

	if !strings.Contains(result.Content, "main.go") {
		t.Errorf("expected non-ignored main.go to be searched, got: %s", result.Content)
	}
	if strings.Contains(result.Content, "secret.go") {
		t.Errorf("expected aiignored secret.go to be excluded, got: %s", result.Content)
	}
}

// TestRipgrepTool_NoCheckerSearchesAiignoredFile is the no-regression guard:
// with NO checker in context, ripgrep must behave exactly as today. Since rg
// does not natively honour .aiignore, the aiignored file must still be found.
func TestRipgrepTool_NoCheckerSearchesAiignoredFile(t *testing.T) {
	base := t.TempDir()
	ignoreWriteFile(t, filepath.Join(base, ".aiignore"), "secret.go\n")
	ignoreWriteFile(t, filepath.Join(base, "secret.go"), "package secret\n\n// needle here\n")

	// No WithIgnoreChecker: plain context.
	ctx := context.Background()

	tool := NewRipgrepTool()
	input, _ := json.Marshal(RipgrepInput{Pattern: "needle", Path: base})
	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Fatalf("unexpected error result: %s", result.Content)
	}

	if !strings.Contains(result.Content, "secret.go") {
		t.Errorf("expected secret.go to STILL be found with no checker (no regression), got: %s", result.Content)
	}
}

// TestRipgrepTool_HonorsRootAiignoreUnderWorkDir verifies the .aiignore
// honouring works when the search root is a work-directory that is distinct
// from the workspace in context. resolvePath symlink-resolves the search root
// when a workspace is present, and os.Stat still locates the .aiignore at the
// resolved root, so the --ignore-file flag is applied correctly.
func TestRipgrepTool_HonorsRootAiignoreUnderWorkDir(t *testing.T) {
	workspace := t.TempDir()
	workDir := t.TempDir()

	wsResolved, err := filepath.EvalSymlinks(workspace)
	if err != nil {
		wsResolved = workspace
	}
	// The checker only needs to be non-nil for ripgrep; its root form is
	// irrelevant because rg is never queried through it per-entry.
	ctx := tools.WithWorkspacePath(
		tools.WithIgnoreChecker(context.Background(), ignoreCheckerForDir(t, workDir)),
		wsResolved,
	)

	ignoreWriteFile(t, filepath.Join(workDir, ".aiignore"), "hidden.go\n")
	ignoreWriteFile(t, filepath.Join(workDir, "visible.go"), "package main\n\n// target token\n")
	ignoreWriteFile(t, filepath.Join(workDir, "hidden.go"), "package hidden\n\n// target token\n")

	tool := NewRipgrepTool()
	input, _ := json.Marshal(RipgrepInput{Pattern: "target token", Path: workDir})
	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Fatalf("unexpected error result: %s", result.Content)
	}

	if !strings.Contains(result.Content, "visible.go") {
		t.Errorf("expected visible.go to be searched under work-dir, got: %s", result.Content)
	}
	if strings.Contains(result.Content, "hidden.go") {
		t.Errorf("expected aiignored hidden.go to be excluded under work-dir, got: %s", result.Content)
	}
}
