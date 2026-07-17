package builtins

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/v0lka/sp4rk/ignore"
	"github.com/v0lka/sp4rk/tools"
)

// ignoreCheckerForDir builds a single-root IgnoreChecker rooted at dir WITHOUT
// symlink-resolving the root. This deliberately matches the path form glob
// uses: when no workspace is in context, resolvePath returns the search path
// as-is (raw), so the checker root must also be raw for filepath.Rel inside
// the resolver to succeed. Tests that DO set a workspace (and therefore cause
// resolvePath to symlink-resolve) build their checker from resolved roots
// explicitly (see TestGlobTool_HonorsIgnoreAtWorkDirRoot).
func ignoreCheckerForDir(t *testing.T, dir string) ignore.IgnoreChecker {
	t.Helper()
	r, err := ignore.NewResolver(dir)
	if err != nil {
		t.Fatalf("ignore.NewResolver(%s): %v", dir, err)
	}
	return r
}

// ignoreResultHas reports whether relPath appears as an exact line in the glob
// output (results are root-relative paths joined by newlines).
func ignoreResultHas(content, relPath string) bool {
	for _, l := range strings.Split(content, "\n") {
		if strings.TrimSpace(l) == relPath {
			return true
		}
	}
	return false
}

func ignoreWriteFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

// TestGlobTool_HonorsGitignoreWithChecker verifies that, when an IgnoreChecker
// is plumbed through the context, files matched by a root .gitignore are
// omitted from glob results (files + nested files).
func TestGlobTool_HonorsGitignoreWithChecker(t *testing.T) {
	base := t.TempDir()
	if err := os.MkdirAll(filepath.Join(base, "nested"), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	ignoreWriteFile(t, filepath.Join(base, ".gitignore"), "*.log\n")
	ignoreWriteFile(t, filepath.Join(base, "app.go"), "package main\n")
	ignoreWriteFile(t, filepath.Join(base, "app.log"), "irrelevant\n")
	ignoreWriteFile(t, filepath.Join(base, "nested", "debug.log"), "irrelevant\n")

	ctx := tools.WithIgnoreChecker(context.Background(), ignoreCheckerForDir(t, base))

	tool := NewGlobTool()
	input, _ := json.Marshal(GlobInput{Pattern: "**/*", Path: base, Type: "all"})
	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Fatalf("unexpected error result: %s", result.Content)
	}

	if !ignoreResultHas(result.Content, "app.go") {
		t.Errorf("expected non-ignored app.go to be present, got: %s", result.Content)
	}
	if ignoreResultHas(result.Content, "app.log") {
		t.Errorf("expected gitignored app.log to be omitted, got: %s", result.Content)
	}
	if ignoreResultHas(result.Content, "nested/debug.log") {
		t.Errorf("expected gitignored nested/debug.log to be omitted, got: %s", result.Content)
	}
}

// TestGlobTool_HonorsAiignoreWithChecker verifies that .aiignore patterns are
// honoured by glob when a checker is in context.
func TestGlobTool_HonorsAiignoreWithChecker(t *testing.T) {
	base := t.TempDir()
	ignoreWriteFile(t, filepath.Join(base, ".aiignore"), "secret.txt\n")
	ignoreWriteFile(t, filepath.Join(base, "keep.txt"), "data\n")
	ignoreWriteFile(t, filepath.Join(base, "secret.txt"), "topsecret\n")

	ctx := tools.WithIgnoreChecker(context.Background(), ignoreCheckerForDir(t, base))

	tool := NewGlobTool()
	input, _ := json.Marshal(GlobInput{Pattern: "*.txt", Path: base})
	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Fatalf("unexpected error result: %s", result.Content)
	}

	if !ignoreResultHas(result.Content, "keep.txt") {
		t.Errorf("expected keep.txt to be present, got: %s", result.Content)
	}
	if ignoreResultHas(result.Content, "secret.txt") {
		t.Errorf("expected aiignored secret.txt to be omitted, got: %s", result.Content)
	}
}

// TestGlobTool_NoCheckerReturnsIgnoredFiles is the no-regression guard: with
// NO checker in context, glob must return exactly today's behaviour, including
// files that would be ignored.
func TestGlobTool_NoCheckerReturnsIgnoredFiles(t *testing.T) {
	base := t.TempDir()
	ignoreWriteFile(t, filepath.Join(base, ".gitignore"), "*.log\n")
	ignoreWriteFile(t, filepath.Join(base, "app.go"), "package main\n")
	ignoreWriteFile(t, filepath.Join(base, "app.log"), "irrelevant\n")

	// No WithIgnoreChecker: plain context.
	ctx := context.Background()

	tool := NewGlobTool()
	input, _ := json.Marshal(GlobInput{Pattern: "**/*", Path: base, Type: "all"})
	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Fatalf("unexpected error result: %s", result.Content)
	}

	if !ignoreResultHas(result.Content, "app.go") {
		t.Errorf("expected app.go to be present, got: %s", result.Content)
	}
	if !ignoreResultHas(result.Content, "app.log") {
		t.Errorf("expected app.log to STILL be present with no checker (no regression), got: %s", result.Content)
	}
}

// TestGlobTool_HonorsIgnoreAtWorkDirRoot verifies that ignore filtering works
// for a work-directory root that is distinct from the workspace in context.
// The checker is a Multi spanning a (dummy) workspace plus the work dir; glob
// runs against the work dir and must still omit its ignored files. This proves
// the "any work-directory root" requirement.
func TestGlobTool_HonorsIgnoreAtWorkDirRoot(t *testing.T) {
	workspace := t.TempDir()
	workDir := t.TempDir()

	ignoreWriteFile(t, filepath.Join(workDir, ".gitignore"), "*.tmp\n")
	ignoreWriteFile(t, filepath.Join(workDir, "keep.go"), "package x\n")
	ignoreWriteFile(t, filepath.Join(workDir, "drop.tmp"), "data\n")

	wsResolved, err := filepath.EvalSymlinks(workspace)
	if err != nil {
		wsResolved = workspace
	}
	wdResolved, err := filepath.EvalSymlinks(workDir)
	if err != nil {
		wdResolved = workDir
	}
	checker, err := ignore.NewMulti(wsResolved, wdResolved)
	if err != nil {
		t.Fatalf("ignore.NewMulti: %v", err)
	}

	// Workspace in context is the OTHER root, proving the checker applies to
	// the work-directory root independently of the workspace.
	ctx := tools.WithWorkspacePath(tools.WithIgnoreChecker(context.Background(), checker), wsResolved)

	tool := NewGlobTool()
	input, _ := json.Marshal(GlobInput{Pattern: "**/*", Path: workDir, Type: "all"})
	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Fatalf("unexpected error result: %s", result.Content)
	}

	if !ignoreResultHas(result.Content, "keep.go") {
		t.Errorf("expected keep.go to be present, got: %s", result.Content)
	}
	if ignoreResultHas(result.Content, "drop.tmp") {
		t.Errorf("expected gitignored drop.tmp to be omitted under work-dir root, got: %s", result.Content)
	}
}

// TestGlobTool_NestedIgnoreFile verifies that a nested .gitignore (inside a
// subdirectory) is honoured by glob via the checker.
func TestGlobTool_NestedIgnoreFile(t *testing.T) {
	base := t.TempDir()
	if err := os.MkdirAll(filepath.Join(base, "sub"), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	// sub/.gitignore ignores *.bak; a .bak at the ROOT is NOT ignored.
	ignoreWriteFile(t, filepath.Join(base, "sub", ".gitignore"), "*.bak\n")
	ignoreWriteFile(t, filepath.Join(base, "root.bak"), "data\n")
	ignoreWriteFile(t, filepath.Join(base, "sub", "local.bak"), "data\n")
	ignoreWriteFile(t, filepath.Join(base, "sub", "keep.go"), "package sub\n")

	ctx := tools.WithIgnoreChecker(context.Background(), ignoreCheckerForDir(t, base))

	tool := NewGlobTool()
	input, _ := json.Marshal(GlobInput{Pattern: "**/*", Path: base, Type: "all"})
	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Fatalf("unexpected error result: %s", result.Content)
	}

	if !ignoreResultHas(result.Content, "root.bak") {
		t.Errorf("expected root.bak (not covered by nested ignore) to be present, got: %s", result.Content)
	}
	if ignoreResultHas(result.Content, "sub/local.bak") {
		t.Errorf("expected sub/local.bak (covered by nested .gitignore) to be omitted, got: %s", result.Content)
	}
}
