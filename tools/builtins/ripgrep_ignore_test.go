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

// TestRipgrepTool_HonorsRootAiignoreWithChecker verifies that, when an
// IgnoreChecker is plumbed through the context, ripgrep honours a root-level
// .aiignore: files listed in .aiignore are excluded from search results while
// non-ignored files are still searched. (.aiignore is a c0wrk-specific name
// that ripgrep does NOT honour by default.)
//
// Mechanism: ripgrep post-filters every emitted match/context path through the
// checker (the same single ignore authority glob uses). rg still respects
// .gitignore natively; the checker additionally applies .aiignore rules rg
// cannot see.
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
// from the workspace in context. The checker is consulted per-entry now, and
// the IgnoreChecker snapshots ignore files at construction — so the .aiignore
// must be written BEFORE the checker is built.
func TestRipgrepTool_HonorsRootAiignoreUnderWorkDir(t *testing.T) {
	workspace := t.TempDir()
	workDir := t.TempDir()

	// Write ignore files FIRST: the resolver walks the root once at
	// construction and will not see files created afterwards.
	ignoreWriteFile(t, filepath.Join(workDir, ".aiignore"), "hidden.go\n")
	ignoreWriteFile(t, filepath.Join(workDir, "visible.go"), "package main\n\n// target token\n")
	ignoreWriteFile(t, filepath.Join(workDir, "hidden.go"), "package hidden\n\n// target token\n")

	wsResolved, err := filepath.EvalSymlinks(workspace)
	if err != nil {
		wsResolved = workspace
	}
	// The checker is consulted per-entry now, so its root form matters. It is
	// rooted at the work-dir (raw form); NewResolver canonicalizes that to the
	// resolved form, which matches the paths rg emits under a resolved search
	// root (a workspace in context makes resolvePath resolve workDir). The
	// workspace in context is a DIFFERENT root, proving the checker applies to
	// the search root independently.
	ctx := tools.WithWorkspacePath(
		tools.WithIgnoreChecker(context.Background(), ignoreCheckerForDir(t, workDir)),
		wsResolved,
	)

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

// TestRipgrepTool_PostFiltersNestedAiignore verifies that ripgrep now honours a
// NESTED .aiignore (one inside a subdirectory), not just the root one. This is
// the behaviour the old --ignore-file approach could not provide (rg's
// --ignore-file only loads a single root-level file). With per-entry post-
// filtering through the checker, nested ignore files are honoured exactly as
// glob honours them — both tools share one source of truth.
func TestRipgrepTool_PostFiltersNestedAiignore(t *testing.T) {
	base := t.TempDir()
	if err := os.MkdirAll(filepath.Join(base, "pkg"), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	// A NESTED .aiignore inside pkg/ that ignores *.gen.go there.
	ignoreWriteFile(t, filepath.Join(base, "pkg", ".aiignore"), "*.gen.go\n")
	ignoreWriteFile(t, filepath.Join(base, "pkg", "real.go"), "package pkg\n\n// marker token\n")
	ignoreWriteFile(t, filepath.Join(base, "pkg", "gen.gen.go"), "package pkg\n\n// marker token\n")

	ctx := tools.WithIgnoreChecker(context.Background(), ignoreCheckerForDir(t, base))

	tool := NewRipgrepTool()
	input, _ := json.Marshal(RipgrepInput{Pattern: "marker token", Path: base})
	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Fatalf("unexpected error result: %s", result.Content)
	}

	if !strings.Contains(result.Content, "real.go") {
		t.Errorf("expected non-ignored pkg/real.go to be searched, got: %s", result.Content)
	}
	if strings.Contains(result.Content, "gen.gen.go") {
		t.Errorf("expected nested-aiignored pkg/gen.gen.go to be excluded, got: %s", result.Content)
	}
}

// TestRipgrepTool_PostFiltersGitignoredFile verifies that the post-filter does
// not regress gitignore handling: a file matched by a root .gitignore (which rg
// also honours natively) is still excluded, and the count reflects only the
// kept matches.
func TestRipgrepTool_PostFiltersGitignoredFile(t *testing.T) {
	base := t.TempDir()
	ignoreWriteFile(t, filepath.Join(base, ".gitignore"), "*.log\n")
	ignoreWriteFile(t, filepath.Join(base, "keep.go"), "package main\n\n// token here\n")
	ignoreWriteFile(t, filepath.Join(base, "drop.log"), "irrelevant token here\n")

	ctx := tools.WithIgnoreChecker(context.Background(), ignoreCheckerForDir(t, base))

	tool := NewRipgrepTool()
	input, _ := json.Marshal(RipgrepInput{Pattern: "token", Path: base})
	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Fatalf("unexpected error result: %s", result.Content)
	}

	if !strings.Contains(result.Content, "keep.go") {
		t.Errorf("expected keep.go to be searched, got: %s", result.Content)
	}
	if strings.Contains(result.Content, "drop.log") {
		t.Errorf("expected gitignored drop.log to be excluded, got: %s", result.Content)
	}
}
