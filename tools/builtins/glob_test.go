package builtins

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/v0lka/sp4rk/tools"
)

// setupGlobTestDir creates a temp directory with nested structure for glob tests.
func setupGlobTestDir(t *testing.T) string {
	t.Helper()

	base := t.TempDir()

	dirs := []string{
		filepath.Join(base, "sub"),
		filepath.Join(base, "other", "deep"),
	}
	for _, d := range dirs {
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatalf("failed to create dir %s: %v", d, err)
		}
	}

	files := []string{
		filepath.Join(base, "sub", "file.go"),
		filepath.Join(base, "sub", "file.txt"),
		filepath.Join(base, "other", "deep", "file.go"),
	}
	for _, f := range files {
		if err := os.WriteFile(f, []byte("content"), 0o644); err != nil {
			t.Fatalf("failed to create file %s: %v", f, err)
		}
	}

	return base
}

func TestGlobTool_Name(t *testing.T) {
	tool := NewGlobTool()
	if tool.Name() != "glob" {
		t.Errorf("expected Name() = %q, got %q", "glob", tool.Name())
	}
}

func TestGlobTool_DefaultPolicy(t *testing.T) {
	tool := NewGlobTool()
	if tool.DefaultPolicy() != tools.PolicyAlwaysAllow {
		t.Errorf("expected DefaultPolicy() = PolicyAlwaysAllow, got %v", tool.DefaultPolicy())
	}
}

func TestGlobTool_FindGoFiles(t *testing.T) {
	base := setupGlobTestDir(t)
	tool := NewGlobTool()

	input, _ := json.Marshal(GlobInput{
		Pattern: "**/*.go",
		Path:    base,
	})

	result, err := tool.Execute(context.Background(), input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Fatalf("expected no error, got: %s", result.Content)
	}

	lines := strings.Split(strings.TrimSpace(result.Content), "\n")
	if len(lines) != 2 {
		t.Errorf("expected 2 .go files, got %d: %v", len(lines), lines)
	}

	content := result.Content
	if !strings.Contains(content, "file.go") {
		t.Errorf("expected result to contain file.go, got: %s", content)
	}
}

func TestGlobTool_FindTxtFiles(t *testing.T) {
	base := setupGlobTestDir(t)
	tool := NewGlobTool()

	input, _ := json.Marshal(GlobInput{
		Pattern: "**/*.txt",
		Path:    base,
	})

	result, err := tool.Execute(context.Background(), input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Fatalf("expected no error, got: %s", result.Content)
	}

	lines := strings.Split(strings.TrimSpace(result.Content), "\n")
	if len(lines) != 1 {
		t.Errorf("expected 1 .txt file, got %d: %v", len(lines), lines)
	}
	if !strings.Contains(result.Content, "file.txt") {
		t.Errorf("expected result to contain file.txt, got: %s", result.Content)
	}
}

func TestGlobTool_FindDirs(t *testing.T) {
	base := setupGlobTestDir(t)
	tool := NewGlobTool()

	input, _ := json.Marshal(GlobInput{
		Pattern: "**",
		Path:    base,
		Type:    "dirs",
	})

	result, err := tool.Execute(context.Background(), input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Fatalf("expected no error, got: %s", result.Content)
	}

	content := result.Content
	if !strings.Contains(content, "sub") {
		t.Errorf("expected result to contain 'sub', got: %s", content)
	}
	if !strings.Contains(content, "other") {
		t.Errorf("expected result to contain 'other', got: %s", content)
	}
}

func TestGlobTool_MaxResults(t *testing.T) {
	base := setupGlobTestDir(t)
	tool := NewGlobTool()

	input, _ := json.Marshal(GlobInput{
		Pattern: "**/*.go",
		Path:    base,
	})

	result, err := tool.Execute(context.Background(), input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Fatalf("expected no error, got: %s", result.Content)
	}

	// No more per-tool truncation; central layer handles it.
	// All matching files should be returned regardless of max_results.
	if strings.Contains(result.Content, "(results limited to") {
		t.Errorf("did not expect truncation message, got: %s", result.Content)
	}

	// Should have more than 1 match since truncation was removed.
	lines := strings.Split(strings.TrimSpace(result.Content), "\n")
	if len(lines) <= 1 {
		t.Errorf("expected more than 1 match since truncation is removed, got %d lines: %s", len(lines), result.Content)
	}
}

func TestGlobTool_NonExistentPath(t *testing.T) {
	tool := NewGlobTool()

	input, _ := json.Marshal(GlobInput{
		Pattern: "**/*.go",
		Path:    "/nonexistent/path/that/does/not/exist",
	})

	result, err := tool.Execute(context.Background(), input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Errorf("expected IsError=true for non-existent path, got content: %s", result.Content)
	}
}

func TestGlobTool_NoMatches(t *testing.T) {
	base := setupGlobTestDir(t)
	tool := NewGlobTool()

	input, _ := json.Marshal(GlobInput{
		Pattern: "**/*.xyz",
		Path:    base,
	})

	result, err := tool.Execute(context.Background(), input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Fatalf("expected no error, got: %s", result.Content)
	}
	if result.Content != "no matching files found" {
		t.Errorf("expected 'no matching files found', got: %s", result.Content)
	}
}

func TestGlobTool_InvalidJSON(t *testing.T) {
	tool := NewGlobTool()

	result, err := tool.Execute(context.Background(), json.RawMessage(`{invalid`))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Errorf("expected IsError=true for invalid JSON")
	}
}

func TestGlobTool_DefaultMaxResults(t *testing.T) {
	base := t.TempDir()
	tool := NewGlobTool()

	// Create many files to potentially exceed the default limit
	for i := 0; i < 250; i++ {
		f := filepath.Join(base, fmt.Sprintf("file%d.txt", i))
		if err := os.WriteFile(f, []byte("content"), 0o644); err != nil {
			t.Fatalf("failed to create file: %v", err)
		}
	}

	input, _ := json.Marshal(GlobInput{
		Pattern: "*.txt",
		Path:    base,
		// MaxResults not specified, should use default of 200
	})

	result, err := tool.Execute(context.Background(), input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Fatalf("expected no error, got: %s", result.Content)
	}

	// No per-tool truncation; all 250 files should be returned.
	if strings.Contains(result.Content, "(results limited to") {
		t.Errorf("did not expect truncation message, got: %s", result.Content)
	}
}

func TestGlobTool_WorkspaceFallback(t *testing.T) {
	workspace := t.TempDir()
	if err := os.WriteFile(filepath.Join(workspace, "hello.txt"), []byte("hi"), 0o644); err != nil {
		t.Fatal(err)
	}

	ctx := tools.WithWorkspacePath(context.Background(), workspace)
	tool := NewGlobTool()

	// Call with no path - should use workspace
	input, _ := json.Marshal(GlobInput{Pattern: "*.txt"})
	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatal(err)
	}
	if result.IsError {
		t.Fatalf("unexpected error: %s", result.Content)
	}
	if !strings.Contains(result.Content, "hello.txt") {
		t.Errorf("expected hello.txt in results, got: %s", result.Content)
	}
}

func TestGlobTool_NoPathNoWorkspace(t *testing.T) {
	ctx := context.Background() // no workspace
	tool := NewGlobTool()

	input, _ := json.Marshal(GlobInput{Pattern: "*.txt"})
	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatal(err)
	}
	if !result.IsError {
		t.Error("expected error when no path and no workspace")
	}
}

// cancelOnFirstIgnore is a test IgnoreChecker that cancels a context the first
// time it is consulted, simulating a mid-walk cancellation deterministically
// (no sleeps): the walk's next filesystem access observes the cancellation.
type cancelOnFirstIgnore struct {
	once   sync.Once
	cancel context.CancelFunc
}

func (c *cancelOnFirstIgnore) Ignored(string, bool) bool {
	c.once.Do(c.cancel)
	return false
}

// TestGlobTool_SymlinkLoopTerminates is the regression repro for the FS-walk
// runaway that hung `glob`. A root containing a self-referential symlink
// (`self -> .`) and a second root containing a symlink to the filesystem root
// (`link -> /`) must not hang the walk: it is non-following
// (doublestar.WithNoFollow) and bounded (entries budget + timeout), so it
// returns within the 2s deadline. Containment is proven by asserting the exact
// result set: a following walk would have traversed `self`/`link` and emitted
// paths under them (e.g. `link/etc/hosts`), which — slash-relative and without a
// `..` prefix — would not trip an IsAbs/`..` check, so the concrete set is the
// assertion that can actually fail.
func TestGlobTool_SymlinkLoopTerminates(t *testing.T) {
	selfRoot := t.TempDir()
	if err := os.Symlink(".", filepath.Join(selfRoot, "self")); err != nil {
		t.Skipf("symlinks unsupported on this filesystem: %v", err)
	}
	if err := os.WriteFile(filepath.Join(selfRoot, "root.go"), []byte("package x\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	escapeRoot := t.TempDir()
	if err := os.MkdirAll(filepath.Join(escapeRoot, "sub"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(escapeRoot, "sub", "keep.go"), []byte("package x\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink("/", filepath.Join(escapeRoot, "link")); err != nil {
		t.Skipf("symlinks unsupported on this filesystem: %v", err)
	}

	cases := []struct {
		name    string
		root    string
		pattern string
		want    []string // exact expected result set (order-insensitive)
	}{
		// The symlink entry itself is listed, but neither loop is traversed:
		// no `self/...` or `link/...` path is ever emitted.
		{"self-loop **/*", selfRoot, "**/*", []string{"root.go", "self"}},
		{"self-loop incident pattern", selfRoot, "**/flowsh@*", nil},
		{"root-escape **/*", escapeRoot, "**/*", []string{"link", "sub", "sub/keep.go"}},
		{"root-escape incident pattern", escapeRoot, "**/flowsh@*", nil},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tool := NewGlobTool()
			input, _ := json.Marshal(GlobInput{Pattern: tc.pattern, Path: tc.root, Type: "all"})

			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()

			type outcome struct {
				result tools.ToolResult
				err    error
			}
			// Buffered so the worker never blocks even if the deadline branch
			// below wins the select.
			done := make(chan outcome, 1)
			go func() {
				res, err := tool.Execute(ctx, input)
				done <- outcome{result: res, err: err}
			}()

			var got outcome
			select {
			case got = <-done:
			case <-ctx.Done():
				t.Fatalf("glob did not terminate within the 2s deadline (pattern %q)", tc.pattern)
			}

			if got.err != nil {
				t.Fatalf("unexpected error: %v", got.err)
			}
			if got.result.IsError {
				t.Fatalf("unexpected error result: %s", got.result.Content)
			}

			if diff := diffGlobResultSet(got.result.Content, tc.want); diff != "" {
				t.Errorf("unexpected result set for pattern %q: %s (content: %s)", tc.pattern, diff, got.result.Content)
			}
		})
	}
}

// TestGlobTool_LiteralSymlinkPrefixIsFollowed pins the glob containment
// contract around doublestar.WithNoFollow's literal-prefix caveat: a symlink
// named before the pattern's first meta character (here `alias/**`) is still
// FOLLOWED by the walk, but the per-entry containment check keeps every
// resolved-outside entry out of the results — an in-root alias still lists,
// and an alias pointing outside the session roots yields nothing.
func TestGlobTool_LiteralSymlinkPrefixIsFollowed(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "real"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "real", "keep.go"), []byte("package x\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "top.go"), []byte("package x\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink("real", filepath.Join(root, "alias")); err != nil {
		t.Skipf("symlinks unsupported on this filesystem: %v", err)
	}

	// Workspace context so per-entry containment is enforced (glob keeps its
	// fail-open behavior when no session roots are attached).
	ctx := tools.WithWorkspacePath(context.Background(), root)

	tool := NewGlobTool()
	run := func(pattern string) string {
		t.Helper()
		input, _ := json.Marshal(GlobInput{Pattern: pattern, Path: root, Type: "all"})
		res, err := tool.Execute(ctx, input)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if res.IsError {
			t.Fatalf("unexpected error result: %s", res.Content)
		}
		return res.Content
	}

	// `**/*` does not traverse the symlinked directory `alias`.
	if diff := diffGlobResultSet(run("**/*"), []string{"alias", "top.go", "real", "real/keep.go"}); diff != "" {
		t.Errorf("`**/*` must not traverse the symlinked dir: %s", diff)
	}
	// The literal symlink prefix is followed by the walker, but every entry
	// under it resolves inside the workspace, so the alias listing survives
	// containment intact.
	if diff := diffGlobResultSet(run("alias/**"), []string{"alias", "alias/keep.go"}); diff != "" {
		t.Errorf("`alias/**` must still list the in-root alias: %s", diff)
	}
}

// TestGlobTool_OutOfRootSymlinkPrefixYieldsNothing is the regression repro for
// review finding #61: with a workspace attached, `glob({"pattern":"link/*"})`
// where link points OUTSIDE the workspace must return zero out-of-root
// entries (and must not fail the whole walk) — the literal-prefix symlink the
// walker still follows must not leak outside-root names into the results.
func TestGlobTool_OutOfRootSymlinkPrefixYieldsNothing(t *testing.T) {
	ws := t.TempDir()
	outside := t.TempDir()
	if err := os.WriteFile(filepath.Join(outside, "a.conf"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(outside, "b.conf"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(outside, "note.txt"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(ws, "real.go"), []byte("package x\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, filepath.Join(ws, "link")); err != nil {
		t.Skipf("symlinks unsupported on this filesystem: %v", err)
	}

	ctx := tools.WithWorkspacePath(context.Background(), ws)
	tool := NewGlobTool()
	run := func(pattern, typeFilter string) []string {
		t.Helper()
		input, _ := json.Marshal(GlobInput{Pattern: pattern, Path: ws, Type: typeFilter})
		res, err := tool.Execute(ctx, input)
		if err != nil {
			t.Fatalf("unexpected error for pattern %q: %v", pattern, err)
		}
		if res.IsError {
			t.Fatalf("out-of-root link must not error the walk, got: %s", res.Content)
		}
		return globResultLines(res.Content)
	}

	for _, tc := range []struct {
		name    string
		pattern string
	}{
		{"star under link", "link/*"},
		{"recursive conf under link", "link/**/*.conf"},
		{"doublestar all under link", "link/**/*"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := run(tc.pattern, "all")
			if len(got) != 0 {
				t.Errorf("glob(%q) leaked out-of-root entries, got: %v", tc.pattern, got)
			}
		})
	}

	// The in-root sibling is unaffected under the same workspace context.
	// The `link` entry itself is dropped too: its resolved path leaves the
	// workspace, and the containment contract is resolve-then-drop for every
	// entry, symlink or not.
	allInput, _ := json.Marshal(GlobInput{Pattern: "**/*", Path: ws, Type: "all"})
	allRes, err := tool.Execute(ctx, allInput)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if allRes.IsError {
		t.Fatalf("unexpected error result: %s", allRes.Content)
	}
	if diff := diffGlobResultSet(allRes.Content, []string{"real.go"}); diff != "" {
		t.Errorf("in-root entries must survive containment: %s", diff)
	}
}

// diffGlobResultSet compares a glob output against the exact expected set
// (order-insensitive) and returns "" on a match, or a human-readable difference
// otherwise. An empty expected set renders as the glob's "no matching files
// found" placeholder.
func diffGlobResultSet(content string, want []string) string {
	got := globResultLines(content)
	sort.Strings(got)
	sort.Strings(want)
	if slices.Equal(got, want) {
		return ""
	}
	return fmt.Sprintf("got %v, want %v", got, want)
}

// globResultLines splits a glob output into result lines, dropping blank lines,
// the "no matching files found" placeholder, and any trailing warning suffix a
// bound abort appends (the warning follows the matches after a blank line, and a
// match-less abort renders the warning alone).
func globResultLines(content string) []string {
	body := content
	if i := strings.Index(content, "\n\n"); i >= 0 {
		body = content[:i]
	}
	var out []string
	for _, line := range strings.Split(body, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || line == "no matching files found" || strings.HasPrefix(line, "warning:") {
			continue
		}
		out = append(out, line)
	}
	return out
}

// TestGlobTool_ContextCancelMidWalk verifies that a context canceled while the
// walk is in progress interrupts it promptly and reports a clear error instead
// of hanging.
func TestGlobTool_ContextCancelMidWalk(t *testing.T) {
	base := t.TempDir()
	// Several sibling directories guarantee further directory reads after the
	// first callback fires, so the cancellation is observed mid-walk.
	for i := 0; i < 16; i++ {
		dir := filepath.Join(base, fmt.Sprintf("d%02d", i))
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "file.go"), []byte("package x\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	tool := NewGlobTool()
	input, _ := json.Marshal(GlobInput{Pattern: "**/*", Path: base, Type: "all"})

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	ctx = tools.WithIgnoreChecker(ctx, &cancelOnFirstIgnore{cancel: cancel})

	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Fatalf("expected an error result after mid-walk cancellation, got: %s", result.Content)
	}
	if !strings.Contains(result.Content, "glob canceled") {
		t.Errorf("expected a cancellation message, got: %s", result.Content)
	}
}
