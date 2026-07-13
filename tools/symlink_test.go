package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"testing"

	"github.com/v0lka/sp4rk/pathutil"
)

// osAbsPath joins an OS-absolute root (the system temp dir) with the given
// relative components, returning a genuinely absolute path on every platform.
// Several path-extraction tests need OS-absolute fixtures (a leading "/" is
// NOT absolute on Windows) so that resolvePathCandidate returns the path
// unchanged instead of joining it with the workspace. Because these fixtures
// are real OS paths, the tests behave identically on macOS and Windows.
func osAbsPath(parts ...string) string {
	all := make([]string, 0, len(parts)+1)
	all = append(all, os.TempDir())
	all = append(all, parts...)
	return filepath.Join(all...)
}

// ── extractAllPathsFromJSON tests ─────────────────────────────────────────

func TestExtractAllPathsFromJSON_Absolute(t *testing.T) {
	// Use a genuinely OS-absolute path so it is returned unchanged (a leading
	// "/" is not absolute on Windows, where it would otherwise be joined with
	// the workspace and doubled).
	file := osAbsPath("workspace", "file.txt")
	ws := osAbsPath("workspace")
	input, _ := json.Marshal(map[string]string{"file_path": file})
	paths := extractAllPathsFromJSON(input, ws)
	want := filepath.Clean(file)
	if len(paths) != 1 || paths[0] != want {
		t.Fatalf("expected [%s], got %v", want, paths)
	}
}

func TestExtractAllPathsFromJSON_Relative(t *testing.T) {
	input := json.RawMessage(`{"file_path": "config/file.txt"}`)
	paths := extractAllPathsFromJSON(input, "/workspace")
	expected := filepath.Clean("/workspace/config/file.txt")
	if len(paths) != 1 || paths[0] != expected {
		t.Fatalf("expected [%s], got %v", expected, paths)
	}
}

func TestExtractAllPathsFromJSON_RelativeNoWorkspace(t *testing.T) {
	input := json.RawMessage(`{"file_path": "config/file.txt"}`)
	paths := extractAllPathsFromJSON(input, "")
	if len(paths) != 0 {
		t.Fatalf("expected empty when no workspace, got %v", paths)
	}
}

func TestExtractAllPathsFromJSON_Multiple(t *testing.T) {
	input := json.RawMessage(`{"src": "/workspace/a", "dst": "/workspace/b"}`)
	paths := extractAllPathsFromJSON(input, "/workspace")
	if len(paths) != 2 {
		t.Fatalf("expected 2 paths, got %d: %v", len(paths), paths)
	}
}

func TestExtractAllPathsFromJSON_SkipNonPaths(t *testing.T) {
	input := json.RawMessage(`{"name": "myconfig", "timeout": "30s"}`)
	paths := extractAllPathsFromJSON(input, "/workspace")
	if len(paths) != 0 {
		t.Fatalf("expected no paths, got %v", paths)
	}
}

func TestExtractAllPathsFromJSON_SkipURLs(t *testing.T) {
	input := json.RawMessage(`{"url": "https://example.com/path"}`)
	paths := extractAllPathsFromJSON(input, "/workspace")
	if len(paths) != 0 {
		t.Fatalf("expected no paths (URL filtered), got %v", paths)
	}
}

func TestExtractAllPathsFromJSON_SkipFileURL(t *testing.T) {
	input := json.RawMessage(`{"url": "file:///etc/hosts"}`)
	paths := extractAllPathsFromJSON(input, "/workspace")
	if len(paths) != 0 {
		t.Fatalf("expected no paths (file:// URL filtered), got %v", paths)
	}
}

func TestExtractAllPathsFromJSON_Deduplicate(t *testing.T) {
	// OS-absolute fixtures so the paths are returned unchanged on every OS.
	x := osAbsPath("workspace", "x")
	ws := osAbsPath("workspace")
	input, _ := json.Marshal(map[string]string{"a": x, "b": x})
	paths := extractAllPathsFromJSON(input, ws)
	want := filepath.Clean(x)
	if len(paths) != 1 || paths[0] != want {
		t.Fatalf("expected deduplicated [%s], got %v", want, paths)
	}
}

func TestExtractAllPathsFromJSON_NestedJSON(t *testing.T) {
	input := json.RawMessage(`{"files": [{"path": "/workspace/a"}, {"path": "/workspace/b"}]}`)
	paths := extractAllPathsFromJSON(input, "/workspace")
	if len(paths) != 2 {
		t.Fatalf("expected 2 paths in nested JSON, got %d: %v", len(paths), paths)
	}
}

func TestExtractAllPathsFromJSON_InvalidJSON(t *testing.T) {
	input := json.RawMessage(`{bad json`)
	paths := extractAllPathsFromJSON(input, "/workspace")
	if len(paths) != 0 {
		t.Fatalf("expected empty for invalid JSON, got %v", paths)
	}
}

func TestExtractAllPathsFromJSON_DotDotTraversal(t *testing.T) {
	input := json.RawMessage(`{"file_path": "../../etc/passwd"}`)
	paths := extractAllPathsFromJSON(input, "/workspace/project")
	// Should resolve to /workspace/etc/passwd NOT /etc/passwd
	// filepath.Join handles .. traversal
	expected := filepath.Clean("/workspace/project/../../etc/passwd")
	cleaned := filepath.Clean(expected)
	if len(paths) != 1 || paths[0] != cleaned {
		t.Fatalf("expected [%s], got %v", cleaned, paths)
	}
}

// ── extractBashPaths tests ────────────────────────────────────────────────

func TestExtractBashPaths_Simple(t *testing.T) {
	// OS-absolute target outside the working directory, written with forward
	// slashes (bash convention) via ToSlash so Windows backslashes are not
	// treated as shell escapes by the parser.
	target := osAbsPath("etc", "hosts")
	cmd := "cat " + filepath.ToSlash(target)
	paths, suspicious := extractBashPaths(cmd, osAbsPath("wd"), osAbsPath("ws"))
	if suspicious {
		t.Fatal("expected not suspicious")
	}
	want := filepath.Clean(target)
	if len(paths) != 1 || paths[0] != want {
		t.Fatalf("expected [%s], got %v", want, paths)
	}
}

func TestExtractBashPaths_Multiple(t *testing.T) {
	paths, suspicious := extractBashPaths("cp /tmp/src /tmp/dst", "", "/workspace")
	if suspicious {
		t.Fatal("expected not suspicious")
	}
	if len(paths) != 2 {
		t.Fatalf("expected 2 paths, got %d: %v", len(paths), paths)
	}
}

func TestExtractBashPaths_Relative(t *testing.T) {
	paths, suspicious := extractBashPaths("cat data/file.txt", "/workspace", "")
	if suspicious {
		t.Fatal("expected not suspicious")
	}
	expected := filepath.Clean("/workspace/data/file.txt")
	if len(paths) != 1 || paths[0] != expected {
		t.Fatalf("expected [%s], got %v", expected, paths)
	}
}

func TestExtractBashPaths_RelativeFallback(t *testing.T) {
	paths, suspicious := extractBashPaths("cat data/file.txt", "", "/workspace")
	if suspicious {
		t.Fatal("expected not suspicious")
	}
	expected := filepath.Clean("/workspace/data/file.txt")
	if len(paths) != 1 || paths[0] != expected {
		t.Fatalf("expected [%s], got %v", expected, paths)
	}
}

func TestExtractBashPaths_VariableExpansion(t *testing.T) {
	// ${HOME} (braced) delimits the variable from the following path and is
	// detected as unexpandable; the trailing OS-absolute literal is extracted
	// and the result is marked suspicious.
	suffix := osAbsPath("lit", ".config")
	cmd := "cat ${HOME}" + filepath.ToSlash(suffix)
	paths, suspicious := extractBashPaths(cmd, osAbsPath("wd"), osAbsPath("ws"))
	if !suspicious {
		t.Fatal("expected suspicious flag for $var")
	}
	want := filepath.Clean(suffix)
	if len(paths) != 1 || paths[0] != want {
		t.Fatalf("expected [%s] from literal parts, got %v", want, paths)
	}
}

func TestExtractBashPaths_VariableExpansionInPath(t *testing.T) {
	// ${HOME} delimits the variable; the trailing literal path is extracted
	// and the result is marked suspicious.
	suffix := osAbsPath("path", "to", "file")
	cmd := "cat ${HOME}" + filepath.ToSlash(suffix)
	paths, suspicious := extractBashPaths(cmd, osAbsPath("wd"), osAbsPath("ws"))
	if !suspicious {
		t.Fatal("expected suspicious flag for $var")
	}
	want := filepath.Clean(suffix)
	if len(paths) != 1 || paths[0] != want {
		t.Fatalf("expected [%s] from literal parts, got %v", want, paths)
	}
}

func TestExtractBashPaths_CommandSubstitution(t *testing.T) {
	paths, suspicious := extractBashPaths("cat $(echo /tmp)", "", "/workspace")
	if !suspicious {
		t.Fatal("expected suspicious flag for $(...)")
	}
	if len(paths) != 0 {
		t.Fatalf("expected no extractable paths from $(...), got %v", paths)
	}
}

func TestExtractBashPaths_QuotedStrings(t *testing.T) {
	target := osAbsPath("etc", "passwd")
	cmd := `cat "` + filepath.ToSlash(target) + `"`
	paths, suspicious := extractBashPaths(cmd, osAbsPath("wd"), osAbsPath("ws"))
	if suspicious {
		t.Fatal("expected not suspicious")
	}
	want := filepath.Clean(target)
	if len(paths) != 1 || paths[0] != want {
		t.Fatalf("expected [%s], got %v", want, paths)
	}
}

func TestExtractBashPaths_Redirects(t *testing.T) {
	target := osAbsPath("out.txt")
	cmd := "echo hi > " + filepath.ToSlash(target)
	paths, suspicious := extractBashPaths(cmd, osAbsPath("wd"), osAbsPath("ws"))
	if suspicious {
		t.Fatal("expected not suspicious")
	}
	want := filepath.Clean(target)
	found := false
	for _, p := range paths {
		if p == want {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("expected redirect target %s in paths, got %v", want, paths)
	}
}

func TestExtractBashPaths_ChainedCommands(t *testing.T) {
	a := osAbsPath("a")
	b := osAbsPath("b")
	cmd := "cd " + filepath.ToSlash(a) + " && ls " + filepath.ToSlash(b)
	paths, suspicious := extractBashPaths(cmd, osAbsPath("wd"), osAbsPath("ws"))
	if suspicious {
		t.Fatal("expected not suspicious")
	}
	sort.Strings(paths)
	expected := []string{filepath.Clean(a), filepath.Clean(b)}
	sort.Strings(expected)
	if len(paths) != 2 || paths[0] != expected[0] || paths[1] != expected[1] {
		t.Fatalf("expected %v, got %v", expected, paths)
	}
}

func TestExtractBashPaths_QuotedWithSpaces(t *testing.T) {
	src := osAbsPath("my file.txt") // contains a space
	dst := osAbsPath("dst")
	cmd := `cp "` + filepath.ToSlash(src) + `" "` + filepath.ToSlash(dst) + `/"`
	paths, suspicious := extractBashPaths(cmd, osAbsPath("wd"), osAbsPath("ws"))
	if suspicious {
		t.Fatal("expected not suspicious")
	}
	wantSrc := filepath.Clean(src)
	wantDst := filepath.Clean(dst)
	foundSrc := false
	foundDst := false
	for _, p := range paths {
		if p == wantSrc {
			foundSrc = true
		}
		if p == wantDst {
			foundDst = true
		}
	}
	if !foundSrc || !foundDst {
		t.Fatalf("expected %s and %s, got %v", wantSrc, wantDst, paths)
	}
}

func TestExtractBashPaths_WorkingDirectory(t *testing.T) {
	paths, suspicious := extractBashPaths("ls ./file.txt", "/workspace/subdir", "")
	if suspicious {
		t.Fatal("expected not suspicious")
	}
	expected := filepath.Clean("/workspace/subdir/file.txt")
	if len(paths) != 1 || paths[0] != expected {
		t.Fatalf("expected [%s], got %v", expected, paths)
	}
}

func TestExtractBashPaths_InvalidSyntax(t *testing.T) {
	_, suspicious := extractBashPaths("for i in; do echo", "", "/workspace")
	if !suspicious {
		t.Fatal("expected suspicious flag for invalid syntax")
	}
}

func TestExtractBashPaths_Backtick(t *testing.T) {
	paths, suspicious := extractBashPaths("cat `echo /tmp`", "", "/workspace")
	if !suspicious {
		t.Fatal("expected suspicious flag for backtick")
	}
	if len(paths) != 0 {
		t.Fatalf("expected no extractable paths from backtick, got %v", paths)
	}
}

func TestExtractBashPaths_ProcSubst(t *testing.T) {
	paths, suspicious := extractBashPaths("diff <(cat /a) <(cat /b)", "", "/workspace")
	if !suspicious {
		t.Fatal("expected suspicious flag for process substitution")
	}
	// The /a and /b are inside <(...) which we skip
	if len(paths) != 0 {
		t.Fatalf("expected no paths from process substitution, got %v", paths)
	}
}

func TestExtractBashPaths_SingleQuotes(t *testing.T) {
	target := osAbsPath("etc", "hosts")
	cmd := "cat '" + filepath.ToSlash(target) + "'"
	paths, suspicious := extractBashPaths(cmd, osAbsPath("wd"), osAbsPath("ws"))
	if suspicious {
		t.Fatal("expected not suspicious")
	}
	want := filepath.Clean(target)
	if len(paths) != 1 || paths[0] != want {
		t.Fatalf("expected [%s], got %v", want, paths)
	}
}

func TestExtractBashPaths_EscapedSpaces(t *testing.T) {
	// Backslash-escaped spaces are POSIX shell semantics. On Windows '\' is a
	// path separator, so filepath.Clean would mangle the preserved escape and
	// the assertion is not meaningful there.
	if runtime.GOOS == "windows" {
		t.Skip("backslash-escaped spaces are POSIX shell semantics")
	}
	paths, suspicious := extractBashPaths(`cat /tmp/my\ file.txt`, "", "/workspace")
	if suspicious {
		t.Fatal("expected not suspicious")
	}
	// The backslash is preserved in the Lit value by the parser
	found := false
	for _, p := range paths {
		if p == "/tmp/my\\ file.txt" {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("expected /tmp/my\\ file.txt (backslash preserved), got %v", paths)
	}
}

// ── walkSymlinkComponents tests ───────────────────────────────────────────

func TestWalkSymlinkComponents_NoSymlink(t *testing.T) {
	dir := t.TempDir()
	normalPath := filepath.Join(dir, "normal", "file.txt")
	_ = os.MkdirAll(filepath.Dir(normalPath), 0o755)
	_ = os.WriteFile(normalPath, []byte("hello"), 0o644)

	result := walkSymlinkComponents(normalPath, dir)
	if result != nil {
		t.Fatalf("expected nil for non-symlink path, got %+v", result)
	}
}

func TestWalkSymlinkComponents_SimpleSymlink(t *testing.T) {
	dir := t.TempDir()
	targetFile := filepath.Join(dir, "real", "target.txt")
	_ = os.MkdirAll(filepath.Dir(targetFile), 0o755)
	_ = os.WriteFile(targetFile, []byte("data"), 0o644)

	symlinkPath := filepath.Join(dir, "link")
	targetDir := filepath.Join(dir, "real")
	_ = os.Symlink(targetDir, symlinkPath)

	nestedPath := filepath.Join(symlinkPath, "target.txt")

	results := walkSymlinkComponents(nestedPath, dir)
	if len(results) == 0 {
		t.Fatal("expected symlink traversal, got none")
	}
	result := results[0]
	if result.SymlinkAt != symlinkPath {
		t.Fatalf("expected symlink at %s, got %s", symlinkPath, result.SymlinkAt)
	}
	expectedResolved, _ := filepath.EvalSymlinks(targetFile)
	if result.FullResolved != expectedResolved {
		t.Fatalf("expected full resolved %s, got %s", expectedResolved, result.FullResolved)
	}
}

func TestWalkSymlinkComponents_DeepSymlink(t *testing.T) {
	dir := t.TempDir()

	// Create: dir/deep/symlink → dir/outside/
	outsideDir := filepath.Join(dir, "outside")
	_ = os.MkdirAll(outsideDir, 0o755)
	_ = os.WriteFile(filepath.Join(outsideDir, "secret"), []byte("x"), 0o644)

	deepDir := filepath.Join(dir, "deep")
	_ = os.MkdirAll(deepDir, 0o755)
	symlinkAt := filepath.Join(deepDir, "symlink")
	_ = os.Symlink(outsideDir, symlinkAt)

	nestedPath := filepath.Join(symlinkAt, "secret")

	results := walkSymlinkComponents(nestedPath, dir)
	if len(results) == 0 {
		t.Fatal("expected symlink traversal for deep symlink")
	}
	result := results[0]
	if result.SymlinkAt != symlinkAt {
		t.Fatalf("expected symlink at %s, got %s", symlinkAt, result.SymlinkAt)
	}
	expectedResolved, _ := filepath.EvalSymlinks(filepath.Join(outsideDir, "secret"))
	if result.FullResolved != expectedResolved {
		t.Fatalf("expected full resolved outside/secret, got %s", result.FullResolved)
	}
}

func TestWalkSymlinkComponents_NonExistentPath(t *testing.T) {
	// Use a path that doesn"t traverse OS-level symlinks (macOS /tmp -> /private/tmp)
	path := "/does/not/exist/at/all"
	result := walkSymlinkComponents(path, "")
	if result != nil {
		t.Fatalf("expected nil for non-existent path, got %+v", result)
	}
}

func TestWalkSymlinkComponents_NonExistentParent(t *testing.T) {
	dir := t.TempDir()
	nope := filepath.Join(dir, "nope", "file.txt")
	result := walkSymlinkComponents(nope, dir)
	if result != nil {
		t.Fatalf("expected nil for non-existent parent, got %+v", result)
	}
}

func TestWalkSymlinkComponents_Empty(t *testing.T) {
	result := walkSymlinkComponents("", "")
	if result != nil {
		t.Fatalf("expected nil for empty path, got %+v", result)
	}
}

func TestWalkSymlinkComponents_LastComponentSymlink(t *testing.T) {
	dir := t.TempDir()
	target := filepath.Join(dir, "real.file")
	_ = os.WriteFile(target, []byte("data"), 0o644)
	symlinkPath := filepath.Join(dir, "link.file")
	_ = os.Symlink(target, symlinkPath)

	results := walkSymlinkComponents(symlinkPath, dir)
	if len(results) == 0 {
		t.Fatal("expected symlink traversal")
	}
	result := results[0]
	if result.SymlinkAt != symlinkPath {
		t.Fatalf("expected symlink at %s, got %s", symlinkPath, result.SymlinkAt)
	}
	expectedResolved, _ := filepath.EvalSymlinks(target)
	if result.FullResolved != expectedResolved {
		t.Fatalf("expected full resolved %s, got %s", expectedResolved, result.FullResolved)
	}
}

// ── detectSymlinksInToolInput tests ───────────────────────────────────────

func TestDetectSymlinks_BashExecWithSymlink(t *testing.T) {
	dir := t.TempDir()
	realDir := filepath.Join(dir, "real")
	_ = os.MkdirAll(realDir, 0o755)
	symlinkPath := filepath.Join(dir, "link")
	_ = os.Symlink(realDir, symlinkPath)

	// Command targets a file through the symlink. Use forward slashes (bash
	// convention) via ToSlash so Windows backslashes are not treated as shell
	// escapes by the parser.
	command := "cat " + filepath.ToSlash(filepath.Join(symlinkPath, "file.txt"))
	input, _ := json.Marshal(map[string]string{"command": command, "working_directory": dir})

	ctx := WithWorkspacePath(context.Background(), dir)
	inside, outside, suspicious := DetectSymlinksInToolInput(ctx, "bash_exec", input, nil)

	if suspicious {
		t.Fatal("expected not suspicious")
	}
	if len(inside)+len(outside) == 0 {
		t.Fatal("expected symlink traversals found")
	}
}

func TestDetectSymlinks_BashExecClean(t *testing.T) {
	dir := t.TempDir()
	command := "echo hello"
	input, _ := json.Marshal(map[string]string{"command": command})

	ctx := WithWorkspacePath(context.Background(), dir)
	inside, outside, suspicious := DetectSymlinksInToolInput(ctx, "bash_exec", input, nil)

	if suspicious {
		t.Fatal("expected not suspicious")
	}
	if len(inside) != 0 || len(outside) != 0 {
		t.Fatalf("expected no traversals for clean command, got inside=%d outside=%d", len(inside), len(outside))
	}
}

func TestDetectSymlinks_BashExecSuspicious(t *testing.T) {
	input, _ := json.Marshal(map[string]string{"command": "cat $HOME/file"})

	ctx := context.Background()
	inside, outside, suspicious := DetectSymlinksInToolInput(ctx, "bash_exec", input, nil)

	if !suspicious {
		t.Fatal("expected suspicious for $var expansion")
	}
	if len(inside) != 0 || len(outside) != 0 {
		t.Fatalf("expected no traversals, got inside=%d outside=%d", len(inside), len(outside))
	}
}

func TestDetectSymlinks_StructuredWithSymlink(t *testing.T) {
	dir := t.TempDir()
	realDir := filepath.Join(dir, "real")
	_ = os.MkdirAll(realDir, 0o755)
	symlinkPath := filepath.Join(dir, "link")
	_ = os.Symlink(realDir, symlinkPath)

	nestedPath := filepath.Join(symlinkPath, "file.txt")
	input, _ := json.Marshal(map[string]string{"file_path": nestedPath})
	schema := json.RawMessage(`{"type":"object","properties":{"file_path":{"type":"string"}}}`)

	ctx := WithWorkspacePath(context.Background(), dir)
	inside, outside, suspicious := DetectSymlinksInToolInput(ctx, "write_file", input, schema)

	if suspicious {
		t.Fatal("expected not suspicious for structured tool")
	}
	if len(inside)+len(outside) == 0 {
		t.Fatal("expected symlink traversals found for structured tool")
	}
}

func TestDetectSymlinks_StructuredClean(t *testing.T) {
	dir := t.TempDir()
	normalPath := filepath.Join(dir, "normal", "file.txt")
	_ = os.MkdirAll(filepath.Dir(normalPath), 0o755)
	_ = os.WriteFile(normalPath, []byte("x"), 0o644)

	input, _ := json.Marshal(map[string]string{"file_path": normalPath})
	schema := json.RawMessage(`{"type":"object","properties":{"file_path":{"type":"string"}}}`)
	ctx := WithWorkspacePath(context.Background(), dir)
	inside, outside, suspicious := DetectSymlinksInToolInput(ctx, "read_file", input, schema)

	if suspicious {
		t.Fatal("expected not suspicious")
	}
	if len(inside) != 0 || len(outside) != 0 {
		t.Fatalf("expected no traversals, got inside=%d outside=%d", len(inside), len(outside))
	}
}

// ── isPathOutside tests ───────────────────────────────────────────────────

func TestIsPathOutside_InsideWorkspace(t *testing.T) {
	ok, err := pathutil.IsWithinPath("/workspace/project", "/workspace/project/file.txt")
	if err != nil || !ok {
		t.Fatal("expected inside workspace")
	}
}

func TestIsPathOutside_OutsideWorkspace(t *testing.T) {
	ok, _ := pathutil.IsWithinPath("/workspace", "/etc/passwd")
	if ok {
		t.Fatal("expected outside workspace")
	}
}

func TestIsPathOutside_EmptyWorkspace(t *testing.T) {
	ok, err := pathutil.IsWithinPath("", "/tmp/anything")
	if err == nil || ok {
		t.Fatal("expected (false, error) for empty workspace (fail closed)")
	}
}

func TestIsPathOutside_WorkspaceIsFile(t *testing.T) {
	ok, _ := pathutil.IsWithinPath("/workspace/other/dir", "/workspace")
	if ok {
		t.Fatal("expected outside — path is parent of workspace")
	}
}

// ── formatSymlinkReasoning tests ──────────────────────────────────────────

func TestFormatSymlinkReasoning_Outside(t *testing.T) {
	traversals := []SymlinkTraversal{
		{OriginalPath: "/workspace/link", SymlinkAt: "/workspace/link", FullResolved: "/etc/cron.d"},
	}
	msg := FormatSymlinkReasoning(nil, traversals, false)
	if !stringsContains(msg, "OUTSIDE the workspace") {
		t.Fatalf("expected OUTSIDE warning, got: %s", msg)
	}
	if !stringsContains(msg, "/workspace/link") {
		t.Fatalf("expected original path in message, got: %s", msg)
	}
	if !stringsContains(msg, "/etc/cron.d") {
		t.Fatalf("expected resolved path in message, got: %s", msg)
	}
}

func TestFormatSymlinkReasoning_Inside(t *testing.T) {
	traversals := []SymlinkTraversal{
		{OriginalPath: "/workspace/link", SymlinkAt: "/workspace/link", FullResolved: "/workspace/real"},
	}
	msg := FormatSymlinkReasoning(traversals, nil, false)
	if !stringsContains(msg, "within workspace") {
		t.Fatalf("expected within workspace, got: %s", msg)
	}
}

func TestFormatSymlinkReasoning_Suspicious(t *testing.T) {
	msg := FormatSymlinkReasoning(nil, nil, true)
	if !stringsContains(msg, "unresolved shell expansions") {
		t.Fatalf("expected suspicious warning, got: %s", msg)
	}
}

func TestFormatSymlinkReasoning_Both(t *testing.T) {
	inside := []SymlinkTraversal{
		{OriginalPath: "/ws/a", SymlinkAt: "/ws/a", FullResolved: "/ws/b"},
	}
	outside := []SymlinkTraversal{
		{OriginalPath: "/ws/c", SymlinkAt: "/ws/c", FullResolved: "/etc/x"},
	}
	msg := FormatSymlinkReasoning(inside, outside, false)
	if !stringsContains(msg, "OUTSIDE") {
		t.Fatalf("expected OUTSIDE warning, got: %s", msg)
	}
	if !stringsContains(msg, "within workspace") {
		t.Fatalf("expected within workspace, got: %s", msg)
	}
}

func TestFormatSymlinkReasoning_OutsideTruncation(t *testing.T) {
	// More than 10 outside traversals — should truncate.
	outside := make([]SymlinkTraversal, 0, 15)
	for i := 0; i < 15; i++ {
		outside = append(outside, SymlinkTraversal{
			OriginalPath: "/ws/link",
			SymlinkAt:    "/ws/link",
			FullResolved: "/etc/x",
		})
	}
	msg := FormatSymlinkReasoning(nil, outside, false)
	if !stringsContains(msg, "and 5 more symlink") {
		t.Fatalf("expected truncation hint, got: %s", msg)
	}
}

func TestFormatSymlinkReasoning_InsideTruncation(t *testing.T) {
	// More than 10 inside traversals — should truncate.
	inside := make([]SymlinkTraversal, 0, 12)
	for i := 0; i < 12; i++ {
		inside = append(inside, SymlinkTraversal{
			OriginalPath: "/ws/link",
			SymlinkAt:    "/ws/link",
			FullResolved: "/ws/real",
		})
	}
	msg := FormatSymlinkReasoning(inside, nil, false)
	if !stringsContains(msg, "and 2 more symlink") {
		t.Fatalf("expected truncation hint, got: %s", msg)
	}
}

func TestFormatSymlinkReasoning_Empty(t *testing.T) {
	msg := FormatSymlinkReasoning(nil, nil, false)
	if msg != "" {
		t.Fatalf("expected empty string for no traversals, got: %s", msg)
	}
}

// ── looksLikePath tests ──────────────────────────────────────────────────

func TestLooksLikePath_Dot(t *testing.T) {
	if looksLikePath(".") {
		t.Error("'.' should not look like a path")
	}
	if looksLikePath("..") {
		t.Error("'..' should not look like a path")
	}
}

func TestLooksLikePath_WindowsDriveLetter(t *testing.T) {
	if !looksLikePath(`C:\Windows\System32`) {
		t.Error("Windows drive-letter path should be recognized")
	}
	if !looksLikePath(`D:/Projects/code`) {
		t.Error("Windows drive-letter with forward slash should be recognized")
	}
}

func TestLooksLikePath_UNC(t *testing.T) {
	if !looksLikePath(`\\server\share\path`) {
		t.Error("UNC path should be recognized")
	}
}

func TestLooksLikePath_NonPath(t *testing.T) {
	if looksLikePath("hello") {
		t.Error("plain string should not look like a path")
	}
	if looksLikePath("") {
		t.Error("empty string should not look like a path")
	}
}

func TestLooksLikePath_MultilineAndControlChars(t *testing.T) {
	// The Case 2 reproduction: an edit_file old_string snippet (multiline Go
	// code containing "/") must not be treated as a path.
	codeSnippet := "\t\tif err := rows.Scan(&info.ID); err != nil {\n\t\t\t// path/to/something\n\t\t}"
	if looksLikePath(codeSnippet) {
		t.Error("multiline code snippet should not look like a path")
	}

	if looksLikePath("foo/bar\nbaz") {
		t.Error("string with newline should not look like a path")
	}
	if looksLikePath("foo/bar\tbaz") {
		t.Error("string with tab should not look like a path")
	}
	if looksLikePath("foo/bar\rbaz") {
		t.Error("string with carriage return should not look like a path")
	}
	if looksLikePath("foo/bar\x00baz") {
		t.Error("string with NUL should not look like a path")
	}
}

func TestLooksLikePath_TooLong(t *testing.T) {
	long := "/" + strings.Repeat("a", maxPathCandidateLen)
	if looksLikePath(long) {
		t.Error("string exceeding maxPathCandidateLen should not look like a path")
	}
}

func TestLooksLikePath_NormalPathsStillAccepted(t *testing.T) {
	cases := []string{
		"src/a.go",
		"/Users/x/y",
		"./relative/path",
		"dir/subdir/file.txt",
	}
	for _, c := range cases {
		if !looksLikePath(c) {
			t.Errorf("expected looksLikePath=true for normal path %q", c)
		}
	}
	// URLs are still rejected.
	if looksLikePath("https://example.com/a/b") {
		t.Error("URL should not look like a (local) path")
	}
}

// ── looksLikeWindowsDriveLetter tests ────────────────────────────────────

func TestLooksLikeWindowsDriveLetter_Short(t *testing.T) {
	if looksLikeWindowsDriveLetter("C:") {
		t.Error("'C:' is too short for drive letter")
	}
	if looksLikeWindowsDriveLetter("ab") {
		t.Error("'ab' is too short")
	}
}

func TestLooksLikeWindowsDriveLetter_NoColon(t *testing.T) {
	if looksLikeWindowsDriveLetter("CD\\foo") {
		t.Error("no colon should not match")
	}
}

func TestLooksLikeWindowsDriveLetter_Digit(t *testing.T) {
	// Drive letters must be a-z or A-Z, digits are not valid.
	if looksLikeWindowsDriveLetter("1:\\foo") {
		t.Error("digit prefix should not match")
	}
}

// ── resolvePathCandidate tests ───────────────────────────────────────────

func TestResolvePathCandidate_Empty(t *testing.T) {
	if got := resolvePathCandidate("", "/ws"); got != "" {
		t.Errorf("expected empty for empty string, got %q", got)
	}
}

func TestResolvePathCandidate_RelativeNoWorkspace(t *testing.T) {
	if got := resolvePathCandidate("file.txt", ""); got != "" {
		t.Errorf("expected empty for relative path with no workspace, got %q", got)
	}
}

// ── extractBashPathsFromInput tests ──────────────────────────────────────

func TestExtractBashPathsFromInput_EmptyCommand(t *testing.T) {
	input := json.RawMessage(`{"command":""}`)
	paths, suspicious := extractBashPathsFromInput(input, "/ws")
	if len(paths) != 0 {
		t.Fatalf("expected no paths for empty command, got %v", paths)
	}
	if suspicious {
		t.Error("expected not suspicious for empty command")
	}
}

func TestExtractBashPathsFromInput_InvalidJSON(t *testing.T) {
	input := json.RawMessage(`{bad`)
	paths, suspicious := extractBashPathsFromInput(input, "/ws")
	if len(paths) != 0 {
		t.Fatalf("expected no paths for invalid JSON, got %v", paths)
	}
	if suspicious {
		t.Error("expected not suspicious for invalid JSON")
	}
}

func TestExtractBashPathsFromInput_WithWorkingDir(t *testing.T) {
	// Use a path with separator so looksLikePath matches.
	input := json.RawMessage(`{"command":"cat subdir/file.txt","working_directory":"/custom/wd"}`)
	paths, suspicious := extractBashPathsFromInput(input, "/ws")
	if suspicious {
		t.Fatal("expected not suspicious")
	}
	expected := filepath.Clean("/custom/wd/subdir/file.txt")
	if len(paths) != 1 || paths[0] != expected {
		t.Fatalf("expected [%s], got %v", expected, paths)
	}
}

// ── walkSymlinkComponents edge cases ─────────────────────────────────────

func TestWalkSymlinkComponents_RootOnly(t *testing.T) {
	// A path that is just "/" has no components after splitting.
	result := walkSymlinkComponents("/", "")
	if result != nil {
		t.Fatalf("expected nil for root-only path, got %+v", result)
	}
}

// ── checkPathsForSymlinks tests ──────────────────────────────────────────

func TestCheckPathsForSymlinks_NoSymlinks(t *testing.T) {
	dir := t.TempDir()
	normalPath := filepath.Join(dir, "normal", "file.txt")
	_ = os.MkdirAll(filepath.Dir(normalPath), 0o755)
	_ = os.WriteFile(normalPath, []byte("hello"), 0o644)

	inside, outside := checkPathsForSymlinks([]string{normalPath}, []string{dir})
	if len(inside) != 0 || len(outside) != 0 {
		t.Fatalf("expected no traversals, got inside=%d outside=%d", len(inside), len(outside))
	}
}

func TestCheckPathsForSymlinks_Empty(t *testing.T) {
	inside, outside := checkPathsForSymlinks(nil, []string{"/ws"})
	if len(inside) != 0 || len(outside) != 0 {
		t.Fatalf("expected no traversals for empty input, got inside=%d outside=%d", len(inside), len(outside))
	}
}

// ── helpers ───────────────────────────────────────────────────────────────

func stringsContains(s, sub string) bool {
	return sub == "" || len(s) >= len(sub) && containsSub(s, sub)
}

func containsSub(s, sub string) bool {
	for i := 0; i <= len(s)-len(sub); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}

// ── invalid-path error handling (ENAMETOOLONG etc.) ──────────────────────

// TestIsInvalidPathError verifies the helper classifies only invalid-path
// errors (ENAMETOOLONG, ENOTDIR, EINVAL) as invalid, leaving ENOENT and
// permission errors to the escalation path.
func TestIsInvalidPathError(t *testing.T) {
	t.Run("nil", func(t *testing.T) {
		if isInvalidPathError(nil) {
			t.Error("expected false for nil")
		}
	})

	t.Run("ENAMETOOLONG from real Lstat", func(t *testing.T) {
		// A single component longer than NAME_MAX (255) yields ENAMETOOLONG.
		ws := t.TempDir()
		long := filepath.Join(ws, strings.Repeat("a", 300))
		_, err := os.Lstat(long)
		if err == nil {
			t.Skip("could not produce ENAMETOOLONG on this platform")
		}
		if !isInvalidPathError(err) {
			t.Errorf("expected isInvalidPathError=true for %v", err)
		}
	})

	t.Run("ENOENT is not invalid", func(t *testing.T) {
		_, err := os.Lstat(filepath.Join(t.TempDir(), "nope"))
		if err == nil {
			t.Fatal("expected an error")
		}
		if isInvalidPathError(err) {
			t.Errorf("expected isInvalidPathError=false for ENOENT %v", err)
		}
	})
}

// TestCheckPathsForSymlinks_LongComponentNoEscalation reproduces Case 2: a
// string mistaken for a path (a code blob) joined onto the workspace produces
// a single component exceeding NAME_MAX. os.Lstat returns ENAMETOOLONG, which
// must NOT be escalated as an "unresolvable symlink outside workspace".
func TestCheckPathsForSymlinks_LongComponentNoEscalation(t *testing.T) {
	ws := t.TempDir()
	// A 300-byte component — longer than NAME_MAX (255). This mimics a code
	// snippet (e.g. an edit_file old_string) glued onto the workspace root.
	candidate := filepath.Join(ws, strings.Repeat("a", 300))

	inside, outside := checkPathsForSymlinks([]string{candidate}, []string{ws})
	if len(inside) != 0 || len(outside) != 0 {
		t.Fatalf("expected no traversals for over-long component, got inside=%v outside=%v", inside, outside)
	}
}

// TestCheckPathsForSymlinks_AllowedRootClassifiedInside proves that a symlink
// traversal whose resolved target lies inside an additional allowed root is
// classified as 'inside' (treated identically to the workspace and temp dir).
// Without the allowed root, the same traversal would be 'outside'.
func TestCheckPathsForSymlinks_AllowedRootClassifiedInside(t *testing.T) {
	ws := t.TempDir()
	auxRoot := t.TempDir()

	// Target directory lives inside the allowed (auxiliary) root.
	targetDir := filepath.Join(auxRoot, "real")
	if err := os.MkdirAll(targetDir, 0o755); err != nil {
		t.Fatal(err)
	}
	// Symlink lives inside the workspace but points into the allowed root.
	link := filepath.Join(ws, "link")
	if err := os.Symlink(targetDir, link); err != nil {
		t.Fatal(err)
	}
	throughLink := filepath.Join(link, "file.txt")

	// With only the workspace root, the resolved target escapes → outside.
	inside, outside := checkPathsForSymlinks([]string{throughLink}, []string{ws})
	if len(inside) != 0 || len(outside) == 0 {
		t.Fatalf("expected traversal outside with workspace-only roots, got inside=%d outside=%d",
			len(inside), len(outside))
	}

	// With the allowed root added, the resolved target is contained → inside.
	inside, outside = checkPathsForSymlinks([]string{throughLink}, []string{ws, auxRoot})
	if len(inside) == 0 || len(outside) != 0 {
		t.Fatalf("expected traversal inside with allowed root, got inside=%d outside=%d",
			len(inside), len(outside))
	}
}

// TestDetectSymlinks_AllowedRootClassifiedInside proves the end-to-end
// DetectSymlinksInToolInput path also honors allowed roots from the context.
func TestDetectSymlinks_AllowedRootClassifiedInside(t *testing.T) {
	ws := t.TempDir()
	auxRoot := t.TempDir()

	targetDir := filepath.Join(auxRoot, "real")
	if err := os.MkdirAll(targetDir, 0o755); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(ws, "link")
	if err := os.Symlink(targetDir, link); err != nil {
		t.Fatal(err)
	}

	input, _ := json.Marshal(map[string]string{"path": filepath.Join(link, "file.txt")})
	schema := json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"}}}`)

	// With the allowed root in context, the traversal is 'inside'.
	ctx := WithWorkspacePath(context.Background(), ws)
	ctx = WithAllowedRoots(ctx, []string{auxRoot})
	inside, outside, suspicious := DetectSymlinksInToolInput(ctx, "write_file", input, schema)
	if suspicious {
		t.Fatal("expected not suspicious")
	}
	if len(inside) == 0 || len(outside) != 0 {
		t.Fatalf("expected traversal inside allowed root, got inside=%d outside=%d",
			len(inside), len(outside))
	}
}

// TestIsInvalidPathError_PermissionNotInvalid is a positive control: a real
// permission error (EACCES) must NOT be classified as an invalid-path error,
// so it is never short-circuited by the new early-break and still reaches the
// unchanged escalation branch (fail-closed for unreadable symlinks).
//
// Note: a full end-to-end "permission-denied component escalates" test is not
// portable here because t.TempDir() lives under macOS /var → /private/var,
// and an unreadable subdir makes pathutil.IsWithinPath unable to resolve the
// candidate's prefix, returning within=false regardless of this change. The
// escalation branch itself is byte-for-byte unchanged; this assertion proves
// the new invalid-path short-circuit cannot swallow permission errors.
func TestIsInvalidPathError_PermissionNotInvalid(t *testing.T) {
	if os.Geteuid() == 0 {
		t.Skip("permission test is meaningless when running as root")
	}
	ws := t.TempDir()
	secret := filepath.Join(ws, "secret")
	if err := os.Mkdir(secret, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(secret, 0o000); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chmod(secret, 0o700) })

	_, err := os.Lstat(filepath.Join(secret, "hidden"))
	if err == nil {
		t.Skip("could not produce a permission error on this platform")
	}
	if isInvalidPathError(err) {
		t.Errorf("expected isInvalidPathError=false for permission error %v", err)
	}
}

// ── field-aware path extraction (schema-driven) ──────────────────────────

func TestPathFieldNamesFromSchema(t *testing.T) {
	globSchema := json.RawMessage(`{"type":"object","properties":{"pattern":{"type":"string"},"path":{"type":"string"},"type":{"type":"string","enum":["files","dirs","all"]}}}`)
	editSchema := json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"},"old_string":{"type":"string"},"new_string":{"type":"string"}},"required":["path","old_string","new_string"]}`)
	writeSchema := json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}}}`)
	mcpSchema := json.RawMessage(`{"type":"object","properties":{"file_path":{"type":"string"},"query":{"type":"string"}}}`)
	noneSchema := json.RawMessage(`{"type":"object","properties":{"command":{"type":"string"},"timeout":{"type":"integer"}}}`)

	t.Run("glob: only path, not pattern/type", func(t *testing.T) {
		f := pathFieldNamesFromSchema(globSchema)
		if !f["path"] {
			t.Error("expected 'path' recognized")
		}
		if f["pattern"] || f["type"] {
			t.Error("pattern/type must not be recognized as path fields")
		}
	})

	t.Run("edit_file: path only, not old_string/new_string", func(t *testing.T) {
		f := pathFieldNamesFromSchema(editSchema)
		if !f["path"] {
			t.Error("expected 'path' recognized")
		}
		if f["old_string"] || f["new_string"] {
			t.Error("content fields must not be recognized as path fields")
		}
	})

	t.Run("write_file: path only, not content", func(t *testing.T) {
		f := pathFieldNamesFromSchema(writeSchema)
		if !f["path"] {
			t.Error("expected 'path' recognized")
		}
		if f["content"] {
			t.Error("content must not be recognized as a path field")
		}
	})

	t.Run("MCP-style file_path suffix recognized", func(t *testing.T) {
		f := pathFieldNamesFromSchema(mcpSchema)
		if !f["file_path"] {
			t.Error("expected 'file_path' recognized via _path suffix")
		}
		if f["query"] {
			t.Error("'query' must not be recognized as a path field")
		}
	})

	t.Run("no path field -> empty", func(t *testing.T) {
		f := pathFieldNamesFromSchema(noneSchema)
		if len(f) != 0 {
			t.Errorf("expected no path fields, got %v", f)
		}
	})

	t.Run("empty schema -> empty", func(t *testing.T) {
		if len(pathFieldNamesFromSchema(nil)) != 0 {
			t.Error("expected empty for nil schema")
		}
	})
}

// TestDetectSymlinks_FieldAwareExcludesContent verifies that a symlink-escaping
// path placed in a CONTENT field (edit_file old_string) is NOT scanned, while
// the same path in the `path` field IS detected. This is the structural fix
// for Case 2.
func TestDetectSymlinks_FieldAwareExcludesContent(t *testing.T) {
	ws := t.TempDir()
	outside := t.TempDir() // a directory outside the workspace

	// A symlink inside the workspace that escapes to `outside`.
	linkInWS := filepath.Join(ws, "escape")
	if err := os.Symlink(outside, linkInWS); err != nil {
		t.Fatal(err)
	}
	pathThroughLink := filepath.Join(linkInWS, "target.txt")

	editSchema := json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"},"old_string":{"type":"string"},"new_string":{"type":"string"}}}`)
	ctx := WithWorkspacePath(context.Background(), ws)

	t.Run("symlink-escape via path field IS detected", func(t *testing.T) {
		input, _ := json.Marshal(map[string]string{
			"path":       pathThroughLink,
			"old_string": "x",
			"new_string": "y",
		})
		_, outsideTrav, _ := DetectSymlinksInToolInput(ctx, "edit_file", input, editSchema)
		if len(outsideTrav) == 0 {
			t.Fatal("expected symlink escape detected via 'path' field")
		}
	})

	t.Run("symlink-escape in old_string is NOT scanned", func(t *testing.T) {
		normalFile := filepath.Join(ws, "normal.txt")
		_ = os.WriteFile(normalFile, []byte("x"), 0o644)
		input, _ := json.Marshal(map[string]string{
			"path":       normalFile,
			"old_string": "// see " + pathThroughLink + " for details",
			"new_string": "y",
		})
		insideTrav, outsideTrav, _ := DetectSymlinksInToolInput(ctx, "edit_file", input, editSchema)
		if len(insideTrav)+len(outsideTrav) != 0 {
			t.Fatalf("expected content field (old_string) NOT scanned, got inside=%v outside=%v", insideTrav, outsideTrav)
		}
	})
}

// TestDetectSymlinks_NoPathFieldFallsBack verifies that a tool whose schema has
// no recognizable path field falls back to scanning all strings (preserving
// detection for unconventional tools / no regression).
func TestDetectSymlinks_NoPathFieldFallsBack(t *testing.T) {
	ws := t.TempDir()
	realDir := filepath.Join(ws, "real")
	_ = os.MkdirAll(realDir, 0o755)
	link := filepath.Join(ws, "link")
	_ = os.Symlink(realDir, link)

	// Schema with only a non-path string field "payload".
	schema := json.RawMessage(`{"type":"object","properties":{"payload":{"type":"string"}}}`)
	input, _ := json.Marshal(map[string]string{"payload": filepath.Join(link, "f.txt")})
	ctx := WithWorkspacePath(context.Background(), ws)

	insideTrav, _, _ := DetectSymlinksInToolInput(ctx, "some_tool", input, schema)
	if len(insideTrav) == 0 {
		t.Fatal("expected fallback scan to detect the symlink inside the workspace")
	}
}

// TestDetectSymlinks_MixedFieldsLogsOmission verifies the path-field allowlist
// log/skip behavior: when a schema declares a recognized path field alongside
// unrecognized string fields, scanning is narrowed to the recognized field and
// a warning is logged naming the fields that are NOT scanned — so the omission
// is observable rather than silent.
func TestDetectSymlinks_MixedFieldsLogsOmission(t *testing.T) {
	ws := t.TempDir()
	outside := t.TempDir() // a directory outside the workspace

	// A symlink inside the workspace that escapes to `outside`.
	linkInWS := filepath.Join(ws, "escape")
	if err := os.Symlink(outside, linkInWS); err != nil {
		t.Fatal(err)
	}

	// Capture slog output — DetectSymlinksInToolInput logs via slog.Default.
	var buf bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelWarn})))
	t.Cleanup(func() { slog.SetDefault(prev) })

	// Schema with a recognized `path` field AND an unrecognized `target` field.
	schema := json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"},"target":{"type":"string"}}}`)

	// A path through the symlink in the RECOGNIZED field — must be detected.
	detected := filepath.Join(linkInWS, "a.txt")
	// A path through the symlink in the UNRECOGNIZED field — must NOT be scanned.
	dropped := filepath.Join(linkInWS, "b.txt")

	input, _ := json.Marshal(map[string]string{
		"path":   detected,
		"target": dropped,
	})
	ctx := WithWorkspacePath(context.Background(), ws)

	_, outsideTrav, _ := DetectSymlinksInToolInput(ctx, "some_tool", input, schema)
	// Only the recognized field is scanned, so exactly one traversal (the path
	// through `linkInWS/a.txt`); the `target` value is dropped, not scanned.
	if len(outsideTrav) != 1 {
		t.Fatalf("DetectSymlinksInToolInput outside = %d traversals, want 1 (recognized field only): %v",
			len(outsideTrav), outsideTrav)
	}
	if outsideTrav[0].OriginalPath != detected {
		t.Errorf("outsideTrav[0].OriginalPath = %q, want %q (the recognized 'path' field)",
			outsideTrav[0].OriginalPath, detected)
	}

	// The omission must be logged, not silent.
	logged := buf.String()
	if !strings.Contains(logged, "some_tool") {
		t.Errorf("expected warning to reference the tool name 'some_tool'; got:\n%s", logged)
	}
	if !strings.Contains(logged, "target") {
		t.Errorf("expected warning to name the unscanned field 'target'; got:\n%s", logged)
	}
	if !strings.Contains(logged, "path") {
		t.Errorf("expected warning to list the scanned path field 'path'; got:\n%s", logged)
	}
}

// TestCheckPathsForSymlinks_SymlinkBeforeENOTDIR guards the component-walk
// ordering invariant: a symlink component is detected (via Lstat+ModeSymlink)
// at its own prefix BEFORE an ENOTDIR can occur on a deeper component (a
// symlink pointing to a regular file, followed by a subpath). The recorded
// symlink traversal must NOT be erased by the trailing ENOTDIR, which
// isInvalidPathError stops on rather than escalating. A future refactor that
// reorders the walk would break this and fail the test.
func TestCheckPathsForSymlinks_SymlinkBeforeENOTDIR(t *testing.T) {
	ws := t.TempDir()

	// A regular file inside the workspace.
	regularFile := filepath.Join(ws, "regular.txt")
	if err := os.WriteFile(regularFile, []byte("data"), 0o644); err != nil {
		t.Fatal(err)
	}

	// A symlink inside the workspace pointing to the regular file.
	link := filepath.Join(ws, "link")
	if err := os.Symlink(regularFile, link); err != nil {
		t.Fatal(err)
	}

	// A path that traverses the symlink then continues into a subpath of a
	// regular file — os.Lstat on this final component yields ENOTDIR.
	throughLink := filepath.Join(link, "subpath")

	inside, outside := checkPathsForSymlinks([]string{throughLink}, []string{ws})
	got := append([]SymlinkTraversal(nil), inside...)
	got = append(got, outside...)
	if len(got) != 1 {
		t.Fatalf("checkPathsForSymlinks(%q) = %d traversal(s), want exactly 1 (the symlink at %s reported before the trailing ENOTDIR): %+v",
			throughLink, len(got), link, got)
	}
	if got[0].SymlinkAt != link {
		t.Errorf("traversal SymlinkAt = %q, want %q", got[0].SymlinkAt, link)
	}
	// The trailing ENOTDIR (on <link>/subpath) must NOT escalate: the symlink
	// itself was resolved by the time the walk reached the bad component, so
	// the recorded traversal is resolved, not Unresolvable.
	if got[0].Unresolvable {
		t.Errorf("traversal is Unresolvable=true; the trailing ENOTDIR must not erase or escalate the already-resolved symlink")
	}
}
