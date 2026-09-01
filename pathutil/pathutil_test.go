package pathutil

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestIsWithinPath_Inside(t *testing.T) {
	dir := t.TempDir()
	parent := filepath.Join(dir, "ws")
	child := filepath.Join(parent, "sub", "file.txt")
	_ = os.MkdirAll(filepath.Dir(child), 0o755)

	ok, err := IsWithinPath(parent, child)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !ok {
		t.Error("child should be within parent")
	}
}

func TestIsWithinPath_SamePath(t *testing.T) {
	dir := t.TempDir()
	ok, err := IsWithinPath(dir, dir)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !ok {
		t.Error("same path should be considered within")
	}
}

func TestIsWithinPath_Outside(t *testing.T) {
	parent := t.TempDir()
	child := t.TempDir() // different directory

	ok, err := IsWithinPath(parent, child)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ok {
		t.Error("sibling directory should not be within parent")
	}
}

func TestIsWithinPath_Parent(t *testing.T) {
	dir := t.TempDir()
	child := filepath.Join(dir, "sub")
	_ = os.MkdirAll(child, 0o755)

	// child is within parent → true; parent is within child → false
	ok, err := IsWithinPath(child, dir)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ok {
		t.Error("parent should not be within child")
	}
}

func TestIsWithinPath_SymlinkWithin(t *testing.T) {
	dir := t.TempDir()
	realDir := filepath.Join(dir, "real")
	_ = os.MkdirAll(realDir, 0o755)
	link := filepath.Join(dir, "link")
	_ = os.Symlink(realDir, link)
	child := filepath.Join(link, "file.txt")

	ok, err := IsWithinPath(dir, child)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !ok {
		t.Error("symlink pointing inside workspace should be within")
	}
}

func TestIsWithinPath_SymlinkEscape(t *testing.T) {
	dir := t.TempDir()
	outside := filepath.Join(dir, "outside", "real")
	_ = os.MkdirAll(outside, 0o755)
	link := filepath.Join(dir, "ws", "escape")
	_ = os.MkdirAll(filepath.Dir(link), 0o755)
	_ = os.Symlink(outside, link)

	ws := filepath.Join(dir, "ws")
	ok, err := IsWithinPath(ws, link)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ok {
		t.Error("symlink pointing outside workspace should not be within")
	}
}

func TestIsWithinPath_NonExistent(t *testing.T) {
	dir := t.TempDir()
	parent := filepath.Join(dir, "ws")
	child := filepath.Join(parent, "sub", "nonexistent.txt")
	// Neither ws/ nor sub/ exists.

	ok, err := IsWithinPath(parent, child)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !ok {
		t.Error("non-existent path under non-existent parent should be considered within")
	}
}

func TestIsWithinPath_EmptyParent(t *testing.T) {
	ok, err := IsWithinPath("", "/some/path")
	if err == nil {
		t.Error("empty parent should return an error (fail closed)")
	}
	if ok {
		t.Error("empty parent should return false")
	}
}

func TestIsWithinPath_DifferentVolumes(t *testing.T) {
	// Containment is impossible when parent and child live on different
	// volumes (e.g. different Windows drives). filepath.VolumeName is empty on
	// Unix, so the mismatch only surfaces there with an explicit volume prefix
	// — which Unix paths don't carry. Construct a genuine cross-volume pair on
	// Windows and verify the contract; on Unix the inputs share a volume and
	// the mismatch path cannot be exercised.
	parent, child := "/vol/a", "/vol/b"
	if runtime.GOOS == "windows" {
		parent = `C:\vol\a`
		child = `D:\vol\b`
	}
	ok, err := IsWithinPath(parent, child)
	if err == nil {
		// Same volume (e.g. Unix): containment is a plain prefix check with no
		// error — the volume-mismatch contract is not exercisable here, but
		// the path is still not contained.
		if ok {
			t.Error("ok should be false when child is not a descendant of parent (same volume)")
		}
		return
	}
	if ok {
		t.Error("ok should be false when parent and child are on different volumes")
	}
}

// TestIsWithinPath_RootParent guards the regression where a filesystem-root
// parent stopped being recognized as containing its descendants: terminating
// the root with another separator produced "//" (or "C:\\") on Windows, which
// no real child starts with.
func TestIsWithinPath_RootParent(t *testing.T) {
	root := string(filepath.Separator)
	descendant := filepath.Join(root, "usr", "bin")

	ok, err := IsWithinPath(root, descendant)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !ok {
		t.Errorf("filesystem root %q must contain its descendant %q", root, descendant)
	}

	// The root is also within itself.
	ok, err = IsWithinPath(root, root)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !ok {
		t.Error("filesystem root should be within itself")
	}
}

// TestIsWithinPath_CaseSensitiveDiffersOnNonExistent contrasts the two
// variants on the SAME path pair. Case sensitivity only matters for paths
// that do not exist on disk: once a prefix exists, EvalSymlinks (inside
// ResolveExistingPrefix) canonicalizes its casing, so even the case-sensitive
// primitive matches by location. For non-existent parents the casing is
// preserved verbatim, which is exactly where IsWithinPath must reject a
// differing-case child while IsWithinPathFold accepts it.
func TestIsWithinPath_CaseSensitiveDiffersOnNonExistent(t *testing.T) {
	dir := t.TempDir()
	// Neither parent nor child exists, so casing cannot be canonicalized.
	parent := filepath.Join(dir, "Workspace")
	child := filepath.Join(dir, "workspace", "file.txt")

	ok, err := IsWithinPath(parent, child)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ok {
		t.Error("case-sensitive IsWithinPath must reject differing-case non-existent path")
	}

	// The fold variant must accept the exact same pair.
	okFold, errFold := IsWithinPathFold(parent, child)
	if errFold != nil {
		t.Fatalf("unexpected fold error: %v", errFold)
	}
	if !okFold {
		t.Error("fold variant must accept the differing-case path")
	}
}

func TestIsWithinPathFold_InsideCaseInsensitive(t *testing.T) {
	dir := t.TempDir()
	parent := filepath.Join(dir, "Project")
	_ = os.MkdirAll(parent, 0o755)
	// Child uses a different casing for the parent component but stays inside.
	child := filepath.Join(dir, "project", "sub", "file.txt")

	ok, err := IsWithinPathFold(parent, child)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !ok {
		t.Error("case-insensitive variant should recognize differing-case child as within parent")
	}
}

func TestIsWithinPathFold_SamePathCaseInsensitive(t *testing.T) {
	dir := t.TempDir()
	// Two casings of the same on-disk location.
	parent := dir
	child := flipCaseTail(dir)

	ok, err := IsWithinPathFold(parent, child)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !ok {
		t.Error("same path with differing case should be within itself (fold)")
	}
}

func TestIsWithinPathFold_NonExistentCaseInsensitive(t *testing.T) {
	// The decisive case for tool-argument locality: neither parent nor child
	// exists, so EvalSymlinks cannot carry canonical casing. Fold must still
	// recognize the differing-case child as within the parent.
	dir := t.TempDir()
	parent := filepath.Join(dir, "Workspace")
	child := filepath.Join(dir, "workspace", "new", "file.txt")

	ok, err := IsWithinPathFold(parent, child)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !ok {
		t.Error("non-existent differing-case path should be within parent (fold)")
	}
}

func TestIsWithinPathFold_OutsideStillRejected(t *testing.T) {
	dir := t.TempDir()
	parent := filepath.Join(dir, "Workspace")
	_ = os.MkdirAll(parent, 0o755)
	// Genuinely different path, not just a casing variant.
	child := filepath.Join(dir, "Other", "file.txt")
	_ = os.MkdirAll(filepath.Dir(child), 0o755)

	ok, err := IsWithinPathFold(parent, child)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ok {
		t.Error("genuinely different path must still be rejected by fold variant")
	}
}

func TestIsWithinPathFold_ParentEscapeStillRejected(t *testing.T) {
	dir := t.TempDir()
	// Parent is a deeper path; child is its ancestor with a different case —
	// fold must NOT let an ancestor sneak in as "within" a descendant.
	parent := filepath.Join(dir, "Workspace", "Sub")
	_ = os.MkdirAll(parent, 0o755)
	ancestor := filepath.Join(dir, "workspace") // "workspace" == "Workspace" by fold, but is an ANCESTOR

	ok, err := IsWithinPathFold(parent, ancestor)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ok {
		t.Error("ancestor must not be considered within a descendant, even when case matches by fold")
	}
}

func TestIsWithinPathFold_DotDotEscapeStillRejected(t *testing.T) {
	dir := t.TempDir()
	parent := filepath.Join(dir, "Workspace")
	_ = os.MkdirAll(parent, 0o755)
	// A ".." escape with matching case must still be rejected.
	child := filepath.Join(dir, "workspace", "..", "secret.txt")

	ok, err := IsWithinPathFold(parent, child)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ok {
		t.Error("'..' escape must still be rejected by fold variant")
	}
}

func TestIsWithinPathFold_EmptyParent(t *testing.T) {
	ok, err := IsWithinPathFold("", "/some/path")
	if err == nil {
		t.Error("empty parent should return an error (fail closed)")
	}
	if ok {
		t.Error("empty parent should return false")
	}
}

// flipCaseTail returns a same-location-but-different-case variant of an
// absolute path by re-casing its last component. It mirrors the production
// fold (strings.ToLower) and falls back to ToUpper when the name is already
// all-lower-case, so the result always differs as long as the component has a
// letter. Used to build a case-variant of a real temp dir without depending
// on filesystem case-sensitivity at test time.
func flipCaseTail(p string) string {
	dir, base := filepath.Split(p)
	if base == "" {
		return p
	}
	flipped := strings.ToLower(base)
	if flipped == base {
		flipped = strings.ToUpper(base)
	}
	if flipped == base {
		return p // no cased letters to flip
	}
	return dir + flipped
}

func TestSplitPathComponents_Absolute(t *testing.T) {
	// Build the path with the OS separator so it is genuinely absolute on
	// every platform (Windows needs a drive letter, handled by VolumeName).
	abs := filepath.Join(string(filepath.Separator), "home", "user", "file.txt")
	result := SplitPathComponents(abs)
	if len(result) != 3 || result[0] != "home" || result[1] != "user" || result[2] != "file.txt" {
		t.Errorf("got %v, want [home user file.txt]", result)
	}
}

// TestDetectCaseInsensitive returns the filesystem's actual case-sensitivity
// for a real writable directory. The assertion is platform-dependent: a temp
// directory under /tmp on Linux reports false, while macOS APFS and Windows
// report true. We assert against runtime.GOOS so the test is portable and
// documents the expected per-platform outcome.
func TestDetectCaseInsensitive(t *testing.T) {
	dir := t.TempDir()
	got := DetectCaseInsensitive(dir)
	want := runtime.GOOS == "darwin" || runtime.GOOS == "windows"
	if got != want {
		t.Errorf("DetectCaseInsensitive(%q) = %v, want %v (GOOS=%s)", dir, got, want, runtime.GOOS)
	}
	// The probe must not leave artefacts behind.
	if entries, _ := os.ReadDir(dir); len(entries) != 0 {
		names := make([]string, 0, len(entries))
		for _, e := range entries {
			names = append(names, e.Name())
		}
		t.Errorf("DetectCaseInsensitive left probe artefacts: %v", names)
	}
}

// TestDetectCaseInsensitive_NonExistentClimbsToAncestor verifies that a
// non-existent target directory falls back to the nearest existing ancestor,
// so detection returns the filesystem's true case-sensitivity rather than the
// fail-safe false.
func TestDetectCaseInsensitive_NonExistentClimbsToAncestor(t *testing.T) {
	dir := t.TempDir()
	target := filepath.Join(dir, "DoesNotExist", "subdir", "leaf")
	got := DetectCaseInsensitive(target)
	want := runtime.GOOS == "darwin" || runtime.GOOS == "windows"
	if got != want {
		t.Errorf("DetectCaseInsensitive(%q) = %v, want %v (GOOS=%s)", target, got, want, runtime.GOOS)
	}
}

// TestDetectCaseInsensitive_NoExistingAncestorFailsClosed documents the
// fail-safe contract: the probe always returns a deterministic bool without
// panicking. We cannot reliably construct a path with no existing ancestor on
// a real system (the filesystem root always exists), so we assert only that
// the call is panic-free and yields a bool.
func TestDetectCaseInsensitive_NoExistingAncestorFailsClosed(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("DetectCaseInsensitive panicked: %v", r)
		}
	}()
	_ = DetectCaseInsensitive("/nonexistent-root-probe/a/b/c")
}

// TestDetectCaseInsensitive_MemoizedPerDir verifies the process-lifetime
// memoization: a second call for the same directory must reuse the cached
// result instead of re-probing the filesystem (each probe creates and deletes
// a temporary CaseSense-*.probe file, so hosts that call this per request
// would otherwise churn the directory). The cache is white-box seeded with the
// OPPOSITE of the freshly probed value — if the function consulted the
// filesystem again it would return the real value and fail the assertion.
// A second, distinct directory proves one directory's entry does not leak
// into another's answer.
func TestDetectCaseInsensitive_MemoizedPerDir(t *testing.T) {
	dir := t.TempDir()
	probed := DetectCaseInsensitive(dir) // first call probes for real

	key, ok := existingAncestorDir(filepath.Clean(dir))
	if !ok {
		t.Fatalf("existingAncestorDir(%q) found no existing ancestor", dir)
	}
	seeded := !probed
	caseInsensitiveMu.Lock()
	caseInsensitiveKnown[key] = seeded
	caseInsensitiveMu.Unlock()

	if got := DetectCaseInsensitive(dir); got != seeded {
		t.Errorf("DetectCaseInsensitive(%q) = %v after seeding cache with %v; the second call re-probed instead of reusing the memoized result", dir, got, seeded)
	}

	other := t.TempDir()
	if got := DetectCaseInsensitive(other); got != probed {
		t.Errorf("DetectCaseInsensitive(%q) = %v, want independently probed %v (per-directory cache isolation)", other, got, probed)
	}
}

func TestSplitPathComponents_Root(t *testing.T) {
	result := SplitPathComponents(string(filepath.Separator))
	if len(result) != 0 {
		t.Errorf("root path should yield empty slice, got %v", result)
	}
}

func TestSplitPathComponents_Empty(t *testing.T) {
	result := SplitPathComponents("")
	if len(result) != 0 {
		t.Errorf("empty path should yield empty slice, got %v", result)
	}
}

func TestResolveExistingPrefix_AllExist(t *testing.T) {
	dir := t.TempDir()
	filePath := filepath.Join(dir, "file.txt")
	_ = os.WriteFile(filePath, []byte("content"), 0o644)

	result := ResolveExistingPrefix(filePath)
	// On macOS, /var → /private/var symlinks can cause resolution;
	// compare resolved-to-resolved.
	expected, _ := filepath.EvalSymlinks(filePath)
	if result != expected {
		t.Errorf("got %q, want %q", result, expected)
	}
}

func TestResolveExistingPrefix_PartialExist(t *testing.T) {
	dir := t.TempDir()
	nonexistent := filepath.Join(dir, "sub", "nonexistent.txt")

	result := ResolveExistingPrefix(nonexistent)
	// The dir prefix may be symlink-resolved (macOS /var → /private/var).
	// Use ResolveExistingPrefix on the expected result for fair comparison.
	expected, _ := filepath.EvalSymlinks(dir)
	expected = filepath.Join(expected, "sub", "nonexistent.txt")
	if result != expected {
		t.Errorf("got %q, want %q", result, expected)
	}
}

func TestResolveExistingPrefix_NoneExist(t *testing.T) {
	input := "/nonexistent/root/path/file.txt"
	result := ResolveExistingPrefix(input)
	// A completely non-existent path is returned unchanged modulo separator
	// normalization (Windows uses '\'). Compare on a normalized form so the
	// test is portable.
	if filepath.ToSlash(result) != input {
		t.Errorf("completely non-existent path should be unchanged: got %q", result)
	}
}

func TestResolveExistingPrefix_WithSymlink(t *testing.T) {
	dir := t.TempDir()
	realDir := filepath.Join(dir, "real")
	_ = os.MkdirAll(realDir, 0o755)
	link := filepath.Join(dir, "link")
	_ = os.Symlink(realDir, link)

	child := filepath.Join(link, "newfile.txt")
	result := ResolveExistingPrefix(child)
	// realDir may also be resolved (macOS /var → /private/var).
	resolvedReal, _ := filepath.EvalSymlinks(realDir)
	expected := filepath.Join(resolvedReal, "newfile.txt")
	if result != expected {
		t.Errorf("symlink prefix should be resolved: got %q, want %q", result, expected)
	}
}
