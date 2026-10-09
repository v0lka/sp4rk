package safeio

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// TestWriteFile_WritesRegularFile exercises the happy path of WriteFile — the
// hardened replacement for os.WriteFile — on every platform.
func TestWriteFile_WritesRegularFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "out.txt")

	if err := WriteFile(path, []byte("hello\n"), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("os.ReadFile: %v", err)
	}
	if string(got) != "hello\n" {
		t.Fatalf("WriteFile wrote %q, want %q", got, "hello\n")
	}
}

// TestWriteFile_TruncatesExisting proves the O_TRUNC behavior: an existing
// longer file is replaced wholesale, not partially overwritten.
func TestWriteFile_TruncatesExisting(t *testing.T) {
	path := filepath.Join(t.TempDir(), "out.txt")
	if err := os.WriteFile(path, []byte("0123456789"), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := WriteFile(path, []byte("ab"), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("os.ReadFile: %v", err)
	}
	if string(got) != "ab" {
		t.Fatalf("WriteFile left %q, want truncated %q", got, "ab")
	}
}

// TestWriteFile_PermBitsWithinRequested checks the permission argument: the
// created file's perm bits must be a subset of the requested ones (the umask
// may remove bits but never add them).
func TestWriteFile_PermBitsWithinRequested(t *testing.T) {
	path := filepath.Join(t.TempDir(), "out.txt")
	if err := WriteFile(path, []byte("x"), 0o600); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	st, err := os.Stat(path)
	if err != nil {
		t.Fatalf("os.Stat: %v", err)
	}
	if extra := st.Mode().Perm() &^ fs.FileMode(0o600).Perm(); extra != 0 {
		t.Fatalf("WriteFile(0o600) created mode %v, has bits outside the request", st.Mode())
	}
}

// TestOpenFileNoFollow_CreatesRegularFile covers the happy path of
// OpenFileNoFollow on every platform: create, write, read back.
func TestOpenFileNoFollow_CreatesRegularFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "out.txt")

	f, err := OpenFileNoFollow(path, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		t.Fatalf("OpenFileNoFollow: %v", err)
	}
	if _, err := f.WriteString("data\n"); err != nil {
		_ = f.Close()
		t.Fatalf("WriteString: %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("os.ReadFile: %v", err)
	}
	if string(got) != "data\n" {
		t.Fatalf("file holds %q, want %q", got, "data\n")
	}
}

// TestWriteFileAtomic_WritesRegularFile covers the happy path of
// WriteFileAtomic: the data lands at the target as a regular file, and no
// temporary file is left behind in the directory.
func TestWriteFileAtomic_WritesRegularFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "out.txt")

	if err := WriteFileAtomic(path, []byte("hello\n"), 0o644); err != nil {
		t.Fatalf("WriteFileAtomic: %v", err)
	}

	st, err := os.Lstat(path)
	if err != nil {
		t.Fatalf("os.Lstat: %v", err)
	}
	if !st.Mode().IsRegular() {
		t.Fatalf("target is %v, want a regular file", st.Mode())
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("os.ReadFile: %v", err)
	}
	if string(got) != "hello\n" {
		t.Fatalf("WriteFileAtomic wrote %q, want %q", got, "hello\n")
	}

	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("os.ReadDir: %v", err)
	}
	if len(entries) != 1 || entries[0].Name() != "out.txt" {
		names := make([]string, 0, len(entries))
		for _, e := range entries {
			names = append(names, e.Name())
		}
		t.Fatalf("directory holds %v after WriteFileAtomic, want only [out.txt] (temp leftover?)", names)
	}
}

// TestWriteFileAtomic_ReplacesExisting proves a pre-existing file is replaced
// wholesale with the new contents.
func TestWriteFileAtomic_ReplacesExisting(t *testing.T) {
	path := filepath.Join(t.TempDir(), "out.txt")
	if err := os.WriteFile(path, []byte("old-contents"), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := WriteFileAtomic(path, []byte("new"), 0o644); err != nil {
		t.Fatalf("WriteFileAtomic: %v", err)
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("os.ReadFile: %v", err)
	}
	if string(got) != "new" {
		t.Fatalf("WriteFileAtomic left %q, want %q", got, "new")
	}
}

// TestWriteFileAtomic_PermBitsWithinRequested checks the chmod-to-perm
// behavior: the published file's perm bits must be a subset of the requested
// ones.
func TestWriteFileAtomic_PermBitsWithinRequested(t *testing.T) {
	path := filepath.Join(t.TempDir(), "out.txt")
	if err := WriteFileAtomic(path, []byte("x"), 0o600); err != nil {
		t.Fatalf("WriteFileAtomic: %v", err)
	}
	st, err := os.Stat(path)
	if err != nil {
		t.Fatalf("os.Stat: %v", err)
	}
	if extra := st.Mode().Perm() &^ fs.FileMode(0o600).Perm(); extra != 0 {
		t.Fatalf("WriteFileAtomic(0o600) published mode %v, has bits outside the request", st.Mode())
	}
}

// TestWriteFileAtomic_PermExactOnUnix pins the documented perm semantics:
// WriteFileAtomic applies perm EXACTLY via fchmod, NOT filtered through the
// process umask — unlike WriteFile/os.WriteFile, whose open(2) call applies
// it. Callers passing the bits they want get exactly those bits, regardless
// of how restrictive the ambient umask is.
func TestWriteFileAtomic_PermExactOnUnix(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("fchmod semantics are unix-specific; Windows maps modes onto the read-only attribute")
	}
	path := filepath.Join(t.TempDir(), "out.txt")
	if err := WriteFileAtomic(path, []byte("x"), 0o640); err != nil {
		t.Fatalf("WriteFileAtomic: %v", err)
	}
	st, err := os.Stat(path)
	if err != nil {
		t.Fatalf("os.Stat: %v", err)
	}
	if got := st.Mode().Perm(); got != 0o640 {
		t.Fatalf("WriteFileAtomic(0o640) published mode %v, want exactly 0640 (fchmod is not umask-filtered)", st.Mode())
	}
}

// TestWriteFileAtomic_MissingParentDirIsError pins that a missing parent is an
// error, not a silent success (MkdirAllReal is the documented way to build it).
func TestWriteFileAtomic_MissingParentDirIsError(t *testing.T) {
	err := WriteFileAtomic(filepath.Join(t.TempDir(), "missing", "out.txt"), []byte("x"), 0o644)
	if err == nil {
		t.Fatal("WriteFileAtomic into a missing parent dir succeeded, want an error")
	}
}

// TestMkdirAllReal_CreatesRealDirs covers the happy path: every component of a
// fresh nested path is created as a genuine directory (never a symlink).
func TestMkdirAllReal_CreatesRealDirs(t *testing.T) {
	root := t.TempDir()
	nested := filepath.Join(root, "a", "b", "c")

	if err := MkdirAllReal(nested, 0o755); err != nil {
		t.Fatalf("MkdirAllReal: %v", err)
	}

	for p := filepath.Join(root, "a"); ; p = filepath.Dir(p) {
		st, err := os.Lstat(p)
		if err != nil {
			t.Fatalf("os.Lstat(%s): %v", p, err)
		}
		if !st.IsDir() || st.Mode()&fs.ModeSymlink != 0 {
			t.Fatalf("component %s: mode %v, want a real directory", p, st.Mode())
		}
		if p == root {
			break
		}
	}
}

// TestMkdirAllReal_AcceptsExistingRealDir proves idempotence on a path whose
// components all already exist as real directories.
func TestMkdirAllReal_AcceptsExistingRealDir(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "exists")
	if err := os.Mkdir(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := MkdirAllReal(dir, 0o755); err != nil {
		t.Fatalf("MkdirAllReal(existing real dir): %v", err)
	}
	if err := MkdirAllReal(filepath.Join(dir, "sub"), 0o755); err != nil {
		t.Fatalf("MkdirAllReal(under existing real dir): %v", err)
	}
}

// TestMkdirAllReal_RefusesFileComponent proves a regular file in a component
// position is refused with ErrNotDirectory instead of being followed, and
// nothing is created through it.
func TestMkdirAllReal_RefusesFileComponent(t *testing.T) {
	root := t.TempDir()
	blocker := filepath.Join(root, "blocker")
	if err := os.WriteFile(blocker, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}

	err := MkdirAllReal(filepath.Join(blocker, "sub"), 0o755)
	if err == nil {
		t.Fatal("MkdirAllReal through a regular file succeeded, want a refusal")
	}
	if !errors.Is(err, ErrNotDirectory) {
		t.Fatalf("MkdirAllReal err = %v, want errors.Is(err, ErrNotDirectory)", err)
	}
	if _, statErr := os.Lstat(filepath.Join(blocker, "sub")); statErr == nil {
		t.Fatal("something was created through the file component")
	}
}

// TestMkdirAllReal_EmptyPathIsError matches os.MkdirAll(""), which fails.
func TestMkdirAllReal_EmptyPathIsError(t *testing.T) {
	if err := MkdirAllReal("", 0o755); err == nil {
		t.Fatal("MkdirAllReal(\"\") succeeded, want an error")
	}
}

// TestCheckRealDirsBelow_RefusesSymlinkBelowRoot pins the compensation
// contract: every component strictly below the given root must be a real
// directory — a symlink shipped inside the controlled subtree is refused,
// while symlinked ancestors ABOVE the root (macOS /var, a symlinked home) are
// not inspected at all.
func TestCheckRealDirsBelow_RefusesSymlinkBelowRoot(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("symlink creation requires privileges on Windows")
	}
	root := t.TempDir()
	link := filepath.Join(root, "link")
	if err := os.Symlink(t.TempDir(), link); err != nil {
		t.Fatal(err)
	}

	// The link itself, in component position below root: refused.
	err := CheckRealDirsBelow(root, filepath.Join(link, "sub"))
	if !errors.Is(err, ErrSymlink) {
		t.Fatalf("CheckRealDirsBelow err = %v, want errors.Is(err, ErrSymlink)", err)
	}
	var sym *SymlinkError
	if !errors.As(err, &sym) {
		t.Fatalf("CheckRealDirsBelow err = %T(%v), want a *SymlinkError", err, err)
	}

	// A real subtree below root passes.
	realSub := filepath.Join(root, "real", "deep")
	if err := os.MkdirAll(realSub, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := CheckRealDirsBelow(root, realSub); err != nil {
		t.Fatalf("CheckRealDirsBelow(real subtree): %v", err)
	}
	// The root itself is never inspected: root==path is a no-op.
	if err := CheckRealDirsBelow(root, root); err != nil {
		t.Fatalf("CheckRealDirsBelow(root): %v", err)
	}
	// A path outside root is invalid input, not a pass.
	err = CheckRealDirsBelow(filepath.Join(root, "real"), filepath.Join(root, "link"))
	if err == nil || !errors.Is(err, fs.ErrInvalid) {
		t.Fatalf("CheckRealDirsBelow(outside root) = %v, want ErrInvalid", err)
	}
}

func TestMkdirAllRealWithin_CreatesRealDirs(t *testing.T) {
	root := t.TempDir()
	dir := filepath.Join(root, "a", "b", "c")
	if err := MkdirAllRealWithin(root, dir, 0o755); err != nil {
		t.Fatalf("MkdirAllRealWithin: %v", err)
	}
	fi, err := os.Lstat(dir)
	if err != nil || !fi.IsDir() {
		t.Fatalf("dir not created as a real directory: %v", err)
	}
	// Equal to the boundary itself is trivially within.
	if err := MkdirAllRealWithin(root, root, 0o755); err != nil {
		t.Fatalf("MkdirAllRealWithin(root): %v", err)
	}
}

func TestMkdirAllRealWithin_RefusesEscapingLeafLink(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("symlink creation requires privileges on Windows")
	}
	root := t.TempDir()
	outside := t.TempDir()
	dir := filepath.Join(root, "projects", "p1", "s1", "plans")
	if err := os.MkdirAll(filepath.Dir(dir), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, dir); err != nil {
		t.Fatal(err)
	}

	err := MkdirAllRealWithin(root, dir, 0o755)
	if !errors.Is(err, ErrSymlink) {
		t.Fatalf("escaping leaf link: err = %v, want errors.Is(err, ErrSymlink)", err)
	}
	// Nothing may have been created at the link target.
	entries, _ := os.ReadDir(outside)
	if len(entries) != 0 {
		t.Fatalf("the link target was modified: %v", entries)
	}
}

func TestMkdirAllRealWithin_RefusesEscapingIntermediateLink(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("symlink creation requires privileges on Windows")
	}
	root := t.TempDir()
	outside := t.TempDir()
	// The ESCAPING component is an ancestor, not the leaf: a planted
	// <root>/.agents must not adopt <root>/.agents/skills either.
	if err := os.Symlink(outside, filepath.Join(root, ".agents")); err != nil {
		t.Fatal(err)
	}
	dir := filepath.Join(root, ".agents", "skills")

	err := MkdirAllRealWithin(root, dir, 0o755)
	if !errors.Is(err, ErrSymlink) {
		t.Fatalf("escaping intermediate link: err = %v, want errors.Is(err, ErrSymlink)", err)
	}
	entries, _ := os.ReadDir(outside)
	if len(entries) != 0 {
		t.Fatalf("the link target was modified: %v", entries)
	}
}

func TestMkdirAllRealWithin_AllowsInBoundaryOperatorLink(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("symlink creation requires privileges on Windows")
	}
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "real"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(filepath.Join(root, "real"), filepath.Join(root, "alias")); err != nil {
		t.Fatal(err)
	}
	// An operator link resolving WITHIN the boundary is intent, not escape.
	dir := filepath.Join(root, "alias", "sub")
	if err := MkdirAllRealWithin(root, dir, 0o755); err != nil {
		t.Fatalf("in-boundary link: %v", err)
	}
	if fi, err := os.Lstat(filepath.Join(root, "real", "sub")); err != nil || !fi.IsDir() {
		t.Fatalf("the tree was not created through the in-boundary link: %v", err)
	}
}

func TestMkdirAllRealWithin_AllowsSymlinkedRoot(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("symlink creation requires privileges on Windows")
	}
	root := t.TempDir()
	alias := filepath.Join(t.TempDir(), "agent-link")
	if err := os.Symlink(root, alias); err != nil {
		t.Fatal(err)
	}
	// A symlinked ANCHOR (operator intent: whole agent dir moved) resolves:
	// the boundary is the resolved root, and the created tree lands in it.
	dir := filepath.Join(alias, "logs")
	if err := MkdirAllRealWithin(alias, dir, 0o755); err != nil {
		t.Fatalf("symlinked root: %v", err)
	}
	if fi, err := os.Lstat(filepath.Join(root, "logs")); err != nil || !fi.IsDir() {
		t.Fatalf("the tree was not created inside the resolved root: %v", err)
	}
}

func TestMkdirAllRealWithin_DanglingLeafLinkStillRefused(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("symlink creation requires privileges on Windows")
	}
	root := t.TempDir()
	dir := filepath.Join(root, "logs")
	if err := os.Symlink(filepath.Join(root, "gone"), dir); err != nil {
		t.Fatal(err)
	}
	if err := MkdirAllRealWithin(root, dir, 0o755); !errors.Is(err, ErrSymlink) {
		t.Fatalf("dangling leaf link: err = %v, want ErrSymlink", err)
	}
}
