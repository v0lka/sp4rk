//go:build unix

package safeio

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"syscall"
	"testing"
	"time"
)

// TestWriteFile_RefusesFifoWithoutBlocking is the write-side twin of
// TestReadFile_RefusesFifoWithoutBlocking: a FIFO planted at a write path must
// be refused promptly, never awaited — a write-open of a FIFO with no reader
// blocks inside open(2) until a reader appears, uninterruptibly. The watchdog
// turns a regression into a test failure instead of a hung CI job.
func TestWriteFile_RefusesFifoWithoutBlocking(t *testing.T) {
	fifo := filepath.Join(t.TempDir(), "fifo")
	if err := syscall.Mkfifo(fifo, 0o644); err != nil {
		t.Fatalf("mkfifo: %v", err)
	}

	done := make(chan error, 1)
	go func() {
		done <- WriteFile(fifo, []byte("x"), 0o644)
	}()
	select {
	case err := <-done:
		if err == nil {
			t.Fatal("WriteFile(FIFO) succeeded, want a refusal")
		}
	case <-time.After(10 * time.Second):
		t.Fatal("WriteFile hung on a FIFO (blocking write-open)")
	}
}

// TestWriteFile_RefusesFifoViaFstat forces the post-open guard: with a reader
// already holding the FIFO, the non-blocking write-open succeeds, so only the
// fstat on the open descriptor can refuse it.
func TestWriteFile_RefusesFifoViaFstat(t *testing.T) {
	fifo := filepath.Join(t.TempDir(), "fifo")
	if err := syscall.Mkfifo(fifo, 0o644); err != nil {
		t.Fatalf("mkfifo: %v", err)
	}

	// Hold the read end without blocking (a blocking read-open of a FIFO waits
	// for a writer), so the subsequent write-open has a reader and returns.
	reader, err := os.OpenFile(fifo, os.O_RDONLY|syscall.O_NONBLOCK, 0)
	if err != nil {
		t.Fatalf("open reader: %v", err)
	}
	defer func() { _ = reader.Close() }()

	err = WriteFile(fifo, []byte("x"), 0o644)
	if err == nil {
		t.Fatal("WriteFile(FIFO) with a reader succeeded, want a non-regular refusal")
	}
	if !errors.Is(err, ErrNotRegular) {
		t.Fatalf("WriteFile(FIFO) err = %v, want errors.Is(err, ErrNotRegular)", err)
	}
}

// TestWriteFile_RefusesFinalSymlink proves O_NOFOLLOW: a symbolic link at the
// final path is refused with ELOOP and the link's TARGET is never written
// through.
func TestWriteFile_RefusesFinalSymlink(t *testing.T) {
	dir := t.TempDir()
	target := filepath.Join(dir, "target.txt")
	if err := os.WriteFile(target, []byte("keep"), 0o644); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(dir, "link.txt")
	if err := os.Symlink(target, link); err != nil {
		t.Fatal(err)
	}

	err := WriteFile(link, []byte("clobber"), 0o644)
	if err == nil {
		t.Fatal("WriteFile(final symlink) succeeded, want an ELOOP refusal")
	}
	if !errors.Is(err, syscall.ELOOP) {
		t.Fatalf("WriteFile(final symlink) err = %v, want errors.Is(err, syscall.ELOOP)", err)
	}

	got, err := os.ReadFile(target)
	if err != nil {
		t.Fatalf("os.ReadFile(target): %v", err)
	}
	if string(got) != "keep" {
		t.Fatalf("symlink target was written through: %q, want %q", got, "keep")
	}
}

// TestOpenFileNoFollow_RefusesFinalSymlink pins the same O_NOFOLLOW guarantee
// at the OpenFileNoFollow level, including with O_CREATE.
func TestOpenFileNoFollow_RefusesFinalSymlink(t *testing.T) {
	dir := t.TempDir()
	target := filepath.Join(dir, "target.txt")
	if err := os.WriteFile(target, []byte("keep"), 0o644); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(dir, "link.txt")
	if err := os.Symlink(target, link); err != nil {
		t.Fatal(err)
	}

	f, err := OpenFileNoFollow(link, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err == nil {
		_ = f.Close()
		t.Fatal("OpenFileNoFollow(final symlink) succeeded, want an ELOOP refusal")
	}
	if !errors.Is(err, syscall.ELOOP) {
		t.Fatalf("OpenFileNoFollow(final symlink) err = %v, want errors.Is(err, syscall.ELOOP)", err)
	}
	if _, statErr := os.Lstat(filepath.Join(dir, "created-through-link")); !errors.Is(statErr, fs.ErrNotExist) {
		t.Fatalf("unexpected file created: %v", statErr)
	}
}

// TestWriteFileAtomic_ReplacesFinalSymlinkAtomically proves the rename-based
// publish replaces a planted symlink ITSELF — never writing through it — and
// leaves the link's original target untouched.
func TestWriteFileAtomic_ReplacesFinalSymlinkAtomically(t *testing.T) {
	dir := t.TempDir()
	target := filepath.Join(dir, "target.txt")
	if err := os.WriteFile(target, []byte("keep"), 0o644); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(dir, "link.txt")
	if err := os.Symlink(target, link); err != nil {
		t.Fatal(err)
	}

	if err := WriteFileAtomic(link, []byte("new"), 0o644); err != nil {
		t.Fatalf("WriteFileAtomic(final symlink): %v", err)
	}

	st, err := os.Lstat(link)
	if err != nil {
		t.Fatalf("os.Lstat(link): %v", err)
	}
	if st.Mode()&fs.ModeSymlink != 0 {
		t.Fatal("WriteFileAtomic left the symlink in place, want it replaced")
	}
	if !st.Mode().IsRegular() {
		t.Fatalf("published entry has mode %v, want a regular file", st.Mode())
	}

	got, err := os.ReadFile(link)
	if err != nil {
		t.Fatalf("os.ReadFile(link): %v", err)
	}
	if string(got) != "new" {
		t.Fatalf("published file holds %q, want %q", got, "new")
	}

	kept, err := os.ReadFile(target)
	if err != nil {
		t.Fatalf("os.ReadFile(target): %v", err)
	}
	if string(kept) != "keep" {
		t.Fatalf("the symlink's original target was modified: %q, want %q", kept, "keep")
	}

	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("os.ReadDir: %v", err)
	}
	if len(entries) != 2 {
		names := make([]string, 0, len(entries))
		for _, e := range entries {
			names = append(names, e.Name())
		}
		t.Fatalf("directory holds %v, want only [link.txt target.txt] (temp leftover?)", names)
	}
}

// TestMkdirAllReal_ResolvesAncestorSymlink pins the resolution contract that
// keeps macOS (/var -> /private/var, t.TempDir) and operator-symlinked agent
// directories working: a pre-existing symlinked ancestor is resolved once to
// its real target, and the missing tail is created under the RESOLVED prefix
// as genuine directories.
func TestMkdirAllReal_ResolvesAncestorSymlink(t *testing.T) {
	dir := t.TempDir()
	realDir := filepath.Join(dir, "real")
	if err := os.Mkdir(realDir, 0o755); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(dir, "link")
	if err := os.Symlink(realDir, link); err != nil {
		t.Fatal(err)
	}

	nested := filepath.Join(link, "sub", "deeper")
	if err := MkdirAllReal(nested, 0o755); err != nil {
		t.Fatalf("MkdirAllReal(through symlinked ancestor): %v", err)
	}
	// The tree must exist under the symlink's TARGET as real directories.
	for p := filepath.Join(realDir, "sub", "deeper"); ; p = filepath.Dir(p) {
		st, err := os.Lstat(p)
		if err != nil {
			t.Fatalf("os.Lstat(%s): %v", p, err)
		}
		if !st.IsDir() || st.Mode()&fs.ModeSymlink != 0 {
			t.Fatalf("component %s: mode %v, want a real directory", p, st.Mode())
		}
		if p == realDir {
			break
		}
	}
	// Idempotent on the same path.
	if err := MkdirAllReal(nested, 0o755); err != nil {
		t.Fatalf("MkdirAllReal(idempotent): %v", err)
	}
}

// TestMkdirAllReal_FinalSymlinkToDirResolves pins that a FINAL component that
// is a symlink to a real directory resolves like any ancestor: the call
// succeeds, matching os.MkdirAll on a symlinked directory.
func TestMkdirAllReal_FinalSymlinkToDirResolves(t *testing.T) {
	dir := t.TempDir()
	realDir := filepath.Join(dir, "real")
	if err := os.Mkdir(realDir, 0o755); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(dir, "link")
	if err := os.Symlink(realDir, link); err != nil {
		t.Fatal(err)
	}
	if err := MkdirAllReal(link, 0o755); err != nil {
		t.Fatalf("MkdirAllReal(final symlink to dir): %v", err)
	}
}

// TestMkdirAllReal_RefusesDanglingSymlink pins that a link whose target does
// not exist is refused wherever it sits: there is nothing to resolve to, and
// creating the missing tree under a broken link's name would silently adopt a
// path the operator left dangling.
func TestMkdirAllReal_RefusesDanglingSymlink(t *testing.T) {
	dir := t.TempDir()

	// Ancestor position.
	dangling := filepath.Join(dir, "dangling")
	if err := os.Symlink(filepath.Join(dir, "missing"), dangling); err != nil {
		t.Fatal(err)
	}
	err := MkdirAllReal(filepath.Join(dangling, "sub"), 0o755)
	if err == nil {
		t.Fatal("MkdirAllReal through a dangling symlink succeeded, want a refusal")
	}
	if !errors.Is(err, ErrSymlink) {
		t.Fatalf("MkdirAllReal err = %v, want errors.Is(err, ErrSymlink)", err)
	}
	var sym *SymlinkError
	if !errors.As(err, &sym) {
		t.Fatalf("MkdirAllReal err = %T(%v), want a *SymlinkError", err, err)
	}
	if _, statErr := os.Lstat(filepath.Join(dir, "missing")); !errors.Is(statErr, fs.ErrNotExist) {
		t.Fatalf("creation was redirected through the dangling link: %v", statErr)
	}

	// Final position.
	err = MkdirAllReal(dangling, 0o755)
	if err == nil {
		t.Fatal("MkdirAllReal(final dangling symlink) succeeded, want a refusal")
	}
	if !errors.Is(err, ErrSymlink) {
		t.Fatalf("MkdirAllReal err = %v, want errors.Is(err, ErrSymlink)", err)
	}
}

// TestMkdirAllReal_SymlinkToFileIsNotDirectory pins that a link resolving to
// a regular file is refused as a non-directory, with nothing created through
// the file.
func TestMkdirAllReal_SymlinkToFileIsNotDirectory(t *testing.T) {
	dir := t.TempDir()
	file := filepath.Join(dir, "file")
	if err := os.WriteFile(file, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(dir, "link")
	if err := os.Symlink(file, link); err != nil {
		t.Fatal(err)
	}

	err := MkdirAllReal(filepath.Join(link, "sub"), 0o755)
	if err == nil {
		t.Fatal("MkdirAllReal through a symlink-to-file succeeded, want a refusal")
	}
	if !errors.Is(err, ErrNotDirectory) {
		t.Fatalf("MkdirAllReal err = %v, want errors.Is(err, ErrNotDirectory)", err)
	}
	if _, statErr := os.Lstat(filepath.Join(file, "sub")); statErr == nil {
		t.Fatal("something was created through the file the link resolves to")
	}
}

// TestMkdirAllReal_RefusesFifoComponent proves a FIFO planted in a component
// position is refused via Lstat — which can never block — instead of being
// followed into a blocking open. Resolution only ever follows symlinks; a
// FIFO is not one and never becomes a target.
func TestMkdirAllReal_RefusesFifoComponent(t *testing.T) {
	dir := t.TempDir()
	fifo := filepath.Join(dir, "fifo")
	if err := syscall.Mkfifo(fifo, 0o644); err != nil {
		t.Fatalf("mkfifo: %v", err)
	}

	err := MkdirAllReal(filepath.Join(fifo, "sub"), 0o755)
	if err == nil {
		t.Fatal("MkdirAllReal through a FIFO component succeeded, want a refusal")
	}
	if !errors.Is(err, ErrNotDirectory) {
		t.Fatalf("MkdirAllReal err = %v, want errors.Is(err, ErrNotDirectory)", err)
	}
}
