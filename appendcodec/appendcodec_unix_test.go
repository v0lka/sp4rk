//go:build unix

package appendcodec

import (
	"errors"
	"os"
	"path/filepath"
	"syscall"
	"testing"
)

// TestDetectMagicRefusesSymlink pins the link-safe guarantee of DetectMagic:
// a symlink at the final path component is refused (ELOOP) instead of being
// resolved, so the sniff can never be redirected to a planted target.
// Windows has the same guarantee — its OpenFileNoFollow opens with
// FILE_FLAG_OPEN_REPARSE_POINT and refuses a reparse-point handle with
// syscall.ELOOP, pinned by safeio_windows_test.go. This file stays
// unix-tagged only because it asserts syscall.ELOOP directly.
func TestDetectMagicRefusesSymlink(t *testing.T) {
	dir := t.TempDir()
	target := filepath.Join(dir, "target.log")
	if err := os.WriteFile(target, []byte("line\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(dir, "link.log")
	if err := os.Symlink(target, link); err != nil {
		t.Fatal(err)
	}

	_, err := DetectMagic(link, nil)
	var pe *os.PathError
	if !errors.As(err, &pe) || !errors.Is(pe.Err, syscall.ELOOP) {
		t.Fatalf("DetectMagic through symlink: error = %v, want ELOOP", err)
	}
}

// TestOpenAppendRefusesSymlink pins the same guarantee for the append open:
// no write can be redirected through a final-component symlink.
func TestOpenAppendRefusesSymlink(t *testing.T) {
	dir := t.TempDir()
	target := filepath.Join(dir, "target.log")
	if err := os.WriteFile(target, []byte("line\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(dir, "link.log")
	if err := os.Symlink(target, link); err != nil {
		t.Fatal(err)
	}

	w, err := OpenAppend(link, CodecNone, 0)
	if err == nil {
		_ = w.Close()
		t.Fatal("OpenAppend through symlink: want error, got nil")
	}
	var pe *os.PathError
	if !errors.As(err, &pe) || !errors.Is(pe.Err, syscall.ELOOP) {
		t.Fatalf("OpenAppend through symlink: error = %v, want ELOOP", err)
	}
	// The refusal must not have followed the link and appended to the target.
	data, rerr := os.ReadFile(target)
	if rerr != nil {
		t.Fatalf("read append target: %v", rerr)
	}
	if len(data) != len("line\n") {
		t.Errorf("append target was modified through the symlink: %q", data)
	}
}

// TestOpenAppendErrLockedOnAliasedPath pins the remaining ErrLocked path for
// the compressed codecs: the same file reached under two differently-
// normalized keys (via a symlinked parent directory) misses the process-local
// sharing registry, so the second open runs into the advisory lock, which
// conflicts across open file descriptions even within one process. The safe
// degradation is the ErrLocked failure — never interleaved frames. The same
// registry-miss + lock-conflict code path serves genuine cross-process
// conflicts. CodecNone takes no lock (compatibility mode), so it has the
// inverted test below.
func TestOpenAppendErrLockedOnAliasedPath(t *testing.T) {
	root := t.TempDir()
	realDir := filepath.Join(root, "real")
	if err := os.Mkdir(realDir, 0o755); err != nil {
		t.Fatal(err)
	}
	aliasDir := filepath.Join(root, "alias")
	if err := os.Symlink(realDir, aliasDir); err != nil {
		t.Fatal(err)
	}

	w1, err := OpenAppend(filepath.Join(realDir, "log"), CodecGzip, 0)
	if err != nil {
		t.Fatalf("OpenAppend real: %v", err)
	}
	if _, err := w1.Write([]byte("held\n")); err != nil {
		t.Fatalf("write: %v", err)
	}

	w2, err := OpenAppend(filepath.Join(aliasDir, "log"), CodecGzip, 0)
	if err == nil {
		_ = w2.Close()
		_ = w1.Close()
		t.Fatal("OpenAppend aliased path: want ErrLocked, got nil")
	}
	if !errors.Is(err, ErrLocked) {
		_ = w1.Close()
		t.Fatalf("error = %v, want ErrLocked", err)
	}

	// The refused open must not have disturbed the live writer.
	if _, err := w1.Write([]byte("still fine\n")); err != nil {
		t.Fatalf("write after refused open: %v", err)
	}
	requireClose(t, w1)
}

// TestOpenAppendNoneAllowsAliasedPath pins the compatibility codec's
// inverted case: CodecNone takes no advisory lock and consults no registry,
// so the same file reached under two differently-normalized keys (via a
// symlinked parent directory — a registry miss for the compressed codecs)
// opens just fine, and both independent handles append side by side like
// the pre-appendcodec plain writers did.
func TestOpenAppendNoneAllowsAliasedPath(t *testing.T) {
	root := t.TempDir()
	realDir := filepath.Join(root, "real")
	if err := os.Mkdir(realDir, 0o755); err != nil {
		t.Fatal(err)
	}
	aliasDir := filepath.Join(root, "alias")
	if err := os.Symlink(realDir, aliasDir); err != nil {
		t.Fatal(err)
	}

	w1, err := OpenAppend(filepath.Join(realDir, "log"), CodecNone, 0)
	if err != nil {
		t.Fatalf("OpenAppend real: %v", err)
	}
	if _, err := w1.Write([]byte("from real\n")); err != nil {
		t.Fatalf("write 1: %v", err)
	}

	w2, err := OpenAppend(filepath.Join(aliasDir, "log"), CodecNone, 0)
	if err != nil {
		_ = w1.Close()
		t.Fatalf("OpenAppend aliased path: %v (compat mode must not lock)", err)
	}
	if _, err := w2.Write([]byte("from alias\n")); err != nil {
		t.Fatalf("write 2: %v", err)
	}

	requireClose(t, w1)
	requireClose(t, w2)

	data, rerr := os.ReadFile(filepath.Join(realDir, "log"))
	if rerr != nil {
		t.Fatalf("read: %v", rerr)
	}
	if got := string(data); got != "from real\nfrom alias\n" {
		t.Errorf("file = %q, want both records", got)
	}
}
