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
// Windows is covered by the safeio parity note — its OpenFileNoFollow still
// resolves symlinks and the unix-tagged assertion has no equivalent there.
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
