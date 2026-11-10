//go:build windows

package safeio

import (
	"errors"
	"io"
	"os"
	"path/filepath"
	"syscall"
	"testing"
)

// errorPrivilegeNotHeld is Windows' ERROR_PRIVILEGE_NOT_HELD (1314); the
// stdlib syscall package does not export it by name.
const errorPrivilegeNotHeld = syscall.Errno(1314)

// requireFileSymlink plants a symlink at link pointing to target, skipping
// the test when the host lacks the privilege: file-symlink creation on
// Windows needs elevated rights or Developer Mode, so a non-elevated dev
// machine cannot pin this invariant locally — the elevated CI Windows leg
// exercises it. A symlink failure that is NOT the privilege refusal is
// fatal, not skipped.
func requireFileSymlink(t *testing.T, target, link string) {
	t.Helper()
	err := os.Symlink(target, link)
	if err == nil {
		return
	}
	var errno syscall.Errno
	if errors.As(err, &errno) && (errno == errorPrivilegeNotHeld || errno == syscall.ERROR_ACCESS_DENIED) {
		t.Skipf("file symlinks on Windows require elevated privileges: %v", err)
	}
	t.Fatalf("plant symlink: %v", err)
}

// TestOpenFileNoFollow_RefusesFinalSymlink pins the O_NOFOLLOW contract: a
// file symlink at the final path component is refused with ELOOP, the link's
// target is never resolved, and both the link and its target survive.
func TestOpenFileNoFollow_RefusesFinalSymlink(t *testing.T) {
	dir := t.TempDir()
	target := filepath.Join(dir, "target")
	if err := os.WriteFile(target, []byte("secret"), 0o644); err != nil {
		t.Fatalf("seed target: %v", err)
	}
	link := filepath.Join(dir, "link")
	requireFileSymlink(t, target, link)

	f, err := OpenFileNoFollow(link, os.O_RDONLY, 0)
	if err == nil {
		_ = f.Close()
		t.Fatal("OpenFileNoFollow opened through a final symlink, want an ELOOP refusal")
	}
	if !errors.Is(err, syscall.ELOOP) {
		t.Fatalf("OpenFileNoFollow err = %v, want a *fs.PathError wrapping syscall.ELOOP", err)
	}

	data, rerr := os.ReadFile(target)
	if rerr != nil || string(data) != "secret" {
		t.Fatalf("symlink target was disturbed: %q, %v", data, rerr)
	}
	if _, lerr := os.Lstat(link); lerr != nil {
		t.Fatalf("symlink itself must survive: %v", lerr)
	}
}

// TestWriteFile_RefusesFinalSymlink pins the O_TRUNC write shape against a
// final symlink. This is the one Windows shape where the stdlib can surface
// an error other than the documented ELOOP: syscall.Open implements O_TRUNC
// as a post-open Ftruncate on the reparse-point handle (before OpenFileNoFollow's
// fstat ever sees it), and a SetEndOfFile failure there escapes the open as
// its own *fs.PathError. Per MS "Symbolic Link Effects on File Systems
// Functions" the truncation affects the LINK, never the target, so the
// realistic outcome is ELOOP with an empty-$DATA side effect on the link —
// the byte-divergence from unix, where the link stays byte-identical. The
// last assertion pins the link-survives property so the divergence stays
// documented: the link itself still resolves after the refusal.
func TestWriteFile_RefusesFinalSymlink(t *testing.T) {
	dir := t.TempDir()
	target := filepath.Join(dir, "victim")
	if err := os.WriteFile(target, []byte("original"), 0o644); err != nil {
		t.Fatalf("seed victim: %v", err)
	}
	link := filepath.Join(dir, "link")
	requireFileSymlink(t, target, link)

	if err := WriteFile(link, []byte("payload"), 0o644); err == nil {
		t.Fatal("WriteFile wrote through a final symlink, want a refusal")
	} else if !errors.Is(err, syscall.ELOOP) {
		t.Fatalf("WriteFile err = %v, want a *fs.PathError wrapping syscall.ELOOP", err)
	}

	data, rerr := os.ReadFile(target)
	if rerr != nil || string(data) != "original" {
		t.Fatalf("victim was truncated or written through the link: %q, %v", data, rerr)
	}
	if _, lerr := os.Readlink(link); lerr != nil {
		t.Fatalf("symlink itself must survive and still resolve: %v", lerr)
	}
}

// TestOpenFileNoFollow_AppendCreateDoesNotWriteThroughLink pins the
// write-open shape the append codec uses: O_RDWR|O_CREATE|O_APPEND against a
// planted symlink must refuse, never append through the link.
func TestOpenFileNoFollow_AppendCreateDoesNotWriteThroughLink(t *testing.T) {
	dir := t.TempDir()
	target := filepath.Join(dir, "victim")
	if err := os.WriteFile(target, []byte("original"), 0o644); err != nil {
		t.Fatalf("seed victim: %v", err)
	}
	link := filepath.Join(dir, "link")
	requireFileSymlink(t, target, link)

	f, err := OpenFileNoFollow(link, os.O_RDWR|os.O_CREATE|os.O_APPEND, 0o644)
	if err == nil {
		_ = f.Close()
		t.Fatal("append-create open followed a final symlink, want an ELOOP refusal")
	}
	if !errors.Is(err, syscall.ELOOP) {
		t.Fatalf("OpenFileNoFollow err = %v, want a *fs.PathError wrapping syscall.ELOOP", err)
	}

	data, rerr := os.ReadFile(target)
	if rerr != nil || string(data) != "original" {
		t.Fatalf("victim was written through the link: %q, %v", data, rerr)
	}
}

// TestOpenFileNoFollow_CreateRefusesDanglingLinkWithoutCreatingTarget pins
// that O_CREATE against a DANGLING symlink refuses and leaves the link's
// target uncreated — CreateFile must open the reparse point itself, not
// materialize the destination.
func TestOpenFileNoFollow_CreateRefusesDanglingLinkWithoutCreatingTarget(t *testing.T) {
	dir := t.TempDir()
	target := filepath.Join(dir, "victim")
	link := filepath.Join(dir, "link")
	requireFileSymlink(t, target, link)

	f, err := OpenFileNoFollow(link, os.O_RDWR|os.O_CREATE|os.O_APPEND, 0o644)
	if err == nil {
		_ = f.Close()
		t.Fatal("create open followed a dangling symlink, want an ELOOP refusal")
	}
	if !errors.Is(err, syscall.ELOOP) {
		t.Fatalf("OpenFileNoFollow err = %v, want a *fs.PathError wrapping syscall.ELOOP", err)
	}
	if _, serr := os.Stat(target); !os.IsNotExist(serr) {
		t.Fatalf("dangling link's target must stay uncreated: %v", serr)
	}
}

// TestOpenFileNoFollow_RegularFileUnaffected pins that the reparse-point
// flag changes nothing for ordinary files: the flag pass-through must not
// disturb create/append/truncate/read on a regular target.
func TestOpenFileNoFollow_RegularFileUnaffected(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "plain")

	f, err := OpenFileNoFollow(path, os.O_RDWR|os.O_CREATE|os.O_APPEND, 0o644)
	if err != nil {
		t.Fatalf("create-append open: %v", err)
	}
	if _, werr := f.WriteString("data"); werr != nil {
		t.Fatalf("write: %v", werr)
	}
	if cerr := f.Close(); cerr != nil {
		t.Fatalf("close: %v", cerr)
	}

	f, err = OpenFileNoFollow(path, os.O_RDONLY, 0)
	if err != nil {
		t.Fatalf("read open: %v", err)
	}
	data, rerr := io.ReadAll(f)
	_ = f.Close()
	if rerr != nil || string(data) != "data" {
		t.Fatalf("read after append = %q, %v; want \"data\"", data, rerr)
	}

	// The truncate shape WriteFile uses must also stay plain.
	f, err = OpenFileNoFollow(path, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o644)
	if err != nil {
		t.Fatalf("truncate open: %v", err)
	}
	if _, werr := f.WriteString("x"); werr != nil {
		t.Fatalf("write after truncate: %v", werr)
	}
	if cerr := f.Close(); cerr != nil {
		t.Fatalf("close: %v", cerr)
	}
	data, rerr = os.ReadFile(path)
	if rerr != nil || string(data) != "x" {
		t.Fatalf("read after truncate = %q, %v; want \"x\"", data, rerr)
	}
}
