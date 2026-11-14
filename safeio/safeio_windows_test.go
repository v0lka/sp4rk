//go:build windows

package safeio

import (
	"errors"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"syscall"
	"testing"
)

// errorPrivilegeNotHeld is Windows' ERROR_PRIVILEGE_NOT_HELD (1314); the
// stdlib syscall package does not export it by name.
const errorPrivilegeNotHeld = syscall.Errno(1314)

// createNoWindow is the Windows process creation flag CREATE_NO_WINDOW
// (0x08000000): the child allocates no console window at all. It is inlined
// here rather than imported from sysproc so safeio keeps its leaf-package
// import set; the SDK's own children get the same flag via sysproc.HideConsole.
const createNoWindow = 0x08000000

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

// requireJunction plants a directory junction at link pointing to the
// directory target (created here). Junction creation — unlike file-symlink
// creation — needs no elevation, so this plant runs on every Windows host,
// including the non-elevated ones where requireFileSymlink skips.
func requireJunction(t *testing.T, target, link string) {
	t.Helper()
	if err := os.MkdirAll(target, 0o755); err != nil {
		t.Fatalf("seed junction target: %v", err)
	}
	cmd := exec.CommandContext(t.Context(), "cmd", "/c", "mklink", "/J", link, target)
	// mklink is not interactive, so its console must never flash. This sets
	// CREATE_NO_WINDOW in CreationFlags — the same flag sysproc.HideConsole
	// OR-s in for SDK-spawned children — because it suppresses console
	// allocation entirely. SysProcAttr.HideWindow would not do: it only adds
	// STARTF_USESHOWWINDOW + SW_HIDE to the startup info, so the console is
	// still created and can flash before it is hidden. The constant is
	// inlined because safeio is a leaf package and does not import sysproc
	// for a test.
	cmd.SysProcAttr = &syscall.SysProcAttr{CreationFlags: createNoWindow}
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("plant directory junction: %v (output: %s)", err, out)
	}
}

// TestOpenFileNoFollow_RefusesJunction pins the reparse branch the symlink
// tests above cannot reach: Go maps IO_REPARSE_TAG_MOUNT_POINT to
// fs.ModeIrregular (not fs.ModeSymlink), and OpenFileNoFollow refuses both
// with syscall.ELOOP. A junction is a directory reparse point, and a read open
// is the shape that gets a handle at all: syscall.Open adds
// FILE_FLAG_BACKUP_SEMANTICS for a read access mode, which lets CreateFile
// open the junction itself rather than failing on a directory. That is also
// why this test needs no elevation, unlike the file-symlink tests above.
func TestOpenFileNoFollow_RefusesJunction(t *testing.T) {
	dir := t.TempDir()
	target := filepath.Join(dir, "target")
	link := filepath.Join(dir, "junction")
	requireJunction(t, target, link)

	f, err := OpenFileNoFollow(link, os.O_RDONLY, 0)
	if err == nil {
		_ = f.Close()
		t.Fatal("OpenFileNoFollow opened through a directory junction, want an ELOOP refusal")
	}
	if !errors.Is(err, syscall.ELOOP) {
		t.Fatalf("OpenFileNoFollow err = %v, want a *fs.PathError wrapping syscall.ELOOP", err)
	}
	if _, lerr := os.Lstat(link); lerr != nil {
		t.Fatalf("junction itself must survive the refusal: %v", lerr)
	}
}

// TestWriteFile_RefusesJunction pins the write shape: a junction is never
// written through. syscall.Open adds FILE_FLAG_BACKUP_SEMANTICS only for a
// read access mode, so a write open of a directory junction is refused by
// CreateFile before OpenFileNoFollow's reparse check is reached, and the
// stdlib maps that ERROR_ACCESS_DENIED to EISDIR for a target carrying
// FILE_ATTRIBUTE_DIRECTORY. Whichever refusal fires (EISDIR here, ELOOP from
// the reparse branch), the junction's target stays untouched.
func TestWriteFile_RefusesJunction(t *testing.T) {
	dir := t.TempDir()
	target := filepath.Join(dir, "target")
	link := filepath.Join(dir, "junction")
	requireJunction(t, target, link)
	keep := filepath.Join(target, "keep")
	if err := os.WriteFile(keep, []byte("original"), 0o644); err != nil {
		t.Fatalf("seed junction target content: %v", err)
	}

	err := WriteFile(link, []byte("payload"), 0o644)
	if err == nil {
		t.Fatal("WriteFile wrote through a directory junction, want a refusal")
	}
	if !errors.Is(err, syscall.EISDIR) && !errors.Is(err, syscall.ELOOP) {
		t.Fatalf("WriteFile err = %v, want a refusal wrapping syscall.EISDIR (directory handle) or syscall.ELOOP (reparse check)", err)
	}
	data, rerr := os.ReadFile(keep)
	if rerr != nil || string(data) != "original" {
		t.Fatalf("junction target was disturbed: %q, %v", data, rerr)
	}
}
