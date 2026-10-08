package builtins

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/v0lka/sp4rk/pathutil"
	"github.com/v0lka/sp4rk/tools"
)

// The tests in this file pin OS component-order resolution of "symlink/.."
// spellings. The OS resolves a path one component at a time: a symlink is
// expanded when encountered, and a following ".." climbs from the link
// TARGET's directory. filepath.Clean and filepath.Join evaluate ".."
// lexically, erasing the link before any symlink resolution runs — so the
// spelling "link/../report.txt" (link → outside the workspace) was silently
// resolved to the unrelated in-workspace report.txt, and delete_file
// destroyed that file instead of refusing the out-of-workspace target the
// spelling actually addresses.
//
// Every ambiguous spelling below is therefore built by RAW concatenation:
// filepath.Join in the test itself would erase "link/.." before the resolver
// ever sees it (the exact bug class under test).

const (
	dotDotFileSpelling      = "link/../report.txt"
	dotDotSubSpelling       = "link/../sub"
	realDotDotSpelling      = "realdir/../file.txt"
	innerLinkDotDotSpelling = "sub/link/../f.txt"
)

// dotDotFixture holds the shared layout for the symlink/../ tests:
//
//	ws/link               → <outsideParent>/out   (intermediate symlink leaving the workspace)
//	ws/report.txt         = "keep"                (the file the lexical bug deleted)
//	<outsideParent>/out   (directory the link points at)
//	<outsideParent>/report.txt = "out"            (the file POSIX resolution addresses)
type dotDotFixture struct {
	ws            string
	outsideParent string
	outside       string
	link          string
	keepFile      string
	posixTarget   string
}

// newDotDotFixture builds the fixture, skipping the test when the filesystem
// does not support symlinks.
func newDotDotFixture(t *testing.T) dotDotFixture {
	t.Helper()
	ws := t.TempDir()
	outsideParent := t.TempDir()
	outside := filepath.Join(outsideParent, "out")
	if err := os.MkdirAll(outside, 0o755); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(ws, "link")
	mustSymlink(t, outside, link)
	keepFile := filepath.Join(ws, "report.txt")
	if err := os.WriteFile(keepFile, []byte("keep"), 0o644); err != nil {
		t.Fatal(err)
	}
	posixTarget := filepath.Join(outsideParent, "report.txt")
	if err := os.WriteFile(posixTarget, []byte("out"), 0o644); err != nil {
		t.Fatal(err)
	}
	return dotDotFixture{
		ws:            ws,
		outsideParent: outsideParent,
		outside:       outside,
		link:          link,
		keepFile:      keepFile,
		posixTarget:   posixTarget,
	}
}

// rawAbsSpelling prefixes a raw relative spelling with an absolute base
// without any Clean.
func rawAbsSpelling(base, rel string) string {
	return base + string(filepath.Separator) + rel
}

func TestResolvePathUnresolved_SymlinkDotDot(t *testing.T) {
	f := newDotDotFixture(t)
	ctx := tools.WithWorkspacePath(context.Background(), f.ws)

	t.Run("relative spelling is refused — POSIX resolution leaves the workspace", func(t *testing.T) {
		got := resolvePathUnresolved(ctx, dotDotFileSpelling)
		if got != "" {
			t.Fatalf("resolvePathUnresolved(%q) = %q, want \"\" (refused)", dotDotFileSpelling, got)
		}
	})

	t.Run("absolute spelling addresses the link target's parent, not the lexical parent", func(t *testing.T) {
		got := resolvePathUnresolved(ctx, rawAbsSpelling(f.ws, dotDotFileSpelling))
		want := filepath.Join(pathutil.ResolveExistingPrefix(f.outsideParent), "report.txt")
		if got != want {
			t.Fatalf("resolvePathUnresolved = %q, want %q", got, want)
		}
	})
}

func TestResolvePath_SymlinkDotDot(t *testing.T) {
	f := newDotDotFixture(t)
	ctx := tools.WithWorkspacePath(context.Background(), f.ws)

	// resolvePath (the read/write/edit resolver) shares the contract: a
	// relative spelling that POSIX-resolves outside the workspace is refused,
	// not silently rewritten to a same-named file inside it.
	got := resolvePath(ctx, dotDotFileSpelling)
	if got != "" {
		t.Fatalf("resolvePath(%q) = %q, want \"\" (refused)", dotDotFileSpelling, got)
	}
}

func TestResolvePath_RealDirDotDot_StillResolves(t *testing.T) {
	// Parity guard: ".." climbing over a PLAIN directory keeps working —
	// only the symlink/../ sequence changes meaning (to the OS-correct one).
	ws := t.TempDir()
	if err := os.MkdirAll(filepath.Join(ws, "realdir"), 0o755); err != nil {
		t.Fatal(err)
	}
	file := filepath.Join(ws, "file.txt")
	if err := os.WriteFile(file, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	ctx := tools.WithWorkspacePath(context.Background(), ws)

	got := resolvePath(ctx, realDotDotSpelling)
	if want := pathutil.ResolveExistingPrefix(file); got != want {
		t.Fatalf("resolvePath(%q) = %q, want %q", realDotDotSpelling, got, want)
	}
}

func TestResolvePath_IntermediateSymlinkInsideWorkspace_DotDotClimbsFromTarget(t *testing.T) {
	// ws/sub/link → ws/other: "sub/link/../f.txt" must POSIX-resolve to
	// ws/f.txt (climb from the link TARGET), not to the lexically cleaned
	// ws/sub/f.txt.
	ws := t.TempDir()
	if err := os.MkdirAll(filepath.Join(ws, "sub"), 0o755); err != nil {
		t.Fatal(err)
	}
	other := filepath.Join(ws, "other")
	if err := os.MkdirAll(other, 0o755); err != nil {
		t.Fatal(err)
	}
	mustSymlink(t, other, filepath.Join(ws, "sub", "link"))
	file := filepath.Join(ws, "f.txt")
	if err := os.WriteFile(file, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	ctx := tools.WithWorkspacePath(context.Background(), ws)

	got := resolvePath(ctx, innerLinkDotDotSpelling)
	if want := pathutil.ResolveExistingPrefix(file); got != want {
		t.Fatalf("resolvePath(%q) = %q, want %q", innerLinkDotDotSpelling, got, want)
	}
}

func TestDeleteFileTool_RelativeSymlinkDotDot_Refused(t *testing.T) {
	f := newDotDotFixture(t)
	ctx := tools.WithWorkspacePath(context.Background(), f.ws)

	input, err := json.Marshal(DeleteFileInput{Path: dotDotFileSpelling})
	if err != nil {
		t.Fatal(err)
	}
	res, err := NewDeleteFileTool().Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected transport error: %v", err)
	}
	if !res.IsError {
		t.Fatalf("expected refusal for %q, got success: %s", dotDotFileSpelling, res.Content)
	}
	if !strings.Contains(res.Content, "outside the session workspace") {
		t.Errorf("expected an outside-the-workspace refusal, got: %s", res.Content)
	}
	if data, err := os.ReadFile(f.keepFile); err != nil || string(data) != "keep" {
		t.Errorf("the unrelated in-workspace file must survive: data=%q err=%v", data, err)
	}
	if data, err := os.ReadFile(f.posixTarget); err != nil || string(data) != "out" {
		t.Errorf("the POSIX-addressed outside file must survive: data=%q err=%v", data, err)
	}
	if _, err := os.Lstat(f.link); err != nil {
		t.Errorf("the link must survive: %v", err)
	}
}

func TestDeleteFileTool_AbsoluteSymlinkDotDot_AddressesPosixTarget(t *testing.T) {
	f := newDotDotFixture(t)
	ctx := tools.WithWorkspacePath(context.Background(), f.ws)

	// Absolute paths bypass the containment refusal (access control stays
	// with the Judge/confirmation flow), so Execute acts on the
	// POSIX-addressed target — the file inside the link target's parent —
	// and never on the same-named in-workspace file the lexical cleaning
	// used to substitute.
	input, err := json.Marshal(DeleteFileInput{Path: rawAbsSpelling(f.ws, dotDotFileSpelling)})
	if err != nil {
		t.Fatal(err)
	}
	res, err := NewDeleteFileTool().Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected transport error: %v", err)
	}
	if res.IsError {
		t.Fatalf("absolute symlink/../ delete failed: %s", res.Content)
	}
	if _, err := os.Lstat(f.posixTarget); !os.IsNotExist(err) {
		t.Errorf("expected the POSIX-addressed outside file to be removed, Lstat err = %v", err)
	}
	if data, err := os.ReadFile(f.keepFile); err != nil || string(data) != "keep" {
		t.Errorf("the unrelated in-workspace file must survive: data=%q err=%v", data, err)
	}
	if _, err := os.Lstat(f.link); err != nil {
		t.Errorf("the link must survive: %v", err)
	}
}

func TestDeleteFileTool_Judge_RelativeSymlinkDotDot_EscalatesOutsideRoots(t *testing.T) {
	f := newDotDotFixture(t)
	ctx := tools.WithWorkspacePath(context.Background(), f.ws)

	input, err := json.Marshal(DeleteFileInput{Path: dotDotFileSpelling})
	if err != nil {
		t.Fatal(err)
	}
	outcome := NewDeleteFileTool().Judge(ctx, input)
	if outcome.Allow {
		t.Fatalf("Judge allowed %q (reason: %s), want an outside-session-roots escalation", dotDotFileSpelling, outcome.Reason)
	}
	if outcome.ReasonCode != tools.ReasonCodeOutsideSessionRoots {
		t.Errorf("Judge reason code = %q, want %q", outcome.ReasonCode, tools.ReasonCodeOutsideSessionRoots)
	}
	if data, err := os.ReadFile(f.keepFile); err != nil || string(data) != "keep" {
		t.Errorf("the unrelated in-workspace file must survive: data=%q err=%v", data, err)
	}
}

func TestDeleteDirectoryTool_RelativeSymlinkDotDot_Refused(t *testing.T) {
	ws := t.TempDir()
	outsideParent := t.TempDir()
	if err := os.MkdirAll(filepath.Join(outsideParent, "out"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(outsideParent, "sub"), 0o755); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(ws, "link")
	mustSymlink(t, filepath.Join(outsideParent, "out"), link)
	wsSub := filepath.Join(ws, "sub")
	if err := os.MkdirAll(filepath.Join(wsSub, "nested"), 0o755); err != nil {
		t.Fatal(err)
	}
	keepFile := filepath.Join(wsSub, "nested", "f.txt")
	if err := os.WriteFile(keepFile, []byte("keep"), 0o644); err != nil {
		t.Fatal(err)
	}

	// "link/../sub" POSIX-resolves to <outsideParent>/sub — outside the
	// workspace — so it must be refused, not lexically collapsed to the
	// unrelated in-workspace ws/sub and deleted.
	ctx := tools.WithWorkspacePath(context.Background(), ws)
	input, err := json.Marshal(DeleteDirectoryInput{Path: dotDotSubSpelling, Recursive: true})
	if err != nil {
		t.Fatal(err)
	}
	res, err := NewDeleteDirectoryTool().Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected transport error: %v", err)
	}
	if !res.IsError {
		t.Fatalf("expected refusal for %q, got success: %s", dotDotSubSpelling, res.Content)
	}
	if !strings.Contains(res.Content, "outside the session workspace") {
		t.Errorf("expected an outside-the-workspace refusal, got: %s", res.Content)
	}
	if _, err := os.Stat(keepFile); err != nil {
		t.Errorf("the unrelated in-workspace tree must survive: %v", err)
	}
	if _, err := os.Stat(filepath.Join(outsideParent, "sub")); err != nil {
		t.Errorf("the POSIX-addressed outside directory must survive: %v", err)
	}
	if _, err := os.Lstat(link); err != nil {
		t.Errorf("the link must survive: %v", err)
	}
}
