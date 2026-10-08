package builtins

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/v0lka/sp4rk/tools"
)

// The tests in this file pin the POSIX rm semantics of the destructive tools
// (review finding #2): delete_file and delete_directory remove the SYMLINK
// itself, never the target it points to — regardless of where the target
// lives. The judge variant evaluates the link path, so an in-root link is
// auto-approved and an out-of-root link escalates; either way the confirmed
// or auto-approved unlink can only ever remove the link.

// mustSymlink creates newname → oldname, skipping the test when the
// filesystem does not support symlinks.
func mustSymlink(t *testing.T, oldname, newname string) {
	t.Helper()
	if err := os.Symlink(oldname, newname); err != nil {
		t.Skipf("symlinks unsupported on this filesystem: %v", err)
	}
}

func TestDeleteFileTool_SymlinkInRoot_RemovesLinkNotTarget(t *testing.T) {
	dir := t.TempDir()
	realFile := filepath.Join(dir, "realFile.txt")
	if err := os.WriteFile(realFile, []byte("keep me"), 0o644); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(dir, "link")
	mustSymlink(t, realFile, link)

	// No workspace in context: the absolute link path is used as spelled.
	input, err := json.Marshal(DeleteFileInput{Path: link})
	if err != nil {
		t.Fatal(err)
	}
	res, err := NewDeleteFileTool().Execute(context.Background(), input)
	if err != nil {
		t.Fatalf("unexpected transport error: %v", err)
	}
	if res.IsError {
		t.Fatalf("delete_file on an in-root symlink failed: %s", res.Content)
	}

	if _, err := os.Lstat(link); !os.IsNotExist(err) {
		t.Errorf("expected the link to be removed, Lstat err = %v", err)
	}
	data, err := os.ReadFile(realFile)
	if err != nil {
		t.Fatalf("the symlink target must survive, got read error: %v", err)
	}
	if string(data) != "keep me" {
		t.Errorf("target content = %q, want %q", data, "keep me")
	}
}

func TestDeleteFileTool_SymlinkOutsideRoot_RemovesLinkOnly(t *testing.T) {
	dir := t.TempDir()
	outside := t.TempDir()
	outTarget := filepath.Join(outside, "victim.txt")
	if err := os.WriteFile(outTarget, []byte("out"), 0o644); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(dir, "link")
	mustSymlink(t, outTarget, link)

	// Execute contract (documented deviation choice): even for a link whose
	// target leaves the session roots, the unlink removes just the link —
	// the judge escalates the out-of-root scope, but a confirmed call can
	// never destroy the out-of-root target through the link.
	input, err := json.Marshal(DeleteFileInput{Path: link})
	if err != nil {
		t.Fatal(err)
	}
	res, err := NewDeleteFileTool().Execute(context.Background(), input)
	if err != nil {
		t.Fatalf("unexpected transport error: %v", err)
	}
	if res.IsError {
		t.Fatalf("delete_file on an out-of-root symlink failed: %s", res.Content)
	}
	if _, err := os.Lstat(link); !os.IsNotExist(err) {
		t.Errorf("expected the link to be removed, Lstat err = %v", err)
	}
	if data, err := os.ReadFile(outTarget); err != nil || string(data) != "out" {
		t.Errorf("out-of-root target must survive, got data=%q err=%v", data, err)
	}
}

func TestDeleteFileTool_RelativeSymlinkInWorkspace_RemovesLinkNotTarget(t *testing.T) {
	ws := t.TempDir()
	realFile := filepath.Join(ws, "realFile.txt")
	if err := os.WriteFile(realFile, []byte("keep"), 0o644); err != nil {
		t.Fatal(err)
	}
	mustSymlink(t, "realFile.txt", filepath.Join(ws, "link"))

	ctx := tools.WithWorkspacePath(context.Background(), ws)
	input, err := json.Marshal(DeleteFileInput{Path: "link"})
	if err != nil {
		t.Fatal(err)
	}
	res, err := NewDeleteFileTool().Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected transport error: %v", err)
	}
	if res.IsError {
		t.Fatalf("relative in-workspace symlink delete failed: %s", res.Content)
	}
	if _, err := os.Lstat(filepath.Join(ws, "link")); !os.IsNotExist(err) {
		t.Errorf("expected the link to be removed, Lstat err = %v", err)
	}
	if _, err := os.Stat(realFile); err != nil {
		t.Errorf("target must survive: %v", err)
	}
}

func TestDeleteFileTool_RelativeSymlinkOutsideWorkspace_Refused(t *testing.T) {
	ws := t.TempDir()
	outside := t.TempDir()
	outTarget := filepath.Join(outside, "victim.txt")
	if err := os.WriteFile(outTarget, []byte("out"), 0o644); err != nil {
		t.Fatal(err)
	}
	// Relative symlink whose target leaves the workspace: resolvePathUnresolved
	// refuses it outright (fail-closed) — the link survives.
	mustSymlink(t, outTarget, filepath.Join(ws, "escape"))

	ctx := tools.WithWorkspacePath(context.Background(), ws)
	input, err := json.Marshal(DeleteFileInput{Path: "escape"})
	if err != nil {
		t.Fatal(err)
	}
	res, err := NewDeleteFileTool().Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected transport error: %v", err)
	}
	if !res.IsError {
		t.Fatalf("expected refusal for a relative symlink escaping the workspace, got: %s", res.Content)
	}
	if !strings.Contains(res.Content, "outside the session workspace") {
		t.Errorf("expected an outside-the-workspace refusal, got: %s", res.Content)
	}
	if _, err := os.Lstat(filepath.Join(ws, "escape")); err != nil {
		t.Errorf("the refused link must survive: %v", err)
	}
	if _, err := os.Stat(outTarget); err != nil {
		t.Errorf("the out-of-root target must survive: %v", err)
	}
}

func TestDeleteDirectoryTool_SymlinkRecursive_RemovesLinkOnly(t *testing.T) {
	dir := t.TempDir()
	realDir := filepath.Join(dir, "realdir")
	if err := os.MkdirAll(realDir, 0o755); err != nil {
		t.Fatal(err)
	}
	inner := filepath.Join(realDir, "inner.txt")
	if err := os.WriteFile(inner, []byte("tree"), 0o644); err != nil {
		t.Fatal(err)
	}
	linkDir := filepath.Join(dir, "linkdir")
	mustSymlink(t, realDir, linkDir)

	input, err := json.Marshal(DeleteDirectoryInput{Path: linkDir, Recursive: true})
	if err != nil {
		t.Fatal(err)
	}
	res, err := NewDeleteDirectoryTool().Execute(context.Background(), input)
	if err != nil {
		t.Fatalf("unexpected transport error: %v", err)
	}
	if res.IsError {
		t.Fatalf("recursive delete_directory on an in-root symlink failed: %s", res.Content)
	}

	if _, err := os.Lstat(linkDir); !os.IsNotExist(err) {
		t.Errorf("expected the link to be removed, Lstat err = %v", err)
	}
	if _, err := os.Stat(inner); err != nil {
		t.Errorf("the target tree must survive untouched: %v", err)
	}
}

func TestDeleteDirectoryTool_SymlinkOutsideRoot_RemovesLinkOnly(t *testing.T) {
	dir := t.TempDir()
	outTree := filepath.Join(t.TempDir(), "tree")
	if err := os.MkdirAll(outTree, 0o755); err != nil {
		t.Fatal(err)
	}
	inner := filepath.Join(outTree, "inner.txt")
	if err := os.WriteFile(inner, []byte("tree"), 0o644); err != nil {
		t.Fatal(err)
	}
	linkDir := filepath.Join(dir, "linkdir")
	mustSymlink(t, outTree, linkDir)

	input, err := json.Marshal(DeleteDirectoryInput{Path: linkDir, Recursive: true})
	if err != nil {
		t.Fatal(err)
	}
	res, err := NewDeleteDirectoryTool().Execute(context.Background(), input)
	if err != nil {
		t.Fatalf("unexpected transport error: %v", err)
	}
	if res.IsError {
		t.Fatalf("recursive delete_directory on an out-of-root symlink failed: %s", res.Content)
	}
	if _, err := os.Lstat(linkDir); !os.IsNotExist(err) {
		t.Errorf("expected the link to be removed, Lstat err = %v", err)
	}
	if _, err := os.Stat(inner); err != nil {
		t.Errorf("the out-of-root tree must survive untouched: %v", err)
	}
}

func TestDeleteFileTool_Judge_SymlinkContainment(t *testing.T) {
	ws := t.TempDir()
	outside := t.TempDir()

	cases := []struct {
		name      string
		target    string // symlink target (relative to its own dir when relative)
		linkIn    string // where the link lives
		dangling  bool   // leave the target non-existent so the link dangles
		wantAllow bool
		wantCode  tools.JudgeReasonCode
	}{
		{
			name:      "in-root link to in-root file is allowed",
			target:    filepath.Join(ws, "real.txt"),
			linkIn:    ws,
			wantAllow: true,
		},
		{
			name:      "in-root link to out-of-root target escalates outside roots",
			target:    filepath.Join(outside, "victim.txt"),
			linkIn:    ws,
			wantAllow: false,
			wantCode:  tools.ReasonCodeOutsideSessionRoots,
		},
		{
			// A DANGLING link has no resolvable target: it is judged by its
			// longest existing prefix (its parent), so an in-root dangling
			// link auto-approves — the unlink can only remove the link.
			name:      "in-root dangling link is allowed",
			target:    filepath.Join(outside, "does-not-exist.txt"),
			linkIn:    ws,
			dangling:  true,
			wantAllow: true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// The out-of-root target must exist so the link resolves THROUGH
			// it and the containment check sees the outside destination (a
			// dangling link is judged by its parent instead).
			if !tc.dangling {
				if err := os.WriteFile(tc.target, []byte("x"), 0o644); err != nil {
					t.Fatal(err)
				}
			}

			link := filepath.Join(tc.linkIn, "judge-link")
			mustSymlink(t, tc.target, link)
			t.Cleanup(func() { _ = os.Remove(link) })

			ctx := tools.WithWorkspacePath(context.Background(), ws)
			input, _ := json.Marshal(DeleteFileInput{Path: link})
			outcome := NewDeleteFileTool().Judge(ctx, input)

			if outcome.Allow != tc.wantAllow {
				t.Fatalf("Judge allow = %v, want %v (reason: %s)", outcome.Allow, tc.wantAllow, outcome.Reason)
			}
			if outcome.ReasonCode != tc.wantCode {
				t.Errorf("Judge reason code = %q, want %q", outcome.ReasonCode, tc.wantCode)
			}
		})
	}
}

// TestDeleteTools_AbsoluteInWorkspaceSymlink_WithWorkspaceCtx pins the
// finding's exact repro shape: an ABSOLUTE in-workspace symlink resolved
// under a workspace-attached context (the session-roots judge is active)
// must remove the link and leave the target intact — for both destructive
// tools. The absolute/no-workspace tests above also pass on the pre-fix
// code (resolvePath returned absolute paths as spelled there); this test
// fails on it, because the pre-fix resolvePath resolved the link and
// os.Remove destroyed the target.
func TestDeleteTools_AbsoluteInWorkspaceSymlink_WithWorkspaceCtx(t *testing.T) {
	t.Run("delete_file", func(t *testing.T) {
		ws := t.TempDir()
		realFile := filepath.Join(ws, "realFile.txt")
		if err := os.WriteFile(realFile, []byte("keep"), 0o644); err != nil {
			t.Fatal(err)
		}
		link := filepath.Join(ws, "link")
		mustSymlink(t, "realFile.txt", link)

		ctx := tools.WithWorkspacePath(context.Background(), ws)
		input, err := json.Marshal(DeleteFileInput{Path: link})
		if err != nil {
			t.Fatal(err)
		}
		res, err := NewDeleteFileTool().Execute(ctx, input)
		if err != nil {
			t.Fatalf("unexpected transport error: %v", err)
		}
		if res.IsError {
			t.Fatalf("delete_file on an in-workspace symlink failed: %s", res.Content)
		}
		if _, err := os.Lstat(link); !os.IsNotExist(err) {
			t.Errorf("expected the link to be removed, Lstat err = %v", err)
		}
		if data, err := os.ReadFile(realFile); err != nil || string(data) != "keep" {
			t.Errorf("target must survive: data=%q err=%v", data, err)
		}
	})

	t.Run("delete_directory", func(t *testing.T) {
		ws := t.TempDir()
		realDir := filepath.Join(ws, "realDir")
		if err := os.MkdirAll(filepath.Join(realDir, "nested"), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(realDir, "nested", "f.txt"), []byte("keep"), 0o644); err != nil {
			t.Fatal(err)
		}
		linkDir := filepath.Join(ws, "linkDir")
		mustSymlink(t, "realDir", linkDir)

		ctx := tools.WithWorkspacePath(context.Background(), ws)
		input, err := json.Marshal(DeleteDirectoryInput{Path: linkDir})
		if err != nil {
			t.Fatal(err)
		}
		res, err := NewDeleteDirectoryTool().Execute(ctx, input)
		if err != nil {
			t.Fatalf("unexpected transport error: %v", err)
		}
		if res.IsError {
			t.Fatalf("delete_directory on an in-workspace symlink failed: %s", res.Content)
		}
		if _, err := os.Lstat(linkDir); !os.IsNotExist(err) {
			t.Errorf("expected the link to be removed, Lstat err = %v", err)
		}
		if _, err := os.Stat(filepath.Join(realDir, "nested", "f.txt")); err != nil {
			t.Errorf("target tree must survive: %v", err)
		}
	})
}
