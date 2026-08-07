// SPDX-License-Identifier: Apache-2.0

package tools

import (
	"context"
	"path"
	"path/filepath"
	"strings"
	"testing"
)

// TestPathsOutsideRoots_EmptyRootsReturnsNil verifies the no-roots contract:
// with no workspace/temp/allowed roots configured, containment cannot be
// enforced and PathsOutsideRoots returns nil (mirroring workdir.go).
func TestPathsOutsideRoots_EmptyRootsReturnsNil(t *testing.T) {
	got := PathsOutsideRoots(context.Background(), "cat /etc/passwd", ShellBash, "")
	if got != nil {
		t.Fatalf("expected nil when SessionRoots is empty, got %v", got)
	}
}

// TestPathsOutsideRoots_AbsoluteOutsideReported verifies an absolute path
// outside the workspace root is reported.
func TestPathsOutsideRoots_AbsoluteOutsideReported(t *testing.T) {
	ws := t.TempDir()
	ctx := WithWorkspacePath(context.Background(), ws)
	got := PathsOutsideRoots(ctx, "cat /etc/passwd", ShellBash, ws)
	if !sliceContains(got, "/etc/passwd") {
		t.Fatalf("expected /etc/passwd reported as outside roots, got %v", got)
	}
}

// TestPathsOutsideRoots_AbsoluteInsideNotReported verifies an absolute path
// inside the workspace root is NOT reported.
func TestPathsOutsideRoots_AbsoluteInsideNotReported(t *testing.T) {
	ws := t.TempDir()
	ctx := WithWorkspacePath(context.Background(), ws)
	inside := filepath.Join(ws, "data.txt")
	got := PathsOutsideRoots(ctx, "cat "+inside, ShellBash, ws)
	if len(got) != 0 {
		t.Fatalf("expected no outside paths for an in-root file, got %v", got)
	}
}

// TestPathsOutsideRoots_TildeReported verifies "~/.ssh/id_rsa" resolves to the
// home directory and is reported when home is outside the roots.
func TestPathsOutsideRoots_TildeReported(t *testing.T) {
	ws := t.TempDir()
	fakeHome := t.TempDir()
	t.Setenv("HOME", fakeHome)

	ctx := WithWorkspacePath(context.Background(), ws)
	got := PathsOutsideRoots(ctx, "cat ~/.ssh/id_rsa", ShellBash, ws)

	want := filepath.Clean(filepath.Join(fakeHome, ".ssh", "id_rsa"))
	if !sliceContains(got, want) {
		t.Fatalf("expected %q reported (home outside roots), got %v", want, got)
	}
}

// TestPathsOutsideRoots_BashEnvExpand verifies "$HOME/.config" expands via
// os.Getenv and is reported when outside the roots.
func TestPathsOutsideRoots_BashEnvExpand(t *testing.T) {
	ws := t.TempDir()
	fakeHome := t.TempDir()
	t.Setenv("HOME", fakeHome)

	ctx := WithWorkspacePath(context.Background(), ws)
	got := PathsOutsideRoots(ctx, "ls $HOME/.config", ShellBash, ws)

	want := filepath.Clean(filepath.Join(fakeHome, ".config"))
	if !sliceContains(got, want) {
		t.Fatalf("expected %q reported, got %v", want, got)
	}
}

// TestPathsOutsideRoots_PoshEnvExpand verifies "$env:USERPROFILE\.ssh" expands
// under the PowerShell dialect and is reported when outside the roots.
func TestPathsOutsideRoots_PoshEnvExpand(t *testing.T) {
	ws := t.TempDir()
	fakeProfile := t.TempDir()
	t.Setenv("USERPROFILE", fakeProfile)

	ctx := WithWorkspacePath(context.Background(), ws)
	got := PathsOutsideRoots(ctx, `Get-Content $env:USERPROFILE\.ssh\config`, ShellPosh, ws)

	want := filepath.Clean(filepath.Join(fakeProfile, ".ssh", "config"))
	if !sliceContains(got, want) {
		t.Fatalf("expected %q reported, got %v", want, got)
	}
}

// TestResolveShellPathTokens_PoshEnvUserprofile is the cross-platform testable
// surrogate for the platform-gated PoshExecTool.Judge containment check
// (posh.go is //go:build windows, so its Judge test never runs on linux/macos
// in CI). It directly verifies that the "$env:USERPROFILE" idiom — the one the
// posh Judge expands via PathsOutsideRoots(ctx, cmd, ShellPosh, workDir) —
// resolves to the configured profile directory. It is environment-driven
// (t.Setenv) so it is deterministic on every platform.
func TestResolveShellPathTokens_PoshEnvUserprofile(t *testing.T) {
	fakeProfile := t.TempDir()
	t.Setenv("USERPROFILE", fakeProfile)

	got := ResolveShellPathTokens(`Get-Content $env:USERPROFILE\.ssh\config`, ShellPosh, "")

	want := filepath.Clean(filepath.Join(fakeProfile, ".ssh", "config"))
	if !sliceContains(got, want) {
		t.Fatalf("expected %q resolved from $env:USERPROFILE, got %v", want, got)
	}
}

// TestResolveShellPathTokens_UnsetEnvVarSkipped verifies that a reference to
// an unset environment variable is skipped (not reported, no crash) rather
// than emitted as a literal or empty path.
func TestResolveShellPathTokens_UnsetEnvVarSkipped(t *testing.T) {
	t.Setenv("NOPE_SHELLPATHS_UNDEF", "")
	t.Setenv("NOPE", "")

	got := ResolveShellPathTokens("cat $NOPE/foo/bar", ShellBash, "/tmp/ws")
	for _, p := range got {
		if p == "$NOPE/foo/bar" || filepath.Base(p) == "foo" || filepath.Base(p) == "bar" {
			t.Fatalf("expected unset $NOPE reference to be skipped, got %v", got)
		}
	}
}

// TestPathsOutsideRoots_RelativeDotDotEscapes verifies a relative ".." token is
// resolved against workDir and reported when it escapes the roots, while a
// plain relative name ("foo.txt") is never reported.
func TestPathsOutsideRoots_RelativeDotDotEscapes(t *testing.T) {
	ws := t.TempDir()
	ctx := WithWorkspacePath(context.Background(), ws)
	workDir := ws // base inside the root so the ".." can be observed escaping it

	// "../sibling" from ws resolves to the parent of ws, which is outside ws.
	got := PathsOutsideRoots(ctx, "cat ../sibling/secret foo.txt", ShellBash, workDir)

	want := filepath.Clean(filepath.Join(workDir, "..", "sibling", "secret"))
	if !sliceContains(got, want) {
		t.Fatalf("expected escaped relative path %q reported, got %v", want, got)
	}
	for _, p := range got {
		if filepath.Base(p) == "foo.txt" {
			t.Fatalf("plain relative name must not be reported, got %v", got)
		}
	}
}

// TestPathsOutsideRoots_RelativeDotDotEscapesRootBoundary drives the
// acceptance criterion that "../../etc" escapes roots and is reported.
func TestPathsOutsideRoots_RelativeDotDotEscapesRootBoundary(t *testing.T) {
	ws := t.TempDir()
	ctx := WithWorkspacePath(context.Background(), ws)
	// workDir is the root itself; two ".." steps escape it to the parent area.
	workDir := ws

	got := PathsOutsideRoots(ctx, "cat ../../etc/passwd", ShellBash, workDir)

	want := filepath.Clean(filepath.Join(workDir, "..", "..", "etc", "passwd"))
	if !sliceContains(got, want) {
		t.Fatalf("expected %q reported, got %v", want, got)
	}
}

// TestPathsOutsideRoots_RelativeDotDotEmptyWorkDirFallsBackToWorkspace
// verifies the SHOULD-FIX regression: when working_directory is omitted (the
// common case), the Judge must resolve relative ".." tokens against the
// workspace root — the same base Execute uses (cmd.Dir = workspace). Without
// the fallback, "cat ../../etc/passwd" with no working_directory would bypass
// containment under an always-allow policy while Execute reads the file.
func TestPathsOutsideRoots_RelativeDotDotEmptyWorkDirFallsBackToWorkspace(t *testing.T) {
	ws := t.TempDir()
	ctx := WithWorkspacePath(context.Background(), ws)

	// Empty workDir — must fall back to the workspace path from context.
	got := PathsOutsideRoots(ctx, "cat ../../etc/passwd", ShellBash, "")

	want := filepath.Clean(filepath.Join(ws, "..", "..", "etc", "passwd"))
	if !sliceContains(got, want) {
		t.Fatalf("expected %q reported with empty workDir (workspace fallback), got %v", want, got)
	}
}

// TestPathsOutsideRoots_AuxiliaryRootNotReported verifies a path inside an
// additional allowed root (auxiliary working directory) is NOT reported.
func TestPathsOutsideRoots_AuxiliaryRootNotReported(t *testing.T) {
	ws := t.TempDir()
	auxRoot := t.TempDir()

	ctx := WithWorkspacePath(context.Background(), ws)
	ctx = WithAllowedRoots(ctx, []string{auxRoot})

	insideAux := filepath.Join(auxRoot, "build", "out.json")
	got := PathsOutsideRoots(ctx, "cat "+insideAux, ShellBash, ws)
	if sliceContains(got, insideAux) {
		t.Fatalf("path inside an allowed root must not be reported, got %v", got)
	}
}

// TestResolveShellPathTokens_URLSchemeSkipped verifies that URL scheme tails
// (e.g. "s://" extracted from "https://host/path" by pathRegex's drive-letter
// alternative) are NOT treated as filesystem paths. Without this skip, a
// command like "git clone https://github.com/org/repo" would be falsely
// flagged as referencing a Windows drive path on Windows hosts.
func TestResolveShellPathTokens_URLSchemeSkipped(t *testing.T) {
	ws := t.TempDir()
	ctx := WithWorkspacePath(context.Background(), ws)

	for _, cmd := range []string{
		"git clone https://github.com/org/repo",
		"curl ftp://example.com/file",
		"wget http://localhost:8080/api",
	} {
		got := PathsOutsideRoots(ctx, cmd, ShellBash, ws)
		for _, p := range got {
			if strings.HasSuffix(p, "://github.com/org/repo") ||
				strings.HasSuffix(p, "://example.com/file") ||
				strings.HasSuffix(p, "://localhost:8080/api") {
				t.Fatalf("URL scheme fragment must not be reported as a path; cmd=%q got=%v", cmd, got)
			}
		}
	}
}

// TestResolveShellPathTokens_DedupeAndClean exercises deduplication and
// filepath.Clean normalization across absolute, env and ".." tokens.
func TestResolveShellPathTokens_DedupeAndClean(t *testing.T) {
	t.Setenv("SHLP_HOME", "/opt/expanded")
	got := ResolveShellPathTokens(
		"cp /var/log/syslog /var/log/syslog $SHLP_HOME/x/.. ", ShellBash, "")

	wantSyslog := "/var/log/syslog"
	wantExpanded := path.Clean("/opt/expanded/x/..") // == /opt/expanded
	if !sliceContains(got, wantSyslog) {
		t.Fatalf("expected %q in resolved tokens, got %v", wantSyslog, got)
	}
	if !sliceContains(got, wantExpanded) {
		t.Fatalf("expected %q in resolved tokens, got %v", wantExpanded, got)
	}
	// Dedup: /var/log/syslog appears twice in the command but once in output.
	count := 0
	for _, p := range got {
		if p == wantSyslog {
			count++
		}
	}
	if count != 1 {
		t.Fatalf("expected /var/log/syslog once after dedup, got %d occurrences in %v", count, got)
	}
}

// sliceContains reports whether ss contains s.
func sliceContains(ss []string, s string) bool {
	for _, v := range ss {
		if v == s {
			return true
		}
	}
	return false
}
