// SPDX-License-Identifier: Apache-2.0

package tools

import (
	"context"
	"os"
	"path"
	"path/filepath"
	"runtime"
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

// TestResolveShellPathTokens_UnsetEnvVarSuffixResolved verifies the two sides of
// an empty/unset environment variable: a bare reference names no path and is
// skipped, while the same reference concatenated with an absolute path suffix
// resolves to that suffix — mirroring the shell's expansion of an empty variable
// ("$UNSET/etc/passwd" → "/etc/passwd"), so a prompt-injection command hiding an
// absolute path behind an empty var still surfaces as an out-of-root reference.
func TestResolveShellPathTokens_UnsetEnvVarSuffixResolved(t *testing.T) {
	t.Setenv("NOPE_SHELLPATHS_UNDEF", "")
	t.Setenv("NOPE", "")

	// Bare unset var names no path.
	if got := ResolveShellPathTokens("cat $NOPE", ShellBash, "/tmp/ws"); len(got) != 0 {
		t.Fatalf("expected bare unset $NOPE to be skipped, got %v", got)
	}
	// Unset var + absolute suffix expands to the suffix.
	got := ResolveShellPathTokens("cat $NOPE/etc/passwd", ShellBash, "/tmp/ws")
	if !sliceContains(got, "/etc/passwd") {
		t.Fatalf("expected /etc/passwd resolved from $NOPE/etc/passwd, got %v", got)
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

// TestPathsOutsideRoots_RelativePathNotExtracted is the regression test for the
// false-positive where pathRegex (via FindAllString) matched a "/" ANYWHERE in
// the command, not only at a token boundary. A relative path such as
// "frontend/src/main.tsx" or "go test ./core/..." had its embedded "/src/..." or
// "/core/..." extracted as a spurious POSIX absolute path and flagged as
// out-of-root, triggering constant confirmation prompts. After the fix the "/"
// following a path-component character is treated as a separator inside the
// relative path and ignored.
func TestPathsOutsideRoots_RelativePathNotExtracted(t *testing.T) {
	ws := t.TempDir()
	ctx := WithWorkspacePath(context.Background(), ws)

	for _, cmd := range []string{
		"go test ./core/...",
		"git diff --stat frontend/src/main.tsx",
		"cat backend/config/config.go",
		"ls core/tools",
		"npm run build -- --config frontend/vite.config.ts",
	} {
		got := PathsOutsideRoots(ctx, cmd, ShellBash, ws)
		if len(got) != 0 {
			t.Errorf("relative cmd %q must not be flagged as out-of-root, got %v", cmd, got)
		}
	}
}

// TestPathsOutsideRoots_DotDotEllipsisNotParentRef is the regression test for
// the false-positive where relDotDotRe matched the ".." inside a run of 3+
// dots (an ellipsis, e.g. the go-test recursive pattern "go test ..."). The
// extracted ".." resolved against the workspace to its PARENT directory and was
// flagged as out-of-root. This affected bash ("go test ...") and especially
// posh ("go test .\core\..."), whose backslash separators are not consumed by
// pathRegex the way bash's "/" is. After the fix a ".." immediately followed
// by another dot is recognized as an ellipsis, not a parent-directory ref.
func TestPathsOutsideRoots_DotDotEllipsisNotParentRef(t *testing.T) {
	ws := t.TempDir()
	ctx := WithWorkspacePath(context.Background(), ws)

	commands := map[ShellKind][]string{
		ShellBash: {
			"go test ./...",
			"go test ...",
			"go test ./core/...",
			"go test ./core/... ./backend/...",
		},
		ShellPosh: {
			"go test ...",
			"go test .\\core\\...",
			"go test .\\... .\\backend\\...",
		},
	}
	for shell, cmds := range commands {
		for _, cmd := range cmds {
			got := PathsOutsideRoots(ctx, cmd, shell, ws)
			if len(got) != 0 {
				t.Errorf("[%s] ellipsis cmd %q must not be flagged as out-of-root, got %v", shell, cmd, got)
			}
		}
	}
}

// TestPathsOutsideRoots_GenuineEscapeStillFlagged verifies the boundary fix
// does not weaken genuine parent-directory escapes: a real "../.." (bash) or
// "..\.." (posh) that resolves outside the workspace is still reported.
func TestPathsOutsideRoots_GenuineEscapeStillFlagged(t *testing.T) {
	ws := t.TempDir()
	ctx := WithWorkspacePath(context.Background(), ws)

	got := PathsOutsideRoots(ctx, "cat ../../etc/passwd", ShellBash, ws)
	// The exact resolved path depends on how deeply nested ws is; what matters
	// is that the genuine "../.." escape resolves OUTSIDE the workspace and is
	// reported (it must end in "etc/passwd" two levels above ws). Normalize the
	// resolved path's separators to "/" before the suffix check: on Windows
	// resolveRelativeToken returns a filepath.Clean-ed path with backslash
	// separators, so a literal "etc/passwd" suffix would never match there.
	if len(got) == 0 || !strings.HasSuffix(filepath.ToSlash(got[0]), "etc/passwd") {
		t.Fatalf("expected ../../etc/passwd -> <ws parent-parent>/etc/passwd flagged as out-of-root, got %v", got)
	}
}

// TestPathsOutsideRoots_PrefixedEscapeStillFlagged is the regression test for
// the security hole introduced by the original leading-boundary fix: a genuine
// parent-directory escape written with a relative prefix (the canonical shell
// ways to express an escape) was silently dropped because the "/"-leading
// pathRegex token that pathRegex matched INSIDE the escape was suppressed
// without resolving the enclosing word. After the fix the whole relative-path
// word is resolved against workDir, so these escapes surface as real
// out-of-root paths and the always-allow escalation is re-armed.
func TestPathsOutsideRoots_PrefixedEscapeStillFlagged(t *testing.T) {
	ws := t.TempDir()
	ctx := WithWorkspacePath(context.Background(), ws)

	for _, cmd := range []string{
		"cat ./../etc/passwd",
		"cat ./../../etc/passwd",
		"cat a/../../etc/passwd",
		"cd subdir/../..",
	} {
		got := PathsOutsideRoots(ctx, cmd, ShellBash, ws)
		if len(got) == 0 {
			t.Errorf("prefixed escape cmd %q must be flagged as out-of-root, got %v", cmd, got)
		}
	}
}

// TestPathsOutsideRoots_RelativeWordInRootNotFlagged pins Issue #2: a relative
// path that resolves INSIDE the workspace (even with ".." components) must not
// be flagged. "a/../b" resolves to ws/b (in-root); "x../../etc/passwd" has its
// ".." glued to a filename char, so the whole token resolves to a path still
// rooted under ws and is correctly not flagged.
func TestPathsOutsideRoots_RelativeWordInRootNotFlagged(t *testing.T) {
	ws := t.TempDir()
	ctx := WithWorkspacePath(context.Background(), ws)

	for _, cmd := range []string{
		"cat a/../b",
		"cat x../../etc/passwd",
		"cat ./subdir/../file.txt",
	} {
		got := PathsOutsideRoots(ctx, cmd, ShellBash, ws)
		if len(got) != 0 {
			t.Errorf("in-root relative cmd %q must not be flagged, got %v", cmd, got)
		}
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

// TestPathsOutsideRoots_HarmlessDeviceNotReported verifies that harmless
// special-device paths are NOT reported as outside the session roots, even
// though they are not contained in any root. The harmless device is host-
// dependent: POSIX exempts /dev/null and /dev/full; Windows exempts the
// reserved NUL device, while /dev/null and /dev/full are ordinary out-of-root
// paths there.
func TestPathsOutsideRoots_HarmlessDeviceNotReported(t *testing.T) {
	ws := t.TempDir()
	ctx := WithWorkspacePath(context.Background(), ws)
	cmds := []string{
		"cat foo > /dev/null",
		"echo hi > /dev/null",
		"cat /dev/null",
		"echo x > /dev/full",
	}
	if runtime.GOOS == "windows" {
		cmds = []string{
			"cat foo > NUL",
			"echo hi > NUL",
			"cat NUL",
			"echo x > NUL",
		}
	}
	for _, cmd := range cmds {
		got := PathsOutsideRoots(ctx, cmd, ShellBash, ws)
		if len(got) != 0 {
			t.Errorf("harmless-device cmd %q must not be flagged as out-of-root, got %v", cmd, got)
		}
	}
}

// TestPathsOutsideRoots_HarmlessDeviceAlongsideOutsided verifies that a
// harmless device does not mask genuinely out-of-root paths: the real escape
// (/etc/passwd) is still reported when it appears alongside the harmless
// device (/dev/null on POSIX, the reserved NUL device on Windows).
func TestPathsOutsideRoots_HarmlessDeviceAlongsideOutside(t *testing.T) {
	ws := t.TempDir()
	ctx := WithWorkspacePath(context.Background(), ws)
	cmd := "cat /etc/passwd > /dev/null"
	harmless := "/dev/null"
	if runtime.GOOS == "windows" {
		cmd = "cat /etc/passwd > NUL"
		harmless = "NUL"
	}
	got := PathsOutsideRoots(ctx, cmd, ShellBash, ws)
	if !sliceContains(got, "/etc/passwd") {
		t.Fatalf("expected /etc/passwd reported alongside %s, got %v", harmless, got)
	}
	for _, p := range got {
		if p == harmless {
			t.Fatalf("%s must not be reported as out-of-root, got %v", harmless, got)
		}
	}
}

// TestExistingPaths filters a slice of paths down to those that exist on the
// host filesystem. This is the strict on-disk existence primitive; the
// shell-exec Judges use the anchored variant (TestExistingOrAnchoredPaths).
func TestExistingPaths(t *testing.T) {
	existing1 := t.TempDir()
	existing2 := filepath.Join(t.TempDir(), "file.txt")
	if err := os.WriteFile(existing2, []byte("x"), 0o644); err != nil {
		t.Fatalf("setup file: %v", err)
	}
	missingDir := filepath.Join(t.TempDir(), "does", "not", "exist")
	missingFile := filepath.Join(t.TempDir(), "missing.txt")

	in := []string{existing1, existing2, missingDir, missingFile, ""}
	got := ExistingPaths(in)

	if len(got) != 2 {
		t.Fatalf("expected 2 existing paths, got %d: %v", len(got), got)
	}
	if !sliceContains(got, existing1) || !sliceContains(got, existing2) {
		t.Fatalf("expected %q and %q to remain, got %v", existing1, existing2, got)
	}
	for _, p := range got {
		if p == missingDir || p == missingFile || p == "" {
			t.Fatalf("non-existent/empty path must be dropped, got %v", got)
		}
	}
}

// TestExistingOrAnchoredPaths verifies the ancestor-anchored filter used by the
// shell-exec Judges: a path is retained when it exists, OR when its nearest
// existing ancestor directory exists (a write target in a real directory). A
// wholly non-existent subtree (no existing ancestor below the volume root) is
// dropped.
func TestExistingOrAnchoredPaths(t *testing.T) {
	existingFile := filepath.Join(t.TempDir(), "file.txt")
	if err := os.WriteFile(existingFile, []byte("x"), 0o644); err != nil {
		t.Fatalf("setup file: %v", err)
	}
	existingDir := t.TempDir()

	// A write target whose leaf does not exist but whose parent directory does.
	anchoredTarget := filepath.Join(existingDir, "newchild", "newleaf")

	// A wholly non-existent subtree: no ancestor below the volume root exists.
	// Use an unlikely name so it does not collide with a real directory.
	whollyMissing := string(filepath.Separator) + "zzz-qqq-rrr-nono-" + "leaf"

	in := []string{existingFile, anchoredTarget, whollyMissing, ""}
	got := ExistingOrAnchoredPaths(in)

	if !sliceContains(got, existingFile) {
		t.Errorf("existing file %q must be retained, got %v", existingFile, got)
	}
	if !sliceContains(got, anchoredTarget) {
		t.Errorf("anchored write target %q (parent %q exists) must be retained, got %v", anchoredTarget, existingDir, got)
	}
	for _, p := range got {
		if p == whollyMissing {
			t.Errorf("wholly non-existent path %q must be dropped, got %v", whollyMissing, got)
		}
		if p == "" {
			t.Errorf("empty path must be dropped, got %v", got)
		}
	}
}

// TestExistingOrAnchoredPaths_SystemDirWriteTarget verifies the
// prompt-injection-relevant case: a write into an existing system directory
// whose leaf does not yet exist is retained, so it would escalate a shell
// command under auto-approval. The anchor directory is the nearest existing
// system directory for the running OS — /etc on macOS/Linux, %SystemRoot%
// (e.g. C:\Windows) on Windows — so the retention is exercised on every
// platform rather than only where /etc happens to exist.
func TestExistingOrAnchoredPaths_SystemDirWriteTarget(t *testing.T) {
	anchorDir := systemWriteAnchorDir(t)
	target := filepath.Join(anchorDir, "cron.d", "sp4rk-unrelated-leaf-marker")

	got := ExistingOrAnchoredPaths([]string{target})
	if len(got) != 1 {
		t.Fatalf("expected the anchored system-dir write target to be retained, got %v", got)
	}
	if got[0] != target {
		t.Errorf("expected the target path verbatim, got %q", got[0])
	}
}

// systemWriteAnchorDir returns a real, existing OS system directory that can
// anchor a write target whose own leaf does not exist: /etc on POSIX, and
// %SystemRoot% (e.g. C:\Windows, always set on supported Windows releases) on
// Windows. It lets the cross-platform SystemDirWriteTarget test exercise the
// prompt-injection-relevant retention on every OS. It skips (rather than fails)
// in the unlikely event the anchor cannot be resolved.
func systemWriteAnchorDir(t *testing.T) string {
	t.Helper()
	if runtime.GOOS == "windows" {
		sysRoot := os.Getenv("SystemRoot") // e.g. C:\Windows; set on every Windows host.
		if sysRoot == "" {
			t.Skip("%SystemRoot% is unset; cannot resolve a Windows system-dir anchor")
		}
		return sysRoot
	}
	return "/etc"
}

// TestPathsOutsideRoots_PureSeparatorRunNotPath is the regression test for the
// false positive where pathRegex matched a bare separator run ("//") and
// ResolveShellPathTokens resolved it (path.Clean("//") == "/") to the
// filesystem root, flagging innocent commands as out-of-root and triggering a
// needless confirmation. The "//" is a shell-language artifact — the trailing
// address of a sed substitution ("sed 's/.*function //'"), a comment marker
// ("echo \"// TODO fix\"") or an integer-division operator
// ("echo $(( total // count ))") — and names no location, so it is skipped
// before dispatch. Both sed-bearing user commands from the bug report are
// exercised, under both shell dialects.
func TestPathsOutsideRoots_PureSeparatorRunNotPath(t *testing.T) {
	ws := t.TempDir()
	ctx := WithWorkspacePath(context.Background(), ws)

	commands := map[ShellKind][]string{
		ShellBash: {
			// Both user commands from the bug report, verbatim.
			"rg 'func ' -n core/conductor.go | sed 's/.*function //' | sort -u",
			"grep -rn 'function ' src/ | sed 's/.*function //' | head -5",
			// Comment-marker spellings, including the literal "// ..." body.
			`echo "// ..."`,
			`echo "// TODO fix" >> notes.md`,
			// Integer-division spellings.
			"echo $(( x // y ))",
			"echo $(( total // count ))",
		},
		ShellPosh: {
			"rg 'func ' -n core/conductor.go | sed 's/.*function //' | sort -u",
			"grep -rn 'function ' src/ | sed 's/.*function //' | head -5",
			`echo "// ..."`,
			`echo "// TODO fix" >> notes.md`,
			"echo $(( x // y ))",
			"echo $(( total // count ))",
		},
	}
	for shell, cmds := range commands {
		for _, cmd := range cmds {
			got := PathsOutsideRoots(ctx, cmd, shell, ws)
			if len(got) != 0 {
				t.Errorf("[%s] separator-artifact cmd %q must not be flagged as out-of-root, got %v", shell, cmd, got)
			}
		}
	}
}

// TestPathsOutsideRoots_SeparatorRunSkipKeepsRealPaths verifies the
// separator-run skip does not weaken detection: genuine out-of-root references
// — including forms built from separator runs ("cat //etc/passwd") and
// single-component roots ("rm -rf /.") — are still reported under BOTH shell
// dialects (pathRegex and the skip are shared between them).
func TestPathsOutsideRoots_SeparatorRunSkipKeepsRealPaths(t *testing.T) {
	ws := t.TempDir()
	ctx := WithWorkspacePath(context.Background(), ws)

	for _, shell := range []ShellKind{ShellBash, ShellPosh} {
		for _, cmd := range []string{
			"cat /etc/passwd",
			"echo x > /etc/cron.d/newjob",
			"cat //etc/passwd",
			"rm -rf /.",
		} {
			got := PathsOutsideRoots(ctx, cmd, shell, ws)
			if len(got) == 0 {
				t.Errorf("[%s] cmd %q must still be flagged as out-of-root", shell, cmd)
			}
		}
	}
}

// TestPathsOutsideRoots_DriveSeparatorRunForms pins the drive-letter half of
// the separator-run skip at the command level. "C:\\" — a drive prefix
// followed by a pure separator run — is an escaped-drive-root artifact and is
// never reported, on any host. The genuine forms "C:\" (drive root) and
// "C:\\Windows\win.ini" (escaped drive path) still name real locations, but a
// Windows drive path is only absolute on Windows ([isAbsResolved]): on POSIX
// hosts the resolver drops it by design, so the still-reported assertions are
// guarded by GOOS.
func TestPathsOutsideRoots_DriveSeparatorRunForms(t *testing.T) {
	ws := t.TempDir()
	ctx := WithWorkspacePath(context.Background(), ws)

	for _, shell := range []ShellKind{ShellBash, ShellPosh} {
		cmd := `Get-Content C:\\`
		got := PathsOutsideRoots(ctx, cmd, shell, ws)
		if len(got) != 0 {
			t.Errorf("[%s] cmd %q must not be flagged (escaped drive root is an artifact), got %v", shell, cmd, got)
		}
	}

	if runtime.GOOS != "windows" {
		return
	}
	for _, shell := range []ShellKind{ShellBash, ShellPosh} {
		for _, cmd := range []string{
			`Get-Content C:\`,
			`Get-Content C:\\Windows\win.ini`,
		} {
			got := PathsOutsideRoots(ctx, cmd, shell, ws)
			if len(got) == 0 {
				t.Errorf("[%s] cmd %q must still be flagged on Windows (drive path outside roots)", shell, cmd)
			}
		}
	}
}

// TestIsPureSeparatorRunToken covers the token classifier behind the
// separator-run skip shared by ResolveShellPathTokens and ExtractPaths: POSIX
// runs of two or more slashes and drive prefixes followed by two or more pure
// separators are artifacts; everything carrying at least one path component —
// including the single-separator drive root "C:\" — is not.
func TestIsPureSeparatorRunToken(t *testing.T) {
	positive := []string{"//", "///", "////", `C:\\`, `D:\\`, `C://`}
	for _, tok := range positive {
		if !isPureSeparatorRunToken(tok) {
			t.Errorf("isPureSeparatorRunToken(%q) = false, want true", tok)
		}
	}
	negative := []string{
		"", ":", "..", "a/b", "/",
		"/.", "//etc/passwd", "/dev/null", "/etc/cron.d/newjob",
		`C:\`, `C:/`, `C:\Windows`, `C:\\Windows`,
	}
	for _, tok := range negative {
		if isPureSeparatorRunToken(tok) {
			t.Errorf("isPureSeparatorRunToken(%q) = true, want false", tok)
		}
	}
}

// TestUnresolvablePathTokens verifies that path-like tokens the resolver
// cannot assess are surfaced (so the Judges escalate them hard), while
// resolvable tilde/env forms and non-tilde "~" occurrences are not.
func TestUnresolvablePathTokens(t *testing.T) {
	cases := []struct {
		name    string
		command string
		shell   ShellKind
		want    []string
	}{
		{
			name:    "tilde-user",
			command: "cat ~root/.ssh/id_rsa",
			shell:   ShellBash,
			want:    []string{"~root"},
		},
		{
			name:    "tilde-current-user-home-not-unresolvable",
			command: "cat ~/.ssh/id_rsa",
			shell:   ShellBash,
			want:    nil,
		},
		{
			name:    "git-revision-tilde-not-tilde-expansion",
			command: "git log HEAD~3",
			shell:   ShellBash,
			want:    nil,
		},
		{
			name:    "param-default-value",
			command: "cat ${VAR:-/etc/passwd}",
			shell:   ShellBash,
			want:    []string{"${VAR:-/etc/passwd}"},
		},
		{
			name:    "param-benign-default-not-unresolvable",
			command: "echo ${GREETING:-hello}",
			shell:   ShellBash,
			want:    nil,
		},
		{
			name:    "param-plain-var-not-unresolvable",
			command: "cat ${VAR}/config",
			shell:   ShellBash,
			want:    nil,
		},
		{
			name:    "posh-has-no-unresolvable-idioms",
			command: "cat ~user/.ssh/id_rsa",
			shell:   ShellPosh,
			want:    nil,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := UnresolvablePathTokens(tc.command, tc.shell)
			if len(got) != len(tc.want) {
				t.Fatalf("UnresolvablePathTokens(%q, %q) = %v, want %v", tc.command, tc.shell, got, tc.want)
			}
			for i := range got {
				if got[i] != tc.want[i] {
					t.Fatalf("UnresolvablePathTokens(%q, %q) = %v, want %v", tc.command, tc.shell, got, tc.want)
				}
			}
		})
	}
}

// TestHasRelativeEscape verifies that a ".." parent-ref inside a relative path
// is detected (so the JSON judge fast-path fails closed) while plain relative
// names, absolute paths, ellipses, and ".."-prefixed filenames are not.
func TestHasRelativeEscape(t *testing.T) {
	positive := []string{
		"a/../../etc/passwd",
		"../foo",
		"subdir/../..",
		"../etc/passwd",
	}
	for _, s := range positive {
		if !HasRelativeEscape(s) {
			t.Errorf("HasRelativeEscape(%q) = false, want true", s)
		}
	}
	negative := []string{
		"frontend/src/main.tsx",
		"/ws/notes.txt",
		"...",
		"..config",
		"",
		"plain",
	}
	for _, s := range negative {
		if HasRelativeEscape(s) {
			t.Errorf("HasRelativeEscape(%q) = true, want false", s)
		}
	}
}

func TestResolveShellPathTokens_InCommandBindingUnionsEnv(t *testing.T) {
	// A variable assigned a literal value in-command does NOT shadow the
	// process env: when both a binding and a non-empty, differing env value
	// exist, BOTH expansions are reported (fail-closed over-report). This test
	// kept next to the resolver it exercises; broader coverage — including the
	// prefix-assignment form and the parser-layer (extractBashPaths) union —
	// lives in shellEnvBindings_test.go.
	envVal := filepath.ToSlash(osAbsPath("from", "env"))
	t.Setenv("D", envVal)
	bind := filepath.ToSlash(osAbsPath("tmp", "build"))
	got := ResolveShellPathTokens(`D=`+bind+`; cat "$D/a"`, ShellBash, osAbsPath("wd"))
	want := []string{
		filepath.Clean(osAbsPath("tmp", "build")),
		filepath.Clean(osAbsPath("tmp", "build", "a")),
		filepath.Clean(osAbsPath("from", "env", "a")),
	}
	if len(got) != len(want) {
		t.Fatalf("union of binding and env expansions = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("union of binding and env expansions = %v, want %v", got, want)
		}
	}

	// Unbound $D (no in-command assignment) falls back to the process env.
	back := ResolveShellPathTokens(`cat "$D/a"`, ShellBash, osAbsPath("wd"))
	wantBack := []string{filepath.Clean(osAbsPath("from", "env", "a"))}
	if len(back) != len(wantBack) || back[0] != wantBack[0] {
		t.Fatalf("env fallback = %v, want %v", back, wantBack)
	}
}

func TestResolveShellPathTokens_LeadingSlashEnvTokenResolvesAbsolute(t *testing.T) {
	// A word-initial "/" before an env token — "/$D/passwd" — makes the
	// runtime expansion absolute, but the token itself starts at "$D" and
	// resolves to the relative join "etc/passwd", which the absolute-only
	// filter drops: without the reconstruction the out-of-root reference
	// would go unreported while the binding cleared the parser layer's
	// suspicious flag. The absolutized variant "/etc/passwd" must be
	// reported (and escalate via PathsOutsideRoots). With the env pinned
	// empty the EMPTY expansion is also a possible runtime value (the
	// binding may never execute), so the suffix-alone "/passwd" is reported
	// as an equal-standing candidate.
	t.Setenv("D", "")
	got := ResolveShellPathTokens(`D=etc; cat "/$D/passwd"`, ShellBash, osAbsPath("wd"))
	if !sliceContains(got, "/etc/passwd") || !sliceContains(got, "/passwd") {
		t.Fatalf("leading-slash env token = %v, want [/etc/passwd /passwd]", got)
	}

	ws := t.TempDir()
	ctx := WithWorkspacePath(context.Background(), ws)
	if out := PathsOutsideRoots(ctx, `D=etc; cat "/$D/passwd"`, ShellBash, ws); !sliceContains(out, "/etc/passwd") {
		t.Fatalf("expected /etc/passwd reported outside roots, got %v", out)
	}

	// The unbound variant keeps surfacing through the suffix-alone fallback.
	if got := ResolveShellPathTokens(`cat "/$UNSET/passwd"`, ShellBash, osAbsPath("wd")); !sliceContains(got, "/passwd") {
		t.Fatalf("unbound leading-slash env token = %v, want /passwd included", got)
	}

	// A "/" that continues a relative word ("a/$D/passwd" → runtime
	// "a/etc/passwd", relative) must NOT be absolutized: the runtime path
	// stays under the working directory.
	got = ResolveShellPathTokens(`D=etc; cat a/$D/passwd`, ShellBash, osAbsPath("wd"))
	for _, p := range got {
		if p == "/etc/passwd" {
			t.Fatalf("mid-word relative env reference absolutized: %v", got)
		}
	}
}

func TestResolveShellPathTokens_RebindingReportsAllValues(t *testing.T) {
	// A name assigned more than once in the command is position-ambiguous
	// (the pre-pass cannot tell which value a reference sees — a decoy
	// re-binding after the reference must not mask the earlier out-of-root
	// binding), so EVERY distinct literal value is reported as a candidate
	// (fail-closed union), never just the last one.
	t.Setenv("D", "")
	first := filepath.ToSlash(osAbsPath("etc", "passwd"))
	decoy := filepath.ToSlash(osAbsPath("ws", "safe"))
	got := ResolveShellPathTokens(`D=`+first+`; cat "/$D/x"; D=`+decoy, ShellBash, osAbsPath("wd"))
	for _, want := range []string{
		filepath.Clean(osAbsPath("etc", "passwd")),
		filepath.Clean(osAbsPath("ws", "safe")),
		filepath.Clean(osAbsPath("etc", "passwd", "x")),
		filepath.Clean(osAbsPath("ws", "safe", "x")),
	} {
		if !sliceContains(got, want) {
			t.Fatalf("re-bound var candidates = %v, want %q included", got, want)
		}
	}
}

func TestStripPosixRootOverDrive_ForOS(t *testing.T) {
	// Exercises the Windows drive-path branch on every host: the resolver can
	// only reach it on a Windows runner, so the OS-parameterized core is
	// pinned directly (the isNullDeviceForOS convention). The leading "/" an
	// absolute-shaped word contributes over a drive-letter candidate must
	// yield the native drive path on Windows, while POSIX hosts — and every
	// assembly that is not exactly root + drive letter + separator — keep the
	// assembled form.
	tests := []struct {
		name string
		p    string
		goos string
		want string
	}{
		{"windows drive path", "/C:/Users/u/etc/passwd/x", "windows", "C:/Users/u/etc/passwd/x"},
		{"windows backslash value", `/C:\Users\u/x`, "windows", `C:\Users\u/x`},
		{"lowercase drive", "/c:/tmp/x", "windows", "c:/tmp/x"},
		{"posix absolute untouched", "/etc/passwd/x", "windows", "/etc/passwd/x"},
		{"double root from posix-absolute value untouched", "//etc/passwd/x", "windows", "//etc/passwd/x"},
		{"bare drive colon untouched", "/C:", "windows", "/C:"},
		{"colon without trailing separator untouched", "/C:temp/x", "windows", "/C:temp/x"},
		{"multi-letter name untouched", "/CD:/x", "windows", "/CD:/x"},
		{"drive not at root untouched", "/xC:/y", "windows", "/xC:/y"},
		{"linux keeps assembled drive form", "/C:/Users/u/x", "linux", "/C:/Users/u/x"},
		{"darwin keeps backslash form", `/C:\Users\u/x`, "darwin", `/C:\Users\u/x`},
		{"empty", "", "windows", ""},
		{"too short", "/C", "windows", "/C"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := stripPosixRootOverDrive(tt.p, tt.goos); got != tt.want {
				t.Errorf("stripPosixRootOverDrive(%q, %q) = %q, want %q", tt.p, tt.goos, got, tt.want)
			}
		})
	}
}
