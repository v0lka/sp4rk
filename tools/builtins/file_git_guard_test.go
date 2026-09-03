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

// newGitGuardWorkspace creates a workspace containing a realistic git
// internals layout (a top-level .git directory with config and objects) plus
// regular dotfiles that must remain writable.
func newGitGuardWorkspace(t *testing.T) string {
	t.Helper()
	ws := t.TempDir()
	gitDir := filepath.Join(ws, ".git")
	if err := os.MkdirAll(filepath.Join(gitDir, "objects", "ab"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(gitDir, "config"), []byte("[core]"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(ws, ".gitignore"), []byte("*.log\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(ws, ".github", "workflows"), 0o755); err != nil {
		t.Fatal(err)
	}
	return ws
}

// TestMutatingFileTools_Judge_GitInternalPathIsHard pins the git-internal
// guard for every mutating file tool: a target with a ".git" path component
// at or below the workspace root escalates as a HARD reason classified
// ReasonCodeGitInternal — a fired security control that must never be
// auto-resolved, only confirmed interactively (mirroring the symlink-escape
// severity contract).
func TestMutatingFileTools_Judge_GitInternalPathIsHard(t *testing.T) {
	ws := newGitGuardWorkspace(t)
	ctx := tools.WithWorkspacePath(context.Background(), ws)
	target := filepath.Join(ws, ".git", "config")

	tests := []struct {
		name  string
		tool  tools.Tool
		input map[string]any
	}{
		{
			name:  "write_file",
			tool:  NewWriteFileTool(),
			input: map[string]any{"path": target, "content": "evil"},
		},
		{
			name:  "edit_file",
			tool:  NewEditFileTool(),
			input: map[string]any{"path": target, "old_string": "[core]", "new_string": "pwned"},
		},
		{
			name:  "delete_file",
			tool:  NewDeleteFileTool(),
			input: map[string]any{"path": target},
		},
		{
			name:  "create_directory",
			tool:  NewCreateDirectoryTool(),
			input: map[string]any{"path": filepath.Join(ws, ".git", "hooks")},
		},
		{
			name:  "delete_directory",
			tool:  NewDeleteDirectoryTool(),
			input: map[string]any{"path": filepath.Join(ws, ".git", "objects"), "recursive": true},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			judger, ok := tt.tool.(tools.ToolJudger)
			if !ok {
				t.Fatalf("%s does not implement ToolJudger", tt.tool.Name())
			}
			input, err := json.Marshal(tt.input)
			if err != nil {
				t.Fatal(err)
			}
			outcome := judger.Judge(ctx, input)
			if outcome.Allow {
				t.Fatalf("expected allow=false for a .git target, got %+v", outcome)
			}
			if outcome.Severity != tools.JudgeSeverityHard {
				t.Fatalf("severity = %v, want hard", outcome.Severity)
			}
			if outcome.ReasonCode != tools.ReasonCodeGitInternal {
				t.Fatalf("reason code = %q, want %q", outcome.ReasonCode, tools.ReasonCodeGitInternal)
			}
			if !strings.Contains(outcome.Reason, ".git") {
				t.Errorf("reason should mention .git, got: %q", outcome.Reason)
			}
		})
	}
}

// TestJudge_GitInternal_CoversNestedReposAndDotGitTarget verifies the guard
// fires for nested repositories (submodule-style layouts) and for the ".git"
// path itself, including the file form a submodule uses as a gitdir pointer.
func TestJudge_GitInternal_CoversNestedReposAndDotGitTarget(t *testing.T) {
	ws := newGitGuardWorkspace(t)
	ctx := tools.WithWorkspacePath(context.Background(), ws)

	// Nested repo: <ws>/vendor/lib/.git/index.
	nestedDir := filepath.Join(ws, "vendor", "lib", ".git")
	if err := os.MkdirAll(nestedDir, 0o755); err != nil {
		t.Fatal(err)
	}
	write := NewWriteFileTool()
	input, _ := json.Marshal(map[string]string{"path": filepath.Join(nestedDir, "index"), "content": "x"})
	if outcome := write.Judge(ctx, input); outcome.Allow || outcome.Severity != tools.JudgeSeverityHard || outcome.ReasonCode != tools.ReasonCodeGitInternal {
		t.Fatalf("nested repo .git/index: got %+v, want hard %s", outcome, tools.ReasonCodeGitInternal)
	}

	// Submodule gitdir-pointer file: <ws>/sub/.git is a FILE, targeted directly.
	subGitFile := filepath.Join(ws, "sub", ".git")
	if err := os.MkdirAll(filepath.Dir(subGitFile), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(subGitFile, []byte("gitdir: ../../.git\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	del := NewDeleteFileTool()
	input, _ = json.Marshal(map[string]string{"path": subGitFile})
	if outcome := del.Judge(ctx, input); outcome.Allow || outcome.Severity != tools.JudgeSeverityHard || outcome.ReasonCode != tools.ReasonCodeGitInternal {
		t.Fatalf("submodule .git pointer file: got %+v, want hard %s", outcome, tools.ReasonCodeGitInternal)
	}

	// The .git directory itself as the operation target.
	mkdir := NewCreateDirectoryTool()
	input, _ = json.Marshal(map[string]string{"path": filepath.Join(ws, ".git")})
	if outcome := mkdir.Judge(ctx, input); outcome.Allow || outcome.Severity != tools.JudgeSeverityHard || outcome.ReasonCode != tools.ReasonCodeGitInternal {
		t.Fatalf("bare .git target: got %+v, want hard %s", outcome, tools.ReasonCodeGitInternal)
	}
}

// TestJudge_GitInternal_CaseSensitivity pins component-matching semantics to
// the session case-sensitivity flag: on a case-insensitive session a ".GIT"
// component IS the git directory (flagged); on a case-sensitive session it is
// an ordinary directory (not flagged) — mirroring tools.IsWithinRoot.
//
// The target deliberately uses a ".GIT" component that exists on disk in NO
// spelling (under a freshly created "sub" directory). A ".GIT" component that
// aliases the real on-disk ".git" directory is a different, platform-specific
// story: Windows filepath.EvalSymlinks canonicalizes existing path components
// to their on-disk spelling, so such a target arrives at the guard already
// spelled ".git" and is correctly flagged regardless of the session flag —
// the write genuinely hits git internals, and not flagging it would be a
// bypass. A component that exists in no spelling survives resolution verbatim
// on every platform, so the two subtests below exercise the guard's own case
// handling — and only that.
func TestJudge_GitInternal_CaseSensitivity(t *testing.T) {
	ws := newGitGuardWorkspace(t)
	if err := os.MkdirAll(filepath.Join(ws, "sub"), 0o755); err != nil {
		t.Fatal(err)
	}
	base := tools.WithWorkspacePath(context.Background(), ws)
	target := filepath.Join(ws, "sub", ".GIT", "config")
	write := NewWriteFileTool()
	input, err := json.Marshal(map[string]string{"path": target, "content": "x"})
	if err != nil {
		t.Fatal(err)
	}

	caseInsensitive := tools.WithCaseInsensitivePaths(base, true)
	if outcome := write.Judge(caseInsensitive, input); outcome.Allow || outcome.Severity != tools.JudgeSeverityHard || outcome.ReasonCode != tools.ReasonCodeGitInternal {
		t.Fatalf("case-insensitive session .GIT: got %+v, want hard %s", outcome, tools.ReasonCodeGitInternal)
	}

	caseSensitive := tools.WithCaseInsensitivePaths(base, false)
	outcome := write.Judge(caseSensitive, input)
	if !outcome.Allow {
		t.Fatalf("case-sensitive session .GIT is an ordinary dir: got %+v, want allow", outcome)
	}
	if outcome.ReasonCode != "" {
		t.Fatalf("case-sensitive session .GIT should carry no reason code, got %q", outcome.ReasonCode)
	}
}

// caseVariantPath returns the path with its first letter re-cased, or "" when
// the path contains no ASCII letter to vary (e.g. an all-numeric tail).
func caseVariantPath(p string) string {
	for i := 0; i < len(p); i++ {
		c := p[i]
		if c >= 'a' && c <= 'z' {
			return p[:i] + string(c-'a'+'A') + p[i+1:]
		}
		if c >= 'A' && c <= 'Z' {
			return p[:i] + string(c-'A'+'a') + p[i+1:]
		}
	}
	return ""
}

// TestJudge_GitInternal_CaseMismatchedWorkspacePrefix pins the
// fold-consistency contract between the guard's scope check and locality
// auto-approval: when the session folds letter case (case-insensitive
// filesystem), a .git target spelled with a case-mismatched workspace prefix
// denotes the SAME on-disk location as the workspace itself — it must hit the
// guard, not slip past it into auto-approval. Without folding
// (case-sensitive session) the variant path is a genuinely different,
// non-existent location and stays with the plain containment escalation.
func TestJudge_GitInternal_CaseMismatchedWorkspacePrefix(t *testing.T) {
	ws := newGitGuardWorkspace(t)
	variant := caseVariantPath(ws)
	if variant == "" {
		t.Skip("workspace path has no case-varying character")
	}
	write := NewWriteFileTool()
	marshal := func(path string) json.RawMessage {
		t.Helper()
		input, err := json.Marshal(map[string]string{"path": path, "content": "x"})
		if err != nil {
			t.Fatal(err)
		}
		return input
	}

	folded := tools.WithCaseInsensitivePaths(tools.WithWorkspacePath(context.Background(), ws), true)
	outcome := write.Judge(folded, marshal(filepath.Join(variant, ".git", "config")))
	if outcome.Allow || outcome.Severity != tools.JudgeSeverityHard || outcome.ReasonCode != tools.ReasonCodeGitInternal {
		t.Fatalf("case-mismatched .git target under folding: got %+v, want hard %s", outcome, tools.ReasonCodeGitInternal)
	}

	// No over-blocking: the same prefix spelling without a .git component
	// still auto-approves as a local write.
	outcome = write.Judge(folded, marshal(filepath.Join(variant, "notes.txt")))
	if !outcome.Allow {
		t.Fatalf("case-mismatched plain file under folding should stay allowed, got %+v (reason: %q)", outcome, outcome.Reason)
	}
}

// TestJudge_GitInternal_RegularDotfilesUnaffected guards against over-blocking:
// regular dotfiles and dot-directories (.gitignore, .github) contain no ".git"
// component and stay auto-approvable.
func TestJudge_GitInternal_RegularDotfilesUnaffected(t *testing.T) {
	ws := newGitGuardWorkspace(t)
	ctx := tools.WithWorkspacePath(context.Background(), ws)
	write := NewWriteFileTool()

	for _, target := range []string{
		filepath.Join(ws, ".gitignore"),
		filepath.Join(ws, ".github", "workflows", "ci.yml"),
	} {
		input, _ := json.Marshal(map[string]string{"path": target, "content": "x"})
		outcome := write.Judge(ctx, input)
		if !outcome.Allow {
			t.Errorf("%s: expected allow=true, got %+v (reason: %q)", target, outcome, outcome.Reason)
		}
		if outcome.ReasonCode != "" {
			t.Errorf("%s: expected no reason code, got %q", target, outcome.ReasonCode)
		}
	}
}

// TestJudge_GitInternal_ScopedToWorkspaceRoot pins the guard's scope: paths
// outside the workspace are handled by the outside-session-roots escalation
// (soft), and the temp directory — an equal session root, but not the
// workspace — is not guarded by this control.
func TestJudge_GitInternal_ScopedToWorkspaceRoot(t *testing.T) {
	ws := newGitGuardWorkspace(t)
	other := t.TempDir()
	tempDir := t.TempDir()
	ctx := tools.WithTempDir(tools.WithWorkspacePath(context.Background(), ws), tempDir)
	write := NewWriteFileTool()

	// Outside every session root: denied, but by the SOFT containment rule,
	// not the git-internal guard.
	input, _ := json.Marshal(map[string]string{"path": filepath.Join(other, "repo", ".git", "config"), "content": "x"})
	outcome := write.Judge(ctx, input)
	if outcome.Allow {
		t.Fatal("expected out-of-roots .git write to be denied")
	}
	if outcome.Severity != tools.JudgeSeveritySoft || outcome.ReasonCode != tools.ReasonCodeOutsideSessionRoots {
		t.Fatalf("out-of-roots .git write should stay soft/%s, got %+v", tools.ReasonCodeOutsideSessionRoots, outcome)
	}

	// Inside the temp dir: a peer root, auto-approved like any other write.
	input, _ = json.Marshal(map[string]string{"path": filepath.Join(tempDir, ".git", "config"), "content": "x"})
	outcome = write.Judge(ctx, input)
	if !outcome.Allow {
		t.Fatalf("temp-dir .git write is outside the guard's scope: got %+v (reason: %q)", outcome, outcome.Reason)
	}
}

// TestRegistry_GitInternalWriteEscalatesAsHardReason proves the guard is
// routed through the unified confirmation funnel: with write_file relaxed to
// PolicyAlwaysAllow, the judge's hard outcome reaches ConfirmationRequest
// with the git_internal_path classification, and a user denial blocks the
// write.
func TestRegistry_GitInternalWriteEscalatesAsHardReason(t *testing.T) {
	ws := newGitGuardWorkspace(t)
	ctx := tools.WithWorkspacePath(context.Background(), ws)
	target := filepath.Join(ws, ".git", "config")

	reg := tools.NewToolRegistry()
	reg.Register(NewWriteFileTool())
	reg.SetPolicyOverride("write_file", tools.PolicyAlwaysAllow)

	var gotReq tools.ConfirmationRequest
	reg.SetConfirmFunc(func(_ context.Context, req tools.ConfirmationRequest) (tools.ConfirmationResponse, error) {
		gotReq = req
		return tools.ConfirmDeny, nil
	})

	input, _ := json.Marshal(map[string]string{"path": target, "content": "evil"})
	res, err := reg.Execute(ctx, "write_file", input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !res.IsError {
		t.Fatalf("expected denied tool result, got %+v", res)
	}
	if !strings.Contains(gotReq.JudgeReasoning, ".git") {
		t.Errorf("confirmation reasoning should mention .git, got: %q", gotReq.JudgeReasoning)
	}
	if gotReq.JudgeSeverity != tools.JudgeSeverityHard {
		t.Errorf("confirmation severity = %v, want hard", gotReq.JudgeSeverity)
	}
	if gotReq.JudgeReasonCode != tools.ReasonCodeGitInternal {
		t.Errorf("confirmation reason code = %q, want %q", gotReq.JudgeReasonCode, tools.ReasonCodeGitInternal)
	}

	// The denial must have blocked the write.
	after, readErr := os.ReadFile(target)
	if readErr != nil {
		t.Fatal(readErr)
	}
	if string(after) != "[core]" {
		t.Fatalf("git config was modified despite denial: %q", after)
	}
}
