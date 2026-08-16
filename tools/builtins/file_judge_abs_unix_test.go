//go:build !windows

package builtins

import (
	"context"
	"os"
	"testing"

	"github.com/v0lka/sp4rk/tools"
)

// TestJudgeWriteInSessionRoots_UnresolvablePathFailsClosed covers the write
// judge's filepath.Abs failure branch: when a path cannot be made absolute
// (here: the process working directory has been deleted, so os.Getwd fails),
// the judge must fail closed — deny with a HARD severity and a reason —
// mirroring the read-side judge and the web_fetch judge's "cannot determine
// target URL": an input that cannot be assessed at all is never soft (soft
// reasons may be auto-resolved by Smart Approve). The branch used to return a
// no-concern outcome, silently auto-approving an unresolvable write target.
//
// Unix-only: Windows cannot delete a process's current working directory, and
// macOS still resolves a deleted cwd, so the branch is only reachable on
// platforms where os.Getwd fails (Linux).
func TestJudgeWriteInSessionRoots_UnresolvablePathFailsClosed(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)
	if err := os.RemoveAll(dir); err != nil {
		t.Fatalf("failed to remove working directory: %v", err)
	}
	if wd, err := os.Getwd(); err == nil && wd != "" {
		t.Skipf("platform still resolves the deleted working directory (%s); cannot exercise the filepath.Abs failure branch", wd)
	}

	outcome := judgeWriteInSessionRoots(context.Background(), "relative/out.txt")
	if outcome.Allow {
		t.Fatal("expected unresolvable target path to be denied (fail closed)")
	}
	if outcome.Reason != "cannot determine target path" {
		t.Errorf("reason = %q, want %q", outcome.Reason, "cannot determine target path")
	}
	if outcome.Severity != tools.JudgeSeverityHard {
		t.Errorf("severity = %v, want %v (unassessable input must escalate as hard)", outcome.Severity, tools.JudgeSeverityHard)
	}
}
