//go:build !windows

package builtins

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/v0lka/sp4rk/tools"
)

// bash_exec severity classification. A Windows counterpart for posh_exec
// lives in posh_judge_severity_windows_test.go; both Judges share the same
// structure (blacklist → hard, containment → soft).

func TestBashExecTool_JudgeSeverity_BlacklistIsHard(t *testing.T) {
	tool, err := NewBashExecTool([]string{`rm\s+-rf`})
	if err != nil {
		t.Fatalf("failed to construct tool: %v", err)
	}
	input, _ := json.Marshal(map[string]string{"command": "rm -rf /"})
	outcome := tool.Judge(context.Background(), input)
	if outcome.Allow {
		t.Fatal("expected blacklist match to be denied")
	}
	if outcome.Reason == "" {
		t.Fatal("expected non-empty reason for blacklist match")
	}
	if outcome.Severity != tools.JudgeSeverityHard {
		t.Fatalf("blacklist match severity = %v, want hard", outcome.Severity)
	}
}

func TestBashExecTool_JudgeSeverity_PathContainmentIsSoft(t *testing.T) {
	tool, err := NewBashExecTool(nil)
	if err != nil {
		t.Fatalf("failed to construct tool: %v", err)
	}
	ws := t.TempDir()
	ctx := tools.WithWorkspacePath(context.Background(), ws)
	// /etc/hosts exists on Unix and lies outside the workspace root.
	input, _ := json.Marshal(map[string]string{"command": "cat /etc/hosts"})
	outcome := tool.Judge(ctx, input)
	if outcome.Allow {
		t.Fatal("expected out-of-root reference to be denied")
	}
	if !strings.Contains(outcome.Reason, "outside session roots") {
		t.Fatalf("unexpected reason: %q", outcome.Reason)
	}
	if outcome.Severity != tools.JudgeSeveritySoft {
		t.Fatalf("containment severity = %v, want soft", outcome.Severity)
	}
}

func TestBashExecTool_JudgeSeverity_NoConcern(t *testing.T) {
	tool, err := NewBashExecTool(nil)
	if err != nil {
		t.Fatalf("failed to construct tool: %v", err)
	}
	input, _ := json.Marshal(map[string]string{"command": "ls ."})
	outcome := tool.Judge(context.Background(), input)
	// No-concern is the zero outcome: allow=false with an EMPTY reason so the
	// registry does not escalate (workspace auto-approval semantics apply).
	if outcome.Allow || outcome.Reason != "" {
		t.Fatalf("expected zero (no-concern) outcome, got %+v", outcome)
	}
}
