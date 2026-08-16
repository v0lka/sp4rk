//go:build windows

package builtins

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/v0lka/sp4rk/tools"
)

// posh_exec severity classification, the Windows counterpart of
// bash_judge_severity_test.go (both shell Judges share structure:
// blacklist → hard, containment → soft, no concern → zero outcome).

func TestPoshExecTool_JudgeSeverity_BlacklistIsHard(t *testing.T) {
	tool, err := NewPoshExecTool([]string{`Remove-Item\s+-Recurse`})
	if err != nil {
		t.Fatalf("failed to construct tool: %v", err)
	}
	input, _ := json.Marshal(map[string]string{"command": "Remove-Item -Recurse C:\\"})
	outcome := tool.Judge(context.Background(), input)
	if outcome.Allow {
		t.Fatal("expected blacklist match to be denied")
	}
	if outcome.Severity != tools.JudgeSeverityHard {
		t.Fatalf("blacklist match severity = %v, want hard", outcome.Severity)
	}
}

func TestPoshExecTool_JudgeSeverity_PathContainmentIsSoft(t *testing.T) {
	tool, err := NewPoshExecTool(nil)
	if err != nil {
		t.Fatalf("failed to construct tool: %v", err)
	}
	ws := t.TempDir()
	ctx := tools.WithWorkspacePath(context.Background(), ws)
	// C:\Windows\win.ini exists on Windows and lies outside the workspace
	// root (mirrors the bash counterpart's /etc/hosts).
	input, _ := json.Marshal(map[string]string{"command": "Get-Content C:\\Windows\\win.ini"})
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

func TestPoshExecTool_JudgeSeverity_NoConcern(t *testing.T) {
	tool, err := NewPoshExecTool(nil)
	if err != nil {
		t.Fatalf("failed to construct tool: %v", err)
	}
	ws := t.TempDir()
	ctx := tools.WithWorkspacePath(context.Background(), ws)
	// A benign in-workspace command: no-concern is the zero outcome
	// (allow=false with an EMPTY reason so the registry does not escalate;
	// workspace auto-approval semantics apply).
	input, _ := json.Marshal(map[string]string{"command": "Get-ChildItem ."})
	outcome := tool.Judge(ctx, input)
	if outcome.Allow || outcome.Reason != "" {
		t.Fatalf("expected zero (no-concern) outcome, got %+v", outcome)
	}
}
