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

func TestBashExecTool_JudgeSeverity_BoundVarContainmentIsSoft(t *testing.T) {
	tool, err := NewBashExecTool(nil)
	if err != nil {
		t.Fatalf("failed to construct tool: %v", err)
	}
	ws := t.TempDir()
	ctx := tools.WithWorkspacePath(context.Background(), ws)
	// D is bound in-command to /etc; "$D/hosts" resolves to /etc/hosts, which
	// exists on Unix and lies outside the workspace root. The bound variable is
	// not a hard (unresolvable) token — containment escalation is soft.
	input, _ := json.Marshal(map[string]string{"command": `D=/etc; cat "$D/hosts"`})
	outcome := tool.Judge(ctx, input)
	if outcome.Allow {
		t.Fatal("expected out-of-root reference (via bound var) to be denied")
	}
	if !strings.Contains(outcome.Reason, "outside session roots") {
		t.Fatalf("unexpected reason: %q", outcome.Reason)
	}
	if outcome.Severity != tools.JudgeSeveritySoft {
		t.Fatalf("bound-var containment severity = %v, want soft", outcome.Severity)
	}
}

// TestBashExecTool_ReasonCodes pins the severity↔reason-code pairing for
// every escalation branch of the bash judge. Reason codes are a
// cross-repository contract (see tools.JudgeReasonCode): prose may be
// reworded freely, these pairs may not drift.
func TestBashExecTool_ReasonCodes(t *testing.T) {
	ws := t.TempDir()
	ctx := tools.WithWorkspacePath(context.Background(), ws)

	tool, err := NewBashExecTool([]string{`rm\s+-rf`})
	if err != nil {
		t.Fatalf("failed to construct tool: %v", err)
	}

	tests := []struct {
		name     string
		command  string
		wantSev  tools.JudgeSeverity
		wantCode tools.JudgeReasonCode
	}{
		{
			name:     "blacklist match is hard command_blacklist",
			command:  "rm -rf /",
			wantSev:  tools.JudgeSeverityHard,
			wantCode: tools.ReasonCodeCommandBlacklist,
		},
		{
			name:     "unresolvable path token is hard unresolvable_path_token",
			command:  "cat ~root/secret",
			wantSev:  tools.JudgeSeverityHard,
			wantCode: tools.ReasonCodeUnresolvablePathToken,
		},
		{
			name:     "existing out-of-root path is soft outside_session_roots",
			command:  "cat /etc/hosts",
			wantSev:  tools.JudgeSeveritySoft,
			wantCode: tools.ReasonCodeOutsideSessionRoots,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input, _ := json.Marshal(map[string]string{"command": tt.command})
			outcome := tool.Judge(ctx, input)
			if outcome.Allow {
				t.Fatalf("expected escalation (allow=false), got %+v", outcome)
			}
			if outcome.Severity != tt.wantSev {
				t.Errorf("severity = %v, want %v", outcome.Severity, tt.wantSev)
			}
			if outcome.ReasonCode != tt.wantCode {
				t.Errorf("reason code = %q, want %q", outcome.ReasonCode, tt.wantCode)
			}
		})
	}
}

// TestBashExecTool_Judge_ReboundVarIsHard closes the invisible-rebinding
// hole at the Judge layer: a variable rebound by a construct the static
// walker cannot see through ("read D < cfg" — attacker-controlled file
// content) must keep every "$D" reference UNASSESSABLE, escalating hard, so
// a decoy literal binding can never auto-approve the command.
func TestBashExecTool_Judge_ReboundVarIsHard(t *testing.T) {
	t.Setenv("D", "")
	tool, err := NewBashExecTool(nil)
	if err != nil {
		t.Fatalf("failed to construct tool: %v", err)
	}
	ws := t.TempDir()
	ctx := tools.WithWorkspacePath(context.Background(), ws)
	input, _ := json.Marshal(map[string]string{"command": `D=safe; read -r D < cfg; cat "$D/x"`})
	outcome := tool.Judge(ctx, input)
	if outcome.Allow {
		t.Fatal("expected rebound-var reference to be denied")
	}
	if !strings.Contains(outcome.Reason, "$D") {
		t.Fatalf("expected reason to name the unassessable token, got %q", outcome.Reason)
	}
	if outcome.Severity != tools.JudgeSeverityHard {
		t.Fatalf("rebound-var severity = %v, want hard", outcome.Severity)
	}
	if outcome.ReasonCode != tools.ReasonCodeUnresolvablePathToken {
		t.Fatalf("rebound-var reason code = %q, want %q", outcome.ReasonCode, tools.ReasonCodeUnresolvablePathToken)
	}
}

// TestBashExecTool_Judge_DeadBranchDecoyStillContained closes the
// empty-expansion masking hole at the Judge layer: a decoy binding in a
// branch that never executes ("if false; then D=safe; fi") must not mask the
// EMPTY expansion of the unset variable — "$D/etc/passwd" runtime-reads
// /etc/passwd, so the suffix-alone candidate must escalate containment.
func TestBashExecTool_Judge_DeadBranchDecoyStillContained(t *testing.T) {
	t.Setenv("D", "")
	tool, err := NewBashExecTool(nil)
	if err != nil {
		t.Fatalf("failed to construct tool: %v", err)
	}
	ws := t.TempDir()
	ctx := tools.WithWorkspacePath(context.Background(), ws)
	input, _ := json.Marshal(map[string]string{"command": `if false; then D=safe; fi; cat "$D/etc/passwd"`})
	outcome := tool.Judge(ctx, input)
	if outcome.Allow {
		t.Fatal("expected dead-branch decoy with absolute suffix to be denied")
	}
	if !strings.Contains(outcome.Reason, "outside session roots") {
		t.Fatalf("unexpected reason: %q", outcome.Reason)
	}
	if outcome.Severity != tools.JudgeSeveritySoft {
		t.Fatalf("dead-branch containment severity = %v, want soft", outcome.Severity)
	}
}

// TestBashExecTool_Judge_EnvClearingStaysContained closes the
// environment-clearing hole at the Judge layer: "env -i" and "exec -c" run
// the child with an environment the judge's process cannot observe, so a
// variable that holds an in-root value HERE expands to NOTHING there —
// "$WS/etc/passwd" runtime-reads /etc/passwd. sudo(8)'s default env_reset
// has the same shape. The whole command must go opaque and escalate hard
// instead of auto-approving on the in-root candidate.
func TestBashExecTool_Judge_EnvClearingStaysContained(t *testing.T) {
	tool, err := NewBashExecTool(nil)
	if err != nil {
		t.Fatalf("failed to construct tool: %v", err)
	}
	ws := t.TempDir()
	t.Setenv("WS", ws)
	ctx := tools.WithWorkspacePath(context.Background(), ws)
	for name, command := range map[string]string{
		"env -i":                   `env -i bash -c 'cat "$WS/etc/passwd"'`,
		"env --ignore-environment": `env --ignore-environment bash -c 'cat "$WS/etc/passwd"'`,
		"exec -c":                  `exec -c bash -c 'cat "$WS/etc/passwd"'`,
		"sudo":                     `sudo cat "$WS/etc/passwd"`,
		"su -l":                    `su -l root -c 'cat "$WS/etc/passwd"'`,
		"doas":                     `doas cat "$WS/etc/passwd"`,
		"ssh":                      `ssh host 'cat "$WS/etc/passwd"'`,
	} {
		t.Run(name, func(t *testing.T) {
			input, _ := json.Marshal(map[string]string{"command": command})
			outcome := tool.Judge(ctx, input)
			if outcome.Allow {
				t.Fatalf("expected %s command to be denied", name)
			}
			if outcome.Severity != tools.JudgeSeverityHard {
				t.Fatalf("%s severity = %v, want hard", name, outcome.Severity)
			}
			if outcome.ReasonCode != tools.ReasonCodeUnresolvablePathToken {
				t.Fatalf("%s reason code = %q, want %q", name, outcome.ReasonCode, tools.ReasonCodeUnresolvablePathToken)
			}
		})
	}
}
