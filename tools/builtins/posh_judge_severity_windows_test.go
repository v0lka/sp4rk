//go:build windows

package builtins

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/v0lka/sp4rk/tools"
)

// posh_exec severity classification, the Windows counterpart of
// bash_judge_severity_test.go (both shell Judges share structure).

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
