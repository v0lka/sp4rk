//go:build windows

package builtins

import (
	"testing"

	"github.com/v0lka/sp4rk/tools"
)

// TestPoshExecTool_Group pins the Windows-only posh_exec tool to the execute
// group; the all-builtins iteration test in the root sp4rk package cannot
// construct it on non-Windows platforms.
func TestPoshExecTool_Group(t *testing.T) {
	tool, err := NewPoshExecTool(nil)
	if err != nil {
		t.Fatalf("failed to construct tool: %v", err)
	}
	if got := tool.Group(); got != tools.GroupExecute {
		t.Fatalf("posh_exec group = %q, want %q", got, tools.GroupExecute)
	}
}
