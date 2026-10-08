//go:build windows

package builtins

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

// TestPoshExecTool_ZeroTimeoutsConstructor is the Windows regression test for
// review finding 31: a zero-value BashTimeouts used to derive a zero per-call
// deadline, cancelling every command before it started.
func TestPoshExecTool_ZeroTimeoutsConstructor(t *testing.T) {
	tool, err := NewPoshExecToolWithTimeouts(nil, BashTimeouts{})
	if err != nil {
		t.Fatalf("NewPoshExecToolWithTimeouts(BashTimeouts{{}}): %v", err)
	}

	input, err := json.Marshal(map[string]string{"command": "Write-Output ok"})
	if err != nil {
		t.Fatalf("marshal input: %v", err)
	}

	result, err := tool.Execute(context.Background(), input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Fatalf("zero-value timeouts made the tool non-functional: %s", result.Content)
	}
	if !strings.Contains(result.Content, "ok") {
		t.Errorf("expected output to contain 'ok', got %q", result.Content)
	}
}

// TestPoshExecTool_OutputCap is the Windows regression test for review
// finding 5: combined child output must be buffered under a byte cap, and the
// result must carry an incompleteness marker naming the cap.
func TestPoshExecTool_OutputCap(t *testing.T) {
	tool := mustNewPoshExecTool(t, nil)

	// Emit 12 MiB — 2 MiB past the 10 MiB cap.
	input, err := json.Marshal(map[string]string{
		"command": "Write-Output ([string]::new('a', 12582912))",
		"timeout": "30s",
	})
	if err != nil {
		t.Fatalf("marshal input: %v", err)
	}

	result, err := tool.Execute(context.Background(), input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Fatalf("unexpected error result: %s", result.Content)
	}
	if !strings.Contains(result.Content, "output truncated") {
		t.Errorf("expected a truncation marker in the result, got %d bytes without one", len(result.Content))
	}
	if len(result.Content) > maxShellOutputBytes+256 {
		t.Errorf("result length = %d, want <= cap (%d) + marker overhead", len(result.Content), maxShellOutputBytes)
	}
}
