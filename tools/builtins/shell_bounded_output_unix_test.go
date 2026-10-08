//go:build !windows

package builtins

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

// TestBashExecTool_ZeroTimeoutsConstructor is the regression test for review
// finding 31: a zero-value BashTimeouts used to derive a zero per-call
// deadline, cancelling every command before it started ("context deadline
// exceeded" from CombinedOutput without the command ever running).
func TestBashExecTool_ZeroTimeoutsConstructor(t *testing.T) {
	tool, err := NewBashExecToolWithTimeouts(nil, BashTimeouts{})
	if err != nil {
		t.Fatalf("NewBashExecToolWithTimeouts(BashTimeouts{{}}): %v", err)
	}

	input, err := json.Marshal(map[string]string{"command": "echo ok"})
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

// TestBashExecTool_OutputCap is the regression test for review finding 5:
// combined child output must be buffered under a byte cap — a command that
// emits far more than the cap must not grow the buffer unboundedly, and the
// result must carry an incompleteness marker naming the cap.
func TestBashExecTool_OutputCap(t *testing.T) {
	tool := mustNewBashExecTool(t, nil)

	// Emit 12 MiB — 2 MiB past the 10 MiB cap.
	input, err := json.Marshal(map[string]string{
		"command": "head -c 12582912 /dev/zero | tr '\\0' 'a'",
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
	// The stored prefix is capped; only the marker may extend the result a
	// little past it.
	if len(result.Content) > maxShellOutputBytes+256 {
		t.Errorf("result length = %d, want <= cap (%d) + marker overhead", len(result.Content), maxShellOutputBytes)
	}
}

// TestBashExecTool_UnderCapOutputUnchanged verifies the cap does not alter
// results below it (no spurious marker).
func TestBashExecTool_UnderCapOutputUnchanged(t *testing.T) {
	tool := mustNewBashExecTool(t, nil)

	input, err := json.Marshal(map[string]string{"command": "echo hello"})
	if err != nil {
		t.Fatalf("marshal input: %v", err)
	}

	result, err := tool.Execute(context.Background(), input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Content != "hello\n" {
		t.Errorf("expected content %q, got %q", "hello\n", result.Content)
	}
	if strings.Contains(result.Content, "output truncated") {
		t.Errorf("sub-cap output must not carry a truncation marker: %q", result.Content)
	}
}

// TestBashExecTool_NegativeRequestParamTimeout guards the Execute-side
// fallback: a negative requested timeout used to flow into
// context.WithTimeout verbatim (an already-expired context).
func TestBashExecTool_NegativeRequestParamTimeout(t *testing.T) {
	tool := mustNewBashExecTool(t, nil)

	input, err := json.Marshal(map[string]string{"command": "echo ok", "timeout": "-5s"})
	if err != nil {
		t.Fatalf("marshal input: %v", err)
	}

	result, err := tool.Execute(context.Background(), input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Fatalf("negative requested timeout broke the call: %s", result.Content)
	}
	if !strings.Contains(result.Content, "ok") {
		t.Errorf("expected output to contain 'ok', got %q", result.Content)
	}
}
