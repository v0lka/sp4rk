//go:build !windows

package sysproc

import (
	"context"
	"os/exec"
	"testing"
)

func TestHideConsole(t *testing.T) {
	cmd := exec.CommandContext(context.Background(), "echo", "test")

	// HideConsole must not crash when called before Start/Run on any platform.
	HideConsole(cmd)

	// On non-Windows platforms, HideConsole is a no-op and must not
	// modify SysProcAttr.
	if cmd.SysProcAttr != nil {
		t.Error("HideConsole must not modify SysProcAttr on non-Windows platforms")
	}
}

func TestAssignKillOnCloseJobNoOp(t *testing.T) {
	cmd := exec.CommandContext(context.Background(), "echo", "test")

	// AssignKillOnCloseJob must be a no-op on non-Windows platforms: a nil
	// error and a safe-to-call cleanup, callable even before Start.
	cleanup, err := AssignKillOnCloseJob(cmd)
	if err != nil {
		t.Fatalf("AssignKillOnCloseJob returned error on non-Windows: %v", err)
	}
	if cleanup == nil {
		t.Fatal("AssignKillOnCloseJob returned nil cleanup on non-Windows")
	}
	// cleanup must be safely callable and idempotent.
	cleanup()
	cleanup()
}
