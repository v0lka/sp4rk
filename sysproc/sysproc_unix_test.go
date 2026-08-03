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
