//go:build windows

package sysproc

import (
	"os/exec"
	"syscall"
	"testing"
)

func TestHideConsole_Windows(t *testing.T) {
	cmd := exec.Command("cmd", "/c", "echo test")

	HideConsole(cmd)

	if cmd.SysProcAttr == nil {
		t.Fatal("HideConsole must initialize SysProcAttr on Windows")
	}
	if cmd.SysProcAttr.CreationFlags&createNoWindow == 0 {
		t.Error("HideConsole must set CREATE_NO_WINDOW in CreationFlags on Windows")
	}
}

func TestHideConsole_PreservesExistingFlags(t *testing.T) {
	const newProcessGroup = 0x00000200
	cmd := exec.Command("cmd", "/c", "echo test")
	cmd.SysProcAttr = &syscall.SysProcAttr{
		CreationFlags: newProcessGroup,
	}

	HideConsole(cmd)

	if cmd.SysProcAttr.CreationFlags&newProcessGroup == 0 {
		t.Error("HideConsole must preserve existing CreationFlags (CREATE_NEW_PROCESS_GROUP)")
	}
	if cmd.SysProcAttr.CreationFlags&createNoWindow == 0 {
		t.Error("HideConsole must add CREATE_NO_WINDOW to existing CreationFlags")
	}
}
