//go:build windows

package sysproc

import (
	"context"
	"os/exec"
	"strconv"
	"syscall"
	"time"
)

// createNoWindow is the Windows process creation flag CREATE_NO_WINDOW
// (0x08000000). It prevents the child process from allocating a console
// window, which is essential when a GUI-subsystem application spawns helper
// processes (shell tools, ripgrep, MCP servers, …) — otherwise each child
// flashes or leaves open a console window on screen.
//
// Note: Go's syscall.SysProcAttr.HideWindow field is not sufficient here; it
// hides a window that has already been created via ShowWindow(SW_HIDE) and
// still allows a brief flash. CREATE_NO_WINDOW suppresses allocation entirely.
const createNoWindow = 0x08000000

// HideConsole configures cmd so the child process does not allocate a visible
// console window. It preserves any CreationFlags already set by the caller
// (e.g. CREATE_NEW_PROCESS_GROUP) by OR-ing CREATE_NO_WINDOW into them. On
// non-Windows platforms this is a no-op (see sysproc_unix.go).
//
// HideConsole must be called before cmd.Start/Run; mutating SysProcAttr after
// the process has started has no effect.
func HideConsole(cmd *exec.Cmd) {
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.CreationFlags |= createNoWindow
}

// SetProcessGroup is a no-op on Windows. A "process group" is a POSIX concept
// (see the Unix implementation, where it is the target of a whole-tree signal);
// Windows containment instead uses Job Objects or a tree kill, performed by
// KillTree. It exists so cross-platform callers can configure a command
// unconditionally before cmd.Start.
func SetProcessGroup(_ *exec.Cmd) {}

// KillTree force-kills cmd and the descendant process tree it spawned. It is
// the last-resort reaper for a child that ignores every graceful shutdown
// request — e.g. a launcher-spawned MCP server (npx → node) that does not exit
// on stdin EOF: a bare cmd.Process.Kill terminates only the launcher and
// orphans the server, which keeps running holding its pipes.
//
// It uses taskkill /T, which terminates the process AND the children Windows
// currently records under it (the analogue of the Unix process-group kill). The
// kill-on-close Job Object helper (AssignKillOnCloseJob) cannot be used here:
// job membership is inherited only at process creation, so a job attached after
// the fact would miss the launched server. The MCP stdio transport owns
// cmd.Start, leaving no point at which the job could be attached before the
// launcher forks its server.
func KillTree(cmd *exec.Cmd) error {
	if cmd == nil || cmd.Process == nil {
		return nil
	}
	// Bound taskkill: this runs on a close path (often during shutdown), so it
	// must not be able to stall on a wedged helper. taskkill /F is immediate in
	// practice; the bound is a safety net.
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	tree := exec.CommandContext(ctx, "taskkill", "/T", "/F", "/PID", strconv.Itoa(cmd.Process.Pid))
	HideConsole(tree)
	if err := tree.Run(); err != nil {
		// taskkill unavailable or the tree already gone: fall back to the
		// direct process kill (never worse than the previous behavior).
		if killErr := cmd.Process.Kill(); killErr != nil {
			return err
		}
	}
	return nil
}
