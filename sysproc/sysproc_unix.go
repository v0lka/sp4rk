//go:build !windows

package sysproc

import (
	"os/exec"
	"syscall"
)

// HideConsole is a no-op on non-Windows platforms, where spawning a child
// process never creates a console window. It exists so callers can apply it
// unconditionally without their own platform build tags.
func HideConsole(_ *exec.Cmd) {}

// SetProcessGroup places cmd in its own process group so the whole tree it
// spawns can later be signalled as one (see KillTree). Without it a launcher
// (npx, pnpm dlx, bunx, …) and the server it spawns are separate targets and a
// kill reaches only the launcher, orphaning the server.
//
// It must be called before cmd.Start/Run — SysProcAttr only takes effect at
// process creation. On Windows this is a no-op (see sysproc_windows.go), so
// callers apply it unconditionally.
func SetProcessGroup(cmd *exec.Cmd) {
	if cmd == nil {
		return
	}
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.Setpgid = true
}

// KillTree force-kills cmd and every process in its process group. It is the
// last-resort reaper for a child that ignores every graceful shutdown request —
// e.g. a launcher-spawned MCP server (npx → node) that does not exit on stdin
// EOF: a bare cmd.Process.Kill terminates only the launcher and orphans the
// server, which keeps running holding its pipes.
//
// The signal targets the process group made at spawn time by SetProcessGroup
// (the child is its group leader), so it reaches the launcher and every
// descendant that did not leave the group. If the group signal fails — e.g. the
// child already exited and its pid was reused — it falls back to killing just
// the process (never worse than the previous behavior).
func KillTree(cmd *exec.Cmd) error {
	if cmd == nil || cmd.Process == nil {
		return nil
	}
	if err := syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL); err != nil {
		return cmd.Process.Kill()
	}
	return nil
}

// AssignKillOnCloseJob is a no-op on non-Windows platforms, where the OS does
// not use Job Objects. It returns a no-op cleanup and a nil error so
// cross-platform callers can defer it unconditionally, mirroring HideConsole.
func AssignKillOnCloseJob(_ *exec.Cmd) (cleanup func(), err error) {
	return func() {}, nil
}
