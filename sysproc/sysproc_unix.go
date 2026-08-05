//go:build !windows

package sysproc

import "os/exec"

// HideConsole is a no-op on non-Windows platforms, where spawning a child
// process never creates a console window. It exists so callers can apply it
// unconditionally without their own platform build tags.
func HideConsole(_ *exec.Cmd) {}

// AssignKillOnCloseJob is a no-op on non-Windows platforms, where the OS does
// not use Job Objects. It returns a no-op cleanup and a nil error so
// cross-platform callers can defer it unconditionally, mirroring HideConsole.
func AssignKillOnCloseJob(_ *exec.Cmd) (cleanup func(), err error) {
	return func() {}, nil
}
