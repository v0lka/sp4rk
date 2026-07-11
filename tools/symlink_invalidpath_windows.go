//go:build windows

package tools

import (
	"errors"
	"syscall"
)

// Windows system error codes that mark a candidate string as an invalid
// filesystem path rather than a genuine unreadable symlink. Go's syscall
// package does not export these values. os.Lstat surfaces them as raw
// syscall.Errno values when a path component is too long or syntactically
// malformed — cases that do not match the invented POSIX errnos checked in
// isInvalidPathError (on Windows ENAMETOOLONG and EINVAL are
// APPLICATION_ERROR+offset; see zerrors_windows.go).
const (
	// windowsErrorInvalidName is ERROR_INVALID_NAME (123): "The filename,
	// directory name, or volume label syntax is incorrect." os.Lstat returns
	// it for a component exceeding NAME_MAX or with invalid syntax.
	windowsErrorInvalidName syscall.Errno = 123
	// windowsErrorFilenameTooLong is ERROR_FILENAME_EXCED_RANGE (206): "The
	// filename or extension is too long."
	windowsErrorFilenameTooLong syscall.Errno = 206
)

// isInvalidPathErrorOS adds Windows-specific invalid-path detection on top of
// the POSIX errno checks in isInvalidPathError. errors.Is unwraps the
// *fs.PathError returned by os.Lstat and compares the inner syscall.Errno to
// the constants above.
func isInvalidPathErrorOS(err error) bool {
	if errors.Is(err, windowsErrorInvalidName) {
		return true
	}
	if errors.Is(err, windowsErrorFilenameTooLong) {
		return true
	}
	return false
}
