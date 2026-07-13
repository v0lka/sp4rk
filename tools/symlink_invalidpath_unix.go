//go:build !windows

package tools

// isInvalidPathErrorOS adds platform-specific invalid-path detection on top
// of the POSIX errno checks in isInvalidPathError. On non-Windows platforms
// the POSIX errnos (ENAMETOOLONG, ENOTDIR, EINVAL) are sufficient, so this is
// a no-op.
func isInvalidPathErrorOS(_ error) bool {
	return false
}
