//go:build !windows

package llm

import "syscall"

// isTransientConnErrno reports whether a raw syscall.Errno represents a
// transient connection-level failure that is safe to retry. On Unix-like
// systems the POSIX errno sentinels (ECONNREFUSED, ECONNRESET, EHOSTUNREACH)
// are the real OS errno values, so a direct comparison is authoritative.
//
// The Windows variant (errors_windows.go) bridges a long-standing gap: on
// Windows the syscall package's POSIX errno names are invented values that
// never occur in real network errors, so it matches the raw Winsock codes
// instead.
func isTransientConnErrno(errno syscall.Errno) bool {
	switch errno {
	case syscall.ECONNREFUSED, syscall.ECONNRESET, syscall.EHOSTUNREACH:
		return true
	}
	return false
}
