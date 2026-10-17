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

// transportKindForErrno maps a raw syscall.Errno to the TransportKind
// diagnostic label ("reset" or "connect"), or "" when the errno carries no
// transport signal. On Unix-like systems the POSIX errno sentinels
// (ECONNRESET, ECONNREFUSED, …) are the real OS errno values, so a direct
// comparison is authoritative.
func transportKindForErrno(errno syscall.Errno) string {
	switch errno {
	case syscall.ECONNRESET, syscall.EPIPE:
		return "reset"
	case syscall.ECONNREFUSED, syscall.EHOSTUNREACH, syscall.ENETUNREACH,
		syscall.ETIMEDOUT, syscall.ECONNABORTED:
		return "connect"
	}
	return ""
}
