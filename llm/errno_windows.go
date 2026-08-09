//go:build windows

package llm

import "syscall"

// On Windows the syscall package exports POSIX-style errno constants that are
// pure inventions: in zerrors_windows.go, ECONNREFUSED, EHOSTUNREACH and the
// rest are defined as (1<<29)+iota and NEVER appear in real network errors.
// The actual failures surface as raw Winsock (WSA*) codes, most of which the
// syscall package does not export. As a result, errors.Is(err,
// syscall.ECONNREFUSED) always returns false for real dial failures on
// Windows, which silently broke retry classification (e.g. the redacted
// *url.Error path in provider_google.go) — a connection-refused error came
// back as non-retryable.
//
// The fix is to match the raw WSA errno values directly. WSAECONNRESET and
// WSAECONNABORTED are the exceptions: the syscall package DOES export them
// with their real values, so they are referenced by name. WSAECONNREFUSED and
// WSAEHOSTUNREACH are not exported, so they are defined here.
const (
	wsaECONNREFUSED syscall.Errno = 10061 // WSAECONNREFUSED: connection refused
	wsaEHOSTUNREACH syscall.Errno = 10065 // WSAEHOSTUNREACH: no route to host
)

// isTransientConnErrno reports whether a raw syscall.Errno represents a
// transient connection-level failure that is safe to retry. Two families of
// values are matched on Windows:
//
//  1. The raw Winsock (WSA*) codes that the dial path actually produces —
//     these are what real network failures carry.
//  2. The invented POSIX errno sentinels exported by the syscall package
//     (ECONNREFUSED, ECONNRESET, EHOSTUNREACH). These never occur in real
//     Windows network errors, but they are matched for parity with the Unix
//     helper and so that code or tests constructing them directly classify
//     intuitively.
func isTransientConnErrno(errno syscall.Errno) bool {
	switch errno {
	case syscall.WSAECONNRESET, syscall.WSAECONNABORTED, wsaECONNREFUSED, wsaEHOSTUNREACH,
		syscall.ECONNREFUSED, syscall.ECONNRESET, syscall.EHOSTUNREACH:
		return true
	}
	return false
}
