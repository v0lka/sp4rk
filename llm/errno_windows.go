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
// with their real values, so they are referenced by name. WSAECONNREFUSED,
// WSAEHOSTUNREACH, WSAENETUNREACH and WSAETIMEDOUT are not exported, so they
// are defined here.
const (
	wsaECONNREFUSED syscall.Errno = 10061 // WSAECONNREFUSED: connection refused
	wsaEHOSTUNREACH syscall.Errno = 10065 // WSAEHOSTUNREACH: no route to host
	wsaENETUNREACH  syscall.Errno = 10051 // WSAENETUNREACH: network unreachable
	wsaETIMEDOUT    syscall.Errno = 10060 // WSAETIMEDOUT: connection timed out
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

// transportKindForErrno maps a raw syscall.Errno to the TransportKind
// diagnostic label ("reset" or "connect"), or "" when the errno carries no
// transport signal. It mirrors isTransientConnErrno's two families: the raw
// Winsock codes that real Windows network failures carry, plus the invented
// POSIX sentinels for parity with the Unix helper and directly constructed
// errors. Without the raw-WSA cases every real HTTP-0 failure would fall
// through to TransportKind's url.Error fallback and misclassify as
// "connect" (e.g. a mid-flight reset, WSAECONNRESET 10054, as "connect"
// instead of "reset"). Note wsaETIMEDOUT labels "connect", matching the
// POSIX switch's ETIMEDOUT arm: Errno.Timeout() only recognizes the
// invented Windows sentinel, so an OS-level Winsock connect timeout reaches
// this helper and gets the coarser "connect" label on Windows where Unix
// reports "timeout".
func transportKindForErrno(errno syscall.Errno) string {
	switch errno {
	case syscall.ECONNRESET, syscall.EPIPE,
		syscall.WSAECONNRESET, syscall.WSAECONNABORTED:
		return "reset"
	case syscall.ECONNREFUSED, syscall.EHOSTUNREACH, syscall.ENETUNREACH,
		syscall.ETIMEDOUT, syscall.ECONNABORTED,
		wsaECONNREFUSED, wsaEHOSTUNREACH, wsaENETUNREACH, wsaETIMEDOUT:
		return "connect"
	}
	return ""
}
