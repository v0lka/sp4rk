//go:build windows

package llm

import (
	"net"
	"net/url"
	"os"
	"syscall"
	"testing"
)

// TestClassifyNetError_WSAECONNRefused is a Windows-only regression test for
// connection-refused retry classification. On Windows the TCP dial path does
// NOT produce the invented syscall.ECONNREFUSED sentinel; it produces the raw
// Winsock code WSAECONNREFUSED (10061) — visible in the error text as
// "connectex: No connection could be made because the target machine actively
// refused it." errors.Is(err, syscall.ECONNREFUSED) returns false for that, so
// a real connection-refused used to come back as non-retryable. This test
// verifies both the full classifyNetError path and the helper treat such an
// error as retryable.
func TestClassifyNetError_WSAECONNRefused(t *testing.T) {
	// A real-looking dial error carrying the raw Winsock errno.
	dialErr := &net.OpError{
		Op:  "dial",
		Net: "tcp",
		Err: &os.SyscallError{Syscall: "connectex", Err: wsaECONNREFUSED},
	}
	if !classifyNetError(dialErr) {
		t.Errorf("classifyNetError(WSAECONNREFUSED dial error) = false, want true")
	}

	// Every errno the Windows helper promises to match must classify as
	// transient — both the raw Winsock codes (real failures) and the invented
	// POSIX sentinels (parity with the Unix helper and tests that build them).
	matched := []syscall.Errno{
		syscall.WSAECONNRESET, syscall.WSAECONNABORTED,
		wsaECONNREFUSED, wsaEHOSTUNREACH,
		syscall.ECONNREFUSED, syscall.ECONNRESET, syscall.EHOSTUNREACH,
	}
	for _, errno := range matched {
		if !isTransientConnErrno(errno) {
			t.Errorf("isTransientConnErrno(%d) = false, want true", errno)
		}
	}
}

// TestTransportKindErrno_WinsockClassifies is a Windows-only regression test
// for HTTP-0 transport classification: real Winsock failures carry raw WSA*
// codes, not the invented POSIX sentinels, so without the raw-code cases in
// transportKindForErrno every one of them fell through to the url.Error
// fallback and misclassified as "connect" — a mid-flight reset as "connect"
// instead of "reset".
func TestTransportKindErrno_WinsockClassifies(t *testing.T) {
	tests := []struct {
		name  string
		errno syscall.Errno
		want  string
	}{
		{"WSAECONNRESET", syscall.WSAECONNRESET, "reset"},
		{"WSAECONNABORTED", syscall.WSAECONNABORTED, "reset"},
		{"WSAECONNREFUSED", wsaECONNREFUSED, "connect"},
		{"WSAEHOSTUNREACH", wsaEHOSTUNREACH, "connect"},
		{"WSAENETUNREACH", wsaENETUNREACH, "connect"},
		{"WSAETIMEDOUT", wsaETIMEDOUT, "connect"},
		{"ECONNRESET sentinel", syscall.ECONNRESET, "reset"},
		{"ECONNREFUSED sentinel", syscall.ECONNREFUSED, "connect"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := transportKindForErrno(tt.errno); got != tt.want {
				t.Errorf("transportKindForErrno(%s = %d) = %q, want %q", tt.name, tt.errno, got, tt.want)
			}
		})
	}

	// End-to-end: the exact shape a mid-call reset surfaces in — *url.Error
	// wrapping *net.OpError wrapping the raw Winsock errno — must classify as
	// "reset", not fall through to the url.Error "connect" fallback.
	resetErr := &url.Error{
		Op:  "Post",
		URL: "https://api.example.com/v1/chat/completions",
		Err: &net.OpError{Op: "read", Net: "tcp", Err: syscall.WSAECONNRESET},
	}
	if got := TransportKind(resetErr); got != "reset" {
		t.Errorf("TransportKind(WSAECONNRESET url.Error) = %q, want reset", got)
	}
	refusedErr := &url.Error{
		Op:  "Post",
		URL: "https://api.example.com/v1/chat/completions",
		Err: &net.OpError{Op: "dial", Net: "tcp", Err: wsaECONNREFUSED},
	}
	if got := TransportKind(refusedErr); got != "connect" {
		t.Errorf("TransportKind(WSAECONNREFUSED url.Error) = %q, want connect", got)
	}
}
