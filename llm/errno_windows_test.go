//go:build windows

package llm

import (
	"net"
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
