//go:build windows

package tools

import (
	"io/fs"
	"syscall"
	"testing"
)

// TestIsInvalidPathError_WindowsErrno pins the Windows system error codes that
// os.Lstat surfaces for an invalid or over-long path component. On Windows the
// POSIX errnos are invented values (APPLICATION_ERROR+offset) that never match
// these raw codes, so isInvalidPathErrorOS must recognize them directly. The
// real-OS test TestIsInvalidPathError/ENAMETOOLONG_from_real_Lstat exercises
// this end to end via os.Lstat; this test fixes the exact errno mapping
// deterministically so it cannot silently regress across Windows versions.
func TestIsInvalidPathError_WindowsErrno(t *testing.T) {
	cases := []struct {
		name   string
		errno  syscall.Errno
		expect bool
	}{
		{name: "ERROR_INVALID_NAME is invalid", errno: windowsErrorInvalidName, expect: true},
		{name: "ERROR_FILENAME_EXCED_RANGE is invalid", errno: windowsErrorFilenameTooLong, expect: true},
		{name: "ENOENT is not invalid", errno: syscall.ENOENT, expect: false},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := &fs.PathError{Op: "CreateFile", Path: `C:\bad`, Err: tc.errno}
			got := isInvalidPathError(err)
			if got != tc.expect {
				t.Errorf("isInvalidPathError(%v) = %v, want %v", err, got, tc.expect)
			}
		})
	}
}
