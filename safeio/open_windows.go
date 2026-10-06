//go:build windows

package safeio

import (
	"io/fs"
	"os"
)

// Open opens path for reading and verifies, via the open handle's fstat, that
// it is a regular file.
//
// Windows named pipes live in the \\.\pipe\ namespace and cannot appear at a
// filesystem path, so the unix O_NONBLOCK dance is unnecessary here. The
// handle-based regularity check still closes the Stat→Open TOCTOU: what is
// opened is what is checked, and only regular files are ever read.
//
// A non-regular target returns a *NotRegularError; an open failure (including a
// missing file) returns os.Open's *fs.PathError unchanged.
func Open(path string) (*os.File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	if err := checkRegularHandle(f, path); err != nil {
		return nil, err
	}
	return f, nil
}

// OpenFile opens path with the caller's flags and permission bits, refusing a
// non-regular pre-existing target so a write-open can never block. It is the
// safe replacement for os.OpenFile on any path that could be a non-regular
// file.
//
// Named pipes cannot appear at a filesystem path on Windows, so no non-blocking
// open dance is needed; the handle-based regularity check is sufficient. A
// newly created target (O_CREATE) is a regular file and is returned; a
// non-regular pre-existing target returns a *NotRegularError; any open failure
// returns an *fs.PathError.
func OpenFile(path string, flag int, perm fs.FileMode) (*os.File, error) {
	f, err := os.OpenFile(path, flag, perm)
	if err != nil {
		return nil, err
	}
	if err := checkRegularHandle(f, path); err != nil {
		return nil, err
	}
	return f, nil
}

// checkRegularHandle verifies by fstat that the open handle f names a regular
// file. On a non-regular handle it closes f and returns a *NotRegularError; on
// a stat failure it closes f and returns an *fs.PathError. On success it
// returns nil and leaves f open for the caller.
func checkRegularHandle(f *os.File, path string) error {
	st, err := f.Stat()
	if err != nil {
		_ = f.Close()
		return &fs.PathError{Op: "open", Path: path, Err: err}
	}
	if !st.Mode().IsRegular() {
		_ = f.Close()
		return &NotRegularError{Path: path}
	}
	return nil
}
