//go:build unix

package safeio

import (
	"io/fs"
	"os"
	"syscall"
)

// Open opens path for reading without ever blocking and without a Stat→Open
// window: the open itself is non-blocking, and regularity is verified by fstat
// on the ALREADY-OPEN descriptor, so what is read is exactly what was checked.
//
// A FIFO planted at path is refused by the fstat instead of hanging the open(2)
// syscall — a read-only O_NONBLOCK open of a FIFO returns immediately even with
// no writer, so no writer is ever awaited — and a FIFO swapped in between a
// stat and an open by a racing local adversary cannot defeat the guard the way
// the earlier Stat→Open pair could (a TOCTOU).
//
// A non-regular target returns a *NotRegularError; a missing file returns an
// *fs.PathError wrapping the underlying ENOENT.
func Open(path string) (*os.File, error) {
	fd, err := syscall.Open(path, syscall.O_RDONLY|syscall.O_NONBLOCK|syscall.O_CLOEXEC, 0)
	if err != nil {
		return nil, &fs.PathError{Op: "open", Path: path, Err: err}
	}
	if err := checkRegularFD(fd, path); err != nil {
		return nil, err
	}
	return os.NewFile(uintptr(fd), path), nil
}

// OpenFile opens path with the caller's flags and permission bits, refusing a
// non-regular pre-existing target so a write-open can never block. It is the
// safe replacement for os.OpenFile on any path that could be a non-regular
// file.
//
// O_NONBLOCK is OR-ed into the caller's flags before the open: a write-open
// (O_WRONLY) of a FIFO without a reader otherwise blocks inside open(2) until a
// reader appears — uninterruptible. With O_NONBLOCK that open returns ENXIO
// immediately (reported as an *fs.PathError), and when a reader IS present (or
// the flags include O_RDWR) the open succeeds and the post-open fstat refuses
// the FIFO with a *NotRegularError. Either way the caller never waits. On a
// regular file O_NONBLOCK has no effect, so the returned descriptor behaves
// like one from os.OpenFile.
//
// A newly created target (O_CREATE) is a regular file and is returned; a
// non-regular pre-existing target returns a *NotRegularError; any open failure
// (missing without O_CREATE, permission denied, a directory) returns an
// *fs.PathError.
func OpenFile(path string, flag int, perm fs.FileMode) (*os.File, error) {
	fd, err := syscall.Open(path, flag|syscall.O_NONBLOCK|syscall.O_CLOEXEC, uint32(perm.Perm()))
	if err != nil {
		return nil, &fs.PathError{Op: "open", Path: path, Err: err}
	}
	if err := checkRegularFD(fd, path); err != nil {
		return nil, err
	}
	return os.NewFile(uintptr(fd), path), nil
}

// checkRegularFD verifies by fstat that the already-open descriptor fd names a
// regular file. On a non-regular descriptor it closes fd and returns a
// *NotRegularError; on an fstat failure it closes fd and returns an
// *fs.PathError. On success it returns nil and leaves fd open for the caller.
func checkRegularFD(fd int, path string) error {
	var st syscall.Stat_t
	if err := syscall.Fstat(fd, &st); err != nil {
		_ = syscall.Close(fd)
		return &fs.PathError{Op: "open", Path: path, Err: err}
	}
	if st.Mode&syscall.S_IFMT != syscall.S_IFREG {
		_ = syscall.Close(fd)
		return &NotRegularError{Path: path}
	}
	return nil
}
