// Package safeio provides hardened, non-blocking filesystem opens: a path is
// opened only after — or while — confirming that it is a regular file, so a
// FIFO, socket, or device planted at an open path can never block the caller.
//
// A read-open (O_RDONLY) of a FIFO blocks until a writer appears, and a
// write-open (O_WRONLY) blocks until a reader appears; both blocks happen
// inside the open(2) syscall, where no context check, deadline, or timeout can
// interrupt them. A single leftover named pipe beneath a directory an
// application reads, walks, or is pointed at is therefore enough to hang a
// user-facing RPC, a background pass, or a shutdown join indefinitely — the
// reported beach-ball freeze. Route every open of a workspace- or
// user-controlled path through this package so that class of hang is closed in
// one place rather than per call site.
//
// On unix the open itself is O_NONBLOCK and regularity is decided by fstat on
// the already-open descriptor, with no Stat→Open window a racing local
// adversary could exploit (see Open and OpenFile). On Windows named pipes live
// in the \\.\pipe\ namespace and cannot appear at a filesystem path, so a plain
// open plus a handle-based regularity check is sufficient (see Open and
// OpenFile).
package safeio

import (
	"errors"
	"io"
	"io/fs"
)

// ErrNotRegular marks a path that is not a regular file — a FIFO, socket,
// device, or directory — opened for reading or writing. Open, OpenFile, and
// ReadFile report it via a *NotRegularError; callers that must distinguish it
// from a genuine open/read failure test it with errors.Is(err, ErrNotRegular).
var ErrNotRegular = errors.New("not a regular file")

// NotRegularError reports that a path exists but is not a regular file. Its
// message preserves the historical "<path> is not a regular file" wording used
// by the git-config scanner, and it matches errors.Is(err, ErrNotRegular).
type NotRegularError struct {
	Path string
}

func (e *NotRegularError) Error() string { return e.Path + " is not a regular file" }

// Is makes errors.Is(err, ErrNotRegular) report true for any NotRegularError.
func (e *NotRegularError) Is(target error) bool { return target == ErrNotRegular }

// ReadFile reads the named file, asserting first that it is a regular file. It
// is the safe replacement for os.ReadFile on any path that could be a
// non-regular file: a FIFO is refused instead of blocking the open forever.
//
// A missing file returns the underlying open error (testable with
// errors.Is(err, fs.ErrNotExist)); a present-but-non-regular path returns a
// *NotRegularError.
func ReadFile(path string) ([]byte, error) {
	f, err := Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()
	data, err := io.ReadAll(f)
	if err != nil {
		return nil, &fs.PathError{Op: "read", Path: path, Err: err}
	}
	return data, nil
}

// IsRegular reports whether path names a regular file. A FIFO, socket, device,
// or directory reports (false, nil); a path that cannot be opened (missing,
// permission denied) reports (false, err). Open refuses a non-regular target
// via fstat on the already-open descriptor, so the probe itself never blocks.
func IsRegular(path string) (bool, error) {
	f, err := Open(path)
	if err != nil {
		if errors.Is(err, ErrNotRegular) {
			return false, nil
		}
		return false, err
	}
	_ = f.Close()
	return true, nil
}
