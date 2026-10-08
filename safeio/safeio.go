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
	"fmt"
	"io"
	"io/fs"
	"math"
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

// DefaultMaxFileSize is the read size cap ReadFile enforces: files larger than
// this many bytes are refused instead of being read into memory. It is sized
// well above any legitimate SKILL.md, AGENT.md, or skill-resource document
// while still bounding a single read to a sane allocation. Use ReadFileLimited
// to impose a different cap.
const DefaultMaxFileSize = int64(10 << 20) // 10 MiB

// ErrTooLarge marks a file that exceeds the read size cap enforced by ReadFile
// and ReadFileLimited. Callers that must distinguish it from a genuine
// open/read failure test it with errors.Is(err, ErrTooLarge).
var ErrTooLarge = errors.New("file too large")

// TooLargeError reports that a path exists and is a regular file but exceeds
// the read size cap. It matches errors.Is(err, ErrTooLarge).
type TooLargeError struct {
	Path  string
	Limit int64
}

// Error returns the refusal message, including the path and the byte cap.
func (e *TooLargeError) Error() string {
	return fmt.Sprintf("%s is too large: exceeds the %d byte read limit", e.Path, e.Limit)
}

// Is makes errors.Is(err, ErrTooLarge) report true for any TooLargeError.
func (e *TooLargeError) Is(target error) bool { return target == ErrTooLarge }

// ReadFile reads the named file, asserting first that it is a regular file. It
// is the safe replacement for os.ReadFile on any path that could be a
// non-regular file: a FIFO is refused instead of blocking the open forever,
// and a file larger than DefaultMaxFileSize bytes is refused instead of being
// read into memory.
//
// A missing file returns the underlying open error (testable with
// errors.Is(err, fs.ErrNotExist)); a present-but-non-regular path returns a
// *NotRegularError; an oversized path returns a *TooLargeError. Use
// ReadFileLimited to read with a different size cap.
func ReadFile(path string) ([]byte, error) {
	return ReadFileLimited(path, DefaultMaxFileSize)
}

// ReadFileLimited reads the named file like ReadFile but refuses files larger
// than limit bytes with a *TooLargeError rather than reading them into memory.
// A file of exactly limit bytes is read in full. A negative limit is treated
// as zero.
func ReadFileLimited(path string, limit int64) ([]byte, error) {
	if limit < 0 {
		limit = 0
	}
	f, err := Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()
	// Read up to limit+1 bytes so an oversized file is detected without
	// buffering more than one byte past the cap. The guard keeps limit+1 from
	// overflowing: a wrapped-negative LimitReader would silently return EOF.
	var r io.Reader = f
	if limit < math.MaxInt64 {
		r = io.LimitReader(f, limit+1)
	}
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, &fs.PathError{Op: "read", Path: path, Err: err}
	}
	if int64(len(data)) > limit {
		return nil, &TooLargeError{Path: path, Limit: limit}
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
