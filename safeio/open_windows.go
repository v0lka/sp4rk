//go:build windows

package safeio

import (
	"io/fs"
	"os"
	"syscall"
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

// OpenFileNoFollow is OpenFile with one extra guard: the open never resolves
// a symbolic link — or any other reparse point — at the FINAL path
// component. It is the Windows counterpart of unix O_NOFOLLOW, with the same
// contract: a link at path makes the open fail with a *fs.PathError wrapping
// syscall.ELOOP instead of redirecting the caller to the link's target.
//
// The mechanism is the syscall layer's own flag pass-through, not a
// hand-rolled CreateFile: os.OpenFile maps the caller's flag through
// syscall.Open, whose high 12 bits (fileFlagsMask) are forwarded to
// CreateFile as FILE_FLAG_* values when they are in the supported set
// (validFileFlagsMask) — and FILE_FLAG_OPEN_REPARSE_POINT is in that set.
// (The pass-through needs the fileFlagsMask-capable syscall.Open; go.mod
// pins the toolchain that has it, so no version shim is carried here.)
// With the flag set, CreateFile opens a reparse-point target as the reparse
// point ITSELF (the link, never its destination), and a non-reparse target
// behaves exactly as it does without the flag. Every other OpenFile semantic
// — share modes, the permission security descriptor, append and truncate
// handling — therefore stays the stdlib's own, on the SAME single open:
// there is no Stat→Open window for a racing local adversary to slip a link
// through, matching the unix kernel-level O_NOFOLLOW guarantee.
//
// The post-open fstat then classifies the handle: fileStat's mode reports
// fs.ModeSymlink for IO_REPARSE_TAG_SYMLINK and fs.ModeIrregular for every
// other non-surrogate reparse tag (junctions, app exec links; the DEDUP tag
// is deliberately reported as a regular file and stays openable). A handle
// carrying either bit is refused: the descriptor is closed and
// syscall.ELOOP is returned, so callers can treat both platforms with one
// errors.Is against syscall.ELOOP. A regular file is returned exactly as
// OpenFile would return it.
//
// All the OpenFile hardening still applies: the handle-based regularity
// check refuses a non-regular pre-existing target with a *NotRegularError.
//
// The guard covers only the final component — intermediate symlinks are
// still resolved, as with unix O_NOFOLLOW; use MkdirAllReal to build
// intermediate paths out of real directories only.
func OpenFileNoFollow(path string, flag int, perm fs.FileMode) (*os.File, error) {
	f, err := os.OpenFile(path, flag|syscall.FILE_FLAG_OPEN_REPARSE_POINT, perm)
	if err != nil {
		return nil, err
	}
	st, err := f.Stat()
	if err != nil {
		_ = f.Close()
		return nil, &fs.PathError{Op: "open", Path: path, Err: err}
	}
	if mode := st.Mode(); mode&fs.ModeSymlink != 0 || mode&fs.ModeIrregular != 0 {
		_ = f.Close()
		return nil, &fs.PathError{Op: "open", Path: path, Err: syscall.ELOOP}
	}
	if !st.Mode().IsRegular() {
		_ = f.Close()
		return nil, &NotRegularError{Path: path}
	}
	return f, nil
}
