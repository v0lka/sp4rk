//go:build windows

package appendcodec

import (
	"errors"
	"os"

	"golang.org/x/sys/windows"
)

// lockOffsetHigh is the high dword of the lock byte range's start offset:
// 0x7FFFFFFF_00000000 — 16 EiB into the file, far beyond any reachable size
// (NTFS caps files at 16 TiB). Windows region locks taken with LockFileEx
// are mandatory for other handles — a reader touching the locked range would
// fail with ERROR_LOCK_VIOLATION — so the range must sit where no content
// byte can ever be: locking byte 0 would block every rotation reader of the
// live log. A lock beyond EOF is still a valid, conflicting target for a
// second LockFileEx, which is what makes "one path, one writer" enforceable.
const lockOffsetHigh = 0x7FFFFFFF

// lockExclusive takes an exclusive byte-range lock (LockFileEx,
// FAIL_IMMEDIATELY) on the unreachable beyond-EOF range described by
// lockOffsetHigh through the descriptor's handle. Locks conflict across
// handles — in this process or in another — so a second OpenAppend on the
// same path fails immediately instead of interleaving its frames into the
// first writer's stream. The lock dies with the handle: Close releases it by
// closing the file, and so does a crash.
func lockExclusive(f *os.File) error {
	ol := &windows.Overlapped{OffsetHigh: lockOffsetHigh}
	return windows.LockFileEx(windows.Handle(f.Fd()),
		windows.LOCKFILE_EXCLUSIVE_LOCK|windows.LOCKFILE_FAIL_IMMEDIATELY,
		0, 1, 0, ol)
}

// isLockConflict reports whether err from lockExclusive means the file is
// already locked by another live writer, as opposed to an I/O failure.
func isLockConflict(err error) bool {
	return errors.Is(err, windows.ERROR_LOCK_VIOLATION)
}
