//go:build unix

package appendcodec

import (
	"errors"
	"os"

	"golang.org/x/sys/unix"
)

// lockExclusive takes an exclusive advisory lock (flock(2), non-blocking) on
// the descriptor's open file description. Locks conflict across open file
// descriptions — in this process or in another — so a second OpenAppend on
// the same path fails immediately instead of interleaving its frames into
// the first writer's stream. The lock dies with the descriptor: Close
// releases it by closing the file, and so does a crash.
//
// golang.org/x/sys/unix, not stdlib syscall: syscall.Flock is undefined on
// solaris/illumos, while x/sys/unix has it (and the LOCK_/E constants) on
// every GOOS the "unix" build tag covers.
func lockExclusive(f *os.File) error {
	return unix.Flock(int(f.Fd()), unix.LOCK_EX|unix.LOCK_NB)
}

// isLockConflict reports whether err from lockExclusive means the file is
// already locked by another live writer, as opposed to an environment
// failure (EWOULDBLOCK and EAGAIN are the same errno almost everywhere; both
// spellings are checked for portability).
func isLockConflict(err error) bool {
	return errors.Is(err, unix.EWOULDBLOCK) || errors.Is(err, unix.EAGAIN)
}
