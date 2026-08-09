//go:build windows

package sysproc

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"sync"
	"unsafe"

	"golang.org/x/sys/windows"
)

// ErrProcessNotStarted is returned by [AssignKillOnCloseJob] when the supplied
// [exec.Cmd] has no running process yet (cmd.Process == nil). The function must
// be called immediately after a successful [exec.Cmd.Start] so that a process
// handle exists to attach to the job.
//
// Match it with errors.Is(err, ErrProcessNotStarted).
var ErrProcessNotStarted = errors.New("sysproc: AssignKillOnCloseJob requires a started process (cmd.Process is nil)")

// AssignKillOnCloseJob places the already-started process represented by cmd —
// and, by inheritance, its entire descendant tree — into a Windows Job Object
// configured with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE. Closing the last
// outstanding handle to the job terminates every process still in it. The
// returned cleanup closes that handle, so invoking it (typically via defer)
// kills the process and any surviving children/grandchildren.
//
// The motivating use case is guaranteeing that long-lived helper processes
// spawned by a command — for example a browser together with its console
// window launched by a shell — cannot outlive the host when the host exits or
// aborts the command.
//
// Requirements:
//   - cmd.Process must be non-nil; call AssignKillOnCloseJob as the very first
//     action after a successful [exec.Cmd.Start].
//   - Windows 8 or later, so that nested jobs are supported. Nested jobs allow
//     the assignment to succeed even when the host process (e.g. the
//     c0wrk-desktop app) is itself already a member of a job.
//
// On success cleanup is idempotent: it closes the job handle on the first call
// and is a safe no-op on subsequent calls (it never double-closes). Closing the
// last handle to the job object is what triggers kill-on-job-close, so the
// caller must keep a reference to cleanup for the lifetime it wants the
// descendants to survive.
//
// Best-effort degradation: if attaching the process to the job fails (for
// example on a pre-Windows 8 host whose own process belongs to a non-nestable
// job), AssignKillOnCloseJob closes the freshly created job handle — so no
// kernel handle is leaked — returns a no-op cleanup, and returns the error so
// the caller can fall back to its own termination strategy.
//
// Race window: the process is assigned to the job after [exec.Cmd.Start] has
// returned, so there is a brief window in which the started process could fork
// a child that escapes job membership. This is accepted for the powershell.exe
// use case: powershell.exe initializes and parses its command line before
// executing -Command, so any grandchildren (a browser, its console window, …)
// are spawned only after the assignment and therefore inherit job membership.
// Callers that spawn fast-forking grandchildren should call
// AssignKillOnCloseJob before the process can reasonably have spawned any.
func AssignKillOnCloseJob(cmd *exec.Cmd) (cleanup func(), err error) {
	if cmd == nil || cmd.Process == nil {
		return noopCleanup, fmt.Errorf("sysproc: %w", ErrProcessNotStarted)
	}

	job, err := createKillOnCloseJob()
	if err != nil {
		return noopCleanup, err
	}

	if err := assignToJob(job, cmd.Process); err != nil {
		// Best-effort: do not leak the job handle on assignment failure.
		_ = windows.CloseHandle(job)
		return noopCleanup, err
	}

	var closeOnce sync.Once
	return func() {
		// Closing the last handle to the job object terminates all members
		// because JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE is set. sync.Once makes
		// the closure safe to call concurrently and guarantees the handle is
		// closed exactly once (never double-closed), honoring the idempotency
		// contract documented above.
		closeOnce.Do(func() {
			_ = windows.CloseHandle(job)
		})
	}, nil
}

// createKillOnCloseJob creates an empty Job Object configured so that closing
// its last outstanding handle kills every member process. The caller owns the
// returned handle and is responsible for closing it (directly, or via the
// cleanup closure produced by AssignKillOnCloseJob).
//
// JOB_OBJECT_LIMIT_BREAKAWAY_OK is intentionally NOT set, so any child spawned
// by a member process inherits job membership automatically and is likewise
// killed when the job handle is closed.
func createKillOnCloseJob() (windows.Handle, error) {
	job, err := windows.CreateJobObject(nil, nil)
	if err != nil {
		return 0, fmt.Errorf("sysproc: CreateJobObject: %w", err)
	}

	info := windows.JOBOBJECT_EXTENDED_LIMIT_INFORMATION{
		BasicLimitInformation: windows.JOBOBJECT_BASIC_LIMIT_INFORMATION{
			LimitFlags: windows.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
		},
	}
	if _, err := windows.SetInformationJobObject(
		job,
		windows.JobObjectExtendedLimitInformation,
		uintptr(unsafe.Pointer(&info)),
		uint32(unsafe.Sizeof(info)),
	); err != nil {
		_ = windows.CloseHandle(job)
		return 0, fmt.Errorf("sysproc: SetInformationJobObject: %w", err)
	}

	return job, nil
}

// assignToJob attaches the running process to the job object. It is a
// package-level indirection so that tests can substitute a failing
// implementation to exercise AssignKillOnCloseJob's error/cleanup path.
var assignToJob = assignProcessToJob

// assignProcessToJob attaches proc to job using a process handle obtained via
// [os.Process.WithHandle]. WithHandle (Go 1.26+) guarantees the handle stays
// valid for the duration of the callback; it returns an error only if a handle
// cannot be acquired (e.g. the process was already released/waited). Nested
// jobs (Windows 8+) allow this to succeed even when the host process is itself
// inside a job.
func assignProcessToJob(job windows.Handle, proc *os.Process) error {
	var assignErr error
	if err := proc.WithHandle(func(handle uintptr) {
		assignErr = windows.AssignProcessToJobObject(job, windows.Handle(handle))
	}); err != nil {
		return fmt.Errorf("sysproc: obtain process handle: %w", err)
	}
	return assignErr
}

// noopCleanup is a safe no-op cleanup returned alongside errors so callers can
// defer it unconditionally without checking for nil.
func noopCleanup() {}
