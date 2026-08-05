//go:build windows

package sysproc

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"
	"unsafe"

	"golang.org/x/sys/windows"
)

// startSleeping starts a process that idles for long enough to outlive the
// test. "ping -n 60 127.0.0.1" emits one ICMP echo per second for ~60s and is
// available on every supported Windows version.
func startSleeping(t *testing.T) *exec.Cmd {
	t.Helper()
	cmd := exec.CommandContext(context.Background(), "ping", "-n", "60", "127.0.0.1")
	if err := cmd.Start(); err != nil {
		t.Fatalf("start ping: %v", err)
	}
	// Guarantee teardown even if an assertion aborts the test early.
	t.Cleanup(func() {
		_ = cmd.Process.Kill()
		_, _ = cmd.Process.Wait()
	})
	return cmd
}

// TestCreateKillOnCloseJob_FlagSet (criterion 2): the created job object must
// carry JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, verified by querying the job.
func TestCreateKillOnCloseJob_FlagSet(t *testing.T) {
	job, err := createKillOnCloseJob()
	if err != nil {
		t.Fatalf("createKillOnCloseJob: %v", err)
	}
	defer func() { _ = windows.CloseHandle(job) }()

	var info windows.JOBOBJECT_EXTENDED_LIMIT_INFORMATION
	if err := windows.QueryInformationJobObject(
		job,
		windows.JobObjectExtendedLimitInformation,
		uintptr(unsafe.Pointer(&info)),
		uint32(unsafe.Sizeof(info)),
		nil,
	); err != nil {
		t.Fatalf("QueryInformationJobObject: %v", err)
	}

	if info.BasicLimitInformation.LimitFlags&windows.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE == 0 {
		t.Errorf("job missing JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE; got flags=0x%x",
			info.BasicLimitInformation.LimitFlags)
	}
	// Breakaway must NOT be set so children inherit job membership.
	if info.BasicLimitInformation.LimitFlags&windows.JOB_OBJECT_LIMIT_BREAKAWAY_OK != 0 {
		t.Errorf("job must not set JOB_OBJECT_LIMIT_BREAKAWAY_OK; got flags=0x%x",
			info.BasicLimitInformation.LimitFlags)
	}
}

// TestAssignKillOnCloseJob_HappyPath (criteria 1 & 3): a started process yields
// a non-nil cleanup and nil error; invoking cleanup terminates the job member,
// and the cleanup closure is idempotent (safe no-op on a second call).
func TestAssignKillOnCloseJob_HappyPath(t *testing.T) {
	cmd := startSleeping(t)

	cleanup, err := AssignKillOnCloseJob(cmd)
	if err != nil {
		t.Fatalf("AssignKillOnCloseJob: %v", err)
	}
	if cleanup == nil {
		t.Fatal("cleanup must be non-nil on success")
	}

	// cleanup() closes the job handle -> kill-on-job-close terminates ping.
	// Wait must return well before ping's own 60s lifetime elapses.
	waitErr := make(chan error, 1)
	go func() { waitErr <- cmd.Wait() }()
	cleanup()
	select {
	case <-waitErr:
		// Process was killed by job close (or had already exited): expected.
	case <-time.After(5 * time.Second):
		t.Fatal("cleanup did not terminate the job member within 5s")
	}

	// Idempotency: a second call must be a safe no-op, not a panic.
	cleanup()
}

// TestAssignKillOnCloseJob_RequiresStartedProcess (criterion "return a sentinel
// error otherwise"): a cmd with no running process returns ErrProcessNotStarted
// and a non-nil (no-op) cleanup.
func TestAssignKillOnCloseJob_RequiresStartedProcess(t *testing.T) {
	cmd := exec.CommandContext(context.Background(), "ping", "-n", "60", "127.0.0.1") // intentionally not started

	cleanup, err := AssignKillOnCloseJob(cmd)
	if !errors.Is(err, ErrProcessNotStarted) {
		t.Fatalf("want ErrProcessNotStarted, got %v", err)
	}
	if cleanup == nil {
		t.Fatal("cleanup must be non-nil (no-op) even on the not-started error")
	}
	cleanup() // must not panic

	// A nil cmd must be defended against rather than panic.
	cleanup2, err2 := AssignKillOnCloseJob(nil)
	if !errors.Is(err2, ErrProcessNotStarted) {
		t.Fatalf("nil cmd: want ErrProcessNotStarted, got %v", err2)
	}
	if cleanup2 == nil {
		t.Fatal("nil cmd: cleanup must be non-nil (no-op)")
	}
	cleanup2()
}

// TestAssignKillOnCloseJob_AssignmentFailureNoLeak (criterion 4): when
// attaching the process to the job fails, AssignKillOnCloseJob must close the
// freshly created job handle (no leak), return a clear error, and hand back a
// no-op cleanup.
func TestAssignKillOnCloseJob_AssignmentFailureNoLeak(t *testing.T) {
	var capturedJob windows.Handle
	prev := assignToJob
	t.Cleanup(func() { assignToJob = prev })
	assignToJob = func(job windows.Handle, proc *os.Process) error {
		capturedJob = job
		return errors.New("simulated assignment failure")
	}

	cmd := startSleeping(t)

	cleanup, err := AssignKillOnCloseJob(cmd)
	if err == nil {
		t.Fatal("want error on simulated assignment failure")
	}
	if cleanup == nil {
		t.Fatal("cleanup must be non-nil (no-op) on assignment failure")
	}
	cleanup() // must be a safe no-op

	// No handle leak: AssignKillOnCloseJob must have closed the job it created.
	// A second Close on an already-closed handle fails (ERROR_INVALID_HANDLE).
	if err := windows.CloseHandle(capturedJob); err == nil {
		t.Error("job handle leaked: CloseHandle succeeded after AssignKillOnCloseJob should have closed it")
	}
}

// stillActive is the STILL_ACTIVE exit-code sentinel (STATUS_PENDING). A
// process reports this from GetExitCodeProcess while it is still running.
const stillActive uint32 = 259

// findChildProcess enumerates running processes and returns the PID of the
// first whose image name matches exe (case-insensitive) and whose recorded
// parent is parentPID. ParentProcessID is fixed at the process's creation, so
// it keeps identifying an orphan even after the parent has exited. It returns
// ok=false if no such process is found or the process snapshot cannot be taken.
func findChildProcess(exe string, parentPID int) (pid int, ok bool) {
	snapshot, err := windows.CreateToolhelp32Snapshot(windows.TH32CS_SNAPPROCESS, 0)
	if err != nil {
		return 0, false
	}
	defer func() { _ = windows.CloseHandle(snapshot) }()

	var entry windows.ProcessEntry32
	entry.Size = uint32(unsafe.Sizeof(entry))
	if err := windows.Process32First(snapshot, &entry); err != nil {
		return 0, false
	}
	for {
		name := windows.UTF16ToString(entry.ExeFile[:])
		if strings.EqualFold(name, exe) && int(entry.ParentProcessID) == parentPID {
			return int(entry.ProcessID), true
		}
		// Process32Next returns an error (ERROR_NO_MORE_FILES) once the
		// snapshot is exhausted; that is the normal termination, not a fault.
		if err := windows.Process32Next(snapshot, &entry); err != nil {
			break
		}
	}
	return 0, false
}

// TestAssignKillOnCloseJob_TerminatesGrandchild reproduces the exact scenario
// from the bug report: a spawned shell launches a DETACHED, long-lived
// grandchild (via cmd's "start /B") that outlives the shell itself. Closing the
// kill-on-close Job Object handle must terminate that orphaned grandchild too —
// proving process-tree containment works end to end, not just for direct
// children.
//
// To eliminate AssignKillOnCloseJob's documented post-Start race window, the
// shell first idles for a couple of seconds ("ping -n 3") so the grandchild is
// guaranteed to be spawned AFTER the job assignment and therefore inherits job
// membership deterministically.
func TestAssignKillOnCloseJob_TerminatesGrandchild(t *testing.T) {
	// Vehicle: cmd.exe launches a long-lived, detached ping grandchild. The
	// leading "ping -n 3" delay guarantees the grandchild is spawned only after
	// AssignKillOnCloseJob has attached cmd.exe to the job, so it reliably
	// inherits membership (no race escape).
	cmd := exec.CommandContext(context.Background(), "cmd", "/c",
		"ping -n 3 127.0.0.1 >NUL & start /B ping -n 60 127.0.0.1 >NUL")
	HideConsole(cmd)
	if err := cmd.Start(); err != nil {
		t.Skipf("spawn vehicle unavailable (cannot start cmd.exe): %v", err)
	}
	cmdPID := cmd.Process.Pid

	cleanup, err := AssignKillOnCloseJob(cmd)
	if err != nil {
		_ = cmd.Process.Kill()
		_, _ = cmd.Process.Wait()
		t.Skipf("kill-on-close jobs unsupported on this host: %v", err)
	}

	// Wait for cmd.exe to exit. After the leading delay it spawns the detached
	// grandchild and returns, leaving ping orphaned but still a job member.
	waitErr := make(chan error, 1)
	go func() { waitErr <- cmd.Wait() }()
	select {
	case <-waitErr:
		// cmd.exe has exited; the grandchild is now orphaned but still alive.
	case <-time.After(15 * time.Second):
		cleanup()
		t.Fatal("cmd.exe did not exit within 15s; orphaning step did not complete")
	}

	// Locate the orphaned grandchild: a ping.exe whose parent was cmd.exe.
	grandchildPID, ok := findChildProcess("ping.exe", cmdPID)
	if !ok {
		cleanup()
		t.Skip("no detached ping grandchild found; spawn vehicle did not behave as expected on this host")
	}

	// Open a handle so we can confirm liveness and guarantee teardown. The
	// handle keeps the process kernel object queryable until we close it.
	const access = windows.SYNCHRONIZE | windows.PROCESS_QUERY_LIMITED_INFORMATION | windows.PROCESS_TERMINATE
	h, err := windows.OpenProcess(access, false, uint32(grandchildPID))
	if err != nil {
		cleanup()
		t.Skipf("cannot open grandchild (already gone or access denied): %v", err)
	}
	// Hard teardown: never leak the grandchild even if an assertion aborts
	// before cleanup() runs. cleanup() is idempotent, so the deferred call is a
	// safe no-op once it has already executed.
	t.Cleanup(func() {
		cleanup()
		_ = windows.TerminateProcess(h, 1)
		_ = windows.CloseHandle(h)
	})

	// Precondition: the grandchild really is still alive and outlived cmd.exe.
	var code uint32
	if err := windows.GetExitCodeProcess(h, &code); err != nil {
		t.Fatalf("GetExitCodeProcess: %v", err)
	}
	if code != stillActive {
		t.Fatalf("grandchild not still-active before cleanup (exit code %d); cannot prove containment", code)
	}

	// The core assertion: closing the job handle kills the orphaned grandchild.
	cleanup()
	const waitMS = uint32(5 * time.Second / time.Millisecond)
	event, werr := windows.WaitForSingleObject(h, waitMS)
	if werr != nil {
		t.Fatalf("WaitForSingleObject: %v", werr)
	}
	if event != windows.WAIT_OBJECT_0 {
		t.Errorf("grandchild survived job close (WaitForSingleObject event=0x%x); process-tree containment failed", event)
	}
}
