// Package sysproc configures process-creation attributes for child processes
// spawned by host applications and the SDK's own helper processes.
//
// On Windows the host application is typically built as a GUI-subsystem binary
// with no attached console. Any child process started via [os/exec] allocates a
// fresh console window by default, which appears as a flashing terminal window
// on screen — most visibly for the shell-execution tools (posh_exec), the
// ripgrep search tool, the version probes run by the env-info collector, and
// the long-lived MCP server processes spawned via stdio transport. HideConsole
// suppresses those windows by setting CREATE_NO_WINDOW.
//
// Call HideConsole on every [exec.Cmd] that is not an interactive pseudo
// terminal session. An interactive PTY/ConPTY session routes the child's
// console through the pseudo terminal and therefore must keep its default
// creation behaviour. The function is a no-op on non-Windows platforms, so
// callers do not need their own build tags.
//
// # Process-tree containment (Windows): Job Objects
//
// Some commands spawn long-lived helper processes — or detached console
// windows — that can outlive the command that launched them. The motivating
// case is posh_exec running PowerShell that launches a browser: the browser
// process and its console window are children of powershell.exe, not of the
// host, so killing powershell.exe (e.g. via [exec.Cmd.Cancel]) leaves them
// running as orphans that persist on screen.
//
// AssignKillOnCloseJob solves this. It places an already-started process —
// and, by inheritance, its entire descendant tree — into a Windows Job Object
// configured with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE. The returned cleanup
// closure closes the job handle (typically via defer, after [exec.Cmd.Wait]),
// which terminates every process still in the job: the command itself on
// timeout or cancellation, and any surviving children/grandchildren (a
// browser, its console window, …) on normal completion too. Because
// JOB_OBJECT_LIMIT_BREAKAWAY_OK is intentionally not set, every child a member
// process later spawns inherits job membership and is likewise cleaned up.
//
// The model is assign-after-Start: the caller calls [exec.Cmd.Start], then
// AssignKillOnCloseJob(cmd) as the very next action, then defers the returned
// cleanup. There is a brief race window between Start and assignment in which
// a fast-forking grandchild could escape job membership; this is accepted for
// the powershell.exe use case, which parses its command line before executing
// -Command, so grandchildren are spawned only after assignment and inherit
// membership.
//
// AssignKillOnCloseJob is best-effort: on failure (e.g. a pre-Windows 8 host
// inside a non-nestable job) it leaks no handle, returns a no-op cleanup so the
// caller can defer it unconditionally, and returns the error so the caller can
// fall back to its own termination strategy (such as [exec.Cmd.Cancel]).
//
// HideConsole (CREATE_NO_WINDOW, which hides the console window from the
// start) and the Job Object (which guarantees cleanup of anything that
// escapes the hidden buffer) are complementary: together they hide the shell
// window AND guarantee that no orphaned grandchildren or stray console windows
// persist after the host finishes, times out, or cancels the command.
//
// # Dependencies
//
// HideConsole itself is implemented with the Go standard library
// (os/exec, syscall under the windows build tag). The Job-Object helper, by
// contrast, requires golang.org/x/sys/windows (CreateJobObject,
// SetInformationJobObject, AssignProcessToJobObject, CloseHandle) and is
// therefore confined to a //go:build windows file. The non-Windows build has
// no such dependency; the package remains a no-op outside Windows.
package sysproc
