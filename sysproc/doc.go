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
package sysproc
