package tools

// Tool name constants for built-in tools. These mirror the names used by
// github.com/v0lka/sp4rk/tools/builtins during registration and are consumed by the executor
// for tool-specific behavior (truncation hints, caching, etc.).
const (
	ToolReadFile  = "read_file"
	ToolWriteFile = "write_file"
	ToolEditFile  = "edit_file"
	ToolRipgrep   = "ripgrep"
	ToolGrep      = "grep"
	ToolGlob      = "glob"
	ToolWebFetch  = "web_fetch"
	ToolBashExec  = "bash_exec"
	ToolPoshExec  = "posh_exec"
	ToolBatch     = "batch"
)

// shellExecTools are the built-in tools that execute a shell command string
// (bash_exec, posh_exec). They are the highest-uncertainty tools and are
// intentionally deprioritized (Tier 3) in grouped tool lists so that more
// purpose-built tools are preferred.
var shellExecTools = map[string]bool{
	ToolBashExec: true,
	ToolPoshExec: true,
}

// IsShellExecTool reports whether name is one of the shell-executing built-in
// tools (bash_exec or posh_exec).
func IsShellExecTool(name string) bool {
	return shellExecTools[name]
}
