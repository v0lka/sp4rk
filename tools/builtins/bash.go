//go:build !windows

package builtins

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os/exec"
	"regexp"
	"syscall"
	"time"

	"github.com/v0lka/sp4rk/tools"
)

const toolBashDescription = `Purpose: execute a shell command via bash -c — the fallback for what no dedicated tool covers: builds, test runs, package managers, git operations, system tasks.
Use when: reading (read_file), editing (edit_file), listing (list_directory) and searching (ripgrep, glob) all have dedicated tools — reach for bash only when they cannot do the job.
Inputs: command (pipes, redirects, chained commands OK); optional working_directory (absolute path for the command's execution context; defaults to workspace root); optional timeout, a JSON string with a unit suffix, e.g. "30s" or "2m" (default "60s", capped at the configured max). The timeout value MUST be quoted in the tool call - write "timeout": "30s"; an unquoted bare token like 30s is invalid JSON and the call is rejected.
Outputs: combined stdout and stderr; failing exit codes surface as errors carrying the output. Keep output minimal (e.g. pipe through tail) to avoid flooding context.
Example: "go test ./core/... 2>&1 | tail -30".
Anti-example: do not "cat src/app.go" (read_file), "find . -name '*.ts'" (glob) or "grep -rn TODO ." (ripgrep) — dedicated tools respect path policy and return structured results.`

// BashExecTool executes bash commands in a shell.
type BashExecTool struct {
	*tools.BaseTool
	blacklist []string
	compiled  []*regexp.Regexp
	timeouts  BashTimeouts
}

// NewBashExecTool creates a new BashExecTool with the given blacklist.
func NewBashExecTool(blacklist []string) (*BashExecTool, error) {
	return NewBashExecToolWithTimeouts(blacklist, DefaultBashTimeouts())
}

// NewBashExecToolWithTimeouts creates a new BashExecTool with the given blacklist and timeouts.
func NewBashExecToolWithTimeouts(blacklist []string, timeouts BashTimeouts) (*BashExecTool, error) {
	compiled := make([]*regexp.Regexp, 0, len(blacklist))
	for _, pattern := range blacklist {
		re, err := regexp.Compile(pattern)
		if err != nil {
			return nil, fmt.Errorf("invalid bash blacklist pattern %q: %w", pattern, err)
		}
		compiled = append(compiled, re)
	}
	return &BashExecTool{
		BaseTool: &tools.BaseTool{
			ToolName:        "bash_exec",
			ToolGroup:       tools.GroupExecute,
			ToolDescription: toolBashDescription,
			Schema:          json.RawMessage(`{"type": "object", "properties": {"command": {"type": "string", "description": "The bash command to execute. Supports pipes, redirects, and chained commands."}, "timeout": {"type": "string", "description": "Optional timeout as a quoted JSON string with a unit suffix, e.g. \"30s\" or \"2m\". The value MUST be a string in double quotes: write \"timeout\": \"30s\" - an unquoted bare token like 30s is invalid JSON and the call is rejected. Default: \"60s\"; maximum: \"120s\"."}, "working_directory": {"type": "string", "description": "Absolute path to use as the working directory for command execution. If omitted, defaults to the workspace root when available."}}, "required": ["command"]}`),
			Policy:          tools.PolicyUserConfirm,
			Untrusted:       true,
		},
		blacklist: blacklist,
		compiled:  compiled,
		timeouts:  timeouts,
	}, nil
}

// bashInput represents the input parameters for bash command execution.
type bashInput struct {
	Command          string `json:"command"`
	Timeout          string `json:"timeout"`
	WorkingDirectory string `json:"working_directory"`
}

// Judge evaluates whether a bash command is safe to execute.
//
// Deterministic pipeline, in order:
//
//  1. Blacklist match — hard, ReasonCodeCommandBlacklist, reason "command
//     matches blacklist pattern: ...". The blacklist is operator policy and
//     always wins; the reason must never be weakened.
//  2. Flowsh criteria — the host pre-computes the deterministic analysis
//     ([tools.AnalyzeShellCommandForJudge]; criteria C1–C8 in
//     tools/shellanalysis.go) and attaches it to ctx via
//     [tools.WithShellAnalysis]; the Judge reads it through
//     [tools.ShellJudgeOutcome] and returns its winning outcome verbatim
//     (hard canonical C1–C5, hard non-canonical C6, soft C7/C8). The Judge
//     never runs the analysis engine itself — no recomputation.
//
// The former static checks — unresolvable path tokens (hard) and shell-path
// containment (soft) — were removed by explicit decision: tokens the static
// walker cannot see through are covered by the C6 unbounded criterion, and
// out-of-root scope by C4/C8, both of which the flowsh effect IR assesses
// more precisely than token walking.
//
// When NOTHING is attached the Judge returns an empty outcome and defers to
// the advisory judges — the deterministic floor (C1–C8) is then absent for
// this call, so a host that wants the shell escalations must attach the
// analysis itself via [tools.WithShellAnalysis]. When the attached analysis
// carries an ERROR (e.g. a knowledge-base load failure — logged), the Judge
// instead FAILS CLOSED with the hard canonical command_analysis_unavailable
// reason (see [tools.ShellJudgeOutcome]).
func (t *BashExecTool) Judge(ctx context.Context, input json.RawMessage) tools.JudgeOutcome {
	var params bashInput
	if err := json.Unmarshal(input, &params); err != nil {
		return tools.JudgeOutcome{} // Defer to LLM Judge on parse error
	}

	for i, re := range t.compiled {
		if re.MatchString(params.Command) {
			return tools.JudgeOutcome{
				Reason:     "command matches blacklist pattern: " + t.blacklist[i],
				Severity:   tools.JudgeSeverityHard,
				ReasonCode: tools.ReasonCodeCommandBlacklist,
			}
		}
	}

	return tools.ShellJudgeOutcome(ctx, "bash_exec")
}

// Execute runs the bash command and returns the result.
func (t *BashExecTool) Execute(ctx context.Context, input json.RawMessage) (tools.ToolResult, error) {
	var params bashInput
	if err := json.Unmarshal(input, &params); err != nil {
		return tools.ParseInputError(err)
	}

	if params.Command == "" {
		return tools.ToolResult{Content: "validation error: command is required", IsError: true}, nil
	}

	// Parse timeout (default 60s, max from config)
	command := params.Command
	timeoutStr := params.Timeout
	if timeoutStr == "" {
		timeoutStr = "60s"
	}
	timeout, err := time.ParseDuration(timeoutStr)
	if err != nil {
		return tools.ToolResult{
			Content: fmt.Sprintf("invalid timeout duration: %v", err),
			IsError: true,
		}, nil
	}
	// Enforce maximum timeout from config
	if timeout > t.timeouts.MaxTimeout {
		timeout = t.timeouts.MaxTimeout
	}

	// Create context with timeout
	timeoutCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	// Create command
	cmd := exec.CommandContext(timeoutCtx, "bash", "-c", command)

	// Put the command and all children in a new process group so we can
	// kill the entire tree on timeout (exec.CommandContext only kills the
	// parent, leaving orphaned children that hold pipes open).
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}

	// Cancel kills the entire process group instead of just the parent.
	cmd.Cancel = func() error {
		if cmd.Process == nil {
			return nil
		}
		return syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
	}
	// Grace period for pipe readers to drain after the process group is killed.
	cmd.WaitDelay = t.timeouts.WaitDelay

	// Set working directory: prefer explicit param, fall back to workspace root.
	// Validate that the resolved directory is within the workspace or the system
	// temp directory to prevent arbitrary filesystem access (S-2).
	workDir := params.WorkingDirectory
	if workDir == "" {
		workDir = tools.WorkspacePathFrom(ctx)
	}
	if workDir != "" {
		if err := validateWorkDir(ctx, workDir, tools.SessionRoots(ctx)); err != nil {
			return tools.ToolResult{
				Content: fmt.Sprintf("working_directory rejected: %v", err),
				IsError: true,
			}, nil
		}
		cmd.Dir = workDir
	}

	// Execute and capture combined output
	output, err := cmd.CombinedOutput()

	if err != nil {
		result := string(output) + "\n" + err.Error()
		if errors.Is(timeoutCtx.Err(), context.DeadlineExceeded) {
			result += "\n[Process killed: timeout exceeded]"
		}
		return tools.ToolResult{
			Content: result,
			IsError: true,
		}, nil
	}

	return tools.ToolResult{
		Content: string(output),
		IsError: false,
	}, nil
}
