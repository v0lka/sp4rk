//go:build windows

package builtins

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os/exec"
	"regexp"
	"strings"
	"syscall"
	"time"

	"github.com/v0lka/sp4rk/sysproc"
	"github.com/v0lka/sp4rk/tools"
)

const toolPoshDescription = `Execute commands via Windows PowerShell (powershell.exe -NoProfile -NonInteractive -Command). Use this for build commands, running scripts, installing packages, git operations, and system tasks on Windows. Returns combined stdout and stderr. Commands time out after 60 seconds by default (configurable up to 120s). An optional working_directory can be set for the command's execution context.`

// PoshExecTool executes PowerShell commands via powershell.exe on Windows.
// It is the Windows counterpart of BashExecTool: same blacklist/Judge model,
// same timeout/working_directory containment rules, but adapted to the
// Windows process model.
type PoshExecTool struct {
	*tools.BaseTool
	blacklist []string
	compiled  []*regexp.Regexp
	timeouts  BashTimeouts
}

// createNewProcessGroup is the Windows process creation flag (0x00000200)
// that places the child in a new process group, isolating it from the
// parent's console group so a Ctrl+C sent to the host does not propagate.
const createNewProcessGroup = 0x00000200

// NewPoshExecTool creates a new PoshExecTool with the given blacklist and
// default timeouts.
func NewPoshExecTool(blacklist []string) (*PoshExecTool, error) {
	return NewPoshExecToolWithTimeouts(blacklist, DefaultBashTimeouts())
}

// NewPoshExecToolWithTimeouts creates a new PoshExecTool with the given
// blacklist and timeouts.
func NewPoshExecToolWithTimeouts(blacklist []string, timeouts BashTimeouts) (*PoshExecTool, error) {
	compiled := make([]*regexp.Regexp, 0, len(blacklist))
	for _, pattern := range blacklist {
		re, err := regexp.Compile(pattern)
		if err != nil {
			return nil, fmt.Errorf("invalid posh blacklist pattern %q: %w", pattern, err)
		}
		compiled = append(compiled, re)
	}
	return &PoshExecTool{
		BaseTool: &tools.BaseTool{
			ToolName:        "posh_exec",
			ToolGroup:       tools.GroupExecute,
			ToolDescription: toolPoshDescription,
			Schema:          json.RawMessage(`{"type": "object", "properties": {"command": {"type": "string", "description": "The PowerShell command to execute. Supports pipes, redirects, and chained commands."}, "timeout": {"type": "string", "description": "Timeout as a Go duration string, e.g. \"30s\" or \"2m\". Default: 60s, maximum: 120s."}, "working_directory": {"type": "string", "description": "Absolute path to use as the working directory for command execution. If omitted, defaults to the workspace root when available."}}, "required": ["command"]}`),
			Policy:          tools.PolicyUserConfirm,
			Untrusted:       true,
		},
		blacklist: blacklist,
		compiled:  compiled,
		timeouts:  timeouts,
	}, nil
}

// poshInput represents the input parameters for PowerShell command execution.
type poshInput struct {
	Command          string `json:"command"`
	Timeout          string `json:"timeout"`
	WorkingDirectory string `json:"working_directory"`
}

// Judge evaluates whether a PowerShell command is safe to execute.
// It checks the command against compiled blacklist patterns.
//
// Severity mirrors BashExecTool.Judge: a blacklist match is hard, a
// path-containment escalation is soft.
func (t *PoshExecTool) Judge(ctx context.Context, input json.RawMessage) tools.JudgeOutcome {
	var params poshInput
	if err := json.Unmarshal(input, &params); err != nil {
		return tools.JudgeOutcome{} // Defer to LLM Judge on parse error
	}

	for i, re := range t.compiled {
		if re.MatchString(params.Command) {
			return tools.JudgeOutcome{
				Reason:   "command matches blacklist pattern: " + t.blacklist[i],
				Severity: tools.JudgeSeverityHard,
			}
		}
	}

	// Containment check: reject commands that reference filesystem paths
	// outside the configured session roots (workspace + auxiliary roots).
	// Mirrors BashExecTool.Judge exactly, differing only in ShellKind
	// (ShellPosh) so PowerShell env syntax like "$env:VAR" is recognized.
	// A path is escalated when it, or its nearest existing ancestor directory,
	// exists and is outside the roots — retaining write/create targets whose
	// leaf does not yet exist but whose parent directory does, so a write
	// into an existing out-of-root directory still triggers a prompt under
	// auto-approval. A wholly non-existent subtree (a fabricated token with
	// no real anchor) is dropped.
	outside := tools.PathsOutsideRoots(ctx, params.Command, tools.ShellPosh, params.WorkingDirectory)
	if outside = tools.ExistingOrAnchoredPaths(outside); len(outside) > 0 {
		return tools.JudgeOutcome{
			Reason:   "command references existing path(s) outside session roots: " + strings.Join(outside, ", "),
			Severity: tools.JudgeSeveritySoft,
		}
	}

	return tools.JudgeOutcome{} // No concern to report; workspace auto-approval semantics apply.
}

// Execute runs the PowerShell command and returns the result.
func (t *PoshExecTool) Execute(ctx context.Context, input json.RawMessage) (tools.ToolResult, error) {
	var params poshInput
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

	// Create command: Windows PowerShell, no profile, non-interactive.
	cmd := exec.CommandContext(timeoutCtx, "powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command)

	// Place the command in a new process group so cmd.Cancel can target
	// powershell.exe in isolation (CREATE_NEW_PROCESS_GROUP). The full process
	// tree — powershell.exe plus any grandchildren it spawns, e.g. a browser
	// and the console-subsystem helper window a Playwright-driven
	// chrome-headless-shell leaves behind — is additionally contained in a
	// kill-on-close Job Object attached after Start (see below). On normal
	// completion, timeout, or cancel the whole tree is terminated, so no
	// orphaned windows linger after the command ends. CREATE_NO_WINDOW
	// (HideConsole, below) and the Job Object are complementary: the former
	// hides the shell's own window and default-flag children that share its
	// console buffer; the latter guarantees cleanup of anything that escapes
	// by allocating its own window.
	cmd.SysProcAttr = &syscall.SysProcAttr{CreationFlags: createNewProcessGroup}
	// Suppress the console window that a GUI-subsystem host would otherwise
	// allocate for the child process (CREATE_NO_WINDOW). This is OR-ed with
	// createNewProcessGroup so both flags remain in effect.
	sysproc.HideConsole(cmd)

	// Cancel kills the powershell.exe parent on timeout/cancel; the Job
	// Object attached after Start then reaps any surviving grandchildren.
	cmd.Cancel = func() error {
		if cmd.Process == nil {
			return nil
		}
		return cmd.Process.Kill()
	}
	// Grace period for pipe readers to drain after the process is killed.
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

	// Capture combined stdout+stderr ourselves rather than using
	// CombinedOutput: CombinedOutput runs Start+Wait as one atomic step,
	// leaving no window to attach the process tree to its Job Object between
	// them. Both streams share one buffer so ordering is preserved.
	var buf bytes.Buffer
	cmd.Stdout = &buf
	cmd.Stderr = &buf

	if err := cmd.Start(); err != nil {
		return tools.ToolResult{
			Content: fmt.Sprintf("failed to start command: %v", err),
			IsError: true,
		}, nil
	}

	// Attach the started process tree to a kill-on-close Job Object so that
	// any grandchildren (browsers, helper windows, ...) are terminated when
	// the job handle is closed below. Closing the handle performs a hard
	// kill (no graceful-shutdown window) on EVERY path — normal completion,
	// timeout, and cancel alike — so any descendant still flushing when the
	// command ends is terminated immediately; this is the intended
	// "no orphans linger" behaviour for the browser/console-window use case.
	// Best-effort: if assignment fails we surface it to the caller and fall
	// back to the parent-only kill that cmd.Cancel performs above, never
	// failing the command itself.
	jobCleanup, jobErr := sysproc.AssignKillOnCloseJob(cmd)
	// jobCleanup is always non-nil (a no-op on assignment failure); closing
	// the job handle after Wait kills every surviving tree member.
	defer jobCleanup()

	err = cmd.Wait()
	output := buf.Bytes()

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

	res := tools.ToolResult{
		Content: string(output),
		IsError: false,
	}
	// Surface a degraded-containment note rather than logging via a global
	// logger: the agent/user is the right audience, and a GUI host has no
	// stderr to surface a log.Printf through.
	if jobErr != nil {
		res.Content += "\n[warning: job-object tree containment unavailable (" + jobErr.Error() + "), falling back to parent-only kill]"
	}
	return res, nil
}
