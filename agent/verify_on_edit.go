package agent

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"time"
)

// EditVerifyResult is the outcome of one verify-on-edit command run.
type EditVerifyResult struct {
	// Output is the combined stdout/stderr of the verification command.
	// The executor truncates it to the configured cap before injecting it.
	Output string
	// ExitCode is the command's exit code. Negative when the command could
	// not run at all (Err non-nil) or was killed by a timeout (TimedOut).
	ExitCode int
	// TimedOut reports that the verification command exceeded its timeout.
	TimedOut bool
	// Timeout is the effective time limit the runner enforced for the run
	// (0 = unknown/unset). It is echoed in the timeout note so the model and
	// the user see the limit that actually applied — which may be lower than
	// the configured verify-on-edit timeout when a bash-tool max-timeout cap
	// clamps it.
	Timeout time.Duration
	// Err is set when the verification command could not be started at all
	// (infrastructure failure — not a failing test suite).
	Err error
}

// EditVerifyRunner executes a user-configured verification command (tests,
// linter, build) after file edits. The command MUST come from user
// configuration — never from the model — so running it requires no
// interactive confirmation. Implementations are responsible for their own
// timeout handling and for combining stdout/stderr into EditVerifyResult.
type EditVerifyRunner func(ctx context.Context) EditVerifyResult

// DefaultVerifyOnEditCap is the default cap (in chars) applied to the
// verification output injected into the observation.
const DefaultVerifyOnEditCap = 4000

// verifyOnEditTools are the tool names that count as a file edit. Only
// content-changing file tools trigger verification; directory operations
// and deletions do not produce code that a test/linter run would validate
// differently, and bash itself is not an "edit" (a bash run may mutate
// files, but the configured command already runs against the final state
// at the end of the group).
var verifyOnEditTools = map[string]struct{}{
	"write_file": {},
	"edit_file":  {},
}

// IsFileEditTool reports whether the given tool name is a file-edit tool
// tracked by the verify-on-edit hook.
func IsFileEditTool(name string) bool {
	_, ok := verifyOnEditTools[name]
	return ok
}

// FormatVerifyNote renders an EditVerifyResult as a system observation note.
// The output portion is truncated to cap chars; the note is empty when the
// result carries no information (no output, no error, exit 0 passes still
// produce a note so the model sees the cycle ran).
func FormatVerifyNote(res EditVerifyResult, maxChars int) string {
	if maxChars <= 0 {
		maxChars = DefaultVerifyOnEditCap
	}
	out := truncateVerifyOutput(res.Output, maxChars)
	var b strings.Builder
	switch {
	case res.Err != nil:
		b.WriteString("[verify_on_edit] could not run verification command: ")
		b.WriteString(res.Err.Error())
		if out != "" {
			b.WriteString("\nOutput:\n")
			b.WriteString(out)
		}
	case res.TimedOut:
		b.WriteString("[verify_on_edit] verification command timed out")
		if res.Timeout > 0 {
			b.WriteString(" (limit " + res.Timeout.String() + ")")
		}
		b.WriteString(". The edit was NOT verified — re-run the command manually or raise the timeout budget (the verification timeout is additionally capped by the bash tool's max timeout). Partial output:\n")
		b.WriteString(out)
	case res.ExitCode > 0:
		b.WriteString("[verify_on_edit] VERIFICATION FAILED (exit " +
			strconv.Itoa(res.ExitCode) + "). Fix these failures before finishing:\n")
		b.WriteString(out)
	case res.ExitCode < 0:
		// Negative exit code means the command produced no exit status at
		// all — it was blocked by policy/blacklist or killed by a signal.
		// That is NOT a failed verification run: telling the model "fix
		// these failures" would send it chasing failures that do not exist.
		b.WriteString("[verify_on_edit] verification command did not complete (blocked or killed — no exit code). The edit was NOT verified; inspect the output and re-run the command manually:\n")
		b.WriteString(out)
	default:
		b.WriteString("[verify_on_edit] verification command passed (exit 0). Output:\n")
		b.WriteString(out)
	}
	return b.String()
}

// truncateVerifyOutput caps the output, appending an explicit truncation
// marker so the model knows output was cut. The cut is rune-safe: byte
// slicing could split a multi-byte rune (test output commonly contains
// ✓/✗/box-drawing characters) and inject invalid UTF-8 into the model
// context and the persisted observation.
func truncateVerifyOutput(output string, maxChars int) string {
	output = strings.TrimSpace(output)
	runes := []rune(output)
	if len(runes) <= maxChars {
		return output
	}
	return string(runes[:maxChars]) + fmt.Sprintf("\n[...truncated %d chars...]", len(runes)-maxChars)
}

// runVerifyOnEditHook implements the debounce logic around the runner.
//
// It is invoked from the tool-call processing paths with the name of the
// tool just executed, whether its result was an error, and whether this is
// the last call of the current response group. A successful file edit
// (write_file/edit_file) marks the run "dirty"; the verification command
// runs ONCE at the end of the response group in which at least one edit
// succeeded, and its formatted output is appended to the last call's
// observation (both LLM context and frontend — the hook runs before the
// ToolResult event emission). Reads, failed edits, and HITL-rejected calls
// never mark the run dirty, so verification never re-runs without new edits.
func (e *Executor) runVerifyOnEditHook(ctx context.Context, toolName string, resultIsError, lastCallInGroup bool, state *runState, observation string) string {
	if e.verifyOnEdit == nil {
		return observation
	}
	if IsFileEditTool(toolName) && !resultIsError {
		state.pendingVerifyEdit = true
	}
	if !lastCallInGroup || !state.pendingVerifyEdit {
		return observation
	}
	state.pendingVerifyEdit = false
	res := e.verifyOnEdit(ctx)
	note := FormatVerifyNote(res, e.verifyOnEditCap)
	if note == "" {
		return observation
	}
	if observation == "" {
		return note
	}
	return observation + "\n\n" + note
}

// flushPendingVerifyOnEdit runs the pending verification, if any, and
// returns its formatted note ("" when nothing awaits verification). It is
// the last-resort flush for finishes that bypass the per-group hook sites:
// early-return interceptions (HITL rejections, circuit breakers, parse-error
// aborts) can skip the last call of a response group, and an implicit
// text-only finish afterwards would otherwise drop the pending verification
// silently. Called from the implicit-finish acceptance branches so the note
// lands in the final output instead of vanishing.
func (e *Executor) flushPendingVerifyOnEdit(ctx context.Context, state *runState) string {
	if e.verifyOnEdit == nil || !state.pendingVerifyEdit {
		return ""
	}
	state.pendingVerifyEdit = false
	return FormatVerifyNote(e.verifyOnEdit(ctx), e.verifyOnEditCap)
}

// SetVerifyOnEdit installs a post-edit verification hook. After every
// response group containing at least one successful write_file/edit_file
// call, the hook runs once (debounced per group) and its output is appended
// to the group's last observation as a [verify_on_edit] system note.
// maxOutputChars caps the injected output; <= 0 selects
// DefaultVerifyOnEditCap. A nil runner (the default) disables the hook.
func (e *Executor) SetVerifyOnEdit(runner EditVerifyRunner, maxOutputChars int) {
	e.verifyOnEdit = runner
	if maxOutputChars > 0 {
		e.verifyOnEditCap = maxOutputChars
	} else {
		e.verifyOnEditCap = DefaultVerifyOnEditCap
	}
}
