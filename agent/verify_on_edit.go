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

// defaultVerifyOnEditCheckpointBudget bounds the verify-on-edit flush at a
// pause checkpoint (see flushPendingVerifyOnEditAtCheckpoint). It is
// deliberately short and independent of the runner's own ceiling (the host
// typically caps that at its bash max timeout — minutes), so a pause remains a
// prompt cooperative stop instead of waiting out the user's whole test/linter
// run.
const defaultVerifyOnEditCheckpointBudget = 5 * time.Second

// verifyOnEditCheckpointAbandonGrace is how long the checkpoint flush waits
// PAST the budget for an already-cancelled runner to deliver its final result
// before dropping it (see flushPendingVerifyOnEditAtCheckpoint). The runner's
// context fires at the budget; the grace is only a fairness window so the
// runner's real timeout note wins over the synthetic dropped-verification
// note. It is NOT part of the command's budget: the command is cancelled at
// the budget either way.
const verifyOnEditCheckpointAbandonGrace = 250 * time.Millisecond

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
// circuit breakers and parse-error aborts can skip the last call of a response
// group, and an implicit text-only finish afterwards would otherwise drop the
// pending verification silently. Called from the implicit-finish acceptance
// branches so the note lands in the final output instead of vanishing.
func (e *Executor) flushPendingVerifyOnEdit(ctx context.Context, state *runState) string {
	if e.verifyOnEdit == nil || !state.pendingVerifyEdit {
		return ""
	}
	state.pendingVerifyEdit = false
	return FormatVerifyNote(e.verifyOnEdit(ctx), e.verifyOnEditCap)
}

// flushPendingVerifyOnEditAtCheckpoint flushes a pending verify-on-edit at a
// terminal boundary that returns a resumable checkpoint — a pause — and records
// the note in state.allSteps as a user-nudge step, so a resumed run sees the
// verification of an edit performed earlier in the same response group. It is
// deliberately NOT called from the arms that return no checkpoint (a tool-call
// ceiling or an infrastructure/cancellation error): with no consumer the note
// would be discarded, so running the command there would be wasted work. It is a
// no-op when nothing is pending (no runner installed, or no edit since the last
// flush).
//
// pendingVerifyEdit lives on the loop-local runState, not on the returned
// Steps, so without this the pending verification would be dropped the moment
// the run leaves the tool-dispatch path — and a resumed run, which starts with
// a fresh runState, would never verify the edit (the same silent loss the
// HITL-reject path already guards against).
//
// The flush is bounded by its OWN short budget
// (defaultVerifyOnEditCheckpointBudget), independent of the runner's ceiling:
//
//   - the runner's context carries the budget as a deadline, so a
//     cancellation-honoring runner is stopped at the budget and its (partial)
//     result is formatted and injected as usual — the [verify_on_edit] timeout
//     note already tells the model the edit was NOT verified;
//   - the select below additionally guarantees the flush RETURNS within the
//     budget (plus a short abandon grace that only waits for an
//     already-cancelled runner's final result) even for a runner that ignores
//     cancellation. A result arriving after that is dropped at the checkpoint
//     and the edit stays unverified: the injected note says so, and the
//     command re-runs at the next regular verification (or is re-run
//     manually). The abandoned runner's goroutine ends when the runner
//     finally returns — bounded by the runner's own timeout handling, the
//     same containment the tool-call watchdog documents for a tool that
//     ignores its cancellation.
func (e *Executor) flushPendingVerifyOnEditAtCheckpoint(ctx context.Context, state *runState) {
	if e.verifyOnEdit == nil || !state.pendingVerifyEdit {
		return
	}
	state.pendingVerifyEdit = false

	budget := e.verifyOnEditCheckpointBudget
	if budget <= 0 {
		budget = defaultVerifyOnEditCheckpointBudget
	}
	flushCtx, cancel := context.WithTimeout(ctx, budget)
	defer cancel()

	// Buffered: an abandoned runner's goroutine must never block on the send.
	done := make(chan EditVerifyResult, 1)
	go func() { done <- e.verifyOnEdit(flushCtx) }()

	// The hard bound is the budget plus the abandon grace: at the budget the
	// runner's context is already cancelled, so the grace only waits for an
	// honoring runner to deliver its final (timeout) result — a runner that
	// ignores cancellation is dropped right after it.
	timer := time.NewTimer(budget + verifyOnEditCheckpointAbandonGrace)
	defer timer.Stop()

	var res EditVerifyResult
	select {
	case res = <-done:
	case <-timer.C:
		// Dismiss a still-running command (cooperative, like the watchdog's
		// cancelCall) and proceed with the pause on time.
		cancel()
		state.allSteps = append(state.allSteps, Step{UserNudge: fmt.Sprintf(
			"[verify_on_edit] verification did not complete within the %s pause-checkpoint budget; the pause proceeds without it and the edit remains UNVERIFIED — re-run the command manually or after the next edit.",
			budget)})
		return
	}
	if note := FormatVerifyNote(res, e.verifyOnEditCap); note != "" {
		state.allSteps = append(state.allSteps, Step{UserNudge: note})
	}
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
