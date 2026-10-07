package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/v0lka/sp4rk/tools"
)

// defaultToolWatchdogInterval is the cadence at which executeToolCall polls the
// cooperative pause checker while a tool call is in flight. It is deliberately
// short so a pause is observed within a fraction of a second even when the
// underlying tool is blocked (a blocking syscall cannot be interrupted, so the
// watchdog must poll).
const defaultToolWatchdogInterval = 250 * time.Millisecond

// defaultToolCallTimeoutExemptTools names the built-in tools exempt from the
// per-tool-call ceiling. They are the reserved GroupSystem orchestration tools
// whose Execute blocks by design — on a human (an ask_user prompt, a plan or
// goal awaiting approval) or on sub-work (a blocking delegate running its
// subagents, a synchronously executed plan) — so bounding them by a raw wall
// clock would fail a perfectly healthy run at the ceiling. A host that raises
// its OWN human prompt inside Execute (e.g. c0wrk's per-tool confirmation gate)
// does not rely on this set: it bounds that wait just under the ceiling itself
// and yields a clean denial, so a slow answer lets the run continue instead of
// aborting it. Hosts extend or replace the set via SetToolCallTimeoutExempt.
var defaultToolCallTimeoutExemptTools = map[string]struct{}{
	"ask_user":     {},
	"declare_plan": {},
	"propose_goal": {},
	"delegate":     {},
	"execute_plan": {},
}

// toolCallOutcome carries the result of the detached tools.Execute call back to
// executeToolCall. The channel it travels on is buffered (capacity 1) so the
// producing goroutine always sends and exits, even when executeToolCall has
// already returned via the pause/timeout/cancel arms — that is what keeps the
// goroutine from leaking on the abandoned paths.
type toolCallOutcome struct {
	result tools.ToolResult
	err    error
}

// executeToolCall dispatches a single tool call through e.tools.Execute while
// remaining responsive to signals that must interrupt a stuck call. It runs the
// tool in its own goroutine and selects over four events:
//
//   - the tool completing normally — its result (and error) are returned as-is;
//   - ctx.Done() — the run's context was cancelled or its deadline elapsed
//     (shutdown/cancel), returning ctx.Err();
//   - the cooperative pause checker tripping (polled every
//     toolWatchdogInterval) — returning ErrPaused so the caller can emit a
//     resumable checkpoint. c0wrk pause does NOT cancel ctx, so this poll is
//     the only way a pause is observed while a tool is blocked;
//   - e.toolCallTimeout elapsing — returning ErrToolTimeout. A zero timeout
//     (the default) disables this arm, and tools named in
//     e.toolCallTimeoutExempt are never bounded at all.
//
// The tool call runs on a context derived from ctx that is cancelled when the
// watchdog abandons the wait (a pause or a timeout), so a still-pending
// interactive prompt or blocking sub-work is dismissed rather than orphaned —
// that is what makes the timeout arm's ErrToolTimeout actually stop the work
// instead of leaving a late "Allow" to execute a mutation with no owning run.
// ctx itself is never cancelled here.
//
// IMPORTANT: cancelling the derived context is a cooperative request, not a
// guarantee. A tool blocked in an uninterruptible syscall, or already past its
// last cancellation check, can still run to completion after the watchdog has
// returned, so a caller that marks work terminal — a pause or timeout
// checkpoint, or an error return at a dispatch site — must treat the tool's
// side effects as having possibly occurred. The detached goroutine itself is
// bounded: it ends as soon as Execute returns or panics (its send on the size-1
// channel never blocks), so it cannot leak once the tool finishes — the one
// case that can outlive the run is a tool that never returns, which no watchdog
// can reclaim.
//
// A panic inside e.tools.Execute is recovered on the detached goroutine (where
// it now runs) and delivered as the outcome's error, preserving the
// subagent-level panic containment that held before the call moved off the
// guarded Run goroutine.
//
// executeToolCall is called from the single-tool and batch dispatch sites in
// executor_run.go. It must be called from the Run goroutine (the pause checker
// and toolCallTimeout fields are read here without synchronization, matching
// their set-before-Run contract).
func (e *Executor) executeToolCall(ctx context.Context, name string, input json.RawMessage) (tools.ToolResult, error) {
	// Derive a cancellable context for the tool call so the abandon paths can
	// dismiss a still-pending call (see the doc comment). ctx itself is never
	// cancelled here — only this child, and the deferred cancel releases it on
	// every exit.
	callCtx, cancelCall := context.WithCancel(ctx)
	defer cancelCall()

	done := make(chan toolCallOutcome, 1)
	go func() {
		defer func() {
			if r := recover(); r != nil {
				done <- toolCallOutcome{err: fmt.Errorf("tool %q panicked: %v", name, r)}
			}
		}()
		res, err := e.tools.Execute(callCtx, name, input)
		done <- toolCallOutcome{result: res, err: err}
	}()

	// A tool that has ALREADY completed must win over any arm: select picks
	// uniformly at random among ready cases, so a completion ready at the same
	// instant as ctx.Done()/a pause tick/the timeout must be drained first, or a
	// finished tool's real result would be discarded (see drainToolResult, used
	// here and inside every blocking arm below).
	if res, ok, err := drainToolResult(done); ok {
		return res, err
	}

	// Pause polling is armed only when a checker is installed; otherwise the
	// tick channel stays nil and its select arm is permanently disabled.
	var tickCh <-chan time.Time
	if e.pauseChecker != nil {
		interval := e.toolWatchdogInterval
		if interval <= 0 {
			// Defensive: a zero interval would panic time.NewTicker. NewExecutor
			// always defaults it, but a raw &Executor{} (present in tests) must
			// not panic here; fall back to the documented cadence rather than
			// silently disabling the mid-call pause poll.
			interval = defaultToolWatchdogInterval
		}
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		tickCh = ticker.C
	}

	// The timeout arm is armed only when a positive ceiling is configured and
	// the tool is not exempt; otherwise the timer channel stays nil (disabled).
	// A zero value means "no timeout", preserving the pre-watchdog behavior for
	// callers that do not opt in.
	var timeoutCh <-chan time.Time
	if e.toolCallTimeout > 0 && !e.isToolCallTimeoutExempt(name) {
		timer := time.NewTimer(e.toolCallTimeout)
		defer timer.Stop()
		timeoutCh = timer.C
	}

	for {
		select {
		case out := <-done:
			return out.result, out.err
		case <-ctx.Done():
			// A tool that completed at the same instant must still win: its
			// result is real, and cancelling must not discard completed work.
			if res, ok, err := drainToolResult(done); ok {
				return res, err
			}
			return tools.ToolResult{}, ctx.Err()
		case <-tickCh:
			// Mirror the step-boundary precedence (executor.go): a cancelled
			// or deadline-exceeded run must report cancellation, never pause,
			// even when the pause ticker fires on the same selection. Without
			// this check, ctx.Done() and a tripping pause ready at once select
			// uniformly at random, so an already-cancelled run could surface as
			// a resumable pause instead of the cancellation it actually is.
			if err := ctx.Err(); err != nil {
				if res, ok, dErr := drainToolResult(done); ok {
					return res, dErr
				}
				return tools.ToolResult{}, err
			}
			// A tool that completed on this tick must win over the pause, so a
			// finished call is not dropped for a pause that will be observed at
			// the next step boundary anyway.
			if res, ok, dErr := drainToolResult(done); ok {
				return res, dErr
			}
			if e.pauseChecker(ctx) {
				e.log().Debug("tool call watchdog: pause observed while tool in flight",
					"tool", name)
				return tools.ToolResult{}, ErrPaused
			}
		case <-timeoutCh:
			// Mirror the pause-tick arm: a cancelled/deadline run must report
			// cancellation, not a timeout, when the ceiling and ctx.Done() are
			// ready in the same selection. Otherwise a Cancel or quit that
			// coincides with the ceiling could be misreported as a failed run.
			if err := ctx.Err(); err != nil {
				if res, ok, dErr := drainToolResult(done); ok {
					return res, dErr
				}
				return tools.ToolResult{}, err
			}
			// A tool that completed at the ceiling must win over the timeout,
			// so a finished call is not reported as timed out and re-issued.
			if res, ok, dErr := drainToolResult(done); ok {
				return res, dErr
			}
			e.log().Debug("tool call watchdog: tool call exceeded the configured timeout",
				"tool", name, "timeout", e.toolCallTimeout.String())
			// Cancel the tool's own context so the pending prompt / blocking
			// sub-work is dismissed rather than orphaned.
			cancelCall()
			return tools.ToolResult{}, ErrToolTimeout
		}
	}
}

// drainToolResult non-blockingly receives a tool result that has ALREADY been
// delivered, so every blocking arm can honour a completion ready at the same
// instant the arm fires: select picks uniformly at random among ready cases, so
// without this a finished tool's real result could be discarded in favour of the
// arm (a pause, a timeout, or a cancellation). It reports whether a result was
// available.
func drainToolResult(done <-chan toolCallOutcome) (tools.ToolResult, bool, error) {
	select {
	case out := <-done:
		return out.result, true, out.err
	default:
		return tools.ToolResult{}, false, nil
	}
}

// isPauseError reports whether err is the cooperative-pause sentinel.
func isPauseError(err error) bool { return errors.Is(err, ErrPaused) }

// isToolTimeoutError reports whether err is the tool-call timeout sentinel
// (possibly wrapped).
func isToolTimeoutError(err error) bool { return errors.Is(err, ErrToolTimeout) }

// isContextError reports whether err is a context cancellation or deadline
// error, either bare or wrapped.
func isContextError(err error) bool {
	return errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded)
}
