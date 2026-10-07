package agent

import (
	"context"
	"encoding/json"
	"errors"
	"time"

	"github.com/v0lka/sp4rk/tools"
)

// defaultToolWatchdogInterval is the cadence at which executeToolCall polls the
// cooperative pause checker while a tool call is in flight. It is deliberately
// short so a pause is observed within a fraction of a second even when the
// underlying tool is blocked (a blocking syscall cannot be interrupted, so the
// watchdog must poll).
const defaultToolWatchdogInterval = 250 * time.Millisecond

// toolWatchdogInterval is the live pause-poll cadence used by executeToolCall.
// It is a package-level var (not a const) purely as a test seam: production
// keeps defaultToolWatchdogInterval, and tests shorten it so a pause is
// observed without a multi-hundred-millisecond wait. It must not be mutated
// while an executeToolCall is running.
var toolWatchdogInterval = defaultToolWatchdogInterval

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
//     (the default) disables this arm.
//
// IMPORTANT: the underlying tool call is NOT cancelled when the pause, timeout,
// or cancellation arm fires. Go cannot preempt a blocked syscall, so the tool
// keeps running in its detached goroutine until it returns on its own; only the
// WAIT is abandoned. Callers must therefore treat the tool's side effects as
// having possibly occurred, and must not assume the call was stopped. The
// goroutine itself is bounded: it ends as soon as Execute returns (its send on
// the size-1 channel never blocks), so no goroutine leaks once the tool
// finishes — the one case that can outlive the run is a tool that never
// returns, which no watchdog can reclaim.
//
// executeToolCall is called from the single-tool and batch dispatch sites in
// executor_run.go. It must be called from the Run goroutine (the pause checker
// and toolCallTimeout fields are read here without synchronization, matching
// their set-before-Run contract).
func (e *Executor) executeToolCall(ctx context.Context, name string, input json.RawMessage) (tools.ToolResult, error) {
	done := make(chan toolCallOutcome, 1)
	go func() {
		res, err := e.tools.Execute(ctx, name, input)
		done <- toolCallOutcome{result: res, err: err}
	}()

	// Pause polling is armed only when a checker is installed; otherwise the
	// tick channel stays nil and its select arm is permanently disabled.
	var tickCh <-chan time.Time
	if e.pauseChecker != nil {
		ticker := time.NewTicker(toolWatchdogInterval)
		defer ticker.Stop()
		tickCh = ticker.C
	}

	// The timeout arm is armed only when a positive ceiling is configured;
	// otherwise the timer channel stays nil (disabled). A zero value means
	// "no timeout", preserving the pre-watchdog behavior for callers that do
	// not opt in.
	var timeoutCh <-chan time.Time
	if e.toolCallTimeout > 0 {
		timer := time.NewTimer(e.toolCallTimeout)
		defer timer.Stop()
		timeoutCh = timer.C
	}

	for {
		select {
		case out := <-done:
			return out.result, out.err
		case <-ctx.Done():
			return tools.ToolResult{}, ctx.Err()
		case <-tickCh:
			if e.pauseChecker(ctx) {
				e.log().Debug("tool call watchdog: pause observed while tool in flight",
					"tool", name)
				return tools.ToolResult{}, ErrPaused
			}
		case <-timeoutCh:
			e.log().Debug("tool call watchdog: tool call exceeded the configured timeout",
				"tool", name, "timeout", e.toolCallTimeout.String())
			return tools.ToolResult{}, ErrToolTimeout
		}
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
