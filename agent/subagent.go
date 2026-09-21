package agent

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"runtime/debug"
	"sync"
	"time"

	"github.com/v0lka/sp4rk/tools"
)

// SubAgentTask bundles an agent with its task tools, context manager, and events.
type SubAgentTask struct {
	StepID         string
	Executor       *Executor
	CM             ContextManager
	TaskTools      []tools.ToolDescriptor
	TaskDesc       string             // task description (for SubAgentLaunch event)
	Emitter        Events             // event emitter (nil-safe)
	TodoUpdateFunc StepTodoUpdateFunc // optional callback for update_checklist tool
}

// RunSubAgent starts the executor in a goroutine and returns a channel for the result.
// The goroutine respects context cancellation — when ctx is cancelled,
// executor.Run will return because its LLM calls and tool executions use the same context.
// emitter is optional (nil-safe) for console output.
func RunSubAgent(ctx context.Context, stepID string, executor *Executor, cm ContextManager, taskTools []tools.ToolDescriptor, taskDesc string, emitter Events, todoUpdateFunc StepTodoUpdateFunc) (resultCh <-chan SubAgentResult) {
	// Use NoopEvents if nil to avoid nil checks
	if emitter == nil {
		emitter = &NoopEvents{}
	}
	ch := make(chan SubAgentResult, 1)

	go func() {
		defer close(ch)
		startTime := time.Now()

		// Log through the executor's configured logger rather than the
		// process-global slog, matching the SDK-wide logger convention. The
		// executor is non-nil on every real path; the discard fallback keeps the
		// panic handler from itself panicking on a hypothetical nil executor.
		logger := slog.New(slog.DiscardHandler)
		if executor != nil {
			logger = executor.log()
		}

		// A panic anywhere in sub-agent execution must not tear down the host
		// process. Execution runs on this goroutine, so the host cannot wrap it
		// with its own recover — guard here, log the stack, emit a failed
		// completion, and hand the conductor an error so the step fails in
		// isolation and execution continues.
		defer func() {
			if r := recover(); r != nil {
				logger.Error("subagent execution panicked",
					"step_id", stepID,
					"panic", r,
					"stack", string(debug.Stack()),
				)
				panicErr := fmt.Errorf("subagent %s panicked: %v", stepID, r)
				emitter.SubAgentComplete(stepID, false, time.Since(startTime), panicErr.Error())
				ch <- SubAgentResult{
					StepID: stepID,
					Error:  panicErr,
				}
			}
		}()

		// Emit subagent launch
		emitter.SubAgentLaunch(stepID, taskDesc)

		// Set task context for tool execution
		ctx = tools.WithTaskContext(ctx, taskDesc)

		// Set step ID so file tracker and other context-aware tools know the current step
		ctx = WithStepID(ctx, stepID)

		// Set checklist update callback for update_checklist tool
		if todoUpdateFunc != nil {
			ctx = WithStepTodoUpdateFunc(ctx, todoUpdateFunc)
		}

		result, err := executor.Run(ctx, taskTools, cm)

		duration := time.Since(startTime)
		success := err == nil && result.Finished

		// Defense-in-depth: even if the executor returned Finished=true, a
		// failure-mode where the model printed tool-call syntax as text
		// (instead of emitting a tool_use block) is NOT a success. The
		// handleImplicitFinish detector should have aborted such cases with
		// Finished=false, but this guard catches any escape. The model's text
		// may land in Output (a normal finish) or in Summary (a stop-tool
		// termination, whose Output holds the tool's short confirmation while
		// the model's prose is carried in Summary), so probe both.
		if success && (DetectToolCallSyntaxInContent(result.Output) || DetectToolCallSyntaxInContent(result.Summary)) {
			success = false
			if err == nil {
				err = errors.New("model printed tool-call syntax as text instead of using tool_use blocks")
			}
		}

		// Compute the terminal failure reason once so the emitted event and the
		// returned SubAgentResult.Error agree. On success errMsg stays ""; on a
		// hard error it is err.Error(); on an incomplete (never-finished) run it
		// prefers the executor's structured AbortReason, then falls back to Output
		// (a short "Aborted: …" message for the circuit-breaker / fruitless /
		// max-steps branches), then to a generic one when both are empty.
		// AbortReason exists precisely so a prose answer (mutation-gate rejection)
		// is never surfaced as the failure cause.
		errMsg := ""
		if err != nil {
			errMsg = err.Error()
		} else if !result.Finished {
			errMsg = "step execution did not complete within max steps"
			if result.AbortReason != "" {
				errMsg = result.AbortReason
			} else if result.Output != "" {
				errMsg = result.Output
			}
		}

		// Emit subagent completion. A cooperative pause (PauseChecker tripped at
		// a step boundary, surfaced as ErrPaused) is a recoverable checkpoint,
		// not a failure — emit a distinct SubAgentPaused event instead of
		// SubAgentComplete(success=false) so hosts can tell the two apart.
		if isPaused(err) {
			emitter.SubAgentPaused(stepID, duration)
		} else {
			emitter.SubAgentComplete(stepID, success, duration, errMsg)
		}

		if err != nil {
			var steps []Step
			if result != nil {
				steps = result.Steps
			}
			ch <- SubAgentResult{StepID: stepID, Steps: steps, Error: err}
			return
		}

		// Treat incomplete execution (no proper finish) as a step failure,
		// reusing the reason computed above as the error message.
		if !result.Finished {
			ch <- SubAgentResult{StepID: stepID, Output: result.Output, Steps: result.Steps, Summary: result.Summary, Error: errors.New(errMsg)}
			return
		}

		ch <- SubAgentResult{
			StepID:  stepID,
			Output:  result.Output,
			Steps:   result.Steps,
			Summary: result.Summary,
		}
	}()

	return ch
}

// RunSubAgentsParallelOption configures RunSubAgentsParallel.
type RunSubAgentsParallelOption func(*runSubAgentsParallelConfig)

// runSubAgentsParallelConfig carries the resolved options.
type runSubAgentsParallelConfig struct {
	// maxConcurrency caps how many subagents execute at once. <= 0 means
	// unlimited (the historical, unbounded fan-out).
	maxConcurrency int
}

// WithMaxParallelSubagents caps the number of subagents that execute
// concurrently within a single RunSubAgentsParallel invocation. When n <= 0 the
// fan-out is unbounded (the historical behavior). A positive n bounds peak
// concurrency — and therefore the peak rate of subagent lifecycle events —
// without changing the result set or its input order: subagents beyond the cap
// are queued and started as running slots free up. The cap is per-invocation
// (see the RunSubAgentsParallel note); it is not a process-wide limit.
func WithMaxParallelSubagents(n int) RunSubAgentsParallelOption {
	return func(c *runSubAgentsParallelConfig) { c.maxConcurrency = n }
}

// RunSubAgentsParallel runs multiple SubAgents concurrently and collects results.
// Returns results in input order (not completion order); a slow agent blocks
// all subsequent results from being returned.
//
// By default every subagent is launched at once (unbounded fan-out). Pass
// WithMaxParallelSubagents to cap how many run concurrently within THIS
// invocation. The cap is per-call, not process-wide: a host that fires several
// RunSubAgentsParallel calls (e.g. a delegate tool and a plan-wave dispatcher
// in the same turn) or nests a delegation must enforce any global limit itself,
// because each call resolves its own semaphore from its own option.
func RunSubAgentsParallel(ctx context.Context, agents []SubAgentTask, opts ...RunSubAgentsParallelOption) (results []SubAgentResult) {
	if len(agents) == 0 {
		return nil
	}

	var cfg runSubAgentsParallelConfig
	for _, opt := range opts {
		if opt != nil {
			opt(&cfg)
		}
	}

	// Resolve the effective cap: <= 0 or a cap >= the task count means
	// "run everything at once".
	maxConcurrency := cfg.maxConcurrency
	if maxConcurrency <= 0 || maxConcurrency > len(agents) {
		maxConcurrency = len(agents)
	}

	// Index-addressed results preserve input order regardless of completion
	// order; each worker writes only its own slot. A semaphore bounds how many
	// workers hold a running slot at once — a worker blocked on acquisition has
	// not called RunSubAgent yet, so no SubAgentLaunch event fires for it until
	// a slot frees, which is exactly what lowers the peak event rate.
	results = make([]SubAgentResult, len(agents))
	sem := make(chan struct{}, maxConcurrency)
	var wg sync.WaitGroup
	for i := range agents {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			sem <- struct{}{}        // acquire a running slot
			defer func() { <-sem }() // release it once this subagent settles
			ag := agents[i]
			results[i] = <-RunSubAgent(ctx, ag.StepID, ag.Executor, ag.CM, ag.TaskTools, ag.TaskDesc, ag.Emitter, ag.TodoUpdateFunc)
		}(i)
	}
	wg.Wait()

	return results
}

// isPaused reports whether err is the cooperative pause sentinel (ErrPaused)
// returned by Executor.Run when a PauseChecker trips at a step boundary. A
// paused sub-agent is a recoverable checkpoint, not a failure: hosts receive
// the SubAgentPaused event (instead of SubAgentComplete(success=false)) and
// the preserved trajectory in SubAgentResult for a later resume.
func isPaused(err error) bool { return errors.Is(err, ErrPaused) }
