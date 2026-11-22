package agent

import (
	"context"
	"encoding/json"
	"errors"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/v0lka/sp4rk/llm"
	"github.com/v0lka/sp4rk/tools"
)

// blockingToolExecutor is a ToolExecutor whose Execute blocks until release is
// closed (or ctx is cancelled), signalling started the first time it runs. It
// exists to prove that a tool which never returns cannot block the ReAct loop:
// the tool-call watchdog abandons the wait and the run completes.
type blockingToolExecutor struct {
	started chan struct{}
	release chan struct{}
	once    sync.Once
	runs    atomic.Int32
}

func newBlockingToolExecutor() *blockingToolExecutor {
	return &blockingToolExecutor{
		started: make(chan struct{}),
		release: make(chan struct{}),
	}
}

func (b *blockingToolExecutor) Execute(ctx context.Context, _ string, _ json.RawMessage) (tools.ToolResult, error) {
	b.runs.Add(1)
	b.once.Do(func() { close(b.started) })
	select {
	case <-b.release:
		return tools.ToolResult{Content: "released"}, nil
	case <-ctx.Done():
		return tools.ToolResult{}, ctx.Err()
	}
}

func (b *blockingToolExecutor) GetToolSource(string) string { return "core" }

func (b *blockingToolExecutor) IsToolUntrusted(string) bool { return false }

func (b *blockingToolExecutor) CacheStrategy(context.Context, string, json.RawMessage) tools.CacheMode {
	return tools.CacheModeDefault
}

// delayedToolExecutor returns successfully after a fixed delay, modelling a
// slow-but-finite tool. Used to prove the watchdog is opt-in: with no timeout
// the run still completes normally.
type delayedToolExecutor struct {
	delay  time.Duration
	result tools.ToolResult
	runs   atomic.Int32
}

func (d *delayedToolExecutor) Execute(ctx context.Context, _ string, _ json.RawMessage) (tools.ToolResult, error) {
	d.runs.Add(1)
	select {
	case <-time.After(d.delay):
		return d.result, nil
	case <-ctx.Done():
		return tools.ToolResult{}, ctx.Err()
	}
}

func (d *delayedToolExecutor) GetToolSource(string) string { return "core" }

func (d *delayedToolExecutor) IsToolUntrusted(string) bool { return false }

func (d *delayedToolExecutor) CacheStrategy(context.Context, string, json.RawMessage) tools.CacheMode {
	return tools.CacheModeDefault
}

// blockToolDescriptors is the single-tool catalog handed to Run by the watchdog
// tests.
func blockToolDescriptors() []tools.ToolDescriptor {
	return []tools.ToolDescriptor{{Name: "block_tool", Description: "blocks forever", Source: "core"}}
}

// runExecutorBounded runs exec.Run in a goroutine and fails the test if it does
// not return within the deadline. It proves the executor loop cannot be wedged
// by a stuck tool (no goroutine-driven hang).
func runExecutorBounded(ctx context.Context, t *testing.T, exec *Executor, defs []tools.ToolDescriptor) (*ExecutorResult, error) {
	t.Helper()
	type outcome struct {
		result *ExecutorResult
		err    error
	}
	done := make(chan outcome, 1)
	go func() {
		res, err := exec.Run(ctx, defs, newMockContextManager())
		done <- outcome{result: res, err: err}
	}()
	select {
	case out := <-done:
		return out.result, out.err
	case <-time.After(10 * time.Second):
		t.Fatal("Run did not return despite a blocking tool — the ReAct loop is wedged")
		return nil, nil
	}
}

func TestExecutor_Run_ToolCallTimeout_BlockingToolReturns(t *testing.T) {
	blocking := newBlockingToolExecutor()
	t.Cleanup(func() { close(blocking.release) }) // unblock the detached goroutine

	mockLLM := &mockLLMCaller{
		responses: []*llm.ChatResponse{
			llmResponseWithToolCall("call the blocking tool", "block_tool", json.RawMessage(`{}`)),
		},
	}
	exec := newExecutorDefaultHITL(mockLLM, blocking, &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig, withToolWatchdogInterval(5*time.Millisecond))
	exec.SetToolCallTimeout(40 * time.Millisecond)

	result, err := runExecutorBounded(context.Background(), t, exec, blockToolDescriptors())

	if !errors.Is(err, ErrToolTimeout) {
		t.Fatalf("expected ErrToolTimeout, got %v", err)
	}
	if result != nil {
		t.Errorf("expected a nil result on a tool-call timeout, got %+v", result)
	}
	if !strings.Contains(err.Error(), "block_tool") {
		t.Errorf("timeout error should name the tool, got %q", err.Error())
	}
	if got := blocking.runs.Load(); got != 1 {
		t.Errorf("blocking tool executed %d times, want 1", got)
	}
}

func TestExecutor_Run_ToolCallWatchdog_PauseWhileToolBlocked(t *testing.T) {
	blocking := newBlockingToolExecutor()
	t.Cleanup(func() { close(blocking.release) })

	mockLLM := &mockLLMCaller{
		responses: []*llm.ChatResponse{
			llmResponseWithToolCall("call the blocking tool", "block_tool", json.RawMessage(`{}`)),
		},
	}
	exec := newExecutorDefaultHITL(mockLLM, blocking, &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig, withToolWatchdogInterval(5*time.Millisecond))

	// The checker only trips once the tool has actually started, so the
	// step-boundary check (which runs BEFORE dispatch) cannot fire first —
	// this proves the pause is observed mid-tool-call, not merely at a
	// boundary. c0wrk pause does not cancel ctx, so this is the only path
	// that can observe it while a tool is stuck.
	var tripped atomic.Bool
	exec.SetPauseChecker(func(context.Context) bool {
		select {
		case <-blocking.started:
			tripped.Store(true)
			return true
		default:
			return false
		}
	})

	result, err := runExecutorBounded(context.Background(), t, exec, blockToolDescriptors())

	if !errors.Is(err, ErrPaused) {
		t.Fatalf("expected ErrPaused, got %v", err)
	}
	if result == nil {
		t.Fatal("expected a non-nil paused checkpoint")
	}
	if result.Finished {
		t.Error("expected Finished=false for a paused checkpoint")
	}
	if !tripped.Load() {
		t.Error("expected the pause checker to have tripped while the tool was in flight")
	}
}

func TestExecutor_Run_ToolCallTimeoutDisabled_RunCompletes(t *testing.T) {
	// With no timeout configured (the default) a slow-but-finite tool must run
	// to completion: the watchdog is strictly opt-in.
	delayed := &delayedToolExecutor{delay: 20 * time.Millisecond, result: tools.ToolResult{Content: "hello world"}}
	mockLLM := &mockLLMCaller{
		responses: []*llm.ChatResponse{
			llmResponseWithToolCall("read", "read_file", json.RawMessage(`{"path": "/tmp/x"}`)),
			llmResponseFinish("got it", "file content here"),
		},
	}
	exec := newExecutorDefaultHITL(mockLLM, delayed, &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig, withToolWatchdogInterval(5*time.Millisecond))
	// Deliberately no SetToolCallTimeout: the zero value disables the ceiling.

	result, err := runExecutorBounded(context.Background(), t, exec, []tools.ToolDescriptor{
		{Name: "read_file", Description: "read a file", Source: "core"},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.Finished {
		t.Error("expected Finished=true")
	}
	if result.Output != "file content here" {
		t.Errorf("Output = %q, want %q", result.Output, "file content here")
	}
	if got := delayed.runs.Load(); got != 1 {
		t.Errorf("slow tool executed %d times, want 1", got)
	}
}

func TestExecutor_ExecuteToolCall_ContextCancellationUnblocks(t *testing.T) {
	blocking := newBlockingToolExecutor()
	t.Cleanup(func() { close(blocking.release) })

	exec := NewExecutor(&mockLLMCaller{}, blocking, 10)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go func() {
		<-blocking.started
		cancel()
	}()

	_, err := exec.executeToolCall(ctx, "block_tool", json.RawMessage(`{}`))
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
}

// TestExecutor_ExecuteToolCall_CancelWinsOverPauseTick pins the watchdog's
// cancellation outcome for an UNTRIPPED pause signal: when a tool is in
// flight and the context is cancelled while the checker stays false (a
// genuine stop), the run must report the cancellation, never a resumable
// pause — mirroring the step-boundary check. Every select outcome converges
// on it: the dead-context classification consults the checker, and an
// untripped checker never translates. (A CANCELED context with a TRIPPED
// signal is the opposite case: the pause's abandonment manifestation, covered
// by TestExecutor_ExecuteToolCall_CanceledWhilePauseTripped. A DEADLINE death
// with a tripped signal — the remaining load-bearing case of the tick arm's
// ctx-first check — is pinned by
// TestExecutor_ExecuteToolCall_DeadlineBeatsTrippedPauseTick.)
func TestExecutor_ExecuteToolCall_CancelWinsOverPauseTick(t *testing.T) {
	blocking := newBlockingToolExecutor()
	t.Cleanup(func() { close(blocking.release) })

	exec := NewExecutor(&mockLLMCaller{}, blocking, 10, withToolWatchdogInterval(2*time.Millisecond))
	// A checker installed but never tripped: the pause tick is armed (the
	// ticker runs) yet the signal stays false, so cancellation keeps its
	// authority through every arm.
	exec.SetPauseChecker(func(context.Context) bool { return false })

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() {
		<-blocking.started
		cancel()
	}()

	_, err := exec.executeToolCall(ctx, "block_tool", json.RawMessage(`{}`))
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation must win over an untripped pause tick: got %v, want context.Canceled", err)
	}
}

// TestExecutor_ExecuteToolCall_DeadlineBeatsTrippedPauseTick pins the tick
// arm's remaining load-bearing ctx-first case: a deadline-exceeded run
// context is a budget death that stays loud even when the pause checker is
// tripped — the tripped tick must never mask it as a resumable pause. (A
// CANCELED death with a tripped signal is the opposite case: the abandonment
// manifestation, checkpointing as ErrPaused.) Every select outcome converges
// on this result: the ctx.Done arm and the tick arm's ctx-first check resolve
// through pauseManifestedAsCancel, which deliberately excludes
// DeadlineExceeded, and a deadline echo passes through resolveAbandonedDrain's
// Canceled-only gate untouched.
func TestExecutor_ExecuteToolCall_DeadlineBeatsTrippedPauseTick(t *testing.T) {
	blocking := newBlockingToolExecutor()
	t.Cleanup(func() { close(blocking.release) })

	exec := NewExecutor(&mockLLMCaller{}, blocking, 10, withToolWatchdogInterval(time.Millisecond))
	// A checker installed and always tripped: without the ctx-first check in
	// the tick arm this deadline-exceeded run would be reported as a plain
	// ErrPaused — the exact regression this test guards.
	exec.SetPauseChecker(func(context.Context) bool { return true })

	ctx, cancel := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
	defer cancel()

	_, err := exec.executeToolCall(ctx, "block_tool", json.RawMessage(`{}`))
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("a deadline-exceeded run must stay loud even with a tripped pause signal: got %v, want context.DeadlineExceeded", err)
	}
	if errors.Is(err, ErrPaused) {
		t.Fatalf("the tripped tick must never mask a budget death as a pause: %v", err)
	}
}

// TestExecutor_ExecuteToolCall_CanceledWhilePauseTripped_CheckpointsAsPause
// pins the abandonment manifestation: a CANCELED context while the pause
// signal is tripped is the pause reaching this executor through an abandoned
// ancestor context (a host dismissed a blocking tool on pause — e.g. a
// delegate wave), so executeToolCall reports the cooperative ErrPaused instead
// of the bare cancellation that would fail the subagent with "context
// canceled". The tool is deliberately context-oblivious (like a stuck syscall
// or the blocking delegate wave itself): a context-respecting tool would echo
// the cancellation through its own exit and be classified instead by
// resolveAbandonedDrain.
func TestExecutor_ExecuteToolCall_CanceledWhilePauseTripped_CheckpointsAsPause(t *testing.T) {
	oblivious := &ctxObliviousBlockingTool{
		started: make(chan struct{}),
		release: make(chan struct{}),
	}
	t.Cleanup(func() { close(oblivious.release) })

	exec := NewExecutor(&mockLLMCaller{}, oblivious, 10)
	exec.SetPauseChecker(func(context.Context) bool { return true })

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() {
		<-oblivious.started
		cancel()
	}()

	_, err := exec.executeToolCall(ctx, "block_tool", json.RawMessage(`{}`))
	if !errors.Is(err, ErrPaused) {
		t.Fatalf("a canceled context with a tripped pause signal must checkpoint as ErrPaused, got %v", err)
	}
	if errors.Is(err, context.Canceled) {
		t.Fatalf("the abandonment manifestation must surface as ErrPaused, not the bare cancellation: %v", err)
	}
}

// TestExecutor_ExecuteToolCall_ContextRespectingToolEcho_CheckpointsAsPause
// pins the drained-outcome classification: a tool that respects its context
// returns the cancellation through its own exit, and that drained outcome is
// the abandonment echo — with a tripped pause signal it must checkpoint as
// ErrPaused, not surface the bare "context canceled" as the tool result.
func TestExecutor_ExecuteToolCall_ContextRespectingToolEcho_CheckpointsAsPause(t *testing.T) {
	blocking := newBlockingToolExecutor()
	t.Cleanup(func() { close(blocking.release) })

	exec := NewExecutor(&mockLLMCaller{}, blocking, 10)
	exec.SetPauseChecker(func(context.Context) bool { return true })

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() {
		<-blocking.started
		cancel()
	}()

	_, err := exec.executeToolCall(ctx, "block_tool", json.RawMessage(`{}`))
	if !errors.Is(err, ErrPaused) {
		t.Fatalf("the abandonment echo of a context-respecting tool must checkpoint as ErrPaused, got %v", err)
	}
	if errors.Is(err, context.Canceled) {
		t.Fatalf("the abandonment echo must surface as ErrPaused, not the bare cancellation: %v", err)
	}
}

// ctxObliviousBlockingTool blocks until release regardless of context
// cancellation — the stand-in for a stuck syscall or a blocking delegate wave
// that cannot observe its derived context dying.
type ctxObliviousBlockingTool struct {
	started   chan struct{}
	release   chan struct{}
	onceStart sync.Once
	runs      atomic.Int32
}

func (b *ctxObliviousBlockingTool) Execute(_ context.Context, _ string, _ json.RawMessage) (tools.ToolResult, error) {
	b.runs.Add(1)
	b.onceStart.Do(func() { close(b.started) })
	<-b.release
	return tools.ToolResult{Content: "released"}, nil
}

func (b *ctxObliviousBlockingTool) GetToolSource(string) string { return "core" }

func (b *ctxObliviousBlockingTool) IsToolUntrusted(string) bool { return false }

func (b *ctxObliviousBlockingTool) CacheStrategy(context.Context, string, json.RawMessage) tools.CacheMode {
	return tools.CacheModeDefault
}

// TestExecutor_ExecuteToolCall_CancelWinsOverTimeoutArm pins the timeout arm's
// cancellation precedence: a run cancelled while a tool is in flight must report
// the cancellation, not ErrToolTimeout, even when the per-call ceiling fires on
// the same selection (mirroring the pause-tick arm).
func TestExecutor_ExecuteToolCall_CancelWinsOverTimeoutArm(t *testing.T) {
	blocking := newBlockingToolExecutor()
	t.Cleanup(func() { close(blocking.release) })

	exec := NewExecutor(&mockLLMCaller{}, blocking, 10, withToolWatchdogInterval(time.Millisecond))
	exec.SetToolCallTimeout(time.Millisecond)
	// A checker installed but never tripped: the pause tick is armed yet the
	// signal stays false, so the cancellation keeps its authority over the
	// timeout arm. (A tripped signal with a canceled context is the pause's
	// abandonment manifestation and checkpoints as ErrPaused instead — that
	// precedence is pinned separately.)
	exec.SetPauseChecker(func(context.Context) bool { return false })

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() {
		<-blocking.started
		cancel()
	}()

	_, err := exec.executeToolCall(ctx, "block_tool", json.RawMessage(`{}`))
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation must win over a simultaneously-ready timeout arm: got %v, want context.Canceled", err)
	}
}

func TestNewExecutor_ToolCallTimeoutSurface(t *testing.T) {
	if DefaultToolCallTimeout != 5*time.Minute {
		t.Fatalf("DefaultToolCallTimeout = %v, want 5m", DefaultToolCallTimeout)
	}
	exec := NewExecutor(&mockLLMCaller{}, newMockToolExecutor(), 5, WithToolCallTimeout(2*time.Second))
	if exec.toolCallTimeout != 2*time.Second {
		t.Fatalf("WithToolCallTimeout wiring: toolCallTimeout = %v, want 2s", exec.toolCallTimeout)
	}
	exec.SetToolCallTimeout(0)
	if exec.toolCallTimeout != 0 {
		t.Fatalf("SetToolCallTimeout(0) must disable the ceiling, got %v", exec.toolCallTimeout)
	}
}

// --- Exemption set ----------------------------------------------------------

// TestNewExecutor_ToolCallTimeoutExemptDefault pins the built-in default: a
// fresh executor exempts exactly the documented orchestration tools and
// nothing else.
func TestNewExecutor_ToolCallTimeoutExemptDefault(t *testing.T) {
	exec := NewExecutor(&mockLLMCaller{}, newMockToolExecutor(), 5)
	for _, name := range DefaultToolCallTimeoutExemptTools() {
		if !exec.isToolCallTimeoutExempt(name) {
			t.Errorf("new executor must exempt %q by default", name)
		}
	}
	if exec.isToolCallTimeoutExempt("any_mcp_tool") {
		t.Error("a tool outside the default set must not be exempt by default")
	}
}

// TestDefaultToolCallTimeoutExemptTools_BuiltinSet pins the accessor's shape
// (sorted, fresh copy) that host config surfaces use as their default.
func TestDefaultToolCallTimeoutExemptTools_BuiltinSet(t *testing.T) {
	want := []string{"ask_user", "declare_plan", "delegate", "execute_plan", "propose_goal"}
	got := DefaultToolCallTimeoutExemptTools()
	if !slices.Equal(got, want) {
		t.Fatalf("DefaultToolCallTimeoutExemptTools() = %v, want %v", got, want)
	}
	// The returned slice is a copy: mutating it must not corrupt the default.
	got[0] = "mutated"
	if again := DefaultToolCallTimeoutExemptTools(); !slices.Equal(again, want) {
		t.Fatalf("mutating the returned slice changed the default set: %v", again)
	}
}

// TestExecutor_SetToolCallTimeoutExempt_ExemptToolSurvivesCeiling pins the
// host extension point: a tool named in the replacement set is never bounded,
// even with a ceiling far below the tool's run time — the scenario of an
// MCP-backed tool that legitimately waits on a slow server.
func TestExecutor_SetToolCallTimeoutExempt_ExemptToolSurvivesCeiling(t *testing.T) {
	blocking := newBlockingToolExecutor()
	var releaseOnce sync.Once
	t.Cleanup(func() { releaseOnce.Do(func() { close(blocking.release) }) })

	mockLLM := &mockLLMCaller{
		responses: []*llm.ChatResponse{
			llmResponseWithToolCall("call the blocking tool", "block_tool", json.RawMessage(`{}`)),
			llmResponseFinish("done", "finished"),
		},
	}
	exec := NewExecutor(mockLLM, blocking, 10)
	exec.SetToolCallTimeout(20 * time.Millisecond)
	// The extension the setter exists for: cover this host's own long-running
	// tool. Without it the 20ms ceiling abandons the call and fails the run.
	exec.SetToolCallTimeoutExempt("block_tool")

	// Release the tool as soon as it is in flight; ordering is channel-driven
	// (no sleeps), so the ceiling can only fire if the exemption is broken.
	go func() {
		<-blocking.started
		releaseOnce.Do(func() { close(blocking.release) })
	}()

	result, err := runExecutorBounded(context.Background(), t, exec, blockToolDescriptors())
	if err != nil {
		t.Fatalf("exempt tool must not be bounded by the ceiling: %v", err)
	}
	released := false
	for _, s := range result.Steps {
		if strings.Contains(s.Observation, "released") {
			released = true
			break
		}
	}
	if !released {
		t.Error("expected the exempt blocking tool's result in the trajectory")
	}
}

// TestExecutor_SetToolCallTimeoutExempt_ClearRestoresBound: clearing the set
// (no names) bounds every tool again.
func TestExecutor_SetToolCallTimeoutExempt_ClearRestoresBound(t *testing.T) {
	blocking := newBlockingToolExecutor()
	t.Cleanup(func() { close(blocking.release) })

	mockLLM := &mockLLMCaller{
		responses: []*llm.ChatResponse{
			llmResponseWithToolCall("call the blocking tool", "block_tool", json.RawMessage(`{}`)),
			llmResponseFinish("done", "finished"),
		},
	}
	exec := NewExecutor(mockLLM, blocking, 10)
	exec.SetToolCallTimeout(20 * time.Millisecond)
	exec.SetToolCallTimeoutExempt("block_tool")
	exec.SetToolCallTimeoutExempt() // clear

	_, err := runExecutorBounded(context.Background(), t, exec, blockToolDescriptors())
	if !errors.Is(err, ErrToolTimeout) {
		t.Fatalf("cleared exemption must restore the ceiling: got %v, want ErrToolTimeout", err)
	}
}
