package agent

import (
	"context"
	"encoding/json"
	"errors"
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

// withToolWatchdogInterval shortens the pause-poll cadence for one test and
// restores it on cleanup. The executor's watchdog reads this package var; tests
// are not parallel, so the override cannot race.
func withToolWatchdogInterval(t *testing.T, d time.Duration) {
	t.Helper()
	prev := toolWatchdogInterval
	toolWatchdogInterval = d
	t.Cleanup(func() { toolWatchdogInterval = prev })
}

// blockToolDescriptors is the single-tool catalog handed to Run by the watchdog
// tests.
func blockToolDescriptors() []tools.ToolDescriptor {
	return []tools.ToolDescriptor{{Name: "block_tool", Description: "blocks forever", Source: "core"}}
}

// runExecutorBounded runs exec.Run in a goroutine and fails the test if it does
// not return within the deadline. It proves the executor loop cannot be wedged
// by a stuck tool (no goroutine-driven hang).
func runExecutorBounded(t *testing.T, exec *Executor, ctx context.Context, defs []tools.ToolDescriptor) (*ExecutorResult, error) {
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
	withToolWatchdogInterval(t, 5*time.Millisecond)

	blocking := newBlockingToolExecutor()
	t.Cleanup(func() { close(blocking.release) }) // unblock the detached goroutine

	mockLLM := &mockLLMCaller{
		responses: []*llm.ChatResponse{
			llmResponseWithToolCall("call the blocking tool", "block_tool", json.RawMessage(`{}`)),
		},
	}
	exec := newExecutorDefaultHITL(mockLLM, blocking, &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
	exec.SetToolCallTimeout(40 * time.Millisecond)

	result, err := runExecutorBounded(t, exec, context.Background(), blockToolDescriptors())

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
	withToolWatchdogInterval(t, 5*time.Millisecond)

	blocking := newBlockingToolExecutor()
	t.Cleanup(func() { close(blocking.release) })

	mockLLM := &mockLLMCaller{
		responses: []*llm.ChatResponse{
			llmResponseWithToolCall("call the blocking tool", "block_tool", json.RawMessage(`{}`)),
		},
	}
	exec := newExecutorDefaultHITL(mockLLM, blocking, &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)

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

	result, err := runExecutorBounded(t, exec, context.Background(), blockToolDescriptors())

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
	withToolWatchdogInterval(t, 5*time.Millisecond)

	delayed := &delayedToolExecutor{delay: 20 * time.Millisecond, result: tools.ToolResult{Content: "hello world"}}
	mockLLM := &mockLLMCaller{
		responses: []*llm.ChatResponse{
			llmResponseWithToolCall("read", "read_file", json.RawMessage(`{"path": "/tmp/x"}`)),
			llmResponseFinish("got it", "file content here"),
		},
	}
	exec := newExecutorDefaultHITL(mockLLM, delayed, &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
	// Deliberately no SetToolCallTimeout: the zero value disables the ceiling.

	result, err := runExecutorBounded(t, exec, context.Background(), []tools.ToolDescriptor{
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
