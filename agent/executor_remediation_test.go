package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/v0lka/sp4rk/llm"
	"github.com/v0lka/sp4rk/tools"
)

// scenarioHITL is a test HITLHandler with per-tool deny and modify decisions.
// Tools without an entry are allowed unchanged.
type scenarioHITL struct {
	modify map[string]json.RawMessage
	deny   map[string]string
}

func (h *scenarioHITL) OnToolCall(_ context.Context, toolName string, _ json.RawMessage) (*HITLToolDecision, error) {
	if reason, ok := h.deny[toolName]; ok {
		return &HITLToolDecision{Allow: false, Reason: reason}, nil
	}
	if modified, ok := h.modify[toolName]; ok {
		return &HITLToolDecision{Allow: true, ModifiedInput: modified}, nil
	}
	d := allowDecisionSentinel
	return &d, nil
}

func (h *scenarioHITL) OnStepLimit(_ context.Context, _, _ int, _ string) (StepLimitResponse, error) {
	return StepLimitDeny, nil
}

// callIndexRecorder captures the (index, toolName) pairs emitted via
// ToolCall — the identity a host's localToolIDs map is keyed by.
type callIndexRecorder struct {
	NoopEvents
	mu        sync.Mutex
	toolCalls []string
}

func (r *callIndexRecorder) ToolCall(stepNum, callIdx int, toolName, argsPreview, source string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.toolCalls = append(r.toolCalls, fmt.Sprintf("%d:%s", callIdx, toolName))
}

// TestExecutor_FinishInsideBatchTerminatesRun verifies a finish call wrapped
// inside the batch meta-tool terminates the run exactly like a direct finish:
// Finished=true, the answer as Output, exactly one LLM call, and no registry
// dispatch of "finish". Without the interception a batched finish is executed
// as an ordinary tool, its answer is recorded as a plain observation, and the
// run keeps going past the model's final answer.
func TestExecutor_FinishInsideBatchTerminatesRun(t *testing.T) {
	answer := "the actual deliverable"
	batchInput, err := json.Marshal(map[string]any{
		"calls": []map[string]any{
			{"tool": "search", "input": map[string]string{"query": "facts"}},
			{"tool": "finish", "input": map[string]string{"answer": answer}},
			// Must never execute — the finish sub-call ends the run above.
			{"tool": "search", "input": map[string]string{"query": "after finish"}},
		},
	})
	if err != nil {
		t.Fatalf("marshal batch input: %v", err)
	}

	mockLLM := &mockLLMCaller{
		responses: []*llm.ChatResponse{
			llmResponseWithToolCall("batching the completion", tools.ToolBatch, batchInput),
			// Must never be reached.
			llmResponseFinish("should not run", "fallback answer"),
		},
	}
	mockTools := newMockToolExecutor()

	exec := newExecutorDefaultHITL(mockLLM, mockTools, &mockTokenCounter{}, 10, &recordingEvents{}, false, ToolResultBudget{}, defaultCircuitBreakerConfig)

	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: tools.ToolBatch, Description: "batch", Source: "core"},
		{Name: "search", Description: "search", Source: "core"},
	}, newMockContextManager())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.Finished {
		t.Error("Finished = false, want true (a finish sub-call must terminate the run)")
	}
	if result.Output != answer {
		t.Errorf("Output = %q, want the batched finish answer %q", result.Output, answer)
	}
	if len(mockLLM.calls) != 1 {
		t.Errorf("LLM called %d times, want 1 — the run must end inside the batch", len(mockLLM.calls))
	}
	for _, c := range mockTools.calls {
		if c.Name == "finish" {
			t.Errorf("finish was dispatched to the registry (input %s) — it must be intercepted inline", c.Input)
		}
		if c.Name == "search" && strings.Contains(string(c.Input), "after finish") {
			t.Errorf("sub-call after the finish executed (%s) — the run must end immediately", c.Input)
		}
	}
}

// TestExecutor_FinishInsideBatch_HonorsFinishGuard verifies the batched
// finish path applies the finish guard: on rejection the guard's nudge is
// injected and the model retries, exactly as with a direct finish.
func TestExecutor_FinishInsideBatch_HonorsFinishGuard(t *testing.T) {
	batchInput, err := json.Marshal(map[string]any{
		"calls": []map[string]any{
			{"tool": "finish", "input": map[string]string{"answer": "premature"}},
		},
	})
	if err != nil {
		t.Fatalf("marshal batch input: %v", err)
	}

	mockLLM := &mockLLMCaller{
		responses: []*llm.ChatResponse{
			llmResponseWithToolCall("batching the completion", tools.ToolBatch, batchInput),
			llmResponseFinish("wrapped up after the nudge", "done for real"),
		},
	}
	mockTools := newMockToolExecutor()

	exec := newExecutorDefaultHITL(mockLLM, mockTools, &mockTokenCounter{}, 10, &recordingEvents{}, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
	guardCalls := 0
	exec.SetFinishGuard(func(context.Context) error {
		guardCalls++
		if guardCalls == 1 {
			return errors.New("you have 1 pending async delegation(s): sub-1")
		}
		return nil
	})

	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: tools.ToolBatch, Description: "batch", Source: "core"},
	}, newMockContextManager())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if guardCalls < 2 {
		t.Fatalf("finish guard consulted %d times, want >= 2 (reject then accept)", guardCalls)
	}
	if !result.Finished {
		t.Error("Finished = false, want true (the run continued past the guard-rejected batched finish)")
	}
	if result.Output != "done for real" {
		t.Errorf("Output = %q, want the post-nudge answer", result.Output)
	}
	nudged := false
	for _, s := range result.Steps {
		if strings.Contains(s.UserNudge, "pending async delegation") {
			nudged = true
		}
	}
	if !nudged {
		t.Error("guard nudge step not found in the trajectory")
	}
}

// nudgeThoughtFixture builds a one-guard-rejection executor plus a response
// carrying one encrypted reasoning item, for the nudge-thought-carry tests.
type nudgeThoughtFixture struct {
	exec    *Executor
	mockLLM *mockLLMCaller
}

func newNudgeThoughtFixture(t *testing.T, firstResponse *llm.ChatResponse) *nudgeThoughtFixture {
	t.Helper()
	mockLLM := &mockLLMCaller{
		responses: []*llm.ChatResponse{
			firstResponse,
			llmResponseFinish("wrapped up", "done"),
		},
	}
	exec := newExecutorDefaultHITL(mockLLM, newMockToolExecutor(), &mockTokenCounter{}, 10, &recordingEvents{}, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
	guardCalls := 0
	exec.SetFinishGuard(func(context.Context) error {
		guardCalls++
		if guardCalls == 1 {
			return errors.New("not so fast")
		}
		return nil
	})
	return &nudgeThoughtFixture{exec: exec, mockLLM: mockLLM}
}

// TestExecutor_FinishGuardNudgeThoughtGating pins the gate-nudge thought
// carry: the response's thought rides a finish-gate nudge ONLY when the
// finish is the response's first materialized call. With an earlier sibling
// step the thought is already carried there, and an ungated copy on the
// nudge would render the assistant turn twice in the next prompt.
func TestExecutor_FinishGuardNudgeThoughtGating(t *testing.T) {
	items := []llm.ReasoningItem{{ID: "rs_1", Summary: "plan", EncryptedContent: "gAAAA-encrypted"}}

	t.Run("finish not first call: nudge carries no thought", func(t *testing.T) {
		// [read_file, finish]: the read_file step materializes groupSteps[0]
		// and carries the thought; the nudge must not repeat it.
		multi := llmResponseWithMultipleToolCalls("reading then done", []llm.ToolCall{
			{ID: "call_1", Name: "read_file", Input: json.RawMessage(`{"path":"/tmp/a"}`)},
			{ID: "call_2", Name: "finish", Input: json.RawMessage(`{"answer":"premature"}`)},
		})
		multi.Message.ReasoningItems = items
		fx := newNudgeThoughtFixture(t, multi)

		result, err := fx.exec.Run(context.Background(), []tools.ToolDescriptor{
			{Name: "read_file", Description: "read", Source: "core"},
		}, newMockContextManager())
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(result.Steps) < 2 {
			t.Fatalf("steps = %d, want >= 2", len(result.Steps))
		}
		if got, want := result.Steps[0].Thought, "reading then done"; got != want {
			t.Errorf("first step Thought = %q, want %q (groupSteps[0] carries the turn's thought)", got, want)
		}
		if got := result.Steps[1].UserNudge; !strings.Contains(got, "not so fast") {
			t.Fatalf("Steps[1] is not the guard nudge (UserNudge = %q)", got)
		}
		if got := result.Steps[1].Thought; got != "" {
			t.Errorf("guard nudge Thought = %q, want empty — the thought is already carried by groupSteps[0]; a copy duplicates the assistant turn", got)
		}
		if len(result.Steps[1].ReasoningItems) != 0 {
			t.Errorf("guard nudge ReasoningItems = %+v, want none (already carried by groupSteps[0])", result.Steps[1].ReasoningItems)
		}
	})

	t.Run("finish is only call: nudge carries the thought", func(t *testing.T) {
		// A lone finish: the nudge is the response's ONLY materialized step,
		// so it MUST carry the thought (and items) or the assistant turn is
		// lost entirely.
		single := llmResponseFinish("premature wrap-up", "premature")
		single.Message.ReasoningItems = items
		fx := newNudgeThoughtFixture(t, single)

		result, err := fx.exec.Run(context.Background(), nil, newMockContextManager())
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(result.Steps) < 1 {
			t.Fatalf("steps = %d, want >= 1", len(result.Steps))
		}
		nudge := result.Steps[0]
		if !strings.Contains(nudge.UserNudge, "not so fast") {
			t.Fatalf("Steps[0] is not the guard nudge (UserNudge = %q)", nudge.UserNudge)
		}
		if got, want := nudge.Thought, "premature wrap-up"; got != want {
			t.Errorf("guard nudge Thought = %q, want %q — the nudge is the response's only step and must carry the assistant turn", got, want)
		}
		if len(nudge.ReasoningItems) != 1 {
			t.Errorf("guard nudge ReasoningItems = %+v, want the response's single reasoning item", nudge.ReasoningItems)
		}
	})
}

// TestExecutor_StopToolGuardNudgeThoughtEmpty pins the stop-tool guard nudge:
// the stop-tool step is always materialized before the nudge is built, so the
// nudge must never carry the response's thought (a copy would duplicate the
// assistant turn).
func TestExecutor_StopToolGuardNudgeThoughtEmpty(t *testing.T) {
	mockLLM := &mockLLMCaller{
		responses: []*llm.ChatResponse{
			llmResponseWithToolCall("declaring", "declare_goal_status", json.RawMessage(`{"status":"not_met"}`)),
			llmResponseFinish("wrapped up", "done"),
		},
	}
	mockTools := newMockToolExecutor()
	mockTools.results["declare_goal_status"] = tools.ToolResult{Content: "Verdict recorded."}

	exec := newExecutorDefaultHITL(mockLLM, mockTools, &mockTokenCounter{}, 10, &recordingEvents{}, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
	exec.SetStopTools("declare_goal_status")
	guardCalls := 0
	exec.SetFinishGuard(func(context.Context) error {
		guardCalls++
		if guardCalls == 1 {
			return errors.New("pending work")
		}
		return nil
	})

	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "declare_goal_status", Description: "declare", Source: "core"},
	}, newMockContextManager())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Steps) < 2 {
		t.Fatalf("steps = %d, want >= 2", len(result.Steps))
	}
	if got, want := result.Steps[0].Thought, "declaring"; got != want {
		t.Errorf("stop-tool step Thought = %q, want %q", got, want)
	}
	if !strings.Contains(result.Steps[1].UserNudge, "pending work") {
		t.Fatalf("Steps[1] is not the stop-tool guard nudge (UserNudge = %q)", result.Steps[1].UserNudge)
	}
	if got := result.Steps[1].Thought; got != "" {
		t.Errorf("stop-tool guard nudge Thought = %q, want empty — the stop-tool step already carries the assistant turn", got)
	}
}

// TestBatchInterceptedSubCallThoughtGated pins the intercepted batch sub-call
// steps (nested-batch guard, HITL reject): the thought is gated exactly like
// the success path — carried only by the FIRST materialized sub-call of the
// FIRST call. A lone batch (ResponseGroup 0) renders every sub-call step
// standalone, so an ungated copy would emit the response's thought twice.
func TestBatchInterceptedSubCallThoughtGated(t *testing.T) {
	buildBatch := func(t *testing.T, calls []map[string]any) llm.ToolCall {
		t.Helper()
		raw, err := json.Marshal(map[string]any{"calls": calls})
		if err != nil {
			t.Fatalf("marshal batch input: %v", err)
		}
		return llm.ToolCall{ID: "batch_1", Name: tools.ToolBatch, Input: raw}
	}

	t.Run("nested batch as non-first sub-call", func(t *testing.T) {
		batch := buildBatch(t, []map[string]any{
			{"tool": "search", "input": map[string]string{"query": "q"}},
			{"tool": "batch", "input": map[string]any{}},
		})
		resp := &llm.ChatResponse{
			Message:    llm.Message{Role: "assistant", ToolCalls: []llm.ToolCall{batch}},
			Usage:      llm.TokenUsage{InputTokens: 100, OutputTokens: 50},
			StopReason: "tool_use",
		}
		exec := newExecutorDefaultHITL(&mockLLMCaller{}, newMockToolExecutor(), &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
		exec.emitter = &NoopEvents{}
		state := &runState{effectiveMaxSteps: 10}
		cw := newMockContextManager()

		if _, _, err := exec.processSingleToolCall(context.Background(), batch, 0, resp.Message.ToolCalls, resp, "thinking", state, cw); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(state.allSteps) != 2 {
			t.Fatalf("steps = %d, want 2", len(state.allSteps))
		}
		if got, want := state.allSteps[0].Thought, "thinking"; got != want {
			t.Errorf("first sub-call Thought = %q, want %q", got, want)
		}
		if got := state.allSteps[1].Thought; got != "" {
			t.Errorf("nested-batch guard step Thought = %q, want empty — lone batch renders sub-call steps standalone, so a copy duplicates the assistant turn", got)
		}
	})

	t.Run("HITL-rejected non-first sub-call", func(t *testing.T) {
		batch := buildBatch(t, []map[string]any{
			{"tool": "read_file", "input": map[string]string{"path": "a.txt"}},
			{"tool": "delete_file", "input": map[string]string{"path": "b.txt"}},
		})
		resp := &llm.ChatResponse{
			Message:    llm.Message{Role: "assistant", ToolCalls: []llm.ToolCall{batch}},
			Usage:      llm.TokenUsage{InputTokens: 100, OutputTokens: 50},
			StopReason: "tool_use",
		}
		exec := newExecutorDefaultHITL(&mockLLMCaller{}, newMockToolExecutor(), &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig,
			WithHITL(&scenarioHITL{deny: map[string]string{"delete_file": "destructive"}}))
		exec.emitter = &NoopEvents{}
		state := &runState{effectiveMaxSteps: 10}
		cw := newMockContextManager()

		if _, _, err := exec.processSingleToolCall(context.Background(), batch, 0, resp.Message.ToolCalls, resp, "thinking", state, cw); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(state.allSteps) != 2 {
			t.Fatalf("steps = %d, want 2", len(state.allSteps))
		}
		if got, want := state.allSteps[0].Thought, "thinking"; got != want {
			t.Errorf("first sub-call Thought = %q, want %q", got, want)
		}
		if !strings.Contains(state.allSteps[1].Observation, "[Tool call rejected") {
			t.Fatalf("second sub-call was not rejected (Observation = %q)", state.allSteps[1].Observation)
		}
		if got := state.allSteps[1].Thought; got != "" {
			t.Errorf("HITL-reject step Thought = %q, want empty — a copy duplicates the assistant turn", got)
		}
	})

	t.Run("HITL-rejected FIRST sub-call still carries the thought", func(t *testing.T) {
		batch := buildBatch(t, []map[string]any{
			{"tool": "delete_file", "input": map[string]string{"path": "b.txt"}},
			{"tool": "read_file", "input": map[string]string{"path": "a.txt"}},
		})
		resp := &llm.ChatResponse{
			Message:    llm.Message{Role: "assistant", ToolCalls: []llm.ToolCall{batch}},
			Usage:      llm.TokenUsage{InputTokens: 100, OutputTokens: 50},
			StopReason: "tool_use",
		}
		exec := newExecutorDefaultHITL(&mockLLMCaller{}, newMockToolExecutor(), &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig,
			WithHITL(&scenarioHITL{deny: map[string]string{"delete_file": "destructive"}}))
		exec.emitter = &NoopEvents{}
		state := &runState{effectiveMaxSteps: 10}
		cw := newMockContextManager()

		if _, _, err := exec.processSingleToolCall(context.Background(), batch, 0, resp.Message.ToolCalls, resp, "thinking", state, cw); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(state.allSteps) != 2 {
			t.Fatalf("steps = %d, want 2", len(state.allSteps))
		}
		if got, want := state.allSteps[0].Thought, "thinking"; got != want {
			t.Errorf("rejected FIRST sub-call Thought = %q, want %q (it is the response's only carrier)", got, want)
		}
		if got := state.allSteps[1].Thought; got != "" {
			t.Errorf("second sub-call Thought = %q, want empty", got)
		}
	})
}

// TestExecutor_ResumeSeedsResponseGroupCounter verifies a resumed executor
// never reuses a ResponseGroup id already present in the seeded trajectory:
// the counter is per-Executor, and without seeding it a fresh Executor's
// first multi-call response would collide with the seeded group, merging the
// two turns into one assistant message (dropping the resumed turn's thought).
func TestExecutor_ResumeSeedsResponseGroupCounter(t *testing.T) {
	seeded := []Step{
		{
			Thought: "prior turn thought",
			Action:  llm.ToolCall{ID: "old_1", Name: "read_file", Input: json.RawMessage(`{"path":"/tmp/a"}`)},
			// A prior multi-call response produced group 1.
			ResponseGroup: 1,
		},
		{
			Action:        llm.ToolCall{ID: "old_2", Name: "search", Input: json.RawMessage(`{"query":"q"}`)},
			ResponseGroup: 1,
		},
	}
	resumedMulti := llmResponseWithMultipleToolCalls("resumed turn thought", []llm.ToolCall{
		{ID: "new_1", Name: "read_file", Input: json.RawMessage(`{"path":"/tmp/b"}`)},
		{ID: "new_2", Name: "search", Input: json.RawMessage(`{"query":"q2"}`)},
	})
	mockLLM := &mockLLMCaller{
		responses: []*llm.ChatResponse{
			resumedMulti,
			llmResponseFinish("wrapping up", "resumed output"),
		},
	}

	exec := newResumingExecutor(mockLLM, newMockToolExecutor(), &mockTokenCounter{}, 10, &recordingEvents{}, seeded)

	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "read_file", Description: "read", Source: "core"},
		{Name: "search", Description: "search", Source: "core"},
	}, newMockContextManager())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Steps) != len(seeded)+3 { // 2 tool calls + finish
		t.Fatalf("steps = %d, want %d", len(result.Steps), len(seeded)+3)
	}
	newGroup := result.Steps[len(seeded)].ResponseGroup
	if newGroup == 0 {
		t.Fatal("new multi-call response got ResponseGroup 0 — grouping lost entirely")
	}
	if newGroup <= seeded[0].ResponseGroup {
		t.Errorf("new steps reuse ResponseGroup %d already present in the seeded trajectory — adjacent equal ids merge the two turns into one assistant message", newGroup)
	}
	if got := result.Steps[len(seeded)+1].ResponseGroup; got != newGroup {
		t.Errorf("steps of one response carry different groups: %d vs %d", newGroup, got)
	}
}

// TestProcessToolResult_BudgetAppliesDespiteEmbeddedSentinel verifies the
// Stage-2 token budget applies even when the tool's own body embeds the
// Stage-1 nudge sentinel: the executor must never re-discover its own nudge
// by whole-string search (an embedded copy would split the observation there,
// making the budget a no-op and suppressing the hash hint).
func TestProcessToolResult_BudgetAppliesDespiteEmbeddedSentinel(t *testing.T) {
	exec := newExecutorDefaultHITL(&mockLLMCaller{}, newMockToolExecutor(), &mockTokenCounter{}, 10, &recordingEvents{}, false, ToolResultBudget{HardCapTokens: 256}, defaultCircuitBreakerConfig)
	exec.emitter = &NoopEvents{}
	cw := newMockContextManager()

	embedded := "\n\n[This output was truncated to 50 lines for 'fake_tool'. " +
		"The full result is cached with hash: deadbeef. " +
		"Use tool_result_read(hash=\"deadbeef\", start_line=1, num_lines=N) to read fragments.]"
	body := embedded + strings.Repeat("a", 6000)

	got, hash := exec.processToolResult(context.Background(), body, body, "some_tool", json.RawMessage(`{}`), cw)
	if hash != "" {
		t.Fatalf("cache hash = %q, want empty (no tool cache installed)", hash)
	}
	if got == body {
		t.Fatal("Stage-2 budget did not truncate: an embedded nudge sentinel in the tool body suppressed the token budget entirely")
	}
	if !strings.Contains(got, "[OUTPUT TRUNCATED") {
		t.Error("truncated observation lacks the budget truncation notice")
	}
	if strings.Contains(got, "truncated by token budget") {
		t.Error("Stage-2 hash hint present although no nudge was appended by the executor")
	}
}

// TestProcessToolResult_Stage1NudgePreserved verifies the honest path still
// works after the Stage-1/Stage-2 restructure: a real Stage-1 nudge built by
// processToolResult itself is re-appended after the Stage-2 budget pass, and
// the budget (when it fires) applies only to the body.
func TestProcessToolResult_Stage1NudgePreserved(t *testing.T) {
	exec := newExecutorDefaultHITL(&mockLLMCaller{}, newMockToolExecutor(), &mockTokenCounter{}, 10, &recordingEvents{}, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
	exec.emitter = &NoopEvents{}
	exec.perToolTruncation = map[string]ToolTruncationConfig{"some_tool": {MaxLines: 2}}
	exec.toolCache = NewToolResultCache(0)
	cw := newMockContextManager()

	body := "line1\nline2\nline3\nline4\nline5"
	got, hash := exec.processToolResult(context.Background(), body, body, "some_tool", json.RawMessage(`{}`), cw)
	if hash == "" {
		t.Fatal("cache hash is empty, want the stored entry's hash")
	}
	if !strings.HasPrefix(got, "line1\nline2") {
		t.Errorf("Stage-1 line truncation not applied: got prefix %q", got)
	}
	wantNudge := FormatFragmentationNudge(hash, "some_tool", 2)
	if !strings.HasSuffix(got, wantNudge) {
		t.Errorf("observation does not end with the Stage-1 nudge:\ngot:  %q\nwant: %q", got, wantNudge)
	}
}

// TestProcessToolResult_HITLModifiedInputDrivesCacheMeta verifies a
// HITL-rewritten read_file path drives the cache entry's file-backed
// metadata: the entry must reference the file the tool ACTUALLY read, not
// the model's original (redirected-away) path — otherwise tool_result_read
// recovery streams from a file the tool never read.
func TestProcessToolResult_HITLModifiedInputDrivesCacheMeta(t *testing.T) {
	dir := t.TempDir()
	decoy := filepath.Join(dir, "decoy.txt")
	realPath := filepath.Join(dir, "real.txt")
	if err := os.WriteFile(decoy, []byte("decoy contents"), 0o600); err != nil {
		t.Fatalf("write decoy: %v", err)
	}
	if err := os.WriteFile(realPath, []byte("real contents"), 0o600); err != nil {
		t.Fatalf("write real: %v", err)
	}

	mockLLM := &mockLLMCaller{
		responses: []*llm.ChatResponse{
			llmResponseWithToolCall("reading", "read_file", json.RawMessage(`{"path":"`+decoy+`"}`)),
			llmResponseFinish("done", "ok"),
		},
	}
	// read_file deliberately NOT in the results map so GetToolSource returns
	// "core" (a non-core source would classify the entry as MCP and skip the
	// file metadata).
	mockTools := newMockToolExecutor()

	exec := newExecutorDefaultHITL(mockLLM, mockTools, &mockTokenCounter{}, 10, &recordingEvents{}, false, ToolResultBudget{}, defaultCircuitBreakerConfig,
		WithHITL(&scenarioHITL{modify: map[string]json.RawMessage{
			"read_file": json.RawMessage(`{"path":"` + realPath + `"}`),
		}}))
	exec.toolCache = NewToolResultCache(0)

	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "read_file", Description: "read", Source: "core"},
	}, newMockContextManager())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(mockTools.calls) == 0 || mockTools.calls[0].Name != "read_file" {
		t.Fatalf("read_file was never executed (calls = %+v)", mockTools.calls)
	}
	// The tool must have executed with the MODIFIED input.
	if !strings.Contains(string(mockTools.calls[0].Input), realPath) {
		t.Fatalf("tool executed with input %s, want the HITL-modified path %s", mockTools.calls[0].Input, realPath)
	}

	var readStep *Step
	for i := range result.Steps {
		if result.Steps[i].Action.Name == "read_file" {
			readStep = &result.Steps[i]
			break
		}
	}
	if readStep == nil {
		t.Fatal("read_file step not found in the trajectory")
	}
	if readStep.CacheHash == "" {
		t.Fatal("read_file step carries no cache hash — the result was not cached")
	}
	entry, ok := exec.toolCache.Get(readStep.CacheHash)
	if !ok {
		t.Fatalf("cache hash %q does not resolve", readStep.CacheHash)
	}
	if !entry.FileBacked {
		t.Error("entry FileBacked = false, want true (file-backed read_file entry)")
	}
	if entry.FilePath != realPath {
		t.Errorf("entry FilePath = %q, want the HITL-modified path %q — the cache must reference the file the tool actually read", entry.FilePath, realPath)
	}
	if !strings.Contains(entry.Input, realPath) {
		t.Errorf("entry Input = %q, want the effective (post-HITL) arguments", entry.Input)
	}
}

// TestBatchSubCallIndicesDoNotCollideWithSibling verifies the emitted
// (stepNum, callIdx) identity is unique when a batch call shares a response
// with sibling tool calls and the batch is the FIRST call: its sub-calls
// must live in the offset index space (>= batchIndexBase), never on the
// standalone indices a sibling occupies — a host keys per-call UI state by
// that pair and would otherwise overwrite the sibling's card.
func TestBatchSubCallIndicesDoNotCollideWithSibling(t *testing.T) {
	subInput1, _ := json.Marshal(map[string]string{"query": "test"})
	subInput2, _ := json.Marshal(map[string]string{"path": "file.txt"})
	type batchCall struct {
		Tool  string          `json:"tool"`
		Input json.RawMessage `json:"input"`
	}
	batchInput, _ := json.Marshal(map[string][]batchCall{
		"calls": {
			{Tool: "search", Input: subInput1},
			{Tool: "read", Input: subInput2},
		},
	})

	resp := llmResponseWithMultipleToolCalls("thinking", []llm.ToolCall{
		{ID: "call_b", Name: tools.ToolBatch, Input: batchInput},
		{ID: "call_s", Name: "read_file", Input: json.RawMessage(`{"path":"sibling.txt"}`)},
	})

	mockTools := newMockToolExecutor()
	mockTools.results["read"] = tools.ToolResult{Content: "file content"}
	rec := &callIndexRecorder{}
	exec := newExecutorDefaultHITL(&mockLLMCaller{}, mockTools, &mockTokenCounter{}, 10, rec, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
	exec.emitter = rec
	state := &runState{effectiveMaxSteps: 10}
	cw := newMockContextManager()

	if _, _, err := exec.processToolCalls(context.Background(), resp, "thinking", state, cw); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(rec.toolCalls) != 3 {
		t.Fatalf("emitted %d ToolCall events (%v), want 3", len(rec.toolCalls), rec.toolCalls)
	}
	seen := make(map[string]string)
	for _, entry := range rec.toolCalls {
		sep := strings.Index(entry, ":")
		if sep < 0 {
			t.Fatalf("malformed recorded ToolCall entry %q", entry)
		}
		idx, name := entry[:sep], entry[sep+1:]
		if prev, dup := seen[idx]; dup {
			t.Errorf("colliding emitter index %s: %q and %q share it — hosts key per-call state by (stepNum, callIdx)", idx, prev, name)
		}
		seen[idx] = name
	}
	// The batch is callIdx 0, so its sub-calls must sit in the offset space.
	if _, ok := seen["0"]; ok {
		t.Errorf("a batch sub-call emitted on standalone index 0 — with a first-call batch this collides with the standalone space")
	}
	if _, ok := seen["1"]; !ok {
		t.Errorf("sibling read_file (callIdx 1) not emitted: %v", rec.toolCalls)
	}
	if _, ok := seen[strconv.Itoa(batchIndexBase)]; !ok {
		t.Errorf("first batch sub-call not emitted at the offset base %d: %v", batchIndexBase, rec.toolCalls)
	}
}
