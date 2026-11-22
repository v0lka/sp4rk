package agent

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/v0lka/sp4rk/llm"
	"github.com/v0lka/sp4rk/tools"
)

// --- ThinkingBlocks carriage gating (first materialized step only) ---
//
// The signed thinking blocks of one LLM response must ride exactly ONE
// materialized step per response — the response's first. Grouped rendering
// reads the blocks from groupSteps[0] only, but a lone batch (ResponseGroup 0)
// renders every sub-call step standalone, so an ungated copy on a later step
// would put the same signed pair on the wire twice; persisted steps would
// carry N copies per multi-call group (checkpoint/trajectory and any
// host-rebuild history re-emit them as duplicate assistant turns).

// sigTestBlocks is the signed thinking-block pair test responses carry.
func sigTestBlocks() []llm.ThinkingBlock {
	return []llm.ThinkingBlock{
		{Type: "thinking", Thinking: "reasoned about it", Signature: "SIG_TEST"},
	}
}

// responseWithBlocks returns a multi-call response carrying thinking blocks.
func responseWithBlocks(thought string, calls []llm.ToolCall) *llm.ChatResponse {
	resp := llmResponseWithMultipleToolCalls(thought, calls)
	resp.Message.ThinkingBlocks = sigTestBlocks()
	return resp
}

// countSerializedBlockCarriers marshals the steps and counts how many carry
// thinking_blocks — the persistence-level duplicate check.
func countSerializedBlockCarriers(t *testing.T, steps []Step) int {
	t.Helper()
	blob, err := json.Marshal(steps)
	if err != nil {
		t.Fatalf("marshal steps: %v", err)
	}
	return strings.Count(string(blob), `"thinking_blocks"`)
}

// denyNamedToolHITL allows every call except one tool name, which it rejects
// (simulating a HITL denial on a specific sibling).
type denyNamedToolHITL struct{ deny string }

func (h *denyNamedToolHITL) OnToolCall(_ context.Context, name string, _ json.RawMessage) (*HITLToolDecision, error) {
	if name == h.deny {
		return &HITLToolDecision{Allow: false, Reason: "denied by test"}, nil
	}
	return nil, nil
}

func (h *denyNamedToolHITL) OnStepLimit(_ context.Context, _, _ int, _ string) (StepLimitResponse, error) {
	return StepLimitDeny, nil
}

var _ HITLHandler = (*denyNamedToolHITL)(nil)

// TestThinkingBlocks_ExecutedPathGatedToFirstStep pins the executed-path gate:
// a two-call response materializes steps where only the FIRST carries the
// signed blocks; the serialized trajectory must not repeat them.
func TestThinkingBlocks_ExecutedPathGatedToFirstStep(t *testing.T) {
	mockLLM := &mockLLMCaller{
		responses: []*llm.ChatResponse{
			responseWithBlocks("reading two files", []llm.ToolCall{
				{ID: "call_a", Name: "read_file", Input: json.RawMessage(`{"path":"a.txt"}`)},
				{ID: "call_b", Name: "glob", Input: json.RawMessage(`{"pattern":"*.txt"}`)},
			}),
			llmResponseFinish("done", "ok"),
		},
	}
	mockTools := newMockToolExecutor()
	mockTools.results["read_file"] = tools.ToolResult{Content: "a"}
	mockTools.results["glob"] = tools.ToolResult{Content: "a.txt"}

	exec := newExecutorDefaultHITL(mockLLM, mockTools, &mockTokenCounter{}, 10, &recordingEvents{}, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "read_file", Description: "read", Source: "core"},
		{Name: "glob", Description: "glob", Source: "core"},
	}, newMockContextManager())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(result.Steps) < 3 {
		t.Fatalf("steps = %d, want >= 3 (two tool calls + finish)", len(result.Steps))
	}
	if len(result.Steps[0].ThinkingBlocks) == 0 || result.Steps[0].ThinkingBlocks[0].Signature != "SIG_TEST" {
		t.Errorf("first step must carry the response's signed blocks, got %+v", result.Steps[0].ThinkingBlocks)
	}
	if len(result.Steps[1].ThinkingBlocks) != 0 {
		t.Errorf("second step (callIdx 1) must not carry blocks, got %+v", result.Steps[1].ThinkingBlocks)
	}
	if n := countSerializedBlockCarriers(t, result.Steps); n != 1 {
		t.Errorf("serialized trajectory carries thinking_blocks %d times, want exactly 1 per response group", n)
	}
}

// TestThinkingBlocks_RejectAtCallIdxOneGated pins the HITL-reject gate: when
// the SECOND sibling is rejected, the reject step must not duplicate the
// first step's signed blocks.
func TestThinkingBlocks_RejectAtCallIdxOneGated(t *testing.T) {
	mockLLM := &mockLLMCaller{
		responses: []*llm.ChatResponse{
			responseWithBlocks("read then write", []llm.ToolCall{
				{ID: "call_a", Name: "read_file", Input: json.RawMessage(`{"path":"a.txt"}`)},
				{ID: "call_b", Name: "bash_exec", Input: json.RawMessage(`{"command":"rm -rf /"}`)},
			}),
			llmResponseFinish("done", "ok"),
		},
	}
	mockTools := newMockToolExecutor()
	mockTools.results["read_file"] = tools.ToolResult{Content: "a"}

	exec := newExecutorDefaultHITL(mockLLM, mockTools, &mockTokenCounter{}, 10, &recordingEvents{}, false, ToolResultBudget{}, defaultCircuitBreakerConfig, WithHITL(&denyNamedToolHITL{deny: "bash_exec"}))
	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "read_file", Description: "read", Source: "core"},
		{Name: "bash_exec", Description: "shell", Source: "core"},
	}, newMockContextManager())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(result.Steps) < 3 {
		t.Fatalf("steps = %d, want >= 3 (executed call + rejected call + finish)", len(result.Steps))
	}
	reject := result.Steps[1]
	if !strings.Contains(reject.Observation, "[Tool call rejected") {
		t.Fatalf("steps[1] is not the reject step (observation %q)", reject.Observation)
	}
	if len(result.Steps[0].ThinkingBlocks) == 0 {
		t.Errorf("first step must carry the response's signed blocks, got none")
	}
	if len(reject.ThinkingBlocks) != 0 {
		t.Errorf("rejected step (callIdx 1) must not carry blocks, got %+v", reject.ThinkingBlocks)
	}
	if n := countSerializedBlockCarriers(t, result.Steps); n != 1 {
		t.Errorf("serialized trajectory carries thinking_blocks %d times, want exactly 1", n)
	}
}

// TestThinkingBlocks_LoneBatchGatedToFirstSubCall pins the batch gate for the
// wire-visible case: a lone batch (single batch call in the response →
// ResponseGroup 0) renders every sub-call step STANDALONE, so an ungated
// second sub-call would emit the same signed pair on the wire twice. Only the
// first sub-call step may carry the blocks.
func TestThinkingBlocks_LoneBatchGatedToFirstSubCall(t *testing.T) {
	type batchCall struct {
		Tool  string          `json:"tool"`
		Input json.RawMessage `json:"input"`
	}
	batchInput, err := json.Marshal(map[string][]batchCall{
		"calls": {
			{Tool: "read_file", Input: json.RawMessage(`{"path":"a.txt"}`)},
			{Tool: "read_file", Input: json.RawMessage(`{"path":"b.txt"}`)},
		},
	})
	if err != nil {
		t.Fatalf("marshal batch input: %v", err)
	}

	mockLLM := &mockLLMCaller{
		responses: []*llm.ChatResponse{
			responseWithBlocks("batching reads", []llm.ToolCall{
				{ID: "call_batch", Name: tools.ToolBatch, Input: batchInput},
			}),
			llmResponseFinish("done", "ok"),
		},
	}
	mockTools := newMockToolExecutor()
	mockTools.results["read_file"] = tools.ToolResult{Content: "content"}

	exec := newExecutorDefaultHITL(mockLLM, mockTools, &mockTokenCounter{}, 10, &recordingEvents{}, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "read_file", Description: "read", Source: "core"},
	}, newMockContextManager())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(result.Steps) < 3 {
		t.Fatalf("steps = %d, want >= 3 (two batch sub-calls + finish)", len(result.Steps))
	}
	if result.Steps[0].ResponseGroup != 0 {
		t.Fatalf("lone batch must materialize as ResponseGroup 0 (standalone), got %d", result.Steps[0].ResponseGroup)
	}
	if len(result.Steps[0].ThinkingBlocks) == 0 {
		t.Errorf("first sub-call step must carry the response's signed blocks, got none")
	}
	if len(result.Steps[1].ThinkingBlocks) != 0 {
		t.Errorf("second sub-call step (subIdx 1) must not carry blocks — a lone batch renders standalone and would duplicate the signed pair on the wire, got %+v", result.Steps[1].ThinkingBlocks)
	}
	if n := countSerializedBlockCarriers(t, result.Steps); n != 1 {
		t.Errorf("serialized trajectory carries thinking_blocks %d times, want exactly 1", n)
	}
}

// TestThinkingBlocks_RepeatInterceptAtCallIdxOneGated pins the repeat-breaker
// gate: when the SECOND identical sibling is intercepted by the repeat-identical
// circuit breaker, the nudge step must not duplicate the first step's blocks.
func TestThinkingBlocks_RepeatInterceptAtCallIdxOneGated(t *testing.T) {
	cb := defaultCircuitBreakerConfig
	cb.RepeatNudgeThreshold = 2 // intercept the second identical call

	lpInput := json.RawMessage(`{"limit":10}`)
	mockLLM := &mockLLMCaller{
		responses: []*llm.ChatResponse{
			responseWithBlocks("listing twice", []llm.ToolCall{
				{ID: "call_a", Name: "list_projects", Input: lpInput},
				{ID: "call_b", Name: "list_projects", Input: lpInput},
			}),
			llmResponseFinish("done", "ok"),
		},
	}
	mockTools := newMockToolExecutor()
	mockTools.results["list_projects"] = tools.ToolResult{Content: "project alpha"}

	exec := newExecutorDefaultHITL(mockLLM, mockTools, &mockTokenCounter{}, 10, &recordingEvents{}, false, ToolResultBudget{}, cb)
	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "list_projects", Description: "list", Source: "core"},
	}, newMockContextManager())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(result.Steps) < 3 {
		t.Fatalf("steps = %d, want >= 3 (executed call + intercepted call + finish)", len(result.Steps))
	}
	intercept := result.Steps[1]
	if !intercept.IsError {
		t.Fatalf("steps[1] is not the repeat-intercept nudge step (IsError=false)")
	}
	if len(result.Steps[0].ThinkingBlocks) == 0 {
		t.Errorf("first step must carry the response's signed blocks, got none")
	}
	if len(intercept.ThinkingBlocks) != 0 {
		t.Errorf("intercepted step (callIdx 1) must not carry blocks, got %+v", intercept.ThinkingBlocks)
	}
	if n := countSerializedBlockCarriers(t, result.Steps); n != 1 {
		t.Errorf("serialized trajectory carries thinking_blocks %d times, want exactly 1", n)
	}
}

// TestThinkingBlocks_ShortCircuitReplayStepCarriesNoDuplicates pins the
// short-circuit replay path: the replayed sibling (callIdx >= 1) mirrors the
// executed step without duplicating the signed blocks — the group's blocks
// stay on groupSteps[0] so the grouped assistant turn carries them exactly
// once.
func TestThinkingBlocks_ShortCircuitReplayStepCarriesNoDuplicates(t *testing.T) {
	lpInput := json.RawMessage(`{"limit":10}`)
	sfInput := json.RawMessage(`{"query":"alpha"}`)
	mockLLM := &mockLLMCaller{
		responses: []*llm.ChatResponse{
			responseWithBlocks("listing and searching", []llm.ToolCall{
				{ID: "call_a", Name: "list_projects", Input: lpInput},
				{ID: "call_b", Name: "search_facts", Input: sfInput},
				{ID: "call_c", Name: "list_projects", Input: lpInput}, // replay
			}),
			llmResponseFinish("done", "ok"),
		},
	}
	mockTools := newMockToolExecutor()
	mockTools.results["list_projects"] = tools.ToolResult{Content: "project alpha\nproject beta"}
	mockTools.results["search_facts"] = tools.ToolResult{Content: "fact: alpha exists"}

	exec := newExecutorDefaultHITL(mockLLM, mockTools, &mockTokenCounter{}, 10, &recordingEvents{}, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "list_projects", Description: "list", Source: "core"},
		{Name: "search_facts", Description: "search", Source: "core"},
	}, newMockContextManager())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(result.Steps) != 4 {
		t.Fatalf("steps = %d, want 4 (list_projects, search_facts, replay, finish)", len(result.Steps))
	}
	if !strings.Contains(result.Steps[2].Observation, "[unchanged since turn") {
		t.Fatalf("steps[2] is not the replay step (observation %q)", result.Steps[2].Observation)
	}
	if len(result.Steps[0].ThinkingBlocks) == 0 {
		t.Errorf("the executed first step (groupSteps[0]) must carry the response's signed blocks, got none")
	}
	for i := 1; i < len(result.Steps); i++ {
		if len(result.Steps[i].ThinkingBlocks) != 0 {
			t.Errorf("step %d must not carry blocks (only groupSteps[0] does), got %+v", i, result.Steps[i].ThinkingBlocks)
		}
	}
	if n := countSerializedBlockCarriers(t, result.Steps); n != 1 {
		t.Errorf("serialized trajectory carries thinking_blocks %d times, want exactly 1", n)
	}
}

// TestReplayShortCircuitObservation_CallIdxZeroCarriesBlocks unit-pins the
// replay arm's callIdx == 0 branch: if a replay ever materializes the
// response's first step, it must carry the reasoning fields AND the signed
// thinking blocks as one bundle — a replay turn whose reasoning is carried
// without its blocks breaks the round-trip contract. The callIdx >= 1 arm
// stays block-free (the executed first step of the group carries them).
func TestReplayShortCircuitObservation_CallIdxZeroCarriesBlocks(t *testing.T) {
	exec := newExecutorDefaultHITL(&mockLLMCaller{}, newMockToolExecutor(), &mockTokenCounter{}, 10, &recordingEvents{}, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
	resp := responseWithBlocks("thinking", []llm.ToolCall{{ID: "call_a", Name: "list_projects", Input: json.RawMessage(`{"limit":10}`)}})
	resp.Message.ReasoningContent = "encrypted reasoning chain"
	entry := shortCircuitEntry{observation: "cached output", cacheHash: "hash1", turn: 1}
	action := llm.ToolCall{ID: "call_a", Name: "list_projects", Input: json.RawMessage(`{"limit":10}`)}

	for _, tc := range []struct {
		name        string
		callIdx     int
		wantBlocks  bool
		wantThought string
	}{
		{name: "callIdx 0 carries the full bundle", callIdx: 0, wantBlocks: true, wantThought: "thinking"},
		{name: "callIdx 1 stays block-free", callIdx: 1, wantBlocks: false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			state := &runState{}
			cw := newMockContextManager()
			act := exec.replayShortCircuitObservation(context.Background(), entry, action, tc.callIdx, "thinking", resp, state, cw)
			if act != actionNone {
				t.Fatalf("loopAction = %v, want actionNone", act)
			}
			if len(state.allSteps) != 1 {
				t.Fatalf("steps recorded = %d, want 1", len(state.allSteps))
			}
			step := state.allSteps[0]
			if tc.wantBlocks {
				if len(step.ThinkingBlocks) == 0 || step.ThinkingBlocks[0].Signature != "SIG_TEST" {
					t.Errorf("replay arm callIdx=0 must carry the signed blocks, got %+v", step.ThinkingBlocks)
				}
				if step.Thought != tc.wantThought {
					t.Errorf("replay arm callIdx=0 must carry the thought %q, got %q", tc.wantThought, step.Thought)
				}
				if step.ReasoningContent == "" {
					t.Errorf("replay arm callIdx=0 must carry the reasoning content, got empty")
				}
			} else {
				if len(step.ThinkingBlocks) != 0 {
					t.Errorf("replay arm callIdx=1 must not carry blocks, got %+v", step.ThinkingBlocks)
				}
				if step.Thought != "" {
					t.Errorf("replay arm callIdx=1 must not carry the thought, got %q", step.Thought)
				}
			}
		})
	}
}
