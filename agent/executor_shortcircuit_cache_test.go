package agent

import (
	"context"
	"encoding/json"
	"strconv"
	"strings"
	"testing"

	"github.com/v0lka/sp4rk/llm"
	"github.com/v0lka/sp4rk/tools"
)

// --- Short-circuit cache (executor-level replay of whitelisted discovery tools) ---

// TestShortCircuitCache_ReplayIdenticalCallBetweenWhitelistedCalls verifies the
// core AC: a repeated identical list_projects call with only OTHER WHITELISTED
// activity in between (search_facts) is served from the executor short-circuit
// cache — the tool executes ONCE, the replay observation carries the original
// content plus the "[unchanged since turn N]" flag, and the replay step is a
// non-error step in the trajectory.
func TestShortCircuitCache_ReplayIdenticalCallBetweenWhitelistedCalls(t *testing.T) {
	lpInput := json.RawMessage(`{"limit":10}`)
	sfInput := json.RawMessage(`{"query":"alpha"}`)

	mockLLM := &mockLLMCaller{
		responses: []*llm.ChatResponse{
			llmResponseWithToolCall("listing projects", "list_projects", lpInput),
			llmResponseWithToolCall("searching facts", "search_facts", sfInput),
			llmResponseWithToolCall("listing projects again", "list_projects", lpInput),
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

	// The whitelisted repeat must NOT re-execute: exactly two real executions
	// (list_projects once, search_facts once).
	if len(mockTools.calls) != 2 {
		t.Fatalf("tool executions = %d (%+v), want 2 (list_projects once + search_facts once)", len(mockTools.calls), mockTools.calls)
	}
	if mockTools.calls[0].Name != "list_projects" || mockTools.calls[1].Name != "search_facts" {
		t.Fatalf("executed tools = [%s, %s], want [list_projects, search_facts]", mockTools.calls[0].Name, mockTools.calls[1].Name)
	}

	if len(result.Steps) != 4 {
		t.Fatalf("steps = %d, want 4 (list_projects, search_facts, list_projects replay, finish)", len(result.Steps))
	}
	original, replay := result.Steps[0], result.Steps[2]
	if original.Action.Name != "list_projects" || replay.Action.Name != "list_projects" {
		t.Fatalf("steps 0/2 are %q/%q, want list_projects/list_projects", original.Action.Name, replay.Action.Name)
	}

	// The original observation carries no flag.
	if original.Observation != "project alpha\nproject beta" {
		t.Fatalf("original observation = %q, want the raw tool output", original.Observation)
	}
	// The replay observation = cached content + "[unchanged since turn N]"
	// where N is the turn the cached result was produced at (turn 1).
	wantReplay := "project alpha\nproject beta\n\n[unchanged since turn 1]"
	if replay.Observation != wantReplay {
		t.Fatalf("replay observation = %q, want %q", replay.Observation, wantReplay)
	}
	if replay.IsError {
		t.Error("replay step IsError = true, want false (a replay is a successful result)")
	}
}

// TestShortCircuitCache_NonWhitelistedExecutionInvalidates verifies the second
// AC: a store_fact execution between two identical list_projects calls
// invalidates the cache, so the repeat is a FRESH execution (no flag, tool
// executed twice).
func TestShortCircuitCache_NonWhitelistedExecutionInvalidates(t *testing.T) {
	lpInput := json.RawMessage(`{"limit":10}`)
	sfInput := json.RawMessage(`{"fact":"something happened"}`)

	mockLLM := &mockLLMCaller{
		responses: []*llm.ChatResponse{
			llmResponseWithToolCall("listing projects", "list_projects", lpInput),
			llmResponseWithToolCall("storing a fact", "store_fact", sfInput),
			llmResponseWithToolCall("listing projects again", "list_projects", lpInput),
			llmResponseFinish("done", "ok"),
		},
	}
	mockTools := newMockToolExecutor()
	mockTools.results["list_projects"] = tools.ToolResult{Content: "project alpha"}
	mockTools.results["store_fact"] = tools.ToolResult{Content: "stored"}

	exec := newExecutorDefaultHITL(mockLLM, mockTools, &mockTokenCounter{}, 10, &recordingEvents{}, false, ToolResultBudget{}, defaultCircuitBreakerConfig)

	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "list_projects", Description: "list", Source: "core"},
		{Name: "store_fact", Description: "store", Source: "core"},
	}, newMockContextManager())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// list_projects must have executed twice: the store_fact execution is
	// non-whitelisted and invalidates the whole replay cache.
	lpExecutions := 0
	for _, c := range mockTools.calls {
		if c.Name == "list_projects" {
			lpExecutions++
		}
	}
	if lpExecutions != 2 {
		t.Fatalf("list_projects executions = %d, want 2 (cache must be invalidated by store_fact)", lpExecutions)
	}
	if len(result.Steps) != 4 {
		t.Fatalf("steps = %d, want 4", len(result.Steps))
	}
	repeat := result.Steps[2]
	if repeat.Action.Name != "list_projects" {
		t.Fatalf("step 2 is %q, want list_projects", repeat.Action.Name)
	}
	if strings.Contains(repeat.Observation, "[unchanged") {
		t.Fatalf("repeat observation = %q, want a fresh result without the unchanged flag", repeat.Observation)
	}
	if repeat.Observation != "project alpha" {
		t.Fatalf("repeat observation = %q, want the fresh tool output", repeat.Observation)
	}
}

// TestShortCircuitCache_TruncatedResultReplayPreservesRetrievalPath verifies
// the third AC: a whitelisted result that was TRUNCATED (Stage 2 token budget)
// is cached-on-truncate as before, and its replay serves the identical
// already-processed observation verbatim — same CacheHash on both steps, hash
// still resolvable in ToolResultCache, so the model keeps the documented
// tool_result_read recovery path (the cache-on-truncate invariant).
func TestShortCircuitCache_TruncatedResultReplayPreservesRetrievalPath(t *testing.T) {
	sfInput := json.RawMessage(`{"query":"big"}`)
	bigOutput := strings.Repeat("fact line\n", 5000) // 50 KB — far above the budget floor

	mockLLM := &mockLLMCaller{
		responses: []*llm.ChatResponse{
			llmResponseWithToolCall("searching", "search_facts", sfInput),
			llmResponseWithToolCall("searching again", "search_facts", sfInput),
			llmResponseFinish("done", "ok"),
		},
	}
	mockTools := newMockToolExecutor()
	mockTools.results["search_facts"] = tools.ToolResult{Content: bigOutput}

	exec := newExecutorDefaultHITL(mockLLM, mockTools, &mockTokenCounter{}, 10, &recordingEvents{}, false,
		ToolResultBudget{HardCapTokens: 200, MaxFillFraction: 0.4}, defaultCircuitBreakerConfig)
	exec.toolCache = NewToolResultCache(0)

	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "search_facts", Description: "search", Source: "core"},
	}, newMockContextManager())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Executed exactly once despite the identical repeat.
	if len(mockTools.calls) != 1 {
		t.Fatalf("search_facts executions = %d, want 1", len(mockTools.calls))
	}
	if len(result.Steps) != 3 {
		t.Fatalf("steps = %d, want 3 (search_facts, search_facts replay, finish)", len(result.Steps))
	}
	original, replay := result.Steps[0], result.Steps[1]

	// The original was truncated and cached on truncate: hash + recovery nudge.
	if original.CacheHash == "" {
		t.Fatal("original step carries no cache hash — the truncated result was not cached (cache-on-truncate broken)")
	}
	if !strings.Contains(original.Observation, original.CacheHash) {
		t.Fatalf("original observation does not embed the recovery hash %q", original.CacheHash)
	}
	if entry, ok := exec.toolCache.Get(original.CacheHash); !ok {
		t.Fatalf("cache hash %q does not resolve after the replay — retrieval path broken", original.CacheHash)
	} else if entry.Content != bigOutput {
		t.Error("cached entry content is not the FULL pre-truncation result")
	}

	// The replay serves the identical processed observation verbatim + flag:
	// same hash, same nudge, no second truncation pass.
	if replay.CacheHash != original.CacheHash {
		t.Fatalf("replay step CacheHash = %q, want the original %q", replay.CacheHash, original.CacheHash)
	}
	if !strings.HasPrefix(replay.Observation, original.Observation) {
		t.Fatalf("replay observation must repeat the original processed observation verbatim;\nreplay: %q\noriginal: %q", replay.Observation, original.Observation)
	}
	if !strings.Contains(replay.Observation, "[unchanged since turn 1]") {
		t.Fatalf("replay observation = %q, want the unchanged flag", replay.Observation)
	}
	if strings.Count(replay.Observation, "truncated by token budget") != 1 {
		t.Fatalf("replay observation contains %d Stage-2 recovery notices, want exactly 1 (no double processing)",
			strings.Count(replay.Observation, "truncated by token budget"))
	}
}

// TestShortCircuitCache_BatchSubCallInvalidates verifies the batch path's
// bookkeeping: a whitelisted sub-call executed inside batch (the batch path
// never replays) records correctly, and a non-whitelisted sub-call (store_fact)
// inside batch invalidates the cache so a later standalone identical call is a
// fresh execution.
func TestShortCircuitCache_BatchSubCallInvalidates(t *testing.T) {
	lpInput := json.RawMessage(`{"limit":10}`)
	type batchCall struct {
		Tool  string          `json:"tool"`
		Input json.RawMessage `json:"input"`
	}
	batchInput, err := json.Marshal(map[string][]batchCall{
		"calls": {{Tool: "store_fact", Input: json.RawMessage(`{"fact":"changed"}`)}},
	})
	if err != nil {
		t.Fatalf("marshal batch input: %v", err)
	}

	mockLLM := &mockLLMCaller{
		responses: []*llm.ChatResponse{
			llmResponseWithToolCall("listing projects", "list_projects", lpInput),
			llmResponseWithMultipleToolCalls("batching", []llm.ToolCall{
				{ID: "call_b", Name: tools.ToolBatch, Input: batchInput},
			}),
			llmResponseWithToolCall("listing projects again", "list_projects", lpInput),
			llmResponseFinish("done", "ok"),
		},
	}
	mockTools := newMockToolExecutor()
	mockTools.results["list_projects"] = tools.ToolResult{Content: "project alpha"}
	mockTools.results["store_fact"] = tools.ToolResult{Content: "stored"}

	exec := newExecutorDefaultHITL(mockLLM, mockTools, &mockTokenCounter{}, 10, &recordingEvents{}, false, ToolResultBudget{}, defaultCircuitBreakerConfig)

	_, err = exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "list_projects", Description: "list", Source: "core"},
		{Name: "store_fact", Description: "store", Source: "core"},
	}, newMockContextManager())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	lpExecutions := 0
	for _, c := range mockTools.calls {
		if c.Name == "list_projects" {
			lpExecutions++
		}
	}
	if lpExecutions != 2 {
		t.Fatalf("list_projects executions = %d, want 2 (batched store_fact must invalidate the cache)", lpExecutions)
	}
}

// TestShortCircuitCache_RepeatBreakerStillFires verifies the cache does not
// weaken loop protection: the 3rd consecutive identical call is intercepted by
// the repeat-identical circuit breaker (nudge step, IsError) exactly as
// without the cache, and the replayed 2nd call still executed the tool only
// once overall.
func TestShortCircuitCache_RepeatBreakerStillFires(t *testing.T) {
	lpInput := json.RawMessage(`{"limit":10}`)

	mockLLM := &mockLLMCaller{
		responses: []*llm.ChatResponse{
			llmResponseWithToolCall("listing", "list_projects", lpInput),
			llmResponseWithToolCall("listing again", "list_projects", lpInput),
			llmResponseWithToolCall("listing yet again", "list_projects", lpInput),
			llmResponseFinish("done", "ok"),
		},
	}
	mockTools := newMockToolExecutor()
	mockTools.results["list_projects"] = tools.ToolResult{Content: "project alpha"}

	exec := newExecutorDefaultHITL(mockLLM, mockTools, &mockTokenCounter{}, 10, &recordingEvents{}, false, ToolResultBudget{}, defaultCircuitBreakerConfig)

	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "list_projects", Description: "list", Source: "core"},
	}, newMockContextManager())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(mockTools.calls) != 1 {
		t.Fatalf("list_projects executions = %d, want 1 (2nd call replays, 3rd is intercepted)", len(mockTools.calls))
	}

	var sawReplay, sawNudge bool
	for _, s := range result.Steps {
		if strings.Contains(s.Observation, "[unchanged since turn 1]") {
			sawReplay = true
			if s.IsError {
				t.Error("replay step IsError = true, want false")
			}
		}
		if strings.Contains(s.Observation, "called the same tool with identical arguments") {
			sawNudge = true
			if !s.IsError {
				t.Error("breaker nudge step IsError = false, want true (the tool call was NOT executed)")
			}
		}
	}
	if !sawReplay {
		t.Error("no replay step with the unchanged flag found")
	}
	if !sawNudge {
		t.Error("no repeat-identical breaker nudge step found — loop protection weakened")
	}
}

// TestShortCircuitCacheHelpers exercises the cache helpers directly: key
// normalization via compactJSON, error eviction of the whitelisted key, whole
// map invalidation on non-whitelisted execution, and the overflow cap.
func TestShortCircuitCacheHelpers(t *testing.T) {
	exec := &Executor{}

	// Non-whitelisted tool results are never stored.
	exec.shortCircuitStoreResult("write_file", json.RawMessage(`{}`), 1, "obs", "hash", false)
	if exec.shortCircuitCache != nil {
		t.Fatalf("shortCircuitCache = %v, want nil (write_file is not whitelisted)", exec.shortCircuitCache)
	}

	// Whitelisted successes store; compactJSON makes whitespace-variant input
	// spellings resolve to the same entry.
	exec.shortCircuitStoreResult("search_facts", json.RawMessage(`{"query":"x"}`), 3, "cached obs", "h123", false)
	if entry, ok := exec.lookupShortCircuitCache("search_facts", json.RawMessage(`{ "query" : "x" }`)); !ok {
		t.Fatal("whitespace-variant identical input did not hit the cache")
	} else if entry.turn != 3 || entry.observation != "cached obs" || entry.cacheHash != "h123" {
		t.Fatalf("entry = %+v, want the stored values", entry)
	}

	// A whitelisted ERROR evicts only its own key.
	exec.shortCircuitNoteExecution("search_facts", json.RawMessage(`{"query":"x"}`), true)
	if _, ok := exec.lookupShortCircuitCache("search_facts", json.RawMessage(`{"query":"x"}`)); ok {
		t.Fatal("errored whitelisted key still cached — a stale success would mask the failure")
	}

	// Non-whitelisted execution invalidates everything.
	exec.shortCircuitStoreResult("search_facts", json.RawMessage(`{"query":"x"}`), 4, "obs2", "", false)
	exec.shortCircuitStoreResult("list_projects", json.RawMessage(`{}`), 4, "obs3", "", false)
	exec.shortCircuitNoteExecution("store_fact", json.RawMessage(`{}`), false)
	if exec.shortCircuitCache != nil {
		t.Fatalf("cache = %v, want nil after a non-whitelisted execution", exec.shortCircuitCache)
	}

	// Overflow resets the map instead of growing without bound.
	for i := 0; i < shortCircuitCacheMaxEntries+5; i++ {
		exec.shortCircuitStoreResult("search_facts", json.RawMessage(`{"i":`+strconv.Itoa(i)+`}`), i, "obs", "", false)
	}
	if len(exec.shortCircuitCache) > shortCircuitCacheMaxEntries {
		t.Fatalf("cache length = %d, want <= %d", len(exec.shortCircuitCache), shortCircuitCacheMaxEntries)
	}
	if _, ok := exec.lookupShortCircuitCache("search_facts", json.RawMessage(`{"i":0}`)); ok {
		t.Error("oldest pre-overflow entry survived the cap reset")
	}
	if _, ok := exec.lookupShortCircuitCache("search_facts", json.RawMessage(`{"i":`+strconv.Itoa(shortCircuitCacheMaxEntries+4)+`}`)); !ok {
		t.Error("newest post-overflow entry missing")
	}
}
