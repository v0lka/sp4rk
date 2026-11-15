package agent

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/v0lka/sp4rk/llm"
)

// shortCircuitCacheMaxEntries bounds the executor short-circuit cache. Any
// non-whitelisted execution clears the cache, so steady-state growth is
// driven only by runs that call nothing but whitelisted discovery tools with
// distinct arguments; the cap is a defensive ceiling (the oldest content is
// NOT tracked — on overflow the whole map is dropped, which costs
// re-executions, never correctness).
const shortCircuitCacheMaxEntries = 64

// shortCircuitUnchangedFlagFmt is appended to a replayed observation when an
// identical-input call to a whitelisted discovery tool is served from the
// executor short-circuit cache instead of being re-executed. It names the
// turn (executor step number) whose observation is being replayed so the
// model can locate the original result in its own history.
const shortCircuitUnchangedFlagFmt = "\n\n[unchanged since turn %d]"

// shortCircuitEntry is one replayable observation.
//
// observation holds the FINAL processed observation — post per-tool
// truncation (Stage 1), post token-budget truncation (Stage 2), fragmentation
// nudges included — exactly as the original step delivered it to the model.
// Replaying it verbatim is what preserves the cache-on-truncate invariant: a
// truncated result's recovery hash stays embedded in the replayed text and
// the original ToolResultCache entry is untouched, so tool_result_read still
// resolves the dropped content. The replay therefore MUST NOT pass through
// processToolResult again (that would re-truncate already-truncated text and
// store a junk ToolResultCache entry for it).
type shortCircuitEntry struct {
	observation string // final processed observation of the original call
	cacheHash   string // ToolResultCache hash the original step carried ("" when not truncated and not eagerly cached)
	turn        int    // executor step number the result was produced at
}

// shortCircuitCacheKey builds the cache key for a tool call: the tool name
// and its compacted input JSON. compactJSON normalizes insignificant
// whitespace, so two spellings of the same arguments share one entry —
// the same normalization the repeat-identical circuit breaker uses.
func shortCircuitCacheKey(toolName string, input json.RawMessage) string {
	return toolName + "\x00" + compactJSON(input)
}

// lookupShortCircuitCache returns the cached entry for an identical-input
// call to a whitelisted discovery tool. Only entries stored since the last
// non-whitelisted execution are visible: every non-whitelisted execution
// clears the whole map (shortCircuitNoteExecution), so a hit guarantees no
// state-changing tool ran between the cached result and this call.
func (e *Executor) lookupShortCircuitCache(toolName string, input json.RawMessage) (shortCircuitEntry, bool) {
	if e.shortCircuitCache == nil {
		return shortCircuitEntry{}, false
	}
	entry, ok := e.shortCircuitCache[shortCircuitCacheKey(toolName, input)]
	return entry, ok
}

// shortCircuitNoteExecution records the cache-level consequence of a tool
// call whose execution has completed (or whose abandonment has been converted
// into an error result). Called by both the standalone path and the batch
// sub-call loop immediately after executeToolCall's error handling:
//
//   - a non-whitelisted tool MAY have changed the world, so the entire cache
//     is invalidated — including when the execution failed (the side effects
//     of a failed call must be treated as having possibly occurred);
//   - a whitelisted tool that returned an ERROR result evicts only its own
//     key: the error may be transient, so a previously cached success for the
//     same call must not keep being replayed while the tool is failing;
//   - a whitelisted SUCCESS stores nothing here — the replayable observation
//     (post-truncation, nudges included) only exists after processToolResult,
//     and shortCircuitStoreResult records it there.
func (e *Executor) shortCircuitNoteExecution(toolName string, input json.RawMessage, isError bool) {
	if _, cacheable := shortCircuitCacheableTools[toolName]; !cacheable {
		e.shortCircuitCache = nil
		return
	}
	if isError && e.shortCircuitCache != nil {
		delete(e.shortCircuitCache, shortCircuitCacheKey(toolName, input))
	}
}

// shortCircuitStoreResult stores the final processed observation of a
// successful whitelisted discovery call as replayable. Must be called after
// processToolResult so the stored text is the exact observation the original
// step delivered (truncation notices and cache-hash nudges included).
func (e *Executor) shortCircuitStoreResult(toolName string, input json.RawMessage, turn int, observation, cacheHash string, isError bool) {
	if isError {
		return
	}
	if _, cacheable := shortCircuitCacheableTools[toolName]; !cacheable {
		return
	}
	if len(e.shortCircuitCache) >= shortCircuitCacheMaxEntries {
		e.shortCircuitCache = make(map[string]shortCircuitEntry)
	}
	if e.shortCircuitCache == nil {
		e.shortCircuitCache = make(map[string]shortCircuitEntry)
	}
	e.shortCircuitCache[shortCircuitCacheKey(toolName, input)] = shortCircuitEntry{
		observation: observation,
		cacheHash:   cacheHash,
		turn:        turn,
	}
}

// replayShortCircuitObservation serves an identical-input call to a
// whitelisted discovery tool from the cache: it emits the replayed tool
// result (cached observation + the "[unchanged since turn N]" flag), records
// the step in the trajectory and context window, and reports whether a step
// was materialized.
//
// The replayed step mirrors the executed step in every field the pipeline
// derives from the tool result: CacheHash reuses the original step's hash
// (identical content dedups to the same short hash, so history mutation
// resolves both steps to the same entry) and IsError is always false (only
// successful results are cached). The post-execution result-size circuit
// breakers (fruitless / same-tool repetition) are deliberately NOT advanced
// by a replay: they can only under-count here, never over-count, and the
// loop pattern a replay belongs to — an identical repeat — is already counted
// by checkRepeatIdenticalTool, which runs BEFORE this path on every call.
// processToolResult, the parse-error tracker and verify-on-edit are skipped:
// the stored observation is already fully processed, and none of them can
// apply to a replayed pure-read result.
func (e *Executor) replayShortCircuitObservation(
	ctx context.Context,
	entry shortCircuitEntry,
	action llm.ToolCall,
	callIdx int,
	thought string,
	resp *llm.ChatResponse,
	state *runState,
	cw ContextManager,
) loopAction {
	observation := entry.observation + fmt.Sprintf(shortCircuitUnchangedFlagFmt, entry.turn)
	e.repeat.lastIsError = false
	e.emitter.ExecutorDiagnostic(state.stepNum, "short_circuit_cache_hit", map[string]any{
		"tool":        action.Name,
		"cached_turn": entry.turn,
	})

	isUntrusted := e.tools.IsToolUntrusted(action.Name)

	stepThought := ""
	stepReasoning := ""
	var stepReasoningItems []llm.ReasoningItem
	if callIdx == 0 {
		stepThought = thought
		stepReasoning = resp.Message.ReasoningContent
		stepReasoningItems = resp.Message.ReasoningItems
	}
	step := Step{
		Thought:          stepThought,
		ReasoningContent: stepReasoning,
		ReasoningItems:   stepReasoningItems,
		Action:           action,
		Observation:      observation,
		IsUntrusted:      isUntrusted,
		IsError:          false,
		CacheHash:        entry.cacheHash,
		TokensUsed:       resp.Usage.InputTokens + resp.Usage.OutputTokens,
		ResponseGroup:    state.responseGroup,
	}
	state.allSteps = append(state.allSteps, step)
	cw.AddStep(step)

	// A host-designated stop tool must terminate the run from a replay exactly
	// as it would from a fresh execution (the whitelist never names stop tools
	// today; the call keeps the semantics uniform if that ever changes).
	return e.stopToolTermination(ctx, action.Name, observation, thought, false, resp, state, cw)
}
