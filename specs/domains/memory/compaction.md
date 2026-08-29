# Compaction Strategies

## Role

Compress conversation history when the context window approaches capacity, preserving the most relevant information for continued execution. Also covers tool-output pruning and regular history mutation, which run independently of fill-triggered compaction.

## Key Files

- `github.com/v0lka/sp4rk/memory` — `SlidingWindowStrategy`, `SummarizationStrategy`, `HierarchicalStrategy`, `NewCompactionStrategy`, `CompactionConfig`, `CompactionDeps`
- `github.com/v0lka/sp4rk/memory` — `CompactConversationHistory` (manual, message-level compaction)
- `github.com/v0lka/sp4rk/memory` — `ContextWindow` (orchestrates pruning + strategy compaction), `CompactionThresholds`
- `github.com/v0lka/sp4rk/memory` — `ToolOutputPruning`, `HistoryMutation`
- `github.com/v0lka/sp4rk/agent` — `CompactionStrategy` interface, `Step`, `ToolResultCache`
- `github.com/v0lka/sp4rk/llm` — `TokenCounter`, `Message`

## Behavior

Three strategies implement the `agent.CompactionStrategy` interface. Each transforms a slice of steps into a smaller slice of messages.

### Sliding window

Keeps the first `KeepFirst` and last `KeepLast` steps, dropping the middle. A system message marks the gap. No LLM calls are required — the simplest and fastest strategy. If the total step count does not exceed `KeepFirst + KeepLast`, all steps are kept verbatim. Best for code tasks (recent edits must stay visible).

```
[sys] [step1] … [stepK] … [stepN-2] [stepN-1] [stepN]
      ├── keep_first ──┤ discard ├── keep_last ────────┤
```

### Summarization

Groups the oldest steps into blocks and uses an LLM (`Summarize` callback) to summarize each block into a compact system message. The most recent `KeepLast` steps are preserved verbatim. Each block's text is truncated to `MaxSummarizeTokens` before being sent; if summarization fails, a fallback indicator message is emitted. Best for research tasks (preserves synthesized findings).

### Hierarchical

Divides steps into three zones with different compression levels:

| Zone | Default ratio | Treatment |
| ---- | ------------- | --------- |
| Distant (oldest) | `0.4` | Aggressive summarization — large blocks (~15 steps), observations truncated to 60% of the base value. |
| Middle | `0.3` | Moderate summarization — smaller blocks (~5 steps). |
| Recent | `0.3` | Kept verbatim. |

Ratios are normalized to sum to `1.0`. For very small step counts (≤ 5), all steps are returned verbatim. Best for long-running complex tasks needing balanced retention.

### NewCompactionStrategy (factory)

```go
func NewCompactionStrategy(name string, cfg CompactionConfig, deps CompactionDeps) agent.CompactionStrategy
```

Recognized names: `"sliding_window"`, `"summarization"`, `"hierarchical"`; any other name falls back to `SlidingWindowStrategy`. `CompactionConfig` holds fields for all three strategies (only the relevant ones are used); `CompactionDeps` carries the `TokenCounter`, the `Summarize` callback, and `MaxSummarizeTokens` (default `16000`) required by the LLM-based strategies. Zero fields are defaulted: `sliding_window` applies `KeepFirst`→`3` and `KeepLast`→`10` (the unrecognized-name fallback strategy inherits these same defaults, so a zero-valued config always preserves a head and a tail of the step history rather than erasing it); `BlockSize`→`10`, `KeepLast`→`5`, `ObservationTruncate`→`500` for the LLM-based strategies; hierarchical ratios→`0.4/0.3/0.3`.

### CompactConversationHistory (manual, message-level)

```go
func CompactConversationHistory(ctx context.Context, msgs []llm.Message, budgetTokens int, strategy string, cfg CompactionConfig, deps CompactionDeps) ([]llm.Message, error)
```

Message-level counterpart to the step strategies for callers that hold raw conversation histories (user/assistant exchanges) rather than a `ContextWindow` — e.g. a session orchestrator compacting cross-task dialogue context on demand. Strategies reuse the same names, config fields, defaults, and zone semantics as `NewCompactionStrategy`:

- `sliding_window` — keep first `KeepFirst` + last `KeepLast` messages verbatim, one system note marks the omitted middle; no LLM call; short histories (`len <= KeepFirst+KeepLast`) are returned unchanged in count-based mode — in token-budget mode the no-op decision is made on tokens alone, so an over-budget short history is still windowed (and then trimmed to the ceiling).
- `summarization` — keep last `KeepLast` messages verbatim; older messages grouped into `BlockSize` blocks, each summarized via `deps.Summarize` into a system message.
- `hierarchical` — distant zone (one aggressive summary block over the whole zone), middle zone (per-`BlockSize` summaries), recent zone verbatim.

Differences from the step-based path: unknown strategy names fail closed with an error (no silent fallback); LLM strategies require a non-nil `deps.Summarize` and propagate its error (no fallback indicator); the input slice is never mutated; user messages with content blocks are flattened (`userMessageText`) before summarization. `budgetTokens > 0` is honored as a hard ceiling — the result is trimmed oldest-first until it fits (a nil `TokenCounter` skips trimming) — and the very last message is always preserved.

In token-budget mode (`budgetTokens > 0` and a non-nil `deps.TokenCounter`), each strategy sizes its verbatim zones as a **share of the budget** so the result keeps as much recent context as the budget allows: `sliding_window` keeps a verbatim tail of at most ~80% of the budget (capped at `KeepLast` messages) plus a head of at most ~15% (capped at `KeepFirst`); `summarization` keeps a verbatim tail of at most ~70% (capped at `KeepLast`); `hierarchical` keeps a verbatim recent zone of at most ~50% (capped at the ratio-derived zone; zone overflow folds into the summarized middle zone instead of being dropped). Each share cap yields to the never-remove-last invariant: the final message is kept verbatim even when it alone exceeds its zone's share (up to the full budget; a last message beyond the budget is the documented exception). The shares deliberately leave headroom below 1.0 for the summary/omission-note messages the strategies add, before the oldest-first trim enforces the hard ceiling.

### Tool-output pruning (separate mechanism)

`ToolOutputPruning` replaces individual tool-result bodies with a placeholder while keeping the surrounding assistant/tool message structure intact:

- The last `KeepLastN` tool-result steps are always protected.
- Any step whose tool name is in `ProtectedTools` is protected regardless of position.
- When context fill is below `ThresholdPercent`, **all** tool outputs are preserved (pruning skipped).
- Entire response groups are protected if any step in the group is protected.

This begins at the `Pruning.ThresholdPercent` floor (default `50`) and runs continuously above it on every `BuildPrompt`, independent of the `CheckFill` status; it is separate from strategy compaction (below), which is `CheckFill`-triggered.

### Regular history mutation (separate mechanism)

`HistoryMutation` runs on **every** `BuildPrompt` (not fill-triggered) to reduce O(n²) replay cost. Information is preserved — evicted content is recoverable via `agent.ToolResultCache`:

- **Cache reference eviction**: a tool result older than `ToolResultEvictionStep` steps with a `CacheHash` is replaced with a cache-reference nudge (`tool_result_read(hash=…, …)`).
- **Step-status eviction**: `update_checklist` (bookkeeping) results are evicted immediately when `EvictStepStatus` is set.
- **Dedup repeated reads**: if the same file (same path + mtime → same cache hash) was read earlier, the later result is replaced with a cache reference.

Mutation runs **first** in message construction, before pruning and injection defense; pruning is skipped for outputs already replaced by a cache reference (overwriting the placeholder would destroy the cache hash and break recoverability).

### Trigger thresholds

```
Tool-output pruning:    continuous floor at Pruning.ThresholdPercent (default 50%).
                        Runs on every BuildPrompt once fill >= floor, independent
                        of the CheckFill status.

Strategy compaction:    CheckFill-driven; begins at "compact" (PredictivePercent,
                        default 85%). The same cw.Compact(ctx) runs for every
                        triggering status:
                            85% (compact)  92% (warning)  98% (emergency)  ->  identical action

Regular history mutation: runs on EVERY BuildPrompt (not fill-triggered)
```

### Per-step pruning overrides

A step may override pruning config via `PruningOverride{KeepLastN, ProtectedTools}` (zero values mean "use the global default"), typically derived from a step profile's role.

## Error Handling

- **Summarization failure**: a fallback indicator message is emitted instead of aborting.
- **Empty history**: compaction is a no-op (returns an empty slice, no error).
- **`budgetTokens` currently unused**: the parameter is part of the `CompactionStrategy.Compact` signature but is not honored by any shipped strategy (sliding-window, summarization, and hierarchical all use only `ctx` and `steps`).
- **Unrecognized domain→strategy**: `sliding_window` is the universal fallback.

## Invariants

- Compaction never removes the system prompt or the last message (current LLM turn).
- Compaction preserves the untrusted-content boundary: after compaction, verbatim tool messages retained in the frozen prefix whose source step was flagged `IsUntrusted` are re-wrapped via `security.WrapUntrustedContent` (`memory.ContextWindow.wrapUntrustedInFrozenPrefix`). The summarization strategy likewise wraps untrusted observations. Untrusted content never re-enters LLM context raw after compaction.
- Tool-output pruning runs before strategy compaction.
- Protected tools are never pruned regardless of `KeepLastN`.
- Mutation runs before pruning; cache-reference placeholders are not overwritten.
- After compaction, fill percentage is below the warning threshold.

## Related Specs

- [README.md](README.md) — context management overview
- [../orchestration/executor.md](../orchestration/executor.md) — per-iteration fill check
- [../orchestration/conductor.md](../orchestration/conductor.md) — compaction strategy selection per run
- [../tool-system/README.md](../tool-system/README.md) — `ToolResultCache` and cache-reference recovery
