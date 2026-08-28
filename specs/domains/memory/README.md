# Memory

## Purpose

Manages the agent's working memory during execution: a managed representation of the LLM context window, three pluggable compaction strategies, selective tool-output pruning, and regular history mutation that reduces the O(n²) replay cost of long conversations. Each [Conductor](../orchestration/conductor.md) run and each [subagent](../orchestration/subagents.md) gets its own isolated `ContextManager`.

## Key Files

- `github.com/v0lka/sp4rk/memory` — `ContextWindow`, `ContextWindowConfig`, `NewContextWindow`, `BuildPrompt`/`AddStep`/`SeedSteps`/`SetPendingUserInterjection`/`ConsumePendingUserInterjection`/`Compact`/`CheckFill`
- `github.com/v0lka/sp4rk/memory` (compaction) — `CompactionStrategy` implementations (`SlidingWindowStrategy`, `SummarizationStrategy`, `HierarchicalStrategy`), `NewCompactionStrategy` factory, `CompactionConfig`, `CompactionDeps`
- `github.com/v0lka/sp4rk/memory` — `CompactionThresholds`, `ToolOutputPruning`, `HistoryMutation`
- `github.com/v0lka/sp4rk/security` — untrusted-content wrapping for tool output
- `github.com/v0lka/sp4rk/llm` — `TokenCounter`, `ContextTokenTracker`, `ModelMetadata`, `Message`, `ContentBlock`
- `github.com/v0lka/sp4rk/prompt` — `CacheBreakMarker` / `SplitCacheBreak`

## Core Types

```go
// Managed representation of the LLM context window.
type ContextWindow struct { /* unexported */ }

type ContextWindowConfig struct {
    SystemPrompt            string             // may contain a CacheBreakMarker
    ModelMeta               llm.ModelMetadata
    Tracker                 *llm.ContextTokenTracker // nil ⇒ discard tracker (accounting disabled)
    Thresholds              CompactionThresholds
    Strategy                agent.CompactionStrategy  // nil ⇒ compaction disabled
    SafetyMarginPercent     int                       // default 5
    InjectionDefenseEnabled bool
    Pruning                 ToolOutputPruning
}

type CompactionStrategy interface {
    Compact(ctx context.Context, steps []agent.Step, budgetTokens int) []llm.Message
}

type CompactionThresholds struct {
    PredictivePercent int // triggers "compact"
    WarningPercent    int // triggers "warning"
    EmergencyPercent  int // triggers "emergency"
}
```

## Flow

```
Executor.Run iteration
  │
  ├─ BuildPrompt()        → system message(s) → prior conversation → task → plan → step history
  │      └─ history mutation + tool-output pruning run here on EVERY call:
  │           pruning active whenever fill >= Pruning.ThresholdPercent (default 50%),
  │           independent of the CheckFill status below
  ├─ LLM call             → response tokens counted → tracker updated
  └─ CheckFill()          → status drives the response:
        ├─ "ok"          continue normally
        ├─ "compact"     strategy compaction runs (cw.Compact)
        ├─ "warning"     strategy compaction runs (cw.Compact) — same action as "compact"
        ├─ "emergency"   strategy compaction runs (cw.Compact) — same action as "compact"
        └─ "reject"      context too full even after compaction
```

`CheckFill` maps the current fill percentage to a status: `>= 100` ⇒ `reject`; `>= EmergencyPercent` ⇒ `emergency`; `>= WarningPercent` ⇒ `warning`; `>= PredictivePercent` ⇒ `compact`; otherwise `ok`.

History mutation runs on **every** `BuildPrompt` (unconditionally), and tool-output pruning runs on every `BuildPrompt` once fill reaches the `Pruning.ThresholdPercent` floor (default `50`) — neither depends on the `CheckFill` status. Evicted content is preserved via the `agent.ToolResultCache` (recoverable via a `tool_result_read` tool).

When `SetPendingUserInterjection` holds a resume nudge, every `BuildPrompt` appends it as the final user message after the seeded step history. Prompt construction does not consume it: the executor calls `ConsumePendingUserInterjection` only after an LLM response succeeds. Therefore a context-window error followed by compaction and a second `BuildPrompt` retains the same nudge, while later successful turns do not repeat it.

### Content wrapping for untrusted tools

When `InjectionDefenseEnabled` is `true`, tool outputs from untrusted sources (flagged via the step's `IsUntrusted`) are wrapped in `<untrusted-content>` tags before becoming an LLM message. Wrapping happens after pruning/mutation and before message construction — the last point before content reaches the LLM API.

## Invariants

- The context window never exceeds the model's limit (compaction prevents overflow).
- The system prompt is always preserved (never compacted away); `BuildPrompt` splits it on `CacheBreakMarker` into multiple system messages for provider-level prompt caching.
- Compaction is triggered proactively (before overflow, not after).
- History mutation and pruning run first in message construction; outputs already replaced by a cache reference or step-status eviction text are not overwritten by the pruning placeholder.
- Entire response groups are protected if any step in the group is protected (prevents malformed assistant/tool message pairs).
- Each step/subagent has its own `ContextManager` (no sharing between parallel steps).
- `SeedSteps` wholesale-replaces the step history: it clears compaction state and recalculates the token-tracker delta for the batch, so fill accounting stays correct until the next `CorrectTokenCount`.
- A non-empty pending user interjection remains the final prompt message across repeated builds and is cleared only by explicit `ConsumePendingUserInterjection` after successful delivery.

## Configuration

`ContextWindowConfig` is the single source of memory tuning. Defaults:

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `SafetyMarginPercent` | `5` | Context window fraction reserved for counting inaccuracy. |
| `Thresholds.PredictivePercent` | `85` | Strategy compaction trigger — "compact" status (>=85%) runs `cw.Compact`. |
| `Thresholds.WarningPercent` | `92` | "warning" status — runs the same `cw.Compact` as "compact". |
| `Thresholds.EmergencyPercent` | `98` | "emergency" status — runs the same `cw.Compact` as "compact"/"warning". |
| `Pruning.KeepLastN` | `3` | Recent tool results always kept. |
| `Pruning.ThresholdPercent` | `50` | Fill % below which pruning is skipped entirely. |
| `ModelMeta.ContextWindow` (when 0) | `128000` | Fallback so compaction still works for unknown models. |
| `ModelMeta.OutputLimit` (when 0) | `4096` | Fallback output limit. |

`HistoryMutation` (set via `SetHistoryMutation`, before the first `BuildPrompt`):

| Field | Description |
| ----- | ----------- |
| `ToolResultEvictionStep` | Steps after which a tool result is replaced with a cache reference (`0` disables). |
| `EvictStepStatus` | Immediately evict `update_checklist` (bookkeeping) results. |
| `DedupRepeatedReads` | Replace duplicate file reads (same path + mtime → same cache hash) with a reference. |

## Extension Points

- **New compaction strategy**: implement `agent.CompactionStrategy` and register it in the `NewCompactionStrategy` factory (or pass it directly to `ContextWindowConfig.Strategy`).
- **Custom thresholds / pruning**: override compaction trigger percentages and `KeepLastN`/`ProtectedTools`.
- **Alternative token counter**: swap the `llm.TokenCounter` implementation; the `ContextTokenTracker` corrects estimates with API-reported actuals.
- **Per-step pruning overrides**: `PruningOverride` (carried through `ContextManagerFactory`) lets a step supply its own `KeepLastN`/`ProtectedTools`.
- **Optional `ContextManager` capabilities**: `TaskAware` (`SetTask`), `BlockTaskAware` (`SetTaskWithBlocks`), `ConversationAware` (`SetPriorConversation`), `TrackerProvider` (`ContextTracker`), `StepSeedable` (`SeedSteps`), orchestration `InterjectionAware` (`SetPendingUserInterjection`), and agent `InterjectionConsumer` (`ConsumePendingUserInterjection`) are type-asserted by the Conductor/executor and wired when implemented. `BlockTaskAware` carries structured content blocks (text + images) for a multimodal user message — `BuildPrompt` emits a `Message` with `ContentBlocks` and `NormalizeContentBlocks` prepends the task text when the blocks lack a text block. `SeedSteps` wholesale-replaces the step history and clears compaction state, enabling resume from a checkpoint; it is required when the Conductor's `ResumeSteps` is set. The interjection pair keeps a one-shot resume nudge through reactive-compaction retries and retires it after successful delivery.

## Related Specs

- [compaction.md](compaction.md) — strategy details, pruning, history mutation
- [blackboard.md](blackboard.md) — inter-step shared state
- [../orchestration/executor.md](../orchestration/executor.md) — the executor drives threshold-gated compaction each iteration
- [../orchestration/conductor.md](../orchestration/conductor.md) — wires the ContextManager per run; `CompactOnStart` forces one compaction pass before the first LLM call
- [../prompt-building.md](../prompt-building.md) — `CacheBreakMarker` and provider-level prompt caching
