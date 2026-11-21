# Framework (Root Package)

## Purpose

The root package (`github.com/v0lka/sp4rk`) is the engine's only entry point: it owns the `Framework` (the shared LLM router, model registry, tool registry, MCP gateway, and tool-result cache), assembles configuration through the classic `Config` and the fluent `NewF`/`RunF`/`TaskF` builders, and drives the two run shapes — a single ReAct loop and the Plan&Execute pipeline.

## Key Files

- `github.com/v0lka/sp4rk` (`framework.go`) — `Framework`, `Config`/`LLMConfig`/`MCPConfig`/`ExecutionConfig`/`CompactionConfig`, `New`, `NewConductor`, `Execute`, `Shutdown`, `RestoreBlackboard`, `ToolRegistry`/`LLMRouter`/`ContextFactory` accessors, `buildContextWindow`, `newBlackboard`
- `github.com/v0lka/sp4rk` (`builder.go`) — `FrameworkBuilder`, `NewF`, `Build`, `mergeConfig`, `Config`/`Options` escape hatches
- `github.com/v0lka/sp4rk` (`options.go`) — the `Option` functional-options set and the matching `FrameworkBuilder` setters (`ConfirmFunc`, `AutoApprove`, `HITL`, `MaxSteps`, `Streaming`, `Logger`, `NoAutoFinish`)
- `github.com/v0lka/sp4rk` (`run.go`) — `RunBuilder`, `Framework.RunF`, `FrameworkBuilder.Run`, `RunBuilder.Ask`
- `github.com/v0lka/sp4rk` (`task.go`) — `TaskBuilder`, `Framework.TaskF`, `FrameworkBuilder.Task`, `Execute`, `runPlanned`, `runStep`, `planCompletionStatus`
- `github.com/v0lka/sp4rk` (`tools.go`) — tool bundles `FileTools`/`MemoryTools`/`CodeTools`/`FinishTool`/`AllBuiltinTools`/`Tools`
- `github.com/v0lka/sp4rk` (`mcp.go`) — `MCPStdio`/`MCPHTTP` entry constructors and builder methods
- `github.com/v0lka/sp4rk` (`provider.go`) — `Anthropic`/`OpenAI`/`OpenAICompatible` entry constructors and builder methods
- `github.com/v0lka/sp4rk` (`prompts.go`) — `DefaultPromptSet` (planner prompts) and `DefaultReflectorPrompt`

## Core Types

```go
// Shared infrastructure container; creates per-task conductors.
type Framework struct { /* llmRouter, tools, modelReg, mcpGateway, toolCache, logger */ }

// Classic configuration; zero-value numeric fields use the sentinel convention.
type Config struct {
    LLM                 LLMConfig                    // providers + default model + retry/backoff + output reserve
    MCP                 *MCPConfig                   // nil = no MCP integration
    Execution           ExecutionConfig              // MaxSteps, budgets, Streaming
    Compaction          CompactionConfig             // strategy + 85/92/98 thresholds
    HITL                agent.HITLHandler            // nil = defaults (allow calls, deny extensions)
    ConfirmFunc         tools.ConfirmFunc            // fail-closed confirmation channel
    Checkpointer        orchestration.Checkpointer   // nil = in-memory blackboards
    OnBlackboardChanged func(changeType string)      // fires only with a Checkpointer
    Logger              *slog.Logger                 // nil = slog.Default()
}

// Fluent builders over the same *Framework.
type FrameworkBuilder struct { /* options + accumulated error */ } // NewF().…Build()
type RunBuilder       struct { /* ctx, fw, events, prompt */ }     // fw.RunF(ctx).…Ask(msg)
type TaskBuilder      struct { /* planner/reflector/models/budgets */ } // fw.TaskF(ctx, task).…Execute()
```

## Flow

```
New(cfg) or NewF()…Build()
  │  builds llm.Router + ModelRegistry, ToolRegistry (+ ConfirmFunc when set),
  │  tool-result cache (TTL from ToolCacheTTLSeconds, default 5m; negative = no cache),
  │  MCP gateway (best-effort start; failure is a warning).
  │  Fluent only: registers accumulated tools, then the finish tool
  │  (autoFinish defaults true; NoAutoFinish opts out).
  ▼
RunF(ctx)…Ask(msg) ────────────► Framework.Execute ─► one Conductor.Run (single ReAct loop)
TaskF(ctx, task)…Execute()
  ├─ without .Plan():  one Conductor.Run on the task message
  └─ with .Plan():     runPlanned — Planner.Plan → DAG waves (ready steps run
                       sequentially) → per-step retry/reflect/replan with
                       BuildCarryForward on adopted replans → planCompletionStatus
  │
  ▼
each Execute/TaskF builds a fresh blackboard (ID prefix "execute"/"taskf"):
CheckpointedBlackboard when Config.Checkpointer is set, else MapBlackboard;
Shutdown() (sync.Once) stops the MCP gateway.
```

## Invariants

- `New` requires at least one LLM provider and returns an error otherwise.
- Numeric `Config` fields follow the sentinel convention: `0` means "use the default", a negative value means "explicitly 0/disabled" (e.g. `MaxSteps`, `MaxRetries`, `LLM.MaxRetries`, `ToolCacheTTLSeconds`); fields with no meaningful disabled state (`SafetyMarginPercent`, compaction thresholds, `MaxDependencyContextChars`) treat only `0` as unset.
- The fluent build delegates to `New` and returns the same `*Framework` type; `mergeConfig` allocates fresh slice/map storage so the base `Config` and its collections are never aliased or mutated.
- `NewF` sets `autoFinish: true`; the finish tool registers after all accumulated tools, and `NoAutoFinish`/`WithoutAutoFinish` disables the registration.
- A `Streaming` builder call always overrides a `Streaming: true` arriving from a base `Config` (the explicit value wins; tracked separately from the `0`-means-default sentinel).
- `TaskBuilder` defaults: `MaxRetries` 2, `MaxReplans` 3, `Compaction` `sliding_window`; without `.Plan()`, `Execute` runs a single `Conductor.Run`.
- Planning runs on `planModel` (else the active model), execution on `execModel` (the Conductor is pinned to it via `SetModel`), and the planning model is restored afterwards; mid-loop switch failures are non-fatal and surfaced via `OnServiceMeta`.
- A cooperative pause suspends the DAG immediately: the step is not retried, reflected on, or recorded as failed; its trajectory stays on the blackboard for resume.
- Terminal status derives with precedence aborted > paused > failed > partial > success; `partial` accompanies `ErrExecutionIncomplete` and fires only when no step failed and nothing was aborted or paused (a cyclic or dangling `DependsOn` graph).
- Exhausting the replan budget reports `OnReplanFailed` and stops with a partial result rather than looping.
- `OnBlackboardChanged` fires only when a `Checkpointer` is configured (the hook rides `CheckpointedBlackboard.SetOnChanged`).
- Every task enters the engine through `Execute`, `NewConductor`, `RunF`, or `TaskF`.

## Configuration

| Parameter | Default | Notes |
| --------- | ------- | ----- |
| `Execution.MaxSteps` | `50` | Negative disables the loop entirely. |
| `Execution.MaxRetries` | `2` (resolved) | Resolved by `New` but consumed by no engine path today — the Plan&Execute retry budget is `TaskBuilder.MaxRetries`. |
| `Execution.SafetyMarginPercent` | `5` | Also fed to the LLM router. |
| `Execution.PreWarningPercent` | disabled (`0`) | Pre-compaction store_fact nudge threshold. |
| `Execution.ToolCacheTTLSeconds` | `300` | `0` → 5 minutes; negative disables the cache. |
| `Execution.Streaming` | `false` | Opt-in `AssistantChunk` deltas. |
| `LLM.MaxRetries` / `InitialBackoff` / `MaxBackoff` | `3` / `1s` / `30s` | Resolved by `llm.NewRouter`; negative = explicitly 0. |
| `LLM.OutputTokenReserve` | `4096` | Reserves window space for model output. |
| `Compaction.Strategy` | `sliding_window` | Unrecognized names fall back to sliding window. |
| `Compaction` thresholds | `85` / `92` / `98` | Predictive / warning / emergency. |
| `TaskBuilder.MaxRetries` / `MaxReplans` | `2` / `3` | Per-step retry budget; global replan budget. |

Events default to `agent.NoopEvents` (`RunBuilder`) / `orchestration.NoopEvents` (`TaskBuilder`) when unset. `TaskBuilder.Workspace` attaches the workspace path via `tools.WithWorkspacePath`.

## Extension Points

- `FrameworkBuilder.Config(cfg)` / `.Options(…)` / `WithConfig` surface classic-API fields not yet exposed as dedicated builder methods.
- `fw.ContextFactory()` hands the planner (or any host component) the same `ContextManager` behavior the Framework-built Conductors use.
- `TaskBuilder.Planner(p)` / `.Reflector(r)` inject custom implementations; `.Models(plan, exec)` splits planning from execution; `.Skills(...)` feeds the planner.
- `DefaultPromptSet()` returns an overridable per-field planner prompt set; `DefaultReflectorPrompt()` is a plain string.
- Tool bundles compose: `Tools(ts...)` groups custom tools with the shipped bundles in one `WithTools` call.

## Related Specs

- [../architecture/layers.md](../architecture/layers.md) — the root layer's placement and import rules
- [../architecture/data-flow.md](../architecture/data-flow.md) — request lifecycle, statuses, blackboard wiring
- [../architecture/security-model.md](../architecture/security-model.md) — the fail-closed `ConfirmFunc` gate the Config wires
- [../contracts/agent-execution.md](../contracts/agent-execution.md) — the host callbacks (`Events`, `ConfirmFunc`, `HITLHandler`)
- [../domains/orchestration/README.md](../domains/orchestration/README.md) — the `Conductor` primitive each run drives
