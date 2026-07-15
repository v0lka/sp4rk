# Conductor

## Role

The SDK-level single-loop task owner: a `Conductor` runs **one ReAct loop** that owns one task end-to-end, wrapping the lower-level `agent.Executor` and wiring blackboard-backed stores into the execution context so context-aware tools see shared state.

## Key Files

- `github.com/v0lka/sp4rk/orchestration` — `Conductor`, `ConductorConfig`, `NewConductor`, `Conductor.Run`, `WithDelegationRegistry`, `PendingDelegations`
- `github.com/v0lka/sp4rk/agent` — `Executor.Run` (the ReAct loop the Conductor launches), `LLMCaller`, `ToolExecutor`, `Events`, `HITLHandler`
- `github.com/v0lka/sp4rk/orchestration` (adapters) — `NewStepOutputStore`, `NewFactStore`, `NewAttachmentStore`, `NewFinalResultStore` (Blackboard → `agent.*Store`)
- `github.com/v0lka/sp4rk/llm` — `ModelRegistry`, `TokenCounter`, `Message`
- `github.com/v0lka/sp4rk/memory` — `ContextManager` returned by `ContextManagerFactory`

## Behavior

A Conductor is reusable across steps: the system-prompt factory receives the step description at `Run` time, so the same instance adapts to each step it executes.

### Construction

`NewConductor(cfg ConductorConfig) *Conductor` creates a Conductor. If `MaxSteps` is zero it defaults to `80`. `ContextFactory` and `SystemPrompt` are required (see [README.md](README.md#configuration) for the full config). When `ResumeSteps` is non-empty, `Run` resumes from a checkpoint (see [Resume from a checkpoint](#resume-from-a-checkpoint)).

### Lifecycle

```
conductor.Run(ctx, message, bb, availableTools, events, compactionStrategy)
│
├─ 1. Resolve compaction strategy ("sliding_window" | "summarization" | "hierarchical";
│      empty → "sliding_window").
│
├─ 2. Build a ContextManager via ContextFactory (type-asserted against the
│      optional TaskAware / ConversationAware / TrackerProvider capabilities;
│      additionally asserted against StepSeedable when ResumeSteps is set).
│
├─ 3. Inject blackboard-backed stores into ctx:
│      ├─ StepOutputStore  (read_step_output / list_step_outputs)
│      ├─ FactStore        (store_fact / search_facts)
│      ├─ AttachmentStore  (read_attachment)
│      └─ FinalResultStore (read_final_result)
│
├─ 4. When ResumeSteps is set, seed the ContextManager (via StepSeedable) and the
│      Executor (via WithResumeSteps) with the prior steps so the loop continues
│      from len(steps)+1; then launch a single Executor.Run with the step
│      description, available tools, system prompt, and finish-join guard.
│
├─ 5. Map the ExecutorResult onto ExecutionStatus:
│      ├─ Finished == true                          → ExecutionStatusSuccess
│      ├─ loop ended without finish (budget/abort)  → ExecutionStatusPartial
│      ├─ error & cancelled (context.Canceled or    → ExecutionStatusCancelled
│      │  ctx.Err() set)
│      └─ Executor returned any other error         → ExecutionStatusFailed
│
└─ 6. Return *ExecutionResult (always non-nil, even alongside a non-nil error,
         which carries best-effort output). The returned Blackboard is the same
         instance passed in, now populated with any reflections recorded.
```

### Finish-join guard

A caller may inject a `PendingDelegations` implementation via `WithDelegationRegistry`. The Conductor's finish guard checks it before accepting `finish`: if any async delegations are still pending, `finish` is rejected with a nudge listing them. This prevents an agent from silently abandoning background work. `PendingDelegations` is a minimal interface the SDK defines to avoid a circular dependency with higher layers:

```go
type PendingDelegations interface {
    ListPending() []string
}
```

`Run` returns an error only when the context factory or system-prompt factory is missing, or when the underlying executor returns an error. A non-nil error is still accompanied by a non-nil `*ExecutionResult` carrying best-effort output.

### Optional ContextManager capabilities

The Conductor type-asserts the `ContextManager` returned by the factory against four named capability interfaces (the SDK's `memory.ContextWindow` implements all four):

| Capability | Purpose |
| ---------- | ------- |
| `TaskAware` (`SetTask`) | Receives the formatted task content (the user message). |
| `ConversationAware` (`SetPriorConversation`) | Receives prior conversation messages rendered before the current task, when `ConductorConfig.ConversationHistory` is set. Without this, a follow-up like "implement variant a" has no referent. |
| `TrackerProvider` (`ContextTracker`) | Exposes the token tracker so API-reported token corrections flow back into fill accounting. |
| `StepSeedable` (`SeedSteps`) | Receives prior ReAct steps when `ConductorConfig.ResumeSteps` is set, so a resumed run renders them in `BuildPrompt` as assistant+tool messages. Unlike the three above, this one is asserted only when `ResumeSteps` is non-empty and `Run` fails fast if it is absent (see [Resume from a checkpoint](#resume-from-a-checkpoint)). |

A custom `ContextManager` that does not implement these still works — the corresponding features are simply skipped, except `StepSeedable` which is required when resuming.

### Resume from a checkpoint

When `ConductorConfig.ResumeSteps` is non-empty, `Run` resumes execution from a checkpoint instead of starting fresh. It makes a single defensive copy of the steps and seeds both the `ContextManager` (via its `StepSeedable` capability — see [Optional ContextManager capabilities](#optional-contextmanager-capabilities)) and the `Executor` (via `agent.WithResumeSteps`) with that copy. The executor's step counter then continues from `len(steps)+1` and the full trajectory (seeded plus new steps) syncs to the `TrajectoryStore`.

Budget: the resumed steps count against the shared `MaxSteps` budget, not in addition to it. A meaningful resume needs `MaxSteps` meaningfully larger than `len(ResumeSteps)`; when `len(ResumeSteps) >= MaxSteps` the resumed loop has little or no room for new steps.

`ResumeSteps` is zero-value by default (nil/empty), which is fully backward-compatible: `Run` starts a fresh loop at step 1 with no seeding.

## Error Handling

- **LLM/tool fatal error**: the executor returns a non-nil error; `Run` wraps it and still returns a non-nil `*ExecutionResult` with best-effort output and `Status == ExecutionStatusFailed`.
- **Budget exhausted / circuit-breaker abort** (no error, `Finished == false`): mapped to `Status == ExecutionStatusPartial` (resumable).
- **Context cancelled**: propagated immediately as an executor error.
- **Missing required factory**: `Run` returns an error before the loop starts.
- **Resume without `StepSeedable`**: when `ResumeSteps` is configured but the `ContextManager` produced by the factory does not implement `StepSeedable`, `Run` returns an error before the loop starts (the steps could not be seeded into the prompt, so a silent incoherent resume is avoided).

## Invariants

- Exactly one `Executor.Run` instance is active per Conductor run.
- `Conductor.Run` injects the four blackboard-backed store adapters into `ctx` before launching the executor.
- The returned `*ExecutionResult` is always non-nil (even when the error is non-nil).
- The returned `Blackboard` is the same instance passed in.
- When a `PendingDelegations` registry is in `ctx`, `finish` is rejected while it reports pending async work.
- When `ResumeSteps` is set, `Run` fails fast unless the `ContextManager` implements `StepSeedable`; on success it seeds both the `ContextManager` and the `Executor` with the same defensive copy of the steps.
- A Conductor instance is reusable across steps; per-run state lives on the `ContextManager`, not on the Conductor.

## Related Specs

- [README.md](README.md) — orchestration domain overview
- [executor.md](executor.md) — the ReAct loop primitive the Conductor wraps
- [../memory/blackboard.md](../memory/blackboard.md) — the shared state container wired into the run
- [../memory/README.md](../memory/README.md) — ContextManager capabilities
- [subagents.md](subagents.md) — delegated execution primitive (drives the pending-delegations guard)
