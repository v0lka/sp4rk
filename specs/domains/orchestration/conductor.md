# Conductor

## Role

The SDK-level single-loop task owner: a `Conductor` runs **one ReAct loop** that owns one task end-to-end, wrapping the lower-level `agent.Executor` and wiring blackboard-backed stores into the execution context so context-aware tools see shared state.

## Key Files

- `github.com/v0lka/sp4rk/orchestration` — `Conductor`, `ConductorConfig`, `NewConductor`, `Conductor.Run`, `WithDelegationRegistry`, `PendingDelegations`
- `github.com/v0lka/sp4rk/agent` — `Executor.Run` (the ReAct loop the Conductor launches), `LLMCaller`, `ToolExecutor`, `Events`, `HITLHandler`, `EditVerifyRunner`, `ErrPaused`, `InterjectionConsumer`
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
│      optional TaskAware / BlockTaskAware / ConversationAware / TrackerProvider /
│      InterjectionAware capabilities; additionally asserted against StepSeedable
│      when ResumeSteps is set).
│
├─ 3. Inject blackboard-backed stores into ctx:
│      ├─ StepOutputStore  (read_step_output / list_step_outputs)
│      ├─ FactStore        (store_fact / search_facts)
│      ├─ AttachmentStore  (read_attachment)
│      └─ FinalResultStore (read_final_result)
│
├─ 4. When ResumeSteps is set, seed the ContextManager (via StepSeedable) and the
│      Executor (via WithResumeSteps) with the prior steps so the loop continues
│      from len(steps)+1. When PendingUserInterjection is set, install the one-shot
│      nudge through InterjectionAware. When CompactOnStart is set, run one forced
│      compaction pass on the ContextManager before the executor launches (see
│      [Compact on start](#compact-on-start)). Install PauseChecker,
│      UserMessageSource, and VerifyOnEdit on the Executor when configured; then
│      launch one Executor.Run with the finish-join guard.
│
├─ 5. Map the ExecutorResult onto ExecutionStatus:
│      ├─ Finished == true                          → ExecutionStatusSuccess
│      ├─ loop ended without finish (budget/abort)  → ExecutionStatusPartial
│      ├─ errors.Is(err, agent.ErrPaused)           → ExecutionStatusPaused
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

### Step-boundary signals (PauseChecker, UserMessageSource)

Two optional `ConductorConfig` callbacks are installed on the executor so the host can interact with a run at its step boundaries:

| Config | Executor hook | Effect |
| ------ | ------------- | ------ |
| `PauseChecker func(ctx) bool` | `SetPauseChecker` | A true return stops the loop cooperatively with `agent.ErrPaused`; `Run` maps it to `ExecutionStatusPaused` and leaves `Output` empty (a clean, resumable checkpoint — never the raw sentinel string). |
| `UserMessageSource func(ctx) string` | `SetUserMessageSource` | A non-empty return is delivered to the model as the final `{role:user}` message of the very next LLM request (see [executor.md § Live user messages](executor.md#live-user-messages-usermessagesource)). The source drains one message per boundary. |

Both are polled inside `Executor.Run` in that order — pause first, then the message poll — so a pausing run never consumes a queued message: the host's queue keeps it for the resumed run. The host threads one universal pause signal and one live-message queue here so any conductor run (normal or specialized) is pauseable and steerable. Nil (default) disables each; both are fully backward-compatible.

### Optional ContextManager capabilities

The Conductor type-asserts the `ContextManager` returned by the factory against six named capability interfaces (the SDK's `memory.ContextWindow` implements all six):

| Capability | Purpose |
| ---------- | ------- |
| `TaskAware` (`SetTask`) | Receives the formatted task content (the user message). Used when `ConductorConfig.ContentBlocks` is empty. |
| `BlockTaskAware` (`SetTaskWithBlocks`) | Receives the formatted task text together with structured content blocks (text + images). Used when `ConductorConfig.ContentBlocks` is non-empty, giving the blocks precedence over the plain `Content` string so providers render a multimodal user message. |
| `ConversationAware` (`SetPriorConversation`) | Receives prior conversation messages rendered before the current task, when `ConductorConfig.ConversationHistory` is set. Without this, a follow-up like "implement variant a" has no referent. |
| `TrackerProvider` (`ContextTracker`) | Exposes the token tracker so API-reported token corrections flow back into fill accounting. |
| `StepSeedable` (`SeedSteps`) | Receives prior ReAct steps when `ConductorConfig.ResumeSteps` is set, so a resumed run renders them in `BuildPrompt` as assistant+tool messages. Unlike the other capabilities, this one is asserted only when `ResumeSteps` is non-empty and `Run` fails fast if it is absent (see [Resume from a checkpoint](#resume-from-a-checkpoint)). |
| `InterjectionAware` (`SetPendingUserInterjection`) | Receives the one-shot `PendingUserInterjection` for resume-with-nudge. `BuildPrompt` keeps it as the final user message until the executor's `InterjectionConsumer` retires it after a successful LLM response. |

A custom `ContextManager` that does not implement these still works — the corresponding features are simply skipped. `StepSeedable` is the sole fail-fast exception because omitting seeded trajectory messages would make resume incoherent; a missing `InterjectionAware` drops only the optional nudge.

### Resume from a checkpoint

When `ConductorConfig.ResumeSteps` is non-empty, `Run` resumes execution from a checkpoint instead of starting fresh. It makes a single defensive copy of the steps and seeds both the `ContextManager` (via its `StepSeedable` capability — see [Optional ContextManager capabilities](#optional-contextmanager-capabilities)) and the `Executor` (via `agent.WithResumeSteps`) with that copy. The executor's step counter then continues from `len(steps)+1` and the full trajectory (seeded plus new steps) syncs to the `TrajectoryStore`.

Budget: the resumed steps count against the shared `MaxSteps` budget, not in addition to it. A meaningful resume needs `MaxSteps` meaningfully larger than `len(ResumeSteps)`; when `len(ResumeSteps) >= MaxSteps` the resumed loop has little or no room for new steps.

`ResumeSteps` is zero-value by default (nil/empty), which is fully backward-compatible: `Run` starts a fresh loop at step 1 with no seeding.

### Resume with a user interjection

`ConductorConfig.PendingUserInterjection` carries a one-shot message for a resumed run. When non-empty and the context manager implements `InterjectionAware`, `Run` calls `SetPendingUserInterjection` after seeding steps. The context window renders it as the final user message after the pending tool result on every prompt build until the executor receives a successful LLM response and calls `InterjectionConsumer.ConsumePendingUserInterjection`. This preserves the nudge across a context-window failure, compaction, and retry without duplicating it after success. Empty is the backward-compatible default.

### Compact on start

`ConductorConfig.CompactOnStart` forces one compaction pass on the `ContextManager` immediately after the resume/nudge seeding and before the executor launches — hence before the first LLM call — regardless of the fill thresholds that gate the Executor's reactive compaction. Its purpose is manual compaction of a paused task: the host re-launches the Conductor with `ResumeSteps` + `CompactOnStart` so the seeded trajectory is compressed up front rather than waiting for a threshold to trip mid-run.

When `Compact` reports a result, `Run` emits a `ContextCompaction` event carrying the result's before/after fill percentages with an empty step ID — the pass belongs to the run, not to an executor step. A nil result (nothing to compact: empty steps or a nil strategy) emits no event.

The pass covers only the start of the run: the Executor's threshold-driven reactive compaction inside the loop is unchanged and still fires when fill warrants it. `false` (the default) keeps compaction purely threshold-driven and is backward-compatible.

### Post-edit verification

`ConductorConfig.VerifyOnEdit` and `VerifyOnEditMaxOutputChars` are forwarded to `Executor.SetVerifyOnEdit`. The runner executes a command selected by user configuration, not model output, once after each response group containing a successful `write_file`/`edit_file`; its capped result becomes a `[verify_on_edit]` observation. Nil disables the hook, and a non-positive cap selects `agent.DefaultVerifyOnEditCap`.

## Error Handling

- **LLM/tool fatal error**: the executor returns a non-nil error; `Run` wraps it and still returns a non-nil `*ExecutionResult` with best-effort output and `Status == ExecutionStatusFailed`.
- **Budget exhausted / circuit-breaker abort** (no error, `Finished == false`): mapped to `Status == ExecutionStatusPartial` (resumable).
- **Context cancelled**: mapped to `ExecutionStatusCancelled` and propagated as an executor error.
- **Cooperative pause**: `agent.ErrPaused` is returned for `errors.Is` inspection, while the non-nil `ExecutionResult` has `Status == ExecutionStatusPaused`, empty `Output`, and the resumable trajectory in `Steps`.
- **Verification failure/timeout/infrastructure failure**: remains a `[verify_on_edit]` observation and does not change the execution status by itself.
- **Missing required factory**: `Run` returns an error before the loop starts.
- **Resume without `StepSeedable`**: when `ResumeSteps` is configured but the `ContextManager` produced by the factory does not implement `StepSeedable`, `Run` returns an error before the loop starts (the steps could not be seeded into the prompt, so a silent incoherent resume is avoided).

## Invariants

- Exactly one `Executor.Run` instance is active per Conductor run.
- `Conductor.Run` injects the four blackboard-backed store adapters into `ctx` before launching the executor.
- The returned `*ExecutionResult` is always non-nil (even when the error is non-nil).
- The returned `Blackboard` is the same instance passed in.
- When a `PendingDelegations` registry is in `ctx`, `finish` is rejected while it reports pending async work.
- When `ResumeSteps` is set, `Run` fails fast unless the `ContextManager` implements `StepSeedable`; on success it seeds both the `ContextManager` and the `Executor` with the same defensive copy of the steps.
- A configured `PendingUserInterjection` is appended after seeded history and remains pending through reactive-compaction retries until one LLM response succeeds.
- When `CompactOnStart` is set, `Run` performs one start-of-run compaction pass after seeding and before the executor launches; the Executor's threshold-driven reactive compaction inside the loop is unchanged.
- Pause is checked before the live-message source at every boundary, so a paused run returns a checkpoint without consuming the next queued message.
- A pending verify-on-edit run is flushed before a paused or finished result is returned.
- A Conductor instance is reusable across steps; per-run state lives on the `ContextManager`, not on the Conductor.

## Related Specs

- [README.md](README.md) — orchestration domain overview
- [executor.md](executor.md) — the ReAct loop primitive the Conductor wraps
- [../memory/blackboard.md](../memory/blackboard.md) — the shared state container wired into the run
- [../memory/README.md](../memory/README.md) — ContextManager capabilities
- [subagents.md](subagents.md) — delegated execution primitive (drives the pending-delegations guard)
