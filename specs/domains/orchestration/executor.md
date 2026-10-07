# Executor

## Role

The ReAct loop primitive (Thought → Action → Observation) with circuit breakers, mutation/checklist gates, two-stage tool-result truncation, and implicit-finish detection. It is the load-bearing primitive shared by two callers: the [Conductor](conductor.md) (a top-level `Executor.Run` that owns a task end-to-end) and **subagents** (isolated `Executor.Run` instances launched in goroutines). The Executor is agnostic to which caller invoked it.

## Key Files

- `github.com/v0lka/sp4rk/agent` — `Executor`, `NewExecutor`, `Executor.Run`, `ExecutorResult`, `Step`, `FinishTool`, `CircuitBreakerConfig`, `ToolResultBudget`, `ToolResultCache`, `DetectToolCallSyntaxInContent`, configuration setters (`Set*`), context helpers
- `github.com/v0lka/sp4rk/agent` (streaming) — `WithStreaming` / `Executor.SetStreaming`, the `ChatRequest.DeltaSink` wiring, and the `emitAssistantEvents` finalization
- `github.com/v0lka/sp4rk/agent` (executor internals) — single-call dispatch, batch meta-tool interception, implicit-finish handling, mutation/checklist gate logic
- `github.com/v0lka/sp4rk/agent` (checklist helpers) — shared Markdown-checkbox parsing (`ParseTodoLine`, `ParseTodoItems`, `TodoItem`) used by both the executor (checklist diffing) and the `update_checklist` tool
- `github.com/v0lka/sp4rk/agent` — `ContextManager` / `CompactionStrategy` / `FillCheck` interfaces, `LLMCaller`, `ToolExecutor`, `Events`, `HITLHandler`
- `github.com/v0lka/sp4rk/tools/builtins` — `batch` meta-tool descriptor (intercepted at the executor, never executed directly)

## Behavior

The Executor is **not safe for concurrent use on a single instance** — `Run` must be called one at a time. Callers that need parallelism create a fresh `Executor` per step (see [subagents.md](subagents.md)).

### NewExecutor

```go
func NewExecutor(llmRouter LLMCaller, toolRegistry ToolExecutor, maxSteps int, opts ...Option) *Executor
```

The event emitter and the HITL handler are **nil-safe** — `nil` is replaced with `NoopEvents` and `NoopHITLHandler`. Options: `WithTokenCounter`, `WithEvents`, `WithSuppressAssistantEvents` (hides streaming events for sub-steps), `WithStreaming` (streams LLM text deltas live as `AssistantChunk` events; see [Streaming](#streaming)), `WithToolResultBudget` (defaults to `DefaultToolResultBudget()`), `WithCircuitBreaker` (defaults to `DefaultCircuitBreakerConfig()`), `WithHITL`, `WithResumeSteps` (seeds prior ReAct steps to resume from a checkpoint; see [Resume from a checkpoint](#resume-from-a-checkpoint)), `WithPauseChecker`, `WithUserMessageSource`, and `WithToolCallTimeout` (the per-tool-call ceiling — zero disables; installed at runtime via `SetToolCallTimeout`, with `SetToolCallTimeoutExempt` naming the interactive/long-running tools exempt from it). The equivalent runtime setters are available for pause/message hooks (`SetStreaming` mirrors `WithStreaming`); `SetVerifyOnEdit` installs the independent post-edit verifier.

### Run

```go
func (e *Executor) Run(ctx context.Context, taskTools []tools.ToolDescriptor, cw ContextManager) (*ExecutorResult, error)
```

`ctx` carries cancellation, workspace path, trajectory store, and other injected dependencies. `taskTools` are the tools available for this run; the `finish` tool is appended automatically if absent. A non-nil error indicates a fatal failure (LLM error, context cancellation). A `nil` error with `Finished == false` means the budget was exhausted or a circuit breaker aborted the loop.

### ReAct loop lifecycle (per iteration)

1. **Trajectory sync** — if a `TrajectoryStore` is in `ctx`, the current step history is synced so tools (e.g. a reflector) can read it.
2. **Step-limit boundary** — if the budget is reached, `HITLHandler.OnStepLimit` decides whether to grant more steps.
3. **Cooperative pause check** — if a pause checker is installed (`WithPauseChecker`/`SetPauseChecker`), a true return stops the loop immediately with `ErrPaused` and the trajectory so far (a resumable checkpoint). The checker is also polled *while a tool call is in flight* (on the executor's `toolWatchdogInterval` ticker, see [Per-tool-call ceiling and watchdog](#per-tool-call-ceiling-and-watchdog)), so a long tool reaches its checkpoint without waiting for the boundary.
4. **Live user message poll** — if a user-message source is installed (`WithUserMessageSource`/`SetUserMessageSource`), a non-empty return is appended to the trajectory and the context manager as a nudge-only step, so it renders as a `{role:user}` message in this iteration's LLM call.
5. **LLM call** — the prompt is built from the context manager and sent. If the provider reports a context-window-exceeded error and no delta of that attempt was delivered live, reactive compaction runs and the call is retried; after a delivered delta the error is returned instead (a retry would mix two attempts' visible text).
6. **Implicit-finish check** — if the model returns no tool calls, the executor decides whether this is a legitimate finish or a failure mode (printed tool-call syntax); a nudge may force an explicit `finish`.
7. **Truncation detection** — `max_tokens` with tool calls present ⇒ nudge injected, truncation counter checked against the circuit breaker.
8. **Tool execution** — each call runs after HITL approval; results are truncated in two stages, cached if applicable, recorded as observations.
9. **Compaction** — context fill is checked; compaction runs and `ContextFill`/`ContextCompaction` events fire when thresholds cross.

The loop terminates when `finish` is called (`Finished: true`) or the budget is exhausted (`Finished: false`).

### Nudge convention

Every corrective nudge the executor injects (circuit breakers, mutation/checklist gates, implicit-finish failure modes, wrap-up warnings) is delivered as a user-channel message prefixed with `[System]` — an operator-side directive the model is taught to treat as system-grade guidance delivered through the user channel. The engine's one-shot service client ([../oneshot.md](../oneshot.md)) repeats exactly this convention for single structured calls: each repair turn pairs an assistant echo of the model's own failed output with a `[System]`-prefixed user message restating the required format, with the same bound — two nudges, then a final refusal — where the executor aborts (`Finished: false`) and the one-shot client resolves its `OnFailurePolicy`.

### Circuit breakers

`CircuitBreakerConfig` holds thresholds that protect the executor from unproductive loops. When a threshold is crossed, a nudge is injected and, if the behavior persists, the loop aborts with `Finished: false`.

| Detection | Trigger | Abort action |
| --------- | ------- | ------------ |
| Repeat | Consecutive identical tool calls (name + args) | `HITLHandler.OnStepLimit` |
| Truncation | Consecutive `max_tokens`-truncated responses with tool calls | `HITLHandler.OnStepLimit` |
| Parse error | Consecutive parse errors on the same tool | `HITLHandler.OnStepLimit` |
| Fruitless | Consecutive minimal-result calls | `HITLHandler.OnStepLimit` |
| Same tool | Same tool with varied args but similar results | `HITLHandler.OnStepLimit` |

On an abort, `HITLHandler.OnStepLimit` is invoked with the trigger reason and the same four options as the step limit: **AllowOnce** (reset the breaker's consecutive counter + nudge), **AllowMore** (at the step-limit boundary, grants a full batch of `maxSteps` additional iterations; inside a circuit breaker, a reprieve equivalent to AllowOnce — resets the counter so the loop continues within its remaining budget, but grants no extra iterations), **AllowAlways** (disable that breaker + nudge), **Deny** (stop). If `HITLHandler` is nil, the executor aborts immediately (headless/test behavior).

### Mutation gate

When `SetMutationRequired(true)` is set, the finish call is intercepted before completion. The gate checks whether any mutating tool executed **successfully** during the current step (scanning the trajectory for mutating tool names, excluding rejected/errored calls). No mutation + first attempt → inject a mutation nudge and retry; no mutation + second attempt → return `Finished: false`. Rejected tool calls and ambiguous tools (e.g. shell execution) do **not** count as mutations. The host enables this gate selectively (e.g. for code-modification steps).

### Checklist gate

Enabled by default (`SetChecklistGateEnabled`, default `true`); it activates only when an `update_checklist` tool is present in the run's tool set. It spans four mechanisms — two finish-time soft sub-gates and two proactive mid-step nudges — all aimed at keeping checklist progress visible throughout a step rather than batched near the end.

A **productive call** is any tool call except the empty-Action nudges, the `finish` terminator, and `update_checklist` (a bookkeeping call, not task progress). The productive-call count drives the trivial-step test and the staleness counter below, so a checklist update — successful or not — never inflates either.

Finish-time sub-gates (each one nudge attempt before finish is accepted):

- **Missing-checklist**: a non-trivial step (more than `checklistTrivialThreshold`, 2, productive calls) with no successful `update_checklist` call → inject a missing-checklist nudge, retry.
- **Unchecked-items**: the last successful checklist has unchecked items → inject an unchecked-items nudge, retry.

Proactive mid-step nudges (run after tool execution, before compaction, while the step is still in progress):

- **Staleness nudge**: once a checklist exists, if the agent makes `checklistStalenessThreshold` (3) or more productive calls since its last successful `update_checklist`, a nudge is injected prompting an immediate incremental update. The counter resets after each successful update, and the nudge is capped at `checklistStaleNudgeCap` (2) injections per step to avoid nudge fatigue. Emits an `ExecutorDiagnostic` with event `checklist_stale_nudge`.
- **Batching detection**: after each successful `update_checklist`, the new checklist is diffed against the previous one by item text. Marking more than one previously-unchecked item complete in a single call appends a correction suffix to that call's observation; marking exactly one earns brief positive reinforcement. The first update (initialization) is exempt. Emits an `ExecutorDiagnostic` with event `checklist_batched_update` when a batch is detected.

### Streaming

Streaming is opt-in via `WithStreaming(true)` (or `Executor.SetStreaming`). When enabled and assistant events are not suppressed (`WithSuppressAssistantEvents`), the executor installs `ChatRequest.DeltaSink` on every loop LLM call and forwards each non-empty text delta to `Events.AssistantChunk` as it arrives; the flag `runState.assistantStreamed` records whether any delta actually arrived. On an implicit finish — the response that ends the loop — the executor emits **only** `AssistantDone` when the text was streamed live (the deltas were the chunks), and falls back to a single full-text `AssistantChunk` + `AssistantDone` when nothing streamed — so a provider that ignores `DeltaSink` (e.g. the Google path) degrades gracefully to the pre-streaming behavior. A response that ends in tool calls emits its streamed deltas without an `AssistantDone` terminator; the terminator belongs to the response that finishes the loop. **No Run ever leaves a live stream open:** when a Run exits with deltas still unterminated — a successful explicit `finish` (whose answer travels in the `ExecutorResult`, not in an implicit assistant response), an error, a pause or the step limit — the Run-exit finalizer emits one closing `AssistantDone` carrying exactly what the open stream accumulated (text plus the summed usage of the responses that produced it; a response aborted mid-stream contributes no usage, since none completed), so a subsequent Run over the same `Events` consumer starts from a clean accumulator instead of inheriting the previous Run's text. With streaming off (the default) no sink is set and one `AssistantChunk` is emitted at the implicit finish, exactly as before. See [../llm-providers.md](../llm-providers.md#streaming) for the transport hook.

Deltas are emitted before the executor knows a response is final, so a response later discarded by a nudge will already have been partially streamed. A reactive-compaction retry is never issued after a delivered delta (the executor returns the error instead — see step 5 above), and the Run-exit finalizer closes any still-open stream so partial text never leaks into a subsequent Run. Streaming affects only event delivery timing — the assembled `ChatResponse` driving the loop is identical to the synchronous result.

### Implicit finish & failure-mode detection

When the LLM returns no tool calls, the executor decides whether to accept an implicit finish, nudge, or abort:

- Up to a small budget of general nudges are injected before accepting an implicit finish. In `suppressAssistantEvents` mode, a finish nudge requires an explicit `finish` call.
- **Failure mode**: `DetectToolCallSyntaxInContent` matches the model "printing" a tool invocation as text instead of emitting a `tool_use` block — either a fenced code block whose language tag looks like a tool name (`` ```bash_exec ``), or a lone JSON tool-call envelope printed as the whole response (the finish args `{"answer": "..."}`, a `{"name": ..., "arguments": {...}}` / `{"tool": ..., "args": {...}}` pair, or an OpenAI-style `{"function": {"name": ..., "arguments": ...}}`), optionally padded with the service keys `id`/`type`/`index` (e.g. Anthropic's `{"type":"tool_use","id":…,"name":…,"input":{…}}`) and optionally wrapped in one ```` ```json ```` fence. A dedicated nudge is injected a few times; after that the executor aborts with `Finished: false` (never a silent success). This is what prevents a leaked `{"answer": "..."}` from being surfaced verbatim as the final answer.

### Finish guard

`SetFinishGuard(func(ctx) error)` lets a caller block premature completion. It is a **hard gate**: every `finish` call re-invokes the guard, and a non-nil error rejects `finish` with a nudge and retries the action every time — finish is never auto-accepted while the guard still errors. (Contrast the mutation and checklist gates, which are soft: after one nudge attempt each, finish is accepted regardless.) This is how the Conductor's pending-delegations join check is expressed.

`SetStopTools(names ...string)` registers ordinary (host-provided) tools as **turn terminators**. A successful call to a listed tool ends the run with `Finished=true` — the tool's observation becomes the run output — instead of continuing to the next step; a *failed* call to a stop tool does not terminate the run (the error observation is returned to the model as usual). It is distinct from the inline `finish` tool: it lets the host model a bounded-turn protocol where the executor — not the model — ends the turn. A host goal-loop protocol, for example, registers the tool the agent uses to declare its goal verdict as a stop tool, so the per-turn working run ends the moment the verdict is declared — which is what makes a goal turn one bounded attempt that advances the turn counter and the turn budget. Threaded from the host via `orchestration.ConductorConfig.StopTools`.

Two semantics matter for reuse. First, the terminator fires for a stop-tool call made **inside `batch`** as well as standalone (shared `processSingleToolCall`/`processBatchTool` helper), so a host protocol cannot be silently defeated by batching the call. Second, a stop tool is a terminal boundary exactly like `finish`, so it **honours the `SetFinishGuard` join gate**: when the guard rejects (e.g. pending async delegations), the run injects the guard's nudge and retries instead of terminating — a stop tool cannot bypass the pending-async gate. Because a stop tool has no `answer` argument like `finish`, the run also records the model's own final assistant text in `ExecutorResult.Summary` (mirrored on `orchestration.ExecutionResult.Summary`), so a host can recover the turn's modeled output when `Output` holds only the tool's short confirmation.

### Resume from a checkpoint

`WithResumeSteps(steps []Step)` seeds the executor with pre-existing ReAct steps so `Run` continues from where it left off instead of starting fresh. When non-empty, `Run` seeds `state.allSteps` with them before the loop: the step counter starts at `len(steps)+1`, and the trajectory synced to the `TrajectoryStore` on every iteration includes the seeded steps (so tools such as a reflector see the complete history). The caller is responsible for seeding the `ContextManager` with the same steps (e.g. via `memory.ContextWindow.SeedSteps`) so they are rendered as assistant+tool messages in `BuildPrompt` — the executor itself does not push resumed steps into the context manager.

Budget: the resumed steps are counted against the shared `maxSteps` budget, not in addition to it. The loop runs until `stepNum <= maxSteps+1`, so a meaningful resume needs `maxSteps` meaningfully larger than `len(steps)`; otherwise the resumed loop has little or no room for new steps.

A zero value (nil/empty steps, or the option omitted) restores the default fresh-start behavior: the loop starts at step 1 with no seeded history.

### Resume-with-nudge interjection

A Conductor may set a one-shot pending user message on the `ContextManager` through the orchestration-level `InterjectionAware.SetPendingUserInterjection` capability. `BuildPrompt` appends that message as the final user turn after the seeded trajectory on every build until it is explicitly retired. After the first **successful** LLM response, the executor type-asserts the context manager against `agent.InterjectionConsumer` and calls `ConsumePendingUserInterjection`.

Consume-on-success is required for reactive compaction: when the first LLM call returns a context-window error, the executor compacts and rebuilds the prompt; the pending nudge remains present in the retry and is consumed only after that retry succeeds. A custom context manager without `InterjectionConsumer` degrades gracefully and retains the nudge until its owner clears it.

### Live user messages (UserMessageSource)

`WithUserMessageSource(source)` / `SetUserMessageSource(fn)` installs a live user-message source polled at every step boundary, immediately after the pause check and before the LLM call. A non-empty return is delivered to the model in the **very next LLM request**:

- the message is appended to the trajectory as a **nudge-only step** (`Step{UserNudge: msg}`) and pushed to the ContextManager via `AddStep` — the same rendering path as executor nudges and the resume-with-nudge interjection, so it lands as the final `{role:user}` message next to the pending tool result;
- one message is delivered per boundary: the source **drains** its backing queue on return, later boundaries inject the next queued message (FIFO);
- the poll order after the pause check guarantees a pausing run never swallows a queued message — the queue keeps holding it for the resumed run;
- an `ExecutorDiagnostic` with event `live_user_message` fires per delivery.

A nil/omitted source disables the injection entirely (default, backward-compatible). The source must be cheap and safe to call from the Run goroutine (invoked once per step). The executor makes no assumption about the queue's owner: the host owns the queue and its epilogue semantics — what happens to undelivered messages when the run finishes is a host concern.

### Post-edit verification

`SetVerifyOnEdit(runner EditVerifyRunner, maxOutputChars int)` installs a mechanical verification cycle whose command is supplied by user configuration rather than model output. A successful `write_file` or `edit_file` marks the current LLM response group dirty; the executor runs the verifier exactly once at the group's end, even when several edits occurred, and appends a `[verify_on_edit]` note to the final call's observation before events and prompt history receive it. Reads, unsuccessful tool results, and HITL-rejected edits never mark the group dirty.

`EditVerifyResult` carries combined output, exit code, timeout state/effective limit, and a runner infrastructure error. `FormatVerifyNote` distinguishes pass, verification failure, timeout, blocked/killed execution without an exit code, and inability to start. The command outcome remains an observation rather than an executor Go error. Output is capped by Unicode code points (`DefaultVerifyOnEditCap == 4000` when the configured cap is non-positive), preserving valid UTF-8 and adding an explicit truncation marker.

Pending verification is flushed on the paths that return a consumer for the note — a HITL rejection, a finish, and a pause checkpoint — so a successful edit cannot disappear from a resumable trajectory or final output without its verification result. The terminal arms that return no result (a tool-call ceiling, or an infrastructure/cancellation error) do not flush: with no consumer the note would be discarded, so running the command there would be wasted work. A nil runner is the default and leaves edit observations unchanged.

### Tool result cache & two-stage truncation

Every cacheable tool result is stored in `ToolResultCache` (keyed by `SHA256(toolName + "\x00" + content)`) before truncation:

- **Stage 1 — per-tool line/byte truncation** (`ToolTruncationConfig`): byte truncation is UTF-8 safe. Defaults ship for `read_file`, `ripgrep`, `glob`, `list_directory`, `web_fetch`, `bash_exec`.
- **Stage 2 — token budget** (`ToolResultBudget`): `HardCapTokens` / `MaxFillFraction` (defaults `30000` / `0.4`). When a result exceeds the budget it is truncated and a fragmentation nudge is appended telling the model how to retrieve fragments via a `tool_result_read` tool using the cache hash.

Cache behaviours: identical content from different tools gets different hashes; dedup of repeated identical calls; file coherence for file-based tools (`read_file`/`write_file`/`edit_file`) via path+mtime+size; MCP-sourced entries expire after a TTL while non-MCP entries never expire; meta-tools (`finish`, `tool_result_read`, `store_fact`, …) are excluded by default and additional names can be added via `AddNonCacheableTools`.

Cache mode selection: `read_file` is file-backed by default (streamed from disk) while mutation tools are content-backed. A read tool opts into content-backed caching when `agent.ToolExecutor.CacheStrategy` returns `tools.CacheModeContentBacked` (the tool implements `tools.ContentBackedReader`); the file coherence metadata (path+mtime+size) is still attached so source-file changes are detected, but the result is stored in memory and `tool_result_read` paginates it rather than re-reading raw bytes from disk.

### Batch meta-tool

The `batch` tool lets the model dispatch multiple tool calls in one turn. It is intercepted by the executor before reaching the registry; its own `Execute()` returns an error. Sub-calls go through the full policy + truncation + caching pipeline, are emitted with a `(batched)` suffix, and per-sub-call errors do not abort the batch.

### Per-tool-call ceiling and watchdog

`WithToolCallTimeout` / `SetToolCallTimeout` bound a **single** tool call. `executeToolCall`:

- runs `e.tools.Execute` on a detached, **panic-recovering** goroutine (a tool panic is contained and delivered as the call's error rather than aborting the process);
- drains an already-completed result *before* racing the context, so a finished tool always wins over a simultaneously-ready `ctx.Done()`, pause-tick, or timeout arm;
- abandons the WAIT — cancelling the tool's derived call context (a cooperative request, so an uninterruptible tool may still finish; treat its side effects as having possibly occurred) — when the ceiling fires, returning an error wrapping `ErrToolTimeout` (the dispatch site wraps it with the tool name: `tool "name": tool call timed out`);
- polls the `PauseChecker` on the executor's `toolWatchdogInterval` ticker (default `defaultToolWatchdogInterval`, 250 ms) for as long as a tool is in flight, so a cooperative pause is observed *mid-call*.

Zero (the default) disables the ceiling. `SetToolCallTimeoutExempt(names...)` names the interactive/long-running orchestration tools (e.g. `ask_user`, `declare_plan`, `delegate`, `execute_plan`) that block on a human or on sub-work by design and are therefore never bounded.

## Error Handling

- **Fatal LLM/tool error**: `Run` returns a non-nil error.
- **Context cancelled**: propagated immediately, no retry.
- **Cooperative pause**: `Run` returns `ErrPaused` plus the trajectory accumulated through the previous completed boundary; pending post-edit verification is flushed before the checkpoint is returned.
- **Tool call ceiling exceeded**: `Run` returns an error wrapping `ErrToolTimeout` (naming the tool). This arm returns no result, so a pending post-edit verification is deliberately NOT flushed here — with no consumer the note would be discarded, so running the command would be wasted work. Exempt tools (see `SetToolCallTimeoutExempt`) are never bounded.
- **Verification command failure/timeout**: represented in a `[verify_on_edit]` observation and does not become a `Run` error.
- **Budget exhausted without finish**: `Finished: false`, treated as incomplete (not an error).
- **Tool not found / parse failure**: surfaced as `ToolResult{IsError: true}`, not a Go error.

## Invariants

- The `finish` tool is always available in every run (appended automatically if absent).
- A single `Executor` instance is never used concurrently — parallel callers create one per step.
- When `WithResumeSteps` supplies prior steps, the step counter starts at `len(steps)+1` and the full trajectory (seeded plus new steps) is synced to the `TrajectoryStore`; the resumed steps are counted against the shared `maxSteps` budget.
- A pending resume interjection remains the final user message across every reactive-compaction retry and is consumed only after a successful LLM response.
- The step-boundary order is cancellation, pause check, then one live user-message poll; a pause never drains the host queue.
- Verify-on-edit runs once per dirty response group, only after a successful `write_file`/`edit_file`, and every pending run is flushed before finish or pause returns control.
- When `mutationRequired` is set, finish without a prior successful mutating tool is rejected (nudge then `Finished: false`).
- Both checklist sub-gates are soft: after one nudge attempt, finish is accepted regardless.
- Tool results from untrusted sources are wrapped in `<untrusted-content>` tags before becoming an LLM message (when injection defense is enabled on the `ContextManager`).
- Every `Step` carries `IsUntrusted` (set after tool execution via `tool.IsUntrusted()` or MCP source check) and `CacheHash` (empty for a non-cacheable tool that was not truncated).
- No tool — cacheable or non-cacheable — is ever truncated without a retrieval path. When a non-cacheable tool's result is truncated at Stage 1 (line/byte) or Stage 2 (token budget), the executor caches it on demand (`ToolResultCache.Store`) and embeds the resulting hash in the truncation notice so the model can re-read the full result via `tool_result_read`. Only small non-truncated non-cacheable results stay out of the cache.
- `batch` is intercepted before the registry; its sub-calls are cached individually.
- Streaming is off by default; when enabled it installs `ChatRequest.DeltaSink`, forwards text deltas to `AssistantChunk`, and emits `AssistantDone` exactly once per implicit-finish response (a tool-call turn delivers streamed deltas without a terminator) — a provider that streams nothing still yields one full-text `AssistantChunk`.

## Related Specs

- [README.md](README.md) — orchestration overview
- [conductor.md](conductor.md) — the top-level Executor caller
- [subagents.md](subagents.md) — isolated Executor instances in goroutines
- [reflector.md](reflector.md) — reads the trajectory via `TrajectoryStore`
- [../oneshot.md](../oneshot.md) — the one-shot service client repeating this nudge convention for single structured calls
- [../memory/compaction.md](../memory/compaction.md) — compaction strategies driving the per-iteration fill check
- [../tool-system/README.md](../tool-system/README.md) — tool execution pipeline and trust classification
