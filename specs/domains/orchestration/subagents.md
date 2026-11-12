# Subagents

## Role

The delegated-execution primitive: goroutine-based wrappers around an `Executor` that run independent steps concurrently. The SDK exposes `RunSubAgent` (launch one isolated `Executor.Run` in a goroutine) and `RunSubAgentsParallel` (run several concurrently and collect results). Subagents power parallel plan execution and any host-defined delegation mechanism. This document describes the **primitive** the SDK exposes — host-built delegation tools (`delegate`, `cancel_delegation`, etc.) are constructed atop it.

## Key Files

- `github.com/v0lka/sp4rk/agent` — `SubAgentTask`, `RunSubAgent`, `RunSubAgentsParallel`, `SubAgentResult`, `DetectToolCallSyntaxInContent`, `StepTodoUpdateFunc`, context helpers (`WithStepID`, `WithStepTodoUpdateFunc`)
- `github.com/v0lka/sp4rk/agent` — `Executor`, `ContextManager`, `Events` (`SubAgentLaunch`/`SubAgentComplete` hooks), `NewModelProviderOverrideCaller` (provider+model pinning; the legacy `NewModelOverrideCaller` is its model-only form)
- `github.com/v0lka/sp4rk/tools` — `ToolDescriptor`, `WithTaskContext`
- `github.com/v0lka/sp4rk/orchestration` — `CompletedStep`, DAG utilities (`FindReadySteps`) used to select independent steps, `PlanStep.Agent` (names a Subagent Profile that should execute a step)
- `github.com/v0lka/sp4rk/agents` — `AgentManager`, `Agent` profile discovery/parsing (resolves `PlanStep.Agent` to a profile)

## Behavior

A plan is a DAG of steps. Steps with no dependencies between them can run in parallel to reduce total latency. The SDK achieves this by giving each independent step its own `Executor` (an `Executor` is not safe for concurrent use on a single instance) and launching it in a goroutine via `RunSubAgent`. Each subagent runs to completion and reports its result on a channel.

### SubAgentTask

```go
type SubAgentTask struct {
    StepID         string
    Executor       *Executor          // fresh executor for this step; never shared
    CM             ContextManager     // isolated prompt history + compaction
    TaskTools      []tools.ToolDescriptor
    TaskDesc       string             // emitted in the SubAgentLaunch event
    Emitter        Events             // nil-safe (replaced with NoopEvents)
    TodoUpdateFunc StepTodoUpdateFunc // optional update_checklist callback
}
```

### RunSubAgent

```go
func RunSubAgent(ctx, stepID string, executor *Executor, cm ContextManager,
    taskTools []tools.ToolDescriptor, taskDesc, model string, emitter Events,
    todoUpdateFunc StepTodoUpdateFunc) (resultCh <-chan SubAgentResult)
```

The returned channel is buffered (capacity 1) and closed when the goroutine finishes. Read exactly one `SubAgentResult`. What it does:

1. **Emits `SubAgentLaunch`** — `emitter.SubAgentLaunch(stepID, taskDesc, model)` fires before the executor starts.
2. **Injects context** — task description (`WithTaskContext`), step ID (`WithStepID`), and (if provided) the checklist callback (`WithStepTodoUpdateFunc`).
3. **Runs the executor** — `executor.Run(ctx, taskTools, cm)` inside the goroutine.
4. **Defense-in-depth check** — even if the executor reports `Finished: true`, the output is scanned for printed tool-call syntax; if detected the result is treated as a failure.
5. **Emits `SubAgentComplete`** — with the outcome, wall-clock duration, and the failure reason (`""` on success).
6. **Sends the result** on the channel.

### Context cancellation

The goroutine respects context cancellation. Because the executor's LLM calls and tool executions use the same context, cancelling `ctx` causes `executor.Run` to return promptly. The result channel is always closed via `defer`, so receivers never block forever. The host expresses delegation cancellation by cancelling the subagent context (typically a child of a parent context).

### SubAgentResult & success semantics

```go
type SubAgentResult struct {
    StepID  string
    Output  string
    Error   error
    Steps   []Step
    Summary string // model's final text on stop-tool termination (mirrors ExecutorResult.Summary)
}
```

A subagent is **successful** only when `executor.Run` returns no error *and* `result.Finished == true`. Two failure cases produce an `Error`:

- **Executor error** — the LLM call or a tool returned a fatal error, or the context was cancelled. `Steps` is populated from the partial result when available.
- **Did not finish** — the step budget was exhausted or a circuit breaker aborted. The failure reason prefers the executor's structured `AbortReason`, then falls back to `Output` (a short `Aborted: …` message for the circuit-breaker / fruitless / max-steps branches), then to the generic `step execution did not complete within max steps` — a mutation-gate prose answer is never surfaced as the failure cause.

### RunSubAgentsParallel

```go
func RunSubAgentsParallel(ctx context.Context, agents []SubAgentTask,
    opts ...RunSubAgentsParallelOption) []SubAgentResult
```

Runs multiple `SubAgentTask`s concurrently and collects all results. Results are collected in **input order** (not completion order); a slow agent blocks subsequent results from being returned. Blocks until all subagents have reported. Returns `nil` for an empty slice. By default every subagent launches at once (unbounded fan-out). `WithMaxParallelSubagents(n)` caps how many run concurrently within a single invocation: subagents beyond the cap are queued and started as running slots free up, while the result set and its input order stay unchanged (`n <= 0` keeps the unbounded default). The cap is per-invocation, not process-wide — a host firing several `RunSubAgentsParallel` calls (or nesting a delegation) enforces any global limit itself. For streaming results as they arrive, launch subagents individually and select on their channels.

### Defense-in-depth

`DetectToolCallSyntaxInContent` reports whether content contains tool-call syntax printed as text (a fenced code block with a tool-name language tag, or a lone JSON tool-call envelope such as `{"answer": "..."}` or `{"name": "...", "arguments": {...}}`, optionally padded with the service keys `id`/`type`/`index`). `RunSubAgent` applies it as a second guard after the executor's own implicit-finish detector, probing both `Output` and `Summary` (`Summary` carries the model's prose on a stop-tool termination, where `Output` is only the stop tool's short confirmation):

```go
if success && (DetectToolCallSyntaxInContent(result.Output) || DetectToolCallSyntaxInContent(result.Summary)) {
    success = false
    err = errors.New("model printed tool-call syntax as text instead of using tool_use blocks")
}
```

The function is exported so callers can use it independently.

### Subagent Profiles

A subagent may be launched under a named **Subagent Profile** (an `AGENT.md`-declared persona; see [../agents.md](../agents.md)). A profile is discovered and parsed by `agents.AgentManager` (mirroring `skills.SkillManager`); the execution layer resolves the profile name — set on a plan step via `PlanStep.Agent`, or chosen directly by a host delegation tool — to an `*agents.Agent` and applies its fields to the launched subagent:

| Profile field | Applied to the subagent as |
| ------------- | -------------------------- |
| `Body` | the core directive / system prompt (replaces the generic orchestrator default) |
| `Tools` (`ToolPreference`) | the tool budget (`nil`=all, `"read-only"`, or a comma-list of tool-group tokens). Resolve it with `ToolPreferenceWithError()`, which returns an error for an invalid field; the legacy `ToolPreference() any` silently maps an invalid field to `nil` (the full toolset) and must not be used for security-sensitive resolution |
| `MaxSteps` | the ReAct iteration cap (0/absent → derived from task complexity) |
| `Model` | forced per-call via `NewModelProviderOverrideCaller` (provider+model pinning); the legacy model-only `NewModelOverrideCaller` is kept |
| `AllowRedelegate` | whether the subagent may launch further subagents |
| `Hidden` | discovery visibility only (never blocks invocation) |

`NewModelProviderOverrideCaller(inner LLMCaller, provider, model string)` wraps an `LLMCaller` so every `Call` pins `req.Provider` and/or `req.Model` before delegating to `inner`; each override applies independently and only when non-empty, and both empty returns `inner` unchanged, so the override applies conditionally without callers branching. It relies on the [LLM router contract](../../contracts/llm-providers.md) that `req.Model` is filled with the active model only when empty — setting it beforehand bypasses the router's active-model selection for that caller — and that a non-empty `req.Provider` routes the call to the named configured provider instead of the active one (unknown name: immediate non-retryable error, zero attempts; see [ADR-011](../../decisions/011-per-request-provider-override.md)). A pinned caller is immune to a mid-task `SetModel`. `NewModelOverrideCaller(inner LLMCaller, model string)` is the legacy model-only form: it sets `req.Model` and never touches provider routing (the request still routes to the active provider).

Profile application is an execution-layer concern: `RunSubAgent`/`RunSubAgentsParallel` themselves are profile-agnostic primitives. The host or Conductor resolves the profile and constructs the `Executor`, `ContextManager`, `TaskTools`, and caller before handing them to `RunSubAgent` as a `SubAgentTask`.

## Error Handling

- **Executor fatal error**: `Error` is set; `Steps` carries the partial trajectory when available.
- **Budget exhausted / circuit-breaker abort**: `Finished == false`; the abort reason is wrapped into `Error`.
- **Context cancelled** (parent cancellation or explicit cancel): the goroutine returns promptly; `Error` reflects the cancellation.
- **Subagent panic**: contained by the launch goroutine's `recover`, logged with the stack, emitted as `SubAgentComplete(stepID, false, duration, panicMsg)`, and delivered as an `Error` result — the step fails in isolation and execution continues.

## Invariants

- Each `SubAgentTask` has its own `Executor` and `ContextManager` — sharing a single `Executor` across concurrent tasks violates the executor's single-execution invariant.
- The result channel is always closed; exactly one `SubAgentResult` is sent.
- A subagent never shares its `ContextManager` with its launcher or with other subagents.
- `SubAgentLaunch` always fires before the executor runs. `SubAgentComplete` fires after on completion, failure, or cancellation, carrying the failure reason in its `errMsg` argument (`""` on success) — **except** on the cooperative-pause path, where `SubAgentPaused` fires instead (a pause is a recoverable checkpoint, not a termination).
- `success` requires `result.Finished` and that neither `result.Output` nor `result.Summary` matches `DetectToolCallSyntaxInContent`.

## Related Specs

- [README.md](README.md) — orchestration overview
- [executor.md](executor.md) — the ReAct loop primitive each subagent runs
- [conductor.md](conductor.md) — the top-level caller that a host's delegation tool launches subagents from
- [../agents.md](../agents.md) — Subagent Profiles that configure a launched subagent
- [../memory/README.md](../memory/README.md) — isolated context per subagent
