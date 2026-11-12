# Subagents

The SDK supports parallel plan execution through **subagents** — goroutine-based wrappers around an `Executor` that run independent plan steps concurrently. This document covers the `SubAgentTask` bundle, the `RunSubAgent` launcher, parallel execution, context injection, event emission, and the `TrajectoryStore` used to capture executor steps.

```go
import (
    "context"

    "github.com/v0lka/sp4rk/agent"
    "github.com/v0lka/sp4rk/tools"
)
```

## Overview

A plan is a DAG of steps. Steps with no dependencies between them can run in parallel to reduce total latency. The SDK achieves this by giving each independent step its own `Executor` (since an `Executor` is not safe for concurrent use on a single instance) and launching it in a goroutine via `RunSubAgent`. Each subagent runs to completion and reports its result on a channel.

## SubAgentTask

`SubAgentTask` bundles everything a subagent needs to run a step: the executor, its context manager, its task tools, a description, an event emitter, and an optional checklist callback.

```go
type SubAgentTask struct {
    StepID         string
    Executor       *Executor
    CM             ContextManager
    TaskTools      []tools.ToolDescriptor
    TaskDesc       string             // task description (for SubAgentLaunch event)
    Model          string             // forced model identity, "" = host default (for SubAgentLaunch event)
    Emitter        Events        // event emitter (nil-safe)
    TodoUpdateFunc StepTodoUpdateFunc // optional callback for update_checklist tool
}
```

| Field | Description |
|-------|-------------|
| `StepID` | The plan-step identifier. Injected into the context so context-aware tools know the current step. |
| `Executor` | A fresh `Executor` for this step. Must not be shared across concurrent subagents. |
| `CM` | The `ContextManager` owning this step's prompt history and compaction. |
| `TaskTools` | The tools available to this step. A registered `finish` tool is not required: the executor appends the `finish` definition to every LLM request at call time, so completion works even when `TaskTools` has no `finish` entry (registering it only makes it visible in registry listings and group-filtered toolsets). |
| `TaskDesc` | A human-readable task description, emitted in the `SubAgentLaunch` event. |
| `Model` | The model identity the sub-agent is pinned to (`""` = host default), emitted in the `SubAgentLaunch` event. |
| `Emitter` | The `Events` sink for this subagent. `nil` is replaced with `NoopEvents`. |
| `TodoUpdateFunc` | Optional callback invoked when the step's checklist is updated. Wired into the context for the `update_checklist` tool. |

## RunSubAgent

`RunSubAgent` starts the executor in a goroutine and returns a channel for the result. It is the primary entry point for launching a single subagent.

```go
func RunSubAgent(
    ctx context.Context,
    stepID string,
    executor *Executor,
    cm ContextManager,
    taskTools []tools.ToolDescriptor,
    taskDesc string,
    model string,
    emitter Events,
    todoUpdateFunc StepTodoUpdateFunc,
) (resultCh <-chan SubAgentResult)
```

The returned channel is buffered (capacity 1) and closed when the goroutine finishes. Read exactly one `SubAgentResult` from it.

### What RunSubAgent does

1. **Emits `SubAgentLaunch`** — `emitter.SubAgentLaunch(stepID, taskDesc, model)` fires immediately, before the executor starts.
2. **Injects context** — the task description, step ID, and (if provided) the checklist callback are attached to the context (see [Context injection](#context-injection)).
3. **Runs the executor** — `executor.Run(ctx, taskTools, cm)` is called inside the goroutine.
4. **Defense-in-depth check** — even if the executor reports `Finished: true`, the output is scanned for printed tool-call syntax (a failure mode where the model writes a fenced code block with a tool-name tag instead of emitting a `tool_use` block). If detected, the result is treated as a failure.
5. **Emits `SubAgentComplete`** — `emitter.SubAgentComplete(stepID, success, duration, errMsg)` fires with the outcome, wall-clock duration, and the failure reason (`""` on success). A cooperative pause (`agent.ErrPaused`) emits `SubAgentPaused(stepID, duration)` **instead**.
6. **Sends the result** — a `SubAgentResult` is sent on the channel.
7. **Recovers panics** — the goroutine body is wrapped in a deferred `recover()` registered *before* `SubAgentLaunch` fires, so a panic anywhere in subagent execution cannot tear down the host process (execution runs on this goroutine, which the host cannot wrap with its own `recover`). The panic is logged as `"subagent execution panicked"` with `step_id`, `panic`, and a `stack` from `debug.Stack()`, through the **executor's** logger (a discard handler when `executor` is nil, so the handler cannot itself panic), then reported as a failure: `emitter.SubAgentComplete(stepID, false, time.Since(startTime), panicErr.Error())` fires and the channel receives `SubAgentResult{StepID: stepID, Error: panicErr}`, where `panicErr` is `fmt.Errorf("subagent %s panicked: %v", stepID, r)`. `Output` and `Steps` are empty on this path, and the remaining steps above are skipped — the recovered send is the only value the channel carries before it is closed.

### Context cancellation

The goroutine respects context cancellation. Because the executor's LLM calls and tool executions use the same context, cancelling `ctx` causes `executor.Run` to return promptly. The result channel is always closed (via `defer`), so receivers never block forever.

```go
ctx, cancel := context.WithCancel(context.Background())
defer cancel()

resultCh := agent.RunSubAgent(ctx, "step_2", exec, cm, taskTools, "Refactor auth module", "", events, nil)

// Later, if you need to abort:
cancel()

_ = <-resultCh // always returns, even after cancellation
```

## SubAgentResult

`SubAgentResult` is the outcome of a subagent run.

```go
type SubAgentResult struct {
    StepID string
    Output string
    Error  error
    Steps  []Step // actual executor steps (tool calls + observations)
    // Summary carries the model's own final text when the subagent terminated
    // on a host-designated stop tool (see Executor.SetStopTools) rather than the
    // inline finish tool. Mirrors ExecutorResult.Summary; empty otherwise.
    Summary string
}
```

| Field | Description |
|-------|-------------|
| `StepID` | The plan-step identifier this result corresponds to. |
| `Output` | Depends on the outcome. **Success** — the `finish` answer (or, when the run ended on a host-designated stop tool, that tool's short confirmation, with the model's prose in `Summary`). **Hard error** (executor errored / cancelled, or the tool-call-syntax guard fired) — **empty**; the result carries only `StepID`, `Steps` and `Error`. **Did not finish** — the executor's own partial output, while the abort reason goes into `Error`. |
| `Error` | Non-nil if the executor errored, was cancelled, did not finish, leaked tool-call syntax, or panicked. |
| `Steps` | The full sequence of executor steps (tool calls + observations). Empty when the executor returned no result. |
| `Summary` | The model's own final text when the subagent terminated on a host-designated stop tool (`Executor.SetStopTools`) rather than the inline `finish` tool — `Output` then holds only the stop tool's short confirmation. Empty for runs that did not end on a stop tool. |

### Success vs. failure semantics

A subagent is considered **successful** only when `executor.Run` returns no error *and* `result.Finished == true`. Four failure cases produce an `Error`:

- **Executor error** — the LLM call or a tool returned a fatal error, or the context was cancelled. `Output` is empty; `Steps` is populated from the partial result when available.
- **Did not finish** — the step budget was exhausted or a circuit breaker aborted. The error message is the executor's structured `AbortReason` when set, otherwise its `Output` (a short `"Aborted: …"` message for the circuit-breaker / fruitless branches), otherwise a generic `"step execution did not complete within max steps"`. `Output` itself still carries the executor's partial output.
- **Leaked tool-call syntax** — the [defense-in-depth guard](#defense-in-depth-detecttoolcallsyntaxincontent) turned an otherwise-successful run into an error. Like the executor-error case it takes the `err != nil` return path, so the result carries only `StepID`, `Steps` and `Error` — `Output` and `Summary` are **empty** even though the executor did finish.
- **Panic** — the [panic-recovery guard](#what-runsubagent-does) caught a panic inside subagent execution. `Error` is `"subagent <stepID> panicked: <value>"` and both `Output` and `Steps` are empty.

```go
result := <-resultCh
if result.Error != nil {
    log.Printf("step %s failed: %v", result.StepID, result.Error)
    return
}
log.Printf("step %s done: %s", result.StepID, result.Output)
```

## Context injection

`RunSubAgent` attaches several values to the context before calling `executor.Run`. These are read by context-aware tools during execution.

| Helper | Purpose |
|--------|---------|
| `tools.WithTaskContext(ctx, taskDesc)` | Attaches the task description so tools know what the agent is working on. |
| `agent.WithStepID(ctx, stepID)` | Attaches the plan-step ID. Used by the file tracker and other context-aware tools. |
| `agent.WithStepTodoUpdateFunc(ctx, fn)` | Attaches the checklist callback so the `update_checklist` tool can emit to-do updates. |

These are the same helpers used in standalone executor runs (see [Agent Executor](agent-executor.md#context-helpers)); `RunSubAgent` simply wires them automatically.

## Event emission

Subagents emit sub-agent lifecycle events through the provided `Events` sink:

- **`SubAgentLaunch(stepID, description, model)`** — fired when the goroutine starts, before the executor runs.
- **`SubAgentComplete(stepID, success, duration, errMsg)`** — fired when the goroutine finishes, with the success flag, wall-clock duration, and the failure reason (`""` on success). On the non-error paths the `SubAgentResult` also carries `Summary` (the model's final text when the run ended on a host-designated stop tool); it is empty when the run returned an error.
- **`SubAgentPaused(stepID, duration)`** — fired instead of `SubAgentComplete` when the subagent stopped at a cooperative pause checkpoint (`agent.ErrPaused`). A pause is a recoverable checkpoint, not a failure: the trajectory is preserved in the `SubAgentResult` for a later resume. So `SubAgentComplete` does **not** always fire — do not pair launch/complete counters without accounting for paused runs. On a paused run the result carries `StepID`, `Steps` (the resumable trajectory) and `Error` = `agent.ErrPaused`; `Output` and `Summary` are empty. Detect the pause with `errors.Is(result.Error, agent.ErrPaused)`.

`SubAgentComplete` with `success == false` has **four** distinct trigger paths, not one: a hard executor error (an LLM call or tool failed, or the context was cancelled); a run that never finished (step budget exhausted, circuit-breaker abort, mutation gate); the [tool-call-syntax defense guard](#defense-in-depth-detecttoolcallsyntaxincontent) — an otherwise-successful run whose `Output`/`Summary` leaks tool-call syntax is forced to `success = false` and gets `errMsg` `"model printed tool-call syntax as text instead of using tool_use blocks"`; and the [panic-recovery guard](#what-runsubagent-does) — a panic inside subagent execution is recovered, logged with a stack trace, and surfaced as `SubAgentComplete(stepID, false, …)` whose `errMsg` is `"subagent <stepID> panicked: <value>"`. On the normal completion path only a cooperative pause diverts to `SubAgentPaused`; the panic guard always reports a failure.

If `emitter` is `nil`, `RunSubAgent` substitutes `NoopEvents`, so the events are silently dropped. See [Events](events.md#sub-agent-events) for the full method signatures.

## Defense-in-depth: DetectToolCallSyntaxInContent

A known LLM failure mode is *printing* a tool invocation as prose — writing a fenced code block with a tool-name language tag (e.g. ` ```bash_exec `), or the JSON form of the call (e.g. `{"answer": "..."}` for the finish tool, or `{"name": "...", "arguments": {...}}`), instead of emitting a proper `tool_use` block. The executor's implicit-finish detector should catch this and abort with `Finished: false`, but `RunSubAgent` adds a second guard as defense-in-depth. It probes both `Output` (a normal finish) and `Summary` (a stop-tool termination, whose `Output` is only the stop tool's short confirmation while the model's prose is carried in `Summary`):

```go
if success && (DetectToolCallSyntaxInContent(result.Output) || DetectToolCallSyntaxInContent(result.Summary)) {
    success = false
    if err == nil {
        err = errors.New("model printed tool-call syntax as text instead of using tool_use blocks")
    }
}
```

`DetectToolCallSyntaxInContent` reports whether content contains tool-call syntax printed as text. It is an exported function so callers can use it independently:

```go
if agent.DetectToolCallSyntaxInContent(output) {
    // the model wrote a tool call as text — not a real finish
}
```

> **Caveat:** the JSON half of the detector is a heuristic on the *shape* of the output, so a genuine answer deliberately shaped like a leaked call — a lone `{"answer": "<text>"}` object, or a `{"name": …, "arguments": {…}}` / `{"tool": …, "args": {…}}` envelope, optionally padded with the service keys `id`/`type`/`index` (e.g. Anthropic's `{"type":"tool_use","id":…,"name":…,"input":{…}}`) — is treated as a leak. A task that instructs the model to emit exactly such an object as its final answer can therefore have its run reported as a failure. A multi-field answer that carries an unrelated extra key (e.g. `{"name":…,"arguments":…,"summary":…}`) is not affected.

## Parallel plan execution

`RunSubAgentsParallel` runs multiple `SubAgentTask`s concurrently and collects all results. It is the convenience entry point for executing independent DAG steps in parallel.

```go
func RunSubAgentsParallel(ctx context.Context, agents []SubAgentTask, opts ...RunSubAgentsParallelOption) []SubAgentResult
```

Results are collected in input order (not completion order); a slow agent blocks all subsequent results from being returned. The function blocks until all subagents have reported. If `agents` is empty, it returns `nil`.

By default every subagent is launched at once (unbounded fan-out). Pass `agent.WithMaxParallelSubagents(n)` to bound how many run concurrently *within the call* — subagents beyond the cap are queued and started as slots free up, with the result set and its input order unchanged. The cap is **per-invocation**, not process-wide: two concurrent `RunSubAgentsParallel` calls (or a nested delegation) each apply their own limit, so a host that needs a global cap must enforce it itself.

```go
tasks := []agent.SubAgentTask{
    {StepID: "step_2", Executor: execA, CM: cmA, TaskTools: toolsA, TaskDesc: "Write tests", Emitter: events},
    {StepID: "step_3", Executor: execB, CM: cmB, TaskTools: toolsB, TaskDesc: "Update docs", Emitter: events},
    {StepID: "step_4", Executor: execC, CM: cmC, TaskTools: toolsC, TaskDesc: "Refactor API", Emitter: events},
}

results := agent.RunSubAgentsParallel(ctx, tasks)
for _, r := range results {
    if r.Error != nil {
        log.Printf("%s failed: %v", r.StepID, r.Error)
    } else {
        log.Printf("%s done: %s", r.StepID, r.Output)
    }
}
```

> **Note:** each `SubAgentTask` must have its own `Executor` and `ContextManager`. Sharing a single `Executor` across concurrent tasks violates the executor's single-execution invariant. The orchestration layer creates a fresh executor and context manager per step.

### Manual parallel execution

For finer control (e.g. streaming results as they arrive), launch subagents individually and select on their channels:

```go
channels := make([]<-chan agent.SubAgentResult, len(tasks))
for i, t := range tasks {
    channels[i] = agent.RunSubAgent(ctx, t.StepID, t.Executor, t.CM,
        t.TaskTools, t.TaskDesc, t.Model, t.Emitter, t.TodoUpdateFunc)
}

// Collect results as they complete.
for _, ch := range channels {
    r := <-ch
    handleResult(r)
}
```

## TrajectoryStore

`TrajectoryStore` captures the executor's step history so tools (e.g. a reflector) can access the trajectory during execution. The executor syncs its steps to the store at each loop iteration.

```go
type TrajectoryStore interface {
    Sync(steps []Step)
    Steps() []Step
}
```

Inject a store into the context before running a subagent (or any executor):

```go
type trajStore struct {
    mu    sync.Mutex
    steps []agent.Step
}

func (s *trajStore) Sync(steps []agent.Step) { s.mu.Lock(); s.steps = steps; s.mu.Unlock() }
func (s *trajStore) Steps() []agent.Step     { s.mu.Lock(); defer s.mu.Unlock(); return s.steps }

store := &trajStore{}
ctx = agent.WithTrajectoryStore(ctx, store)

resultCh := agent.RunSubAgent(ctx, "step_2", exec, cm, taskTools, "Investigate bug", "", events, nil)
<-resultCh

// The store now holds the full trajectory, synced at each iteration.
trajectory := store.Steps()
log.Printf("captured %d steps", len(trajectory))
```

Retrieve the store from inside a tool via the context:

```go
if ts := agent.TrajectoryStoreFromContext(ctx); ts != nil {
    for _, step := range ts.Steps() {
        // inspect thought, action, observation...
    }
}
```

`TrajectoryStore` is particularly useful in plan-and-reflect orchestration: the orchestrator hands the captured trajectory (`store.Steps()`) to `Reflector.Reflect` as an explicit argument — the reflector analyzes the trajectory it is given and does not fetch the store from the context itself. See [Events](events.md#orchestrationevents) for the `OnReflected` hook.

## Subagent Profiles

A subagent may be launched under a named **Subagent Profile** (an `AGENT.md`-declared persona; see [Subagent Profiles](agents.md)). A profile is discovered and parsed by `agents.AgentManager` (mirroring `skills.SkillManager`); the execution layer resolves the profile name — set on a plan step via `PlanStep.Agent`, or chosen directly by a host delegation tool — to an `*agents.Agent` and applies its fields to the launched subagent:

| Profile field | Applied to the subagent as |
| --- | --- |
| `Agent.Body` | the core directive / system prompt (replaces the generic orchestrator default) |
| `Agent.Metadata.Tools` (`ToolPreference`) | the tool budget (`nil`=all, `"read-only"`, or a comma-list of tool-group tokens). Resolve it with `ToolPreferenceWithError()`, which returns an error for an invalid field; the legacy `ToolPreference() any` silently maps an invalid field to `nil` (the full toolset) and must not be used for security-sensitive resolution |
| `Agent.Metadata.MaxSteps` | the ReAct iteration cap (0/absent → derived from task complexity) |
| `Agent.Metadata.Model` | forced per-call via `NewModelProviderOverrideCaller` (provider+model pinning); the legacy model-only `NewModelOverrideCaller` is kept |
| `Agent.Metadata.AllowRedelegate` | whether the subagent may launch further subagents |
| `Agent.Metadata.Hidden` | discovery visibility only (never blocks invocation) |

Only `Body` (alongside `DirPath`) is a field of `agents.Agent` itself — the frontmatter fields (`Tools`, `MaxSteps`, `Model`, `AllowRedelegate`, `Hidden`, …) live on `agents.AgentMetadata`, reached through `Agent.Metadata`.

`agent.NewModelProviderOverrideCaller(inner LLMCaller, provider, model string)` wraps an `LLMCaller` so every `Call` pins `req.Provider` and/or `req.Model` before delegating to `inner`; each override applies independently and only when non-empty, and both empty returns `inner` unchanged, so the override applies conditionally without callers branching. It relies on the [LLM router contract](llm-providers.md) that `req.Model` is filled with the active model only when empty — setting it beforehand bypasses the router's active-model selection for that caller — and that a non-empty `req.Provider` routes the call to the named configured provider instead of the active one (unknown name: immediate non-retryable error, zero attempts; see [ADR-011](../specs/decisions/011-per-request-provider-override.md)). A pinned caller is immune to a mid-task `SetModel`. `agent.NewModelOverrideCaller(inner LLMCaller, model string)` is the legacy model-only form: it sets `req.Model` and never touches provider routing (the request still routes to the active provider).

Profile application is an execution-layer concern: `RunSubAgent`/`RunSubAgentsParallel` themselves are profile-agnostic primitives. The **host's** execution layer resolves the profile and constructs the `Executor`, `ContextManager`, `TaskTools`, and caller before handing them to `RunSubAgent` as a `SubAgentTask` — the SDK ships no resolver, and `orchestration.Conductor` passes `PlanStep.Agent` through untouched.

## Putting it together

A complete parallel execution with **per-step** trajectory capture.

`RunSubAgentsParallel` takes a single `ctx`, and `SubAgentTask` has no per-task context field — so every subagent it launches sees the *same* `TrajectoryStore`. Because the executor replaces the store's whole contents on each sync (`ts.Sync(state.allSteps)`), concurrent subagents sharing one store clobber each other. **Per-step trajectory capture is therefore not achievable through the parallel helper.** To capture each step's trajectory separately, launch the subagents manually with a per-task context:

```go
// One trajectory store per step, attached to that step's own context.
storeA, storeB := &trajStore{}, &trajStore{}
ctxA := agent.WithTrajectoryStore(ctx, storeA)
ctxB := agent.WithTrajectoryStore(ctx, storeB)

chA := agent.RunSubAgent(ctxA, "step_2", newExecutorForStep("step_2"),
    newContextManagerForStep("step_2"), taskTools, "Implement feature A", "", events, nil)
chB := agent.RunSubAgent(ctxB, "step_3", newExecutorForStep("step_3"),
    newContextManagerForStep("step_3"), taskTools, "Implement feature B", "", events, nil)

// Collect in launch order; select on the channels instead to stream arrivals.
for _, r := range []agent.SubAgentResult{<-chA, <-chB} {
    if r.Error != nil {
        log.Printf("step %s failed: %v (output: %s)", r.StepID, r.Error, r.Output)
        continue
    }
    log.Printf("step %s succeeded: %s", r.StepID, r.Output)
}

// Each store holds only its own step's trajectory.
log.Printf("step_2 trajectory: %d steps\n", len(storeA.Steps()))
log.Printf("step_3 trajectory: %d steps\n", len(storeB.Steps()))
```

> **Note:** `RunSubAgentsParallel(ctx, tasks, opts...)` is still the right choice when you do *not* need per-step trajectories — a single shared store then simply holds whichever subagent synced last. It also offers `agent.WithMaxParallelSubagents(n)` to bound fan-out; the manual loop above launches every subagent at once and has no such cap.

## See also

- [Subagent Profiles](agents.md) — `AGENT.md`-declared personas that configure a launched subagent
- [Planner](planner.md) — `PlanStep.Agent` targets a named profile
- [LLM providers](llm-providers.md) — the model-override router contract `NewModelOverrideCaller` relies on
- [Agent executor](agent-executor.md) — the ReAct loop each subagent runs
