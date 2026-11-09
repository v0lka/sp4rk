# Orchestration

The `orchestration` package provides the primitives that turn a single ReAct agent loop into a multi-step, self-correcting execution engine. It defines the shared state container (the **Blackboard**), the plan data model (a DAG of steps), the single-loop executor (**Conductor**), DAG traversal utilities, and a persistence layer for checkpointing state across restarts.

This document covers every exported type and function in the package. For the planning component that produces DAGs, see [planner.md](planner.md); for the self-correction component, see [reflector.md](reflector.md).

```go
import "github.com/v0lka/sp4rk/orchestration"
```

---

## Table of contents

- [Conductor](#conductor)
  - [ConductorConfig](#conductorconfig)
  - [NewConductor](#newconductor)
  - [Run](#run)
  - [Cleanup](#cleanup)
  - [SetReasoningEffort](#setreasoningeffort)
  - [SetModel](#setmodel)
  - [WithDelegationRegistry](#withdelegationregistry)
- [Interfaces](#interfaces)
  - [Planner](#planner)
  - [Reflector](#reflector)
  - [Events](#events)
  - [BlackboardFactory](#blackboardfactory)
  - [SystemPromptFactory](#systempromptfactory)
  - [ContextManagerFactory](#contextmanagerfactory)
  - [Optional ContextManager capabilities](#optional-contextmanager-capabilities)
  - [PruningOverride](#pruningoverride)
  - [StepScopable and RetryScopable](#stepscopable-and-retryscopable)
- [Resume with a user interjection](#resume-with-a-user-interjection)
- [Verify edits mechanically](#verify-edits-mechanically)
- [Blackboard](#blackboard)
  - [Blackboard interface](#blackboard-interface)
  - [MapBlackboard](#mapblackboard)
  - [GenerateSummary](#generatesummary)
  - [Fact memory](#fact-memory)
- [Plan data model](#plan-data-model)
  - [Plan](#plan)
  - [PlanStep](#planstep)
  - [CompletedStep](#completedstep)
  - [StepResult](#stepresult)
  - [PlanStepEvent](#planstepevent)
  - [BlackboardEntry](#blackboardentry)
- [Execution outcome](#execution-outcome)
  - [ExecutionStatus](#executionstatus)
  - [ExecutionResult](#executionresult)
  - [ErrExecutionIncomplete](#errexecutionincomplete)
- [Reflection](#reflection)
- [Fact](#fact)
- [Attachment](#attachment)
- [DAG utilities](#dag-utilities)
  - [FindReadySteps](#findreadysteps)
  - [BuildCarryForward](#buildcarryforward)
  - [BuildPlanExecutionSteps](#buildplanexecutionsteps)
  - [AggregateOutput](#aggregateoutput)
- [Persistence](#persistence)
  - [Checkpointer](#checkpointer)
  - [CheckpointedBlackboard](#checkpointedblackboard)
  - [RestoreBlackboard](#restoreblackboard)
  - [StepDumpTracker](#stepdumptracker)
- [Adapters](#adapters)
- [Complete Plan & Execute example](#complete-plan--execute-example)

---

## Conductor

The `Conductor` is the SDK-level primitive that runs a **single ReAct loop** owning one task end-to-end. It wraps the lower-level `agent.Executor` and wires in the blackboard-backed stores (step outputs, facts, attachments, final result) so that built-in tools such as `read_step_output`, `store_fact`, `search_facts`, `read_attachment`, and `read_final_result` can access shared state during execution.

A Conductor is reusable across steps: the system prompt factory receives the step description at `Run` time, so the same instance adapts to each step it executes.

```go
type Conductor struct {
    cfg ConductorConfig
    mu  sync.RWMutex
}
```

### ConductorConfig

`ConductorConfig` holds every dependency a Conductor needs. All fields are populated by the caller before `NewConductor`.

```go
type ConductorConfig struct {
    LLM               agent.LLMCaller
    Tools             agent.ToolExecutor
    ToolRegistry      *tools.ToolRegistry
    TokenCounter      llm.TokenCounter
    Model             string
    ModelRegistry     *llm.ModelRegistry
    ContextFactory    ContextManagerFactory
    SystemPrompt      SystemPromptFactory
    MaxSteps          int
    ToolResultBudget  agent.ToolResultBudget
    CircuitBreaker    agent.CircuitBreakerConfig
    HITLHandler       agent.HITLHandler
    ToolCache         *agent.ToolResultCache
    PerToolTruncation map[string]agent.ToolTruncationConfig
    ReasoningEffort   string
    PreWarningPercent int
    Streaming bool
    VerifyOnEdit agent.EditVerifyRunner
    VerifyOnEditMaxOutputChars int
    NonCacheableTools []string
    StopTools []string
    ConversationHistory []llm.Message
    ResumeSteps []agent.Step
    ContentBlocks []llm.ContentBlock
    PendingUserInterjection string
    PauseChecker func(context.Context) bool
    ToolCallTimeout time.Duration
    ToolCallTimeoutExemptTools []string
    UserMessageSource func(context.Context) string
    CompactOnStart bool
}
```

| Field | Purpose |
| --- | --- |
| `LLM` | The LLM caller used by the underlying executor. |
| `Tools` | The tool executor that dispatches tool calls. |
| `ToolRegistry` | Registry of available tool descriptors. |
| `TokenCounter` | Counts tokens for context-window management. |
| `Model` | Active model name; resolved against `ModelRegistry` for metadata. |
| `ModelRegistry` | Resolves model metadata (context window, output limit, tokenizer). The `Resolve` ok flag is discarded: whenever the resolved `ContextWindow` is `0` — an unknown model, a nil `ModelRegistry` (resolution skipped entirely), or a known model whose resolved window is zero — a usable fallback (`ContextWindow=128000`, `OutputLimit=4096`, `TokenizerType="approximate"`) is applied so compaction still works. |
| `ContextFactory` | **Required.** Creates a `ContextManager` for the run. |
| `SystemPrompt` | **Required.** Factory that builds the system prompt from the step description and model metadata. |
| `MaxSteps` | Per-run ReAct step budget. Defaults to `80` when zero. |
| `ToolResultBudget` | Limits how much tool-result text is retained in context. |
| `CircuitBreaker` | Detects repetitive/looping tool calls and aborts. |
| `HITLHandler` | Human-in-the-loop handler invoked on tool calls / step-limit extensions. |
| `ToolCache` | Optional content-addressed cache for tool results, keyed by a git-style short hash of `SHA256(toolName + "\x00" + content)` (or of the file's path+mtime+size for file-backed entries) — never by call arguments, so identical outputs dedupe to one entry while identical calls returning different outputs stay distinct. This hash is what backs `tool_result_read`. |
| `PerToolTruncation` | Per-tool Stage 1 truncation configuration. |
| `ReasoningEffort` | Reasoning effort passed to reasoning-capable models (e.g. `"low"`, `"medium"`, `"high"`). |
| `PreWarningPercent` | Context-fill percentage that triggers a pre-compaction `store_fact` nudge. `0` disables it. |
| `Streaming` | Enables live LLM text streaming on the main-loop `Executor`: text deltas are forwarded as `agent.Events.AssistantChunk` as they arrive. Opt-in; `false` (default) preserves the non-streaming behavior. |
| `VerifyOnEdit` | Trusted, user-configured `agent.EditVerifyRunner`. Runs once after each LLM response group containing a successful `write_file`/`edit_file`; nil disables verification. |
| `VerifyOnEditMaxOutputChars` | Rune-safe cap for the injected `[verify_on_edit]` output. `<= 0` selects `agent.DefaultVerifyOnEditCap` (4000). |
| `NonCacheableTools` | Additional tool names whose results must not be cached (e.g. meta-tools whose output is inherently volatile). Extends the SDK-provided defaults. |
| `StopTools` | Ordinary tool names that **terminate the run** when they execute successfully (see `agent.Executor.SetStopTools`): the run ends with `Finished=true`, the call's observation as `Output`, and the model's final text in `ExecutionResult.Summary`. The finish guard still applies; the mutation/checklist gates do not. |
| `ConversationHistory` | Prior user/assistant exchanges from the session. When non-empty, the Conductor injects it into the `ContextManager` so the LLM sees the dialogue leading up to the current message. Without this, a follow-up like "implement variant a" has no referent. |
| `ResumeSteps` | Prior ReAct steps to resume from a checkpoint instead of starting fresh. When non-empty, `Run` seeds the `ContextManager` (via its `StepSeedable` capability) and the `Executor` (via `agent.WithResumeSteps`) with a defensive copy, so the step counter continues from `len(steps)+1` and the full trajectory syncs to the `TrajectoryStore`. The steps count against `MaxSteps`, not in addition to it. Requires a `StepSeedable` `ContextManager`; `Run` fails fast otherwise. Nil/empty (the default) is fully backward-compatible. |
| `ContentBlocks` | Optional text/image blocks for a multimodal task. When non-empty, requires the optional `BlockTaskAware` path; providers render blocks ahead of plain content. |
| `PendingUserInterjection` | One-shot resume nudge appended after seeded history as the final user message. Requires `InterjectionAware`; an `InterjectionConsumer` retires it only after a successful LLM response. Empty disables it. |
| `PauseChecker` | Cooperative pause callback. A true return becomes `ExecutionStatusPaused` with an empty output and preserved trajectory. It is checked at every step boundary and, while a tool call is in flight, polled on the executor's `toolWatchdogInterval` ticker (keep it cheap and idempotent). Nil disables pause. |
| `ToolCallTimeout` | Per-tool-call ceiling installed via `SetToolCallTimeout`. A single tool call whose wait exceeds it is abandoned and the tool's context is cancelled (a cooperative request — an uninterruptible tool may still finish) and the run returns an error wrapping `agent.ErrToolTimeout`, named with the tool. Interactive/long-running tools can be exempted via `agent.Executor.SetToolCallTimeoutExempt`. Zero (default) disables it. |
| `ToolCallTimeoutExemptTools` | Replaces the executor's set of tool names exempt from the `ToolCallTimeout` ceiling (installed via `agent.Executor.SetToolCallTimeoutExempt`). A `nil` slice keeps the executor's built-in default (`agent.DefaultToolCallTimeoutExemptTools`); a non-nil slice — possibly empty — replaces it wholesale. |
| `UserMessageSource` | Host-owned live-message source polled after pause at each boundary. One non-empty message is delivered in the next LLM request. Nil disables live steering. |
| `CompactOnStart` | Forces one compaction pass right after resume/nudge seeding — before the first LLM call — regardless of fill thresholds. Intended for manual compaction of a paused task: relaunch with `ResumeSteps` + `CompactOnStart` and the seeded trajectory is compressed up front. Emits a `ContextCompaction` event (empty step ID) only when `Compact` reports a result; a nil result (nothing to compact) emits nothing. The Executor's threshold-driven reactive compaction may still fire later. `false` (default) keeps compaction purely threshold-driven. |

### NewConductor

```go
func NewConductor(cfg ConductorConfig) *Conductor
```

Creates a Conductor from the given config. If `MaxSteps` is zero it defaults to `80`.

```go
conductor := orchestration.NewConductor(orchestration.ConductorConfig{
    LLM:            router,
    Tools:          registry,
    ToolRegistry:   registry,
    TokenCounter:   llm.NewSimpleTokenCounter(),
    Model:          "claude-sonnet-4-5",
    ModelRegistry:  modelRegistry,
    ContextFactory: contextFactory,
    SystemPrompt:   systemPromptFactory,
    MaxSteps:       15,
})
defer conductor.Cleanup()
```

### Run

```go
func (c *Conductor) Run(
    ctx context.Context,
    message string,
    bb Blackboard,
    availableTools []tools.ToolDescriptor,
    events agent.Events,
    compactionStrategy string,
) (*ExecutionResult, error)
```

Launches the Conductor's single ReAct loop for one task.

**Parameters**

| Parameter | Description |
| --- | --- |
| `ctx` | Context. The caller may inject a `PendingDelegations` registry via `WithDelegationRegistry` to enable the finish-join guard. |
| `message` | The task description for this run (typically a plan step's `Description`). |
| `bb` | The blackboard holding shared task state. The Conductor injects `StepOutputStore`, `FactStore`, `AttachmentStore`, and `FinalResultStore` adapters derived from it into `ctx`. |
| `availableTools` | Tool descriptors the executor may call. |
| `events` | Event sink for observing the ReAct loop. May be `nil`; a `NoopEvents` instance is used in that case. |
| `compactionStrategy` | Context compaction strategy: `"sliding_window"`, `"summarization"`, or `"hierarchical"`. Empty defaults to `"sliding_window"`. |

**Returns**

An `*ExecutionResult` whose `Status` is `ExecutionStatusSuccess`, `ExecutionStatusPartial` (the loop ended without finishing), `ExecutionStatusPaused` (a cooperative, resumable checkpoint), `ExecutionStatusCancelled` (the run was cancelled — the executor returned an error satisfying `errors.Is(err, context.Canceled)`, or `ctx.Err()` was non-nil), or `ExecutionStatusFailed` (any other error occurred). A paused result preserves `Steps` and deliberately leaves `Output` empty rather than exposing the pause sentinel. The returned `Blackboard` is the same instance passed in, now populated with any reflections recorded during the run.

`Run` returns an error only when the context factory or system prompt factory is missing, when `ResumeSteps` is set but the `ContextManager` does not implement `StepSeedable`, or when the underlying executor returns an error. Only the executor-error path pairs a non-nil error with a non-nil `*ExecutionResult` carrying best-effort output; the missing-factory checks and the `StepSeedable` check fail fast and return `(nil, err)`, so callers must nil-check `result` before dereferencing it. `agent.ErrPaused` is mapped to `ExecutionStatusPaused`; callers may still use `errors.Is(err, agent.ErrPaused)` while resuming from `result.Steps`.

```go
result, err := conductor.Run(ctx, step.Description, bb, availableTools, events, "sliding_window")
if err != nil {
    // result is nil on the fail-fast paths (missing factory, ResumeSteps
    // without StepSeedable) — always nil-check it before dereferencing.
    if result != nil {
        log.Printf("run error: %v (status=%s)", err, result.Status)
    } else {
        log.Printf("run error: %v", err)
    }
}
if result != nil && result.Status == orchestration.ExecutionStatusSuccess {
    fmt.Println(result.Output)
}
```

### Cleanup

```go
func (c *Conductor) Cleanup()
```

Releases resources held by the Conductor. Currently a no-op; per-step dump cleanup is owned by the session layer. Safe to call via `defer`.

### SetReasoningEffort

```go
func (c *Conductor) SetReasoningEffort(effort string)
```

Updates the reasoning effort applied to subsequent runs. Useful when the same Conductor is reused across phases that benefit from different reasoning depths.

### SetModel

```go
func (c *Conductor) SetModel(model string)
```

Overrides the model whose metadata the Conductor resolves for the system prompt and the context window (`ConductorConfig.Model`). Use it when the model serving the calls is chosen — or switched — after construction, so the per-step context budget (window, output reserve, tokenizer) is sized against that execution model rather than the framework's construction-time default. Safe to call concurrently with `Run`; `Run` snapshots the value under the same lock.

### WithDelegationRegistry

```go
func WithDelegationRegistry(ctx context.Context, reg PendingDelegations) context.Context
```

Injects a `PendingDelegations` implementation into the context. The Conductor's finish-join guard checks it before allowing a `finish` call: if any async delegations are still pending, `finish` is rejected with a nudge listing them. This prevents the Conductor from silently abandoning background work.

`PendingDelegations` is a minimal interface the SDK defines to avoid a circular dependency with higher layers:

```go
type PendingDelegations interface {
    ListPending() []string
}
```

```go
ctx = orchestration.WithDelegationRegistry(ctx, myRegistry)
result, err := conductor.Run(ctx, task, bb, tools, events, "sliding_window")
```

---

## Interfaces

### Planner

```go
type Planner interface {
    Plan(ctx context.Context, task string, availableTools []tools.ToolDescriptor, reflections []Reflection, availableSkills []skills.SkillDescriptor, singleStep bool, conversationHistory []llm.Message) (*Plan, error)
    Replan(ctx context.Context, originalPlan *Plan, completed []CompletedStep, failedStep CompletedStep, reflection *Reflection, sessionReflections []Reflection, availableSkills []skills.SkillDescriptor) (*Plan, error)
    PlanContinuation(ctx context.Context, originalRequest string, existingPlan *Plan, completedSteps []CompletedStep, newMessage string, availableTools []tools.ToolDescriptor, availableSkills []skills.SkillDescriptor, singleStep bool, conversationHistory []llm.Message, taskComplete bool) (*Plan, error)
}
```

Generates and regenerates DAG execution plans. The reference implementation `planner.Planner` satisfies this interface (verified by a compile-time `var _ orchestration.Planner = (*Planner)(nil)` check in the `planner` package). See [planner.md](planner.md) for the reference implementation.

### Reflector

```go
type Reflector interface {
    Reflect(ctx context.Context, trajectory []agent.Step, plan *Plan, prevReflections []Reflection) (*Reflection, error)
}
```

Analyzes failures and produces corrective insights. See [reflector.md](reflector.md) for the reference implementation.

### Events

`Events` extends `agent.Events` with orchestration-lifecycle hooks. The `On*` hooks are emitted by the host orchestrator as it drives the plan — `Conductor.Run` never calls them. The embedded `agent.Events` methods *are* driven by the Conductor and its executor: `Conductor.Run` emits `ContextCompaction` itself for the `CompactOnStart` pass and hands the same sink to the executor via `agent.WithEvents(events)`, which drives the rest (`AssistantChunk`, `ExecutorDiagnostic`, …).

```go
type Events interface {
    agent.Events
    OnPlanGenerated(stepCount int, steps []PlanStepEvent)
    OnStepStarted(stepID, description, summary string)
    OnStepCompleted(stepID string, success bool, duration time.Duration, errMsg string)
    OnReflected(reflection *Reflection, attempt, maxAttempts int)
    OnRetry(attempt, maxAttempts int)
    OnStepRetry(stepID string, attempt, maxAttempts int)
    OnService(content string)
    OnServiceMeta(content string, meta map[string]any)
    OnReplanFailed(err error)
    OnStepTodoUpdate(stepID string, items []agent.TodoItem)
}
```

`NoopEvents` is a no-op implementation that satisfies the interface; embed it and override only the hooks you care about:

```go
type myEvents struct {
    orchestration.NoopEvents
}

func (e *myEvents) OnStepCompleted(id string, ok bool, d time.Duration, _ string) {
    fmt.Printf("step %s done=%v in %v\n", id, ok, d)
}
```

### BlackboardFactory

```go
type BlackboardFactory func(taskID string) Blackboard
```

Creates a `Blackboard` for a new task, keyed by `taskID`. Like [`StepScopable and RetryScopable`](#stepscopable-and-retryscopable), it is a hook the `orchestration` package declares for host orchestrators to supply — the package itself never calls it — so a host decides what backs each task: `NewMapBlackboard` for in-memory runs, or `NewCheckpointedBlackboard`/`RestoreBlackboard` when the task must survive a restart.

### SystemPromptFactory

```go
type SystemPromptFactory func(ctx context.Context, stepDescription string, modelMeta llm.ModelMetadata) string
```

Builds the system prompt for a step executor. `ctx` carries the workspace path; `stepDescription` is the step's task text; `modelMeta` provides model capabilities (context window, output limit) so the prompt can adapt.

```go
systemPromptFactory := func(_ context.Context, stepDescription string, _ llm.ModelMetadata) string {
    return fmt.Sprintf(`You are a task execution agent.

## Task
%s

Use the available tools. Verify your work before calling finish.`, stepDescription)
}
```

### ContextManagerFactory

```go
type ContextManagerFactory func(
    systemPrompt string,
    modelMeta llm.ModelMetadata,
    compactionStrategy string,
    pruningOverrides ...PruningOverride,
) agent.ContextManager
```

Creates a `ContextManager` for a new task step. `pruningOverrides`, when provided, override the global pruning configuration with step-specific `KeepLastN` and `ProtectedTools` values.

### Optional ContextManager capabilities

The Conductor type-asserts the `ContextManager` returned by the factory against six named capability interfaces; the SDK's `memory.ContextWindow` implements all six:

```go
// Receives the formatted task content (the user message).
type TaskAware interface {
    SetTask(task string)
}

// Receives a multimodal task when ConductorConfig.ContentBlocks is non-empty.
type BlockTaskAware interface {
    SetTaskWithBlocks(task string, blocks []llm.ContentBlock)
}

// Receives prior conversation messages (previous user/assistant exchanges).
type ConversationAware interface {
    SetPriorConversation(msgs []llm.Message)
}

// Exposes the token tracker for API-reported count correction.
type TrackerProvider interface {
    ContextTracker() *llm.ContextTokenTracker
}

// Receives prior ReAct steps when resuming. Required when ResumeSteps is set.
type StepSeedable interface {
    SeedSteps(steps []agent.Step)
}

// Receives the one-shot resume nudge after seeded history.
type InterjectionAware interface {
    SetPendingUserInterjection(msg string)
}
```

A custom `ContextManager` that omits an optional capability still runs, and the corresponding feature is skipped. `StepSeedable` is the exception: when `ResumeSteps` is non-empty, `Run` fails fast if seeding is impossible. `InterjectionAware` accepts `PendingUserInterjection`; after a successful LLM response the executor calls `ConsumePendingUserInterjection()` through its separate `agent.InterjectionConsumer` capability. This consume-on-success split keeps the nudge present if the first call overflows context and is retried after reactive compaction.

### PruningOverride

```go
type PruningOverride struct {
    KeepLastN      int      // 0 = use global default
    ProtectedTools []string // nil = use global default
}
```

Carries per-step overrides for tool-output pruning. Zero values mean "use the global default".

### StepScopable and RetryScopable

These optional interfaces let an `Events` implementation scope events to a specific plan step or retry attempt:

```go
type StepScopable interface {
    WithStepID(id string) Events
}

type RetryScopable interface {
    WithRetryAttempt(attempt int) Events
}
```

The `orchestration` package declares them but does not itself type-assert an `Events` value against them — a host orchestrator that wants scoped emitters performs the check and wraps the sink, so downstream handlers know which step or retry attempt produced each event.

---

## Resume with a user interjection

Use `PendingUserInterjection` when resuming a paused/checkpointed run and the user has supplied a one-shot correction that must land next to the pending tool result:

```go
conductor := orchestration.NewConductor(orchestration.ConductorConfig{
    LLM:                     router,
    Tools:                   registry,
    ToolRegistry:            registry,
    ContextFactory:          contextFactory,
    SystemPrompt:            systemPromptFactory,
    MaxSteps:                25,
    ResumeSteps:             checkpoint,
    PendingUserInterjection: "Use approach B and keep the public API stable.",
})

result, err := conductor.Run(ctx, task, bb, registry.List(), events, "sliding_window")
```

The context manager appends the interjection as the **final** user message after `ResumeSteps`. It remains pending across prompt rebuilds and is consumed only after one successful LLM response, preventing a context-overflow/reactive-compaction retry from silently losing the correction. This differs from `UserMessageSource`, which is a host-owned queue polled throughout a currently running loop.

## Verify edits mechanically

Configure `VerifyOnEdit` with a trusted runner created from user/host settings, not from model text:

```go
conductor := orchestration.NewConductor(orchestration.ConductorConfig{
    // ...required dependencies...
    VerifyOnEdit: func(ctx context.Context) agent.EditVerifyResult {
        return runConfiguredTests(ctx) // host implementation
    },
    VerifyOnEditMaxOutputChars: 6000,
})
```

After a response group containing one or more successful `write_file`/`edit_file` calls, the executor invokes the runner once and appends a `[verify_on_edit]` note to the last observation. Failed or rejected edits do not trigger it. The note distinguishes test failure (positive exit code) from timeout, blocked/killed execution, or infrastructure failure; non-successful runs say that the edit was not verified. See [Agent Executor → Verify-on-edit](agent-executor.md#verify-on-edit) for the result contract and truncation behavior.

---

## Blackboard

The Blackboard is the structured, thread-safe container for all shared task state: the original request, the plan, per-step results, reflections, the final result, and keyword-tagged facts. Every step executor reads from and writes to the same blackboard instance, which is how steps communicate.

### Blackboard interface

```go
type Blackboard interface {
    GetOriginalRequest() string
    GetPlan() *Plan
    GetStepResult(stepID string) (StepResult, bool)
    GetStepSummary(stepID string) string
    GetAllStepResults() map[string]StepResult
    GetReflections() []Reflection
    GetFinalResult() string
    SetOriginalRequest(req string)
    SetPlan(plan *Plan)
    SetStepResult(stepID string, output string, err error, steps []agent.Step)
    AddReflection(r Reflection)
    SetFinalResult(result string)
    Search(query string) []BlackboardEntry

    // Fact memory for inter-step communication
    StoreFact(fact Fact)
    SearchFacts(keywords []string) []Fact
    GetFacts() []Fact

    // Attachment memory (user-attached files converted to markdown)
    AddAttachment(a Attachment)
    GetAttachments() []Attachment
    GetAttachment(id string) (Attachment, bool)
    RemoveAttachment(id string) bool
}
```

All methods are safe for concurrent use. Read methods return defensive copies, so callers can mutate returned slices without racing with the blackboard.

| Method | Behaviour |
| --- | --- |
| `GetOriginalRequest` / `SetOriginalRequest` | The original user request string. |
| `GetPlan` / `SetPlan` | The current plan. `GetPlan` returns a deep copy or `nil`. |
| `GetStepResult` | Returns the `StepResult` for a step ID and whether it exists. |
| `GetStepSummary` | Returns just the auto-generated summary for a step, or `""`. |
| `GetAllStepResults` | Defensive copy of all step results, keyed by step ID. |
| `GetReflections` | Defensive copy of all reflections, in insertion order. |
| `GetFinalResult` / `SetFinalResult` | The final result string. |
| `SetStepResult` | Records a completed step. Auto-generates a summary from `output` (skipping leading blank lines and markdown headings, then the first paragraph, capped at the configured max length). |
| `AddReflection` | Appends a reflection. |
| `Search` | Case-insensitive substring search across step summaries, full outputs, and reflection summaries. Returns `[]BlackboardEntry`. |
| `StoreFact` / `SearchFacts` / `GetFacts` | Fact memory — see [Fact memory](#fact-memory). |
| `AddAttachment` | Stores a user-attached file (converted to markdown), replacing any existing attachment with the same `ID`; defaults `AttachedAt` to now when it is zero. |
| `GetAttachments` / `GetAttachment` | Defensive copy of all attachments, or of the one with the given ID plus whether it exists. |
| `RemoveAttachment` | Removes the attachment with the given ID; returns `true` when one was removed. |

### MapBlackboard

`MapBlackboard` is the reference thread-safe, map-backed implementation. It is the default blackboard for in-memory tasks.

```go
func NewMapBlackboard(opts ...MapBlackboardOption) *MapBlackboard
```

```go
bb := orchestration.NewMapBlackboard()
bb.SetOriginalRequest("Build a Go project that prints hello")
```

**WithMaxSummaryLen**

```go
func WithMaxSummaryLen(n int) MapBlackboardOption
```

Sets a character-based cap on auto-generated step summaries. A value of `0` uses the default of `500` characters. The summary skips leading blank lines and leading markdown headings (ATX `# …` and setext `===`/`---` underlines), then takes the first paragraph (text up to the first double-newline) or the first `n` characters, whichever is shorter, appending `…` when truncated. The configured cap is readable back via the `MaxSummaryLen() int` getter.

```go
bb := orchestration.NewMapBlackboard(orchestration.WithMaxSummaryLen(800))
```

`MapBlackboard` also exposes three methods used by the persistence layer to hydrate state without regenerating summaries:

- `SetStepResultRaw(stepID string, sr StepResult)` — stores a pre-built `StepResult` directly.
- `SetFacts(facts []Fact)` — replaces the entire facts slice (deep-copied).
- `SetAttachments(attachments []Attachment)` — replaces the entire attachments slice (copied).

### GenerateSummary

```go
func GenerateSummary(output string, maxLen int) string
```

Creates a summary from a step's output: the first meaningful paragraph (up to the first double-newline) or the first `maxLen` characters, whichever is shorter. Leading blank lines and leading markdown headings (ATX `# …` and setext `===`/`---` underlines) are skipped so the summary captures actual content rather than a title; truncation happens on a rune boundary via [`strutil.TruncateUTF8`](utilities.md#truncateutf8), which appends `…`. A `maxLen` of `0` uses the default of `500`, and empty output yields an empty summary.

This is the helper `MapBlackboard.SetStepResult` calls to auto-generate `StepResult.Summary` (capped by `WithMaxSummaryLen`, above); it is exported so a host can produce summaries with identical rules.

### Fact memory

Facts are keyword-tagged pieces of information that steps write for later retrieval by other steps. This is the primary inter-step communication channel beyond explicit step outputs.

- `StoreFact(fact Fact)` appends a fact.
- `SearchFacts(keywords []string) []Fact` returns facts where at least one keyword matches (case-insensitive), **sorted by number of matching keywords descending** — most relevant first.
- `GetFacts() []Fact` returns a defensive copy of all stored facts.

```go
bb.StoreFact(orchestration.Fact{
    Keywords: []string{"auth", "middleware", "jwt"},
    Content:  "Auth middleware lives in internal/auth/middleware.go and validates JWTs.",
    Author:   "step_1",
})

matches := bb.SearchFacts([]string{"auth", "jwt"})
for _, f := range matches {
    fmt.Printf("[%s] %s\n", strings.Join(f.Keywords, ", "), f.Content)
}
```

---

## Plan data model

### Plan

```go
type Plan struct {
    Steps              []PlanStep `json:"steps"`
    ExplorationContext string     `json:"exploration_context,omitempty"`
}
```

A DAG of execution steps. `ExplorationContext` holds a summary of any codebase exploration the planner performed before producing the plan (empty for direct, one-shot planning).

### PlanStep

```go
type PlanStep struct {
    ID             string   `json:"id"`
    Summary        string   `json:"summary"`
    Description    string   `json:"description"`
    DependsOn      []string `json:"depends_on"`
    Parallelizable bool     `json:"parallelizable"`
    EstimatedTools []string `json:"estimated_tools"`
    Agent          string   `json:"agent,omitempty"`
    Profile        any      `json:"profile,omitempty"`
}
```

| Field | Description |
| --- | --- |
| `ID` | Stable step identifier (e.g. `"step_1"`). Used in `DependsOn` and as the blackboard key. |
| `Summary` | Short, human-readable label (5–7 words). |
| `Description` | Full task text passed to the step executor. |
| `DependsOn` | IDs of steps that must complete successfully before this one can start. |
| `Parallelizable` | Hint that this step can run concurrently with its siblings. |
| `EstimatedTools` | Tools the step is expected to use. |
| `Agent` | Optionally names a **Subagent Profile** (an `AGENT.md`-declared persona — see [Subagent Profiles](agents.md)) that should execute this step. When non-empty, the step runs with that profile's system prompt, tools, max-steps, and model instead of the generic orchestrator defaults. The name is resolved by the **host's** execution layer — the SDK ships no resolver, and `orchestration.Conductor` passes `PlanStep.Agent` through untouched (it is only declared and copied by `copyPlan`). Orthogonal to `Profile`. |
| `Profile` | Optional step-level configuration. During JSON deserialization this is `map[string]any`; consumers convert it to a domain-specific profile (e.g. `*planner.AgentProfile`). The field type is `any`. |

### CompletedStep

```go
type CompletedStep struct {
    StepID string       `json:"step_id"`
    Output string       `json:"output"`
    Error  error        `json:"-"`
    Steps  []agent.Step `json:"steps,omitempty"`
}
```

The result of an executed plan step. `Steps` holds the actual executor trajectory (tool calls + observations) when captured, which reflectors and evaluators use as evidence. `Error` is `nil` on success.

### StepResult

```go
type StepResult struct {
    StepID     string
    Summary    string
    FullOutput string
    Error      error
    Steps      []agent.Step
}
```

The blackboard's representation of a completed step. `Summary` is auto-generated from `FullOutput`; `Steps` is the executor trajectory.

### PlanStepEvent

```go
type PlanStepEvent struct {
    ID          string   `json:"id"`
    Summary     string   `json:"summary"`
    Description string   `json:"description"`
    Status      string   `json:"status"`
    DependsOn   []string `json:"depends_on"`
}
```

A step representation for event emission. The SDK's own producer — `TaskF`'s `planStepsToEvents`, which feeds `OnPlanGenerated` — leaves `Status` empty; the field is a hook for host-side lifecycle annotation, and a host that re-emits enriched events may populate it with values such as `"pending"`, `"running"`, `"completed"`, or `"failed"`.

### BlackboardEntry

```go
type BlackboardEntry struct {
    Type    string // "step_result", "criterion", "reflection", etc.
    Key     string
    Summary string
}
```

A single search result from `Blackboard.Search`.

---

## Execution outcome

### ExecutionStatus

`ExecutionStatus` classifies the terminal outcome of a plan execution. It is the typed success contract: callers must consult `Status` instead of parsing output suffixes.

```go
type ExecutionStatus string

const (
    ExecutionStatusSuccess   ExecutionStatus = "success"
    ExecutionStatusPartial   ExecutionStatus = "partial"
    ExecutionStatusFailed    ExecutionStatus = "failed"
    ExecutionStatusAborted   ExecutionStatus = "aborted"
    ExecutionStatusCancelled ExecutionStatus = "cancelled"
    ExecutionStatusPaused    ExecutionStatus = "paused"
)
```

| Status | Meaning |
| --- | --- |
| `success` | All plan steps completed without errors. |
| `partial` | Some steps were never attempted (execution incomplete); accompanies `ErrExecutionIncomplete`. |
| `failed` | All steps were attempted, some failed, and the retry budget is exhausted. |
| `aborted` | The reflector recommended aborting after step failures. |
| `cancelled` | The context was cancelled mid-execution. |
| `paused` | The executor stopped at a step boundary (or, since the tool-call watchdog, while a tool call was in flight) because a cooperative pause signal tripped (`agent.ErrPaused`). The trajectory so far is preserved in the result's `Steps`; execution may be resumed. Distinct from `cancelled` (user/interrupt) and `partial` (budget exhausted). |

### ExecutionResult

```go
type ExecutionResult struct {
    Output       string          `json:"output"`
    Summary      string          `json:"summary,omitempty"`
    Steps        []agent.Step    `json:"-"`
    Plan         *Plan           `json:"plan,omitempty"`
    Blackboard   Blackboard      `json:"-"`
    AttemptCount int             `json:"attempt_count,omitempty"`
    Reflections  []Reflection    `json:"reflections,omitempty"`
    Status       ExecutionStatus `json:"status,omitempty"`
    FailedSteps  int             `json:"failed_steps,omitempty"`
}
```

| Field | Description |
| --- | --- |
| `Output` | Aggregated or best-effort output text. |
| `Summary` | The model's own final text when the run terminated on a host-designated stop tool rather than the inline `finish` tool (`Output` is then just the tool's confirmation). Empty otherwise. |
| `Steps` | The executor trajectory (not JSON-serialized). On a paused result it is the resumable checkpoint a caller feeds back via `ResumeSteps`. |
| `Plan` | The plan that was executed (when applicable). |
| `Blackboard` | The blackboard instance (not JSON-serialized). |
| `AttemptCount` | Number of execution attempts. |
| `Reflections` | Reflections recorded during execution. |
| `Status` | Terminal status — see [ExecutionStatus](#executionstatus). |
| `FailedSteps` | Steps that finished with an error in the final attempt. |

### ErrExecutionIncomplete

```go
var ErrExecutionIncomplete = errors.New("plan execution incomplete")
```

Indicates a plan execution ended before all steps completed (e.g. step limit reached, context cancelled). It accompanies a non-nil `*ExecutionResult` carrying best-effort output. Callers should `errors.Is`-check it and use the returned result for partial output. Within the SDK its only producer is the planned `TaskBuilder.Execute` path: `planCompletionStatus` wraps it (with the completed/total counts) when plan steps were never attempted, and `runPlanned` returns it alongside the partial result.

```go
result, err := fw.TaskF(ctx, task).System(prompt).Plan().Reflect().Execute()
if errors.Is(err, orchestration.ErrExecutionIncomplete) {
    // result is non-nil on this path — the partial run keeps its best-effort output.
    log.Printf("partial completion (status=%s): %d/%d step results",
        result.Status, len(result.Blackboard.GetAllStepResults()), len(result.Plan.Steps))
}
```

---

## Reflection

```go
type Reflection struct {
    Summary         string    `json:"summary"`
    Hypotheses      []string  `json:"hypotheses"`
    SuggestedAction string    `json:"suggested_action"` // "retry" | "replan" | "abort"
    Reasoning       string    `json:"reasoning"`
    FailureAnalysis string    `json:"failure_analysis"`
    RootCause       string    `json:"root_cause"`
    ActionPlan      string    `json:"action_plan"`
    Timestamp       time.Time `json:"timestamp"`
}
```

The structured result of failure analysis. `SuggestedAction` drives the orchestrator's recovery decision:

| Value | When to use |
| --- | --- |
| `"retry"` | Try the step again with adjustments informed by the reflection. |
| `"replan"` | The plan itself is wrong; generate a new plan via `Planner.Replan`. |
| `"abort"` | The failure is unrecoverable; stop execution. |

See [reflector.md](reflector.md) for how reflections are produced.

---

## Fact

```go
type Fact struct {
    Keywords []string `json:"keywords"` // 3-10 keywords for retrieval
    Content  string   `json:"content"`  // the fact text
    Author   string   `json:"author"`   // step ID that wrote it
}
```

A keyword-tagged piece of information for inter-step communication. Steps store facts via `Blackboard.StoreFact` and retrieve them via `SearchFacts`/`GetFacts`. The `Author` field records which step wrote the fact, aiding traceability. Keywords drive relevance ranking in `SearchFacts`. The 3–10 keyword rule is enforced by the `store_fact` builtin tool, **not** by `Blackboard.StoreFact`, which is a bare append with no validation.

---

## Attachment

```go
type Attachment struct {
    ID              string    `json:"id"`
    OriginalName    string    `json:"original_name"`
    OriginalPath    string    `json:"original_path"`
    Format          string    `json:"format"` // normalized extension without dot, e.g. "pdf"
    SizeBytes       int64     `json:"size_bytes"`
    MarkdownContent string    `json:"markdown_content"`
    AttachedAt      time.Time `json:"attached_at"`
}
```

A user-attached file that has been converted to markdown and is available to agents as read-only context. Attachment IDs are surfaced to the model in the user message attachment list; agents read `MarkdownContent` via the `read_attachment` tool (backed by [`NewAttachmentStore`](#adapters)). Hosts add attachments to the `Blackboard` via `AddAttachment` before a run.

---

## DAG utilities

The package provides pure functions for traversing and aggregating plan DAGs. They are the building blocks an orchestrator uses to drive execution.

### FindReadySteps

```go
func FindReadySteps(plan *Plan, completed map[string]CompletedStep) []PlanStep
```

Returns plan steps whose dependencies are **all completed successfully**. Steps that are already completed, or that depend on a failed step, are excluded. Returns `nil` if `plan` is nil.

```go
for {
    ready := orchestration.FindReadySteps(plan, completed)
    if len(ready) == 0 {
        break // all done or blocked by failures
    }
    for _, step := range ready {
        // execute step...
    }
}
```

### BuildCarryForward

```go
func BuildCarryForward(completed []CompletedStep, newPlan *Plan) map[string]CompletedStep
```

Maps previously completed step outputs onto a new (replanned) plan. A step is a carry-forward candidate when its ID appears in the new plan **and** it completed without error. Steps whose dependencies (in the new plan) include a step that is **not** carried forward are transitively excluded — a replanned step invalidates all its downstream dependents. Returns `nil` if no steps can be preserved, signalling that full re-execution is required.

```go
carried := orchestration.BuildCarryForward(completedSteps, newPlan)
// `carried` can be seeded into the `completed` map before executing newPlan
```

### BuildPlanExecutionSteps

```go
func BuildPlanExecutionSteps(completedList []CompletedStep, plan *Plan) []agent.Step
```

Converts completed plan steps into an execution trajectory (`[]agent.Step`) for reflectors and evaluators. When a `CompletedStep` carries the actual executor steps (tool calls + observations), those are used directly so the evaluator sees real evidence. Otherwise a fallback summary step is constructed from the completion output.

### AggregateOutput

```go
func AggregateOutput(completedSteps map[string]CompletedStep, plan *Plan, preCompletedIDs map[string]bool) string
```

Combines outputs from **terminal steps** — steps that no other step depends on. If no terminal outputs exist, all step outputs are collected instead. `preCompletedIDs`, when non-nil, lists step IDs that were pre-completed from a previous turn's blackboard; these are excluded so continuation messages return only newly produced output.

```go
finalOutput := orchestration.AggregateOutput(completed, plan, nil)
```

---

## Persistence

The persistence layer lets you checkpoint blackboard state to an external backend and restore it later — essential for surviving restarts or resuming long-running tasks.

### Checkpointer

```go
type Checkpointer interface {
    SaveCheckpoint(ctx context.Context, id string, bb Blackboard) error
    LoadCheckpoint(ctx context.Context, id string) (Blackboard, error)
    DeleteCheckpoint(ctx context.Context, id string) error
}
```

Provides persistence for blackboard state. Implementations are responsible for the serialization format and backend storage. All methods must be safe for concurrent use. `LoadCheckpoint` returns `nil, nil` when the checkpoint does not exist. When no `Checkpointer` is configured, persistence is simply disabled — the `Framework.RestoreBlackboard` method returns an error if called without one.

### CheckpointedBlackboard

`CheckpointedBlackboard` wraps a `MapBlackboard` and persists every write operation through a `Checkpointer`. Read methods delegate to the embedded `MapBlackboard`; write methods delegate **and** persist.

```go
func NewCheckpointedBlackboard(
    id string,
    cp Checkpointer,
    logger *slog.Logger,
    timeout time.Duration,
    opts ...MapBlackboardOption,
) *CheckpointedBlackboard
```

| Parameter | Description |
| --- | --- |
| `id` | Checkpoint identifier. |
| `cp` | The Checkpointer backend. |
| `logger` | Optional structured logger (`nil`-safe). |
| `timeout` | Max time for a single checkpoint write. `0` uses the default (5s). |
| `opts` | `MapBlackboardOption`s (e.g. `WithMaxSummaryLen`). |

All persistence calls are **best-effort**: errors are logged but do not propagate to callers. Persistence operations run on a single background worker goroutine with a timeout and panic recovery, so a slow or panicking backend cannot hang the agent.

**Lifecycle methods**

- `SetOnChanged(fn func(changeType string))` — optional callback invoked after every successful write. `changeType` is one of the seven values `"original_request"`, `"plan"`, `"step_result"`, `"reflection"`, `"fact"`, `"attachment"`, `"final_result"`.
- `SetPersistContext(ctx context.Context)` — sets the context used for persistence operations (carries cancellation/tracing). Defaults to `context.Background()`.
- `Shutdown()` — closes the persistence channel and waits for the worker to drain, bounded by a grace period (`shutdownGrace`; `<= 0` falls back to `defaultShutdownGrace = 30 * time.Second`). A worker still busy when the grace expires is logged ("persistence worker did not stop before the shutdown grace expired; abandoning it") and abandoned instead of hanging shutdown. Idempotent via `sync.Once` — safe to call multiple times. **Always call this** when the blackboard is no longer needed to prevent goroutine leaks.
- `ID() string` — returns the checkpoint identifier.

```go
cp := myCheckpointer{} // implements orchestration.Checkpointer
bb := orchestration.NewCheckpointedBlackboard("task-42", cp, logger, 5*time.Second)
defer bb.Shutdown()

bb.SetOnChanged(func(change string) {
    fmt.Println("blackboard changed:", change)
})

bb.SetPlan(plan)        // persisted
bb.SetStepResult(...)   // persisted
bb.StoreFact(...)       // persisted
```

### RestoreBlackboard

```go
func RestoreBlackboard(
    ctx context.Context,
    id string,
    cp Checkpointer,
    logger *slog.Logger,
    timeout time.Duration,
    opts ...MapBlackboardOption,
) (*CheckpointedBlackboard, error)
```

Loads a blackboard state from a Checkpointer and hydrates a fresh `CheckpointedBlackboard`. Returns `nil, nil` if the checkpoint does not exist. The restored blackboard is fully writable and will persist subsequent changes under the same `id`.

```go
bb, err := orchestration.RestoreBlackboard(ctx, "task-42", cp, logger, 5*time.Second)
if err != nil {
    return err
}
if bb == nil {
    // no checkpoint found — start fresh
    bb = orchestration.NewCheckpointedBlackboard("task-42", cp, logger, 5*time.Second)
}
defer bb.Shutdown()
```

### StepDumpTracker

```go
func NewStepDumpTracker(dir string, logger *slog.Logger, opts ...StepDumpOption) *StepDumpTracker
```

A best-effort manager for **per-step LLM dump files**, used when the host records each plan step's LLM traffic to its own file for offline debugging. Pass a directory and a logger (`nil` falls back to `slog.Default()`); `NewStepDumpTracker` creates the directory (logging a warning on failure, but not returning an error — dumps are best-effort debugging aids). Pass an empty `dir` to disable the tracker.

Without options the dumps are plain JSONL (`appendcodec.CodecNone`). Two options select compression, via the `github.com/v0lka/sp4rk/appendcodec` package:

```go
// WithCodec sets the dump codec: appendcodec.CodecNone (plain JSONL, the
// default), CodecGzip or CodecZstd. The codec extends the filename —
// step_<id>.jsonl, step_<id>.jsonl.gz or step_<id>.jsonl.zst.
func WithCodec(codec appendcodec.Codec) StepDumpOption

// WithCodecWindowLog sizes the zstd encoder window (1<<windowLog, clamped
// by appendcodec; <=0 keeps the encoder default). Ignored by other codecs.
func WithCodecWindowLog(windowLog int) StepDumpOption
```

```go
type StepDumpTracker struct{ /* unexported */ }

// OpenStepDump returns an io.Writer for the step's dump file — an
// appendcodec wrapper that appends codec-framed records (each completed
// Write is already decodable on disk) — or nil when disabled/closed. The
// file is created on first call and the wrapper is cached — repeated calls
// for the same stepID (including retries) append through the same encoder.
func (t *StepDumpTracker) OpenStepDump(stepID string) io.Writer

// CloseAll closes every tracked writer: each wrapper's Close writes the
// codec footer (gzip trailer / zstd frame epilogue; nothing for CodecNone)
// and then releases the file descriptor. Idempotent.
func (t *StepDumpTracker) CloseAll() error
```

Key properties:

- **Path-traversal safe** — the filename is `step_<filepath.Base(stepID)>.jsonl` plus the codec's extension, so any directory components in `stepID` are stripped and the file always lands inside `dir`.
- **Codec-consistent** — codec-specific filenames mean different codecs never collide on one path, and `appendcodec.OpenAppend` still verifies the existing file's magic before appending, refusing (with a warning, nil writer) foreign content planted at the codec's own path.
- **Idempotent** — opening a dump for an already-seen step returns the cached wrapper, so retries and sub-steps append rather than overwrite (and a second live encoder — which would interleave compressed frames — is never created). Across trackers, `appendcodec`'s advisory path lock enforces the same rule: a concurrent second open of a dump file fails with `appendcodec.ErrLocked` and degrades to the ordinary warning + nil writer path, instead of interleaving frames.
- **Graceful degradation** — every failure (directory creation, file open, write) is swallowed with a warning; dumps are debugging aids and never affect execution.

A host that wants per-step dumps opens each step's writer with `OpenStepDump` and feeds it into the executor context itself via [`agent.WithDumpWriter`](agent-executor.md#llm-debugging-callers) (typically wrapped around a [`NewDumpCaller`](agent-executor.md#newdumpcaller--full-jsonl-requestresponse-dumps) caller through the `CallerForStep` hook), so each step produces a self-contained `step_<id>.jsonl[.gz|.zst]`. The SDK does not wire this for you: the Conductor never opens or consumes these dumps (its `Cleanup` is a no-op) and the executor never reads the context key back.

---

## Adapters

The package provides four small adapters that wrap a `Blackboard` as the `agent.*Store` interfaces consumed by built-in tools. The Conductor injects these into the context automatically, but they are also useful standalone.

```go
func NewStepOutputStore(bb Blackboard) agent.StepOutputStore
func NewFactStore(bb Blackboard) agent.FactStore
func NewAttachmentStore(bb Blackboard) agent.AttachmentStore
func NewFinalResultStore(bb Blackboard) agent.FinalResultStore
```

- `NewStepOutputStore` exposes step outputs to the `read_step_output` tool, distinguishing three cases. A step that completed **without error** is visible with its full output. A step whose error is the **cooperative-pause sentinel** — matched by `errors.Is(err, agent.ErrPaused)`, plus a trailing-suffix string match on the sentinel's message for hosts that rebuild the error after persistence — is *also* visible, surfaced as the marker `"[paused checkpoint — no final output yet]"`; hiding it would read as vanished/failed. Only **genuinely failed** steps (any other error) are excluded. Visible outputs are listed in deterministic (step-ID) order.
- `NewFactStore` exposes fact memory to the `store_fact` / `search_facts` tools.
- `NewAttachmentStore` exposes user-attached files (converted to markdown) to the `read_attachment` tool. Host applications add attachments to the blackboard before a run; the attachment IDs are surfaced in the user message.
- `NewFinalResultStore` exposes the prior task's final result to the `read_final_result` tool — useful for continuation agents when the conversation history alone is insufficient (e.g. after a restart, or when the result was too large to inject verbatim).

---

## Complete Plan & Execute example

This example shows the full Plan & Execute pattern: a `Planner` generates a DAG, a `Conductor` executes each step, and a `Reflector` provides self-correction on failure. It is adapted from the SDK's example 06. Note the `ConfirmFunc` in the config: `write_file`, `edit_file`, and `create_directory` default to `tools.PolicyUserConfirm`, and the registry is fail-closed — with no `ConfirmFunc` configured, every such call is denied rather than executed.

```go
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"sync"

	"github.com/v0lka/sp4rk"
	"github.com/v0lka/sp4rk/agent"
	"github.com/v0lka/sp4rk/agent/reflector"
	"github.com/v0lka/sp4rk/llm"
	"github.com/v0lka/sp4rk/orchestration"
	"github.com/v0lka/sp4rk/planner"
	"github.com/v0lka/sp4rk/tools"
	"github.com/v0lka/sp4rk/tools/builtins"
)

// trajectoryStore captures the executor's step history for reflection.
type trajectoryStore struct {
	mu    sync.Mutex
	steps []agent.Step
}

func (s *trajectoryStore) Sync(steps []agent.Step) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.steps = steps
}

func (s *trajectoryStore) Steps() []agent.Step {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.steps
}

func run() error {
	fw, err := sp4rk.New(sp4rk.Config{
		LLM: sp4rk.LLMConfig{
			Providers: []llm.ProviderEntry{{
				Name:         "anthropic",
				ProviderType: "anthropic",
				APIKey:       os.Getenv("ANTHROPIC_API_KEY"),
				Models:       []string{"claude-sonnet-4-5"},
			}},
		},
		Execution: sp4rk.ExecutionConfig{
			MaxSteps:   15, // per-step ReAct budget
			MaxRetries: 2,  // retries per plan step on failure
		},
		// The registry is FAIL-CLOSED for PolicyUserConfirm tools
		// (write_file, edit_file, create_directory). This example runs in a
		// throwaway temp workspace, so we auto-approve mutations.
		ConfirmFunc: func(_ context.Context, _ tools.ConfirmationRequest) (tools.ConfirmationResponse, error) {
			return tools.ConfirmAllowOnce, nil
		},
	})
	if err != nil {
		return err
	}
	defer func() { _ = fw.Shutdown() }()

	// Register tools.
	registry := fw.ToolRegistry()
	registry.Register(builtins.NewReadFileTool())
	registry.Register(builtins.NewWriteFileTool())
	registry.Register(builtins.NewEditFileTool())
	registry.Register(builtins.NewListDirectoryTool())
	registry.Register(builtins.NewGlobTool())
	registry.Register(builtins.NewCreateDirectoryTool())
	registry.Register(agent.NewFinishTool())

	workspaceDir, err := os.MkdirTemp("", "example-*")
	if err != nil {
		return err
	}
	defer func() { _ = os.RemoveAll(workspaceDir) }()

	// --- Planner ---
	plannerCfg := planner.DefaultConfig()
	plannerCfg.Prompts = planner.PromptSet{
		BasePrompt: `You are a task planning agent. Break down the user's task into steps.

## Available Tools
AVAILABLE-TOOLS

Create at most MAX-STEPS steps. Use "depends_on" to chain steps.

MODE-PREAMBLE

## Output Format
Return ONLY a valid JSON object:
MODE-JSON-EXAMPLE`,
		PlanPreamble:      "Break the task into logical, sequential steps.",
		MultiStepGuidance: "Prefer fewer, well-defined steps with acceptance criteria.",
	}
	plannerCfg.Model = "claude-sonnet-4-5"

	pl, err := planner.NewPlanner(fw.LLMRouter(), plannerCfg)
	if err != nil {
		return err
	}

	// --- Reflector ---
	rf := reflector.New(fw.LLMRouter(), reflector.Config{
		SystemPrompt: `You are a reflection agent. Analyze the failed execution.
Return JSON with "summary", "root_cause", "suggested_action" (retry/replan/abort), and "action_plan".`,
	})

	// --- Conductor ---
	systemPromptFactory := func(_ context.Context, stepDescription string, _ llm.ModelMetadata) string {
		return fmt.Sprintf(`You are a task execution agent.

## Task
%s

Use the available tools. Verify your work before calling finish.`, stepDescription)
	}
	conductor, err := fw.NewConductor(systemPromptFactory)
	if err != nil {
		return err
	}
	defer conductor.Cleanup()

	// --- The task ---
	task := fmt.Sprintf(`Create a small Go project in %s:
1. Create a directory called "myproject"
2. Write a main.go file that prints "Hello from planned agent!"
3. Read the file back to verify it was written correctly`, workspaceDir)

	ctx := tools.WithWorkspacePath(context.Background(), workspaceDir)
	availableTools := registry.List()

	// --- Step 1: Plan ---
	bb := orchestration.NewMapBlackboard()
	bb.SetOriginalRequest(task)

	plan, err := pl.Plan(ctx, task, availableTools, nil, nil, false, nil)
	if err != nil {
		return fmt.Errorf("planning failed: %w", err)
	}
	fmt.Printf("Plan: %d steps\n", len(plan.Steps))

	// --- Step 2: Execute the DAG with retry + reflect ---
	completed := make(map[string]orchestration.CompletedStep)
	var reflections []orchestration.Reflection
	maxRetries := 2

	for {
		readySteps := orchestration.FindReadySteps(plan, completed)
		if len(readySteps) == 0 {
			break
		}

		for _, step := range readySteps {
			success := false
			for attempt := 1; attempt <= maxRetries+1; attempt++ {
				store := &trajectoryStore{}
				stepCtx := agent.WithTrajectoryStore(ctx, store)

				result, runErr := conductor.Run(
					stepCtx, step.Description, bb, availableTools,
					&agent.NoopEvents{}, "sliding_window",
				)
				trajectory := store.Steps()

				if runErr == nil && result.Status == orchestration.ExecutionStatusSuccess {
					completed[step.ID] = orchestration.CompletedStep{
						StepID: step.ID,
						Output: result.Output,
						Steps:  trajectory,
					}
					bb.SetStepResult(step.ID, result.Output, nil, trajectory)
					success = true
					break
				}

				// Failure — reflect and decide.
				if attempt <= maxRetries {
					reflection, reflectErr := rf.Reflect(stepCtx, trajectory, plan, reflections)
					if reflectErr == nil && reflection != nil {
						bb.AddReflection(*reflection)
						reflections = append(reflections, *reflection)
						fmt.Printf("Reflection: %s → %s\n",
							reflection.RootCause, reflection.SuggestedAction)
						if reflection.SuggestedAction == "abort" {
							break
						}
						// For "replan", a full implementation would call
						// pl.Replan(...) and restart with the new plan.
					}
				}
			}
			if !success {
				completed[step.ID] = orchestration.CompletedStep{
					StepID: step.ID,
					Error:  fmt.Errorf("step failed after %d attempts", maxRetries+1),
				}
			}
		}
	}

	// --- Step 3: Aggregate results ---
	finalOutput := orchestration.AggregateOutput(completed, plan, nil)
	fmt.Printf("Completed %d/%d steps, %d reflections\n",
		len(completed), len(plan.Steps), len(reflections))
	fmt.Println(finalOutput)
	return nil
}

func main() {
	if err := run(); err != nil {
		log.Fatalf("%v", err)
	}
}
```

The pattern is:

1. **Plan** — `Planner.Plan` produces a `*Plan` (a DAG of `PlanStep`s).
2. **Execute** — `FindReadySteps` selects steps whose dependencies are met; `Conductor.Run` executes each step's ReAct loop, recording results on the blackboard.
3. **Reflect** — on failure, `Reflector.Reflect` analyses the trajectory and returns a `Reflection` whose `SuggestedAction` drives retry, replan, or abort.
4. **Aggregate** — `AggregateOutput` combines terminal step outputs into the final result.
