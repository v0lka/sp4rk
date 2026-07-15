# Blackboard

## Role

The structured, thread-safe container for all shared task state: the original request, the plan, per-step results, reflections, the final result, and keyword-tagged facts. Every step executor reads from and writes to the same blackboard instance, which is how steps communicate and how state is checkpointed across restarts.

## Key Files

- `github.com/v0lka/sp4rk/orchestration` — `Blackboard` interface, `MapBlackboard` (reference in-memory implementation), `Plan`, `StepResult`, `Reflection`, `Fact`, `Attachment`, `BlackboardEntry`, `CompletedStep`
- `github.com/v0lka/sp4rk/orchestration` (adapters) — `NewStepOutputStore`, `NewFactStore`, `NewAttachmentStore`, `NewFinalResultStore` (Blackboard → `agent.*Store` interfaces consumed by built-in tools)
- `github.com/v0lka/sp4rk/orchestration` (persistence) — `Checkpointer`, `CheckpointedBlackboard`, `RestoreBlackboard`
- `github.com/v0lka/sp4rk/agent` — `Step`

## Behavior

### Blackboard interface

```go
type Blackboard interface {
    GetOriginalRequest() string
    SetOriginalRequest(req string)
    GetPlan() *Plan
    SetPlan(plan *Plan)
    GetStepResult(stepID string) (StepResult, bool)
    GetStepSummary(stepID string) string
    GetAllStepResults() map[string]StepResult
    SetStepResult(stepID, output string, err error, steps []agent.Step)
    GetReflections() []Reflection
    AddReflection(r Reflection)
    GetFinalResult() string
    SetFinalResult(result string)
    Search(query string) []BlackboardEntry
    StoreFact(fact Fact)
    SearchFacts(keywords []string) []Fact
    GetFacts() []Fact
    AddAttachment(a Attachment)
    GetAttachments() []Attachment
    GetAttachment(id string) (Attachment, bool)
    RemoveAttachment(id string) bool
}
```

All methods are safe for concurrent use. Read methods return defensive copies, so callers can mutate returned slices without racing the blackboard. `SetStepResult` auto-generates a summary from `output` (first paragraph, capped at the configured max length). `Search` is a case-insensitive substring search across step summaries, full outputs, and reflection summaries.

### Fact memory

Facts are the primary inter-step communication channel beyond explicit step outputs:

```go
type Fact struct {
    Keywords []string // 3-5 recommended; drive retrieval ranking
    Content  string
    Author   string   // step ID that wrote the fact
}
```

- `StoreFact(fact)` appends a fact.
- `SearchFacts(keywords)` returns facts where at least one keyword matches (case-insensitive), **sorted by number of matching keywords descending** — most relevant first.
- `GetFacts()` returns a defensive copy of all facts.

### Attachment memory

Attachments are user-attached files converted to markdown and surfaced to agents as read-only context. Host applications add them before a run; the agent reads the converted content via the `read_attachment` tool (the IDs are listed in the user message). Like facts, attachments live on the blackboard and survive checkpoints.

```go
type Attachment struct {
    ID              string    // stable identifier surfaced in the user message attachment list
    OriginalName    string
    OriginalPath    string
    Format          string    // normalized extension without the dot, e.g. "pdf"
    SizeBytes       int64
    MarkdownContent string    // converted content read by the read_attachment tool
    AttachedAt      time.Time
}
```

- `AddAttachment(a)` appends an attachment. When `AttachedAt` is zero it is set to the current time. When an attachment with the same `ID` already exists it is **replaced** (replace-on-conflict semantics) rather than duplicated.
- `GetAttachments()` returns a defensive copy of all attachments (`nil` when none are stored).
- `GetAttachment(id)` returns a defensive copy of the attachment with that `ID` (`Attachment{}, false` when absent).
- `RemoveAttachment(id)` removes the attachment with that `ID` and returns whether anything was removed. The dead trailing slot is zeroed so a removed attachment (which may hold a large `MarkdownContent`) is eligible for garbage collection.
- `SetAttachments(attachments)` replaces the whole slice (used by persistence restoration) and defensively copies the input.

### MapBlackboard

`NewMapBlackboard(opts ...MapBlackboardOption)` is the reference thread-safe, map-backed implementation — the default for in-memory tasks. `WithMaxSummaryLen(n)` caps auto-generated step summaries (default `500` characters; first paragraph up to `n` chars, `...` when truncated). It also exposes `SetStepResultRaw`, `SetFacts`, and `SetAttachments`, used by the persistence layer to hydrate state without regenerating summaries.

### Adapters

Three small adapters wrap a `Blackboard` as the `agent.*Store` interfaces consumed by built-in tools. The [Conductor](../orchestration/conductor.md) injects these into the context automatically:

| Adapter | Exposes | Consumer tool |
| ------- | ------- | ------------- |
| `NewStepOutputStore` | successful step outputs (only error-free steps), deterministic step-ID order | `read_step_output` / `list_step_outputs` |
| `NewFactStore` | fact memory | `store_fact` / `search_facts` |
| `NewAttachmentStore` | user-attached files converted to markdown | `read_attachment` |
| `NewFinalResultStore` | the prior task's final result | `read_final_result` (recovery for continuation agents) |

### Persistence

The persistence layer checkpoints blackboard state to an external backend and restores it later — essential for surviving restarts or resuming long-running tasks.

```go
type Checkpointer interface {
    SaveCheckpoint(ctx, id string, bb Blackboard) error
    LoadCheckpoint(ctx, id string) (Blackboard, error) // nil, nil when absent
    DeleteCheckpoint(ctx, id string) error
}
```

`CheckpointedBlackboard` wraps a `MapBlackboard` and persists every write through a `Checkpointer`. All persistence calls are **best-effort**: errors are logged but do not propagate, and operations run on a single background worker goroutine with a timeout and panic recovery, so a slow or panicking backend cannot hang the agent. `SetOnChanged(fn)` invokes a callback after successful writes; `Shutdown()` closes the persistence channel and waits for the worker (always call this to prevent goroutine leaks).

`RestoreBlackboard(ctx, id, cp, logger, timeout, opts)` loads state from a `Checkpointer` and hydrates a fresh `CheckpointedBlackboard` (returns `nil, nil` when the checkpoint does not exist). When no `Checkpointer` is configured, persistence is disabled.

### DAG data model

`Plan` is a DAG of `PlanStep`s (`ID`, `Summary`, `Description`, `DependsOn`, `Parallelizable`, `EstimatedTools`, `Profile`). `CompletedStep` / `StepResult` record an executed step's output, error, and full executor trajectory (`Steps`). The trajectory is the evidence reflectors and evaluators use. See [../orchestration/planner.md](../orchestration/planner.md) for plan generation.

## Error Handling

- **Persistence failure** (`CheckpointedBlackboard`): logged and non-fatal; in-memory state remains consistent.
- **Checkpoint not found**: `LoadCheckpoint`/`RestoreBlackboard` return `nil, nil` (start fresh).
- **Read methods**: always return defensive copies; never panic on missing keys (return zero values / `false`).

## Invariants

- All `Blackboard` methods are safe for concurrent use.
- Read methods return defensive copies.
- `SearchFacts` results are ranked by number of matching keywords (descending).
- `AddAttachment` replaces an existing attachment with the same ID; no two stored attachments ever share an ID.
- Attachment reads return defensive copies; `GetAttachments` returns `nil` when no attachments are stored.
- `RemoveAttachment` zeroes the removed slot so a removed attachment's `MarkdownContent` is eligible for garbage collection.
- A `StepResult`'s `Steps` trajectory is immutable once stored.
- `CheckpointedBlackboard` persistence is best-effort and never blocks the agent on a slow backend.
- `Shutdown()` is idempotent and always closes the persistence worker.

## Related Specs

- [README.md](README.md) — context management overview
- [../orchestration/README.md](../orchestration/README.md) — blackboard wired into each Conductor run
- [../orchestration/conductor.md](../orchestration/conductor.md) — injects the store adapters into the run context
- [../orchestration/reflector.md](../orchestration/reflector.md) — reflections persist on the blackboard
