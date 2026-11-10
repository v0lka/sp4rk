# Example 09 — Context Window & Memory Management

A focused look at the SDK's **memory subsystem**: how the context window fills,
how compaction reclaims space, and how to observe both in real time.

Compaction is configured as a single field in other examples (the capstone sets
`CompactionConfig.Strategy = "sliding_window"` and never triggers it). This
example makes the subsystem the focus: it configures the thresholds, references
all three strategies, and streams the `ContextFill` / `ContextCompaction` events.

This example ships in **two variants**:

| Variant     | File            | Command                 |
|-------------|-----------------|-------------------------|
| **Fluent**  | `main_fluent.go`| `go run -tags fluent .` |
| **Classic** | `main.go`       | `go run .`              |

See [`docs/memory.md`](../../docs/memory.md).

## What you will learn

- `sp4rk.CompactionConfig` — the three trigger thresholds + strategy name
- The three strategies: `sliding_window`, `summarization`, `hierarchical`
- The `ContextFill` event (emitted after each completed step iteration — the terminal finish iteration exits before the post-step pass — reporting fill %, tokens used/max, status)
- The `ContextCompaction` event (emitted when space is reclaimed: before → after %)
- Per-task strategy override via `TaskBuilder.Compaction(...)` (fluent variant)

## How it works

```
each completed step iteration ──► ContextWindow.CheckFill()
                                      │  fill % vs thresholds → status: ok|compact|warning|emergency|reject
                                      ▼
                                 emitter.ContextFill(percent, used, max, status, stepID)
                                      │
                          status ∈ {compact,warning,emergency}?
                                      │ yes
                                      ▼
                                 ContextWindow.Compact(ctx)  ──► emitter.ContextCompaction(before, after, stepID)
```

The three thresholds (default → demo value):

| Threshold           | Default | Demo | Meaning                                   |
|---------------------|---------|------|-------------------------------------------|
| `PredictivePercent` | 85      | 10   | proactive compaction kicks in             |
| `WarningPercent`    | 92      | 50   | same compaction, at a higher fill level   |
| `EmergencyPercent`  | 98      | 70   | last-resort compaction to avoid rejection |

The demo lowers `PredictivePercent` and seeds several files so the fill climbs
quickly and a `ContextCompaction` event can fire on a short run. On a
large-context model at the default 85%, compaction only triggers at scale —
which is exactly why the threshold is configurable.

### The three strategies

| Strategy          | How it reclaims space                            | Needs an LLM summarizer? |
|-------------------|--------------------------------------------------|--------------------------|
| `sliding_window`  | keep first N + last M **steps**, middle → […]    | No                       |
| `summarization`   | LLM-summarize older blocks into compact summaries | Yes                      |
| `hierarchical`    | tiered resolution: distant/middle/recent detail  | Yes                      |

`sliding_window` works out of the box. The other two need an LLM summarizer
(`memory.CompactionDeps.Summarize`); with none, `memory.NewCompactionStrategy`
fails closed and returns the sliding-window strategy instead. The Framework
always builds its context windows with `Summarize: nil` and `sp4rk.Config`
exposes no summarizer hook, so a Framework-driven run — including this
example's fluent `.Compaction("hierarchical")` override — compacts with the
sliding window in practice. To exercise the LLM-backed strategies, build the
`memory.ContextWindow` yourself and supply it through an
`orchestration.ContextManagerFactory` set on `ConductorConfig.ContextFactory`, or
the separately-declared `planner.ContextManagerFactory` set on
`planner.Config.ContextFactory`. The two factory types are not interchangeable: `orchestration.ContextManagerFactory` takes a trailing variadic `...PruningOverride` while `planner.ContextManagerFactory` does not. `sp4rk.Config` exposes no such hook — the
Framework hard-wires its own factory, and `fw.ContextFactory()` only hands that
built-in factory back read-only.
Tool-output pruning (keeping only the last N results, protecting named tools)
runs alongside every strategy.

## Run it

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
cd examples/09-context-memory
go run -tags fluent .   # or: go run .
```

Each step prints a `📊 Context` line; when the fill crosses the predictive
threshold, a `♻️ Compaction` line shows the reclaimed percentage.
