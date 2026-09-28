# Reflector

## Role

Provides structured self-correction. After a step fails, a `Reflector` analyses the execution trajectory — the sequence of thoughts, actions, and observations the agent produced — and returns an `orchestration.Reflection` containing a root-cause analysis and a suggested recovery action. The reference implementation in `agent/reflector` satisfies `orchestration.Reflector` and drops into any orchestrator that supports reflection-driven retry loops.

## Key Files

- `github.com/v0lka/sp4rk/agent/reflector` — `Reflector`, `New`, `Config`, `Reflect`, `SetReasoningEffort`
- `github.com/v0lka/sp4rk/orchestration` — `Reflector` interface, `Reflection` type, `Plan`, `Blackboard.AddReflection`/`GetReflections`
- `github.com/v0lka/sp4rk/agent` — `LLMCaller`, `Step` (trajectory), `TrajectoryStore`
- `github.com/v0lka/sp4rk/llm` — `ExtractJSON`

## Behavior

The `Reflector` is a stateless wrapper around an `LLMCaller`: each `Reflect` call sends the trajectory, the plan, and any prior reflections to the LLM and parses the JSON response into an `orchestration.Reflection`.

### Config

```go
type Config struct {
    SystemPrompt  string // reflection system prompt; instructs the LLM on analysis + JSON schema
    AnalyzeFooter string // appended to the user message; defaults to a standard analysis request

    ReasoningEffort string // initial reasoning effort; may be changed at runtime via SetReasoningEffort
    Family          string // optional model identity for reasoning-tier resolution
    Model           string
}
```

The system prompt should instruct the model to return a JSON object matching the `Reflection` schema. The reflector runs the call through the one-shot client (see [../oneshot.md](../oneshot.md)) and extracts JSON from the response (tolerating surrounding prose or markdown fences) before unmarshalling it. The reasoning effort is resolved per call: an explicit `SetReasoningEffort` (or `Config.ReasoningEffort`) override wins; otherwise the service-call policy applies — reasoning tier Off, resolved to the serving model's native disable spelling via `llm.ReasoningForCall` from `Family`/`Model` (`""` — no field sent — when the identity is empty or unknown).

### Reflect

```go
func (r *Reflector) Reflect(ctx, trajectory []agent.Step, plan *orchestration.Plan, prevReflections []orchestration.Reflection) (*orchestration.Reflection, error)
```

| Parameter | Description |
| --------- | ----------- |
| `ctx` | Carries workspace/environment info appended to the system prompt as a compact environment block. |
| `trajectory` | The executor's step history (thought → action → observation) produced during the failed run. |
| `plan` | The plan being executed, providing step descriptions/dependencies. May be `nil`. |
| `prevReflections` | Reflections from earlier failures in the same task — the reflector learns from these to avoid repeating mistakes. |

### How the trajectory is built

The reflector constructs a user message presenting the trajectory in step-by-step **Thought / Action / Observation** format. When a plan is provided, it is appended as a section listing each step's ID, description, and dependencies. The trajectory is captured by injecting an `agent.TrajectoryStore` into the context before running the executor; the executor syncs its step history to the store at each iteration, so the full trajectory is available after the run.

### Previous reflections

When `prevReflections` is non-empty, the reflector includes them under a `## Previous Reflections` section, prefixed with a note to learn from them and avoid repeating the same mistakes. Each prior reflection's summary, root cause, action plan, and suggested action are listed.

### Response parsing

- `"retry"`, `"replan"`, and `"abort"` are accepted as-is for `SuggestedAction`.
- An empty or unrecognised value defaults to `"retry"` (the safest recovery attempt).
- If `Summary` is empty, it is set to `"Execution analysis unavailable"`.
- `Timestamp` is always set to `time.Now()`.

### Reflection

```go
type Reflection struct {
    Summary         string
    Hypotheses      []string
    SuggestedAction string   // "retry" | "replan" | "abort"
    Reasoning       string
    FailureAnalysis string
    RootCause       string
    ActionPlan      string
    Timestamp       time.Time
}
```

`SuggestedAction` drives the orchestrator's recovery decision:

| Value | Orchestrator response |
| ----- | --------------------- |
| `"retry"` | Re-run the step (optionally injecting the reflection's `ActionPlan` into the next attempt). |
| `"replan"` | Call `Planner.Replan` to generate a corrected plan, then resume. |
| `"abort"` | Stop execution immediately. |

## Integration with the Blackboard

Reflections persist on the `orchestration.Blackboard` (`AddReflection` appends; `GetReflections` returns them in insertion order) so they survive the retry loop and are available to the planner during replanning. A typical retry loop records each reflection and passes the accumulated list to the next `Reflect` call. The `ExecutionResult.Reflections` field also carries the accumulated reflections, so callers can inspect what went wrong after the task completes.

## Error Handling

- Returns an error if the LLM call fails (transport errors are never retried here — provider-level retry is the Router's), returns nil, or stays unparseable after the one-shot client's repair loop: exactly two corrective nudges (assistant echo of the failed output + a `[System]` user message restating the required JSON schema — see [../oneshot.md](../oneshot.md)), then a final refusal wrapping the last parse error. The sanitize layer (below) is untouched by the repair loop.
- An unrecognised/empty `SuggestedAction` is normalized to `"retry"` rather than treated as an error.

## Invariants

- The `Reflector` is stateless across calls; all context is passed explicitly.
- `agent/reflector.Reflector` satisfies `orchestration.Reflector` (compile-time check).
- `Reflection.Timestamp` is always set by the reflector.
- `SuggestedAction` is always one of `"retry"` / `"replan"` / `"abort"` after parsing.
- Reflections are append-only on the blackboard and returned in insertion order.

## Related Specs

- [README.md](README.md) — orchestration overview
- [planner.md](planner.md) — `SuggestedAction == "replan"` triggers `Planner.Replan`
- [executor.md](executor.md) — provides the trajectory via `TrajectoryStore`
- [conductor.md](conductor.md) — runs the steps the reflector analyses
- [../oneshot.md](../oneshot.md) — the one-shot client the reflection call runs through
- [../memory/blackboard.md](../memory/blackboard.md) — reflection persistence
