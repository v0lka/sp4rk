# Router

## Role

Classifies user requests by domain, complexity, and matched skills, and (optionally) matched tools. The Router is a pure classification primitive: it produces a `RoutingDecision` and never mutates state. Downstream components (the host orchestrator, the planner, the context manager) consume the decision to select a compaction strategy, step profiles, and activated skills.

## Key Files

- `github.com/v0lka/sp4rk/agent/router` — `Router`, `Config`, `New`, `Route`, `RoutingDecision`, `SetReasoningEffort`, `SetToolMatching`, domain constants
- `github.com/v0lka/sp4rk/llm` — `LLMCaller`, `ExtractJSON`, `Message`
- `github.com/v0lka/sp4rk/prompt` — prompt builder (system prompt assembled with cache-aware substitutions)
- `github.com/v0lka/sp4rk/agent/router` (types) — the router defines its own minimal `SkillDescriptor{Name, Description}` used for matching; it does not import `skills`. (`skills.SkillDescriptor` is the richer type consumed by the planner.)

## Behavior

### Input

`Route(ctx, userMessage, availableTools, history, availableSkills)` takes:

- the user message (string)
- available tools (`[]tools.ToolDescriptor`, grouped by priority tier for the prompt)
- conversation history (the last `HistoryWindow` messages, default 10)
- available skill descriptors (name + description)

### Output: RoutingDecision

```go
type RoutingDecision struct {
    Domain             string   // "code" | "research" | "general" | "mixed"
    Complexity         int      // 1-5
    NeedsClarification bool     // retained from the JSON contract
    MatchedSkills      []string // skills selected by the router
    MatchedTools       []string // tools selected by the router (only when tool matching is enabled)
}
```

### Domain values

| Domain | Meaning | Compaction strategy (downstream) |
| ------ | ------- | -------------------------------- |
| `code` | File modifications, implementation, build/test | `sliding_window` |
| `research` | Information gathering, analysis | `summarization` |
| `general` | Mixed or unclear primary activity | `sliding_window` (`hierarchical` when complexity >= 4) |
| `mixed` | Explicitly mixed activities | `sliding_window` (`hierarchical` when complexity >= 4) |

### Complexity scale

Complexity (1–5) is advisory information for downstream components (e.g. host delegation guidance or planner step profiles). It is not mapped to a fixed step count by the Router itself.

### Skill matching

The router prompt includes the full list of available skills (name + description). The LLM selects which skills are relevant. `MatchedSkills` is deduplicated and trimmed during validation. Merging router-matched skills with explicitly user-activated skills is the host's responsibility, not the Router's.

### Tool matching (optional)

Semantic tool selection is opt-in. It is disabled by default and enabled either by setting `Config.ToolMatching = true` at construction or by calling `Router.SetToolMatching(true)`. When enabled:

- A tool-selection instruction is injected into the prompt (via the `TOOL-MATCHING` placeholder), directing the LLM to pick the relevant tools from the available set and return them in `matched_tools`.
- The JSON output schema (injected via the `JSON-OUTPUT-SCHEMA` placeholder) includes a `matched_tools` array alongside the base fields.
- The repair prompt used after an invalid-JSON retry echoes the active schema, including `matched_tools`.
- `RoutingDecision.MatchedTools` is populated, deduplicated, and trimmed during validation.

When disabled, the tool-selection instruction is omitted, the schema resolves to the default (without `matched_tools`), and behavior is identical to before the feature existed. `MatchedTools` is the host's to consume (e.g. to narrow the tool pool handed to a step); the Router itself does not modify the tool registry.

### Process

1. Build the system prompt from the caller-supplied template. The template must contain `AVAILABLE-TOOLS` and `AVAILABLE-SKILLS` placeholders. Tool/skill lists and any project-context section are substituted via single-pass data substitution (placeholders inside these externally-influenced values are never expanded). The template may optionally include `TOOL-MATCHING` and `JSON-OUTPUT-SCHEMA` placeholders; these resolve to in-package trusted constants (iterative substitution): when tool matching is off, `TOOL-MATCHING` resolves to empty and `JSON-OUTPUT-SCHEMA` resolves to the default schema. A template without these placeholders behaves exactly as before — `Replace` is a no-op for absent placeholders.
2. Construct messages: system + history (last `HistoryWindow`) + `"Classify this request: {msg}"`.
3. Apply the reasoning effort set via `SetReasoningEffort`.
4. Call the LLM.
5. Extract JSON from the response via `llm.ExtractJSON` (handles surrounding prose and markdown fences).
6. Unmarshal into `RoutingDecision`; validate and clamp.

### Validation rules

- Domain: must be one of `{"code", "research", "general", "mixed"}`; otherwise `"general"`.
- Complexity: clamped to `[1, 5]`.
- MatchedSkills: deduplicated, empty entries removed (first-occurrence order preserved).
- MatchedTools: deduplicated, empty entries removed (first-occurrence order preserved) — applied whenever present, regardless of whether tool matching is enabled.

### Optional dependencies

- `SetModelRegistry(*llm.ModelRegistry)` — model metadata resolution.
- `SetReasoningEffort(effort string)` — sets the reasoning effort applied to the LLM call.
- `Config.ToolMatching` / `SetToolMatching(enabled bool)` — enable semantic tool selection (see [Tool matching](#tool-matching-optional)); defaults to false. `Config.ToolMatching` is equivalent to calling `SetToolMatching(true)` at construction.
- `Config.AppendContextSections` — a function producing additional prompt sections (e.g. project conventions) inserted via a `PROJECT-CONTEXT` placeholder; if the template lacks that placeholder the section is appended for backward compatibility.

## Error Handling

- **LLM call failure**: returns an error wrapping the failure (no fallback routing).
- **JSON parse failure**: one retry with a repair prompt that echoes the active JSON output schema (including `matched_tools` when tool matching is enabled), asking the LLM to fix its JSON.
- **Second parse failure**: returns an error.

## Invariants

- `Route` always returns a valid `RoutingDecision` on success (never nil).
- Domain is always from the valid set after validation; complexity is always in `[1, 5]`.
- `MatchedSkills` and `MatchedTools` are always deduplicated and trimmed when non-empty; a nil or empty input is returned unchanged so callers can distinguish "not set" from an explicitly empty set.
- The Router never modifies the tool registry or any other state — it is pure classification.

## Related Specs

- [README.md](README.md) — orchestration overview
- [planner.md](planner.md) — consumes domain/complexity to drive exploration vs direct planning
- [../memory/compaction.md](../memory/compaction.md) — domain → strategy mapping
- [../skills.md](../skills.md) — skill discovery and descriptors
