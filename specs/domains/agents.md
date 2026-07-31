# Subagent Profiles

## Purpose

Discovery, parsing, and management of **Subagent Profiles** — `AGENT.md` documents that declare specialized subagent personas and tool budgets. A Subagent Profile is to a delegated subagent what a [Skill](skills.md) is to an activated skill: a named, directory-scoped declaration whose YAML frontmatter carries metadata and whose markdown body is the agent's core directive (the system prompt applied when a subagent is launched under that profile). `AgentManager` discovers profiles from configured directories, parses them, and serves their metadata/bodies. The package is intentionally self-contained and mirrors the `skills` package's Manager/Parser design.

## Key Files

- `github.com/v0lka/sp4rk/agents` — `AgentManager`, `NewAgentManager`, `Scan`, `List`, `Get`
- `github.com/v0lka/sp4rk/agents` (parsing) — `ParseAgent`, `ParseError`, `parseFrontmatter`
- `github.com/v0lka/sp4rk/agents` (types) — `Agent`, `AgentMetadata`, `AgentDescriptor`, `ToolPreference`

## Core Types

```go
type AgentManager struct { /* unexported */ }

// A fully loaded Subagent Profile — metadata, directive body, and directory path.
type Agent struct {
    Metadata AgentMetadata
    Body     string // markdown body after the YAML frontmatter (the agent's core directive)
    DirPath  string // absolute path to the agent directory
}

type AgentMetadata struct {
    Name            string `yaml:"name"`                        // required
    Description     string `yaml:"description"`                 // required
    Tools           string `yaml:"tools,omitempty"`             // "all"(default) | "read-only" | comma-list
    MaxSteps        int    `yaml:"max-steps,omitempty"`         // ReAct cap; 0/absent = derive
    Model           string `yaml:"model,omitempty"`             // per-agent model override
    AllowRedelegate bool   `yaml:"allow-redelegate,omitempty"` // nested delegation (default false)
    Hidden          bool   `yaml:"hidden,omitempty"`            // hide from #-autocomplete (default false)
    Color           string `yaml:"color,omitempty"`             // UI accent color for the agent badge
}

// Lightweight discovery-time representation — what List() returns.
type AgentDescriptor struct {
    Name        string `json:"name"`
    Description string `json:"description"`
    Hidden      bool   `json:"hidden,omitempty"`
}
```

## Flow

```
NewAgentManager(dirs []string, logger)        // dirs in priority order, highest first
  │
  ├─ Scan()                                   // clears catalog, then walks dirs in reverse
  │      priority order so higher-priority entries overwrite lower ones of the same
  │      name; invalid AGENT.md files are logged at debug and skipped
  │        ├─ follows directory symlinks (resolved via os.Stat)
  │        └─ ParseAgent each AGENT.md → validate → catalog (keyed by name)
  │
  ├─ List()  → []AgentDescriptor              // name + description + hidden flag
  └─ Get(name) → (*Agent, bool)               // full body + metadata
```

`Descriptor()` returns the lightweight `AgentDescriptor` for one agent; `List()` collects them for discovery (#-autocomplete and the "Available Subagents" UI), where hidden agents are filtered out by the consumer. `Get()` returns the full profile including hidden agents — hiding affects discovery only, never invocation.

## AGENT.md format & validation

A profile lives at `<agents-dir>/<name>/AGENT.md`. The file is YAML frontmatter (delimited by `---`) followed by a markdown body. `ParseAgent(agentMDPath, dirPath)` reads, splits, and validates it, returning a `*Agent` or a `*ParseError`.

| Field | Rule |
| ----- | ---- |
| `name` | Required. Lowercase alphanumeric and hyphens, no leading/trailing hyphens (`^[a-z0-9]([a-z0-9-]*[a-z0-9])?$`). Must match the parent directory name. |
| `description` | Required (non-empty). |
| unknown keys | Silently ignored — not-yet-supported frontmatter fields (temperature/top-p, per-tool policy, mode, resources) do not break parsing. |

A profile whose `name` does not match its parent directory name is rejected. A missing opening or closing `---` frontmatter delimiter is a parse error; the `---` must be at the start of a line so that `---` inside YAML string values is not mistaken for a delimiter.

## ToolPreference

`ToolPreference()` translates the declarative `tools` frontmatter field into the shape a delegating runtime expects (the `tools` field of a delegation task, consumed by the runtime's tool resolver). It returns `any` so its result can be assigned directly to a delegation task's `tools` field:

| `tools` value | Returns | Meaning |
| ------------- | ------- | ------- |
| `""` or `"all"` (default) | `nil` | grant the full mutating toolset (resolver treats nil as all tools) |
| `"read-only"` | `"read-only"` | mandatory read-only/MCP base only |
| comma-list (e.g. `"edit_file,bash_exec"`) | `[]string` | named mutating tools; the read-only/MCP base is always added on top by the resolver |

## How a profile configures a subagent

When a subagent is launched under a named profile, the execution layer applies the profile's fields:

| Field | Applied as |
| ----- | ---------- |
| `Body` | the subagent's core directive / system prompt |
| `Tools` (`ToolPreference`) | the subagent's tool budget |
| `MaxSteps` | the ReAct iteration cap (0/absent → derived from task complexity) |
| `Model` | forced via `agent.NewModelOverrideCaller` (wraps the caller so `req.Model` is set before the router resolves it) |
| `AllowRedelegate` | whether the subagent may launch further subagents |
| `Hidden` | discovery visibility only (does not affect invocation) |
| `Color` | UI badge accent (host/UI concern) |

`NewModelOverrideCaller(inner LLMCaller, model string)` returns `inner` unchanged when `model` is empty, so the override applies conditionally without callers branching on it. It relies on the [LLM router contract](../contracts/llm-providers.md) that `req.Model` is only filled when empty — setting it beforehand bypasses the router's active-model selection for that caller.

A plan step targets a profile via the `PlanStep.Agent` field (see [orchestration/planner.md](orchestration/planner.md)); the name is resolved by the execution layer (e.g. the Conductor's agent resolver).

## Invariants

- Directories are scanned in **reverse priority order** so higher-priority entries win on name collision (first occurrence / highest priority wins).
- `Scan` is idempotent (clears the catalog first); safe to call repeatedly.
- Invalid `AGENT.md` files are skipped at debug level, never fatal; a directory that is not an agent is ignored.
- `Scan` follows directory symlinks.
- A profile's `name` must match its parent directory name.
- `List()` returns lightweight descriptors including hidden agents (the consumer filters hidden out); `Get()` returns the full profile for any name, hidden or not.
- Unknown frontmatter keys are ignored, not errors.

## Configuration

`NewAgentManager(dirs []string, logger)` takes discovery directories in priority order (highest first). A nil logger is tolerated and falls back to a default text handler. The host wires the manager and resolves profile names to profiles at execution time — the SDK's `AgentManager` only discovers/parses/serves; launching a subagent under a profile, applying its model/tools/max-steps, and enforcing `allow-redelegate` are execution-layer concerns (see [orchestration/subagents.md](orchestration/subagents.md)).

## Extension Points

- **Custom discovery**: pass a resolved directory list to `NewAgentManager`; layer multiple roots for priority overrides.
- **Additional frontmatter**: declare new fields in `AgentMetadata`; unknown keys are ignored until modeled, so profiles with not-yet-supported fields still parse.
- **Integration**: the host binds `AgentManager.Get(name)` to its delegation tool's agent resolver, mapping each profile field onto the subagent it launches.

## Related Specs

- [skills.md](skills.md) — the sibling package whose Manager/Parser design this mirrors
- [orchestration/subagents.md](orchestration/subagents.md) — the delegated-execution primitive a profile configures
- [orchestration/planner.md](orchestration/planner.md) — `PlanStep.Agent` targets a profile
- [../contracts/llm-providers.md](../contracts/llm-providers.md) — the model-override router contract `NewModelOverrideCaller` relies on
