# Subagent Profiles

The `agents` package provides discovery, parsing, and management of **Subagent Profiles** — `AGENT.md` documents that declare specialized subagent personas and tool budgets. A Subagent Profile is to a delegated subagent what a [Skill](skills.md) is to an activated skill: a named, directory-scoped declaration whose YAML frontmatter carries metadata and whose markdown body is the agent's core directive (the system prompt applied when a subagent is launched under that profile).

```go
import "github.com/v0lka/sp4rk/agents"
```

The package mirrors the `skills` package's Manager/Parser design, but is intentionally self-contained: it imports only the Go standard library, `gopkg.in/yaml.v3`, and the engine's `safeio` package (which supplies the hardened, non-blocking `AGENT.md` read and its size cap). `safeio` is the `agents` package's only direct engine import, and itself pulls in only `pathutil` — launching a subagent under a profile, applying its model/tools/max-steps, and enforcing `allow-redelegate` are execution-layer concerns (see [Subagents](subagents.md)).

## AgentManager

`AgentManager` discovers, parses, and serves Subagent Profiles from configured directories. Directories are scanned in priority order; the first occurrence of an agent name wins, so higher-priority directories override lower-priority ones.

### NewAgentManager

```go
func NewAgentManager(dirs []string, logger *slog.Logger) *AgentManager
```

`dirs` is a list of discovery directories in priority order (highest priority first). Call `Scan()` to populate the catalog. A nil `logger` is tolerated and falls back to a default text handler.

```go
mgr := agents.NewAgentManager([]string{
    "/home/user/project/.agents/agents", // highest priority
    "/home/user/.agents/agents",         // user-level
    "/usr/local/share/agents/agents",    // system-level
}, logger)
if err := mgr.Scan(); err != nil {
    log.Fatal(err)
}
```

### Scan

```go
func (m *AgentManager) Scan() error
```

`Scan` walks all discovery directories and loads valid agents. It clears any existing catalog first, so it is safe to call repeatedly (e.g. after agents are added or removed). Directories are walked in **reverse priority order** so that higher-priority entries overwrite lower-priority ones with the same name.

An existing `AGENT.md` that fails parsing or validation is skipped and logged as a warning (`skipped invalid agent`) so a broken profile is visible to operators. An `AGENT.md` that exceeds the `safeio` read cap is skipped with its own warning (`skipped oversized agent`). Ordinary directories that do not contain `AGENT.md` are debug-level discovery noise and are ignored.

**Symlink following:** `Scan` follows symlinks that point to directories (resolved via `os.Stat`, since `os.ReadDir` reports symlinks as non-directories even when they point to one).

### List

```go
func (m *AgentManager) List() []AgentDescriptor
```

Returns lightweight descriptors for all discovered agents. Each descriptor carries the name, description, a hidden flag, and the profile's `model` override when set, making it suitable for the discovery phase (`#`-autocomplete and the "Available Subagents" UI). Hidden agents are included here; the consumer filters them out as needed.

### Get

```go
func (m *AgentManager) Get(name string) (*Agent, bool)
```

Returns the full `Agent` by name, or `(_, false)` if not found. Hidden agents are returned here just like any other — hiding affects discovery only, never invocation.

## Agent

`Agent` represents a fully loaded Subagent Profile — metadata, directive body, and filesystem path.

```go
type Agent struct {
    Metadata AgentMetadata
    Body     string // Markdown body after the YAML frontmatter (the agent's core directive)
    DirPath  string // Absolute path to the agent directory
}
```

`Descriptor()` returns the lightweight `AgentDescriptor` for this agent.

## AgentMetadata

`AgentMetadata` holds the parsed YAML frontmatter fields of an `AGENT.md` file.

```go
type AgentMetadata struct {
    Name            string `yaml:"name"`                       // required
    Description     string `yaml:"description"`                // required
    Tools           string `yaml:"tools,omitempty"`            // "all" (default) | "read-only" | comma-list
    MaxSteps        int    `yaml:"max-steps,omitempty"`        // ReAct cap; 0/absent = derive from complexity
    Model           string `yaml:"model,omitempty"`            // per-agent model override
    AllowRedelegate bool   `yaml:"allow-redelegate,omitempty"` // permit nested delegation (default false)
    Hidden          bool   `yaml:"hidden,omitempty"`           // hide from #-autocomplete (default false)
    Color           string `yaml:"color,omitempty"`            // UI accent color for the agent badge
    Skills          string `yaml:"skills,omitempty"`           // comma-list of required skill names (e.g. "code-review,git-conventions"); "" = none
}
```

Only the v1 fields above are modeled. Unknown frontmatter keys are silently ignored by the parser (unknown-field-ignore, not an error), so not-yet-supported fields (temperature/top-p, per-tool policy, mode, resources) do not break parsing.

### ToolPreferenceWithError

```go
func (a *Agent) ToolPreferenceWithError() (any, error)
```

Translates the declarative `tools` frontmatter field into the shape a delegating runtime expects (the `tools` field of a delegation task, consumed by the runtime's tool resolver). It returns `any` so its result can be assigned directly to a delegation task's `tools` field (itself typed `any`); the error return surfaces an invalid programmatically-constructed field instead of silently widening it:

| `tools` value | Returns | Meaning |
| --- | --- | --- |
| `""` or `"all"` (default; surrounding whitespace tolerated) | `nil`, no error | grant the full mutating toolset (a resolver treats nil as all tools) |
| `"read-only"` | `"read-only"`, no error | the read-only preset — a conforming delegating resolver grants local-read + remote-read on top of the always-included `system` group, with no MCP groups |
| comma-list of tool-group tokens (e.g. `"local-read,execute"`) | `[]string`, no error | granted capability groups (kebab-case; underscores normalize); a conforming resolver always adds the `system` group |
| unknown token, empty item, duplicate group, or a preset mixed into the comma-list — `"all"`/`"read-only"` are whole-field values only and `cannot be combined with other items` | error | fail-closed — the token is never silently dropped; a partially-dropped list could widen to the full toolset |

Valid group tokens come from `ToolGroupTokens()`, which returns every accepted `tools:` group token in canonical kebab-case, in the same order as `tools.AllToolGroups()` (`execute`, `local-read`, `local-write`, `remote-read`, `remote-write`, `system`, `local-mcp`, `remote-mcp`):

```go
func ToolGroupTokens() []string
func NormalizeToolGroupToken(token string) (string, bool)
```

`NormalizeToolGroupToken` is what implements the "underscores normalize" behaviour in the table above: it canonicalizes a single token, accepting both the kebab and the underscore spelling (`"local-read"` and `"local_read"`) and returning the canonical kebab form, with `false` as the second result for a token that is not a declared tool group. AGENT.md files are validated at parse time (both `ToolPreferenceWithError()` and the parser apply the same `validateToolsField` rules and messages); the error return guards profiles constructed programmatically.

The legacy widening method remains source-compatible:

```go
func (a *Agent) ToolPreference() any
```

It calls `ToolPreferenceWithError` and discards the error. Invalid programmatic metadata therefore maps to `nil`, which means the full toolset. Existing integrations keep compiling, but security-sensitive/new code must call `ToolPreferenceWithError()` and propagate the error.

> **Migration note:** an earlier public draft documented `ToolPreference() (any, error)`. The strict method is now named `ToolPreferenceWithError`; replace `pref, err := profile.ToolPreference()` with `pref, err := profile.ToolPreferenceWithError()`. The one-result `ToolPreference()` method is retained only for compatibility.

> **Migration note:** an earlier form of the `tools:` field accepted comma-separated tool **names** (e.g. `edit_file,bash_exec`). That form is now a validation error — the field accepts group tokens only. Express `edit_file,bash_exec` as `local-write,execute` (and `read_file` as `local-read`). A profile whose `tools:` fails validation is skipped from the catalog and logged as a warning (`skipped invalid agent`) rather than silently dropped.

### RequiredSkills

```go
func (a *Agent) RequiredSkills() ([]string, error)
```

Translates the declarative `skills` frontmatter field into the list of skill names a subagent launched under this profile must have activated:

| `skills` value | Returns | Meaning |
| --- | --- | --- |
| `""` (absent or empty) | `nil`, no error | the profile requires no particular skills |
| comma-list of skill names (e.g. `"code-review,git-conventions"`) | `[]string`, no error | trimmed skill names the runtime must have activated |
| invalid name shape, empty item (stray comma), or duplicate name | error | fail-closed — a typo fails the profile loudly instead of silently changing which skills the subagent gets |

Every item must have the same shape as an agent name (lowercase alphanumeric with hyphens, no leading/trailing hyphens); surrounding whitespace is tolerated. The parser applies the same `validateSkillsField` rules (and messages) at load time, so the parse-time and programmatic paths cannot diverge. This package only parses the names — whether a named skill actually exists is resolved by the delegating runtime against its skill catalog at launch time.

## AgentDescriptor

`AgentDescriptor` is the lightweight discovery-time representation of an agent — name, description, a hidden flag, and the profile's `model` override (when set). It is what `List()` returns.

```go
type AgentDescriptor struct {
    Name        string `json:"name"`
    Description string `json:"description"`
    Model       string `json:"model,omitempty"` // per-agent model override (mirrors AgentMetadata.Model)
    Hidden      bool   `json:"hidden,omitempty"`
}
```

## AGENT.md Format

A profile lives at `<agents-dir>/<name>/AGENT.md`. The file consists of YAML frontmatter delimited by `---` lines, followed by a markdown body.

```markdown
---
name: code-reviewer
description: A meticulous code reviewer that checks style, correctness, and tests.
tools: read-only
max-steps: 15
model: claude-sonnet-4-5
allow-redelegate: false
hidden: false
color: blue
---

# Code Reviewer

You review changesets for correctness, style adherence, and test coverage.
Report findings as a structured list. Do not modify files.
```

### Validation rules

`ParseAgent` enforces the following constraints:

| Field | Rule |
| --- | --- |
| `name` | Required. Lowercase alphanumeric and hyphens, no leading/trailing hyphens (regex `^[a-z0-9]([a-z0-9-]*[a-z0-9])?$`). Must match the parent directory name. |
| `description` | Required (non-empty). |
| `tools` | Optional. `"all"`/empty, `"read-only"`, or a comma-list of capability-group tokens (`ToolGroupTokens()`); validated at parse time (`validateToolsField`). |
| `skills` | Optional. Empty, or a comma-list of skill names with the same shape as `name`; validated at parse time (`validateSkillsField`). |
| unknown keys | Silently ignored — not-yet-supported frontmatter fields do not break parsing. |

A profile whose `name` does not match its parent directory name is rejected. A missing opening or closing `---` frontmatter delimiter is a parse error; the `---` must be at the start of a line so that `---` inside YAML string values is not mistaken for a delimiter.

### ParseAgent

```go
func ParseAgent(agentMDPath, dirPath string) (*Agent, error)
```

Reads and validates an `AGENT.md` file. `agentMDPath` is the path to `AGENT.md`; `dirPath` is the absolute path to the agent directory (used for `DirPath` and name validation). On validation failure, returns a `*ParseError` describing the problem. The file is read through `safeio.ReadFile`, which refuses a non-regular `AGENT.md` (e.g. a planted FIFO) without blocking on the open and refuses a file larger than `safeio.DefaultMaxFileSize` (10 MiB) instead of reading it into memory.

## How a profile configures a subagent

When a subagent is launched under a named profile, the execution layer resolves the profile and applies its fields to the launched subagent:

| Profile field | Applied to the subagent as |
| --- | --- |
| `Body` | the subagent's core directive / system prompt (replaces the generic orchestrator default) |
| `Tools` (`ToolPreferenceWithError`) | the subagent's tool budget |
| `MaxSteps` | the ReAct iteration cap (0/absent → derived from task complexity) |
| `Model` | forced via `agent.NewModelProviderOverrideCaller` (pins `req.Provider` + `req.Model` per call — see the [LLM provider contract](llm-providers.md) and [ADR-012](../specs/decisions/012-per-request-provider-override.md)); the legacy model-only `agent.NewModelOverrideCaller` is kept and delegates with an empty provider. Carried on `AgentDescriptor.Model` for discovery-time visibility |
| `AllowRedelegate` | whether the subagent may launch further subagents |
| `Hidden` | discovery visibility only (never blocks invocation) |
| `Color` | UI badge accent (host/UI concern) |

### Forcing a per-agent provider and model

`Model` is applied by wrapping the LLM caller so every call pins `req.Provider` and/or `req.Model` before the router resolves them:

```go
// agent.NewModelProviderOverrideCaller wraps an LLMCaller so each Call pins
// req.Provider and req.Model (each applies only when non-empty). The provider
// is a host-side routing decision (ADR-012) — profiles carry only model:.
caller := agent.NewModelProviderOverrideCaller(router, "openai_compatible", agentProfile.Model)
```

`NewModelProviderOverrideCaller(inner LLMCaller, provider, model string)` pins `req.Provider` and/or `req.Model` on every call; each override applies independently and only when non-empty, and both empty returns `inner` unchanged, so the override applies conditionally without callers branching on it. It relies on the [LLM router contract](llm-providers.md) that `req.Model` is only filled when empty — setting it beforehand bypasses the router's active-model selection for that caller — and that a non-empty `req.Provider` routes the call to the named configured provider (unknown name: immediate non-retryable error, zero attempts; see [ADR-012](../specs/decisions/012-per-request-provider-override.md)). A caller pinned this way is immune to a mid-task `SetModel`. `NewModelOverrideCaller(inner LLMCaller, model string)` is the legacy model-only form and never touches provider routing.

## Targeting a profile from a plan step

A plan step targets a profile via the `PlanStep.Agent` field (see [Planner](planner.md)). When non-empty, the step runs with that profile's system prompt, tools, max-steps, and model instead of the generic orchestrator defaults. The name is resolved by the **host's** execution layer — the SDK ships no resolver, and `orchestration.Conductor` passes `PlanStep.Agent` through untouched. `Agent` round-trips through JSON so a declared plan survives blackboard persistence and task continuation.

`Agent` and the existing `Profile` field are independent: `Profile` is typed `any` on `PlanStep` (it deserializes as `map[string]any`, and consumers convert it to a domain-specific profile such as `*planner.AgentProfile`) and carries inline step-level configuration on the plan object itself, while `Agent` targets a named, externally-declared profile by directory name.

## Complete Example

This example mirrors the skill-discovery flow: seed a sample agent profile on disk, scan the directory, and list the discovered agents.

```go
package main

import (
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/v0lka/sp4rk/agents"
)

func main() {
	// Create a temporary agents directory and seed a sample agent.
	baseDir, err := os.MkdirTemp("", "agents-*")
	if err != nil {
		log.Fatal(err)
	}
	defer os.RemoveAll(baseDir)

	agentDir := filepath.Join(baseDir, "code-reviewer")
	_ = os.MkdirAll(agentDir, 0o755)
	content := "---\n" +
		"name: code-reviewer\n" +
		"description: A meticulous code reviewer that checks style, correctness, and tests.\n" +
		"tools: read-only\n" +
		"---\n\n" +
		"# Code Reviewer\n\n" +
		"You review changesets for correctness, style adherence, and test coverage.\n"
	_ = os.WriteFile(filepath.Join(agentDir, "AGENT.md"), []byte(content), 0o644)

	// Discover agents from the directory.
	mgr := agents.NewAgentManager([]string{baseDir}, nil)
	if err := mgr.Scan(); err != nil {
		log.Fatal(err)
	}

	discovered := mgr.List()
	fmt.Printf("Discovered agents: %d\n", len(discovered))
	for _, a := range discovered {
		fmt.Printf("  • %s: %s\n", a.Name, a.Description)
	}

	// Fetch the full profile body.
	agent, ok := mgr.Get("code-reviewer")
	if ok {
		fmt.Printf("\nAgent body:\n%s\n", agent.Body)
		pref, err := agent.ToolPreferenceWithError()
		if err != nil {
			log.Fatalf("invalid tools field: %v", err)
		}
		fmt.Printf("Tool preference: %v\n", pref)
	}
}
```

Output:

```
Discovered agents: 1
  • code-reviewer: A meticulous code reviewer that checks style, correctness, and tests.

Agent body:
# Code Reviewer

You review changesets for correctness, style adherence, and test coverage.

Tool preference: read-only
```

## See also

- [Skills](skills.md) — the sibling package whose Manager/Parser design this mirrors
- [Subagents](subagents.md) — the delegated-execution primitive a profile configures
- [Planner](planner.md) — `PlanStep.Agent` targets a profile
- [LLM providers](llm-providers.md) — the model-override router contract `NewModelOverrideCaller` relies on
