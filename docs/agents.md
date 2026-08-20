# Subagent Profiles

The `agents` package provides discovery, parsing, and management of **Subagent Profiles** — `AGENT.md` documents that declare specialized subagent personas and tool budgets. A Subagent Profile is to a delegated subagent what a [Skill](skills.md) is to an activated skill: a named, directory-scoped declaration whose YAML frontmatter carries metadata and whose markdown body is the agent's core directive (the system prompt applied when a subagent is launched under that profile).

```go
import "github.com/v0lka/sp4rk/agents"
```

The package mirrors the `skills` package's Manager/Parser design, but is intentionally self-contained: it imports only the Go standard library plus `gopkg.in/yaml.v3` and `log/slog`. It does not depend on any engine package — launching a subagent under a profile, applying its model/tools/max-steps, and enforcing `allow-redelegate` are execution-layer concerns (see [Subagents](subagents.md)).

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

An existing `AGENT.md` that fails parsing or validation is skipped and logged as a warning (`skipped invalid agent`) so a broken profile is visible to operators. Ordinary directories that do not contain `AGENT.md` are debug-level discovery noise and are ignored.

**Symlink following:** `Scan` follows symlinks that point to directories (resolved via `os.Stat`, since `os.ReadDir` reports symlinks as non-directories even when they point to one).

### List

```go
func (m *AgentManager) List() []AgentDescriptor
```

Returns lightweight descriptors for all discovered agents. Each descriptor carries the name, description, and a hidden flag, making it suitable for the discovery phase (`#`-autocomplete and the "Available Subagents" UI). Hidden agents are included here; the consumer filters them out as needed.

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
| unknown token, empty item, or duplicate group | error | fail-closed — the token is never silently dropped; a partially-dropped list could widen to the full toolset |

Valid group tokens come from `ToolGroupTokens()` (`execute`, `local-read`, `local-write`, `remote-read`, `remote-write`, `system`, `local-mcp`, `remote-mcp`). AGENT.md files are validated at parse time (both `ToolPreferenceWithError()` and the parser apply the same `validateToolsField` rules and messages); the error return guards profiles constructed programmatically.

The legacy widening method remains source-compatible:

```go
func (a *Agent) ToolPreference() any
```

It calls `ToolPreferenceWithError` and discards the error. Invalid programmatic metadata therefore maps to `nil`, which means the full toolset. Existing integrations keep compiling, but security-sensitive/new code must call `ToolPreferenceWithError()` and propagate the error.

> **Migration note:** an earlier public draft documented `ToolPreference() (any, error)`. The strict method is now named `ToolPreferenceWithError`; replace `pref, err := profile.ToolPreference()` with `pref, err := profile.ToolPreferenceWithError()`. The one-result `ToolPreference()` method is retained only for compatibility.

> **Migration note:** an earlier form of the `tools:` field accepted comma-separated tool **names** (e.g. `edit_file,bash_exec`). That form is now a validation error — the field accepts group tokens only. Express `edit_file,bash_exec` as `local-write,execute` (and `read_file` as `local-read`). A profile whose `tools:` fails validation is skipped from the catalog and logged as a warning (`skipped invalid agent`) rather than silently dropped.

## AgentDescriptor

`AgentDescriptor` is the lightweight discovery-time representation of an agent — name, description, and a hidden flag. It is what `List()` returns.

```go
type AgentDescriptor struct {
    Name        string `json:"name"`
    Description string `json:"description"`
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
| unknown keys | Silently ignored — not-yet-supported frontmatter fields do not break parsing. |

A profile whose `name` does not match its parent directory name is rejected. A missing opening or closing `---` frontmatter delimiter is a parse error; the `---` must be at the start of a line so that `---` inside YAML string values is not mistaken for a delimiter.

### ParseAgent

```go
func ParseAgent(agentMDPath, dirPath string) (*Agent, error)
```

Reads and validates an `AGENT.md` file. `agentMDPath` is the path to `AGENT.md`; `dirPath` is the absolute path to the agent directory (used for `DirPath` and name validation). On validation failure, returns a `*ParseError` describing the problem.

## How a profile configures a subagent

When a subagent is launched under a named profile, the execution layer resolves the profile and applies its fields to the launched subagent:

| Profile field | Applied to the subagent as |
| --- | --- |
| `Body` | the subagent's core directive / system prompt (replaces the generic orchestrator default) |
| `Tools` (`ToolPreferenceWithError`) | the subagent's tool budget |
| `MaxSteps` | the ReAct iteration cap (0/absent → derived from task complexity) |
| `Model` | forced via `agent.NewModelOverrideCaller` |
| `AllowRedelegate` | whether the subagent may launch further subagents |
| `Hidden` | discovery visibility only (never blocks invocation) |
| `Color` | UI badge accent (host/UI concern) |

### Forcing a per-agent model

`Model` is applied by wrapping the LLM caller so every call sets `req.Model` before the router resolves it:

```go
// agent.NewModelOverrideCaller wraps an LLMCaller so each Call forces req.Model.
caller := agent.NewModelOverrideCaller(router, agentProfile.Model)
```

`NewModelOverrideCaller(inner LLMCaller, model string)` returns `inner` unchanged when `model` is empty, so the override applies conditionally without callers branching on it. It relies on the [LLM router contract](llm-providers.md) that `req.Model` is only filled when empty — setting it beforehand bypasses the router's active-model selection for that caller while still routing to the active provider.

## Targeting a profile from a plan step

A plan step targets a profile via the `PlanStep.Agent` field (see [Planner](planner.md)). When non-empty, the step runs with that profile's system prompt, tools, max-steps, and model instead of the generic orchestrator defaults. The name is resolved by the execution layer (e.g. the Conductor's agent resolver). `Agent` round-trips through JSON so a declared plan survives blackboard persistence and task continuation.

`Agent` and the existing `Profile` field (`*planner.AgentProfile`) are independent: `Agent` targets a named, externally-declared profile by directory name, while `Profile` carries inline step-level configuration on the plan object itself.

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
