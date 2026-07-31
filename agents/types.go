// Package agents provides discovery, parsing, and management of Subagent
// Profiles — AGENT.md files that declare specialized subagent personas and
// tool budgets. A Subagent Profile is to a delegated subagent what a Skill
// (see package github.com/v0lka/sp4rk/skills) is to an activated skill.
//
// A profile lives at `<agents-dir>/<name>/AGENT.md` (analogous to
// `<skills-dir>/<name>/SKILL.md` for skills). Its YAML frontmatter carries the
// agent's metadata (name, description, tool preference, step cap, model
// override, redelegation permission, visibility, badge color); the markdown
// body is the agent's core directive — the system prompt applied when a
// subagent is launched under that profile.
//
// The package is intentionally self-contained: it imports only the Go standard
// library plus gopkg.in/yaml.v3 and log/slog. It mirrors the skills package's
// Manager/Parser design.
package agents

import "strings"

// Agent represents a fully loaded Subagent Profile — metadata, directive body,
// and the filesystem path of its directory.
type Agent struct {
	Metadata AgentMetadata
	Body     string // Markdown body after the YAML frontmatter (the agent's core directive)
	DirPath  string // Absolute path to the agent directory
}

// AgentMetadata holds the parsed YAML frontmatter fields of an AGENT.md file.
//
// Only the v1 fields are modeled here. Unknown frontmatter keys are silently
// ignored by the parser (unknown-field-ignore, not an error) so that
// not-yet-supported fields (temperature/top-p, per-tool policy, mode, resources)
// do not break parsing.
type AgentMetadata struct {
	Name            string `yaml:"name"`                       // required: agent identifier, must match dir name
	Description     string `yaml:"description"`                // required: #-autocomplete + "Available Subagents"
	Tools           string `yaml:"tools,omitempty"`            // "all" (default) | "read-only" | comma-list of mutating tools
	MaxSteps        int    `yaml:"max-steps,omitempty"`        // ReAct iteration cap; 0/absent = derive from complexity
	Model           string `yaml:"model,omitempty"`            // per-agent model override
	AllowRedelegate bool   `yaml:"allow-redelegate,omitempty"` // permit nested delegation (default false)
	Hidden          bool   `yaml:"hidden,omitempty"`           // hide from #-autocomplete (default false)
	Color           string `yaml:"color,omitempty"`            // UI accent color for the agent badge
}

// AgentDescriptor is the lightweight discovery-time representation of an agent
// (name + description + hidden flag). It is returned by List() for use by
// #-autocomplete and the "Available Subagents" UI, where hidden agents are
// filtered out by the consumer.
type AgentDescriptor struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Hidden      bool   `json:"hidden,omitempty"`
}

// Descriptor returns the lightweight AgentDescriptor for this agent.
func (a *Agent) Descriptor() AgentDescriptor {
	return AgentDescriptor{
		Name:        a.Metadata.Name,
		Description: a.Metadata.Description,
		Hidden:      a.Metadata.Hidden,
	}
}

// ToolPreference translates the declarative `tools` frontmatter field into the
// shape a delegating runtime expects (e.g. the `tools` field of a delegation
// task, consumed by the runtime's tool resolver):
//
//   - "" or "all" (the default) → nil, meaning "grant the full mutating
//     toolset" (a resolver treats a nil request as all tools).
//   - "read-only" → the string "read-only", meaning "mandatory read-only/MCP
//     base only".
//   - a comma-separated list (e.g. "edit_file,bash_exec") → a []string of the
//     individual mutating tool names. The read-only/MCP base is always added on
//     top by the resolver. NOTE: a consumer that expects []any can convert the
//     returned []string accordingly.
//
// The method returns `any` so its result can be assigned directly to a
// delegation task's `tools` field (which is itself typed `any`).
func (a *Agent) ToolPreference() any {
	tools := a.Metadata.Tools
	switch tools {
	case "", "all":
		return nil
	case "read-only":
		return "read-only"
	default:
		var names []string
		for _, part := range strings.Split(tools, ",") {
			if name := strings.TrimSpace(part); name != "" {
				names = append(names, name)
			}
		}
		if len(names) == 0 {
			return nil
		}
		return names
	}
}
