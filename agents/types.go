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

import (
	"fmt"
	"strings"
)

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
	Tools           string `yaml:"tools,omitempty"`            // "all" (default) | "read-only" | comma-list of tool GROUPS (kebab-case, e.g. "local-read,execute")
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

// ToolPreference returns the tool preference for this agent in the shape a
// delegating runtime expects (see [Agent.ToolPreferenceWithError] for the full
// contract). It is the backward-compatible form: an invalid `tools` field is
// silently mapped to nil (meaning "grant the full toolset"). Callers that must
// distinguish an invalid `tools` field from the default should use
// [Agent.ToolPreferenceWithError] instead.
func (a *Agent) ToolPreference() any {
	pref, _ := a.ToolPreferenceWithError()
	return pref
}

// ToolPreferenceWithError translates the declarative `tools` frontmatter field
// into the shape a delegating runtime expects (e.g. the `tools` field of a
// delegation task, consumed by the runtime's tool resolver):
//
//   - "" or "all" (the default) → nil, meaning "grant the full toolset"
//     (a resolver treats a nil request as all tools).
//   - "read-only" → the string "read-only", meaning the read-only preset
//     (local-read + remote-read groups on top of the always-granted system
//     group — no MCP).
//   - a comma-separated list of tool-group tokens (e.g. "local-read,execute")
//     → a []string of canonical kebab-case group tokens. The resolver always
//     adds the system group on top of the granted groups. NOTE: a consumer
//     that expects []any can convert the returned []string accordingly.
//
// The method returns `any` so its result can be assigned directly to a
// delegation task's `tools` field (which is itself typed `any`). Surrounding
// whitespace is tolerated. Any token that is not a declared tool group — or a
// duplicate group after canonicalization — yields an error instead of being
// silently dropped: silently dropping tokens could turn a narrow request into
// the full toolset. The list form is validated with the same rules (and the
// same error messages) the parser applies to AGENT.md files (see
// validateToolsField), so the parse-time and programmatic paths can never
// diverge; the error return matters for profiles built programmatically,
// which never pass through the parser.
func (a *Agent) ToolPreferenceWithError() (any, error) {
	tools := strings.TrimSpace(a.Metadata.Tools)
	switch tools {
	case "", "all":
		return nil, nil
	case "read-only":
		return "read-only", nil
	default:
		if err := validateToolsField(tools); err != nil {
			return nil, err
		}
		// Validation passed, so every item canonicalizes cleanly.
		var groups []string
		for _, part := range strings.Split(tools, ",") {
			canonical, _ := NormalizeToolGroupToken(part)
			groups = append(groups, canonical)
		}
		return groups, nil
	}
}

// toolGroupTokens are the accepted `tools:` group tokens in canonical
// kebab-case spelling. They mirror the sdktools ToolGroup values ("local_read"
// etc. — underscore there, kebab here) one-to-one. This package is deliberately
// self-contained (stdlib + yaml only), so the set is duplicated rather than
// imported from the tools package; a root-level test cross-checks the two sets
// against each other to catch drift.
var toolGroupTokens = []string{
	"execute", "local-read", "local-write", "remote-read", "remote-write",
	"system", "local-mcp", "remote-mcp",
}

// ToolGroupTokens returns every accepted `tools:` group token in canonical
// kebab-case, in the same order as tools.AllToolGroups().
func ToolGroupTokens() []string {
	return append([]string(nil), toolGroupTokens...)
}

// NormalizeToolGroupToken canonicalizes a single tool-group token: kebab and
// underscore spellings ("local-read" / "local_read") are both accepted and
// normalize to the kebab form. The second return value is false for tokens
// that are not declared tool groups.
func NormalizeToolGroupToken(token string) (string, bool) {
	canonical := strings.ReplaceAll(strings.TrimSpace(token), "_", "-")
	for _, t := range toolGroupTokens {
		if canonical == t {
			return canonical, true
		}
	}
	return "", false
}

// validateToolsField validates the declarative `tools` frontmatter field:
// "" or "all" (full toolset), "read-only" (read-only preset), or a non-empty
// comma list of tool-group tokens. Surrounding whitespace is tolerated.
// "all"/"read-only" cannot be mixed with group tokens, empty items (stray
// commas) are rejected, and a group repeated after canonicalization (including
// mixed kebab/underscore spellings of the same group) is rejected — a typo or
// a stray duplicate must fail the profile loudly, not silently widen or
// narrow the toolset.
func validateToolsField(tools string) error {
	switch strings.TrimSpace(tools) {
	case "", "all", "read-only":
		return nil
	}
	seen := make(map[string]struct{}, 4)
	for _, part := range strings.Split(tools, ",") {
		token := strings.TrimSpace(part)
		if token == "" {
			return fmt.Errorf("tools: empty item in group list %q", tools)
		}
		if token == "all" || token == "read-only" {
			return fmt.Errorf("tools: %q cannot be combined with other items in %q", token, tools)
		}
		canonical, ok := NormalizeToolGroupToken(token)
		if !ok {
			return fmt.Errorf("tools: unknown tool group %q (valid groups: %s; also accepted: \"all\", \"read-only\")",
				token, strings.Join(toolGroupTokens, ", "))
		}
		if _, dup := seen[canonical]; dup {
			return fmt.Errorf("tools: duplicate group %q in %q", token, tools)
		}
		seen[canonical] = struct{}{}
	}
	return nil
}
