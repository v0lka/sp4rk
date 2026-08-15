package sp4rk

import (
	"strings"
	"testing"

	"github.com/v0lka/sp4rk/agent"
	"github.com/v0lka/sp4rk/agents"
	"github.com/v0lka/sp4rk/skills"
	"github.com/v0lka/sp4rk/tools"
	"github.com/v0lka/sp4rk/tools/builtins"
	"github.com/v0lka/sp4rk/tools/builtins/websearch"
)

// expectedToolGroups pins the capability-group taxonomy for every built-in
// tool. The platform shell tool (bash_exec on Unix, posh_exec on Windows) is
// contributed by shellToolExpectations in the build-tagged files; MCP tools
// are transport-derived and covered in tools/mcp.
var expectedToolGroups = map[string]tools.ToolGroup{
	// local_read
	"read_file":      tools.GroupLocalRead,
	"list_directory": tools.GroupLocalRead,
	"glob":           tools.GroupLocalRead,
	"ripgrep":        tools.GroupLocalRead,
	// local_write
	"write_file":       tools.GroupLocalWrite,
	"edit_file":        tools.GroupLocalWrite,
	"create_directory": tools.GroupLocalWrite,
	"delete_file":      tools.GroupLocalWrite,
	"delete_directory": tools.GroupLocalWrite,
	// remote_read
	"web_fetch":  tools.GroupRemoteRead,
	"web_search": tools.GroupRemoteRead,
	// system
	"store_fact":          tools.GroupSystem,
	"search_facts":        tools.GroupSystem,
	"update_checklist":    tools.GroupSystem,
	"tool_result_read":    tools.GroupSystem,
	"read_step_output":    tools.GroupSystem,
	"list_step_outputs":   tools.GroupSystem,
	"read_final_result":   tools.GroupSystem,
	"read_attachment":     tools.GroupSystem,
	"semantic_search":     tools.GroupSystem,
	"batch":               tools.GroupSystem,
	"finish":              tools.GroupSystem,
	"read_skill_resource": tools.GroupSystem,
}

// allTools assembles every constructible built-in tool, including the
// parameterized ones AllBuiltinTools excludes plus the platform shell tool.
func allTools() []tools.Tool {
	ts := AllBuiltinTools()
	// Parameterized tools (excluded from AllBuiltinTools).
	ts = append(ts,
		builtins.NewWebFetchTool(builtins.WebFetchLimits{}),
		builtins.NewVectorSearchTool(nil, nil),
		websearch.NewTool(nil, websearch.Limits{}),
		agent.NewFinishTool(),
		skills.NewReadSkillResourceTool(nil),
	)
	return append(ts, platformShellTools()...)
}

// TestEveryBuiltinToolDeclaresValidGroup iterates ALL built-in tools and
// asserts each declares a valid (non-unknown, since none exists) group.
func TestEveryBuiltinToolDeclaresValidGroup(t *testing.T) {
	ts := allTools()
	if len(ts) < len(expectedToolGroups) {
		t.Fatalf("assembled %d tools, expected at least %d — construction list is stale", len(ts), len(expectedToolGroups))
	}
	for _, tool := range ts {
		g := tool.Group()
		if !tools.IsValidToolGroup(g) {
			t.Errorf("tool %q declares invalid/empty group %q", tool.Name(), g)
		}
	}
}

// TestBuiltinToolGroups_ExactMapping verifies each tool's exact group against
// the pinned taxonomy, so a group change is a deliberate, visible decision.
func TestBuiltinToolGroups_ExactMapping(t *testing.T) {
	expectations := make(map[string]tools.ToolGroup, len(expectedToolGroups)+1)
	for name, want := range expectedToolGroups {
		expectations[name] = want
	}
	for name, want := range shellToolExpectations {
		expectations[name] = want
	}
	seen := make(map[string]bool, len(expectations))
	for _, tool := range allTools() {
		name := tool.Name()
		want, ok := expectations[name]
		if !ok {
			t.Errorf("tool %q has no pinned group expectation — add it to expectedToolGroups", name)
			continue
		}
		seen[name] = true
		if got := tool.Group(); got != want {
			t.Errorf("tool %q group = %q, want %q", name, got, want)
		}
	}
	for name := range expectations {
		if !seen[name] {
			t.Errorf("expected tool %q was not assembled by allTools", name)
		}
	}
}

// TestAgentToolGroupTokensMatchToolGroups guards the deliberate duplication
// between the agents package's kebab-case `tools:` tokens (self-contained
// package, cannot import tools) and the authoritative tools.ToolGroup values.
// Drift between the two sets would silently reject valid profiles or accept
// groups that no tool can ever declare.
func TestAgentToolGroupTokensMatchToolGroups(t *testing.T) {
	t.Parallel()

	tokens := agents.ToolGroupTokens()
	groups := tools.AllToolGroups()
	if len(tokens) != len(groups) {
		t.Fatalf("agents.ToolGroupTokens() has %d entries, tools.AllToolGroups() has %d — sets drifted", len(tokens), len(groups))
	}
	for i, g := range groups {
		// The kebab token is the underscore ToolGroup value with '-' in place
		// of '_', in the same declaration order.
		want := strings.ReplaceAll(string(g), "_", "-")
		if tokens[i] != want {
			t.Errorf("token[%d] = %q, want %q (group %q)", i, tokens[i], want, g)
		}
		if _, ok := agents.NormalizeToolGroupToken(string(g)); !ok {
			t.Errorf("underscore spelling %q of group %q must be accepted by agents.NormalizeToolGroupToken", g, g)
		}
	}
}
