package tools

// ToolGroup classifies a tool by the principal kind of capability or side
// effect it exposes. It is a coarse, security-relevant taxonomy used for
// group-based tool selection (e.g. read-only subagent profiles) and policy
// reasoning — not a display label.
//
// Every tool belongs to exactly one group; there is deliberately no "unknown"
// value. Tools that embed [BaseTool] must set its ToolGroup field; a tool
// whose group is not declared matches no group-based allow-list (fail-closed
// for group filtering).
type ToolGroup string

const (
	// GroupExecute marks tools that execute arbitrary shell commands
	// (bash_exec, posh_exec). The highest-uncertainty capability: a single
	// call can read, write, and exfiltrate.
	GroupExecute ToolGroup = "execute"
	// GroupLocalRead marks tools that only read local filesystem state
	// (read_file, list_directory, glob, ripgrep).
	GroupLocalRead ToolGroup = "local_read"
	// GroupLocalWrite marks tools that mutate local filesystem state
	// (write_file, edit_file, delete_file, create_directory, delete_directory).
	GroupLocalWrite ToolGroup = "local_write"
	// GroupRemoteRead marks tools that read data from remote/network sources
	// (web_fetch, web_search).
	GroupRemoteRead ToolGroup = "remote_read"
	// GroupRemoteWrite marks tools that send or mutate data on remote systems.
	// No sp4rk builtin currently declares it; the group exists so that
	// outbound-mutating tools (custom tools, future builtins) get a truthful
	// tag instead of degrading into a less-accurate one.
	GroupRemoteWrite ToolGroup = "remote_write"
	// GroupSystem marks orchestration/state tools with no direct filesystem
	// or network side effect of their own (fact memory, checklist, step
	// outputs, tool-result cache, attachments, vector search, batch, finish,
	// skill resources).
	GroupSystem ToolGroup = "system"
	// GroupLocalMCP marks tools served by an MCP server running as a local
	// process over the stdio transport.
	GroupLocalMCP ToolGroup = "local_mcp"
	// GroupRemoteMCP marks tools served by a remote MCP server reached over
	// the http transport.
	GroupRemoteMCP ToolGroup = "remote_mcp"
)

// toolGroups is the authoritative set of declared groups. Membership here is
// the definition of "declared": a group value outside this set is treated as
// not declared by every group-based check.
var toolGroups = map[ToolGroup]bool{
	GroupExecute:     true,
	GroupLocalRead:   true,
	GroupLocalWrite:  true,
	GroupRemoteRead:  true,
	GroupRemoteWrite: true,
	GroupSystem:      true,
	GroupLocalMCP:    true,
	GroupRemoteMCP:   true,
}

// AllToolGroups returns every declared ToolGroup, in declaration order.
func AllToolGroups() []ToolGroup {
	return []ToolGroup{
		GroupExecute,
		GroupLocalRead,
		GroupLocalWrite,
		GroupRemoteRead,
		GroupRemoteWrite,
		GroupSystem,
		GroupLocalMCP,
		GroupRemoteMCP,
	}
}

// IsValidToolGroup reports whether g is one of the declared groups.
func IsValidToolGroup(g ToolGroup) bool {
	return toolGroups[g]
}

// MCPToolGroup returns the ToolGroup for an MCP server transport type
// ("stdio" | "http"): stdio servers run as local processes → local_mcp,
// http servers are remote → remote_mcp. Transport "" (unset) and any other
// value default to the stdio/local mapping, mirroring Server.Connect's
// defaulting of an unspecified transport to stdio. The mcp package uses this
// to tag server tools unless the operator set a per-server override.
func MCPToolGroup(transport string) ToolGroup {
	if transport == "http" {
		return GroupRemoteMCP
	}
	return GroupLocalMCP
}
