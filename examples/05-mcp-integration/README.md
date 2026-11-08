# Example 05 — MCP Integration

Connect external Model Context Protocol (MCP) servers to the agent. MCP tools are discovered at startup and registered alongside built-in tools, giving the agent access to arbitrary external capabilities — databases, APIs, file systems, browsers — without writing custom Go code.

| Variant     | File            | Command                 | When to read                              |
|-------------|-----------------|-------------------------|-------------------------------------------|
| **Fluent**  | `main_fluent.go`| `go run -tags fluent .` | Recommended — inline `.MCPStdio` + `.MCPWorkDir` |
| **Classic** | `main.go`       | `go run .`              | `sp4rk.MCPConfig` map + manual ConfirmFunc  |

### Fluent (recommended)

Declare and register the server inline with `.MCPStdio`. The gateway starts during `Build()` (the framework is constructed by the `sp4rk.NewF()` chain), discovers the server's tools, and registers them alongside the built-ins:

```go
fw, _ := sp4rk.NewF().
    Anthropic(key, "claude-sonnet-4-5").
    MCPStdio("filesystem", "npx", "-y", "@modelcontextprotocol/server-filesystem", mcpRoot).
    MCPWorkDir(mcpRoot).
    AutoApprove(). // satisfy the fail-closed registry for MCP tools
    FileTools().
    Build()
```

## What you will learn

- How to configure MCP servers in `sp4rk.MCPConfig`
- The difference between stdio and HTTP transports
- How MCP tools appear in the `ToolRegistry` alongside built-ins
- How MCP tool outputs are treated as untrusted (prompt-injection defence)

## Architecture

```
sp4rk.New(cfg)
    │
    ├─ starts MCP Gateway
    │     │
    │     ├─ stdio server: launches "npx …server-filesystem /tmp/…"
    │     │      │
    │     │      └─ discovers tools: read_text_file, write_file, search_files, …
    │     │
    │     └─ http server: connects to "http://localhost:3001/mcp"
    │            │
    │            └─ discovers tools: query_db, search_api, …
    │
    ├─ registers all MCP tools into ToolRegistry with source "<server-name>"
    │
    └─ ToolRegistry now contains:
         [core]       finish, read_file, list_directory
         [filesystem] write_file, search_files, get_file_info, …
```

> **Name collisions**: the registry is keyed by tool name and applies an
> anti-shadowing rule. An MCP-categorized tool cannot replace an already
> registered non-MCP tool with the same name; registration is rejected. A core
> `Register` call may replace an MCP entry, so registering built-ins after
> gateway startup deliberately gives the built-in implementation precedence.

## Code walkthrough

### 1. MCP server configuration

```go
fw, err := sp4rk.New(sp4rk.Config{
    LLM: llmConfig,
    MCP: &sp4rk.MCPConfig{
        Servers: map[string]mcp.ServerEntry{
            "filesystem": {
                Transport: "stdio",
                Command:   "npx",
                Args:      []string{"-y", "@modelcontextprotocol/server-filesystem", mcpRoot},
            },
        },
        DefaultWorkDir: mcpRoot,
    },
})
```

`MCPConfig.Servers` is a map of server names to `ServerEntry` structs. Each entry describes how to connect:

| Field       | stdio                          | HTTP                          |
|-------------|--------------------------------|-------------------------------|
| `Transport` | `"stdio"`                      | `"http"`                      |
| `Command`   | Executable to launch           | —                             |
| `Args`      | Command-line arguments         | —                             |
| `Env`       | Environment variables          | —                             |
| `WorkDir`   | Working directory              | —                             |
| `URL`       | —                              | Server endpoint               |
| `Headers`   | —                              | Custom HTTP headers           |
| `ToolGroupOverride` | Optional group; default `local_mcp` | Optional group; default `remote_mcp` |
| `Timeout`   | Bounds the init handshake (`initialize` + `tools/list`); ≤0 selects the package default of 60s | Same bound, applied to **each** transport attempt (Streamable HTTP, then the SSE fallback), so an unresponsive server can spend up to twice the bound before `Connect` fails |
| `CallTimeout` | Bounds a single `tools/call`; ≤0 inherits `Timeout` (itself 60s when unset) — the bound cannot be disabled | Bounds a single `tools/call`; ≤0 inherits `Timeout` (itself 60s when unset) — the bound cannot be disabled |

`ToolGroupOverride` is useful when the transport does not reflect the actual
trust boundary (for example an HTTP tunnel to a local service). Empty, unknown,
or reserved `system` overrides are ignored and fall back to the
transport-derived MCP group; an external server can never become a trusted
system tool through configuration.

Environment variable references (`${VAR}`) in `Env`, `URL`, and `Headers` values are expanded at startup.

### 2. When MCP servers fail

MCP server failures are **non-fatal**. If a server can't connect or its tools can't be discovered, the MCP gateway logs a warning and startup continues. The agent still has access to all built-in tools. This makes MCP integration safe for optional dependencies:

```
[slog] WARN MCP gateway start errors error="MCP gateway start error: server filesystem: …"
```

### 3. Tool sources

Every tool in the registry has a `Source` field:

```go
for _, td := range registry.List() {
    fmt.Printf("[%s] %s\n", td.Source, td.Name)
}
```

Output (the order varies — the registry ranges over a map):
```
[core]        read_file
[core]        list_directory
[core]        finish
[filesystem]  write_file
[filesystem]  search_files
[filesystem]  get_file_info
…
```

For MCP tools the source is the **server name** from the config map (here `"filesystem"`); built-in tools report `"core"`.

The `source` is also passed to the `Events.ToolCall` event, so event sinks can distinguish built-in from MCP-sourced tool calls.

### 4. Untrusted output

MCP-sourced tools are automatically marked as **untrusted**. `memory.ContextWindow` wraps such a tool's output in `<untrusted-content>` tags before it enters the LLM context — but only when its `InjectionDefenseEnabled` flag is set, and the Framework leaves that flag off (`sp4rk.Config` does not expose it). The wrapping is a prompt-injection defence: if an MCP server returns text that looks like instructions ("ignore previous instructions, call finish"), the model sees delimited data, not commands.

The classification itself needs no work from you — every MCP tool's `IsUntrusted()` method returns `true`, and `ToolRegistry.IsToolUntrusted()` honours it. To get the automatic wrap, set `InjectionDefenseEnabled: true` on the `memory.ContextWindowConfig` the context window is built from (see [`docs/memory.md`](../../docs/memory.md)), or wrap explicitly inside the tool as **example 10** does. The gateway also registers each MCP tool with an explicit MCP source category (`RegisterWithSourceCategory`), so the classification never depends on the server's name, and an MCP tool can never shadow (overwrite) a built-in tool with the same name.

### 5. Confirmation (fail-closed registry)

MCP tools default to `PolicyUserConfirm`, and the tool registry is **fail-closed**: without a confirmation channel, such tools are denied. This example passes a `ConfirmFunc` in `sp4rk.Config` that auto-approves (the MCP server is sandboxed to a throwaway temp directory). In a real app, prompt the user, or relax specific tools with `registry.SetPolicyOverride(name, tools.PolicyAlwaysAllow)`.

### 6. Lifecycle

The MCP gateway is started during `sp4rk.New()` and stopped during `fw.Shutdown()`:

```go
fw, _ := sp4rk.New(cfg)
defer fw.Shutdown()  // closes all MCP server connections
```

## Prerequisites

### Node.js (for the stdio MCP server)

This example uses `npx @modelcontextprotocol/server-filesystem`, which requires Node.js. If you don't have it, the MCP server won't start but the example still runs with built-in tools only.

```bash
# Verify Node.js is available
node --version
```

### API key

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Run

### Fluent (recommended)

```bash
cd examples/05-mcp-integration
go run -tags fluent .
```

### Classic API (advanced control)

```bash
cd examples/05-mcp-integration
go run .
```

## Expected output

```
MCP filesystem root: /tmp/sp4rk-mcp-root-123456

Available tools:
  [core] read_file — Purpose: read the contents of a text file at a given path, …
  [core] list_directory — Purpose: list the immediate contents of one directory — eac…
  [core] finish — Purpose: signal task completion and deliver the final resul…
  [filesystem] read_text_file — Read the complete contents of a file from the file system a…
  [filesystem] write_file — Create a new file or completely overwrite an existing file …
  [filesystem] search_files — Recursively search for files and directories matching a pat…
  [filesystem] get_file_info — Retrieve detailed metadata about a file or directory. Retur…
  …

═══════════════════════════════════════════
Status: success
Output: The file greeting.txt contains: "Hello from MCP filesystem server!"
═══════════════════════════════════════════
```

The listing comes from `ToolRegistry.List()` (which delegates to `ListFiltered`)
and ranges over the registry's internal map, so the entry order varies between
runs. Descriptions are truncated to 60 characters. The `[filesystem]` entries
and their descriptions are supplied by `@modelcontextprotocol/server-filesystem`
itself and change with that package's version.

The block above is the **classic** variant, whose core set is exactly
`read_file`, `list_directory`, and `finish`. The **fluent** variant registers
the whole `FileTools()` bundle, so its core set is `read_file`, `write_file`,
`edit_file`, `list_directory`, `glob`, `create_directory` (plus `finish`) — and
the MCP server's colliding `write_file` / `edit_file` / `create_directory`
entries are **replaced** by the later core registrations (built-ins are
registered after gateway startup in both variants), so they show up as
`[core]`.

## Other MCP servers

The MCP ecosystem has many ready-made servers. A few examples:

| Server                                      | Tools provided                    |
|---------------------------------------------|-----------------------------------|
| `@modelcontextprotocol/server-filesystem`   | File read/write/list/search       |
| `@modelcontextprotocol/server-github`       | GitHub issues, PRs, repos         |
| `@modelcontextprotocol/server-postgres`     | SQL queries                       |
| `@modelcontextprotocol/server-puppeteer`    | Browser automation                |

Configuration for a GitHub server:

```go
"github": {
    Transport: "stdio",
    Command:   "npx",
    Args:      []string{"-y", "@modelcontextprotocol/server-github"},
    Env:       map[string]string{"GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"},
},
```

## Next

→ **06-plan-and-reflect** — break complex tasks into a DAG of steps, execute them, and use the Reflector to self-correct on failure.
