# sp4rk Agent SDK — Examples
A progression of 11 examples that show how to build AI agents with the `github.com/v0lka/sp4rk` Go framework, from a minimal single-call agent through focused subsystem deep-dives to a full-stack system that exercises every major SDK subsystem.

## Layout

```
examples/
├── go.mod                      # standalone module — imports the SDK as an external dep
├── go.sum                      # dependency checksums for that module
├── README.md                   # this file
├── 01-minimal-agent/           # minimal full agent (Framework + finish + Execute)
├── 02-custom-tools/            # custom Tool implementation + built-in tools
├── 03-event-streaming/         # custom Events sink for live execution observability
├── 04-human-in-the-loop/       # custom HITLHandler for tool-call confirmation
├── 05-mcp-integration/         # Model Context Protocol server integration
├── 06-plan-and-reflect/        # Planner → DAG → Conductor → Reflector orchestration
├── 07-multi-provider-routing/  # multi-provider Router, runtime model switching
├── 08-parallel-subagents/      # RunSubAgent / RunSubAgentsParallel concurrency
├── 09-context-memory/          # ContextWindow, compaction strategies, fill events
├── 10-security-and-safety/     # prompt-injection defense, ToolJudger, ToolPolicy
└── 11-full-power/              # every SDK subsystem combined in one agent
```

## Progression

| #  | Example              | Concepts introduced                                             |
|----|----------------------|-----------------------------------------------------------------|
| 01 | minimal-agent        | `sp4rk.New`, `Framework.Execute`, `FinishTool`, system prompt     |
| 02 | custom-tools         | `tools.Tool`, optional `GroupProvider`, `BaseTool`, capability groups, built-in tools |
| 03 | event-streaming      | `Events` interface, live thought/tool/result observation        |
| 04 | human-in-the-loop    | `HITLHandler`, tool-call confirmation, step-limit decisions     |
| 05 | mcp-integration      | `MCPConfig`, stdio/HTTP MCP servers, external tool discovery    |
| 06 | plan-and-reflect     | `Planner`, DAG execution, `Reflector`, retry loop (+ abort)     |
| 07 | multi-provider-routing | `Router`, `SetModel`, composite IDs, phase-based `Models()`   |
| 08 | parallel-subagents   | `RunSubAgent`/`RunSubAgentsParallel`, per-step Executor/CM     |
| 09 | context-memory       | `ContextWindow`, compaction strategies, `ContextFill` events   |
| 10 | security-and-safety  | `WrapUntrustedContent`, `ToolJudger`, `JudgeStrict`, fail-safe confirmation |
| 11 | full-power           | multi-provider, Plan→Reflect, MCP, events, HITL, skills, Subagent Profiles, compaction, facts |

Each example is a self-contained `package main` with its own `README.md`.

## Prerequisites

### Go

Go 1.27.3+ is required (matches the SDK's `go.mod`).

### API keys

Live agent runs need at least one LLM provider API key:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
# or
export OPENAI_API_KEY="sk-..."
```

Examples default to Anthropic (`claude-sonnet-4-5`). See each example's README for provider-specific configuration.

### Module setup

The examples live in a **separate Go module** (`sp4rk-examples`) that imports the SDK as an external dependency via a `replace` directive:

```go
// go.mod
require github.com/v0lka/sp4rk v0.0.0-00010101000000-000000000000
replace github.com/v0lka/sp4rk => ..
```

This lets the examples compile against the local SDK source tree without publishing to a module proxy. The `..` points to the SDK module root — the repository root, which declares `module github.com/v0lka/sp4rk` — one directory above `examples/`. A real consumer would simply run:

```bash
go get github.com/v0lka/sp4rk@latest
```

Before running an example for the first time, resolve dependencies:

```bash
cd examples
GOWORK=off go mod tidy
```

## Offline build and tests

The module compiles and its local tests run without API keys, LLM requests, or
an MCP server. `GOWORK=off` is required when the parent checkout is part of a
workspace that does not include the standalone examples module:

```bash
cd examples
GOWORK=off go build ./...
GOWORK=off go test ./...
```

The tests exercise deterministic public contracts such as capability-group
metadata, strict-judge fail-safe behavior, and Subagent Profile tool-group
parsing. Example 10 also prints its deterministic security demonstrations before
its optional live-agent portion; the other `go run` commands are live examples
and need the provider credentials listed above.

## Running an example

Examples 01–10 each ship in **two variants** selected by a build tag:

| Command                 | Variant  | File            |
|-------------------------|----------|-----------------|
| `go run -tags fluent .` | Fluent   | `main_fluent.go` (`//go:build fluent`) — recommended, concise |
| `go run .`              | Classic  | `main.go` (`//go:build !fluent`) — full low-level control        |

```bash
cd examples/01-minimal-agent
go run -tags fluent .   # recommended (fluent API)
go run .                # classic SDK API
```

Both variants build the same *type* (`*sp4rk.Framework`) and produce equivalent console output — the fluent builders live in the root `sp4rk` package and return the original SDK types. They do not necessarily register the same tools: most fluent variants (02–06, 08, 09) use the `FileTools()` bundle, so their tool set can be larger than the classic variant's hand-registered tools (see example 02); examples 01 and 07 add no bundle in their fluent variant (only the auto-registered `finish`), and 10 adds a single `fetch_webpage` tool via `.Tools(…)`. See [`docs/fluent-api.md`](../docs/fluent-api.md).

**11-full-power is the exception:** it ships a single untagged `main.go` that is fluent-first and drops to the classic API where the fluent façade has no helper. Because the file carries no build tag, both `go run .` and `go run -tags fluent .` build and run it identically.

Each example prints its output to stdout. Examples that require interactive input (e.g. 04-human-in-the-loop) will prompt on stdin.

## Key SDK packages

| Package                              | Purpose                                             |
|--------------------------------------|-----------------------------------------------------|
| `github.com/v0lka/sp4rk`         | Top-level `Framework`, `Config`, `Execute`; fluent builders `NewF`/`RunF`/`TaskF` + bundles |
| `…/agent`                            | `Executor`, `Events`, `HITLHandler`, `FinishTool`   |
| `…/llm`                              | `Router`, `ProviderEntry`, `ModelMetadata`          |
| `…/tools`                            | `Tool` interface, `ToolRegistry`, `BaseTool`        |
| `…/tools/builtins`                   | Built-in tools (read_file, write_file, bash_exec, …) |
| `…/tools/mcp`                        | MCP gateway, `ServerEntry`                          |
| `…/orchestration`                    | `Conductor`, `Blackboard`, `Plan`, DAG utilities    |
| `…/planner`                          | `Planner`, `Config`, `PromptSet`                    |
| `…/agent/reflector`                  | `Reflector` for failure analysis                    |
| `…/prompt`                           | `Builder` / `SystemPromptBuilder`, cache-break support |
| `…/skills`                           | `SkillManager`, skill discovery                     |
| `…/agents`                           | `AgentManager`, Subagent Profile (`AGENT.md`) discovery |
| `…/memory`                           | `ContextWindow`, compaction strategies              |
| `…/security`                         | `WrapUntrustedContent`, `StripUntrustedTags`        |
| `…/agent/router`                     | `Router` classifying requests by domain, complexity, and matched skills |
| `…/oneshot`                          | One-shot structured-output LLM calls with a parse-failure retry loop (used by the planner) |
| `…/tools/builtins/websearch`         | `web_search` tool over external search APIs (Brave, DuckDuckGo, Exa, SearXNG, Tavily) |
| `…/pathutil`                         | `IsWithinPath`/`IsWithinPathFold`, `SplitPathComponents`, `ResolveExistingPrefix` |
| `…/strutil`                          | `TruncateUTF8` and related string helpers           |
| `…/ignore`                           | Multi-root `.gitignore`/`.aiignore` resolver        |
| `…/safeio`                           | Hardened, non-blocking filesystem opens (regular-file check) |
| `…/sysproc`                          | Child-process console suppression + process-tree containment |
| `…/embedding`                        | Optional ONNX-based local `Embedder`, `Tokenizer`, and chunker |
