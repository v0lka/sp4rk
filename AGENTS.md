# AGENTS.md

Guidance for coding agents working on **sp4rk** — a standalone Go framework for building AI agent systems with Plan & Execute orchestration, tool integration, and multi-provider LLM support. sp4rk is a reusable engine: a ReAct execution loop, a DAG-based planner, a reflector for failure analysis, multi-provider LLM routing, a thread-safe tool registry, MCP integration, context-window management with compaction, human-in-the-loop hooks, structured event streaming, skill discovery, local ONNX embeddings, and prompt-injection defenses. It has no opinion about your UI, storage, or deployment.

## Specifications

Formal design specifications live in [`specs/`](specs/) (architecture, domains, contracts, decisions — see [`specs/INDEX.md`](specs/INDEX.md)). Prose guides with runnable examples live in [`docs/`](docs/). Before making structural changes, read the relevant reference:

- [Architecture spec](specs/architecture/layers.md) / [Architecture guide](docs/architecture.md) — layered design and package layout.
- [Agent executor spec](specs/domains/orchestration/executor.md) / [guide](docs/agent-executor.md) — the ReAct execution loop.
- [Orchestration spec](specs/domains/orchestration/README.md) / [guide](docs/orchestration.md) — Plan & Execute mode and the Conductor.
- [Tool-system spec](specs/domains/tool-system/README.md) / [Tools guide](docs/tools.md) / [MCP guide](docs/mcp-integration.md) — the tool registry and gateway.
- [Security spec](specs/architecture/security-model.md) / [guide](docs/security.md) — tool policies and the untrusted-content boundary.
- See [README.md](README.md) and [docs/README.md](docs/README.md) for the full guide index.

## Project shape

- Single Go module: `github.com/v0lka/sp4rk` (this directory). Go `1.26.3` (see `go.mod`).
- The root `sp4rk` package is the public entry point. There is no internal/host-application layer here — every package is a reusable engine primitive.
- **This is an SDK, not an application.** Every exported method, struct, interface, constant, and function is part of the public API consumed by downstream host applications — *even when it is only referenced from tests within this repo*. An exported symbol with no non-test call site inside sp4rk is **not** dead code: tests here stand in for real consumers, and the absence of a production call site is expected and normal. When reviewing, do **not** flag "exported X is only used in tests" as unreachable/dead code, and do not suggest un-exporting or removing such symbols on that basis alone. (Genuinely unused symbols are still caught by `go vet` / `unused`, which exclude test references appropriately.)

### Package map

| Package | Purpose |
| --- | --- |
| `github.com/v0lka/sp4rk` | Top-level `Framework`, `Config`, `New`, `Execute`, `NewConductor`, `Shutdown`; fluent builders `NewF`/`RunF`/`TaskF` plus option/tool/MCP helpers (recommended entry point) |
| `…/agent` | ReAct `Executor` loop, `Events`, `HITLHandler`, `FinishTool`, `RunSubAgent`, tool-result cache |
| `…/agent/reflector` | `Reflector` for execution failure analysis and self-correction |
| `…/agent/router` | `Router` that classifies requests by domain and complexity |
| `…/llm` | `Router`, `Provider`, `ProviderEntry`, `ModelRegistry`, token counting, Anthropic/OpenAI providers |
| `…/tools` | `Tool` interface, `BaseTool`, thread-safe `ToolRegistry`, `ToolPolicy`, `ToolDescriptor` |
| `…/tools/builtins` | Built-in tools: file I/O, shell, glob, ripgrep, web fetch, vector search, facts, checklist |
| `…/tools/mcp` | MCP `Gateway`, `Server`, `ServerEntry`, and tool proxying |
| `…/orchestration` | `Conductor`, `Blackboard`, `Plan`, DAG utilities, `Checkpointer`, orchestration interfaces |
| `…/planner` | `Planner`, `Config`, `PromptSet`, `AgentProfile` for DAG plan generation |
| `…/prompt` | Fluent prompt `Builder` with cache-break support and substitutions |
| `…/skills` | `SkillManager` for agentskills.io skill discovery, parsing, and serving |
| `…/memory` | `ContextWindow`, compaction strategies (sliding-window, summarization, hierarchical), pruning |
| `…/security` | Prompt-injection defense: `WrapUntrustedContent`, `StripUntrustedTags` |
| `…/embedding` | ONNX-based `Embedder`, `Tokenizer`, and chunker for local vector embeddings |
| `…/pathutil` | Reusable filesystem-path algorithms: `IsWithinPath`, `SplitPathComponents`, `ResolveExistingPrefix` |
| `…/strutil` | String helpers: `TruncateUTF8` and related utilities |
| `…/ignore` | Multi-root `.gitignore`/`.aiignore` resolver: `Resolver`, `Multi`, `IgnoreChecker` (satisfied structurally by `tools.IgnoreChecker`) |

### Entry points

- **Fluent (recommended):** `sp4rk.NewF()` returns a `*FrameworkBuilder`; `.Build()` yields a `*sp4rk.Framework`. Run a single ReAct loop with `fw.RunF(ctx)…Ask(…)`; multi-step work with `fw.TaskF(ctx)…`. The builders return the real `*Framework` and `*orchestration.ExecutionResult` types, so fluent and classic code mix freely.
- **Classic:** `sp4rk.New(sp4rk.Config{…})` → `*Framework`, then `fw.Execute(ctx, systemPrompt, events, ask)` for a single loop, or `fw.NewConductor(...)` for Plan & Execute.
- `fw.Shutdown()` must be called (typically `defer`) to release provider/MCP resources.

## Commands

The root `Makefile` provides `build`, `test`, and `lint` targets; otherwise run Go tooling directly from the module root.

- `go test ./…` (or `make test`) — run the full test suite.
- `go test ./agent -run TestExecutor -v` — single test. Tests are in-package (`package agent`, not `agent_test`); many packages ship a `testhelpers_test.go`.
- `go vet ./…` — vet.
- `golangci-lint run` (or `make lint`) — lint. The config is `.golangci.yml` (v2 schema).
- `go run ./examples/01-minimal-agent` — **wrong**: `examples/` is a *separate* Go module (`sp4rk-examples`) that imports sp4rk as an external dependency. Run examples from inside it instead:
  - `cd examples && go run ./01-minimal-agent`
  - The eleven examples progress from a minimal agent through focused subsystem deep-dives to a full-stack system; see [`examples/README.md`](examples/README.md).

## Conventions & gotchas

- **Logging:** `log/slog` everywhere. Pass `*slog.Logger` through constructors; do not use a global `slog`.
- **Errors:** `errorlint` + `perfsprint` are on. Wrap with `%w`, use `errors.Is/As`, never `fmt.Errorf` where `errors.New` suffices, and never `fmt.Sprintf("%s", s)`. `noctx`, `bodyclose`, `sqlclosecheck` are also enforced.
- **Linters enabled** (`.golangci.yml`): `errcheck` (incl. type assertions), `govet`, `staticcheck`, `ineffassign`, `unused`, `errorlint`, `nilerr`, `gocritic` (diagnostic+performance+style, except `hugeParam`/`rangeValCopy`), `revive`, `prealloc`, `unconvert`, `wastedassign`, `copyloopvar`, `durationcheck`, `whitespace`, `depguard`. In this module `revive`'s `exported` and `var-naming` rules are **enabled** (they are disabled in some sibling configs) — keep exported identifiers and naming idiomatic.
- **Tool registry pattern:** built-in tools live in `tools/builtins/`; MCP-backed tools are added at runtime via `tools/mcp/gateway.go`. To add a built-in tool, implement the `tools.Tool` interface and register it on the `ToolRegistry`.
- **`finish` tool:** auto-registered by the fluent builders (`NewF()` sets `autoFinish: true`), so a fluent agent can signal completion without an explicit `Register` call. The **classic** `sp4rk.New` path does **not** auto-register it — call `fw.ToolRegistry().Register(agent.NewFinishTool())` yourself, or the loop runs until the step budget is exhausted and returns a "partial" status. Disable fluent auto-registration with `NoAutoFinish()`.
- **Non-cacheable tools:** the `Executor` caches tool results; SDK-internal meta-tools are listed in `defaultNonCacheableTools` (`finish`, `tool_result_read`, `read_step_output`, `list_step_outputs`, `read_final_result`, `read_attachment`). Application-layer meta-tools (e.g. delegation/plan tools) should be added via `Executor.AddNonCacheableTools(names…)` — they are excluded from both caching and Stage-2 hash hints.
- **Conductor vs Executor:** the Plan & Execute `Conductor` runs each plan step as its own ReAct loop — both are `Executor.Run` instances. When you change the execution loop, it affects single-shot `Execute`, subagent runs, and conductor steps alike.
- **Prompts:** system prompts are assembled via the fluent `prompt.Builder` (`prompt`) with cache-break and substitution support. Prefer configurable prompt factories over hard-coded strings.
- **Path & string helpers:** use `pathutil` (`IsWithinPath`, `SplitPathComponents`, `ResolveExistingPrefix`) and `strutil` (`TruncateUTF8`) rather than hand-rolling path/string logic.
- **Ignore filtering:** `glob` and `ripgrep` honour `.gitignore`/`.aiignore` through a single shared authority — the `tools.IgnoreChecker` plumbed through context by the host (typically `ignore.Multi` over the workspace + work-dir roots). `tools` never imports `ignore`: both define `Ignored(absPath, isDir) bool`, and `ignore.Resolver`/`ignore.Multi` satisfy `tools.IgnoreChecker` structurally. A `nil` checker (none attached) means **no** ignore filtering — the opt-in, no-regression default; `rg` still honours `.gitignore` natively. Negation patterns (`!`) are unsupported.
- **Security:** untrusted tool output (web, MCP, filesystem) is wrapped in `<untrusted-content>` boundary tags via `security` (`WrapUntrustedContent` / `StripUntrustedTags`) before it enters LLM context.
- **ONNX Runtime is OPTIONAL:** only the local embedding subsystem (`embedding`) needs it. The rest of the framework runs without it. See [docs/embedding.md](docs/embedding.md).

## Pre-PR checklist

`go vet ./… && golangci-lint run && go test ./…`. All three must be clean.
