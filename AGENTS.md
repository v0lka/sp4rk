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
| `…/agent/router` | `Router` that classifies requests by domain, complexity, and matched skills (optional semantic tool matching via `SetToolMatching`) |
| `…/llm` | `Router`, `Provider`, `ProviderEntry`, `ModelRegistry`, token counting, OpenAI/Anthropic/Google providers, multi-protocol routing (`APIProtocol`/`DetectProtocol`), multimodal `ContentBlock` |
| `…/tools` | `Tool` interface, `BaseTool`, thread-safe `ToolRegistry`, `ToolPolicy`, `ToolDescriptor` |
| `…/tools/builtins` | Built-in tools: file I/O, shell, glob, ripgrep, web fetch, vector search, facts, checklist |
| `…/tools/mcp` | MCP `Gateway`, `Server`, `ServerEntry`, and tool proxying |
| `…/orchestration` | `Conductor`, `Blackboard`, `Plan`, DAG utilities, `Checkpointer`, orchestration interfaces |
| `…/planner` | `Planner`, `Config`, `PromptSet`, `AgentProfile` for DAG plan generation |
| `…/prompt` | Fluent prompt `Builder` with cache-break support and substitutions |
| `…/skills` | `SkillManager` for agentskills.io skill discovery, parsing, and serving |
| `…/agents` | `AgentManager` for Subagent Profile (`AGENT.md`) discovery, parsing, and serving |
| `…/memory` | `ContextWindow`, compaction strategies (sliding-window, summarization, hierarchical), pruning |
| `…/security` | Prompt-injection defense: `WrapUntrustedContent`, `StripUntrustedTags` |
| `…/embedding` | ONNX-based `Embedder`, `Tokenizer`, and chunker for local vector embeddings |
| `…/pathutil` | Reusable filesystem-path algorithms: `IsWithinPath`, `SplitPathComponents`, `ResolveExistingPrefix` |
| `…/strutil` | String helpers: `TruncateUTF8` and related utilities |
| `…/ignore` | Multi-root `.gitignore`/`.aiignore` resolver: `Resolver`, `Multi`, `IgnoreChecker` (satisfied structurally by `tools.IgnoreChecker`) |
| `…/sysproc` | Child-process console-window suppression (`HideConsole`) + process-tree containment (`AssignKillOnCloseJob`): OR-edits `CREATE_NO_WINDOW` on Windows and assigns started processes to a kill-on-close Job Object; no-op elsewhere. `HideConsole` is stdlib-only; the Windows Job-Object helper depends on `golang.org/x/sys/windows` (behind a `//go:build windows` tag) |

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
- **Non-cacheable tools:** the `Executor` caches tool results; SDK-internal meta-tools are listed in `defaultNonCacheableTools` (`finish`, `tool_result_read`, `read_step_output`, `list_step_outputs`, `read_final_result`, `read_attachment`). Application-layer meta-tools (e.g. delegation/plan tools) should be added via `Executor.AddNonCacheableTools(names…)` — they skip the *eager* caching of small results (and the Stage-2 hash hint for small results). **No tool is ever truncated without a retrieval path:** when a non-cacheable tool's result is truncated at Stage 1 (line/byte limit) or Stage 2 (token budget), the executor caches the full result on demand and embeds the hash in the truncation notice, so the LLM can recover it via `tool_result_read`. Only small, non-truncated non-cacheable results stay out of the cache.
- **Conductor vs Executor:** the Plan & Execute `Conductor` runs each plan step as its own ReAct loop — both are `Executor.Run` instances. When you change the execution loop, it affects single-shot `Execute`, subagent runs, and conductor steps alike.
- **Subagent Profiles:** the `agents` package (`AgentManager`) discovers/parses `AGENT.md` profiles — it is fully self-contained (stdlib + `yaml.v3` + `log/slog` only; no engine imports), mirroring `skills` but with no engine coupling. A `PlanStep.Agent` names a profile resolved by the execution layer; the profile's `Model` is forced via `agent.NewModelOverrideCaller`, which relies on the router contract that `req.Model` is filled only when empty. Applying a profile (prompt/tools/max-steps/model/redelegation policy) is an execution-layer concern, not a `RunSubAgent` primitive — the host wires it before launching. `PlanStep.Agent` and `PlanStep.Profile` (`*planner.AgentProfile`) are independent.
- **Prompts:** system prompts are assembled via the fluent `prompt.Builder` (`prompt`) with cache-break and substitution support. Prefer configurable prompt factories over hard-coded strings.
- **Path & string helpers:** use `pathutil` (`IsWithinPath`, `SplitPathComponents`, `ResolveExistingPrefix`) and `strutil` (`TruncateUTF8`) rather than hand-rolling path/string logic.
- **Child-process console suppression:** every `exec.Cmd` the SDK spawns that is not an interactive PTY session calls `sysproc.HideConsole(cmd)` before starting it, so a GUI-subsystem host does not flash or leave open a console window for `posh_exec`, `ripgrep`, env-info version probes, or stdio MCP servers. It OR-edits `CREATE_NO_WINDOW` (0x08000000) into `SysProcAttr.CreationFlags` on Windows, preserving any flags already set; it is a no-op elsewhere, so call it unconditionally — never branch on `runtime.GOOS`. Do not apply it to an interactive pseudo-terminal session (ConPTY).
- **Process-tree containment:** `posh_exec` (and equivalent shell tools) call `sysproc.AssignKillOnCloseJob(cmd)` as the very next action after `cmd.Start()` and `defer` the returned cleanup, placing the process and its entire descendant tree in a kill-on-close Windows Job Object so orphaned grandchildren (e.g. a browser launched by PowerShell, and its console window) cannot outlive the host — killing the shell alone would leave them running. It is best-effort (returns a no-op cleanup and a sentinel error on a pre-Windows 8 / non-nestable host) and a no-op on non-Windows, so call it unconditionally like `HideConsole` and `defer` the cleanup regardless of the error.
- **Ignore filtering:** `glob` and `ripgrep` honour `.gitignore`/`.aiignore` through a single shared authority — the `tools.IgnoreChecker` plumbed through context by the host (typically `ignore.Multi` over the workspace + work-dir roots). `tools` never imports `ignore`: both define `Ignored(absPath, isDir) bool`, and `ignore.Resolver`/`ignore.Multi` satisfy `tools.IgnoreChecker` structurally. A `nil` checker (none attached) means **no** ignore filtering — the opt-in, no-regression default; `rg` still honours `.gitignore` natively. Negation patterns (`!`) are unsupported.
- **Security:** untrusted tool output (web, MCP, filesystem) is wrapped in `<untrusted-content>` boundary tags via `security` (`WrapUntrustedContent` / `StripUntrustedTags`) before it enters LLM context.
- **Multi-protocol routing:** a single OpenAI-compatible `ProviderEntry` dispatches each model to its native wire protocol via `DetectProtocol` — `/responses` (gpt-5/codex), `/messages` (Claude, via a co-located AnthropicProvider), `:generateContent` (Gemini/Gemma, via `googleCompletion`), or `/chat/completions` (default). There is no separate `"google"` `ProviderType`. For a locally-served model whose name matches a family token but speaks a different protocol, set `ModelMetadata.Protocol` to override substring detection. The router's `prepareRequest` resolves metadata once and threads the resolved protocol into `ChatRequest.Protocol`; the provider honors `req.Protocol` over its own `DetectProtocol`, so a tier-1 override takes effect for router-driven calls.
- **Multimodal content blocks:** `llm.Message.ContentBlocks` carries text/image blocks; providers render them when non-empty (`NormalizeContentBlocks` prepends task text when blocks lack a text block; `ValidateContentBlocks` enforces required fields). Feed a multimodal task via `ConductorConfig.ContentBlocks` → `BlockTaskAware.SetTaskWithBlocks`. Token counters estimate images at 765 tokens (default) / 85 (Anthropic-family).
- **Model registry resolution:** `Resolve` is a 7-tier case-insensitive lookup (overrides → observed runtime entries written by `SetRuntimeMetadata`, which supersede the built-in spec for how a model is actually served → built-in → fuzzy vendor-prefix/separator-insensitive, which also consults the runtime index → cache → HuggingFace/registered sources → fallback with optimistic `Attachment`). `ResolveLocal` is the strictly network-free variant (local tiers only) for synchronous UI paths; `ResolveBuiltInModel` is catalog-only (no network); `SetCachedMetadata` stores late-learned metadata at the cache tier; `RuntimeMetadata` reads a runtime entry back as stored (un-enriched). A failed HuggingFace probe is negatively cached for 10 minutes (`context.Canceled` from the caller excepted; `Invalidate` clears both entries). A **partial** override (one that leaves some scalar fields unset, e.g. a protocol-only `{Protocol: ChatCompletions}`) inherits its unset `ContextWindow`/`OutputLimit`/`TokenizerType`/`Capabilities` from the lower non-network tiers — a zero `ModelCapabilities` counts as "unset" — so pinning the protocol does not collapse the context window or disable capabilities. A fully-specified override is returned verbatim. The fuzzy tier is O(1) via normalized-ID indexes built at construction (the runtime index is rebuilt on every runtime write).
- **ONNX Runtime is OPTIONAL:** only the local embedding subsystem (`embedding`) needs it. The rest of the framework runs without it. See [docs/embedding.md](docs/embedding.md).

## Pre-PR checklist

`go vet ./… && golangci-lint run && go test ./…`. All three must be clean.
