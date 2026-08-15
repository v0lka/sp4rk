# Contract: Tool System

> This contract documents the public tool interface an embedding application implements and consumes to give an agent capabilities. It is the boundary between the generic tool layer (`github.com/v0lka/sp4rk/tools`) and the host application that registers tools, enforces policy, and wires in human-in-the-loop confirmation.

## Boundary Rule

The host application consumes the tool types from `github.com/v0lka/sp4rk/tools` directly. The `tools` package depends only on sibling sp4rk packages (`llm`, `pathutil`, `strutil`); it never imports host-application code. An embedder plugs in capabilities by **implementing** the `Tool` interface (typically by embedding `BaseTool`), **registering** tools through `ToolRegistry`, and supplying a `ConfirmFunc` for human-in-the-loop gating. The registry is the single execution choke point and satisfies `agent.ToolExecutor`.

## Interfaces

| Interface / Type | Package | Implemented / Consumed By | Purpose |
| --- | --- | --- | --- |
| `Tool` | tools | Implemented by built-ins / host / MCP | Unified tool interface: `Name`, `Description`, `InputSchema`, `Execute`, `DefaultPolicy`, `IsUntrusted`, and the **required** `Group() ToolGroup` |
| `BaseTool` | tools | Embedded by concrete tools | Default impls of `Name`/`Description`/`InputSchema`/`DefaultPolicy`/`IsUntrusted` so a tool implements only `Execute`; carries the `ToolGroup` field (the declaration point for the group) |
| `ToolResult` | tools | Consumed by executor/host | Execution result: `{Content string; IsError bool}` |
| `ToolPolicy` | tools | Set per tool / host override | Security policy enum: `PolicyAlwaysAllow`, `PolicyAlwaysDeny`, `PolicyUserConfirm` |
| `ToolGroup` | tools | Declared by every tool | Capability-group enum (`tools/group.go`): `execute`, `local_read`, `local_write`, `remote_read`, `remote_write`, `system`, `local_mcp`, `remote_mcp`; helpers `AllToolGroups()`, `IsValidToolGroup()`, `MCPToolGroup(transport)`. No `unknown` value — a tool whose group is undeclared matches no group allow-list (fail-closed) |
| `ToolDescriptor` | tools | Consumed by planner/executor | Tool metadata (name, description, schema, source, source category, `Group`) with no execution; `ListFiltered` populates `Group` from the live tool |
| `ToolRegistry` | tools | Constructed by host | Thread-safe tool store and the single execution choke point; satisfies `agent.ToolExecutor` |
| `ToolJudger` | tools | Optionally implemented by a tool | Tool-specific safety heuristic: `Judge(ctx, input) JudgeOutcome{Allow, Reason, Severity}` — severity classifies the reason as `hard` (fired security control: blacklist, SSRF, symlink escape) or `soft` (scope question: path containment). The severity reaches the host via `ConfirmationRequest.JudgeSeverity` (zero value `hard`), which hosts use to decide whether an escalation can be auto-resolved |
| `ContentBackedReader` | tools | Optionally implemented by a read tool | Per-input opt-in to content-backed caching: `IsContentBacked(ctx, input) bool` for read tools returning a transformed/decoded view of a file |
| `CacheMode` | tools | Returned by `ToolRegistry.CacheStrategy` | Cache mode enum: `CacheModeDefault` (keep the file-backed heuristic), `CacheModeContentBacked` (cache the result in memory) |
| `ConfirmFunc` | tools | Implemented by host | Confirmation callback consulted for `PolicyUserConfirm` and judge-escalated calls |
| `ConfirmationRequest` | tools | Built by registry | `{ToolName, Input json.RawMessage, JudgeReasoning, JudgeSeverity}` describing a call needing confirmation; `JudgeSeverity` is `hard` for plain `PolicyUserConfirm` gates and unclassified judge outcomes, `soft` only for explicitly soft judge reasons |
| `ConfirmationResponse` | tools | Returned by host | Decision enum: `ConfirmAllowOnce`, `ConfirmDeny`, `ConfirmDenyAndStop` |
| `ToolSourceCategory` | tools | Set at registration | Origin classifier: `SourceCategoryCore`, `SourceCategoryMCP` (drives untrusted-output handling) |
| `IgnoreChecker` | tools | Satisfied structurally by `ignore.Resolver`/`ignore.Multi` | Reports whether an absolute path is ignored by `.gitignore`/`.aiignore` rules: `Ignored(absPath string, isDir bool) bool`. Read-style tools (glob, ripgrep) consult it to honour ignore rules for the workspace and any work-directory root |
| `WithIgnoreChecker` / `IgnoreCheckerFrom` | tools | Attached/consumed via context | Context plumbing for the ignore checker; `IgnoreCheckerFrom` returns `nil` when none is attached, and callers MUST then skip ignore filtering and keep their pre-ignore behaviour (graceful, no regression) |
| `StripParamsFromSchema` | tools | Consumed by MCP gateway / host | Schema utility that removes named properties from a JSON Schema (`properties` + `required`), e.g. to hide source-specific parameters from the LLM |

> The `ToolJudge` type in `github.com/v0lka/sp4rk/tools/judge.go` is a **separate**, LLM-powered safety evaluator (verdicts `VerdictAllow`/`VerdictConfirm`). It is distinct from the `ToolJudger` interface a tool may implement; do not confuse the two. The judge's response parser is tolerant of LLM formatting variations (markdown decoration, list markers, code fences, lowercase keys, inline single-line answers, and JSON) and classifies the verdict by **whole-token, case-insensitive matching**: `ALLOW`/`ALLOWED`/`APPROVE`/`APPROVED`/`SAFE` → `VerdictAllow`; `CONFIRM`/`CONFIRMED`/`DENY`/`DENIED`/`BLOCK`/`BLOCKED`/`REJECT`/`MANUAL`/`DISALLOW`/`DISAPPROVE` → `VerdictConfirm`. Any unrecognized token — including negations of allow-words (e.g. `DISALLOW`/`DISAPPROVE`, which contain `ALLOW`/`APPROVE` as substrings) — and any LLM error fail **safe** to `VerdictConfirm`, so an ambiguous or adversarial verdict never auto-allows a potentially destructive call.

## Initialization

At startup the host builds the tool surface in this order:

1. Construct a `ToolRegistry` via `NewToolRegistry()` (empty) and optionally `SetLogger`.
2. Implement each built-in/host tool by embedding `BaseTool` and providing `Execute`; set a `DefaultPolicy`, the `Untrusted` flag, **and the `ToolGroup`** on the base. Register each via the registry's registration methods (with a `ToolSourceCategory`).
3. Implement a `ConfirmFunc` that routes `ConfirmationRequest`s to the user (e.g. a UI prompt) and returns a `ConfirmationResponse`. Call `SetConfirmFunc(fn)`.
4. Apply policy overrides: per-tool via `SetPolicyOverride` (engine primitive), or — the recommended host pattern — key policy off the tool's **group** (c0wrk's registry wrapper resolves every non-`system` tool's policy from a group map; see the host's ADR-024). MCP servers can pin their group via `ServerConfig.ToolGroupOverride` (otherwise stdio becomes `local_mcp`, http becomes `remote_mcp`). Optionally attach a `SchemaSanitizer` to the MCP gateway config (see [../domains/tool-system/mcp-gateway.md](../domains/tool-system/mcp-gateway.md)) to transform schemas before they reach the LLM.
5. Register MCP tools through the MCP gateway's `RegisterTools(registry)` (see [../decisions/002-skills-mcp-in-sdk.md](../decisions/002-skills-mcp-in-sdk.md)), which registers them with `SourceCategoryMCP`.
6. The host passes the `ToolRegistry` to the agent executor as its `ToolExecutor`.

`ToolJudger` is **optional**: only tools that opt into self-judging implement it. No tool is required to implement it. `Group()`, by contrast, is **mandatory** on every `Tool` (zero value = undeclared = matches no allow-list).

## Data Flow Across Boundary

- **Host → registry:** tool registration (name, `Tool`, source, category), `SetConfirmFunc`, `SetPolicyOverride`.
- **executor → registry:** `Execute(ctx, name, input json.RawMessage)` and the `agent.ToolExecutor` helpers `GetToolSource(name)` / `IsToolUntrusted(name)` / `CacheStrategy(ctx, name, input)` (returns a `CacheMode`).
- **registry → Tool:** `Execute(ctx, input json.RawMessage)` after policy is satisfied.
- **registry → ContentBackedReader:** during `CacheStrategy`, if the tool implements `ContentBackedReader`, `IsContentBacked(ctx, input)` is consulted per-input to choose content-backed vs file-backed caching.
- **registry → ConfirmFunc:** a `ConfirmationRequest` whenever the effective policy is `PolicyUserConfirm` or a judge escalates; the host returns a `ConfirmationResponse`.
- **registry → ToolJudger:** before an `AlwaysAllow` tool executes, `Judge(ctx, input)` is consulted; a `false` verdict with reasoning escalates to confirmation. `JudgeOutcome.Severity` is delivered to the host as `ConfirmationRequest.JudgeSeverity` — `hard` (never auto-resolvable) or `soft` (a strict judge may allow it); an unclassified outcome is `hard` (fail-closed).
- **host → tools (context):** the host attaches an `IgnoreChecker` via `WithIgnoreChecker(ctx, checker)` so read tools (glob, ripgrep) honour `.gitignore`/`.aiignore`. The checker is typically built from `ignore.Multi` over the workspace and work-directory roots; `IgnoreCheckerFrom(ctx)` returns `nil` when absent, in which case tools keep their pre-ignore behaviour (no filtering).
- **Tool → registry:** `ToolResult` (`{Content, IsError}`) and an error.
- **registry → executor:** `ToolResult` plus the untrusted-source classification (MCP tools and tools with `IsUntrusted()==true` are flagged untrusted so observations are wrapped defensively before entering LLM context).

Data is plain Go values and `json.RawMessage`. Schemas may be transformed (e.g. source-specific parameters stripped) before they reach the LLM via the MCP gateway's `SchemaSanitizer`.

## Error Propagation

- **Fail-closed confirmation:** if the effective policy is `PolicyUserConfirm` and **no** `ConfirmFunc` is configured, the call is **denied** (never executed silently). Mutating tools never run without an explicit confirmation channel or an explicit policy override.
- **Judge escalation is not an error:** a `ToolJudger` returning `allow=false` produces a `ConfirmationRequest` (with the judge reasoning and severity) routed through `ConfirmFunc`; the outcome is `Allow`/`Deny`/`DenyAndStop`, not a Go error.
- **`ConfirmationResponse` semantics:** `ConfirmAllowOnce` permits the single call; `ConfirmDeny` rejects it (becomes the tool's observation); `ConfirmDenyAndStop` rejects and cancels the entire task.
- **`ConfirmDeny` and judge-rejection** are **not** Go errors — they become in-loop observations the model can react to.
- **Tool execution failure** is represented as a `ToolResult` with `IsError=true` (a recoverable, in-loop result fed back as the observation); infrastructure-level failures surface as a Go `error`.
- **LLM-powered `ToolJudge`** fails **safe**: on any LLM error it returns `VerdictConfirm` (escalate to the user) rather than auto-allowing.

## Breaking Change Checklist

- If you change the `Tool` interface, you MUST update `BaseTool`, every built-in/MCP/host tool, and the registry's call sites. Every tool must still declare a valid `ToolGroup` (sp4rk's `TestEveryBuiltinToolDeclaresValidGroup` enforces this for builtins).
- If you change `ToolGroup` (add/rename a group), you MUST update `AllToolGroups`, `IsValidToolGroup`, `MCPToolGroup`, the agents package's `toolGroupTokens` (a cross-check test pins them together), and every host that keys policy or tool budgets off groups.
- If you change `JudgeOutcome`/`JudgeSeverity`, you MUST update every `ToolJudger` implementation and the host escalation paths that read `Severity` (hard reasons must never become auto-approvable).
- If you change `ToolRegistry` registration or `Execute`, you MUST verify it still satisfies `agent.ToolExecutor` and update the MCP gateway's `RegisterTools`.
- If you change the policy enforcement semantics (fail-closed behavior, judge escalation, confirmation gating), you MUST update the host's `ConfirmFunc` plumbing and document the new guarantee.
- If you change `ToolPolicy`, `ConfirmFunc`, `ConfirmationRequest`, or `ConfirmationResponse`, you MUST update every host confirmation path (UI, CLI mode) and serialization.
- If you change `ToolJudger`, you MUST update every tool that implements it and the registry's judge-invocation path.
- If you change `ToolResult`, you MUST update every tool implementation, the executor's observation handling, and the `Step.IsError` mapping.
- If you change `ToolSourceCategory` or the untrusted-output classification, you MUST update `IsToolUntrusted`/`GetToolSource` consumers and the prompt-injection defense wrapping.
- If you change `CacheMode` or the `ContentBackedReader` contract, you MUST update `ToolRegistry.CacheStrategy` and the executor's cache-mode dispatch (`buildCacheMeta`).
- If you change `StripParamsFromSchema` or the `SchemaSanitizer` contract, you MUST update MCP schema handling and any host that transforms schemas before exposing them to the LLM.
- If you change `IgnoreChecker` or its context plumbing (`WithIgnoreChecker`/`IgnoreCheckerFrom`), you MUST update the `ignore.Resolver`/`ignore.Multi` implementations that satisfy it structurally and every read tool (glob, ripgrep) that consults it; preserve the `nil` ⇒ no-filtering default.
