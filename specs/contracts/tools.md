# Contract: Tool System

> This contract documents the public tool interface an embedding application implements and consumes to give an agent capabilities. It is the boundary between the generic tool layer (`github.com/v0lka/sp4rk/tools`) and the host application that registers tools, enforces policy, and wires in human-in-the-loop confirmation.

## Boundary Rule

The host application consumes the tool types from `github.com/v0lka/sp4rk/tools` directly. The `tools` package depends only on sibling sp4rk packages (`llm`, `pathutil`, `strutil`); it never imports host-application code. An embedder plugs in capabilities by **implementing** the `Tool` interface (typically by embedding `BaseTool`), **registering** tools through `ToolRegistry`, and supplying a `ConfirmFunc` for human-in-the-loop gating. The registry is the single execution choke point and satisfies `agent.ToolExecutor`.

## Interfaces

| Interface / Type | Package | Implemented / Consumed By | Purpose |
| --- | --- | --- | --- |
| `Tool` | tools | Implemented by built-ins / host / MCP | Unified tool interface: `Name`, `Description`, `InputSchema`, `Execute`, `DefaultPolicy`, `IsUntrusted` |
| `BaseTool` | tools | Embedded by concrete tools | Default impls of `Name`/`Description`/`InputSchema`/`DefaultPolicy`/`IsUntrusted` so a tool implements only `Execute` |
| `ToolResult` | tools | Consumed by executor/host | Execution result: `{Content string; IsError bool}` |
| `ToolPolicy` | tools | Set per tool / host override | Security policy enum: `PolicyAlwaysAllow`, `PolicyAlwaysDeny`, `PolicyUserConfirm` |
| `ToolDescriptor` | tools | Consumed by planner/executor | Tool metadata (name, description, schema, source, source category) with no execution |
| `ToolRegistry` | tools | Constructed by host | Thread-safe tool store and the single execution choke point; satisfies `agent.ToolExecutor` |
| `ToolJudger` | tools | Optionally implemented by a tool | Tool-specific safety heuristic: `Judge(ctx, input)(allow bool, reasoning string)` |
| `ContentBackedReader` | tools | Optionally implemented by a read tool | Per-input opt-in to content-backed caching: `IsContentBacked(ctx, input) bool` for read tools returning a transformed/decoded view of a file |
| `CacheMode` | tools | Returned by `ToolRegistry.CacheStrategy` | Cache mode enum: `CacheModeDefault` (keep the file-backed heuristic), `CacheModeContentBacked` (cache the result in memory) |
| `ConfirmFunc` | tools | Implemented by host | Confirmation callback consulted for `PolicyUserConfirm` and judge-escalated calls |
| `ConfirmationRequest` | tools | Built by registry | `{ToolName, Input json.RawMessage, JudgeReasoning}` describing a call needing confirmation |
| `ConfirmationResponse` | tools | Returned by host | Decision enum: `ConfirmAllowOnce`, `ConfirmDeny`, `ConfirmDenyAndStop` |
| `ToolSourceCategory` | tools | Set at registration | Origin classifier: `SourceCategoryCore`, `SourceCategoryMCP` (drives untrusted-output handling) |
| `IgnoreChecker` | tools | Satisfied structurally by `ignore.Resolver`/`ignore.Multi` | Reports whether an absolute path is ignored by `.gitignore`/`.aiignore` rules: `Ignored(absPath string, isDir bool) bool`. Read-style tools (glob, ripgrep) consult it to honour ignore rules for the workspace and any work-directory root |
| `WithIgnoreChecker` / `IgnoreCheckerFrom` | tools | Attached/consumed via context | Context plumbing for the ignore checker; `IgnoreCheckerFrom` returns `nil` when none is attached, and callers MUST then skip ignore filtering and keep their pre-ignore behaviour (graceful, no regression) |
| `ParamManager` | tools | Provided by host (optional) | Auto-injected parameter management (`SanitizeSchema` + `InjectParams`), e.g. project path |
| `AutoInjectedParamProject` | tools | Constant | `"project"` — the auto-injected param name stripped from tool schemas before the LLM sees them |

> The `ToolJudge` type in `github.com/v0lka/sp4rk/tools/judge.go` is a **separate**, LLM-powered safety evaluator (verdicts `VerdictAllow`/`VerdictConfirm`, fail-safe to confirm on LLM error). It is distinct from the `ToolJudger` interface a tool may implement; do not confuse the two.

## Initialization

At startup the host builds the tool surface in this order:

1. Construct a `ToolRegistry` via `NewToolRegistry()` (empty) and optionally `SetLogger`.
2. Implement each built-in/host tool by embedding `BaseTool` and providing `Execute`; set a `DefaultPolicy` and the `Untrusted` flag on the base. Register each via the registry's registration methods (with a `ToolSourceCategory`).
3. Implement a `ConfirmFunc` that routes `ConfirmationRequest`s to the user (e.g. a UI prompt) and returns a `ConfirmationResponse`. Call `SetConfirmFunc(fn)`.
4. Optionally install a `ParamManager` for auto-injected parameters, and apply per-tool policy overrides via `SetPolicyOverride`.
5. Register MCP tools through the MCP gateway's `RegisterTools(registry)` (see [../decisions/002-skills-mcp-in-sdk.md](../decisions/002-skills-mcp-in-sdk.md)), which registers them with `SourceCategoryMCP`.
6. The host passes the `ToolRegistry` to the agent executor as its `ToolExecutor`.

`ToolJudger` is **optional**: only tools that opt into self-judging implement it. No tool is required to implement it.

## Data Flow Across Boundary

- **Host → registry:** tool registration (name, `Tool`, source, category), `SetConfirmFunc`, `SetPolicyOverride`, optional `ParamManager`.
- **executor → registry:** `Execute(ctx, name, input json.RawMessage)` and the `agent.ToolExecutor` helpers `GetToolSource(name)` / `IsToolUntrusted(name)` / `CacheStrategy(ctx, name, input)` (returns a `CacheMode`).
- **registry → Tool:** `Execute(ctx, input json.RawMessage)` after policy is satisfied.
- **registry → ContentBackedReader:** during `CacheStrategy`, if the tool implements `ContentBackedReader`, `IsContentBacked(ctx, input)` is consulted per-input to choose content-backed vs file-backed caching.
- **registry → ConfirmFunc:** a `ConfirmationRequest` whenever the effective policy is `PolicyUserConfirm` or a judge escalates; the host returns a `ConfirmationResponse`.
- **registry → ToolJudger:** before an `AlwaysAllow` tool executes, `Judge(ctx, input)` is consulted; a `false` verdict with reasoning escalates to confirmation.
- **host → tools (context):** the host attaches an `IgnoreChecker` via `WithIgnoreChecker(ctx, checker)` so read tools (glob, ripgrep) honour `.gitignore`/`.aiignore`. The checker is typically built from `ignore.Multi` over the workspace and work-directory roots; `IgnoreCheckerFrom(ctx)` returns `nil` when absent, in which case tools keep their pre-ignore behaviour (no filtering).
- **Tool → registry:** `ToolResult` (`{Content, IsError}`) and an error.
- **registry → executor:** `ToolResult` plus the untrusted-source classification (MCP tools and tools with `IsUntrusted()==true` are flagged untrusted so observations are wrapped defensively before entering LLM context).

Data is plain Go values and `json.RawMessage`. Auto-injected parameters are injected at execution time and stripped from schemas before they reach the LLM.

## Error Propagation

- **Fail-closed confirmation:** if the effective policy is `PolicyUserConfirm` and **no** `ConfirmFunc` is configured, the call is **denied** (never executed silently). Mutating tools never run without an explicit confirmation channel or an explicit policy override.
- **Judge escalation is not an error:** a `ToolJudger` returning `allow=false` produces a `ConfirmationRequest` (with the judge reasoning) routed through `ConfirmFunc`; the outcome is `Allow`/`Deny`/`DenyAndStop`, not a Go error.
- **`ConfirmationResponse` semantics:** `ConfirmAllowOnce` permits the single call; `ConfirmDeny` rejects it (becomes the tool's observation); `ConfirmDenyAndStop` rejects and cancels the entire task.
- **`ConfirmDeny` and judge-rejection** are **not** Go errors — they become in-loop observations the model can react to.
- **Tool execution failure** is represented as a `ToolResult` with `IsError=true` (a recoverable, in-loop result fed back as the observation); infrastructure-level failures surface as a Go `error`.
- **LLM-powered `ToolJudge`** fails **safe**: on any LLM error it returns `VerdictConfirm` (escalate to the user) rather than auto-allowing.

## Breaking Change Checklist

- If you change the `Tool` interface, you MUST update `BaseTool`, every built-in/MCP/host tool, and the registry's call sites.
- If you change `ToolRegistry` registration or `Execute`, you MUST verify it still satisfies `agent.ToolExecutor` and update the MCP gateway's `RegisterTools`.
- If you change the policy enforcement semantics (fail-closed behavior, judge escalation, confirmation gating), you MUST update the host's `ConfirmFunc` plumbing and document the new guarantee.
- If you change `ToolPolicy`, `ConfirmFunc`, `ConfirmationRequest`, or `ConfirmationResponse`, you MUST update every host confirmation path (UI, CLI mode) and serialization.
- If you change `ToolJudger`, you MUST update every tool that implements it and the registry's judge-invocation path.
- If you change `ToolResult`, you MUST update every tool implementation, the executor's observation handling, and the `Step.IsError` mapping.
- If you change `ToolSourceCategory` or the untrusted-output classification, you MUST update `IsToolUntrusted`/`GetToolSource` consumers and the prompt-injection defense wrapping.
- If you change `CacheMode` or the `ContentBackedReader` contract, you MUST update `ToolRegistry.CacheStrategy` and the executor's cache-mode dispatch (`buildCacheMeta`).
- If you change `ParamManager` or `AutoInjectedParamProject`, you MUST update schema sanitizers, MCP schema handling, and the host's injection logic.
- If you change `IgnoreChecker` or its context plumbing (`WithIgnoreChecker`/`IgnoreCheckerFrom`), you MUST update the `ignore.Resolver`/`ignore.Multi` implementations that satisfy it structurally and every read tool (glob, ripgrep) that consults it; preserve the `nil` ⇒ no-filtering default.
