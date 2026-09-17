# Tool System

## Purpose

Provides the tool abstraction for the agent: a single `Tool` interface, a thread-safe `ToolRegistry`, the execution pipeline with fail-closed policy enforcement, and metadata for the planner/executor to reason about available tools without holding live references. Tools are the agent's interface to the outside world (filesystem, shell, web, external MCP servers). Host applications extend the SDK by registering built-in tools, custom tools, and MCP-proxied tools into the registry.

## Key Files

- `github.com/v0lka/sp4rk/tools` — `Tool` interface, `BaseTool`, `ToolResult`, `ToolPolicy`, `ToolJudger` (`JudgeOutcome` + `JudgeSeverity` + `JudgeReasonCode`), `ToolJudge` (`StrictJudgeRequest`, `Judge`/`JudgeStrict`), `ToolDescriptor` (carries `Group`), `ToolRegistry`, `StripParamsFromSchema`
- `github.com/v0lka/sp4rk/tools/group.go` — `ToolGroup` and the 8 declared groups (`execute`, `local_read`, `local_write`, `remote_read`, `remote_write`, `system`, `local_mcp`, `remote_mcp`), `AllToolGroups()`, `IsValidToolGroup()`, `MCPToolGroup(transport)`
- `github.com/v0lka/sp4rk/tools` (registry) — `Register`/`RegisterWithSource`/`RegisterWithSourceCategory`, `Execute` (pre-dispatch input validation + fail-closed policy enforcement), `List`/`ListFiltered`, MCP shadowing protection
- `github.com/v0lka/sp4rk/tools/inputvalidate.go` — `ValidateToolInput(tool, schema, input)` and `InputValidationError`: the recursive structural tool-input validator `Execute` enforces before any policy (see [Input Validation](#input-validation) below); `SetInputValidationEnabled` on the registry is the opt-out handle
- `github.com/v0lka/sp4rk/tools` (context helpers) — `WithWorkspacePath`, `WithTempDir`, `WithAllowedRoots`, `SessionRoots`, `WithTaskContext`. `SessionRoots` returns the deduplicated union of workspace + temp + additional allowed roots consulted by every path-containment check.
- `github.com/v0lka/sp4rk/tools/builtins` — built-in tool catalog
- `github.com/v0lka/sp4rk/tools/mcp` — MCP gateway (dynamic tool discovery/proxying)
- `github.com/v0lka/sp4rk/security` — untrusted-content wrapping for tool output

## Core Types

```go
// Every tool — built-in, custom, or MCP-proxied — implements Tool.
type Tool interface {
    Name() string
    Description() string
    InputSchema() json.RawMessage
    Execute(ctx context.Context, input json.RawMessage) (ToolResult, error)
    DefaultPolicy() ToolPolicy
    IsUntrusted() bool
    // Note: Group() is NOT part of Tool. A tool declares its capability
    // group by additionally implementing the optional GroupProvider interface
    // (tools/group.go); ToolGroupOf(t) returns "" for non-implementers
    // (undeclared = matches no allow-list, fail-closed).
}

type ToolResult struct {
    Content string
    IsError bool
}

type ToolPolicy int
const (
    PolicyAlwaysAllow ToolPolicy = iota
    PolicyAlwaysDeny
    PolicyUserConfirm
)

// Optional per-tool safety heuristic. JudgeOutcome carries Allow, Reason,
// Severity, and ReasonCode: JudgeSeverityHard (the zero value — fail-closed)
// marks a fired security control (blocklist pattern, a flowsh criterion,
// SSRF); JudgeSeveritySoft
// marks a scope question (path containment). ReasonCode is the typed, stable
// classification of the reason delivered to the host as
// ConfirmationRequest.JudgeReasonCode. Allow=false with an empty Reason means
// "no tool-specific concern" and the tool proceeds.
type ToolJudger interface {
    Judge(ctx context.Context, input json.RawMessage) JudgeOutcome
}

// Metadata-only representation for the planner/executor.
type ToolDescriptor struct {
    Name           string
    Description    string
    InputSchema    json.RawMessage
    Source         string             // "core" | <server-name>
    SourceCategory ToolSourceCategory // "core" | "mcp"
    Group          ToolGroup          // capability group; zero value "" matches no group allow-list (fail-closed)
}
```

## Flow

```
ToolRegistry.Execute(ctx, name, input)
│
├─ 1. Look up the tool by name → not found ⇒ error ToolResult
├─ 2. Pre-dispatch input validation (ValidateToolInput — fail-open; see
│     [Input Validation](#input-validation); opt out via
│     SetInputValidationEnabled(false)) → structurally invalid ⇒ error
│     ToolResult naming the offending path and the valid parameters
└─ 3. Resolve effective policy (per-tool override, else the tool's DefaultPolicy):
      ├─ PolicyAlwaysAllow  → execute (escalate to confirmation if a ToolJudger flags it)
      ├─ PolicyAlwaysDeny   → reject with an error result
      └─ PolicyUserConfirm  → consult ConfirmFunc; DENIED (fail-closed) if none configured
```

The executor calls the registry through the narrow `ToolExecutor` interface (`Execute`, `GetToolSource`, `IsToolUntrusted`, `CacheStrategy`). `GetToolSource` returns `"core"` or the MCP server name; `IsToolUntrusted` reports whether a tool's output is from an untrusted source (`tool.IsUntrusted()` true **or** MCP source category) — driving the `<untrusted-content>` wrapping of observations. `CacheStrategy(ctx, name, input)` reports the cache mode the executor should use for a tool's result; a read tool opts into content-backed caching by implementing the optional `ContentBackedReader` interface, otherwise `CacheModeDefault` keeps the existing file-backed heuristic.

The separate LLM-backed `ToolJudge` exposes two host-invoked modes. Advisory `Judge` may auto-allow internal/session-local calls and caches by tool, input, and session roots. `JudgeStrict(ctx, StrictJudgeRequest)` always evaluates the current task/source/input/environment/session-directory envelope with the LLM, applies no fast path or cache, and fail-safes every provider, timeout, construction, or parse failure to `VerdictConfirm`. Tool input and session-directory strings are wrapped as untrusted data in the strict JSON envelope.

> **Breaking change (cache_mode):** `CacheStrategy` is a required method on the exported `agent.ToolExecutor` interface. External implementors (custom registries, wrappers, mocks) must add it — returning `tools.CacheModeDefault` preserves prior behavior. On the `Tool` side the capability remains optional via `ContentBackedReader`.

## Input Validation

`ToolRegistry.Execute` validates every call's input against the target tool's `InputSchema()` BEFORE any policy, override, or confirmation flow is consulted. Provider-side JSON-schema validation comes for free on native tool definitions (one typed tool call per tool), but it is absent whenever arguments travel as a free-form object — meta-tool envelopes (`batch`, E2S-style `action.args`), replayed or queued calls — and `json.Unmarshal` silently ignores unknown fields, so a wrong argument name would execute the tool with defaulted parameters. The registry-level check closes that gap for every call routed through `Execute` (a host wrapper that shadows `Execute` applies its own check — the SDK-level enforcement only covers the SDK path).

The validator (`ValidateToolInput`) is deliberately lightweight — no external JSON-schema dependency — and **recursive**:

- **What it checks** — the input is a JSON object; every `required` key is present; declared property types match (JSON type names; `integer` accepts only numbers with a zero fractional part — `1`, `1.0` and `2e3` qualify, `1.5` does not — since `encoding/json` reports every number as one `number` kind and the literal's textual form decides; an array-typed `["string","null"]` accepts either member); unknown keys are rejected against the declared property set; recursion descends into nested object properties and array `items`, with error paths like `tasks[2].id` naming the offending value. Depth is capped (`maxValidationDepth` = 16).
- **Closed-set semantics** — a schema level without `additionalProperties` (or with an explicit JSON `null`, which counts as absent) is a **closed set**: keys not declared in `properties` are rejected. `additionalProperties: true` (or the object form, a schema for extra keys) reopens that level.
- **Fail-open rules** — anything the validator does not model never blocks a call: an unparseable schema or input at a level, `$ref` subtrees (no resolver by design), levels without a `properties` object, unmodeled type forms (`oneOf`/`anyOf`, an explicit `null`/empty `type`, non-string junk), and values nested deeper than the depth cap. A violation is returned as an `InputValidationError` rendered into an error `ToolResult` (`IsError: true`, nil Go error) that lists the valid parameters, so the model can fix the arguments and retry without ever reaching the tool or the confirmation flow.
- **Disable handle** — `SetInputValidationEnabled(false)` restores the pre-validation behavior (inputs are handed to the tool untouched, before any policy is consulted). Validation is enabled by default (`NewToolRegistry`).

This is defense-in-depth (OWASP ASI02-R2), not a policy gate: it complements, never replaces, per-tool validation and host security gates.

## Invariants

- Tool names are unique within the registry.
- The registry is thread-safe (`sync.RWMutex`).
- `Execute` validates the call input structurally (`ValidateToolInput`) before any policy or override is consulted; the check fails OPEN — an unmodeled schema construct never blocks a call the tool itself would accept. Opt out with `SetInputValidationEnabled(false)` (enabled by default).
- `Execute` is **fail-closed**: a `PolicyUserConfirm` tool with no `ConfirmFunc` configured is DENIED — mutating tools never execute silently.
- A `PolicyAlwaysAllow` tool may implement `ToolJudger` to escalate a call to confirmation; a denied escalation is also fail-closed.
- `ToolJudge.JudgeStrict` performs one independent LLM evaluation per invocation and always maps ambiguous or failed evaluation to `VerdictConfirm`.
- An MCP tool may **not** shadow an already-registered non-MCP tool of the same name (`RegisterWithSourceCategory` errors; the legacy path logs and skips). A built-in tool can always replace an MCP tool; an MCP server re-registering its own tools is allowed.
- MCP tools default to `PolicyUserConfirm` and always report `IsUntrusted() == true`.
- Built-in untrusted tools set `Untrusted: true` on their `BaseTool`.

## Configuration

Policy is set per tool at the engine level. Hosts are encouraged to resolve policy from the tool's capability **group** instead of its name; the engine remains agnostic. To relax tools for non-interactive use:

```go
registry.SetPolicyOverride("bash_exec", tools.PolicyAlwaysAllow) // deliberate opt-in
registry.ClearPolicyOverride("bash_exec")
registry.SetConfirmFunc(myConfirmFunc)   // consulted for PolicyUserConfirm + judge escalation
registry.SetInputValidationEnabled(false) // opt out of pre-dispatch input validation (enabled by default)
```

Stage 1 truncation limits are configured per tool on the executor (see [../orchestration/executor.md](../orchestration/executor.md)); tool result budget (Stage 2) is configured on the executor's `ToolResultBudget`.

## Extension Points

- **New built-in tool**: embed `tools.BaseTool`, implement `Execute`, declare the capability group (`BaseTool.ToolGroup` — an undeclared group matches no group allow-list, fail-closed), optionally implement `ToolJudger`, set `Untrusted: true` for external-output tools, and register. See [builtins.md](builtins.md).
- **Transformed read view (content-backed cache)**: a read tool that returns a transformed/decoded representation of a file (not its raw bytes) implements `tools.ContentBackedReader` (`IsContentBacked(ctx, input) bool`). When it reports `true` for an input, `ToolRegistry.CacheStrategy` returns `CacheModeContentBacked`, so the executor caches the result in memory while still attaching file coherence metadata (path+mtime+size). The decision is per-input, so the same tool can stay file-backed for plain text and content-backed for transformed formats.
- **Custom policy enforcement layer**: hosts may wrap the registry and shadow `Execute` (calling `tool.Execute` directly after their own checks); the SDK-level enforcement only applies to calls routed through `ToolRegistry.Execute`.
- **Schema sanitization**: `StripParamsFromSchema(schema, paramsToRemove)` removes named properties from a JSON Schema's `properties` object and (if present) from `required` — used to hide source-specific parameters from the LLM. The MCP gateway applies it through `GatewayConfig.SchemaSanitizer`; hosts may also call it directly before exposing a schema.
- **MCP servers**: add external tools without writing Go code per server. See [mcp-gateway.md](mcp-gateway.md).

## Related Specs

- [builtins.md](builtins.md) — built-in tool catalog and extension guide
- [mcp-gateway.md](mcp-gateway.md) — MCP server lifecycle and dynamic tool discovery
- [../orchestration/executor.md](../orchestration/executor.md) — tool execution, truncation, caching, trust classification
- [../memory/compaction.md](../memory/compaction.md) — `ToolResultCache` and history mutation
