# Security Model

## Context

sp4rk executes arbitrary tools (filesystem operations, shell commands, web requests) on behalf of an LLM, and ingests external data into the model's context. The engine provides two complementary security primitives: a **tool-execution policy layer** that gates whether and how a tool runs, and a **prompt-injection defense** that delimits untrusted external content before it reaches the model. This spec describes the engine-level primitives only — a host application layers additional wiring (workspace/temp auto-approval, command blacklists, symlink-forced confirmation) on top of these.

## Scope Boundary

sp4rk ships the *primitives*; the host application assembles the *policy*. The SDK provides:

- the `ToolPolicy` enum, per-tool policy resolution, and the `ToolGroup` capability groups (`tools/group.go`),
- the `ToolJudger` interface and the fail-closed confirmation flow (`ConfirmFunc`/`ConfirmationRequest`/`ConfirmationResponse`),
- the path-locality primitive — [`tools.IsWithinRoot`](#path-locality-primitive) and [`AllPathsInSessionRoots`](#path-locality-primitive), consulted by the judge fast-path auto-approval and every tool-argument containment check (see [Path-Locality Primitive](#path-locality-primitive)),
- the prompt-injection defense (`security.WrapUntrustedContent`/`StripUntrustedTags`), and
- the MCP tool-shadowing guard.

The host application owns, and this spec deliberately does **not** cover:

- the *decision* to enable path-locality auto-approval (whether the judge fast-path is on, which session roots are configured, confirmation UX),
- command blacklists (e.g. dangerous-shell patterns),
- symlink-forced confirmation gating.

Those decisions depend on host-specific concepts (session roots, command grammars, OS conventions) and are wired by the embedding application around the SDK registry.

<a id="path-locality-primitive"></a>
### Path-Locality Primitive

`tools.IsWithinRoot(ctx, root, path)` is the canonical path-containment check consulted by the judge fast-path (`AllPathsInSessionRoots`), file-tool argument resolution, symlink inside/outside classification, working-directory validation, cache-coherence gating, and ignore-root selection. It resolves symlinks through the longest existing prefix of both paths, then folds letter case **only when the session flag is set** — determined by probing the filesystem at session-root resolution time (`pathutil.DetectCaseInsensitive`):

- **Case-insensitive filesystems** (macOS APFS, Windows NTFS): the fold matches the on-disk reality, so a not-yet-existing target written with different casing than the session root is still recognized as local.
- **Case-sensitive filesystems** (Linux ext4/tmpfs/btrfs, and case-sensitive APFS volumes): containment is case-sensitive, so distinct-cased siblings such as `/ws/Project` and `/ws/project` are never conflated. Applying the fold unconditionally here would be an authorization bypass — a non-local sibling whose name differs only by case would be auto-approved as "local".

The flag defaults to **case-sensitive** (fail-safe): when detection is impossible (no existing ancestor, unreadable root), containment never widens. Case folding never weakens escape prevention — it can only cause an otherwise-mismatched path to be considered "within" a root; the `..` traversal and symlink-escape logic is unaffected.

The flag is attached automatically by `tools.WithWorkspacePath` (which probes the workspace root) and may be overridden by `tools.WithCaseInsensitivePaths`. Hosts that construct their context without `WithWorkspacePath` (e.g. `WithWorkspacePathNoProbe`) inherit the fail-safe case-sensitive default unless they attach the flag explicitly.

Harmless special-device paths are exempted from path-locality determination via [`tools.IsHarmlessDevicePath`](#path-locality-primitive). A path that is a bit-bucket or error-sink pseudo-device — `/dev/null` and `/dev/full` on POSIX, `NUL` on Windows (matched case-insensitively as the final path component) — cannot leak data outside the workspace nor persist unwanted changes, so `AllPathsInSessionRoots` and `AllPathsInDir` skip it. This is why `cat file > /dev/null`, `read_file /dev/null`, or `write_file NUL` are not forced to confirm as out-of-workspace operations. The symlink gate deliberately does **not** consult it: it resolves every path component via `Lstat` rather than trusting the path string, so harmless-looking names that are actually symlinks (e.g. macOS `/dev/stdin`, which points into `/dev/fd/*`) are still walked and resolved.

### Separator-Run Shell Tokens Are Not Paths

Shell commands routinely contain runs of slashes that are artifacts of the shell language, not filesystem paths: the trailing `//` of a sed address (`sed 's/.*function //'`), a comment marker inside an echoed string (`echo "// TODO fix" >> notes.md`), an integer-division operator (`echo $(( total // count ))`), or an escaped PowerShell drive root (`C:\\`). The path extractors behind path-locality — `tools.ResolveShellPathTokens` (the shell-dialect-aware extractor behind `PathsOutsideRoots`, consulted by the built-in shell tools' `Judge`) and `tools.ExtractPaths` (the JSON-input extractor behind `AllPathsInSessionRoots`) — **skip tokens that consist entirely of separators**: a POSIX run of two or more slashes (`//`, `///`, …), or a two-character drive prefix followed by two or more separators and no path component (`C:\\`). Such a token carries no path component and names no out-of-root location; resolving it anyway would clean a bare `//` to the filesystem root `/`. A phantom root in the extracted set forced a false-positive confirmation of entirely in-root commands — and it poisoned the `AllPathsInSessionRoots` fast-path, which requires *every* extracted path to be in-root.

The skip is narrow by construction (`tools.isPureSeparatorRunToken`) and preserves every containment guarantee:

- A single `/` never matches the absolute-path pattern at all (it requires at least one character after the leading separator), so the POSIX form skips only runs of **two or more** slashes.
- `cat //etc/passwd` is still reported — the token carries the `etc/passwd` path components, so it is not a pure separator run.
- On the drive form a single trailing separator is a drive root (`Get-Content C:\`) and a token with a component is a real path (`Get-Content C:\\Windows\win.ini`); both stay reportable. Only the pure separator run (`C:\\`) is skipped.
- Anchored out-of-root writes still escalate: `cat /etc/passwd`, `echo x > /etc/cron.d/newjob` (leaf missing, parent directory existing), and `rm -rf /.` all resolve outside the roots and are reported.
- A separator-run artifact cannot mask a real escape: a sed `//` and `/etc/passwd` in the same command still fails containment, because each token is extracted independently.
- Command blacklists are untouched: the built-in shell tools' `ToolJudger` matches the raw command string against its blacklist patterns **before** path analysis (and its reason takes precedence), so `rm -rf /` is still escalated by an `rm -rf` pattern regardless of the skip. Its separator-run spelling `rm -rf //` is covered by the same blacklist: the skip removes the `//` token from path extraction (exactly as the bare `/` never matched the absolute-path pattern), so the blacklist is the authoritative backstop for bare-root deletions — a host deploying shell tools without an `rm -rf`-style pattern should add one. See [../domains/tool-system/builtins.md](../domains/tool-system/builtins.md).

## Tool Policies

Every tool carries a `ToolPolicy` (`github.com/v0lka/sp4rk/tools`). It is an integer enum whose string mapping is stable:

```go
type ToolPolicy int

const (
    PolicyAlwaysAllow ToolPolicy = iota // execute without confirmation or judge
    PolicyAlwaysDeny                     // block; tool never executes
    PolicyUserConfirm                    // require user confirmation before executing
)
```

`ParseToolPolicy` maps the configuration strings `"always_allow"`, `"always_deny"`, and `"user_confirm"`; any unrecognized value (including the empty string) falls back to `PolicyUserConfirm` — the safest default.

| Policy string    | Behavior                                                                                                |
| ---------------- | ------------------------------------------------------------------------------------------------------- |
| `always_allow`   | Execute immediately. No confirmation. A `ToolJudger` may still escalate a flagged call (see below).     |
| `user_confirm`   | Block execution and call `ConfirmFunc`. The host must allow, deny, or deny-and-stop. **Fail-closed** when no `ConfirmFunc` is configured. |
| `always_deny`    | Immediately return an error `ToolResult`. The tool is never executed.                                   |

## Policy Resolution

When a tool is about to execute, its effective policy is resolved (first match wins):

```
1. Per-tool override   (registry.SetPolicyOverride(name, policy))
2. Tool's own default  (Tool.DefaultPolicy())
```

Source: `github.com/v0lka/sp4rk/tools` registry `Execute` — if a per-tool override is set (`r.policyOverrides[name]`) it wins; otherwise the tool's own `DefaultPolicy()` is used. There is no registry-level default policy: `ToolRegistry` has no default-policy field and no `SetDefaultPolicy` method.

The engine itself has no concept of "skill policy" or a configuration file; those are host-side layers that call `SetPolicyOverride` on the registry. The engine only resolves between the per-tool override and the tool's own `DefaultPolicy()`.

## The ToolJudger Interface

Tools may optionally implement `ToolJudger` to provide tool-specific safety heuristics:

```go
type JudgeSeverity int // zero value = hard (fail-closed); constants JudgeSeverityHard, JudgeSeveritySoft; String() → "hard"/"soft"

type JudgeOutcome struct {
    Allow    bool
    Reason   string
    Severity JudgeSeverity
}

type ToolJudger interface {
    Judge(ctx context.Context, input json.RawMessage) JudgeOutcome
}
```

For a tool whose effective policy is `PolicyAlwaysAllow`, the registry calls `Judge` **before** execution. If the judge returns `Allow=false` **with non-empty Reason**, the call is escalated to user confirmation via `ConfirmFunc`. An outcome of `Allow=false` with empty Reason is treated as "no concern to report" and the tool proceeds.

The `Severity` classifies the reason so hosts can decide what an escalation means: **`hard`** marks a fired security control (shell blacklist pattern, SSRF) — hosts treat it as never auto-resolvable and do not pass it through an advisory auto-approval path; **`soft`** marks a scope question (path containment outside session roots — evaluated on the symlink-resolved path, so a symlink that escapes the roots is denied containment like any other out-of-roots target) that a strict judge may resolve. The engine escalates both severities identically (user confirmation) and delivers the severity to the host via `ConfirmationRequest.JudgeSeverity`. `JudgeSeverityHard` is the zero value — a judge outcome or escalation that arrives unclassified is treated as `hard` (fail-closed).

This is the SDK hook a host uses to implement its own safety checks (for example, a shell tool whose `Judge` matches a blacklist of dangerous commands, or a file tool whose `Judge` flags paths outside permitted roots). The engine provides the interface and the escalation wiring; the heuristics themselves are tool/host-specific.

The separate `ToolJudge` (`github.com/v0lka/sp4rk/tools` `judge.go`) is an LLM-backed **advisory** safety evaluation invoked on demand by the host — it is not an automatic gate.

### Strict ToolJudge mode

`ToolJudge.JudgeStrict(ctx, StrictJudgeRequest)` is the conservative LLM primitive a host may invoke when deciding whether a soft confirmation escalation can be auto-resolved. `StrictJudgeRequest` supplies `ToolName`, raw `Input`, `TaskContext`, and `ToolSource`; the judge adds compact environment information and the current `SessionRoots(ctx)`.

Strict mode differs from advisory `Judge` in three security-relevant ways:

1. It performs one LLM call for every invocation. Internal-tool and in-session path fast paths are disabled, and no verdict cache is read or written.
2. It serializes a JSON envelope rather than interpolating the tool arguments into instructions. Raw input and host-supplied session-directory values are wrapped in `untrusted-content` boundaries, and directory lines are sanitized before serialization.
3. It accepts only the strict `VERDICT`/`REASON` response contract. Missing provider, request construction failure, provider error, cancellation/timeout, nil response, and unparseable output all return `VerdictConfirm`. Provider error text is excluded from strict logs because it may echo sensitive tool input.

Strict mode remains advisory to the host: it returns an allow/confirm recommendation and never bypasses `PolicyAlwaysDeny`, a hard `JudgeSeverity`, or registry confirmation policy by itself.

## Confirmation Flow

`PolicyUserConfirm` tools (and judge-escalated `PolicyAlwaysAllow` tools) go through the confirmation flow:

```go
type ConfirmationRequest struct {
    ToolName       string          `json:"tool_name"`
    Input          json.RawMessage `json:"input"`
    JudgeReasoning string          `json:"judge_reasoning,omitempty"`
    JudgeSeverity  JudgeSeverity   `json:"judge_severity"` // hard (zero) | soft; plain PolicyUserConfirm gates escalate as hard
    DisableJudge   bool            `json:"disable_judge,omitempty"`
}

type ConfirmationResponse int

const (
    ConfirmAllowOnce    ConfirmationResponse = iota // allow this single execution
    ConfirmDeny                                     // deny this execution
    ConfirmDenyAndStop                              // deny and cancel the entire task
)

type ConfirmFunc func(ctx context.Context, req ConfirmationRequest) (ConfirmationResponse, error)
```

```
registry.Execute()
  → effective policy == PolicyUserConfirm   (or judge-escalated PolicyAlwaysAllow)
  │
  ├─ no ConfirmFunc configured? → FAIL-CLOSED: return error ToolResult
  │                                ("no ConfirmFunc is configured")
  ├─ ConfirmFunc(ctx, ConfirmationRequest{ToolName, Input, JudgeReasoning, JudgeSeverity})
  │      │
  │      ▼
  │   host responds (Allow / Deny / Deny & Stop)
  │
  ├─ ConfirmAllowOnce     → execute tool
  ├─ ConfirmDeny          → return error ToolResult (agent sees the denial)
  └─ ConfirmDenyAndStop   → cancel the task context
```

The host wires `ConfirmFunc` via `sp4rk.Config.ConfirmFunc` (framework) or `registry.SetConfirmFunc` (direct registry use). `WithAutoApprove` installs a callback that approves every `PolicyUserConfirm` call — intended for throwaway/sandboxed workspaces.

**Fail-closed is the default.** Without a `ConfirmFunc`, a `PolicyUserConfirm` tool is denied rather than executing silently, so an injected instruction cannot trigger a mutation in a default-configured engine. To relax individual tools, the host calls `registry.SetPolicyOverride(name, PolicyAlwaysAllow)`.

## MCP Tool-Shadowing Protection

MCP servers are untrusted; a malicious or compromised server could advertise a tool named `read_file` or `bash_exec` to intercept calls intended for built-in tools. The registry stores each tool's source category explicitly at registration (`RegisterWithSourceCategory`) and **rejects any MCP-categorized registration that would overwrite an existing non-MCP tool**. Built-in tools can always replace MCP tools, and an MCP server may re-register its own tools on reconnect.

## MCP stdio Environment Allowlist (Supply-Chain Defense)

A stdio MCP server runs an arbitrary third-party command declared in config, so its subprocess inherits only an allowlisted subset of the host environment (`safeStdioEnv`): `PATH`, `HOME`, `USER`, `SHELL`, `USERPROFILE`, `LANG`, `TERM`, `TMPDIR`, `TEMP`, `TMP`, `APPDATA`, `LOCALAPPDATA`, `SYSTEMROOT`, `COMSPEC`, `PATHEXT`, the proxy variables (`HTTP_PROXY`/`HTTPS_PROXY`/`NO_PROXY`/`ALL_PROXY` in both cases, with embedded credentials stripped), CA trust anchors (`SSL_CERT_FILE`, `SSL_CERT_DIR`, `REQUESTS_CA_BUNDLE`, `CURL_CA_BUNDLE`, `NODE_EXTRA_CA_CERTS`), Python venv/conda activation (`VIRTUAL_ENV`, `CONDA_*`), and `LC_*` locale variables. Host secrets — LLM API keys, proxy credentials — are never forwarded implicitly. Explicit `cfg.Env` entries are applied on top of the allowlist and always win, so a host that wants to pass a secret to a server declares it explicitly, and one that wants stricter isolation can override inherited vars (e.g. `HOME`) through `cfg.Env`.

## Indirect Prompt Injection Defense

sp4rk protects the LLM context from untrusted tool output that could contain hidden instructions (prompt injection). The defense lives in `github.com/v0lka/sp4rk/security` and is applied by the memory layer's `ContextWindow`.

### Content Delimiting (Spotlighting)

Tool output from untrusted sources is wrapped in `<untrusted-content>` XML tags before it enters the LLM context:

```xml
<untrusted-content source="web_fetch">
...external data here...
</untrusted-content>
```

```go
const UntrustedTag = "untrusted-content"

func WrapUntrustedContent(content, source string, metadata map[string]string) string
func StripUntrustedTags(content string) string
```

`WrapUntrustedContent` wraps content in `<untrusted-content>` tags with a `source` attribute (the producing tool's name) and optional metadata attributes. The content is **first sanitized** via `StripUntrustedTags` to prevent tag-breakout attacks.

The wrapping is applied by `memory.ContextWindow.BuildPrompt` (when constructed with `InjectionDefenseEnabled: true`) — the last point before content reaches the LLM API. It runs **after** history mutation and pruning, so the defense always wraps the final content the model sees.

The boundary survives compaction. When a compaction strategy freezes verbatim tool messages into a compacted prefix, `ContextWindow.wrapUntrustedInFrozenPrefix` re-wraps any tool message whose source step was flagged `IsUntrusted` via `WrapUntrustedContent` (the summarization strategy likewise wraps untrusted observations). The strategies build their verbatim tool messages through a path that bypasses the live `BuildPrompt` wrap, so without this re-application previously-fenced untrusted content would re-enter context raw after compaction; the boundary guarantees it never does.

### Trust Classification

A tool marks its output as untrusted via the `IsUntrusted() bool` method on the `Tool` interface (set through `BaseTool.Untrusted`). The executor sets `Step.IsUntrusted` after tool execution; the context builder reads that flag to decide whether to wrap.

Typical untrusted tools:

- Web fetch / search tools — return arbitrary internet content.
- MCP-backed tools — return data from external servers. MCP tools are always considered untrusted regardless of their `IsUntrusted()` value.
- Filesystem tools reading untrusted paths — return file contents that may contain injected instructions.

Tools that return only internally generated, trusted data (e.g. the `finish` tool) do not set the flag, so their output is not wrapped.

### Tag-Breakout Protection

`StripUntrustedTags` escapes literal `<untrusted-content` and `</untrusted-content` patterns to prevent attackers from closing the wrapper early. Only the **leading `<`** of a matching tag is replaced with `&lt;`; the rest of the tag text is preserved as-is. Matching is case-insensitive and tolerates whitespace.

This operates on **literal character sequences only**. HTML-entity-encoded variants (e.g. `&#60;/untrusted-content>`) are **not** escaped — intentionally. LLMs process raw text tokens; they do not decode HTML entities when interpreting context boundaries, so escaping entity-encoded variants would add noise without improving security.

### No LLM-Based Output Judging

The defense does **not** include LLM-based output-content judging for injection detection. Judging whether content constitutes an attack is delegated to external firewall/proxy defenses. This keeps latency predictable and avoids token waste on detection tasks.

### No Domain Gate

All untrusted tools (including web fetchers) receive the same wrapping treatment. There is no domain allowlist or content-type gate before wrapping — the wrapping is unconditional for any tool marked as untrusted.

## Configuration (Engine-Level)

The engine exposes policy as Go values, not a configuration file. A host configures policies programmatically:

```go
// Per-tool overrides (host calls these on the registry)
registry.SetPolicyOverride("write_file", tools.PolicyUserConfirm)
registry.SetPolicyOverride("bash_exec",  tools.PolicyUserConfirm)
registry.SetPolicyOverride("web_search", tools.PolicyAlwaysAllow)
registry.SetPolicyOverride("web_fetch",  tools.PolicyAlwaysAllow)

// The confirmation channel (host supplies this)
framework := sp4rk.New(sp4rk.Config{
    ConfirmFunc: func(ctx context.Context, req tools.ConfirmationRequest) (tools.ConfirmationResponse, error) {
        // prompt the user; return ConfirmAllowOnce / ConfirmDeny / ConfirmDenyAndStop
    },
})

// Prompt-injection defense is enabled on the context window
cw := memory.NewContextWindow(memory.ContextWindowConfig{
    InjectionDefenseEnabled: true, // wrap untrusted tool outputs
    // ...system prompt, model meta, thresholds, strategy...
})
```

File-based defaults, session roots, and blacklist regexes are host-application configuration; the engine only consumes the resulting `ToolPolicy` values and `ConfirmFunc`.

## Invariants

- A `PolicyUserConfirm` tool with no `ConfirmFunc` configured is **always denied** (fail-closed), never executed silently.
- `PolicyAlwaysDeny` is never bypassed — not by a judge, not by a confirmation, not by an override that does not change the policy.
- A `ToolJudger` escalation requires `allow=false` **with non-empty reasoning**; `allow=false` with empty reasoning proceeds.
- Path extraction skips tokens that consist entirely of separators — a POSIX `//` run or a drive prefix followed by only separators (`C:\\`) — because they are shell-language artifacts (sed addresses, comment markers, integer division, escaped drive roots) that name no location. The skip never applies to a token carrying path components: `/etc/passwd`, `//etc/passwd`, `C:\`, and `C:\\Windows\win.ini` all remain reportable, anchored out-of-root writes still escalate, and command blacklists (matched on the raw command string, before path analysis) are unaffected.
- Untrusted tool output is wrapped in `<untrusted-content>` before it reaches the model, unconditionally for any tool whose `IsUntrusted()` is true (or that is MCP-sourced).
- The untrusted-content boundary survives compaction: `ContextWindow` re-wraps untrusted tool messages retained in the frozen compacted prefix, so untrusted content never re-enters LLM context raw after compaction.
- `WrapUntrustedContent` always sanitizes its content with `StripUntrustedTags` first.
- An MCP tool registration can never overwrite an existing non-MCP tool.
- The LLM-powered advisory `ToolJudge` cache key incorporates session roots and partitions cached verdicts by directory scope; its prompt carries the same wrapped scope data.
- `ToolJudge.JudgeStrict` performs an uncached, no-fast-path LLM evaluation per invocation and maps every construction/provider/timeout/parse failure to `VerdictConfirm` without logging potentially sensitive provider diagnostics.
- The LLM-powered `ToolJudge` verdict parser fails **safe** to `VerdictConfirm` on any unrecognized or ambiguous verdict: verdict tokens are matched whole-token (case-insensitive), so negations of allow-words (e.g. `DISALLOW`, `DISAPPROVE`) are never misclassified as `VerdictAllow`. An LLM error likewise yields `VerdictConfirm`. See [../contracts/tools.md](../contracts/tools.md) for the verdict vocabulary.

## Anti-Patterns

- Treating the `ToolJudge` as a primary safety mechanism — it is advisory, not a gate. The hard boundary is the policy layer.
- Relying on path-locality auto-approval as a security control — the SDK ships the primitive (`IsWithinRoot`/`AllPathsInSessionRoots`), but whether the judge fast-path is *enabled* and which roots are configured is host wiring, not an engine guarantee.
- Calling `tool.Execute` directly instead of `registry.Execute` — bypasses policy resolution, the judge gate, the confirmation flow, and MCP shadowing protection.
- Constructing a `ContextWindow` with `InjectionDefenseEnabled: false` in production — disables untrusted-content wrapping for all tool output.
- Expecting the engine to enforce a command blacklist — blacklists are host/tool-specific `ToolJudger` heuristics, not engine behavior.

## Related Specs

- [layers.md](layers.md) - Where the security primitives live in the package hierarchy
- [data-flow.md](data-flow.md) - Tool execution flow and the fail-closed gate
- [../contracts/tools.md](../contracts/tools.md) - The Tool and ToolJudger interface contract
