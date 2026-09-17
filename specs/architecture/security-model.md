# Security Model

## Context

sp4rk executes arbitrary tools (filesystem operations, shell commands, web requests) on behalf of an LLM, and ingests external data into the model's context. The engine provides two complementary security primitives: a **tool-execution policy layer** that gates whether and how a tool runs, and a **prompt-injection defense** that delimits untrusted external content before it reaches the model. On top of these, the engine ships a **deterministic shell-command analysis** (the flowsh criteria C1–C8, see [Shell Command Analysis](#shell-command-analysis)) that hosts attach to every `bash_exec`/`posh_exec` call. This spec describes the engine-level primitives only — a host application layers additional wiring (workspace/temp auto-approval, shell blocklist patterns, symlink-forced confirmation) on top of these.

## Scope Boundary

sp4rk ships the *primitives*; the host application assembles the *policy*. The SDK provides:

- the `ToolPolicy` enum, per-tool policy resolution, and the `ToolGroup` capability groups (`tools/group.go`),
- the `ToolJudger` interface and the fail-closed confirmation flow (`ConfirmFunc`/`ConfirmationRequest`/`ConfirmationResponse`),
- the path-locality primitive — [`tools.IsWithinRoot`](#path-locality-primitive) and [`AllPathsInSessionRoots`](#path-locality-primitive), consulted by the judge fast-path auto-approval and every tool-argument containment check (see [Path-Locality Primitive](#path-locality-primitive)),
- the prompt-injection defense (`security.WrapUntrustedContent`/`StripUntrustedTags`), and
- the MCP tool-shadowing guard.

The host application owns, and this spec deliberately does **not** cover:

- the *decision* to enable path-locality auto-approval (whether the judge fast-path is on, which session roots are configured, confirmation UX),
- shell blocklist patterns (the host-supplied regex list passed to the shell tool constructors; the engine ships none — the structural floor is the [flowsh criteria](#shell-command-analysis)),
- symlink-forced confirmation gating.

Those decisions depend on host-specific concepts (session roots, command grammars, OS conventions) and are wired by the embedding application around the SDK registry.

<a id="path-locality-primitive"></a>
### Path-Locality Primitive

`tools.IsWithinRoot(ctx, root, path)` is the canonical path-containment check consulted by the judge fast-path (`AllPathsInSessionRoots`), file-tool argument resolution, symlink inside/outside classification, working-directory validation, cache-coherence gating, and ignore-root selection. It resolves symlinks through the longest existing prefix of both paths, then folds letter case **only when the session flag is set** — determined by probing the filesystem at session-root resolution time (`pathutil.DetectCaseInsensitive`; the probe's result is memoized per probed directory for the process lifetime, so repeated calls never re-create the temporary probe file):

- **Case-insensitive filesystems** (macOS APFS, Windows NTFS): the fold matches the on-disk reality, so a not-yet-existing target written with different casing than the session root is still recognized as local.
- **Case-sensitive filesystems** (Linux ext4/tmpfs/btrfs, and case-sensitive APFS volumes): containment is case-sensitive, so distinct-cased siblings such as `/ws/Project` and `/ws/project` are never conflated. Applying the fold unconditionally here would be an authorization bypass — a non-local sibling whose name differs only by case would be auto-approved as "local".

The flag defaults to **case-sensitive** (fail-safe): when detection is impossible (no existing ancestor, unreadable root), containment never widens. Case folding never weakens escape prevention — it can only cause an otherwise-mismatched path to be considered "within" a root; the `..` traversal and symlink-escape logic is unaffected.

The flag is attached automatically by `tools.WithWorkspacePath` (which probes the workspace root) and may be overridden by `tools.WithCaseInsensitivePaths`. Hosts that construct their context without `WithWorkspacePath` (e.g. `WithWorkspacePathNoProbe`) inherit the fail-safe case-sensitive default unless they attach the flag explicitly.

Harmless special-device paths are exempted from path-locality determination via [`tools.IsHarmlessDevicePath`](#path-locality-primitive). A path that is a bit-bucket or error-sink pseudo-device — `/dev/null` and `/dev/full` on POSIX, `NUL` on Windows (matched case-insensitively as the final path component) — cannot leak data outside the workspace nor persist unwanted changes, so `AllPathsInSessionRoots` and `AllPathsInDir` skip it. This is why `cat file > /dev/null`, `read_file /dev/null`, or `write_file NUL` are not forced to confirm as out-of-workspace operations. The symlink gate deliberately does **not** consult it: it resolves every path component via `Lstat` rather than trusting the path string, so harmless-looking names that are actually symlinks (e.g. macOS `/dev/stdin`, which points into `/dev/fd/*`) are still walked and resolved.

### Separator-Run Shell Tokens Are Not Paths

Shell commands routinely contain runs of slashes that are artifacts of the shell language, not filesystem paths: the trailing `//` of a sed address (`sed 's/.*function //'`), a comment marker inside an echoed string (`echo "// TODO fix" >> notes.md`), an integer-division operator (`echo $(( total // count ))`), or an escaped PowerShell drive root (`C:\\`). The path extractors behind path-locality — `tools.ResolveShellPathTokens` (the shell-dialect-aware extractor behind `PathsOutsideRoots`) and `tools.ExtractPaths` (the JSON-input extractor behind `AllPathsInSessionRoots`) — **skip tokens that consist entirely of separators**: a POSIX run of two or more slashes (`//`, `///`, …), or a two-character drive prefix followed by two or more separators and no path component (`C:\\`). Such a token carries no path component and names no out-of-root location; resolving it anyway would clean a bare `//` to the filesystem root `/`. A phantom root in the extracted set forced a false-positive confirmation of entirely in-root commands — and it poisoned the `AllPathsInSessionRoots` fast-path, which requires *every* extracted path to be in-root.

The skip is narrow by construction (`tools.isPureSeparatorRunToken`) and preserves every containment guarantee:

- A single `/` never matches the absolute-path pattern at all (it requires at least one character after the leading separator), so the POSIX form skips only runs of **two or more** slashes.
- `cat //etc/passwd` is still reported — the token carries the `etc/passwd` path components, so it is not a pure separator run.
- On the drive form a single trailing separator is a drive root (`Get-Content C:\`) and a token with a component is a real path (`Get-Content C:\\Windows\win.ini`); both stay reportable. Only the pure separator run (`C:\\`) is skipped.
- Anchored out-of-root writes still escalate: `cat /etc/passwd`, `echo x > /etc/cron.d/newjob` (leaf missing, parent directory existing), and `rm -rf /.` all resolve outside the roots and are reported.
- A separator-run artifact cannot mask a real escape: a sed `//` and `/etc/passwd` in the same command still fails containment, because each token is extracted independently.
- Shell blocklists and flowsh criteria are untouched: the built-in shell tools' `ToolJudger` matches the raw command string against its host-supplied blocklist patterns **before** anything else (and that reason takes precedence), and the structural flowsh criteria (C3/C4 on system-path and out-of-roots writes) key on effect targets, not on extracted string tokens — so `rm -rf /` escalates either way: via a host blocklist pattern when the user supplied one, and structurally via C4 (irreversible class-E write targeting the filesystem root, outside any session root) even with an empty blocklist. The separator-run skip cannot mask it: the skip removes the `//` token from path extraction (exactly as the bare `/` never matched the absolute-path pattern), but C4 sees the flowsh FSWrite target, which is unaffected. See [../domains/tool-system/builtins.md](../domains/tool-system/builtins.md).

### Assignment-Form Command Substitutions Are Assessable

The shell path extractors (`tools.ResolveShellPathTokens`, the shell-dialect-aware extractor behind `PathsOutsideRoots`, and the extractors behind `tools.DetectSymlinksInToolInput`) treat a shell expansion whose value cannot be determined statically as **unresolved**: the extractors flag it, and the host's symlink gate receives the `suspicious` flag. (The built-in shell tools' `Judge` no longer consumes these extractors — its static unresolvable-token and containment stages were removed in favor of the [flowsh criteria](#shell-command-analysis); the extractors remain the path source for symlink detection.) One expansion shape is the exception — a **validated assignment-form command substitution**: a pure `VAR=$(...)` (or `VAR="$(...)"` wrapping exactly one substitution) whose inner command is **fully assessable** is *resolved*, not unresolved, so it does not raise the `suspicious` flag.

"Fully assessable" reuses the same assessment pipeline, applied recursively (`tools.commandSubstitutionAssessable`): the inner command must parse; a fresh binding summary (`tools.collectShellEnvBindings`) must report no opaque construct (`eval`/`source`/`let`) and no dynamic binding; every inner word must be statically assessable — literal fragments are, a pure nested substitution recurses, and a plain `$NAME`/`${NAME}` reference is assessable only when `NAME` is itself bound solely by an assessable substitution — and `tools.UnresolvablePathTokens(inner, ShellBash)` must be empty.

When a name is promoted (`tools.envBindings.cmdSubst`), a later `$VAR` reference is **assessable**: it does not escalate, and it does not resolve to a literal either — the name stays deliberately **not resolvable** because its value is the substitution's unknown output. The approved substitution's **inner literal paths are surfaced** to the symlink walk (`tools.extractBashPaths` recurses into it), so a target reachable through a symlink only inside the substitution is still detected.

The exception is fail-closed by a **union** over all bindings (`tools.envBindings.resolveCmdSubst`): a name is promoted only when *every* recorded binding of it is a pure assessable substitution and it carries no literal value and no other dynamic binding. Any single literal, non-assessable, or differently-shaped binding — or an opaque construct — poisons the whole name back to dynamic, and the call stays unresolved.

The scope is deliberately narrow:

- A **bare `$(...)` in argument position** (`cat $(echo x)`) is not eligible — there is no assignment binding to promote — and stays unresolved.
- A reference to a name **not** bound by an assessable substitution — including environment variables such as `$HOME` — stays unresolved.
- The substitution's **stdout is not analyzed**: only the inner command's static assessability and its literal paths are assessed. This is an accepted residual risk of the exception (a substitution whose inner command reads a file can yield a value no pass inspects).

`posh_exec` mirrors the bash rule with a static binding `$NAME = <literal RHS>`. Its char-level tokenizer bounds the RHS at the next statement/pipeline terminator and treats the binding as static only when the RHS carries no expandable token (`$`-expansion, backtick escape, or expandable double-quoted content); a bare `$NAME` reference is then literal only when the name is bound **exactly once** with a static RHS. Recognition is fail-closed for the same cases as bash: a suffixed reference (`$X/secret`), a name bound more than once, an unbound name, and `$(...)`/`(...)` all keep every reference unresolved.

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

type JudgeReasonCode string // stable, machine-checkable classification of the reason; e.g. "command_blacklist", "ssrf_private_address", "outside_session_roots", "unassessable_path"

type JudgeOutcome struct {
    Allow      bool
    Reason     string
    Severity   JudgeSeverity
    ReasonCode JudgeReasonCode
}

type ToolJudger interface {
    Judge(ctx context.Context, input json.RawMessage) JudgeOutcome
}
```

For a tool whose effective policy is `PolicyAlwaysAllow`, the registry calls `Judge` **before** execution. If the judge returns `Allow=false` **with non-empty Reason**, the call is escalated to user confirmation via `ConfirmFunc`. An outcome of `Allow=false` with empty Reason is treated as "no concern to report" and the tool proceeds.

The `Severity` classifies the reason so hosts can decide what an escalation means: **`hard`** marks a fired security control (shell blocklist pattern, a flowsh criterion, SSRF) — hosts treat it as never auto-resolvable and do not pass it through an advisory auto-approval path; **`soft`** marks a scope question (path containment outside session roots — evaluated on the symlink-resolved path, so a symlink that escapes the roots is denied containment like any other out-of-roots target; credential access without an exfil pair) that a strict judge may resolve. The engine escalates both severities identically (user confirmation) and delivers the severity to the host via `ConfirmationRequest.JudgeSeverity`. `JudgeSeverityHard` is the zero value — a judge outcome or escalation that arrives unclassified is treated as `hard` (fail-closed).

`ReasonCode` is the typed classification of the reason — the machine-checkable counterpart of the human-readable `Reason` prose, delivered to the host as `ConfirmationRequest.JudgeReasonCode`. Codes are a cross-repository contract: hosts key deterministic policy decisions off the code instead of matching prose, so a published code is never renamed or reused (new classifications add new codes). The empty value means unclassified — hosts decide unclassified outcomes by their own fail-closed policy. The built-in judges publish `command_blacklist`, `outside_session_roots`, `ssrf_private_address`, `ssrf_protection_degraded`, `unassessable_url`, `unassessable_path`, `git_internal_path`, and the flowsh criteria `command_exfil_flow`, `command_privilege_escalation`, `command_system_write`, `command_destructive_outside_roots`, `command_download_cradle`, `command_unbounded_analysis`, `credential_access`, and `command_analysis_unavailable` (a failed analyzer — the shell Judge fails closed) (see [Shell Command Analysis](#shell-command-analysis) and [../domains/tool-system/builtins.md](../domains/tool-system/builtins.md)); `unresolvable_path_token` is retained as a published code for contract stability but is no longer fired by any built-in judge; `symlink_escape` and `symlink_suspicious` are host-set classifications.

This is the SDK hook a host uses to implement its own safety checks (for example, a file tool whose `Judge` flags paths outside permitted roots). The engine provides the interface and the escalation wiring; the heuristics themselves are tool/host-specific — for the shell tools the heuristic is split between the host-supplied blocklist patterns and the engine's flowsh criteria below.

The separate `ToolJudge` (`github.com/v0lka/sp4rk/tools` `judge.go`) is an LLM-backed **advisory** safety evaluation invoked on demand by the host — it is not an automatic gate.

### Strict ToolJudge mode

`ToolJudge.JudgeStrict(ctx, StrictJudgeRequest)` is the conservative LLM primitive a host may invoke when deciding whether a soft confirmation escalation can be auto-resolved. `StrictJudgeRequest` supplies `ToolName`, raw `Input`, `TaskContext`, `ToolSource`, and optionally `AnalysisContext` (the one-line flowsh digest for shell tools — empty for non-shell tools, and the envelope field is omitted entirely); the judge adds compact environment information and the current `SessionRoots(ctx)`.

Strict mode differs from advisory `Judge` in three security-relevant ways:

1. It performs one LLM call for every invocation. Internal-tool and in-session path fast paths are disabled, and no verdict cache is read or written.
2. It serializes a JSON envelope rather than interpolating the tool arguments into instructions. Raw input, host-supplied session-directory values, and the host's judge reasoning (host-generated, but it may quote fragments of the untrusted command, e.g. an unresolvable path-like token) are wrapped in `untrusted-content` boundaries and line-sanitized before serialization.
3. It accepts only the strict `VERDICT`/`REASON` response contract. Missing provider, request construction failure, provider error, cancellation/timeout, nil response, and unparseable output all return `VerdictConfirm`. Provider error text is excluded from strict logs because it may echo sensitive tool input.

Strict mode remains advisory to the host: it returns an allow/confirm recommendation and never bypasses `PolicyAlwaysDeny`, a hard `JudgeSeverity`, or registry confirmation policy by itself.

<a id="shell-command-analysis"></a>
## Shell Command Analysis (flowsh criteria)

The engine ships a **deterministic structural analysis** for `bash_exec`/`posh_exec` commands, built on [`github.com/v0lka/flowsh`](https://github.com/v0lka/flowsh) (AST-based, bash and PowerShell dialects, never executes the analyzed command). It runs **once per call**: the host calls `tools.AnalyzeShellCommandForJudge(ctx, toolName, input)` and attaches the result with `tools.WithShellAnalysis`; the built-in shell tools' `Judge` then returns the winning criterion's outcome verbatim (`tools.ShellJudgeOutcome`) — the Judge never recomputes. With nothing attached it returns an empty outcome and defers to the LLM judges; with an **error** attached (a failed analyzer/KB load — sticky) it **fails closed** with the hard canonical `command_analysis_unavailable` reason, so a call never runs with the floor silently absent.

The criteria are evaluated in fixed priority (C1 > … > C8; all fired criteria are recorded in the digest, the winner sets the outcome). Scores/grades are never thresholds — only structural facts (effect kinds, targets, destructive KB class, exfiltration pairs) fire criteria:

| # | Condition | Reason code | Severity | Canonical |
| --- | --- | --- | --- | --- |
| C1 | exfiltration pair (secret read → tainted network egress) | `command_exfil_flow` | hard | yes |
| C2 | privilege-escalation effect (e.g. SUID install) | `command_privilege_escalation` | hard | yes |
| C3 | direct write/metadata on a system path (`/etc`, `/usr`, `/boot`, `/bin`, `/sbin`; Windows `c:\windows`, `c:\program files`, `c:\program files (x86)`, component-boundary matched) or a non-harmless raw device | `command_system_write` | hard | yes |
| C4 | irreversible destructive command (KB class D/E) writing outside the session roots | `command_destructive_outside_roots` | hard | yes |
| C5 | analyzer ⊤/conservative **with** network egress (download-cradle shape) | `command_download_cradle` | hard | yes |
| C6 | analyzer ⊤/conservative **without** network egress, **or** an irreversible write whose target the analyzer could not resolve (⊤ target — e.g. abbreviated PowerShell parameters) | `command_unbounded_analysis` | hard | **no** |
| C7 | credential access without an exfil pair | `credential_access` | soft | — |
| C8 | direct FS effect outside the session roots — writes/metadata on a non-system path, and reads even of a system path (system writes/metadata are C3's; raw-device reads are exempt) | `outside_session_roots` (reused) | soft | — |

**Canonicality** (also exposed as `ShellAnalysis.Canonical` and per-criterion in the digest) marks reasons a host must never auto-override: C1–C5 are canonical; C6 — the analyzer's ⊤ *limitation*, which routine local scripts, unfamiliar CLIs and an unresolved destructive-write target produce — is hard but non-canonical, so a strict judge may clear it. Containment (C4/C8) consults only path-shaped FS-effect targets (operand noise like `-30`, `+x`, `s/foo/bar/g` is discarded); relative targets anchor to the command's `working_directory` else the ctx workspace; empty session roots disable C4/C8; POSIX-absolute targets are `path.Clean`ed so classification is host-independent.

The engine deliberately ships **no blocklist patterns**: the constructor-supplied regex list is host policy, and these criteria are the dialect-aware floor. The shell tools' `Judge` checks the host blocklist **first** (a match wins over every criterion), then returns the attached analysis's winning outcome. The static unresolvable-token and shell-path-containment stages the `Judge` formerly ran were **removed** by explicit decision: C6 covers input the static walker cannot see through, C4/C8 cover out-of-root scope, and the flowsh effect IR assesses both more precisely than token walking.

The compact digest (`sp4rk-shell-analysis/v1`: schemaVersion, lang, top/conservative, effects, score with exfil pairs, destructive classes, fired criteria; no why-traces, no input echo) is judge-facing evidence: `JudgeStrict` carries it in `StrictJudgeRequest.AnalysisContext` behind an untrusted `shell_analysis` boundary, and the advisory `Judge` renders it as a `## Static Analysis Report` block (participating in the advisory cache key). Judge prompts are taught the semantics — the digest is data, `score.grade` is the *inherent* destructiveness of the command text, `top`/`conservative` are analyzer limits, and nothing follows from the digest's absence.

## Confirmation Flow

`PolicyUserConfirm` tools (and judge-escalated `PolicyAlwaysAllow` tools) go through the confirmation flow:

```go
type ConfirmationRequest struct {
    ToolName        string          `json:"tool_name"`
    Input           json.RawMessage `json:"input"`
    JudgeReasoning  string          `json:"judge_reasoning,omitempty"`
    JudgeSeverity   JudgeSeverity   `json:"judge_severity"` // hard (zero) | soft; plain PolicyUserConfirm gates escalate as hard
    JudgeReasonCode JudgeReasonCode `json:"judge_reason_code,omitempty"` // typed classification; empty for plain gates and unclassified outcomes
    DisableJudge    bool            `json:"disable_judge,omitempty"`
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
  ├─ ConfirmFunc(ctx, ConfirmationRequest{ToolName, Input, JudgeReasoning, JudgeSeverity, JudgeReasonCode})
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
- A published `JudgeReasonCode` is a stable contract: it is never renamed or reused, and the empty value means unclassified (hosts decide unclassified escalations by their own fail-closed policy, never by matching the `Reason` prose).
- Path extraction skips tokens that consist entirely of separators — a POSIX `//` run or a drive prefix followed by only separators (`C:\\`) — because they are shell-language artifacts (sed addresses, comment markers, integer division, escaped drive roots) that name no location. The skip never applies to a token carrying path components: `/etc/passwd`, `//etc/passwd`, `C:\`, and `C:\\Windows\win.ini` all remain reportable, anchored out-of-root writes still escalate, and neither the host blocklist (matched on the raw command string, before any analysis) nor the flowsh write criteria (keyed on effect targets, not extracted tokens) is affected.
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
- Expecting the engine to ship blocklist patterns — the regex list is host/tool-specific policy (the engine's shell floor is the structural [flowsh criteria](#shell-command-analysis)); also do not expect the flowsh criteria to fire without host wiring: the host must attach the analysis per call (`AnalyzeShellCommandForJudge` + `WithShellAnalysis`), otherwise the shell tools' `Judge` defers to the LLM judges.

## Related Specs

- [layers.md](layers.md) - Where the security primitives live in the package hierarchy
- [data-flow.md](data-flow.md) - Tool execution flow and the fail-closed gate
- [../contracts/tools.md](../contracts/tools.md) - The Tool and ToolJudger interface contract
