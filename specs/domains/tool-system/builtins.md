# Built-in Tools

## Role

The SDK ships a catalog of filesystem, search, web, execution, and agent-infrastructure tools that implement `tools.Tool`. They are the out-of-the-box capabilities an agent can call; hosts register the subset they need into a `ToolRegistry`. `ask_user` is **not** an SDK built-in — interactive prompting is a host-application concern.

## Key Files

- `github.com/v0lka/sp4rk/tools/builtins` — built-in tool implementations, `file_reader.go` (`ReadFileRange`, streaming O(1)-memory line reader), `limits.go` (per-tool truncation limit types)
- `github.com/v0lka/sp4rk/tools` (names) — `ToolReadFile`/`ToolWriteFile`/`ToolBashExec`/… name constants mirroring registration names, and `IsShellExecTool` (classifies the shell-executing tools `bash_exec`/`posh_exec`)
- `github.com/v0lka/sp4rk/tools/builtins` (web search) — `web_search/` provider abstraction (brave, duckduckgo, exa, tavily)
- `github.com/v0lka/sp4rk/tools/builtins` (blackboard-backed) — `read_step_output`, `list_step_outputs`, `read_final_result`, `read_attachment`, `tool_result_read`, `update_checklist`
- `github.com/v0lka/sp4rk/agent` — `FinishTool`
- `github.com/v0lka/sp4rk/skills` — `ReadSkillResourceTool`
- `github.com/v0lka/sp4rk/sysproc` — `HideConsole` (suppresses console windows for `posh_exec`, `ripgrep`, and env-info version probes under a GUI-subsystem host)

## Behavior

### Tool Catalog

| Tool | Category | Default Policy | Untrusted | Description |
| ---- | -------- | -------------- | --------- | ----------- |
| `bash_exec` | Execution | `user_confirm` | yes | Shell command execution with timeout, the host-supplied blocklist (no patterns ship), and the deterministic flowsh criteria C1–C9. |
| `posh_exec` | Execution | `user_confirm` | yes | Windows PowerShell command execution with timeout, the host-supplied blocklist (no patterns ship), and the deterministic flowsh criteria C1–C9. |
| `read_file` | File | `always_allow` | yes | Read file contents (streaming, O(1) memory, default 2000-line window). |
| `write_file` | File | `user_confirm` | no | Create/overwrite a file. |
| `edit_file` | File | `user_confirm` | no | Apply targeted find-and-replace edits. |
| `list_directory` | File | `always_allow` | yes | List directory contents. |
| `create_directory` | File | `user_confirm` | no | Create a directory recursively. |
| `delete_directory` | File | `user_confirm` | no | Remove a directory recursively. |
| `delete_file` | File | `user_confirm` | no | Remove a single file. |
| `glob` | Search | `always_allow` | yes | Glob-pattern file matching; honours `.gitignore`/`.aiignore` via the context `IgnoreChecker` when present. |
| `ripgrep` | Search | `always_allow` | yes | Fast regex content search (shells out to `rg`); honours `.gitignore`/`.aiignore` via the context `IgnoreChecker` when present. |
| `semantic_search` | Search | `always_allow` | yes | Vector similarity search (optional; see [../embedding.md](../embedding.md)). |
| `web_fetch` | Web | `always_allow` | yes | Fetch URL content as markdown. |
| `web_search` | Web | `always_allow` | yes | Search the web (optional; needs a search-provider config). |
| `finish` | Agent | internal | no | Signal task/step completion (auto-appended to every run). |
| `batch` | Agent | `always_allow` | no | Dispatch multiple tool calls in one turn (intercepted at the executor level). |
| `read_step_output` | Agent | `always_allow` | no | Read a specific completed step's output (blackboard-backed). |
| `list_step_outputs` | Agent | `always_allow` | no | List completed step outputs. |
| `read_final_result` | Agent | `always_allow` | no | Read the prior task's final result (continuation recovery). |
| `read_attachment` | Agent | `always_allow` | yes | Read the markdown content of a user-attached file by ID (blackboard-backed). |
| `update_checklist` | Agent | `always_allow` | no | Update a step/sub-task checklist; validates Markdown checkboxes. |
| `store_fact` | Agent | `always_allow` | no | Store a keyword-tagged fact to the blackboard. |
| `search_facts` | Agent | `always_allow` | no | Search blackboard facts by keyword. |
| `tool_result_read` | Agent | internal | yes | Read cached tool-result fragments by hash (streaming, O(1) memory for file-backed entries). |
| `read_skill_resource` | Agent | `always_allow` | no | Read a resource file from an activated skill directory (path-traversal safe). |

`ask_user` is intentionally absent from the SDK. Interactive multi-question prompting requires host UI plumbing, so it lives in the host application; the SDK only supplies the primitives such a tool builds on.

### Trust classification

A tool's output is **untrusted** when it may carry adversarial content (web, MCP, filesystem reads of external data). Such tools set `Untrusted: true` on their `BaseTool`; their observations are wrapped in `<untrusted-content>` tags before entering the LLM context when injection defense is enabled (see [../memory/README.md](../memory/README.md)). Mutating filesystem tools (`write_file`, `edit_file`, `delete_*`, `create_directory`) are not marked untrusted.

### File tools

File tools resolve paths via context helpers (`WorkspacePathFrom`/`TempDirFrom`). Relative paths are joined with the workspace root and must stay within it; absolute paths are symlink-resolved and returned regardless of containment (containment is a policy concern, not a parse failure). Containment checks consult `SessionRoots(ctx)` — the union of workspace, temp directory, and any additional roots attached via `WithAllowedRoots` — so all roots are equal peers for path-locality auto-approval, judge fast-paths, symlink classification, and shell working-directory validation (`AllPathsInSessionRoots`, `isPathInSessionRoots`, `validateWorkDir`). `read_file` uses `ReadFileRange` for O(1)-memory streaming of line ranges and is file-backed by default in the `ToolResultCache` (zero content bytes stored; fragments streamed from disk on demand); a read wrapper that implements `ContentBackedReader` opts into content-backed caching for transformed views. Binary files (null bytes in the leading window) are detected and rejected.

**Git-internal guard (mutating tools).** `write_file`, `edit_file`, `delete_file`, `create_directory`, and `delete_directory` share one judge (`judgeWriteInSessionRoots`) that first checks the symlink-resolved target for a ".git" path component at or below the workspace root. A match escalates as a fired security control (`git_internal_path`, hard) **before** the soft containment check: writing the object database, refs, config, or hooks can rewrite history or plant executable code, so the confirmation must never be auto-resolved. Scope is decided by the canonical `tools.IsWithinRoot` containment check (case-folding exactly when locality auto-approval folds), so a case-mismatched spelling of a workspace prefix cannot slip the guard into auto-approval on case-insensitive filesystems; the temp directory and other session roots are outside the guard (their writes are governed by ordinary containment), and regular dotfiles (`.gitignore`, `.github`) are not matched — only the exact ".git" component is.

### Shell tools (`bash_exec`, `posh_exec`)

Both shell tools share one safety model, evaluated by their `ToolJudger.Judge` before execution. Every escalation carries a typed `tools.JudgeReasonCode` — the machine-checkable classification delivered to the host as `ConfirmationRequest.JudgeReasonCode` — alongside its prose reason and severity (see [Judge reason codes](#judge-reason-codes)):

1. **Blocklist** — the raw command string is matched against the constructor-supplied regex list (an invalid pattern fails construction). The list is **host policy**: the engine ships no patterns, and the host decides whether it is empty (c0wrk ships it empty by default and calls it the blocklist). A match returns `allow=false` with the blocklist reason (`command_blacklist`, hard), which **takes precedence** over every criterion.
2. **Flowsh criteria C1–C9** — the host pre-computes the deterministic analysis once per call (`tools.AnalyzeShellCommandForJudge`) and attaches it to the context (`tools.WithShellAnalysis`); the Judge returns the winning criterion's outcome verbatim via `tools.ShellJudgeOutcome` — hard canonical C1–C5, hard non-canonical C6/C7, soft C8/C9. The Judge never runs the analysis engine itself. When no analysis is attached, the Judge returns an empty outcome and defers to the LLM judges; when the attached one carries an **error** (a failed analyzer/KB load — sticky), it **fails closed** with the hard canonical `command_analysis_unavailable` reason.

The former static stages — **unresolvable path tokens** (`unresolvable_path_token`, hard, bash-only) and **shell-path containment** via `PathsOutsideRoots`/`ExistingOrAnchoredPaths` (`outside_session_roots`, soft) — were **removed** by explicit decision: tokens the static walker cannot see through are covered by the C6 unbounded criterion, and out-of-root scope by C4/C9, both of which the flowsh effect IR assesses more precisely than token walking. The published `unresolvable_path_token` code is retained for contract stability but is no longer fired.

**The symlink walk extracts literal paths only.** `DetectSymlinksInToolInput` pulls literal paths from shell commands (bash: literal word fragments from the `mvdan.cc/sh` AST; PowerShell: quote-aware tokenization with comment skipping, `$`-carrying tokens skipped) and does **not** assess expansion suspicion at all: dynamic constructs are the flowsh criteria's domain on the same call. The former `suspicious` return value, its `symlink_suspicious` escalation, and the validated command-substitution machinery that decided when an expansion was benign (`shellEnvBindings.go`, the shell-path resolvers, the PowerShell static-binding tokenizer) were deleted ([decision 006](../../decisions/006-symlink-walk-literal-paths-only.md)); `symlink_suspicious` is retained in the code contract but never fired. See [../../architecture/security-model.md](../../architecture/security-model.md#the-symlink-walk-is-a-literal-path-extractor).

**Separator-run tokens are skipped.** A token consisting entirely of separators — a POSIX run of two or more slashes (`//`, `///`) or a two-character drive prefix followed by only separators (`C:\\`) — is a shell-language artifact, not a path: the trailing `//` of a sed address (`sed 's/.*function //'`), a comment marker (`echo "// TODO fix" >> notes.md`), an integer-division operator (`$(( total // count ))`), or an escaped PowerShell drive root. It carries no path component and names no out-of-root location; resolving a bare `//` would clean it to the filesystem root `/` and force a false-positive confirmation of an entirely in-root command. The skip lives in `tools.isPureSeparatorRunToken`, applied in the JSON-input extractor `tools.ExtractPaths` behind `AllPathsInSessionRoots` (where a phantom root previously defeated the fast-path's *all paths in-root* auto-allow).

Guarantees the skip preserves (regression-tested in `tools/judge_test.go`):

- `cat /etc/passwd`, `echo x > /etc/cron.d/newjob`, and `rm -rf /.` still escalate — their tokens carry real path components, and structurally the flowsh C3/C4/C9 criteria key on effect targets, not extracted tokens.
- `cat //etc/passwd` still reports: `//etc/passwd` carries the `etc/passwd` components, so it is not a pure separator run.
- `Get-Content C:\` (single separator — the drive root) and `Get-Content C:\\Windows\win.ini` are still extracted; only the pure `C:\\` run is skipped.
- The blocklist fires on the raw command string **before** any analysis, so `rm -rf /` is still escalated by a host-supplied `rm -rf` pattern regardless of the skip — and with an empty host blocklist it escalates structurally: the flowsh FSWrite target for the filesystem root is outside every session root, so C4 (irreversible class-E destructive write out-of-roots) fires; its separator-run spelling `rm -rf //` is covered identically because the criteria `path.Clean` POSIX-absolute targets (`//` → `/`).
- A separator-run artifact cannot mask a real escape (`sed 's/.*function //'` alongside `/etc/passwd` still fails containment — tokens are extracted independently), and alongside only in-root paths it no longer injects a phantom root, so judge fast-path auto-approval behaves correctly.

**Shell invocation override (`tools.ShellInvocation`).** Both shell tools carry a launch shape — by default `bash -c <command>` and `powershell.exe -NoProfile -NonInteractive -Command <command>`, constructed via `NewBashExecToolWithInvocation`/`NewPoshExecToolWithInvocation` (the legacy constructors delegate with the defaults). A host MAY let its operator override the launch shape: `Binary` + `Args` argv template in which exactly one element equals `{command}` (`tools.ShellCommandPlaceholder`) and is replaced by the agent's command **as a single argv element** — the command crosses the process boundary verbatim, with no extra quoting layer or string splicing. An override is invalid (fails construction) when the binary is empty or itself the placeholder, the placeholder is missing, duplicated, or embedded inside another argument, or the declared `Kind` is outside the **closed** shell set: `bash`, `sh`, `zsh`, `ksh`, `dash` (bash family) and `powershell`, `pwsh` (PowerShell family). The set is closed because three things key on the declared kind: (1) the **tool description** — a same-family override rewrites only the "Purpose:" header to name the actual launch command and the declared shell, while a cross-family override (the declared kind belongs to the other syntax family) also replaces the family-specific guidance tail with a dialect-neutral one, so the description never mixes syntax families (the default invocation keeps the legacy description byte-for-byte); the description is part of every tool catalog, so it is the channel that informs every agent able to call the tool; (2) the **PowerShell UTF-8 bootstrap** (`posh_exec.commandText`) is applied only for PowerShell-family kinds; (3) the **flowsh analysis dialect** — `tools.ShellAnalysisLangForTool` resolves the dialect from the tool's `DeclaredShellKind` (falling back to the legacy `bash_exec`/`posh_exec` name mapping), and `tools.AnalyzeShellCommandForJudgeWithDialect` accepts the explicit dialect. Shells outside the two families (fish, nushell, csh, cmd.exe, …) have no analyzer; they are rejected at construction rather than silently dropping the deterministic floor or failing closed on every call. Process containment (Unix process-group kill, Windows Job Object), `working_directory` validation, timeouts, blocklist matching, and the C1–C9 criteria are unchanged — the override changes *how* a command is launched, never *what* is checked.

### Judge reason codes

Every built-in `ToolJudger` escalation classifies its reason with a typed `tools.JudgeReasonCode` — the stable, machine-checkable contract the host receives as `ConfirmationRequest.JudgeReasonCode` (see [../../contracts/tools.md](../../contracts/tools.md)); the prose `Reason` may be reworded freely while a published code is never renamed or reused. The built-in mapping:

| Code | Severity | Emitted by |
| ---- | -------- | ---------- |
| `command_blacklist` | hard | `bash_exec`, `posh_exec` — the raw command string matched a host-supplied blocklist pattern (checked first; wins over every criterion) |
| `command_exfil_flow` | hard | `bash_exec`, `posh_exec` — flowsh C1: exfiltration pair (secret read → tainted network egress); canonical |
| `command_privilege_escalation` | hard | `bash_exec`, `posh_exec` — flowsh C2: privilege-escalation effect (e.g. SUID install); canonical |
| `command_system_write` | hard | `bash_exec`, `posh_exec` — flowsh C3: direct write/metadata on a system path (POSIX `/etc`, `/usr`, `/boot`, `/bin`, `/sbin`; Windows `c:\windows`, `c:\program files`, `c:\program files (x86)`, component-boundary) or non-harmless raw device; canonical |
| `command_destructive_outside_roots` | hard | `bash_exec`, `posh_exec` — flowsh C4: irreversible destructive (KB class D/E) write outside the session roots; canonical |
| `command_download_cradle` | hard | `bash_exec`, `posh_exec` — flowsh C5: an established network→code-execution **cradle flow** (fetched content reaching a shell/interpreter: a pipe to `sh`, a sourced/process-substituted fetch, a command-substitution sink); keyed on the proven flow, not on a NetEgress/CodeExec co-occurrence; canonical |
| `command_unbounded_analysis` | hard | `bash_exec`, `posh_exec` — flowsh C6: analyzer ⊤/conservative without a cradle flow, or an irreversible write whose target could not be resolved (⊤ target, e.g. abbreviated PowerShell parameters); **non-canonical** — an analysis limitation a strict judge may clear |
| `command_external_content_ingest` | hard | `bash_exec`, `posh_exec` — flowsh C7: an established network→filesystem **ingest flow** (a download client — `curl -o`/`-O`, `wget -O`/default — wrote content it fetched to a file; a stdout fetch is not an ingest); **non-canonical** — a flow a strict judge may clear |
| `credential_access` | soft | `bash_exec`, `posh_exec` — flowsh C8: credential access without an exfil pair |
| `command_analysis_unavailable` | hard | `bash_exec`, `posh_exec` — the deterministic analysis could not be produced (flowsh analyzer/KB init failure): the Judge **fails closed**; canonical |
| `unresolvable_path_token` | hard | retained for contract stability — no built-in judge fires it since the static unresolvable-token stage was removed (flowsh C6 supersedes it) |
| `symlink_suspicious` | — | retained for contract stability — no built-in path fires it since the expansion-suspicion checks were removed from the symlink walk (decision 006); was host-set alongside `symlink_escape` |
| `outside_session_roots` | soft | `bash_exec`, `posh_exec` (flowsh C9: direct FS effect outside the session roots — writes/metadata on a non-system path, and reads even of a system path), file tools — a fully assessed path resolved outside the session roots |
| `unassessable_path` | hard | file tools — the target path could not be determined at all |
| `git_internal_path` | hard | file tools — the mutating target contains a ".git" path component at or below the workspace root (repository object database, refs, config, hooks; nested repos and worktrees included) |
| `unassessable_url` | hard | `web_fetch` — the target URL could not be determined at all |
| `ssrf_private_address` | hard | `web_fetch` — the URL resolves to a private/reserved address |
| `ssrf_protection_degraded` | hard | `web_fetch` — the SSRF CIDR check is unavailable (fail-closed) |

Allowed outcomes carry an empty code (nothing to classify), as do plain `PolicyUserConfirm` escalations no judge classified. `symlink_escape` exists in the taxonomy for hosts that run symlink detection over tool input; no SDK built-in emits it. `symlink_suspicious` is retained for contract stability but is no longer fired by the symlink walk (see [decision 006](../../decisions/006-symlink-walk-literal-paths-only.md)).

### Web search providers

`web_search` is optional: it is silently not registered when no search-provider config/API key is supplied. The provider abstraction supports Brave, DuckDuckGo, Exa, and Tavily.

### Ignore filtering (glob, ripgrep)

`glob` and `ripgrep` honour `.gitignore` and `.aiignore` rules through a single shared authority — the `IgnoreChecker` plumbed through tool context by the host. The checker (typically `ignore.Multi` over the workspace and any work-directory roots) is consulted per result so both tools agree on what is hidden. A `nil` checker (none attached) means **no** ignore filtering — the pre-ignore default — so wiring the checker is opt-in and never a regression.

- **glob** resolves each `doublestar.GlobWalk` entry to an absolute path and drops it when `checker.Ignored(absEntry, isDir)` is true. An ignored directory is skipped, and its file children are skipped too because the checker considers ancestor directories when deciding whether a path is ignored.
- **ripgrep** relies on `rg`'s native `.gitignore` handling and additionally **post-filters** every emitted match *and* context-line path through the same `IgnoreChecker` (via `isIgnoredPath`). This catches `.aiignore` rules (root *and* nested) and any resolver-only rule `rg` cannot see. The trade-off is that `rg` searches (and the tool then discards) `.aiignore`-matched files; for the typical secret-suppression use case these are few, so the cost is negligible. Nested `.aiignore` files are fully honoured (a prior limitation, now resolved).

### Windows console suppression

The tools that shell out to a child process — `posh_exec` (PowerShell), `ripgrep` (the `rg` binary), and the env-info runtime version probes — call `sysproc.HideConsole(cmd)` before starting it. When the host application is a Windows GUI-subsystem binary with no attached console, every child otherwise allocates a fresh console window that flashes or stays open on screen; `HideConsole` OR-edits `CREATE_NO_WINDOW` (0x08000000) into the child's `SysProcAttr.CreationFlags`, preserving any flags already set (e.g. `CREATE_NEW_PROCESS_GROUP` for `posh_exec`). The call is a no-op on non-Windows platforms, so the tools apply it unconditionally without platform branches. `bash_exec` is unaffected: it runs an already-attached interactive shell and is not invoked from this code path.

### Agent-infrastructure tools

Blackboard-backed tools (`read_step_output`, `list_step_outputs`, `read_final_result`, `read_attachment`, `store_fact`, `search_facts`) read and write shared blackboard state through the `agent.*Store` adapters the [Conductor](../orchestration/conductor.md) injects (see [../memory/blackboard.md](../memory/blackboard.md)). `read_attachment` reads a user-attached file's converted markdown from the `AttachmentStore` by ID (the IDs are listed in the user message). `update_checklist` validates Markdown checkboxes and emits to-do updates via a context-injected callback; `read_step_output` is likewise context-aware. `tool_result_read` validates cache coherence on every read (file mtime+size for file tools; TTL for MCP tools).

## Error Handling

- **Tool not found**: `ToolRegistry.Execute` returns an `IsError` `ToolResult` (does not panic).
- **Parse failure**: idiomatically returned via `tools.ParseInputError` — an `IsError` result, nil Go error (clean message, not an infrastructure failure).
- **Shell blocklist match / timeout**: `IsError: true` with a descriptive message; timeout messages include the configured value.
- **ripgrep exit 1**: not an error (no matches); exit ≥ 2 produces `IsError` with stderr.
- **Optional tool absence**: a missing dependency (e.g. no search API key) silently skips registration — no error at registration time.

## Invariants

- `finish` is always available (auto-appended to every run if absent).
- `batch` is intercepted at the executor level before reaching the registry; its own `Execute()` returns an error.
- Blackboard-backed tools read only error-free completed steps; outputs are listed in deterministic step-ID order.
- `read_skill_resource` resolves paths via `skills.SafeResolvePath` (path-traversal safe).
- Shell path extraction skips pure separator-run tokens (`//`, `C:\\`) as shell-language artifacts; tokens carrying real components (`/etc/passwd`, `//etc/passwd`, `C:\`, `C:\\Windows\win.ini`), anchored out-of-root writes, the raw-command-string blocklist, and the flowsh write criteria (keyed on effect targets, not extracted tokens) are unaffected.
- Every built-in judge escalation carries a `JudgeReasonCode` matching its escalation branch; allowed outcomes and plain `PolicyUserConfirm` gates carry none.
- Untrusted-output tools always set `Untrusted: true` and are wrapped when injection defense is enabled.
- `glob` and `ripgrep` share a single ignore authority (`IgnoreChecker` from context); a `nil` checker means no filtering (the opt-in, no-regression default).
- Tool-name constants live in `tools` (`names.go`) and mirror the registration names used by `tools/builtins`; `IsShellExecTool` classifies `bash_exec`/`posh_exec` as the highest-uncertainty, intentionally deprioritized tools in grouped tool lists.

## Extension Guide

To add a new built-in tool:

1. Define a struct embedding `*tools.BaseTool`.
2. In the constructor, set `ToolName`, `ToolDescription`, `Schema` (JSON Schema), and `Policy` (`PolicyAlwaysAllow` / `PolicyUserConfirm`).
3. Implement `Execute(ctx, input)` — use `ParseInputError` for JSON parse failures and `ErrorResult` for logical errors.
4. Set `Untrusted: true` if the tool returns external data (web, MCP, filesystem reads of external content).
5. Optionally implement `ToolJudger` for tool-specific safety escalation on `PolicyAlwaysAllow` tools; classify each denied branch with a `tools.JudgeReasonCode` (reuse a published code where the semantics match, add a new one otherwise).
6. For a read tool that returns a transformed/decoded view of a file (not raw bytes), implement `tools.ContentBackedReader` (`IsContentBacked`) so the executor caches the result in memory while keeping file coherence metadata; leave it unimplemented to keep the default file-backed behavior.
7. Register the tool in the `ToolRegistry` alongside the `finish` tool.

## Related Specs

- [README.md](README.md) — tool system overview and execution pipeline
- [mcp-gateway.md](mcp-gateway.md) — dynamic MCP tool discovery
- [../../contracts/tools.md](../../contracts/tools.md) — `Tool`/`BaseTool`/`ToolJudger`/`IgnoreChecker` interfaces and the host-to-registry data flow
- [../orchestration/executor.md](../orchestration/executor.md) — `batch` interception, two-stage truncation, `ToolResultCache`
- [../memory/blackboard.md](../memory/blackboard.md) — store adapters backing the blackboard tools
- [../embedding.md](../embedding.md) — `semantic_search` vector tool
