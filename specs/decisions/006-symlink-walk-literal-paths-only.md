# ADR-006: The Symlink Walk Is a Literal-Path Extractor

## Status

Accepted

## Context

`DetectSymlinksInToolInput` carried two responsibilities: symlink detection over extracted paths, and a `suspicious` flag raised for `bash_exec`/`posh_exec` commands containing variable expansions (`$var`, `$(cmd)`, backticks, `$env:...`, process substitution) that "may hide additional paths". Hosts escalated that flag as a hard `symlink_suspicious` reason through their confirmation funnels and strict LLM judges.

After the deterministic flowsh shell-command analysis landed (criteria C1–C8, `AnalyzeShellCommandForJudge`), the flag became pure duplication: constructs the static token walk cannot see through are assessed by the C6 unbounded criterion, and out-of-root scope by C4/C8 — from the effect IR, more precisely than token walking. The shell tools' built-in `Judge` had already dropped its static unresolvable-token and containment stages for exactly this reason; the symlink walk was the last consumer of the machinery.

That machinery — the validated command-substitution assessment (`shellEnvBindings.go`), the shell-path resolvers (`shellpaths.go`'s `ResolveShellPathTokens`/`PathsOutsideRoots`/`UnresolvablePathTokens`), and the PowerShell static-binding tokenizer — existed mostly to decide when an expansion was benign enough *not* to fire the flag: roughly 3300 lines plus their tests. Operationally, hosts reported the flag cluttering the LLM judge's work and inflating per-session token spend.

## Decision

The symlink walk is a **pure literal-path extractor**:

- `DetectSymlinksInToolInput` returns `(inside, outside)` only — the `suspicious` flag is gone. Shell commands contribute their literal paths (unquoted, single-quoted, double-quoted literals); dynamic parts contribute nothing and never escalate.
- `FormatSymlinkReasoning` formats traversals only.
- `ReasonCodeSymlinkSuspicious` stays in the published reason-code contract (wire stability; the `unresolvable_path_token` precedent) but nothing fires it.
- `shellEnvBindings.go`, the shell-path resolvers in `shellpaths.go`, and their tests are deleted. The three helpers other packages still use (`isASCIILetter`, `isPathComponentChar`, `isPureSeparatorRunToken`) moved to their consumers (`shellanalysis.go`, `judge.go`).

## Consequences

- Fewer false-positive escalations for ordinary shell idioms; host LLM judges stop spending tokens on expansion noise.
- A symlink reachable only *through* a variable (`X=$(cat link/secret); cat $X`) is no longer surfaced by the walk — accepted residual: the flowsh criteria assess the command holistically and never depended on symlink-walk coverage of dynamic constructs.
- The `tools` package shrinks by ~3300 lines and its test surface accordingly.
- Hosts that keyed policy off `symlink_suspicious` still parse the retained code; nothing fires it.

## Alternatives Considered

- **Keep the flag, widen the benign-exception machinery.** Rejected: every widening added code to avoid firing information flowsh already provides.
- **Downgrade the escalation to soft.** Rejected: soft reasons still consult strict judges (same token spend) while weakening posture; the redundancy argument removes the check, not its severity.
