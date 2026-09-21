# ADR-007: The C10 Exec-Scope Criterion and Digest v4

## Status

Accepted

## Context

The deterministic flowsh shell-command analysis maps a command's effect IR onto
the fixed-priority criteria in `tools/shellanalysis.go`. Before this change the
criteria covered out-of-root *filesystem* scope (C4, destructive write outside
the roots; C9, FS effect outside the roots) but not out-of-root *code
execution*: a verification driver pointed at code outside the trusted session
roots — `npx vitest run /tmp/debug.test.ts`,
`./node_modules/.bin/vitest run /tmp/x.test.ts` — carried no dedicated reason.

flowsh v0.4.0 (`bind/runner.go`) resolves package runners and
`node_modules/.bin` wrappers onto the driven binary's knowledge-base signature.
Those calls therefore became bounded — they no longer fell to the C6 unbounded
reason — yet their out-of-root operand still had no criterion of its own. The
host application's ADR-059 ("Package-Runner Resolution and the Exec-Scope
Criterion") specifies the change; this ADR records the sp4rk half.

## Decision

Add the exec-scope criterion C10, `command_exec_outside_roots`, and bump the
digest schema to `sp4rk-shell-analysis/v4`.

- C10 fires when a directly performed `KindCodeExec`/`KindProcSpawn` effect has
  a path-shaped, resolvable target outside every session root — the exec
  sibling of C4/C9. The invoked binary's own path is not an operand and never
  triggers it, and no system-path exclusion applies.
- It is hard but NON-canonical (absent from `canonicalReasonCodes`), so the
  advisory and strict judges may positively clear a scratch script the session
  itself wrote to the host temp directory.
- It fires only on a BOUNDED report (`!top && !conservative &&` no unresolved
  irreversible write): an unbounded call remains the territory of C5/C6.
- Winner selection is by severity first, ties broken by the fixed priority
  order (`shellWinningCriterion`), so the hard C10 is never masked by the
  lower-numbered soft C9.
- Bump `github.com/v0lka/flowsh` to v0.4.0 for the runner/bin-path resolution
  the in-root driver forms rely on.

## Consequences

- Running code the session roots do not vouch for escalates to confirmation
  instead of passing silently or riding on the C6 unbounded reason.
- The workspace-scoped verification marker stays off for an out-of-root exec
  (its own containment condition), so the judge decides without positive
  evidence — fail-closed.
- The digest gains one published reason code; hosts keying deterministic policy
  off `JudgeReasonCode` see it, and the vocabulary guard pins it.
- The criteria set is now C1–C10; every pre-C10 criteria reference in code,
  tests, specs and docs is updated to match.

## Alternatives Considered

- **Fold the exec-scope question into C6.** Rejected: C6 is "the analyzer could
  not bound the command", an analysis limitation; a bounded driver with an
  out-of-root operand is a resolved scope finding, not an unboundedness one.
- **Make C10 canonical.** Rejected: executing an out-of-root file is a judgment
  shape (a scratch script in the host temp directory is routine), so the judges
  must be able to clear it.
- **Keep positional winner selection.** Rejected: it would let the soft C9
  dominate the hard C10, downgrading a hard exec-scope signal to a soft scope
  note.

## Related

- [ADR-006](006-symlink-walk-literal-paths-only.md) — the C4/C9 scope criteria
  this criterion joins.
