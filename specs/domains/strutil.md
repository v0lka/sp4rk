# String Utilities

## Purpose

Leaf string helpers shared across the engine: rune-safe truncation for display-facing text (step titles, tool-call previews, judge reasoning) and an invisible-character trim set for deciding whether LLM output carries visible content. The package is imported by `agent`, `llm`, `memory`, `orchestration`, `planner`, and `tools`, and imports nothing engine-internal.

## Key Files

- `github.com/v0lka/sp4rk/strutil/truncate.go` - the entire exported surface: `InvisibleTrimSet`, `HasVisibleContent`, `TruncateUTF8`, `TruncateUTF8AtLineBoundary`

## Core Types

```go
// InvisibleTrimSet is the cutset of trailing invisible characters common in
// LLM output: space, tab, newline, carriage return, NUL, form feed, vertical
// tab, zero-width space (U+200B), zero-width non-joiner (U+200C),
// zero-width joiner (U+200D), and byte-order mark (U+FEFF).
const InvisibleTrimSet = " \t\n\r\x00\f\v\u200b\u200c\u200d\ufeff"

// HasVisibleContent reports whether s contains any non-trailing-invisible content.
func HasVisibleContent(s string) bool

// TruncateUTF8 truncates s to at most maxChars runes, appending "…" if truncated.
func TruncateUTF8(s string, maxChars int) string

// TruncateUTF8AtLineBoundary truncates s to at most maxChars runes, then rewinds
// to the last newline to avoid splitting mid-line.
func TruncateUTF8AtLineBoundary(s string, maxChars int) string
```

## Flow

`TruncateUTF8` truncation decision:

1. `maxChars <= 0` returns the empty string.
2. A string of at most `maxChars` runes returns unchanged.
3. Otherwise the result keeps `maxChars-1` leading runes plus `…` (a total of `maxChars` runes); at `maxChars == 1` it keeps the first rune plus `…`.

`TruncateUTF8AtLineBoundary` truncation decision:

1. `maxChars <= 0` returns the empty string.
2. A string of at most `maxChars` runes returns unchanged.
3. Otherwise the cut position rewinds from rune index `maxChars-1` to the last `'\n'` (inclusive — the newline is kept), yielding a line-aligned prefix.
4. With no newline within the window, the result is exactly the first `maxChars` runes, with no ellipsis appended.

`HasVisibleContent` applies `strings.TrimRight(s, InvisibleTrimSet)` and reports whether anything remains.

## Invariants

- Truncation is rune-counted, so a multi-byte UTF-8 sequence is always kept or dropped whole — the result is always valid UTF-8.
- `TruncateUTF8AtLineBoundary` always returns a prefix of the input; every truncated result that contains a newline within the window ends at a `'\n'` boundary.
- A `maxChars` of zero or less always yields the empty string.
- `HasVisibleContent` agrees with `strings.TrimRight(s, InvisibleTrimSet) != ""`, so a string it rejects trims to empty under the same cutset.

## Configuration

`maxChars` is a rune count (not a byte count) with no internal default; callers pass display budgets. `InvisibleTrimSet` is a fixed constant — callers compose it into their own `Trim`/`TrimRight`/`TrimLeft` calls.

## Extension Points

New string helpers join this package when they are engine-generic and dependency-free: the package imports only the standard library, and consumers rely on it staying free of engine-internal imports. Helpers that need path logic belong in `pathutil` instead.

## Related Specs

- [Architecture Layers](../architecture/layers.md) - strutil is a supporting leaf package consumed across all layers with no upward dependencies
