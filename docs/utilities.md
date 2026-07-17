# Utilities

The SDK ships small, dependency-light packages with reusable algorithms: `pathutil` for filesystem-path operations, `strutil` for string helpers, and `ignore` for multi-root `.gitignore`/`.aiignore` resolution.

## pathutil

```go
import "github.com/v0lka/sp4rk/pathutil"
```

The `pathutil` package provides reusable filesystem-path algorithms with **zero project-specific knowledge**. It contains pure algorithmic primitives that are safe to use from any layer. Project-specific path construction and directory layout live elsewhere — `pathutil` knows nothing about it.

### IsWithinPath

```go
func IsWithinPath(parent, child string) (bool, error)
```

`IsWithinPath` returns `true` if `child` is equal to or a descendant of `parent`. Both paths are symlink-resolved through their longest existing prefix (`ResolveExistingPrefix`) before comparison, so it correctly handles OS-level symlinks like macOS `/var → /private/var` even when the paths do not exist on disk yet.

It returns an error when `parent` is empty (containment cannot be determined — fail closed; callers must guard empty roots explicitly before calling) or when `filepath.Rel` fails (e.g. the paths are on different volumes).

```go
ok, err := pathutil.IsWithinPath("/home/user/project", "/home/user/project/src/main.go")
// ok == true, err == nil

ok, err = pathutil.IsWithinPath("/home/user/project", "/home/user/../etc/passwd")
// ok == false — the resolved path escapes the parent

// Handles macOS /var → /private/var symlink:
ok, err = pathutil.IsWithinPath("/var/log", "/var/log/app.log")
// ok == true even though /var is a symlink to /private/var
```

The containment check works by computing the relative path from `parent` to `child`:

- `rel == "."` means the paths are the same → within.
- `rel` starting with `".."` means `child` escapes above `parent` → not within.
- otherwise → within.

### SplitPathComponents

```go
func SplitPathComponents(absPath string) []string
```

`SplitPathComponents` splits a cleaned absolute path into non-empty components, stripping the root separator.

```go
pathutil.SplitPathComponents("/home/user/file.txt")
// → ["home", "user", "file.txt"]

pathutil.SplitPathComponents("/")
// → []
```

Empty components (e.g. from consecutive separators) are filtered out.

### ResolveExistingPrefix

```go
func ResolveExistingPrefix(path string) string
```

`ResolveExistingPrefix` resolves symlinks on the **longest existing prefix** of `path`, then joins the non-existent suffix back. This is used when validating paths for files or directories that may not exist yet (e.g. write or mkdir tool targets) — `filepath.EvalSymlinks` fails on non-existent paths, so this function walks up the path until it finds a component that exists, resolves it, and reattaches the remainder.

```go
// If "/ws/link" is a symlink to "/real/path" but "/ws/link/newfile.txt"
// does not exist yet:
resolved := pathutil.ResolveExistingPrefix("/ws/link/newfile.txt")
// → "/real/path/newfile.txt"
```

The algorithm:

1. Try `filepath.EvalSymlinks(path)`. If it succeeds, return the resolved path.
2. If it fails with "not exist", move to the parent directory and retry.
3. When an existing ancestor is found, resolve it and reattach the relative suffix.
4. If the root is reached without finding anything, return the path unchanged.
5. On permission or other errors, return the path unchanged.

### Complete pathutil example

```go
package main

import (
	"fmt"

	"github.com/v0lka/sp4rk/pathutil"
)

func main() {
	root := "/home/user/project"

	// Containment check — used to validate that a target path stays
	// within an allowed workspace.
	targets := []string{
		"/home/user/project/src/main.go",
		"/home/user/project/../../etc/passwd",
	}
	for _, t := range targets {
		ok, err := pathutil.IsWithinPath(root, t)
		fmt.Printf("%-45s within=%v err=%v\n", t, ok, err)
	}

	// Split a path into components.
	comps := pathutil.SplitPathComponents("/home/user/project/src/main.go")
	fmt.Println("\ncomponents:", comps)

	// Resolve symlinks on the longest existing prefix (safe for paths
	// that do not exist yet).
	resolved := pathutil.ResolveExistingPrefix("/home/user/project/new/dir/file.txt")
	fmt.Println("resolved:", resolved)
}
```

## strutil

```go
import "github.com/v0lka/sp4rk/strutil"
```

The `strutil` package provides shared string helpers.

### TruncateUTF8

```go
func TruncateUTF8(s string, maxBytes int) string
```

`TruncateUTF8` returns `s` truncated to at most `maxBytes` bytes, respecting UTF-8 rune boundaries so the result is always valid UTF-8. If `s` is already shorter than `maxBytes` (or `maxBytes` is non-positive), `s` is returned unchanged.

This is the recommended replacement for byte-slice truncation expressions like `s[:N]` when the input may contain multi-byte UTF-8 characters that the downstream consumer (LLM API, logger, frontend) expects to be valid. A naive `s[:N]` cut can split a multi-byte rune in half, producing invalid UTF-8 that causes encoding errors downstream.

```go
// A 4-byte emoji followed by ASCII.
s := "🎉 Hello, world!"

// TruncateUTF8 respects rune boundaries — the result is always valid UTF-8.
truncated := strutil.TruncateUTF8(s, 6)
// "🎉 H" — 4 bytes (emoji) + 1 byte (space) + 1 byte ('H') = 6 bytes.

// When maxBytes falls inside a multi-byte rune, the cut backs up.
// E.g. TruncateUTF8(s, 3) returns "" — a naive s[:3] would split the emoji.

// No-op when the string is already short enough.
strutil.TruncateUTF8("short", 100) // → "short"

// No-op when maxBytes is non-positive.
strutil.TruncateUTF8("anything", 0) // → "anything"
```

The implementation decrements `maxBytes` until `utf8.RuneStart(s[maxBytes])` is true, ensuring the cut never lands in the middle of a multi-byte rune.

### TruncateUTF8AtLineBoundary

```go
func TruncateUTF8AtLineBoundary(s string, maxBytes int) string
```

`TruncateUTF8AtLineBoundary` truncates `s` to at most `maxBytes` bytes using `TruncateUTF8`, then snaps the result back to the last newline so the returned string ends on a complete line. If the truncated string contains no newline, or the only newline is at index 0, the UTF-8-safe truncated value is returned unchanged.

Use this when downstream consumers expect line-oriented output (e.g. log lines, plan exploration summaries) and a cut mid-line would be confusing.

```go
// Truncate to ~4000 bytes, ending on a line boundary.
summary := strutil.TruncateUTF8AtLineBoundary(longText, 4000)
```

### Complete strutil example

```go
package main

import (
	"fmt"

	"github.com/v0lka/sp4rk/strutil"
)

func main() {
	texts := []string{
		"Hello, world!",            // ASCII only
		"café résumé naïve",        // Latin-1 supplement (2-byte runes)
		"🎉🚀✨ emoji parade",       // 4-byte runes
	}

	for _, t := range texts {
		fmt.Printf("original:  %q (%d bytes)\n", t, len(t))
		for _, n := range []int{4, 8, 12} {
			fmt.Printf("  truncate(%d): %q (%d bytes, valid UTF-8)\n",
				n, strutil.TruncateUTF8(t, n), len(strutil.TruncateUTF8(t, n)))
		}
		fmt.Println()
	}
}
```

## ignore

```go
import "github.com/v0lka/sp4rk/ignore"
```

The `ignore` package is a multi-root ignore resolver that loads `.gitignore` and `.aiignore` files (at the root and in every nested directory) for each root and answers whether an arbitrary path is ignored by the patterns of the root that contains it. It is a **pure algorithmic building block**: it performs no hidden-dotfile or binary-file filtering. Those universal guards are caller-side concerns layered on top.

It imports only `pathutil` and an external glob library (`doublestar`) — no engine packages — so the host application wires it into tool context rather than the engine importing it.

### NewResolver and Multi

```go
func NewResolver(root string) (*Resolver, error)
func NewMulti(roots ...string) (*Multi, error)
```

`NewResolver` walks `root` once, collecting every `.gitignore` and `.aiignore` file (root plus nested directories) and compiling their patterns into globs anchored relative to the root. `root` may be absolute or relative; it is canonicalized to an absolute, symlink-resolved form (via `pathutil.ResolveExistingPrefix`) so queries work regardless of the path form callers supply.

`NewMulti` builds a `Resolver` per root and answers queries by delegating to whichever root contains the path. Both `Resolver` and `Multi` satisfy the `IgnoreChecker` interface (`Ignored(absPath string, isDir bool) bool`), which the `tools` package defines itself — `tools` never imports `ignore`.

```go
// Single-root resolver.
r, err := ignore.NewResolver("/home/user/project")

// Multi-root resolver (workspace + a separate work directory).
m, err := ignore.NewMulti(workspace, workDir)

// Both are usable as a tools.IgnoreChecker:
var checker tools.IgnoreChecker = m
ctx := tools.WithIgnoreChecker(ctx, checker)
```

### Ignored

```go
func (r *Resolver) Ignored(absPath string, isDir bool) bool
func (m *Multi) Ignored(absPath string, isDir bool) bool
```

`Ignored` reports whether an absolute path is ignored. `absPath` is canonicalized via longest-existing-prefix symlink resolution and then converted to a root-relative path, making the resolver robust to either path form callers supply (raw `/tmp/...` or resolved `/private/tmp/...`). Paths outside all known roots are never ignored (matching the `IgnoreChecker` contract).

Directory semantics follow standard gitignore: if any ancestor directory is ignored, the path is ignored too.

### Pattern semantics and limitations

- A leading slash anchors a pattern to its file's directory; a bare name with no slash matches at any depth beneath that directory.
- A trailing slash marks a rule as directory-only.
- **Negation patterns** (lines beginning with `!`) are **unsupported** — they are silently skipped.
- The `.git` directory is always pruned during the walk (never meaningful source, and can be enormous). Any directory that is itself ignored by the patterns collected so far is pruned too.

### Complete ignore example

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/v0lka/sp4rk/ignore"
	"github.com/v0lka/sp4rk/tools"
)

func main() {
	// Build a resolver over the workspace; .gitignore + .aiignore at the
	// root and in nested directories are honoured.
	checker, err := ignore.NewResolver(".")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("root:", checker.Root())

	// Attach it to tool context so glob/ripgrep honour the rules.
	ctx := tools.WithIgnoreChecker(context.Background(), checker)

	_ = ctx // pass to the agent/executor
}
```
