# Utilities

The SDK ships small, dependency-light packages with reusable algorithms: `pathutil` for filesystem-path operations, `strutil` for string helpers, `sysproc` for child-process console-window suppression, and `ignore` for multi-root `.gitignore`/`.aiignore` resolution.

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
func TruncateUTF8(s string, maxChars int) string
```

`TruncateUTF8` returns `s` truncated to at most `maxChars` runes, respecting UTF-8 boundaries so the result is always valid UTF-8. When truncation occurs, the result ends with `"…"` (U+2026) to indicate content was cut. If `s` is already `maxChars` runes or shorter, it is returned unchanged. If `maxChars` is non-positive, an empty string is returned.

This is the recommended replacement for byte-slice truncation expressions like `s[:N]` when the input may contain multi-byte UTF-8 characters that the downstream consumer (LLM API, logger, frontend) expects to be valid. A naive `s[:N]` cut can split a multi-byte rune in half, producing invalid UTF-8 that causes encoding errors downstream.

```go
// A 4-byte emoji followed by ASCII.
s := "🎉 Hello, world!" // 13 runes

// TruncateUTF8 truncates by rune count and appends "…" when needed.
truncated := strutil.TruncateUTF8(s, 6)
// "🎉 He…" — first 5 runes + "…" = 6 runes total.

// No-op when the string fits.
strutil.TruncateUTF8("short", 100) // → "short"

// Non-positive maxChars returns an empty string.
strutil.TruncateUTF8("anything", 0) // → ""
```

### HasVisibleContent

```go
func HasVisibleContent(s string) bool
```

`HasVisibleContent` applies the same trailing trim set as LLM response normalization (`InvisibleTrimSet`) and reports whether anything remains. Empty strings and strings made entirely of spaces, control whitespace, nulls, zero-width characters, or a byte-order mark return `false`. Use it before accepting model output so the visibility check and the later `strings.TrimRight(..., strutil.InvisibleTrimSet)` normalization cannot disagree.

```go
strutil.HasVisibleContent("answer\u200b") // true
strutil.HasVisibleContent(" \t\u200b")  // false
```

### TruncateUTF8AtLineBoundary

```go
func TruncateUTF8AtLineBoundary(s string, maxChars int) string
```

`TruncateUTF8AtLineBoundary` truncates `s` to at most `maxChars` runes, then rewinds to the last newline so the returned string ends on a complete line. Unlike `TruncateUTF8`, it does **not** append an ellipsis. If the truncated prefix contains no newline, or the only newline is at index 0, the raw truncation to `maxChars` runes is returned unchanged.

Use this when downstream consumers expect line-oriented output (e.g. log lines, plan exploration summaries) and a cut mid-line would be confusing.

```go
// Truncate to ~4000 runes, ending on a line boundary.
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

## sysproc

```go
import "github.com/v0lka/sp4rk/sysproc"
```

The `sysproc` package configures process-creation attributes for child processes spawned by host applications and the SDK's own helper processes. It exists because a Windows host built as a GUI-subsystem binary has no attached console, so any child started via `os/exec` allocates a fresh console window by default — a terminal that flashes or stays open on screen. `HideConsole` suppresses that window, and `AssignKillOnCloseJob` guarantees the whole descendant tree is cleaned up.

The package has no engine imports; it is a near-leaf consumed by `tools`, `tools/builtins`, and `tools/mcp`. Callers apply it without their own platform build tags. `HideConsole` is stdlib-only; the Windows process-tree-containment helper (`AssignKillOnCloseJob`) requires `golang.org/x/sys/windows`, confined to a `//go:build windows` file, so the package is no longer strictly stdlib-only on Windows.

### HideConsole

```go
func HideConsole(cmd *exec.Cmd)
```

`HideConsole` configures `cmd` so the child process does not allocate a visible console window. On Windows it OR-edits the `CREATE_NO_WINDOW` flag (0x08000000) into `cmd.SysProcAttr.CreationFlags`, preserving any flags the caller already set (e.g. `CREATE_NEW_PROCESS_GROUP`). On other platforms it is a no-op, so callers apply it unconditionally without branching on the OS. It must be called before `cmd.Start`/`cmd.Run`; mutating `SysProcAttr` after the process has started has no effect.

```go
cmd := exec.CommandContext(ctx, "rg", args...)
sysproc.HideConsole(cmd)
out, err := cmd.Output()
```

### When to call it

Apply `HideConsole` to every `exec.Cmd` that is **not** an interactive pseudo-terminal (PTY/ConPTY) session:

- `posh_exec` (PowerShell), `ripgrep` (the `rg` binary), and the env-info runtime version probes all call it before starting the child.
- stdio MCP servers are spawned through a custom command factory that always calls it.

Do **not** call it for an interactive PTY/ConPTY session, which routes the child's console through the pseudo terminal and must keep its default creation behaviour.

### Why CREATE_NO_WINDOW, not HideWindow

Go's `syscall.SysProcAttr.HideWindow` field hides a window that has already been created via `ShowWindow(SW_HIDE)` and still allows a brief flash. `CREATE_NO_WINDOW` suppresses console allocation entirely, so no window ever appears — including no flash.

### AssignKillOnCloseJob

```go
func AssignKillOnCloseJob(cmd *exec.Cmd) (cleanup func(), err error)
```

`AssignKillOnCloseJob` places an already-started process — and, by inheritance, its entire descendant tree — into a Windows Job Object configured with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`. The returned cleanup closure closes the job handle (typically via `defer`, after `cmd.Wait`), terminating every process still in the job: the command on timeout/cancellation, plus any surviving children and grandchildren on normal completion. It solves the orphan problem where a command's long-lived children (e.g. a browser and its console window launched by `posh_exec`'s PowerShell) outlive the command's own kill scope.

The model is **assign-after-Start**: call `cmd.Start()`, then `AssignKillOnCloseJob(cmd)` as the very next action, then `defer` the returned cleanup. `cmd.Process` must be non-nil. It is best-effort — on failure it leaks no handle, returns a no-op cleanup so the caller can defer it unconditionally, and returns the error so the caller can fall back to `cmd.Cancel`.

This is the complement to `HideConsole`: together they hide the shell window AND guarantee no orphaned grandchildren or stray console windows persist after the host finishes, times out, or cancels. On Windows it requires `golang.org/x/sys/windows` (Windows 8+ for nested jobs) and lives behind a `//go:build windows` tag; on other platforms it is a no-op that returns a no-op cleanup and a nil error, so callers can defer it unconditionally without their own build tags — mirroring `HideConsole`.

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
// Pass a logger for debug-level diagnostics; nil suppresses.
m, err := ignore.NewMulti(logger, workspace, workDir)

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
