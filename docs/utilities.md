# Utilities

The SDK ships small, dependency-light packages with reusable algorithms: `pathutil` for filesystem-path operations, `safeio` for hardened non-blocking file I/O, `strutil` for string helpers, `sysproc` for child-process console-window suppression, and `ignore` for multi-root `.gitignore`/`.aiignore` resolution.

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

Both `parent` and `child` must be absolute paths — passing a relative path produces undefined results, so callers must resolve to absolute form first. The comparison is case-**sensitive**; `IsWithinPathFold(parent, child)` (same signature) is the case-insensitive variant that path-locality decisions need on case-insensitive filesystems (the default on macOS APFS and Windows NTFS), where `/Users/Foo` and `/Users/foo` denote the same location. `DetectCaseInsensitive(dir)` reports whether the filesystem at `dir` folds case, memoized per probed directory for the process lifetime. When `dir` does not exist (a brand-new session workspace), the probe climbs to the nearest existing ancestor directory and returns **that ancestor's** verdict: case-sensitivity is a property of the filesystem mount and is shared by the whole volume, so an ancestor answers correctly for the target — which means a non-existent `dir` on a case-insensitive mount still yields `true`, not `false`. The climb is **existence-only** — it never consults writability — so a `dir` that exists but is unwritable is **not** climbed past: the probe runs in `dir` itself, fails to create its temporary file, and the fail-safe `false` is returned and memoized. The fail-safe `false` is returned only when no usable existing ancestor is reachable at all (an unreachable root), or when the probe itself fails in that ancestor (an unwritable directory); a failed probe is memoized too, so a permanently read-only root is not re-probed forever.

It returns an error when `parent` is empty (containment cannot be determined — fail closed; callers must guard empty roots explicitly before calling) or when `parent` and `child` live on different volumes (e.g. different Windows drives — `filepath.VolumeName` is empty on Unix, so that check is a no-op there).

```go
ok, err := pathutil.IsWithinPath("/home/user/project", "/home/user/project/src/main.go")
// ok == true, err == nil

ok, err = pathutil.IsWithinPath("/home/user/project", "/home/user/../etc/passwd")
// ok == false — the resolved path escapes the parent

// Handles macOS /var → /private/var symlink:
ok, err = pathutil.IsWithinPath("/var/log", "/var/log/app.log")
// ok == true even though /var is a symlink to /private/var
```

The containment decision is an explicit prefix test, not a `filepath.Rel` computation:

- The resolved `parent` is terminated with a separator (unless it already ends in one — a root parent must never become `//` or `C:\\`, which would match no real child).
- `child` is within `parent` only when it is **identical** to the un-terminated resolved parent, or **starts with** the separator-terminated parent.
- Terminating with a separator is what stops a prefix-only sibling whose last component merely starts with the parent's (e.g. `/us` against parent `/usr`) from matching.

The child is resolved in OS component order from its **raw** spelling — it is deliberately *not* lexically cleaned first, because `filepath.Clean` would collapse a `symlink/..` sequence and erase the link before it was ever resolved. A path like `<parent>/<link-to-outside>/../x` therefore resolves outside the parent and is correctly rejected; cleaning it first would wrongly approve the escape. Only the trusted parent keeps its `filepath.Clean`.

### SplitPathComponents

```go
func SplitPathComponents(absPath string) []string
```

`SplitPathComponents` splits a cleaned absolute path into non-empty components, stripping the root separator and any volume prefix (e.g. `C:` on Windows, `\\server\share` for a UNC path — a no-op on Unix, where the volume is always empty) so a volume is never mistaken for a path component. `SplitPathComponents("C:\Users\file.txt")` → `["Users", "file.txt"]` on Windows.

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

1. Try `filepath.EvalSymlinks(candidate)` (starting with `candidate == path`). If it succeeds, return the resolved path.
2. If it fails with `os.ErrNotExist`, cut the last **raw** component off `candidate` and retry. The cut is byte-level and deliberately does not use `filepath.Dir` — that would `Clean` the path and erase an intermediate `symlink/..` spelling before the next resolution attempt.
3. When an existing ancestor is found, resolve it and reattach the raw remainder (`filepath.Join(resolved, path[len(candidate):])`).
4. When no separator is left to climb (a relative path at its top), return the path unchanged.
5. On permission or other errors, return the path unchanged.

The path is evaluated in **OS component order**: symlinks are resolved as they are encountered and `..` segments climb from the resolved location, so `<link>/..` refers to the link *target's* parent directory, not the link's lexical parent. Callers deciding containment must therefore pass the path **un-cleaned** (see `IsWithinPath` above).

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

## safeio

```go
import "github.com/v0lka/sp4rk/safeio"
```

The `safeio` package provides **hardened, non-blocking filesystem opens**. A path is opened only after — or while — confirming that it is a regular file, so a FIFO, socket, or device planted at an open path can never block the caller.

### Why it exists

A read-open (`O_RDONLY`) of a FIFO blocks until a writer appears, and a write-open (`O_WRONLY`) blocks until a reader appears. **Both blocks happen inside the `open(2)` syscall**, where no context check, deadline, or timeout can interrupt them. A single leftover named pipe beneath a directory an application reads, walks, or is pointed at is therefore enough to hang a user-facing RPC, a background pass, or a shutdown join indefinitely.

Route every open of a workspace- or user-controlled path through `safeio` so that class of hang is closed in one place rather than per call site. Inside the SDK it is used by `skills`, `agents`, `ignore`, `orchestration`, and `tools/builtins`.

Platform behaviour:

- **unix** — the open itself is `O_NONBLOCK`, and regularity is decided by `fstat` on the *already-open* descriptor. There is no `Stat`→`Open` window a racing local adversary could exploit (no TOCTOU).
- **Windows** — named pipes live in the `\\.\pipe\` namespace and cannot appear at a filesystem path, so a plain open plus a handle-based regularity check is sufficient.

### Open / OpenFile / OpenFileNoFollow

```go
func Open(path string) (*os.File, error)
func OpenFile(path string, flag int, perm fs.FileMode) (*os.File, error)
func OpenFileNoFollow(path string, flag int, perm fs.FileMode) (*os.File, error)
```

`Open` is the safe replacement for `os.Open`; `OpenFile` for `os.OpenFile`. `OpenFile` OR-s `O_NONBLOCK` into your flags, so a write-open of a reader-less FIFO returns `ENXIO` immediately (as an `*fs.PathError`) instead of waiting, and when a reader *is* present the post-open `fstat` refuses the FIFO with a `*NotRegularError`. Either way the caller never waits. On a regular file `O_NONBLOCK` has no effect, so the returned descriptor behaves like one from `os.OpenFile`.

`OpenFileNoFollow` adds `O_NOFOLLOW`: the kernel refuses to open a symbolic link at the **final** path component (`ELOOP`, wrapped in `*fs.PathError`) instead of resolving it, so a write-open can never be redirected through a planted symlink. Intermediate symlinks are still resolved by the kernel — use `MkdirAllReal` to build intermediate paths out of real directories only.

> **Windows parity:** `OpenFileNoFollow` is at parity with unix there — `os.OpenFile` carries `FILE_FLAG_OPEN_REPARSE_POINT` (forwarded to `CreateFile` by the syscall layer's own flag pass-through), so a reparse point at the final path component is opened **as the reparse point itself**, never its destination. The post-open handle classification refuses it with the same `*fs.PathError` wrapping `syscall.ELOOP` unix produces, so one `errors.Is(err, syscall.ELOOP)` covers both platforms. Intermediate symlinks are still resolved, as on unix — use `MkdirAllReal` to build intermediate paths out of real directories only. `WriteFileAtomic` remains the belt-and-braces option whose rename replaces a link itself instead of following it.

### ReadFile / ReadFileLimited

```go
func ReadFile(path string) ([]byte, error)
func ReadFileLimited(path string, limit int64) ([]byte, error)
func IsRegular(path string) (bool, error)

const DefaultMaxFileSize = int64(10 << 20) // 10 MiB
```

`ReadFile` is the safe replacement for `os.ReadFile`: it asserts regularity first (a FIFO is refused instead of blocking forever) and refuses a file larger than `DefaultMaxFileSize` instead of reading it into memory. `ReadFileLimited` applies a different cap — a file of exactly `limit` bytes is read in full, and a negative limit is treated as zero. `IsRegular` probes without blocking: a FIFO, socket, device, or directory reports `(false, nil)`, while a path that cannot be opened reports `(false, err)`.

### WriteFile / WriteFileAtomic

```go
func WriteFile(path string, data []byte, perm fs.FileMode) error
func WriteFileAtomic(path string, data []byte, perm fs.FileMode) error
```

`WriteFile` is the hardened replacement for `os.WriteFile`. The final path component is never followed on any platform: a symbolic link at `path` makes the open fail with `ELOOP` (wrapped in `*fs.PathError`) and the link's target is never written through. A missing file is created; **a missing parent directory is an error** — call `MkdirAllReal` first.

`WriteFileAtomic` writes a temporary file in the same directory, chmods it to `perm`, writes, `fsync`s, then `os.Rename`s it over the target, so readers never observe a partial file. The permission bits are applied with `Chmod`, which is **not** filtered by the process umask — pass the exact bits you want (e.g. `0600` for private state) and mask the mode yourself if umask filtering is intended.

### MkdirAllReal / MkdirAllRealWithin / CheckRealDirsBelow

```go
func MkdirAllReal(path string, perm fs.FileMode) error
func MkdirAllRealWithin(root, dir string, perm fs.FileMode) error
func CheckRealDirsBelow(from, path string) error
```

`MkdirAllReal` creates `path` with any missing parents, like `os.MkdirAll`, while guaranteeing that every directory it *creates* is a **real** directory:

- Pre-existing symbolic links anywhere in the path are **resolved once** to their real targets before anything is created. Ancestor links are ordinary operator/system structure (macOS makes `/var` a link to `/private/var`), so refusing them would break normal environments. Each resolved target must be a real directory: a link to a file (or other non-directory) at a component position returns a `*fs.PathError` wrapping `ErrNotDirectory`, and a dangling link returns a `*SymlinkError`.
- The missing tail is created component by component. A link swapped into the component *being created* makes its `Mkdir` fail with `EEXIST`, and the re-check refuses whatever appeared instead of following it. A link swapped into an *already-created* intermediate component between two `Mkdir` calls can still redirect the remainder — pair it with `CheckRealDirsBelow` where the residual window matters.

`MkdirAllRealWithin` adds a containment boundary: every pre-existing symbolic link anywhere in `dir`'s path must resolve to a target inside `root`, or the whole call fails with a `*SymlinkError` and nothing is created. `root` is resolved first (so an operator symlink on the anchor itself stays "operator intent"); links *below* the boundary are content-controlled and are refused when they escape it. Both paths must be absolute.

`CheckRealDirsBelow(from, path)` verifies that every component of `path` strictly below `from` is a real directory, creating and following nothing. `from` and its ancestors are not inspected.

### Errors

Every refusal is a typed error that also matches a sentinel via `errors.Is`:

| Sentinel | Typed error | Meaning |
| --- | --- | --- |
| `ErrNotRegular` | `*NotRegularError` | Path exists but is not a regular file (FIFO, socket, device, directory) |
| `ErrTooLarge` | `*TooLargeError` | Regular file exceeds the read size cap (`Path`, `Limit`) |
| `ErrSymlink` | `*SymlinkError` | A symlink where a real directory was required and could not be honored (dangling, loop, or swapped into a component being created) |
| `ErrNotDirectory` | — (wrapped in `*fs.PathError`) | An existing path component is not a directory |

A genuinely missing file passes the underlying open error through, so `errors.Is(err, fs.ErrNotExist)` still works.

```go
data, err := safeio.ReadFile(filepath.Join(workspace, "SKILL.md"))
switch {
case errors.Is(err, safeio.ErrNotRegular):
    return fmt.Errorf("refusing to read a non-regular file at %s", workspace)
case errors.Is(err, safeio.ErrTooLarge):
    return errors.New("document exceeds the read size cap")
case errors.Is(err, fs.ErrNotExist):
    return nil // absent is fine — nothing to load
case err != nil:
    return err
}
```

### Complete safeio example

```go
package main

import (
	"errors"
	"fmt"
	"io/fs"
	"log"
	"os"
	"path/filepath"

	"github.com/v0lka/sp4rk/safeio"
)

func main() {
	root := "/tmp/sp4rk-safeio-demo"

	// Build the directory tree out of real directories only.
	if err := safeio.MkdirAllReal(filepath.Join(root, "skills", "demo"), 0o755); err != nil {
		log.Fatalf("mkdirall: %v", err)
	}
	defer func() { _ = os.RemoveAll(root) }()

	target := filepath.Join(root, "skills", "demo", "SKILL.md")

	// Atomic write: temp file + fsync + rename, so readers never see a half
	// document and a symlink at target is never written through.
	if err := safeio.WriteFileAtomic(target, []byte("# demo\n"), 0o600); err != nil {
		log.Fatalf("write: %v", err)
	}

	// Non-blocking read with the default 10 MiB cap.
	data, err := safeio.ReadFile(target)
	if err != nil {
		log.Fatalf("read: %v", err)
	}
	fmt.Printf("read %d bytes\n", len(data))

	// A directory is not a regular file: refused, never opened.
	ok, err := safeio.IsRegular(filepath.Join(root, "skills"))
	fmt.Println("dir is regular:", ok, "err:", err)

	_, err = safeio.ReadFile(filepath.Join(root, "missing.md"))
	fmt.Println("missing is ErrNotExist:", errors.Is(err, fs.ErrNotExist))
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

`TruncateUTF8` returns `s` truncated to at most `maxChars` runes, respecting UTF-8 boundaries so the result is always valid UTF-8. When truncation occurs, the result ends with `"…"` (U+2026) to indicate content was cut: for `maxChars > 1` it is the first `maxChars-1` runes plus the ellipsis, and for `maxChars == 1` the single rune plus the ellipsis (two runes). If `s` is already `maxChars` runes or shorter, it is returned unchanged. If `maxChars` is non-positive, an empty string is returned.

This is the recommended replacement for byte-slice truncation expressions like `s[:N]` when the input may contain multi-byte UTF-8 characters that the downstream consumer (LLM API, logger, frontend) expects to be valid. A naive `s[:N]` cut can split a multi-byte rune in half, producing invalid UTF-8 that causes encoding errors downstream.

```go
// A 4-byte emoji followed by ASCII.
s := "🎉 Hello, world!" // 15 runes

// TruncateUTF8 truncates by rune count and appends "…" when needed.
truncated := strutil.TruncateUTF8(s, 6)
// "🎉 Hel…" — first 5 runes + "…" = 6 runes total.

// No-op when the string fits.
strutil.TruncateUTF8("short", 100) // → "short"

// Non-positive maxChars returns an empty string.
strutil.TruncateUTF8("anything", 0) // → ""
```

### InvisibleTrimSet

```go
const InvisibleTrimSet = " \t\n\r\x00\f\v\u200b\u200c\u200d\ufeff"
```

`InvisibleTrimSet` is the cutset of trailing invisible characters common in LLM output: spaces, tabs, newlines, carriage returns, null, form feed, vertical tab, zero-width space (U+200B), zero-width non-joiner (U+200C), zero-width joiner (U+200D), and the byte-order mark (U+FEFF). Apply it with `strings.TrimRight(s, strutil.InvisibleTrimSet)` when normalizing model output so the trim and `HasVisibleContent`'s verdict cannot disagree.

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

`TruncateUTF8AtLineBoundary` truncates `s` to at most `maxChars` runes, then rewinds to the last newline so the returned string ends on a complete line. Unlike `TruncateUTF8`, it does **not** append an ellipsis. If the truncated prefix contains no newline at all, the raw truncation to `maxChars` runes is returned unchanged; a newline at index 0 *is* honoured, so the result can be a single `"\n"`. A non-positive `maxChars` returns an empty string.

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
		"🎉🚀✨ emoji parade",       // 3- and 4-byte runes
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

The package has no engine imports; it is a near-leaf consumed by `tools`, `tools/builtins`, `tools/mcp`, and `embedding`. Callers apply it without their own platform build tags. `HideConsole` is stdlib-only; the Windows process-tree-containment helper (`AssignKillOnCloseJob`) requires `golang.org/x/sys/windows`, confined to a `//go:build windows` file, so the package is no longer strictly stdlib-only on Windows.

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

The model is **assign-after-Start**: call `cmd.Start()`, then `AssignKillOnCloseJob(cmd)` as the very next action, then `defer` the returned cleanup. `cmd.Process` must be non-nil — a `cmd` that has not been started yet (or is nil itself) returns the exported sentinel `sysproc.ErrProcessNotStarted`, which is the documented `errors.Is` match target (`errors.Is(err, sysproc.ErrProcessNotStarted)`). **The sentinel is Windows-only**: it is declared solely in the `//go:build windows` file `sysproc/job_windows.go`, so any `errors.Is(err, sysproc.ErrProcessNotStarted)` check must itself live in a Windows-tagged file — referencing the identifier from untagged code does not compile on Linux or macOS (`undefined: sysproc.ErrProcessNotStarted`). It is best-effort — on failure it leaks no handle, returns a no-op cleanup so the caller can defer it unconditionally, and returns the error (wrapping `ErrProcessNotStarted` for the not-started case) so the caller can fall back to `cmd.Cancel`. On other platforms it is a no-op — the OS has no Job Objects — returning a no-op cleanup and an always-`nil` error, so the sentinel is unreachable there. The function itself is all-platform, so callers apply and `defer` it unconditionally without branching on the OS, mirroring `HideConsole`.

This is the complement to `HideConsole`: together they hide the shell window AND guarantee no orphaned grandchildren or stray console windows persist after the host finishes, times out, or cancels.

### SetProcessGroup / KillTree

```go
func SetProcessGroup(cmd *exec.Cmd)
func KillTree(cmd *exec.Cmd) error
```

`SetProcessGroup` places `cmd` in its own process group so the whole tree it spawns can later be signalled as one. Without it a launcher (`npx`, `pnpm dlx`, `bunx`, …) and the server it spawns are separate targets and a kill reaches only the launcher, orphaning the server. It must be called **before** `cmd.Start`/`cmd.Run` — `SysProcAttr` only takes effect at process creation. On Windows it is a no-op, so callers apply it unconditionally.

`KillTree` force-kills `cmd` and, on unix, every process in its process group: the last-resort reaper for a child that ignores every graceful shutdown request (e.g. a launcher-spawned MCP server, `npx → node`, that does not exit on stdin EOF). The signal targets the group made at spawn time by `SetProcessGroup` (the child is its group leader), so it reaches the launcher and every descendant that did not leave the group; if the group signal fails — e.g. the child already exited and its pid was reused — it falls back to killing just the process. A nil `cmd` or a nil `cmd.Process` returns nil without signalling. `KillTree` is implemented on **both** platforms and returns a single `error` — there is no cleanup closure to `defer`. On Windows it runs `taskkill /T /F /PID <pid>` under a ~10 second bound (the kill-on-close Job Object helper cannot be used there: job membership is inherited only at process creation, and the MCP stdio transport owns `cmd.Start`, leaving no point to attach a job before the launcher forks its server), falling back to a direct `cmd.Process.Kill()` when `taskkill` is unavailable or the tree is already gone. The Job-Object / `golang.org/x/sys/windows` / `//go:build windows` story belongs to [`AssignKillOnCloseJob`](#assignkillonclosejob) above.

## ignore

```go
import "github.com/v0lka/sp4rk/ignore"
```

The `ignore` package is a multi-root ignore resolver that loads `.gitignore` and `.aiignore` files (at the root and in every nested directory) for each root and answers whether an arbitrary path is ignored by the patterns of the root that contains it. It is a **pure algorithmic building block**: it performs no hidden-dotfile or binary-file filtering. Those universal guards are caller-side concerns layered on top.

It imports only `pathutil`, the engine's `safeio` leaf, and an external glob library (`doublestar`) — no other engine package — so the host application wires it into tool context rather than the engine importing it. Ignore files are read through `safeio.Open`, so a FIFO or socket planted at a `.gitignore`/`.aiignore` path is refused instead of blocking the load walk.

### NewResolver and Multi

```go
func NewResolver(root string) (*Resolver, error)
func NewResolverContext(ctx context.Context, root string) (*Resolver, error)
func NewResolverWithOptions(root string, opts Options) (*Resolver, error)
func NewResolverWithOptionsContext(ctx context.Context, root string, opts Options) (*Resolver, error)
func NewMulti(logger *slog.Logger, roots ...string) (*Multi, error)
func NewMultiFromResolvers(logger *slog.Logger, resolvers ...*Resolver) *Multi
```

`NewResolver` walks `root` once, collecting every `.gitignore` and `.aiignore` file (root plus nested directories) and compiling their patterns into globs anchored relative to the root. `root` may be absolute or relative; it is canonicalized to an absolute, symlink-resolved form (via `pathutil.ResolveExistingPrefix`) so queries work regardless of the path form callers supply. `NewResolver` is `NewResolverContext` over `context.Background()`: the walk cannot be abandoned. `NewResolverContext` checks `ctx` on every directory entry and, once it is done, abandons the rest of the tree and returns a nil `Resolver` with an error wrapping ctx's error (detect via `errors.Is(err, context.Canceled)`); a cancelled build never publishes a half-collected resolver. A nil `ctx` means `context.Background()`.

`NewMulti` builds a `Resolver` per root and answers queries by delegating to whichever root contains the path. It requires at least one root and fails all-or-nothing when any root's resolver cannot be built; `logger` drives debug-level diagnostics (pass nil to suppress). `NewMultiFromResolvers` wraps pre-built resolvers instead — no re-walk, nil entries filtered, `nil` returned when nothing non-nil is supplied — for callers that want per-root error handling (skip an unreadable work directory rather than fail the whole workspace). Both `Resolver` and `Multi` satisfy [`ignore.IgnoreChecker`](#ignored) and, structurally, the identically-shaped `tools.IgnoreChecker` (`Ignored(absPath string, isDir bool) bool`): `ignore` declares its own exported `IgnoreChecker`, while `tools` declares a separate copy of the same shape and never imports `ignore` — so the host, not the engine, wires an `ignore` value into tool context.

### Options: honoured-file subset and extra patterns

```go
type Options struct {
    IgnoreFiles   []string // nil = both .gitignore and .aiignore; non-nil = the exact honoured set
    ExtraPatterns []string // gitignore-style lines compiled on top of the honoured files
}
```

The zero `Options` is exactly `NewResolver`: both ignore files honoured, no extra patterns.

- **`IgnoreFiles`** restricts which ignore file names the load walk honours. `nil` (the zero value) means both `.gitignore` and `.aiignore`. A non-nil slice is the exact honoured set: every entry must be `".gitignore"` or `".aiignore"` — anything else fails the build with an error (`ignore: IgnoreFiles entry %q must be ".gitignore" or ".aiignore"`), wrapped per root as `ignore: options for root %q: …`, so a typo cannot silently disable an ignore file. An **empty non-nil** slice honours no ignore file at all, leaving only `ExtraPatterns` in effect.
- **`ExtraPatterns`** are gitignore-style lines compiled on top of whatever the honoured ignore files contribute, scoped as if written in a root ignore file: a leading (or internal) slash anchors the pattern to the root, a bare name matches at any depth, and a trailing slash marks it directory-only. Blank lines, comments, and negation lines are skipped exactly as in an ignore file. Compiled patterns carry the origin `"config"`, so — like `.gitignore`-sourced rules — they are excluded from the `.aiignore`-scoped query (`IgnoredByAIIgnore`).

Options are validated before any filesystem access, so an invalid `Options` fails fast without touching the root.

### IgnoredByAIIgnore / Match / Root

```go
func (r *Resolver) IgnoredByAIIgnore(absPath string, isDir bool) bool
func (m *Multi) IgnoredByAIIgnore(absPath string, isDir bool) bool
func (r *Resolver) Match(relPath string, isDir bool) bool
func (r *Resolver) Root() string
func (m *Multi) Resolvers() []*Resolver
func (m *Multi) RootFor(absPath string) *Resolver
```

`IgnoredByAIIgnore` reports whether a path is ignored by `.aiignore`-sourced rules only — for callers that already obtain a `.gitignore` verdict from git itself (which honours negation and the global gitignore that the resolver does not), so layering only `.aiignore` rules on top preserves git's un-ignore semantics. Limitation: the load walk prunes any directory ignored by *either* ignore file, so a nested `.aiignore` beneath a `.gitignore`-only-ignored directory is never collected and cannot contribute rules (a false negative that is safe for the OR-merge consumer). `Match` answers the same question for a root-relative path; `Root` returns the cleaned absolute root; `Resolvers` returns a copy of the per-root resolvers; `RootFor` returns the `Resolver` whose root contains the path (nil when none does).

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
type IgnoreChecker interface {
    Ignored(absPath string, isDir bool) bool
}

func (r *Resolver) Ignored(absPath string, isDir bool) bool
func (m *Multi) Ignored(absPath string, isDir bool) bool
```

`IgnoreChecker` is the abstraction tools depend on: given an absolute path and whether it is a directory, report whether it is ignored. Both `Resolver` and `Multi` satisfy it.

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
