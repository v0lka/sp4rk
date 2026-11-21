# Child-Process Containment

## Role

Cross-platform child-process containment helpers: console-window suppression on Windows, unix process-group placement and whole-tree killing, and a Windows kill-on-close Job Object that cleans up an entire descendant tree. Every helper is callable on every platform — no-op where the mechanism does not exist — so callers apply them unconditionally without build tags.

## Key Files

- `github.com/v0lka/sp4rk/sysproc/doc.go` - package contract: the console-flash problem, the orphaned-grandchild problem, and the assign-after-Start model
- `github.com/v0lka/sp4rk/sysproc/sysproc_unix.go` - non-Windows builds: `HideConsole` no-op, `SetProcessGroup` (`Setpgid`), `KillTree` (process-group `SIGKILL`), `AssignKillOnCloseJob` no-op
- `github.com/v0lka/sp4rk/sysproc/sysproc_windows.go` - Windows builds: `HideConsole` (`CREATE_NO_WINDOW`), `SetProcessGroup` no-op, `KillTree` (bounded `taskkill /T /F`)
- `github.com/v0lka/sp4rk/sysproc/job_windows.go` - `AssignKillOnCloseJob` (`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` via `golang.org/x/sys/windows`) and `ErrProcessNotStarted`

## Behavior

**Console suppression (`HideConsole`).** On Windows, OR-s `CREATE_NO_WINDOW` (0x08000000) into `SysProcAttr.CreationFlags`, preserving any flags already set (e.g. `CREATE_NEW_PROCESS_GROUP`). `CREATE_NO_WINDOW` suppresses console allocation entirely — the stdlib's `HideWindow` field only hides an already-created window and still flashes. The call must precede `Start`/`Run`; it is a no-op elsewhere. Interactive PTY/ConPTY sessions route the child's console through the pseudo terminal and must NOT apply it.

**Process-group placement (`SetProcessGroup`).** On unix, sets `SysProcAttr.Setpgid` so the child leads its own process group, making the whole tree it spawns signalable as one (the target of `KillTree`); without it, a launcher (npx, pnpm dlx, bunx) and the server it spawns are separate kill targets. Must be called before `Start`. No-op on Windows, where containment instead uses `KillTree`'s tree kill or the Job Object.

**Tree kill (`KillTree`).** The last-resort reaper for a child that ignores graceful shutdown (e.g. an npx-launched MCP server that ignores stdin EOF — a bare `Process.Kill` orphans the server holding its pipes). On unix it signals `-pid` with `SIGKILL`, reaching the group leader and every descendant that stayed in the group; a failed group signal falls back to killing just the process. On Windows it runs `taskkill /T /F /PID` with its own `HideConsole`, bounded by a 10-second context; a taskkill failure falls back to the direct process kill. A nil cmd or nil Process is a nil-error no-op on both platforms.

**Job-Object containment (`AssignKillOnCloseJob`).** Windows only; assign-after-`Start` contract: call it as the very next action after a successful `Start`, then defer the returned cleanup. It creates a Job Object with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` (deliberately without `JOB_OBJECT_LIMIT_BREAKAWAY_OK`, so every child a member spawns inherits membership), assigns the running process via `os.Process.WithHandle`, and returns an idempotent cleanup that closes the job handle exactly once (`sync.Once`) — closing the last handle terminates every process still in the job, so deferred cleanup after `Wait` also reaps surviving grandchildren (a browser, its console window). Nested-job support (Windows 8+) lets the assignment succeed even when the host is itself in a job. A brief race window between `Start` and assignment exists in which a fast-forking grandchild could escape membership; the motivating `powershell.exe -Command` use case parses its command line before executing, so grandchildren spawn only after assignment.

## Error Handling

- `AssignKillOnCloseJob` returns `ErrProcessNotStarted` (match via `errors.Is`) when `cmd.Process` is nil, and wraps job-creation or assignment failures otherwise. On any failure it closes the freshly created job handle — no kernel handle leaks — and returns a no-op cleanup so callers can `defer` it unconditionally alongside the error, falling back to their own termination strategy.
- `KillTree` reports the taskkill error only when the direct-kill fallback also fails; a group-signal failure on unix degrades to the single-process kill.
- Every non-Windows build of every function is a no-op with a nil error (`AssignKillOnCloseJob` returns `func(){}, nil`).

## Invariants

- Every function is callable on every platform without build tags: the no-op builds exist precisely so callers apply containment unconditionally.
- `HideConsole` and `AssignKillOnCloseJob` never mutate an already-started process's creation attributes — they configure before `Start` or attach after it, never both.
- The Job Object's `BREAKAWAY_OK` limit is intentionally absent: member processes' descendants inherit job membership.
- The returned cleanup is idempotent and concurrency-safe: the job handle closes exactly once, and error paths never leak it.
- `HideConsole` itself is stdlib-only; the Job-Object helper's `golang.org/x/sys/windows` dependency stays confined to `//go:build windows` files, keeping the non-Windows build free of it.

## Related Specs

- [Architecture Layers](../architecture/layers.md) - sysproc is a leaf utility package (stdlib plus `golang.org/x/sys` on Windows); the import rule keeps it free of upward dependencies
