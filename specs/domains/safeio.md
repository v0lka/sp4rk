# Safe File I/O

## Purpose

Hardened, non-blocking filesystem I/O primitives: every open of a fixed, user- or workspace-controlled path is confirmed to be a regular file before — or while — it is used, so a FIFO, socket, or device planted at a path is refused instead of hanging the caller inside `open(2)`. The package also provides atomic file publication and real-directory creation that refuse symlink redirection where a real directory was required. It is consumed by the `skills` and `agents` discovery scanners, the `ignore` resolver, the `tools/builtins` file tools, and the orchestration step-dump tracker.

## Key Files

- `github.com/v0lka/sp4rk/safeio/safeio.go` - package contract, the refusal error taxonomy, and capped reads (`ReadFile`, `ReadFileLimited`, `IsRegular`)
- `github.com/v0lka/sp4rk/safeio/write.go` - hardened and atomic writes (`WriteFile`, `WriteFileAtomic`) plus real-directory creation (`MkdirAllReal`, `CheckRealDirsBelow`, `MkdirAllRealWithin`)
- `github.com/v0lka/sp4rk/safeio/open_unix.go` - unix safe-open (`//go:build unix`): `syscall.Open` with `O_NONBLOCK|O_CLOEXEC`, regularity by fstat on the open descriptor, `O_NOFOLLOW` in `OpenFileNoFollow`
- `github.com/v0lka/sp4rk/safeio/open_windows.go` - Windows safe-open (`//go:build windows`): `os.Open`/`os.OpenFile` with a handle-based regularity check; `OpenFileNoFollow` carries `FILE_FLAG_OPEN_REPARSE_POINT` and refuses a final reparse point with `syscall.ELOOP`

## Core Types

```go
// safeio.go — refusal taxonomy
var ErrNotRegular = errors.New("not a regular file")
var ErrTooLarge   = errors.New("file too large")

type NotRegularError struct{ Path string }             // Error(); Is() → matches ErrNotRegular
type TooLargeError struct{ Path string; Limit int64 } // Error(); Is() → matches ErrTooLarge

const DefaultMaxFileSize = int64(10 << 20) // 10 MiB — the read cap ReadFile enforces

func ReadFile(path string) ([]byte, error)                      // ReadFileLimited(path, DefaultMaxFileSize)
func ReadFileLimited(path string, limit int64) ([]byte, error)
func IsRegular(path string) (bool, error)
```

```go
// write.go — directory guarantees
var ErrSymlink      = errors.New("symlink in path")
var ErrNotDirectory = errors.New("not a directory")

type SymlinkError struct{ Path string } // Error(); Is() → matches ErrSymlink

func WriteFile(path string, data []byte, perm fs.FileMode) error
func WriteFileAtomic(path string, data []byte, perm fs.FileMode) error
func MkdirAllReal(path string, perm fs.FileMode) error
func CheckRealDirsBelow(from, path string) error
func MkdirAllRealWithin(root, dir string, perm fs.FileMode) error
```

```go
// open_unix.go / open_windows.go — identical exported signatures on both platforms
func Open(path string) (*os.File, error)
func OpenFile(path string, flag int, perm fs.FileMode) (*os.File, error)
func OpenFileNoFollow(path string, flag int, perm fs.FileMode) (*os.File, error)
```

## Flow

Every open funnels through the shared guard; writes and directory creation build on it:

```
        Open / OpenFile / OpenFileNoFollow
        unix: syscall.Open(flags | O_NONBLOCK | O_CLOEXEC)   (never blocks in open(2))
        windows: os.Open / os.OpenFile   (named pipes cannot sit at a filesystem path)
                        │
                        ▼
        regularity check on the ALREADY-OPEN descriptor (fstat / handle Stat)
                │                           │
        regular │                 non-regular (FIFO, socket, device, directory)
                ▼                           ▼
        caller proceeds              *NotRegularError (errors.Is → ErrNotRegular)
        write-open of a readerless FIFO (unix):    OpenFileNoFollow:
        O_NONBLOCK makes the open fail fast          unix: +O_NOFOLLOW —
        with ENXIO (*fs.PathError)                   final symlink → ELOOP
        OpenFileNoFollow (windows): +FILE_FLAG_OPEN_REPARSE_POINT — the
        reparse point opens as itself and the post-open fstat refuses it
        → ELOOP (same errors.Is(syscall.ELOOP) contract as unix)

  reads:  ReadFileLimited reads at most limit+1 bytes;
          over the cap → *TooLargeError (errors.Is → ErrTooLarge)

  writes: WriteFile        → OpenFileNoFollow(O_WRONLY|O_CREATE|O_TRUNC)
          WriteFileAtomic  → CreateTemp(dir, base+".tmp*") → Chmod(perm) → Write
                             → Sync → Close → Rename over path (atomic replace)
```

Directory creation (`MkdirAllReal`, extended by `MkdirAllRealWithin`):

1. Resolve each pre-existing symlink component once to its real target (operator intent: macOS `/var`, a symlinked agent directory); each resolved target must be a real directory.
2. A dangling or looping link returns `*SymlinkError`; any other pre-existing non-directory component returns an `ErrNotDirectory`-wrapped `*fs.PathError`.
3. Create the missing tail component by component; an `EEXIST` from a swap-in (including a symlink) triggers a re-check that refuses whatever appeared.
4. `MkdirAllRealWithin` additionally verifies — before creating anything and again after — that the fully resolved tree lands inside the resolved `root` boundary, failing with `*SymlinkError` and creating nothing otherwise.

## Invariants

- Every open performed by the package returns or fails without waiting: a FIFO at a probed path is refused, never awaited.
- Regularity is always decided on the already-open descriptor, so the file that is checked is exactly the file that is read or written — there is no Stat→Open window.
- `ReadFile` always bounds a single read at `DefaultMaxFileSize` bytes and `ReadFileLimited` at `limit` bytes; a file of exactly the cap reads in full, and a file over it is refused with `ErrTooLarge` instead of being read into memory.
- Every refusal class is distinguishable with `errors.Is`: `ErrNotRegular`, `ErrTooLarge`, `ErrSymlink`, `ErrNotDirectory`, and (final symlink at an open target, both platforms) `syscall.ELOOP`.
- A missing file always surfaces the underlying open error (testable with `errors.Is(err, fs.ErrNotExist)`).
- `WriteFileAtomic` always publishes by rename: a reader of the target sees either the complete old or the complete new contents, and on any failure the target keeps its old contents and the temporary file is removed.
- `WriteFileAtomic` applies `perm` exactly via fchmod, independent of the process umask; `WriteFile` and `MkdirAllReal` pass `perm` through `open(2)`/`mkdir(2)`, which apply the umask.
- `MkdirAllReal` always resolves pre-existing symlinks before creating anything and always re-checks a component whose `Mkdir` failed with `EEXIST`.
- `MkdirAllRealWithin` always checks containment against the fully resolved root both before and after creation; an escaping link fails with nothing created.
- On Windows, `checkRealDirInfo` refuses a directory junction exactly like a symlink (`os.Lstat` reports it `ModeIrregular` there).
- The package imports only the standard library and `github.com/v0lka/sp4rk/pathutil`; it never imports higher engine layers.

## Configuration

| Parameter | Default / behavior |
| --------- | ------------------ |
| `DefaultMaxFileSize` | `int64(10 << 20)` — 10 MiB read cap enforced by `ReadFile` |
| `ReadFileLimited` `limit` | any cap; a negative limit is treated as zero; exactly `limit` bytes reads in full |
| `perm` arguments | `WriteFile`: umask-filtered by `open(2)`; `WriteFileAtomic`: exact bits via fchmod; `MkdirAllReal`/`MkdirAllRealWithin`: umask-filtered by `mkdir(2)`; on Windows modes map onto the read-only attribute only |
| `OpenFile`/`OpenFileNoFollow` `flag` | caller's `os` flags; on unix `O_NONBLOCK|O_CLOEXEC` (plus `O_NOFOLLOW` in `OpenFileNoFollow`) are OR-ed in — on a regular file `O_NONBLOCK` has no effect |

## Extension Points

- New hardened read or write operations route through `Open`, `OpenFile`, or `OpenFileNoFollow` so the regularity guard and refusal taxonomy apply uniformly; direct use of `os.Open`/`os.ReadFile`/`os.WriteFile` on fixed or workspace-controlled paths is the anti-pattern this package exists to replace.
- New directory-tree operations build on `MkdirAllReal`, pairing it with `CheckRealDirsBelow` (content-controlled subtrees) or `MkdirAllRealWithin` (containment boundaries).
- Platform divergence stays confined to `open_unix.go` and `open_windows.go` behind identical exported signatures; behavior differences are documented in the platform files' doc comments (with the nofollow contract itself unified at `errors.Is(err, syscall.ELOOP)` on both).

## Related Specs

- [Architecture Layers](../architecture/layers.md) - safeio is a leaf utility package; the import rule keeps it free of upward dependencies
- [Skills](skills.md) - skill discovery reads SKILL.md and skill resources through safeio's capped, non-blocking reads
- [Subagent Profiles](agents.md) - AGENT.md discovery reads profiles through safeio's capped, non-blocking reads
