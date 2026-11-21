# Append-Safe Log Codecs

## Role

Append-safe compression codecs for rotation-style logs: `OpenAppend` returns an `io.WriteCloser` whose every completed `Write` is a decodable prefix of the file — visible to any concurrent reader and recoverable after a crash-before-Close — while appending onto an existing gzip/zstd stream as a valid multi-member / multi-frame continuation.

## Key Files

- `github.com/v0lka/sp4rk/appendcodec/appendcodec.go` - package contract, the `Codec` constants (`CodecNone`/`CodecGzip`/`CodecZstd`), and `ParseCodec`/`ErrUnknownCodec`
- `github.com/v0lka/sp4rk/appendcodec/detect.go` - `DetectMagic` codec sniffing (`sniff`, `textPrefix`) through `safeio.OpenFileNoFollow`
- `github.com/v0lka/sp4rk/appendcodec/writer.go` - `OpenAppend`, the refcounted shared-stream registry, `appendWriter`/`queuedWriter`, magic verification, zstd window clamping, and the error taxonomy (`ErrClosed`, `ErrMagicMismatch`, `ErrLocked`, `MagicMismatchError`)
- `github.com/v0lka/sp4rk/appendcodec/lock_unix.go` - unix advisory lock (`flock(2)`, `LOCK_EX|LOCK_NB` via `golang.org/x/sys/unix`)
- `github.com/v0lka/sp4rk/appendcodec/lock_windows.go` - Windows exclusive `LockFileEx` range lock on a beyond-EOF offset (2^63)

## Behavior

**Core property.** `Write` blocks until the codec has flushed its encoding of the record to the file (gzip `Flush` / zstd `Flush` after every encoder write; a direct `write(2)` for `CodecNone`), so a record is decodable the moment `Write` returns. Nothing fsyncs: the guarantee ends at process death, and power-loss durability is the caller's concern.

**Open path.** `OpenAppend(path, codec, windowLog)` validates the codec before any filesystem side effect, resolves the registry key (`filepath.Abs` + `Clean`; lower-cased on Windows), then branches by codec:

- `CodecGzip`/`CodecZstd` open through `safeio.OpenFileNoFollow` (`O_RDWR|O_CREATE|O_APPEND`), take the exclusive advisory lock, read the magic prefix from the SAME descriptor, and verify it against the requested codec before building the encoder (`compress/gzip` at the default level; `klauspost/compress` zstd with `WithEncoderConcurrency(1)` and, for a positive `windowLog`, `WithWindowSize(1<<windowLog)` clamped into `[MinWindowLog, MaxWindowLog]` = 10..29). There is no Stat→Open window: inspection and append share one descriptor, and the path is locked while the prefix is inspected.
- `CodecNone` opens the same flags through `safeio.OpenFileNoFollow` with no lock and no registry — the pre-appendcodec compatibility semantics: independent plain handles, any number of writers in this or other processes appending side by side (each `write(2)` atomic at end of file). The magic sniff is retained but unsynchronized (best-effort).

**One path, one encoder (compressed codecs).** An in-process registry (`streamRegistry`) maps normalized absolute paths to live `sharedStream` values. A same-path second `OpenAppend` returns an independent `queuedWriter` handle that queues its records through the one live encoder's mutex; the joining open's own parameters (including `windowLog`) are ignored — the first open's encoder serves the stream for its lifetime. Handles are refcounted: the LAST `Close` writes the codec footer, closes the descriptor, releases the lock, and drops the registry entry. Across processes the lock cannot be shared, so a conflicting open fails with `ErrLocked`; this process reaches the same error on a registry miss caused by a differently-normalizing alias (symlinked parent directory, hardlink, Windows 8.3 short-name). A Windows case alias is not such a case — the key is lower-cased, so case variants hit one entry (on a case-sensitive Windows directory two case variants are distinct files that then share one entry).

**Close semantics.** `Close` is idempotent per handle. A handle whose own `Close` already returned keeps appending through the live stream until the last handle closes — `Write` reports `ErrClosed` only after the stream itself is shut down. A failed footer close still closes the descriptor and reports the footer error joined with any close error.

**Detection.** `DetectMagic(path, log)` sniffs at most 6 leading bytes: empty or text-shaped prefixes (optional UTF-8 BOM, printable ASCII, any ≥0x80 byte assumed UTF-8, JSON included) report `CodecNone`; complete gzip (`1f 8b`) or zstd (`28 b5 2f fd`) magic plus at least one stream byte reports that codec; a magic-only file is deliberately NOT recognized as its codec (no decoder can read it); binary content that is neither reports `CodecNone` with a single Warn record when a logger is supplied.

## Error Handling

| Error | Raised when | Distinguish via |
| ----- | ----------- | --------------- |
| `ErrMagicMismatch` (`*MagicMismatchError` carries `Path`, `Codec`, `Magic`) | existing content cannot be appended to with the requested codec (gzip onto text, zstd onto gzip, a different codec onto a live stream) | `errors.Is` |
| `ErrLocked` | the advisory lock is held by an open that cannot share the stream — another process, or a same-process alias that misses the registry; compressed codecs only | `errors.Is` |
| `ErrClosed` | `Write` on a stream whose last handle already closed (per-handle before that: no error) | `errors.Is` |
| `ErrUnknownCodec` | `ParseCodec` with an unknown name, or `OpenAppend`/`checkAppendMagic` with an undeclared codec value — before any filesystem side effect | `errors.Is` |
| `*fs.PathError` | open/read/lock failures (missing file, `ELOOP` from `OpenFileNoFollow`, `safeio.ErrNotRegular`, environment lock failures) | wrapped verbatim |

Open failures leave the file untouched; a magic conflict closes the descriptor and returns without writing.

## Invariants

- Every completed `Write` is a decodable prefix of the file: a concurrent reader and a crash-recovery decoder always recover every acknowledged record.
- For the compressed codecs, a path always has exactly one live encoder: concurrent encoders whose frame writes would interleave cannot exist while advisory locks hold (local filesystems, and network filesystems with a functioning lock manager).
- Every open goes through `safeio`: a symlink at the final component fails with `ELOOP` and a non-regular pre-existing target is refused with `ErrNotRegular`, on every platform; the magic is read from the same descriptor that is appended through.
- The compressed-codec lock lives with the descriptor: only the last `Close` or process death releases it, so a compressed handle dropped without `Close` holds the descriptor, the lock, and the registry entry until process death — in-process opens then silently join the leaked stream while every other process gets `ErrLocked`.
- A join matches by path string, not file identity: a renamed or replaced file keeps receiving records through the original descriptor while the new file at the path stays empty, so a rotator closes every handle on a path before renaming or replacing it.
- `CodecNone` takes no lock and consults no registry: opens always return independent handles, and concurrent writers interleave only between whole writes, never inside one.
- `DetectMagic` and `OpenAppend` never follow a symlink and never block on a non-regular file.
- Crash-truncated appends stay decodable up to the truncation point; a later `OpenAppend` lands behind that point, where sequential decoders stop — recovery (truncate to the last decodable boundary) is the caller's job.

## Related Specs

- [safeio.md](safeio.md) - the link-safe, non-blocking opens and the regularity guard every appendcodec open goes through
- [ADR-010: appendcodec gzip/zstd, xz rejected](../decisions/010-appendcodec-gzip-zstd-xz-rejected.md) - the decision record behind the codec set (xz has no usable Flush)
- [Architecture Layers](../architecture/layers.md) - appendcodec is a leaf utility package (stdlib, klauspost/compress, safeio); the import rule keeps it free of upward dependencies
