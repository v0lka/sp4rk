# ADR-010: Append-safe codec writers — gzip and zstd, xz rejected

## Status

Accepted

## Context

Rotation-style logs need to append compressed records to a file that stays
decodable while it is being written: a reader must see every acknowledged
record before the writer closes, and a writer that dies before Close must
leave the file a decodable prefix rather than garbage. A codec is usable here
only if its encoder can flush the encoding of every completed `Write` to the
file without terminating the stream, and if appending to an existing file of
the same codec yields one continuous, decodable stream (gzip multi-member /
zstd multi-frame). Additionally, the open path must be link-safe and
TOCTOU-free: the existing content is inspected through the same descriptor
that is later appended through, and symlinks and non-regular files are
refused (via `safeio.OpenFileNoFollow`). The link-safety half — the refusal
of a final-component symlink (ELOOP) — is a unix guarantee; on Windows,
`safeio.OpenFileNoFollow` still resolves final-component symlinks (the safeio
Windows parity note). The TOCTOU closure — inspection through the same
descriptor that is appended through — holds on both platforms.

Candidate encoder libraries: the standard library's `compress/gzip`,
`github.com/klauspost/compress/zstd`, and `github.com/ulikunitz/xz`.

## Decision

The new `appendcodec` package supports exactly three codecs:

- `CodecNone` — plain pass-through bytes; the compatibility codec: opens
  take no advisory lock and return independent handles, so concurrent
  writers append side by side exactly as they did before this package
  existed.
- `CodecGzip` — standard library `compress/gzip` (default level).
- `CodecZstd` — `github.com/klauspost/compress` **v1.20.3** (pinned direct
  dependency), encoder built with `WithEncoderConcurrency(1)` and an optional
  `WithWindowSize(1<<windowLog)` clamped to `[MinWindowLog, MaxWindowLog]`
  (klauspost `MinWindowSize` 1 KiB … `MaxWindowSize` 512 MiB).

`OpenAppend` guarantees the append contract by running the encoder's `Flush`
after every `Write` (gzip flushes the pending deflate blocks; klauspost zstd
dispatches complete non-final blocks — the frame epilogue is written only by
`Close`, which is where the footer lands), and by serializing `Write`/`Flush`
and `Close` behind a mutex so concurrent goroutines cannot interleave
records. For the compressed codecs one path has exactly one live *encoder*,
enforced: `OpenAppend` takes an exclusive advisory lock on the file for the
lifetime of its descriptor — `flock(2)` on unix; on Windows a beyond-EOF
`LockFileEx` range, because Windows region locks are mandatory for other
handles and the range therefore sits 2^63 ≈ 8 EiB into the file, where no
content byte can ever land, so rotation readers and content writes stay
unblocked. Within one process the one-encoder rule is invisible: a second
`OpenAppend` for the same path does not fail — it returns an independent,
refcounted handle whose records queue through the one live encoder (never a
second encoder), and the last handle's `Close` writes the footer and
releases the lock. Opening the live path with a different codec fails with
an error matching `ErrMagicMismatch`. `ErrLocked` remains for genuine
cross-process conflicts (compressed codecs only): another process holding
the path fails the open instead of queueing, because records cannot be
merged across process boundaries — and the lock dies with the descriptor (a
last Close or a crash releases it). `CodecNone` is deliberately outside
this machinery: the compatibility codec takes no lock and consults no
registry — every open returns an independent plain `O_APPEND` handle, and
any number of writers, in this process or others, may append side by side
(each `write(2)` lands atomically at the end of file), preserving the
pre-appendcodec multi-writer behavior. Only the magic sniff survives:
plain bytes on a gzip/zstd file still fail with `ErrMagicMismatch`,
best-effort because nothing serializes the check.
`DetectMagic` sniffs the first bytes of a file through
`safeio.OpenFileNoFollow`: gzip magic `1f 8b`, zstd magic `28 b5 2f fd`,
text content (JSON or otherwise) → `CodecNone`, anything binary and
unrecognised → `CodecNone` plus one Warn record on the supplied
`*slog.Logger`. A file consisting of exactly the magic bytes is not
recognised as that codec — no sequential decoder can read a magic-only
head — so appending the codec onto it fails with `*MagicMismatchError`
instead of writing members no decoder can reach.

**xz is NOT included.** `github.com/ulikunitz/xz` was rejected by the
operator for the rotation wrapper: its xz writer has no usable `Flush` on the
write path — the xz container cannot emit a decodable prefix mid-stream — so
it cannot honour the append contract this package exists for. This decision
is recorded here so the rejection is not re-litigated by accident; the
`Codec` enum intentionally has no `CodecXz`, and `ParseCodec("xz")` reports
`ErrUnknownCodec`.

## Consequences

- Every record acknowledged by `OpenAppend`'s writer is decodable
  immediately — before `Close`, and after a crash that never reaches `Close`
  (the decoder recovers all flushed records and stops with a truncation
  error at the tail).
- Appends compose: reopening the same path with the detected codec and
  appending produces a valid multi-member gzip / multi-frame zstd stream.
  `OpenAppend` verifies the existing magic against the requested codec on
  the same descriptor (refuses a mismatch with `*MagicMismatchError`
  matching `ErrMagicMismatch`) instead of producing undecodable files.
- For the compressed codecs one path has exactly one live encoder, enforced
  by the advisory `OpenAppend` lock: a second encoder cannot interleave
  frames into the stream. Same-process writers share the encoder through
  refcounted queued handles; a cross-process second open fails with
  `ErrLocked`. Sequential reopen-after-Close (the rotation flow) is
  unaffected. `CodecNone` keeps the pre-appendcodec semantics — no lock, no
  queue: concurrent writers append side by side, each `write(2)` atomic at
  the end of file (compatibility with hosts that already fan plain JSONL
  logs out over several writers); only the best-effort magic sniff is
  retained.
- Appending to a crash-truncated file is NOT a recovery path: the truncated
  member/frame head never becomes decodable, so a sequential decoder stops
  there with a truncation error and never reaches the records appended after
  it — they sit in the file, unreachable. A flow that reopens a crashed file
  must recover it first (scan to the last decodable boundary and truncate
  there) or accept that post-crash appends are unreadable; no recovery helper
  ships yet.
- The codec surface is closed: adding a codec means an ADR superseding this
  one. klauspost/compress becomes a pinned direct dependency of the module
  (v1.20.3; upgrades are normal dependency maintenance, the pin records the
  version the append behaviour was verified against).
- Memory for zstd is bounded by the caller-chosen window (`windowLog`);
  `Concurrency(1)` keeps the encoder single-goroutine and deterministic.
- DetectMagic's text heuristic classifies plain/UTF-8 text as known-uncompressed
  (no warning); only binary non-magic content warns. A binary blob that
  happens to start with six printable bytes is classified as text — an
  accepted limitation for a log-file sniffer.

## Alternatives Considered

- **xz (`github.com/ulikunitz/xz`)** — rejected: no usable `Flush` on the
  write path; a rotation wrapper built on it could not make records visible
  before `Close` (operator decision; see Context).
- **Brotli, lz4** — rejected for now: no consumer demand for the rotation
  wrapper; the closed `Codec` enum keeps the surface minimal until one
  exists.
- **Detect-open-append as two separate opens (detect, then reopen)** —
  rejected: a local adversary could swap the file between the two opens
  (TOCTOU); the shipped design reads the magic from the descriptor it
  appends through.
- **Advisory lock for CodecNone too** — rejected: plain records need no
  encoder to serialize (each `write(2)` is already atomic at the end of
  file), and locking would break hosts that fan plain JSONL logs out over
  several writers — across processes the second open would fail with
  `ErrLocked` where it previously just appended. Hence the codec-scoped
  contract: locks and queues for gzip/zstd, compatibility append for none.
- **Buffer writes and flush on a timer** — rejected: the contract is that
  every acknowledged record is decodable; a timer leaves an unbounded window
  of acknowledged-but-not-decodable records before a crash.

## Related

- [`safeio`](../../safeio/) — `OpenFileNoFollow` (link-safe, non-blocking
  opens with fstat-verified regularity) used by both `DetectMagic` and
  `OpenAppend`.
