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
refused (via `safeio.OpenFileNoFollow`).

Candidate encoder libraries: the standard library's `compress/gzip`,
`github.com/klauspost/compress/zstd`, and `github.com/ulikunitz/xz`.

## Decision

The new `appendcodec` package supports exactly three codecs:

- `CodecNone` — plain pass-through bytes.
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
records. `DetectMagic` sniffs the first bytes of a file through
`safeio.OpenFileNoFollow`: gzip magic `1f 8b`, zstd magic `28 b5 2f fd`,
text content (JSON or otherwise) → `CodecNone`, anything binary and
unrecognised → `CodecNone` plus one Warn record on the supplied `*slog.Logger`.

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
- **Buffer writes and flush on a timer** — rejected: the contract is that
  every acknowledged record is decodable; a timer leaves an unbounded window
  of acknowledged-but-not-decodable records before a crash.

## Related

- [`safeio`](../../safeio/) — `OpenFileNoFollow` (link-safe, non-blocking
  opens with fstat-verified regularity) used by both `DetectMagic` and
  `OpenAppend`.
