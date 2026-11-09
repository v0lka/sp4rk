// Package appendcodec provides append-safe compression codecs for
// rotation-style logs: an io.WriteCloser that appends gzip or zstd streams to
// an existing file while guaranteeing that every completed Write is a
// decodable prefix — visible to any concurrent reader and recoverable by a
// decoder after the writer died before Close. Nothing here fsyncs: the
// guarantee ends at process death, and surviving a power loss is the
// caller's concern. A companion DetectMagic sniffs the codec of an existing
// file from its leading bytes without ever following a symlink.
//
// The core property — a write is decodable by any reader the moment Write
// returns — is what makes rotation wrappers possible: a reader (or a decoder
// of a file whose writer crashed before Close) always recovers every record
// that was acknowledged, and the appended-to file stays a valid continuation
// of the existing stream (gzip multi-member / zstd multi-frame). The codecs
// are deliberately limited to gzip and zstd: xz was rejected because
// github.com/ulikunitz/xz has no usable Flush on its write path and
// therefore cannot honour the append contract (see ADR-010 in
// specs/decisions/).
//
// Every open goes through safeio: DetectMagic refuses to traverse a symlink
// at the final path component (ELOOP on unix; see the safeio Windows parity
// note), refuses non-regular files instead of blocking on a planted FIFO, and
// OpenAppend reads the existing magic from the SAME descriptor it appends
// through, so there is no Stat→Open window in which the file could be swapped
// for a differently-compressed target. OpenAppend also verifies the detected
// magic against the requested codec and refuses a mismatch with an error
// matching ErrMagicMismatch rather than producing an undecodable file.
//
// Writer concurrency contract: a writer is safe for concurrent use — a mutex
// serializes Write (+ the trailing Flush) so concurrent goroutines never
// interleave each other's records, provided each Write carries one whole
// record (a line). For the compressed codecs the stronger rule holds: one
// path, one encoder. OpenAppend holds an exclusive advisory lock on the file
// for the lifetime of the underlying descriptor (flock on unix, a
// beyond-EOF LockFileEx range on Windows), so concurrent encoders — which
// would interleave their frames' write(2)s and corrupt the stream — cannot
// exist. A second OpenAppend on the same path from the SAME process does not
// fail: it returns an independent handle that queues its records through the
// one live encoder, and handles are refcounted — the codec footer and the
// lock are released by the LAST handle's Close. Opening the live path with a
// different codec fails with an error matching ErrMagicMismatch. From
// ANOTHER process the lock cannot be shared, so the conflicting open fails
// with ErrLocked instead of interleaving frames.
//
// CodecNone deliberately keeps the pre-appendcodec semantics: no lock, no
// registry, no queue. Every open returns an independent plain O_APPEND
// handle, and any number of writers — in this process or others — may
// append side by side: each write(2) lands atomically at the end of file,
// so records of concurrent writers interleave only between write calls,
// never inside one. This preserves compatibility with hosts that already
// fan plain JSONL logs out over several writers. Only the sniff guard
// survives: appending plain bytes onto a gzip/zstd file still fails with
// ErrMagicMismatch (best-effort — nothing serializes the check against
// concurrent opens), and no ErrLocked exists for a none-codec path.
//
// Crash-truncated appends: a file whose writer crashed mid-member (no Close,
// no footer) stays decodable up to the truncation point — but a later
// OpenAppend appends behind that point, and sequential decoders stop at the
// truncation error without ever reaching the newer records. A flow that
// reopens a crashed file should recover it first (scan to the last decodable
// boundary and truncate there); this package ships no recovery helper yet.
package appendcodec

import (
	"errors"
	"fmt"
)

// Codec identifies the compression codec of an append-target file.
type Codec int

const (
	// CodecNone writes plain bytes with no framing. Flush and the footer are
	// no-ops: every Write is a direct write(2), and Close just closes the
	// descriptor. It is the compatibility codec: opens take no advisory lock
	// and return independent handles, so concurrent writers append side by
	// side exactly as before this package existed (see the package
	// concurrency contract).
	CodecNone Codec = iota
	// CodecGzip appends RFC 1952 gzip members. Appending to a file that
	// already contains gzip data produces a valid multi-member gzip stream.
	CodecGzip
	// CodecZstd appends zstd frames (klauspost/compress, Concurrency 1).
	// Appending to a file that already contains zstd frames produces a valid
	// concatenated-frame stream.
	CodecZstd
)

// String returns the canonical lowercase codec name used by ParseCodec.
func (c Codec) String() string {
	switch c {
	case CodecNone:
		return "none"
	case CodecGzip:
		return "gzip"
	case CodecZstd:
		return "zstd"
	default:
		return fmt.Sprintf("codec(%d)", int(c))
	}
}

// ErrUnknownCodec is reported by ParseCodec for a name that is not one of the
// String values of the Codec constants. Test it with errors.Is.
var ErrUnknownCodec = errors.New("appendcodec: unknown codec")

// ParseCodec maps a canonical codec name ("none", "gzip", "zstd") to its
// Codec. Any other name reports a wrapped ErrUnknownCodec.
func ParseCodec(name string) (Codec, error) {
	switch name {
	case "none":
		return CodecNone, nil
	case "gzip":
		return CodecGzip, nil
	case "zstd":
		return CodecZstd, nil
	default:
		return CodecNone, fmt.Errorf("%w: %q", ErrUnknownCodec, name)
	}
}
