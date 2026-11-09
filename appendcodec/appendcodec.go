// Package appendcodec provides append-safe compression codecs for
// rotation-style logs: an io.WriteCloser that appends gzip or zstd streams to
// an existing file while guaranteeing that every completed Write is already a
// decodable prefix on disk. A companion DetectMagic sniffs the codec of an
// existing file from its leading bytes without ever following a symlink.
//
// The core property — a write is durable the moment Write returns — is what
// makes rotation wrappers possible: a reader (or a decoder of a file whose
// writer crashed before Close) always recovers every record that was
// acknowledged, and the appended-to file stays a valid continuation of the
// existing stream (gzip multi-member / zstd multi-frame). The codecs are
// deliberately limited to gzip and zstd: xz was rejected because
// github.com/ulikunitz/xz has no usable Flush on its write path and therefore
// cannot honour the append contract (see ADR-010 in specs/decisions/).
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
// Writer concurrency contract: a single writer returned by OpenAppend is safe
// for concurrent use — a mutex serializes Write (+ the trailing Flush) so
// concurrent goroutines never interleave each other's records, provided each
// Write carries one whole record (a line). Opening several writers on the
// same compressed path is NOT supported: concurrent encoders would interleave
// their frames' write(2)s and corrupt the stream. One path, one writer.
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
	// descriptor.
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
