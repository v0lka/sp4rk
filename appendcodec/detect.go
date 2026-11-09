package appendcodec

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"

	"github.com/v0lka/sp4rk/safeio"
)

// magicLen is the number of leading bytes DetectMagic needs to distinguish
// every supported container: 2 for the gzip magic, 4 for the zstd magic, plus
// slack for a UTF-8 BOM ahead of JSON text.
const magicLen = 6

// DetectMagic sniffs the codec of an existing file from its first bytes and
// never follows a symlink: the open refuses a symlink at the final path
// component with ELOOP (unix; on Windows see the safeio parity note), and a
// non-regular pre-existing target (FIFO, socket, device, directory) is
// refused with safeio's *NotRegularError instead of blocking inside open(2).
//
// Results: an empty file and any file whose leading bytes are text — JSON
// text ('{' or '[' after an optional BOM and blanks), plain ASCII, or UTF-8
// — report CodecNone with a nil error. A file whose leading bytes carry a
// container magic plus at least one byte of stream reports CodecGzip or
// CodecZstd; a file consisting of exactly the magic bytes is not recognised
// as that codec (see sniff). Binary content that is
// neither reports CodecNone with no error but emits a single Warn record
// ("unknown magic; treating as uncompressed") carrying the path and the hex
// magic when log is non-nil — a nil log drops the warning. An open failure
// (missing file, ELOOP, not regular) is returned as-is with CodecNone.
func DetectMagic(path string, log *slog.Logger) (Codec, error) {
	f, err := safeio.OpenFileNoFollow(path, os.O_RDONLY, 0)
	if err != nil {
		return CodecNone, err
	}
	defer func() { _ = f.Close() }()

	prefix := make([]byte, magicLen)
	n, rerr := io.ReadFull(f, prefix)
	if rerr != nil && !errors.Is(rerr, io.EOF) && !errors.Is(rerr, io.ErrUnexpectedEOF) {
		return CodecNone, &os.PathError{Op: "read", Path: path, Err: rerr}
	}

	codec, known := sniff(prefix[:n])
	if known {
		return codec, nil
	}
	if log != nil {
		log.Warn("appendcodec: unknown magic; treating as uncompressed",
			"path", path, "magic", fmt.Sprintf("% x", prefix[:n]))
	}
	return CodecNone, nil
}

// gzipMagic and zstdMagic are the container signatures DetectMagic and
// OpenAppend recognise: gzip starts with 1f 8b, zstd frames with 28 b5 2f fd.
var (
	gzipMagic = []byte{0x1f, 0x8b}
	zstdMagic = []byte{0x28, 0xb5, 0x2f, 0xfd}
)

// sniff identifies a codec from a leading prefix of at most magicLen bytes.
// An empty prefix (missing or zero-length file) is a known CodecNone — there
// is nothing to be incompatible with. A prefix that carries the COMPLETE
// container magic plus at least one byte beyond it reports that codec — a
// real stream member is always longer than the magic alone (a gzip member is
// a 10-byte header + data + trailer; a zstd frame header alone exceeds the
// magic) — while a file consisting of exactly the magic bytes is deliberately
// NOT recognised as that codec: no sequential decoder can read it, so
// treating it as its codec would let OpenAppend append members that are
// unreachable (it instead fails with *MagicMismatchError; DetectMagic reports
// it as unknown-or-text). A prefix that looks like text (JSON or otherwise) is
// a known CodecNone. Anything else — binary content that is neither container
// magic nor text — reports (CodecNone, false) so callers can distinguish
// "plain" from "unknown" (DetectMagic warns; OpenAppend treats unknown
// content as appendable only for CodecNone).
func sniff(prefix []byte) (Codec, bool) {
	if len(prefix) == 0 {
		return CodecNone, true
	}
	if len(prefix) > len(gzipMagic) && bytes.HasPrefix(prefix, gzipMagic) {
		return CodecGzip, true
	}
	if len(prefix) > len(zstdMagic) && bytes.HasPrefix(prefix, zstdMagic) {
		return CodecZstd, true
	}
	if textPrefix(prefix) {
		return CodecNone, true
	}
	return CodecNone, false
}

// textPrefix reports whether the prefix looks like a text file: an optional
// UTF-8 BOM, then bytes that are printable ASCII (blank bytes included), with
// any byte >= 0x80 assumed to be part of UTF-8 multibyte text rather than
// binary. JSON text ('{'/'[' after optional blanks) is a special case of
// this, and is what rotation wrappers of structured logs encounter. A
// control byte (NUL, escapes, DEL) makes the prefix binary. The heuristic
// exists to classify plain logs as known-uncompressed; genuinely undecidable
// prefixes (all blanks so far) are treated as text.
func textPrefix(prefix []byte) bool {
	i := 0
	if bytes.HasPrefix(prefix, []byte{0xEF, 0xBB, 0xBF}) {
		i = 3
	}
	for ; i < len(prefix); i++ {
		b := prefix[i]
		switch {
		case b == '\t' || b == '\r' || b == '\n':
		case b < 0x20 || b == 0x7F:
			return false
		}
	}
	return true
}
