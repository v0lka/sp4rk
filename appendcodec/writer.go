package appendcodec

import (
	"compress/gzip"
	"errors"
	"fmt"
	"io"
	"os"
	"sync"

	"github.com/klauspost/compress/zstd"
	"github.com/v0lka/sp4rk/safeio"
)

// Bounds for the windowLog parameter of OpenAppend on CodecZstd: the zstd
// window size is 1<<windowLog, and klauspost/compress accepts window sizes
// from MinWindowSize (1 KiB) through MaxWindowSize (512 MiB) — logs 10
// through 29. Values outside the range are clamped into it. windowLog <= 0
// (and the whole parameter for CodecGzip and CodecNone) means the encoder
// default. CodecGzip and CodecNone ignore the parameter: gzip carries no
// window option and plain writes carry no window at all.
const (
	MinWindowLog = 10 // 1 KiB, klauspost/compress zstd MinWindowSize
	MaxWindowLog = 29 // 512 MiB, klauspost/compress zstd MaxWindowSize
)

// ErrClosed is returned by Write on a writer that has already been Closed.
// Close itself is idempotent and reports nil after the first call. Test it
// with errors.Is.
var ErrClosed = errors.New("appendcodec: writer already closed")

// ErrMagicMismatch is returned by OpenAppend when the existing content of the
// target file cannot be appended to with the requested codec (gzip onto plain
// text, zstd onto gzip, any codec onto itself mismatched). Test it with
// errors.Is; the concrete type is *MagicMismatchError.
var ErrMagicMismatch = errors.New("appendcodec: existing content does not match codec")

// MagicMismatchError reports the codec requested for OpenAppend and the
// unrecognised-or-conflicting leading bytes found in the target file. It
// matches errors.Is(err, ErrMagicMismatch).
type MagicMismatchError struct {
	Path  string
	Codec Codec
	Magic []byte
}

func (e *MagicMismatchError) Error() string {
	return fmt.Sprintf("appendcodec: cannot append %s to %s: existing magic % x", e.Codec, e.Path, e.Magic)
}

// Is makes errors.Is(err, ErrMagicMismatch) report true for every
// MagicMismatchError.
func (e *MagicMismatchError) Is(target error) bool { return target == ErrMagicMismatch }

// codecStream is the codec encoder facade both backing encoders satisfy:
// *gzip.Writer from the standard library and *zstd.Encoder from
// klauspost/compress. A nil codecStream is the CodecNone pass-through.
type codecStream interface {
	io.Writer
	Flush() error
	Close() error
}

// OpenAppend opens path for codec-compressed appending and returns a writer
// whose every completed Write is already on disk (the codec's Flush runs
// after each Write, so a concurrent reader — and a decoder of a file whose
// writer crashed before Close — always sees every acknowledged record), and
// whose Close writes the codec footer (gzip member trailer / zstd frame
// epilogue; nothing for CodecNone) and releases the descriptor.
//
// The open is link-safe and TOCTOU-free: safeio.OpenFileNoFollow refuses a
// symlink at the final path component (ELOOP on unix; Windows parity per
// safeio) and any non-regular pre-existing target instead of blocking, and
// the existing magic is read from the SAME O_RDWR|O_APPEND|O_CREATE
// descriptor that is then appended through — there is no window in which the
// path can be swapped for a differently-compressed file between inspection
// and append. The detected magic is verified against codec: appending is
// allowed when the file is empty, already carries the requested codec's
// magic (gzip multi-member / zstd multi-frame), or — for CodecNone — is not
// gzip/zstd at all (plain or JSON text is fine; unrecognised content is also
// allowed, because a plain pass-through cannot corrupt it). A conflict fails
// with a *MagicMismatchError and leaves the file untouched.
//
// windowLog sizes the zstd encoder window as 1<<windowLog, clamped to
// [MinWindowLog, MaxWindowLog]; values <= 0 use the encoder default. It is
// ignored by CodecGzip and CodecNone. The zstd encoder always runs with
// WithEncoderConcurrency(1); gzip runs at the standard-library default level.
//
// The returned writer is safe for concurrent Write calls from multiple
// goroutines: a mutex serializes each Write together with its Flush, so
// records never interleave. Write after Close reports ErrClosed; Close is
// idempotent. One path should have exactly one live writer.
func OpenAppend(path string, codec Codec, windowLog int) (io.WriteCloser, error) {
	if codec == CodecZstd && windowLog > 0 {
		windowLog = clampWindowLog(windowLog)
	}
	f, err := safeio.OpenFileNoFollow(path, os.O_RDWR|os.O_CREATE|os.O_APPEND, 0o644)
	if err != nil {
		return nil, err
	}

	prefix := make([]byte, magicLen)
	n, rerr := f.ReadAt(prefix, 0)
	if rerr != nil && !errors.Is(rerr, io.EOF) && !errors.Is(rerr, io.ErrUnexpectedEOF) {
		_ = f.Close()
		return nil, &os.PathError{Op: "read", Path: path, Err: rerr}
	}
	prefix = prefix[:n]
	if err := checkAppendMagic(codec, path, prefix); err != nil {
		_ = f.Close()
		return nil, err
	}

	w := &appendWriter{f: f}
	switch codec {
	case CodecGzip:
		w.cs = gzip.NewWriter(f)
	case CodecZstd:
		opts := []zstd.EOption{zstd.WithEncoderConcurrency(1)}
		if windowLog > 0 {
			opts = append(opts, zstd.WithWindowSize(1<<windowLog))
		}
		enc, zerr := zstd.NewWriter(f, opts...)
		if zerr != nil {
			_ = f.Close()
			return nil, fmt.Errorf("appendcodec: zstd encoder for %s: %w", path, zerr)
		}
		w.cs = enc
	case CodecNone:
		// nil codecStream: Write goes straight to f.
	default:
		_ = f.Close()
		return nil, fmt.Errorf("%w: %s", ErrUnknownCodec, codec)
	}
	return w, nil
}

// clampWindowLog folds a caller-supplied zstd windowLog exponent into the
// range klauspost/compress accepts, so an out-of-range knob degrades to the
// nearest working window instead of failing the open.
func clampWindowLog(log2 int) int {
	if log2 < MinWindowLog {
		return MinWindowLog
	}
	if log2 > MaxWindowLog {
		return MaxWindowLog
	}
	return log2
}

// checkAppendMagic validates that prefix — the leading bytes read from the
// append target path — is compatible with appending codec. Empty is
// compatible with everything; matching container magic is compatible
// (multi-member / multi-frame); CodecNone is compatible with any content
// that is not the other container's magic (JSON text and unrecognised
// content included).
func checkAppendMagic(codec Codec, path string, prefix []byte) error {
	magic, known := sniff(prefix)
	switch codec {
	case CodecGzip:
		if len(prefix) == 0 || (known && magic == CodecGzip) {
			return nil
		}
	case CodecZstd:
		if len(prefix) == 0 || (known && magic == CodecZstd) {
			return nil
		}
	case CodecNone:
		if !known || magic == CodecNone {
			return nil
		}
	default:
		return fmt.Errorf("%w: %s", ErrUnknownCodec, codec)
	}
	return &MagicMismatchError{Path: path, Codec: codec, Magic: prefix}
}

// appendWriter is the io.WriteCloser OpenAppend returns: the open descriptor
// plus, for gzip/zstd, the codec encoder. The mutex makes Write (+ its
// trailing Flush) and Close mutually exclusive, so concurrent writers cannot
// interleave records or race the footer.
type appendWriter struct {
	mu     sync.Mutex
	f      *os.File
	cs     codecStream // nil for CodecNone
	closed bool
}

// Write appends p as the next record and blocks until the codec has flushed
// its encoding of it to the file, so the record is decodable the moment this
// call returns — before Close, and even if the process dies immediately
// after. The error, if any, may come from the encoder write or from the
// flush; n counts the bytes the encoder accepted either way.
func (w *appendWriter) Write(p []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed {
		return 0, ErrClosed
	}
	var (
		n   int
		err error
	)
	if w.cs != nil {
		n, err = w.cs.Write(p)
		if err == nil {
			err = w.cs.Flush()
		}
	} else {
		n, err = w.f.Write(p)
	}
	return n, err
}

// Close writes the codec footer (gzip member trailer / zstd frame epilogue;
// none for CodecNone) and closes the descriptor. It is idempotent: a second
// Close reports nil and does nothing. A Close that fails on the footer still
// closes the descriptor, reporting the footer error, joined with any close
// error.
func (w *appendWriter) Close() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed {
		return nil
	}
	w.closed = true
	var err error
	if w.cs != nil {
		err = w.cs.Close()
	}
	if ferr := w.f.Close(); err == nil {
		err = ferr
	} else if ferr != nil {
		err = errors.Join(err, ferr)
	}
	return err
}
