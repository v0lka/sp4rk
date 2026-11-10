package appendcodec

import (
	"compress/gzip"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"

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

// ErrClosed is returned by Write on a writer whose underlying stream has
// been closed. For CodecNone every open is an independent stream, so the
// error follows that writer's own Close. For the compressed codecs the
// handles of one path share one stream, and it closes with the LAST handle
// Close: until then a handle keeps appending through the live encoder —
// whether or not it has Closed itself — and Write reports no error. Close
// itself is idempotent and reports nil after the first call. Test it with
// errors.Is.
//
// This is deliberately unlike the io.WriteCloser convention (a
// write-after-close returns an error): while a sibling handle keeps the
// shared stream alive, a handle whose OWN Close already returned keeps
// writing — "handle A: A.Close(); A.Write(p)" appends p to the stream and
// reports (len(p), nil) until the LAST handle's Close makes the write
// surface ErrClosed. Hosts that must not append behind their own Close
// should track handle liveness themselves.
var ErrClosed = errors.New("appendcodec: writer already closed")

// ErrMagicMismatch is returned by OpenAppend when the existing content of the
// target file cannot be appended to with the requested codec (gzip onto plain
// text, zstd onto gzip, any codec onto itself mismatched). Test it with
// errors.Is; the concrete type is *MagicMismatchError.
var ErrMagicMismatch = errors.New("appendcodec: existing content does not match codec")

// ErrLocked is returned by OpenAppend when another PROCESS holds the target
// path. It applies only to the compressed codecs (CodecGzip, CodecZstd),
// which serialize every writer of a path behind one advisory lock and one
// encoder; CodecNone is the compatibility codec — it takes no lock at all
// and never reports this error. Within one process there is no such error
// for a compressed path either: further OpenAppend calls for the same path
// share the live writer — each returns an independent handle whose records
// queue through the one encoder's mutex, so nothing interleaves and the OS
// lock is taken only once. Across processes the lock cannot be shared or
// merged, so a conflicting open fails immediately with ErrLocked instead of
// interleaving its frames into the other process's stream. The lock dies
// with the descriptor — released by the LAST handle Close, and by process
// death. Test it with errors.Is.
var ErrLocked = errors.New("appendcodec: path already locked by another process")

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
// whose every completed Write is decodable by any reader the moment the call
// returns (the codec's Flush runs after each Write, so a concurrent reader —
// and a decoder of a file whose writer crashed before Close — always sees
// every acknowledged record; nothing fsyncs, so power-loss durability is the
// caller's concern), and whose Close writes the codec footer (gzip member
// trailer / zstd frame epilogue; nothing for CodecNone) and releases the
// descriptor.
//
// The open is link-safe and TOCTOU-free: safeio.OpenFileNoFollow refuses a
// symlink at the final path component (ELOOP, on every platform) and any
// non-regular pre-existing target instead of blocking, and
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
// Concurrency is scoped by codec:
//
//   - CodecGzip and CodecZstd: one path has exactly one live encoder —
//     OpenAppend holds an exclusive advisory lock on the file for the
//     descriptor's lifetime (flock on unix; a beyond-EOF LockFileEx range on
//     Windows, where region locks are mandatory — the range sits past any
//     reachable file size so content reads and writes stay unblocked) — and
//     concurrent encoders, whose frame write(2)s would interleave and
//     corrupt the stream, cannot exist. Within this process that constraint
//     is invisible: a second OpenAppend for the same path returns an
//     independent handle that queues its records through the one live
//     encoder (the mutex under the stream serializes each Write with its
//     Flush, so records never interleave), and handles are refcounted —
//     Close retires only its handle; the LAST Close writes the footer,
//     closes the descriptor and releases the lock. The joining open's own
//     parameters — windowLog included — are ignored: the first open's
//     encoder serves the stream for its lifetime. Opening the same live
//     path with a different codec fails with an error matching
//     ErrMagicMismatch. From another process the lock cannot be shared, so
//     the open fails with ErrLocked. Write after the last Close reports
//     ErrClosed; Close is idempotent.
//
//   - CodecNone: the compatibility codec, keeping the semantics this package
//     replaced — no advisory lock, no sharing registry, no queue. Every open
//     returns an independent plain O_APPEND handle, and any number of
//     writers — in this process or others — may hold the path at once; each
//     write(2) lands atomically at the then-current end of file, so records
//     of concurrent writers interleave only between write calls, never
//     inside one. The only retained guard is the magic sniff: plain bytes
//     are refused on a gzip/zstd file (and compressed codecs refuse plain
//     files) with ErrMagicMismatch — best-effort, because nothing
//     serializes the check against concurrent opens.
func OpenAppend(path string, codec Codec, windowLog int) (io.WriteCloser, error) {
	// Unknown codec values fail here, before any filesystem side effect —
	// no file is created, no lock is taken.
	switch codec {
	case CodecNone, CodecGzip, CodecZstd:
	default:
		return nil, fmt.Errorf("%w: %s", ErrUnknownCodec, codec)
	}

	key, err := filepath.Abs(path)
	if err != nil {
		return nil, fmt.Errorf("appendcodec: resolve %s: %w", path, err)
	}
	key = filepath.Clean(key)
	if runtime.GOOS == "windows" {
		// Windows paths are case-insensitive: fold the case so aliases such
		// as C:\Logs\A and c:\logs\a hit one registry entry instead of
		// degrading a same-process share to ErrLocked. Caveat: on a
		// case-sensitive directory (fsutil file setCaseSensitiveInfo, WSL
		// interop) two case-variant paths are distinct files that now share
		// one entry — the second open queues through the first file's
		// encoder instead of erroring (see ADR-010's Windows notes).
		key = strings.ToLower(key)
	}

	// CodecNone takes the compatibility path — no lock, no registry. See the
	// Concurrency section above.
	if codec == CodecNone {
		return openAppendNone(path)
	}

	// The registry lock is held across the whole open: concurrent first
	// opens of one compressed path serialize here, so the second finds the
	// entry the first created instead of racing it to the file lock.
	// OpenAppend runs once per stream, not per record, so the held lock
	// costs nothing.
	streamRegistry.mu.Lock()
	defer streamRegistry.mu.Unlock()

	if s, ok := streamRegistry.m[key]; ok {
		if s.codec != codec {
			return nil, fmt.Errorf("%w: %s is already open with codec %s, requested %s",
				ErrMagicMismatch, path, s.codec, codec)
		}
		s.refs++
		return &queuedWriter{s: s}, nil
	}

	w, err := openAppendFile(path, codec, windowLog)
	if err != nil {
		return nil, err
	}
	s := &sharedStream{appendWriter: w, codec: codec, key: key, refs: 1}
	streamRegistry.m[key] = s
	return &queuedWriter{s: s}, nil
}

// openAppendNone is the CodecNone open path: the pre-appendcodec semantics
// kept for compatibility. It takes no advisory lock and consults no sharing
// registry — every call returns an independent plain O_APPEND handle, and
// concurrent writers (in this process or another) append side by side, each
// write(2) atomic at the end of file. The one retained guard is the magic
// sniff read from the same descriptor: appending plain bytes onto a
// gzip/zstd stream would strand the existing members, so that mismatch
// still fails with ErrMagicMismatch. Without the lock the check is not
// serialized against concurrent opens, so it is best-effort; mixing codecs
// on one path is a caller error.
func openAppendNone(path string) (*appendWriter, error) {
	f, err := safeio.OpenFileNoFollow(path, os.O_RDWR|os.O_CREATE|os.O_APPEND, 0o644)
	if err != nil {
		return nil, err
	}
	prefix, err := readMagicPrefix(f, path)
	if err != nil {
		_ = f.Close()
		return nil, err
	}
	if err := checkAppendMagic(CodecNone, path, prefix); err != nil {
		_ = f.Close()
		return nil, err
	}
	return &appendWriter{f: f}, nil
}

// readMagicPrefix reads the leading magicLen bytes of f for the append
// compatibility check, tolerating a file shorter than the sniff window.
func readMagicPrefix(f *os.File, path string) ([]byte, error) {
	prefix := make([]byte, magicLen)
	n, rerr := f.ReadAt(prefix, 0)
	if rerr != nil && !errors.Is(rerr, io.EOF) && !errors.Is(rerr, io.ErrUnexpectedEOF) {
		return nil, &os.PathError{Op: "read", Path: path, Err: rerr}
	}
	return prefix[:n], nil
}

// sharedStream is the registry entry behind every handle OpenAppend hands
// out for one compressed path (gzip/zstd): the locked descriptor's
// appendWriter (its mutex is the write queue all handles share), the codec
// the stream was opened with, the normalized registry key, and the count of
// live handles.
type sharedStream struct {
	*appendWriter
	codec Codec
	key   string
	refs  int
}

// streamRegistry maps normalized absolute paths to this process's live
// compressed streams. It turns a same-process second OpenAppend for a
// gzip/zstd path into a queued handle instead of a lock conflict; paths that
// normalize differently (a symlinked parent directory aliases the same file
// under two keys) miss the registry and degrade to the advisory lock's
// ErrLocked — never to interleaved frames. CodecNone streams never enter
// the registry: the compatibility codec has no lock and no shared encoder.
var streamRegistry = struct {
	mu sync.Mutex
	m  map[string]*sharedStream
}{m: make(map[string]*sharedStream)}

// queuedWriter is one independent handle on a compressed sharedStream
// (gzip/zstd). Write queues behind the stream's mutex, so records written
// through different handles of one path never interleave; Close retires
// only this handle — the last Close writes the codec footer, closes the
// descriptor, releases the OS lock and drops the registry entry — and is
// idempotent per handle, so a double Close never counts a reference twice.
// Note the io.WriteCloser divergence this enables (see ErrClosed): after
// this handle's own Close, while a sibling keeps the stream alive, Write
// still succeeds and appends through the one encoder — ErrClosed surfaces
// only once the LAST handle's Close shut the stream down.
// CodecNone never produces a queuedWriter: its opens return independent
// plain handles.
type queuedWriter struct {
	s      *sharedStream
	closed atomic.Bool
}

func (q *queuedWriter) Write(p []byte) (int, error) { return q.s.Write(p) }

func (q *queuedWriter) Close() error {
	if !q.closed.CompareAndSwap(false, true) {
		return nil
	}
	// The registry lock is held across the whole teardown — entry drop and
	// descriptor close alike. Dropping the entry first and only then closing
	// (the footer write is real disk I/O) would leave the descriptor open
	// and locked but unlisted: a concurrent same-process OpenAppend for the
	// path would miss the registry, go fresh to the file and hit the
	// still-held advisory lock as a spurious ErrLocked — an error the
	// contract reserves for cross-process conflicts. Lock order is
	// registry.mu -> appendWriter.mu (Write and appendWriter.Close never
	// take registry.mu), so this cannot deadlock.
	streamRegistry.mu.Lock()
	defer streamRegistry.mu.Unlock()
	q.s.refs--
	if q.s.refs > 0 {
		return nil
	}
	if streamRegistry.m[q.s.key] == q.s {
		delete(streamRegistry.m, q.s.key)
	}
	return q.s.Close()
}

// openAppendFile is OpenAppend's locked path, used only for the compressed
// codecs (CodecGzip, CodecZstd — CodecNone takes the lock-free
// openAppendNone): open through safeio, take the advisory lock, verify the
// existing magic against the codec and build the encoder. The caller
// (OpenAppend) owns the registry bookkeeping.
func openAppendFile(path string, codec Codec, windowLog int) (*appendWriter, error) {
	if codec == CodecZstd && windowLog > 0 {
		windowLog = clampWindowLog(windowLog)
	}
	f, err := safeio.OpenFileNoFollow(path, os.O_RDWR|os.O_CREATE|os.O_APPEND, 0o644)
	if err != nil {
		return nil, err
	}
	if err := lockExclusive(f); err != nil {
		_ = f.Close()
		if isLockConflict(err) {
			return nil, fmt.Errorf("%w: %s", ErrLocked, path)
		}
		return nil, &os.PathError{Op: "lock", Path: path, Err: err}
	}

	// The path is locked: no other locked OpenAppend can write concurrently
	// while the prefix is inspected, so the content is stable for the check
	// below. (A concurrent CodecNone writer would not be locked out — the
	// compatibility codec takes no lock — so this check remains best-effort
	// against mixed-codec misuse.)
	prefix, rerr := readMagicPrefix(f, path)
	if rerr != nil {
		_ = f.Close()
		return nil, rerr
	}
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
