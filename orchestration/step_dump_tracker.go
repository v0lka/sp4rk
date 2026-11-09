package orchestration

import (
	"errors"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"sync"

	"github.com/v0lka/sp4rk/appendcodec"
)

// StepDumpOption customises the compression codec the tracker applies to its
// dump files. The zero configuration is appendcodec.CodecNone — plain JSONL
// files, byte-identical to the pre-codec tracker behaviour.
type StepDumpOption func(*stepDumpCodecConfig)

// stepDumpCodecConfig carries the codec knobs resolved from StepDumpOptions.
type stepDumpCodecConfig struct {
	codec     appendcodec.Codec
	windowLog int
}

// WithCodec selects the compression codec for every dump file the tracker
// opens: appendcodec.CodecNone (plain JSONL, the default), CodecGzip or
// CodecZstd. The codec also extends the filename — step_<id>.jsonl,
// step_<id>.jsonl.gz or step_<id>.jsonl.zst respectively — so different
// codecs never collide on the same path. Any other value is rejected at
// construction: NewStepDumpTracker logs one warning and falls back to
// CodecNone.
func WithCodec(codec appendcodec.Codec) StepDumpOption {
	return func(c *stepDumpCodecConfig) { c.codec = codec }
}

// WithCodecWindowLog sizes the zstd encoder window as 1<<windowLog for
// CodecZstd dumps; appendcodec.OpenAppend clamps the value into
// [appendcodec.MinWindowLog, appendcodec.MaxWindowLog] and values <= 0 keep
// the encoder default. The option is ignored by CodecNone and CodecGzip.
func WithCodecWindowLog(windowLog int) StepDumpOption {
	return func(c *stepDumpCodecConfig) { c.windowLog = windowLog }
}

// StepDumpTracker manages per-step LLM dump files. It is safe for concurrent use.
type StepDumpTracker struct {
	mu        sync.Mutex
	dir       string
	codec     appendcodec.Codec
	windowLog int
	writers   map[string]io.WriteCloser
	closed    bool
	logger    *slog.Logger
}

// NewStepDumpTracker creates a tracker for per-step dump files.
// Creates dir via os.MkdirAll; logs a warning on failure but does not
// return an error (dumps are best-effort debugging aids). Without options
// the tracker writes plain JSONL (appendcodec.CodecNone); pass
// WithCodec/WithCodecWindowLog to compress the dumps.
func NewStepDumpTracker(dir string, logger *slog.Logger, opts ...StepDumpOption) *StepDumpTracker {
	if logger == nil {
		logger = slog.Default()
	}
	cfg := stepDumpCodecConfig{codec: appendcodec.CodecNone}
	for _, opt := range opts {
		if opt != nil {
			opt(&cfg)
		}
	}
	switch cfg.codec {
	case appendcodec.CodecNone, appendcodec.CodecGzip, appendcodec.CodecZstd:
	default:
		// An invalid codec would fail every OpenAppend with ErrUnknownCodec
		// after creating an empty dump file; degrade once, here, instead.
		logger.Warn("step_dump_tracker: unknown codec; falling back to plain JSONL",
			"codec", cfg.codec.String())
		cfg.codec = appendcodec.CodecNone
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		logger.Warn("step_dump_tracker: failed to create directory",
			"dir", dir, "error", err)
	}
	return &StepDumpTracker{
		dir:       dir,
		codec:     cfg.codec,
		windowLog: cfg.windowLog,
		writers:   make(map[string]io.WriteCloser),
		logger:    logger,
	}
}

// stepDumpFilename maps a stepID and codec to the dump file's base name: the
// stepID is reduced to its final path element so the file always stays
// within the tracker directory, and the codec appends its extension
// (.jsonl, .jsonl.gz, .jsonl.zst).
func stepDumpFilename(stepID string, codec appendcodec.Codec) string {
	name := "step_" + filepath.Base(stepID) + ".jsonl"
	switch codec {
	case appendcodec.CodecGzip:
		name += ".gz"
	case appendcodec.CodecZstd:
		name += ".zst"
	}
	return name
}

// OpenStepDump returns an io.Writer for the step's dump file — an
// appendcodec wrapper that appends codec-framed records (plain bytes for
// CodecNone) and whose eventual Close writes the codec footer. The file is
// created/opened on first call and the wrapper is cached for subsequent
// calls (idempotent — retries append to the same file through the same
// encoder; a second live encoder on the same compressed path would
// interleave frames and corrupt the stream).
// Returns nil if the tracker is disabled (empty dir), closed, or file
// creation fails.
func (t *StepDumpTracker) OpenStepDump(stepID string) io.Writer {
	if t.dir == "" {
		return nil
	}
	filename := stepDumpFilename(stepID, t.codec)

	t.mu.Lock()
	defer t.mu.Unlock()

	if t.closed {
		return nil
	}
	if w, ok := t.writers[filename]; ok {
		return w
	}
	// appendcodec.OpenAppend goes through safeio.OpenFileNoFollow, which ORs
	// O_NONBLOCK, refuses to follow a symlink at the final path component
	// (ELOOP), and fstat-checks the descriptor, so a FIFO or symlink planted
	// at the dump path is refused instead of blocking this write-open forever
	// (a write-open of a FIFO with no reader blocks inside open(2)). The
	// filename is already sanitized to a base name above, and OpenAppend
	// verifies the existing magic against the codec before appending.
	w, err := appendcodec.OpenAppend(filepath.Join(t.dir, filename), t.codec, t.windowLog)
	if err != nil {
		t.logger.Warn("step_dump_tracker: failed to open dump file",
			"filename", filename, "error", err)
		return nil
	}
	t.writers[filename] = w
	return w
}

// CloseAll closes all tracked dump writers. Each wrapper's Close first
// writes the codec footer (gzip member trailer / zstd frame epilogue;
// nothing for CodecNone) and then releases the underlying file descriptor,
// so closing the wrappers is closing the files. Idempotent — safe to call
// multiple times.
func (t *StepDumpTracker) CloseAll() error {
	t.mu.Lock()
	defer t.mu.Unlock()

	t.closed = true
	var errs []error
	for name, w := range t.writers {
		if err := w.Close(); err != nil {
			errs = append(errs, err)
		}
		delete(t.writers, name)
	}
	return errors.Join(errs...)
}
