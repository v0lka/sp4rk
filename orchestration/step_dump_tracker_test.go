package orchestration

import (
	"bytes"
	"compress/gzip"
	"errors"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/klauspost/compress/zstd"

	"github.com/v0lka/sp4rk/appendcodec"
)

func TestStepDumpTracker_OpenAndClose(t *testing.T) {
	dir := t.TempDir()
	tracker := NewStepDumpTracker(dir, slog.Default())

	w1 := tracker.OpenStepDump("step-1")
	w2 := tracker.OpenStepDump("step-2")
	if w1 == nil {
		t.Fatal("expected non-nil writer for step-1")
	}
	if w2 == nil {
		t.Fatal("expected non-nil writer for step-2")
	}

	// Write to verify files are writable
	if _, err := io.WriteString(w1, "dump1\n"); err != nil {
		t.Fatalf("write step-1: %v", err)
	}
	if _, err := io.WriteString(w2, "dump2\n"); err != nil {
		t.Fatalf("write step-2: %v", err)
	}

	if err := tracker.CloseAll(); err != nil {
		t.Fatalf("CloseAll: %v", err)
	}

	// Verify files exist on disk
	data1, err := os.ReadFile(filepath.Join(dir, "step_step-1.jsonl"))
	if err != nil {
		t.Fatalf("read step-1 file: %v", err)
	}
	if !strings.Contains(string(data1), "dump1") {
		t.Errorf("step-1 file missing content: %q", string(data1))
	}
	data2, err := os.ReadFile(filepath.Join(dir, "step_step-2.jsonl"))
	if err != nil {
		t.Fatalf("read step-2 file: %v", err)
	}
	if !strings.Contains(string(data2), "dump2") {
		t.Errorf("step-2 file missing content: %q", string(data2))
	}
}

func TestStepDumpTracker_IdempotentOpen(t *testing.T) {
	dir := t.TempDir()
	tracker := NewStepDumpTracker(dir, slog.Default())
	// Close tracked files so Windows can remove the temp dir (open files are
	// locked on Windows, unlike Unix where they can be unlinked while open).
	t.Cleanup(func() { _ = tracker.CloseAll() })

	w1 := tracker.OpenStepDump("step-1")
	w2 := tracker.OpenStepDump("step-1")
	if w1 != w2 {
		t.Error("expected same writer for repeated OpenStepDump with same stepID")
	}
}

func TestStepDumpTracker_CloseAllIsIdempotent(t *testing.T) {
	dir := t.TempDir()
	tracker := NewStepDumpTracker(dir, slog.Default())

	_ = tracker.OpenStepDump("step-1")
	if err := tracker.CloseAll(); err != nil {
		t.Fatalf("first CloseAll: %v", err)
	}
	if err := tracker.CloseAll(); err != nil {
		t.Fatalf("second CloseAll: %v", err)
	}
}

func TestStepDumpTracker_OpenAfterClose(t *testing.T) {
	dir := t.TempDir()
	tracker := NewStepDumpTracker(dir, slog.Default())

	_ = tracker.OpenStepDump("step-1")
	_ = tracker.CloseAll()

	w := tracker.OpenStepDump("step-2")
	if w != nil {
		t.Error("expected nil after CloseAll")
	}
}

func TestStepDumpTracker_ConcurrentOpen(t *testing.T) {
	dir := t.TempDir()
	tracker := NewStepDumpTracker(dir, slog.Default())
	// Close tracked files so Windows can remove the temp dir (open files are
	// locked on Windows, unlike Unix where they can be unlinked while open).
	t.Cleanup(func() { _ = tracker.CloseAll() })

	const goroutines = 10
	var wg sync.WaitGroup
	writers := make([]io.Writer, goroutines)

	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			writers[idx] = tracker.OpenStepDump("step-shared")
		}(i)
	}
	wg.Wait()

	first := writers[0]
	for i := 1; i < goroutines; i++ {
		if writers[i] != first {
			t.Errorf("goroutine %d got different writer", i)
		}
	}
}

func TestStepDumpTracker_NonexistentDir(t *testing.T) {
	// Use a path that cannot be created (nested under a file)
	tmpDir := t.TempDir()
	filePath := filepath.Join(tmpDir, "not-a-dir")
	if err := os.WriteFile(filePath, []byte("block"), 0o644); err != nil {
		t.Fatalf("setup: %v", err)
	}
	badDir := filepath.Join(filePath, "sub", "deep")

	logger := expectedDiagnostics(t,
		expectedDiagnostic{message: "step_dump_tracker: failed to create directory", attrs: map[string]any{"dir": badDir, "error": func(v any) bool { err, ok := v.(*os.PathError); return ok && err.Op == "mkdir" }}},
		expectedDiagnostic{message: "step_dump_tracker: failed to open dump file", attrs: map[string]any{"filename": "step_step-1.jsonl", "error": func(v any) bool { err, ok := v.(*os.PathError); return ok && err.Op == "open" }}},
	)
	tracker := NewStepDumpTracker(badDir, logger)
	w := tracker.OpenStepDump("step-1")
	if w != nil {
		t.Error("expected nil when directory creation fails")
	}
}

// decodeGzip fully decodes a gzip stream. io.ReadAll reports
// ErrUnexpectedEOF when the stream is missing its member trailer, so a clean
// read to EOF doubles as a footer-on-close assertion.
func decodeGzip(t *testing.T, raw []byte) string {
	t.Helper()
	r, err := gzip.NewReader(bytes.NewReader(raw))
	if err != nil {
		t.Fatalf("gzip.NewReader: %v", err)
	}
	out, err := io.ReadAll(r)
	if err != nil {
		t.Fatalf("gzip read: %v", err)
	}
	return string(out)
}

// decodeZstd fully decodes a zstd frame stream.
func decodeZstd(t *testing.T, raw []byte) string {
	t.Helper()
	r, err := zstd.NewReader(bytes.NewReader(raw))
	if err != nil {
		t.Fatalf("zstd.NewReader: %v", err)
	}
	defer r.Close()
	out, err := io.ReadAll(r)
	if err != nil {
		t.Fatalf("zstd read: %v", err)
	}
	return string(out)
}

func TestStepDumpTracker_CodecFilename(t *testing.T) {
	tests := []struct {
		codec    appendcodec.Codec
		filename string
		decode   func(t *testing.T, raw []byte) string
	}{
		{
			codec:    appendcodec.CodecNone,
			filename: "step_step-1.jsonl",
			decode: func(t *testing.T, raw []byte) string {
				return string(raw)
			},
		},
		{codec: appendcodec.CodecGzip, filename: "step_step-1.jsonl.gz", decode: decodeGzip},
		{codec: appendcodec.CodecZstd, filename: "step_step-1.jsonl.zst", decode: decodeZstd},
	}
	for _, tt := range tests {
		t.Run(tt.codec.String(), func(t *testing.T) {
			dir := t.TempDir()
			tracker := NewStepDumpTracker(dir, slog.Default(), WithCodec(tt.codec))
			// Close tracked files so Windows can remove the temp dir (open
			// files are locked on Windows, unlike Unix where they can be
			// unlinked while open).
			t.Cleanup(func() { _ = tracker.CloseAll() })

			w := tracker.OpenStepDump("step-1")
			if w == nil {
				t.Fatal("expected non-nil writer")
			}
			if _, err := io.WriteString(w, "dump1\n"); err != nil {
				t.Fatalf("write: %v", err)
			}
			if err := tracker.CloseAll(); err != nil {
				t.Fatalf("CloseAll: %v", err)
			}

			// The codec names the file: only the codec-specific extension
			// exists, and its content decodes back to what was written.
			raw, err := os.ReadFile(filepath.Join(dir, tt.filename))
			if err != nil {
				t.Fatalf("read %s: %v", tt.filename, err)
			}
			if got := tt.decode(t, raw); got != "dump1\n" {
				t.Errorf("decoded content = %q, want %q", got, "dump1\n")
			}
		})
	}
}

func TestStepDumpTracker_CloseAllWritesFooter(t *testing.T) {
	// gzip: CloseAll must leave a complete multi-member-decodable stream —
	// decodeGzip's clean read to EOF fails with ErrUnexpectedEOF on a footer-
	// less file, so this pins the wrapper-close (footer) before file-release
	// ordering in CloseAll.
	dir := t.TempDir()
	tracker := NewStepDumpTracker(dir, slog.Default(), WithCodec(appendcodec.CodecGzip))
	w := tracker.OpenStepDump("step-1")
	if w == nil {
		t.Fatal("expected non-nil writer")
	}
	if _, err := io.WriteString(w, "payload\n"); err != nil {
		t.Fatalf("write: %v", err)
	}
	if err := tracker.CloseAll(); err != nil {
		t.Fatalf("CloseAll: %v", err)
	}
	raw, err := os.ReadFile(filepath.Join(dir, "step_step-1.jsonl.gz"))
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if got := decodeGzip(t, raw); got != "payload\n" {
		t.Errorf("decoded content = %q, want %q", got, "payload\n")
	}

	// Same property for zstd: the frames CloseAll leaves behind must decode
	// cleanly with the standard decoder.
	zdir := t.TempDir()
	ztracker := NewStepDumpTracker(zdir, slog.Default(), WithCodec(appendcodec.CodecZstd))
	zw := ztracker.OpenStepDump("step-1")
	if zw == nil {
		t.Fatal("expected non-nil writer")
	}
	if _, err := io.WriteString(zw, "payload\n"); err != nil {
		t.Fatalf("zstd write: %v", err)
	}
	if err := ztracker.CloseAll(); err != nil {
		t.Fatalf("zstd CloseAll: %v", err)
	}
	zraw, err := os.ReadFile(filepath.Join(zdir, "step_step-1.jsonl.zst"))
	if err != nil {
		t.Fatalf("zstd read: %v", err)
	}
	if got := decodeZstd(t, zraw); got != "payload\n" {
		t.Errorf("zstd decoded content = %q, want %q", got, "payload\n")
	}
}

func TestStepDumpTracker_CodecWriterCached(t *testing.T) {
	dir := t.TempDir()
	tracker := NewStepDumpTracker(dir, slog.Default(),
		WithCodec(appendcodec.CodecGzip), WithCodecWindowLog(10))
	// Close tracked files so Windows can remove the temp dir (open files are
	// locked on Windows, unlike Unix where they can be unlinked while open).
	t.Cleanup(func() { _ = tracker.CloseAll() })

	w1 := tracker.OpenStepDump("step-1")
	w2 := tracker.OpenStepDump("step-1")
	if w1 == nil || w2 == nil {
		t.Fatal("expected non-nil writers")
	}
	if w1 != w2 {
		t.Error("expected the same cached wrapper for repeated OpenStepDump")
	}

	// One cached encoder must frame the interleaved writes as a single
	// stream: two live encoders on one compressed path would corrupt it.
	if _, err := io.WriteString(w1, "a\n"); err != nil {
		t.Fatalf("write w1: %v", err)
	}
	if _, err := io.WriteString(w2, "b\n"); err != nil {
		t.Fatalf("write w2: %v", err)
	}
	if err := tracker.CloseAll(); err != nil {
		t.Fatalf("CloseAll: %v", err)
	}
	raw, err := os.ReadFile(filepath.Join(dir, "step_step-1.jsonl.gz"))
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if got := decodeGzip(t, raw); got != "a\nb\n" {
		t.Errorf("decoded content = %q, want %q", got, "a\nb\n")
	}
}

func TestStepDumpTracker_GzipAppendsToExistingFile(t *testing.T) {
	dir := t.TempDir()
	first := NewStepDumpTracker(dir, slog.Default(), WithCodec(appendcodec.CodecGzip))
	w := first.OpenStepDump("step-1")
	if w == nil {
		t.Fatal("expected non-nil writer")
	}
	if _, err := io.WriteString(w, "first\n"); err != nil {
		t.Fatalf("first write: %v", err)
	}
	if err := first.CloseAll(); err != nil {
		t.Fatalf("first CloseAll: %v", err)
	}

	// A fresh tracker over the same directory appends as a new gzip member
	// (rotation-style continuation), not an overwrite.
	second := NewStepDumpTracker(dir, slog.Default(), WithCodec(appendcodec.CodecGzip))
	// Close tracked files so Windows can remove the temp dir (open files are
	// locked on Windows, unlike Unix where they can be unlinked while open).
	t.Cleanup(func() { _ = second.CloseAll() })
	w2 := second.OpenStepDump("step-1")
	if w2 == nil {
		t.Fatal("expected non-nil writer")
	}
	if _, err := io.WriteString(w2, "second\n"); err != nil {
		t.Fatalf("second write: %v", err)
	}
	if err := second.CloseAll(); err != nil {
		t.Fatalf("second CloseAll: %v", err)
	}

	raw, err := os.ReadFile(filepath.Join(dir, "step_step-1.jsonl.gz"))
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if got := decodeGzip(t, raw); got != "first\nsecond\n" {
		t.Errorf("decoded content = %q, want %q", got, "first\nsecond\n")
	}
}

func TestStepDumpTracker_CodecMagicMismatchRefused(t *testing.T) {
	dir := t.TempDir()
	// Codec-specific filenames keep codecs from colliding, so a mismatch can
	// only arise from foreign content planted at the codec's own path (a
	// renamed leftover, a hand-copied file). Plant plain JSONL text at the
	// zstd dump path: the zstd tracker must refuse to open with exactly one
	// warning and leave the existing file untouched.
	target := filepath.Join(dir, "step_step-1.jsonl.zst")
	if err := os.WriteFile(target, []byte("{\"json\":true}\n"), 0o644); err != nil {
		t.Fatalf("setup: %v", err)
	}

	logger := expectedDiagnostics(t,
		expectedDiagnostic{
			message: "step_dump_tracker: failed to open dump file",
			attrs: map[string]any{
				"filename": "step_step-1.jsonl.zst",
				"error": func(v any) bool {
					err, ok := v.(error)
					return ok && errors.Is(err, appendcodec.ErrMagicMismatch)
				},
			},
		},
	)
	tracker := NewStepDumpTracker(dir, logger, WithCodec(appendcodec.CodecZstd))
	if w := tracker.OpenStepDump("step-1"); w != nil {
		t.Error("expected nil writer on codec/magic mismatch")
	}
	data, err := os.ReadFile(target)
	if err != nil {
		t.Fatalf("read existing dump: %v", err)
	}
	if string(data) != "{\"json\":true}\n" {
		t.Errorf("existing dump modified: %q", string(data))
	}
}
