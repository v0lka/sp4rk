package appendcodec

import (
	"bytes"
	"compress/gzip"
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/klauspost/compress/zstd"
)

// warnCapture is a slog handler that records Warn-level messages so tests can
// assert the exact expected diagnostics at their source, including payload
// attributes and call counts.
type warnCapture struct {
	mu      sync.Mutex
	records []slog.Record
}

func (h *warnCapture) Enabled(_ context.Context, level slog.Level) bool {
	return level >= slog.LevelWarn
}

func (h *warnCapture) Handle(_ context.Context, r slog.Record) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.records = append(h.records, r.Clone())
	return nil
}

func (h *warnCapture) WithAttrs([]slog.Attr) slog.Handler { return h }

func (h *warnCapture) WithGroup(string) slog.Handler { return h }

// warns returns the recorded Warn messages with their attribute sets.
func (h *warnCapture) warns() []string {
	h.mu.Lock()
	defer h.mu.Unlock()
	out := make([]string, 0, len(h.records))
	for _, r := range h.records {
		var sb strings.Builder
		sb.WriteString(r.Message)
		r.Attrs(func(a slog.Attr) bool {
			fmt.Fprintf(&sb, " %s=%v", a.Key, a.Value)
			return true
		})
		out = append(out, sb.String())
	}
	return out
}

func (h *warnCapture) len() int {
	h.mu.Lock()
	defer h.mu.Unlock()
	return len(h.records)
}

// writePlainFile writes raw bytes to path, failing the test on error.
func writePlainFile(t *testing.T, path, data string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(data), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

// writeGzipFile compresses data into a gzip file with a closed footer.
func writeGzipFile(t *testing.T, path string, data []byte) {
	t.Helper()
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o644)
	if err != nil {
		t.Fatalf("open %s: %v", path, err)
	}
	gw := gzip.NewWriter(f)
	if _, err := gw.Write(data); err != nil {
		t.Fatalf("gzip write: %v", err)
	}
	if err := gw.Close(); err != nil {
		t.Fatalf("gzip close: %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
}

// writeZstdFile compresses data into a zstd file with a closed footer.
func writeZstdFile(t *testing.T, path string, data []byte) {
	t.Helper()
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o644)
	if err != nil {
		t.Fatalf("open %s: %v", path, err)
	}
	enc, err := zstd.NewWriter(f, zstd.WithEncoderConcurrency(1))
	if err != nil {
		t.Fatalf("zstd encoder: %v", err)
	}
	if _, err := enc.Write(data); err != nil {
		t.Fatalf("zstd write: %v", err)
	}
	if err := enc.Close(); err != nil {
		t.Fatalf("zstd close: %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
}

// decodeFile decodes the full current content of path through codec,
// tolerating the io.ErrUnexpectedEOF a truncated (crash-without-Close) gzip
// member or zstd frame produces after yielding all decodable records.
func decodeFile(t *testing.T, path string, codec Codec) string {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	var out []byte
	switch codec {
	case CodecNone:
		out = raw
	case CodecGzip:
		zr, err := gzip.NewReader(bytes.NewReader(raw))
		if err != nil {
			t.Fatalf("gzip reader: %v", err)
		}
		out, err = io.ReadAll(zr)
		if err != nil && !errors.Is(err, io.ErrUnexpectedEOF) {
			t.Fatalf("gzip decode: %v", err)
		}
	case CodecZstd:
		zr, err := zstd.NewReader(bytes.NewReader(raw))
		if err != nil {
			t.Fatalf("zstd reader: %v", err)
		}
		defer zr.Close()
		out, err = io.ReadAll(zr)
		if err != nil && !errors.Is(err, io.ErrUnexpectedEOF) {
			t.Fatalf("zstd decode: %v", err)
		}
	default:
		t.Fatalf("unexpected codec %s", codec)
	}
	return string(out)
}

// requireClose closes w and fails the test on error; must be deferred/wired
// through t.Cleanup by every test that opens a writer.
func requireClose(t *testing.T, w io.WriteCloser) {
	t.Helper()
	if err := w.Close(); err != nil {
		t.Errorf("Close: %v", err)
	}
}

func TestMain(m *testing.M) {
	switch os.Getenv("APPENDCODEC_TEST_HELPER") {
	case "":
		os.Exit(m.Run())
	case "hold-open":
		// Re-executed test binary: hold an OpenAppend writer on the path
		// from APPENDCODEC_TEST_PATH with the codec from
		// APPENDCODEC_TEST_CODEC long enough for the parent process to
		// attempt its own open, then exit.
		crossProcessHelperMain()
	}
	os.Exit(0)
}

// crossProcessHelperMain is the hold-open helper body: open, write one
// record, hold the descriptor, write a final record, Close, exit. Any
// failure is reported on stderr with a non-zero exit code.
func crossProcessHelperMain() {
	path := os.Getenv("APPENDCODEC_TEST_PATH")
	codec, err := ParseCodec(os.Getenv("APPENDCODEC_TEST_CODEC"))
	if err != nil {
		fmt.Fprintf(os.Stderr, "helper: bad codec: %v\n", err)
		os.Exit(3)
	}
	w, err := OpenAppend(path, codec, 0)
	if err != nil {
		fmt.Fprintf(os.Stderr, "helper: open: %v\n", err)
		os.Exit(3)
	}
	if _, err := w.Write([]byte("from child\n")); err != nil {
		fmt.Fprintf(os.Stderr, "helper: write: %v\n", err)
		os.Exit(3)
	}
	time.Sleep(1500 * time.Millisecond)
	if _, err := w.Write([]byte("child late\n")); err != nil {
		fmt.Fprintf(os.Stderr, "helper: late write: %v\n", err)
		os.Exit(3)
	}
	if err := w.Close(); err != nil {
		fmt.Fprintf(os.Stderr, "helper: close: %v\n", err)
		os.Exit(3)
	}
	os.Exit(0)
}

func TestCodecString(t *testing.T) {
	for c, want := range map[Codec]string{
		CodecNone: "none", CodecGzip: "gzip", CodecZstd: "zstd", Codec(42): "codec(42)",
	} {
		if got := c.String(); got != want {
			t.Errorf("Codec(%d).String() = %q, want %q", int(c), got, want)
		}
	}
}

func TestParseCodec(t *testing.T) {
	for name, want := range map[string]Codec{"none": CodecNone, "gzip": CodecGzip, "zstd": CodecZstd} {
		got, err := ParseCodec(name)
		if err != nil {
			t.Fatalf("ParseCodec(%q): %v", name, err)
		}
		if got != want {
			t.Errorf("ParseCodec(%q) = %s, want %s", name, got, want)
		}
	}
	_, err := ParseCodec("xz")
	if !errors.Is(err, ErrUnknownCodec) {
		t.Errorf("ParseCodec(\"xz\") error = %v, want ErrUnknownCodec", err)
	}
}

func TestDetectMagic(t *testing.T) {
	dir := t.TempDir()
	payload := []byte("hello compressed world\n")

	cases := []struct {
		name    string
		file    string
		write   func(t *testing.T, path string)
		want    Codec
		wantN   int // expected Warn-record count
		wantSub string
	}{
		{name: "gzip", file: "a.gz", want: CodecGzip, write: func(t *testing.T, p string) { writeGzipFile(t, p, payload) }},
		{name: "zstd", file: "a.zst", want: CodecZstd, write: func(t *testing.T, p string) { writeZstdFile(t, p, payload) }},
		{name: "json", file: "a.jsonl", want: CodecNone, write: func(t *testing.T, p string) {
			writePlainFile(t, p, "{\"ts\":1,\"msg\":\"x\"}\n{\"ts\":2}\n")
		}},
		{name: "json-array", file: "a.json", want: CodecNone, write: func(t *testing.T, p string) {
			writePlainFile(t, p, "  [\n  1, 2\n]\n")
		}},
		{name: "json-bom", file: "a.bom.json", want: CodecNone, write: func(t *testing.T, p string) {
			writePlainFile(t, p, string([]byte{0xEF, 0xBB, 0xBF})+"{\"a\":1}")
		}},
		{name: "plain-text", file: "a.log", want: CodecNone, write: func(t *testing.T, p string) {
			writePlainFile(t, p, "plain log line\n")
		}},
		{name: "empty", file: "empty.log", want: CodecNone, write: func(t *testing.T, p string) {
			writePlainFile(t, p, "")
		}},
		{name: "unknown-magic", file: "a.bin", want: CodecNone, wantN: 1,
			wantSub: "unknown magic; treating as uncompressed",
			write: func(t *testing.T, p string) {
				writePlainFile(t, p, string([]byte{0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x01}))
			}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(dir, tc.file)
			tc.write(t, path)

			var capture warnCapture
			log := slog.New(&capture)
			got, err := DetectMagic(path, log)
			if err != nil {
				t.Fatalf("DetectMagic(%s): %v", path, err)
			}
			if got != tc.want {
				t.Errorf("DetectMagic(%s) = %s, want %s", path, got, tc.want)
			}
			if got := capture.len(); got != tc.wantN {
				t.Fatalf("warn records = %d, want %d (%v)", got, tc.wantN, capture.warns())
			}
			if tc.wantN > 0 {
				msg := capture.warns()[0]
				if !strings.Contains(msg, tc.wantSub) {
					t.Errorf("warn message %q missing %q", msg, tc.wantSub)
				}
				if !strings.Contains(msg, `path=`+path) {
					t.Errorf("warn message %q missing path attribute", msg)
				}
				if !strings.Contains(msg, `magic=de ad be ef 00 01`) {
					t.Errorf("warn message %q missing hex magic attribute", msg)
				}
			}
		})
	}
}

func TestDetectMagicMissingFile(t *testing.T) {
	_, err := DetectMagic(filepath.Join(t.TempDir(), "absent.log"), nil)
	if err == nil {
		t.Fatal("DetectMagic on missing file: want error, got nil")
	}
	if !errors.Is(err, os.ErrNotExist) {
		t.Errorf("error = %v, want os.ErrNotExist", err)
	}
}

func TestOpenAppendRoundtrip(t *testing.T) {
	for _, codec := range []Codec{CodecNone, CodecGzip, CodecZstd} {
		t.Run(codec.String(), func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "appended."+codec.String())
			records := []string{
				"first line\n",
				strings.Repeat("μ-emoji ✅ payload ", 400) + "\n", // multibyte, well past any single buffer
				"last line\n",
			}
			w, err := OpenAppend(path, codec, 10)
			if err != nil {
				t.Fatalf("OpenAppend: %v", err)
			}
			for _, r := range records {
				if n, err := w.Write([]byte(r)); err != nil || n != len(r) {
					t.Fatalf("Write: n=%d err=%v", n, err)
				}
			}
			requireClose(t, w)

			if got := decodeFile(t, path, codec); got != strings.Join(records, "") {
				t.Errorf("decoded content mismatch: %d bytes", len(got))
			}
		})
	}
}

func TestOpenAppendVisibleBeforeClose(t *testing.T) {
	for _, codec := range []Codec{CodecGzip, CodecZstd} {
		t.Run(codec.String(), func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "live."+codec.String())
			w, err := OpenAppend(path, codec, 10)
			if err != nil {
				t.Fatalf("OpenAppend: %v", err)
			}
			first := "visible-before-close line\n"
			if _, err := w.Write([]byte(first)); err != nil {
				t.Fatalf("Write: %v", err)
			}

			// The record must already decode from disk while the writer is
			// still open — this is the property rotation readers rely on.
			if got := decodeFile(t, path, codec); got != first {
				t.Errorf("pre-Close decode = %q, want %q", got, first)
			}

			second := "second line before close\n"
			if _, err := w.Write([]byte(second)); err != nil {
				t.Fatalf("Write: %v", err)
			}
			if got := decodeFile(t, path, codec); got != first+second {
				t.Errorf("pre-Close decode = %q, want both records", got)
			}

			requireClose(t, w)
			if got := decodeFile(t, path, codec); got != first+second {
				t.Errorf("post-Close decode = %q, want both records", got)
			}
		})
	}
}

func TestOpenAppendCrashWithoutClose(t *testing.T) {
	for _, codec := range []Codec{CodecGzip, CodecZstd} {
		t.Run(codec.String(), func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "crashed."+codec.String())
			w, err := OpenAppend(path, codec, 10)
			if err != nil {
				t.Fatalf("OpenAppend: %v", err)
			}
			want := &strings.Builder{}
			for i := 0; i < 5; i++ {
				rec := fmt.Sprintf("record %d — survives the crash\n", i)
				if _, err := w.Write([]byte(rec)); err != nil {
					t.Fatalf("Write: %v", err)
				}
				want.WriteString(rec)
			}
			// Simulate a crashed writer: process death closes the descriptor
			// without running the encoder's Close, so the codec footer is
			// never written. Drop the writer WITHOUT Close and release only
			// the raw fd — exactly the on-disk bytes a crash leaves behind.
			qw, ok := w.(*queuedWriter)
			if !ok {
				t.Fatalf("OpenAppend returned %T, want *queuedWriter", w)
			}
			aw := qw.s.appendWriter
			aw.mu.Lock()
			aw.closed = true
			fdErr := aw.f.Close()
			aw.mu.Unlock()
			if fdErr != nil {
				t.Fatalf("raw fd close: %v", fdErr)
			}

			// A decoder of the crashed file must recover every acknowledged
			// record despite the missing footer.
			if got := decodeFile(t, path, codec); got != want.String() {
				t.Errorf("crash decode = %q (%d bytes), want %q (%d bytes)", got, len(got), want.String(), want.Len())
			}
		})
	}
}

func TestOpenAppendConcurrentWriters(t *testing.T) {
	const (
		goroutines = 8
		linesEach  = 50
	)
	for _, codec := range []Codec{CodecNone, CodecGzip, CodecZstd} {
		t.Run(codec.String(), func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "concurrent."+codec.String())
			w, err := OpenAppend(path, codec, 10)
			if err != nil {
				t.Fatalf("OpenAppend: %v", err)
			}
			var wg sync.WaitGroup
			for g := 0; g < goroutines; g++ {
				wg.Add(1)
				go func(g int) {
					defer wg.Done()
					for i := 0; i < linesEach; i++ {
						line := fmt.Sprintf("writer-%02d line-%03d %s\n", g, i, strings.Repeat("x", g+1))
						if _, err := w.Write([]byte(line)); err != nil {
							t.Errorf("write from goroutine %d: %v", g, err)
							return
						}
					}
				}(g)
			}
			wg.Wait()
			requireClose(t, w)

			got := decodeFile(t, path, codec)
			lines := strings.Split(strings.TrimSuffix(got, "\n"), "\n")
			if len(lines) != goroutines*linesEach {
				t.Fatalf("decoded %d lines, want %d", len(lines), goroutines*linesEach)
			}
			seen := make(map[string]int, goroutines*linesEach)
			for _, l := range lines {
				if !strings.HasPrefix(l, "writer-") || !strings.Contains(l, " line-") {
					t.Fatalf("corrupted line %q — records were interleaved", l)
				}
				seen[l]++
			}
			for g := 0; g < goroutines; g++ {
				for i := 0; i < linesEach; i++ {
					line := fmt.Sprintf("writer-%02d line-%03d %s", g, i, strings.Repeat("x", g+1))
					if seen[line] != 1 {
						t.Errorf("line %q seen %d times, want exactly once", line, seen[line])
					}
				}
			}
		})
	}
}

func TestOpenAppendMagicMismatch(t *testing.T) {
	dir := t.TempDir()
	payload := []byte("body\n")

	type fixture struct {
		name  string
		build func(t *testing.T, path string)
	}
	fixtures := []fixture{
		{"plain", func(t *testing.T, p string) { writePlainFile(t, p, "{\"json\":true}\n") }},
		{"gzip", func(t *testing.T, p string) { writeGzipFile(t, p, payload) }},
		{"zstd", func(t *testing.T, p string) { writeZstdFile(t, p, payload) }},
		{"unknown", func(t *testing.T, p string) { writePlainFile(t, p, string([]byte{0xCA, 0xFE, 0xBA, 0xBE})) }},
	}

	// expected: codec onto pre-existing content kind → error ("" means OK).
	expect := map[string]map[Codec]string{
		"plain":   {CodecGzip: "err", CodecZstd: "err", CodecNone: ""},
		"gzip":    {CodecGzip: "", CodecZstd: "err", CodecNone: "err"},
		"zstd":    {CodecGzip: "err", CodecZstd: "", CodecNone: "err"},
		"unknown": {CodecGzip: "err", CodecZstd: "err", CodecNone: ""},
	}

	for _, fx := range fixtures {
		for codec, want := range expect[fx.name] {
			t.Run(fx.name+"+"+codec.String(), func(t *testing.T) {
				path := filepath.Join(dir, fx.name+"."+codec.String()+".out")
				fx.build(t, path)
				w, err := OpenAppend(path, codec, 10)
				if want == "" {
					if err != nil {
						t.Fatalf("OpenAppend: unexpected error %v", err)
					}
					requireClose(t, w)
					return
				}
				if err == nil {
					_ = w.Close()
					t.Fatal("OpenAppend: want mismatch error, got nil")
				}
				if !errors.Is(err, ErrMagicMismatch) {
					t.Fatalf("error = %v, want ErrMagicMismatch", err)
				}
				var mme *MagicMismatchError
				if !errors.As(err, &mme) {
					t.Fatalf("error %v is not *MagicMismatchError", err)
				}
				if mme.Path != path || mme.Codec != codec {
					t.Errorf("error fields: path=%q codec=%s, want %q %s", mme.Path, mme.Codec, path, codec)
				}
				if len(mme.Magic) == 0 {
					t.Errorf("error carries no magic bytes")
				}
			})
		}
	}
}

func TestOpenAppendRotationAppendsSameCodec(t *testing.T) {
	// The rotation scenario: reopen the same path with the detected codec and
	// append a second record — the file must decode as one continuous stream.
	for _, codec := range []Codec{CodecGzip, CodecZstd} {
		t.Run(codec.String(), func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "rotated."+codec.String())
			w, err := OpenAppend(path, codec, 10)
			if err != nil {
				t.Fatalf("OpenAppend 1: %v", err)
			}
			first := "generation one\n"
			if _, err := w.Write([]byte(first)); err != nil {
				t.Fatalf("Write: %v", err)
			}
			requireClose(t, w)

			got, err := DetectMagic(path, nil)
			if err != nil {
				t.Fatalf("DetectMagic: %v", err)
			}
			if got != codec {
				t.Fatalf("DetectMagic after generation one = %s, want %s", got, codec)
			}
			w, err = OpenAppend(path, got, 10)
			if err != nil {
				t.Fatalf("OpenAppend 2: %v", err)
			}
			second := "generation two\n"
			if _, err := w.Write([]byte(second)); err != nil {
				t.Fatalf("Write: %v", err)
			}
			requireClose(t, w)

			if got := decodeFile(t, path, codec); got != first+second {
				t.Errorf("decoded = %q, want concatenated generations", got)
			}
		})
	}
}

func TestOpenAppendZstdWindowLog(t *testing.T) {
	dir := t.TempDir()
	cases := []struct {
		name      string
		windowLog int
	}{
		{name: "min", windowLog: MinWindowLog},
		{name: "clamped-low", windowLog: 1},
		{name: "clamped-high", windowLog: 99},
		{name: "default", windowLog: 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(dir, "win-"+tc.name+".zst")
			w, err := OpenAppend(path, CodecZstd, tc.windowLog)
			if err != nil {
				t.Fatalf("OpenAppend(windowLog=%d): %v", tc.windowLog, err)
			}
			rec := "window fits\n"
			if _, err := w.Write([]byte(rec)); err != nil {
				t.Fatalf("Write: %v", err)
			}
			requireClose(t, w)
			if got := decodeFile(t, path, CodecZstd); got != rec {
				t.Errorf("decode = %q, want %q", got, rec)
			}
		})
	}
	if clampWindowLog(1) != MinWindowLog || clampWindowLog(99) != MaxWindowLog || clampWindowLog(20) != 20 {
		t.Errorf("clampWindowLog bounds wrong")
	}
}

func TestAppendWriterClosedSemantics(t *testing.T) {
	path := filepath.Join(t.TempDir(), "closed.log")
	w, err := OpenAppend(path, CodecGzip, 0)
	if err != nil {
		t.Fatalf("OpenAppend: %v", err)
	}
	requireClose(t, w)

	if _, err := w.Write([]byte("x")); !errors.Is(err, ErrClosed) {
		t.Errorf("Write after Close error = %v, want ErrClosed", err)
	}
	if err := w.Close(); err != nil {
		t.Errorf("second Close = %v, want nil (idempotent)", err)
	}
}

func TestOpenAppendTruncatedMagicRefused(t *testing.T) {
	// A file cut off exactly at the container magic bytes is readable by no
	// sequential decoder, so appending that codec after the magic-only head
	// would produce members no decoder can reach. The sniff must refuse to
	// classify the truncated magic as its codec, and OpenAppend must fail
	// with ErrMagicMismatch instead of writing the unreachable data.
	dir := t.TempDir()
	cases := []struct {
		name  string
		magic []byte
		codec Codec
	}{
		{name: "gzip", magic: []byte{0x1f, 0x8b}, codec: CodecGzip},
		{name: "zstd", magic: []byte{0x28, 0xb5, 0x2f, 0xfd}, codec: CodecZstd},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(dir, "truncated-"+tc.name)
			if err := os.WriteFile(path, tc.magic, 0o644); err != nil {
				t.Fatalf("write magic-only file: %v", err)
			}

			got, err := DetectMagic(path, nil)
			if err != nil {
				t.Fatalf("DetectMagic: %v", err)
			}
			if got == tc.codec {
				t.Errorf("DetectMagic classified truncated %s magic as %s", tc.name, got)
			}

			w, err := OpenAppend(path, tc.codec, 0)
			if err == nil {
				_ = w.Close()
				t.Fatal("OpenAppend onto magic-only file: want error, got nil")
			}
			if !errors.Is(err, ErrMagicMismatch) {
				t.Fatalf("error = %v, want ErrMagicMismatch", err)
			}
			var mme *MagicMismatchError
			if !errors.As(err, &mme) {
				t.Fatalf("error %v is not *MagicMismatchError", err)
			}
			if !bytes.Equal(mme.Magic, tc.magic) {
				t.Errorf("error magic = % x, want % x", mme.Magic, tc.magic)
			}
		})
	}
}

func TestOpenAppendSecondWriterSharesStream(t *testing.T) {
	// The one-encoder contract is invisible within a process: a second
	// OpenAppend on a live path succeeds and returns an independent handle
	// that queues its records through the first writer's encoder; handles
	// are refcounted, so the first Close only retires its handle and the
	// last Close writes the footer and releases the lock. After the last
	// Close the path is open again and the new writer continues the stream
	// rotation-style. Records from both handles stay in write order and
	// decodable end to end (gzip multi-member / zstd multi-frame).
	for _, codec := range []Codec{CodecNone, CodecGzip, CodecZstd} {
		t.Run(codec.String(), func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "shared."+codec.String())
			w1, err := OpenAppend(path, codec, 10)
			if err != nil {
				t.Fatalf("OpenAppend 1: %v", err)
			}
			a := "written by handle one\n"
			if _, err := w1.Write([]byte(a)); err != nil {
				t.Fatalf("write 1: %v", err)
			}

			w2, err := OpenAppend(path, codec, 10)
			if err != nil {
				t.Fatalf("second OpenAppend on live path: %v", err)
			}
			b := "queued by handle two\n"
			if _, err := w2.Write([]byte(b)); err != nil {
				t.Fatalf("write 2: %v", err)
			}
			c := "still fine\n"
			if _, err := w1.Write([]byte(c)); err != nil {
				t.Fatalf("write after second open: %v", err)
			}

			// The first Close only retires its handle: handle two keeps
			// writing through the shared encoder.
			requireClose(t, w1)
			d := "after first close\n"
			if _, err := w2.Write([]byte(d)); err != nil {
				t.Fatalf("write after first close: %v", err)
			}

			// The last Close writes the footer and releases the lock.
			requireClose(t, w2)
			if _, err := w2.Write([]byte("late\n")); !errors.Is(err, ErrClosed) {
				t.Fatalf("write after last close: err = %v, want ErrClosed", err)
			}

			w3, err := OpenAppend(path, codec, 10)
			if err != nil {
				t.Fatalf("OpenAppend after last Close: %v", err)
			}
			e := "generation two\n"
			if _, err := w3.Write([]byte(e)); err != nil {
				t.Fatalf("write 3: %v", err)
			}
			requireClose(t, w3)

			want := a + b + c + d + e
			if got := decodeFile(t, path, codec); got != want {
				t.Errorf("decoded = %q, want %q", got, want)
			}
		})
	}
}

func TestQueuedWriterCloseIdempotentPerHandle(t *testing.T) {
	// A double Close of ONE handle retires its reference exactly once: the
	// sibling handle keeps writing and the shared stream outlives the
	// misuse. Without the per-handle guard the extra decrement would retire
	// the sibling's reference and, on the last live one, tear down the
	// still-shared stream under it. CodecNone has no queued handles — its
	// *appendWriter is already idempotent — and is included for uniform
	// coverage.
	for _, codec := range []Codec{CodecNone, CodecGzip, CodecZstd} {
		t.Run(codec.String(), func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "double-close."+codec.String())
			w1, err := OpenAppend(path, codec, 10)
			if err != nil {
				t.Fatalf("OpenAppend 1: %v", err)
			}
			a := "before misuse\n"
			if _, err := w1.Write([]byte(a)); err != nil {
				t.Fatalf("write 1: %v", err)
			}
			w2, err := OpenAppend(path, codec, 10)
			if err != nil {
				t.Fatalf("OpenAppend 2: %v", err)
			}

			// The misuse: handle one is closed twice; the second Close
			// reports nil and counts nothing.
			requireClose(t, w1)
			requireClose(t, w1)

			// The sibling must be unaffected and keep writing.
			b := "sibling survives\n"
			if _, err := w2.Write([]byte(b)); err != nil {
				t.Fatalf("write after sibling double close: %v", err)
			}
			requireClose(t, w2)

			want := a + b
			if got := decodeFile(t, path, codec); got != want {
				t.Errorf("decoded = %q, want %q", got, want)
			}
		})
	}
}

func TestOpenAppendSharedPathCodecConflict(t *testing.T) {
	// A second open of a LIVE path with a different codec would append into
	// the wrong container; it fails with ErrMagicMismatch and leaves the
	// live writer undisturbed.
	path := filepath.Join(t.TempDir(), "conflict.gz")
	w, err := OpenAppend(path, CodecGzip, 0)
	if err != nil {
		t.Fatalf("OpenAppend gzip: %v", err)
	}
	if _, err := w.Write([]byte("kept\n")); err != nil {
		t.Fatalf("write: %v", err)
	}
	for _, codec := range []Codec{CodecZstd, CodecNone} {
		_, err := OpenAppend(path, codec, 0)
		if !errors.Is(err, ErrMagicMismatch) {
			_ = w.Close()
			t.Fatalf("OpenAppend %s on live gzip path: err = %v, want ErrMagicMismatch", codec, err)
		}
	}
	requireClose(t, w)
	if got := decodeFile(t, path, CodecGzip); got != "kept\n" {
		t.Errorf("decoded = %q, want untouched stream", got)
	}
}

// TestOpenAppendCrossProcess pins the cross-process half of the concurrency
// contract by re-executing the test binary as a hold-open helper. With
// CodecNone — the compatibility codec — no advisory lock exists, so the
// parent's concurrent open succeeds and both writers land their records in
// the file (the pre-appendcodec behavior). With CodecGzip the helper holds
// the lock, so the parent's open fails with ErrLocked and leaves the
// helper's stream untouched.
func TestOpenAppendCrossProcess(t *testing.T) {
	for _, tc := range []struct {
		name    string
		codec   Codec
		wantErr error
	}{
		{name: "none-no-lock", codec: CodecNone},
		{name: "gzip-locked", codec: CodecGzip, wantErr: ErrLocked},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "xproc."+tc.codec.String())
			ctx, cancel := context.WithCancel(context.Background())
			t.Cleanup(cancel)
			cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^$") // re-executes this test binary itself
			cmd.Env = append(os.Environ(),
				"APPENDCODEC_TEST_HELPER=hold-open",
				"APPENDCODEC_TEST_PATH="+path,
				"APPENDCODEC_TEST_CODEC="+tc.codec.String(),
			)
			var stderr bytes.Buffer
			cmd.Stderr = &stderr
			if err := cmd.Start(); err != nil {
				t.Fatalf("start helper: %v", err)
			}
			t.Cleanup(func() {
				if cmd.Process != nil {
					_ = cmd.Process.Kill()
				}
			})

			// Wait until the helper's first record reaches the file: from
			// then on its writer is definitely open and holding whatever
			// lock its codec takes. (Size, not content — a compressed
			// record does not contain its plaintext.)
			deadline := time.Now().Add(10 * time.Second)
			for {
				info, serr := os.Stat(path)
				if serr == nil && info.Size() > 0 {
					break
				}
				if time.Now().After(deadline) {
					t.Fatalf("helper produced no output in time; stderr: %s", stderr.String())
				}
				time.Sleep(20 * time.Millisecond)
			}

			w, err := OpenAppend(path, tc.codec, 0)
			if tc.wantErr != nil {
				if !errors.Is(err, tc.wantErr) {
					t.Fatalf("OpenAppend while helper holds path: err = %v, want %v", err, tc.wantErr)
				}
			} else {
				if err != nil {
					t.Fatalf("OpenAppend while helper holds path: %v (compat codec must not lock)", err)
				}
				if _, err := w.Write([]byte("from parent\n")); err != nil {
					t.Fatalf("parent write: %v", err)
				}
				if err := w.Close(); err != nil {
					t.Fatalf("parent close: %v", err)
				}
			}

			if err := cmd.Wait(); err != nil {
				t.Fatalf("helper exited %v; stderr: %s", err, stderr.String())
			}

			if tc.wantErr != nil {
				// The refused open must not have disturbed the helper's
				// stream.
				if got := decodeFile(t, path, tc.codec); got != "from child\nchild late\n" {
					t.Errorf("decoded = %q, want only the helper's records", got)
				}
				return
			}
			data, rerr := os.ReadFile(path)
			if rerr != nil {
				t.Fatalf("read: %v", rerr)
			}
			// Both records must be present and intact; the order between
			// the two independent writers is not deterministic.
			for _, want := range []string{"from child\n", "child late\n", "from parent\n"} {
				if !bytes.Contains(data, []byte(want)) {
					t.Errorf("file %q missing record %q", data, want)
				}
			}
		})
	}
}

// TestOpenAppendNoneIndependentHandles pins the compatibility behavior of
// CodecNone: two opens of one path return INDEPENDENT plain handles — not
// queued through a shared encoder, not refcounted, no registry entry — and
// concurrent writers through different handles interleave only between
// write calls, never inside one.
func TestOpenAppendNoneIndependentHandles(t *testing.T) {
	path := filepath.Join(t.TempDir(), "independent.log")
	w1, err := OpenAppend(path, CodecNone, 0)
	if err != nil {
		t.Fatalf("OpenAppend 1: %v", err)
	}
	w2, err := OpenAppend(path, CodecNone, 0)
	if err != nil {
		t.Fatalf("OpenAppend 2 on live path: %v (compat mode must not fail)", err)
	}
	if _, ok := w1.(*appendWriter); !ok {
		t.Errorf("CodecNone open returned %T, want *appendWriter (independent handle)", w1)
	}
	if _, ok := w2.(*appendWriter); !ok {
		t.Errorf("CodecNone second open returned %T, want *appendWriter (independent handle)", w2)
	}
	streamRegistry.mu.Lock()
	_, inRegistry := streamRegistry.m[path]
	streamRegistry.mu.Unlock()
	if inRegistry {
		t.Errorf("CodecNone registered its path in the shared-stream registry, want none (compat mode has no shared encoder)")
	}

	// Both handles write concurrently; O_APPEND keeps every write(2) intact.
	const linesEach = 25
	var wg sync.WaitGroup
	wg.Add(2)
	for _, hw := range []struct {
		handle io.WriteCloser
		tag    string
	}{{w1, "h1"}, {w2, "h2"}} {
		go func(tag string, w io.WriteCloser) {
			defer wg.Done()
			for i := 0; i < linesEach; i++ {
				if _, err := fmt.Fprintf(w, "%s line %d\n", tag, i); err != nil {
					t.Errorf("write %s: %v", tag, err)
					return
				}
			}
		}(hw.tag, hw.handle)
	}
	wg.Wait()

	// Closing one handle must not affect the other: independent fds, no
	// shared refcount.
	requireClose(t, w1)
	if _, err := w2.Write([]byte("h2 after h1 close\n")); err != nil {
		t.Fatalf("write after first close: %v", err)
	}
	requireClose(t, w2)

	data, rerr := os.ReadFile(path)
	if rerr != nil {
		t.Fatalf("read: %v", rerr)
	}
	lines := strings.Split(strings.TrimSuffix(string(data), "\n"), "\n")
	if len(lines) != 2*linesEach+1 {
		t.Fatalf("decoded %d lines, want %d", len(lines), 2*linesEach+1)
	}
	counts := map[string]int{"h1": 0, "h2": 0}
	for _, l := range lines {
		switch {
		case strings.HasPrefix(l, "h1 line "):
			counts["h1"]++
		case strings.HasPrefix(l, "h2"):
			counts["h2"]++
		default:
			t.Fatalf("corrupted line %q — records were interleaved", l)
		}
	}
	if counts["h1"] != linesEach || counts["h2"] != linesEach+1 {
		t.Errorf("line counts h1=%d h2=%d, want %d/%d", counts["h1"], counts["h2"], linesEach, linesEach+1)
	}
}
