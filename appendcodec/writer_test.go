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
	"path/filepath"
	"strings"
	"sync"
	"testing"

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
			aw, ok := w.(*appendWriter)
			if !ok {
				t.Fatalf("OpenAppend returned %T, want *appendWriter", w)
			}
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

func TestOpenAppendLocksPathAgainstSecondWriter(t *testing.T) {
	// The one-writer contract is enforced, not just documented: while one
	// writer is live, a second OpenAppend on the same path fails with
	// ErrLocked (cross-platform: flock on unix, a beyond-EOF LockFileEx
	// range on Windows); after Close the path is open again and the new
	// writer continues the stream rotation-style.
	for _, codec := range []Codec{CodecNone, CodecGzip, CodecZstd} {
		t.Run(codec.String(), func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "locked."+codec.String())
			w1, err := OpenAppend(path, codec, 10)
			if err != nil {
				t.Fatalf("OpenAppend 1: %v", err)
			}
			first := "held by writer one\n"
			if _, err := w1.Write([]byte(first)); err != nil {
				t.Fatalf("write 1: %v", err)
			}

			w2, err := OpenAppend(path, codec, 10)
			if err == nil {
				_ = w2.Close()
				t.Fatal("second OpenAppend on locked path: want ErrLocked, got nil")
			}
			if !errors.Is(err, ErrLocked) {
				t.Fatalf("error = %v, want ErrLocked", err)
			}

			// The refused open must not have disturbed the live writer.
			if _, err := w1.Write([]byte("still fine\n")); err != nil {
				t.Fatalf("write after refused open: %v", err)
			}

			requireClose(t, w1)
			w3, err := OpenAppend(path, codec, 10)
			if err != nil {
				t.Fatalf("OpenAppend after Close: %v", err)
			}
			second := "generation two\n"
			if _, err := w3.Write([]byte(second)); err != nil {
				t.Fatalf("write 3: %v", err)
			}
			requireClose(t, w3)

			if got := decodeFile(t, path, codec); got != first+"still fine\n"+second {
				t.Errorf("decoded = %q, want all three records", got)
			}
		})
	}
}
