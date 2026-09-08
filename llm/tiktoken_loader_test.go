package llm

import (
	"errors"
	"io/fs"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/pkoukk/tiktoken-go"
)

// failingRoundTripper fails every request, proving a code path performs no
// network I/O: if the loader under test reached the transport, the error
// surfaces (or the test fails) instead of silently passing.
type failingRoundTripper struct{}

func (failingRoundTripper) RoundTrip(*http.Request) (*http.Response, error) {
	return nil, errors.New("network access is not allowed in this test")
}

// newFailingBpeLoader returns a loader that must never touch the network.
func newFailingBpeLoader() *boundedBpeLoader {
	return &boundedBpeLoader{client: &http.Client{Transport: failingRoundTripper{}}}
}

// bpeFixture is a tiny vocabulary in tiktoken's on-disk format:
// "<base64(token)> <rank>" per line. Decodings: IQ== → "!", Ig== → "\"",
// AQID → "\x01\x02\x03" (binary token), aGVsbG8= → "hello".
const bpeFixture = "IQ== 0\n" +
	"Ig== 1\n" +
	"AQID 7\n" +
	"aGVsbG8= 42\n"

func writeBPEFixture(t *testing.T, dir, name, content string) string {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	return path
}

func TestParseBPERanks(t *testing.T) {
	t.Parallel()

	tests := map[string]struct {
		content string
		want    map[string]int
		wantErr string
	}{
		"valid vocabulary":         {content: bpeFixture, want: map[string]int{"!": 0, "\"": 1, "\x01\x02\x03": 7, "hello": 42}},
		"trailing newline is fine": {content: bpeFixture + "\n", want: map[string]int{"!": 0, "\"": 1, "\x01\x02\x03": 7, "hello": 42}},
		"blank lines ignored":      {content: "\n\n" + bpeFixture + "\n\n", want: map[string]int{"!": 0, "\"": 1, "\x01\x02\x03": 7, "hello": 42}},
		"only blank lines":         {content: "\n\n", wantErr: "empty tiktoken bpe vocabulary"},
		"empty content":            {content: "", wantErr: "empty tiktoken bpe vocabulary"},
		"missing rank":             {content: "aGVsbG8=\n", wantErr: `malformed tiktoken bpe line "aGVsbG8="`},
		"extra column":             {content: "aGVsbG8= 1 2\n", wantErr: `malformed tiktoken bpe line "aGVsbG8= 1 2"`},
		"invalid base64":           {content: "!!! 1\n", wantErr: "malformed tiktoken bpe line"},
		"invalid rank":             {content: "aGVsbG8= ten\n", wantErr: "malformed tiktoken bpe line"},
	}

	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			got, err := parseBPERanks([]byte(tt.content))
			if tt.wantErr != "" {
				if err == nil {
					t.Fatalf("parseBPERanks() error = nil, want %q", tt.wantErr)
				}
				if !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("parseBPERanks() error = %q, want containing %q", err.Error(), tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("parseBPERanks() unexpected error: %v", err)
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("parseBPERanks() = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestBoundedBpeLoader_ParityWithUpstream verifies the bounded loader parses
// a local vocabulary file exactly like tiktoken-go's default loader. Both
// read local paths without network access, so the comparison is hermetic.
func TestBoundedBpeLoader_ParityWithUpstream(t *testing.T) {
	t.Setenv("TIKTOKEN_CACHE_DIR", t.TempDir()) // keep the upstream loader's cache writes out of the real TempDir

	fixture := writeBPEFixture(t, t.TempDir(), "fixture.tiktoken", bpeFixture)

	upstream, err := tiktoken.NewDefaultBpeLoader().LoadTiktokenBpe(fixture)
	if err != nil {
		t.Fatalf("upstream loader: %v", err)
	}

	got, err := newFailingBpeLoader().LoadTiktokenBpe(fixture)
	if err != nil {
		t.Fatalf("bounded loader: %v", err)
	}

	if !reflect.DeepEqual(got, upstream) {
		t.Errorf("bounded loader result differs from upstream:\n got %v\nwant %v", got, upstream)
	}
}

func TestBoundedBpeLoader_CacheHitSkipsHTTP(t *testing.T) {
	t.Setenv("TIKTOKEN_CACHE_DIR", t.TempDir())

	const blobURL = "https://openaipublic.blob.core.windows.net/encodings/fake.tiktoken"
	cachePath, err := bpeCachePath(blobURL)
	if err != nil {
		t.Fatalf("bpeCachePath: %v", err)
	}
	if err := os.MkdirAll(filepath.Dir(cachePath), 0o755); err != nil {
		t.Fatalf("mkdir cache dir: %v", err)
	}
	if err := os.WriteFile(cachePath, []byte(bpeFixture), 0o644); err != nil {
		t.Fatalf("seed cache: %v", err)
	}

	got, err := newFailingBpeLoader().LoadTiktokenBpe(blobURL)
	if err != nil {
		t.Fatalf("LoadTiktokenBpe(cache hit): %v", err)
	}
	want := map[string]int{"!": 0, "\"": 1, "\x01\x02\x03": 7, "hello": 42}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("cached load = %v, want %v", got, want)
	}
}

func TestBoundedBpeLoader_DownloadCachesAndServes(t *testing.T) {
	t.Setenv("TIKTOKEN_CACHE_DIR", t.TempDir())

	var hits int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		hits++
		w.Header().Set("Content-Type", "application/octet-stream")
		_, _ = w.Write([]byte(bpeFixture))
	}))
	defer srv.Close()

	blobURL := srv.URL + "/encodings/fake.tiktoken"
	want := map[string]int{"!": 0, "\"": 1, "\x01\x02\x03": 7, "hello": 42}

	// First load: downloads from the server and populates the cache.
	got, err := (&boundedBpeLoader{client: srv.Client()}).LoadTiktokenBpe(blobURL)
	if err != nil {
		t.Fatalf("first load (download): %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("downloaded load = %v, want %v", got, want)
	}
	if hits != 1 {
		t.Fatalf("server hits after first load = %d, want 1", hits)
	}

	cachePath, err := bpeCachePath(blobURL)
	if err != nil {
		t.Fatalf("bpeCachePath: %v", err)
	}
	if _, err := os.Stat(cachePath); err != nil {
		t.Fatalf("cache file must exist after download: %v", err)
	}

	// Second load: served from cache even with the network cut off.
	got, err = newFailingBpeLoader().LoadTiktokenBpe(blobURL)
	if err != nil {
		t.Fatalf("second load (cache): %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("cached load = %v, want %v", got, want)
	}
	if hits != 1 {
		t.Errorf("server hits after cached load = %d, want 1 (no re-download)", hits)
	}
}

func TestBoundedBpeLoader_HTTPErrorStatus(t *testing.T) {
	t.Setenv("TIKTOKEN_CACHE_DIR", t.TempDir())

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "not found", http.StatusNotFound)
	}))
	defer srv.Close()

	_, err := (&boundedBpeLoader{client: srv.Client()}).LoadTiktokenBpe(srv.URL + "/missing.tiktoken")
	if err == nil {
		t.Fatal("expected an error for a 404 vocabulary response, got nil")
	}
	if !strings.Contains(err.Error(), "404") {
		t.Errorf("error %q should mention the HTTP status 404", err.Error())
	}
}

// TestBoundedBpeLoader_HTTPTimeout pins the property that motivated the
// loader: a server that accepts the connection but never sends data must
// produce an error within the client timeout, not hang forever. The handler
// parks on the request context, which the client timeout cancels — the test
// contains no sleeps and asserts a wall-clock ceiling instead of trusting
// the error alone.
func TestBoundedBpeLoader_HTTPTimeout(t *testing.T) {
	t.Setenv("TIKTOKEN_CACHE_DIR", t.TempDir())

	srv := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
		<-r.Context().Done() // stall until the client's timeout cancels the request
	}))
	defer srv.Close()

	loader := &boundedBpeLoader{client: &http.Client{Timeout: 100 * time.Millisecond}}

	start := time.Now()
	_, err := loader.LoadTiktokenBpe(srv.URL + "/stalled.tiktoken")
	elapsed := time.Since(start)

	if err == nil {
		t.Fatal("expected a timeout error from a stalled download, got nil")
	}
	if elapsed > 2*time.Second {
		t.Errorf("stalled download returned after %v, want an error within ~100ms (bounded), took %v", elapsed, elapsed)
	}
}

// TestBoundedBpeLoader_LocalPathErrors covers the local-file branch error
// path, which the parity test only exercises on the happy path.
func TestBoundedBpeLoader_LocalPathErrors(t *testing.T) {
	t.Parallel()

	_, err := newFailingBpeLoader().LoadTiktokenBpe(filepath.Join(t.TempDir(), "missing.tiktoken"))
	if err == nil {
		t.Fatal("expected an error for a missing local vocabulary file, got nil")
	}
	if !errors.Is(err, fs.ErrNotExist) {
		t.Errorf("error should wrap fs.ErrNotExist, got: %v", err)
	}
}
