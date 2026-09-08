package llm

import (
	"context"
	"crypto/sha1" //nolint:gosec // the digest only names a local cache file, matching tiktoken-go's key format
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/pkoukk/tiktoken-go"
)

// tiktoken-go (v0.1.8) fetches BPE vocabularies with a bare http.Get that has
// no timeout (its load.go readFile): once a connection is established, a
// stalled transfer blocks forever, because neither TCP keepalive nor any
// application deadline applies to the read. On CI runners every leg starts
// with a cold cache (TIKTOKEN_CACHE_DIR unset, empty TempDir), so the llm
// package's token-counting tests always hit that unbounded path — a single
// stalled download wedges the whole test binary until the harness kills it.
// The same hazard applies to production hosts calling NewTiktokenCounter on
// a flaky network.
//
// boundedBpeLoader replicates tiktoken-go v0.1.8's default loader (cache
// directory selection, sha1-of-URL cache key, tmp-file+rename publication,
// line format "<base64-token> <rank>") but performs the download with a
// bounded HTTP client. Two deliberate deviations, both strictly safer:
//
//   - the response status is checked (upstream would try to parse an error
//     page and fail with a confusing decode message);
//   - the response body is size-capped, so a runaway download cannot write
//     gigabytes to disk within the timeout window.
//
// The package replaces the library's global loader in init, so every
// GetEncoding call in this process — from tests and hosts alike — is
// bounded. A host may still install its own loader afterwards via
// tiktoken.SetBpeLoader, which overrides ours (it runs later in program
// startup than any package init).
const (
	// boundedBPEHTTPTimeout bounds each vocabulary download, including
	// connection, TLS handshake, and the full body transfer.
	boundedBPEHTTPTimeout = 30 * time.Second

	// maxBPEFileBytes caps the downloaded vocabulary size. The largest known
	// public vocabulary (o200k_base) is ~3.6 MiB; 64 MiB leaves a wide margin
	// while still bounding disk writes on a misbehaving endpoint.
	maxBPEFileBytes = 64 << 20
)

// boundedBpeLoader is a tiktoken.BpeLoader whose network reads cannot hang.
// The zero value is not usable; construct it with newBoundedBpeLoader.
type boundedBpeLoader struct {
	client *http.Client
}

// newBoundedBpeLoader returns a loader whose downloads are bounded by d.
func newBoundedBpeLoader(d time.Duration) *boundedBpeLoader {
	return &boundedBpeLoader{
		client: &http.Client{Timeout: d},
	}
}

// init installs the bounded loader as tiktoken-go's global default, so the
// unbounded http.Get in the library's own loader is never reached. See the
// boundedBpeLoader doc comment for why this package overrides it globally.
func init() {
	tiktoken.SetBpeLoader(newBoundedBpeLoader(boundedBPEHTTPTimeout))
}

// LoadTiktokenBpe implements tiktoken.BpeLoader. tiktoken-go passes the full
// URL of a vocabulary file (e.g.
// https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken);
// non-URL values are treated as local file paths, mirroring the library.
func (l *boundedBpeLoader) LoadTiktokenBpe(tiktokenBpeFile string) (map[string]int, error) {
	if !strings.HasPrefix(tiktokenBpeFile, "http://") && !strings.HasPrefix(tiktokenBpeFile, "https://") {
		// Local path: read directly, no caching — same observable result as
		// the upstream loader, without touching the shared cache directory.
		contents, err := os.ReadFile(tiktokenBpeFile)
		if err != nil {
			return nil, fmt.Errorf("read tiktoken bpe %s: %w", tiktokenBpeFile, err)
		}
		return parseBPERanks(contents)
	}

	cachePath, err := bpeCachePath(tiktokenBpeFile)
	if err != nil {
		return nil, err
	}
	if contents, err := os.ReadFile(cachePath); err == nil {
		return parseBPERanks(contents)
	}

	contents, err := l.download(tiktokenBpeFile)
	if err != nil {
		return nil, err
	}
	if err := writeBPECache(cachePath, contents); err != nil {
		return nil, err
	}
	return parseBPERanks(contents)
}

// download fetches the vocabulary over HTTP with the loader's bounded client.
func (l *boundedBpeLoader) download(blobURL string) ([]byte, error) {
	req, err := http.NewRequestWithContext(context.Background(), http.MethodGet, blobURL, http.NoBody)
	if err != nil {
		return nil, fmt.Errorf("build tiktoken bpe request %s: %w", blobURL, err)
	}
	resp, err := l.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("fetch tiktoken bpe %s: %w", blobURL, err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("fetch tiktoken bpe %s: unexpected status %s", blobURL, resp.Status)
	}

	// Read one extra byte to distinguish "exactly maxBPEFileBytes" (fine)
	// from "more than maxBPEFileBytes" (reject).
	contents, err := io.ReadAll(io.LimitReader(resp.Body, maxBPEFileBytes+1))
	if err != nil {
		return nil, fmt.Errorf("fetch tiktoken bpe %s: %w", blobURL, err)
	}
	if len(contents) > maxBPEFileBytes {
		return nil, fmt.Errorf("fetch tiktoken bpe %s: response exceeds %d bytes", blobURL, maxBPEFileBytes)
	}
	return contents, nil
}

// bpeCachePath returns the on-disk cache path for a vocabulary URL, using the
// same directory selection and key format as tiktoken-go v0.1.8: the
// TIKTOKEN_CACHE_DIR or DATA_GYM_CACHE_DIR environment variable, falling back
// to <TempDir>/data-gym-cache, keyed by the hex sha1 of the URL.
func bpeCachePath(blobURL string) (string, error) {
	cacheDir := strings.TrimSpace(os.Getenv("TIKTOKEN_CACHE_DIR"))
	if cacheDir == "" {
		cacheDir = strings.TrimSpace(os.Getenv("DATA_GYM_CACHE_DIR"))
	}
	if cacheDir == "" {
		cacheDir = filepath.Join(os.TempDir(), "data-gym-cache")
	}
	key := fmt.Sprintf("%x", sha1.Sum([]byte(blobURL)))
	return filepath.Join(cacheDir, key), nil
}

// writeBPECache publishes contents at cachePath atomically: write to a
// temporary file in the same directory, close it, then rename over the
// target. The close-before-rename order is required on Windows, which refuses
// to rename an open file.
func writeBPECache(cachePath string, contents []byte) error {
	cacheDir := filepath.Dir(cachePath)
	if err := os.MkdirAll(cacheDir, 0o755); err != nil {
		return fmt.Errorf("create tiktoken cache dir: %w", err)
	}

	tmp, err := os.CreateTemp(cacheDir, filepath.Base(cachePath)+".*.tmp")
	if err != nil {
		return fmt.Errorf("create tiktoken cache temp file: %w", err)
	}
	tmpName := tmp.Name()
	defer func() {
		if tmpName != "" {
			_ = os.Remove(tmpName)
		}
	}()

	if _, err := tmp.Write(contents); err != nil {
		_ = tmp.Close()
		return fmt.Errorf("write tiktoken cache: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("write tiktoken cache: %w", err)
	}
	if err := os.Rename(tmpName, cachePath); err != nil {
		return fmt.Errorf("publish tiktoken cache: %w", err)
	}
	tmpName = "" // published; the deferred cleanup must not remove it
	return nil
}

// parseBPERanks decodes tiktoken's vocabulary line format: one entry per
// line, "<base64(token)> <rank>", blank lines ignored. Unlike the upstream
// parser it reports malformed lines (including lines without a rank) instead
// of panicking on an out-of-range index.
func parseBPERanks(contents []byte) (map[string]int, error) {
	bpeRanks := make(map[string]int)
	for _, line := range strings.Split(string(contents), "\n") {
		if line == "" {
			continue
		}
		parts := strings.Split(line, " ")
		if len(parts) != 2 {
			return nil, fmt.Errorf("malformed tiktoken bpe line %q: want \"<base64 token> <rank>\"", line)
		}
		token, err := base64.StdEncoding.DecodeString(parts[0])
		if err != nil {
			return nil, fmt.Errorf("malformed tiktoken bpe line %q: %w", line, err)
		}
		rank, err := strconv.Atoi(parts[1])
		if err != nil {
			return nil, fmt.Errorf("malformed tiktoken bpe line %q: %w", line, err)
		}
		bpeRanks[string(token)] = rank
	}
	if len(bpeRanks) == 0 {
		return nil, errors.New("empty tiktoken bpe vocabulary")
	}
	return bpeRanks, nil
}
