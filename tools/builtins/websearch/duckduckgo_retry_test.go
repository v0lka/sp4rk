package websearch

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

// ddgAnomalyPage is the anti-bot challenge page DuckDuckGo serves behind a
// 202 (and occasionally a 200); the signature below is the marker the
// provider fails loudly on.
const ddgAnomalyPage = `<!DOCTYPE html>
<html><head><title> unfortunately, bots use DuckDuckGo too</title></head>
<body><form action="//duckduckgo.com/anomaly.js?a=b">
Please complete the following challenge… <img alt="CAPTCHA">
</form></body></html>`

// waitRecorder is a test wait function: it records every requested duration
// instead of sleeping (virtual time — the production waitCtx sleeps for
// real). It also fails a test that tries to wait at all unless arming is
// allowed, so a regression that reintroduces real sleeps is impossible.
type waitRecorder struct {
	mu    sync.Mutex
	waits []time.Duration
}

func (w *waitRecorder) wait(_ context.Context, d time.Duration) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.waits = append(w.waits, d)
	return nil
}

func (w *waitRecorder) recorded() []time.Duration {
	w.mu.Lock()
	defer w.mu.Unlock()
	return append([]time.Duration(nil), w.waits...)
}

// newSilentProvider returns a provider whose waits are recorded rather than
// slept and whose jitter is pinned to zero (the request pacer shares the
// provider's jitter seam), so pacing and backoff durations are deterministic.
func newSilentProvider(timeout time.Duration, rec *waitRecorder) *DuckDuckGoProvider {
	p := NewDuckDuckGoProviderWithClient(timeout, nil)
	p.wait = rec.wait
	p.jitter = func() float64 { return 0 }
	return p
}

// TestDuckDuckGoProvider_Anomaly202FailsLoudly pins the correctness bug
// behind the silent-empty-results reports: HTTP 202 carries the anomaly
// CAPTCHA page and must surface as an explicit error after the bounded
// retries — never as a successful empty result.
func TestDuckDuckGoProvider_Anomaly202FailsLoudly(t *testing.T) {
	var mu sync.Mutex
	hits := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		mu.Lock()
		hits++
		mu.Unlock()
		w.WriteHeader(http.StatusAccepted)
		_, _ = w.Write([]byte(ddgAnomalyPage))
	}))
	defer server.Close()

	rec := &waitRecorder{}
	provider := newSilentProvider(30*time.Second, rec)
	provider.SetBaseURL(server.URL)

	results, err := provider.Search(context.Background(), "test", 5)
	if err == nil {
		t.Fatalf("expected an explicit error for the 202 anomaly response, got %d results", len(results))
	}
	if !strings.Contains(err.Error(), "rate limited") || !strings.Contains(err.Error(), "HTTP 202") {
		t.Errorf("error = %q, want it to name the anomaly rate limit and HTTP 202", err.Error())
	}
	if got := hits; got != ddgMaxAttempts {
		t.Errorf("expected %d attempts (bounded retries), got %d", ddgMaxAttempts, got)
	}
	// Four waits interleave: backoff(1), pacer slot before attempt 2,
	// backoff(2), pacer slot before attempt 3. The backoffs are exact
	// computed values (jitter pinned to zero); the pacer slots sit anywhere
	// in [interval, 2·interval] — with the fake waits not advancing the
	// clock, slots accumulate (4s, then 8s), while in real time the backoff
	// advances toward the next slot and the remainder is waited out. Both
	// satisfy the invariant the pacer exists for: consecutive requests are
	// at least ddgMinRequestInterval apart.
	got := rec.recorded()
	if len(got) != 4 {
		t.Fatalf("waits = %v, want exactly 4 (two backoffs interleaved with two pacer slots)", got)
	}
	assertDurationClose(t, 0, got[0], ddgBaseBackoff)
	assertDurationClose(t, 2, got[2], ddgBaseBackoff<<1)
	for _, i := range []int{1, 3} {
		if got[i] < ddgMinRequestInterval-waitEpsilon || got[i] > 2*ddgMinRequestInterval {
			t.Errorf("pacer wait[%d] = %v, want within [%v, %v]", i, got[i], ddgMinRequestInterval, 2*ddgMinRequestInterval)
		}
	}
}

// TestDuckDuckGoProvider_Anomaly200BodyFailsLoudly verifies the anomaly scan
// also catches the challenge page served behind a 200: the page has no
// results, so without the body check the provider would report a silent
// empty result again.
func TestDuckDuckGoProvider_Anomaly200BodyFailsLoudly(t *testing.T) {
	var mu sync.Mutex
	hits := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		mu.Lock()
		hits++
		mu.Unlock()
		_, _ = w.Write([]byte(ddgAnomalyPage))
	}))
	defer server.Close()

	provider := newSilentProvider(30*time.Second, &waitRecorder{})
	provider.SetBaseURL(server.URL)

	results, err := provider.Search(context.Background(), "test", 5)
	if err == nil {
		t.Fatalf("expected an explicit error for the 200 anomaly body, got %d results", len(results))
	}
	if !strings.Contains(err.Error(), "rate limited") {
		t.Errorf("error = %q, want it to name the anomaly rate limit", err.Error())
	}
	if got := hits; got != ddgMaxAttempts {
		t.Errorf("expected %d attempts, got %d", ddgMaxAttempts, got)
	}
}

// TestDuckDuckGoProvider_AnomalyThenSuccess verifies the bounded retry loop
// recovers when the flag clears between attempts: the first 202 is retried
// after the first backoff step and the second attempt returns results.
func TestDuckDuckGoProvider_AnomalyThenSuccess(t *testing.T) {
	mockHTML := `<!DOCTYPE html>
<html><body>
<div class="result">
	<a class="result__a" href="https://example.com/1">Result 1</a>
	<a class="result__snippet">Snippet 1</a>
</div>
</body></html>`

	var mu sync.Mutex
	hits := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		mu.Lock()
		hits++
		flagged := hits == 1
		mu.Unlock()
		if flagged {
			w.WriteHeader(http.StatusAccepted)
			_, _ = w.Write([]byte(ddgAnomalyPage))
			return
		}
		_, _ = w.Write([]byte(mockHTML))
	}))
	defer server.Close()

	provider := newSilentProvider(30*time.Second, &waitRecorder{})
	provider.SetBaseURL(server.URL)

	results, err := provider.Search(context.Background(), "test", 5)
	if err != nil {
		t.Fatalf("Search() returned error: %v", err)
	}
	if len(results) != 1 || results[0].Title != "Result 1" {
		t.Errorf("expected the retried search to succeed, got %+v", results)
	}
}

// TestDuckDuckGoProvider_NonAnomalyErrorNotRetried verifies only anomaly
// responses are retried: a plain HTTP error surfaces immediately.
func TestDuckDuckGoProvider_NonAnomalyErrorNotRetried(t *testing.T) {
	var mu sync.Mutex
	hits := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		mu.Lock()
		hits++
		mu.Unlock()
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte("internal server error"))
	}))
	defer server.Close()

	rec := &waitRecorder{}
	provider := newSilentProvider(30*time.Second, rec)
	provider.SetBaseURL(server.URL)

	_, err := provider.Search(context.Background(), "test", 5)
	if err == nil {
		t.Fatal("expected error for HTTP 500, got nil")
	}
	if !strings.Contains(err.Error(), "HTTP 500") {
		t.Errorf("error = %q, want it to contain %q", err.Error(), "HTTP 500")
	}
	if got := hits; got != 1 {
		t.Errorf("expected exactly 1 request (no retry for non-anomaly errors), got %d", got)
	}
	if got := rec.recorded(); len(got) != 0 {
		t.Errorf("expected no backoff waits, got %v", got)
	}
}

// TestDuckDuckGoProvider_BackoffHonoursContext verifies a cancelled context
// interrupts the backoff wait instead of retrying.
func TestDuckDuckGoProvider_BackoffHonoursContext(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusAccepted)
		_, _ = w.Write([]byte(ddgAnomalyPage))
	}))
	defer server.Close()

	provider := NewDuckDuckGoProviderWithClient(30*time.Second, nil)
	provider.SetBaseURL(server.URL)
	provider.jitter = func() float64 { return 0 }

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := provider.Search(ctx, "test", 5)
	if err == nil {
		t.Fatal("expected an error for the cancelled context, got nil")
	}
	if !errors.Is(err, context.Canceled) {
		t.Errorf("error = %v, want it to wrap context.Canceled", err)
	}
}

// TestDuckDuckGoProvider_PacesConsecutiveRequests verifies the client-side
// rate limit: the second Search on the same provider instance waits out the
// minimum interval (plus jitter) before its request.
func TestDuckDuckGoProvider_PacesConsecutiveRequests(t *testing.T) {
	mockHTML := `<!DOCTYPE html>
<html><body><div class="result">
	<a class="result__a" href="https://example.com/1">Result 1</a>
	<a class="result__snippet">Snippet 1</a>
</div></body></html>`

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(mockHTML))
	}))
	defer server.Close()

	for _, tc := range []struct {
		name      string
		jitter    float64
		wantPause time.Duration
	}{
		{name: "no jitter", jitter: 0, wantPause: ddgMinRequestInterval},
		{name: "mid jitter", jitter: 0.5, wantPause: ddgMinRequestInterval + ddgMinRequestInterval/4},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rec := &waitRecorder{}
			provider := newSilentProvider(30*time.Second, rec)
			provider.SetBaseURL(server.URL)
			provider.jitter = func() float64 { return tc.jitter }

			for i := 0; i < 2; i++ {
				if _, err := provider.Search(context.Background(), "test", 5); err != nil {
					t.Fatalf("Search() call %d returned error: %v", i+1, err)
				}
			}
			got := rec.recorded()
			if len(got) != 1 {
				t.Fatalf("expected exactly one pacing wait (between the two searches), got %v", got)
			}
			assertDurationClose(t, 0, got[0], tc.wantPause)
		})
	}
}

// TestDuckDuckGoProvider_FirstRequestNotDelayed verifies a fresh provider
// issues its first request without pacing wait.
func TestDuckDuckGoProvider_FirstRequestNotDelayed(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`<!DOCTYPE html><html><body></body></html>`))
	}))
	defer server.Close()

	rec := &waitRecorder{}
	provider := newSilentProvider(30*time.Second, rec)
	provider.SetBaseURL(server.URL)

	if _, err := provider.Search(context.Background(), "test", 5); err != nil {
		t.Fatalf("Search() returned error: %v", err)
	}
	if got := rec.recorded(); len(got) != 0 {
		t.Errorf("expected no wait on the first request, got %v", got)
	}
}

// waitEpsilon is the wall-clock slack allowed on recorded wait durations:
// real time advances by microseconds between calls, so a pacer slot computed
// from an earlier `now` can come out slightly below the ideal value.
const waitEpsilon = 100 * time.Millisecond

// assertDurationClose checks got is within [want-waitEpsilon, want] — the
// recorded duration must never exceed the requested value.
func assertDurationClose(t *testing.T, i int, got, want time.Duration) {
	t.Helper()
	if got > want || got < want-waitEpsilon {
		t.Errorf("wait[%d] = %v, want %v (±%v scheduling epsilon)", i, got, want, waitEpsilon)
	}
}

// Compile-time guard: the recorder must satisfy the wait signature used by
// the provider.
var _ func(ctx context.Context, d time.Duration) error = (*waitRecorder)(nil).wait
