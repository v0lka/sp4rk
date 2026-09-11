package builtins

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// HTML small enough to pass the body limit; used by retry tests.
const retryTestHTML = `<html><body><h1>ok</h1></body></html>`

func TestWebFetchTool_RetrySucceedsAfterTransientFailures(t *testing.T) {
	var hits atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if hits.Add(1) <= 2 {
			// First two attempts fail with a transient server-side status.
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		w.Header().Set("Content-Type", "text/html")
		_, _ = w.Write([]byte(retryTestHTML))
	}))
	defer server.Close()

	tool := newTestWebFetchTool(WebFetchLimits{Timeout: 5 * time.Second, Retries: 2})

	input, _ := json.Marshal(map[string]string{"url": server.URL})
	result, err := tool.Execute(context.Background(), input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Errorf("expected success after retries, got error: %s", result.Content)
	}
	if !strings.Contains(result.Content, "ok") {
		t.Errorf("expected fetched content, got: %s", result.Content)
	}
	if got := hits.Load(); got != 3 {
		t.Errorf("expected 3 requests (1 initial + 2 retries), got %d", got)
	}
}

func TestWebFetchTool_RetriesExhausted(t *testing.T) {
	var hits atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer server.Close()

	tool := newTestWebFetchTool(WebFetchLimits{Timeout: 5 * time.Second, Retries: 2})

	input, _ := json.Marshal(map[string]string{"url": server.URL})
	result, err := tool.Execute(context.Background(), input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Error("expected IsError=true after exhausting retries")
	}
	if !strings.Contains(result.Content, "503") {
		t.Errorf("expected error to contain the last HTTP status, got: %s", result.Content)
	}
	if got := hits.Load(); got != 3 {
		t.Errorf("expected exactly 3 requests (1 initial + 2 retries), got %d", got)
	}
}

func TestWebFetchTool_ZeroRetriesSingleAttempt(t *testing.T) {
	var hits atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer server.Close()

	tool := newTestWebFetchTool(WebFetchLimits{Timeout: 5 * time.Second})

	input, _ := json.Marshal(map[string]string{"url": server.URL})
	result, err := tool.Execute(context.Background(), input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Error("expected IsError=true for 503 without retries")
	}
	if got := hits.Load(); got != 1 {
		t.Errorf("expected exactly 1 request with Retries=0, got %d", got)
	}
}

func TestWebFetchTool_NoRetryOnClientError(t *testing.T) {
	var hits atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.WriteHeader(http.StatusNotFound)
	}))
	defer server.Close()

	tool := newTestWebFetchTool(WebFetchLimits{Timeout: 5 * time.Second, Retries: 2})

	input, _ := json.Marshal(map[string]string{"url": server.URL})
	result, err := tool.Execute(context.Background(), input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Error("expected IsError=true for HTTP 404")
	}
	// 404 is deterministic for the same URL — retrying cannot help.
	if got := hits.Load(); got != 1 {
		t.Errorf("expected no retry for deterministic 404, got %d requests", got)
	}
}

func TestWebFetchTool_NoRetryOnBodyLimit(t *testing.T) {
	var hits atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		_, _ = w.Write([]byte(strings.Repeat("x", maxWebFetchBodyBytes+100)))
	}))
	defer server.Close()

	tool := newTestWebFetchTool(WebFetchLimits{Timeout: 5 * time.Second, Retries: 2})

	input, _ := json.Marshal(map[string]string{"url": server.URL})
	result, err := tool.Execute(context.Background(), input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Error("expected IsError=true for body exceeding the fetch cap")
	}
	if !strings.Contains(result.Content, "byte limit") {
		t.Errorf("expected body-limit error, got: %s", result.Content)
	}
	// The cap is deterministic for the same URL — re-downloading 10 MB per
	// retry would be pure waste.
	if got := hits.Load(); got != 1 {
		t.Errorf("expected no retry for body-limit failure, got %d requests", got)
	}
}

func TestWebFetchTool_NoRetryOnCancelledContext(t *testing.T) {
	var hits atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		_, _ = w.Write([]byte(retryTestHTML))
	}))
	defer server.Close()

	tool := newTestWebFetchTool(WebFetchLimits{Timeout: 5 * time.Second, Retries: 2})

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	input, _ := json.Marshal(map[string]string{"url": server.URL})
	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Error("expected IsError=true for a cancelled context")
	}
	// Caller cancellation is never retried — the request must not even be sent.
	if got := hits.Load(); got != 0 {
		t.Errorf("expected the cancelled request not to reach the server, got %d requests", got)
	}
}

func TestWebFetchTool_RetryUsesDoubledTimeout(t *testing.T) {
	// Channel-gated handler: attempt 0's response is held until the test
	// releases it AFTER Execute has returned, so attempt 0 can fail only via
	// the 200ms client timeout — no dependence on a handler sleep racing the
	// timeout on a loaded runner. Attempt 1 answers instantly and must
	// succeed under the doubled (400ms) timeout, proving the retry applies
	// the doubled per-attempt budget.
	release := make(chan struct{})
	var hits atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if hits.Add(1) == 1 {
			<-release // park attempt 0 until the client has timed out
			return    // abandoned response; the client is gone by now
		}
		w.Header().Set("Content-Type", "text/html")
		_, _ = w.Write([]byte(retryTestHTML))
	}))
	defer func() {
		close(release) // let the parked handler return before Close waits on it
		server.Close()
	}()

	tool := newTestWebFetchTool(WebFetchLimits{Timeout: 200 * time.Millisecond, Retries: 1})

	input, _ := json.Marshal(map[string]string{"url": server.URL})
	result, err := tool.Execute(context.Background(), input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Errorf("expected the doubled-timeout retry to succeed, got error: %s", result.Content)
	}
	if got := hits.Load(); got != 2 {
		t.Errorf("expected 2 requests (timed-out initial + successful retry), got %d", got)
	}
}

func TestWebFetchTool_RetryBackoffApplied(t *testing.T) {
	// A retryable 503 followed by success must wait for the backoff delay
	// before re-issuing — an immediate burst is exactly what rate limiting
	// is meant to stop.
	var hits atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if hits.Add(1) == 1 {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		w.Header().Set("Content-Type", "text/html")
		_, _ = w.Write([]byte(retryTestHTML))
	}))
	defer server.Close()

	tool := newTestWebFetchTool(WebFetchLimits{Timeout: 5 * time.Second, Retries: 2})

	input, _ := json.Marshal(map[string]string{"url": server.URL})
	start := time.Now()
	result, err := tool.Execute(context.Background(), input)
	elapsed := time.Since(start)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Errorf("expected success after retry, got error: %s", result.Content)
	}
	if elapsed < retryBackoffBase {
		t.Errorf("expected at least one backoff delay of %v before the retry, elapsed %v", retryBackoffBase, elapsed)
	}
}

func TestWebFetchTool_RetryHonorsRetryAfterHeader(t *testing.T) {
	// A 429 carrying Retry-After: 1 must delay the retry by ~1s (the
	// requested delay, under the maxRetryDelay ceiling), not by the plain
	// backoff.
	var hits atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if hits.Add(1) == 1 {
			w.Header().Set("Retry-After", "1")
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
		w.Header().Set("Content-Type", "text/html")
		_, _ = w.Write([]byte(retryTestHTML))
	}))
	defer server.Close()

	tool := newTestWebFetchTool(WebFetchLimits{Timeout: 5 * time.Second, Retries: 1})

	input, _ := json.Marshal(map[string]string{"url": server.URL})
	start := time.Now()
	result, err := tool.Execute(context.Background(), input)
	elapsed := time.Since(start)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Errorf("expected success after Retry-After delay, got error: %s", result.Content)
	}
	if elapsed < time.Second {
		t.Errorf("expected the origin-requested 1s delay before the retry, elapsed %v", elapsed)
	}
	if elapsed > 3*time.Second {
		t.Errorf("retry waited far longer than the requested delay, elapsed %v", elapsed)
	}
}

func TestWebFetchTool_TotalBudgetBoundsSlowDrip(t *testing.T) {
	// A server that drip-feeds headers slower than the per-attempt timeout
	// must not hold the call for the unbounded sum of doubled timeouts: with
	// base 300ms and Retries 2 the budget is 300ms<<3 = 2.4s even though the
	// naive doubling schedule would allow 300+600+1200 = 2.1s of attempts
	// PLUS backoff — here every attempt hangs to its per-attempt timeout, so
	// only the budget ends the call.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Park until the client abandons the attempt (its per-attempt timeout
		// or the wall-clock budget closes the connection, which cancels the
		// request context) instead of sleeping a fixed 10s. httptest.Server's
		// Close waits for in-flight handlers, so a fixed sleep would keep the
		// whole test (and the package run) blocked for ~10s of teardown after
		// the call itself has already returned.
		select {
		case <-r.Context().Done():
		case <-time.After(10 * time.Second): // longer than any per-attempt timeout
		}
		w.Header().Set("Content-Type", "text/html")
		_, _ = w.Write([]byte(retryTestHTML))
	}))
	defer server.Close()

	tool := newTestWebFetchTool(WebFetchLimits{Timeout: 300 * time.Millisecond, Retries: 2})

	input, _ := json.Marshal(map[string]string{"url": server.URL})
	start := time.Now()
	result, err := tool.Execute(context.Background(), input)
	elapsed := time.Since(start)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Error("expected IsError=true when the budget expires")
	}
	if elapsed > 5*time.Second {
		t.Errorf("call ran %v; the wall-clock budget (2.4s) did not bound it", elapsed)
	}
}

func TestRetryDelay(t *testing.T) {
	cases := []struct {
		name    string
		err     error
		attempt int
		want    time.Duration
	}{
		{"first retry backs off 250ms", &httpStatusError{code: 503}, 0, 250 * time.Millisecond},
		{"second retry backs off 500ms", &httpStatusError{code: 503}, 1, 500 * time.Millisecond},
		{"third retry backs off 1s", &httpStatusError{code: 503}, 2, time.Second},
		{"backoff saturates at cap", &httpStatusError{code: 503}, 30, maxRetryDelay},
		{"retry-after seconds honored", &httpStatusError{code: 429, retryAfter: 2 * time.Second}, 0, 2 * time.Second},
		{"retry-after capped at ceiling", &httpStatusError{code: 429, retryAfter: time.Hour}, 0, maxRetryDelay},
		{"retry-after beats plain backoff", &httpStatusError{code: 429, retryAfter: 2 * time.Second}, 5, 2 * time.Second},
		{"zero retry-after falls back to backoff", &httpStatusError{code: 429}, 5, maxRetryDelay},
		{"transport error uses plain backoff", fmt.Errorf("request failed: %w", errors.New("connection reset")), 0, 250 * time.Millisecond},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := retryDelay(tc.err, tc.attempt); got != tc.want {
				t.Errorf("retryDelay(%v, %d) = %v, want %v", tc.err, tc.attempt, got, tc.want)
			}
		})
	}
}

func TestParseRetryAfter(t *testing.T) {
	now := time.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC)
	cases := []struct {
		name   string
		header string
		want   time.Duration
	}{
		{"empty", "", 0},
		{"seconds", "30", 30 * time.Second},
		{"zero seconds", "0", 0},
		{"negative seconds", "-5", 0},
		{"http date in the future", now.Add(90 * time.Second).Format(http.TimeFormat), 90 * time.Second},
		{"http date in the past", now.Add(-time.Hour).Format(http.TimeFormat), 0},
		{"garbage", "soon", 0},
		{"fractional seconds unsupported", "1.5", 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := parseRetryAfter(tc.header, now)
			if got == tc.want {
				return
			}
			// The HTTP-date case may differ by sub-second rounding of the
			// formatted timestamp; allow a small tolerance.
			diff := got - tc.want
			if diff < 0 {
				diff = -diff
			}
			if diff > time.Second {
				t.Errorf("parseRetryAfter(%q) = %v, want ~%v", tc.header, got, tc.want)
			}
		})
	}
}

func TestTotalFetchBudget(t *testing.T) {
	cases := []struct {
		name    string
		base    time.Duration
		retries int
		want    time.Duration
	}{
		{"no retries means no budget", 30 * time.Second, 0, 0},
		{"negative retries means no budget", 30 * time.Second, -1, 0},
		{"non-positive base means no budget", 0, 3, 0},
		{"one retry", time.Second, 1, 4 * time.Second},
		{"two retries", time.Second, 2, 8 * time.Second},
		{"three retries covers full schedule", time.Second, 3, 16 * time.Second},
		{"saturates beyond three retries", time.Second, 7, 16 * time.Second},
		{"overflow clamps to max", time.Duration(1) << 62, 3, math.MaxInt64},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := totalFetchBudget(tc.base, tc.retries); got != tc.want {
				t.Errorf("totalFetchBudget(%v, %d) = %v, want %v", tc.base, tc.retries, got, tc.want)
			}
		})
	}
}

func TestAttemptTimeout(t *testing.T) {
	base := 30 * time.Second
	cases := []struct {
		name    string
		base    time.Duration
		attempt int
		want    time.Duration
	}{
		{"attempt 0 keeps base", base, 0, 30 * time.Second},
		{"attempt 1 doubles", base, 1, 60 * time.Second},
		{"attempt 2 quadruples", base, 2, 120 * time.Second},
		{"attempt 3", base, 3, 240 * time.Second},
		{"zero base stays unlimited", 0, 2, 0},
		{"negative base unchanged", -time.Second, 1, -time.Second},
		{"negative attempt unchanged", base, -1, base},
		{"overflow clamps to max", time.Duration(1) << 62, 5, math.MaxInt64},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := attemptTimeout(tc.base, tc.attempt); got != tc.want {
				t.Errorf("attemptTimeout(%v, %d) = %v, want %v", tc.base, tc.attempt, got, tc.want)
			}
		})
	}
}

func TestRetryableFetchError(t *testing.T) {
	cases := []struct {
		name string
		err  error
		want bool
	}{
		{"nil error", nil, false},
		{"404 not retried", &httpStatusError{code: http.StatusNotFound, text: "404 Not Found"}, false},
		{"403 not retried", &httpStatusError{code: http.StatusForbidden, text: "403 Forbidden"}, false},
		{"408 retried", &httpStatusError{code: http.StatusRequestTimeout, text: "408 Request Timeout"}, true},
		{"429 retried", &httpStatusError{code: http.StatusTooManyRequests, text: "429 Too Many Requests"}, true},
		{"500 retried", &httpStatusError{code: http.StatusInternalServerError, text: "500 Internal Server Error"}, true},
		{"503 retried", &httpStatusError{code: http.StatusServiceUnavailable, text: "503 Service Unavailable"}, true},
		{"body limit not retried", &bodyLimitError{}, false},
		{"too many redirects not retried", fmt.Errorf("request failed: %w", errTooManyRedirects), false},
		{"private redirect not retried", fmt.Errorf("request failed: %w", errRedirectPrivateRefused), false},
		{"failed SSRF redirect check not retried", fmt.Errorf("request failed: %w", errRedirectSSRFCheckFailed), false},
		{"wrapped status error classified", fmt.Errorf("outer: %w", &httpStatusError{code: http.StatusBadGateway, text: "502 Bad Gateway"}), true},
		{"transport error retried", fmt.Errorf("request failed: %w", errors.New("connection reset by peer")), true},
		{"per-attempt timeout retried", fmt.Errorf("request failed: %w", errors.New("context deadline exceeded (Client.Timeout exceeded while awaiting headers)")), true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := retryableFetchError(tc.err); got != tc.want {
				t.Errorf("retryableFetchError(%v) = %v, want %v", tc.err, got, tc.want)
			}
		})
	}
}
