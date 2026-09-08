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
	// Base timeout 200ms: attempt 0 must time out against a 300ms-delayed
	// handler, while retry attempt 1 (timeout doubled to 400ms) must succeed
	// against the instant response — proving the doubled timeout is applied.
	var hits atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if hits.Add(1) == 1 {
			time.Sleep(300 * time.Millisecond)
		}
		w.Header().Set("Content-Type", "text/html")
		_, _ = w.Write([]byte(retryTestHTML))
	}))
	defer server.Close()

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
