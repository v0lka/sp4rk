package llm

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// asyncTestClient builds an *http.Client wrapped with the 202-polling
// transport at a fast cadence for tests: a tiny fallback interval and no
// MinInterval floor (the floor gets its own dedicated test).
func asyncTestClient(cfg AsyncAcceptedConfig) *http.Client {
	if cfg.DefaultInterval == 0 {
		cfg.DefaultInterval = 2 * time.Millisecond
	}
	if cfg.MinInterval == 0 {
		cfg.MinInterval = -1
	}
	return WrapHTTPClientAsyncAccepted(nil, cfg)
}

// TestAsyncAcceptedTransport_PollsUntilTerminal exercises the full idiom: the
// POST answers 202 with a relative Location, the first poll answers 202
// again, the second delivers the result. The final body reaches the caller as
// the response to the original request; polls are GETs carrying the original
// Authorization (same host).
func TestAsyncAcceptedTransport_PollsUntilTerminal(t *testing.T) {
	var polls atomic.Int32
	var pollAuth atomic.Value
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/submit":
			w.Header().Set("Location", "/jobs/42")
			w.Header().Set("Retry-After-Ms", "5")
			w.WriteHeader(http.StatusAccepted)
			fmt.Fprint(w, `{"task_id": "42"}`)
		case r.Method == http.MethodGet && r.URL.Path == "/jobs/42":
			pollAuth.Store(r.Header.Get("Authorization"))
			if polls.Add(1) == 1 {
				w.Header().Set("Retry-After-Ms", "5")
				w.WriteHeader(http.StatusAccepted)
				return
			}
			fmt.Fprint(w, `{"result": "done"}`)
		default:
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	t.Cleanup(srv.Close)

	req, _ := http.NewRequest(http.MethodPost, srv.URL+"/submit", strings.NewReader(`{"in":1}`))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer k")
	resp, err := asyncTestClient(AsyncAcceptedConfig{}).Do(req)
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Errorf("status = %d, want 200", resp.StatusCode)
	}
	body, _ := io.ReadAll(resp.Body)
	if string(body) != `{"result": "done"}` {
		t.Errorf("body = %q, want the terminal result", body)
	}
	if got := polls.Load(); got != 2 {
		t.Errorf("polls = %d, want 2 (one pending, one terminal)", got)
	}
	if got, _ := pollAuth.Load().(string); got != "Bearer k" {
		t.Errorf("poll Authorization = %q, want it forwarded on the same host", got)
	}
}

// TestAsyncAcceptedTransport_HonorsRetryAfter verifies the server-provided
// wait governs the cadence: with a tiny client-side default, the elapsed time
// must still cover the Retry-After-Ms the 202 requested.
func TestAsyncAcceptedTransport_HonorsRetryAfter(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			w.Header().Set("Location", "/jobs/1")
			w.Header().Set("Retry-After-Ms", "150")
			w.WriteHeader(http.StatusAccepted)
			return
		}
		fmt.Fprint(w, `ok`)
	}))
	t.Cleanup(srv.Close)

	start := time.Now()
	resp, err := asyncTestClient(AsyncAcceptedConfig{}).Post(srv.URL+"/submit", "application/json", strings.NewReader(`{}`))
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	defer resp.Body.Close()
	if elapsed := time.Since(start); elapsed < 140*time.Millisecond {
		t.Errorf("elapsed = %v, want >= ~150ms (Retry-After-Ms must govern the wait, not the 2ms default)", elapsed)
	}
}

// TestAsyncAcceptedTransport_Passthrough verifies the two untouched paths: a
// 202 without a Location (not the idiom — body must survive) and an ordinary
// synchronous 200.
func TestAsyncAcceptedTransport_Passthrough(t *testing.T) {
	var gets atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet {
			gets.Add(1)
		}
		switch r.URL.Path {
		case "/accepted-no-location":
			w.WriteHeader(http.StatusAccepted)
			fmt.Fprint(w, `{"task_id": "7"}`)
		default:
			fmt.Fprint(w, `sync`)
		}
	}))
	t.Cleanup(srv.Close)
	client := asyncTestClient(AsyncAcceptedConfig{})

	resp, err := client.Post(srv.URL+"/accepted-no-location", "application/json", strings.NewReader(`{}`))
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusAccepted || string(body) != `{"task_id": "7"}` {
		t.Errorf("202-without-Location = %d %q, want it passed through with its body", resp.StatusCode, body)
	}

	resp, err = client.Post(srv.URL+"/sync", "application/json", strings.NewReader(`{}`))
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	body, _ = io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK || string(body) != "sync" {
		t.Errorf("synchronous response = %d %q, want 200 \"sync\" untouched", resp.StatusCode, body)
	}
	if got := gets.Load(); got != 0 {
		t.Errorf("polls = %d, want 0 for passthrough paths", got)
	}
}

// TestAsyncAcceptedTransport_CrossHostStripsCredentials verifies the
// same-or-subdomain rule: a monitor on a differently-named host (localhost vs
// 127.0.0.1 — same server, different hostname) is polled without the
// sensitive headers while benign headers still travel.
func TestAsyncAcceptedTransport_CrossHostStripsCredentials(t *testing.T) {
	var pollAuth, pollAPIKey, pollTrace atomic.Value
	var srvURL string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			w.Header().Set("Location", strings.Replace(srvURL, "127.0.0.1", "localhost", 1)+"/jobs/9")
			w.WriteHeader(http.StatusAccepted)
			return
		}
		pollAuth.Store(r.Header.Get("Authorization"))
		pollAPIKey.Store(r.Header.Get("X-Api-Key"))
		pollTrace.Store(r.Header.Get("X-Trace-Id"))
		fmt.Fprint(w, `ok`)
	}))
	t.Cleanup(srv.Close)
	srvURL = srv.URL

	req, _ := http.NewRequest(http.MethodPost, srv.URL+"/submit", strings.NewReader(`{}`))
	req.Header.Set("Authorization", "Bearer secret")
	req.Header.Set("X-Api-Key", "secret2")
	req.Header.Set("X-Trace-Id", "trace-1")
	resp, err := asyncTestClient(AsyncAcceptedConfig{}).Do(req)
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	resp.Body.Close()

	if got, _ := pollAuth.Load().(string); got != "" {
		t.Errorf("cross-host poll Authorization = %q, want stripped", got)
	}
	if got, _ := pollAPIKey.Load().(string); got != "" {
		t.Errorf("cross-host poll X-Api-Key = %q, want stripped", got)
	}
	if got, _ := pollTrace.Load().(string); got != "trace-1" {
		t.Errorf("cross-host poll X-Trace-Id = %q, want benign headers forwarded", got)
	}
}

// TestAsyncAcceptedTransport_TransientPollFailuresTolerated verifies that
// 5xx answers from the monitor do not abandon the task: the loop keeps
// polling and still delivers the eventual result.
func TestAsyncAcceptedTransport_TransientPollFailuresTolerated(t *testing.T) {
	var polls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			w.Header().Set("Location", "/jobs/1")
			w.WriteHeader(http.StatusAccepted)
			return
		}
		switch polls.Add(1) {
		case 1, 2:
			http.Error(w, "hiccup", http.StatusInternalServerError)
		default:
			fmt.Fprint(w, `ok`)
		}
	}))
	t.Cleanup(srv.Close)

	resp, err := asyncTestClient(AsyncAcceptedConfig{}).Post(srv.URL+"/submit", "application/json", strings.NewReader(`{}`))
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK || string(body) != "ok" {
		t.Errorf("response = %d %q, want the result after tolerated hiccups", resp.StatusCode, body)
	}
	if got := polls.Load(); got != 3 {
		t.Errorf("polls = %d, want 3 (two failures tolerated)", got)
	}
}

// TestAsyncAcceptedTransport_GivesUpAfterConsecutiveFailures verifies the
// tolerance bound: a monitor that keeps failing eventually surfaces its last
// failing response instead of polling forever.
func TestAsyncAcceptedTransport_GivesUpAfterConsecutiveFailures(t *testing.T) {
	var polls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			w.Header().Set("Location", "/jobs/1")
			w.WriteHeader(http.StatusAccepted)
			return
		}
		polls.Add(1)
		http.Error(w, "down", http.StatusServiceUnavailable)
	}))
	t.Cleanup(srv.Close)

	client := asyncTestClient(AsyncAcceptedConfig{MaxConsecutiveErrs: 2})
	resp, err := client.Post(srv.URL+"/submit", "application/json", strings.NewReader(`{}`))
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want the last 503 surfaced for upstream classification", resp.StatusCode)
	}
	if got := polls.Load(); got != 2 {
		t.Errorf("polls = %d, want exactly MaxConsecutiveErrs (2)", got)
	}
}

// TestAsyncAcceptedTransport_ContextCancel verifies a context cancelled
// mid-wait aborts the poll loop with the context error.
func TestAsyncAcceptedTransport_ContextCancel(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			w.Header().Set("Location", "/jobs/1")
			w.Header().Set("Retry-After", "60")
			w.WriteHeader(http.StatusAccepted)
			return
		}
		t.Error("monitor must never be reached: the context is cancelled during the first wait")
	}))
	t.Cleanup(srv.Close)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, srv.URL+"/submit", strings.NewReader(`{}`))
	_, err := asyncTestClient(AsyncAcceptedConfig{}).Do(req)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("error = %v, want context.DeadlineExceeded", err)
	}
}

// TestParseRetryAfter pins the header grammar: millisecond extension first,
// delta-seconds, HTTP-date (future and past), and the unparseable/absent
// fallbacks.
func TestParseRetryAfter(t *testing.T) {
	h := func(k, v string) http.Header {
		hdr := http.Header{}
		hdr.Set(k, v)
		return hdr
	}
	if d, ok := parseRetryAfter(h("Retry-After-Ms", "250")); !ok || d != 250*time.Millisecond {
		t.Errorf("Retry-After-Ms 250 = %v %v, want 250ms true", d, ok)
	}
	if d, ok := parseRetryAfter(h("Retry-After", "3")); !ok || d != 3*time.Second {
		t.Errorf("Retry-After 3 = %v %v, want 3s true", d, ok)
	}
	future := time.Now().Add(2 * time.Second).UTC().Format(http.TimeFormat)
	if d, ok := parseRetryAfter(h("Retry-After", future)); !ok || d <= 0 || d > 2*time.Second {
		t.Errorf("Retry-After future date = %v %v, want ~2s true", d, ok)
	}
	past := time.Now().Add(-time.Hour).UTC().Format(http.TimeFormat)
	if d, ok := parseRetryAfter(h("Retry-After", past)); !ok || d != 0 {
		t.Errorf("Retry-After past date = %v %v, want 0 true (poll now)", d, ok)
	}
	if _, ok := parseRetryAfter(h("Retry-After", "soon")); ok {
		t.Error("unparseable Retry-After must report ok=false")
	}
	if _, ok := parseRetryAfter(http.Header{}); ok {
		t.Error("absent Retry-After must report ok=false")
	}
	// Azure's millisecond spelling is honored too.
	if d, ok := parseRetryAfter(h("x-ms-retry-after-ms", "80")); !ok || d != 80*time.Millisecond {
		t.Errorf("x-ms-retry-after-ms 80 = %v %v, want 80ms true", d, ok)
	}
	// Precedence: the millisecond extensions win over the coarser header,
	// and the OpenAI spelling wins over the Azure one.
	hdr := http.Header{}
	hdr.Set("Retry-After", "60")
	hdr.Set("x-ms-retry-after-ms", "200")
	hdr.Set("Retry-After-Ms", "100")
	if d, ok := parseRetryAfter(hdr); !ok || d != 100*time.Millisecond {
		t.Errorf("precedence = %v %v, want Retry-After-Ms (100ms) to win", d, ok)
	}
}

// TestAsyncAcceptedTransport_MaxPollsExhausted verifies the count budget: a
// monitor answering 202 forever is abandoned after exactly MaxPolls polls,
// the error names the spent budget (and is shaped non-retryable so upper
// layers do not resubmit), and CancelOnAbandon sends the best-effort DELETE —
// carrying auth on the same host.
func TestAsyncAcceptedTransport_MaxPollsExhausted(t *testing.T) {
	var polls atomic.Int32
	var deleteAuth atomic.Value
	deleted := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost:
			w.Header().Set("Location", "/jobs/1")
			w.WriteHeader(http.StatusAccepted)
		case r.Method == http.MethodGet:
			polls.Add(1)
			w.Header().Set("Retry-After-Ms", "1")
			w.WriteHeader(http.StatusAccepted)
		case r.Method == http.MethodDelete && r.URL.Path == "/jobs/1":
			deleteAuth.Store(r.Header.Get("Authorization"))
			close(deleted)
			w.WriteHeader(http.StatusNoContent)
		default:
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
	}))
	t.Cleanup(srv.Close)

	client := asyncTestClient(AsyncAcceptedConfig{MaxPolls: 3, CancelOnAbandon: true})
	req, _ := http.NewRequest(http.MethodPost, srv.URL+"/submit", strings.NewReader(`{}`))
	req.Header.Set("Authorization", "Bearer k")
	_, err := client.Do(req)
	if err == nil || !strings.Contains(err.Error(), "poll budget") {
		t.Fatalf("error = %v, want a poll-budget exhaustion error", err)
	}
	if got := polls.Load(); got != 3 {
		t.Errorf("polls = %d, want exactly MaxPolls (3)", got)
	}
	select {
	case <-deleted:
	case <-time.After(5 * time.Second):
		t.Fatal("abandon-cancel DELETE never reached the monitor")
	}
	if got, _ := deleteAuth.Load().(string); got != "Bearer k" {
		t.Errorf("DELETE Authorization = %q, want it forwarded on the same host", got)
	}
}

// TestAsyncAcceptedTransport_MaxElapsedExhausted verifies the duration
// budget: a wait that would overshoot MaxElapsed is never served — the call
// gives up before sleeping, with zero polls issued.
func TestAsyncAcceptedTransport_MaxElapsedExhausted(t *testing.T) {
	var polls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			w.Header().Set("Location", "/jobs/1")
			w.Header().Set("Retry-After-Ms", "500")
			w.WriteHeader(http.StatusAccepted)
			return
		}
		polls.Add(1)
		w.WriteHeader(http.StatusAccepted)
	}))
	t.Cleanup(srv.Close)

	client := asyncTestClient(AsyncAcceptedConfig{MaxElapsed: 20 * time.Millisecond})
	_, err := client.Post(srv.URL+"/submit", "application/json", strings.NewReader(`{}`))
	if err == nil || !strings.Contains(err.Error(), "polling budget") {
		t.Fatalf("error = %v, want a polling-budget exhaustion error", err)
	}
	if got := polls.Load(); got != 0 {
		t.Errorf("polls = %d, want 0 (the 500ms wait overshoots the 20ms budget up front)", got)
	}
}

// TestAsyncAcceptedTransport_MinIntervalFloor verifies a zero Retry-After
// cannot hot-loop the monitor: every wait is floored by MinInterval.
func TestAsyncAcceptedTransport_MinIntervalFloor(t *testing.T) {
	var polls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			w.Header().Set("Location", "/jobs/1")
			w.Header().Set("Retry-After-Ms", "0")
			w.WriteHeader(http.StatusAccepted)
			return
		}
		if polls.Add(1) == 1 {
			w.Header().Set("Retry-After-Ms", "0")
			w.WriteHeader(http.StatusAccepted)
			return
		}
		fmt.Fprint(w, `ok`)
	}))
	t.Cleanup(srv.Close)

	client := asyncTestClient(AsyncAcceptedConfig{MinInterval: 60 * time.Millisecond})
	start := time.Now()
	resp, err := client.Post(srv.URL+"/submit", "application/json", strings.NewReader(`{}`))
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	resp.Body.Close()
	// Two waits (before each poll), both floored: >= ~120ms total.
	if elapsed := time.Since(start); elapsed < 100*time.Millisecond {
		t.Errorf("elapsed = %v, want >= ~120ms (Retry-After-Ms: 0 must be floored by MinInterval)", elapsed)
	}
	if got := polls.Load(); got != 2 {
		t.Errorf("polls = %d, want 2", got)
	}
}

// TestNewRouter_AsyncAccepted_EntryConfig verifies the declarative wiring:
// the 202-polling transport is on BY DEFAULT (no opt-in flag on the entry)
// and ProviderEntry.AsyncAccepted reaches it, so a router consumer can bound
// polling (here MaxPolls: 2) without building a client — a 202-forever
// endpoint exhausts the entry's budget instead of the default.
func TestNewRouter_AsyncAccepted_EntryConfig(t *testing.T) {
	var polls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			w.Header().Set("Location", "/jobs/1")
			w.WriteHeader(http.StatusAccepted)
			return
		}
		polls.Add(1)
		w.Header().Set("Retry-After-Ms", "1")
		w.WriteHeader(http.StatusAccepted)
	}))
	t.Cleanup(srv.Close)

	router, err := NewRouter(context.Background(), RouterConfig{
		Providers: []ProviderEntry{{
			Name:         "p",
			ProviderType: "openai",
			APIKey:       "k",
			BaseURL:      srv.URL,
			Models:       []string{"test-model"},
			AsyncAccepted: AsyncAcceptedConfig{
				DefaultInterval: 2 * time.Millisecond,
				MinInterval:     -1,
				MaxPolls:        2,
			},
		}},
		MaxRetries: -1, // single attempt: a budget error must not resubmit
	}, nil)
	if err != nil {
		t.Fatalf("NewRouter: %v", err)
	}

	_, err = router.Call(context.Background(), ChatRequest{
		Model:    "test-model",
		Messages: []Message{{Role: "user", Content: "hi"}},
	})
	if err == nil || !strings.Contains(err.Error(), "poll budget") {
		t.Fatalf("error = %v, want the entry-configured poll-budget exhaustion", err)
	}
	if got := polls.Load(); got != 2 {
		t.Errorf("polls = %d, want exactly the entry-configured MaxPolls (2)", got)
	}
}

// TestNewRouter_DisableAsyncAccepted_OptOut verifies the opt-out: with
// DisableAsyncAccepted the client stays unwrapped, so a 202 surfaces raw to
// the SDK (a degenerate success → error) and the monitor is never polled.
func TestNewRouter_DisableAsyncAccepted_OptOut(t *testing.T) {
	var polls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			w.Header().Set("Location", "/jobs/1")
			w.WriteHeader(http.StatusAccepted)
			return
		}
		polls.Add(1)
		w.WriteHeader(http.StatusAccepted)
	}))
	t.Cleanup(srv.Close)

	router, err := NewRouter(context.Background(), RouterConfig{
		Providers: []ProviderEntry{{
			Name:                 "p",
			ProviderType:         "openai",
			APIKey:               "k",
			BaseURL:              srv.URL,
			Models:               []string{"test-model"},
			DisableAsyncAccepted: true,
		}},
		MaxRetries: -1,
	}, nil)
	if err != nil {
		t.Fatalf("NewRouter: %v", err)
	}

	if _, err = router.Call(context.Background(), ChatRequest{
		Model:    "test-model",
		Messages: []Message{{Role: "user", Content: "hi"}},
	}); err == nil {
		t.Fatal("expected an error: a raw 202 is a degenerate success for the SDK")
	}
	if got := polls.Load(); got != 0 {
		t.Errorf("polls = %d, want 0 (opt-out must not poll the monitor)", got)
	}
}

// TestOpenAIProvider_FollowAsyncAccepted_ChatCompletions drives the idiom end
// to end through the provider stack: a /chat/completions endpoint answering
// 202+Location, resolved by the wrapped client, assembled into the ordinary
// ChatResponse — the Provider contract never notices the async hop.
func TestOpenAIProvider_FollowAsyncAccepted_ChatCompletions(t *testing.T) {
	var polls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/chat/completions"):
			w.Header().Set("Location", "/jobs/cc-1")
			w.Header().Set("Retry-After-Ms", "5")
			w.WriteHeader(http.StatusAccepted)
		case r.Method == http.MethodGet && r.URL.Path == "/jobs/cc-1":
			if polls.Add(1) == 1 {
				w.Header().Set("Retry-After-Ms", "5")
				w.WriteHeader(http.StatusAccepted)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprint(w, `{"id":"cc-1","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"Hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5}}`)
		default:
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	t.Cleanup(srv.Close)

	p, err := NewOpenAIProvider(OpenAIProviderConfig{
		Name:       "p",
		APIKey:     "k",
		BaseURL:    srv.URL,
		HTTPClient: asyncTestClient(AsyncAcceptedConfig{}),
	})
	if err != nil {
		t.Fatalf("NewOpenAIProvider: %v", err)
	}
	resp, err := p.ChatCompletion(context.Background(), ChatRequest{
		Model:    "some-async-model",
		Messages: []Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("ChatCompletion: %v", err)
	}
	if resp.Message.Content != "Hello" {
		t.Errorf("content = %q, want Hello", resp.Message.Content)
	}
	if resp.StopReason != "end_turn" {
		t.Errorf("stop reason = %q, want end_turn", resp.StopReason)
	}
	if resp.Usage.InputTokens != 10 || resp.Usage.OutputTokens != 5 {
		t.Errorf("usage = %+v, want 10/5", resp.Usage)
	}
	if got := polls.Load(); got != 2 {
		t.Errorf("polls = %d, want 2", got)
	}
}
