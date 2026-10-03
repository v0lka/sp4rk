package llm

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// staticTokenSource is the minimal TokenSource: it hands out one fixed token
// and counts how many times it was consulted.
type staticTokenSource struct {
	token BearerToken
	calls atomic.Int64
}

func (s *staticTokenSource) Token(ctx context.Context) (BearerToken, error) {
	s.calls.Add(1)
	if err := ctx.Err(); err != nil {
		return BearerToken{}, err
	}
	return s.token, nil
}

type erroringTokenSource struct{ err error }

func (s erroringTokenSource) Token(context.Context) (BearerToken, error) { return BearerToken{}, s.err }

// tokenSourceFunc adapts a function to the TokenSource interface (tests only).
type tokenSourceFunc func(ctx context.Context) (BearerToken, error)

func (f tokenSourceFunc) Token(ctx context.Context) (BearerToken, error) { return f(ctx) }

// TestTokenSourceStaticImplementation pins the interface shape: a type with
// only Token(ctx) (BearerToken, error) satisfies TokenSource, and the
// returned BearerToken fields round-trip untouched.
func TestTokenSourceStaticImplementation(t *testing.T) {
	var src TokenSource = &staticTokenSource{token: BearerToken{
		AccessToken:  "at",
		TokenType:    "Bearer",
		ExpiresAt:    time.Unix(1700000000, 0).UTC(),
		ExtraHeaders: map[string]string{"ChatGPT-Account-Id": "acct"},
	}}
	got, err := src.Token(context.Background())
	if err != nil {
		t.Fatalf("Token() error = %v, want nil", err)
	}
	if got.AccessToken != "at" || got.TokenType != "Bearer" {
		t.Fatalf("Token() = %+v, want fields to round-trip", got)
	}
	if want := time.Unix(1700000000, 0).UTC(); !got.ExpiresAt.Equal(want) {
		t.Fatalf("ExpiresAt = %v, want %v", got.ExpiresAt, want)
	}
	if got.ExtraHeaders["ChatGPT-Account-Id"] != "acct" {
		t.Fatalf("ExtraHeaders = %v, want ChatGPT-Account-Id to round-trip", got.ExtraHeaders)
	}
}

// TestTokenSourceErrorPropagation verifies an erroring implementation surfaces
// its error verbatim.
func TestTokenSourceErrorPropagation(t *testing.T) {
	wantErr := errors.New("signed out")
	src := erroringTokenSource{err: wantErr}
	var _ TokenSource = src
	_, err := src.Token(context.Background())
	if !errors.Is(err, wantErr) {
		t.Fatalf("Token() error = %v, want %v", err, wantErr)
	}
}

// headerRecorder spins up an OpenAI-compatible endpoint that records the
// request headers of the LAST request and replies with a minimal valid
// completion for whichever protocol the request targeted.
func headerRecorder(t *testing.T) (*http.Header, *httptest.Server) {
	t.Helper()
	var mu sync.Mutex
	var got http.Header
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		snapshot := r.Header.Clone()
		mu.Lock()
		got = snapshot
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		body := `{"id":"x","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`
		if strings.Contains(r.URL.Path, "responses") {
			body = `{"id":"x","object":"response","created":1,"model":"m","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}],"status":"completed","usage":{"input_tokens":1,"output_tokens":1}}`
		}
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)
	return &got, srv
}

func tokenSourceRequest(model string) ChatRequest {
	return ChatRequest{
		Model:    model,
		Messages: []Message{{Role: "user", Content: "hi"}},
	}
}

// TestOpenAIProvider_TokenSource_HeadersOnChatCompletions verifies the auth
// seam on the Chat Completions wire: a provider built with a TokenSource
// stamps the middleware-resolved Authorization header (overriding the static
// APIKey credential) and the BearerToken.ExtraHeaders onto every request.
func TestOpenAIProvider_TokenSource_HeadersOnChatCompletions(t *testing.T) {
	got, srv := headerRecorder(t)
	src := &staticTokenSource{token: BearerToken{
		AccessToken:  "tok-123",
		ExtraHeaders: map[string]string{"ChatGPT-Account-Id": "acct-456"},
	}}
	p, err := NewOpenAIProvider(OpenAIProviderConfig{
		Name:        "chatgpt",
		APIKey:      "static-key",
		BaseURL:     srv.URL,
		TokenSource: src,
	})
	if err != nil {
		t.Fatalf("NewOpenAIProvider: %v", err)
	}
	if _, err := p.ChatCompletion(context.Background(), tokenSourceRequest("gpt-4o")); err != nil {
		t.Fatalf("ChatCompletion: %v", err)
	}
	if auth := got.Get("Authorization"); auth != "Bearer tok-123" {
		t.Errorf("Authorization = %q, want %q (token must override the static APIKey)", auth, "Bearer tok-123")
	}
	if acct := got.Get("ChatGPT-Account-Id"); acct != "acct-456" {
		t.Errorf("ChatGPT-Account-Id = %q, want %q", acct, "acct-456")
	}
	if src.calls.Load() != 1 {
		t.Errorf("Token() calls = %d, want 1 (one per outgoing request)", src.calls.Load())
	}
}

// TestOpenAIProvider_TokenSource_HeadersOnResponses verifies the same seam on
// the Responses API wire: GPT-5.x models dispatch through responsesClient,
// which carries the same middleware.
func TestOpenAIProvider_TokenSource_HeadersOnResponses(t *testing.T) {
	got, srv := headerRecorder(t)
	src := &staticTokenSource{token: BearerToken{
		AccessToken:  "tok-resp",
		ExtraHeaders: map[string]string{"ChatGPT-Account-Id": "acct-resp"},
	}}
	p, err := NewOpenAIProvider(OpenAIProviderConfig{
		Name:        "chatgpt",
		APIKey:      "static-key",
		BaseURL:     srv.URL,
		TokenSource: src,
	})
	if err != nil {
		t.Fatalf("NewOpenAIProvider: %v", err)
	}
	if _, err := p.ChatCompletion(context.Background(), tokenSourceRequest("gpt-5.6")); err != nil {
		t.Fatalf("ChatCompletion: %v", err)
	}
	if auth := got.Get("Authorization"); auth != "Bearer tok-resp" {
		t.Errorf("Authorization = %q, want %q", auth, "Bearer tok-resp")
	}
	if acct := got.Get("ChatGPT-Account-Id"); acct != "acct-resp" {
		t.Errorf("ChatGPT-Account-Id = %q, want %q", acct, "acct-resp")
	}
}

// TestOpenAIProvider_TokenSource_AbsentKeepsStaticCredential pins the
// no-TokenSource behavior as historical: the static APIKey credential
// travels on the wire untouched and no extra headers appear.
func TestOpenAIProvider_TokenSource_AbsentKeepsStaticCredential(t *testing.T) {
	got, srv := headerRecorder(t)
	p, err := NewOpenAIProvider(OpenAIProviderConfig{
		Name:    "openai_compatible",
		APIKey:  "sk-static",
		BaseURL: srv.URL,
	})
	if err != nil {
		t.Fatalf("NewOpenAIProvider: %v", err)
	}
	if _, err := p.ChatCompletion(context.Background(), tokenSourceRequest("gpt-4o")); err != nil {
		t.Fatalf("ChatCompletion: %v", err)
	}
	if auth := got.Get("Authorization"); auth != "Bearer sk-static" {
		t.Errorf("Authorization = %q, want %q", auth, "Bearer sk-static")
	}
	if acct := got.Get("ChatGPT-Account-Id"); acct != "" {
		t.Errorf("ChatGPT-Account-Id = %q, want empty (no TokenSource configured)", acct)
	}
}

// TestOpenAIProvider_TokenSource_ErrorAbortsRequest verifies a Token failure
// aborts the request before it hits the wire.
func TestOpenAIProvider_TokenSource_ErrorAbortsRequest(t *testing.T) {
	var hits atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		hits.Add(1)
		_, _ = w.Write([]byte(`{}`))
	}))
	t.Cleanup(srv.Close)
	p, err := NewOpenAIProvider(OpenAIProviderConfig{
		Name:        "chatgpt",
		APIKey:      "static-key",
		BaseURL:     srv.URL,
		TokenSource: erroringTokenSource{err: errors.New("refresh failed")},
	})
	if err != nil {
		t.Fatalf("NewOpenAIProvider: %v", err)
	}
	_, err = p.ChatCompletion(context.Background(), tokenSourceRequest("gpt-4o"))
	if err == nil {
		t.Fatal("ChatCompletion must fail when Token() fails")
	}
	if !strings.Contains(err.Error(), "token source") {
		t.Errorf("error = %v, want it to name the token source", err)
	}
	if hits.Load() != 0 {
		t.Errorf("server hits = %d, want 0 (failed token resolution must not reach the wire)", hits.Load())
	}
}

// TestOpenAIProvider_TokenSource_FreshTokenPerRetryAttempt verifies that a
// permitted HTTP retry carries a freshly resolved credential: with the
// TokenSource seam active, SDK-internal retries are disabled (a credential
// failure must abort, not be retried in-SDK — see NewOpenAIProvider), so the
// ROUTER owns the retry loop and re-enters the provider per attempt, which
// re-resolves the token through the middleware each time. A 429-then-200
// endpoint served through Router.Call (MaxRetries=1) sees the freshly
// generated token of attempt two from a single call.
func TestOpenAIProvider_TokenSource_FreshTokenPerRetryAttempt(t *testing.T) {
	var attempt atomic.Int64
	var mu sync.Mutex
	var auths []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := attempt.Add(1)
		mu.Lock()
		auths = append(auths, r.Header.Get("Authorization"))
		mu.Unlock()
		if n == 1 {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write([]byte(`{"error":{"message":"rate limited"}}`))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"x","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
	}))
	t.Cleanup(srv.Close)

	var generation atomic.Int64
	src := tokenSourceFunc(func(context.Context) (BearerToken, error) {
		return BearerToken{AccessToken: "tok-gen-" + strconv.FormatInt(generation.Add(1), 10)}, nil
	})
	r, err := NewRouter(context.Background(), RouterConfig{
		Providers: []ProviderEntry{{
			Name:         "chatgpt",
			ProviderType: "openai",
			APIKey:       "static-key",
			BaseURL:      srv.URL,
			Models:       []string{"gpt-4o"},
			TokenSource:  src,
		}},
		MaxRetries:     1,
		InitialBackoff: time.Millisecond,
	}, nil)
	if err != nil {
		t.Fatalf("NewRouter: %v", err)
	}
	if _, err := r.Call(context.Background(), tokenSourceRequest("gpt-4o")); err != nil {
		t.Fatalf("Router.Call: %v", err)
	}
	if got := attempt.Load(); got != 2 {
		t.Fatalf("server attempts = %d, want 2 (one 429, one success)", got)
	}
	if got := generation.Load(); got != 2 {
		t.Fatalf("token resolutions = %d, want 2 (fresh credential per router attempt)", got)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(auths) != 2 || auths[0] != "Bearer tok-gen-1" || auths[1] != "Bearer tok-gen-2" {
		t.Errorf("Authorization per attempt = %v, want [Bearer tok-gen-1, Bearer tok-gen-2]", auths)
	}
}

// TestOpenAIProvider_TokenSource_ErrorAbortsRetryChain pins the TokenSource
// contract's abort semantics end to end: a failing Token() call must surface
// as the ORIGINAL error from ONE provider call — the SDK-internal retry loop
// (which retries a nil-response middleware error and could mask it with a
// later success) is disabled for dynamic-credential clients, the router
// classifies the wrapped credential failure non-retryable, and no HTTP
// request ever reaches the wire.
func TestOpenAIProvider_TokenSource_ErrorAbortsRetryChain(t *testing.T) {
	var hits atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		hits.Add(1)
		_, _ = w.Write([]byte(`{}`))
	}))
	t.Cleanup(srv.Close)

	var calls atomic.Int64
	src := tokenSourceFunc(func(context.Context) (BearerToken, error) {
		calls.Add(1)
		if calls.Load() == 1 {
			return BearerToken{}, errors.New("refresh failed")
		}
		// A subsequent resolution WOULD succeed — the failure must still be
		// terminal for this call (no in-SDK re-resolution).
		return BearerToken{AccessToken: "late"}, nil
	})
	r, err := NewRouter(context.Background(), RouterConfig{
		Providers: []ProviderEntry{{
			Name:         "chatgpt",
			ProviderType: "openai",
			APIKey:       "static-key",
			BaseURL:      srv.URL,
			Models:       []string{"gpt-4o"},
			TokenSource:  src,
		}},
		MaxRetries:     2,
		InitialBackoff: time.Millisecond,
	}, nil)
	if err != nil {
		t.Fatalf("NewRouter: %v", err)
	}
	_, err = r.Call(context.Background(), tokenSourceRequest("gpt-4o"))
	if err == nil {
		t.Fatal("Router.Call must fail when Token() fails")
	}
	if !strings.Contains(err.Error(), "token source") {
		t.Errorf("error = %v, want it to name the token source", err)
	}
	if !strings.Contains(err.Error(), "refresh failed") {
		t.Errorf("error = %v, want the ORIGINAL token source error preserved", err)
	}
	if IsRetryable(err) {
		t.Errorf("error = %v, want non-retryable classification (credential failure aborts the retry chain)", err)
	}
	if got := calls.Load(); got != 1 {
		t.Errorf("token resolutions = %d, want 1 (the failure is terminal, no re-resolution)", got)
	}
	if hits.Load() != 0 {
		t.Errorf("server hits = %d, want 0 (failed token resolution must not reach the wire)", hits.Load())
	}
}

// TestNewRouter_ProviderEntryTokenSourcePlumbing proves the auth-seam fields
// travel the whole way — RouterConfig → ProviderEntry →
// createProviderFromConfig → OpenAIProviderConfig → OpenAIProvider → wire
// headers. An entry with a TokenSource yields a provider whose requests
// carry the token-source Authorization header (not the static APIKey), and
// RequireStreaming reaches the provider struct; an entry without them keeps
// the historical static-credential behavior and the zero RequireStreaming.
func TestNewRouter_ProviderEntryTokenSourcePlumbing(t *testing.T) {
	got, srv := headerRecorder(t)
	src := &staticTokenSource{token: BearerToken{
		AccessToken:  "tok-router",
		ExtraHeaders: map[string]string{"ChatGPT-Account-Id": "acct-router"},
	}}
	r, err := NewRouter(context.Background(), RouterConfig{
		Providers: []ProviderEntry{{
			Name:             "chatgpt",
			ProviderType:     "openai",
			APIKey:           "unused-static",
			BaseURL:          srv.URL,
			Models:           []string{"gpt-4o"},
			TokenSource:      src,
			RequireStreaming: true,
		}},
		MaxRetries: -1, // retries disabled: the recorder answers once
	}, nil)
	if err != nil {
		t.Fatalf("NewRouter: %v", err)
	}
	p, ok := r.providers["chatgpt"].(*OpenAIProvider)
	if !ok {
		t.Fatalf("router provider type = %T, want *OpenAIProvider", r.providers["chatgpt"])
	}
	if !p.requireStreaming {
		t.Error("provider requireStreaming = false, want true (threaded from ProviderEntry)")
	}
	if _, err := p.ChatCompletion(context.Background(), tokenSourceRequest("gpt-4o")); err != nil {
		t.Fatalf("ChatCompletion: %v", err)
	}
	if auth := got.Get("Authorization"); auth != "Bearer tok-router" {
		t.Errorf("Authorization = %q, want %q", auth, "Bearer tok-router")
	}
	if acct := got.Get("ChatGPT-Account-Id"); acct != "acct-router" {
		t.Errorf("ChatGPT-Account-Id = %q, want %q", acct, "acct-router")
	}

	// An entry without the seam fields keeps the zero values and the static
	// credential path.
	body2, srv2 := chatBodyRecorder(t)
	r2, err := NewRouter(context.Background(), RouterConfig{
		Providers: []ProviderEntry{{
			Name:         "lmstudio",
			ProviderType: "openai",
			APIKey:       "sk-plain",
			BaseURL:      srv2.URL,
			Models:       []string{"gpt-4o"},
		}},
		MaxRetries: -1,
	}, nil)
	if err != nil {
		t.Fatalf("NewRouter (plain): %v", err)
	}
	p2, ok := r2.providers["lmstudio"].(*OpenAIProvider)
	if !ok {
		t.Fatalf("router provider type = %T, want *OpenAIProvider", r2.providers["lmstudio"])
	}
	if p2.requireStreaming {
		t.Error("provider requireStreaming = true, want the zero value false")
	}
	if _, err := p2.ChatCompletion(context.Background(), tokenSourceRequest("gpt-4o")); err != nil {
		t.Fatalf("ChatCompletion (plain): %v", err)
	}
	if len(*body2) == 0 {
		t.Error("plain entry request body missing")
	}
}

// TestOpenAIProvider_TokenSource_DelegateProtocols pins the auth-seam matrix
// for the delegated protocols: with a TokenSource configured, the
// ProtocolAnthropic (Claude via the co-located Anthropic delegate) and
// ProtocolGoogle (Gemini via googleCompletion) dispatch paths carry the
// resolved bearer + extra headers and NEVER the static key (x-api-key is
// cleared; the ?key= query parameter is dropped), while a failing Token
// aborts with zero wire requests — exactly like the two native OpenAI
// protocols.
func TestOpenAIProvider_TokenSource_DelegateProtocols(t *testing.T) {
	t.Run("router_anthropic_entry_threads_the_token_source", func(t *testing.T) {
		var mu sync.Mutex
		var auth, apiKey string
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			mu.Lock()
			auth = r.Header.Get("Authorization")
			apiKey = r.Header.Get("x-api-key")
			mu.Unlock()
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"id":"msg_1","type":"message","role":"assistant","model":"claude-3-haiku-20240307","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":1}}`))
		}))
		t.Cleanup(srv.Close)

		// The ROUTER's anthropic-typed entry (not the OpenAI provider's
		// internal delegate): a TokenSource here must reach the provider —
		// before the wiring it was silently dropped and the request rode the
		// static API key.
		r, err := NewRouter(context.Background(), RouterConfig{
			Providers: []ProviderEntry{{
				Name:         "anthropic-gw",
				ProviderType: "anthropic",
				APIKey:       "static-key",
				BaseURL:      srv.URL,
				Models:       []string{"claude-3-haiku-20240307"},
				TokenSource:  &staticTokenSource{token: BearerToken{AccessToken: "tok-router"}},
			}},
		}, nil)
		if err != nil {
			t.Fatalf("NewRouter: %v", err)
		}
		resp, err := r.Call(context.Background(), ChatRequest{
			Model:    "claude-3-haiku-20240307",
			Messages: []Message{{Role: "user", Content: "hi"}},
		})
		if err != nil {
			t.Fatalf("router Call: %v", err)
		}
		if resp == nil || resp.Message.Content != "ok" {
			t.Fatalf("response = %+v, want the endpoint's reply", resp)
		}
		mu.Lock()
		defer mu.Unlock()
		if auth != "Bearer tok-router" {
			t.Errorf("Authorization = %q, want %q (the entry's TokenSource must reach the anthropic-typed provider)", auth, "Bearer tok-router")
		}
		if apiKey != "" {
			t.Errorf("x-api-key = %q, want empty (static key cleared by the dynamic credential)", apiKey)
		}
	})

	t.Run("anthropic_bearer_and_no_static_key", func(t *testing.T) {
		var mu sync.Mutex
		var auth, apiKey string
		var acct string
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			mu.Lock()
			auth = r.Header.Get("Authorization")
			apiKey = r.Header.Get("x-api-key")
			acct = r.Header.Get("ChatGPT-Account-Id")
			mu.Unlock()
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"id":"msg_1","type":"message","role":"assistant","model":"claude-3-haiku-20240307","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":1}}`))
		}))
		t.Cleanup(srv.Close)

		p, err := NewOpenAIProvider(OpenAIProviderConfig{
			Name:    "zen",
			APIKey:  "static-key",
			BaseURL: srv.URL,
			TokenSource: &staticTokenSource{token: BearerToken{
				AccessToken:  "tok-delegate",
				ExtraHeaders: map[string]string{"ChatGPT-Account-Id": "acct-1"},
			}},
		})
		if err != nil {
			t.Fatalf("NewOpenAIProvider: %v", err)
		}
		if _, err := p.ChatCompletion(context.Background(), ChatRequest{
			Model:    "claude-3-haiku-20240307",
			Protocol: ProtocolAnthropic,
			Messages: []Message{{Role: "user", Content: "hi"}},
		}); err != nil {
			t.Fatalf("ChatCompletion(anthropic): %v", err)
		}
		mu.Lock()
		defer mu.Unlock()
		if auth != "Bearer tok-delegate" {
			t.Errorf("Authorization = %q, want %q", auth, "Bearer tok-delegate")
		}
		if apiKey != "" {
			t.Errorf("x-api-key = %q, want empty (static key cleared by the dynamic credential)", apiKey)
		}
		if acct != "acct-1" {
			t.Errorf("ChatGPT-Account-Id = %q, want %q", acct, "acct-1")
		}
	})

	t.Run("google_bearer_and_no_static_key", func(t *testing.T) {
		var mu sync.Mutex
		var auth string
		var keyParam string
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			mu.Lock()
			auth = r.Header.Get("Authorization")
			keyParam = r.URL.Query().Get("key")
			mu.Unlock()
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"candidates":[{"content":{"parts":[{"text":"ok"}],"role":"model"}}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1}}`))
		}))
		t.Cleanup(srv.Close)

		p, err := NewOpenAIProvider(OpenAIProviderConfig{
			Name:        "zen",
			APIKey:      "static-key",
			BaseURL:     srv.URL,
			TokenSource: &staticTokenSource{token: BearerToken{AccessToken: "tok-google"}},
		})
		if err != nil {
			t.Fatalf("NewOpenAIProvider: %v", err)
		}
		if _, err := p.ChatCompletion(context.Background(), ChatRequest{
			Model:    "gemini-2.5-pro",
			Protocol: ProtocolGoogle,
			Messages: []Message{{Role: "user", Content: "hi"}},
		}); err != nil {
			t.Fatalf("ChatCompletion(google): %v", err)
		}
		mu.Lock()
		defer mu.Unlock()
		if auth != "Bearer tok-google" {
			t.Errorf("Authorization = %q, want %q", auth, "Bearer tok-google")
		}
		if keyParam != "" {
			t.Errorf("?key= = %q, want empty (static key dropped by the dynamic credential)", keyParam)
		}
	})

	t.Run("token_error_aborts_delegates_without_wire", func(t *testing.T) {
		for _, tc := range []struct {
			name     string
			model    string
			protocol APIProtocol
		}{
			{"anthropic", "claude-3-haiku-20240307", ProtocolAnthropic},
			{"google", "gemini-2.5-pro", ProtocolGoogle},
		} {
			t.Run(tc.name, func(t *testing.T) {
				var hits atomic.Int64
				srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
					hits.Add(1)
					_, _ = w.Write([]byte(`{}`))
				}))
				t.Cleanup(srv.Close)

				p, err := NewOpenAIProvider(OpenAIProviderConfig{
					Name:        "zen",
					APIKey:      "static-key",
					BaseURL:     srv.URL,
					TokenSource: erroringTokenSource{err: errors.New("refresh failed")},
				})
				if err != nil {
					t.Fatalf("NewOpenAIProvider: %v", err)
				}
				_, err = p.ChatCompletion(context.Background(), ChatRequest{
					Model:    tc.model,
					Protocol: tc.protocol,
					Messages: []Message{{Role: "user", Content: "hi"}},
				})
				if err == nil {
					t.Fatal("ChatCompletion must fail when Token() fails")
				}
				if !strings.Contains(err.Error(), "token source") {
					t.Errorf("error = %v, want it to name the token source", err)
				}
				if hits.Load() != 0 {
					t.Errorf("server hits = %d, want 0 (failed token resolution must not reach the wire)", hits.Load())
				}
			})
		}
	})
}
