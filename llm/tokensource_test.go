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

// TestOpenAIProvider_TokenSource_FreshTokenPerRetryAttempt verifies the
// middleware re-resolves credentials for every SDK-internal retry attempt:
// a 429-then-200 endpoint sees the freshly generated token of attempt two
// from a single ChatCompletion call.
func TestOpenAIProvider_TokenSource_FreshTokenPerRetryAttempt(t *testing.T) {
	var attempt atomic.Int64
	var lastAuth atomic.Value // string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := attempt.Add(1)
		lastAuth.Store(r.Header.Get("Authorization"))
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
	if got := attempt.Load(); got != 2 {
		t.Fatalf("server attempts = %d, want 2 (one 429, one success)", got)
	}
	if auth, _ := lastAuth.Load().(string); auth != "Bearer tok-gen-2" {
		t.Errorf("retry attempt Authorization = %q, want freshly resolved %q", auth, "Bearer tok-gen-2")
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
