package mcp

// Regression tests for the origin-pinned redirect policy on the HTTP
// transports (CVE-2026-116685 class): a trusted-but-redirecting endpoint must
// never carry the operator's custom headers — or, on 307/308, the JSON-RPC
// body — to an origin other than the configured endpoint's own.

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mark3labs/mcp-go/mcp"
	mcpserver "github.com/mark3labs/mcp-go/server"
)

// TestOriginPinnedRedirects_Policy exercises the redirect decision table
// directly: same-origin hops are allowed, everything that leaves the origin
// (host, port, scheme downgrade, non-HTTP scheme, hop cap, unparsable base
// URL) is refused. The single exception is an http→https upgrade on the same
// host, mirroring the reference MCP SDK's post-CVE fix.
func TestOriginPinnedRedirects_Policy(t *testing.T) {
	tests := []struct {
		name    string
		initial string
		target  string
		via     int
		wantErr string // empty — the redirect is allowed
	}{
		{
			name:    "same origin",
			initial: "https://api.example.com/mcp",
			target:  "https://api.example.com/mcp2",
		},
		{
			name:    "explicit default port equals implicit",
			initial: "https://api.example.com/mcp",
			target:  "https://api.example.com:443/mcp",
		},
		{
			name:    "host case-insensitive",
			initial: "https://api.example.com/mcp",
			target:  "https://API.Example.COM/mcp",
		},
		{
			name:    "different port",
			initial: "https://api.example.com/mcp",
			target:  "https://api.example.com:8443/mcp",
			wantErr: "refused",
		},
		{
			name:    "different host",
			initial: "https://api.example.com/mcp",
			target:  "https://evil.example.com/mcp",
			wantErr: "refused",
		},
		{
			name:    "suffix-spoofed host",
			initial: "https://api.example.com/mcp",
			target:  "https://api.example.com.evil.com/mcp",
			wantErr: "refused",
		},
		{
			name:    "subdomain of the origin host",
			initial: "https://api.example.com/mcp",
			target:  "https://evil.api.example.com/mcp",
			wantErr: "refused",
		},
		{
			name:    "https to http downgrade",
			initial: "https://api.example.com/mcp",
			target:  "http://api.example.com/mcp",
			wantErr: "downgrade",
		},
		{
			name:    "http to https upgrade",
			initial: "http://api.example.com/mcp",
			target:  "https://api.example.com/mcp",
		},
		{
			name:    "http to https upgrade on another port",
			initial: "http://api.example.com/mcp",
			target:  "https://api.example.com:8443/mcp",
		},
		{
			name:    "non-http scheme",
			initial: "https://api.example.com/mcp",
			target:  "ftp://api.example.com/mcp",
			wantErr: "refused",
		},
		{
			name:    "redirect cap",
			initial: "https://api.example.com/mcp",
			target:  "https://api.example.com/mcp2",
			via:     maxHTTPRedirects,
			wantErr: "stopped after",
		},
		{
			name:    "unparsable initial URL refuses everything",
			initial: "://bad",
			target:  "https://api.example.com/mcp",
			wantErr: "unavailable",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			policy := originPinnedRedirects(tt.initial)
			u, err := url.Parse(tt.target)
			if err != nil {
				t.Fatalf("url.Parse(%q) = %v, want nil", tt.target, err)
			}
			err = policy(&http.Request{URL: u, Header: http.Header{}}, make([]*http.Request, tt.via))
			switch {
			case tt.wantErr == "" && err != nil:
				t.Errorf("originPinnedRedirects(%q) target %q = %v, want allowed", tt.initial, tt.target, err)
			case tt.wantErr != "" && err == nil:
				t.Errorf("originPinnedRedirects(%q) target %q = nil, want error containing %q", tt.initial, tt.target, tt.wantErr)
			case tt.wantErr != "" && !strings.Contains(err.Error(), tt.wantErr):
				t.Errorf("originPinnedRedirects(%q) target %q = %v, want error containing %q", tt.initial, tt.target, err, tt.wantErr)
			}
		})
	}
}

// TestOriginPinnedClient_CrossOriginRedirectRefused verifies the client-level
// behavior against real httptest servers: a 307 to another origin (here:
// another port on the same loopback host) is refused, the redirect target is
// never contacted, and the caller's client is not mutated.
func TestOriginPinnedClient_CrossOriginRedirectRefused(t *testing.T) {
	var targetHits atomic.Int32
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		targetHits.Add(1)
		w.WriteHeader(http.StatusOK)
	}))
	defer target.Close()

	origin := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, target.URL+"/steal", http.StatusTemporaryRedirect)
	}))
	defer origin.Close()

	caller := &http.Client{Timeout: 5 * time.Second}
	pinned := originPinnedClient(caller, origin.URL)

	req, err := http.NewRequestWithContext(context.Background(), http.MethodPost, origin.URL, strings.NewReader(`{"jsonrpc":"2.0","method":"initialize"}`))
	if err != nil {
		t.Fatalf("http.NewRequestWithContext = %v, want nil", err)
	}
	req.Header.Set("X-API-Key", "secret")
	resp, err := pinned.Do(req)
	if err == nil {
		_ = resp.Body.Close()
		t.Fatal("originPinnedClient.Do with a cross-origin redirect = nil error, want refusal")
	}
	if !strings.Contains(err.Error(), "refused") {
		t.Errorf("originPinnedClient.Do error = %v, want it to contain %q", err, "refused")
	}
	if targetHits.Load() != 0 {
		t.Errorf("redirect target was hit %d times, want 0", targetHits.Load())
	}
	if caller.CheckRedirect != nil {
		t.Error("caller client was mutated; CheckRedirect must remain nil on the original")
	}
}

// TestOriginPinnedClient_SameOriginRedirectFollowed verifies that a
// same-origin 307 is followed with the full request replayed — headers and
// body — so legitimate endpoints behind same-origin redirects keep working.
func TestOriginPinnedClient_SameOriginRedirectFollowed(t *testing.T) {
	var captured struct {
		sync.Mutex
		method, key, body string
	}
	origin := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/redirect" {
			http.Redirect(w, r, "/final", http.StatusTemporaryRedirect)
			return
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("reading replayed body: %v", err)
		}
		captured.Lock()
		captured.method, captured.key, captured.body = r.Method, r.Header.Get("X-API-Key"), string(body)
		captured.Unlock()
		w.WriteHeader(http.StatusOK)
	}))
	defer origin.Close()

	pinned := originPinnedClient(&http.Client{Timeout: 5 * time.Second}, origin.URL)
	req, err := http.NewRequestWithContext(context.Background(), http.MethodPost, origin.URL+"/redirect", strings.NewReader(`{"jsonrpc":"2.0"}`))
	if err != nil {
		t.Fatalf("http.NewRequestWithContext = %v, want nil", err)
	}
	req.Header.Set("X-API-Key", "secret")
	resp, err := pinned.Do(req)
	if err != nil {
		t.Fatalf("originPinnedClient.Do with a same-origin redirect = %v, want nil", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("final status = %d, want %d", resp.StatusCode, http.StatusOK)
	}
	captured.Lock()
	defer captured.Unlock()
	if captured.method != http.MethodPost {
		t.Errorf("replayed method = %q, want %q", captured.method, http.MethodPost)
	}
	if captured.key != "secret" {
		t.Errorf("replayed X-API-Key = %q, want %q", captured.key, "secret")
	}
	if captured.body != `{"jsonrpc":"2.0"}` {
		t.Errorf("replayed body = %q, want %q", captured.body, `{"jsonrpc":"2.0"}`)
	}
}

// TestServer_Connect_HTTPCrossOriginRedirectRefused verifies the fix at the
// connectHTTP boundary with the SDK-default client path: BOTH transport legs
// (Streamable HTTP first, then the SSE fallback) refuse the cross-origin
// redirect, so Connect fails and the attacker-chosen target is never contacted.
func TestServer_Connect_HTTPCrossOriginRedirectRefused(t *testing.T) {
	var targetHits atomic.Int32
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		targetHits.Add(1)
		w.WriteHeader(http.StatusOK)
	}))
	defer target.Close()

	origin := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, target.URL+"/mcp", http.StatusTemporaryRedirect)
	}))
	defer origin.Close()

	s := newServer("x-origin")
	err := s.Connect(context.Background(), ServerConfig{
		Transport: "http",
		URL:       origin.URL,
		Headers:   map[string]string{"X-API-Key": "secret"},
		Timeout:   10 * time.Second,
	})
	if err == nil {
		_ = s.Close()
		t.Fatal("Connect with a cross-origin redirecting endpoint = nil error, want refusal")
	}
	if !strings.Contains(err.Error(), "refused") {
		t.Errorf("Connect error = %v, want it to contain %q", err, "refused")
	}
	if targetHits.Load() != 0 {
		t.Errorf("redirect target was hit %d times, want 0", targetHits.Load())
	}
}

// TestServer_Connect_SuppliedClientCrossOriginRedirectRefused verifies the
// host-supplied client path (c0wrk's proxy client is threaded through
// ServerConfig.HTTPClient): the redirect pinning applies on a shallow copy —
// the connection is refused and the caller's client keeps CheckRedirect nil.
func TestServer_Connect_SuppliedClientCrossOriginRedirectRefused(t *testing.T) {
	var targetHits atomic.Int32
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		targetHits.Add(1)
		w.WriteHeader(http.StatusOK)
	}))
	defer target.Close()

	origin := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, target.URL+"/mcp", http.StatusTemporaryRedirect)
	}))
	defer origin.Close()

	supplied := &http.Client{Timeout: 10 * time.Second}
	s := newServer("x-origin-supplied")
	err := s.Connect(context.Background(), ServerConfig{
		Transport:  "http",
		URL:        origin.URL,
		Headers:    map[string]string{"X-API-Key": "secret"},
		HTTPClient: supplied,
		Timeout:    10 * time.Second,
	})
	if err == nil {
		_ = s.Close()
		t.Fatal("Connect with a supplied client and a cross-origin redirect = nil error, want refusal")
	}
	if !strings.Contains(err.Error(), "refused") {
		t.Errorf("Connect error = %v, want it to contain %q", err, "refused")
	}
	if targetHits.Load() != 0 {
		t.Errorf("redirect target was hit %d times, want 0", targetHits.Load())
	}
	if supplied.CheckRedirect != nil {
		t.Error("supplied client was mutated; CheckRedirect must remain nil on the original")
	}
}

// TestServer_Connect_HTTPSameOriginRedirectConnects verifies that a legitimate
// same-origin redirect does not break the transport: the endpoint 308-redirects
// the initialize POST (body and headers replayed) to a real Streamable HTTP
// MCP server on the same origin, and the connection reaches full function.
func TestServer_Connect_HTTPSameOriginRedirectConnects(t *testing.T) {
	mcpSrv := mcpserver.NewMCPServer("redirected", "1.0.0")
	mcpSrv.AddTool(mcp.NewTool("echo", mcp.WithDescription("echo tool")),
		func(_ context.Context, _ mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			return mcp.NewToolResultText("ok"), nil
		})
	streamable := mcpserver.NewStreamableHTTPServer(mcpSrv)

	origin := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/entry" {
			http.Redirect(w, r, "/mcp", http.StatusPermanentRedirect)
			return
		}
		streamable.ServeHTTP(w, r)
	}))
	defer origin.Close()

	s := newServer("same-origin")
	if err := s.Connect(context.Background(), ServerConfig{
		Transport: "http",
		URL:       origin.URL + "/entry",
		Timeout:   10 * time.Second,
	}); err != nil {
		t.Fatalf("Connect behind a same-origin redirect = %v, want nil", err)
	}
	defer func() { _ = s.Close() }()

	if !s.IsConnected() {
		t.Error("server must be connected after a same-origin redirect handshake")
	}
	if err := s.DiscoverTools(context.Background()); err != nil {
		t.Fatalf("DiscoverTools behind a same-origin redirect = %v, want nil", err)
	}
	if tools := s.Tools(); len(tools) != 1 || tools[0].Name != "echo" {
		t.Errorf("Tools() = %+v, want a single \"echo\" tool", tools)
	}
	res, err := s.CallTool(context.Background(), "echo", nil)
	if err != nil {
		t.Fatalf("CallTool(echo) = %v, want nil", err)
	}
	if len(res.Content) == 0 {
		t.Fatalf("CallTool(echo) result = %+v, want one text content part", res)
	}
	tc, ok := mcp.AsTextContent(res.Content[0])
	if !ok || tc.Text != "ok" {
		t.Errorf("CallTool(echo) content = %#v, want text \"ok\"", res.Content[0])
	}
}
