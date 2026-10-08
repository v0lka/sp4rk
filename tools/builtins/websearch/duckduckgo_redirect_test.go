package websearch

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// TestDuckDuckGoProvider_RedirectNotFollowed verifies the client refuses to
// follow redirects (review finding 8). DuckDuckGo was the only search
// provider without the hardening: the stock net/http policy silently
// re-issued the request to any Location, making the host perform a
// server-side request to an arbitrary (possibly internal) URL. The client
// must surface the redirect instead.
func TestDuckDuckGoProvider_RedirectNotFollowed(t *testing.T) {
	target := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
		t.Error("redirect target was hit; the client must not follow redirects")
	}))
	defer target.Close()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, target.URL, http.StatusFound)
	}))
	defer srv.Close()

	p := NewDuckDuckGoProviderWithClient(30*time.Second, nil)
	p.SetBaseURL(srv.URL)

	_, err := p.Search(context.Background(), "test", 5)
	if err == nil {
		t.Fatal("expected an error for a redirect response, got nil")
	}
	if !strings.Contains(err.Error(), "HTTP 302") {
		t.Errorf("error = %q, want it to contain %q (redirect surfaced, not followed)", err.Error(), "HTTP 302")
	}
}

// TestDuckDuckGoProvider_CallerClientRedirectEnforced verifies that a
// caller-supplied client ALSO gets the redirect protection (the stock policy
// would otherwise follow a redirect and hand the response of an
// attacker-chosen host back as the search result). It also verifies the
// caller's client is not mutated.
func TestDuckDuckGoProvider_CallerClientRedirectEnforced(t *testing.T) {
	target := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
		t.Error("redirect target was hit; caller client must not follow redirects")
	}))
	defer target.Close()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, target.URL, http.StatusFound)
	}))
	defer srv.Close()

	// A caller client with no CheckRedirect (would follow redirects by default).
	callerClient := &http.Client{Timeout: 30 * time.Second}
	p := NewDuckDuckGoProviderWithClient(30*time.Second, callerClient)
	p.SetBaseURL(srv.URL)

	_, err := p.Search(context.Background(), "test", 5)
	if err == nil {
		t.Fatal("expected an error for a redirect response, got nil")
	}
	if !strings.Contains(err.Error(), "HTTP 302") {
		t.Errorf("error = %q, want it to contain %q (redirect surfaced, not followed)", err.Error(), "HTTP 302")
	}

	// The caller's original client must not have been mutated.
	if callerClient.CheckRedirect != nil {
		t.Error("caller client was mutated; CheckRedirect must remain nil on the original")
	}
}
