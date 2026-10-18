package websearch

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/v0lka/sp4rk/tools/builtins"
)

func TestSearXNGProvider_Search(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			t.Errorf("Expected GET request, got %s", r.Method)
		}
		if q := r.URL.Query().Get("q"); q != "golang testing" {
			t.Errorf("Expected query 'golang testing', got %s", q)
		}
		if format := r.URL.Query().Get("format"); format != "json" {
			t.Errorf("Expected format 'json', got %s", format)
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"results": [
				{"title": "Go Testing", "url": "https://example.com/testing", "content": "Go has built-in support for testing."},
				{"title": "Go Documentation", "url": "https://go.dev/doc/", "content": "Learn about Go."}
			]
		}`))
	}))
	defer server.Close()

	provider := NewSearXNGProviderWithClient(server.URL, 30*time.Second, nil)

	results, err := provider.Search(context.Background(), "golang testing", 5)
	if err != nil {
		t.Fatalf("Search() returned error: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("Expected 2 results, got %d", len(results))
	}
	if results[0].Title != "Go Testing" || results[0].URL != "https://example.com/testing" || results[0].Snippet != "Go has built-in support for testing." {
		t.Errorf("unexpected first result: %+v", results[0])
	}
	if results[1].Title != "Go Documentation" || results[1].URL != "https://go.dev/doc/" {
		t.Errorf("unexpected second result: %+v", results[1])
	}
}

func TestSearXNGProvider_MaxResults(t *testing.T) {
	payload := `{"results": [
		{"title": "Result", "url": "https://example.com/1", "content": "snippet"},
		{"title": "Result", "url": "https://example.com/2", "content": "snippet"},
		{"title": "Result", "url": "https://example.com/3", "content": "snippet"}
	]}`

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(payload))
	}))
	defer server.Close()

	provider := NewSearXNGProviderWithClient(server.URL, 30*time.Second, nil)
	results, err := provider.Search(context.Background(), "test", 2)
	if err != nil {
		t.Fatalf("Search() returned error: %v", err)
	}
	if len(results) != 2 {
		t.Errorf("Expected 2 results (maxResults limit), got %d", len(results))
	}
}

func TestSearXNGProvider_HTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte("internal server error"))
	}))
	defer server.Close()

	provider := NewSearXNGProviderWithClient(server.URL, 30*time.Second, nil)
	_, err := provider.Search(context.Background(), "test", 5)
	if err == nil {
		t.Fatal("expected error for HTTP 500, got nil")
	}
	if !strings.Contains(err.Error(), "HTTP 500") {
		t.Errorf("error = %q, want it to contain %q", err.Error(), "HTTP 500")
	}
}

func TestSearXNGProvider_HTMLBodyPointsAtJSONSetting(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/html")
		_, _ = w.Write([]byte("<!DOCTYPE html><html><body>search UI</body></html>"))
	}))
	defer server.Close()

	provider := NewSearXNGProviderWithClient(server.URL, 30*time.Second, nil)
	_, err := provider.Search(context.Background(), "test", 5)
	if err == nil {
		t.Fatal("expected error for an HTML response body, got nil")
	}
	if !strings.Contains(err.Error(), "search.formats: [html, json]") {
		t.Errorf("error = %q, want it to point at the JSON-output setting", err.Error())
	}
}

func TestSearXNGProvider_InvalidJSON(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"results": [`))
	}))
	defer server.Close()

	provider := NewSearXNGProviderWithClient(server.URL, 30*time.Second, nil)
	_, err := provider.Search(context.Background(), "test", 5)
	if err == nil {
		t.Fatal("expected error for invalid JSON, got nil")
	}
	if !strings.Contains(err.Error(), "failed to parse response") {
		t.Errorf("error = %q, want it to contain %q", err.Error(), "failed to parse response")
	}
}

func TestSearXNGProvider_RedirectNotFollowed(t *testing.T) {
	target := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
		t.Error("redirect target was hit; the client must not follow redirects")
	}))
	defer target.Close()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, target.URL, http.StatusFound)
	}))
	defer srv.Close()

	provider := NewSearXNGProviderWithClient(srv.URL, 30*time.Second, nil)
	_, err := provider.Search(context.Background(), "test", 5)
	if err == nil {
		t.Fatal("expected an error for a redirect response, got nil")
	}
	if !strings.Contains(err.Error(), "HTTP 302") {
		t.Errorf("error = %q, want it to contain %q (redirect surfaced, not followed)", err.Error(), "HTTP 302")
	}
}

func TestSearXNGProvider_CallerClientRedirectEnforced(t *testing.T) {
	target := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
		t.Error("redirect target was hit; caller client must not follow redirects")
	}))
	defer target.Close()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, target.URL, http.StatusFound)
	}))
	defer srv.Close()

	callerClient := &http.Client{Timeout: 30 * time.Second}
	provider := NewSearXNGProviderWithClient(srv.URL, 30*time.Second, callerClient)
	_, err := provider.Search(context.Background(), "test", 5)
	if err == nil {
		t.Fatal("expected an error for a redirect response, got nil")
	}
	if callerClient.CheckRedirect != nil {
		t.Error("caller client was mutated; CheckRedirect must remain nil on the original")
	}
}

func TestNormalizeSearxngBaseURL(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{name: "empty selects the default", input: "", want: DefaultSearxngBaseURL},
		{name: "whitespace selects the default", input: "   ", want: DefaultSearxngBaseURL},
		{name: "trailing slash trimmed", input: "http://127.0.0.1:8888/", want: "http://127.0.0.1:8888"},
		{name: "value kept as-is", input: "https://searx.internal.example.com", want: "https://searx.internal.example.com"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := normalizeSearxngBaseURL(tc.input); got != tc.want {
				t.Errorf("normalizeSearxngBaseURL(%q) = %q, want %q", tc.input, got, tc.want)
			}
		})
	}
}

// TestSearXNGTool_Execute verifies the tool-level wiring end to end against
// a fake instance.
func TestSearXNGTool_Execute(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"results": [{"title": "Answer", "url": "https://example.com/a", "content": "content"}]}`))
	}))
	defer server.Close()

	tool := NewTool(NewSearXNGProviderWithClient(server.URL, 30*time.Second, nil), builtins.DefaultWebSearchLimits())
	input, _ := json.Marshal(map[string]any{"query": "test", "max_results": 5})
	result, err := tool.Execute(context.Background(), input)
	if err != nil {
		t.Fatalf("Execute() returned error: %v", err)
	}
	if result.IsError {
		t.Fatalf("Execute() returned IsError=true: %s", result.Content)
	}
	if !strings.Contains(result.Content, "Answer") {
		t.Errorf("expected the formatted result to contain the title, got: %s", result.Content)
	}
}

// TestSearXNGProvider_RealSearch is the integration test against a real
// self-hosted instance; it is env-guarded like the other provider
// integration tests (set SEARXNG_TEST_URL, e.g.
// SEARXNG_TEST_URL=http://127.0.0.1:8888; the instance must enable JSON
// output: settings.yml → search.formats: [html, json]).
func TestSearXNGProvider_RealSearch(t *testing.T) {
	instance := os.Getenv("SEARXNG_TEST_URL")
	if instance == "" {
		t.Skip("Skipping integration test: SEARXNG_TEST_URL environment variable not set")
	}

	provider := NewSearXNGProviderWithClient(instance, 30*time.Second, nil)
	results, err := provider.Search(context.Background(), "golang programming language", 3)
	if err != nil {
		t.Fatalf("Search() returned error: %v", err)
	}
	if len(results) == 0 {
		t.Error("Expected results for 'golang programming language'")
	}
}
