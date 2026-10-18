package websearch

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// --- SearXNG Search Provider ---

// DefaultSearxngBaseURL is the default address of a self-hosted SearXNG
// instance (the upstream docker-compose default). The c0wrk config carries
// the same default for search.base_url.
const DefaultSearxngBaseURL = "http://127.0.0.1:8888"

// SearXNGProvider implements SearchProvider against a self-hosted SearXNG
// instance's JSON API. A self-hosted instance gives a ToS-clean,
// rate-limit-free search path (we control the IP and the query rate; SearXNG
// aggregates the upstream engines server-side), unlike the fragile
// DuckDuckGo HTML scraping.
//
// The instance must enable JSON output: settings.yml →
// `search.formats: [html, json]`. Many public instances disable it (and some
// run their own bot checks), so self-hosting is the supported path.
type SearXNGProvider struct {
	baseURL string
	client  *http.Client
}

// NewSearXNGProviderWithClient creates a new SearXNGProvider pointed at the
// given instance base URL (empty selects DefaultSearxngBaseURL) with the
// given timeout and optional HTTP client. If client is nil, a default client
// with the specified timeout is used.
func NewSearXNGProviderWithClient(baseURL string, timeout time.Duration, client *http.Client) *SearXNGProvider {
	baseURL = normalizeSearxngBaseURL(baseURL)

	// Refuse to follow redirects, matching the other search providers: the
	// stock net/http policy silently re-issues the request to any Location —
	// up to 10 hops — so a misconfigured or hijacked base URL could make the
	// host perform a server-side request to an arbitrary URL. This
	// protection must apply to a caller-supplied client too: we make a
	// shallow copy (never mutating the caller's) and set CheckRedirect
	// unconditionally, so the 3xx response is surfaced to Search instead of
	// being followed.
	noRedirect := func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	if client == nil {
		client = &http.Client{Timeout: timeout, CheckRedirect: noRedirect}
	} else {
		c := *client
		c.CheckRedirect = noRedirect
		client = &c
	}
	return &SearXNGProvider{
		baseURL: baseURL,
		client:  client,
	}
}

// normalizeSearxngBaseURL trims whitespace and trailing slashes; an empty
// value selects DefaultSearxngBaseURL.
func normalizeSearxngBaseURL(raw string) string {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return DefaultSearxngBaseURL
	}
	return strings.TrimRight(raw, "/")
}

// Name returns the provider name.
func (p *SearXNGProvider) Name() string { return "searxng" }

// SetBaseURL allows setting a custom base URL (useful for testing).
func (p *SearXNGProvider) SetBaseURL(rawURL string) { p.baseURL = normalizeSearxngBaseURL(rawURL) }

// searxngResponse represents the JSON response from the SearXNG search API.
type searxngResponse struct {
	Results []searxngResult `json:"results"`
}

// searxngResult represents a single search result from SearXNG.
type searxngResult struct {
	Title   string `json:"title"`
	URL     string `json:"url"`
	Content string `json:"content"`
}

// Search performs a web search using the SearXNG JSON API.
//
// maxResults caps the number of returned results client-side (the JSON API
// has no count parameter). A negative count is not an error: like the
// DuckDuckGo loop-guard, it degrades to an empty result — never a
// negative-cap make() panic — so a sign-slipped caller configuration cannot
// crash the host.
func (p *SearXNGProvider) Search(ctx context.Context, query string, maxResults int) ([]SearchResult, error) {
	if maxResults < 0 {
		maxResults = 0
	}
	if p.baseURL == "" {
		return nil, errors.New("searxng: base URL is not configured")
	}

	reqURL := p.baseURL + "/search?" + url.Values{"q": {query}, "format": {"json"}}.Encode()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, reqURL, http.NoBody)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Accept", "application/json")

	resp, err := p.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	// Read the (capped) body once so both the status rendering and the JSON
	// decode use it.
	body, readErr := io.ReadAll(limitBody(resp.Body))
	if readErr != nil {
		return nil, fmt.Errorf("HTTP %d (could not read body: %w)", resp.StatusCode, readErr)
	}

	// Check status code
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
	}

	// Parse response
	var searxResp searxngResponse
	if err := json.Unmarshal(body, &searxResp); err != nil {
		// A 200 with a non-JSON body almost always means the instance has
		// JSON output disabled (or an anti-bot page) — point the operator at
		// the fix instead of a bare decode error.
		if bytes.Contains(bytes.TrimSpace(body), []byte("<")) {
			return nil, errors.New("searxng: the instance returned HTML instead of JSON — enable JSON output in settings.yml (search.formats: [html, json]) and prefer a self-hosted instance")
		}
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	// Convert to the generic SearchResult, honouring maxResults client-side
	// (the JSON API has no count parameter).
	n := min(len(searxResp.Results), maxResults)
	results := make([]SearchResult, 0, n)
	for _, r := range searxResp.Results[:n] {
		results = append(results, SearchResult{
			Title:   r.Title,
			URL:     r.URL,
			Snippet: r.Content,
		})
	}
	return results, nil
}
