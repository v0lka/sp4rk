package websearch

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"math/rand/v2"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"golang.org/x/net/html"
)

// --- DuckDuckGo Search Provider ---

const (
	// ddgAnomalySignature identifies DuckDuckGo's anti-bot "anomaly" page
	// ("Unfortunately, bots use DuckDuckGo too…"). It appears behind a 202
	// Accepted status and, occasionally, behind a 200 — either way the page
	// carries an image CAPTCHA instead of results and must surface as an
	// explicit error, never as a silent empty result (see the 202 handling in
	// DuckDuckGoProvider.attempt).
	ddgAnomalySignature = "bots use DuckDuckGo"

	// ddgMinRequestInterval is the minimum spacing between consecutive
	// requests a single provider instance makes, before jitter. DuckDuckGo's
	// anti-bot flag behaves like a sliding per-IP quota (GET died after ~3–4
	// requests in the same window), so the scraper paces itself.
	ddgMinRequestInterval = 4 * time.Second

	// ddgBaseBackoff is the first backoff step before a retry of an anomaly
	// response; each subsequent step doubles it. Bounded by ddgMaxAttempts.
	ddgBaseBackoff = 2 * time.Second

	// ddgMaxAttempts is the total number of attempts (initial request plus
	// bounded retries) for one Search call. After exhausting it, the anomaly
	// response surfaces as the explicit error below.
	ddgMaxAttempts = 3
)

// ddgAnomalyError marks DuckDuckGo's anti-bot anomaly/CAPTCHA challenge. It
// drives the bounded retry loop: only anomaly responses are retried, every
// other error surfaces immediately.
type ddgAnomalyError struct{ status int }

func (e *ddgAnomalyError) Error() string {
	return fmt.Sprintf(
		"duckduckgo: rate limited (anomaly/CAPTCHA challenge, HTTP %d) — DuckDuckGo is throttling this IP; reduce the query rate, or switch to an API-backed provider (brave/tavily/exa) or a self-hosted SearXNG instance",
		e.status,
	)
}

// requestPacer enforces a minimum interval (plus jitter) between consecutive
// requests issued by one provider instance, so the scraper does not hammer
// DuckDuckGo into flagging the egress IP. Acquire blocks (via the caller's
// wait function) until the slot for the next request is due.
type requestPacer struct {
	mu       sync.Mutex
	next     time.Time
	interval time.Duration
	// jitterFn returns a float in [0,1); never nil on a constructed pacer
	// (the provider delegates its own jitter seam into it, so tests pin one
	// field).
	jitterFn func() float64
}

// acquire reserves the next request slot and waits (through wait) for the
// remaining interval. The first acquisition of a fresh pacer never waits.
func (p *requestPacer) acquire(ctx context.Context, wait func(context.Context, time.Duration) error) error {
	p.mu.Lock()
	now := time.Now()
	d := p.next.Sub(now)
	if d < 0 {
		d = 0
	}
	p.next = now.Add(d + p.interval + p.jitterSpan())
	p.mu.Unlock()
	if d == 0 {
		return nil
	}
	return wait(ctx, d)
}

// jitterSpan returns the randomized extra spacing in [0, interval/2).
func (p *requestPacer) jitterSpan() time.Duration {
	return time.Duration(p.jitterFn() * float64(p.interval) / 2)
}

// waitCtx is the production wait function: it sleeps for d while honouring
// context cancellation. Tests replace it with a recording fake (virtual
// time) so the backoff and pacing paths stay deterministic without sleeps.
func waitCtx(ctx context.Context, d time.Duration) error {
	if d <= 0 {
		return ctx.Err()
	}
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

// DuckDuckGoProvider implements SearchProvider using DuckDuckGo HTML search.
// Note: DuckDuckGo does not have an official API, so we scrape the HTML
// version. This is best-effort by design: the anti-bot anomaly challenge can
// still flag an IP; the provider fails loudly when it does instead of
// reporting a silent empty result.
type DuckDuckGoProvider struct {
	baseURL string
	client  *http.Client

	pacer requestPacer
	// maxAttempts bounds the retry loop for anomaly responses.
	maxAttempts int
	// wait sleeps/backsoff between attempts; waitCtx in production, a
	// recording fake in tests.
	wait func(ctx context.Context, d time.Duration) error
	// jitter returns a float in [0,1); nil means math/rand/v2.Float64. Used
	// for pacing and backoff randomization; a seam for deterministic tests.
	jitter func() float64
}

// NewDuckDuckGoProviderWithClient creates a new DuckDuckGoProvider with the given timeout
// and optional HTTP client. If client is nil, a default client with the specified timeout is used.
func NewDuckDuckGoProviderWithClient(timeout time.Duration, client *http.Client) *DuckDuckGoProvider {
	// Refuse to follow redirects, matching the other search providers. The
	// stock net/http policy silently re-issues the request to any Location —
	// up to 10 hops — so a compromised or MITM'd endpoint (or a custom
	// SetBaseURL) could make the host perform a server-side request to an
	// arbitrary, e.g. internal, URL. This protection must apply to a
	// caller-supplied client too: we make a shallow copy (never mutating the
	// caller's) and set CheckRedirect unconditionally, so the 3xx response is
	// surfaced to Search instead of being followed.
	noRedirect := func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	if client == nil {
		client = &http.Client{Timeout: timeout, CheckRedirect: noRedirect}
	} else {
		c := *client
		c.CheckRedirect = noRedirect
		client = &c
	}
	p := &DuckDuckGoProvider{
		baseURL:     "https://html.duckduckgo.com/html/",
		client:      client,
		maxAttempts: ddgMaxAttempts,
		wait:        waitCtx,
		pacer:       requestPacer{interval: ddgMinRequestInterval},
	}
	// The pacer shares the provider's jitter seam so tests pin one field.
	p.pacer.jitterFn = p.jitterOrRandom
	return p
}

// jitterOrRandom resolves the provider's jitter seam, falling back to
// math/rand/v2 when it is unset.
func (p *DuckDuckGoProvider) jitterOrRandom() float64 {
	if p.jitter != nil {
		return p.jitter()
	}
	return rand.Float64()
}

// Name returns the provider name.
func (p *DuckDuckGoProvider) Name() string { return "duckduckgo" }

// SetBaseURL allows setting a custom base URL (useful for testing).
func (p *DuckDuckGoProvider) SetBaseURL(rawURL string) { p.baseURL = rawURL }

// Search performs a web search using DuckDuckGo HTML search. Consecutive
// requests are paced (minimum interval + jitter); an anomaly/CAPTCHA
// challenge response is retried with bounded exponential backoff and then
// surfaces as an explicit error — never as a silent empty result.
func (p *DuckDuckGoProvider) Search(ctx context.Context, query string, maxResults int) ([]SearchResult, error) {
	for attempt := 1; ; attempt++ {
		if err := p.pacer.acquire(ctx, p.wait); err != nil {
			return nil, fmt.Errorf("duckduckgo: request pacing interrupted: %w", err)
		}

		results, err := p.attempt(ctx, query, maxResults)
		if err == nil {
			return results, nil
		}
		var anom *ddgAnomalyError
		if !errors.As(err, &anom) || attempt >= p.maxAttempts {
			return nil, err
		}

		// Bounded exponential backoff before retrying the anomaly response.
		if err := p.wait(ctx, p.backoff(attempt)); err != nil {
			return nil, fmt.Errorf("duckduckgo: backoff interrupted: %w", err)
		}
	}
}

// attempt issues one request and parses the response. An anomaly challenge
// (202, or any body carrying the anomaly signature) is returned as a
// *ddgAnomalyError; other non-2xx responses are plain errors.
func (p *DuckDuckGoProvider) attempt(ctx context.Context, query string, maxResults int) ([]SearchResult, error) {
	// POST the query form instead of GET: in endpoint fuzzing the POST shape
	// (with a same-origin Referer) survived roughly twice as many requests
	// before the anti-bot flag as GET did. The result markup is identical,
	// so parsing is unchanged. This stays best-effort: DDG can extend the
	// anomaly flag to POST at any time.
	form := url.Values{}
	form.Set("q", query)
	form.Set("kl", "wt-wt") // wt-wt = no region restriction

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL, strings.NewReader(form.Encode()))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("Referer", p.referer())
	// Set a realistic User-Agent to avoid being blocked
	req.Header.Set("User-Agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
	req.Header.Set("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
	req.Header.Set("Accept-Language", "en-US,en;q=0.5")

	resp, err := p.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	// Read the (capped) body once so both the anomaly scan and the error
	// rendering can use it.
	body, readErr := io.ReadAll(limitBody(resp.Body))
	if readErr != nil {
		return nil, fmt.Errorf("HTTP %d (could not read body: %w)", resp.StatusCode, readErr)
	}

	// Fail loudly on the anti-bot anomaly challenge regardless of the status
	// code: 202 Accepted carries the CAPTCHA page, and a 200 may too. Either
	// must surface as an explicit, actionable error — treating 202 as
	// success made the provider return "No results found" with no error.
	if bytes.Contains(body, []byte(ddgAnomalySignature)) {
		return nil, &ddgAnomalyError{status: resp.StatusCode}
	}

	// Check status code
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
	}

	// Parse HTML response
	results, err := parseDuckDuckGoHTML(bytes.NewReader(body), maxResults)
	if err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	return results, nil
}

// backoff returns the wait before retrying after attempt n (1-based):
// base * 2^(n-1) plus jitter in [0, base * 2^(n-1) / 4).
func (p *DuckDuckGoProvider) backoff(attempt int) time.Duration {
	d := ddgBaseBackoff << (attempt - 1)
	return d + time.Duration(p.jitterOrRandom()*float64(d)/4)
}

// referer derives the Referer header from the base URL (a same-origin
// referer, mirroring the browser behaviour of the HTML form). Falls back to
// the canonical html endpoint when the URL does not parse.
func (p *DuckDuckGoProvider) referer() string {
	if u, err := url.Parse(p.baseURL); err == nil && u.Host != "" {
		ref := *u
		ref.RawQuery = ""
		ref.Fragment = ""
		return ref.String()
	}
	return "https://html.duckduckgo.com/html/"
}

// parseDuckDuckGoHTML parses the DuckDuckGo HTML response and extracts search results.
func parseDuckDuckGoHTML(r io.Reader, maxResults int) ([]SearchResult, error) {
	doc, err := html.Parse(r)
	if err != nil {
		return nil, fmt.Errorf("failed to parse HTML: %w", err)
	}

	var results []SearchResult

	// Find all result divs
	var traverse func(*html.Node)
	traverse = func(n *html.Node) {
		if n.Type == html.ElementNode && n.Data == "div" {
			if hasClass(n, "result") {
				// Extract result from this div
				result := extractResultFromDiv(n)
				if result.URL != "" && result.Title != "" {
					results = append(results, result)
					if len(results) >= maxResults {
						return
					}
				}
			}
		}
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			if len(results) >= maxResults {
				return
			}
			traverse(c)
		}
	}
	traverse(doc)

	return results, nil
}

// extractResultFromDiv extracts a single search result from a result div.
func extractResultFromDiv(n *html.Node) SearchResult {
	var result SearchResult

	// Find the anchor with class "result__a" for title and URL
	var findAnchor func(*html.Node)
	findAnchor = func(node *html.Node) {
		if node.Type == html.ElementNode && node.Data == "a" {
			if hasClass(node, "result__a") {
				// Get URL from href attribute
				for _, attr := range node.Attr {
					if attr.Key == "href" {
						result.URL = cleanDuckDuckGoURL(attr.Val)
						break
					}
				}
				// Get title from text content
				result.Title = getTextContent(node)
				return
			}
		}
		for c := node.FirstChild; c != nil; c = c.NextSibling {
			findAnchor(c)
		}
	}
	findAnchor(n)

	// Find snippet in anchor with class "result__snippet"
	var findSnippet func(*html.Node)
	findSnippet = func(node *html.Node) {
		if node.Type == html.ElementNode && node.Data == "a" {
			if hasClass(node, "result__snippet") {
				result.Snippet = getTextContent(node)
				return
			}
		}
		for c := node.FirstChild; c != nil; c = c.NextSibling {
			findSnippet(c)
		}
	}
	findSnippet(n)

	return result
}

// hasClass checks if a node has a specific CSS class.
func hasClass(n *html.Node, class string) bool {
	for _, attr := range n.Attr {
		if attr.Key == "class" {
			classes := strings.Fields(attr.Val)
			for _, c := range classes {
				if c == class {
					return true
				}
			}
		}
	}
	return false
}

// getTextContent extracts the text content from a node and its children.
func getTextContent(n *html.Node) string {
	var sb strings.Builder
	var extract func(*html.Node)
	extract = func(node *html.Node) {
		if node.Type == html.TextNode {
			sb.WriteString(node.Data)
		}
		for c := node.FirstChild; c != nil; c = c.NextSibling {
			extract(c)
		}
	}
	extract(n)
	return strings.TrimSpace(sb.String())
}

// cleanDuckDuckGoURL extracts the actual URL from DuckDuckGo's redirect URL.
// DuckDuckGo uses URLs like: /l/?uddg=https%3A%2F%2Fexample.com&rut=...
func cleanDuckDuckGoURL(ddgURL string) string {
	// Check if it's a DuckDuckGo redirect URL
	if strings.Contains(ddgURL, "/l/?uddg=") {
		// Parse the URL to extract the uddg parameter
		if u, err := url.Parse(ddgURL); err == nil {
			if uddg := u.Query().Get("uddg"); uddg != "" {
				return uddg
			}
			// Try parsing as a relative URL
			if strings.HasPrefix(ddgURL, "//") {
				ddgURL = "https:" + ddgURL
			}
			if u, err := url.Parse(ddgURL); err == nil {
				if uddg := u.Query().Get("uddg"); uddg != "" {
					return uddg
				}
			}
		}
	}

	// If it's already a direct URL, return as-is
	if strings.HasPrefix(ddgURL, "http://") || strings.HasPrefix(ddgURL, "https://") {
		return ddgURL
	}

	// Handle protocol-relative URLs
	if strings.HasPrefix(ddgURL, "//") {
		return "https:" + ddgURL
	}

	return ddgURL
}
