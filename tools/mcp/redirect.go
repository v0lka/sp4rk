package mcp

import (
	"fmt"
	"net/http"
	"net/url"
	"strings"
)

// maxHTTPRedirects bounds how many redirect hops an MCP HTTP transport may
// follow before giving up. It re-implements net/http's built-in cap, which
// applies only when Client.CheckRedirect is nil — once a custom policy is
// installed, the limit is the policy's own responsibility.
const maxHTTPRedirects = 10

// originPinnedClient returns a shallow copy of client whose redirect policy is
// pinned to the origin of initialURL (see originPinnedRedirects). The caller's
// client is NEVER mutated: the host may share one *http.Client (c0wrk's proxy
// client fans out to every MCP server entry) and each server entry must pin
// redirects to its own endpoint's origin, so the copy is exclusive to this
// connection.
func originPinnedClient(client *http.Client, initialURL string) *http.Client {
	pinned := *client
	pinned.CheckRedirect = originPinnedRedirects(initialURL)
	return &pinned
}

// originPinnedRedirects builds the CheckRedirect policy that closes the
// CVE-2026-116685 class for this package's HTTP transports: the stock
// net/http redirect policy silently re-issues a redirect request to any
// Location — up to 10 hops — and on a cross-domain hop it strips only
// Authorization, Cookie and Www-Authenticate, while every operator-configured
// custom header (X-API-Key, Mcp-Session-Id, …) passes through verbatim and a
// 307/308 re-sends the full JSON-RPC request body too. A trusted-but-
// redirecting endpoint (compromised infrastructure, stale DNS, an open
// redirect in front of the server) would therefore hand those credentials and
// payload contents to an attacker-chosen origin — and a Location pointing at
// an internal address turns the host into a request-forgery client.
//
// The policy follows a redirect only when it stays on the endpoint's origin —
// same scheme, same host (case-insensitive), same effective port — with a
// single exception: an insecure→secure upgrade (http→https) on the same host,
// which mirrors the reference MCP SDK's post-CVE fix. Cross-host redirects,
// subdomains, port changes, https→http downgrades and non-HTTP schemes are
// refused; the refusal surfaces as a connection error for the server entry,
// never as a followed request.
func originPinnedRedirects(initialURL string) func(*http.Request, []*http.Request) error {
	origin, parseErr := url.Parse(initialURL)
	var originScheme, originHost, originPort string
	if parseErr == nil {
		originScheme, originHost, originPort = origin.Scheme, origin.Hostname(), effectivePort(origin)
	}
	return func(req *http.Request, via []*http.Request) error {
		if parseErr != nil {
			return fmt.Errorf("mcp http transport: redirect policy for %q unavailable: %w", initialURL, parseErr)
		}
		if len(via) >= maxHTTPRedirects {
			return fmt.Errorf("mcp http transport: stopped after %d redirects", maxHTTPRedirects)
		}
		target := req.URL
		switch target.Scheme {
		case "http", "https":
		default:
			return fmt.Errorf("mcp http transport: redirect to %q refused: scheme %q is not http(s)",
				target.Redacted(), target.Scheme)
		}
		if !strings.EqualFold(target.Hostname(), originHost) {
			return fmt.Errorf("mcp http transport: redirect to %q refused: host %q crosses the origin of %q",
				target.Redacted(), target.Hostname(), origin.Redacted())
		}
		switch {
		case target.Scheme == originScheme:
			if effectivePort(target) != originPort {
				return fmt.Errorf("mcp http transport: redirect to %q refused: port %q leaves the origin of %q",
					target.Redacted(), effectivePort(target), origin.Redacted())
			}
		case originScheme == "http" && target.Scheme == "https":
			// Insecure→secure upgrade on the same host: allowed regardless of
			// the effective port, which necessarily changes (80 → 443).
		default:
			return fmt.Errorf("mcp http transport: redirect to %q refused: %s→%s downgrade",
				target.Redacted(), originScheme, target.Scheme)
		}
		return nil
	}
}

// effectivePort returns the URL's explicit port, or the scheme's default port
// when none is set (http 80, https 443); the empty string for other schemes.
// Two same-scheme origins are compared on the effective port so an explicit
// default port ("https://host:443") never reads as a port change.
func effectivePort(u *url.URL) string {
	if p := u.Port(); p != "" {
		return p
	}
	switch u.Scheme {
	case "http":
		return "80"
	case "https":
		return "443"
	}
	return ""
}
