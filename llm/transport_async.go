package llm

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
)

// Async request-reply over HTTP 202 Accepted (RFC 9110 §15.3.3): an endpoint
// that cannot (or chooses not to) answer a request synchronously replies
//
//	202 Accepted
//	Location: <status-monitor URI>
//	Retry-After: <seconds | HTTP-date>        (optional)
//
// and the client polls the monitor with GET until it answers something other
// than 202 — that final response IS the response to the original request.
// The idiom is transport-level: the request and result bodies are whatever
// the API speaks (e.g. a plain /chat/completions payload), so implementing it
// as an http.RoundTripper makes every provider, protocol and SDK above it
// async-capable with no contract changes — callers simply see a blocking call
// that does not hold a connection while the server works.
//
// NewAsyncAcceptedTransport builds that RoundTripper;
// WrapHTTPClientAsyncAccepted applies it to an *http.Client suitable for
// ProviderEntry.HTTPClient / OpenAIProviderConfig.HTTPClient; and
// ProviderEntry.FollowAsyncAccepted enables it declaratively per provider.
//
// Poll cadence is SERVER-OWNED when the server speaks: Retry-After (seconds
// or HTTP-date) and the OpenAI extension Retry-After-Ms are honored on the
// 202s and on transient poll failures, clamped to MaxInterval. When the
// server is silent, the client falls back to DefaultInterval doubling per
// poll up to asyncAcceptedBackoffCap× — the same cadence shape as the
// background-mode poll loop, for the same reason: gentle on rate limits for
// long tasks, snappy for short ones.
const (
	defaultAsyncAcceptedInterval = 2 * time.Second
	asyncAcceptedBackoffCap      = 8
	asyncAcceptedMaxConsecErrs   = 5
	defaultAsyncAcceptedMaxWait  = 5 * time.Minute
)

// AsyncAcceptedConfig tunes the 202-polling transport. The zero value is
// ready to use.
type AsyncAcceptedConfig struct {
	// DefaultInterval is the poll wait used when the server provides no
	// Retry-After; it doubles per headerless poll up to
	// asyncAcceptedBackoffCap× this value. Zero = 2s.
	DefaultInterval time.Duration
	// MaxInterval caps any single wait, including server-provided
	// Retry-After values — a misconfigured (or hostile) header must not
	// stall a call for an hour; the caller's context remains the overall
	// deadline. Zero = 5m.
	MaxInterval time.Duration
	// MaxConsecutiveErrs bounds how many consecutive transient poll
	// failures (network errors, HTTP 408/429/5xx from the monitor) are
	// tolerated before the call gives up: a blip must not abandon a running
	// server-side task, but a monitor that stops answering must not be
	// polled forever. Zero = 5.
	MaxConsecutiveErrs int
	// Logger receives debug-level poll diagnostics and the warning emitted
	// when polling gives up. Nil = slog.Default().
	Logger *slog.Logger
}

// asyncAcceptedTransport is the RoundTripper implementing the idiom. It is
// stateless across calls (all loop state is per-RoundTrip), so one instance
// is safe for concurrent use by any number of requests.
type asyncAcceptedTransport struct {
	base         http.RoundTripper
	interval     time.Duration
	maxInterval  time.Duration
	maxConsecErr int
	logger       *slog.Logger
}

// NewAsyncAcceptedTransport wraps base (nil = http.DefaultTransport) with the
// 202 Accepted async request-reply idiom: a 202 response carrying a Location
// header is transparently resolved by polling that monitor URI with GET —
// honoring Retry-After / Retry-After-Ms — until it answers non-202, and that
// final response is returned to the caller as the response to the original
// request. A 202 without a Location passes through untouched (there is
// nothing to poll), as does every other status.
//
// Credential safety: poll requests copy the original request's headers, but
// the sensitive ones (Authorization, Cookie, API-key spellings) are forwarded
// ONLY when the monitor is the same host as — or a subdomain of — the
// original request's host, mirroring net/http's redirect rule. A monitor on a
// foreign host is polled unauthenticated rather than leaking credentials.
//
// Cancellation and deadlines ride the original request's context, which
// governs the entire poll loop. NOTE http.Client.Timeout also spans the whole
// loop (it is implemented as a deadline over the full RoundTrip), so a client
// wrapped with this transport should use per-request contexts rather than an
// aggressive client-wide Timeout, or long tasks will be cut off mid-poll.
func NewAsyncAcceptedTransport(base http.RoundTripper, cfg AsyncAcceptedConfig) http.RoundTripper {
	if base == nil {
		base = http.DefaultTransport
	}
	t := &asyncAcceptedTransport{
		base:         base,
		interval:     cfg.DefaultInterval,
		maxInterval:  cfg.MaxInterval,
		maxConsecErr: cfg.MaxConsecutiveErrs,
		logger:       cfg.Logger,
	}
	if t.interval <= 0 {
		t.interval = defaultAsyncAcceptedInterval
	}
	if t.maxInterval <= 0 {
		t.maxInterval = defaultAsyncAcceptedMaxWait
	}
	if t.maxConsecErr <= 0 {
		t.maxConsecErr = asyncAcceptedMaxConsecErrs
	}
	return t
}

// WrapHTTPClientAsyncAccepted returns a copy of client whose transport
// resolves 202 Accepted + Location responses via NewAsyncAcceptedTransport.
// A nil client wraps http.DefaultTransport into a fresh client. The copy
// keeps the original's CheckRedirect and Jar; Timeout is kept too — see the
// NewAsyncAcceptedTransport note about client-wide timeouts spanning the
// whole poll loop.
func WrapHTTPClientAsyncAccepted(client *http.Client, cfg AsyncAcceptedConfig) *http.Client {
	wrapped := &http.Client{}
	var base http.RoundTripper
	if client != nil {
		*wrapped = *client
		base = client.Transport
	}
	wrapped.Transport = NewAsyncAcceptedTransport(base, cfg)
	return wrapped
}

// RoundTrip implements http.RoundTripper.
func (t *asyncAcceptedTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	resp, err := t.base.RoundTrip(req)
	if err != nil || resp.StatusCode != http.StatusAccepted {
		return resp, err
	}
	monitor, lerr := resp.Location() // resolves a relative Location against req.URL
	if lerr != nil {
		// A 202 without a Location is not the polling idiom — there is
		// nothing to poll, so it must surface to the caller untouched.
		return resp, nil
	}

	logger := loggerOrDefault(t.logger)
	logger.Debug("async accepted: polling status monitor",
		"url", req.URL.String(), "monitor", monitor.String())

	wait, hasServerWait := parseRetryAfter(resp.Header)
	drainBody(resp)

	ctx := req.Context()
	grow := t.interval // headerless fallback cadence, doubles per poll
	consecutiveErrs := 0
	for {
		if !hasServerWait {
			wait = grow
			if grow < asyncAcceptedBackoffCap*t.interval {
				grow *= 2
			}
		}
		if wait > t.maxInterval {
			wait = t.maxInterval
		}
		if err := sleepCtx(ctx, wait); err != nil {
			return nil, err
		}

		pollResp, perr := t.base.RoundTrip(newPollRequest(ctx, monitor, req))
		if perr != nil {
			// Same tolerance rationale as the poll loops elsewhere in this
			// package: a transient GET failure must not abandon the running
			// server-side task the 202 acknowledged.
			consecutiveErrs++
			if consecutiveErrs >= t.maxConsecErr {
				logger.Warn("async accepted: giving up after consecutive poll failures",
					"monitor", monitor.String(), "failures", consecutiveErrs, "error", perr)
				return nil, fmt.Errorf("async accepted: polling %s: %w", monitor, perr)
			}
			hasServerWait = false
			continue
		}

		switch {
		case pollResp.StatusCode == http.StatusAccepted:
			// Still running. The monitor may relocate itself (a new
			// Location on a 202 poll); adopt it so a server can hand off
			// status serving mid-task.
			consecutiveErrs = 0
			if next, nerr := pollResp.Location(); nerr == nil {
				monitor = next
			}
			wait, hasServerWait = parseRetryAfter(pollResp.Header)
			drainBody(pollResp)

		case isTransientAsyncPollStatus(pollResp.StatusCode):
			// The monitor itself hiccuped (429/5xx/timeout). Honor its
			// Retry-After if present and keep polling up to the tolerance;
			// past it, return the failing response so the layers above
			// classify it exactly like any other provider failure.
			consecutiveErrs++
			if consecutiveErrs >= t.maxConsecErr {
				logger.Warn("async accepted: giving up after consecutive poll failures",
					"monitor", monitor.String(), "failures", consecutiveErrs,
					"status", pollResp.StatusCode)
				return pollResp, nil
			}
			wait, hasServerWait = parseRetryAfter(pollResp.Header)
			drainBody(pollResp)

		default:
			// Terminal: the result (200), a redirect to it (303), or a
			// definitive error (4xx). It IS the response to the original
			// request; redirects are then followed by the http.Client layer
			// above this transport as usual.
			return pollResp, nil
		}
	}
}

// newPollRequest builds the GET against the status monitor, carrying the
// original request's headers minus body-describing ones — and minus the
// sensitive ones unless the monitor host passes the same-or-subdomain rule
// (see NewAsyncAcceptedTransport).
func newPollRequest(ctx context.Context, monitor *url.URL, orig *http.Request) *http.Request {
	poll := (&http.Request{
		Method: http.MethodGet,
		URL:    monitor,
		Header: make(http.Header, len(orig.Header)),
	}).WithContext(ctx)
	forwardSensitive := isSameOrSubdomainHost(orig.URL, monitor)
	for name, values := range orig.Header {
		if isBodyHeader(name) {
			continue
		}
		if isSensitiveHeader(name) && !forwardSensitive {
			continue
		}
		poll.Header[name] = append([]string(nil), values...)
	}
	return poll
}

// isBodyHeader reports whether the header describes the original request's
// body or its one-shot submission semantics — meaningless (or wrong) on the
// bodyless poll GET.
func isBodyHeader(name string) bool {
	switch http.CanonicalHeaderKey(name) {
	case "Content-Type", "Content-Length", "Content-Encoding", "Transfer-Encoding",
		"Expect", "Idempotency-Key":
		return true
	default:
		return false
	}
}

// isSensitiveHeader reports whether the header carries credentials. The set
// covers the generic HTTP credentials net/http protects on redirects plus the
// API-key spellings the providers in this package use (x-api-key for
// Anthropic, x-goog-api-key for Google).
func isSensitiveHeader(name string) bool {
	switch http.CanonicalHeaderKey(name) {
	case "Authorization", "Proxy-Authorization", "Cookie", "Cookie2",
		"X-Api-Key", "Api-Key", "X-Goog-Api-Key":
		return true
	default:
		return false
	}
}

// isSameOrSubdomainHost reports whether monitor's host equals orig's host or
// is a subdomain of it — net/http's rule for forwarding sensitive headers on
// redirects, applied here to poll requests. Ports are ignored, hostname
// comparison is case-insensitive.
func isSameOrSubdomainHost(orig, monitor *url.URL) bool {
	origHost := strings.ToLower(orig.Hostname())
	monHost := strings.ToLower(monitor.Hostname())
	return monHost == origHost || strings.HasSuffix(monHost, "."+origHost)
}

// isTransientAsyncPollStatus reports whether a monitor response status names
// a transient failure of the POLL itself (not of the task): request timeout,
// rate limit, or any server-side 5xx. Mirrors the retryable statuses of
// classifyHTTPStatus without the provider-error wrapping.
func isTransientAsyncPollStatus(code int) bool {
	return code == http.StatusRequestTimeout || code == http.StatusTooManyRequests || code >= 500
}

// parseRetryAfter extracts the server-requested wait from a response's
// headers, most precise spelling first: the millisecond extensions
// Retry-After-Ms (OpenAI) and x-ms-retry-after-ms (Azure), then Retry-After
// as delta-seconds or an HTTP-date (RFC 9110 §10.2.3; a past date means poll
// now). Returns ok=false when no header parses, leaving cadence to the
// client-side fallback.
func parseRetryAfter(h http.Header) (time.Duration, bool) {
	for _, name := range []string{"Retry-After-Ms", "X-Ms-Retry-After-Ms"} {
		if v := strings.TrimSpace(h.Get(name)); v != "" {
			if ms, err := strconv.ParseInt(v, 10, 64); err == nil && ms >= 0 {
				return time.Duration(ms) * time.Millisecond, true
			}
		}
	}
	v := strings.TrimSpace(h.Get("Retry-After"))
	if v == "" {
		return 0, false
	}
	if secs, err := strconv.ParseInt(v, 10, 64); err == nil && secs >= 0 {
		return time.Duration(secs) * time.Second, true
	}
	if at, err := http.ParseTime(v); err == nil {
		if d := time.Until(at); d > 0 {
			return d, true
		}
		return 0, true
	}
	return 0, false
}

// sleepCtx waits d, aborting with ctx.Err() if the context ends first. A
// non-positive d only checks the context.
func sleepCtx(ctx context.Context, d time.Duration) error {
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

// drainBody consumes and closes an intermediate response's body so the
// underlying connection can be reused for the next poll. The read is bounded:
// an intermediate status body is advisory (e.g. a task id echo) and must not
// become an unbounded allocation.
func drainBody(resp *http.Response) {
	if resp == nil || resp.Body == nil {
		return
	}
	_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 64<<10))
	_ = resp.Body.Close()
}
