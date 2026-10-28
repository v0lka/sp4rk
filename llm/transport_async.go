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
// ProviderEntry.HTTPClient / OpenAIProviderConfig.HTTPClient. Router-built
// providers get it BY DEFAULT (tuned via ProviderEntry.AsyncAccepted, opted
// out via ProviderEntry.DisableAsyncAccepted).
//
// Poll cadence is SERVER-OWNED when the server speaks: Retry-After (seconds
// or HTTP-date) and the millisecond extensions Retry-After-Ms (OpenAI) /
// x-ms-retry-after-ms (Azure) are honored on the 202s and on transient poll
// failures, floored by MinInterval and clamped to MaxInterval. When the
// server is silent, the client falls back to DefaultInterval doubling per
// poll up to asyncAcceptedBackoffCap× — gentle on rate limits for long
// tasks, snappy for short ones.
//
// Polling is BUDGETED by default: MaxPolls and MaxElapsed bound every
// request's status polling (standards bound polling by TIME — Retry-After
// paces, the monitor's own TTL/410 terminates — so a per-request budget here
// is the client-side defense against a monitor that answers 202 forever).
// The floor/cap/budget defaults compose: at the headerless fallback cadence a
// request makes ~7 polls in the first minute and ~3.75/min thereafter, so the
// 30m default elapsed budget and the 120-poll default count budget exhaust
// together, and a server-driven cadence can never exceed 10 polls/s.
const (
	defaultAsyncAcceptedInterval   = 2 * time.Second
	asyncAcceptedBackoffCap        = 8
	asyncAcceptedMaxConsecErrs     = 5
	defaultAsyncAcceptedMinWait    = 100 * time.Millisecond
	defaultAsyncAcceptedMaxWait    = 5 * time.Minute
	defaultAsyncAcceptedMaxPolls   = 120
	defaultAsyncAcceptedMaxElapsed = 30 * time.Minute
	asyncAcceptedCancelTimeout     = 10 * time.Second
)

// AsyncAcceptedConfig tunes the 202-polling transport. The zero value is
// ready to use. The limit fields follow the package's sentinel convention
// (see RouterConfig.MaxRetries): zero selects the documented default, a
// NEGATIVE value disables the limit.
type AsyncAcceptedConfig struct {
	// DefaultInterval is the poll wait used when the server provides no
	// Retry-After; it doubles per headerless poll up to
	// asyncAcceptedBackoffCap× this value. Zero = 2s.
	DefaultInterval time.Duration
	// MinInterval floors every wait, INCLUDING server-provided Retry-After
	// values: a server answering Retry-After-Ms: 0 (or a misconfigured
	// gateway echoing 0) must not turn the loop into a hot hammer on the
	// monitor. Zero = 100ms; negative = no floor.
	MinInterval time.Duration
	// MaxInterval caps any single wait, including server-provided
	// Retry-After values — a misconfigured (or hostile) header must not
	// stall a call for an hour; the caller's context remains the overall
	// deadline. Zero = 5m; negative = no cap.
	MaxInterval time.Duration
	// MaxPolls bounds the number of status polls per request — the count
	// fuse against a monitor that answers 202 forever. On exhaustion the
	// call gives up (and best-effort cancels when CancelOnAbandon is set).
	// Zero = 120; negative = unlimited (the context remains the bound).
	MaxPolls int
	// MaxElapsed bounds the total time spent polling per request, the
	// duration analogue of MaxPolls and the primary budget (it composes
	// with whatever cadence Retry-After dictates, which a count cannot).
	// Zero = 30m; negative = unlimited.
	MaxElapsed time.Duration
	// MaxConsecutiveErrs bounds how many consecutive transient poll
	// failures (network errors, HTTP 408/429/5xx from the monitor) are
	// tolerated before the call gives up: a blip must not abandon a running
	// server-side task, but a monitor that stops answering must not be
	// polled forever. Zero = 5.
	MaxConsecutiveErrs int
	// CancelOnAbandon, when set, best-effort cancels the server-side task
	// whenever this transport abandons it — budget exhaustion, poll-failure
	// give-up, or the caller's context ending mid-wait — by sending DELETE
	// to the status monitor on a detached, time-bounded context (REST's
	// cancel-and-discard idiom; the endpoint must treat it as idempotent).
	// 404/409/410 answers count as success (the task is not running —
	// already gone or already terminal). Leave unset for endpoints without
	// a cancel contract: an unobserved task then simply expires server-side.
	CancelOnAbandon bool
	// Logger receives debug-level poll diagnostics and the warning emitted
	// when polling gives up. Nil = slog.Default().
	Logger *slog.Logger
}

// asyncAcceptedTransport is the RoundTripper implementing the idiom. It is
// stateless across calls (all loop state is per-RoundTrip), so one instance
// is safe for concurrent use by any number of requests.
type asyncAcceptedTransport struct {
	base            http.RoundTripper
	interval        time.Duration
	minInterval     time.Duration
	maxInterval     time.Duration
	maxPolls        int
	maxElapsed      time.Duration
	maxConsecErr    int
	cancelOnAbandon bool
	logger          *slog.Logger
}

// NewAsyncAcceptedTransport wraps base (nil = http.DefaultTransport) with the
// 202 Accepted async request-reply idiom: a 202 response carrying a Location
// header is transparently resolved by polling that monitor URI with GET —
// honoring Retry-After / Retry-After-Ms / x-ms-retry-after-ms — until it
// answers non-202, and that final response is returned to the caller as the
// response to the original request. A 202 without a Location passes through
// untouched (there is nothing to poll), as does every other status.
//
// Credential safety: poll requests copy the original request's headers, but
// the sensitive ones (Authorization, Cookie, API-key spellings) are forwarded
// ONLY when the monitor is the same host as — or a subdomain of — the
// original request's host, mirroring net/http's redirect rule. A monitor on a
// foreign host is polled unauthenticated rather than leaking credentials.
//
// Polling is bounded: MaxElapsed / MaxPolls budget each request's status
// polling (defaults 30m / 120; see AsyncAcceptedConfig), MinInterval floors
// the cadence so a zero Retry-After cannot hammer the monitor, and budget
// exhaustion surfaces as an error (after the optional CancelOnAbandon
// DELETE). Note the budget bounds the POLLING of an accepted task; it does
// not retry or resubmit anything — a retry above this transport submits a
// NEW task.
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
		base:            base,
		interval:        cfg.DefaultInterval,
		minInterval:     cfg.MinInterval,
		maxInterval:     cfg.MaxInterval,
		maxPolls:        cfg.MaxPolls,
		maxElapsed:      cfg.MaxElapsed,
		maxConsecErr:    cfg.MaxConsecutiveErrs,
		cancelOnAbandon: cfg.CancelOnAbandon,
		logger:          cfg.Logger,
	}
	if t.interval <= 0 {
		t.interval = defaultAsyncAcceptedInterval
	}
	switch {
	case t.minInterval == 0:
		t.minInterval = defaultAsyncAcceptedMinWait
	case t.minInterval < 0:
		t.minInterval = 0
	}
	switch {
	case t.maxInterval == 0:
		t.maxInterval = defaultAsyncAcceptedMaxWait
	case t.maxInterval < 0:
		t.maxInterval = time.Duration(1<<62 - 1)
	}
	switch {
	case t.maxPolls == 0:
		t.maxPolls = defaultAsyncAcceptedMaxPolls
	case t.maxPolls < 0:
		t.maxPolls = 0 // 0 = unlimited internally
	}
	switch {
	case t.maxElapsed == 0:
		t.maxElapsed = defaultAsyncAcceptedMaxElapsed
	case t.maxElapsed < 0:
		t.maxElapsed = 0 // 0 = unlimited internally
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

// isAsyncAcceptedHTTPClient reports whether the client's transport is (an
// unwrapped) asyncAcceptedTransport. The OpenAI provider constructors consult
// it to disable the SDK-internal retry loop for such clients: a poll-budget
// exhaustion (or give-up) surfaces as a transport error, which the SDK would
// otherwise retry BENEATH the router's policy — resubmitting the request and
// starting a NEW server-side task per hidden attempt. With SDK retries off,
// the router alone decides (and correctly refuses, since budget errors
// classify non-retryable) — the same router-owns-retries rationale as the
// TokenSource clients.
func isAsyncAcceptedHTTPClient(client *http.Client) bool {
	if client == nil {
		return false
	}
	_, ok := client.Transport.(*asyncAcceptedTransport)
	return ok
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
	start := time.Now()
	grow := t.interval // headerless fallback cadence, doubles per poll
	polls := 0
	consecutiveErrs := 0
	for {
		// Budget checks come first so an exhausted budget never issues one
		// more poll (count) or sleeps past its own limit (elapsed).
		if t.maxPolls > 0 && polls >= t.maxPolls {
			t.cancelAbandoned(req, monitor)
			return nil, t.budgetErr(logger, monitor, polls, start,
				fmt.Sprintf("poll budget (%d polls) exhausted", t.maxPolls))
		}
		if !hasServerWait {
			wait = grow
			if grow < asyncAcceptedBackoffCap*t.interval {
				grow *= 2
			}
		}
		if wait < t.minInterval {
			wait = t.minInterval
		}
		if wait > t.maxInterval {
			wait = t.maxInterval
		}
		if t.maxElapsed > 0 && time.Since(start)+wait > t.maxElapsed {
			t.cancelAbandoned(req, monitor)
			return nil, t.budgetErr(logger, monitor, polls, start,
				fmt.Sprintf("polling budget (%v) exhausted", t.maxElapsed))
		}
		if err := sleepCtx(ctx, wait); err != nil {
			t.cancelAbandoned(req, monitor)
			return nil, err
		}

		polls++
		pollResp, perr := t.base.RoundTrip(newMonitorRequest(ctx, http.MethodGet, monitor, req))
		if perr != nil {
			// A transient GET failure must not abandon the running
			// server-side task the 202 acknowledged.
			consecutiveErrs++
			if consecutiveErrs >= t.maxConsecErr {
				logger.Warn("async accepted: giving up after consecutive poll failures",
					"monitor", monitor.String(), "failures", consecutiveErrs, "error", perr)
				t.cancelAbandoned(req, monitor)
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
				t.cancelAbandoned(req, monitor)
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

// budgetErr logs the give-up and builds the budget-exhaustion error. The
// message deliberately does not match any transient-error pattern: a spent
// budget must surface as non-retryable, or the retry layers above would
// resubmit the request and start a NEW server-side task.
func (t *asyncAcceptedTransport) budgetErr(logger *slog.Logger, monitor *url.URL, polls int, start time.Time, what string) error {
	logger.Warn("async accepted: giving up — "+what,
		"monitor", monitor.String(), "polls", polls, "elapsed", time.Since(start).Round(time.Millisecond))
	return fmt.Errorf("async accepted: %s while polling %s (%d polls over %v); the task may still be running server-side",
		what, monitor, polls, time.Since(start).Round(time.Second))
}

// cancelAbandoned best-effort cancels the server-side task this transport is
// abandoning, when CancelOnAbandon opted in: DELETE on the status monitor —
// REST's cancel-and-discard idiom — so an unobserved task does not keep
// running (and billing). The caller's context has typically already ended
// here, so the DELETE rides a detached, time-bounded context that keeps the
// original context's values. 404/409/410 count as success (the task is not
// running: already gone or already terminal — DELETE's idempotency makes the
// repeat/raced cases converge); failures are logged, never surfaced, since
// the caller is already returning the primary error.
func (t *asyncAcceptedTransport) cancelAbandoned(orig *http.Request, monitor *url.URL) {
	if !t.cancelOnAbandon {
		return
	}
	logger := loggerOrDefault(t.logger)
	cctx, cancel := context.WithTimeout(context.WithoutCancel(orig.Context()), asyncAcceptedCancelTimeout)
	defer cancel()
	resp, err := t.base.RoundTrip(newMonitorRequest(cctx, http.MethodDelete, monitor, orig))
	if err != nil {
		logger.Warn("async accepted: abandon-cancel failed", "monitor", monitor.String(), "error", err)
		return
	}
	defer drainBody(resp)
	switch {
	case resp.StatusCode >= 200 && resp.StatusCode < 300,
		resp.StatusCode == http.StatusNotFound,
		resp.StatusCode == http.StatusConflict,
		resp.StatusCode == http.StatusGone:
		logger.Debug("async accepted: abandoned task cancelled",
			"monitor", monitor.String(), "status", resp.StatusCode)
	default:
		logger.Warn("async accepted: abandon-cancel failed",
			"monitor", monitor.String(), "status", resp.StatusCode)
	}
}

// newMonitorRequest builds a request against the status monitor (GET to poll,
// DELETE to cancel), carrying the original request's headers minus
// body-describing ones — and minus the sensitive ones unless the monitor host
// passes the same-or-subdomain rule (see NewAsyncAcceptedTransport).
func newMonitorRequest(ctx context.Context, method string, monitor *url.URL, orig *http.Request) *http.Request {
	mreq := (&http.Request{
		Method: method,
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
		mreq.Header[name] = append([]string(nil), values...)
	}
	return mreq
}

// isBodyHeader reports whether the header describes the original request's
// body or its one-shot submission semantics — meaningless (or wrong) on the
// bodyless monitor requests.
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
// redirects, applied here to monitor requests. Ports are ignored, hostname
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
