package llm

import (
	"crypto/tls"
	"errors"
	"fmt"
	"io"
	"net"
	"net/url"
	"strings"
	"syscall"
)

// ErrKind is a transport-independent classification of a provider error,
// derived from the HTTP status code when the transport carries one (OpenAI,
// Google, generic request errors) or from the provider's own error type
// field when it does not (the Anthropic SDK parses the JSON error body into
// APIError, which carries no HTTP status). It lets callers reason about the
// failure class (e.g. rate limiting) without knowing which SDK produced the
// error. Empty when no classification is available. Not to be confused with
// the Anthropic SDK's anthropic.ErrType, whose constants carry the
// provider-side "rate_limit_error" spelling — always bridge via the SDK's
// Is*Err() helpers.
type ErrKind string

const (
	// ErrKindRateLimit marks a rate-limit / quota rejection. HTTP 429 on
	// status-carrying transports; the "rate_limit_error" APIError type on
	// the Anthropic transport.
	ErrKindRateLimit ErrKind = "rate_limit"
	// ErrKindOverloaded marks a transient provider overload. HTTP 529 on
	// status-carrying transports; the "overloaded_error" APIError type on
	// the Anthropic transport.
	ErrKindOverloaded ErrKind = "overloaded"
)

// errKindForStatus maps an HTTP status code to its ErrKind classification,
// or "" when the code carries no classification.
func errKindForStatus(statusCode int) ErrKind {
	switch statusCode {
	case 429:
		return ErrKindRateLimit
	case 529:
		return ErrKindOverloaded
	default:
		return ""
	}
}

// Error wraps provider errors with classification metadata.
type Error struct {
	Provider   string  // e.g. "openai", "anthropic"
	StatusCode int     // HTTP status code (0 if not applicable, e.g. network error)
	Retryable  bool    // whether this error is safe to retry
	ErrKind    ErrKind // transport-independent failure class ("" if unknown); derived by NewError/WrapProviderError from the status — hand-built literals must set it explicitly
	Err        error   // the original underlying error
}

// Error formats the error like "llm [provider] error (HTTP status, retryable=bool): original message".
func (e *Error) Error() string {
	return fmt.Sprintf("llm [%s] error (HTTP %d, retryable=%t): %v", e.Provider, e.StatusCode, e.Retryable, e.Err)
}

// Unwrap returns the original underlying error.
func (e *Error) Unwrap() error {
	return e.Err
}

// NewError constructs an Error with the given fields.
func NewError(provider string, statusCode int, retryable bool, err error) *Error {
	return &Error{
		Provider:   provider,
		StatusCode: statusCode,
		Retryable:  retryable,
		ErrKind:    errKindForStatus(statusCode),
		Err:        err,
	}
}

// IsRetryable checks whether the error chain contains a *Error with Retryable == true.
func IsRetryable(err error) bool {
	var llmErr *Error
	if errors.As(err, &llmErr) {
		return llmErr.Retryable
	}
	return false
}

// classifyHTTPStatus returns true for transient HTTP status codes that are safe
// to retry. This is exhaustive over the classes of transient failures a client
// can observe from an LLM provider gateway:
//
//   - 408 Request Timeout: the origin gave up waiting for the client.
//   - 429 Too Many Requests: rate limit / quota — back off and retry.
//   - 500 Internal Server Error: transient upstream faults; the major provider
//     SDKs (OpenAI, Anthropic) treat 500 as retryable for LLM calls.
//   - 502 Bad Gateway, 503 Service Unavailable: gateway/origin unavailable.
//   - 504 Gateway Timeout: the standard (RFC 7231) gateway timeout.
//   - 520-524: Cloudflare edge errors (unknown error, web server down,
//     connection timed out, origin unreachable, origin timeout). 524 in
//     particular is Cloudflare's proprietary equivalent of 504: the edge waited
//     for an origin that never responded in time. Without retry, a single slow
//     generation surfaces as a hard, non-retryable task failure.
//   - 529 Overloaded: Anthropic-specific transient overload.
func classifyHTTPStatus(code int) bool {
	switch code {
	case 408, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 529:
		return true
	default:
		return false
	}
}

// classifyNetError returns true for transient network errors such as timeouts,
// connection refused/reset, DNS errors, and unexpected EOF.
func classifyNetError(err error) bool {
	if err == nil {
		return false
	}

	// Check for net.Error with Timeout()
	var netErr net.Error
	if errors.As(err, &netErr) && netErr.Timeout() {
		return true
	}

	// Check for transient connection failures (ECONNREFUSED, ECONNRESET,
	// EHOSTUNREACH). Match the raw syscall.Errno from the error chain via a
	// platform-specific helper: on Unix the POSIX errno sentinels are the real
	// OS values, while on Windows they are invented and we must match the raw
	// Winsock codes instead (see errno_windows.go). errors.As is used rather
	// than errors.Is so the comparison is uniform across platforms and does not
	// depend on the fragile invented-errno situation on Windows.
	var errno syscall.Errno
	if errors.As(err, &errno) && isTransientConnErrno(errno) {
		return true
	}

	// Check for DNS errors
	var dnsErr *net.DNSError
	if errors.As(err, &dnsErr) {
		return true
	}

	// Check for connection dropped (EOF)
	if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
		return true
	}

	return false
}

// ErrContextWindowExceeded is a sentinel error for pre-call context window validation failures.
// Use errors.Is(err, ErrContextWindowExceeded) to detect this condition.
var ErrContextWindowExceeded = errors.New("context window exceeded")

// TransportKind classifies a connection-level provider error into a short,
// stable diagnostic label: "timeout", "dns", "connect" (refused/unreachable),
// "reset", "tls", "eof", or "other" for any other non-nil error. It is
// observability only — retryability is still decided by classifyNetError and
// classifyHTTPStatus — and exists because an "HTTP 0" failure (the OpenAI-
// compatible SDK surfaced no HTTP response at all) otherwise reaches callers
// as an opaque string with no hint whether the request ever left the process,
// which upstream consumers (e.g. census callers curating per-app failures)
// need to triage reproducible connection-level rejections.
func TransportKind(err error) string {
	if err == nil {
		return ""
	}
	var netErr net.Error
	if errors.As(err, &netErr) && netErr.Timeout() {
		return "timeout"
	}
	var dnsErr *net.DNSError
	if errors.As(err, &dnsErr) {
		return "dns"
	}
	var errno syscall.Errno
	if errors.As(err, &errno) {
		// Per-OS classification: on Windows real failures carry raw Winsock
		// codes, not the invented POSIX sentinels (see errno_windows.go).
		if kind := transportKindForErrno(errno); kind != "" {
			return kind
		}
	}
	var tlsErr *tls.CertificateVerificationError
	if errors.As(err, &tlsErr) {
		return "tls"
	}
	if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
		return "eof"
	}
	// url.Error wraps the whole outbound request/round-trip; its inner error
	// may still classify, and TLS handshake failures surface as *url.Error
	// wrapping an OpError wrapping the tls handshake error. Classify any
	// remaining url.Error (or net.OpError) chain as "connect" — the request
	// never produced a response from the endpoint.
	var urlErr *url.Error
	if errors.As(err, &urlErr) {
		if inner := TransportKind(urlErr.Err); inner != "" && inner != "other" {
			return inner
		}
		return "connect"
	}
	var opErr *net.OpError
	if errors.As(err, &opErr) {
		if inner := TransportKind(opErr.Err); inner != "" && inner != "other" {
			return inner
		}
		return "connect"
	}
	if strings.Contains(err.Error(), "tls:") {
		return "tls"
	}
	return "other"
}

// NewContextWindowError creates a non-retryable Error for context window overflow.
func NewContextWindowError(model string, estimatedTokens, effectiveMax, contextWindow, outputReserve int) *Error {
	return &Error{
		Provider:   "router",
		StatusCode: 0,
		Retryable:  false,
		Err: fmt.Errorf("%w: estimated %d tokens, model %q allows %d (context_window=%d, output_reserve=%d)",
			ErrContextWindowExceeded, estimatedTokens, model, effectiveMax, contextWindow, outputReserve),
	}
}

// WrapProviderError creates an Error by classifying an error based on HTTP status code
// and underlying error type. If statusCode > 0, it uses HTTP status classification.
// Otherwise falls back to network error classification.
func WrapProviderError(provider string, statusCode int, err error) *Error {
	retryable := false
	if statusCode > 0 {
		retryable = classifyHTTPStatus(statusCode)
	}
	if !retryable {
		retryable = classifyNetError(err)
	}
	return NewError(provider, statusCode, retryable, err)
}
