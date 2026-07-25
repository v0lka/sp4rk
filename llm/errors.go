package llm

import (
	"errors"
	"fmt"
	"io"
	"net"
	"syscall"
)

// Error wraps provider errors with classification metadata.
type Error struct {
	Provider   string // e.g. "openai", "anthropic"
	StatusCode int    // HTTP status code (0 if not applicable, e.g. network error)
	Retryable  bool   // whether this error is safe to retry
	Err        error  // the original underlying error
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

	// Check for syscall errors (ECONNREFUSED, ECONNRESET, EHOSTUNREACH)
	if errors.Is(err, syscall.ECONNREFUSED) || errors.Is(err, syscall.ECONNRESET) || errors.Is(err, syscall.EHOSTUNREACH) {
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
