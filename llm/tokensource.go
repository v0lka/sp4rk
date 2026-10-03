package llm

import (
	"context"
	"fmt"
	"net/http"
	"time"
)

// BearerToken is a credential snapshot produced by a TokenSource for a single
// outgoing HTTP request. The renewal material (refresh token, expiry) stays
// inside the TokenSource implementation — the wire snapshot carries only what
// an HTTP request needs.
type BearerToken struct {
	// AccessToken is the bearer credential stamped onto the Authorization
	// header as "Bearer <AccessToken>", overriding whatever static credential
	// the provider was constructed with. Empty = leave the request's existing
	// Authorization header untouched (e.g. a static API key, or none for a
	// local backend).
	AccessToken string

	// TokenType is the token type reported by the issuing endpoint (typically
	// "Bearer"). It is informational; the header always uses the Bearer
	// scheme.
	TokenType string

	// ExpiresAt is when AccessToken stops being valid; the zero value means
	// unknown or non-expiring, as reported by the issuing endpoint.
	ExpiresAt time.Time

	// ExtraHeaders are additional headers set on the request after the
	// Authorization header is applied (e.g. "ChatGPT-Account-Id" for a
	// ChatGPT OAuth-backed account, or vendor beta-routing headers). A key
	// with an empty value removes that header from the request instead of
	// sending an empty-valued one.
	ExtraHeaders map[string]string
}

// TokenSource supplies per-request bearer credentials to a provider. It is
// the seam through which a host application injects short-lived or refreshed
// credentials (OAuth access tokens, subscription-backed sign-ins) without the
// provider knowing how they were obtained: the provider resolves a token
// immediately before every request leaves the process — on every protocol it
// serves (the two OpenAI protocols through an SDK middleware, the Anthropic
// and Google delegates through a wrapping transport / pre-send resolution) —
// so an implementation may refresh an expired credential lazily and every
// attempt carries a freshly resolved one.
//
// Implementations must be safe for concurrent use: a provider may issue
// requests from multiple goroutines through the same TokenSource. While a
// non-expired token is available, Token must not perform a network
// round-trip; renewal traffic belongs to the implementation, not its callers.
type TokenSource interface {
	// Token returns the credentials for the next outgoing request. A non-nil
	// error aborts that request (and its retry chain) before it hits the
	// wire; the provider wraps it so the failure surfaces to the caller. The
	// context is the request's own — honor its cancellation during renewal.
	Token(ctx context.Context) (BearerToken, error)
}

// applyBearerToken stamps a resolved BearerToken onto an outgoing request.
// A non-empty access token overrides the Authorization header and clears
// staticHeader — the credential header the underlying SDK would otherwise
// send ("x-api-key" for the Anthropic SDK; "" when there is none to clear) —
// so a dynamic credential never rides alongside a stale static key. Each
// ExtraHeaders entry is set; an empty value removes that header. An empty
// access token leaves the request's existing credentials untouched.
func applyBearerToken(req *http.Request, tok BearerToken, staticHeader string) {
	if tok.AccessToken != "" {
		req.Header.Set("Authorization", "Bearer "+tok.AccessToken)
		if staticHeader != "" {
			req.Header.Del(staticHeader)
		}
	}
	for k, v := range tok.ExtraHeaders {
		if v == "" {
			req.Header.Del(k)
			continue
		}
		req.Header.Set(k, v)
	}
}

// tokenSourceRoundTripper applies a TokenSource to an HTTP transport for
// SDKs without a middleware hook (go-anthropic): it resolves credentials
// immediately before the request leaves the process and aborts the call —
// no network I/O — when Token fails, mirroring the openai-go
// tokenSourceMiddleware contract (the error wraps "llm: token source:").
type tokenSourceRoundTripper struct {
	ts           TokenSource
	next         http.RoundTripper
	staticHeader string // the SDK's static credential header, cleared when a token resolves
}

// RoundTrip implements http.RoundTripper.
func (t *tokenSourceRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	tok, err := t.ts.Token(req.Context())
	if err != nil {
		return nil, fmt.Errorf("llm: token source: %w", err)
	}
	applyBearerToken(req, tok, t.staticHeader)
	next := t.next
	if next == nil {
		// A client with a nil Transport uses http.DefaultTransport lazily;
		// mirror that so wrapping a default client keeps working.
		next = http.DefaultTransport
	}
	return next.RoundTrip(req)
}
