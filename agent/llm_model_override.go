package agent

import (
	"context"

	"github.com/v0lka/sp4rk/llm"
)

// modelProviderOverrideCaller wraps an LLMCaller so that every Call pins the
// per-request provider (req.Provider) and/or the model (req.Model) before
// delegating. This enables per-subagent provider+model routing without
// constructing a new caller/router.
//
// It relies on two llm.Router contracts (see llm/router.go and ADR-012):
//
//   - req.Provider, when non-empty, routes the request to the named
//     configured provider instead of the active one.
//   - prepareRequest only fills the active model when req.Model == "" —
//     setting req.Model beforehand bypasses the router's active-model
//     selection for this caller.
//
// Both fields are caller-pinned, so a mid-task Router.SetModel does not
// affect a pinned caller: every Call re-stamps the same provider+model.
type modelProviderOverrideCaller struct {
	inner    LLMCaller
	provider string
	model    string
}

// NewModelProviderOverrideCaller wraps an LLMCaller so that every Call pins
// the per-request provider and the model before delegating to inner.
//
// Each of the two overrides applies independently and only when non-empty:
//
//   - provider != "" → req.Provider is set on every call (the router routes
//     to that named provider); an empty provider leaves provider routing to
//     the router's active provider.
//   - model != "" → req.Model is set on every call (the router skips its
//     active-model fill); an empty model inherits whatever model the router
//     resolves.
//
// If BOTH are empty, inner is returned unchanged — this preserves the normal
// inheritance path, so the override can be applied conditionally without
// callers branching on it.
func NewModelProviderOverrideCaller(inner LLMCaller, provider, model string) LLMCaller {
	if provider == "" && model == "" {
		return inner
	}
	return &modelProviderOverrideCaller{inner: inner, provider: provider, model: model}
}

// NewModelOverrideCaller wraps an LLMCaller so that every Call forces req.Model
// to the given model string before delegating to inner. Provider routing is
// untouched (req.Provider passes through empty — the request routes to the
// router's active provider), making this the model-only legacy form of
// [NewModelProviderOverrideCaller].
//
// If model is empty, inner is returned unchanged — this preserves the normal
// inheritance path (the caller uses whatever model the router resolves), so the
// override can be applied conditionally without callers branching on it.
func NewModelOverrideCaller(inner LLMCaller, model string) LLMCaller {
	return NewModelProviderOverrideCaller(inner, "", model)
}

// Call pins the configured provider and/or model onto the request (if any)
// before delegating to the wrapped LLMCaller.
func (c *modelProviderOverrideCaller) Call(ctx context.Context, req llm.ChatRequest) (*llm.ChatResponse, error) {
	if c.provider != "" {
		req.Provider = c.provider
	}
	if c.model != "" {
		req.Model = c.model
	}
	return c.inner.Call(ctx, req)
}
