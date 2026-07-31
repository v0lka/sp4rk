package agent

import (
	"context"

	"github.com/v0lka/sp4rk/llm"
)

// modelOverrideCaller wraps an LLMCaller so that every Call forces req.Model to
// a specific model string before delegating. This enables per-agent model
// overrides without constructing a new caller/provider.
//
// It relies on the contract that llm.Router.prepareRequest only fills the
// active model when req.Model == "" (see llm/router.go). By setting req.Model
// beforehand, the router's active-model selection is bypassed for this caller.
type modelOverrideCaller struct {
	inner LLMCaller
	model string
}

// NewModelOverrideCaller wraps an LLMCaller so that every Call forces req.Model
// to the given model string before delegating to inner.
//
// If model is empty, inner is returned unchanged — this preserves the normal
// inheritance path (the caller uses whatever model the router resolves), so the
// override can be applied conditionally without callers branching on it.
func NewModelOverrideCaller(inner LLMCaller, model string) LLMCaller {
	if model == "" {
		return inner
	}
	return &modelOverrideCaller{inner: inner, model: model}
}

// Call forces req.Model to the configured model (if any) before delegating to
// the wrapped LLMCaller.
func (c *modelOverrideCaller) Call(ctx context.Context, req llm.ChatRequest) (*llm.ChatResponse, error) {
	if c.model != "" {
		req.Model = c.model
	}
	return c.inner.Call(ctx, req)
}
