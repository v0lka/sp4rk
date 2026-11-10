package agent

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"sync/atomic"
)

type toolCallIDContextKey struct{}

type toolCallBinding struct {
	id          string
	tool        string
	inputSHA256 [32]byte
	claimed     atomic.Bool
}

func withToolCallID(ctx context.Context, id, tool string, input json.RawMessage) context.Context {
	// Always shadow a parent ID. A nested executor with a legacy emitter must
	// not inherit its delegating tool's ID and falsely attribute its own calls.
	binding := &toolCallBinding{id: id, tool: tool}
	if id != "" {
		binding.inputSHA256 = sha256.Sum256(input)
	}
	return context.WithValue(ctx, toolCallIDContextKey{}, binding)
}

// WithoutToolCallID clears an inherited SDK call binding when an independent
// dispatcher starts its own tool-call lifecycle.
func WithoutToolCallID(ctx context.Context) context.Context {
	return withToolCallID(ctx, "", "", nil)
}

// ToolCallIDFromContext returns the host emitter's ID for the current SDK
// tool invocation. Empty means no ID was supplied by the emitter or this
// context does not belong to an SDK-dispatched tool call.
func ToolCallIDFromContext(ctx context.Context) string {
	binding, _ := ctx.Value(toolCallIDContextKey{}).(*toolCallBinding)
	if binding == nil {
		return ""
	}
	return binding.id
}

// ClaimToolCallID returns an ID once, only for the exact post-HITL input
// passed to the named tool. It does not identify the pre-HITL card arguments.
func ClaimToolCallID(ctx context.Context, tool string, input json.RawMessage) string {
	binding, _ := ctx.Value(toolCallIDContextKey{}).(*toolCallBinding)
	if binding == nil || binding.id == "" || binding.tool != tool ||
		binding.inputSHA256 != sha256.Sum256(input) ||
		!binding.claimed.CompareAndSwap(false, true) {
		return ""
	}
	return binding.id
}
