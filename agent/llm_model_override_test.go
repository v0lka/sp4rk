package agent

import (
	"context"
	"testing"

	"github.com/v0lka/sp4rk/llm"
)

// TestNewModelOverrideCaller_EmptyModel_PassesThroughInner verifies that an
// empty model string is a no-op: the wrapper returns the inner caller unchanged,
// preserving the normal inheritance path (the router resolves the model).
func TestNewModelOverrideCaller_EmptyModel_PassesThroughInner(t *testing.T) {
	inner := &mockLLMCaller{}

	got := NewModelOverrideCaller(inner, "")

	if got != inner {
		t.Fatalf("NewModelOverrideCaller(inner, \"\") = %T, want inner (%T) unchanged", got, inner)
	}

	// The inner caller's Call should still work and should record the request
	// model verbatim (no override applied).
	req := llm.ChatRequest{Model: "router-selected"}
	if _, err := got.Call(context.Background(), req); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(inner.calls) != 1 {
		t.Fatalf("expected 1 recorded call, got %d", len(inner.calls))
	}
	if inner.calls[0].Model != "router-selected" {
		t.Errorf("empty-model passthrough altered req.Model = %q, want %q",
			inner.calls[0].Model, "router-selected")
	}
}

// TestNewModelOverrideCaller_ForcesModel verifies that a non-empty model string
// forces req.Model to that value on every Call, overriding both an empty model
// and a pre-set model.
func TestNewModelOverrideCaller_ForcesModel(t *testing.T) {
	const overrideModel = "claude-haiku"
	inner := &mockLLMCaller{}
	caller := NewModelOverrideCaller(inner, overrideModel)

	cases := []struct {
		name  string
		model string // model the caller places in the request before Call
	}{
		{"empty request model is filled", ""},
		{"pre-set request model is overridden", "claude-opus"},
		{"override is stable across calls", "gpt-4o"},
	}

	for i, tc := range cases {
		req := llm.ChatRequest{Model: tc.model}
		if _, err := caller.Call(context.Background(), req); err != nil {
			t.Fatalf("[%s] unexpected error: %v", tc.name, err)
		}
		if len(inner.calls) <= i {
			t.Fatalf("[%s] expected call %d to be recorded", tc.name, i)
		}
		if got := inner.calls[i].Model; got != overrideModel {
			t.Errorf("[%s] req.Model = %q, want %q", tc.name, got, overrideModel)
		}
	}

	// Sanity: the override must not mutate the caller's own request copy in a way
	// that leaks back — the original request passed by the caller stays intact.
	src := llm.ChatRequest{Model: "original"}
	if _, err := caller.Call(context.Background(), src); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if src.Model != "original" {
		t.Errorf("override mutated the caller's original request: got %q, want %q",
			src.Model, "original")
	}
}
