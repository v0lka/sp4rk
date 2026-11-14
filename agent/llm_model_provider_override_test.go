package agent

import (
	"context"
	"strings"
	"testing"

	"github.com/v0lka/sp4rk/llm"
)

// streamingInnerCaller is an LLMCaller that invokes the request's DeltaSink
// (if any) before returning its response — the minimal streaming contract the
// wrapper must preserve.
type streamingInnerCaller struct {
	deltas   []llm.StreamDelta
	response *llm.ChatResponse
}

func (c *streamingInnerCaller) Call(_ context.Context, req llm.ChatRequest) (*llm.ChatResponse, error) {
	if req.DeltaSink != nil {
		for _, d := range c.deltas {
			if err := req.DeltaSink(d); err != nil {
				return nil, err
			}
		}
	}
	return c.response, nil
}

// TestNewModelProviderOverrideCaller_EmptyPins_ReturnsInner verifies that with
// both overrides empty the wrapper returns the inner caller unchanged,
// preserving the normal inheritance path.
func TestNewModelProviderOverrideCaller_EmptyPins_ReturnsInner(t *testing.T) {
	inner := &mockLLMCaller{}

	got := NewModelProviderOverrideCaller(inner, "", "")

	if got != inner {
		t.Fatalf("NewModelProviderOverrideCaller(inner, \"\", \"\") = %T, want inner (%T) unchanged", got, inner)
	}
}

// TestNewModelProviderOverrideCaller_PinsProviderAndModel verifies that a
// non-empty provider and model are stamped onto EVERY call — overriding empty
// and pre-set request values alike — and that the caller's original request is
// not mutated. Because the stamping is stateless, a mid-task router SetModel
// cannot affect a pinned caller: each Call re-asserts the same pair (the
// router half of that guarantee is tested in the llm package:
// TestRouter_Call_ProviderOverride_SurvivesSetModel).
func TestNewModelProviderOverrideCaller_PinsProviderAndModel(t *testing.T) {
	const (
		pinnedProvider = "openai_compatible"
		pinnedModel    = "qwen3:32b"
	)
	inner := &mockLLMCaller{}
	caller := NewModelProviderOverrideCaller(inner, pinnedProvider, pinnedModel)

	cases := []struct {
		name     string
		provider string
		model    string
	}{
		{"empty request fields are filled", "", ""},
		{"pre-set request fields are overridden", "other", "other-model"},
		{"override is stable across calls", "third", "third-model"},
	}

	for i, tc := range cases {
		req := llm.ChatRequest{Provider: tc.provider, Model: tc.model}
		if _, err := caller.Call(context.Background(), req); err != nil {
			t.Fatalf("[%s] unexpected error: %v", tc.name, err)
		}
		if len(inner.calls) <= i {
			t.Fatalf("[%s] expected call %d to be recorded", tc.name, i)
		}
		if got := inner.calls[i].Provider; got != pinnedProvider {
			t.Errorf("[%s] req.Provider = %q, want %q", tc.name, got, pinnedProvider)
		}
		if got := inner.calls[i].Model; got != pinnedModel {
			t.Errorf("[%s] req.Model = %q, want %q", tc.name, got, pinnedModel)
		}
	}

	// The override must not mutate the caller's original request.
	src := llm.ChatRequest{Provider: "orig", Model: "original"}
	if _, err := caller.Call(context.Background(), src); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if src.Provider != "orig" || src.Model != "original" {
		t.Errorf("override mutated the caller's original request: got provider %q model %q, want %q/%q",
			src.Provider, src.Model, "orig", "original")
	}
}

// TestNewModelProviderOverrideCaller_ProviderOnly verifies that an empty model
// leaves req.Model untouched (the router resolves it) while the provider is
// still pinned.
func TestNewModelProviderOverrideCaller_ProviderOnly(t *testing.T) {
	inner := &mockLLMCaller{}
	caller := NewModelProviderOverrideCaller(inner, "openai_compatible", "")

	if _, err := caller.Call(context.Background(), llm.ChatRequest{Model: ""}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(inner.calls) != 1 {
		t.Fatalf("expected 1 recorded call, got %d", len(inner.calls))
	}
	if got := inner.calls[0].Provider; got != "openai_compatible" {
		t.Errorf("req.Provider = %q, want %q", got, "openai_compatible")
	}
	if got := inner.calls[0].Model; got != "" {
		t.Errorf("provider-only override altered req.Model = %q, want \"\" (router resolves)", got)
	}
}

// TestNewModelOverrideCaller_LegacyLeavesProviderUntouched verifies the legacy
// constructor is the model-only form of the provider-aware wrapper: it pins
// req.Model and never touches req.Provider (an empty request provider stays
// empty — the request routes to the router's active provider).
func TestNewModelOverrideCaller_LegacyLeavesProviderUntouched(t *testing.T) {
	inner := &mockLLMCaller{}
	caller := NewModelOverrideCaller(inner, "claude-haiku")

	if _, err := caller.Call(context.Background(), llm.ChatRequest{}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(inner.calls) != 1 {
		t.Fatalf("expected 1 recorded call, got %d", len(inner.calls))
	}
	if got := inner.calls[0].Model; got != "claude-haiku" {
		t.Errorf("req.Model = %q, want %q", got, "claude-haiku")
	}
	if got := inner.calls[0].Provider; got != "" {
		t.Errorf("legacy override set req.Provider = %q, want \"\" (provider routing untouched)", got)
	}
}

// TestNewModelProviderOverrideCaller_DeltaSinkStreams verifies the streaming
// seam is preserved through the wrapper: a non-nil DeltaSink on the request
// reaches the inner caller intact and deltas flow through it.
func TestNewModelProviderOverrideCaller_DeltaSinkStreams(t *testing.T) {
	inner := &streamingInnerCaller{
		deltas: []llm.StreamDelta{
			{Text: "Hel"},
			{Text: "lo"},
		},
		response: &llm.ChatResponse{
			Message:    llm.Message{Role: "assistant", Content: "Hello"},
			StopReason: "end_turn",
		},
	}
	caller := NewModelProviderOverrideCaller(inner, "openai_compatible", "qwen3:32b")

	var got strings.Builder
	req := llm.ChatRequest{
		Provider: "will-be-overwritten",
		Model:    "will-be-overwritten",
		DeltaSink: func(d llm.StreamDelta) error {
			got.WriteString(d.Text)
			return nil
		},
	}
	resp, err := caller.Call(context.Background(), req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.String() != "Hello" {
		t.Errorf("streamed deltas = %q, want %q", got.String(), "Hello")
	}
	if resp == nil || resp.Message.Content != "Hello" {
		t.Errorf("assembled response = %+v, want content %q", resp, "Hello")
	}
}
