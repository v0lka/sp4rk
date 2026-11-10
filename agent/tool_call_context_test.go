package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/v0lka/sp4rk/llm"
	"github.com/v0lka/sp4rk/tools"
	"testing"
)

type traceEmitter struct {
	NoopEvents
	ids []string
}

func (e *traceEmitter) ToolCallWithID(_, _ int, _, _, _ string) string {
	id := fmt.Sprintf("call-%d", len(e.ids)+1)
	e.ids = append(e.ids, id)
	return id
}

type traceTools struct {
	*mockToolExecutor
	ids    []string
	claims []string
}

func (e *traceTools) Execute(ctx context.Context, name string, input json.RawMessage) (tools.ToolResult, error) {
	e.ids = append(e.ids, ToolCallIDFromContext(ctx))
	e.claims = append(e.claims, ClaimToolCallID(ctx, name, input))
	return e.mockToolExecutor.Execute(ctx, name, input)
}

func TestToolCallIDContextSingleBatchDenial(t *testing.T) {
	for _, tc := range []struct {
		name, tool string
		input      json.RawMessage
		want       int
		deny       bool
	}{
		{"single", "synthetic", json.RawMessage(`{}`), 1, false},
		{"batch", tools.ToolBatch, json.RawMessage(`{"calls":[{"tool":"synthetic","input":{}},{"tool":"synthetic","input":{}}]}`), 2, false},
		{"denied", "synthetic", json.RawMessage(`{}`), 0, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			emitter := &traceEmitter{}
			toolExec := &traceTools{mockToolExecutor: newMockToolExecutor()}
			model := &mockLLMCaller{responses: []*llm.ChatResponse{llmResponseWithToolCall("synthetic", tc.tool, tc.input), llmResponseFinish("done", "done")}}
			opts := []Option{WithEvents(emitter), WithTokenCounter(&mockTokenCounter{}), WithCircuitBreaker(defaultCircuitBreakerConfig)}
			if tc.deny {
				opts = append(opts, WithHITL(&denyingHITLHandler{}))
			}
			exec := NewExecutor(model, toolExec, 2, opts...)
			_, err := exec.Run(context.Background(), []tools.ToolDescriptor{{Name: tc.tool}, {Name: "synthetic"}}, newMockContextManager())
			if err != nil {
				t.Fatal(err)
			}
			if len(toolExec.ids) != tc.want {
				t.Fatalf("dispatch IDs=%v, want %d", toolExec.ids, tc.want)
			}
			for i, id := range toolExec.ids {
				if id == "" || id != emitter.ids[i] || toolExec.claims[i] != id {
					t.Fatalf("dispatch %d id=%q claim=%q event=%q", i, id, toolExec.claims[i], emitter.ids[i])
				}
			}
			if tc.deny && len(emitter.ids) != 1 {
				t.Fatalf("denied card IDs=%v", emitter.ids)
			}
		})
	}
}

type cancelTraceTools struct {
	*mockToolExecutor
	started chan string
}

func (e *cancelTraceTools) Execute(ctx context.Context, _ string, _ json.RawMessage) (tools.ToolResult, error) {
	e.started <- ToolCallIDFromContext(ctx)
	<-ctx.Done()
	return tools.ToolResult{}, ctx.Err()
}

func TestToolCallIDContextCancellation(t *testing.T) {
	emitter := &traceEmitter{}
	toolExec := &cancelTraceTools{mockToolExecutor: newMockToolExecutor(), started: make(chan string, 1)}
	model := &mockLLMCaller{responses: []*llm.ChatResponse{llmResponseWithToolCall("synthetic", "synthetic", json.RawMessage(`{}`))}}
	exec := NewExecutor(model, toolExec, 10, WithEvents(emitter), WithTokenCounter(&mockTokenCounter{}), WithCircuitBreaker(defaultCircuitBreakerConfig))
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		_, err := exec.Run(ctx, []tools.ToolDescriptor{{Name: "synthetic"}}, newMockContextManager())
		done <- err
	}()
	id := <-toolExec.started
	cancel()
	err := <-done
	if id == "" || len(emitter.ids) != 1 || id != emitter.ids[0] {
		t.Fatalf("in-flight ID=%q events=%v", id, emitter.ids)
	}
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation error=%v", err)
	}
	if ToolCallIDFromContext(context.Background()) != "" {
		t.Fatal("ID escaped to unrelated context")
	}
}

func TestToolCallIDContextShadowsParent(t *testing.T) {
	parent := withToolCallID(context.Background(), "outer", "synthetic", json.RawMessage(`{}`))
	if got := ToolCallIDFromContext(WithoutToolCallID(parent)); got != "" {
		t.Fatalf("nested call inherited outer ID %q", got)
	}
	if got := ToolCallIDFromContext(withToolCallID(parent, "inner", "synthetic", json.RawMessage(`{}`))); got != "inner" {
		t.Fatalf("nested ID=%q", got)
	}
}

func TestToolCallIDClaimMatchesOnce(t *testing.T) {
	type unrelatedContextKey struct{}
	input := json.RawMessage(`{"phase":"operator"}`)
	ctx := withToolCallID(context.Background(), "call-a", "c0pen_http_check", input)
	if got := ClaimToolCallID(ctx, "other", input); got != "" {
		t.Fatalf("wrong tool claimed %q", got)
	}
	if got := ClaimToolCallID(ctx, "c0pen_http_check", json.RawMessage(`{"phase":"model"}`)); got != "" {
		t.Fatalf("wrong input claimed %q", got)
	}
	if got := ClaimToolCallID(ctx, "c0pen_http_check", input); got != "call-a" {
		t.Fatalf("matching tool claimed %q", got)
	}
	if got := ClaimToolCallID(context.WithValue(ctx, unrelatedContextKey{}, true), "c0pen_http_check", input); got != "" {
		t.Fatalf("inherited binding was reused: %q", got)
	}
}

type modifyingCallHITL struct {
	NoopHITLHandler
	replacement json.RawMessage
}

func (h modifyingCallHITL) OnToolCall(context.Context, string, json.RawMessage) (*HITLToolDecision, error) {
	return &HITLToolDecision{Allow: true, ModifiedInput: h.replacement}, nil
}

type postHITLTraceTools struct {
	*mockToolExecutor
	modelInput json.RawMessage
	seen       []string
	wrong      []string
	claims     []string
}

func (e *postHITLTraceTools) Execute(ctx context.Context, name string, input json.RawMessage) (tools.ToolResult, error) {
	e.seen = append(e.seen, string(input))
	e.wrong = append(e.wrong, ClaimToolCallID(ctx, name, e.modelInput))
	e.claims = append(e.claims, ClaimToolCallID(ctx, name, input))
	return e.mockToolExecutor.Execute(ctx, name, input)
}

func TestToolCallIDClaimUsesPostHITLInputSingleAndBatch(t *testing.T) {
	modelInput := json.RawMessage(`{"phase":"model"}`)
	operatorInput := json.RawMessage(`{"phase":"operator"}`)
	for _, tc := range []struct {
		name, tool string
		input      json.RawMessage
		want       int
	}{
		{"single", "synthetic", modelInput, 1},
		{"batch", tools.ToolBatch, json.RawMessage(`{"calls":[{"tool":"synthetic","input":{"phase":"model"}},{"tool":"synthetic","input":{"phase":"model"}}]}`), 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			emitter := &traceEmitter{}
			toolExec := &postHITLTraceTools{mockToolExecutor: newMockToolExecutor(), modelInput: modelInput}
			model := &mockLLMCaller{responses: []*llm.ChatResponse{llmResponseWithToolCall("synthetic", tc.tool, tc.input), llmResponseFinish("done", "done")}}
			exec := NewExecutor(model, toolExec, 2, WithEvents(emitter), WithHITL(modifyingCallHITL{replacement: operatorInput}), WithTokenCounter(&mockTokenCounter{}), WithCircuitBreaker(defaultCircuitBreakerConfig))
			if _, err := exec.Run(context.Background(), []tools.ToolDescriptor{{Name: tc.tool}, {Name: "synthetic"}}, newMockContextManager()); err != nil {
				t.Fatal(err)
			}
			if len(toolExec.claims) != tc.want || len(emitter.ids) != tc.want {
				t.Fatalf("claims=%v events=%v, want %d", toolExec.claims, emitter.ids, tc.want)
			}
			for i := range toolExec.claims {
				if toolExec.seen[i] != string(operatorInput) || toolExec.wrong[i] != "" || toolExec.claims[i] != emitter.ids[i] {
					t.Fatalf("call %d: seen=%q wrong=%q claimed=%q event=%q", i, toolExec.seen[i], toolExec.wrong[i], toolExec.claims[i], emitter.ids[i])
				}
			}
		})
	}
}
