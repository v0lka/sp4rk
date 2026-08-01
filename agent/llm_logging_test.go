package agent

import (
	"bytes"
	"context"
	"errors"
	"log/slog"
	"testing"

	"github.com/v0lka/sp4rk/llm"
)

func TestLoggingCaller_CallSuccess_LogsTokenUsage(t *testing.T) {
	inner := &mockLLMCaller{
		responses: []*llm.ChatResponse{
			{
				Message: llm.Message{Role: "assistant", Content: "hi"},
				Usage:   llm.TokenUsage{InputTokens: 100, OutputTokens: 50},
			},
		},
	}

	// Capture slog output.
	var buf bytes.Buffer
	handler := slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug})
	origLogger := slog.Default()
	slog.SetDefault(slog.New(handler))
	defer slog.SetDefault(origLogger)

	caller := NewLoggingLLMCaller(inner, "openai", slog.New(handler))
	req := llm.ChatRequest{Model: "gpt-4o"}
	resp, err := caller.Call(context.Background(), req)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Usage.InputTokens != 100 || resp.Usage.OutputTokens != 50 {
		t.Fatalf("unexpected usage: %+v", resp.Usage)
	}

	logged := buf.String()
	for _, want := range []string{"llm: token usage", "provider=openai", "model=gpt-4o", "input_tokens=100", "output_tokens=50", "total_tokens=150", "stopReason", "toolCallCount"} {
		if !bytes.Contains([]byte(logged), []byte(want)) {
			t.Errorf("log output missing %q; got: %s", want, logged)
		}
	}
}

func TestLoggingCaller_CallError_LogsWarn(t *testing.T) {
	inner := &mockLLMCaller{errors: []error{errors.New("provider down")}}

	// Capture slog output at DEBUG so all levels are recorded.
	var buf bytes.Buffer
	handler := slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug})
	origLogger := slog.Default()
	slog.SetDefault(slog.New(handler))
	defer slog.SetDefault(origLogger)

	caller := NewLoggingLLMCaller(inner, "anthropic", slog.New(handler))
	_, err := caller.Call(context.Background(), llm.ChatRequest{Model: "claude-3"})

	if err == nil {
		t.Fatal("expected error")
	}

	// A failed LLM call must be logged at WARN (not DEBUG) with the provider
	// and the underlying error, so it surfaces at the default log level.
	logged := buf.String()
	for _, want := range []string{"llm: call failed", "level=WARN", "provider=anthropic", "error=\"provider down\""} {
		if !bytes.Contains([]byte(logged), []byte(want)) {
			t.Errorf("log output missing %q; got: %s", want, logged)
		}
	}
}

func TestLoggingCaller_CallError_VisibleAtDefaultLevel(t *testing.T) {
	inner := &mockLLMCaller{errors: []error{errors.New("boom")}}

	// Default handler level is INFO: DEBUG records must be filtered out, but a
	// WARN-level failure log must still appear. This is the bug being fixed —
	// previously the failure was DEBUG and invisible at the default level.
	var buf bytes.Buffer
	handler := slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelInfo})
	origLogger := slog.Default()
	slog.SetDefault(slog.New(handler))
	defer slog.SetDefault(origLogger)

	caller := NewLoggingLLMCaller(inner, "openai", slog.New(handler))
	_, err := caller.Call(context.Background(), llm.ChatRequest{Model: "gpt-4o"})

	if err == nil {
		t.Fatal("expected error")
	}

	logged := buf.String()
	if !bytes.Contains([]byte(logged), []byte("llm: call failed")) {
		t.Errorf("failure log should be visible at default (INFO) level; got: %s", logged)
	}
	if !bytes.Contains([]byte(logged), []byte("level=WARN")) {
		t.Errorf("failure log should be at WARN level; got: %s", logged)
	}
	// The DEBUG-only request log must NOT leak through at INFO level.
	if bytes.Contains([]byte(logged), []byte("llm: request")) {
		t.Errorf("request (DEBUG) log leaked through INFO level; got: %s", logged)
	}
}

func TestLoggingCaller_DelegatesToInner(t *testing.T) {
	wantResp := &llm.ChatResponse{
		Message:    llm.Message{Role: "assistant", Content: "result"},
		Usage:      llm.TokenUsage{InputTokens: 10, OutputTokens: 20},
		StopReason: "end_turn",
	}
	inner := &mockLLMCaller{responses: []*llm.ChatResponse{wantResp}}

	// Suppress log output for this test.
	handler := slog.NewTextHandler(&bytes.Buffer{}, &slog.HandlerOptions{Level: slog.LevelDebug})
	origLogger := slog.Default()
	slog.SetDefault(slog.New(handler))
	defer slog.SetDefault(origLogger)

	caller := NewLoggingLLMCaller(inner, "test", slog.New(handler))
	req := llm.ChatRequest{Model: "m1", Messages: []llm.Message{{Role: "user", Content: "hello"}}}
	resp, err := caller.Call(context.Background(), req)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp != wantResp {
		t.Errorf("expected response to be delegated; got %+v", resp)
	}
	if len(inner.calls) != 1 {
		t.Fatalf("expected 1 call; got %d", len(inner.calls))
	}
	if inner.calls[0].Model != "m1" {
		t.Errorf("expected model m1; got %s", inner.calls[0].Model)
	}
}
