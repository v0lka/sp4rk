package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/v0lka/sp4rk/llm"
	"github.com/v0lka/sp4rk/tools/internal/judge_prompts"
)

func strictResponse(content string) *llm.ChatResponse {
	return &llm.ChatResponse{Message: llm.Message{Content: content}}
}

func TestJudgeStrictPromptCoversOWASPASI(t *testing.T) {
	prompt := judge_prompts.JudgeStrictSystem
	for i := 1; i <= 10; i++ {
		category := "ASI" + string(rune('0'+i/10)) + string(rune('0'+i%10))
		if !strings.Contains(prompt, category) {
			t.Errorf("strict prompt does not mention %s", category)
		}
	}
	for _, required := range []string{
		"mandatory risks", "context makes them applicable", "Path locality alone is never sufficient",
		"Only the verdict tokens ALLOW and CONFIRM are valid",
	} {
		if !strings.Contains(prompt, required) {
			t.Errorf("strict prompt missing policy phrase %q", required)
		}
	}
}

func TestParseStrictJudgeResponse(t *testing.T) {
	tests := []struct {
		name    string
		content string
		verdict JudgeVerdict
		reason  string
	}{
		{name: "allow", content: "VERDICT: ALLOW\nREASON: no material ASI risk", verdict: VerdictAllow, reason: "no material ASI risk"},
		{name: "confirm", content: "VERDICT: CONFIRM\nREASON: ASI05 risk", verdict: VerdictConfirm, reason: "ASI05 risk"},
		{name: "advisory alias rejected", content: "VERDICT: SAFE\nREASON: looks safe", verdict: VerdictConfirm, reason: judgeUnparsedReason},
		{name: "lowercase rejected", content: "VERDICT: allow\nREASON: looks safe", verdict: VerdictConfirm, reason: judgeUnparsedReason},
		{name: "missing reason", content: "VERDICT: ALLOW", verdict: VerdictConfirm, reason: judgeUnparsedReason},
		{name: "prose rejected", content: "Sure.\nVERDICT: ALLOW\nREASON: safe", verdict: VerdictConfirm, reason: judgeUnparsedReason},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			verdict, reason := parseStrictJudgeResponse(tt.content)
			if verdict != tt.verdict || reason != tt.reason {
				t.Fatalf("got (%v, %q), want (%v, %q)", verdict, reason, tt.verdict, tt.reason)
			}
		})
	}
}

func TestJudgeStrictAlwaysCallsLLMAndIncludesContext(t *testing.T) {
	provider := &mockLLMProvider{response: strictResponse("VERDICT: ALLOW\nREASON: bounded read")}
	judge := NewToolJudge(provider, "test-model", 10, nil)
	judge.SetIsInternalFn(func(string) bool { return true })

	ctx := WithWorkspacePath(context.Background(), t.TempDir())
	ctx = WithEnvInfo(ctx, &EnvInfo{OS: "TestOS"})
	input := json.RawMessage(`{"path":"` + WorkspacePathFrom(ctx) + `/file.txt"}`)
	request := StrictJudgeRequest{
		ToolName:    "read_file",
		Input:       input,
		TaskContext: "inspect the requested file",
		ToolSource:  "mcp-filesystem",
	}

	for range 2 {
		verdict, _, err := judge.JudgeStrict(ctx, request)
		if err != nil {
			t.Fatalf("JudgeStrict returned error: %v", err)
		}
		if verdict != VerdictAllow {
			t.Fatalf("expected ALLOW, got %v", verdict)
		}
	}

	requests := provider.snapshot()
	if len(requests) != 2 {
		t.Fatalf("expected one LLM call per strict evaluation, got %d", len(requests))
	}
	for _, got := range requests {
		if got.Messages[0].Content != judge_prompts.JudgeStrictSystem {
			t.Error("strict evaluation did not use strict system prompt")
		}
		var envelope strictJudgeEnvelope
		if err := json.Unmarshal([]byte(got.Messages[1].Content), &envelope); err != nil {
			t.Fatalf("decode strict envelope: %v", err)
		}
		if envelope.TaskContext != request.TaskContext || envelope.ToolSource != request.ToolSource {
			t.Errorf("strict context missing from envelope: %+v", envelope)
		}
		if !strings.Contains(envelope.Environment, "TestOS") {
			t.Errorf("compact environment missing from envelope: %q", envelope.Environment)
		}
		if envelope.Input != string(input) {
			t.Errorf("input changed in envelope: got %s want %s", envelope.Input, input)
		}
	}
}

func TestJudgeStrictCallsLLMForMalformedInput(t *testing.T) {
	provider := &mockLLMProvider{response: strictResponse("VERDICT: CONFIRM\nREASON: malformed input requires review")}
	judge := NewToolJudge(provider, "test-model", 0, nil)
	input := json.RawMessage(`{"unterminated"`)

	verdict, _, err := judge.JudgeStrict(context.Background(), StrictJudgeRequest{
		ToolName:    "write_file",
		Input:       input,
		TaskContext: "update configuration",
		ToolSource:  "core",
	})
	if err != nil || verdict != VerdictConfirm {
		t.Fatalf("got (%v, %v), want provider CONFIRM", verdict, err)
	}
	requests := provider.snapshot()
	if len(requests) != 1 {
		t.Fatalf("expected malformed input to still reach LLM, got %d calls", len(requests))
	}
	var envelope strictJudgeEnvelope
	if err := json.Unmarshal([]byte(requests[0].Messages[1].Content), &envelope); err != nil {
		t.Fatalf("decode strict envelope: %v", err)
	}
	if envelope.Input != string(input) {
		t.Fatalf("malformed input changed: got %q want %q", envelope.Input, input)
	}
}

func TestJudgeStrictDoesNotReuseVerdictAcrossContexts(t *testing.T) {
	provider := &mockLLMProvider{response: strictResponse("VERDICT: CONFIRM\nREASON: context requires review")}
	judge := NewToolJudge(provider, "test-model", 10, nil)
	input := json.RawMessage(`{"command":"tool --check"}`)

	requests := []StrictJudgeRequest{
		{ToolName: "bash_exec", Input: input, TaskContext: "audit project", ToolSource: "core"},
		{ToolName: "bash_exec", Input: input, TaskContext: "deploy release", ToolSource: "mcp-remote"},
	}
	for _, request := range requests {
		if _, _, err := judge.JudgeStrict(context.Background(), request); err != nil {
			t.Fatalf("JudgeStrict returned error: %v", err)
		}
	}

	got := provider.snapshot()
	if len(got) != 2 {
		t.Fatalf("expected isolated LLM evaluations, got %d calls", len(got))
	}
	if got[0].Messages[1].Content == got[1].Messages[1].Content {
		t.Fatal("different task/source contexts produced identical strict prompts")
	}
}

func TestJudgeStrictFailsSafe(t *testing.T) {
	t.Run("provider error", func(t *testing.T) {
		provider := &mockLLMProvider{err: errors.New("provider echoed secret-token")}
		judge := NewToolJudge(provider, "test-model", 0, nil)
		verdict, reason, err := judge.JudgeStrict(context.Background(), StrictJudgeRequest{ToolName: "bash_exec", Input: json.RawMessage(`{"token":"secret-token"}`)})
		if err != nil || verdict != VerdictConfirm || reason != strictJudgeFailureReason {
			t.Fatalf("got (%v, %q, %v), want fail-safe CONFIRM", verdict, reason, err)
		}
	})

	t.Run("cancelled context", func(t *testing.T) {
		provider := &mockLLMProvider{handler: func(ctx context.Context, _ llm.ChatRequest) (*llm.ChatResponse, error) {
			<-ctx.Done()
			return nil, ctx.Err()
		}}
		judge := NewToolJudge(provider, "test-model", 0, nil)
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		verdict, reason, err := judge.JudgeStrict(ctx, StrictJudgeRequest{ToolName: "write_file", Input: json.RawMessage(`{}`)})
		if err != nil || verdict != VerdictConfirm || reason != strictJudgeFailureReason {
			t.Fatalf("got (%v, %q, %v), want fail-safe CONFIRM", verdict, reason, err)
		}
	})

	t.Run("deadline exceeded", func(t *testing.T) {
		provider := &mockLLMProvider{handler: func(ctx context.Context, _ llm.ChatRequest) (*llm.ChatResponse, error) {
			<-ctx.Done()
			return nil, ctx.Err()
		}}
		judge := NewToolJudge(provider, "test-model", 0, nil)
		ctx, cancel := context.WithTimeout(context.Background(), time.Millisecond)
		defer cancel()
		verdict, reason, err := judge.JudgeStrict(ctx, StrictJudgeRequest{ToolName: "write_file", Input: json.RawMessage(`{}`)})
		if err != nil || verdict != VerdictConfirm || reason != strictJudgeFailureReason {
			t.Fatalf("got (%v, %q, %v), want timeout fail-safe CONFIRM", verdict, reason, err)
		}
	})

	t.Run("nil response", func(t *testing.T) {
		judge := NewToolJudge(&mockLLMProvider{}, "test-model", 0, nil)
		verdict, reason, err := judge.JudgeStrict(context.Background(), StrictJudgeRequest{ToolName: "write_file", Input: json.RawMessage(`{}`)})
		if err != nil || verdict != VerdictConfirm || reason != strictJudgeFailureReason {
			t.Fatalf("got (%v, %q, %v), want fail-safe CONFIRM", verdict, reason, err)
		}
	})

	t.Run("unparseable", func(t *testing.T) {
		judge := NewToolJudge(&mockLLMProvider{response: strictResponse("probably okay")}, "test-model", 0, nil)
		verdict, reason, err := judge.JudgeStrict(context.Background(), StrictJudgeRequest{ToolName: "write_file", Input: json.RawMessage(`{}`)})
		if err != nil || verdict != VerdictConfirm || reason != judgeUnparsedReason {
			t.Fatalf("got (%v, %q, %v), want unparseable CONFIRM", verdict, reason, err)
		}
	})
}

func TestJudgeStrictDoesNotLogSensitiveArguments(t *testing.T) {
	const secret = "super-secret-token"
	var logs bytes.Buffer
	logger := slog.New(slog.NewTextHandler(&logs, &slog.HandlerOptions{Level: slog.LevelDebug}))
	provider := &mockLLMProvider{err: errors.New("provider failure containing " + secret)}
	judge := NewToolJudge(provider, "test-model", 0, logger)

	_, _, _ = judge.JudgeStrict(context.Background(), StrictJudgeRequest{
		ToolName: "bash_exec",
		Input:    json.RawMessage(`{"token":"` + secret + `"}`),
	})
	if strings.Contains(logs.String(), secret) {
		t.Fatalf("strict judge logs leaked sensitive tool arguments: %s", logs.String())
	}
}

func TestJudgeStrictConcurrentAccess(t *testing.T) {
	provider := &mockLLMProvider{response: strictResponse("VERDICT: ALLOW\nREASON: bounded read")}
	judge := NewToolJudge(provider, "test-model", 4, nil)

	const workers = 32
	var wg sync.WaitGroup
	wg.Add(workers)
	for i := range workers {
		go func() {
			defer wg.Done()
			request := StrictJudgeRequest{
				ToolName:    "read_file",
				Input:       json.RawMessage(`{"path":"file.txt"}`),
				TaskContext: "worker " + string(rune('A'+i)),
				ToolSource:  "core",
			}
			verdict, _, err := judge.JudgeStrict(context.Background(), request)
			if err != nil || verdict != VerdictAllow {
				t.Errorf("JudgeStrict got (%v, %v)", verdict, err)
			}
		}()
	}
	wg.Wait()

	if got := len(provider.snapshot()); got != workers {
		t.Fatalf("expected %d independent concurrent calls, got %d", workers, got)
	}
}
