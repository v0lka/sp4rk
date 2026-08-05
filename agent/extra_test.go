package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/v0lka/sp4rk/llm"
	"github.com/v0lka/sp4rk/tools"
)

// ============================================================================
// Fruitless Result Detector: OnStepLimit paths for abort
// ============================================================================

func TestExecutor_Run_FruitlessDetector_Abort_StepLimitAllowOnce(t *testing.T) {
	fruitlessConfig := CircuitBreakerConfig{
		RepeatNudgeThreshold:         50,
		RepeatAbortThreshold:         100,
		TruncationAbortThreshold:     100,
		ParseErrorAbortThreshold:     100,
		FruitlessNudgeThreshold:      5,
		FruitlessAbortThreshold:      8,
		FruitlessMaxResultLen:        32,
		SameToolRepeatNudgeThreshold: 50,
		SameToolRepeatAbortThreshold: 100,
		SameToolResultSizeDelta:      64,
	}

	responses := make([]*llm.ChatResponse, 10)
	for i := 0; i < 9; i++ {
		responses[i] = llmResponseWithToolCall(
			fmt.Sprintf("search %d", i+1),
			"search",
			json.RawMessage(fmt.Sprintf(`{"q":"test%d"}`, i+1)),
		)
	}
	responses[9] = llmResponseFinish("done", "completed")

	mockLLM := &mockLLMCaller{responses: responses}
	mockTools := newMockToolExecutor()
	mockTools.results["search"] = tools.ToolResult{Content: "short", IsError: false}

	cm := newMockContextManager()
	exec := newExecutorDefaultHITL(mockLLM, mockTools, &mockTokenCounter{}, 20, nil, false, ToolResultBudget{}, fruitlessConfig)

	callCount := 0
	exec.SetHITLHandler(&testStepLimitAdapter{fn: func(ctx context.Context, currentStep int, maxSteps int, reason string) (StepLimitResponse, error) {
		callCount++
		return StepLimitAllowOnce, nil
	}})

	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "search", Description: "search", Source: "core"},
	}, cm)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.Finished {
		t.Error("expected Finished=true after fruitless abort → AllowOnce → finish")
	}
	if callCount != 1 {
		t.Errorf("expected OnStepLimit to be called once, got %d", callCount)
	}
}

func TestExecutor_Run_FruitlessDetector_Abort_StepLimitAllowAlways(t *testing.T) {
	fruitlessConfig := CircuitBreakerConfig{
		RepeatNudgeThreshold:         50,
		RepeatAbortThreshold:         100,
		TruncationAbortThreshold:     100,
		ParseErrorAbortThreshold:     100,
		FruitlessNudgeThreshold:      5,
		FruitlessAbortThreshold:      8,
		FruitlessMaxResultLen:        32,
		SameToolRepeatNudgeThreshold: 50,
		SameToolRepeatAbortThreshold: 100,
		SameToolResultSizeDelta:      64,
	}

	responses := make([]*llm.ChatResponse, 10)
	for i := 0; i < 9; i++ {
		responses[i] = llmResponseWithToolCall(
			fmt.Sprintf("search %d", i+1),
			"search",
			json.RawMessage(fmt.Sprintf(`{"q":"test%d"}`, i+1)),
		)
	}
	responses[9] = llmResponseFinish("done", "completed")

	mockLLM := &mockLLMCaller{responses: responses}
	mockTools := newMockToolExecutor()
	mockTools.results["search"] = tools.ToolResult{Content: "short", IsError: false}

	cm := newMockContextManager()
	exec := newExecutorDefaultHITL(mockLLM, mockTools, &mockTokenCounter{}, 20, nil, false, ToolResultBudget{}, fruitlessConfig)

	callCount := 0
	exec.SetHITLHandler(&testStepLimitAdapter{fn: func(ctx context.Context, currentStep int, maxSteps int, reason string) (StepLimitResponse, error) {
		callCount++
		return StepLimitAllowAlways, nil
	}})

	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "search", Description: "search", Source: "core"},
	}, cm)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.Finished {
		t.Error("expected Finished=true after fruitless abort → AllowAlways → finish")
	}
	if callCount != 1 {
		t.Errorf("expected OnStepLimit to be called once, got %d", callCount)
	}
}

func TestExecutor_Run_FruitlessDetector_Abort_StepLimitDeny(t *testing.T) {
	fruitlessConfig := CircuitBreakerConfig{
		RepeatNudgeThreshold:         50,
		RepeatAbortThreshold:         100,
		TruncationAbortThreshold:     100,
		ParseErrorAbortThreshold:     100,
		FruitlessNudgeThreshold:      5,
		FruitlessAbortThreshold:      8,
		FruitlessMaxResultLen:        32,
		SameToolRepeatNudgeThreshold: 50,
		SameToolRepeatAbortThreshold: 100,
		SameToolResultSizeDelta:      64,
	}

	responses := make([]*llm.ChatResponse, 15)
	for i := 0; i < 15; i++ {
		responses[i] = llmResponseWithToolCall(
			fmt.Sprintf("search %d", i+1),
			"search",
			json.RawMessage(fmt.Sprintf(`{"q":"test%d"}`, i+1)),
		)
	}

	mockLLM := &mockLLMCaller{responses: responses}
	mockTools := newMockToolExecutor()
	mockTools.results["search"] = tools.ToolResult{Content: "short", IsError: false}

	cm := newMockContextManager()
	exec := newExecutorDefaultHITL(mockLLM, mockTools, &mockTokenCounter{}, 20, nil, false, ToolResultBudget{}, fruitlessConfig)

	exec.SetHITLHandler(&testStepLimitAdapter{fn: func(ctx context.Context, currentStep int, maxSteps int, reason string) (StepLimitResponse, error) {
		return StepLimitDeny, nil
	}})

	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "search", Description: "search", Source: "core"},
	}, cm)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Finished {
		t.Error("expected Finished=false after fruitless abort → Deny")
	}
	if !strings.Contains(result.Output, "empty or minimal results") {
		t.Errorf("expected fruitless abort message, got %q", result.Output)
	}
}

// ============================================================================
// SameTool Repetition: OnStepLimit paths for abort
// ============================================================================

func TestExecutor_Run_SameToolRepeat_Abort_StepLimitAllowOnce(t *testing.T) {
	sameToolConfig := CircuitBreakerConfig{
		RepeatNudgeThreshold:         50,
		RepeatAbortThreshold:         100,
		TruncationAbortThreshold:     100,
		ParseErrorAbortThreshold:     100,
		FruitlessNudgeThreshold:      50,
		FruitlessAbortThreshold:      100,
		FruitlessMaxResultLen:        32,
		SameToolRepeatNudgeThreshold: 8,
		SameToolRepeatAbortThreshold: 12,
		SameToolResultSizeDelta:      64,
	}

	responses := make([]*llm.ChatResponse, 14)
	for i := 0; i < 13; i++ {
		responses[i] = llmResponseWithToolCall(
			fmt.Sprintf("search %d", i+1),
			"search",
			json.RawMessage(fmt.Sprintf(`{"q":"query%d"}`, i+1)),
		)
	}
	responses[13] = llmResponseFinish("done", "completed")

	mockLLM := &mockLLMCaller{responses: responses}
	mockTools := newMockToolExecutor()
	mockTools.results["search"] = tools.ToolResult{Content: strings.Repeat("x", 50), IsError: false}

	cm := newMockContextManager()
	exec := newExecutorDefaultHITL(mockLLM, mockTools, &mockTokenCounter{}, 20, nil, false, ToolResultBudget{}, sameToolConfig)

	callCount := 0
	exec.SetHITLHandler(&testStepLimitAdapter{fn: func(ctx context.Context, currentStep int, maxSteps int, reason string) (StepLimitResponse, error) {
		callCount++
		return StepLimitAllowOnce, nil
	}})

	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "search", Description: "search", Source: "core"},
	}, cm)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.Finished {
		t.Error("expected Finished=true after same-tool abort → AllowOnce → finish")
	}
	if callCount != 1 {
		t.Errorf("expected OnStepLimit to be called once, got %d", callCount)
	}
}

func TestExecutor_Run_SameToolRepeat_Abort_StepLimitAllowAlways(t *testing.T) {
	sameToolConfig := CircuitBreakerConfig{
		RepeatNudgeThreshold:         50,
		RepeatAbortThreshold:         100,
		TruncationAbortThreshold:     100,
		ParseErrorAbortThreshold:     100,
		FruitlessNudgeThreshold:      50,
		FruitlessAbortThreshold:      100,
		FruitlessMaxResultLen:        32,
		SameToolRepeatNudgeThreshold: 8,
		SameToolRepeatAbortThreshold: 12,
		SameToolResultSizeDelta:      64,
	}

	responses := make([]*llm.ChatResponse, 14)
	for i := 0; i < 13; i++ {
		responses[i] = llmResponseWithToolCall(
			fmt.Sprintf("search %d", i+1),
			"search",
			json.RawMessage(fmt.Sprintf(`{"q":"query%d"}`, i+1)),
		)
	}
	responses[13] = llmResponseFinish("done", "completed")

	mockLLM := &mockLLMCaller{responses: responses}
	mockTools := newMockToolExecutor()
	mockTools.results["search"] = tools.ToolResult{Content: strings.Repeat("x", 50), IsError: false}

	cm := newMockContextManager()
	exec := newExecutorDefaultHITL(mockLLM, mockTools, &mockTokenCounter{}, 20, nil, false, ToolResultBudget{}, sameToolConfig)

	callCount := 0
	exec.SetHITLHandler(&testStepLimitAdapter{fn: func(ctx context.Context, currentStep int, maxSteps int, reason string) (StepLimitResponse, error) {
		callCount++
		return StepLimitAllowAlways, nil
	}})

	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "search", Description: "search", Source: "core"},
	}, cm)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.Finished {
		t.Error("expected Finished=true after same-tool abort → AllowAlways → finish")
	}
	if callCount != 1 {
		t.Errorf("expected OnStepLimit to be called once, got %d", callCount)
	}
}

func TestExecutor_Run_SameToolRepeat_Abort_StepLimitDeny(t *testing.T) {
	sameToolConfig := CircuitBreakerConfig{
		RepeatNudgeThreshold:         50,
		RepeatAbortThreshold:         100,
		TruncationAbortThreshold:     100,
		ParseErrorAbortThreshold:     100,
		FruitlessNudgeThreshold:      50,
		FruitlessAbortThreshold:      100,
		FruitlessMaxResultLen:        32,
		SameToolRepeatNudgeThreshold: 8,
		SameToolRepeatAbortThreshold: 12,
		SameToolResultSizeDelta:      64,
	}

	responses := make([]*llm.ChatResponse, 15)
	for i := 0; i < 15; i++ {
		responses[i] = llmResponseWithToolCall(
			fmt.Sprintf("search %d", i+1),
			"search",
			json.RawMessage(fmt.Sprintf(`{"q":"query%d"}`, i+1)),
		)
	}

	mockLLM := &mockLLMCaller{responses: responses}
	mockTools := newMockToolExecutor()
	mockTools.results["search"] = tools.ToolResult{Content: strings.Repeat("x", 50), IsError: false}

	cm := newMockContextManager()
	exec := newExecutorDefaultHITL(mockLLM, mockTools, &mockTokenCounter{}, 20, nil, false, ToolResultBudget{}, sameToolConfig)

	exec.SetHITLHandler(&testStepLimitAdapter{fn: func(ctx context.Context, currentStep int, maxSteps int, reason string) (StepLimitResponse, error) {
		return StepLimitDeny, nil
	}})

	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "search", Description: "search", Source: "core"},
	}, cm)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Finished {
		t.Error("expected Finished=false after same-tool abort → Deny")
	}
	if !strings.Contains(result.Output, "similar results") {
		t.Errorf("expected same-tool abort message, got %q", result.Output)
	}
}

// ============================================================================
// Parse Error: OnStepLimit paths for abort + reset test
// ============================================================================

func TestExecutor_Run_ConsecutiveParseErrors_Abort_StepLimitAllowOnce(t *testing.T) {
	responses := make([]*llm.ChatResponse, 4)
	for i := range responses {
		responses[i] = llmResponseWithToolCall(
			fmt.Sprintf("attempt %d", i+1),
			"create_file",
			json.RawMessage(fmt.Sprintf(`{"bad_input":%d}`, i+1)),
		)
	}

	mockLLM := &mockLLMCaller{responses: responses}
	mockTools := newMockToolExecutor()
	mockTools.results["create_file"] = tools.ToolResult{
		Content: "failed to parse input: invalid field type",
		IsError: true,
	}

	cm := newMockContextManager()
	exec := newExecutorDefaultHITL(mockLLM, mockTools, &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)

	callCount := 0
	exec.SetHITLHandler(&testStepLimitAdapter{fn: func(ctx context.Context, currentStep int, maxSteps int, reason string) (StepLimitResponse, error) {
		callCount++
		return StepLimitAllowOnce, nil
	}})

	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "create_file", Description: "create a file", Source: "core"},
	}, cm)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// After AllowOnce resets the counter, the 4th parse error bumps count to 1.
	// Then mock LLM returns default end_turn → implicit finish → Finished=true.
	if !result.Finished {
		t.Error("expected Finished=true after AllowOnce → one more parse error → end_turn")
	}
	if callCount != 1 {
		t.Errorf("expected OnStepLimit to be called once, got %d", callCount)
	}
}

func TestExecutor_Run_ConsecutiveParseErrors_Abort_StepLimitAllowAlways(t *testing.T) {
	responses := make([]*llm.ChatResponse, 5)
	for i := 0; i < 4; i++ {
		responses[i] = llmResponseWithToolCall(
			fmt.Sprintf("attempt %d", i+1),
			"create_file",
			json.RawMessage(fmt.Sprintf(`{"bad_input":%d}`, i+1)),
		)
	}
	responses[4] = llmResponseFinish("done", "completed")

	mockLLM := &mockLLMCaller{responses: responses}
	mockTools := newMockToolExecutor()
	mockTools.results["create_file"] = tools.ToolResult{
		Content: "failed to parse input: invalid field type",
		IsError: true,
	}

	cm := newMockContextManager()
	exec := newExecutorDefaultHITL(mockLLM, mockTools, &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)

	callCount := 0
	exec.SetHITLHandler(&testStepLimitAdapter{fn: func(ctx context.Context, currentStep int, maxSteps int, reason string) (StepLimitResponse, error) {
		callCount++
		return StepLimitAllowAlways, nil
	}})

	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "create_file", Description: "create a file", Source: "core"},
	}, cm)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.Finished {
		t.Error("expected Finished=true after AllowAlways → finish")
	}
	if callCount != 1 {
		t.Errorf("expected OnStepLimit to be called once, got %d", callCount)
	}
}

func TestExecutor_Run_ConsecutiveParseErrors_Abort_StepLimitDeny(t *testing.T) {
	responses := make([]*llm.ChatResponse, 3)
	for i := range responses {
		responses[i] = llmResponseWithToolCall(
			fmt.Sprintf("attempt %d", i+1),
			"create_file",
			json.RawMessage(fmt.Sprintf(`{"bad_input":%d}`, i+1)),
		)
	}

	mockLLM := &mockLLMCaller{responses: responses}
	mockTools := newMockToolExecutor()
	mockTools.results["create_file"] = tools.ToolResult{
		Content: "failed to parse input: invalid field type",
		IsError: true,
	}

	cm := newMockContextManager()
	exec := newExecutorDefaultHITL(mockLLM, mockTools, &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)

	exec.SetHITLHandler(&testStepLimitAdapter{fn: func(ctx context.Context, currentStep int, maxSteps int, reason string) (StepLimitResponse, error) {
		return StepLimitDeny, nil
	}})

	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "create_file", Description: "create a file", Source: "core"},
	}, cm)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Finished {
		t.Error("expected Finished=false after parse error abort → Deny")
	}
	if !strings.Contains(result.Output, "failed to parse input") {
		t.Errorf("expected parse error abort message, got %q", result.Output)
	}
}

func TestExecutor_Run_ParseErrorResetsOnSuccess(t *testing.T) {
	cfg := defaultCircuitBreakerConfig
	cfg.ParseErrorAbortThreshold = 3

	responses := []*llm.ChatResponse{
		llmResponseWithToolCall("attempt 1", "tool_a", json.RawMessage(`{"x":"y"}`)),
		llmResponseWithToolCall("success", "tool_b", json.RawMessage(`{"good":"input"}`)),
		llmResponseWithToolCall("attempt 2", "tool_a", json.RawMessage(`{"x":"y"}`)),
		llmResponseFinish("done", "completed"),
	}

	mockLLM := &mockLLMCaller{responses: responses}
	mockTools := newMockToolExecutor()
	mockTools.results["tool_a"] = tools.ToolResult{
		Content: "failed to parse input: invalid field type",
		IsError: true,
	}
	mockTools.results["tool_b"] = tools.ToolResult{
		Content: "successful result data",
		IsError: false,
	}

	cm := newMockContextManager()
	exec := newExecutorDefaultHITL(mockLLM, mockTools, &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, cfg)

	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "tool_a", Description: "tool a", Source: "core"},
		{Name: "tool_b", Description: "tool b", Source: "core"},
	}, cm)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.Finished {
		t.Error("expected Finished=true (parse counter should reset after success)")
	}
}

// ============================================================================
// Truncation: OnStepLimit paths for abort
// ============================================================================

func TestExecutor_Run_ConsecutiveTruncation_Abort_StepLimitAllowOnce(t *testing.T) {
	responses := make([]*llm.ChatResponse, 4)
	for i := 0; i < 3; i++ {
		responses[i] = &llm.ChatResponse{
			Message: llm.Message{
				Role:    "assistant",
				Content: fmt.Sprintf("attempt %d", i+1),
				ToolCalls: []llm.ToolCall{
					{ID: fmt.Sprintf("call_%d", i), Name: "write_file", Input: json.RawMessage(`{"content":"tr`)},
				},
			},
			StopReason: "max_tokens",
			Usage:      llm.TokenUsage{InputTokens: 100, OutputTokens: 4096},
		}
	}
	responses[3] = llmResponseFinish("done", "completed")

	mockLLM := &mockLLMCaller{responses: responses}
	mockTools := newMockToolExecutor()
	cm := newMockContextManager()

	exec := newExecutorDefaultHITL(mockLLM, mockTools, &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
	callCount := 0
	exec.SetHITLHandler(&testStepLimitAdapter{fn: func(ctx context.Context, currentStep int, maxSteps int, reason string) (StepLimitResponse, error) {
		callCount++
		return StepLimitAllowOnce, nil
	}})

	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "write_file", Description: "write a file", Source: "core"},
	}, cm)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.Finished {
		t.Error("expected Finished=true after truncation abort → AllowOnce → finish")
	}
	if callCount != 1 {
		t.Errorf("expected OnStepLimit to be called once, got %d", callCount)
	}
}

func TestExecutor_Run_ConsecutiveTruncation_Abort_StepLimitAllowAlways(t *testing.T) {
	responses := make([]*llm.ChatResponse, 5)
	for i := 0; i < 4; i++ {
		responses[i] = &llm.ChatResponse{
			Message: llm.Message{
				Role:    "assistant",
				Content: fmt.Sprintf("attempt %d", i+1),
				ToolCalls: []llm.ToolCall{
					{ID: fmt.Sprintf("call_%d", i), Name: "write_file", Input: json.RawMessage(`{"content":"tr`)},
				},
			},
			StopReason: "max_tokens",
			Usage:      llm.TokenUsage{InputTokens: 100, OutputTokens: 4096},
		}
	}
	responses[4] = llmResponseFinish("done", "completed")

	mockLLM := &mockLLMCaller{responses: responses}
	mockTools := newMockToolExecutor()
	cm := newMockContextManager()

	exec := newExecutorDefaultHITL(mockLLM, mockTools, &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
	callCount := 0
	exec.SetHITLHandler(&testStepLimitAdapter{fn: func(ctx context.Context, currentStep int, maxSteps int, reason string) (StepLimitResponse, error) {
		callCount++
		return StepLimitAllowAlways, nil
	}})

	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "write_file", Description: "write a file", Source: "core"},
	}, cm)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.Finished {
		t.Error("expected Finished=true after truncation abort → AllowAlways → finish")
	}
	if callCount != 1 {
		t.Errorf("expected OnStepLimit to be called once, got %d", callCount)
	}
}

// ============================================================================
// Repeat Identical Tool: OnStepLimit paths for abort
// ============================================================================

func TestExecutor_Run_CircuitBreaker_Abort_StepLimitAllowOnce(t *testing.T) {
	responses := []*llm.ChatResponse{
		llmResponseWithToolCall("call 1", "search", json.RawMessage(`{"q":"same"}`)),
		llmResponseWithToolCall("call 2", "search", json.RawMessage(`{"q":"same"}`)),
		llmResponseWithToolCall("call 3", "search", json.RawMessage(`{"q":"same"}`)),
		llmResponseWithToolCall("call 4", "search", json.RawMessage(`{"q":"same"}`)),
		llmResponseFinish("done", "finished at last"),
	}

	mockLLM := &mockLLMCaller{responses: responses}
	mockTools := newMockToolExecutor()
	mockTools.results["search"] = tools.ToolResult{Content: "found", IsError: false}

	cm := newMockContextManager()
	exec := newExecutorDefaultHITL(mockLLM, mockTools, &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)

	callCount := 0
	exec.SetHITLHandler(&testStepLimitAdapter{fn: func(ctx context.Context, currentStep int, maxSteps int, reason string) (StepLimitResponse, error) {
		callCount++
		return StepLimitAllowOnce, nil
	}})

	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "search", Description: "search", Source: "core"},
	}, cm)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.Finished {
		t.Error("expected Finished=true after circuit breaker abort → AllowOnce → finish")
	}
	if callCount != 1 {
		t.Errorf("expected OnStepLimit to be called once, got %d", callCount)
	}
}

func TestExecutor_Run_CircuitBreaker_Abort_StepLimitAllowAlways(t *testing.T) {
	responses := []*llm.ChatResponse{
		llmResponseWithToolCall("call 1", "search", json.RawMessage(`{"q":"same"}`)),
		llmResponseWithToolCall("call 2", "search", json.RawMessage(`{"q":"same"}`)),
		llmResponseWithToolCall("call 3", "search", json.RawMessage(`{"q":"same"}`)),
		llmResponseWithToolCall("call 4", "search", json.RawMessage(`{"q":"same"}`)),
		llmResponseFinish("done", "finished"),
	}

	mockLLM := &mockLLMCaller{responses: responses}
	mockTools := newMockToolExecutor()
	mockTools.results["search"] = tools.ToolResult{Content: "found", IsError: false}

	cm := newMockContextManager()
	exec := newExecutorDefaultHITL(mockLLM, mockTools, &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)

	callCount := 0
	exec.SetHITLHandler(&testStepLimitAdapter{fn: func(ctx context.Context, currentStep int, maxSteps int, reason string) (StepLimitResponse, error) {
		callCount++
		return StepLimitAllowAlways, nil
	}})

	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "search", Description: "search", Source: "core"},
	}, cm)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.Finished {
		t.Error("expected Finished=true after circuit breaker abort → AllowAlways → finish")
	}
	if callCount != 1 {
		t.Errorf("expected OnStepLimit to be called once, got %d", callCount)
	}
}

func TestExecutor_Run_CircuitBreaker_Abort_StepLimitDeny(t *testing.T) {
	responses := []*llm.ChatResponse{
		llmResponseWithToolCall("call 1", "search", json.RawMessage(`{"q":"same"}`)),
		llmResponseWithToolCall("call 2", "search", json.RawMessage(`{"q":"same"}`)),
		llmResponseWithToolCall("call 3", "search", json.RawMessage(`{"q":"same"}`)),
		llmResponseWithToolCall("call 4", "search", json.RawMessage(`{"q":"same"}`)),
	}

	mockLLM := &mockLLMCaller{responses: responses}
	mockTools := newMockToolExecutor()
	mockTools.results["search"] = tools.ToolResult{Content: "found", IsError: false}

	cm := newMockContextManager()
	exec := newExecutorDefaultHITL(mockLLM, mockTools, &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)

	exec.SetHITLHandler(&testStepLimitAdapter{fn: func(ctx context.Context, currentStep int, maxSteps int, reason string) (StepLimitResponse, error) {
		return StepLimitDeny, nil
	}})

	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "search", Description: "search", Source: "core"},
	}, cm)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Finished {
		t.Error("expected Finished=false after circuit breaker abort → Deny")
	}
	if !strings.Contains(result.Output, "Aborted") {
		t.Errorf("expected abort message, got %q", result.Output)
	}
}

func TestExecutor_Run_CircuitBreaker_ErrorAware_AbortWithDeny(t *testing.T) {
	responses := []*llm.ChatResponse{
		llmResponseWithToolCall("call 1", "search", json.RawMessage(`{"q":"same"}`)),
		llmResponseWithToolCall("call 2", "search", json.RawMessage(`{"q":"same"}`)),
		llmResponseWithToolCall("call 3", "search", json.RawMessage(`{"q":"same"}`)),
	}

	mockLLM := &mockLLMCaller{responses: responses}
	mockTools := newMockToolExecutor()
	mockTools.results["search"] = tools.ToolResult{Content: "error: not found", IsError: true}

	cm := newMockContextManager()
	exec := newExecutorDefaultHITL(mockLLM, mockTools, &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)

	exec.SetHITLHandler(&testStepLimitAdapter{fn: func(ctx context.Context, currentStep int, maxSteps int, reason string) (StepLimitResponse, error) {
		return StepLimitDeny, nil
	}})

	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "search", Description: "search", Source: "core"},
	}, cm)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Finished {
		t.Error("expected Finished=false after error-aware circuit breaker abort → Deny")
	}
}

// ============================================================================
// handleImplicitFinish: finish_nudge path (suppressAssistantEvents=true)
// ============================================================================

func TestExecutor_Run_ImplicitFinish_FinishNudge(t *testing.T) {
	// suppressAssistantEvents=true enables the finish_nudge path in handleImplicitFinish
	responses := []*llm.ChatResponse{
		{Message: llm.Message{Role: "assistant", Content: "answer"}, StopReason: "end_turn"},
		llmResponseFinish("done", "completed"),
	}
	mockLLM := &mockLLMCaller{responses: responses}
	mockTools := newMockToolExecutor()
	cm := newMockContextManager()
	exec := newExecutorDefaultHITL(mockLLM, mockTools, &mockTokenCounter{}, 5, nil, true, ToolResultBudget{}, defaultCircuitBreakerConfig)
	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "search", Description: "search", Source: "core"},
	}, cm)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.Finished {
		t.Error("expected Finished=true after finish_nudge → finish call")
	}
}

func TestExecutor_Run_ImplicitFinish_NonEndTurn_Nudge(t *testing.T) {
	// Non-end_turn stop reason with no tool calls — enters the nudge path
	responses := []*llm.ChatResponse{
		{Message: llm.Message{Role: "assistant", Content: "no tools"}, StopReason: "stop"},
		llmResponseWithToolCall("now", "search", json.RawMessage(`{"q":"x"}`)),
		llmResponseFinish("done", "finished"),
	}
	mockLLM := &mockLLMCaller{responses: responses}
	mockTools := newMockToolExecutor()
	mockTools.results["search"] = tools.ToolResult{Content: "result", IsError: false}
	cm := newMockContextManager()
	exec := newExecutorDefaultHITL(mockLLM, mockTools, &mockTokenCounter{}, 5, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "search", Description: "search", Source: "core"},
	}, cm)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.Finished {
		t.Error("expected Finished=true after non-end_turn nudge → tool call → finish")
	}
}

func TestExecutor_Run_ImplicitFinish_NonEndTurn_ImplicitFinish(t *testing.T) {
	// Non-end_turn with nudge already attempted — implicit finish path
	responses := []*llm.ChatResponse{
		{Message: llm.Message{Role: "assistant", Content: "no tools"}, StopReason: "stop"},
		{Message: llm.Message{Role: "assistant", Content: "still no tools"}, StopReason: "stop"},
	}
	mockLLM := &mockLLMCaller{responses: responses}
	mockTools := newMockToolExecutor()
	cm := newMockContextManager()
	exec := newExecutorDefaultHITL(mockLLM, mockTools, &mockTokenCounter{}, 5, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "search", Description: "search", Source: "core"},
	}, cm)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.Finished {
		t.Error("expected Finished=true after nudge attempted → implicit finish")
	}
}

func TestExecutor_Run_ImplicitFinish_NonEndTurn_FinishNudge(t *testing.T) {
	// suppressAssistantEvents=true, non-end_turn → finish_nudge path (line 193)
	responses := []*llm.ChatResponse{
		{Message: llm.Message{Role: "assistant", Content: "no tools"}, StopReason: "stop"},
		llmResponseFinish("done", "finished via finish tool"),
	}
	mockLLM := &mockLLMCaller{responses: responses}
	mockTools := newMockToolExecutor()
	cm := newMockContextManager()
	exec := newExecutorDefaultHITL(mockLLM, mockTools, &mockTokenCounter{}, 5, nil, true, ToolResultBudget{}, defaultCircuitBreakerConfig)
	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "search", Description: "search", Source: "core"},
	}, cm)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.Finished {
		t.Error("expected Finished=true after non-end_turn finish_nudge → finish")
	}
}

// ============================================================================
// processToolResult: cache + truncation nudge path
// ============================================================================

func TestExecutor_Run_ProcessToolResult_TruncationCache(t *testing.T) {
	// Trigger per-tool truncation which sets wasTruncated=true,
	// then the tool cache path appends a fragmentation nudge.
	tb := ToolResultBudget{HardCapTokens: 1 << 30}
	ptt := map[string]ToolTruncationConfig{
		"search": {MaxLines: 1, MaxBytes: 50},
	}
	responses := []*llm.ChatResponse{
		llmResponseWithToolCall("search", "search", json.RawMessage(`{"q":"test"}`)),
		llmResponseFinish("done", "done"),
	}
	mockLLM := &mockLLMCaller{responses: responses}
	mockTools := newMockToolExecutor()
	// Return a long result that will be truncated
	mockTools.results["search"] = tools.ToolResult{Content: "line1\nline2\nline3\nline4\nline5", IsError: false}
	cm := newMockContextManager()
	tc := NewToolResultCache(5 * time.Minute)
	exec := newExecutorDefaultHITL(mockLLM, mockTools, &mockTokenCounter{}, 5, nil, false, tb, defaultCircuitBreakerConfig)
	exec.SetPerToolTruncation(ptt)
	exec.SetToolCache(tc)
	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "search", Description: "search", Source: "core"},
	}, cm)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.Finished {
		t.Error("expected Finished=true")
	}
	// Verify the result output or step observation contains the truncation nudge
}

// ============================================================================
// log() with non-nil logger
// ============================================================================

func TestExecutor_Log_WithLogger(t *testing.T) {
	mockLLM := &mockLLMCaller{}
	mockTools := newMockToolExecutor()
	exec := newExecutorDefaultHITL(mockLLM, mockTools, &mockTokenCounter{}, 5, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
	// SetLogger sets e.logger to non-nil, so log() returns it (not discard)
	logger := slog.New(slog.NewTextHandler(&nopWriter{}, nil))
	exec.SetLogger(logger)
	// log() should return the set logger, not the discard one.
	// We verify that log() doesn't return a discard logger by checking
	// that the returned logger is the same one we set.
	got := exec.log()
	if got != logger {
		t.Error("log() should return the SetLogger value, not discard")
	}
}

type nopWriter struct{}

func (w *nopWriter) Write(p []byte) (int, error) { return len(p), nil }

// ============================================================================
// Fruitless detector: AllowAlways disables threshold (line 893 sets to 0)
// Verify that AllowAlways properly prevents subsequent abort
// ============================================================================

func TestExecutor_Run_FruitlessDetector_AllowAlways_NotReAbort(t *testing.T) {
	cfg := CircuitBreakerConfig{
		RepeatNudgeThreshold:         50,
		RepeatAbortThreshold:         100,
		TruncationAbortThreshold:     100,
		ParseErrorAbortThreshold:     100,
		FruitlessNudgeThreshold:      5,
		FruitlessAbortThreshold:      8,
		FruitlessMaxResultLen:        32,
		SameToolRepeatNudgeThreshold: 50,
		SameToolRepeatAbortThreshold: 100,
		SameToolResultSizeDelta:      64,
	}
	// After AllowAlways disables fruitless abort, subsequent fruitless results
	// should NOT abort. Provide 12 fruitless responses + finish.
	responses := make([]*llm.ChatResponse, 13)
	for i := 0; i < 12; i++ {
		responses[i] = llmResponseWithToolCall(
			fmt.Sprintf("fruitless %d", i+1),
			"search",
			json.RawMessage(fmt.Sprintf(`{"q":"test%d"}`, i+1)),
		)
	}
	responses[12] = llmResponseFinish("done", "completed")
	mockLLM := &mockLLMCaller{responses: responses}
	mockTools := newMockToolExecutor()
	mockTools.results["search"] = tools.ToolResult{Content: "short", IsError: false}
	cm := newMockContextManager()
	exec := newExecutorDefaultHITL(mockLLM, mockTools, &mockTokenCounter{}, 20, nil, false, ToolResultBudget{}, cfg)
	exec.SetHITLHandler(&testStepLimitAdapter{fn: func(ctx context.Context, currentStep int, maxSteps int, reason string) (StepLimitResponse, error) {
		return StepLimitAllowAlways, nil
	}})
	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "search", Description: "search", Source: "core"},
	}, cm)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.Finished {
		t.Error("expected Finished=true after AllowAlways — should not re-abort on subsequent fruitless results")
	}
}

// ============================================================================
// getTruncationHint: cover all tool-specific branches
// ============================================================================

func TestGetTruncationHint_AllTools(t *testing.T) {
	tests := []struct {
		toolName string
		want     string
	}{
		{tools.ToolReadFile, "Re-read the file with start_line/end_line to see specific sections, or use ripgrep to search for specific content."},
		{tools.ToolRipgrep, "Narrow your search pattern or add path filters to reduce results."},
		{tools.ToolGrep, "Narrow your search pattern or add path filters to reduce results."},
		{tools.ToolGlob, "Use a more specific glob pattern to reduce results."},
		{tools.ToolWebFetch, "The page content was truncated. Ask the user to open the URL directly, or try fetching a more specific page."},
		{"unknown_tool", "Break into smaller operations or use targeted queries."},
	}
	for _, tt := range tests {
		t.Run(tt.toolName, func(t *testing.T) {
			got := getTruncationHint(tt.toolName)
			if got != tt.want {
				t.Errorf("getTruncationHint(%q) = %q, want %q", tt.toolName, got, tt.want)
			}
		})
	}
}

// ============================================================================
// applyToolResultBudget: HardCapTokens = 0 path
// ============================================================================

func TestApplyToolResultBudget_NoHardCap(t *testing.T) {
	mockLLM := &mockLLMCaller{}
	mockTools := newMockToolExecutor()
	// HardCapTokens=0 → early return, no truncation
	exec := newExecutorDefaultHITL(mockLLM, mockTools, &mockTokenCounter{}, 5, nil, false, ToolResultBudget{HardCapTokens: 0, MaxFillFraction: 0.5}, defaultCircuitBreakerConfig)
	cm := newMockContextManager()
	result := exec.applyToolResultBudget("some observation", cm, "search", "")
	if result != "some observation" {
		t.Errorf("expected unchanged observation, got %q", result)
	}
}

// ============================================================================
// CallLLM with nil response (defensive check)
// ============================================================================

func TestCallLLMWithReactiveCompaction_NilResponse(t *testing.T) {
	mockLLM := &mockLLMCaller{
		responses: []*llm.ChatResponse{nil}, // nil response triggers defensive check
	}
	mockTools := newMockToolExecutor()
	cm := newMockContextManager()
	exec := newExecutorDefaultHITL(mockLLM, mockTools, &mockTokenCounter{}, 5, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
	state := &runState{effectiveMaxSteps: 5, stepNum: 1}
	resp, _, err := exec.callLLMWithReactiveCompaction(context.Background(), state, cm, nil)
	if err == nil {
		t.Error("expected error for nil LLM response")
	}
	if resp != nil {
		t.Error("expected nil response")
	}
}

// ============================================================================
// handleStepLimitBoundary: error from OnStepLimit
// ============================================================================

func TestHandleStepLimitBoundary_CallbackError(t *testing.T) {
	mockLLM := &mockLLMCaller{}
	mockTools := newMockToolExecutor()
	cm := newMockContextManager()
	exec := newExecutorDefaultHITL(mockLLM, mockTools, &mockTokenCounter{}, 5, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
	exec.SetHITLHandler(&testStepLimitAdapter{fn: func(ctx context.Context, currentStep int, maxSteps int, reason string) (StepLimitResponse, error) {
		return StepLimitDeny, errors.New("callback error")
	}})
	// Run with only 1 step to hit the boundary quickly
	responses := []*llm.ChatResponse{
		llmResponseWithToolCall("step1", "search", json.RawMessage(`{"q":"x"}`)),
		llmResponseWithToolCall("step2", "search", json.RawMessage(`{"q":"y"}`)),
		llmResponseWithToolCall("step3", "search", json.RawMessage(`{"q":"z"}`)),
		llmResponseWithToolCall("step4", "search", json.RawMessage(`{"q":"w"}`)),
		llmResponseWithToolCall("step5", "search", json.RawMessage(`{"q":"v"}`)),
		llmResponseFinish("done", "finished"),
	}
	mockLLM.responses = responses
	exec.llm = mockLLM
	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "search", Description: "search", Source: "core"},
	}, cm)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Finished {
		t.Error("expected Finished=false when callback returns error")
	}
}

// ============================================================================
// Reactive compaction on context-exceeded error
// ============================================================================

func TestExecutor_Run_ReactiveCompaction(t *testing.T) {
	mockLLM := &mockLLMCaller{
		errors: []error{errors.New("context length exceeded")},
		responses: []*llm.ChatResponse{
			llmResponseFinish("recovered", "done after compaction"),
		},
	}
	mockTools := newMockToolExecutor()
	cm := newMockContextManager()
	exec := newExecutorDefaultHITL(mockLLM, mockTools, &mockTokenCounter{}, 5, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
	result, err := exec.Run(context.Background(), nil, cm)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.Finished {
		t.Error("expected Finished=true after reactive compaction + recovery")
	}
}

// ============================================================================
// DetectToolCallSyntaxInContent: failure-mode detector
// ============================================================================

func TestDetectToolCallSyntaxInContent(t *testing.T) {
	tests := []struct {
		name    string
		content string
		want    bool
	}{
		{"bash_exec fenced", "Let me check.\n```bash_exec\ncommand\n```\n", true},
		{"read_file fenced", "I'll read the file.\n```read_file\ncmd\n", true},
		{"edit_file fenced", "```edit_file\n", true},
		{"plain text no tools", "The answer is yes.", false},
		{"markdown code block no underscore", "```go\nfmt.Println()\n```", false},
		{"empty", "", false},
		{"indented fenced tool", "  ```bash_exec\n", true},
		{"tool with suffix", "```bash_exec (batched)\n", true},
		{"legit explanation mentioning tool name", "Use the `bash_exec` tool to run commands.", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := DetectToolCallSyntaxInContent(tt.content); got != tt.want {
				t.Errorf("DetectToolCallSyntaxInContent(%q) = %v, want %v", tt.content, got, tt.want)
			}
		})
	}
}

// ============================================================================
// handleImplicitFinish: tool-call syntax failure-mode path
// ============================================================================

func TestExecutor_Run_ToolCallSyntaxNudge_ThenAbort(t *testing.T) {
	// Model repeatedly prints tool-call syntax as text instead of using
	// tool_use blocks. After 3 special nudges, the executor should abort
	// with Finished=false.
	syntaxResp := &llm.ChatResponse{
		Message:    llm.Message{Role: "assistant", Content: "Let me check.\n```bash_exec\ncommand\n```"},
		StopReason: "end_turn",
		Usage:      llm.TokenUsage{InputTokens: 50, OutputTokens: 50},
	}
	mockLLM := &mockLLMCaller{
		responses: []*llm.ChatResponse{syntaxResp, syntaxResp, syntaxResp, syntaxResp},
	}
	cm := newMockContextManager()
	exec := newExecutorDefaultHITL(mockLLM, newMockToolExecutor(), &mockTokenCounter{}, 20, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)

	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "bash_exec", Description: "run bash", Source: "core"},
	}, cm)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Finished {
		t.Error("expected Finished=false (abort after 3 tool-call syntax nudges)")
	}
	if !strings.Contains(result.Output, "Aborted") {
		t.Errorf("expected abort message, got %q", result.Output)
	}
}

func TestExecutor_Run_ToolCallSyntaxNudge_ThenRecovery(t *testing.T) {
	// Model prints tool-call syntax once, then after a nudge recovers and
	// uses a real tool_use block (finish).
	mockLLM := &mockLLMCaller{
		responses: []*llm.ChatResponse{
			{
				Message:    llm.Message{Role: "assistant", Content: "Let me check.\n```bash_exec\necho hi"},
				StopReason: "end_turn",
				Usage:      llm.TokenUsage{InputTokens: 50, OutputTokens: 50},
			},
			llmResponseFinish("done", "completed"),
		},
	}
	mockTools := newMockToolExecutor()
	cm := newMockContextManager()
	exec := newExecutorDefaultHITL(mockLLM, mockTools, &mockTokenCounter{}, 20, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)

	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "finish", Description: "finish", Source: "core"},
	}, cm)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.Finished {
		t.Error("expected Finished=true after recovery from tool-call syntax nudge")
	}
}

// ============================================================================
// File-backed cache entry tests
// ============================================================================

func TestBuildCacheMeta_FileBacked_ReadFile(t *testing.T) {
	tmpDir := t.TempDir()
	testFile := tmpDir + "/test.go"
	if err := os.WriteFile(testFile, []byte("content"), 0o644); err != nil {
		t.Fatal(err)
	}
	mockTools := newMockToolExecutor()
	exec := newExecutorDefaultHITL(&mockLLMCaller{}, mockTools, &mockTokenCounter{}, 5, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)

	input, _ := json.Marshal(map[string]string{"path": testFile})
	meta := exec.buildCacheMeta(context.Background(), tools.ToolReadFile, input)

	if !meta.FileBacked {
		t.Error("expected FileBacked = true for read_file")
	}
	if meta.FilePath != testFile {
		t.Errorf("FilePath = %q, want %q", meta.FilePath, testFile)
	}
}

func TestBuildCacheMeta_FileBacked_WriteFile(t *testing.T) {
	tmpDir := t.TempDir()
	testFile := tmpDir + "/test.go"
	if err := os.WriteFile(testFile, []byte("content"), 0o644); err != nil {
		t.Fatal(err)
	}
	mockTools := newMockToolExecutor()
	exec := newExecutorDefaultHITL(&mockLLMCaller{}, mockTools, &mockTokenCounter{}, 5, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)

	input, _ := json.Marshal(map[string]string{"path": testFile})
	meta := exec.buildCacheMeta(context.Background(), tools.ToolWriteFile, input)

	if meta.FileBacked {
		t.Error("expected FileBacked = false for write_file")
	}
}

func TestBuildCacheMeta_FileBacked_EditFile(t *testing.T) {
	tmpDir := t.TempDir()
	testFile := tmpDir + "/test.go"
	if err := os.WriteFile(testFile, []byte("content"), 0o644); err != nil {
		t.Fatal(err)
	}
	mockTools := newMockToolExecutor()
	exec := newExecutorDefaultHITL(&mockLLMCaller{}, mockTools, &mockTokenCounter{}, 5, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)

	input, _ := json.Marshal(map[string]string{"path": testFile})
	meta := exec.buildCacheMeta(context.Background(), tools.ToolEditFile, input)

	if meta.FileBacked {
		t.Error("expected FileBacked = false for edit_file")
	}
}

func TestProcessToolResult_FileBackedNudge(t *testing.T) {
	// Verify that file-backed entries (read_file) get a nudge even without
	// Stage 1 truncation. tool_result_read serves token economy, not just
	// truncation recovery.
	tmpDir := t.TempDir()
	testFile := tmpDir + "/test.go"
	if err := os.WriteFile(testFile, []byte("line1\nline2\nline3"), 0o644); err != nil {
		t.Fatal(err)
	}

	mockTools := newMockToolExecutor()
	exec := newExecutorDefaultHITL(&mockLLMCaller{}, mockTools, &mockTokenCounter{}, 5, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
	tc := NewToolResultCache(5 * time.Minute)
	exec.SetToolCache(tc)

	cm := newMockContextManager()
	input, _ := json.Marshal(map[string]string{"path": testFile})

	observation, cacheHash := exec.processToolResult(context.Background(), "line1\nline2\nline3", "line1\nline2\nline3", tools.ToolReadFile, input, cm)

	if cacheHash == "" {
		t.Fatal("expected non-empty cache hash for file-backed read_file")
	}
	if !strings.Contains(observation, fileBackedNudgePrefix) {
		t.Errorf("expected file-backed nudge in observation, got: %s", observation)
	}
	if !strings.Contains(observation, cacheHash) {
		t.Errorf("expected cache hash %s in nudge, got: %s", cacheHash, observation)
	}
	if !strings.Contains(observation, "tool_result_read") {
		t.Errorf("expected tool_result_read instruction in nudge")
	}
}

func TestProcessToolResult_FileBackedNudge_NoStage1Truncation(t *testing.T) {
	// With no per-tool truncation config, Stage 1 doesn't fire.
	// The file-backed nudge should still be appended.
	tmpDir := t.TempDir()
	testFile := tmpDir + "/test.go"
	if err := os.WriteFile(testFile, []byte("content"), 0o644); err != nil {
		t.Fatal(err)
	}

	mockTools := newMockToolExecutor()
	// No per-tool truncation config — Stage 1 won't fire
	exec := newExecutorDefaultHITL(&mockLLMCaller{}, mockTools, &mockTokenCounter{}, 5, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
	tc := NewToolResultCache(5 * time.Minute)
	exec.SetToolCache(tc)

	cm := newMockContextManager()
	input, _ := json.Marshal(map[string]string{"path": testFile})

	observation, _ := exec.processToolResult(context.Background(), "content", "content", tools.ToolReadFile, input, cm)

	if !strings.Contains(observation, fileBackedNudgePrefix) {
		t.Errorf("expected file-backed nudge even without Stage 1 truncation, got: %s", observation)
	}
}

// contentBackedToolExecutor wraps mockToolExecutor but reports
// CacheModeContentBacked for read_file, simulating a read wrapper that returns
// a transformed view of the file (e.g. a converted document).
type contentBackedToolExecutor struct {
	mockToolExecutor
}

func (m *contentBackedToolExecutor) CacheStrategy(_ context.Context, name string, _ json.RawMessage) tools.CacheMode {
	if name == tools.ToolReadFile {
		return tools.CacheModeContentBacked
	}
	return tools.CacheModeDefault
}

func TestBuildCacheMeta_ContentBacked_ReadFile(t *testing.T) {
	// A read_file tool that opts into content-backed caching (via
	// ContentBackedReader) must NOT set FileBacked, yet still attach file
	// coherence metadata so the executor can detect source-file changes.
	tmpDir := t.TempDir()
	testFile := tmpDir + "/doc.pdf"
	if err := os.WriteFile(testFile, []byte("%PDF-1.4 fake"), 0o644); err != nil {
		t.Fatal(err)
	}

	mockTools := &contentBackedToolExecutor{mockToolExecutor: *newMockToolExecutor()}
	exec := newExecutorDefaultHITL(&mockLLMCaller{}, mockTools, &mockTokenCounter{}, 5, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)

	input, _ := json.Marshal(map[string]string{"path": testFile})
	meta := exec.buildCacheMeta(context.Background(), tools.ToolReadFile, input)

	if meta.FileBacked {
		t.Error("expected FileBacked = false for content-backed read_file")
	}
	// File coherence metadata must still be present.
	if meta.FilePath != testFile {
		t.Errorf("FilePath = %q, want %q", meta.FilePath, testFile)
	}
	if meta.FileSize == 0 {
		t.Error("expected non-zero FileSize for content-backed read_file")
	}
	if meta.FileMtime == 0 {
		t.Error("expected non-zero FileMtime for content-backed read_file")
	}
}

// ============================================================================
// cache-on-truncate for non-cacheable tools (Stage 1 + Stage 2)
// ============================================================================
//
// Regression coverage for the irreversible-truncation bug: processToolResult
// used to skip caching entirely for tools in nonCacheableTools, so a truncated
// result got an "[OUTPUT TRUNCATED]" notice with NO hash — the dropped content
// was permanently lost. Now ANY truncation caches the full result on demand and
// yields a hash so the LLM can recover it via tool_result_read.

func TestProcessToolResult_NonCacheable_Stage2Truncate_HashAndRetrievable(t *testing.T) {
	// A non-cacheable tool whose output exceeds the token budget must be
	// truncated at Stage 2 with a VALID non-empty hash, and tool_result_read
	// (i.e. cache.Get) must return the full original content.
	mockTools := newMockToolExecutor()
	exec := newExecutorDefaultHITL(&mockLLMCaller{}, mockTools, &mockTokenCounter{}, 5, nil, false, ToolResultBudget{
		HardCapTokens:   100,
		MaxFillFraction: 0.5,
	}, defaultCircuitBreakerConfig)
	tc := NewToolResultCache(5 * time.Minute)
	exec.SetToolCache(tc)

	cm := newMockContextManager()
	cm.availableTokens = 200 // adaptive cap = 100 → floor 256 dominates; make content clearly larger

	// "search_facts" is in defaultNonCacheableTools. Use a large result that
	// exceeds the 256-token floor cap (256 tokens ≈ 1024 chars).
	fullContent := strings.Repeat("fact-", 400) // ~2000 chars → 500 tokens, well over cap
	input, _ := json.Marshal(map[string]string{"query": "test"})

	observation, cacheHash := exec.processToolResult(context.Background(), fullContent, fullContent, "search_facts", input, cm)

	if cacheHash == "" {
		t.Fatal("expected non-empty cache hash for truncated non-cacheable tool")
	}
	if !strings.Contains(observation, "OUTPUT TRUNCATED") {
		t.Errorf("expected Stage 2 truncation notice, got: %s", observation)
	}
	// The Stage 2 notice must carry the hash + tool_result_read instruction.
	if !strings.Contains(observation, "Hash: "+cacheHash) {
		t.Errorf("expected truncation notice to reference hash %q, got: %s", cacheHash, observation)
	}
	if !strings.Contains(observation, "tool_result_read") {
		t.Error("expected tool_result_read instruction in truncation notice")
	}

	// The full original content must be retrievable from the cache via the hash.
	entry, ok := tc.Get(cacheHash)
	if !ok {
		t.Fatalf("cache.Get(%q) returned not-found; full result is irretrievable", cacheHash)
	}
	if entry.Content != fullContent {
		t.Errorf("cached content does not match original full result (got %d bytes, want %d)", len(entry.Content), len(fullContent))
	}
	if entry.ToolName != "search_facts" {
		t.Errorf("cached ToolName = %q, want %q", entry.ToolName, "search_facts")
	}
}

func TestProcessToolResult_NonCacheable_Stage1Truncate_FragmentationNudge(t *testing.T) {
	// A non-cacheable tool configured with per-tool truncation (Stage 1) must
	// be truncated at Stage 1 and carry a fragmentation nudge with a valid hash.
	mockTools := newMockToolExecutor()
	exec := newExecutorDefaultHITL(&mockLLMCaller{}, mockTools, &mockTokenCounter{}, 5, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
	exec.SetPerToolTruncation(map[string]ToolTruncationConfig{
		"search_facts": {MaxLines: 2},
	})
	tc := NewToolResultCache(5 * time.Minute)
	exec.SetToolCache(tc)

	cm := newMockContextManager()

	fullContent := "line1\nline2\nline3\nline4\nline5"
	input, _ := json.Marshal(map[string]string{"query": "test"})

	observation, cacheHash := exec.processToolResult(context.Background(), fullContent, fullContent, "search_facts", input, cm)

	if cacheHash == "" {
		t.Fatal("expected non-empty cache hash for Stage 1 truncated non-cacheable tool")
	}
	// Stage 1 fragmentation nudge prefix (not the Stage 2 "[OUTPUT TRUNCATED" notice).
	if !strings.Contains(observation, "This output was truncated to 2 lines for 'search_facts'") {
		t.Errorf("expected Stage 1 fragmentation nudge, got: %s", observation)
	}
	if !strings.Contains(observation, "hash: "+cacheHash) {
		t.Errorf("expected nudge to reference hash %q, got: %s", cacheHash, observation)
	}
	if !strings.Contains(observation, "tool_result_read") {
		t.Error("expected tool_result_read instruction in fragmentation nudge")
	}

	// Full original content retrievable.
	entry, ok := tc.Get(cacheHash)
	if !ok {
		t.Fatalf("cache.Get(%q) returned not-found after Stage 1 truncation", cacheHash)
	}
	if entry.Content != fullContent {
		t.Errorf("cached content does not match original full result (got %q, want %q)", entry.Content, fullContent)
	}
}

func TestProcessToolResult_NonCacheable_RoundTrip_FullContentViaToolResultRead(t *testing.T) {
	// Round-trip reversibility for a non-cacheable tool. processToolResult caches
	// the full result on truncate and yields a hash; re-reading the cache entry
	// through the SAME windowed-line extraction that tool_result_read applies
	// must reconstruct the COMPLETE original content, proving the dropped portion
	// is fully recoverable and never permanently lost.
	//
	// This in-package test replicates tool_result_read's content-backed read
	// algorithm (cache.Get + line-window slicing) because package agent cannot
	// import the builtins package implementing ToolResultReadTool without an
	// import cycle. The Stage 1/Stage 2 tests above prove processToolResult
	// stores the entry with the full content; this test proves that entry is
	// reassemblable into 100% of the original via the tool's own read path.
	mockTools := newMockToolExecutor()
	exec := newExecutorDefaultHITL(&mockLLMCaller{}, mockTools, &mockTokenCounter{}, 5, nil, false, ToolResultBudget{
		HardCapTokens:   100,
		MaxFillFraction: 0.5,
	}, defaultCircuitBreakerConfig)
	tc := NewToolResultCache(5 * time.Minute)
	exec.SetToolCache(tc)

	cm := newMockContextManager()
	cm.availableTokens = 200 // adaptive cap = 100 → 256-token floor dominates

	// "search_facts" is in defaultNonCacheableTools. Build a multi-line payload
	// large enough to force Stage 2 truncation (256 tokens ≈ 1024 chars).
	const numLines = 60
	lines := make([]string, numLines)
	for i := range lines {
		lines[i] = fmt.Sprintf("fact line %02d: %s", i, strings.Repeat("x", 30))
	}
	fullContent := strings.Join(lines, "\n")
	input, _ := json.Marshal(map[string]string{"query": "round-trip"})

	observation, cacheHash := exec.processToolResult(context.Background(), fullContent, fullContent, "search_facts", input, cm)

	// 1. The Stage 2 truncation notice must carry a usable hash the LLM can act on.
	if cacheHash == "" {
		t.Fatal("expected non-empty cache hash for truncated non-cacheable tool")
	}
	if !strings.Contains(observation, "OUTPUT TRUNCATED") {
		t.Fatalf("expected Stage 2 truncation notice, got: %s", observation)
	}
	if !strings.Contains(observation, "Hash: "+cacheHash) {
		t.Fatalf("truncation notice does not expose hash %q; observation: %s", cacheHash, observation)
	}

	// 2. Replicate tool_result_read's content-backed read: fetch the entry, then
	//    reassemble the full content by reading consecutive line windows — exactly
	//    how the LLM recovers the dropped portion fragment by fragment.
	entry, ok := tc.Get(cacheHash)
	if !ok {
		t.Fatalf("cache.Get(%q) returned not-found; round-trip impossible", cacheHash)
	}
	allLines := strings.Split(entry.Content, "\n")
	totalLines := len(allLines)
	if totalLines != numLines {
		t.Fatalf("line count drifted: cache has %d lines, original has %d", totalLines, numLines)
	}

	const windowSize = 10
	var reassembled strings.Builder
	startLine := 1
	for startLine <= totalLines {
		endLine := startLine + windowSize - 1
		if endLine > totalLines {
			endLine = totalLines
		}
		// tool_result_read slices allLines[startLine-1 : endLine] and joins on "\n".
		reassembled.WriteString(strings.Join(allLines[startLine-1:endLine], "\n"))
		if endLine < totalLines {
			reassembled.WriteByte('\n')
		}
		startLine = endLine + 1
	}

	if reassembled.String() != fullContent {
		t.Errorf("round-trip mismatch: reassembled content != original full result\nwant %d bytes\ngot  %d bytes",
			len(fullContent), reassembled.Len())
	}
}

func TestProcessToolResult_NonCacheable_SmallResult_NotCached(t *testing.T) {
	// A small non-truncated result from a non-cacheable tool must NOT enter the
	// cache (optimization preserved): cache.Len() unchanged after processing.
	mockTools := newMockToolExecutor()
	exec := newExecutorDefaultHITL(&mockLLMCaller{}, mockTools, &mockTokenCounter{}, 5, nil, false, ToolResultBudget{
		HardCapTokens:   1000,
		MaxFillFraction: 0.5,
	}, defaultCircuitBreakerConfig)
	tc := NewToolResultCache(5 * time.Minute)
	exec.SetToolCache(tc)

	cm := newMockContextManager()
	cm.availableTokens = 100000

	beforeLen := tc.Len()
	input, _ := json.Marshal(map[string]string{"item": "checklist-ok"})

	// "update_checklist" is in defaultNonCacheableTools and produces a tiny result.
	observation, cacheHash := exec.processToolResult(context.Background(), "ok", "ok", "update_checklist", input, cm)

	if cacheHash != "" {
		t.Errorf("expected empty hash for small non-truncated non-cacheable result, got %q", cacheHash)
	}
	afterLen := tc.Len()
	if afterLen != beforeLen {
		t.Errorf("cache size changed for small non-cacheable result: before=%d after=%d", beforeLen, afterLen)
	}
	if observation != "ok" {
		t.Errorf("expected unchanged observation, got %q", observation)
	}
}

func TestWillBudgetTruncate(t *testing.T) {
	// Direct coverage of the willBudgetTruncate predictor: it must agree with
	// applyToolResultBudget's truncation decision across disabled/under/over
	// cases, so processToolResult's cache-on-truncate prediction never drifts.
	exec := newExecutorDefaultHITL(&mockLLMCaller{}, newMockToolExecutor(), &mockTokenCounter{}, 5, nil, false, ToolResultBudget{
		HardCapTokens:   100,
		MaxFillFraction: 0.5,
	}, defaultCircuitBreakerConfig)
	cm := newMockContextManager()
	cm.availableTokens = 200 // adaptive cap = 100, floor dominates → 256

	// Disabled budget → never truncates.
	execDisabled := newExecutorDefaultHITL(&mockLLMCaller{}, newMockToolExecutor(), &mockTokenCounter{}, 5, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
	if execDisabled.willBudgetTruncate("any", cm) {
		t.Error("expected willBudgetTruncate=false when budget disabled")
	}

	// Under budget (small content) → no truncation.
	if exec.willBudgetTruncate("short", cm) {
		t.Error("expected willBudgetTruncate=false for short content")
	}

	// Over budget (large content > 256 tokens ≈ 1024 chars) → truncation.
	longContent := strings.Repeat("x", 2000) // 500 tokens
	if !exec.willBudgetTruncate(longContent, cm) {
		t.Error("expected willBudgetTruncate=true for over-budget content")
	}
	// Prediction must match the actual truncation decision.
	actual := exec.applyToolResultBudget(longContent, cm, "some_tool", "")
	if !strings.Contains(actual, "OUTPUT TRUNCATED") {
		t.Error("applyToolResultBudget disagrees with willBudgetTruncate: expected truncation")
	}
}
