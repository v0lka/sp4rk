package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"strconv"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/v0lka/sp4rk/llm"
	"github.com/v0lka/sp4rk/tools"
)

// --- Setter method tests ---

func TestSetLogger(t *testing.T) {
	exec := newExecutorDefaultHITL(&mockLLMCaller{}, newMockToolExecutor(), &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
	if exec.logger != nil {
		t.Error("logger should be nil before SetLogger")
	}
	exec.SetLogger(slog.New(slog.DiscardHandler))
	if exec.logger == nil {
		t.Error("logger should not be nil after SetLogger")
	}
}

func TestSetReasoningEffort(t *testing.T) {
	exec := newExecutorDefaultHITL(&mockLLMCaller{}, newMockToolExecutor(), &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
	if exec.reasoningEffort != "" {
		t.Error("reasoningEffort should be empty initially")
	}
	exec.SetReasoningEffort("high")
	if exec.reasoningEffort != "high" {
		t.Errorf("reasoningEffort = %q, want %q", exec.reasoningEffort, "high")
	}
}

func TestSetToolCache(t *testing.T) {
	exec := newExecutorDefaultHITL(&mockLLMCaller{}, newMockToolExecutor(), &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
	if exec.toolCache != nil {
		t.Error("toolCache should be nil initially")
	}
	cache := NewToolResultCache(0)
	exec.SetToolCache(cache)
	if exec.toolCache == nil {
		t.Error("toolCache should not be nil after SetToolCache")
	}
}

func TestAddNonCacheableTools(t *testing.T) {
	exec := newExecutorDefaultHITL(&mockLLMCaller{}, newMockToolExecutor(), &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)

	// NewExecutor initialises the set from defaultNonCacheableTools.
	if exec.nonCacheableTools == nil {
		t.Fatal("nonCacheableTools should be initialised from defaults in NewExecutor")
	}
	for _, name := range []string{"finish", "store_fact", tools.ToolBatch} {
		if _, ok := exec.nonCacheableTools[name]; !ok {
			t.Errorf("default nonCacheableTools missing %q", name)
		}
	}

	// AddNonCacheableTools extends the set without removing defaults.
	exec.AddNonCacheableTools("delegate", "reflect")
	for _, name := range []string{"delegate", "reflect", "finish"} {
		if _, ok := exec.nonCacheableTools[name]; !ok {
			t.Errorf("nonCacheableTools missing %q after AddNonCacheableTools", name)
		}
	}

	// Adding an already-present tool is a no-op (no panic, still present).
	exec.AddNonCacheableTools("finish")
	if _, ok := exec.nonCacheableTools["finish"]; !ok {
		t.Error("finish should still be present after re-adding")
	}
}

func TestAddNonCacheableTools_DoesNotMutateDefault(t *testing.T) {
	exec := newExecutorDefaultHITL(&mockLLMCaller{}, newMockToolExecutor(), &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
	exec.AddNonCacheableTools("my_custom_tool")

	// The package-level default must not be polluted.
	if _, ok := defaultNonCacheableTools["my_custom_tool"]; ok {
		t.Error("AddNonCacheableTools must not mutate the package-level defaultNonCacheableTools")
	}
	// The executor's own set should contain the custom tool.
	if _, ok := exec.nonCacheableTools["my_custom_tool"]; !ok {
		t.Error("executor's nonCacheableTools should contain the custom tool")
	}
}

func TestSetPreWarningPercent(t *testing.T) {
	exec := newExecutorDefaultHITL(&mockLLMCaller{}, newMockToolExecutor(), &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
	if exec.preWarningPercent != 0 {
		t.Error("preWarningPercent should be 0 initially")
	}
	exec.SetPreWarningPercent(75)
	if exec.preWarningPercent != 75 {
		t.Errorf("preWarningPercent = %d, want 75", exec.preWarningPercent)
	}
}

func TestSetPerToolTruncation(t *testing.T) {
	exec := newExecutorDefaultHITL(&mockLLMCaller{}, newMockToolExecutor(), &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
	if exec.perToolTruncation != nil {
		t.Error("perToolTruncation should be nil initially")
	}
	cfg := map[string]ToolTruncationConfig{
		"read_file": {MaxLines: 100},
	}
	exec.SetPerToolTruncation(cfg)
	if exec.perToolTruncation == nil {
		t.Error("perToolTruncation should not be nil after SetPerToolTruncation")
	}
	if exec.perToolTruncation["read_file"].MaxLines != 100 {
		t.Errorf("MaxLines = %d, want 100", exec.perToolTruncation["read_file"].MaxLines)
	}
}

// --- checkFruitlessResult edge cases ---

func TestCheckFruitlessResult_ThresholdZero_Disabled(t *testing.T) {
	cfg := CircuitBreakerConfig{
		RepeatNudgeThreshold:         3,
		RepeatAbortThreshold:         4,
		TruncationAbortThreshold:     3,
		ParseErrorAbortThreshold:     3,
		FruitlessNudgeThreshold:      0,
		FruitlessAbortThreshold:      0,
		FruitlessMaxResultLen:        32,
		SameToolRepeatNudgeThreshold: 50,
		SameToolRepeatAbortThreshold: 60,
		SameToolResultSizeDelta:      64,
	}
	exec := newExecutorDefaultHITL(&mockLLMCaller{}, newMockToolExecutor(), &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, cfg)
	exec.consecutiveFruitlessCount = 100
	act, result, err := exec.checkFruitlessResult(
		context.Background(),
		llm.ToolCall{Name: "search"},
		0, "", false,
		&runState{effectiveMaxSteps: 10},
		newMockContextManager(),
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != nil {
		t.Error("expected nil result when fruitless thresholds are disabled")
	}
	if act != actionNone {
		t.Errorf("expected actionNone, got %v", act)
	}
}

func TestCheckFruitlessResult_ErrorDoesNotCount(t *testing.T) {
	cfg := CircuitBreakerConfig{
		RepeatNudgeThreshold:         3,
		RepeatAbortThreshold:         4,
		TruncationAbortThreshold:     3,
		ParseErrorAbortThreshold:     3,
		FruitlessNudgeThreshold:      3,
		FruitlessAbortThreshold:      6,
		FruitlessMaxResultLen:        32,
		SameToolRepeatNudgeThreshold: 50,
		SameToolRepeatAbortThreshold: 60,
		SameToolResultSizeDelta:      64,
	}
	exec := newExecutorDefaultHITL(&mockLLMCaller{}, newMockToolExecutor(), &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, cfg)
	for i := 0; i < 5; i++ {
		_, _, _ = exec.checkFruitlessResult(
			context.Background(),
			llm.ToolCall{Name: "search"},
			0, "error", true,
			&runState{effectiveMaxSteps: 10},
			newMockContextManager(),
		)
	}
	if exec.consecutiveFruitlessCount != 0 {
		t.Errorf("fruitless count should be 0 for error results, got %d", exec.consecutiveFruitlessCount)
	}
}

func TestCheckFruitlessResult_LargeResultResets(t *testing.T) {
	cfg := CircuitBreakerConfig{
		RepeatNudgeThreshold:         3,
		RepeatAbortThreshold:         4,
		TruncationAbortThreshold:     3,
		ParseErrorAbortThreshold:     3,
		FruitlessNudgeThreshold:      3,
		FruitlessAbortThreshold:      6,
		FruitlessMaxResultLen:        32,
		SameToolRepeatNudgeThreshold: 50,
		SameToolRepeatAbortThreshold: 60,
		SameToolResultSizeDelta:      64,
	}
	exec := newExecutorDefaultHITL(&mockLLMCaller{}, newMockToolExecutor(), &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, cfg)
	_, _, _ = exec.checkFruitlessResult(
		context.Background(),
		llm.ToolCall{Name: "search"},
		0, "small", false,
		&runState{effectiveMaxSteps: 10},
		newMockContextManager(),
	)
	if exec.consecutiveFruitlessCount != 1 {
		t.Errorf("fruitless count should be 1, got %d", exec.consecutiveFruitlessCount)
	}
	_, _, _ = exec.checkFruitlessResult(
		context.Background(),
		llm.ToolCall{Name: "search"},
		0, strings.Repeat("x", 100), false,
		&runState{effectiveMaxSteps: 10},
		newMockContextManager(),
	)
	if exec.consecutiveFruitlessCount != 0 {
		t.Errorf("fruitless count should be 0 after large result, got %d", exec.consecutiveFruitlessCount)
	}
}

// --- checkFruitlessResult: exempt tools (mutating tools, meta-tools) ---

func TestCheckFruitlessResult_ExemptToolsNotCounted(t *testing.T) {
	cfg := CircuitBreakerConfig{
		RepeatNudgeThreshold:         3,
		RepeatAbortThreshold:         5,
		TruncationAbortThreshold:     3,
		ParseErrorAbortThreshold:     3,
		FruitlessNudgeThreshold:      3,
		FruitlessAbortThreshold:      4,
		FruitlessMaxResultLen:        48,
		SameToolRepeatNudgeThreshold: 50,
		SameToolRepeatAbortThreshold: 60,
		SameToolResultSizeDelta:      64,
	}
	// Each exempt tool produces a short successful result (e.g. edit_file
	// returns "successfully edited file" = 24 bytes; a wait tool's "still
	// pending"). None should increment the fruitless counter, even when called
	// many times in a row.
	exemptTools := []string{
		"edit_file", "write_file", "create_directory",
		"delete_file", "delete_directory",
		"update_checklist", "set_step_status", "store_fact",
		"wait", "wait_for_one", "wait_for_all",
	}
	for _, toolName := range exemptTools {
		t.Run(toolName, func(t *testing.T) {
			exec := newExecutorDefaultHITL(&mockLLMCaller{}, newMockToolExecutor(), &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, cfg)
			for i := 0; i < 10; i++ {
				act, result, err := exec.checkFruitlessResult(
					context.Background(),
					llm.ToolCall{Name: toolName},
					0, "ok", false,
					&runState{effectiveMaxSteps: 10},
					newMockContextManager(),
				)
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				if result != nil {
					t.Errorf("expected nil result for exempt tool %s at iteration %d, got abort", toolName, i)
				}
				if act != actionNone {
					t.Errorf("expected actionNone for exempt tool %s at iteration %d, got %v", toolName, i, act)
				}
			}
			if exec.consecutiveFruitlessCount != 0 {
				t.Errorf("fruitless count should be 0 for exempt tool %s after 10 short results, got %d", toolName, exec.consecutiveFruitlessCount)
			}
		})
	}
}

// --- checkSameToolRepetition edge cases ---

func TestCheckSameToolRepetition_StoreFactResets(t *testing.T) {
	cfg := CircuitBreakerConfig{
		RepeatNudgeThreshold:         3,
		RepeatAbortThreshold:         4,
		TruncationAbortThreshold:     3,
		ParseErrorAbortThreshold:     3,
		FruitlessNudgeThreshold:      50,
		FruitlessAbortThreshold:      60,
		FruitlessMaxResultLen:        32,
		SameToolRepeatNudgeThreshold: 3,
		SameToolRepeatAbortThreshold: 6,
		SameToolResultSizeDelta:      64,
	}
	exec := newExecutorDefaultHITL(&mockLLMCaller{}, newMockToolExecutor(), &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, cfg)
	for i := 0; i < 4; i++ {
		_, _, _ = exec.checkSameToolRepetition(
			context.Background(),
			llm.ToolCall{Name: "search"},
			0, strings.Repeat("x", 50),
			tools.ToolResult{Content: strings.Repeat("x", 50)},
			&runState{effectiveMaxSteps: 10},
			newMockContextManager(),
		)
	}
	_, _, _ = exec.checkSameToolRepetition(
		context.Background(),
		llm.ToolCall{Name: "store_fact"},
		0, "stored",
		tools.ToolResult{Content: "stored"},
		&runState{effectiveMaxSteps: 10},
		newMockContextManager(),
	)
	if exec.sameToolConsecutiveCount != 0 {
		t.Errorf("sameToolConsecutiveCount should be 0 after store_fact, got %d", exec.sameToolConsecutiveCount)
	}
}

func TestCheckSameToolRepetition_ExemptToolsNotCounted(t *testing.T) {
	cfg := CircuitBreakerConfig{
		RepeatNudgeThreshold:         3,
		RepeatAbortThreshold:         5,
		TruncationAbortThreshold:     3,
		ParseErrorAbortThreshold:     3,
		FruitlessNudgeThreshold:      50,
		FruitlessAbortThreshold:      60,
		FruitlessMaxResultLen:        48,
		SameToolRepeatNudgeThreshold: 3,
		SameToolRepeatAbortThreshold: 4,
		SameToolResultSizeDelta:      128,
	}
	// Mutating tools and meta-tools legitimately produce bursts of short,
	// similarly-sized successful results (e.g. batch edit_file calls each
	// returning "successfully edited file" = 24 bytes; a wait tool polled on
	// the same ids). They must not trigger the same-tool-repeat detector.
	exemptTools := []string{
		"edit_file", "write_file", "create_directory",
		"delete_file", "delete_directory",
		"update_checklist", "set_step_status", "store_fact",
		"wait", "wait_for_one", "wait_for_all",
	}
	for _, toolName := range exemptTools {
		t.Run(toolName, func(t *testing.T) {
			exec := newExecutorDefaultHITL(&mockLLMCaller{}, newMockToolExecutor(), &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, cfg)
			for i := 0; i < 10; i++ {
				act, result, err := exec.checkSameToolRepetition(
					context.Background(),
					llm.ToolCall{Name: toolName},
					0, "ok",
					tools.ToolResult{Content: "ok"},
					&runState{effectiveMaxSteps: 10},
					newMockContextManager(),
				)
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				if result != nil {
					t.Errorf("expected nil result for exempt tool %s at iteration %d, got abort", toolName, i)
				}
				if act != actionNone {
					t.Errorf("expected actionNone for exempt tool %s at iteration %d, got %v", toolName, i, act)
				}
			}
			if exec.sameToolConsecutiveCount != 0 {
				t.Errorf("sameToolConsecutiveCount should be 0 for exempt tool %s after 10 calls, got %d", toolName, exec.sameToolConsecutiveCount)
			}
		})
	}
}

func TestCheckSameToolRepetition_ThresholdZero_Disabled(t *testing.T) {
	cfg := CircuitBreakerConfig{
		RepeatNudgeThreshold:         3,
		RepeatAbortThreshold:         4,
		TruncationAbortThreshold:     3,
		ParseErrorAbortThreshold:     3,
		FruitlessNudgeThreshold:      50,
		FruitlessAbortThreshold:      60,
		FruitlessMaxResultLen:        32,
		SameToolRepeatNudgeThreshold: 0,
		SameToolRepeatAbortThreshold: 0,
		SameToolResultSizeDelta:      64,
	}
	exec := newExecutorDefaultHITL(&mockLLMCaller{}, newMockToolExecutor(), &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, cfg)
	exec.sameToolConsecutiveCount = 100
	exec.sameToolLastName = "search"
	exec.sameToolLastResultLen = 50
	act, result, err := exec.checkSameToolRepetition(
		context.Background(),
		llm.ToolCall{Name: "search"},
		0, strings.Repeat("x", 50),
		tools.ToolResult{Content: strings.Repeat("x", 50)},
		&runState{effectiveMaxSteps: 10},
		newMockContextManager(),
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != nil {
		t.Error("expected nil result when thresholds are disabled")
	}
	if act != actionNone {
		t.Errorf("expected actionNone, got %v", act)
	}
}

// TestCheckSameToolRepetition_RetryAfterFixDoesNotNudge covers the legitimate
// pattern "call failed → arguments fixed → retried": an error result must not
// count as "similar results" loop evidence, so the follow-up same-tool call
// with changed arguments neither triggers the nudge nor increments the chain.
func TestCheckSameToolRepetition_RetryAfterFixDoesNotNudge(t *testing.T) {
	cfg := CircuitBreakerConfig{
		RepeatNudgeThreshold:         3,
		RepeatAbortThreshold:         4,
		TruncationAbortThreshold:     3,
		ParseErrorAbortThreshold:     3,
		FruitlessNudgeThreshold:      50,
		FruitlessAbortThreshold:      60,
		FruitlessMaxResultLen:        32,
		SameToolRepeatNudgeThreshold: 2, // a plain increment here WOULD nudge
		SameToolRepeatAbortThreshold: 3,
		SameToolResultSizeDelta:      64,
	}
	exec := newExecutorDefaultHITL(&mockLLMCaller{}, newMockToolExecutor(), &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, cfg)

	// 1st call: the tool fails (bad argument).
	errAction := llm.ToolCall{Name: "search", Input: json.RawMessage(`{"query":"misspelled-queary"}`)}
	errResult := tools.ToolResult{Content: "error: invalid query 'misspelled-queary'", IsError: true}
	actionBreak, result, err := exec.checkSameToolRepetition(
		context.Background(),
		errAction,
		0, errResult.Content,
		errResult,
		&runState{effectiveMaxSteps: 10},
		newMockContextManager(),
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if actionBreak != actionNone {
		t.Fatalf("actionBreak = %v, want actionNone after first errored call", actionBreak)
	}
	if result != nil {
		t.Fatalf("first errored call must not produce a nudge result, got %+v", result)
	}
	if exec.sameToolConsecutiveCount != 1 {
		t.Fatalf("sameToolConsecutiveCount = %d after first call, want 1", exec.sameToolConsecutiveCount)
	}
	if !exec.sameToolLastResultIsError {
		t.Fatal("sameToolLastResultIsError must be recorded for the error result")
	}

	// 2nd call: same tool, FIXED arguments, success with a similar-sized result.
	fixedAction := llm.ToolCall{Name: "search", Input: json.RawMessage(`{"query":"corrected query"}`)}
	okResult := tools.ToolResult{Content: "ok: found 3 matches for 'corrected query'"}
	actionBreak, result, err = exec.checkSameToolRepetition(
		context.Background(),
		fixedAction,
		0, okResult.Content,
		okResult,
		&runState{effectiveMaxSteps: 10},
		newMockContextManager(),
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if actionBreak != actionNone {
		t.Fatalf("actionBreak = %v, want actionNone: retry-after-fix must not break the loop", actionBreak)
	}
	if result != nil {
		t.Fatalf("retry-after-fix must NOT trigger the same-tool nudge, got %+v", result)
	}
	if exec.sameToolConsecutiveCount != 1 {
		t.Fatalf("sameToolConsecutiveCount = %d after retry-after-fix, want 1 (reset, not incremented)", exec.sameToolConsecutiveCount)
	}
	if exec.sameToolLastResultIsError {
		t.Fatal("sameToolLastResultIsError must be cleared after the successful retry")
	}
}

// TestCheckSameToolRepetition_ConsecutiveErrorsStillAccumulate pins the
// error-loop half of the retry-after-fix rule: consecutive FAILING calls to
// the same tool with varied arguments and similar-sized results are loop
// evidence, not a retry-after-fix — they must accumulate toward the
// same-tool nudge/abort thresholds exactly like successful repetitions.
// Without the !result.IsError guard on the reset, every failing call reset
// the chain to 1 and the breaker could never fire for varied-args error
// loops (the only breaker covering that shape).
func TestCheckSameToolRepetition_ConsecutiveErrorsStillAccumulate(t *testing.T) {
	cfg := CircuitBreakerConfig{
		RepeatNudgeThreshold:         3,
		RepeatAbortThreshold:         4,
		TruncationAbortThreshold:     3,
		ParseErrorAbortThreshold:     3,
		FruitlessNudgeThreshold:      50,
		FruitlessAbortThreshold:      60,
		FruitlessMaxResultLen:        32,
		SameToolRepeatNudgeThreshold: 3,
		SameToolRepeatAbortThreshold: 4,
		SameToolResultSizeDelta:      64,
	}
	exec := newExecutorDefaultHITL(&mockLLMCaller{}, newMockToolExecutor(), &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, cfg)
	state := &runState{effectiveMaxSteps: 10}

	call := func(query string) (loopAction, *ExecutorResult, error) {
		return exec.checkSameToolRepetition(
			context.Background(),
			llm.ToolCall{Name: "search", Input: json.RawMessage(`{"query":"` + query + `"}`)},
			0, "error: boom",
			tools.ToolResult{Content: "error: boom", IsError: true},
			state,
			newMockContextManager(),
		)
	}

	// 1st failing call: recorded as the chain start (count=1).
	loopAct, result, err := call("attempt-one")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if loopAct != actionNone || result != nil {
		t.Fatalf("first call: loopAct=%v result=%v, want actionNone/nil", loopAct, result)
	}
	if exec.sameToolConsecutiveCount != 1 {
		t.Fatalf("count = %d after 1st failing call, want 1", exec.sameToolConsecutiveCount)
	}

	// 2nd failing call, varied arguments, identical result size: must
	// INCREMENT the chain (retry-after-fix does not apply — no success).
	loopAct, result, err = call("attempt-two-different-args")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if loopAct != actionNone || result != nil {
		t.Fatalf("second call: loopAct=%v result=%v, want actionNone/nil (below nudge threshold)", loopAct, result)
	}
	if exec.sameToolConsecutiveCount != 2 {
		t.Fatalf("count = %d after 2nd consecutive failing call, want 2 (incremented, not reset)", exec.sameToolConsecutiveCount)
	}
	if !exec.sameToolLastResultIsError {
		t.Fatal("sameToolLastResultIsError must stay recorded across consecutive errors")
	}

	// 3rd failing call reaches the nudge threshold → the breaker fires.
	loopAct, result, err = call("attempt-three-other-args")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if loopAct != actionBreak || result != nil || !state.circuitBreakerTriggered {
		t.Fatalf("3rd consecutive failing call must trigger the same-tool nudge: loopAct=%v result=%v triggered=%v", loopAct, result, state.circuitBreakerTriggered)
	}
	if exec.sameToolConsecutiveCount != 3 {
		t.Fatalf("count = %d at nudge, want 3", exec.sameToolConsecutiveCount)
	}
}

// TestCheckRepeatIdenticalTool_ErrorThenIdenticalCall_Nudges pins the behavior
// for the truly pathological pattern "X → error → X identical": the identical
// call is still intercepted by the repeat-error nudge (thresholds are lowered
// by one after an errored identical call), unchanged by the retry-after-fix fix.
func TestCheckRepeatIdenticalTool_ErrorThenIdenticalCall_Nudges(t *testing.T) {
	cfg := CircuitBreakerConfig{
		RepeatNudgeThreshold:         3,
		RepeatAbortThreshold:         4,
		TruncationAbortThreshold:     3,
		ParseErrorAbortThreshold:     3,
		FruitlessNudgeThreshold:      50,
		FruitlessAbortThreshold:      60,
		FruitlessMaxResultLen:        32,
		SameToolRepeatNudgeThreshold: 50,
		SameToolRepeatAbortThreshold: 60,
		SameToolResultSizeDelta:      64,
	}
	exec := newExecutorDefaultHITL(&mockLLMCaller{}, newMockToolExecutor(), &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, cfg)

	action := llm.ToolCall{Name: "search", Input: json.RawMessage(`{"query":"stuck"}`)}
	resp := &llm.ChatResponse{Message: llm.Message{Role: "assistant", ToolCalls: []llm.ToolCall{action}}}
	state := &runState{effectiveMaxSteps: 10, responseGroup: 1}

	// First identical call: recorded (count=1) and allowed through.
	loopAct, result, err := exec.checkRepeatIdenticalTool(context.Background(), action, 0, "", resp, state, newMockContextManager())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if loopAct != actionNone {
		t.Fatalf("loopAct = %v, want actionNone for the first call", loopAct)
	}
	if result != nil {
		t.Fatalf("first call must not produce a result, got %+v", result)
	}
	if exec.repeat.count != 1 {
		t.Fatalf("repeat.count = %d after first call, want 1", exec.repeat.count)
	}

	// The first call executed and FAILED — mirror the executor's post-result state.
	exec.repeat.lastIsError = true

	// Second IDENTICAL call after the error must be intercepted by the nudge.
	loopAct, result, err = exec.checkRepeatIdenticalTool(context.Background(), action, 0, "", resp, state, newMockContextManager())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if loopAct != actionBreak || result != nil || !state.circuitBreakerTriggered {
		t.Fatalf("identical retry after error must trigger the nudge: loopAct=%v result=%v triggered=%v", loopAct, result, state.circuitBreakerTriggered)
	}
	if len(state.allSteps) != 1 {
		t.Fatalf("expected exactly 1 nudge step, got %d", len(state.allSteps))
	}
	if got := state.allSteps[0].Observation; got != repeatErrorNudgeMessage {
		t.Fatalf("nudge text = %q, want repeatErrorNudgeMessage %q", got, repeatErrorNudgeMessage)
	}
}

// --- checkParseErrors edge cases ---

func TestCheckParseErrors_NudgeBeforeAbort(t *testing.T) {
	cfg := CircuitBreakerConfig{
		RepeatNudgeThreshold:         3,
		RepeatAbortThreshold:         4,
		TruncationAbortThreshold:     3,
		ParseErrorAbortThreshold:     3,
		FruitlessNudgeThreshold:      50,
		FruitlessAbortThreshold:      60,
		FruitlessMaxResultLen:        32,
		SameToolRepeatNudgeThreshold: 50,
		SameToolRepeatAbortThreshold: 60,
		SameToolResultSizeDelta:      64,
	}
	exec := newExecutorDefaultHITL(&mockLLMCaller{}, newMockToolExecutor(), &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, cfg)
	obs, act, result, err := exec.checkParseErrors(
		context.Background(),
		llm.ToolCall{Name: "create_file"},
		0, "failed to parse input: bad json",
		tools.ToolResult{Content: "failed to parse input: bad json", IsError: true},
		&runState{effectiveMaxSteps: 10},
		newMockContextManager(),
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != nil {
		t.Error("expected nil result on first parse error")
	}
	if act != actionNone {
		t.Errorf("expected actionNone on first parse error, got %v", act)
	}
	if !strings.Contains(obs, "failed to parse input") {
		t.Error("observation should contain original error")
	}
	if !strings.Contains(obs, "malformed") {
		t.Error("observation should contain nudge about malformed arguments")
	}
}

func TestCheckParseErrors_NudgeContainsDiagnostics(t *testing.T) {
	cfg := CircuitBreakerConfig{
		RepeatNudgeThreshold:         3,
		RepeatAbortThreshold:         4,
		TruncationAbortThreshold:     3,
		ParseErrorAbortThreshold:     3,
		FruitlessNudgeThreshold:      50,
		FruitlessAbortThreshold:      60,
		FruitlessMaxResultLen:        32,
		SameToolRepeatNudgeThreshold: 50,
		SameToolRepeatAbortThreshold: 60,
		SameToolResultSizeDelta:      64,
	}
	exec := newExecutorDefaultHITL(&mockLLMCaller{}, newMockToolExecutor(), &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, cfg)
	raw := `{"path": "/tmp/diag.txt", "content": "RAW_OUTPUT_FRAGMENT_MARKER"}`
	obs, _, _, err := exec.checkParseErrors(
		context.Background(),
		llm.ToolCall{Name: "create_file", Input: json.RawMessage(raw)},
		0,
		"failed to parse input: bad json",
		tools.ToolResult{Content: "failed to parse input: bad json", IsError: true},
		&runState{effectiveMaxSteps: 10},
		newMockContextManager(),
	)
	if err != nil {
		t.Fatalf("checkParseErrors() error = %v", err)
	}
	if !strings.Contains(obs, "failed to parse input: bad json") {
		t.Errorf("nudge should include the parse error description, got: %s", obs)
	}
	if !strings.Contains(obs, "RAW_OUTPUT_FRAGMENT_MARKER") {
		t.Errorf("nudge should quote a fragment of the raw model output, got: %s", obs)
	}
	if !strings.Contains(obs, "ONLY a valid JSON tool call") {
		t.Errorf("nudge should demand a JSON-only response format, got: %s", obs)
	}
}

func TestCheckParseErrors_NudgeTruncatesRawOutput(t *testing.T) {
	cfg := CircuitBreakerConfig{
		RepeatNudgeThreshold:         3,
		RepeatAbortThreshold:         4,
		TruncationAbortThreshold:     3,
		ParseErrorAbortThreshold:     3,
		FruitlessNudgeThreshold:      50,
		FruitlessAbortThreshold:      60,
		FruitlessMaxResultLen:        32,
		SameToolRepeatNudgeThreshold: 50,
		SameToolRepeatAbortThreshold: 60,
		SameToolResultSizeDelta:      64,
	}
	exec := newExecutorDefaultHITL(&mockLLMCaller{}, newMockToolExecutor(), &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, cfg)
	long := `{"prefix_marker":"` + strings.Repeat("a", 1000) + `","tail_marker":true}`
	obs, _, _, err := exec.checkParseErrors(
		context.Background(),
		llm.ToolCall{Name: "create_file", Input: json.RawMessage(long)},
		0,
		"failed to parse input: bad json",
		tools.ToolResult{Content: "failed to parse input: bad json", IsError: true},
		&runState{effectiveMaxSteps: 10},
		newMockContextManager(),
	)
	if err != nil {
		t.Fatalf("checkParseErrors() error = %v", err)
	}
	if !strings.Contains(obs, "prefix_marker") {
		t.Errorf("nudge should quote the beginning of the raw model output, got: %s", obs)
	}
	if strings.Contains(obs, "tail_marker") {
		t.Errorf("nudge should truncate the raw output excerpt to ~%d chars, but it contains the tail", parseErrorNudgeRawPrefix)
	}
	if idx := strings.Index(obs, "Your raw output as received"); idx < 0 {
		t.Errorf("nudge should contain the raw output header, got: %s", obs)
	} else {
		line := obs[idx:]
		if end := strings.IndexByte(line, '\n'); end >= 0 {
			line = line[:end]
		}
		if n := len([]rune(line)); n > parseErrorNudgeRawPrefix+120 {
			t.Errorf("raw output line should be capped near %d chars, got %d", parseErrorNudgeRawPrefix, n)
		}
	}
}

func TestCheckParseErrors_ResetOnSuccess(t *testing.T) {
	cfg := CircuitBreakerConfig{
		RepeatNudgeThreshold:         3,
		RepeatAbortThreshold:         4,
		TruncationAbortThreshold:     3,
		ParseErrorAbortThreshold:     3,
		FruitlessNudgeThreshold:      50,
		FruitlessAbortThreshold:      60,
		FruitlessMaxResultLen:        32,
		SameToolRepeatNudgeThreshold: 50,
		SameToolRepeatAbortThreshold: 60,
		SameToolResultSizeDelta:      64,
	}
	exec := newExecutorDefaultHITL(&mockLLMCaller{}, newMockToolExecutor(), &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, cfg)
	for i := 0; i < 2; i++ {
		_, _, _, _ = exec.checkParseErrors(
			context.Background(),
			llm.ToolCall{Name: "create_file"},
			0, "failed to parse input",
			tools.ToolResult{Content: "failed to parse input", IsError: true},
			&runState{effectiveMaxSteps: 10},
			newMockContextManager(),
		)
	}
	if exec.consecutiveParseErrorCount != 2 {
		t.Errorf("parse error count should be 2, got %d", exec.consecutiveParseErrorCount)
	}
	_, _, _, _ = exec.checkParseErrors(
		context.Background(),
		llm.ToolCall{Name: "search"},
		0, "found results",
		tools.ToolResult{Content: "found results", IsError: false},
		&runState{effectiveMaxSteps: 10},
		newMockContextManager(),
	)
	if exec.consecutiveParseErrorCount != 0 {
		t.Errorf("parse error count should be 0 after success, got %d", exec.consecutiveParseErrorCount)
	}
}

func TestCheckParseErrors_DifferentTool(t *testing.T) {
	cfg := CircuitBreakerConfig{
		RepeatNudgeThreshold:         3,
		RepeatAbortThreshold:         4,
		TruncationAbortThreshold:     3,
		ParseErrorAbortThreshold:     3,
		FruitlessNudgeThreshold:      50,
		FruitlessAbortThreshold:      60,
		FruitlessMaxResultLen:        32,
		SameToolRepeatNudgeThreshold: 50,
		SameToolRepeatAbortThreshold: 60,
		SameToolResultSizeDelta:      64,
	}
	exec := newExecutorDefaultHITL(&mockLLMCaller{}, newMockToolExecutor(), &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, cfg)
	_, _, _, _ = exec.checkParseErrors(
		context.Background(),
		llm.ToolCall{Name: "create_file"},
		0, "failed to parse input",
		tools.ToolResult{Content: "failed to parse input", IsError: true},
		&runState{effectiveMaxSteps: 10},
		newMockContextManager(),
	)
	_, _, _, _ = exec.checkParseErrors(
		context.Background(),
		llm.ToolCall{Name: "write_file"},
		0, "failed to parse input",
		tools.ToolResult{Content: "failed to parse input", IsError: true},
		&runState{effectiveMaxSteps: 10},
		newMockContextManager(),
	)
	if exec.consecutiveParseErrorCount != 1 {
		t.Errorf("parse error count should be 1 for different tool, got %d", exec.consecutiveParseErrorCount)
	}
	if exec.consecutiveParseErrorTool != "write_file" {
		t.Errorf("parse error tool should be write_file, got %q", exec.consecutiveParseErrorTool)
	}
}

func TestCheckParseErrors_NonParseError(t *testing.T) {
	cfg := defaultCircuitBreakerConfig
	exec := newExecutorDefaultHITL(&mockLLMCaller{}, newMockToolExecutor(), &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, cfg)
	obs, act, result, err := exec.checkParseErrors(
		context.Background(),
		llm.ToolCall{Name: "tool"},
		0, "something went wrong",
		tools.ToolResult{Content: "something went wrong", IsError: true},
		&runState{effectiveMaxSteps: 10},
		newMockContextManager(),
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != nil {
		t.Error("non-parse error should not trigger abort")
	}
	if act != actionNone {
		t.Errorf("non-parse error should return actionNone, got %v", act)
	}
	if obs != "something went wrong" {
		t.Errorf("observation should be unchanged, got %q", obs)
	}
}

// --- context With/From tests ---

type testWriterStruct struct{}

func (tw *testWriterStruct) Write(p []byte) (n int, err error) { return len(p), nil }

type testFactStoreStruct struct{}

func (tfs *testFactStoreStruct) StoreFact(keywords []string, content, author string) {}
func (tfs *testFactStoreStruct) SearchFacts(keywords []string) []FactEntry           { return nil }

var _ FactStore = (*testFactStoreStruct)(nil)

func TestWithStepID_StepIDFromContext(t *testing.T) {
	ctx := WithStepID(context.Background(), "step_42")
	if got := StepIDFromContext(ctx); got != "step_42" {
		t.Errorf("StepIDFromContext = %q, want %q", got, "step_42")
	}
}

func TestStepIDFromContext_Empty(t *testing.T) {
	if got := StepIDFromContext(context.Background()); got != "" {
		t.Errorf("StepIDFromContext with no step ID = %q, want empty", got)
	}
}

func TestWithDumpWriter_DumpWriterFromContext(t *testing.T) {
	w := &testWriterStruct{}
	ctx := WithDumpWriter(context.Background(), w)
	if got := DumpWriterFromContext(ctx); got != w {
		t.Error("DumpWriterFromContext should return the set writer")
	}
}

func TestDumpWriterFromContext_Nil(t *testing.T) {
	if got := DumpWriterFromContext(context.Background()); got != nil {
		t.Error("DumpWriterFromContext should return nil when not set")
	}
}

func TestWithFactStore_FactStoreFromContext(t *testing.T) {
	fs := &testFactStoreStruct{}
	ctx := WithFactStore(context.Background(), fs)
	if got := FactStoreFromContext(ctx); got != fs {
		t.Error("FactStoreFromContext should return the set fact store")
	}
}

func TestFactStoreFromContext_Nil(t *testing.T) {
	if got := FactStoreFromContext(context.Background()); got != nil {
		t.Error("FactStoreFromContext should return nil when not set")
	}
}

// --- handleTruncationStopReason edge cases ---

func TestHandleTruncationStopReason_Abort_DefaultHandler(t *testing.T) {
	cfg := CircuitBreakerConfig{
		RepeatNudgeThreshold:         3,
		RepeatAbortThreshold:         4,
		TruncationAbortThreshold:     2,
		ParseErrorAbortThreshold:     3,
		FruitlessNudgeThreshold:      50,
		FruitlessAbortThreshold:      60,
		FruitlessMaxResultLen:        32,
		SameToolRepeatNudgeThreshold: 50,
		SameToolRepeatAbortThreshold: 60,
		SameToolResultSizeDelta:      64,
	}
	exec := newExecutorDefaultHITL(&mockLLMCaller{}, newMockToolExecutor(), &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, cfg)
	exec.emitter = &NoopEvents{}
	exec.consecutiveTruncationCount = 2 // exactly at threshold

	resp := &llm.ChatResponse{
		Message: llm.Message{
			Role: "assistant",
			ToolCalls: []llm.ToolCall{
				{ID: "call_1", Name: "search", Input: []byte(`{"query":"test"}`)},
			},
		},
		Usage: llm.TokenUsage{InputTokens: 100, OutputTokens: 50},
	}
	state := &runState{effectiveMaxSteps: 10}
	cw := newMockContextManager()

	result, act := exec.handleTruncationStopReason(context.Background(), resp, "thinking", state, cw)
	if result == nil {
		t.Fatal("expected non-nil result for truncation abort")
	}
	if result.Finished {
		t.Error("expected Finished=false for truncation abort")
	}
	if act != actionNone {
		t.Errorf("expected actionNone, got %v", act)
	}
}

func TestHandleTruncationStopReason_Abort_StepLimitDeny(t *testing.T) {
	cfg := CircuitBreakerConfig{
		RepeatNudgeThreshold:         3,
		RepeatAbortThreshold:         4,
		TruncationAbortThreshold:     2,
		ParseErrorAbortThreshold:     3,
		FruitlessNudgeThreshold:      50,
		FruitlessAbortThreshold:      60,
		FruitlessMaxResultLen:        32,
		SameToolRepeatNudgeThreshold: 50,
		SameToolRepeatAbortThreshold: 60,
		SameToolResultSizeDelta:      64,
	}
	exec := newExecutorDefaultHITL(&mockLLMCaller{}, newMockToolExecutor(), &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, cfg)
	exec.emitter = &NoopEvents{}
	exec.consecutiveTruncationCount = 2
	exec.SetHITLHandler(&testStepLimitAdapter{fn: func(ctx context.Context, stepNum, effectiveMaxSteps int, reason string) (StepLimitResponse, error) {
		return StepLimitDeny, nil
	}})

	resp := &llm.ChatResponse{
		Message: llm.Message{
			Role: "assistant",
			ToolCalls: []llm.ToolCall{
				{ID: "call_1", Name: "search", Input: []byte(`{"query":"test"}`)},
			},
		},
		Usage: llm.TokenUsage{InputTokens: 100, OutputTokens: 50},
	}
	state := &runState{effectiveMaxSteps: 10}
	cw := newMockContextManager()

	result, act := exec.handleTruncationStopReason(context.Background(), resp, "thinking", state, cw)
	if result == nil {
		t.Fatal("expected non-nil result for truncation abort with StepLimitDeny")
	}
	if result.Finished {
		t.Error("expected Finished=false")
	}
	if act != actionNone {
		t.Errorf("expected actionNone, got %v", act)
	}
}

func TestHandleTruncationStopReason_Abort_OnStepLimitError(t *testing.T) {
	cfg := CircuitBreakerConfig{
		RepeatNudgeThreshold:         3,
		RepeatAbortThreshold:         4,
		TruncationAbortThreshold:     2,
		ParseErrorAbortThreshold:     3,
		FruitlessNudgeThreshold:      50,
		FruitlessAbortThreshold:      60,
		FruitlessMaxResultLen:        32,
		SameToolRepeatNudgeThreshold: 50,
		SameToolRepeatAbortThreshold: 60,
		SameToolResultSizeDelta:      64,
	}
	exec := newExecutorDefaultHITL(&mockLLMCaller{}, newMockToolExecutor(), &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, cfg)
	exec.emitter = &NoopEvents{}
	exec.consecutiveTruncationCount = 2
	exec.SetHITLHandler(&testStepLimitAdapter{fn: func(ctx context.Context, stepNum, effectiveMaxSteps int, reason string) (StepLimitResponse, error) {
		return StepLimitDeny, errors.New("callback error")
	}})

	resp := &llm.ChatResponse{
		Message: llm.Message{
			Role: "assistant",
			ToolCalls: []llm.ToolCall{
				{ID: "call_1", Name: "search", Input: []byte(`{"query":"test"}`)},
			},
		},
		Usage: llm.TokenUsage{InputTokens: 100, OutputTokens: 50},
	}
	state := &runState{effectiveMaxSteps: 10}
	cw := newMockContextManager()

	result, act := exec.handleTruncationStopReason(context.Background(), resp, "thinking", state, cw)
	if result == nil {
		t.Fatal("expected non-nil result when OnStepLimit errors")
	}
	if act != actionNone {
		t.Errorf("expected actionNone, got %v", act)
	}
}

// --- checkParseErrors abort threshold ---

func TestCheckParseErrors_AbortThreshold(t *testing.T) {
	cfg := CircuitBreakerConfig{
		RepeatNudgeThreshold:         3,
		RepeatAbortThreshold:         4,
		TruncationAbortThreshold:     3,
		ParseErrorAbortThreshold:     3,
		FruitlessNudgeThreshold:      50,
		FruitlessAbortThreshold:      60,
		FruitlessMaxResultLen:        32,
		SameToolRepeatNudgeThreshold: 50,
		SameToolRepeatAbortThreshold: 60,
		SameToolResultSizeDelta:      64,
	}
	exec := newExecutorDefaultHITL(&mockLLMCaller{}, newMockToolExecutor(), &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, cfg)
	exec.emitter = &NoopEvents{}
	exec.consecutiveParseErrorCount = 2
	exec.consecutiveParseErrorTool = "create_file"

	_, act, result, err := exec.checkParseErrors(
		context.Background(),
		llm.ToolCall{Name: "create_file"},
		0, "failed to parse input: invalid json",
		tools.ToolResult{Content: "failed to parse input: invalid json", IsError: true},
		&runState{effectiveMaxSteps: 10},
		newMockContextManager(),
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result == nil {
		t.Fatal("expected non-nil result on 3rd parse error (abort)")
	}
	if act != actionNone {
		t.Errorf("expected actionNone for abort, got %v", act)
	}
	if !strings.Contains(result.Output, "failed to parse input") {
		t.Errorf("output should mention parse failure, got %q", result.Output)
	}
}

func TestCheckParseErrors_Abort_StepLimitAllowOnce(t *testing.T) {
	cfg := CircuitBreakerConfig{
		RepeatNudgeThreshold:         3,
		RepeatAbortThreshold:         4,
		TruncationAbortThreshold:     3,
		ParseErrorAbortThreshold:     3,
		FruitlessNudgeThreshold:      50,
		FruitlessAbortThreshold:      60,
		FruitlessMaxResultLen:        32,
		SameToolRepeatNudgeThreshold: 50,
		SameToolRepeatAbortThreshold: 60,
		SameToolResultSizeDelta:      64,
	}
	exec := newExecutorDefaultHITL(&mockLLMCaller{}, newMockToolExecutor(), &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, cfg)
	exec.emitter = &NoopEvents{}
	exec.consecutiveParseErrorCount = 2
	exec.consecutiveParseErrorTool = "create_file"
	exec.SetHITLHandler(&testStepLimitAdapter{fn: func(ctx context.Context, stepNum, effectiveMaxSteps int, reason string) (StepLimitResponse, error) {
		return StepLimitAllowOnce, nil
	}})

	state := &runState{effectiveMaxSteps: 10}
	cw := newMockContextManager()

	obs, act, result, _ := exec.checkParseErrors(
		context.Background(),
		llm.ToolCall{Name: "create_file"},
		0, "failed to parse input: bad json",
		tools.ToolResult{Content: "failed to parse input: bad json", IsError: true},
		state,
		cw,
	)
	if result != nil {
		t.Error("expected nil result when StepLimitAllowOnce")
	}
	if act != actionBreak {
		t.Errorf("expected actionBreak, got %v", act)
	}
	if !state.circuitBreakerTriggered {
		t.Error("circuitBreakerTriggered should be true")
	}
	_ = obs
}

func TestCheckParseErrors_Abort_StepLimitAllowMore(t *testing.T) {
	cfg := CircuitBreakerConfig{
		RepeatNudgeThreshold:         3,
		RepeatAbortThreshold:         4,
		TruncationAbortThreshold:     3,
		ParseErrorAbortThreshold:     3,
		FruitlessNudgeThreshold:      50,
		FruitlessAbortThreshold:      60,
		FruitlessMaxResultLen:        32,
		SameToolRepeatNudgeThreshold: 50,
		SameToolRepeatAbortThreshold: 60,
		SameToolResultSizeDelta:      64,
	}
	exec := newExecutorDefaultHITL(&mockLLMCaller{}, newMockToolExecutor(), &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, cfg)
	exec.emitter = &NoopEvents{}
	exec.consecutiveParseErrorCount = 2
	exec.consecutiveParseErrorTool = "create_file"
	exec.SetHITLHandler(&testStepLimitAdapter{fn: func(ctx context.Context, stepNum, effectiveMaxSteps int, reason string) (StepLimitResponse, error) {
		return StepLimitAllowMore, nil
	}})

	state := &runState{effectiveMaxSteps: 10}
	cw := newMockContextManager()

	_, act, result, _ := exec.checkParseErrors(
		context.Background(),
		llm.ToolCall{Name: "create_file"},
		0, "failed to parse input: bad json",
		tools.ToolResult{Content: "failed to parse input: bad json", IsError: true},
		state,
		cw,
	)
	if result != nil {
		t.Error("expected nil result when StepLimitAllowMore grants a reprieve")
	}
	if act != actionBreak {
		t.Errorf("expected actionBreak, got %v", act)
	}
	if !state.circuitBreakerTriggered {
		t.Error("circuitBreakerTriggered should be true")
	}
	// In a circuit breaker, AllowMore is a reprieve equivalent to AllowOnce:
	// the consecutive counter is reset, but no additional iterations are granted.
	if exec.consecutiveParseErrorCount != 0 {
		t.Errorf("consecutiveParseErrorCount = %d, want 0 (counter reset)", exec.consecutiveParseErrorCount)
	}
	if state.effectiveMaxSteps != 10 {
		t.Errorf("effectiveMaxSteps = %d, want 10 (no budget extension in circuit breakers)", state.effectiveMaxSteps)
	}
	// The AllowMore nudge must branch away from the AllowOnce "ONE more chance" wording.
	foundAllowMoreNudge := false
	for _, s := range state.allSteps {
		if strings.Contains(s.UserNudge, "let you continue") {
			foundAllowMoreNudge = true
		}
		if strings.Contains(s.UserNudge, "ONE more chance") {
			t.Errorf("AllowMore nudge should not reuse AllowOnce wording, got %q", s.UserNudge)
		}
	}
	if !foundAllowMoreNudge {
		t.Error("expected AllowMore ('let you continue') nudge to be injected")
	}
}

// --- applyPerToolTruncation ---

func TestApplyPerToolTruncation_NilConfig(t *testing.T) {
	exec := newExecutorDefaultHITL(&mockLLMCaller{}, newMockToolExecutor(), &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
	content, truncated := exec.applyPerToolTruncation("some content", "read_file")
	if truncated {
		t.Error("should not be truncated with nil config")
	}
	if content != "some content" {
		t.Errorf("content = %q, want %q", content, "some content")
	}
}

func TestApplyPerToolTruncation_ToolNotInConfig(t *testing.T) {
	cfg := map[string]ToolTruncationConfig{"search": {MaxLines: 10}}
	exec := newExecutorDefaultHITL(&mockLLMCaller{}, newMockToolExecutor(), &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
	exec.SetPerToolTruncation(cfg)
	content, truncated := exec.applyPerToolTruncation("some content", "read_file")
	if truncated {
		t.Error("should not be truncated when tool not in config")
	}
	if content != "some content" {
		t.Errorf("content = %q, want unchanged", content)
	}
}

func TestApplyPerToolTruncation_LineTruncation(t *testing.T) {
	cfg := map[string]ToolTruncationConfig{"read_file": {MaxLines: 3}}
	exec := newExecutorDefaultHITL(&mockLLMCaller{}, newMockToolExecutor(), &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
	exec.SetPerToolTruncation(cfg)
	content := "line1\nline2\nline3\nline4\nline5"
	result, truncated := exec.applyPerToolTruncation(content, "read_file")
	if !truncated {
		t.Error("expected truncated=true for content exceeding MaxLines")
	}
	lines := strings.Split(result, "\n")
	if len(lines) != 3 {
		t.Errorf("expected 3 lines after truncation, got %d", len(lines))
	}
}

func TestApplyPerToolTruncation_ByteTruncation(t *testing.T) {
	cfg := map[string]ToolTruncationConfig{"read_file": {MaxBytes: 5}}
	exec := newExecutorDefaultHITL(&mockLLMCaller{}, newMockToolExecutor(), &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
	exec.SetPerToolTruncation(cfg)
	content := "hello world"
	result, truncated := exec.applyPerToolTruncation(content, "read_file")
	if !truncated {
		t.Error("expected truncated=true for content exceeding MaxBytes")
	}
	if len(result) != 5 {
		t.Errorf("expected 5 bytes after truncation, got %d", len(result))
	}
	if result != "hello" {
		t.Errorf("expected 'hello', got %q", result)
	}
}

func TestApplyPerToolTruncation_UTF8Safe(t *testing.T) {
	cfg := map[string]ToolTruncationConfig{"read_file": {MaxBytes: 4}}
	exec := newExecutorDefaultHITL(&mockLLMCaller{}, newMockToolExecutor(), &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
	exec.SetPerToolTruncation(cfg)
	// "héllo" = 6 bytes: h(1) + é(2) + l(1) + l(1) + o(1). MaxBytes=4 cuts at byte 4 (middle of 'é')
	content := "héllo"
	result, truncated := exec.applyPerToolTruncation(content, "read_file")
	if !truncated {
		t.Error("expected truncated=true")
	}
	// Should walk back to valid UTF-8 boundary: "h" is 1 byte, no room for "hé" (3 bytes)
	// MaxBytes=4 means we try "héll" (4 bytes) but 'é' is at bytes 1-2, so bytes 0-3 = "h" + partial "é"
	// But... wait, "héll" = h(1) + é(2) + l(1) = 4 bytes which is a valid UTF-8 boundary
	// Actually: h=1byte, é=2bytes, l=1byte, l=1byte, o=1byte = 6 bytes total
	// bytes[0:4] = h(1) + é(2) + l(1) = "hél" which is valid UTF-8
	if result != "hél" {
		t.Errorf("expected 'hél', got %q (len=%d)", result, len(result))
	}
}

func TestApplyPerToolTruncation_UTF8SafeWalkBack(t *testing.T) {
	cfg := map[string]ToolTruncationConfig{"read_file": {MaxBytes: 5}}
	exec := newExecutorDefaultHITL(&mockLLMCaller{}, newMockToolExecutor(), &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
	exec.SetPerToolTruncation(cfg)
	// "héllo" = h(1) + é(2) + l(1) + l(1) + o(1) = 6 bytes
	// bytes[0:5] = h(1) + é(2) + l(1) + l(1) = "héll" valid UTF-8
	content := "héllo"
	result, truncated := exec.applyPerToolTruncation(content, "read_file")
	if !truncated {
		t.Error("expected truncated=true")
	}
	if result != "héll" {
		t.Errorf("expected 'héll', got %q", result)
	}
}

func TestHandleTruncationStopReason_BelowThreshold(t *testing.T) {
	cfg := CircuitBreakerConfig{
		RepeatNudgeThreshold:         3,
		RepeatAbortThreshold:         4,
		TruncationAbortThreshold:     5,
		ParseErrorAbortThreshold:     3,
		FruitlessNudgeThreshold:      50,
		FruitlessAbortThreshold:      60,
		FruitlessMaxResultLen:        32,
		SameToolRepeatNudgeThreshold: 50,
		SameToolRepeatAbortThreshold: 60,
		SameToolResultSizeDelta:      64,
	}
	exec := newExecutorDefaultHITL(&mockLLMCaller{}, newMockToolExecutor(), &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, cfg)
	exec.emitter = &NoopEvents{}
	exec.consecutiveTruncationCount = 1 // below threshold of 5

	resp := &llm.ChatResponse{
		Message: llm.Message{
			Role: "assistant",
			ToolCalls: []llm.ToolCall{
				{ID: "call_1", Name: "search", Input: []byte(`{"query":"test"}`)},
			},
		},
		Usage: llm.TokenUsage{InputTokens: 100, OutputTokens: 50},
	}
	state := &runState{effectiveMaxSteps: 10}
	cw := newMockContextManager()

	result, act := exec.handleTruncationStopReason(context.Background(), resp, "thinking", state, cw)
	if result != nil {
		t.Error("expected nil result below threshold")
	}
	if act != actionContinue {
		t.Errorf("expected actionContinue, got %v", act)
	}
	if len(state.allSteps) != 1 {
		t.Errorf("expected 1 step to be added, got %d", len(state.allSteps))
	}
	if exec.consecutiveTruncationCount != 2 {
		t.Errorf("expected consecutiveTruncationCount=2, got %d", exec.consecutiveTruncationCount)
	}
}

// --- processBatchTool tests (via processSingleToolCall) ---

func TestProcessBatchTool_ParseError(t *testing.T) {
	exec := newExecutorDefaultHITL(&mockLLMCaller{}, newMockToolExecutor(), &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
	exec.emitter = &NoopEvents{}

	batchInput := json.RawMessage(`invalid json`)
	action := llm.ToolCall{ID: "batch_1", Name: "batch", Input: batchInput}
	resp := &llm.ChatResponse{
		Message:    llm.Message{Role: "assistant", ToolCalls: []llm.ToolCall{action}},
		Usage:      llm.TokenUsage{InputTokens: 100, OutputTokens: 50},
		StopReason: "tool_use",
	}
	state := &runState{effectiveMaxSteps: 10}
	cw := newMockContextManager()

	result, act, err := exec.processSingleToolCall(context.Background(), action, 0, resp.Message.ToolCalls, resp, "thinking", state, cw)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != nil {
		t.Error("expected nil result for batch parse error")
	}
	if act != actionNone {
		t.Errorf("expected actionNone, got %v", act)
	}
}

func TestProcessBatchTool_EmptyCalls(t *testing.T) {
	exec := newExecutorDefaultHITL(&mockLLMCaller{}, newMockToolExecutor(), &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
	exec.emitter = &NoopEvents{}

	batchInput, _ := json.Marshal(map[string]interface{}{"calls": []interface{}{}})
	action := llm.ToolCall{ID: "batch_1", Name: "batch", Input: batchInput}
	resp := &llm.ChatResponse{
		Message:    llm.Message{Role: "assistant", ToolCalls: []llm.ToolCall{action}},
		Usage:      llm.TokenUsage{InputTokens: 100, OutputTokens: 50},
		StopReason: "tool_use",
	}
	state := &runState{effectiveMaxSteps: 10}
	cw := newMockContextManager()

	result, act, err := exec.processSingleToolCall(context.Background(), action, 0, resp.Message.ToolCalls, resp, "thinking", state, cw)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != nil {
		t.Error("expected nil result for empty calls")
	}
	if act != actionNone {
		t.Errorf("expected actionNone, got %v", act)
	}
}

// TestProcessBatchTool_CarriesReasoningItems pins that the batch path carries
// the response's ReasoningItems (including EncryptedContent from a stateless
// Responses backend) onto the FIRST step of the batch group — exactly like
// the standalone tool-call path — so the next BuildPrompt round-trips the
// reasoning chain through the assistant message.
func TestProcessBatchTool_CarriesReasoningItems(t *testing.T) {
	mockTools := newMockToolExecutor()
	mockTools.results["search"] = tools.ToolResult{Content: "search result"}

	exec := newExecutorDefaultHITL(&mockLLMCaller{}, mockTools, &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
	exec.emitter = &NoopEvents{}

	subInput, _ := json.Marshal(map[string]string{"query": "test"})
	type batchCall struct {
		Tool  string          `json:"tool"`
		Input json.RawMessage `json:"input"`
	}
	batchInput, _ := json.Marshal(map[string][]batchCall{
		"calls": {
			{Tool: "search", Input: subInput},
			{Tool: "search", Input: subInput},
		},
	})

	reasoning := []llm.ReasoningItem{
		{ID: "rs_1", Summary: "committed plan", EncryptedContent: "gAAAA-encrypted"},
	}
	action := llm.ToolCall{ID: "batch_1", Name: "batch", Input: batchInput}
	resp := &llm.ChatResponse{
		Message: llm.Message{
			Role:           "assistant",
			ToolCalls:      []llm.ToolCall{action},
			ReasoningItems: reasoning,
		},
		Usage:      llm.TokenUsage{InputTokens: 100, OutputTokens: 50},
		StopReason: "tool_use",
	}
	state := &runState{effectiveMaxSteps: 10}
	cw := newMockContextManager()

	if _, _, err := exec.processSingleToolCall(context.Background(), action, 0, resp.Message.ToolCalls, resp, "thinking", state, cw); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(state.allSteps) != 2 {
		t.Fatalf("steps = %d, want 2 (one per sub-call)", len(state.allSteps))
	}
	first := state.allSteps[0]
	if len(first.ReasoningItems) != 1 || first.ReasoningItems[0].ID != "rs_1" ||
		first.ReasoningItems[0].EncryptedContent != "gAAAA-encrypted" {
		t.Errorf("first batch step ReasoningItems = %+v, want the response's reasoning item (with encrypted content) to ride it", first.ReasoningItems)
	}
	if first.Thought != "thinking" {
		t.Errorf("first batch step Thought = %q, want the response thought", first.Thought)
	}
	second := state.allSteps[1]
	if len(second.ReasoningItems) != 0 {
		t.Errorf("second batch step ReasoningItems = %+v, want none (only the first step carries them)", second.ReasoningItems)
	}
}

// TestProcessBatchTool_InterceptedPathsCarryReasoningItems pins the
// interception branches of the batch path: a rejected (HITL-denied) FIRST
// sub-call, a nested-batch FIRST sub-call and an unparseable batch input all
// materialize their step as groupSteps[0] — the step BuildPrompt reads the
// response's reasoning items from — so each must carry them instead of
// silently losing the encrypted reasoning chain for every batched turn.
func TestProcessBatchTool_InterceptedPathsCarryReasoningItems(t *testing.T) {
	newResp := func(action llm.ToolCall) *llm.ChatResponse {
		return &llm.ChatResponse{
			Message: llm.Message{
				Role: "assistant",
				ToolCalls: []llm.ToolCall{
					{ID: "batch_1", Name: "batch", Input: action.Input},
				},
				ReasoningItems: []llm.ReasoningItem{
					{ID: "rs_1", Summary: "committed plan", EncryptedContent: "gAAAA-encrypted"},
				},
			},
			Usage:      llm.TokenUsage{InputTokens: 100, OutputTokens: 50},
			StopReason: "tool_use",
		}
	}

	t.Run("hitl-rejected first sub-call", func(t *testing.T) {
		mockTools := newMockToolExecutor()
		exec := NewExecutor(&mockLLMCaller{}, mockTools, 10,
			WithTokenCounter(&mockTokenCounter{}),
			WithCircuitBreaker(defaultCircuitBreakerConfig),
			WithHITL(&denyingHITLHandler{}),
		)
		exec.emitter = &NoopEvents{}

		subInput, _ := json.Marshal(map[string]string{"query": "test"})
		type batchCall struct {
			Tool  string          `json:"tool"`
			Input json.RawMessage `json:"input"`
		}
		batchInput, _ := json.Marshal(map[string][]batchCall{
			"calls": {
				{Tool: "search", Input: subInput},
				{Tool: "search", Input: subInput},
			},
		})
		action := llm.ToolCall{ID: "batch_1", Name: "batch", Input: batchInput}
		state := &runState{effectiveMaxSteps: 10}
		cw := newMockContextManager()

		if _, _, err := exec.processSingleToolCall(context.Background(), action, 0, []llm.ToolCall{action}, newResp(action), "thinking", state, cw); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(state.allSteps) != 2 {
			t.Fatalf("steps = %d, want 2 (one rejected step per sub-call)", len(state.allSteps))
		}
		first := state.allSteps[0]
		if len(first.ReasoningItems) != 1 || first.ReasoningItems[0].EncryptedContent != "gAAAA-encrypted" {
			t.Errorf("rejected first sub-call ReasoningItems = %+v, want the response's reasoning item to ride it", first.ReasoningItems)
		}
		if len(state.allSteps[1].ReasoningItems) != 0 {
			t.Errorf("rejected second sub-call ReasoningItems = %+v, want none (only the first step carries them)", state.allSteps[1].ReasoningItems)
		}
	})

	t.Run("nested batch first sub-call", func(t *testing.T) {
		mockTools := newMockToolExecutor()
		exec := newExecutorDefaultHITL(&mockLLMCaller{}, mockTools, &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
		exec.emitter = &NoopEvents{}

		innerInput, _ := json.Marshal(map[string]any{"calls": []any{}})
		type batchCall struct {
			Tool  string          `json:"tool"`
			Input json.RawMessage `json:"input"`
		}
		batchInput, _ := json.Marshal(map[string][]batchCall{
			"calls": {
				{Tool: "batch", Input: innerInput},
			},
		})
		action := llm.ToolCall{ID: "batch_1", Name: "batch", Input: batchInput}
		state := &runState{effectiveMaxSteps: 10}
		cw := newMockContextManager()

		if _, _, err := exec.processSingleToolCall(context.Background(), action, 0, []llm.ToolCall{action}, newResp(action), "thinking", state, cw); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(state.allSteps) != 1 {
			t.Fatalf("steps = %d, want 1 (the nested-batch guard step)", len(state.allSteps))
		}
		first := state.allSteps[0]
		if len(first.ReasoningItems) != 1 || first.ReasoningItems[0].EncryptedContent != "gAAAA-encrypted" {
			t.Errorf("nested-guard first sub-call ReasoningItems = %+v, want the response's reasoning item to ride it", first.ReasoningItems)
		}
	})

	t.Run("unparseable batch input", func(t *testing.T) {
		mockTools := newMockToolExecutor()
		exec := newExecutorDefaultHITL(&mockLLMCaller{}, mockTools, &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
		exec.emitter = &NoopEvents{}

		action := llm.ToolCall{ID: "batch_1", Name: "batch", Input: json.RawMessage(`{"calls": [}`)}
		state := &runState{effectiveMaxSteps: 10}
		cw := newMockContextManager()

		if _, _, err := exec.processSingleToolCall(context.Background(), action, 0, []llm.ToolCall{action}, newResp(action), "thinking", state, cw); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(state.allSteps) != 1 {
			t.Fatalf("steps = %d, want 1 (the parse-error step)", len(state.allSteps))
		}
		first := state.allSteps[0]
		if len(first.ReasoningItems) != 1 || first.ReasoningItems[0].EncryptedContent != "gAAAA-encrypted" {
			t.Errorf("parse-error step ReasoningItems = %+v, want the response's reasoning item to ride it", first.ReasoningItems)
		}
	})
}

// failingHITLHandler fails every OnToolCall with a Go error — the HITL
// backend itself broke, which is neither an allow nor a denial.
type failingHITLHandler struct{}

func (h *failingHITLHandler) OnToolCall(_ context.Context, _ string, _ json.RawMessage) (*HITLToolDecision, error) {
	return nil, errors.New("hitl backend unavailable")
}

func (h *failingHITLHandler) OnStepLimit(_ context.Context, _, _ int, _ string) (StepLimitResponse, error) {
	return StepLimitDeny, nil
}

// TestProcessBatchTool_SynthesisedSubCallErrorsCarryExecutionKind pins the
// typed error class on the batch path's two executor-synthesised failures
// (specs/contracts/tools.md): a sub-call whose HITL handler itself errored
// and a sub-call whose tool returned a Go error both materialize an error
// step carrying ErrorKind == ToolErrorExecution — the arguments were
// acceptable and the environment failed — never the zero kind and never
// invalid_call. The batch path builds these result literals itself (it does
// not go through the registry's constructors), so each site needs its own
// pin next to the standalone path's.
func TestProcessBatchTool_SynthesisedSubCallErrorsCarryExecutionKind(t *testing.T) {
	type batchCall struct {
		Tool  string          `json:"tool"`
		Input json.RawMessage `json:"input"`
	}
	subInput, _ := json.Marshal(map[string]string{"query": "test"})
	batchInput, _ := json.Marshal(map[string][]batchCall{
		"calls": {
			{Tool: "search", Input: subInput},
		},
	})
	newResp := func() *llm.ChatResponse {
		return &llm.ChatResponse{
			Message: llm.Message{
				Role: "assistant",
				ToolCalls: []llm.ToolCall{
					{ID: "batch_1", Name: "batch", Input: batchInput},
				},
			},
			Usage:      llm.TokenUsage{InputTokens: 100, OutputTokens: 50},
			StopReason: "tool_use",
		}
	}
	run := func(t *testing.T, exec *Executor) Step {
		t.Helper()
		action := llm.ToolCall{ID: "batch_1", Name: "batch", Input: batchInput}
		state := &runState{effectiveMaxSteps: 10}
		if _, _, err := exec.processSingleToolCall(context.Background(), action, 0, []llm.ToolCall{action}, newResp(), "thinking", state, newMockContextManager()); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(state.allSteps) != 1 {
			t.Fatalf("steps = %d, want exactly the one intercepted sub-call step", len(state.allSteps))
		}
		return state.allSteps[0]
	}
	assertExecutionStep := func(t *testing.T, step Step, wantSubstring string) {
		t.Helper()
		if !step.IsError {
			t.Fatal("IsError = false, want true")
		}
		if step.ErrorKind != tools.ToolErrorExecution {
			t.Errorf("ErrorKind = %q, want %q (an executor-synthesised failure is execution class)", step.ErrorKind, tools.ToolErrorExecution)
		}
		if !strings.Contains(step.Observation, wantSubstring) {
			t.Errorf("Observation = %q, want it to contain %q", step.Observation, wantSubstring)
		}
	}

	t.Run("HITL handler error", func(t *testing.T) {
		mockTools := newMockToolExecutor()
		exec := NewExecutor(&mockLLMCaller{}, mockTools, 10,
			WithTokenCounter(&mockTokenCounter{}),
			WithCircuitBreaker(defaultCircuitBreakerConfig),
			WithHITL(&failingHITLHandler{}),
		)
		exec.emitter = &NoopEvents{}
		assertExecutionStep(t, run(t, exec), `HITL handler error for tool "search"`)
		if len(mockTools.calls) != 0 {
			t.Errorf("tool executions = %d, want none (a failed HITL handler must not reach the tool)", len(mockTools.calls))
		}
	})

	t.Run("tool executor Go error", func(t *testing.T) {
		mockTools := newMockToolExecutor()
		mockTools.errors["search"] = errors.New("executor backend unavailable")
		exec := newExecutorDefaultHITL(&mockLLMCaller{}, mockTools, &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
		exec.emitter = &NoopEvents{}
		assertExecutionStep(t, run(t, exec), `error executing "search": executor backend unavailable`)
	})
}

// batchRegistryProbe records real-registry dispatch for batch integration tests.
type batchRegistryProbe struct {
	tools.BaseTool
	executions int
}

func (p *batchRegistryProbe) Execute(context.Context, json.RawMessage) (tools.ToolResult, error) {
	p.executions++
	return tools.ToolResult{Content: "sibling executed"}, nil
}

func TestProcessBatchTool_NamespacedMissingContinues(t *testing.T) {
	registry := tools.NewToolRegistry()
	probe := &batchRegistryProbe{BaseTool: tools.BaseTool{
		ToolName: "echo", ToolGroup: tools.GroupLocalRead,
		Schema: json.RawMessage(`{"type":"object"}`), Policy: tools.PolicyAlwaysAllow,
	}}
	registry.Register(probe)
	exec := newExecutorDefaultHITL(&mockLLMCaller{}, registry, &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
	exec.emitter = &NoopEvents{}
	action := llm.ToolCall{ID: "batch_1", Name: "batch", Input: json.RawMessage(`{"calls":[{"tool":"functions.echo","input":{}},{"tool":"echo","input":{}}]}`)}
	resp := &llm.ChatResponse{Message: llm.Message{Role: "assistant", ToolCalls: []llm.ToolCall{action}}}
	state := &runState{effectiveMaxSteps: 10}
	result, act, err := exec.processSingleToolCall(context.Background(), action, 0, resp.Message.ToolCalls, resp, "thinking", state, newMockContextManager())
	if err != nil || result != nil || act != actionNone {
		t.Fatalf("processSingleToolCall(batch) = %v, %v, %v, want nil/actionNone/nil", result, act, err)
	}
	if len(state.allSteps) != 2 {
		t.Fatalf("processSingleToolCall(batch) steps = %d, want 2", len(state.allSteps))
	}
	first, second := state.allSteps[0], state.allSteps[1]
	if first.Action.Name != "functions.echo" || !first.IsError || !strings.Contains(first.Observation, "batch.calls[].tool") {
		t.Errorf("batch first step = %+v, want literal functions.echo and actionable error", first)
	}
	if second.Action.Name != "echo" || second.IsError || second.Observation != "sibling executed" || probe.executions != 1 {
		t.Errorf("batch sibling = %+v, executions %d, want echo success and exactly 1 execution", second, probe.executions)
	}
}

// TestRunNamespacedHintScopedToTaskTools is the regression for the
// catalog-scoped tool-not-found hint: Executor.Run attaches the run's
// effective catalog (taskTools) as the diagnostic scope, so a namespaced
// miss recommends the un-prefixed candidate ONLY when that candidate is part
// of the run's grant. A globally registered tool excluded from taskTools
// (e.g. an execute-group tool under a read-only delegation) must get the
// neutral not-found error — never a hint pointing outside the task-scoped
// grant.
func TestRunNamespacedHintScopedToTaskTools(t *testing.T) {
	newRegistry := func() (*tools.ToolRegistry, *batchRegistryProbe) {
		registry := tools.NewToolRegistry()
		probe := &batchRegistryProbe{BaseTool: tools.BaseTool{
			ToolName: "echo", ToolGroup: tools.GroupLocalRead,
			Schema: json.RawMessage(`{"type":"object"}`), Policy: tools.PolicyAlwaysAllow,
		}}
		registry.Register(probe)
		return registry, probe
	}
	runNamespacedMiss := func(registry *tools.ToolRegistry, taskTools []tools.ToolDescriptor) []Step {
		mockLLM := &mockLLMCaller{responses: []*llm.ChatResponse{
			llmResponseWithToolCall("thinking", "functions.echo", json.RawMessage(`{}`)),
			llmResponseFinish("done", "recovered"),
		}}
		exec := newExecutorDefaultHITL(mockLLM, registry, &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
		exec.emitter = &NoopEvents{}
		result, err := exec.Run(context.Background(), taskTools, newMockContextManager())
		if err != nil || !result.Finished {
			t.Fatalf("Run() = %+v, %v, want finished run", result, err)
		}
		if len(result.Steps) == 0 || result.Steps[0].Action.Name != "functions.echo" || !result.Steps[0].IsError {
			t.Fatalf("Run() steps = %+v, want first step literal functions.echo error", result.Steps)
		}
		return result.Steps
	}

	t.Run("excluded candidate gets no hint", func(t *testing.T) {
		registry, probe := newRegistry()
		steps := runNamespacedMiss(registry, []tools.ToolDescriptor{
			{Name: "read_file", InputSchema: json.RawMessage(`{}`)},
		})
		if obs := steps[0].Observation; obs != "tool not found: functions.echo" {
			t.Errorf("observation = %q, want neutral not-found without catalog hint", obs)
		}
		if probe.executions != 0 {
			t.Errorf("probe executions = %d, want 0", probe.executions)
		}
	})

	t.Run("granted candidate gets hint", func(t *testing.T) {
		registry, probe := newRegistry()
		steps := runNamespacedMiss(registry, []tools.ToolDescriptor{
			{Name: "echo", InputSchema: json.RawMessage(`{}`)},
		})
		obs := steps[0].Observation
		for _, hint := range []string{`"echo"`, "batch.calls[].tool", "direct tool call name"} {
			if !strings.Contains(obs, hint) {
				t.Errorf("observation = %q, want hint %q", obs, hint)
			}
		}
		if probe.executions != 0 {
			t.Errorf("probe executions = %d, want 0 (the hint never dispatches)", probe.executions)
		}
	})

	t.Run("excluded candidate in batch gets no hint", func(t *testing.T) {
		registry, probe := newRegistry()
		mockLLM := &mockLLMCaller{responses: []*llm.ChatResponse{
			llmResponseWithToolCall("thinking", "batch", json.RawMessage(`{"calls":[{"tool":"functions.echo","input":{}}]}`)),
			llmResponseFinish("done", "recovered"),
		}}
		exec := newExecutorDefaultHITL(mockLLM, registry, &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
		exec.emitter = &NoopEvents{}
		result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
			{Name: "batch", InputSchema: json.RawMessage(`{}`)},
			{Name: "read_file", InputSchema: json.RawMessage(`{}`)},
		}, newMockContextManager())
		if err != nil || !result.Finished {
			t.Fatalf("Run() = %+v, %v, want finished run", result, err)
		}
		if len(result.Steps) == 0 || result.Steps[0].Action.Name != "functions.echo" || !result.Steps[0].IsError {
			t.Fatalf("Run() steps = %+v, want first batch sub-step literal functions.echo error", result.Steps)
		}
		if obs := result.Steps[0].Observation; obs != "tool not found: functions.echo" {
			t.Errorf("batch observation = %q, want neutral not-found without catalog hint", obs)
		}
		if probe.executions != 0 {
			t.Errorf("probe executions = %d, want 0", probe.executions)
		}
	})

	t.Run("granted candidate in batch gets hint", func(t *testing.T) {
		registry, probe := newRegistry()
		mockLLM := &mockLLMCaller{responses: []*llm.ChatResponse{
			llmResponseWithToolCall("thinking", "batch", json.RawMessage(`{"calls":[{"tool":"functions.echo","input":{}}]}`)),
			llmResponseFinish("done", "recovered"),
		}}
		exec := newExecutorDefaultHITL(mockLLM, registry, &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
		exec.emitter = &NoopEvents{}
		result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
			{Name: "batch", InputSchema: json.RawMessage(`{}`)},
			{Name: "echo", InputSchema: json.RawMessage(`{}`)},
		}, newMockContextManager())
		if err != nil || !result.Finished {
			t.Fatalf("Run() = %+v, %v, want finished run", result, err)
		}
		if len(result.Steps) == 0 || result.Steps[0].Action.Name != "functions.echo" || !result.Steps[0].IsError {
			t.Fatalf("Run() steps = %+v, want first batch sub-step literal functions.echo error", result.Steps)
		}
		obs := result.Steps[0].Observation
		for _, hint := range []string{`"echo"`, "batch.calls[].tool", "direct tool call name"} {
			if !strings.Contains(obs, hint) {
				t.Errorf("batch observation = %q, want hint %q", obs, hint)
			}
		}
		if probe.executions != 0 {
			t.Errorf("probe executions = %d, want 0 (the hint never dispatches)", probe.executions)
		}
	})
}

func TestProcessBatchTool_SubCalls(t *testing.T) {
	mockTools := newMockToolExecutor()
	mockTools.results["search"] = tools.ToolResult{Content: "search result"}
	mockTools.results["read"] = tools.ToolResult{Content: "file content"}

	exec := newExecutorDefaultHITL(&mockLLMCaller{}, mockTools, &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
	exec.emitter = &NoopEvents{}

	subInput1, _ := json.Marshal(map[string]string{"query": "test"})
	subInput2, _ := json.Marshal(map[string]string{"path": "file.txt"})

	type batchCall struct {
		Tool  string          `json:"tool"`
		Input json.RawMessage `json:"input"`
	}
	batchInput, _ := json.Marshal(map[string][]batchCall{
		"calls": {
			{Tool: "search", Input: subInput1},
			{Tool: "read", Input: subInput2},
		},
	})

	action := llm.ToolCall{ID: "batch_1", Name: "batch", Input: batchInput}
	resp := &llm.ChatResponse{
		Message:    llm.Message{Role: "assistant", ToolCalls: []llm.ToolCall{action}},
		Usage:      llm.TokenUsage{InputTokens: 100, OutputTokens: 50},
		StopReason: "tool_use",
	}
	state := &runState{effectiveMaxSteps: 10}
	cw := newMockContextManager()

	result, act, err := exec.processSingleToolCall(context.Background(), action, 0, resp.Message.ToolCalls, resp, "thinking", state, cw)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != nil {
		t.Error("expected nil result")
	}
	if act != actionNone {
		t.Errorf("expected actionNone, got %v", act)
	}
}

func TestProcessBatchTool_NestedBatch(t *testing.T) {
	mockTools := newMockToolExecutor()

	exec := newExecutorDefaultHITL(&mockLLMCaller{}, mockTools, &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
	exec.emitter = &NoopEvents{}

	nestedInput, _ := json.Marshal(map[string]string{})

	type batchCall struct {
		Tool  string          `json:"tool"`
		Input json.RawMessage `json:"input"`
	}
	batchInput, _ := json.Marshal(map[string][]batchCall{
		"calls": {
			{Tool: "batch", Input: nestedInput},
		},
	})

	action := llm.ToolCall{ID: "batch_1", Name: "batch", Input: batchInput}
	resp := &llm.ChatResponse{
		Message:    llm.Message{Role: "assistant", ToolCalls: []llm.ToolCall{action}},
		Usage:      llm.TokenUsage{InputTokens: 100, OutputTokens: 50},
		StopReason: "tool_use",
	}
	state := &runState{effectiveMaxSteps: 10}
	cw := newMockContextManager()

	result, act, err := exec.processSingleToolCall(context.Background(), action, 0, resp.Message.ToolCalls, resp, "thinking", state, cw)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != nil {
		t.Error("expected nil result for nested batch")
	}
	if act != actionNone {
		t.Errorf("expected actionNone, got %v", act)
	}
}

// reasoningItemsFixture builds a response whose message carries one
// encrypted reasoning item — the shape every ReasoningItems carry test
// starts from.
func reasoningItemsFixture(action llm.ToolCall) *llm.ChatResponse {
	return &llm.ChatResponse{
		Message: llm.Message{
			Role: "assistant",
			ToolCalls: []llm.ToolCall{
				{ID: "call_1", Name: "batch", Input: action.Input},
				action,
			},
			ReasoningItems: []llm.ReasoningItem{
				{ID: "rs_1", Summary: "committed plan", EncryptedContent: "gAAAA-encrypted"},
			},
		},
		Usage:      llm.TokenUsage{InputTokens: 100, OutputTokens: 50},
		StopReason: "tool_use",
	}
}

// TestNonBatchPathsCarryReasoningItems pins the non-batch ReasoningItems
// carries and the one deliberate non-carry: the truncation step (below
// threshold AND both reprieve branches), the repeat-identical interception at
// callIdx 0 (and its absence at callIdx > 0), the standalone HITL-reject at
// callIdx 0, and the stop-tool guard nudge carrying NOTHING while the stop
// tool's own step carries them.
func TestNonBatchPathsCarryReasoningItems(t *testing.T) {
	wantItems := func(t *testing.T, step Step, where string) {
		t.Helper()
		if len(step.ReasoningItems) != 1 || step.ReasoningItems[0].EncryptedContent != "gAAAA-encrypted" {
			t.Errorf("%s ReasoningItems = %+v, want the response's reasoning item to ride it", where, step.ReasoningItems)
		}
	}
	noItems := func(t *testing.T, step Step, where string) {
		t.Helper()
		if len(step.ReasoningItems) != 0 {
			t.Errorf("%s ReasoningItems = %+v, want none", where, step.ReasoningItems)
		}
	}
	action := llm.ToolCall{ID: "call_1", Name: "search", Input: []byte(`{"query":"test"}`)}

	t.Run("truncation below threshold", func(t *testing.T) {
		exec := newExecutorDefaultHITL(&mockLLMCaller{}, newMockToolExecutor(), &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
		exec.emitter = &NoopEvents{}
		resp := reasoningItemsFixture(action)
		resp.StopReason = "max_tokens"
		state := &runState{effectiveMaxSteps: 10}
		cw := newMockContextManager()

		_, act := exec.handleTruncationStopReason(context.Background(), resp, "thinking", state, cw)
		if act != actionContinue {
			t.Fatalf("act = %v, want actionContinue", act)
		}
		if len(state.allSteps) != 1 {
			t.Fatalf("steps = %d, want 1", len(state.allSteps))
		}
		wantItems(t, state.allSteps[0], "truncation step")
	})

	t.Run("truncation reprieve", func(t *testing.T) {
		for _, tc := range []struct {
			name    string
			slResp  StepLimitResponse
			wantNud int
		}{
			{"allow_once", StepLimitAllowOnce, 1},
			{"allow_always", StepLimitAllowAlways, 1},
		} {
			t.Run(tc.name, func(t *testing.T) {
				cfg := defaultCircuitBreakerConfig
				cfg.TruncationAbortThreshold = 1
				exec := newExecutorDefaultHITL(&mockLLMCaller{}, newMockToolExecutor(), &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, cfg)
				exec.emitter = &NoopEvents{}
				exec.SetHITLHandler(&testStepLimitAdapter{fn: func(context.Context, int, int, string) (StepLimitResponse, error) {
					return tc.slResp, nil
				}})
				resp := reasoningItemsFixture(action)
				resp.StopReason = "max_tokens"
				state := &runState{effectiveMaxSteps: 10}
				cw := newMockContextManager()

				_, act := exec.handleTruncationStopReason(context.Background(), resp, "thinking", state, cw)
				if act != actionContinue {
					t.Fatalf("act = %v, want actionContinue (reprieve)", act)
				}
				if len(state.allSteps) != tc.wantNud {
					t.Fatalf("steps = %d, want %d", len(state.allSteps), tc.wantNud)
				}
				wantItems(t, state.allSteps[0], "truncation reprieve step")
			})
		}
	})

	t.Run("repeat-identical interception first call", func(t *testing.T) {
		cfg := defaultCircuitBreakerConfig
		cfg.RepeatNudgeThreshold = 1
		exec := newExecutorDefaultHITL(&mockLLMCaller{}, newMockToolExecutor(), &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, cfg)
		exec.emitter = &NoopEvents{}
		resp := reasoningItemsFixture(action)
		state := &runState{effectiveMaxSteps: 10}
		cw := newMockContextManager()

		_, _, err := exec.checkRepeatIdenticalTool(context.Background(), action, 0, "thinking", resp, state, cw)
		if err != nil {
			t.Fatalf("checkRepeatIdenticalTool: %v", err)
		}
		if len(state.allSteps) != 1 {
			t.Fatalf("steps = %d, want 1 (the intercepted call's step)", len(state.allSteps))
		}
		wantItems(t, state.allSteps[0], "repeat-intercept first-call step")
	})

	t.Run("repeat-identical interception later call carries none", func(t *testing.T) {
		cfg := defaultCircuitBreakerConfig
		cfg.RepeatNudgeThreshold = 1
		exec := newExecutorDefaultHITL(&mockLLMCaller{}, newMockToolExecutor(), &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, cfg)
		exec.emitter = &NoopEvents{}
		resp := reasoningItemsFixture(action)
		state := &runState{effectiveMaxSteps: 10}
		cw := newMockContextManager()

		_, _, err := exec.checkRepeatIdenticalTool(context.Background(), action, 2, "thinking", resp, state, cw)
		if err != nil {
			t.Fatalf("checkRepeatIdenticalTool: %v", err)
		}
		if len(state.allSteps) != 1 {
			t.Fatalf("steps = %d, want 1 (the intercepted call's step)", len(state.allSteps))
		}
		noItems(t, state.allSteps[0], "repeat-intercept later-call step")
	})

	t.Run("standalone HITL-reject first call", func(t *testing.T) {
		exec := NewExecutor(&mockLLMCaller{}, newMockToolExecutor(), 10,
			WithTokenCounter(&mockTokenCounter{}),
			WithCircuitBreaker(defaultCircuitBreakerConfig),
			WithHITL(&denyingHITLHandler{}),
		)
		exec.emitter = &NoopEvents{}
		resp := reasoningItemsFixture(action)
		state := &runState{effectiveMaxSteps: 10}
		cw := newMockContextManager()

		if _, _, err := exec.processSingleToolCall(context.Background(), action, 0, []llm.ToolCall{action}, resp, "thinking", state, cw); err != nil {
			t.Fatalf("processSingleToolCall: %v", err)
		}
		if len(state.allSteps) != 1 {
			t.Fatalf("steps = %d, want 1 (the rejected call's step)", len(state.allSteps))
		}
		wantItems(t, state.allSteps[0], "standalone HITL-reject first-call step")
	})

	t.Run("stop-tool guard nudge carries none while the tool step carries them", func(t *testing.T) {
		mockTools := newMockToolExecutor()
		mockTools.results["verdict"] = tools.ToolResult{Content: "done"}
		exec := newExecutorDefaultHITL(&mockLLMCaller{}, mockTools, &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
		exec.emitter = &NoopEvents{}
		exec.SetStopTools("verdict")
		exec.finishGuard = func(context.Context) error { return errors.New("pending delegations") }

		stopAction := llm.ToolCall{ID: "call_1", Name: "verdict", Input: []byte(`{"v":"ok"}`)}
		resp := reasoningItemsFixture(stopAction)
		state := &runState{effectiveMaxSteps: 10}
		cw := newMockContextManager()

		if _, _, err := exec.processSingleToolCall(context.Background(), stopAction, 0, []llm.ToolCall{stopAction}, resp, "thinking", state, cw); err != nil {
			t.Fatalf("processSingleToolCall: %v", err)
		}
		if len(state.allSteps) != 2 {
			t.Fatalf("steps = %d, want 2 (the stop tool's step + the guard nudge)", len(state.allSteps))
		}
		wantItems(t, state.allSteps[0], "stop tool step")
		noItems(t, state.allSteps[1], "stop-tool guard nudge")
	})

	// The four finish-path nudges (guard, mutation gate, both checklist
	// gates) render standalone: they carry the response's items exactly when
	// finish is the group's FIRST call (the nudge is then the response's
	// only step) and carry NONE with earlier siblings, whose groupSteps[0]
	// is already the single carrier.
	t.Run("finish-guard nudge first call carries them", func(t *testing.T) {
		exec := newExecutorDefaultHITL(&mockLLMCaller{}, newMockToolExecutor(), &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
		exec.emitter = &NoopEvents{}
		exec.finishGuard = func(context.Context) error { return errors.New("pending delegations") }

		finishAction := llm.ToolCall{ID: "call_1", Name: "finish", Input: []byte(`{"answer":"done"}`)}
		resp := reasoningItemsFixture(finishAction)
		resp.Message.ToolCalls = []llm.ToolCall{finishAction}
		state := &runState{effectiveMaxSteps: 10}
		cw := newMockContextManager()

		if _, _, err := exec.processSingleToolCall(context.Background(), finishAction, 0, []llm.ToolCall{finishAction}, resp, "thinking", state, cw); err != nil {
			t.Fatalf("processSingleToolCall: %v", err)
		}
		if len(state.allSteps) != 1 {
			t.Fatalf("steps = %d, want 1 (the guard nudge alone)", len(state.allSteps))
		}
		wantItems(t, state.allSteps[0], "finish-guard nudge (first call)")
	})

	t.Run("finish-guard nudge later call carries none", func(t *testing.T) {
		mockTools := newMockToolExecutor()
		mockTools.results["search"] = tools.ToolResult{Content: "found"}
		exec := newExecutorDefaultHITL(&mockLLMCaller{}, mockTools, &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
		exec.emitter = &NoopEvents{}
		exec.finishGuard = func(context.Context) error { return errors.New("pending delegations") }

		searchAction := llm.ToolCall{ID: "call_1", Name: "search", Input: []byte(`{"query":"test"}`)}
		finishAction := llm.ToolCall{ID: "call_2", Name: "finish", Input: []byte(`{"answer":"done"}`)}
		resp := reasoningItemsFixture(searchAction)
		resp.Message.ToolCalls = []llm.ToolCall{searchAction, finishAction}
		state := &runState{effectiveMaxSteps: 10}
		cw := newMockContextManager()

		if _, _, err := exec.processToolCalls(context.Background(), resp, "thinking", state, cw); err != nil {
			t.Fatalf("processToolCalls: %v", err)
		}
		if len(state.allSteps) != 2 {
			t.Fatalf("steps = %d, want 2 (the search step + the guard nudge)", len(state.allSteps))
		}
		wantItems(t, state.allSteps[0], "search step (group's first)")
		noItems(t, state.allSteps[1], "finish-guard nudge (later call)")
	})

	t.Run("mutation-gate nudge first call carries them, later call carries none", func(t *testing.T) {
		mockTools := newMockToolExecutor()
		mockTools.results["search"] = tools.ToolResult{Content: "found"}
		for _, tc := range []struct {
			name      string
			toolCalls []llm.ToolCall
			wantSteps int
		}{
			{"first call", []llm.ToolCall{{ID: "call_1", Name: "finish", Input: []byte(`{"answer":"done"}`)}}, 1},
			{"later call", []llm.ToolCall{
				{ID: "call_1", Name: "search", Input: []byte(`{"query":"test"}`)},
				{ID: "call_2", Name: "finish", Input: []byte(`{"answer":"done"}`)},
			}, 2},
		} {
			t.Run(tc.name, func(t *testing.T) {
				exec := newExecutorDefaultHITL(&mockLLMCaller{}, mockTools, &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
				exec.emitter = &NoopEvents{}
				exec.SetMutationRequired(true)

				resp := reasoningItemsFixture(tc.toolCalls[0])
				resp.Message.ToolCalls = tc.toolCalls
				state := &runState{effectiveMaxSteps: 10}
				cw := newMockContextManager()

				if _, _, err := exec.processToolCalls(context.Background(), resp, "thinking", state, cw); err != nil {
					t.Fatalf("processToolCalls: %v", err)
				}
				if len(state.allSteps) != tc.wantSteps {
					t.Fatalf("steps = %d, want %d", len(state.allSteps), tc.wantSteps)
				}
				wantItems(t, state.allSteps[0], "group's first step")
				if tc.wantSteps == 2 {
					noItems(t, state.allSteps[1], "mutation-gate nudge (later call)")
				}
			})
		}
	})

	t.Run("checklist-missing nudge first call carries them, later call carries none", func(t *testing.T) {
		mockTools := newMockToolExecutor()
		mockTools.results["search"] = tools.ToolResult{Content: "found"}
		for _, tc := range []struct {
			name      string
			toolCalls []llm.ToolCall
			wantSteps int
		}{
			{"first call", []llm.ToolCall{{ID: "call_1", Name: "finish", Input: []byte(`{"answer":"done"}`)}}, 1},
			{"later call", []llm.ToolCall{
				{ID: "call_1", Name: "search", Input: []byte(`{"query":"test"}`)},
				{ID: "call_2", Name: "finish", Input: []byte(`{"answer":"done"}`)},
			}, 2},
		} {
			t.Run(tc.name, func(t *testing.T) {
				exec := newExecutorDefaultHITL(&mockLLMCaller{}, mockTools, &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
				exec.emitter = &NoopEvents{}
				// checklistGateEnabled is true by default; make the gate
				// active and the step non-trivial (3 productive calls).
				seed := &runState{effectiveMaxSteps: 10, checklistAvailable: true}
				for i := 0; i < checklistTrivialThreshold+1; i++ {
					seed.allSteps = append(seed.allSteps, Step{
						Action:      llm.ToolCall{ID: fmt.Sprintf("seed_%d", i), Name: "search", Input: []byte(`{}`)},
						Observation: "found",
					})
				}

				resp := reasoningItemsFixture(tc.toolCalls[0])
				resp.Message.ToolCalls = tc.toolCalls
				state := seed
				cw := newMockContextManager()

				if _, _, err := exec.processToolCalls(context.Background(), resp, "thinking", state, cw); err != nil {
					t.Fatalf("processToolCalls: %v", err)
				}
				// Seed steps + this response's steps.
				if got := len(state.allSteps) - (checklistTrivialThreshold + 1); got != tc.wantSteps {
					t.Fatalf("steps this response = %d, want %d", got, tc.wantSteps)
				}
				wantItems(t, state.allSteps[checklistTrivialThreshold+1], "group's first step")
				if tc.wantSteps == 2 {
					noItems(t, state.allSteps[checklistTrivialThreshold+2], "checklist-missing nudge (later call)")
				}
			})
		}
	})

	t.Run("checklist-unchecked nudge first call carries them, later call carries none", func(t *testing.T) {
		mockTools := newMockToolExecutor()
		mockTools.results["search"] = tools.ToolResult{Content: "found"}
		for _, tc := range []struct {
			name      string
			toolCalls []llm.ToolCall
			wantSteps int
		}{
			{"first call", []llm.ToolCall{{ID: "call_1", Name: "finish", Input: []byte(`{"answer":"done"}`)}}, 1},
			{"later call", []llm.ToolCall{
				{ID: "call_1", Name: "search", Input: []byte(`{"query":"test"}`)},
				{ID: "call_2", Name: "finish", Input: []byte(`{"answer":"done"}`)},
			}, 2},
		} {
			t.Run(tc.name, func(t *testing.T) {
				exec := newExecutorDefaultHITL(&mockLLMCaller{}, mockTools, &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
				exec.emitter = &NoopEvents{}
				// A last checklist with unchecked items (0/2 done) plus the
				// productive-call seed that makes Enf-1 pass so only Enf-2
				// fires.
				seed := &runState{effectiveMaxSteps: 10, checklistAvailable: true}
				seed.allSteps = append(seed.allSteps, Step{
					Action:      llm.ToolCall{ID: "seed_cl", Name: "update_checklist", Input: []byte(`{}`)},
					Observation: "0/2 done",
				})
				for i := 0; i < checklistTrivialThreshold+1; i++ {
					seed.allSteps = append(seed.allSteps, Step{
						Action:      llm.ToolCall{ID: fmt.Sprintf("seed_%d", i), Name: "search", Input: []byte(`{}`)},
						Observation: "found",
					})
				}

				resp := reasoningItemsFixture(tc.toolCalls[0])
				resp.Message.ToolCalls = tc.toolCalls
				state := seed
				cw := newMockContextManager()

				if _, _, err := exec.processToolCalls(context.Background(), resp, "thinking", state, cw); err != nil {
					t.Fatalf("processToolCalls: %v", err)
				}
				seedLen := 1 + checklistTrivialThreshold + 1
				if got := len(state.allSteps) - seedLen; got != tc.wantSteps {
					t.Fatalf("steps this response = %d, want %d", got, tc.wantSteps)
				}
				wantItems(t, state.allSteps[seedLen], "group's first step")
				if tc.wantSteps == 2 {
					noItems(t, state.allSteps[seedLen+1], "checklist-unchecked nudge (later call)")
				}
			})
		}
	})
}

// --- Batch pre-flight, whole-batch repeat detection, validation breaker ---

// invalidValidationFor builds the *tools.InputValidationError a pre-flight
// rejection carries, digest included — the shape a real registry rejection
// renders (see the schema digest tests in the tools package).
func invalidValidationFor(tool, reason string) error {
	return &tools.InputValidationError{
		Tool:         tool,
		Reason:       reason,
		ValidParams:  []string{"path"},
		SchemaDigest: `Required parameters: path (string). Minimal example: {"path":"<string>"}`,
	}
}

// bc is one batch sub-call fixture entry.
type bc struct {
	tool  string
	input map[string]any
}

// batchCallJSON marshals a batch meta-call input from sub-call fixtures.
func batchCallJSON(t *testing.T, calls ...bc) json.RawMessage {
	t.Helper()
	type batchCall struct {
		Tool  string          `json:"tool"`
		Input json.RawMessage `json:"input"`
	}
	arr := make([]batchCall, 0, len(calls))
	for _, c := range calls {
		in, err := json.Marshal(c.input)
		if err != nil {
			t.Fatalf("marshal sub-call input: %v", err)
		}
		arr = append(arr, batchCall{Tool: c.tool, Input: in})
	}
	b, err := json.Marshal(map[string][]batchCall{"calls": arr})
	if err != nil {
		t.Fatalf("marshal batch input: %v", err)
	}
	return b
}

// batchAction builds the batch meta-call and its response wrapper.
func batchAction(id string, input json.RawMessage) (llm.ToolCall, *llm.ChatResponse) {
	action := llm.ToolCall{ID: id, Name: "batch", Input: input}
	resp := &llm.ChatResponse{
		Message:    llm.Message{Role: "assistant", ToolCalls: []llm.ToolCall{action}},
		Usage:      llm.TokenUsage{InputTokens: 100, OutputTokens: 50},
		StopReason: "tool_use",
	}
	return action, resp
}

// TestProcessBatchTool_PreflightRejectsWholeBatch pins the pre-flight
// contract: a batch carrying TWO invalid sub-calls executes NOTHING and
// returns ONE aggregated rejection listing every offending index with its
// reason and the schema digest (required parameters, types, minimal
// example). The valid prefix must not run.
func TestProcessBatchTool_PreflightRejectsWholeBatch(t *testing.T) {
	mockTools := newMockToolExecutor()
	mockTools.results["good_tool"] = tools.ToolResult{Content: "good result"}
	mockTools.validationFailures["bad_a"] = invalidValidationFor("bad_a", `missing required parameter "path"`)
	mockTools.validationFailures["bad_b"] = invalidValidationFor("bad_b", `parameter "path" must be of type string, got number`)

	exec := newExecutorDefaultHITL(&mockLLMCaller{}, mockTools, &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
	exec.emitter = &NoopEvents{}

	batchInput := batchCallJSON(t,
		bc{tool: "good_tool", input: map[string]any{"query": "x"}},
		bc{tool: "bad_a", input: map[string]any{}},
		bc{tool: "good_tool", input: map[string]any{"query": "y"}},
		bc{tool: "bad_b", input: map[string]any{"path": 42}},
	)
	action, resp := batchAction("batch_1", batchInput)
	state := &runState{effectiveMaxSteps: 10}
	cw := newMockContextManager()

	result, act, err := exec.processSingleToolCall(context.Background(), action, 0, resp.Message.ToolCalls, resp, "thinking", state, cw)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != nil || act != actionNone {
		t.Fatalf("rejection must return to the loop: result=%v act=%v", result, act)
	}

	// NOTHING executed — the valid sub-calls included.
	if len(mockTools.calls) != 0 {
		t.Fatalf("pre-flight rejection must execute nothing, got %d executed calls: %+v", len(mockTools.calls), mockTools.calls)
	}

	// ONE aggregated step, marked as an error.
	if len(state.allSteps) != 1 {
		t.Fatalf("expected exactly 1 rejection step, got %d", len(state.allSteps))
	}
	step := state.allSteps[0]
	if !step.IsError {
		t.Error("rejection step must be IsError")
	}
	obs := step.Observation
	for _, want := range []string{
		"batch rejected",
		"no sub-call was executed",
		`calls[1] ("bad_a")`,
		`calls[3] ("bad_b")`,
		`missing required parameter "path"`,
		`parameter "path" must be of type string, got number`,
		"Required parameters: path (string)",
		`Minimal example: {"path":"<string>"}`,
		// Each rejection carries a truncated verbatim excerpt of the model's
		// own wrong args so it can diff them against the schema digest.
		"your input: {}",
		`your input: {"path":42}`,
	} {
		if !strings.Contains(obs, want) {
			t.Errorf("aggregate rejection must contain %q:\n%s", want, obs)
		}
	}

	// The rejection armed the validation breaker exactly once.
	if exec.consecutiveValidationErrorCount != 1 {
		t.Errorf("consecutiveValidationErrorCount = %d, want 1", exec.consecutiveValidationErrorCount)
	}
}

// TestProcessBatchTool_PreflightRejectionExcerptTruncatesOversizedInput pins
// the excerpt side of the rejection render with a DeepSeek-shaped payload
// from the 11-10 burst: a multi-line shell command emitted as the sub-call
// input, far beyond the excerpt budget. The model must see its own raw args
// (index + tool + its input) with the cut marked, never the whole payload.
func TestProcessBatchTool_PreflightRejectionExcerptTruncatesOversizedInput(t *testing.T) {
	mockTools := newMockToolExecutor()
	mockTools.validationFailures["bash"] = invalidValidationFor("bash", `parameter "timeout_s" must be of type number, got string`)

	exec := newExecutorDefaultHITL(&mockLLMCaller{}, mockTools, &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
	exec.emitter = &NoopEvents{}

	// DeepSeek-shaped oversized payload: a multi-line shell command (newlines
	// JSON-escaped — the batch envelope must stay valid JSON to reach
	// pre-flight) that also exceeds the excerpt budget.
	big := strings.Repeat(`echo line of output\n`, 60) // 1260 bytes as JSON
	rawInput := `{"command": "` + big + `"}`
	batchInput := json.RawMessage(`{"calls":[{"tool":"bash","input":` + rawInput + `}]}`)
	action, resp := batchAction("batch_1", batchInput)
	state := &runState{effectiveMaxSteps: 10}
	cw := newMockContextManager()

	if _, _, err := exec.processSingleToolCall(context.Background(), action, 0, resp.Message.ToolCalls, resp, "thinking", state, cw); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(state.allSteps) != 1 {
		t.Fatalf("expected exactly 1 rejection step, got %d", len(state.allSteps))
	}
	obs := state.allSteps[0].Observation
	for _, want := range []string{
		`calls[0] ("bash")`,
		`parameter "timeout_s" must be of type number, got string`,
		`your input: {"command": "echo line of output`,
		"...(truncated)",
	} {
		if !strings.Contains(obs, want) {
			t.Errorf("rejection must contain %q:\n%s", want, obs)
		}
	}
	// The excerpt must be a PREFIX of the payload, not the whole thing.
	if strings.Contains(obs, big) {
		t.Error("rejection embedded the whole oversized payload; excerpt must be truncated")
	}
	if n := strings.Count(obs, "echo line of output"); n >= 60 {
		t.Errorf("excerpt carried %d of 60 payload lines; truncation did not fire", n)
	}
}

// TestTruncateBatchInputExcerpt pins the excerpt bounds: short payloads pass
// through verbatim, long ASCII payloads cut at the budget with the marker,
// and a cut landing inside a multi-byte rune backs off to the rune boundary.
func TestTruncateBatchInputExcerpt(t *testing.T) {
	if got := truncateBatchInputExcerpt(json.RawMessage(`{"path":"x"}`)); got != `{"path":"x"}` {
		t.Errorf("short excerpt = %q, want verbatim", got)
	}
	long := json.RawMessage(`{"command":"` + strings.Repeat("a", 1000) + `"}`)
	got := truncateBatchInputExcerpt(long)
	if len(got) != maxBatchRejectionExcerpt+len("...(truncated)") {
		t.Errorf("excerpt len = %d, want %d (budget + marker)", len(got), maxBatchRejectionExcerpt+len("...(truncated)"))
	}
	if !strings.HasSuffix(got, "...(truncated)") {
		t.Errorf("excerpt = %q, want the truncation marker suffix", got)
	}
	// Byte 600 of the payload below lands inside the é rune (bytes 599-600);
	// the cut must back off to the rune start so the excerpt stays valid UTF-8.
	multi := json.RawMessage(`{"note":"` + strings.Repeat("x", 590) + "é" + strings.Repeat("y", 40) + `"}`)
	got = truncateBatchInputExcerpt(multi)
	if !utf8.ValidString(got) {
		t.Errorf("excerpt %q is not valid UTF-8; cut must back off to a rune boundary", got)
	}
	if want := `{"note":"` + strings.Repeat("x", 590) + "...(truncated)"; got != want {
		t.Errorf("excerpt = %q, want the rune-straddling cut backed off to %q", got, want)
	}
}

// TestProcessBatchTool_IdenticalInvalidBatchesNudgeThenAbort pins the
// whole-batch repeat ladder with lowered thresholds (nudge 3 / abort 5): the
// THIRD identical invalid batch is nudged (and pre-flight does not even run),
// the FIFTH hits the abort path — and not a single sub-call ever executes.
func TestProcessBatchTool_IdenticalInvalidBatchesNudgeThenAbort(t *testing.T) {
	mockTools := newMockToolExecutor()
	mockTools.validationFailures["bad_a"] = invalidValidationFor("bad_a", `missing required parameter "path"`)

	cfg := CircuitBreakerConfig{
		RepeatNudgeThreshold: 3,
		RepeatAbortThreshold: 5,
		// Validation breaker disabled to isolate the whole-batch repeat ladder.
		ValidationErrorAbortThreshold: 0,
	}
	exec := newExecutorDefaultHITL(&mockLLMCaller{}, mockTools, &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, cfg)
	exec.emitter = &NoopEvents{}

	batchInput := batchCallJSON(t, bc{tool: "bad_a", input: map[string]any{}})
	action, resp := batchAction("batch_1", batchInput)
	state := &runState{effectiveMaxSteps: 10}
	cw := newMockContextManager()
	run := func() (result *ExecutorResult, act loopAction) {
		t.Helper()
		// Mirror the per-wave reset the Run loop performs before each
		// processSingleToolCall wave (response-group head): direct calls
		// bypass it, and a stale flag would turn the batch-5 abort arm into
		// a phantom reprieve.
		state.circuitBreakerTriggered = false
		var err error
		result, act, err = exec.processSingleToolCall(context.Background(), action, 0, resp.Message.ToolCalls, resp, "thinking", state, cw)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		return result, act
	}

	// Batches 1-2: pre-flight rejection, no breaker interference yet.
	for i := 0; i < 2; i++ {
		if result, act := run(); result != nil || act != actionNone {
			t.Fatalf("batch %d must pass through the aggregate rejection: result=%v act=%v", i+1, result, act)
		}
	}
	if exec.batchRepeat.count != 2 {
		t.Fatalf("batchRepeat.count = %d after two identical batches, want 2", exec.batchRepeat.count)
	}
	if exec.consecutiveValidationErrorCount != 2 {
		t.Fatalf("validation count = %d after two rejections, want 2", exec.consecutiveValidationErrorCount)
	}

	// Batch 3: the whole-batch repeat nudge intercepts BEFORE pre-flight.
	result, act := run()
	if result != nil || act != actionBreak || !state.circuitBreakerTriggered {
		t.Fatalf("third identical batch must trigger the nudge: result=%v act=%v triggered=%v", result, act, state.circuitBreakerTriggered)
	}
	nudgeStep := state.allSteps[len(state.allSteps)-1]
	if nudgeStep.Observation != repeatNudgeMessage {
		t.Errorf("nudge observation = %q, want the generic repeat nudge (the rejection was not an executed-call error, so thresholds stay unlowered)", nudgeStep.Observation)
	}
	// The nudge fired before pre-flight: the validation count froze at 2 and
	// the repeat count was not reset.
	if exec.consecutiveValidationErrorCount != 2 {
		t.Errorf("validation count = %d after nudged batch 3, want 2 (pre-flight must not run)", exec.consecutiveValidationErrorCount)
	}
	if exec.batchRepeat.count != 3 {
		t.Errorf("batchRepeat.count = %d after the nudge, want 3 (nudges do not reset)", exec.batchRepeat.count)
	}

	// Batch 4: nudge again (count 4 ≥ 3).
	if result, act := run(); result != nil || act != actionBreak {
		t.Fatalf("fourth identical batch must nudge again: result=%v act=%v", result, act)
	}

	// Batch 5: abort path (NoopHITL denies the reprieve).
	result, act = run()
	if act != actionNone || result == nil {
		t.Fatalf("fifth identical batch must abort: result=%v act=%v", result, act)
	}
	if result.Finished {
		t.Error("repeat abort is an unfinished stop, Finished must be false")
	}
	if !strings.Contains(result.Output, "Aborted: tool 'batch' called 5 times consecutively with identical arguments") {
		t.Errorf("abort output mismatch: %q", result.Output)
	}

	// Not a single sub-call ever executed across the whole ladder.
	if len(mockTools.calls) != 0 {
		t.Errorf("no sub-call may execute across the nudge/abort ladder, got %+v", mockTools.calls)
	}
}

// TestProcessBatchTool_IdenticalWaitPollBatchesExemptFromRepeatBreaker pins
// the whole-batch extension of the repeat exemption: a WHOLE batch whose
// every sub-call is a wait poller, re-issued byte-identically (the
// multi-delegation join poll), never accumulates in the batch namespace —
// the mock's short "ok" sub-call results would otherwise look like an
// identical resend and nudge at 3 / abort at 5. A batch mixing a wait
// poller with a non-exempt tool stays breaker-armed: identical re-issues
// still accumulate and the third trips the nudge.
func TestProcessBatchTool_IdenticalWaitPollBatchesExemptFromRepeatBreaker(t *testing.T) {
	mockTools := newMockToolExecutor()
	mockTools.validationFailures["bad_a"] = invalidValidationFor("bad_a", `missing required parameter "path"`)

	cfg := CircuitBreakerConfig{
		RepeatNudgeThreshold: 3,
		RepeatAbortThreshold: 5,
		// Validation breaker disabled to isolate the repeat exemption.
		ValidationErrorAbortThreshold: 0,
	}
	exec := newExecutorDefaultHITL(&mockLLMCaller{}, mockTools, &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, cfg)
	exec.emitter = &NoopEvents{}
	state := &runState{effectiveMaxSteps: 10}
	cw := newMockContextManager()

	// The join poll: two wait_for_one sub-calls, identical batch input,
	// re-issued past both thresholds.
	pollInput := batchCallJSON(t,
		bc{tool: "wait_for_one", input: map[string]any{"delegate_id": "del_1"}},
		bc{tool: "wait_for_one", input: map[string]any{"delegate_id": "del_2"}},
	)
	pollAction, pollResp := batchAction("batch_poll", pollInput)
	for i := 0; i < 6; i++ {
		state.circuitBreakerTriggered = false
		if _, act, err := exec.processSingleToolCall(context.Background(), pollAction, 0, pollResp.Message.ToolCalls, pollResp, "thinking", state, cw); err != nil || act != actionNone || state.circuitBreakerTriggered {
			t.Fatalf("poll batch %d tripped the repeat breaker: act=%v triggered=%v err=%v (a healthy join poll must never nudge/abort)", i+1, act, state.circuitBreakerTriggered, err)
		}
		if exec.batchRepeat.count != 0 {
			t.Fatalf("batchRepeat.count = %d after poll batch %d, want 0 (a whole-batch wait poll is exempt)", exec.batchRepeat.count, i+1)
		}
	}

	// Control: a batch mixing the poller with a non-exempt tool stays armed —
	// identical re-issues accumulate and the third trips the nudge.
	mixedInput := batchCallJSON(t,
		bc{tool: "wait_for_one", input: map[string]any{"delegate_id": "del_1"}},
		bc{tool: "bad_a", input: map[string]any{}},
	)
	mixedAction, mixedResp := batchAction("batch_mixed", mixedInput)
	for i := 0; i < 2; i++ {
		state.circuitBreakerTriggered = false
		if _, act, err := exec.processSingleToolCall(context.Background(), mixedAction, 0, mixedResp.Message.ToolCalls, mixedResp, "thinking", state, cw); err != nil || act != actionNone {
			t.Fatalf("mixed batch %d must pass through the aggregate rejection: act=%v err=%v", i+1, act, err)
		}
		if exec.batchRepeat.count != i+1 {
			t.Fatalf("batchRepeat.count = %d after mixed batch %d, want %d", exec.batchRepeat.count, i+1, i+1)
		}
	}
	state.circuitBreakerTriggered = false
	if _, act, err := exec.processSingleToolCall(context.Background(), mixedAction, 0, mixedResp.Message.ToolCalls, mixedResp, "thinking", state, cw); err != nil || act != actionBreak || !state.circuitBreakerTriggered {
		t.Fatalf("third identical mixed batch must trip the repeat nudge: act=%v triggered=%v err=%v", act, state.circuitBreakerTriggered, err)
	}
}

// TestValidationBreaker_DifferentInvalidBatchesAbort pins the validation
// circuit breaker on batches that are all DIFFERENT (so the whole-batch
// repeat breaker never accumulates): the third consecutive schema rejection
// trips the abort path with lowered threshold 3.
func TestValidationBreaker_DifferentInvalidBatchesAbort(t *testing.T) {
	mockTools := newMockToolExecutor()
	mockTools.validationFailures["bad_a"] = invalidValidationFor("bad_a", `missing required parameter "path"`)

	cfg := CircuitBreakerConfig{
		// Repeat detection disabled to isolate the validation breaker.
		RepeatNudgeThreshold:          0,
		RepeatAbortThreshold:          0,
		ValidationErrorAbortThreshold: 3,
	}
	exec := newExecutorDefaultHITL(&mockLLMCaller{}, mockTools, &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, cfg)
	exec.emitter = &NoopEvents{}
	state := &runState{effectiveMaxSteps: 10}
	cw := newMockContextManager()

	// Three batches with DIFFERENT inputs (different dedup keys).
	for i, query := range []string{"v1", "v2", "v3"} {
		batchInput := batchCallJSON(t, bc{tool: "bad_a", input: map[string]any{"q": query}})
		action, resp := batchAction("batch_1", batchInput)
		result, act, err := exec.processSingleToolCall(context.Background(), action, 0, resp.Message.ToolCalls, resp, "thinking", state, cw)
		if err != nil {
			t.Fatalf("batch %d: unexpected error: %v", i+1, err)
		}
		if i < 2 {
			if result != nil || act != actionNone {
				t.Fatalf("batch %d must pass through the rejection: result=%v act=%v", i+1, result, act)
			}
			continue
		}
		// Third consecutive validation failure: abort.
		if act != actionNone || result == nil {
			t.Fatalf("third invalid batch must abort: result=%v act=%v", result, act)
		}
		if result.Finished {
			t.Error("validation abort is an unfinished stop, Finished must be false")
		}
		if !strings.Contains(result.Output, "Aborted: tool inputs failed schema validation 3 times consecutively") {
			t.Errorf("abort output mismatch: %q", result.Output)
		}
	}
	if len(mockTools.calls) != 0 {
		t.Errorf("nothing may execute across rejected batches, got %+v", mockTools.calls)
	}
}

// TestValidationBreaker_OnStepLimitReprieve pins the reprieve arm of the
// validation ladder: StepLimitAllowOnce resets the count, appends the
// acknowledgment nudge and breaks the loop without aborting. Driven through
// checkValidationErrors directly — the single-call pipeline reaches a
// registry rejection only through a real ToolRegistry, which this breaker
// unit does not need.
func TestValidationBreaker_OnStepLimitReprieve(t *testing.T) {
	exec := newExecutorDefaultHITL(&mockLLMCaller{}, newMockToolExecutor(), &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, CircuitBreakerConfig{ValidationErrorAbortThreshold: 3})
	exec.emitter = &NoopEvents{}
	exec.SetHITLHandler(&testStepLimitAdapter{fn: func(ctx context.Context, stepNum, effectiveMaxSteps int, reason string) (StepLimitResponse, error) {
		if !strings.Contains(reason, "schema validation") {
			t.Errorf("reprieve reason must name the validation failure: %q", reason)
		}
		return StepLimitAllowOnce, nil
	}})
	state := &runState{effectiveMaxSteps: 10}
	cw := newMockContextManager()
	action := llm.ToolCall{ID: "call_1", Name: "bad_a", Input: json.RawMessage(`{}`)}
	rejection := func(count int) (string, tools.ToolResult) {
		obs := fmt.Sprintf("input for %q is invalid: missing required parameter \"path\"; valid parameters: path. Required parameters: path (string). Minimal example: {\"path\":\"<string>\"}", action.Name)
		return obs, tools.ToolResult{Content: obs, IsError: true}
	}

	// Two rejections below the threshold: counted, nudged, pass through.
	for i := 1; i <= 2; i++ {
		obs, result := rejection(i)
		got, act, execResult, err := exec.checkValidationErrors(context.Background(), action, 0, obs, result, state, cw)
		if err != nil || execResult != nil || act != actionNone {
			t.Fatalf("rejection %d must pass through: act=%v execResult=%v err=%v", i, act, execResult, err)
		}
		if !strings.Contains(got, "rejected by schema validation "+strconv.Itoa(i)) {
			t.Errorf("rejection %d must append the nudge with the count: %q", i, got)
		}
	}

	// Third rejection trips the threshold; the user grants ONE more chance.
	obs, result := rejection(3)
	_, act, execResult, err := exec.checkValidationErrors(context.Background(), action, 0, obs, result, state, cw)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if execResult != nil || act != actionBreak || !state.circuitBreakerTriggered {
		t.Fatalf("reprieve must break the loop without aborting: act=%v execResult=%v triggered=%v", act, execResult, state.circuitBreakerTriggered)
	}
	if exec.consecutiveValidationErrorCount != 0 {
		t.Errorf("count = %d after the reprieve, want 0 (reset)", exec.consecutiveValidationErrorCount)
	}
	last := state.allSteps[len(state.allSteps)-1]
	if !strings.Contains(last.UserNudge, "validation circuit breaker") || !strings.Contains(last.UserNudge, "ONE more chance") {
		t.Errorf("reprieve nudge mismatch: %q", last.UserNudge)
	}

	// A subsequent success resets the tracker (covered here for the direct
	// contract): a non-error result clears the count.
	if _, _, _, err := exec.checkValidationErrors(context.Background(), action, 0, "ok result", tools.ToolResult{Content: "ok"}, state, cw); err != nil {
		t.Fatalf("unexpected error on success: %v", err)
	}
	if exec.consecutiveValidationErrorCount != 0 {
		t.Errorf("count = %d after a success, want 0", exec.consecutiveValidationErrorCount)
	}
}
