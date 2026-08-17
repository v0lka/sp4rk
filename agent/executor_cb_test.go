package agent

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"strings"
	"testing"

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
	// returns "successfully edited file" = 24 bytes). None should increment
	// the fruitless counter, even when called many times in a row.
	exemptTools := []string{
		"edit_file", "write_file", "create_directory",
		"delete_file", "delete_directory",
		"update_checklist", "set_step_status", "store_fact",
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
	// returning "successfully edited file" = 24 bytes). They must not trigger
	// the same-tool-repeat detector.
	exemptTools := []string{
		"edit_file", "write_file", "create_directory",
		"delete_file", "delete_directory",
		"update_checklist", "set_step_status", "store_fact",
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
	if exec.consecutiveRepeatCount != 1 {
		t.Fatalf("consecutiveRepeatCount = %d after first call, want 1", exec.consecutiveRepeatCount)
	}

	// The first call executed and FAILED — mirror the executor's post-result state.
	exec.lastToolResultIsError = true

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
