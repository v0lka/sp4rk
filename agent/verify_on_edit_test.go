package agent

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync/atomic"
	"testing"
	"unicode/utf8"

	"github.com/v0lka/sp4rk/llm"
	"github.com/v0lka/sp4rk/tools"
)

// newVerifyTestExecutor builds an executor whose LLM script edits files and
// finishes, with a counting verify-on-edit runner installed.
func newVerifyTestExecutor(t *testing.T, responses []*llm.ChatResponse, runner EditVerifyRunner, maxChars int) (*Executor, *mockToolExecutor, *int32) {
	t.Helper()
	mockLLM := &mockLLMCaller{responses: responses}
	mockTools := newMockToolExecutor()
	mockTools.results["write_file"] = tools.ToolResult{Content: "written"}
	mockTools.results["edit_file"] = tools.ToolResult{Content: "edited"}
	mockTools.results["read_file"] = tools.ToolResult{Content: "content"}
	exec := newExecutorDefaultHITL(mockLLM, mockTools, &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
	var calls int32
	if runner != nil {
		inner := runner
		runner = func(ctx context.Context) EditVerifyResult {
			atomic.AddInt32(&calls, 1)
			return inner(ctx)
		}
		exec.SetVerifyOnEdit(runner, maxChars)
	}
	return exec, mockTools, &calls
}

func editInput() json.RawMessage { return json.RawMessage(`{"path":"/a.txt","content":"x"}`) }

// TestExecutor_VerifyOnEdit_SuccessfulEdit_RunsCommandAndAppendsOutput: a
// successful edit_file (enabled) runs the configured command and its
// truncated output lands in the step observation.
func TestExecutor_VerifyOnEdit_SuccessfulEdit_RunsCommandAndAppendsOutput(t *testing.T) {
	exec, _, calls := newVerifyTestExecutor(t, []*llm.ChatResponse{
		llmResponseWithToolCall("editing", "edit_file", editInput()),
		llmResponseFinish("done", "finished"),
	}, func(context.Context) EditVerifyResult {
		return EditVerifyResult{Output: "PASS: 3 tests ran", ExitCode: 0}
	}, 0)

	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "edit_file", Description: "edit", Source: "core"},
	}, newMockContextManager())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if *calls != 1 {
		t.Errorf("verify runner calls = %d, want 1", *calls)
	}
	if len(result.Steps) < 1 {
		t.Fatalf("expected at least 1 step")
	}
	obs := result.Steps[0].Observation
	if !strings.Contains(obs, "edited") {
		t.Errorf("observation should keep tool result, got %q", obs)
	}
	if !strings.Contains(obs, "[verify_on_edit]") || !strings.Contains(obs, "PASS: 3 tests ran") {
		t.Errorf("observation should carry verify note, got %q", obs)
	}
}

// TestExecutor_VerifyOnEdit_Disabled_IdenticalToCurrent: with no runner
// installed (default), a successful edit observation is unchanged.
func TestExecutor_VerifyOnEdit_Disabled_IdenticalToCurrent(t *testing.T) {
	exec, _, calls := newVerifyTestExecutor(t, []*llm.ChatResponse{
		llmResponseWithToolCall("editing", "edit_file", editInput()),
		llmResponseFinish("done", "finished"),
	}, nil, 0)

	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "edit_file", Description: "edit", Source: "core"},
	}, newMockContextManager())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if *calls != 0 {
		t.Errorf("verify runner calls = %d, want 0", *calls)
	}
	if got := result.Steps[0].Observation; got != "edited" {
		t.Errorf("observation = %q, want plain tool result %q", got, "edited")
	}
}

// TestExecutor_VerifyOnEdit_Debounce_BatchRunsOnce: multiple edits in one
// response group trigger a single verification run.
func TestExecutor_VerifyOnEdit_Debounce_BatchRunsOnce(t *testing.T) {
	exec, _, calls := newVerifyTestExecutor(t, []*llm.ChatResponse{
		llmResponseWithMultipleToolCalls("editing both", []llm.ToolCall{
			{ID: "call_a", Name: "edit_file", Input: editInput()},
			{ID: "call_b", Name: "write_file", Input: editInput()},
		}),
		llmResponseFinish("done", "finished"),
	}, func(context.Context) EditVerifyResult {
		return EditVerifyResult{Output: "ok", ExitCode: 0}
	}, 0)

	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "edit_file", Description: "edit", Source: "core"},
		{Name: "write_file", Description: "write", Source: "core"},
	}, newMockContextManager())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if *calls != 1 {
		t.Errorf("verify runner calls = %d, want 1 (debounced per group)", *calls)
	}
	// The note rides on the LAST call of the group (write_file), not on
	// the earlier edit_file step.
	var editObs, writeObs string
	for _, s := range result.Steps {
		switch s.Action.Name {
		case "edit_file":
			editObs = s.Observation
		case "write_file":
			writeObs = s.Observation
		}
	}
	if !strings.Contains(writeObs, "[verify_on_edit]") {
		t.Errorf("write_file (last in group) observation should carry verify note, got %q", writeObs)
	}
	if strings.Contains(editObs, "[verify_on_edit]") {
		t.Errorf("edit_file (not last) observation should not carry verify note, got %q", editObs)
	}
}

// TestExecutor_VerifyOnEdit_NoEditNoRun: read-only groups never run the
// verification command (debounce: no new edits — no re-run).
func TestExecutor_VerifyOnEdit_NoEditNoRun(t *testing.T) {
	exec, _, calls := newVerifyTestExecutor(t, []*llm.ChatResponse{
		llmResponseWithToolCall("reading", "read_file", json.RawMessage(`{"path":"/a"}`)),
		llmResponseWithToolCall("reading again", "read_file", json.RawMessage(`{"path":"/b"}`)),
		llmResponseFinish("done", "finished"),
	}, func(context.Context) EditVerifyResult {
		return EditVerifyResult{Output: "ok", ExitCode: 0}
	}, 0)

	_, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "read_file", Description: "read", Source: "core"},
	}, newMockContextManager())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if *calls != 0 {
		t.Errorf("verify runner calls = %d, want 0 (no edits)", *calls)
	}
}

// TestExecutor_VerifyOnEdit_NonZeroExit_MarksVerificationFailed
// verifies that a failing command marks the observation as failed.
func TestExecutor_VerifyOnEdit_NonZeroExit_MarksVerificationFailed(t *testing.T) {
	exec, _, _ := newVerifyTestExecutor(t, []*llm.ChatResponse{
		llmResponseWithToolCall("editing", "edit_file", editInput()),
		llmResponseFinish("done", "finished"),
	}, func(context.Context) EditVerifyResult {
		return EditVerifyResult{Output: "FAIL main_test.go:12", ExitCode: 1}
	}, 0)

	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "edit_file", Description: "edit", Source: "core"},
	}, newMockContextManager())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	obs := result.Steps[0].Observation
	if !strings.Contains(obs, "VERIFICATION FAILED (exit 1)") {
		t.Errorf("observation should mark verification failure, got %q", obs)
	}
	if !strings.Contains(obs, "FAIL main_test.go:12") {
		t.Errorf("observation should include command output, got %q", obs)
	}
}

// TestExecutor_VerifyOnEdit_Timeout_ClearMessage verifies that a timed-out
// command produces an understandable timeout observation.
func TestExecutor_VerifyOnEdit_Timeout_ClearMessage(t *testing.T) {
	exec, _, _ := newVerifyTestExecutor(t, []*llm.ChatResponse{
		llmResponseWithToolCall("editing", "edit_file", editInput()),
		llmResponseFinish("done", "finished"),
	}, func(context.Context) EditVerifyResult {
		return EditVerifyResult{Output: "partial...", TimedOut: true, ExitCode: -1}
	}, 0)

	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "edit_file", Description: "edit", Source: "core"},
	}, newMockContextManager())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	obs := result.Steps[0].Observation
	if !strings.Contains(obs, "timed out") {
		t.Errorf("observation should explain the timeout, got %q", obs)
	}
	if !strings.Contains(obs, "NOT verified") {
		t.Errorf("observation should state the edit was not verified, got %q", obs)
	}
}

// TestExecutor_VerifyOnEdit_OutputTruncatedToCap verifies the injected
// observation is capped at maxOutputChars with a truncation marker.
func TestExecutor_VerifyOnEdit_OutputTruncatedToCap(t *testing.T) {
	exec, _, _ := newVerifyTestExecutor(t, []*llm.ChatResponse{
		llmResponseWithToolCall("editing", "edit_file", editInput()),
		llmResponseFinish("done", "finished"),
	}, func(context.Context) EditVerifyResult {
		return EditVerifyResult{Output: strings.Repeat("x", 10_000), ExitCode: 0}
	}, 100)

	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "edit_file", Description: "edit", Source: "core"},
	}, newMockContextManager())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	obs := result.Steps[0].Observation
	if !strings.Contains(obs, "[...truncated ") {
		t.Errorf("observation should carry truncation marker, got %d chars", len(obs))
	}
	if len(obs) > 400 {
		t.Errorf("observation too long after truncation: %d", len(obs))
	}
}

// TestExecutor_VerifyOnEdit_FinishAfterEdit_StillVerifies: finish in the
// same group as an edit still runs the pending verification; the note is
// attached to the finish output.
func TestExecutor_VerifyOnEdit_FinishAfterEdit_StillVerifies(t *testing.T) {
	exec, _, calls := newVerifyTestExecutor(t, []*llm.ChatResponse{
		llmResponseWithMultipleToolCalls("edit then finish", []llm.ToolCall{
			{ID: "call_a", Name: "edit_file", Input: editInput()},
			{ID: "call_b", Name: "finish", Input: json.RawMessage(`{"answer":"done"}`)},
		}),
	}, func(context.Context) EditVerifyResult {
		return EditVerifyResult{Output: "PASS", ExitCode: 0}
	}, 0)

	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "edit_file", Description: "edit", Source: "core"},
	}, newMockContextManager())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if *calls != 1 {
		t.Errorf("verify runner calls = %d, want 1", *calls)
	}
	if !strings.Contains(result.Output, "[verify_on_edit]") {
		t.Errorf("finish output should carry verify note, got %q", result.Output)
	}
}

// rejectingLastCallHITL allows every tool call except the named one, which
// it rejects — reproducing a final sibling intercepted before normal tool
// execution.
type rejectingLastCallHITL struct {
	NoopHITLHandler
	reject string
}

func (h *rejectingLastCallHITL) OnToolCall(_ context.Context, toolName string, _ json.RawMessage) (*HITLToolDecision, error) {
	if toolName == h.reject {
		return &HITLToolDecision{Allow: false, Reason: "not allowed here"}, nil
	}
	return nil, nil
}

// TestExecutor_VerifyOnEdit_RejectedLastSiblingFlushesBeforePause verifies the
// response-group boundary directly: a successful edit followed by a rejected
// final sibling must run verification before the next boundary can pause the
// executor. The note is persisted on the rejected sibling in the checkpoint.
func TestExecutor_VerifyOnEdit_RejectedLastSiblingFlushesBeforePause(t *testing.T) {
	exec, _, calls := newVerifyTestExecutor(t, []*llm.ChatResponse{
		llmResponseWithMultipleToolCalls("edit then shell", []llm.ToolCall{
			{ID: "call_a", Name: "edit_file", Input: editInput()},
			{ID: "call_b", Name: "bash", Input: json.RawMessage(`{"command":"ls"}`)},
		}),
	}, func(context.Context) EditVerifyResult {
		return EditVerifyResult{Output: "PASS", ExitCode: 0}
	}, 0)
	exec.SetHITLHandler(&rejectingLastCallHITL{reject: "bash"})
	boundaryChecks := 0
	exec.SetPauseChecker(func(context.Context) bool {
		boundaryChecks++
		return boundaryChecks == 2
	})

	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "edit_file", Description: "edit", Source: "core"},
		{Name: "bash", Description: "shell", Source: "core"},
	}, newMockContextManager())
	if !errors.Is(err, ErrPaused) {
		t.Fatalf("expected ErrPaused, got %v", err)
	}
	if result == nil {
		t.Fatal("expected paused checkpoint result")
	}
	if *calls != 1 {
		t.Errorf("verify runner calls = %d, want 1 before pause", *calls)
	}
	if len(result.Steps) != 2 {
		t.Fatalf("len(Steps) = %d, want edit and rejected sibling", len(result.Steps))
	}
	last := result.Steps[len(result.Steps)-1]
	if last.Action.Name != "bash" || !last.IsError {
		t.Fatalf("last step = %#v, want rejected bash call", last)
	}
	if !strings.Contains(last.Observation, "[Tool call rejected:") ||
		!strings.Contains(last.Observation, "[verify_on_edit]") {
		t.Errorf("rejected sibling observation should carry rejection and verify note, got %q", last.Observation)
	}
}

func TestExecutor_VerifyOnEdit_BatchRejectedLastSiblingFlushesBeforePause(t *testing.T) {
	type batchCall struct {
		Tool  string          `json:"tool"`
		Input json.RawMessage `json:"input"`
	}
	batchInput, err := json.Marshal(map[string][]batchCall{
		"calls": {
			{Tool: "edit_file", Input: editInput()},
			{Tool: "bash", Input: json.RawMessage(`{"command":"ls"}`)},
		},
	})
	if err != nil {
		t.Fatalf("marshal batch input: %v", err)
	}

	exec, _, calls := newVerifyTestExecutor(t, []*llm.ChatResponse{
		llmResponseWithToolCall("batch edit then shell", tools.ToolBatch, batchInput),
	}, func(context.Context) EditVerifyResult {
		return EditVerifyResult{Output: "PASS", ExitCode: 0}
	}, 0)
	exec.SetHITLHandler(&rejectingLastCallHITL{reject: "bash"})
	boundaryChecks := 0
	exec.SetPauseChecker(func(context.Context) bool {
		boundaryChecks++
		return boundaryChecks == 2
	})

	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: tools.ToolBatch, Description: "batch", Source: "core"},
		{Name: "edit_file", Description: "edit", Source: "core"},
		{Name: "bash", Description: "shell", Source: "core"},
	}, newMockContextManager())
	if !errors.Is(err, ErrPaused) {
		t.Fatalf("expected ErrPaused, got %v", err)
	}
	if result == nil || len(result.Steps) != 2 {
		t.Fatalf("paused batch result = %#v, want two completed sub-calls", result)
	}
	if *calls != 1 {
		t.Errorf("verify runner calls = %d, want 1 before pause", *calls)
	}
	last := result.Steps[len(result.Steps)-1]
	if last.Action.Name != "bash" || !last.IsError ||
		!strings.Contains(last.Observation, "[verify_on_edit]") {
		t.Errorf("rejected final batch sibling should retain verify note, got %#v", last)
	}
}

// TestExecutor_VerifyOnEdit_ImplicitFinishDoesNotRepeatFlushedVerification:
// after the rejected final sibling flushes the group, a later text-only finish
// must not run verification a second time.
func TestExecutor_VerifyOnEdit_ImplicitFinishDoesNotRepeatFlushedVerification(t *testing.T) {
	exec, _, calls := newVerifyTestExecutor(t, []*llm.ChatResponse{
		llmResponseWithMultipleToolCalls("edit then shell", []llm.ToolCall{
			{ID: "call_a", Name: "edit_file", Input: editInput()},
			{ID: "call_b", Name: "bash", Input: json.RawMessage(`{"command":"ls"}`)},
		}),
		llmResponseEndTurn("all done"),
	}, func(context.Context) EditVerifyResult {
		return EditVerifyResult{Output: "PASS", ExitCode: 0}
	}, 0)
	exec.SetHITLHandler(&rejectingLastCallHITL{reject: "bash"})

	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "edit_file", Description: "edit", Source: "core"},
		{Name: "bash", Description: "shell", Source: "core"},
	}, newMockContextManager())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.Finished {
		t.Fatalf("expected finished run")
	}
	if *calls != 1 {
		t.Errorf("verify runner calls = %d, want exactly 1 for the response group", *calls)
	}
	if strings.Contains(result.Output, "[verify_on_edit]") {
		t.Errorf("implicit finish must not repeat an already-recorded verify note, got %q", result.Output)
	}
	if len(result.Steps) < 2 || !strings.Contains(result.Steps[1].Observation, "[verify_on_edit]") {
		t.Errorf("rejected sibling should retain the verify note in trajectory: %#v", result.Steps)
	}
}

// TestExecutor_VerifyOnEdit_FailedEditDoesNotTrigger: an edit tool result
// marked as error does not trigger verification.
func TestExecutor_VerifyOnEdit_FailedEditDoesNotTrigger(t *testing.T) {
	mockLLM := &mockLLMCaller{responses: []*llm.ChatResponse{
		llmResponseWithToolCall("editing", "edit_file", editInput()),
		llmResponseFinish("done", "finished"),
	}}
	mockTools := newMockToolExecutor()
	mockTools.results["edit_file"] = tools.ToolResult{Content: "path not found", IsError: true}
	cm := newMockContextManager()
	exec := newExecutorDefaultHITL(mockLLM, mockTools, &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
	var calls int32
	exec.SetVerifyOnEdit(func(context.Context) EditVerifyResult {
		atomic.AddInt32(&calls, 1)
		return EditVerifyResult{Output: "ok", ExitCode: 0}
	}, 0)

	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: "edit_file", Description: "edit", Source: "core"},
	}, cm)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if calls != 0 {
		t.Errorf("verify runner calls = %d, want 0 (edit failed)", calls)
	}
	if strings.Contains(result.Steps[0].Observation, "[verify_on_edit]") {
		t.Errorf("failed edit must not carry a verify note")
	}
}

// TestFormatVerifyNote_RunnerError: infrastructure errors produce a clear
// "could not run" note rather than a fake pass/fail.
func TestFormatVerifyNote_RunnerError(t *testing.T) {
	note := FormatVerifyNote(EditVerifyResult{Err: errors.New("bash: not found")}, 0)
	if !strings.Contains(note, "could not run") || !strings.Contains(note, "bash: not found") {
		t.Errorf("note = %q", note)
	}
}

// TestFormatVerifyNote_TruncationIsRuneSafe: the cap must cut on a rune
// boundary — byte slicing would split a multi-byte rune and inject invalid
// UTF-8 into the observation and the model context.
func TestFormatVerifyNote_TruncationIsRuneSafe(t *testing.T) {
	long := strings.Repeat("✓", 300) // 3 bytes per rune
	note := FormatVerifyNote(EditVerifyResult{Output: long, ExitCode: 0}, 100)
	if !utf8.ValidString(note) {
		t.Fatal("truncated note contains invalid UTF-8 (rune split at the cap)")
	}
	if got := strings.Count(note, "✓"); got > 100 {
		t.Errorf("expected at most 100 ✓ runes after truncation, got %d", got)
	}
}

// TestFormatVerifyNote_CapDefaultsAndTruncation verifies cap fallback and truncation.
func TestFormatVerifyNote_CapDefaultsAndTruncation(t *testing.T) {
	long := strings.Repeat("y", DefaultVerifyOnEditCap+50)
	note := FormatVerifyNote(EditVerifyResult{Output: long, ExitCode: 0}, 0)
	if !strings.Contains(note, "[...truncated ") {
		t.Errorf("default cap should truncate")
	}
	short := FormatVerifyNote(EditVerifyResult{Output: "tiny", ExitCode: 0}, 0)
	if !strings.Contains(short, "tiny") {
		t.Errorf("short output should be preserved: %q", short)
	}
}

// TestFormatVerifyNote_NegativeExitCodeIsNotAFailedRun: a negative exit code
// means the command never produced an exit status (blocked by policy or
// killed by a signal). The note must say the edit was NOT verified — it must
// not claim VERIFICATION FAILED, which would send the model chasing test
// failures that do not exist.
func TestFormatVerifyNote_NegativeExitCodeIsNotAFailedRun(t *testing.T) {
	note := FormatVerifyNote(EditVerifyResult{
		Output:   "blocked by security policy: group deny",
		ExitCode: -1,
	}, 0)

	if strings.Contains(note, "VERIFICATION FAILED") {
		t.Errorf("policy-blocked run must not be reported as VERIFICATION FAILED: %q", note)
	}
	if !strings.Contains(note, "did not complete") {
		t.Errorf("note should state the command did not complete: %q", note)
	}
	if !strings.Contains(note, "NOT verified") {
		t.Errorf("note should state the edit was not verified: %q", note)
	}
	if !strings.Contains(note, "blocked by security policy") {
		t.Errorf("note should preserve the runner output: %q", note)
	}
}
