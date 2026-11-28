package agent

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/v0lka/sp4rk/llm"
	"github.com/v0lka/sp4rk/tools"
)

// --- Emitter seam ----------------------------------------------------------

// baseOnlyEmitter implements ONLY the base Events interface. Assigning it to
// an Events variable is the compile-time proof that a pre-seam emitter still
// satisfies the contract; it records every base ToolResult call it receives.
type baseOnlyEmitter struct {
	*NoopEvents
	baseCalls int
}

func (b *baseOnlyEmitter) ToolResult(_, _, _ int, _ string, _ bool) { b.baseCalls++ }

// kindCall is one recorded kind-bearing emission.
type kindCall struct {
	stepNum, callIdx, resultLen int
	preview                     string
	isError                     bool
	kind                        tools.ToolErrorKind
}

// kindEmitter opts into the kind-bearing seam. It also records base calls so
// a test can prove the two are never both invoked for the same result.
type kindEmitter struct {
	*NoopEvents
	baseCalls int
	calls     []kindCall
}

var _ ToolResultKindEvents = (*kindEmitter)(nil)

func (k *kindEmitter) ToolResult(_, _, _ int, _ string, _ bool) { k.baseCalls++ }

func (k *kindEmitter) ToolResultWithKind(stepNum, callIdx, resultLen int, preview string, isError bool, kind tools.ToolErrorKind) {
	k.calls = append(k.calls, kindCall{stepNum: stepNum, callIdx: callIdx, resultLen: resultLen, preview: preview, isError: isError, kind: kind})
}

// TestEmitToolResultWithKind_BaseOnlyEmitter pins the backward-compatible
// arm: an emitter implementing only Events (no ToolResultKindEvents) still
// compiles against the interface and receives exactly the base ToolResult
// call — the kind is simply dropped for it.
func TestEmitToolResultWithKind_BaseOnlyEmitter(t *testing.T) {
	e := &baseOnlyEmitter{}
	var asEvents Events = e // compiles: only the base interface is required
	if _, optsIn := asEvents.(ToolResultKindEvents); optsIn {
		t.Fatal("a base-only emitter must not satisfy ToolResultKindEvents")
	}
	emitToolResultWithKind(asEvents, 3, 1, 12, "preview", true, tools.ToolErrorInvalidCall)
	if e.baseCalls != 1 {
		t.Fatalf("base ToolResult called %d times, want 1", e.baseCalls)
	}
}

// TestEmitToolResultWithKind_KindEmitter pins the kind-bearing arm: the
// optional method receives every argument verbatim, and the base call is NOT
// also invoked for the same result (one result, one event).
func TestEmitToolResultWithKind_KindEmitter(t *testing.T) {
	e := &kindEmitter{}
	var asEvents Events = e
	emitToolResultWithKind(asEvents, 3, 1, 12, "preview", true, tools.ToolErrorInvalidCall)
	if e.baseCalls != 0 {
		t.Fatalf("kind emitter must not also receive the base call, got %d", e.baseCalls)
	}
	if len(e.calls) != 1 {
		t.Fatalf("kind calls = %d, want 1", len(e.calls))
	}
	got := e.calls[0]
	want := kindCall{stepNum: 3, callIdx: 1, resultLen: 12, preview: "preview", isError: true, kind: tools.ToolErrorInvalidCall}
	if got != want {
		t.Fatalf("kind call = %+v, want %+v", got, want)
	}
}

// TestStepErrorKindJSONTag pins the wire shape: the class serialises as
// error_kind and is omitted when unset (a non-error step stays compact).
func TestStepErrorKindJSONTag(t *testing.T) {
	b, err := json.Marshal(Step{IsError: true, ErrorKind: tools.ToolErrorInvalidCall})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), `"error_kind":"invalid_call"`) {
		t.Fatalf("marshalled step missing error_kind: %s", b)
	}
	b, err = json.Marshal(Step{IsError: true})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(b), "error_kind") {
		t.Fatalf("zero ErrorKind must be omitted, got: %s", b)
	}
}

// --- Executor end-to-end: kind flows to the emitter and the Step ----------

// agentTestTool is a minimal tools.Tool for driving a real *tools.ToolRegistry
// from executor tests: the registry's pre-dispatch validator and the batch
// pre-flight then reject input against the declared schema exactly as in
// production, instead of a hand-rolled mock.
type agentTestTool struct {
	tools.BaseTool
	exec func(context.Context, json.RawMessage) (tools.ToolResult, error)
	runs int
}

func (a *agentTestTool) Execute(ctx context.Context, in json.RawMessage) (tools.ToolResult, error) {
	a.runs++
	if a.exec != nil {
		return a.exec(ctx, in)
	}
	return tools.ToolResult{Content: "ok"}, nil
}

func strictSchema() json.RawMessage {
	return json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}`)
}

// stepFor returns the trajectory step whose action is the named tool, or nil.
// A run that ends on finish materialises a finish step too, so tests must
// never assume the tool step is the only one.
func stepFor(res *ExecutorResult, tool string) *Step {
	for i := range res.Steps {
		if res.Steps[i].Action.Name == tool {
			return &res.Steps[i]
		}
	}
	return nil
}

// wantErrorStep asserts that res carries exactly one step for tool, that it is
// an error step, and that its ErrorKind equals kind.
func wantErrorStep(t *testing.T, res *ExecutorResult, tool string, kind tools.ToolErrorKind) {
	t.Helper()
	step := stepFor(res, tool)
	if step == nil {
		t.Fatalf("no step for %q in %+v", tool, res.Steps)
	}
	if !step.IsError {
		t.Fatalf("%q step must be an error step: %+v", tool, step)
	}
	if step.ErrorKind != kind {
		t.Fatalf("%q step ErrorKind = %q, want %q", tool, step.ErrorKind, kind)
	}
}

// TestExecutor_Run_SchemaRejectedCallCarriesInvalidCallKind: a call the
// registry's pre-dispatch validator rejects is emitted as kind=invalid_call
// and the materialised Step carries Step.ErrorKind=invalid_call — while the
// never-executed tool records zero runs.
func TestExecutor_Run_SchemaRejectedCallCarriesInvalidCallKind(t *testing.T) {
	reg := tools.NewToolRegistry()
	strict := &agentTestTool{BaseTool: tools.BaseTool{ToolName: "strict", ToolDescription: "strict", Schema: strictSchema(), Policy: tools.PolicyAlwaysAllow}}
	reg.Register(strict)

	emitter := &kindEmitter{}
	model := &mockLLMCaller{responses: []*llm.ChatResponse{
		llmResponseWithToolCall("call strict", "strict", json.RawMessage(`{}`)),
		llmResponseFinish("done", "done"),
	}}
	exec := newExecutorDefaultHITL(model, reg, &mockTokenCounter{}, 5, emitter, false, ToolResultBudget{}, defaultCircuitBreakerConfig)

	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{{Name: "strict", Description: "strict", Source: "core"}}, newMockContextManager())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if strict.runs != 0 {
		t.Fatalf("schema-rejected input must never reach the tool, got %d runs", strict.runs)
	}
	if len(emitter.calls) != 2 {
		t.Fatalf("kind emissions = %d, want 2 (strict + finish): %+v", len(emitter.calls), emitter.calls)
	}
	rejected := emitter.calls[0]
	if rejected.isError != true || rejected.kind != tools.ToolErrorInvalidCall {
		t.Fatalf("rejected call emission = isError %v kind %q, want true/%q", rejected.isError, rejected.kind, tools.ToolErrorInvalidCall)
	}
	if !strings.Contains(rejected.preview, `input for "strict" is invalid`) {
		t.Fatalf("rejected preview must carry the rendered schema error, got: %q", rejected.preview)
	}
	finish := emitter.calls[1]
	if finish.isError || finish.kind != "" {
		t.Fatalf("finish emission = isError %v kind %q, want false/\"\"", finish.isError, finish.kind)
	}
	wantErrorStep(t, result, "strict", tools.ToolErrorInvalidCall)
}

// TestExecutor_Run_ParseErrorCarriesInvalidCallKind: a tool answering with
// tools.ParseInputError (the shared constructor every builtin and the MCP
// bridge use) is emitted and stamped as an invalid call.
func TestExecutor_Run_ParseErrorCarriesInvalidCallKind(t *testing.T) {
	parseResult, _ := tools.ParseInputError(errors.New("unexpected token"))
	toolExec := newMockToolExecutor()
	toolExec.results["parsey"] = parseResult

	emitter := &kindEmitter{}
	model := &mockLLMCaller{responses: []*llm.ChatResponse{
		llmResponseWithToolCall("call parsey", "parsey", json.RawMessage(`{not json`)),
		llmResponseFinish("done", "done"),
	}}
	exec := newExecutorDefaultHITL(model, toolExec, &mockTokenCounter{}, 5, emitter, false, ToolResultBudget{}, defaultCircuitBreakerConfig)

	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{{Name: "parsey", Description: "parsey", Source: "core"}}, newMockContextManager())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(emitter.calls) != 2 {
		t.Fatalf("kind emissions = %d, want 2: %+v", len(emitter.calls), emitter.calls)
	}
	if got := emitter.calls[0]; got.isError != true || got.kind != tools.ToolErrorInvalidCall {
		t.Fatalf("parse-error emission = isError %v kind %q, want true/%q", got.isError, got.kind, tools.ToolErrorInvalidCall)
	}
	wantErrorStep(t, result, "parsey", tools.ToolErrorInvalidCall)
}

// TestExecutor_Run_BatchPreflightRejectsInvalidCallKind: the batch meta-tool's
// pre-flight aggregate (one sub-call fails schema validation, nothing runs)
// is an invalid call in both the emission and the Step, and no sub-call tool
// ever executes.
func TestExecutor_Run_BatchPreflightRejectsInvalidCallKind(t *testing.T) {
	reg := tools.NewToolRegistry()
	strict := &agentTestTool{BaseTool: tools.BaseTool{ToolName: "strict", ToolDescription: "strict", Schema: strictSchema(), Policy: tools.PolicyAlwaysAllow}}
	reg.Register(strict)

	emitter := &kindEmitter{}
	model := &mockLLMCaller{responses: []*llm.ChatResponse{
		llmResponseWithToolCall("batch it", tools.ToolBatch, json.RawMessage(`{"calls":[{"tool":"strict","input":{}}]}`)),
		llmResponseFinish("done", "done"),
	}}
	exec := newExecutorDefaultHITL(model, reg, &mockTokenCounter{}, 5, emitter, false, ToolResultBudget{}, defaultCircuitBreakerConfig)

	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{
		{Name: tools.ToolBatch, Description: "batch", Source: "core"},
		{Name: "strict", Description: "strict", Source: "core"},
	}, newMockContextManager())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if strict.runs != 0 {
		t.Fatalf("a pre-flight-rejected batch must execute nothing, got %d runs", strict.runs)
	}
	if len(emitter.calls) != 2 {
		t.Fatalf("kind emissions = %d, want 2 (batch aggregate + finish): %+v", len(emitter.calls), emitter.calls)
	}
	aggregate := emitter.calls[0]
	if aggregate.isError != true || aggregate.kind != tools.ToolErrorInvalidCall {
		t.Fatalf("batch aggregate emission = isError %v kind %q, want true/%q", aggregate.isError, aggregate.kind, tools.ToolErrorInvalidCall)
	}
	if !strings.Contains(aggregate.preview, "batch rejected: no sub-call was executed") {
		t.Fatalf("aggregate preview = %q", aggregate.preview)
	}
	wantErrorStep(t, result, tools.ToolBatch, tools.ToolErrorInvalidCall)
}

// TestExecutor_Run_PlainErrorResultKeepsZeroKind: a tool that ran and failed
// with tools.ErrorResult emits and stamps the UNSPECIFIED zero kind — the
// legacy class, so no consumer mistakes a genuine execution failure for a
// malformed call.
func TestExecutor_Run_PlainErrorResultKeepsZeroKind(t *testing.T) {
	toolExec := newMockToolExecutor()
	toolExec.results["failer"] = tools.ErrorResult("disk on fire")

	emitter := &kindEmitter{}
	model := &mockLLMCaller{responses: []*llm.ChatResponse{
		llmResponseWithToolCall("call failer", "failer", json.RawMessage(`{}`)),
		llmResponseFinish("done", "done"),
	}}
	exec := newExecutorDefaultHITL(model, toolExec, &mockTokenCounter{}, 5, emitter, false, ToolResultBudget{}, defaultCircuitBreakerConfig)

	result, err := exec.Run(context.Background(), []tools.ToolDescriptor{{Name: "failer", Description: "failer", Source: "core"}}, newMockContextManager())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(emitter.calls) != 2 {
		t.Fatalf("kind emissions = %d, want 2: %+v", len(emitter.calls), emitter.calls)
	}
	if got := emitter.calls[0]; got.isError != true || got.kind != "" {
		t.Fatalf("ErrorResult emission = isError %v kind %q, want true/\"\"", got.isError, got.kind)
	}
	wantErrorStep(t, result, "failer", "")
}

// --- Repeat breaker: wait tools are exempt ---------------------------------

// waitExemptBreakerConfig arms every post-execution breaker low (repeat 3/5,
// fruitless 3/6, same-tool 3/6) so a NON-exempt tool polled identically
// would be intercepted well before six calls. Only an exemption on all three
// detectors lets a tool survive the whole sequence.
func waitExemptBreakerConfig() CircuitBreakerConfig {
	return CircuitBreakerConfig{
		RepeatNudgeThreshold:         3,
		RepeatAbortThreshold:         5,
		TruncationAbortThreshold:     3,
		ParseErrorAbortThreshold:     3,
		FruitlessNudgeThreshold:      3,
		FruitlessAbortThreshold:      6,
		FruitlessMaxResultLen:        32,
		SameToolRepeatNudgeThreshold: 3,
		SameToolRepeatAbortThreshold: 6,
		SameToolResultSizeDelta:      64,
	}
}

// runIdenticalCalls drives n identical calls to tool with the given result,
// then a finish response, and returns the run result, the executor and the
// recorded executions.
func runIdenticalCalls(t *testing.T, tool string, input json.RawMessage, result tools.ToolResult, n int, cfg CircuitBreakerConfig) (*Executor, *ExecutorResult, *mockToolExecutor) {
	t.Helper()
	responses := make([]*llm.ChatResponse, 0, n+1)
	for i := 0; i < n; i++ {
		responses = append(responses, llmResponseWithToolCall("poll", tool, input))
	}
	responses = append(responses, llmResponseFinish("done", "done"))
	toolExec := newMockToolExecutor()
	toolExec.results[tool] = result
	exec := newExecutorDefaultHITL(&mockLLMCaller{responses: responses}, toolExec, &mockTokenCounter{}, 20, nil, false, ToolResultBudget{}, cfg)
	res, err := exec.Run(context.Background(), []tools.ToolDescriptor{{Name: tool, Description: tool, Source: "core"}}, newMockContextManager())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	return exec, res, toolExec
}

// TestExecutor_Run_WaitToolRepeatExempt: six consecutive IDENTICAL wait calls
// with a short "still pending" result neither nudge nor abort on any breaker
// and all six execute — the polling pattern the wait tools exist for.
func TestExecutor_Run_WaitToolRepeatExempt(t *testing.T) {
	for _, tool := range []string{"wait", "wait_for_one", "wait_for_all"} {
		t.Run(tool, func(t *testing.T) {
			input := json.RawMessage(`{"ids":["d-1"]}`)
			_, res, toolExec := runIdenticalCalls(t, tool, input, tools.ToolResult{Content: "still pending"}, 6, waitExemptBreakerConfig())

			if !res.Finished {
				t.Fatalf("Finished=false, want true (Output=%q)", res.Output)
			}
			if strings.Contains(res.Output, "Aborted") {
				t.Fatalf("wait poll must not abort, Output=%q", res.Output)
			}
			if len(toolExec.calls) != 6 {
				t.Fatalf("executions = %d, want 6 (every poll runs)", len(toolExec.calls))
			}
			for i, s := range res.Steps {
				if s.UserNudge != "" {
					t.Fatalf("step %d carries a breaker nudge: %q", i, s.UserNudge)
				}
				if s.IsError && strings.Contains(s.Observation, "identical arguments") {
					t.Fatalf("step %d carries the repeat-identical interception: %q", i, s.Observation)
				}
			}
		})
	}
}

// TestExecutor_Run_RepeatExemptResetsTracker: an exempt call RESETS the
// shared repeat tracker (rather than leaving a stale count for the next
// non-exempt identical call to inherit), mirroring the same-tool exemption.
func TestCheckRepeatIdenticalTool_RepeatExemptResetsTracker(t *testing.T) {
	for _, tool := range []string{"wait", "wait_for_one", "wait_for_all"} {
		t.Run(tool, func(t *testing.T) {
			exec := newExecutorDefaultHITL(&mockLLMCaller{}, newMockToolExecutor(), &mockTokenCounter{}, 10, nil, false, ToolResultBudget{}, waitExemptBreakerConfig())
			action := llm.ToolCall{Name: tool, Input: json.RawMessage(`{"ids":["d-1"]}`)}
			// Seed the tracker as if four identical calls had already been
			// counted with the last one erroring — one short of the abort.
			exec.repeat.count = 4
			exec.repeat.lastKey = tool + ":" + compactJSON(action.Input)
			exec.repeat.lastIsError = true

			act, result, err := exec.checkRepeatIdenticalTool(context.Background(), action, 0, "poll", nil, &runState{}, newMockContextManager())
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if act != actionNone || result != nil {
				t.Fatalf("exempt call returned action %v result %v, want actionNone/nil", act, result)
			}
			if exec.repeat.count != 0 || exec.repeat.lastKey != "" || exec.repeat.lastIsError {
				t.Fatalf("tracker not reset: count=%d lastKey=%q lastIsError=%v", exec.repeat.count, exec.repeat.lastKey, exec.repeat.lastIsError)
			}
		})
	}
}

// TestExecutor_Run_RepeatBreakerStillTripsForNonExemptTool is the control for
// the wait exemption: the SAME six-call identical pattern on a non-exempt tool
// (read_file) is intercepted by the repeat breaker at the configured 3 (nudge)
// and 5 (abort). Fruitless/same-tool are set high so the repeat breaker is
// the only one in play.
func TestExecutor_Run_RepeatBreakerStillTripsForNonExemptTool(t *testing.T) {
	cfg := CircuitBreakerConfig{
		RepeatNudgeThreshold:         3,
		RepeatAbortThreshold:         5,
		TruncationAbortThreshold:     3,
		ParseErrorAbortThreshold:     3,
		FruitlessNudgeThreshold:      50,
		FruitlessAbortThreshold:      60,
		FruitlessMaxResultLen:        32,
		SameToolRepeatNudgeThreshold: 50,
		SameToolRepeatAbortThreshold: 60,
		SameToolResultSizeDelta:      64,
	}
	// A result long enough to stay clear of the fruitless length even if
	// that detector were armed.
	_, res, toolExec := runIdenticalCalls(t, "read_file", json.RawMessage(`{"path":"a.go"}`), tools.ToolResult{Content: strings.Repeat("x", 200)}, 6, cfg)

	if res.Finished {
		t.Fatalf("Finished=true, want false on repeat abort (Output=%q)", res.Output)
	}
	if !strings.Contains(res.Output, "called 5 times consecutively with identical arguments") {
		t.Fatalf("Output = %q, want the 5-call repeat abort", res.Output)
	}
	// The repeat breaker nudges on EVERY call from the nudge threshold up to
	// the abort threshold (no one-shot flag, unlike fruitless/same-tool), so
	// calls 1,2 execute, 3 and 4 are intercepted by the nudge, 5 by the abort.
	if len(toolExec.calls) != 2 {
		t.Fatalf("executions = %d, want 2", len(toolExec.calls))
	}
	nudges := 0
	for _, s := range res.Steps {
		if s.IsError && strings.Contains(s.Observation, "identical arguments") {
			nudges++
		}
	}
	if nudges != 2 {
		t.Fatalf("repeat nudge steps = %d, want exactly 2 (at the 3rd and 4th calls)", nudges)
	}
}

// TestCircuitBreakerExemptTools_WaitTools pins the table membership directly
// for the fruitless/same-tool exemption, independent of the detector logic.
func TestCircuitBreakerExemptTools_WaitTools(t *testing.T) {
	for _, tool := range []string{"wait", "wait_for_one", "wait_for_all"} {
		if _, ok := circuitBreakerExemptTools[tool]; !ok {
			t.Errorf("%q missing from circuitBreakerExemptTools", tool)
		}
		if _, ok := repeatExemptTools[tool]; !ok {
			t.Errorf("%q missing from repeatExemptTools", tool)
		}
	}
	if _, ok := repeatExemptTools["read_file"]; ok {
		t.Error("read_file must not be repeat-exempt")
	}
}
