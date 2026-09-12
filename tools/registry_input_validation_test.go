package tools

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

// Registry-side input-validation tests: they exercise the
// ToolRegistry.Execute wiring (validation before policy resolution and
// dispatch, the SetInputValidationEnabled escape hatch, the not-found path).
// Pure ValidateToolInput semantics are covered in inputvalidate_test.go.

// registryValidationSchema declares a closed property set with one required
// key (path) and one optional key (query).
const registryValidationSchema = `{"type":"object","properties":{"path":{"type":"string"},"query":{"type":"string"}},"required":["path"]}`

// registryDispatchFlags record whether the tool body and/or the
// confirmation callback were reached during Execute.
type registryDispatchFlags struct {
	execCalled    bool
	confirmCalled bool
}

func (d *registryDispatchFlags) instrumentTool(t *mockTool) {
	t.execFn = func(_ context.Context, _ json.RawMessage) (ToolResult, error) {
		d.execCalled = true
		return ToolResult{Content: "executed"}, nil
	}
}

func (d *registryDispatchFlags) installConfirm(reg *ToolRegistry) {
	reg.SetConfirmFunc(func(_ context.Context, _ ConfirmationRequest) (ConfirmationResponse, error) {
		d.confirmCalled = true
		return ConfirmAllowOnce, nil
	})
}

func newRegistryValidationTool(name string, policy ToolPolicy) *mockTool {
	return &mockTool{
		BaseTool: BaseTool{
			ToolName:        name,
			ToolDescription: name + " for registry input-validation tests",
			Schema:          json.RawMessage(registryValidationSchema),
			Policy:          policy,
		},
	}
}

// TestRegistryExecute_InvalidInput_RejectedBeforeDispatch verifies that an
// input missing a schema-required parameter is rejected as an ErrorResult
// with a nil Go error (the model retries), and that neither the
// confirmation flow nor the tool body is ever reached.
func TestRegistryExecute_InvalidInput_RejectedBeforeDispatch(t *testing.T) {
	reg := NewToolRegistry()
	flags := &registryDispatchFlags{}
	tool := newRegistryValidationTool("needs_path", PolicyUserConfirm)
	flags.instrumentTool(tool)
	flags.installConfirm(reg)
	reg.Register(tool)

	res, err := reg.Execute(context.Background(), "needs_path", json.RawMessage(`{"query":"select"}`))
	if err != nil {
		t.Fatalf("expected nil Go error (model-retryable result), got %v", err)
	}
	if !res.IsError {
		t.Fatalf("expected IsError=true, got %+v", res)
	}
	for _, want := range []string{"needs_path", `missing required parameter "path"`} {
		if !strings.Contains(res.Content, want) {
			t.Errorf("result content %q does not mention %q", res.Content, want)
		}
	}
	if flags.confirmCalled {
		t.Error("confirmFunc must not be called for an input rejected by validation")
	}
	if flags.execCalled {
		t.Error("tool.Execute must not be called for an input rejected by validation")
	}
}

// TestRegistryExecute_InvalidInput_BeforePolicyResolution proves validation
// runs BEFORE policy resolution: even a PolicyAlwaysDeny tool reports the
// validation error, not the policy-block message, for malformed input.
func TestRegistryExecute_InvalidInput_BeforePolicyResolution(t *testing.T) {
	reg := NewToolRegistry()
	flags := &registryDispatchFlags{}
	tool := newRegistryValidationTool("denied_tool", PolicyAlwaysDeny)
	flags.instrumentTool(tool)
	reg.Register(tool)

	res, err := reg.Execute(context.Background(), "denied_tool", json.RawMessage(`{}`))
	if err != nil {
		t.Fatalf("expected nil Go error, got %v", err)
	}
	if !res.IsError {
		t.Fatal("expected IsError=true")
	}
	if !strings.Contains(res.Content, `missing required parameter "path"`) {
		t.Errorf("expected validation error before policy resolution, got %q", res.Content)
	}
	if strings.Contains(res.Content, "blocked by security policy") {
		t.Errorf("policy block message must not preempt validation, got %q", res.Content)
	}
	if flags.execCalled {
		t.Error("tool.Execute must not be called")
	}
}

// TestRegistryExecute_ValidInput_PolicyAlwaysAllow verifies a valid input
// still reaches the tool body under PolicyAlwaysAllow.
func TestRegistryExecute_ValidInput_PolicyAlwaysAllow(t *testing.T) {
	reg := NewToolRegistry()
	flags := &registryDispatchFlags{}
	tool := newRegistryValidationTool("reader", PolicyAlwaysAllow)
	flags.instrumentTool(tool)
	reg.Register(tool)

	res, err := reg.Execute(context.Background(), "reader", json.RawMessage(`{"path":"/ws/file.txt"}`))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.IsError {
		t.Fatalf("unexpected error result: %+v", res)
	}
	if !flags.execCalled {
		t.Error("valid input must reach tool.Execute under PolicyAlwaysAllow")
	}
}

// TestRegistryExecute_ValidInput_PolicyUserConfirm verifies a valid input
// flows through the confirmation callback and then executes as before.
func TestRegistryExecute_ValidInput_PolicyUserConfirm(t *testing.T) {
	reg := NewToolRegistry()
	flags := &registryDispatchFlags{}
	tool := newRegistryValidationTool("mutating", PolicyUserConfirm)
	flags.instrumentTool(tool)
	flags.installConfirm(reg)
	reg.Register(tool)

	res, err := reg.Execute(context.Background(), "mutating", json.RawMessage(`{"path":"/ws/file.txt"}`))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.IsError {
		t.Fatalf("unexpected error result: %+v", res)
	}
	if !flags.confirmCalled {
		t.Error("valid input must still reach confirmFunc under PolicyUserConfirm")
	}
	if !flags.execCalled {
		t.Error("valid input must reach tool.Execute after confirmation")
	}
}

// TestRegistryExecute_ValidInput_PolicyAlwaysDeny verifies a valid input
// under PolicyAlwaysDeny is blocked by the policy exactly as before
// (validation passing does not loosen the policy).
func TestRegistryExecute_ValidInput_PolicyAlwaysDeny(t *testing.T) {
	reg := NewToolRegistry()
	flags := &registryDispatchFlags{}
	tool := newRegistryValidationTool("forbidden", PolicyAlwaysDeny)
	flags.instrumentTool(tool)
	reg.Register(tool)

	res, err := reg.Execute(context.Background(), "forbidden", json.RawMessage(`{"path":"/ws/file.txt"}`))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !res.IsError {
		t.Fatal("expected IsError=true for denied policy")
	}
	if !strings.Contains(res.Content, "blocked by security policy") {
		t.Errorf("expected policy block message, got %q", res.Content)
	}
	if flags.execCalled {
		t.Error("tool.Execute must not be called under PolicyAlwaysDeny")
	}
}

// TestRegistryExecute_InputValidationDisabled_InvalidInputReachesTool
// verifies the escape hatch: with validation disabled, an invalid input is
// handed to the tool untouched (pre-validation behaviour).
func TestRegistryExecute_InputValidationDisabled_InvalidInputReachesTool(t *testing.T) {
	reg := NewToolRegistry()
	flags := &registryDispatchFlags{}
	tool := newRegistryValidationTool("unvalidated", PolicyAlwaysAllow)
	flags.instrumentTool(tool)
	reg.Register(tool)

	reg.SetInputValidationEnabled(false)

	res, err := reg.Execute(context.Background(), "unvalidated", json.RawMessage(`{}`))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.IsError {
		t.Fatalf("unexpected error result: %+v", res)
	}
	if !flags.execCalled {
		t.Error("with validation disabled, invalid input must reach tool.Execute")
	}
}

// TestRegistryExecute_NotFound_NoValidation verifies the not-found path is
// unchanged: an unknown tool reports the not-found error result and never
// reaches validation (there is no schema to validate against).
func TestRegistryExecute_NotFound_NoValidation(t *testing.T) {
	reg := NewToolRegistry()
	reg.SetConfirmFunc(func(_ context.Context, _ ConfirmationRequest) (ConfirmationResponse, error) {
		t.Error("confirmFunc must not be called for an unknown tool")
		return ConfirmDeny, nil
	})

	res, err := reg.Execute(context.Background(), "ghost", json.RawMessage(`{}`))
	if err != nil {
		t.Fatalf("expected nil infrastructure error, got %v", err)
	}
	if !res.IsError || res.Content != "tool not found: ghost" {
		t.Errorf("unexpected result for unknown tool: %+v", res)
	}
}
