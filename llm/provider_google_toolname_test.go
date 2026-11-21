package llm

import (
	"encoding/json"
	"strings"
	"testing"
)

// TestBuildGoogleRequest_MapsToolNamesAndPreservesPairing pins the two Google
// contracts the mapping must never break:
//  1. functionDeclarations, history functionCall parts and functionResponse
//     parts all carry the SAME wire spelling — Google pairs a response with
//     its call by NAME, so a response mapped differently from its call is an
//     HTTP 400 INVALID_ARGUMENT on the first response part;
//  2. dotted MCP names are spelled endpoint-legally everywhere.
func TestBuildGoogleRequest_MapsToolNamesAndPreservesPairing(t *testing.T) {
	req := ChatRequest{
		Model: "gemini-2.5-pro",
		Messages: []Message{
			{Role: "user", Content: "list services"},
			{
				Role: "assistant",
				ToolCalls: []ToolCall{
					{ID: "ps.list", Name: "ps.list", Input: json.RawMessage(`{}`)},
				},
			},
			{Role: "tool", ToolCallID: "ps.list", Content: `{"ok":true}`},
		},
		Tools: []ToolDefinition{
			{Name: "bash_exec", Description: "legal name"},
			{Name: "ps.list", Description: "dotted MCP tool"},
		},
	}
	mapping := newToolNameMapping(req.Tools, defaultToolNamePolicy, nil)
	out := buildGoogleRequest(req, mapping)

	if len(out.Tools) != 1 || len(out.Tools[0].FunctionDeclarations) != 2 {
		t.Fatalf("function declarations = %+v, want one tools entry with two declarations", out.Tools)
	}
	if got, want := out.Tools[0].FunctionDeclarations[0].Name, "bash_exec"; got != want {
		t.Errorf("declaration 0 name = %q, want %q (legal name untouched)", got, want)
	}
	if got, want := out.Tools[0].FunctionDeclarations[1].Name, "ps_list"; got != want {
		t.Errorf("declaration 1 name = %q, want %q", got, want)
	}

	// Collect every functionCall and functionResponse name on the contents wire.
	var callName, respName string
	for _, c := range out.Contents {
		for _, part := range c.Parts {
			if part.FunctionCall != nil {
				callName = part.FunctionCall.Name
			}
			if part.FunctionResp != nil {
				respName = part.FunctionResp.Name
			}
		}
	}
	if got, want := callName, "ps_list"; got != want {
		t.Errorf("history functionCall name = %q, want %q", got, want)
	}
	if got, want := respName, "ps_list"; got != want {
		t.Errorf("functionResponse name = %q, want %q (must match the functionCall name)", got, want)
	}
	if callName != respName {
		t.Errorf("functionCall %q and functionResponse %q names diverge — Google pairs them by name", callName, respName)
	}
}

// TestParseGoogleResponse_ReverseMapsFunctionCallNames pins the dispatch
// contract AND the historical "ID equals the name" format: the wire name is
// reversed to the REGISTERED name before the ToolCall.ID is built from it, so
// the executor's tool-result correlation keeps working for mapped names, and
// the Gemini 3 thought signature still rides behind the name.
func TestParseGoogleResponse_ReverseMapsFunctionCallNames(t *testing.T) {
	mapping := newToolNameMapping([]ToolDefinition{{Name: "ps.list"}, {Name: "bash_exec"}}, defaultToolNamePolicy, nil)

	resp := &googleGenerateResponse{
		Candidates: []googleCandidate{{
			Content: googleContent{
				Role: "model",
				Parts: []googlePart{
					{FunctionCall: &googleFunctionCall{Name: "ps_list", Args: json.RawMessage(`{}`)}},
					{FunctionCall: &googleFunctionCall{Name: "bash_exec", Args: json.RawMessage(`{}`)}},
					{
						FunctionCall:     &googleFunctionCall{Name: "ps_list", Args: json.RawMessage(`{}`)},
						ThoughtSignature: "sigABC==",
					},
				},
			},
			FinishReason: "STOP",
		}},
	}
	result, err := parseGoogleResponse("gemini-2.5-pro", resp, []byte("{}"), "p", mapping)
	if err != nil {
		t.Fatalf("parseGoogleResponse: %v", err)
	}
	if len(result.Message.ToolCalls) != 3 {
		t.Fatalf("tool calls = %d, want 3", len(result.Message.ToolCalls))
	}
	if got, want := result.Message.ToolCalls[0].Name, "ps.list"; got != want {
		t.Errorf("tool call 0 name = %q, want %q (wire name reversed to the registered name)", got, want)
	}
	if got, want := result.Message.ToolCalls[0].ID, "ps.list"; got != want {
		t.Errorf("tool call 0 ID = %q, want %q (ID equals the REAL name, historical format)", got, want)
	}
	if got, want := result.Message.ToolCalls[1].Name, "bash_exec"; got != want {
		t.Errorf("tool call 1 name = %q, want %q (identity name passes through)", got, want)
	}
	// The thought signature still rides behind the REAL name.
	tc := result.Message.ToolCalls[2]
	if got, want := tc.ID, "ps.list"+googleThoughtSigSep+"sigABC=="; got != want {
		t.Errorf("tool call 2 ID = %q, want %q (real name + thought signature)", got, want)
	}
	if !strings.HasPrefix(tc.ID, "ps.list") {
		t.Errorf("tool call 2 ID = %q, want the real name as prefix", tc.ID)
	}
}
