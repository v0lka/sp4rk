package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

// ---------------------------------------------------------------------------
// Finding #52: convertToResponsesTools must install the default object schema
// for a nil/empty InputSchema. FunctionToolParam.Parameters is tagged
// "parameters,omitzero,required", so a nil map is dropped from the wire
// entirely and the endpoint rejects the tool with HTTP 400
// missing_required_parameter — while the same toolset works on the Chat
// protocol, whose builder installs exactly this default.
// ---------------------------------------------------------------------------

func TestConvertToResponsesTools_EmptySchemaGetsDefaultParameters(t *testing.T) {
	tools := []ToolDefinition{
		{Name: "nil_schema", Description: "declares no schema"},
		{Name: "empty_schema", Description: "declares an empty schema", InputSchema: json.RawMessage{}},
		{Name: "real_schema", Description: "declares a real schema", InputSchema: json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}},"required":["q"]}`)},
	}
	result := convertToResponsesTools(tools)
	if len(result) != len(tools) {
		t.Fatalf("got %d tools, want %d", len(result), len(tools))
	}

	for i, name := range []string{"nil_schema", "empty_schema"} {
		if result[i].OfFunction == nil {
			t.Fatalf("tool %d (%s): OfFunction is nil", i, name)
		}
		raw, err := json.Marshal(result[i].OfFunction)
		if err != nil {
			t.Fatalf("tool %d (%s): marshal: %v", i, name, err)
		}
		var wire map[string]any
		if err := json.Unmarshal(raw, &wire); err != nil {
			t.Fatalf("tool %d (%s): unmarshal %s: %v", i, name, raw, err)
		}
		params, ok := wire["parameters"].(map[string]any)
		if !ok {
			t.Fatalf("tool %d (%s): parameters missing from the wire payload %s — omitzero drops the nil map and the endpoint answers 400", i, name, raw)
		}
		if params["type"] != "object" {
			t.Errorf("tool %d (%s): parameters.type = %v, want object", i, name, params["type"])
		}
		if _, ok := params["properties"]; !ok {
			t.Errorf("tool %d (%s): parameters.properties missing in %s", i, name, raw)
		}
	}

	// A real schema must pass through untouched (no default merging).
	if result[2].OfFunction == nil {
		t.Fatal("tool 2 (real_schema): OfFunction is nil")
	}
	raw, err := json.Marshal(result[2].OfFunction)
	if err != nil {
		t.Fatalf("tool 2 (real_schema): marshal: %v", err)
	}
	var wire map[string]any
	if err := json.Unmarshal(raw, &wire); err != nil {
		t.Fatalf("tool 2 (real_schema): unmarshal %s: %v", raw, err)
	}
	params, _ := wire["parameters"].(map[string]any)
	props, ok := params["properties"].(map[string]any)
	if !ok || props["q"] == nil {
		t.Errorf("tool 2 (real_schema): real schema was replaced by the default: %s", raw)
	}
}

// TestOpenAIProvider_ResponsesEmptySchemaToolOnWire is the wire-level
// regression: an empty-schema tool routed to /responses must carry a
// "parameters" object in the request JSON (previously omitted → HTTP 400),
// and the endpoint's success answer is parsed normally.
func TestOpenAIProvider_ResponsesEmptySchemaToolOnWire(t *testing.T) {
	var requestTools []any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var req map[string]any
		if err := json.Unmarshal(raw, &req); err == nil {
			if tools, ok := req["tools"].([]any); ok {
				requestTools = tools
			}
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"resp_1","object":"response","status":"completed","output":[{"type":"message","id":"msg_1","role":"assistant","content":[{"type":"output_text","text":"ok"}]}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`)
	}))
	t.Cleanup(srv.Close)

	p, err := NewOpenAIProvider(OpenAIProviderConfig{Name: "p", APIKey: "k", BaseURL: srv.URL})
	if err != nil {
		t.Fatalf("NewOpenAIProvider: %v", err)
	}

	resp, err := p.ChatCompletion(context.Background(), ChatRequest{
		Model:    "gpt-5.6",
		Protocol: ProtocolResponses,
		Messages: []Message{{Role: "user", Content: "hi"}},
		Tools: []ToolDefinition{
			{Name: "no_schema_tool", Description: "tool with no schema"},
		},
	})
	if err != nil {
		t.Fatalf("ChatCompletion(responses): %v", err)
	}
	if len(requestTools) != 1 {
		t.Fatalf("request carried %d tools, want 1", len(requestTools))
	}
	tool, _ := requestTools[0].(map[string]any)
	params, ok := tool["parameters"].(map[string]any)
	if !ok {
		t.Fatalf("wire request omitted tools[0].parameters: %v", requestTools[0])
	}
	if params["type"] != "object" {
		t.Errorf("tools[0].parameters.type = %v, want object", params["type"])
	}
	if resp.Message.Content != "ok" {
		t.Errorf("content = %q, want ok", resp.Message.Content)
	}
}

// ---------------------------------------------------------------------------
// Finding #60 (Responses API): refusal parts ride message items' content and
// resp.OutputText() assembles output_text parts only, so a refusal or
// content-filtered answer (HTTP 200, no output_text) previously surfaced as a
// silent empty success.
// ---------------------------------------------------------------------------

func TestOpenAIProvider_ResponsesRefusalSurfaced(t *testing.T) {
	const refusalText = "I cannot comply with that request."

	t.Run("refusal-only response", func(t *testing.T) {
		body := `{"id":"resp_r","object":"response","status":"completed","output":[{"type":"message","id":"msg_1","role":"assistant","content":[{"type":"refusal","refusal":"` + refusalText + `"}]}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`
		srv := httptest.NewServer(staticResponsesHandler(t, body))
		t.Cleanup(srv.Close)
		p, err := NewOpenAIProvider(OpenAIProviderConfig{Name: "p", APIKey: "k", BaseURL: srv.URL})
		if err != nil {
			t.Fatalf("NewOpenAIProvider: %v", err)
		}

		resp, err := p.ChatCompletion(context.Background(), ChatRequest{
			Model:    "gpt-5.6",
			Protocol: ProtocolResponses,
			Messages: []Message{{Role: "user", Content: "hi"}},
		})
		if err != nil {
			t.Fatalf("ChatCompletion(responses): %v", err)
		}
		if resp.Message.Content != refusalText {
			t.Errorf("content = %q, want the refusal %q", resp.Message.Content, refusalText)
		}
		if resp.StopReason != "end_turn" {
			t.Errorf("stop reason = %q, want end_turn", resp.StopReason)
		}
	})

	t.Run("refusal does not override output_text", func(t *testing.T) {
		body := `{"id":"resp_r","object":"response","status":"completed","output":[{"type":"message","id":"msg_1","role":"assistant","content":[{"type":"output_text","text":"the answer"},{"type":"refusal","refusal":"` + refusalText + `"}]}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`
		srv := httptest.NewServer(staticResponsesHandler(t, body))
		t.Cleanup(srv.Close)
		p, err := NewOpenAIProvider(OpenAIProviderConfig{Name: "p", APIKey: "k", BaseURL: srv.URL})
		if err != nil {
			t.Fatalf("NewOpenAIProvider: %v", err)
		}

		resp, err := p.ChatCompletion(context.Background(), ChatRequest{
			Model:    "gpt-5.6",
			Protocol: ProtocolResponses,
			Messages: []Message{{Role: "user", Content: "hi"}},
		})
		if err != nil {
			t.Fatalf("ChatCompletion(responses): %v", err)
		}
		if resp.Message.Content != "the answer" {
			t.Errorf("content = %q, want %q (refusal must only fill an empty content)", resp.Message.Content, "the answer")
		}
	})
}

// TestOpenAIProvider_ResponsesStreamRefusalSurfaced verifies that refusal
// deltas (response.refusal.delta) are forwarded to the sink as text deltas and
// that a terminal response whose message item carries the refusal part
// assembles the refusal as the content.
func TestOpenAIProvider_ResponsesStreamRefusalSurfaced(t *testing.T) {
	terminal := `{"id":"resp_1","object":"response","status":"completed","output":[{"type":"message","id":"msg_1","role":"assistant","content":[{"type":"refusal","refusal":"I cannot comply"}]}],"usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		respSSE(w, `{"type":"response.created","response":{"id":"resp_1"}}`)
		respSSE(w, `{"type":"response.refusal.delta","item_id":"msg_1","output_index":0,"content_index":0,"delta":"I cannot "}`)
		respSSE(w, `{"type":"response.refusal.delta","item_id":"msg_1","output_index":0,"content_index":0,"delta":"comply"}`)
		respSSE(w, fmt.Sprintf(`{"type":"response.completed","response":%s}`, terminal))
	}))
	t.Cleanup(srv.Close)

	p, err := NewOpenAIProvider(OpenAIProviderConfig{Name: "p", APIKey: "k", BaseURL: srv.URL})
	if err != nil {
		t.Fatalf("NewOpenAIProvider: %v", err)
	}

	var deltas []StreamDelta
	resp, err := p.ChatCompletion(context.Background(), ChatRequest{
		Model:     "gpt-5.6",
		Protocol:  ProtocolResponses,
		Messages:  []Message{{Role: "user", Content: "hi"}},
		DeltaSink: func(d StreamDelta) error { deltas = append(deltas, d); return nil },
	})
	if err != nil {
		t.Fatalf("ChatCompletion(responses stream): %v", err)
	}

	if resp.Message.Content != "I cannot comply" {
		t.Errorf("content = %q, want %q", resp.Message.Content, "I cannot comply")
	}
	if len(deltas) != 2 || deltas[0].Text != "I cannot " || deltas[1].Text != "comply" {
		t.Errorf("deltas = %+v, want [I cannot , comply]", deltas)
	}
}

// TestOpenAIProvider_ResponsesStreamRefusalFallback covers the Codex-style
// backend shape: refusal deltas are streamed, but the terminal
// response.completed carries an EMPTY output list — the accumulated refusal
// payload must feed the streamed-output fallback so the message is not empty.
func TestOpenAIProvider_ResponsesStreamRefusalFallback(t *testing.T) {
	terminal := `{"id":"resp_2","object":"response","status":"completed","output":[],"usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		respSSE(w, `{"type":"response.created","response":{"id":"resp_2"}}`)
		respSSE(w, `{"type":"response.refusal.delta","item_id":"msg_1","output_index":0,"content_index":0,"delta":"I cannot "}`)
		respSSE(w, `{"type":"response.refusal.delta","item_id":"msg_1","output_index":0,"content_index":0,"delta":"comply"}`)
		respSSE(w, fmt.Sprintf(`{"type":"response.completed","response":%s}`, terminal))
	}))
	t.Cleanup(srv.Close)

	p, err := NewOpenAIProvider(OpenAIProviderConfig{Name: "p", APIKey: "k", BaseURL: srv.URL})
	if err != nil {
		t.Fatalf("NewOpenAIProvider: %v", err)
	}

	resp, err := p.ChatCompletion(context.Background(), ChatRequest{
		Model:     "gpt-5.6",
		Protocol:  ProtocolResponses,
		Messages:  []Message{{Role: "user", Content: "hi"}},
		DeltaSink: func(StreamDelta) error { return nil },
	})
	if err != nil {
		t.Fatalf("ChatCompletion(responses stream): %v", err)
	}
	if resp.Message.Content != "I cannot comply" {
		t.Errorf("content = %q, want the streamed refusal %q (empty terminal output must fall back to the refusal payload)", resp.Message.Content, "I cannot comply")
	}
}

// staticResponsesHandler answers every request with body (application/json).
func staticResponsesHandler(t *testing.T, body string) http.HandlerFunc {
	t.Helper()
	return func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, body)
	}
}
