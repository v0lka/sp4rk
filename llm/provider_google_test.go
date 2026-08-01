package llm

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// TestGoogleCompletion_TextResponse verifies that googleCompletion POSTs to
// {baseURL}/models/{model}:generateContent with a Google contents/parts body,
// carries the API key as ?key=, and parses a text response into a ChatResponse.
func TestGoogleCompletion_TextResponse(t *testing.T) {
	var (
		gotPath    string
		gotQuery   string
		gotBody    []byte
		gotAuthHdr string
	)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotQuery = r.URL.RawQuery
		gotAuthHdr = r.Header.Get("x-goog-api-key")
		b, _ := io.ReadAll(r.Body)
		gotBody = b
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"role":"model","parts":[{"text":"hello from gemini"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":7,"totalTokenCount":17}}`))
	}))
	t.Cleanup(srv.Close)

	resp, err := googleCompletion(context.Background(), srv.Client(), srv.URL, "secret-key", "Zen", ChatRequest{
		Model:     "gemini-1.5-pro",
		MaxTokens: 100,
		Messages:  []Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("googleCompletion failed: %v", err)
	}

	// Endpoint must be the generateContent form.
	wantPath := "/models/gemini-1.5-pro:generateContent"
	if gotPath != wantPath {
		t.Errorf("expected path %q, got %q", wantPath, gotPath)
	}
	// API key carried as ?key= (Google's documented auth form).
	if !strings.Contains(gotQuery, "key=secret-key") {
		t.Errorf("expected query to contain key=secret-key, got %q", gotQuery)
	}
	_ = gotAuthHdr // x-goog-api-key header is not set; auth is via ?key=

	// Request body must be the Google contents/parts shape: a "contents" array,
	// "generationConfig", and the model name embedded in the endpoint path.
	var body map[string]any
	if err := json.Unmarshal(gotBody, &body); err != nil {
		t.Fatalf("failed to unmarshal request body: %v\nbody: %s", err, gotBody)
	}
	contents, ok := body["contents"].([]any)
	if !ok || len(contents) != 1 {
		t.Fatalf("expected contents array of length 1, got: %s", gotBody)
	}
	first, ok := contents[0].(map[string]any)
	if !ok {
		t.Fatalf("expected first content to be a map, got: %v", contents[0])
	}
	if first["role"] != "user" {
		t.Errorf("expected first content role \"user\", got %v", first["role"])
	}
	if gc, ok := body["generationConfig"].(map[string]any); ok {
		maxTok, ok := gc["maxOutputTokens"].(float64)
		if !ok || int(maxTok) != 100 {
			t.Errorf("expected maxOutputTokens=100, got %v", gc["maxOutputTokens"])
		}
	} else {
		t.Errorf("expected generationConfig in body, got: %s", gotBody)
	}

	// Response parsed into ChatResponse.
	if resp == nil {
		t.Fatal("expected non-nil ChatResponse")
	}
	if resp.Message.Content != "hello from gemini" {
		t.Errorf("expected content %q, got %q", "hello from gemini", resp.Message.Content)
	}
	if resp.StopReason != "end_turn" {
		t.Errorf("expected stop_reason end_turn, got %q", resp.StopReason)
	}
	if resp.Usage.InputTokens != 10 || resp.Usage.OutputTokens != 7 {
		t.Errorf("expected usage in=10 out=7, got in=%d out=%d", resp.Usage.InputTokens, resp.Usage.OutputTokens)
	}
}

// TestGoogleCompletion_SystemInstruction verifies a system message is hoisted
// into the top-level systemInstruction field rather than a contents entry.
func TestGoogleCompletion_SystemInstruction(t *testing.T) {
	var gotBody []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}]}`))
	}))
	t.Cleanup(srv.Close)

	_, err := googleCompletion(context.Background(), srv.Client(), srv.URL, "k", "Zen", ChatRequest{
		Model: "gemini-1.5-pro",
		Messages: []Message{
			{Role: "system", Content: "you are helpful"},
			{Role: "user", Content: "hi"},
		},
	})
	if err != nil {
		t.Fatalf("googleCompletion failed: %v", err)
	}

	var body map[string]any
	if err := json.Unmarshal(gotBody, &body); err != nil {
		t.Fatalf("failed to unmarshal request body: %v\nbody: %s", err, gotBody)
	}
	si, ok := body["systemInstruction"].(map[string]any)
	if !ok {
		t.Fatalf("expected systemInstruction in body, got: %s", gotBody)
	}
	parts, ok := si["parts"].([]any)
	if !ok || len(parts) == 0 {
		t.Fatalf("expected systemInstruction parts, got: %v", si["parts"])
	}
	firstPart, ok := parts[0].(map[string]any)
	if !ok || firstPart["text"] != "you are helpful" {
		t.Errorf("expected systemInstruction text, got: %v", parts)
	}
	contents, ok := body["contents"].([]any)
	if !ok || len(contents) != 1 {
		t.Errorf("expected contents to exclude the system message (len 1), got %v", body["contents"])
	}
}

// TestGoogleCompletion_ToolCallResponse verifies a functionCall part is parsed
// into a ToolCall and the stop reason becomes "tool_use".
func TestGoogleCompletion_ToolCallResponse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"get_weather","args":{"city":"London"}}}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":3,"totalTokenCount":13}}`))
	}))
	t.Cleanup(srv.Close)

	resp, err := googleCompletion(context.Background(), srv.Client(), srv.URL, "k", "Zen", ChatRequest{
		Model:    "gemini-1.5-pro",
		Messages: []Message{{Role: "user", Content: "weather in London?"}},
	})
	if err != nil {
		t.Fatalf("googleCompletion failed: %v", err)
	}

	if resp == nil || len(resp.Message.ToolCalls) != 1 {
		t.Fatalf("expected one tool call, got resp=%+v", resp)
	}
	tc := resp.Message.ToolCalls[0]
	if tc.Name != "get_weather" {
		t.Errorf("expected tool name get_weather, got %q", tc.Name)
	}
	var args map[string]string
	if err := json.Unmarshal(tc.Input, &args); err != nil {
		t.Fatalf("failed to unmarshal tool args: %v", err)
	}
	if args["city"] != "London" {
		t.Errorf("expected args city=London, got %q", args["city"])
	}
	if resp.StopReason != "tool_use" {
		t.Errorf("expected stop_reason tool_use, got %q", resp.StopReason)
	}
}

// TestGoogleCompletion_ToolResultUsesFunctionName verifies that a tool result
// is sent back as a functionResponse whose "name" is the function NAME (not the
// ToolCallID). Google correlates functionResponse with functionCall by name, so
// an opaque call ID would cause a 400 "function response name not found". This
// test uses a ToolCallID that deliberately differs from the function name.
func TestGoogleCompletion_ToolResultUsesFunctionName(t *testing.T) {
	var gotBody []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}]}`))
	}))
	t.Cleanup(srv.Close)

	_, err := googleCompletion(context.Background(), srv.Client(), srv.URL, "k", "Zen", ChatRequest{
		Model: "gemini-1.5-pro",
		Messages: []Message{
			{Role: "user", Content: "weather in London?"},
			// Assistant turn issued a tool call with an opaque ID distinct from
			// the function name.
			{
				Role: "assistant",
				ToolCalls: []ToolCall{{
					ID:    "call_abc_123",
					Name:  "get_weather",
					Input: json.RawMessage(`{"city":"London"}`),
				}},
			},
			// Tool result echoes the opaque ToolCallID (as an executor would).
			{Role: "tool", ToolCallID: "call_abc_123", Content: `{"temp":"21C"}`},
			{Role: "user", Content: "thanks"},
		},
	})
	if err != nil {
		t.Fatalf("googleCompletion failed: %v", err)
	}

	var body map[string]any
	if err := json.Unmarshal(gotBody, &body); err != nil {
		t.Fatalf("failed to unmarshal request body: %v\nbody: %s", err, gotBody)
	}
	contents, ok := body["contents"].([]any)
	if !ok {
		t.Fatalf("expected contents array, got: %s", gotBody)
	}

	// Find the functionResponse turn (a "user" role carrying a functionResponse part).
	var fnRespName string
	for _, c := range contents {
		turn, ok := c.(map[string]any)
		if !ok {
			continue
		}
		parts, ok := turn["parts"].([]any)
		if !ok {
			continue
		}
		for _, p := range parts {
			part, ok := p.(map[string]any)
			if !ok {
				continue
			}
			if fr, ok := part["functionResponse"].(map[string]any); ok {
				fnRespName, _ = fr["name"].(string)
			}
		}
	}
	if fnRespName != "get_weather" {
		t.Errorf("functionResponse.name = %q, want %q (function name, not ToolCallID %q)",
			fnRespName, "get_weather", "call_abc_123")
	}
}

// TestGoogleCompletion_ToolsDeclared verifies request tools become a tools[]
// entry with functionDeclarations.
func TestGoogleCompletion_ToolsDeclared(t *testing.T) {
	var gotBody []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"role":"model","parts":[{"text":"no call"}]},"finishReason":"STOP"}]}`))
	}))
	t.Cleanup(srv.Close)

	_, err := googleCompletion(context.Background(), srv.Client(), srv.URL, "k", "Zen", ChatRequest{
		Model:    "gemini-1.5-pro",
		Messages: []Message{{Role: "user", Content: "hi"}},
		Tools: []ToolDefinition{{
			Name:        "search",
			Description: "search the web",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}}}`),
		}},
	})
	if err != nil {
		t.Fatalf("googleCompletion failed: %v", err)
	}

	var body map[string]any
	if err := json.Unmarshal(gotBody, &body); err != nil {
		t.Fatalf("failed to unmarshal request body: %v\nbody: %s", err, gotBody)
	}
	tools, ok := body["tools"].([]any)
	if !ok || len(tools) != 1 {
		t.Fatalf("expected tools array of length 1, got: %s", gotBody)
	}
	toolEntry, ok := tools[0].(map[string]any)
	if !ok {
		t.Fatalf("expected tools[0] to be a map, got: %v", tools[0])
	}
	decls, ok := toolEntry["functionDeclarations"].([]any)
	if !ok || len(decls) == 0 {
		t.Fatalf("expected functionDeclarations, got: %v", toolEntry["functionDeclarations"])
	}
	fd, ok := decls[0].(map[string]any)
	if !ok {
		t.Fatalf("expected decls[0] to be a map, got: %v", decls[0])
	}
	if fd["name"] != "search" {
		t.Errorf("expected function name search, got %v", fd["name"])
	}
	if fd["description"] != "search the web" {
		t.Errorf("expected function description, got %v", fd["description"])
	}
}

// TestGoogleCompletion_ErrorWrappedWithProviderName verifies a non-2xx response
// surfaces an error wrapped with the provider name and status code.
func TestGoogleCompletion_ErrorWrappedWithProviderName(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"error":{"code":500,"message":"internal"}}`))
	}))
	t.Cleanup(srv.Close)

	_, err := googleCompletion(context.Background(), srv.Client(), srv.URL, "k", "Zen", ChatRequest{
		Model:    "gemini-1.5-pro",
		Messages: []Message{{Role: "user", Content: "hi"}},
	})
	if err == nil {
		t.Fatal("expected an error from the failed google request, got nil")
	}
	if !strings.Contains(err.Error(), "Zen") {
		t.Errorf("expected error wrapped with provider name \"Zen\", got: %v", err)
	}
	if !strings.Contains(err.Error(), "500") {
		t.Errorf("expected error to mention HTTP 500, got: %v", err)
	}
	var llmErr *Error
	if !errors.As(err, &llmErr) {
		t.Errorf("expected a classified *llm.Error, got %T: %v", err, err)
	} else if llmErr.StatusCode != 500 {
		t.Errorf("expected status code 500, got %d", llmErr.StatusCode)
	}
}

// TestGoogleCompletion_EmptyCandidatesReturnsError verifies the degenerate
// response guard surfaces a clear error instead of a silent empty reply.
func TestGoogleCompletion_EmptyCandidatesReturnsError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"candidates":[]}`))
	}))
	t.Cleanup(srv.Close)

	_, err := googleCompletion(context.Background(), srv.Client(), srv.URL, "k", "Zen", ChatRequest{
		Model:    "gemini-1.5-pro",
		Messages: []Message{{Role: "user", Content: "hi"}},
	})
	if err == nil {
		t.Fatal("expected an error for empty candidates, got nil")
	}
	if !strings.Contains(err.Error(), "no candidates") {
		t.Errorf("expected error mentioning no candidates, got: %v", err)
	}
}

// TestOpenAIProvider_GoogleDelegate_RoutesToGenerateContent verifies that a
// Gemini model served by an OpenAI-compatible gateway (e.g. Zen) under the
// ProtocolGoogle dispatch path is delegated to googleCompletion: the request
// POSTs to the gateway's {baseURL}/models/{model}:generateContent endpoint
// (NOT /chat/completions), with a Google contents/parts body, and the response
// is parsed into a ChatResponse. The delegation reuses googleCompletion — no
// Google code lives on the OpenAI provider.
func TestOpenAIProvider_GoogleDelegate_RoutesToGenerateContent(t *testing.T) {
	var (
		gotPath string
		gotBody []byte
	)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"role":"model","parts":[{"text":"gemini says hi"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":4,"totalTokenCount":9}}`))
	}))
	t.Cleanup(srv.Close)

	p, err := NewOpenAIProvider(OpenAIProviderConfig{
		Name:    "Zen",
		APIKey:  "k",
		BaseURL: srv.URL,
		Logger:  slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	if err != nil {
		t.Fatalf("NewOpenAIProvider failed: %v", err)
	}

	resp, err := p.ChatCompletion(context.Background(), ChatRequest{
		Model:    "gemini-1.5-pro",
		Messages: []Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("ChatCompletion failed: %v", err)
	}

	// Must hit the Google generateContent endpoint, NOT the OpenAI chat endpoint.
	wantPath := "/models/gemini-1.5-pro:generateContent"
	if gotPath != wantPath {
		t.Errorf("expected Gemini model to POST to %q, got path %q", wantPath, gotPath)
	}
	if strings.Contains(gotPath, "/chat/completions") {
		t.Errorf("Gemini model must NOT POST to /chat/completions, got path %q", gotPath)
	}

	// Body must be Google contents/parts format (a "contents" array), not OpenAI.
	var body map[string]any
	if err := json.Unmarshal(gotBody, &body); err != nil {
		t.Fatalf("failed to unmarshal request body: %v\nbody: %s", err, gotBody)
	}
	if _, ok := body["contents"]; !ok {
		t.Errorf("expected Google body to contain \"contents\", got: %s", gotBody)
	}
	if _, ok := body["messages"]; ok {
		t.Errorf("Google body must NOT contain OpenAI \"messages\", got: %s", gotBody)
	}

	// Response parsed into ChatResponse.
	if resp == nil {
		t.Fatal("expected non-nil ChatResponse")
	}
	if resp.Message.Content != "gemini says hi" {
		t.Errorf("expected parsed content %q, got %q", "gemini says hi", resp.Message.Content)
	}
	if resp.Usage.OutputTokens != 4 {
		t.Errorf("expected output tokens 4, got %d", resp.Usage.OutputTokens)
	}
}

// TestOpenAIProvider_GoogleDelegate_ErrorWrappedWithProviderName verifies that
// when the delegated Google path fails, the error is wrapped with the provider
// name (inherited from the OpenAIProvider config) so it is observable.
func TestOpenAIProvider_GoogleDelegate_ErrorWrappedWithProviderName(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadGateway)
		_, _ = w.Write([]byte(`{"error":{"code":502,"message":"bad gateway"}}`))
	}))
	t.Cleanup(srv.Close)

	p, err := NewOpenAIProvider(OpenAIProviderConfig{
		Name:    "Zen",
		APIKey:  "k",
		BaseURL: srv.URL,
		Logger:  slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	if err != nil {
		t.Fatalf("NewOpenAIProvider failed: %v", err)
	}

	_, err = p.ChatCompletion(context.Background(), ChatRequest{
		Model:    "gemini-1.5-pro",
		Messages: []Message{{Role: "user", Content: "hi"}},
	})
	if err == nil {
		t.Fatal("expected an error from the failed Google request, got nil")
	}
	if !strings.Contains(err.Error(), "Zen") {
		t.Errorf("expected error to be wrapped with provider name \"Zen\", got: %v", err)
	}
}
