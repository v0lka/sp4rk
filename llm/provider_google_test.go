package llm

import (
	"bytes"
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

	resp, err := googleCompletion(context.Background(), googleCompletionConfig{HTTPClient: srv.Client(), BaseURL: srv.URL, APIKey: "secret-key", ProviderName: "Zen"}, ChatRequest{
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

	_, err := googleCompletion(context.Background(), googleCompletionConfig{HTTPClient: srv.Client(), BaseURL: srv.URL, APIKey: "k", ProviderName: "Zen"}, ChatRequest{
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

	resp, err := googleCompletion(context.Background(), googleCompletionConfig{HTTPClient: srv.Client(), BaseURL: srv.URL, APIKey: "k", ProviderName: "Zen"}, ChatRequest{
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

	_, err := googleCompletion(context.Background(), googleCompletionConfig{HTTPClient: srv.Client(), BaseURL: srv.URL, APIKey: "k", ProviderName: "Zen"}, ChatRequest{
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

	_, err := googleCompletion(context.Background(), googleCompletionConfig{HTTPClient: srv.Client(), BaseURL: srv.URL, APIKey: "k", ProviderName: "Zen"}, ChatRequest{
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

	_, err := googleCompletion(context.Background(), googleCompletionConfig{HTTPClient: srv.Client(), BaseURL: srv.URL, APIKey: "k", ProviderName: "Zen"}, ChatRequest{
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

	_, err := googleCompletion(context.Background(), googleCompletionConfig{HTTPClient: srv.Client(), BaseURL: srv.URL, APIKey: "k", ProviderName: "Zen"}, ChatRequest{
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

// TestGoogleCompletion_TransportErrorRedactsAPIKey verifies that when the HTTP
// call fails at the transport layer (connection refused etc.), the propagated
// error does NOT leak the API key, which is sent as a ?key= query parameter.
// Go's *url.Error renders the full request URL (query included) in its Error()
// string, so without redaction the key would flow into logs and error displays.
// Retry classification must be preserved (the underlying net error is kept).
func TestGoogleCompletion_TransportErrorRedactsAPIKey(t *testing.T) {
	// Port 1 reliably refuses connections at the transport layer (no LLM round
	// trip), producing the *url.Error path exercised by this regression test.
	_, err := googleCompletion(context.Background(), googleCompletionConfig{
		HTTPClient:   &http.Client{},
		BaseURL:      "http://127.0.0.1:1",
		APIKey:       "SECRET-API-KEY-12345",
		ProviderName: "Zen",
	}, ChatRequest{
		Model:    "gemini-1.5-pro",
		Messages: []Message{{Role: "user", Content: "hi"}},
	})
	if err == nil {
		t.Fatal("expected a transport error, got nil")
	}
	if strings.Contains(err.Error(), "SECRET-API-KEY-12345") {
		t.Errorf("API key leaked into error string: %v", err)
	}
	// The error must still be a classified *llm.Error mentioning the provider.
	var llmErr *Error
	if !errors.As(err, &llmErr) {
		t.Fatalf("expected a classified *llm.Error, got %T: %v", err, err)
	}
	if !strings.Contains(err.Error(), "Zen") {
		t.Errorf("expected error wrapped with provider name \"Zen\", got: %v", err)
	}
	// Retry classification must survive redaction: connection-refused is a
	// transient net error, so the cloned *url.Error's Unwrap chain must still
	// resolve to syscall.ECONNREFUSED.
	if !llmErr.Retryable {
		t.Errorf("expected connection-refused classified retryable, got retryable=false: %v", err)
	}
}

// TestGoogleCompletion_ResponseSizeCapped verifies the response body read is
// bounded: an endpoint streaming more than maxGoogleResponseBytes must fail
// with a size-limit error instead of buffering the whole body in memory
// (unbounded read → OOM of the host process).
func TestGoogleCompletion_ResponseSizeCapped(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		// Stream more than the cap; the client stops reading after the limit,
		// so the trailing write error is expected and ignored.
		_, _ = w.Write(bytes.Repeat([]byte("a"), maxGoogleResponseBytes+1024))
	}))
	t.Cleanup(srv.Close)

	_, err := googleCompletion(context.Background(), googleCompletionConfig{HTTPClient: srv.Client(), BaseURL: srv.URL, APIKey: "k", ProviderName: "Zen"}, ChatRequest{
		Model:    "gemini-1.5-pro",
		Messages: []Message{{Role: "user", Content: "hi"}},
	})
	if err == nil {
		t.Fatal("expected an error for an over-limit response body, got nil")
	}
	if !strings.Contains(err.Error(), "exceeds") {
		t.Errorf("expected error to mention the size limit, got: %v", err)
	}
	var llmErr *Error
	if !errors.As(err, &llmErr) {
		t.Errorf("expected a classified *llm.Error, got %T: %v", err, err)
	}
}

// TestGoogleCompletion_DefaultHTTPClientHasTimeout verifies the no-client
// fallback is timeout-bounded: http.DefaultClient has no timeout, so a stalled
// response would hang googleCompletion forever.
func TestGoogleCompletion_DefaultHTTPClientHasTimeout(t *testing.T) {
	if googleDefaultHTTPClient.Timeout <= 0 {
		t.Errorf("googleDefaultHTTPClient.Timeout = %v, want > 0 (fallback client must be bounded)", googleDefaultHTTPClient.Timeout)
	}
}

// TestGoogleCompletion_MultiToolCallResultsGroupedIntoSingleTurn verifies that
// the function responses of a multi-tool-call turn are bundled into ONE "user"
// Content turn with one functionResponse part per call. The generateContent API
// requires the response part count to equal the call turn's functionCall part
// count; one "user" turn per tool message is rejected with 400 INVALID_ARGUMENT.
func TestGoogleCompletion_MultiToolCallResultsGroupedIntoSingleTurn(t *testing.T) {
	var gotBody []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"role":"model","parts":[{"text":"done"}]},"finishReason":"STOP"}]}`))
	}))
	t.Cleanup(srv.Close)

	_, err := googleCompletion(context.Background(), googleCompletionConfig{HTTPClient: srv.Client(), BaseURL: srv.URL, APIKey: "k", ProviderName: "Zen"}, ChatRequest{
		Model: "gemini-1.5-pro",
		Messages: []Message{
			{Role: "user", Content: "do two things"},
			{Role: "assistant", ToolCalls: []ToolCall{
				{ID: "call_a", Name: "get_weather", Input: json.RawMessage(`{"city":"London"}`)},
				{ID: "call_b", Name: "get_time", Input: json.RawMessage(`{"tz":"GMT"}`)},
			}},
			{Role: "tool", ToolCallID: "call_a", Content: `{"temp":"21C"}`},
			// An empty tool message carries nothing renderable and must be
			// skipped without breaking the group or emitting an empty turn.
			{Role: "tool"},
			{Role: "tool", ToolCallID: "call_b", Content: `{"now":"12:00"}`},
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
	// Expected turns: user, model (2 functionCalls), user (2 grouped
	// functionResponses), user ("thanks") — the trailing user turn must NOT be
	// merged into the grouped tool turn.
	if len(contents) != 4 {
		t.Fatalf("expected 4 contents turns, got %d: %s", len(contents), gotBody)
	}
	grouped, ok := contents[2].(map[string]any)
	if !ok {
		t.Fatalf("expected contents[2] to be a map, got: %v", contents[2])
	}
	if grouped["role"] != "user" {
		t.Errorf("expected grouped turn role \"user\", got %v", grouped["role"])
	}
	parts, ok := grouped["parts"].([]any)
	if !ok || len(parts) != 2 {
		t.Fatalf("expected grouped turn with 2 functionResponse parts, got: %v", grouped["parts"])
	}
	wantNames := []string{"get_weather", "get_time"}
	for i, want := range wantNames {
		part, ok := parts[i].(map[string]any)
		if !ok {
			t.Fatalf("expected parts[%d] to be a map, got: %v", i, parts[i])
		}
		fr, ok := part["functionResponse"].(map[string]any)
		if !ok {
			t.Fatalf("expected parts[%d] to carry a functionResponse, got: %v", i, part)
		}
		if fr["name"] != want {
			t.Errorf("grouped parts[%d].functionResponse.name = %v, want %q", i, fr["name"], want)
		}
		if _, hasResp := fr["response"]; !hasResp {
			t.Errorf("grouped parts[%d].functionResponse missing response payload: %v", i, fr)
		}
	}
	last, ok := contents[3].(map[string]any)
	if !ok {
		t.Fatalf("expected contents[3] to be a map, got: %v", contents[3])
	}
	lastParts, ok := last["parts"].([]any)
	if !ok || len(lastParts) != 1 {
		t.Fatalf("expected trailing user turn with one part, got: %v", last["parts"])
	}
	if tp, ok := lastParts[0].(map[string]any); !ok || tp["text"] != "thanks" {
		t.Errorf("expected trailing user turn text \"thanks\", got: %v", lastParts[0])
	}
}

// TestGoogleCompletion_ToolSchemaSanitizedForGoogle verifies the request-side
// tool schemas pass through SanitizeSchemaForGoogle: a JSON-array type (the
// in-tree store_fact/search_facts shape) must collapse to a single
// Google-modelled type, $refs must inline, and keywords the Schema proto does
// not model must be stripped — otherwise generateContent rejects the request
// with 400 INVALID_ARGUMENT.
func TestGoogleCompletion_ToolSchemaSanitizedForGoogle(t *testing.T) {
	var gotBody []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}]}`))
	}))
	t.Cleanup(srv.Close)

	_, err := googleCompletion(context.Background(), googleCompletionConfig{HTTPClient: srv.Client(), BaseURL: srv.URL, APIKey: "k", ProviderName: "Zen"}, ChatRequest{
		Model:    "gemini-1.5-pro",
		Messages: []Message{{Role: "user", Content: "hi"}},
		Tools: []ToolDefinition{{
			Name:        "store_fact",
			Description: "store a fact",
			InputSchema: json.RawMessage(`{
				"type": "object",
				"properties": {
					"keywords": {"type": ["array", "string"], "items": {"type": "string"}},
					"maybe":    {"type": ["string", "null"]},
					"ptr":      {"$ref": "#/$defs/thing"}
				},
				"required": ["keywords"],
				"$defs": {"thing": {"type": "string"}},
				"additionalProperties": false,
				"strict": true
			}`),
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
		t.Fatalf("expected one tools entry, got: %s", gotBody)
	}
	toolEntry, ok := tools[0].(map[string]any)
	if !ok {
		t.Fatalf("expected tools[0] to be a map, got: %v", tools[0])
	}
	decls, ok := toolEntry["functionDeclarations"].([]any)
	if !ok || len(decls) != 1 {
		t.Fatalf("expected one functionDeclaration, got: %s", gotBody)
	}
	decl, ok := decls[0].(map[string]any)
	if !ok {
		t.Fatalf("expected decls[0] to be a map, got: %v", decls[0])
	}
	params, ok := decl["parameters"].(map[string]any)
	if !ok {
		t.Fatalf("expected parameters object, got: %s", gotBody)
	}
	if params["type"] != "object" {
		t.Errorf("parameters.type = %v, want \"object\"", params["type"])
	}
	for _, banned := range []string{"additionalProperties", "strict", "$defs", "$ref"} {
		if _, present := params[banned]; present {
			t.Errorf("parameters must not carry %q after sanitization, got: %v", banned, params[banned])
		}
	}
	props, ok := params["properties"].(map[string]any)
	if !ok {
		t.Fatalf("expected properties object, got: %v", params["properties"])
	}
	keywords, ok := props["keywords"].(map[string]any)
	if !ok {
		t.Fatalf("expected keywords property object, got: %v", props["keywords"])
	}
	if keywords["type"] != "string" {
		t.Errorf("keywords.type = %v, want \"string\" (type union must collapse)", keywords["type"])
	}
	if _, hasItems := keywords["items"]; hasItems {
		t.Errorf("keywords.items must be dropped for a scalar type, got: %v", keywords["items"])
	}
	maybe, ok := props["maybe"].(map[string]any)
	if !ok {
		t.Fatalf("expected maybe property object, got: %v", props["maybe"])
	}
	if maybe["type"] != "string" {
		t.Errorf("maybe.type = %v, want \"string\"", maybe["type"])
	}
	if maybe["nullable"] != true {
		t.Errorf("maybe.nullable = %v, want true (dropped \"null\" member must survive as nullable)", maybe["nullable"])
	}
	ptr, ok := props["ptr"].(map[string]any)
	if !ok {
		t.Fatalf("expected ptr property object, got: %v", props["ptr"])
	}
	if ptr["type"] != "string" {
		t.Errorf("ptr.type = %v, want \"string\" ($ref must inline)", ptr["type"])
	}
	if _, hasRef := ptr["$ref"]; hasRef {
		t.Errorf("ptr must not carry $ref after inlining, got: %v", ptr)
	}
	required, ok := params["required"].([]any)
	if !ok || len(required) != 1 || required[0] != "keywords" {
		t.Errorf("required = %v, want [keywords]", params["required"])
	}
}

// TestGoogleCompletion_ThoughtSignatureRoundTrip verifies the Gemini 3 thought
// signature round trip end to end over the wire:
//  1. a response carrying a thought part and a thoughtSignature-bearing
//     functionCall maps the thought to the reasoning channel (not Content) and
//     captures the signature on the ToolCall;
//  2. the follow-up request re-emits the signature verbatim on the
//     functionCall part of the model turn — required by the API, which answers
//     400 INVALID_ARGUMENT ("Function call is missing a thought_signature")
//     otherwise — without re-emitting any thought part.
func TestGoogleCompletion_ThoughtSignatureRoundTrip(t *testing.T) {
	var (
		calls   int
		gotBody []byte
	)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		body, _ := io.ReadAll(r.Body)
		if calls == 1 {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"candidates":[{"content":{"role":"model","parts":[` +
				`{"thought":true,"text":"internal scratch"},` +
				`{"text":"Calling the tool now"},` +
				`{"functionCall":{"name":"get_weather","args":{"city":"London"}},"thoughtSignature":"sigABC=="}` +
				`]},"finishReason":"STOP"}]}`))
			return
		}
		gotBody = body
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"role":"model","parts":[{"text":"21C"}]},"finishReason":"STOP"}]}`))
	}))
	t.Cleanup(srv.Close)
	cfg := googleCompletionConfig{HTTPClient: srv.Client(), BaseURL: srv.URL, APIKey: "k", ProviderName: "Zen"}

	// Phase 1: parse a signature-bearing, thought-carrying response.
	resp, err := googleCompletion(context.Background(), cfg, ChatRequest{
		Model:    "gemini-3-flash",
		Messages: []Message{{Role: "user", Content: "weather in London?"}},
	})
	if err != nil {
		t.Fatalf("googleCompletion (phase 1) failed: %v", err)
	}
	if resp.Message.ReasoningContent != "internal scratch" {
		t.Errorf("Message.ReasoningContent = %q, want %q (thought part must map to the reasoning channel)", resp.Message.ReasoningContent, "internal scratch")
	}
	if resp.Reasoning != "internal scratch" {
		t.Errorf("Reasoning = %q, want %q", resp.Reasoning, "internal scratch")
	}
	if resp.Message.Content != "Calling the tool now" {
		t.Errorf("Content = %q, want %q (thought text must not leak into Content)", resp.Message.Content, "Calling the tool now")
	}
	if len(resp.Message.ToolCalls) != 1 {
		t.Fatalf("expected one tool call, got %d", len(resp.Message.ToolCalls))
	}
	tc := resp.Message.ToolCalls[0]
	wantID := "get_weather" + "\x1f" + "sigABC=="
	if tc.ID != wantID {
		t.Errorf("ToolCall.ID = %q, want %q (signature must ride the ID for the round trip)", tc.ID, wantID)
	}

	// Phase 2: replay the turn — the executor echoes the assistant message and
	// the tool result back; the model turn must carry the signature verbatim.
	_, err = googleCompletion(context.Background(), cfg, ChatRequest{
		Model: "gemini-3-flash",
		Messages: []Message{
			{Role: "user", Content: "weather in London?"},
			{Role: "assistant", Content: resp.Message.Content, ToolCalls: resp.Message.ToolCalls},
			{Role: "tool", ToolCallID: tc.ID, Content: `{"temp":"21C"}`},
		},
	})
	if err != nil {
		t.Fatalf("googleCompletion (phase 2) failed: %v", err)
	}

	var body map[string]any
	if err := json.Unmarshal(gotBody, &body); err != nil {
		t.Fatalf("failed to unmarshal follow-up request body: %v\nbody: %s", err, gotBody)
	}
	contents, ok := body["contents"].([]any)
	if !ok || len(contents) != 3 {
		t.Fatalf("expected 3 contents turns, got: %s", gotBody)
	}
	modelTurn, ok := contents[1].(map[string]any)
	if !ok || modelTurn["role"] != "model" {
		t.Fatalf("expected contents[1] to be the model turn, got: %v", contents[1])
	}
	parts, ok := modelTurn["parts"].([]any)
	if !ok || len(parts) != 2 {
		t.Fatalf("expected model turn with 2 parts (text + functionCall), got: %v", modelTurn["parts"])
	}
	callPart, ok := parts[1].(map[string]any)
	if !ok {
		t.Fatalf("expected parts[1] to be a map, got: %v", parts[1])
	}
	fc, ok := callPart["functionCall"].(map[string]any)
	if !ok || fc["name"] != "get_weather" {
		t.Fatalf("expected parts[1].functionCall get_weather, got: %v", callPart)
	}
	if callPart["thoughtSignature"] != "sigABC==" {
		t.Errorf("functionCall part thoughtSignature = %v, want \"sigABC==\" (must round-trip verbatim)", callPart["thoughtSignature"])
	}
	if _, hasThought := callPart["thought"]; hasThought {
		t.Errorf("replayed functionCall part must not carry \"thought\": true, got: %v", callPart)
	}
	textPart, ok := parts[0].(map[string]any)
	if !ok {
		t.Fatalf("expected parts[0] to be a map, got: %v", parts[0])
	}
	if _, hasThought := textPart["thought"]; hasThought {
		t.Errorf("replayed model turn must not re-emit thought parts, got: %v", parts[0])
	}
	// The composite ToolCallID must still resolve to the function name.
	toolTurn, ok := contents[2].(map[string]any)
	if !ok {
		t.Fatalf("expected contents[2] to be a map, got: %v", contents[2])
	}
	toolParts, ok := toolTurn["parts"].([]any)
	if !ok || len(toolParts) != 1 {
		t.Fatalf("expected one grouped functionResponse turn with one part, got: %v", toolTurn["parts"])
	}
	toolPart, ok := toolParts[0].(map[string]any)
	if !ok {
		t.Fatalf("expected toolParts[0] to be a map, got: %v", toolParts[0])
	}
	fr, ok := toolPart["functionResponse"].(map[string]any)
	if !ok || fr["name"] != "get_weather" {
		t.Errorf("functionResponse.name = %v, want \"get_weather\" (composite ID must resolve to the name)", toolPart)
	}
}

// TestGoogleCompletion_ThoughtlessToolCallKeepsPlainID verifies that a tool
// call response WITHOUT a thoughtSignature keeps the historical ID format (the
// bare function name) and that the replayed functionCall part carries no
// thoughtSignature key at all — the fix must not perturb Gemini 1.5/2.x runs.
func TestGoogleCompletion_ThoughtlessToolCallKeepsPlainID(t *testing.T) {
	var gotBody []byte
	calls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		body, _ := io.ReadAll(r.Body)
		if calls == 1 {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"get_weather","args":{"city":"London"}}}]},"finishReason":"STOP"}]}`))
			return
		}
		gotBody = body
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}]}`))
	}))
	t.Cleanup(srv.Close)
	cfg := googleCompletionConfig{HTTPClient: srv.Client(), BaseURL: srv.URL, APIKey: "k", ProviderName: "Zen"}

	resp, err := googleCompletion(context.Background(), cfg, ChatRequest{
		Model:    "gemini-1.5-pro",
		Messages: []Message{{Role: "user", Content: "weather?"}},
	})
	if err != nil {
		t.Fatalf("googleCompletion failed: %v", err)
	}
	if len(resp.Message.ToolCalls) != 1 {
		t.Fatalf("expected one tool call, got %d", len(resp.Message.ToolCalls))
	}
	if id := resp.Message.ToolCalls[0].ID; id != "get_weather" {
		t.Errorf("ToolCall.ID = %q, want bare function name %q when no signature is present", id, "get_weather")
	}

	_, err = googleCompletion(context.Background(), cfg, ChatRequest{
		Model: "gemini-1.5-pro",
		Messages: []Message{
			{Role: "user", Content: "weather?"},
			{Role: "assistant", ToolCalls: resp.Message.ToolCalls},
			{Role: "tool", ToolCallID: resp.Message.ToolCalls[0].ID, Content: `{"temp":"21C"}`},
		},
	})
	if err != nil {
		t.Fatalf("googleCompletion (replay) failed: %v", err)
	}
	if !strings.Contains(string(gotBody), `"functionCall"`) {
		t.Fatalf("expected a functionCall part in the replayed body, got: %s", gotBody)
	}
	if strings.Contains(string(gotBody), "thoughtSignature") {
		t.Errorf("replayed functionCall part must omit thoughtSignature when none was captured, got: %s", gotBody)
	}
}

// TestSanitizeSchemaForGoogle unit-tests the Google schema sanitizer's
// rewrites: type-union collapse (with nullable preservation), composition
// flattening, unsupported-keyword stripping, $ref inlining and passthrough of
// unparsable input.
func TestSanitizeSchemaForGoogle(t *testing.T) {
	tests := []struct {
		name string
		raw  string
		want string // exact expected JSON, or "" when only shape checks apply
	}{
		{
			name: "type union array+string collapses to string and drops items",
			raw:  `{"type":["array","string"],"items":{"type":"string"}}`,
			want: `{"type":"string"}`,
		},
		{
			name: "type union string+null collapses to string plus nullable",
			raw:  `{"type":["string","null"]}`,
			want: `{"type":"string","nullable":true}`,
		},
		{
			name: "type union integer+null keeps integer",
			raw:  `{"type":["integer","null"]}`,
			want: `{"type":"integer","nullable":true}`,
		},
		{
			name: "null-only union falls back to string plus nullable",
			raw:  `{"type":["null"]}`,
			want: `{"type":"string","nullable":true}`,
		},
		{
			name: "multi non-null union without string keeps first declared member",
			raw:  `{"type":["number","integer"]}`,
			want: `{"type":"number"}`,
		},
		{
			name: "oneOf is renamed to anyOf",
			raw:  `{"oneOf":[{"type":"string"},{"type":"integer"}]}`,
			want: `{"anyOf":[{"type":"string"},{"type":"integer"}]}`,
		},
		{
			name: "single-item allOf is unwrapped into the parent",
			raw:  `{"type":"object","allOf":[{"properties":{"a":{"type":"string"}}}]}`,
			want: `{"type":"object","properties":{"a":{"type":"string"}}}`,
		},
		{
			name: "unsupported keywords are stripped recursively",
			raw:  `{"type":"object","strict":true,"additionalProperties":false,"properties":{"q":{"type":"string","const":"x","uniqueItems":true}}}`,
			want: `{"type":"object","properties":{"q":{"type":"string"}}}`,
		},
		{
			name: "$ref inlines against $defs and $defs is removed",
			raw:  `{"type":"object","properties":{"p":{"$ref":"#/$defs/thing"}},"$defs":{"thing":{"type":"string"}}}`,
			want: `{"type":"object","properties":{"p":{"type":"string"}}}`,
		},
		{
			name: "object type is inferred when properties are present",
			raw:  `{"properties":{"a":{"type":"string"}}}`,
			want: `{"type":"object","properties":{"a":{"type":"string"}}}`,
		},
		{
			name: "valid Google-compatible schema is preserved",
			raw:  `{"type":"object","properties":{"a":{"type":"string","enum":["x"]}},"required":["a"]}`,
			want: `{"type":"object","properties":{"a":{"type":"string","enum":["x"]}},"required":["a"]}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := SanitizeSchemaForGoogle(json.RawMessage(tt.raw))
			var gotVal, wantVal any
			if err := json.Unmarshal(got, &gotVal); err != nil {
				t.Fatalf("output is not valid JSON: %v (%s)", err, got)
			}
			if tt.want == "" {
				return
			}
			if err := json.Unmarshal([]byte(tt.want), &wantVal); err != nil {
				t.Fatalf("test bug: want is not valid JSON: %v", err)
			}
			// Compare structurally: re-marshaling through any sorts map keys,
			// so byte equality of the normalized forms is a fair comparison.
			gotNorm, err := json.Marshal(gotVal)
			if err != nil {
				t.Fatalf("re-marshal failed: %v", err)
			}
			wantNorm, err := json.Marshal(wantVal)
			if err != nil {
				t.Fatalf("re-marshal failed: %v", err)
			}
			if !bytes.Equal(gotNorm, wantNorm) {
				t.Fatalf("SanitizeSchemaForGoogle() = %s, want %s", gotNorm, wantNorm)
			}
		})
	}

	// Passthrough guarantees: unparsable input is returned unchanged.
	rawBad := json.RawMessage(`{"type": `)
	if got := SanitizeSchemaForGoogle(rawBad); string(got) != string(rawBad) {
		t.Errorf("SanitizeSchemaForGoogle(unparsable) = %s, want unchanged %s", got, rawBad)
	}
	if got := SanitizeSchemaForGoogle(nil); got != nil {
		t.Errorf("SanitizeSchemaForGoogle(nil) = %s, want nil", got)
	}
	if got := SanitizeSchemaForGoogle(json.RawMessage{}); len(got) != 0 {
		t.Errorf("SanitizeSchemaForGoogle(empty) = %s, want empty", got)
	}
}
