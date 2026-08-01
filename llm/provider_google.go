package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// This file implements the Google Generative Language ("Gemini") generateContent
// protocol as a non-streaming delegate, mirroring how the OpenAI provider routes
// ProtocolResponses (→ /responses) and ProtocolAnthropic (→ AnthropicProvider).
//
// Google models speak POST {baseURL}/models/{model}:generateContent with a
// contents/parts request shape (see https://ai.google.dev/api/rest/v1beta/generateContent).
// It is intentionally a stateless function rather than a full Provider struct:
// the OpenAI-compatible gateway (e.g. Zen) exposes Gemini behind its own
// baseURL/apiKey/httpClient, so googleCompletion receives those values and
// issues a single HTTP call. There is no official Google SDK in the dependency
// graph, so the request/response are built and decoded with encoding/json over
// net/http — the same raw approach the codebase uses for error-body capture.

// googleGenerateRequest is the Google generateContent request body.
type googleGenerateRequest struct {
	SystemInstruction *googleContent          `json:"systemInstruction,omitempty"`
	Contents          []googleContent         `json:"contents"`
	Tools             []googleToolDecls       `json:"tools,omitempty"`
	GenerationConfig  *googleGenerationConfig `json:"generationConfig,omitempty"`
}

// googleContent is a single conversation turn: a role plus an ordered list of
// parts. Google uses "user" and "model" roles (assistant maps to "model").
type googleContent struct {
	Role  string       `json:"role,omitempty"`
	Parts []googlePart `json:"parts"`
}

// googlePart is a union: exactly one of Text / InlineData / FunctionCall /
// FunctionResp is populated. It is reused for both request construction and
// response decoding (a response part carries Text and/or FunctionCall).
type googlePart struct {
	Text         string                  `json:"text,omitempty"`
	InlineData   *googleInlineData       `json:"inlineData,omitempty"`
	FunctionCall *googleFunctionCall     `json:"functionCall,omitempty"`
	FunctionResp *googleFunctionResponse `json:"functionResponse,omitempty"`
}

type googleInlineData struct {
	MimeType string `json:"mimeType"`
	Data     string `json:"data"` // base64, without the "data:" prefix
}

type googleFunctionCall struct {
	Name string          `json:"name"`
	Args json.RawMessage `json:"args,omitempty"`
}

// googleFunctionResponse carries a tool result back to the model. Google
// identifies function responses by NAME (not by call ID), and requires the
// response payload to be a JSON object.
type googleFunctionResponse struct {
	Name     string          `json:"name"`
	Response json.RawMessage `json:"response"`
}

// googleToolDecls wraps a list of function declarations under the "tools" key.
type googleToolDecls struct {
	FunctionDeclarations []googleFunctionDeclaration `json:"functionDeclarations,omitempty"`
}

type googleFunctionDeclaration struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"` // JSON Schema; omitted when nil/empty
}

type googleGenerationConfig struct {
	MaxOutputTokens int      `json:"maxOutputTokens,omitempty"`
	Temperature     *float64 `json:"temperature,omitempty"`
}

// googleGenerateResponse is the Google generateContent response.
type googleGenerateResponse struct {
	Candidates    []googleCandidate    `json:"candidates"`
	UsageMetadata *googleUsageMetadata `json:"usageMetadata,omitempty"`
}

type googleCandidate struct {
	Content      googleContent `json:"content"`
	FinishReason string        `json:"finishReason"`
}

type googleUsageMetadata struct {
	PromptTokenCount     int `json:"promptTokenCount"`
	CandidatesTokenCount int `json:"candidatesTokenCount"`
	TotalTokenCount      int `json:"totalTokenCount"`
}

// googleFinishReasonMap maps Google finishReason values to our standard stop
// reason format. When the response carries function calls, mapGoogleFinishReason
// returns "tool_use" regardless of finishReason (mirroring the OpenAI Responses
// provider's behavior), since the executor keys tool execution off the presence
// of ToolCalls rather than off the stop reason.
var googleFinishReasonMap = map[string]string{
	"STOP":       "end_turn",
	"MAX_TOKENS": "max_tokens",
	"SAFETY":     "end_turn",
	"RECITATION": "end_turn",
	"OTHER":      "end_turn",
}

// googleCompletion performs a non-streaming Google Generative Language
// generateContent call. It is the delegate for ProtocolGoogle models (Gemini /
// Gemma) served by an OpenAI-compatible gateway (e.g. Zen) that reuses the same
// baseURL/apiKey/httpClient as the OpenAI provider.
//
// baseURL is the provider's configured base URL; the endpoint becomes
// "{baseURL}/models/{model}:generateContent". apiKey is sent as the "?key="
// query parameter rather than the x-goog-api-key header, because ?key= is the
// most broadly supported auth form across Google-compatible gateways; see the
// inline note at the request site for the logging redaction it implies.
// providerName is used to attribute errors. httpClient may be nil
// (→ http.DefaultClient).
func googleCompletion(ctx context.Context, httpClient *http.Client, baseURL, apiKey, providerName string, req ChatRequest) (*ChatResponse, error) {
	// Validate image blocks up front so a missing MediaType/ImageB64 yields a
	// clear local error instead of an opaque API 400.
	for _, msg := range req.Messages {
		if err := ValidateContentBlocks(msg.ContentBlocks); err != nil {
			return nil, fmt.Errorf("%s: %w", providerName, err)
		}
	}

	gReq := buildGoogleRequest(req)

	body, err := json.Marshal(gReq)
	if err != nil {
		return nil, fmt.Errorf("%s: failed to marshal google request: %w", providerName, err)
	}

	// Build the endpoint: {baseURL}/models/{model}:generateContent.
	endpoint := strings.TrimRight(baseURL, "/") + "/models/" + req.Model + ":generateContent"
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("%s: failed to build google request: %w", providerName, err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if apiKey != "" {
		// The API key is sent as the ?key= query parameter rather than the
		// x-goog-api-key header. Both are documented Google auth forms, but
		// ?key= is the most broadly supported across Google-compatible gateways
		// (e.g. OpenCode Zen), which may not forward custom headers. Trade-off:
		// the key is now embedded in the URL, so it can appear in gateway/proxy
		// access logs — any URL logging in this layer or upstream MUST redact
		// RawQuery and never log the full request URL verbatim.
		q := httpReq.URL.Query()
		q.Set("key", apiKey)
		httpReq.URL.RawQuery = q.Encode()
	}

	client := httpClient
	if client == nil {
		client = http.DefaultClient
	}

	resp, err := client.Do(httpReq)
	if err != nil {
		return nil, WrapProviderError(providerName, 0, err)
	}
	defer func() { _ = resp.Body.Close() }()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, WrapProviderError(providerName, resp.StatusCode, fmt.Errorf("google: read response body: %w", err))
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, WrapProviderError(providerName, resp.StatusCode,
			fmt.Errorf("google generateContent failed (HTTP %d): %s",
				resp.StatusCode, truncateForError(respBody)))
	}

	var genResp googleGenerateResponse
	if err := json.Unmarshal(respBody, &genResp); err != nil {
		return nil, fmt.Errorf("%s: failed to decode google response: %w (body: %s)",
			providerName, err, truncateForError(respBody))
	}

	return parseGoogleResponse(req.Model, &genResp, respBody, providerName)
}

// buildGoogleRequest converts a ChatRequest to the Google generateContent body.
func buildGoogleRequest(req ChatRequest) *googleGenerateRequest {
	// Extract the system prompt into systemInstruction.parts (Google has no
	// system role in contents; system text lives under systemInstruction).
	systemPrompt, filtered := ExtractSystemPrompt(req.Messages)

	// Build a ToolCallID → function-name index from assistant turns. Google
	// correlates a functionResponse with its functionCall by NAME, not by call
	// ID, so a tool result must report the function name — even though our
	// tool-result message carries a ToolCallID. For Google-sourced calls the
	// ID we assigned already equals the name (see parseGoogleResponse), but
	// this lookup also covers IDs produced by other layers (e.g. an executor
	// that mints its own opaque IDs) and makes the name resolution explicit.
	callIDToName := buildToolCallIDIndex(filtered)

	out := &googleGenerateRequest{
		Contents: make([]googleContent, 0, len(filtered)),
	}
	if systemPrompt != "" {
		out.SystemInstruction = &googleContent{
			Parts: []googlePart{{Text: systemPrompt}},
		}
	}

	for _, msg := range filtered {
		// Skip messages with no renderable content (Google rejects empty turns).
		if msg.Content == "" && len(msg.ContentBlocks) == 0 && len(msg.ToolCalls) == 0 && msg.ToolCallID == "" {
			continue
		}
		out.Contents = append(out.Contents, convertGoogleMessage(msg, callIDToName))
	}

	// Tools → a single tools[] entry whose functionDeclarations list each tool.
	// Schema is passed through best-effort (InputSchema is already JSON Schema).
	if len(req.Tools) > 0 {
		decls := make([]googleFunctionDeclaration, len(req.Tools))
		for i, tool := range req.Tools {
			decls[i] = googleFunctionDeclaration{
				Name:        tool.Name,
				Description: tool.Description,
				Parameters:  tool.InputSchema,
			}
		}
		out.Tools = []googleToolDecls{{FunctionDeclarations: decls}}
	}

	if req.MaxTokens > 0 || req.Temperature != nil {
		gc := &googleGenerationConfig{}
		if req.MaxTokens > 0 {
			gc.MaxOutputTokens = req.MaxTokens
		}
		gc.Temperature = req.Temperature
		out.GenerationConfig = gc
	}

	return out
}

// convertGoogleMessage maps a Message to a Google content turn. Roles: user
// stays "user"; assistant becomes "model"; a tool result becomes a "user" turn
// carrying a functionResponse part (Google has no dedicated tool role).
//
// callIDToName maps a ToolCall.ID to the function NAME; it is used to resolve
// the name a tool result must report in functionResponse.name, since Google
// correlates function responses by name rather than by call ID.
func convertGoogleMessage(msg Message, callIDToName map[string]string) googleContent {
	switch msg.Role {
	case "user":
		blocks := NormalizeContentBlocks(msg)
		parts := make([]googlePart, 0, len(blocks)+1)
		if blocks != nil {
			for _, blk := range blocks {
				switch blk.Type {
				case "text":
					parts = append(parts, googlePart{Text: blk.Text})
				case "image":
					parts = append(parts, googlePart{InlineData: &googleInlineData{
						MimeType: blk.MediaType,
						Data:     blk.ImageB64,
					}})
				}
			}
		} else if msg.Content != "" {
			parts = append(parts, googlePart{Text: msg.Content})
		}
		return googleContent{Role: "user", Parts: parts}

	case "assistant":
		parts := make([]googlePart, 0, len(msg.ToolCalls)+1)
		if msg.Content != "" {
			parts = append(parts, googlePart{Text: msg.Content})
		}
		for _, tc := range msg.ToolCalls {
			parts = append(parts, googlePart{FunctionCall: &googleFunctionCall{
				Name: tc.Name,
				Args: tc.Input,
			}})
		}
		return googleContent{Role: "model", Parts: parts}

	case "tool":
		// Google identifies a function response by the function NAME, not by
		// call ID. Resolve the name from the preceding assistant turn's tool
		// call (indexed in callIDToName); fall back to the ToolCallID itself
		// when no match is found (best-effort for the single-call case, where
		// Google-sourced IDs already equal the function name). The response
		// payload must be a JSON object: use the content verbatim when it is
		// already a JSON object, otherwise wrap it as {"result": ...}.
		name := msg.ToolCallID
		if n, ok := callIDToName[msg.ToolCallID]; ok {
			name = n
		}
		return googleContent{
			Role: "user",
			Parts: []googlePart{{
				FunctionResp: &googleFunctionResponse{
					Name:     name,
					Response: googleFunctionResponsePayload(msg.Content),
				},
			}},
		}

	default:
		// Unknown role — render as user text so the turn is not dropped.
		return googleContent{Role: "user", Parts: []googlePart{{Text: msg.Content}}}
	}
}

// buildToolCallIDIndex scans conversation messages for assistant tool calls and
// returns a map from ToolCall.ID to the function Name. It is used to resolve the
// function name a tool result must report back to Google's generateContent API,
// which correlates functionResponse with functionCall by name rather than by ID.
func buildToolCallIDIndex(msgs []Message) map[string]string {
	idx := make(map[string]string)
	for _, m := range msgs {
		for _, tc := range m.ToolCalls {
			if tc.ID != "" && tc.Name != "" {
				idx[tc.ID] = tc.Name
			}
		}
	}
	return idx
}

// googleFunctionResponsePayload coerces a tool result string into the JSON
// object Google requires for functionResponse.response.
func googleFunctionResponsePayload(content string) json.RawMessage {
	trimmed := bytes.TrimSpace([]byte(content))
	if len(trimmed) > 0 && trimmed[0] == '{' && json.Valid(trimmed) {
		return trimmed
	}
	wrapped, _ := json.Marshal(map[string]string{"result": content})
	return wrapped
}

// parseGoogleResponse converts a Google generateContent response to a ChatResponse.
// rawBody is the raw HTTP body, used only to enrich diagnostics when the response
// is empty or malformed (e.g. a non-compliant gateway returning a 200 with no
// candidates).
func parseGoogleResponse(model string, resp *googleGenerateResponse, rawBody []byte, providerName string) (*ChatResponse, error) {
	if len(resp.Candidates) == 0 {
		return nil, fmt.Errorf("%s: google response has no candidates (body: %s)",
			providerName, truncateForError(rawBody))
	}

	candidate := resp.Candidates[0]
	msg := Message{Role: "assistant"}
	hasToolCalls := false

	for _, part := range candidate.Content.Parts {
		if part.Text != "" {
			if msg.Content != "" {
				msg.Content += "\n"
			}
			msg.Content += part.Text
		}
		if part.FunctionCall != nil {
			hasToolCalls = true
			// Google has no per-call ID; use the function name as the ID so the
			// executor's tool-result correlation has a stable identifier (it is
			// echoed back as the functionResponse name on the next turn).
			msg.ToolCalls = append(msg.ToolCalls, ToolCall{
				ID:    part.FunctionCall.Name,
				Name:  part.FunctionCall.Name,
				Input: part.FunctionCall.Args,
			})
		}
	}

	// Degenerate-response guard: a 200 with neither text nor tool calls usually
	// indicates a non-compliant gateway or a wrong base_url; surface it as an
	// error so the failure is observable instead of a silent empty reply.
	if msg.Content == "" && !hasToolCalls {
		return nil, fmt.Errorf("%s: google response has no content (finishReason=%q); "+
			"verify base_url/model. Raw body: %s",
			providerName, candidate.FinishReason, truncateForError(rawBody))
	}

	usage := TokenUsage{}
	if resp.UsageMetadata != nil {
		usage.InputTokens = resp.UsageMetadata.PromptTokenCount
		usage.OutputTokens = resp.UsageMetadata.CandidatesTokenCount
	}

	return &ChatResponse{
		Model:      model,
		Message:    msg,
		StopReason: mapGoogleFinishReason(candidate.FinishReason, hasToolCalls),
		Usage:      usage,
	}, nil
}

// mapGoogleFinishReason converts a Google finishReason to the standard stop
// reason. When the response carries function calls it returns "tool_use"
// (mirroring the OpenAI Responses provider), since the executor keys tool
// execution off ToolCalls presence rather than the stop reason.
func mapGoogleFinishReason(reason string, hasToolCalls bool) string {
	if hasToolCalls {
		return "tool_use"
	}
	if reason == "" {
		return "end_turn"
	}
	if mapped, ok := googleFinishReasonMap[reason]; ok {
		return mapped
	}
	return reason
}
