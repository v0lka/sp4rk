package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"strings"
	"time"
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
	// Thought marks a chain-of-thought part (returned by gateways that enable
	// includeThoughts). Thought text is internal reasoning: parseGoogleResponse
	// routes it to the response's reasoning channel — never to the visible
	// Content — and it is deliberately NOT re-emitted on subsequent requests
	// (Google's own SDKs drop thought parts when replaying history).
	Thought bool `json:"thought,omitempty"`
	// ThoughtSignature carries the opaque thinking signature Gemini 3 attaches
	// to a part (alongside a functionCall). The API requires the signature to
	// be returned verbatim on the next request's matching part, even at minimal
	// thinking budget, or the request fails with 400 INVALID_ARGUMENT ("Function
	// call is missing a thought_signature"). parseGoogleResponse captures it on
	// the produced ToolCall (see googleThoughtSigSep) and convertGoogleMessage
	// re-emits it on the functionCall part of the following request.
	ThoughtSignature string `json:"thoughtSignature,omitempty"`
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
	// topP/topK are the Gemini sampling parameters; the API has no
	// presence_penalty/repetition_penalty equivalents, so those ChatRequest
	// fields are never serialized for Google.
	TopP *float64 `json:"topP,omitempty"`
	TopK *int     `json:"topK,omitempty"`
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

// maxGoogleResponseBytes caps how much of a generateContent response body is
// buffered into memory. The delegate decodes the whole body with encoding/json,
// so an unbounded read against a hostile or misbehaving endpoint (routed via a
// custom base URL) could exhaust host memory. 32 MiB comfortably covers the
// largest legitimate responses (long candidates plus inline base64 media).
const maxGoogleResponseBytes = 32 << 20

// googleDefaultHTTPClient bounds Google delegate calls when the host supplies
// no HTTP client. It replaces http.DefaultClient, whose zero timeout lets a
// stalled response hang googleCompletion (and its caller) forever. The generous
// wall clock accommodates long non-streaming generations; hosts that need
// different limits pass their own client via googleCompletionConfig.HTTPClient.
var googleDefaultHTTPClient = &http.Client{Timeout: 10 * time.Minute}

// googleCompletion performs a non-streaming Google Generative Language
// generateContent call. It is the delegate for ProtocolGoogle models (Gemini /
// Gemma) served by an OpenAI-compatible gateway (e.g. Zen) that reuses the same
// baseURL/apiKey/httpClient as the OpenAI provider.
//
// cfg.BaseURL is the provider's configured base URL; the endpoint becomes
// "{baseURL}/models/{model}:generateContent". cfg.APIKey is sent as the "?key="
// query parameter rather than the x-goog-api-key header, because ?key= is the
// most broadly supported auth form across Google-compatible gateways.

// googleCompletionConfig groups parameters for the internal googleCompletion
// function to prevent accidental positional argument swaps (e.g. baseURL ↔ apiKey).
type googleCompletionConfig struct {
	HTTPClient   *http.Client
	BaseURL      string
	APIKey       string
	ProviderName string
	Logger       *slog.Logger
	// TokenSource optionally supplies per-request bearer credentials (see
	// TokenSource). When set, the resolved access token overrides the
	// Authorization header and the static ?key= query parameter is dropped
	// from the URL, and a Token failure aborts the request before any
	// network I/O. nil = static APIKey only (the historical behavior).
	TokenSource TokenSource
}

// googleCompletion calls the Google Generative Language API (Gemini) and
// returns a unified ChatResponse.
// cfg.ProviderName is used to attribute errors. cfg.HTTPClient may be nil
// (→ googleDefaultHTTPClient, a timeout-bounded client).
func googleCompletion(ctx context.Context, cfg googleCompletionConfig, req ChatRequest) (*ChatResponse, error) {
	httpClient := cfg.HTTPClient
	if httpClient == nil {
		httpClient = googleDefaultHTTPClient
	}
	providerName := cfg.ProviderName
	baseURL := cfg.BaseURL
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
	if cfg.APIKey != "" {
		// The API key is sent as the ?key= query parameter rather than the
		// x-goog-api-key header. Both are documented Google auth forms, but
		// ?key= is the most broadly supported across Google-compatible gateways
		// (e.g. OpenCode Zen), which may not forward custom headers. Trade-off:
		// the key is now embedded in the URL, so it can appear in gateway/proxy
		// access logs — any URL logging in this layer or upstream MUST redact
		// RawQuery and never log the full request URL verbatim.
		q := httpReq.URL.Query()
		q.Set("key", cfg.APIKey)
		httpReq.URL.RawQuery = q.Encode()
	}
	// Dynamic credentials (TokenSource) are applied immediately before the
	// request leaves the process, mirroring tokenSourceMiddleware: the
	// resolved access token overrides the Authorization header, the static
	// ?key= parameter is dropped so a dynamic credential never rides
	// alongside a stale static key, and a Token failure aborts the call —
	// zero wire requests — with the "llm: token source:" cause preserved.
	if cfg.TokenSource != nil {
		tok, tokErr := cfg.TokenSource.Token(ctx)
		if tokErr != nil {
			return nil, WrapProviderError(providerName, 0, fmt.Errorf("llm: token source: %w", tokErr))
		}
		applyBearerToken(httpReq, tok, "")
		if tok.AccessToken != "" {
			q := httpReq.URL.Query()
			q.Del("key")
			httpReq.URL.RawQuery = q.Encode()
		}
	}

	resp, err := httpClient.Do(httpReq)
	if err != nil {
		// On a transport-level failure, httpClient.Do returns a *url.Error whose
		// Error() renders the full request URL — including the ?key= query
		// parameter carrying the API key. Propagating it raw would leak the secret
		// into error strings, logs, and UIs. Redact the query first; the underlying
		// net error is preserved via the cloned *url.Error's Unwrap chain, so retry
		// classification (errors.Is/As on net errors) still works.
		return nil, WrapProviderError(providerName, 0, redactKeyFromURLError(err))
	}
	defer func() { _ = resp.Body.Close() }()

	// Cap the buffered body (mirroring tiktoken_loader.go): read one extra
	// byte beyond the limit to distinguish "exactly maxGoogleResponseBytes"
	// (accepted) from "more" (rejected). Without the cap a hostile or broken
	// endpoint streaming an unbounded body would exhaust host memory.
	respBody, err := io.ReadAll(io.LimitReader(resp.Body, maxGoogleResponseBytes+1))
	if err != nil {
		return nil, WrapProviderError(providerName, resp.StatusCode, fmt.Errorf("google: read response body: %w", err))
	}
	if len(respBody) > maxGoogleResponseBytes {
		return nil, WrapProviderError(providerName, resp.StatusCode,
			fmt.Errorf("google: response body exceeds %d byte limit", maxGoogleResponseBytes))
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

	converted, err := parseGoogleResponse(req.Model, &genResp, respBody, providerName)
	if err != nil {
		return nil, err
	}
	logToolCallArguments(cfg.Logger, providerName, converted.Message.ToolCalls)
	return converted, nil
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

	for i := 0; i < len(filtered); {
		msg := filtered[i]
		if msg.Role == "tool" {
			// Bundle the maximal run of consecutive tool messages into ONE
			// "user" Content whose parts carry one functionResponse per call.
			// The generateContent API requires the function responses for a
			// function-call turn to be returned as a single Content turn whose
			// part count equals that turn's functionCall part count; emitting
			// one "user" turn per tool message yields a part-count mismatch
			// and a 400 INVALID_ARGUMENT on the first response part.
			var parts []googlePart
			for i < len(filtered) && filtered[i].Role == "tool" {
				toolMsg := filtered[i]
				i++
				// Skip messages with no renderable content (matches the
				// empty-turn filter below; Google rejects empty turns).
				if toolMsg.Content == "" && len(toolMsg.ContentBlocks) == 0 && toolMsg.ToolCallID == "" {
					continue
				}
				parts = append(parts, googleFunctionResponsePart(toolMsg, callIDToName))
			}
			if len(parts) > 0 {
				out.Contents = append(out.Contents, googleContent{Role: "user", Parts: parts})
			}
			continue
		}
		// Skip messages with no renderable content (Google rejects empty turns).
		if msg.Content == "" && len(msg.ContentBlocks) == 0 && len(msg.ToolCalls) == 0 && msg.ToolCallID == "" {
			i++
			continue
		}
		out.Contents = append(out.Contents, convertGoogleMessage(msg, callIDToName))
		i++
	}

	// Tools → a single tools[] entry whose functionDeclarations list each tool.
	// Schemas pass through SanitizeSchemaForGoogle: Google's Schema.type is a
	// single-valued proto enum, so a JSON-Schema type union ("type": ["array",
	// "string"], as declared by the in-tree fact tools) would be rejected with
	// 400 INVALID_ARGUMENT — the sanitizer collapses unions, inlines $ref and
	// strips keywords the Schema proto does not model.
	if len(req.Tools) > 0 {
		decls := make([]googleFunctionDeclaration, len(req.Tools))
		for i, tool := range req.Tools {
			decls[i] = googleFunctionDeclaration{
				Name:        tool.Name,
				Description: tool.Description,
				Parameters:  SanitizeSchemaForGoogle(tool.InputSchema),
			}
		}
		out.Tools = []googleToolDecls{{FunctionDeclarations: decls}}
	}

	if req.MaxTokens > 0 || req.Temperature != nil || req.TopP != nil || req.TopK != nil {
		gc := &googleGenerationConfig{}
		if req.MaxTokens > 0 {
			gc.MaxOutputTokens = req.MaxTokens
		}
		gc.Temperature = req.Temperature
		gc.TopP = req.TopP
		gc.TopK = req.TopK
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
//
// Note on grouping: a multi-tool-call turn's function responses must reach the
// API bundled in a single "user" Content; buildGoogleRequest merges each run
// of consecutive tool messages into one turn via googleFunctionResponsePart.
// This function (one tool message → one turn) remains for single-tool-result
// calls and direct per-message conversion.
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
			// Re-emit the Gemini 3 thought signature captured on the ToolCall
			// (empty for calls without one — omitempty keeps the part clean).
			// ReasoningContent is deliberately NOT re-emitted: thought parts
			// are internal reasoning and Google's SDKs drop them when
			// replaying history.
			parts = append(parts, googlePart{
				FunctionCall:     &googleFunctionCall{Name: tc.Name, Args: tc.Input},
				ThoughtSignature: googleThoughtSignatureFromID(tc.ID),
			})
		}
		return googleContent{Role: "model", Parts: parts}

	case "tool":
		// Google identifies a function response by the function NAME, not by
		// call ID (resolved in googleFunctionResponsePart).
		return googleContent{
			Role:  "user",
			Parts: []googlePart{googleFunctionResponsePart(msg, callIDToName)},
		}

	default:
		// Unknown role — render as user text so the turn is not dropped.
		return googleContent{Role: "user", Parts: []googlePart{{Text: msg.Content}}}
	}
}

// googleFunctionResponsePart builds the functionResponse part for one tool
// result message. Google identifies a function response by the function NAME,
// not by call ID: the name is resolved from the preceding assistant turn's
// tool call (indexed in callIDToName), falling back to the ToolCallID itself
// when no match is found (best-effort for the single-call case, where
// Google-sourced IDs already equal the function name). The response payload
// must be a JSON object: the content is used verbatim when it is already a
// JSON object, otherwise wrapped as {"result": ...}.
func googleFunctionResponsePart(msg Message, callIDToName map[string]string) googlePart {
	name := msg.ToolCallID
	if n, ok := callIDToName[msg.ToolCallID]; ok {
		name = n
	}
	return googlePart{
		FunctionResp: &googleFunctionResponse{
			Name:     name,
			Response: googleFunctionResponsePayload(msg.Content),
		},
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

// googleThoughtSigSep separates a function name from its round-tripped Gemini
// thought signature inside a ToolCall.ID. The neutral llm.ToolCall struct has
// no signature field, and the signature must survive the executor's message
// history to be re-emitted on the next request's functionCall part, so it rides
// in the ID. The unit separator cannot occur in function names (restricted to
// [a-zA-Z0-9-_.]) nor in Google's base64 signatures ([A-Za-z0-9+/=]), so the
// split is unambiguous; IDs of calls without a signature stay the bare function
// name (bit-identical to the historical format).
const googleThoughtSigSep = "\x1f"

// googleThoughtSignatureID returns the ToolCall.ID encoding name plus an
// optional thought signature (bare name when sig is empty).
func googleThoughtSignatureID(name, sig string) string {
	if sig == "" {
		return name
	}
	return name + googleThoughtSigSep + sig
}

// googleThoughtSignatureFromID extracts the thought signature encoded in a
// ToolCall.ID by googleThoughtSignatureID, or "" when the ID carries none.
func googleThoughtSignatureFromID(id string) string {
	_, sig, found := strings.Cut(id, googleThoughtSigSep)
	if !found {
		return ""
	}
	return sig
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
			if part.Thought {
				// A thought part is internal chain-of-thought: route it to
				// the reasoning channel, never to the visible answer (which
				// the executor would re-send and the user would see).
				if msg.ReasoningContent != "" {
					msg.ReasoningContent += "\n"
				}
				msg.ReasoningContent += part.Text
			} else {
				if msg.Content != "" {
					msg.Content += "\n"
				}
				msg.Content += part.Text
			}
		}
		if part.FunctionCall != nil {
			hasToolCalls = true
			// Google has no per-call ID; use the function name as the ID so the
			// executor's tool-result correlation has a stable identifier (it is
			// echoed back as the functionResponse name on the next turn). A
			// Gemini 3 thought signature rides behind the name (see
			// googleThoughtSigSep) so it survives to the next request.
			msg.ToolCalls = append(msg.ToolCalls, ToolCall{
				ID:    googleThoughtSignatureID(part.FunctionCall.Name, part.ThoughtSignature),
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
		Reasoning:  msg.ReasoningContent,
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

// redactKeyFromURLError strips the API key from a transport error's URL.
// googleCompletion sends the key as a ?key= query parameter; on a network
// failure http.Client.Do returns a *url.Error whose Error() string includes
// the full request URL (key and all). This clones such an error with the query
// removed so the secret never reaches logs or error displays. The underlying net
// error is preserved (the clone keeps the original Err, and *url.Error
// implements Unwrap), so retry classification via errors.Is/As still works.
func redactKeyFromURLError(err error) error {
	var ue *url.Error
	if errors.As(err, &ue) {
		redacted, perr := url.Parse(ue.URL)
		if perr == nil {
			redacted.RawQuery = ""
			return &url.Error{Op: ue.Op, URL: redacted.String(), Err: ue.Err}
		}
		// Unparseable URL: emit a fixed marker rather than risk echoing the
		// original (key-bearing) string.
		return &url.Error{Op: ue.Op, URL: "[redacted]", Err: ue.Err}
	}
	return err
}
