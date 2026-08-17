package llm

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"regexp"
	"strings"

	"github.com/liushuangls/go-anthropic/v2"
)

// defaultAnthropicMaxTokens is the fallback max_tokens sent to the Anthropic
// Messages API when the caller does not specify one. The Anthropic API
// requires max_tokens to be present and > 0 — omitting it (or sending 0)
// results in a 400 "Missing key ['max_tokens']" error. Several callers
// build ChatRequests without MaxTokens, relying on the provider to supply a
// safe default. 8192 is the minimum OutputLimit across all Anthropic models
// in the built-in registry, so it is accepted by every supported model.
const defaultAnthropicMaxTokens = 8192

// anthropicToolIDPattern matches characters not allowed in Anthropic tool call IDs.
// Anthropic only allows [a-zA-Z0-9_-] in tool call IDs.
var anthropicToolIDPattern = regexp.MustCompile(`[^a-zA-Z0-9_-]`)

// sanitizeAnthropicToolID ensures tool call IDs only contain characters allowed by Anthropic API.
func sanitizeAnthropicToolID(id string) string {
	return anthropicToolIDPattern.ReplaceAllString(id, "_")
}

// AnthropicProviderConfig holds configuration for Anthropic provider.
type AnthropicProviderConfig struct {
	Name       string // logical provider name ("anthropic" default; custom name for Anthropic-compatible providers)
	APIKey     string
	BaseURL    string       // empty = default Anthropic; otherwise custom endpoint (Anthropic-compatible proxy)
	HTTPClient *http.Client // optional proxy-configured HTTP client (nil = default)
	Logger     *slog.Logger // optional structured logger (nil = slog.Default())
}

// AnthropicProvider implements LLM Provider using Anthropic's Claude API.
type AnthropicProvider struct {
	client *anthropic.Client
	name   string
	logger *slog.Logger
}

// log returns the provider's logger, defaulting to slog.Default() when unset.
func (p *AnthropicProvider) log() *slog.Logger {
	if p.logger != nil {
		return p.logger
	}
	return slog.Default()
}

// NewAnthropicProvider creates a new Anthropic provider with the given configuration.
//
// If BaseURL is empty, uses the default Anthropic endpoint; otherwise uses the
// custom endpoint (an Anthropic-compatible proxy or gateway).
//
// Note: APIKey is intentionally not validated here. The official Anthropic API
// always requires a key, but local Anthropic-compatible servers may not. An
// empty key for the official endpoint fails at call time with a 401, consistent
// with NewOpenAIProvider's handling of local OpenAI-compatible backends.
func NewAnthropicProvider(cfg AnthropicProviderConfig) (*AnthropicProvider, error) {
	var opts []anthropic.ClientOption
	if cfg.BaseURL != "" {
		opts = append(opts, anthropic.WithBaseURL(normalizeAnthropicBaseURL(cfg.BaseURL)))
	}

	// Always install a response-body-capturing transport. The go-anthropic SDK
	// only surfaces errors for non-2xx HTTP status codes; some
	// Anthropic-compatible endpoints return an error object (or a degenerate
	// empty body) with HTTP 200, which the SDK then silently decodes into an
	// empty MessagesResponse. Capturing the raw body lets parseResponse include
	// it in a descriptive error so such failures are observable instead of
	// surfacing as a silent empty reply. The raw body is read into memory on
	// every response and re-wrapped so the SDK can still decode it: this is a
	// transient ~1× body-size allocation per call (acceptable for the
	// non-streaming CreateMessages path), not a zero-overhead path. The
	// provided HTTP client is cloned (not mutated) so any shared proxy/TLS/
	// timeout configuration is preserved and other consumers of the same client
	// are unaffected.
	httpClient := &http.Client{}
	if cfg.HTTPClient != nil {
		*httpClient = *cfg.HTTPClient
	}
	// The go-anthropic SDK sends the API key in the x-api-key header, which
	// Go does NOT strip on cross-host redirects (only Authorization, Cookie,
	// and Www-Authenticate are). Refuse to follow redirects so the key is
	// never forwarded to a redirect target — a redirect from a /messages
	// endpoint is a misconfiguration or attack, not normal operation. This
	// guard applies unconditionally, including to a caller-supplied client
	// (whose fields are cloned above, never mutated): the credential-leak
	// threat is identical regardless of who supplied the client, and a custom
	// proxy/gateway pointing at an Anthropic-compatible endpoint is precisely
	// where an unexpected redirect is most dangerous.
	httpClient.CheckRedirect = func(*http.Request, []*http.Request) error {
		return http.ErrUseLastResponse
	}
	httpClient.Transport = &capturingTransport{base: httpClient.Transport}
	opts = append(opts, anthropic.WithHTTPClient(httpClient))

	client := anthropic.NewClient(cfg.APIKey, opts...)

	name := cfg.Name
	if name == "" {
		name = "anthropic"
	}

	return &AnthropicProvider{
		client: client,
		name:   name,
		logger: cfg.Logger,
	}, nil
}

// normalizeAnthropicBaseURL ensures the configured base URL ends with "/v1".
//
// The go-anthropic SDK treats the base URL as already including the API version
// path — its built-in default is "https://api.anthropic.com/v1" and it appends
// only "/messages" to produce ".../v1/messages". Anthropic-compatible endpoints
// are conventionally documented with a base URL that EXCLUDES "/v1" (e.g.
// Z.AI's "https://api.z.ai/api/anthropic", matching the ANTHROPIC_BASE_URL
// convention used by the official Anthropic SDK, which appends "/v1/messages").
// Passing such a URL through unchanged makes message calls hit the wrong path
// ("/api/anthropic/messages" instead of "/api/anthropic/v1/messages"), which
// the endpoint answers with a 200 and an empty/non-standard body — the SDK
// returns no error and the provider sees a silently empty response.
//
// URLs that already end with "/v1" (with or without a trailing slash) are left
// untouched, so callers that follow the go-anthropic convention keep working.
func normalizeAnthropicBaseURL(base string) string {
	trimmed := strings.TrimRight(base, "/")
	if strings.HasSuffix(trimmed, "/v1") {
		return trimmed
	}
	return trimmed + "/v1"
}

// bodyCaptureCtxKey is the context key under which ChatCompletion stashes a
// *[]byte that the capturingTransport fills with the raw response body.
type bodyCaptureCtxKey struct{}

// capturingTransport is an http.RoundTripper that reads the full response body,
// copies it into a per-request buffer (when the request context carries one),
// and re-wraps it so the underlying SDK can still decode it.
type capturingTransport struct {
	base http.RoundTripper
}

func (t *capturingTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	base := t.base
	if base == nil {
		base = http.DefaultTransport
	}
	resp, err := base.RoundTrip(req)
	if err != nil {
		// Per the http.RoundTripper contract, the response is undefined (and the
		// http.Client will not close it) when err != nil. A transport that
		// nevertheless returns a non-nil response here would leak its body, so
		// close and discard it.
		if resp != nil {
			_ = resp.Body.Close()
		}
		return nil, err
	}
	body, readErr := io.ReadAll(resp.Body)
	_ = resp.Body.Close() // fully read above; close error is not actionable
	if readErr != nil {
		return nil, fmt.Errorf("anthropic: failed to read response body: %w", readErr)
	}
	if holder, ok := req.Context().Value(bodyCaptureCtxKey{}).(*[]byte); ok && holder != nil {
		*holder = body
	}
	resp.Body = io.NopCloser(bytes.NewReader(body))
	return resp, nil
}

// truncateForError returns a trimmed, length-limited view of b suitable for
// embedding in an error message.
func truncateForError(b []byte) string {
	const maxLen = 2048
	s := strings.TrimSpace(string(b))
	if len(s) > maxLen {
		return s[:maxLen] + " …(truncated)"
	}
	return s
}

// Name returns the provider name.
func (p *AnthropicProvider) Name() string {
	return p.name
}

// ChatCompletion sends a request and returns the full response.
func (p *AnthropicProvider) ChatCompletion(ctx context.Context, req ChatRequest) (*ChatResponse, error) {
	anthropicReq, err := p.buildRequest(req)
	if err != nil {
		return nil, fmt.Errorf("anthropic: failed to build request: %w", err)
	}

	// Stash a buffer the capturingTransport fills with the raw response body,
	// so parseResponse can embed it in an error if the endpoint returns a
	// non-standard or error response with HTTP 200.
	var capturedBody []byte
	ctx = context.WithValue(ctx, bodyCaptureCtxKey{}, &capturedBody)

	resp, err := p.client.CreateMessages(ctx, *anthropicReq)
	if err != nil {
		return nil, p.wrapError(fmt.Errorf("anthropic: API error: %w", err))
	}

	return p.parseResponse(resp, capturedBody)
}

// buildRequest converts ChatRequest to anthropic.MessagesRequest.
func (p *AnthropicProvider) buildRequest(req ChatRequest) (*anthropic.MessagesRequest, error) {
	// Validate image blocks up front so a missing MediaType/ImageB64 yields a
	// clear local error instead of an opaque API 400.
	for _, msg := range req.Messages {
		if err := ValidateContentBlocks(msg.ContentBlocks); err != nil {
			return nil, fmt.Errorf("anthropic: %w", err)
		}
	}
	// Extract system prompt parts from messages (preserves multi-part for caching)
	systemParts, filteredMsgs := ExtractSystemPromptParts(req.Messages)
	var messages []anthropic.Message

	for _, msg := range filteredMsgs {
		// Skip messages with no renderable content (Anthropic API rejects empty
		// messages). ContentBlocks are included so an image-only user message
		// (empty Content, non-empty ContentBlocks) is not silently dropped.
		// ReasoningContent is also checked so an assistant message carrying only
		// reasoning is not silently dropped.
		if msg.Content == "" && len(msg.ContentBlocks) == 0 && len(msg.ToolCalls) == 0 && msg.ToolCallID == "" && msg.ReasoningContent == "" {
			continue
		}
		anthropicMsg, err := p.convertMessage(msg)
		if err != nil {
			return nil, err
		}
		messages = append(messages, anthropicMsg)
	}

	anthropicReq := &anthropic.MessagesRequest{
		Model:     anthropic.Model(req.Model),
		Messages:  messages,
		MaxTokens: req.MaxTokens,
	}
	if anthropicReq.MaxTokens <= 0 {
		anthropicReq.MaxTokens = defaultAnthropicMaxTokens
	}

	// Set system prompt: use MultiSystem with cache control when multiple parts exist
	if len(systemParts) > 1 {
		multiSystem := make([]anthropic.MessageSystemPart, len(systemParts))
		for i, part := range systemParts {
			multiSystem[i] = anthropic.MessageSystemPart{
				Type: "text",
				Text: part,
			}
			// Mark all parts except the last as cacheable (stable content)
			if i < len(systemParts)-1 {
				multiSystem[i].CacheControl = &anthropic.MessageCacheControl{
					Type: anthropic.CacheControlTypeEphemeral,
				}
			}
		}
		anthropicReq.MultiSystem = multiSystem
	} else if len(systemParts) == 1 {
		anthropicReq.System = systemParts[0]
	}

	if req.Temperature != nil {
		temp := float32(*req.Temperature)
		anthropicReq.Temperature = &temp
	}
	// Anthropic Messages API sampling parameters (float32 in the SDK).
	// presence_penalty/repetition_penalty are OpenAI-style controls the
	// Anthropic API does not accept — they are never serialized here.
	if req.TopP != nil {
		topP := float32(*req.TopP)
		anthropicReq.TopP = &topP
	}
	if req.TopK != nil {
		topK := *req.TopK
		anthropicReq.TopK = &topK
	}

	// Apply reasoning effort: "On" enables thinking with budget 32000.
	// The Anthropic API requires max_tokens to be strictly greater than
	// thinking.budget_tokens, and the budget itself must be >= 1024. Clamp
	// the budget for small max_tokens values (e.g. the 8192 fallback) and
	// skip thinking entirely when no valid budget fits.
	if req.ReasoningEffort == "On" {
		budget := 32000
		if anthropicReq.MaxTokens <= budget {
			budget = anthropicReq.MaxTokens / 2
		}
		if budget >= 1024 {
			anthropicReq.Thinking = &anthropic.Thinking{
				Type:         anthropic.ThinkingTypeEnabled,
				BudgetTokens: budget,
			}
			// Anthropic requires temperature to be unset (or 1.0) when
			// thinking is enabled, and equally rejects top_p/top_k overrides
			// in extended-thinking mode, so all sampling knobs are dropped.
			anthropicReq.Temperature = nil
			anthropicReq.TopP = nil
			anthropicReq.TopK = nil
		}
	}

	// Convert tools
	if len(req.Tools) > 0 {
		tools := make([]anthropic.ToolDefinition, len(req.Tools))
		for i, tool := range req.Tools {
			tools[i] = anthropic.ToolDefinition{
				Name:        tool.Name,
				Description: tool.Description,
				InputSchema: SanitizeSchemaForAnthropic(tool.InputSchema),
			}
		}
		anthropicReq.Tools = tools
	}

	return anthropicReq, nil
}

// convertMessage converts a Message to anthropic.Message.
func (p *AnthropicProvider) convertMessage(msg Message) (anthropic.Message, error) {
	switch msg.Role {
	case "user":
		// When ContentBlocks are present, render them as structured content
		// (text and/or image blocks) instead of the plain Content string.
		// NormalizeContentBlocks prepends Content as a text block when the
		// blocks carry no text, so the task text always reaches the model.
		// Text-only messages without ContentBlocks keep the existing path.
		blocks := NormalizeContentBlocks(msg)
		if blocks != nil {
			content := make([]anthropic.MessageContent, 0, len(blocks))
			for _, block := range blocks {
				switch block.Type {
				case "text":
					content = append(content, anthropic.NewTextMessageContent(block.Text))
				case "image":
					content = append(content, anthropic.NewImageMessageContent(anthropic.MessageContentSource{
						Type:      anthropic.MessagesContentSourceTypeBase64,
						MediaType: block.MediaType,
						Data:      block.ImageB64,
					}))
				default:
					// Unknown block types are skipped (consistent with other
					// providers); log at debug so misconfigured callers can
					// diagnose silently dropped content.
					p.log().Debug("anthropic: skipping unknown content block type",
						"block_type", block.Type, "provider", p.name)
				}
			}
			return anthropic.Message{
				Role:    anthropic.RoleUser,
				Content: content,
			}, nil
		}
		return anthropic.Message{
			Role: anthropic.RoleUser,
			Content: []anthropic.MessageContent{
				anthropic.NewTextMessageContent(msg.Content),
			},
		}, nil

	case "assistant":
		var content []anthropic.MessageContent

		// Add text content if present
		if msg.Content != "" {
			content = append(content, anthropic.NewTextMessageContent(msg.Content))
		}

		// Add tool use blocks for tool calls
		for _, tc := range msg.ToolCalls {
			content = append(content, anthropic.NewToolUseMessageContent(sanitizeAnthropicToolID(tc.ID), tc.Name, tc.Input))
		}

		return anthropic.Message{
			Role:    anthropic.RoleAssistant,
			Content: content,
		}, nil

	case "tool":
		return anthropic.Message{
			Role: anthropic.RoleUser,
			Content: []anthropic.MessageContent{
				anthropic.NewToolResultMessageContent(sanitizeAnthropicToolID(msg.ToolCallID), msg.Content, false),
			},
		}, nil

	default:
		return anthropic.Message{}, fmt.Errorf("unsupported message role: %s", msg.Role)
	}
}

// parseResponse converts anthropic.MessagesResponse to ChatResponse.
//
// rawBody is the raw HTTP response body captured by the transport. It is used
// only to enrich diagnostics when the endpoint returns a non-standard response
// (e.g. an Anthropic-compatible endpoint that answers a 200 with an error
// object or an empty body); a nil/empty value is fine for callers that capture
// the response directly (tests).
func (p *AnthropicProvider) parseResponse(resp anthropic.MessagesResponse, rawBody []byte) (*ChatResponse, error) {
	// The go-anthropic SDK only treats non-2xx HTTP statuses as errors. Some
	// Anthropic-compatible endpoints return an explicit {"type":"error",...}
	// object WITH HTTP 200, which the SDK happily decodes into a zero-value
	// MessagesResponse (empty content, empty stop_reason, zero usage) and
	// returns as success. Detect that here so the caller sees a real error
	// instead of a silent empty reply.
	if resp.Type == anthropic.MessagesResponseTypeError {
		detail := truncateForError(rawBody)
		if detail == "" {
			detail = "(no response body captured)"
		}
		return nil, fmt.Errorf("anthropic: endpoint %q returned an error response: %s",
			p.name, detail)
	}

	message := Message{
		Role: "assistant",
	}

	var reasoning string

	// Process content blocks
	for _, block := range resp.Content {
		switch block.Type {
		case anthropic.MessagesContentTypeText:
			if message.Content != "" {
				message.Content += "\n"
			}
			message.Content += block.GetText()

		case anthropic.MessagesContentTypeThinking:
			// Extended thinking content block
			if block.MessageContentThinking != nil {
				if reasoning != "" {
					reasoning += "\n"
				}
				reasoning += block.Thinking
			}

		case anthropic.MessagesContentTypeToolUse:
			if block.MessageContentToolUse != nil {
				message.ToolCalls = append(message.ToolCalls, ToolCall{
					ID:    block.ID,
					Name:  block.Name,
					Input: block.Input,
				})
			}
		}
	}

	// Degenerate response guard: the SDK succeeded (no error) but the response
	// carries neither text content nor tool calls. A well-formed Anthropic
	// Messages response always has a non-empty stop_reason and at least one
	// content block, so this combination indicates a non-compliant
	// Anthropic-compatible endpoint (most often a misconfigured base URL that
	// misses the "/v1" path segment, causing the endpoint to return a 200 with
	// an empty or unrecognized body). Surface it as an error so callers and
	// operators can diagnose it instead of seeing a silent empty result.
	if message.Content == "" && len(message.ToolCalls) == 0 {
		detail := truncateForError(rawBody)
		if detail == "" {
			detail = "(no response body captured)"
		}
		return nil, fmt.Errorf("anthropic: endpoint %q returned a 200 response with no content "+
			"(stop_reason=%q) — this usually indicates a non-compliant Anthropic-compatible "+
			"endpoint; verify the base_url includes the correct API path. Raw body: %s",
			p.name, resp.StopReason, detail)
	}

	return &ChatResponse{
		Message:    message,
		Reasoning:  reasoning,
		StopReason: string(resp.StopReason),
		Usage: TokenUsage{
			InputTokens:  resp.Usage.InputTokens,
			OutputTokens: resp.Usage.OutputTokens,
		},
	}, nil
}

// wrapError maps Anthropic SDK error types to *Error.
func (p *AnthropicProvider) wrapError(err error) error {
	var apiErr *anthropic.APIError
	if errors.As(err, &apiErr) {
		retryable := apiErr.IsRateLimitErr() || apiErr.IsOverloadedErr() || apiErr.IsApiErr()
		return NewError(p.name, 0, retryable, err)
	}
	var reqErr *anthropic.RequestError
	if errors.As(err, &reqErr) {
		return WrapProviderError(p.name, reqErr.StatusCode, err)
	}
	return WrapProviderError(p.name, 0, err)
}
