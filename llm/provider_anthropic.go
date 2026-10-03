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
	"sync"

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
	// TokenSource optionally supplies per-request bearer credentials (see
	// TokenSource). The go-anthropic SDK has no middleware hook, so the
	// credentials are applied through a wrapping http.RoundTripper installed
	// on the provider's client: the resolved access token overrides the
	// Authorization header and clears the SDK's static x-api-key, and a
	// Token failure aborts the request before any bytes reach the wire.
	// nil = static APIKey only (the historical behavior).
	TokenSource TokenSource
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
	// surfacing as a silent empty reply. The transport tee-wraps the response
	// body (instead of reading it into memory up front) so it is captured as
	// the SDK consumes it: streaming responses stay incremental (deltas reach
	// the DeltaSink as they arrive off the wire) and the synchronous path pays
	// the same transient ~1× body-size allocation as before. A body the SDK
	// abandons early (e.g. a cancelled stream) captures partially — enough for
	// diagnostics. The provided HTTP client is cloned (not mutated) so any
	// shared proxy/TLS/timeout configuration is preserved and other consumers
	// of the same client are unaffected.
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
	// Dynamic credentials (TokenSource) are applied beneath the capturing
	// transport: every captured body corresponds to a request that already
	// carries the resolved bearer (and no stale static x-api-key), and a
	// Token failure aborts before any network I/O.
	transportBase := httpClient.Transport
	if cfg.TokenSource != nil {
		transportBase = &tokenSourceRoundTripper{
			ts:           cfg.TokenSource,
			next:         transportBase,
			staticHeader: "x-api-key",
		}
	}
	httpClient.Transport = &capturingTransport{base: transportBase}
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
// *capturedBody that the capturingTransport fills with the raw response body.
type bodyCaptureCtxKey struct{}

// capturedBody accumulates the raw response bytes while the SDK consumes the
// response and hands them to the caller once the body is fully read (or
// abandoned early — then what was read so far). ChatCompletion owns one
// instance per call and shares it with the transport through the request
// context, so the caller can inspect the body after either the synchronous or
// the streaming path completes.
type capturedBody struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

// write appends bytes read off the wire.
func (c *capturedBody) write(p []byte) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.buf.Write(p)
}

// bytes returns everything captured so far. It is safe to call concurrently
// with writes; the returned slice is a copy so later reads of the stream
// cannot mutate it under the caller.
func (c *capturedBody) bytes() []byte {
	c.mu.Lock()
	defer c.mu.Unlock()
	return bytes.Clone(c.buf.Bytes())
}

// capturingTransport is an http.RoundTripper that tee-wraps the response body:
// every byte the SDK reads is mirrored into the per-request capturedBody (when
// the request context carries one) and passed through unchanged. Because the
// wrapper never reads the body itself, server-sent events flow through the
// transport incrementally and streaming callbacks fire live; the synchronous
// path captures the full body once the SDK reads it to EOF.
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
	capture, _ := req.Context().Value(bodyCaptureCtxKey{}).(*capturedBody)
	resp.Body = &capturingBody{rc: resp.Body, capture: capture}
	return resp, nil
}

// capturingBody mirrors every byte read from the underlying body into the
// per-request capturedBody while passing it through to the SDK decoder.
type capturingBody struct {
	rc      io.ReadCloser
	capture *capturedBody
}

func (c *capturingBody) Read(p []byte) (int, error) {
	n, err := c.rc.Read(p)
	if n > 0 && c.capture != nil {
		c.capture.write(p[:n])
	}
	return n, err
}

func (c *capturingBody) Close() error {
	return c.rc.Close()
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

	// Stash a capture buffer the capturingTransport fills with the raw response
	// body (mirrored as the SDK reads it), so parseResponse can embed it in an
	// error if the endpoint returns a non-standard or error response with HTTP
	// 200. The streaming path reads the same buffer after CreateMessagesStream
	// completes, so stream-mode diagnostics see the real body too.
	capture := &capturedBody{}
	ctx = context.WithValue(ctx, bodyCaptureCtxKey{}, capture)

	// Opt-in streaming: a non-nil DeltaSink asks the provider to stream text /
	// reasoning deltas as they arrive, then return the same assembled response.
	if req.DeltaSink != nil {
		return p.chatCompletionStream(ctx, req, anthropicReq, capture)
	}

	resp, err := p.client.CreateMessages(ctx, *anthropicReq)
	if err != nil {
		return nil, p.wrapError(fmt.Errorf("anthropic: API error: %w", err))
	}

	return p.parseResponse(resp, capture.bytes())
}

// chatCompletionStream performs a streaming Messages call. The go-anthropic SDK
// delivers content-block deltas through callbacks and still returns the fully
// assembled MessagesResponse, so the final response is built by the same
// parseResponse used by the synchronous path. req.DeltaSink is non-nil.
//
// The SDK streaming callback cannot return an error, so a delta sink error
// (host-side cancellation) is recorded and immediately cancels the request
// context: the HTTP stream read inside the SDK fails with a context error and
// the provider stops consuming the wire — the endpoint stops generating —
// instead of paying the full generation cost. The sink error itself (not the
// resulting transport error) is surfaced to the caller, matching the OpenAI
// streaming path's observable contract: returning an error from DeltaSink
// aborts the stream and the caller sees that error.
func (p *AnthropicProvider) chatCompletionStream(ctx context.Context, req ChatRequest, base *anthropic.MessagesRequest, capture *capturedBody) (*ChatResponse, error) {
	// Derive a per-call cancellation so a sink error stops the HTTP stream
	// mid-flight (see above) instead of reading the endpoint's output to EOF.
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	streamReq := anthropic.MessagesStreamRequest{MessagesRequest: *base}

	var sinkErr error
	// messageStop tracks whether the stream delivered its terminal
	// message_stop event. The SDK surfaces a plain EOF as a successful
	// (short) response, so without this flag a gateway that closes the
	// SSE stream mid-generation would be accepted as a complete answer.
	messageStop := false
	streamReq.OnMessageStop = func(anthropic.MessagesEventMessageStopData) {
		messageStop = true
	}
	streamReq.OnContentBlockDelta = func(ev anthropic.MessagesEventContentBlockDeltaData) {
		if sinkErr != nil {
			return
		}
		switch ev.Delta.Type {
		case anthropic.MessagesContentTypeTextDelta:
			if text := ev.Delta.GetText(); text != "" {
				sinkErr = req.DeltaSink(StreamDelta{Text: text})
			}
		case anthropic.MessagesContentTypeThinkingDelta:
			if ev.Delta.MessageContentThinking != nil && ev.Delta.Thinking != "" {
				sinkErr = req.DeltaSink(StreamDelta{Reasoning: ev.Delta.Thinking})
			}
		}
		if sinkErr != nil {
			// Stop reading the wire: cancel the request so the SDK's body read
			// fails promptly and the endpoint sees a closed connection.
			cancel()
		}
	}

	resp, err := p.client.CreateMessagesStream(ctx, streamReq)
	if sinkErr != nil {
		// The stream was aborted on purpose; surface the sink error, not the
		// transport failure the cancellation produced (context.Canceled or a
		// truncated-body read error).
		return nil, sinkErr
	}
	if err != nil {
		return nil, p.wrapError(fmt.Errorf("anthropic: API error: %w", err))
	}
	// The stream ended without the terminal message_stop event: a truncated
	// response, not a success. Return an unexpected-EOF error (classified
	// transient by the router) instead of letting the parser accept the
	// partial content as a complete answer.
	if !messageStop {
		return nil, p.wrapError(fmt.Errorf("anthropic: API error: %w", io.ErrUnexpectedEOF))
	}
	return p.parseResponse(resp, capture.bytes())
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

	logToolCallArguments(p.log(), p.name, message.ToolCalls)

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
		// The Anthropic SDK parses the JSON error body into APIError, which
		// carries no HTTP status — classify by the provider's own error type
		// field so transport-independent consumers (e.g. auto-retry arming)
		// still see the failure class.
		errKind := ErrKind("")
		switch {
		case apiErr.IsRateLimitErr():
			errKind = ErrKindRateLimit
		case apiErr.IsOverloadedErr():
			errKind = ErrKindOverloaded
		}
		e := NewError(p.name, 0, retryable, err)
		e.ErrKind = errKind
		return e
	}
	var reqErr *anthropic.RequestError
	if errors.As(err, &reqErr) {
		return WrapProviderError(p.name, reqErr.StatusCode, err)
	}
	return WrapProviderError(p.name, 0, err)
}
