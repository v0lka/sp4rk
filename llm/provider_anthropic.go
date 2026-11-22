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
	// diagnostics. The capture retains only the first maxCapturedBodyBytes
	// (64 KiB) of the body — far more than any diagnostic consumes — so even a
	// hostile, arbitrarily large or never-ending response cannot grow the heap
	// without bound. The provided HTTP client is cloned (not mutated) so any
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
// the streaming path completes. The capture is bounded: write retains only the
// first maxCapturedBodyBytes bytes of the body, so even a hostile or
// misbehaving endpoint streaming an arbitrarily large (or never-ending) body
// cannot grow the heap without bound — the SDK itself never materializes the
// whole body, so an uncapped tee would be a net-new, always-on allocation.
type capturedBody struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

// maxCapturedBodyBytes bounds the per-request response-body capture at 64 KiB.
// The capture exists only for error diagnostics — truncateForError embeds at
// most 2 KiB in an error message — so a bounded prefix is always sufficient,
// while the cap makes the transport's memory footprint constant regardless of
// response size or stream length.
const maxCapturedBodyBytes = 64 * 1024

// write appends bytes read off the wire, retaining only a bounded prefix
// (maxCapturedBodyBytes); bytes past the cap are dropped. Every consumer reads
// only the head of the body (truncateForError), so dropping the tail is
// lossless for diagnostics while capping memory.
func (c *capturedBody) write(p []byte) {
	c.mu.Lock()
	defer c.mu.Unlock()
	room := maxCapturedBodyBytes - c.buf.Len()
	if room <= 0 {
		return
	}
	if len(p) > room {
		p = p[:room]
	}
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
	// Tool names are mapped per request (see toolname.go): the Messages API
	// documents ^[a-zA-Z0-9_-]{1,64}$ for tool names, and an MCP-provided
	// dotted name would fail the whole request with HTTP 400.
	mapping := newToolNameMapping(req.Tools, defaultToolNamePolicy, p.logger)
	anthropicReq, err := p.buildRequest(req, mapping)
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
		return p.chatCompletionStream(ctx, req, anthropicReq, capture, mapping)
	}

	resp, err := p.client.CreateMessages(ctx, *anthropicReq)
	if err != nil {
		return nil, p.wrapError(fmt.Errorf("anthropic: API error: %w", err))
	}

	return p.parseResponse(resp, capture.bytes(), mapping)
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
func (p *AnthropicProvider) chatCompletionStream(ctx context.Context, req ChatRequest, base *anthropic.MessagesRequest, capture *capturedBody, mapping *toolNameMapping) (*ChatResponse, error) {
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
	return p.parseResponse(resp, capture.bytes(), mapping)
}

// buildRequest converts ChatRequest to anthropic.MessagesRequest.
//
// mapping adapts tool names to the endpoint-legal wire spelling: tool
// definitions carry the mapped name and assistant-history tool_use blocks are
// re-spelled the same way (see toolname.go).
func (p *AnthropicProvider) buildRequest(req ChatRequest, mapping *toolNameMapping) (*anthropic.MessagesRequest, error) {
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
		// ReasoningContent deliberately does NOT keep a message alive: a
		// reasoning-only message has no renderable block (the assistant branch
		// emits text and tool_use blocks only), so sending it would marshal a
		// nil Content as "content": null and the API would reject the whole
		// request with HTTP 400 — the same treatment buildGoogleRequest gives
		// such messages.
		if msg.Content == "" && len(msg.ContentBlocks) == 0 && len(msg.ToolCalls) == 0 && msg.ToolCallID == "" {
			continue
		}
		anthropicMsg, err := p.convertMessage(msg, mapping)
		if err != nil {
			return nil, err
		}
		// Choke point: never send a message that rendered to zero content
		// blocks — a nil Content slice marshals as "content": null, which the
		// Messages API rejects with HTTP 400.
		if len(anthropicMsg.Content) == 0 {
			continue
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

	// Cache breakpoints (provider-level prompt caching) are armed only on
	// loop-shaped calls: executor requests (and undeclared ones, which the
	// sampling policy already treats as executor) replay a stable, growing
	// prefix every turn, so each breakpoint's 1.25× cache-write premium is
	// repaid by 0.1× reads on the following turns. Routing / compaction /
	// summarization calls are one-shot structured requests — a cache write
	// there is pure premium with no reuse — so their wire shape is left
	// exactly as before (string system, no cache_control).
	cacheEnabled := !req.CallPurpose.Deterministic()

	// Set system prompt: cache-armed requests always render MultiSystem parts
	// (the wire accepts the array form interchangeably with the plain string),
	// with ONE cache_control breakpoint at the END OF THE STABLE SYSTEM — the
	// boundary the planner's CacheBreakMarker produced (stable prefix first,
	// dynamic suffix last). A breakpoint marks the longest prefix ending at
	// its block, so a single breakpoint on the boundary part caches the whole
	// stable prefix; marking several parts would only spend nested breakpoints
	// from the shared 4-per-request budget (system + tools + history).
	// Non-cached (deterministic-purpose) requests keep the legacy wire shape:
	// MultiSystem only when several parts exist, a plain string system for a
	// single part, and no cache_control anywhere.
	if len(systemParts) > 0 && (cacheEnabled || len(systemParts) > 1) {
		multiSystem := make([]anthropic.MessageSystemPart, len(systemParts))
		for i, part := range systemParts {
			multiSystem[i] = anthropic.MessageSystemPart{
				Type: "text",
				Text: part,
			}
		}
		if cacheEnabled {
			// With a dynamic suffix (marker present, 2+ parts) the boundary is
			// the second-to-last part; a single all-stable part is its own
			// boundary.
			boundary := len(systemParts) - 1
			if len(systemParts) > 1 {
				boundary = len(systemParts) - 2
			}
			multiSystem[boundary].CacheControl = &anthropic.MessageCacheControl{
				Type: anthropic.CacheControlTypeEphemeral,
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
	//
	// Thinking stays ENABLED across tool-use continuations: parseResponse now
	// captures every thinking/redacted_thinking block onto the assistant
	// Message (llm.Message.ThinkingBlocks) and convertMessage re-emits them at
	// the START of the corresponding assistant turn, which is exactly what the
	// Messages API requires of a thinking-enabled continuation ("Expected
	// `thinking` or `redacted_thinking`"). The fail-safe remains only for the
	// genuinely unsupported shape: a history whose final assistant turn
	// carries NO thinking blocks (a foreign or pre-carriage turn) would be
	// rejected with HTTP 400 under thinking, so thinking is disabled for that
	// request — at the cost of extended thinking on that turn only.
	if req.ReasoningEffort == "On" && !assistantHistoryMissingLeadingThinking(filteredMsgs) {
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
			sanitized := SanitizeSchemaForAnthropic(tool.InputSchema)
			var inputSchema any = sanitized
			if len(sanitized) == 0 {
				// A schema-less tool must not go out with an empty input_schema:
				// assigned to the SDK's any-typed field, a (typed-nil)
				// json.RawMessage defeats omitempty and marshals as
				// "input_schema": null — the Messages API requires input_schema
				// to be a JSON Schema object and rejects the whole request with
				// HTTP 400. Mirror convertSchemaToMap (Chat Completions) and
				// install the same default object schema.
				inputSchema = defaultObjectSchema()
			}
			tools[i] = anthropic.ToolDefinition{
				Name:        mapping.wireFor(tool.Name),
				Description: tool.Description,
				InputSchema: inputSchema,
			}
		}
		if cacheEnabled {
			// One breakpoint on the LAST tool definition: a breakpoint marks
			// the longest prefix ending at its block, and tools precede the
			// system in the caching hierarchy, so this caches the entire
			// tool-definitions prefix. Tool definitions are static across a
			// session's turns, so the write premium is repaid every turn.
			tools[len(tools)-1].CacheControl = &anthropic.MessageCacheControl{
				Type: anthropic.CacheControlTypeEphemeral,
			}
		}
		anthropicReq.Tools = tools
	}

	// Stable history boundary: mark the last block of the last rendered
	// message so the ENTIRE prompt up to this request becomes the cached
	// prefix. The agent loop only ever APPENDS to the history between
	// boundaries (compaction advances it in K-step cohorts), so the next
	// request's prefix is byte-identical up to this block and reads the
	// cache back before writing the extended prefix — the incremental
	// pattern that keeps per-turn uncached input at one turn's worth of
	// tokens instead of the full prefill.
	if cacheEnabled && len(messages) > 0 {
		armHistoryCacheBreakpoint(&messages[len(messages)-1])
	}

	return anthropicReq, nil
}

// armHistoryCacheBreakpoint marks the last cacheable content block of msg
// with an ephemeral cache_control breakpoint. Thinking and redacted_thinking
// blocks are skipped (they are never valid cache_control carriers); the
// terminal block of a rendered history message is a text or tool_result
// block, so in practice the walk stops at the first backward step.
func armHistoryCacheBreakpoint(msg *anthropic.Message) {
	for i := len(msg.Content) - 1; i >= 0; i-- {
		t := msg.Content[i].Type
		if t == anthropic.MessagesContentTypeThinking || t == anthropic.MessagesContentTypeRedactedThinking {
			continue
		}
		msg.Content[i].CacheControl = &anthropic.MessageCacheControl{
			Type: anthropic.CacheControlTypeEphemeral,
		}
		return
	}
}

// assistantHistoryMissingLeadingThinking reports whether the FINAL assistant
// turn in the outgoing history lacks its original thinking blocks — the one
// shape the Messages API rejects with HTTP 400 when thinking is enabled
// ("a final assistant message must start with a thinking block"). Turns the
// Anthropic provider produced now always carry ThinkingBlocks (parseResponse),
// so this fires only for foreign histories (other providers, pre-carriage
// persisted sessions) — which keep the legacy behavior of a thinking-disabled
// request. An assistant turn that renders to nothing (the empty-message skip
// in buildRequest) never reaches the wire, so it imposes no requirement and
// produces no fail-safe here.
func assistantHistoryMissingLeadingThinking(msgs []Message) bool {
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role != "assistant" {
			continue
		}
		return len(msgs[i].ThinkingBlocks) == 0
	}
	return false
}

// convertMessage converts a Message to anthropic.Message.
//
// mapping re-spells assistant-history tool_use names with the same wire
// mapping the tool definitions use; the call IDs are already sanitized
// independently (sanitizeAnthropicToolID).
func (p *AnthropicProvider) convertMessage(msg Message, mapping *toolNameMapping) (anthropic.Message, error) {
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

		// Re-emit carried thinking blocks FIRST, in their original order and
		// byte-for-byte: the Messages API requires an assistant turn to START
		// with its (redacted_)thinking block whenever thinking is enabled, and
		// verifies each block's signature. Every captured block is rendered —
		// including blocks whose thinking text or redacted payload is empty:
		// the API documents that it may return such blocks (e.g. generation
		// cut off at max_tokens) and requires them passed back unmodified, in
		// original order. Dropping one would both make the replayed turn
		// differ from the API's own response (HTTP 400 "cannot be modified")
		// and break the leading-thinking invariant the fail-safe below
		// (assistantHistoryMissingLeadingThinking) relies on. (The SDK's
		// omitempty drops an empty thinking/data string from the wire JSON;
		// the block itself — type + signature — still round-trips.)
		for _, tb := range msg.ThinkingBlocks {
			switch tb.Type {
			case "thinking":
				content = append(content, anthropic.MessageContent{
					Type: anthropic.MessagesContentTypeThinking,
					MessageContentThinking: &anthropic.MessageContentThinking{
						Thinking:  tb.Thinking,
						Signature: tb.Signature,
					},
				})
			case "redacted_thinking":
				content = append(content, anthropic.MessageContent{
					Type: anthropic.MessagesContentTypeRedactedThinking,
					MessageContentRedactedThinking: &anthropic.MessageContentRedactedThinking{
						Data: tb.Data,
					},
				})
			default:
				// Unknown thinking-block type: skip rather than corrupt the
				// turn shape; log at debug so misconfigured callers can
				// diagnose silently dropped content.
				p.log().Debug("anthropic: skipping unknown thinking block type",
					"block_type", tb.Type, "provider", p.name)
			}
		}

		// Add text content if present
		if msg.Content != "" {
			content = append(content, anthropic.NewTextMessageContent(msg.Content))
		}

		// Add tool use blocks for tool calls
		for _, tc := range msg.ToolCalls {
			content = append(content, anthropic.NewToolUseMessageContent(sanitizeAnthropicToolID(tc.ID), mapping.wireFor(tc.Name), tc.Input))
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
// mapping reverses the wire tool_use names back to the registered names the
// executor dispatches by (see toolname.go); names outside the request's
// catalog pass through unchanged.
//
// rawBody is the raw HTTP response body captured by the transport. It is used
// only to enrich diagnostics when the endpoint returns a non-standard response
// (e.g. an Anthropic-compatible endpoint that answers a 200 with an error
// object or an empty body); a nil/empty value is fine for callers that capture
// the response directly (tests).
func (p *AnthropicProvider) parseResponse(resp anthropic.MessagesResponse, rawBody []byte, mapping *toolNameMapping) (*ChatResponse, error) {
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

	// Thinking blocks are captured verbatim (text, signature, redacted payload)
	// onto the returned assistant Message so the next request can re-emit them —
	// the Messages API rejects a thinking-enabled continuation whose assistant
	// turns lack their original thinking blocks (see buildRequest). The plain
	// reasoning summary for ChatResponse.Reasoning is still accumulated from
	// thinking blocks only: redacted_thinking blocks carry an encrypted payload,
	// not readable reasoning, so they add nothing to it.
	var thinkingBlocks []ThinkingBlock

	// Process content blocks
	for _, block := range resp.Content {
		switch block.Type {
		case anthropic.MessagesContentTypeText:
			if message.Content != "" {
				message.Content += "\n"
			}
			message.Content += block.GetText()

		case anthropic.MessagesContentTypeThinking:
			// Extended thinking content block: accumulate the readable
			// reasoning for ChatResponse.Reasoning and keep the block —
			// text AND signature, both verbatim (the signature covers the
			// exact thinking text; any alteration invalidates it) — for the
			// wire round-trip.
			if block.MessageContentThinking != nil {
				if reasoning != "" {
					reasoning += "\n"
				}
				reasoning += block.Thinking
				thinkingBlocks = append(thinkingBlocks, ThinkingBlock{
					Type:      "thinking",
					Thinking:  block.Thinking,
					Signature: block.Signature,
				})
			} else {
				// The block type says thinking but the SDK union payload did
				// not decode. Capture a placeholder anyway so the captured
				// set matches the block set the API returned — the Messages
				// API requires EVERY returned thinking block to be passed
				// back (including blocks with an empty thinking field), and
				// the fail-safe (assistantHistoryMissingLeadingThinking)
				// keys on ThinkingBlocks being non-empty.
				thinkingBlocks = append(thinkingBlocks, ThinkingBlock{
					Type: "thinking",
				})
			}

		case anthropic.MessagesContentTypeRedactedThinking:
			// Redacted thinking: the API replaced the reasoning with an
			// encrypted payload. It carries no readable text (nothing to add
			// to Reasoning) but MUST round-trip like a thinking block — the
			// API treats it as satisfying the leading-thinking-block
			// requirement on its assistant turn.
			if block.MessageContentRedactedThinking != nil {
				thinkingBlocks = append(thinkingBlocks, ThinkingBlock{
					Type: "redacted_thinking",
					Data: block.Data,
				})
			} else {
				// Same capture-completeness contract as the thinking case
				// above: the API returned a redacted_thinking block, so the
				// replayed turn must carry one even when the SDK payload did
				// not decode.
				thinkingBlocks = append(thinkingBlocks, ThinkingBlock{
					Type: "redacted_thinking",
				})
			}

		case anthropic.MessagesContentTypeToolUse:
			if block.MessageContentToolUse != nil {
				message.ToolCalls = append(message.ToolCalls, ToolCall{
					ID:    block.ID,
					Name:  mapping.realFor(block.Name),
					Input: block.Input,
				})
			}
		}
	}

	message.ThinkingBlocks = thinkingBlocks

	finalizeToolCallArguments(p.log(), p.name, message.ToolCalls)

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
			// input_tokens counts only the tokens after the last cache
			// breakpoint; the cached prefix arrives separately as
			// cache_creation_input_tokens / cache_read_input_tokens. Sum all
			// three so usage tracking and the executor's near-context-limit
			// warning see the true prompt size instead of the tiny uncached
			// remainder (this provider arms caching on multi-part system
			// prompts, so the cache counters are routinely populated).
			InputTokens:  resp.Usage.InputTokens + resp.Usage.CacheCreationInputTokens + resp.Usage.CacheReadInputTokens,
			OutputTokens: resp.Usage.OutputTokens,
			// The cache split rides alongside the unchanged sum:
			// cache_read_input_tokens are the hits served from the cache,
			// cache_creation_input_tokens the tokens written into it by this
			// request. Both are already included in InputTokens above.
			CachedInputTokens: resp.Usage.CacheReadInputTokens,
			CacheWriteTokens:  resp.Usage.CacheCreationInputTokens,
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
