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
	"regexp"
	"sort"
	"strings"

	oai "github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/packages/param"
)

// OpenAIProviderConfig contains configuration for OpenAI-compatible providers.
type OpenAIProviderConfig struct {
	Name       string // logical provider name ("openai", "deepseek", "grok", etc.)
	APIKey     string
	BaseURL    string       // empty = default OpenAI; otherwise custom endpoint
	HTTPClient *http.Client // optional proxy-configured HTTP client (nil = default)
	Logger     *slog.Logger // optional structured logger (nil = slog.Default())
	// ReasoningWire selects the JSON spelling this endpoint expects for
	// Qwen-family reasoning controls (enable_thinking / reasoning_effort).
	// Zero value = ReasoningWireVendorDefault (top-level request fields),
	// which every OpenAI-compatible server except llama.cpp reads. Set it to
	// ReasoningWireChatTemplateKwargs for a llama.cpp-served endpoint, which
	// ignores a top-level enable_thinking and only honors the one nested in
	// chat_template_kwargs. See ReasoningWire for the full contract.
	ReasoningWire ReasoningWire
	// TokenSource optionally supplies per-request bearer credentials (see
	// TokenSource). When set, every protocol this provider serves resolves
	// credentials through it immediately before a request leaves the process:
	// the two OpenAI SDK clients (Chat Completions and Responses) via a
	// middleware that stamps the returned access token onto the Authorization
	// header — overriding the static APIKey credential — and applies
	// BearerToken.ExtraHeaders; the Anthropic and Google delegates apply the
	// same stamping through their transports, clearing the static x-api-key /
	// ?key= credential so a dynamic token never rides alongside it. A Token
	// failure aborts the request before any network I/O. The SDK-internal
	// retry loops are disabled for dynamic-credential clients (a credential
	// failure must abort, not be retried in-SDK); the router's own retry
	// policy re-enters the resolution per attempt. nil = static APIKey only
	// (the historical behavior).
	TokenSource TokenSource
	// RequireStreaming marks this endpoint as accepting streaming calls ONLY
	// on the wire (e.g. a ChatGPT OAuth backend that rejects any
	// non-streaming request) and rejecting the Responses API output cap
	// (max_output_tokens is dropped from the wire request for such
	// endpoints; they answer it with HTTP 400) and the "minimal" reasoning
	// effort (clamped to "low" on the wire for such endpoints). Seam field:
	// it is carried on the provider (requireStreaming) for the streaming
	// path to act on.
	// Zero value = no such requirement.
	RequireStreaming bool
	// OmitReasoningHistory stops the provider from echoing assistant
	// ReasoningContent back to the endpoint as the "reasoning_content" extra
	// field on subsequent request messages. Zero value = echo (the DeepSeek
	// V4 contract: reasoning_content must ride EVERY assistant message in
	// thinking mode, even an empty one, or the endpoint answers 400). Set it
	// for endpoints whose chat template rejects an unknown
	// "reasoning_content" message field (e.g. a llama.cpp server without a
	// DeepSeek-style template, which fails the request on strict template
	// rendering).
	// Seam field: it is carried on the provider (omitReasoningHistory) for
	// the request-building path to act on.
	OmitReasoningHistory bool
}

// OpenAIProvider implements Provider for OpenAI and compatible APIs.
type OpenAIProvider struct {
	client               *oai.Client        // official SDK for Chat Completions API
	responsesClient      *oai.Client        // official SDK for Responses API
	anthropicDelegate    *AnthropicProvider // serves ProtocolAnthropic models (Claude) via a co-located Anthropic provider using the same baseURL/APIKey/HTTPClient/Logger
	name                 string
	baseURL              string       // empty = default OpenAI; non-empty = compatible provider
	apiKey               string       // API key (passed to the Google delegate; empty for local backends)
	httpClient           *http.Client // optional proxy-configured HTTP client (passed to the Google delegate; nil = http.DefaultClient)
	logger               *slog.Logger
	reasoningWire        ReasoningWire // Qwen reasoning-control spelling for this endpoint (zero = vendor default)
	requireStreaming     bool          // endpoint accepts streaming calls only (seam; see OpenAIProviderConfig.RequireStreaming)
	tokenSource          TokenSource   // dynamic per-request credentials, threaded to the Anthropic/Google delegates too (nil = static key)
	omitReasoningHistory bool          // stop echoing reasoning_content on request assistant messages (seam; see OpenAIProviderConfig.OmitReasoningHistory)
}

// log returns the provider's logger, defaulting to slog.Default() when unset.
func (p *OpenAIProvider) log() *slog.Logger {
	if p.logger != nil {
		return p.logger
	}
	return slog.Default()
}

// NewOpenAIProvider creates a new OpenAI provider.
// If BaseURL is empty, uses default OpenAI endpoint.
// If BaseURL is set, uses custom endpoint (DeepSeek, Grok, OpenRouter, Ollama, LM-Studio).
//
// Note: APIKey is intentionally not validated here. Local models (LM Studio, Ollama)
// using OpenAI-compatible endpoints do not require authentication. This constructor
// must accept empty keys to support local inference backends.
func NewOpenAIProvider(cfg OpenAIProviderConfig) (*OpenAIProvider, error) {
	opts := []option.RequestOption{
		option.WithAPIKey(cfg.APIKey),
	}
	if cfg.BaseURL != "" {
		opts = append(opts, option.WithBaseURL(cfg.BaseURL))
	}
	if cfg.HTTPClient != nil {
		opts = append(opts, option.WithHTTPClient(cfg.HTTPClient))
	}
	if cfg.TokenSource != nil {
		// The SDK's internal retry loop retries a request whose middleware
		// returned (nil, error) — shouldRetry(req, nil) is true for a nil
		// response — so a TokenSource failure would be retried in-SDK, with
		// its own short backoff, BEFORE the router's policy ever sees it:
		// the contract's "abort the retry chain" would not hold (the
		// credentials are re-resolved, a later Token call may even succeed,
		// and the request leaves anyway), and a persistent failure costs
		// extra Token calls and sleep. Disable the SDK-internal retries for
		// dynamic-credential clients and let the ROUTER own the retry
		// policy: its loop re-enters the middleware per attempt (fresh
		// credentials on permitted HTTP retries) and a credential failure is
		// classified non-retryable by WrapProviderError, so it aborts there.
		opts = append(opts, option.WithMaxRetries(0), option.WithMiddleware(tokenSourceMiddleware(cfg.TokenSource)))
	}
	client := oai.NewClient(opts...)

	responsesClient := newResponsesClient(cfg.APIKey, cfg.BaseURL, cfg.HTTPClient, cfg.TokenSource)

	// Build a co-located Anthropic provider so ProtocolAnthropic models (Claude)
	// served by an OpenAI-compatible gateway (e.g. Zen) can be delegated to the
	// existing Anthropic Messages implementation — reusing the same
	// baseURL/APIKey/HTTPClient/Logger, with no code duplication. This is only
	// invoked on the ProtocolAnthropic dispatch path; for purely OpenAI
	// providers the delegate is constructed but never used. The delegate is built
	// from the shared connection fields explicitly (rather than via a direct
	// type conversion) so that adding Anthropic-specific fields to
	// AnthropicProviderConfig later cannot silently drop or mis-map a field.
	// TokenSource threads through too, so a dynamic-credential gateway
	// authenticates Claude exactly like GPT models (see the TokenSource
	// contract) instead of silently falling back to the static key.
	//nolint:staticcheck // S1016: explicit field copy is deliberate — see comment above; survives future divergent fields.
	anthropicDelegate, err := NewAnthropicProvider(AnthropicProviderConfig{
		Name:        cfg.Name,
		APIKey:      cfg.APIKey,
		BaseURL:     cfg.BaseURL,
		HTTPClient:  cfg.HTTPClient,
		Logger:      cfg.Logger,
		TokenSource: cfg.TokenSource,
	})
	if err != nil {
		return nil, fmt.Errorf("openai: failed to build anthropic delegate: %w", err)
	}

	return &OpenAIProvider{
		client:               &client,
		responsesClient:      responsesClient,
		anthropicDelegate:    anthropicDelegate,
		name:                 cfg.Name,
		baseURL:              cfg.BaseURL,
		apiKey:               cfg.APIKey,
		httpClient:           cfg.HTTPClient,
		logger:               cfg.Logger,
		reasoningWire:        cfg.ReasoningWire,
		requireStreaming:     cfg.RequireStreaming,
		tokenSource:          cfg.TokenSource,
		omitReasoningHistory: cfg.OmitReasoningHistory,
	}, nil
}

// tokenSourceMiddleware builds an openai-go request middleware that resolves
// credentials from ts immediately before every request leaves the process.
// For dynamic-credential clients the constructor disables the SDK-internal
// retry loop (option.WithMaxRetries(0)), so the only re-entry into this
// middleware comes from the ROUTER's retry policy — which re-enters per
// attempt, so a refreshed token reaches each new attempt of the call. The
// middleware stamps the access token onto the
// Authorization header (overriding the static option.WithAPIKey credential,
// which the SDK applies before middlewares run) and sets each extra header
// from BearerToken.ExtraHeaders; an empty extra value removes that header.
//
// The middleware deliberately performs no logging: token material (access
// tokens, header values) must never reach a log sink, and a header-stamping
// hot path has nothing else worth reporting. A Token failure aborts the
// request before it hits the wire and is returned to the SDK unwrapped in
// place of an HTTP response.
func tokenSourceMiddleware(ts TokenSource) option.Middleware {
	return func(req *http.Request, next option.MiddlewareNext) (*http.Response, error) {
		tok, err := ts.Token(req.Context())
		if err != nil {
			return nil, fmt.Errorf("llm: token source: %w", err)
		}
		applyBearerToken(req, tok, "")
		return next(req)
	}
}

// Name returns the provider name for logging.
func (p *OpenAIProvider) Name() string {
	return p.name
}

// ChatCompletion sends a request and returns the full response.
func (p *OpenAIProvider) ChatCompletion(ctx context.Context, req ChatRequest) (*ChatResponse, error) {
	// Validate image blocks up front so a missing MediaType/ImageB64 yields a
	// clear local error instead of an opaque API 400.
	for _, msg := range req.Messages {
		if err := ValidateContentBlocks(msg.ContentBlocks); err != nil {
			return nil, fmt.Errorf("openai: %w", err)
		}
	}

	// Dispatch each model to the API protocol it speaks. The OpenAI provider
	// natively handles the two OpenAI protocols:
	//   - ProtocolResponses (GPT-5.x, GPT-6, Codex) → Responses API
	//     (/v1/responses). These models are served exclusively via
	//     /responses — both the official OpenAI endpoint and compatible
	//     gateways (e.g. OpenCode Zen exposes gpt-5.x / gpt-5.x-codex
	//     there) — and sending them to /chat/completions returns a
	//     degenerate HTTP 400 with an empty body.
	//     Therefore, when the Responses endpoint is genuinely missing (HTTP
	//     404/405) we surface a clear "Responses API required but unavailable"
	//     error instead of silently falling back to a Chat Completions path
	//     that is known to fail. The responsesClient is already configured with
	//     the provider's baseURL, so a single code path covers both the
	//     official endpoint and compatible gateways.
	//   - ProtocolChatCompletions (gpt-4o, o-series, gpt-4.1, and most
	//     compatible models) → Chat Completions (/v1/chat/completions).
	//
	// ProtocolAnthropic (Claude) is delegated to a co-located AnthropicProvider
	// built with the same baseURL/APIKey/HTTPClient/Logger, so a Claude model
	// served by an OpenAI-compatible gateway (e.g. Zen) hits the gateway's
	// Anthropic /messages endpoint with no code duplication. ProtocolGoogle
	// (Gemini/Gemma) is delegated to googleCompletion, which POSTs
	// {baseURL}/models/{model}:generateContent with the Google contents/parts
	// format, reusing the same baseURL/apiKey/httpClient.
	// Honor an explicit registry-resolved protocol when set (the documented
	// escape hatch in protocol.go: a caller MAY override ModelMetadata.Protocol,
	// which the router threads into req.Protocol). Fall back to name-based
	// detection only when req.Protocol is empty (e.g. direct provider use
	// without a router/registry), preserving backward compatibility.
	protocol := req.Protocol
	if protocol == "" {
		protocol = DetectProtocol(req.Model)
	}
	switch protocol {
	case ProtocolResponses:
		var (
			resp *ChatResponse
			err  error
		)
		// Streaming wire path when the endpoint demands it (requireStreaming —
		// e.g. a ChatGPT OAuth Codex backend that rejects non-streaming
		// requests) or the caller opted into delta delivery via DeltaSink.
		// Both produce the same assembled ChatResponse as the synchronous
		// path; see responsesAPICompletionStream.
		if p.requireStreaming || req.DeltaSink != nil {
			resp, err = responsesAPICompletionStream(ctx, p.responsesClient, p.name, p.baseURL, req, p.requireStreaming, p.logger)
		} else {
			resp, err = responsesAPICompletion(ctx, p.responsesClient, p.name, p.baseURL, req, p.logger)
		}
		if err == nil {
			return resp, nil
		}
		// GPT-5.x / GPT-6 / Codex models require the Responses API: both the
		// official endpoint and compatible gateways serve them only via
		// /responses, and /chat/completions returns a degenerate HTTP 400
		// with an empty body.
		// When the endpoint is genuinely missing (404/405), surface a clear,
		// actionable error instead of silently falling back to Chat Completions
		// — which would mask the real cause behind an opaque 400 from a path
		// that is known not to work for these models. Other errors (400/500/...)
		// mean the endpoint exists but the request failed, so they are returned
		// as-is.
		if isResponsesEndpointUnsupported(err) {
			p.log().Warn("openai: responses API required for model but unavailable",
				"model", req.Model, "provider", p.name)
			return nil, fmt.Errorf("openai: model %q requires the Responses API (/responses) which is unavailable on provider %q (HTTP 404/405): %w",
				req.Model, p.name, err)
		}
		return resp, err
	case ProtocolAnthropic:
		// A Claude model served by an OpenAI-compatible gateway (e.g. Zen
		// exposing Claude via an Anthropic-compatible /messages endpoint) is
		// delegated to the co-located AnthropicProvider built with the same
		// baseURL/APIKey/HTTPClient/Logger. This reuses the entire existing
		// Anthropic Messages implementation — which already normalizes
		// baseURL→/v1 and POSTs /messages → Zen's /v1/messages — with no code
		// duplication. Error wrapping and body capture happen inside the
		// delegate. The provider name is inherited from this OpenAIProvider.
		return p.anthropicDelegate.ChatCompletion(ctx, req)
	case ProtocolGoogle:
		// A Gemini/Gemma model served by an OpenAI-compatible gateway (e.g.
		// Zen exposing Gemini via its generateContent endpoint) is delegated to
		// googleCompletion, which POSTs {baseURL}/models/{model}:generateContent
		// with the Google contents/parts format. The delegate reuses this
		// provider's baseURL/apiKey/httpClient; error wrapping and body capture
		// happen inside the delegate, with the provider name inherited here.
		// TokenSource threads through, so a dynamic-credential gateway
		// authenticates Gemini exactly like GPT models instead of silently
		// falling back to the static ?key= credential.
		return googleCompletion(ctx, googleCompletionConfig{
			HTTPClient:   p.httpClient,
			BaseURL:      p.baseURL,
			APIKey:       p.apiKey,
			ProviderName: p.name,
			Logger:       p.logger,
			TokenSource:  p.tokenSource,
		}, req)
	case ProtocolChatCompletions:
		// Handled by the shared Chat Completions path below.
	}

	params := p.buildChatParams(req)

	// Opt-in streaming: a non-nil DeltaSink asks the provider to stream text /
	// reasoning deltas as they arrive, then return the same assembled response.
	if req.DeltaSink != nil {
		// Ask the endpoint to include a final usage-only chunk so token
		// accounting matches the synchronous path. Some OpenAI-compatible
		// servers (older vLLM, certain Azure api-versions, proxies) reject the
		// unknown stream_options field with an HTTP 400; retry once without it
		// so enabling streaming stays safe on those endpoints too — usage is
		// then simply absent (zeros), which the executor tolerates. The retry
		// cannot duplicate deltas: a 400 is returned before the first chunk,
		// so no delta reached the sink.
		params.StreamOptions = oai.ChatCompletionStreamOptionsParam{IncludeUsage: param.NewOpt(true)}
		resp, err := p.chatCompletionStream(ctx, req, params)
		if err != nil && isStreamOptionsUnsupported(err) {
			p.log().Debug("openai: endpoint rejected stream_options; retrying stream without usage accounting",
				"provider", p.name, "model", req.Model)
			params.StreamOptions = oai.ChatCompletionStreamOptionsParam{}
			return p.chatCompletionStream(ctx, req, params)
		}
		return resp, err
	}

	resp, err := p.client.Chat.Completions.New(ctx, params)
	if err != nil {
		return nil, p.wrapError(fmt.Errorf("openai chat completion: %w", err))
	}
	// The OpenAI SDK decodes an HTTP 200 response whose JSON body is the
	// literal `null` into a nil response pointer with a nil error (JSON null
	// unmarshalled into a **struct clears the pointer). A compatible gateway
	// can emit that instead of a completion object during a transient upstream
	// fault, so guard before dereferencing resp below — a nil dereference here
	// panics inside the caller's goroutine and, for a sub-agent, tears down the
	// host process. Classified as retryable so the router's backoff loop gets a
	// chance to recover.
	if resp == nil {
		return nil, NewError(p.name, 0, true,
			errors.New("openai chat completion: provider returned HTTP 200 with a null body"))
	}

	if len(resp.Choices) == 0 {
		return nil, WrapProviderError(p.name, 0, errors.New("no choices in response"))
	}

	choice := resp.Choices[0]
	message := p.convertChatResponseMessage(choice.Message)
	stopReason := MapStopReason(choice.FinishReason, openAIStopReasonMap)

	return &ChatResponse{
		Message:    message,
		Reasoning:  message.ReasoningContent,
		StopReason: stopReason,
		Usage: TokenUsage{
			InputTokens:  int(resp.Usage.PromptTokens),
			OutputTokens: int(resp.Usage.CompletionTokens),
		},
	}, nil
}

// chatCompletionStream performs a streaming Chat Completions call. It forwards
// text and reasoning deltas to req.DeltaSink as they arrive off the wire and
// assembles the same *ChatResponse the synchronous path returns (content,
// reasoning, tool calls, usage, stop reason). req.DeltaSink is non-nil.
// params.StreamOptions has been configured by the caller (usage chunk on,
// unless the endpoint rejected it).
//
// A delta sink error aborts the stream immediately (host-side cancellation):
// the stream is closed and the error is returned to the caller.
func (p *OpenAIProvider) chatCompletionStream(ctx context.Context, req ChatRequest, params oai.ChatCompletionNewParams) (*ChatResponse, error) {
	stream := p.client.Chat.Completions.NewStreaming(ctx, params)
	defer func() { _ = stream.Close() }()

	var (
		content   strings.Builder
		reasoning strings.Builder
		toolAcc   = map[int]*streamToolCall{}
		finish    string
		usage     TokenUsage
	)

	for stream.Next() {
		chunk := stream.Current()

		if chunk.JSON.Usage.Valid() {
			usage = TokenUsage{
				InputTokens:  int(chunk.Usage.PromptTokens),
				OutputTokens: int(chunk.Usage.CompletionTokens),
			}
		}

		for _, choice := range chunk.Choices {
			// A multi-choice stream (n > 1, non-standard gateways) interleaves
			// fragments of every choice into one chunk stream; the synchronous
			// path returns Choices[0] only, so consume index 0 alone to keep the
			// assembled response identical. Servers that omit the index field
			// decode it as 0 and are unaffected.
			if choice.Index != 0 {
				continue
			}
			delta := choice.Delta
			if delta.Content != "" {
				content.WriteString(delta.Content)
				if err := req.DeltaSink(StreamDelta{Text: delta.Content}); err != nil {
					return nil, err
				}
			}
			// reasoning_content is a non-standard extension (DeepSeek et al.);
			// it is not a typed field on the delta, so read it from the raw JSON.
			if r := extractReasoningContent(delta.RawJSON()); r != "" {
				reasoning.WriteString(r)
				if err := req.DeltaSink(StreamDelta{Reasoning: r}); err != nil {
					return nil, err
				}
			}
			// Tool-call arguments arrive as per-index fragments across chunks and
			// must be concatenated in order. The function NAME is concatenated
			// too: some OpenAI-compatible gateways fragment it across chunks
			// exactly like the arguments (the official openai-go accumulator
			// concatenates both), so an overwrite here would drop all but the
			// last fragment and hand the executor a wrong or nonexistent tool
			// name ("get_" + "weather" must not become "weather").
			for _, tc := range delta.ToolCalls {
				idx := int(tc.Index)
				acc := toolAcc[idx]
				if acc == nil {
					acc = &streamToolCall{}
					toolAcc[idx] = acc
				}
				if tc.ID != "" {
					acc.id = tc.ID
				}
				acc.name += tc.Function.Name
				acc.args.WriteString(tc.Function.Arguments)
			}
			if choice.FinishReason != "" {
				finish = choice.FinishReason
			}
		}
	}
	if err := stream.Err(); err != nil {
		return nil, p.wrapError(fmt.Errorf("openai chat completion (stream): %w", err))
	}
	// A stream that ended (plain EOF, no SDK-level transport error) without a
	// terminal finish_reason on choice 0 is a truncated response: a
	// compatible gateway that closes the chunked/SSE stream after a few
	// deltas would otherwise be accepted as a successful (short) answer —
	// the empty finish reason must not silently map to end_turn. Surface it
	// as an unexpected-EOF error so the router's classifier sees a transient
	// failure and the caller does not persist a truncated reply.
	if finish == "" {
		return nil, p.wrapError(fmt.Errorf("openai chat completion (stream): %w", io.ErrUnexpectedEOF))
	}

	// Reassemble tool calls in declaration order (index order).
	idxs := make([]int, 0, len(toolAcc))
	for idx := range toolAcc {
		idxs = append(idxs, idx)
	}
	sort.Ints(idxs)
	var toolCalls []ToolCall
	for _, idx := range idxs {
		acc := toolAcc[idx]
		toolCalls = append(toolCalls, ToolCall{
			ID:    acc.id,
			Name:  acc.name,
			Input: json.RawMessage(acc.args.String()),
		})
	}

	message := Message{
		Role:             "assistant",
		Content:          content.String(),
		ReasoningContent: reasoning.String(),
		ToolCalls:        toolCalls,
	}
	logToolCallArguments(p.log(), p.name, message.ToolCalls)

	return &ChatResponse{
		Message:    message,
		Reasoning:  message.ReasoningContent,
		StopReason: MapStopReason(finish, openAIStopReasonMap),
		Usage:      usage,
	}, nil
}

// streamToolCall accumulates the per-index fragments of one streamed tool call.
type streamToolCall struct {
	id   string
	name string
	args strings.Builder
}

// buildChatParams converts our ChatRequest to OpenAI ChatCompletionNewParams.
func (p *OpenAIProvider) buildChatParams(req ChatRequest) oai.ChatCompletionNewParams {
	// Normalize system messages to a single leading system message.
	//
	// sp4rk's prompt assembly can emit system messages after a user/assistant/
	// tool message (e.g. the plan as a trailing system message, or system
	// messages carried inside injected conversation history). Some
	// OpenAI-compatible backends — notably vLLM serving Qwen — apply a strict
	// chat template that rejects any non-leading system message with HTTP 400
	// "System message must be at the beginning" (LM Studio's template is
	// lenient, which is why the same payload works there). Hoist every system
	// message to the front, exactly as the Anthropic and OpenAI Responses
	// providers already do via ExtractSystemPrompt(Parts). This is a no-op for
	// the common case of a single leading system message.
	systemPrompt, filtered := ExtractSystemPrompt(req.Messages)
	messages := make([]oai.ChatCompletionMessageParamUnion, 0, len(filtered)+1)
	if systemPrompt != "" {
		messages = append(messages, oai.SystemMessage(systemPrompt))
	}
	for _, msg := range filtered {
		messages = append(messages, p.convertRequestMessage(msg))
	}

	params := oai.ChatCompletionNewParams{
		Model:    req.Model,
		Messages: messages,
	}

	if req.MaxTokens > 0 {
		params.MaxCompletionTokens = oai.Int(int64(req.MaxTokens))
	}

	if req.Temperature != nil {
		params.Temperature = oai.Float(*req.Temperature)
	}
	// top_p and presence_penalty are part of the official OpenAI Chat
	// Completions schema, so they are sent to every endpoint (strict
	// api.openai.com and compatible gateways alike).
	if req.TopP != nil {
		params.TopP = oai.Float(*req.TopP)
	}
	if req.PresencePenalty != nil {
		params.PresencePenalty = oai.Float(*req.PresencePenalty)
	}

	// Apply reasoning effort as native provider value
	if req.ReasoningEffort != "" {
		family := req.ModelFamily
		if family == "" {
			family = string(DetectFamily(req.Model))
		}
		switch family {
		case "openai_flagship", "openai_standard", "openai_codex":
			params.ReasoningEffort = oai.ReasoningEffort(req.ReasoningEffort)
		case "deepseek":
			applyDeepSeekReasoning(&params, req.ReasoningEffort)
		case "qwen":
			applyQwenReasoning(&params, req.Model, req.ReasoningEffort, p.reasoningWire)
		case "glm":
			applyGLMReasoning(&params, req.Model, req.ReasoningEffort)
		case "kimi":
			applyKimiReasoning(&params, req.Model, req.ReasoningEffort)
		}
	}

	// top_k and repetition_penalty are NOT part of the official OpenAI schema
	// and the strict api.openai.com endpoint rejects unknown request fields.
	// OpenAI-compatible serving stacks (LM Studio, vLLM, llama.cpp, Ollama)
	// read them from the same Chat Completions payload, so they are forwarded
	// only when a custom baseURL is configured. Applied after the reasoning
	// switch above because mergeExtraFields must preserve the extras it may
	// have set ("thinking", "enable_thinking", "chat_template_kwargs", ...).
	if p.baseURL != "" {
		extras := make(map[string]any)
		if req.TopK != nil {
			extras["top_k"] = *req.TopK
		}
		if req.RepetitionPenalty != nil {
			extras["repetition_penalty"] = *req.RepetitionPenalty
		}
		if len(extras) > 0 {
			mergeExtraFields(&params, extras)
		}
	}

	if len(req.Tools) > 0 {
		tools := make([]oai.ChatCompletionToolParam, len(req.Tools))
		for i, tool := range req.Tools {
			tools[i] = oai.ChatCompletionToolParam{
				Function: oai.FunctionDefinitionParam{
					Name:        tool.Name,
					Description: oai.String(tool.Description),
					Parameters:  p.convertSchemaToMap(SanitizeSchemaForOpenAINonStrict(tool.InputSchema)),
				},
			}
		}
		params.Tools = tools
	}

	return params
}

// mergeExtraFields merges extra fields into the OpenAI SDK params without
// clobbering values set by an earlier SetExtraFields call: the SDK's
// SetExtraFields replaces the whole map, so extras set by the reasoning-family
// branches in buildChatParams ("thinking", "enable_thinking", ...) must be
// preserved.
func mergeExtraFields(params *oai.ChatCompletionNewParams, extra map[string]any) {
	merged := make(map[string]any, len(params.ExtraFields())+len(extra))
	for k, v := range params.ExtraFields() {
		merged[k] = v
	}
	for k, v := range extra {
		merged[k] = v
	}
	params.SetExtraFields(merged)
}

// applyQwenReasoning sets the reasoning-related extra fields for Qwen models.
//
// Qwen 3.8+ supports the per-request reasoning_effort parameter (values
// "xhigh"/"medium"/"low") with thinking enabled by default at the native xhigh
// effort:
//
//   - "off"/"Off":    thinking disabled (enable_thinking=false), no reasoning_effort
//   - "On"/"xhigh":   thinking enabled (enable_thinking=true) at the native
//     default effort xhigh, so reasoning_effort is left unset
//   - "medium"/"low": adaptive thinking at the requested effort, expressed via
//     reasoning_effort alone (the control implies thinking enabled)
//
// Pre-3.8 Qwen models (qwen3-*, qwen2.5-*, qwq-*, …) do not know
// reasoning_effort — sending it is at best silently ignored and at worst
// rejected by a strict OpenAI-compatible gateway — so they keep the legacy
// binary control: a known effort value maps to enable_thinking=true, every
// other value fails closed to false (the pre-reasoning_effort behavior of
// enable_thinking = (effort == "On")).
//
// "On" is kept as a legacy alias of "xhigh" for configurations stored with the
// binary value. The disable sentinel is matched case-insensitively (EqualFold)
// ahead of the switch, so the fail-closed default branch cannot silently
// evolve into a vendor-default pass-through that re-enables thinking for the
// sentinel. Any other value fails closed to thinking disabled, preserving
// the pre-reasoning_effort behavior of enable_thinking = (effort == "On").
//
// wire selects the JSON SPELLING of those controls (see ReasoningWire):
// ReasoningWireVendorDefault — the zero value, and the fall-back for any
// unrecognized wire — emits them as top-level request fields, which is what
// vLLM, LM Studio, SGLang, Ollama and DashScope read. A llama.cpp server
// ignores a top-level enable_thinking, so endpoints served by one opt into
// ReasoningWireChatTemplateKwargs and get the identical decision table encoded
// inside "chat_template_kwargs" instead.
func applyQwenReasoning(params *oai.ChatCompletionNewParams, model, effort string, wire ReasoningWire) {
	if wire == ReasoningWireChatTemplateKwargs {
		applyQwenReasoningChatTemplateKwargs(params, model, effort)
		return
	}
	// "off" is the value c0wrk stores in its small-LLM config; "Off" is the
	// canonical spelling. Both disable thinking; the guard runs before the
	// switch so the semantics of the documented sentinel are pinned
	// independently of the default branch below.
	if strings.EqualFold(effort, "off") {
		params.SetExtraFields(map[string]any{
			"enable_thinking": false,
		})
		return
	}
	if !IsQwen38OrLater(model) {
		// Legacy binary control: any known effort value ("On" and its native
		// aliases) requests thinking; anything else keeps the fail-closed
		// disabled default of the historical enable_thinking=(effort=="On").
		on := effort == "On" || effort == "xhigh" || effort == "medium" || effort == "low"
		params.SetExtraFields(map[string]any{
			"enable_thinking": on,
		})
		return
	}
	switch effort {
	case "On", "xhigh":
		params.SetExtraFields(map[string]any{
			"enable_thinking": true,
		})
	case "medium", "low":
		params.SetExtraFields(map[string]any{
			"reasoning_effort": effort,
		})
	default:
		params.SetExtraFields(map[string]any{
			"enable_thinking": false,
		})
	}
}

// applyQwenReasoningChatTemplateKwargs encodes applyQwenReasoning's decision
// table in the llama.cpp spelling: ONE top-level "chat_template_kwargs" object
// holding the controls, instead of top-level fields.
//
// Two server-side facts (tools/server/server-common.cpp,
// oaicompat_chat_params_parse) drive the shape:
//
//   - each kwarg value is JSON-dumped into a key→string map and re-parsed into
//     the template's extra context, and the dumped enable_thinking is compared
//     against the strings "true"/"false". A JSON boolean dumps to exactly that;
//     a JSON *string* dumps WITH quotes and makes the server throw
//     `invalid type for "enable_thinking" (expected boolean, got string)`. So
//     enable_thinking is emitted as a Go bool, never as "true"/"false".
//   - a top-level enable_thinking is never read, which is precisely why this
//     spelling exists: without it, "Off" is a silent no-op and the model keeps
//     thinking.
//
// reasoning_effort is emitted ONLY for the native levels medium/low, and always
// together with enable_thinking=true (a kwarg effort does not by itself switch
// thinking on, unlike the top-level spelling where llama.cpp forwards the value
// itself). The Bonsai chat template raises on a reasoning_effort outside
// {xhigh, medium, low}, so the "Off"/"On" sentinels — which are not native
// levels — must never be forwarded as that kwarg: "On"/"xhigh" mean "thinking
// at the template's native default (xhigh)" and are expressed as
// enable_thinking=true alone.
//
// mergeExtraFields (not SetExtraFields) is used so extras set earlier on the
// same params — and the top_k/repetition_penalty merge that runs after this in
// buildChatParams — are preserved rather than clobbered.
func applyQwenReasoningChatTemplateKwargs(params *oai.ChatCompletionNewParams, model, effort string) {
	kwargs := make(map[string]any, 2)
	switch {
	case strings.EqualFold(effort, "off"):
		kwargs["enable_thinking"] = false
	case !IsQwen38OrLater(model):
		// Legacy binary control, same fail-closed mapping as the
		// vendor-default spelling: a known effort value requests thinking,
		// anything else disables it, and reasoning_effort is never sent
		// (pre-3.8 models do not know the parameter).
		kwargs["enable_thinking"] = effort == "On" || effort == "xhigh" || effort == "medium" || effort == "low"
	case effort == "On", effort == "xhigh":
		kwargs["enable_thinking"] = true
	case effort == "medium", effort == "low":
		kwargs["enable_thinking"] = true
		kwargs["reasoning_effort"] = effort
	default:
		kwargs["enable_thinking"] = false
	}
	mergeExtraFields(params, map[string]any{
		"chat_template_kwargs": kwargs,
	})
}

// applyDeepSeekReasoning sets the thinking control for DeepSeek models.
//
// The DeepSeek Chat Completions API splits the control across two fields
// (api-docs.deepseek.com/api/create-chat-completion): the "thinking" object
// toggles the MODE — thinking.type accepts only "enabled"/"disabled" (the
// serving enum also carries "adaptive", which the docs do not offer for the
// current models) — while the effort is a SEPARATE top-level
// "reasoning_effort" with the documented set none/low/high/max ("none"
// disables thinking; the default effort is "high"). The family's canonical
// options "Off"/"High"/"Max" (see FamilyReasoningOptions) are OPTION
// spellings, not wire values: passing them through verbatim sends an invalid
// thinking.type and the API answers HTTP 422 "unknown variant `Off`,
// expected one of `adaptive`, `enabled`, `disabled`" — the failure that
// broke every c0wrk one-shot service call (title/commit pin the off tier).
// Everything is therefore normalized here:
//
//   - "off"/"none" (case-insensitive; "off" is the value c0wrk stores in
//     its small-LLM config, "none" the GLM-style disable spelling):
//     thinking disabled — {"type": "disabled"} with NO reasoning_effort,
//     the effort field is never sent alongside a disabled mode switch
//   - "low"/"minimal": thinking enabled at effort low ("minimal" is the
//     OpenAI spelling the server maps to low; normalized client-side)
//   - "medium"/"xhigh"/"high": thinking enabled at effort high ("medium"
//     and "xhigh" are the server's own compat spellings of high; normalized
//     client-side so the wire only ever carries documented values)
//   - "max": thinking enabled at effort max
//
// Every value outside the documented set ("On", unknown spellings) fails
// closed to thinking disabled: silently enabling thinking is the opposite
// of the configured intent, and a strict gateway rejects unknown
// thinking.type values outright (mirroring applyGLMReasoning).
func applyDeepSeekReasoning(params *oai.ChatCompletionNewParams, effort string) {
	thinkingDisabled := func() {
		params.SetExtraFields(map[string]any{
			"thinking": map[string]string{"type": "disabled"},
		})
	}
	thinkingAtEffort := func(effort string) {
		params.SetExtraFields(map[string]any{
			"thinking":         map[string]string{"type": "enabled"},
			"reasoning_effort": effort,
		})
	}
	switch {
	case strings.EqualFold(effort, "off"), strings.EqualFold(effort, "none"):
		thinkingDisabled()
	case strings.EqualFold(effort, "low"), strings.EqualFold(effort, "minimal"):
		thinkingAtEffort("low")
	case strings.EqualFold(effort, "medium"), strings.EqualFold(effort, "xhigh"),
		strings.EqualFold(effort, "high"):
		thinkingAtEffort("high")
	case strings.EqualFold(effort, "max"):
		thinkingAtEffort("max")
	default:
		thinkingDisabled()
	}
}

// applyGLMReasoning encodes the model's reasoning option set. GLM 5.2
// supports none/max/high; GLM 5.3+ (flagship and Flash) supports only
// max/high/low. The same ModelReasoningOptions drives the picker, service
// tiers, and this encoder so disabling a thinking-locked model is impossible.
//
// Off/none and unknown efforts select the cheapest legal posture: disabled
// when available, otherwise the minimal enabled effort. On selects the model
// default for legacy host settings. Option matching is case-insensitive but
// the wire receives the canonical spelling. Pre-5.2 models retain binary
// On/Off thinking.type control; an empty effort is omitted by the call site.
func applyGLMReasoning(params *oai.ChatCompletionNewParams, model, effort string) {
	if IsGLM52OrLater(model) {
		options, preferred, _ := ModelReasoningOptions("glm", model)
		native := reasoningDisableSpelling(options)
		if native == "" {
			native = minimalReasoningEffort(options)
		}
		if strings.EqualFold(effort, "on") {
			native = preferred
		} else {
			for _, option := range options {
				if strings.EqualFold(effort, option) {
					native = option
					break
				}
			}
		}
		if strings.EqualFold(native, "none") || strings.EqualFold(native, "off") {
			params.SetExtraFields(map[string]any{
				"thinking": map[string]string{"type": "disabled"},
			})
		} else {
			params.SetExtraFields(map[string]any{
				"thinking":         map[string]string{"type": "enabled"},
				"reasoning_effort": native,
			})
		}
		return
	}
	wireType := "disabled"
	if strings.EqualFold(effort, "on") {
		wireType = "enabled"
	}
	params.SetExtraFields(map[string]any{
		"thinking": map[string]string{"type": wireType},
	})
}

// kimiK3SeriesRe recognizes the K3 series of Kimi (Moonshot AI) models — the
// only Kimi models whose OpenAI-compatible API documents a per-request
// reasoning control (reasoning_effort, values low/high/max; see the catalog
// section comment and the K3 quickstart). It anchors "k3" on identifier
// boundaries (start/end or a separator) so the platform ID "kimi-k3", the
// Kimi Code endpoint aliases "k3"/"k3-256k", and future K3 variants match,
// while unrelated substrings ("k2.7-code", "mk3", a "k30" version token)
// do not. Matching is case-insensitive; the caller passes the bare model
// name already lowercased.
var kimiK3SeriesRe = regexp.MustCompile(`(^|[-_.])k3([-_.]|$)`)

// applyKimiReasoning sets the reasoning control for Kimi (Moonshot AI) models
// served through OpenAI-compatible endpoints (platform.kimi.ai and
// api.kimi.com/coding alike).
//
// Only the K3 series documents a per-request effort field — reasoning_effort
// (low/high/max). Every other kimi model exposes no documented wire spelling
// in this catalog: K2.7 Code runs with thinking always on (a thinking-disable
// routes the request to K2.6), K2.5/K2.6 document thinking/non-thinking modes
// without a documented field, and kimi-k2 carries no reasoning capability at
// all (gated upstream by ReasoningForCall, so it never reaches this switch).
// The effort value is therefore emitted ONLY for K3-series models and fails
// closed to no field for everything else: a strict gateway rejects unknown
// request fields outright, and for the thinking-locked models the omitted
// parameter resolves to the server-managed default — the only control they
// accept. Values outside the documented set fail closed the same way (the
// tier resolver only produces "low" for kimi, but a host may set the field
// directly).
//
// The SDK's typed ReasoningEffort field serializes to the same top-level
// "reasoning_effort" key the K3 API documents, exactly like the
// openai_* families above.
func applyKimiReasoning(params *oai.ChatCompletionNewParams, model, effort string) {
	bare := strings.ToLower(strings.TrimSpace(BareModel(model)))
	if !kimiK3SeriesRe.MatchString(bare) {
		return
	}
	switch effort {
	case "low", "high", "max":
		params.ReasoningEffort = oai.ReasoningEffort(effort)
	}
}

// convertSchemaToMap converts JSON schema bytes to a map[string]any.
func (p *OpenAIProvider) convertSchemaToMap(schema []byte) map[string]any {
	if len(schema) == 0 {
		return map[string]any{
			"type":                 "object",
			"properties":           map[string]any{},
			"required":             []any{},
			"additionalProperties": false,
		}
	}
	var params map[string]any
	if err := json.Unmarshal(schema, &params); err != nil {
		return map[string]any{
			"type":                 "object",
			"properties":           map[string]any{},
			"required":             []any{},
			"additionalProperties": false,
		}
	}
	return params
}

// convertRequestMessage converts our Message to OpenAI's message format.
func (p *OpenAIProvider) convertRequestMessage(msg Message) oai.ChatCompletionMessageParamUnion {
	// Safety net: OpenAI API requires non-empty content for tool-role messages.
	// The context layer should already guarantee this, but we keep this as a defensive measure.
	content := msg.Content
	if msg.Role == "tool" && content == "" {
		content = "(no output)"
	}

	switch msg.Role {
	case "system":
		return oai.SystemMessage(content)
	case "user":
		// When ContentBlocks are present, render them as multipart content
		// (text and/or image_url parts) instead of the plain Content string.
		// NormalizeContentBlocks prepends Content as a text block when the
		// blocks carry no text, so the task text always reaches the model.
		// Text-only messages without ContentBlocks keep the existing path.
		blocks := NormalizeContentBlocks(msg)
		if blocks != nil {
			parts := make([]oai.ChatCompletionContentPartUnionParam, 0, len(blocks))
			for _, block := range blocks {
				switch block.Type {
				case "text":
					parts = append(parts, oai.ChatCompletionContentPartUnionParam{
						OfText: &oai.ChatCompletionContentPartTextParam{
							Text: block.Text,
						},
					})
				case "image":
					parts = append(parts, oai.ChatCompletionContentPartUnionParam{
						OfImageURL: &oai.ChatCompletionContentPartImageParam{
							ImageURL: oai.ChatCompletionContentPartImageImageURLParam{
								URL: "data:" + block.MediaType + ";base64," + block.ImageB64,
							},
						},
					})
				default:
					// Unknown block types are skipped (consistent with other
					// providers); log at debug so misconfigured callers can
					// diagnose silently dropped content.
					p.log().Debug("openai: skipping unknown content block type",
						"block_type", block.Type, "provider", p.name)
				}
			}
			return oai.UserMessage(parts)
		}
		return oai.UserMessage(content)
	case "assistant":
		assistantParam := oai.ChatCompletionAssistantMessageParam{
			Content: oai.ChatCompletionAssistantMessageParamContentUnion{
				OfString: oai.String(content),
			},
		}
		if len(msg.ToolCalls) > 0 {
			toolCalls := make([]oai.ChatCompletionMessageToolCallParam, len(msg.ToolCalls))
			for i, tc := range msg.ToolCalls {
				toolCalls[i] = oai.ChatCompletionMessageToolCallParam{
					ID:   tc.ID,
					Type: "function",
					Function: oai.ChatCompletionMessageToolCallFunctionParam{
						Name:      tc.Name,
						Arguments: string(tc.Input),
					},
				}
			}
			assistantParam.ToolCalls = toolCalls
		}
		// DeepSeek V4 requires reasoning_content to be echoed back for ALL
		// assistant messages in thinking mode, even when empty. Constructed
		// assistant messages (e.g., nudges without tool calls) must also
		// include the field to avoid 400 errors. OmitReasoningHistory opts
		// an endpoint out of the echo: a server whose chat template rejects
		// the unknown field (e.g. llama.cpp without a DeepSeek-style
		// template) fails the whole request on it.
		if !p.omitReasoningHistory {
			assistantParam.SetExtraFields(map[string]any{
				"reasoning_content": msg.ReasoningContent,
			})
		}
		return oai.ChatCompletionMessageParamUnion{
			OfAssistant: &assistantParam,
		}
	case "tool":
		return oai.ToolMessage(content, msg.ToolCallID)
	default:
		return oai.UserMessage(content)
	}
}

// convertChatResponseMessage converts OpenAI's message to our Message format.
func (p *OpenAIProvider) convertChatResponseMessage(msg oai.ChatCompletionMessage) Message {
	result := Message{
		Role:    string(msg.Role),
		Content: msg.Content,
	}

	// Extract reasoning_content from raw JSON (DeepSeek extension).
	result.ReasoningContent = extractReasoningContent(msg.RawJSON())

	if len(msg.ToolCalls) > 0 {
		result.ToolCalls = make([]ToolCall, len(msg.ToolCalls))
		for i, tc := range msg.ToolCalls {
			result.ToolCalls[i] = ToolCall{
				ID:    tc.ID,
				Name:  tc.Function.Name,
				Input: json.RawMessage(tc.Function.Arguments),
			}
		}
	}

	logToolCallArguments(p.log(), p.name, result.ToolCalls)

	return result
}

// extractReasoningContent extracts the "reasoning_content" field from raw JSON.
// This is a DeepSeek-specific extension to the OpenAI chat completions format.
func extractReasoningContent(rawJSON string) string {
	if rawJSON == "" {
		return ""
	}
	var payload struct {
		ReasoningContent string `json:"reasoning_content"`
	}
	if err := json.Unmarshal([]byte(rawJSON), &payload); err != nil {
		return ""
	}
	return payload.ReasoningContent
}

// enrichOpenAIError extracts a human-readable message from an OpenAI SDK *oai.Error.
//
// The OpenAI SDK only parses the nested {"error":{...}} envelope (it unmarshals
// gjson.Get(body, "error").Raw). For responses that use a non-standard error
// shape — e.g. OpenCode Zen returns {"detail":"..."} — the Message field and
// RawJSON() are both empty, and the upstream message survives only in
// Response.Body. Without this enrichment a 400 surfaces as a bare
// "400 Bad Request" with an empty body.
//
// The lookup order is:
//  1. apiErr.Message (standard OpenAI error envelope).
//  2. The raw response body (covers non-standard envelopes), truncated.
//
// Retryable classification is intentionally left untouched — callers classify by
// HTTP status code, not by the message contents.
func enrichOpenAIError(apiErr *oai.Error) string {
	if msg := strings.TrimSpace(apiErr.Message); msg != "" {
		return msg
	}
	return readOpenAIErrorBody(apiErr)
}

// readOpenAIErrorBody reads the raw response body from an *oai.Error, truncated
// to a sane size. It is nil-safe: Response/Body may be unset (e.g. in tests or
// when the error was constructed without an HTTP round-trip).
func readOpenAIErrorBody(apiErr *oai.Error) string {
	if apiErr == nil || apiErr.Response == nil || apiErr.Response.Body == nil {
		return ""
	}
	body, err := io.ReadAll(apiErr.Response.Body)
	if err != nil {
		return ""
	}
	body = bytes.TrimSpace(body)
	const maxErrorBodySize = 4096 // 4 KiB cap to keep error messages bounded
	if len(body) > maxErrorBodySize {
		return string(body[:maxErrorBodySize]) + "..."
	}
	return string(body)
}

// wrapError maps OpenAI SDK error types to *Error.
func (p *OpenAIProvider) wrapError(err error) error {
	var apiErr *oai.Error
	if errors.As(err, &apiErr) {
		if msg := enrichOpenAIError(apiErr); msg != "" {
			return WrapProviderError(p.name, apiErr.StatusCode, fmt.Errorf("%s: %w", msg, err))
		}
		return WrapProviderError(p.name, apiErr.StatusCode, err)
	}
	// Fallback: check for net errors directly
	return WrapProviderError(p.name, 0, err)
}

// isResponsesEndpointUnsupported reports whether err indicates that the
// Responses API endpoint (/v1/responses) is not implemented by the provider —
// i.e. HTTP 404 Not Found or 405 Method Not Allowed. It is used to decide
// whether to surface a clear "Responses API required but unavailable" error for
// codex/gpt-5.x/gpt-6-family models, which are served exclusively via
// /responses and
// for which a Chat Completions fallback is known to fail (degenerate HTTP 400
// with an empty body). Other statuses (400/500/...) mean the endpoint exists but
// the request or processing failed, so they are NOT treated as "unsupported".
func isResponsesEndpointUnsupported(err error) bool {
	var llmErr *Error
	if errors.As(err, &llmErr) {
		return llmErr.StatusCode == http.StatusNotFound || llmErr.StatusCode == http.StatusMethodNotAllowed
	}
	return false
}

// isStreamOptionsUnsupported reports whether err is an HTTP 400 that names the
// stream_options request field — the shape older OpenAI-compatible servers
// (vLLM versions predating stream_options, certain Azure api-versions, proxies)
// return when asked for stream usage accounting. It gates the single retry
// without StreamOptions so that enabling streaming ("always safe" per the
// WithStreaming contract) also holds on those endpoints. A 400 that merely
// mentions the field for a different reason retries once and, if the retry
// fails too, surfaces that second error — the real one.
func isStreamOptionsUnsupported(err error) bool {
	var llmErr *Error
	if !errors.As(err, &llmErr) || llmErr.StatusCode != http.StatusBadRequest {
		return false
	}
	return strings.Contains(err.Error(), "stream_options")
}
