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
	"strings"

	oai "github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

// OpenAIProviderConfig contains configuration for OpenAI-compatible providers.
type OpenAIProviderConfig struct {
	Name       string // logical provider name ("openai", "deepseek", "grok", etc.)
	APIKey     string
	BaseURL    string       // empty = default OpenAI; otherwise custom endpoint
	HTTPClient *http.Client // optional proxy-configured HTTP client (nil = default)
	Logger     *slog.Logger // optional structured logger (nil = slog.Default())
}

// OpenAIProvider implements Provider for OpenAI and compatible APIs.
type OpenAIProvider struct {
	client            *oai.Client        // official SDK for Chat Completions API
	responsesClient   *oai.Client        // official SDK for Responses API
	anthropicDelegate *AnthropicProvider // serves ProtocolAnthropic models (Claude) via a co-located Anthropic provider using the same baseURL/APIKey/HTTPClient/Logger
	name              string
	baseURL           string       // empty = default OpenAI; non-empty = compatible provider
	apiKey            string       // API key (passed to the Google delegate; empty for local backends)
	httpClient        *http.Client // optional proxy-configured HTTP client (passed to the Google delegate; nil = http.DefaultClient)
	logger            *slog.Logger
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
	client := oai.NewClient(opts...)

	responsesClient := newResponsesClient(cfg.APIKey, cfg.BaseURL, cfg.HTTPClient)

	// Build a co-located Anthropic provider so ProtocolAnthropic models (Claude)
	// served by an OpenAI-compatible gateway (e.g. Zen) can be delegated to the
	// existing Anthropic Messages implementation — reusing the same
	// baseURL/APIKey/HTTPClient/Logger, with no code duplication. This is only
	// invoked on the ProtocolAnthropic dispatch path; for purely OpenAI
	// providers the delegate is constructed but never used. The delegate is built
	// from the shared connection fields explicitly (rather than via a direct
	// type conversion) so that adding Anthropic-specific fields to
	// AnthropicProviderConfig later cannot silently drop or mis-map a field.
	//nolint:staticcheck // S1016: explicit field copy is deliberate — see comment above; survives future divergent fields.
	anthropicDelegate, err := NewAnthropicProvider(AnthropicProviderConfig{
		Name:       cfg.Name,
		APIKey:     cfg.APIKey,
		BaseURL:    cfg.BaseURL,
		HTTPClient: cfg.HTTPClient,
		Logger:     cfg.Logger,
	})
	if err != nil {
		return nil, fmt.Errorf("openai: failed to build anthropic delegate: %w", err)
	}

	return &OpenAIProvider{
		client:            &client,
		responsesClient:   responsesClient,
		anthropicDelegate: anthropicDelegate,
		name:              cfg.Name,
		baseURL:           cfg.BaseURL,
		apiKey:            cfg.APIKey,
		httpClient:        cfg.HTTPClient,
		logger:            cfg.Logger,
	}, nil
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
	//   - ProtocolResponses (GPT-5.x, Codex) → Responses API (/v1/responses).
	//     These models are served exclusively via /responses — both the
	//     official OpenAI endpoint and compatible gateways (e.g. OpenCode Zen
	//     exposes gpt-5.x / gpt-5.x-codex there) — and sending them to
	//     /chat/completions returns a degenerate HTTP 400 with an empty body.
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
		resp, err := responsesAPICompletion(ctx, p.responsesClient, p.name, p.baseURL, req, p.logger)
		if err == nil {
			return resp, nil
		}
		// GPT-5.x / Codex models require the Responses API: both the official
		// endpoint and compatible gateways serve them only via /responses, and
		// /chat/completions returns a degenerate HTTP 400 with an empty body.
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
		return googleCompletion(ctx, googleCompletionConfig{
			HTTPClient:   p.httpClient,
			BaseURL:      p.baseURL,
			APIKey:       p.apiKey,
			ProviderName: p.name,
		}, req)
	case ProtocolChatCompletions:
		// Handled by the shared Chat Completions path below.
	}

	params := p.buildChatParams(req)

	resp, err := p.client.Chat.Completions.New(ctx, params)
	if err != nil {
		return nil, p.wrapError(fmt.Errorf("openai chat completion: %w", err))
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
			params.SetExtraFields(map[string]any{
				"thinking": map[string]string{"type": req.ReasoningEffort},
			})
		case "qwen":
			params.SetExtraFields(map[string]any{
				"enable_thinking": req.ReasoningEffort == "On",
			})
		case "glm":
			applyGLMReasoning(&params, req.Model, req.ReasoningEffort)
		}
	}

	if len(req.Tools) > 0 {
		tools := make([]oai.ChatCompletionToolParam, len(req.Tools))
		for i, tool := range req.Tools {
			tools[i] = oai.ChatCompletionToolParam{
				Function: oai.FunctionDefinitionParam{
					Name:        tool.Name,
					Description: oai.String(tool.Description),
					Parameters:  p.convertSchemaToMap(SanitizeSchemaForOpenAI(tool.InputSchema)),
				},
			}
		}
		params.Tools = tools
	}

	return params
}

// applyGLMReasoning sets the reasoning-related extra fields for GLM models.
//
// GLM 5.2+ supports the reasoning_effort parameter (values "max"/"high") which
// is honored when thinking is enabled:
//
//   - "none": thinking disabled, no reasoning_effort
//   - "max":  thinking enabled,  reasoning_effort=max
//   - "high": thinking enabled,  reasoning_effort=high
//
// An empty effort (the UI "Auto"/Default selection) is left unset: GLM 5.2
// enables thinking by default with reasoning_effort=max, so Auto == "max".
//
// Older GLM models (< 5.2) keep the legacy thinking on/off control, passing the
// native effort value ("On"/"Off") directly as thinking.type.
func applyGLMReasoning(params *oai.ChatCompletionNewParams, model, effort string) {
	if IsGLM52OrLater(model) {
		switch effort {
		case "none":
			params.SetExtraFields(map[string]any{
				"thinking": map[string]string{"type": "disabled"},
			})
		case "max":
			params.SetExtraFields(map[string]any{
				"thinking":         map[string]string{"type": "enabled"},
				"reasoning_effort": "max",
			})
		case "high":
			params.SetExtraFields(map[string]any{
				"thinking":         map[string]string{"type": "enabled"},
				"reasoning_effort": "high",
			})
		}
		return
	}
	params.SetExtraFields(map[string]any{
		"thinking": map[string]string{"type": effort},
	})
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
		// include the field to avoid 400 errors.
		assistantParam.SetExtraFields(map[string]any{
			"reasoning_content": msg.ReasoningContent,
		})
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
// codex/gpt-5.x-family models, which are served exclusively via /responses and
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
