package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strings"

	oai "github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/responses"
	"github.com/openai/openai-go/shared"
)

// newResponsesClient creates an official OpenAI SDK client for the Responses API.
// tokenSource optionally injects per-request bearer credentials (nil = static
// apiKey only); see TokenSource and tokenSourceMiddleware.
func newResponsesClient(apiKey, baseURL string, httpClient *http.Client, tokenSource TokenSource) *oai.Client {
	opts := []option.RequestOption{
		option.WithAPIKey(apiKey),
	}
	if baseURL != "" {
		opts = append(opts, option.WithBaseURL(baseURL))
	}
	if httpClient != nil {
		opts = append(opts, option.WithHTTPClient(httpClient))
	}
	if tokenSource != nil {
		opts = append(opts, option.WithMiddleware(tokenSourceMiddleware(tokenSource)))
	}
	client := oai.NewClient(opts...)
	return &client
}

// responsesAPICompletion performs a non-streaming Responses API call.
// baseURL is the provider's configured base URL (empty = official OpenAI).
// logger is used for debug-level diagnostics (nil = slog.Default()).
func responsesAPICompletion(ctx context.Context, client *oai.Client, providerName, baseURL string, req ChatRequest, logger *slog.Logger) (*ChatResponse, error) {
	// Validate image blocks up front so a missing MediaType/ImageB64 yields a
	// clear local error instead of an opaque API 400.
	for _, msg := range req.Messages {
		if err := ValidateContentBlocks(msg.ContentBlocks); err != nil {
			return nil, fmt.Errorf("%s: %w", providerName, err)
		}
	}

	params := buildResponsesParams(req, baseURL, logger)

	resp, err := client.Responses.New(ctx, params)
	if err != nil {
		return nil, wrapResponsesError(providerName, err)
	}

	converted, err := convertResponsesResponse(resp)
	if err != nil {
		return nil, err
	}
	logToolCallArguments(logger, providerName, converted.Message.ToolCalls)
	return converted, nil
}

// responsesAPICompletionStream performs a streaming Responses API call. It is
// selected when the endpoint requires streaming on the wire (requireStreaming,
// e.g. a ChatGPT OAuth Codex backend that answers non-streaming requests with
// an error) or when the caller opted into incremental delivery via
// req.DeltaSink. Text deltas (response.output_text.delta) and reasoning
// summary deltas (response.reasoning_summary_text.delta) are forwarded to
// req.DeltaSink as they arrive (when set); a sink error aborts the stream and
// is surfaced to the caller unchanged (host-side cancellation).
//
// The final ChatResponse is assembled from the terminal event's Response —
// response.completed / response.incomplete carry usage and status — via the
// same convertResponsesResponse the synchronous path uses, so the host still
// receives one fully assembled *ChatResponse, exactly as today. The deltas
// and completed output items (response.output_item.done) are additionally
// accumulated in-call and merged in as a strict fallback (see
// applyStreamedOutputFallback): some backends — observed on ChatGPT OAuth
// Codex endpoints — stream the full completion but deliver the terminal
// Response with an empty output list, which would otherwise surface as an
// empty Message (hosts then fail one-shot JSON extraction). Failure events
// are surfaced as errors instead of a degenerate empty response: an `error`
// event mid-stream and a terminal response.failed both abort with the event's
// code/message (the streaming analogue of the synchronous path's HTTP-level
// failures).
//
// When requireStreaming is set, the request additionally pins Store=false,
// includes reasoning.encrypted_content, drops the max_output_tokens cap and
// clamps the reasoning effort "minimal" down to "low": stateless Codex-style
// backends keep no stored response, so the encrypted reasoning payload is
// the only way to round-trip reasoning items across turns
// (ReasoningItem.EncryptedContent is sent back by convertToResponsesInput),
// and they reject both the output cap and the "minimal" effort outright with
// HTTP 400 "Unsupported parameter: max_output_tokens" / "'minimal' is not
// supported".
func responsesAPICompletionStream(ctx context.Context, client *oai.Client, providerName, baseURL string, req ChatRequest, requireStreaming bool, logger *slog.Logger) (*ChatResponse, error) {
	// Validate image blocks up front so a missing MediaType/ImageB64 yields a
	// clear local error instead of an opaque API 400.
	for _, msg := range req.Messages {
		if err := ValidateContentBlocks(msg.ContentBlocks); err != nil {
			return nil, fmt.Errorf("%s: %w", providerName, err)
		}
	}

	params := buildResponsesParams(req, baseURL, logger)
	if requireStreaming {
		params.Store = param.NewOpt(false)
		params.Include = append(params.Include, responses.ResponseIncludableReasoningEncryptedContent)
		// Codex-style backends reject the output cap outright (HTTP 400
		// "Unsupported parameter: max_output_tokens"), so the cap must not
		// travel on the wire for them. The zero Opt omits the field entirely;
		// req.MaxTokens keeps its meaning only for endpoints that accept it.
		params.MaxOutputTokens = param.Opt[int64]{}
		// They also accept only the low..max effort ladder and reject
		// "minimal" — valid on api.openai.com — with a retryable=false
		// HTTP 400 "'minimal' is not supported with the ... model.
		// Supported values are: 'low', 'medium', 'high', 'xhigh', 'max'".
		// Clamp to the nearest supported rung.
		if params.Reasoning.Effort == "minimal" {
			params.Reasoning.Effort = "low"
		}
	}

	// NewStreaming sets stream:true on the wire request itself.
	stream := client.Responses.NewStreaming(ctx, params)
	defer func() { _ = stream.Close() }()

	var terminal *responses.Response
	var (
		textBuf      strings.Builder                     // response.output_text.delta payload
		reasoningBuf strings.Builder                     // response.reasoning_summary_text.delta payload
		doneItems    []responses.ResponseOutputItemUnion // items from response.output_item.done
	)
	for stream.Next() {
		ev := stream.Current()
		switch ev.Type {
		case "response.output_text.delta":
			if d := ev.Delta.OfString; d != "" {
				textBuf.WriteString(d)
				if req.DeltaSink != nil {
					if err := req.DeltaSink(StreamDelta{Text: d}); err != nil {
						return nil, err
					}
				}
			}
		case "response.reasoning_summary_text.delta":
			if d := ev.Delta.OfString; d != "" {
				reasoningBuf.WriteString(d)
				if req.DeltaSink != nil {
					if err := req.DeltaSink(StreamDelta{Reasoning: d}); err != nil {
						return nil, err
					}
				}
			}
		case "response.output_item.done":
			doneItems = append(doneItems, ev.Item)
		case "error":
			// The flat union carries Code/Message/Param directly; the SDK's
			// stream layer only bails on a top-level `error` member, which the
			// error event itself does not carry (its fields are flat).
			return nil, WrapProviderError(providerName, 0,
				fmt.Errorf("responses API (stream): error event: %s: %s", ev.Code, ev.Message))
		case "response.completed", "response.incomplete":
			r := ev.Response
			terminal = &r
		case "response.failed":
			r := ev.Response
			return nil, WrapProviderError(providerName, 0,
				fmt.Errorf("responses API (stream): response failed: %s: %s", r.Error.Code, r.Error.Message))
		}
	}
	if err := stream.Err(); err != nil {
		return nil, wrapResponsesError(providerName, err)
	}
	if terminal == nil {
		return nil, WrapProviderError(providerName, 0,
			errors.New("responses API (stream): stream ended without a terminal response event"))
	}

	converted, err := convertResponsesResponse(terminal)
	if err != nil {
		return nil, err
	}
	applyStreamedOutputFallback(converted, textBuf.String(), reasoningBuf.String(), doneItems)
	logToolCallArguments(logger, providerName, converted.Message.ToolCalls)
	return converted, nil
}

// buildResponsesParams constructs ResponseNewParams from a ChatRequest.
// baseURL is the provider's configured base URL (empty = official OpenAI);
// it controls whether OpenAI-specific fields like `store` are included.
// logger is used for debug-level diagnostics (nil = slog.Default()).
func buildResponsesParams(req ChatRequest, baseURL string, logger *slog.Logger) responses.ResponseNewParams {
	systemPrompt, filteredMsgs := ExtractSystemPrompt(req.Messages)

	params := responses.ResponseNewParams{
		Model: req.Model,
		Input: responses.ResponseNewParamsInputUnion{
			OfInputItemList: convertToResponsesInput(filteredMsgs, logger),
		},
	}

	// `store` is an OpenAI-specific Responses API parameter. Compatible
	// providers (custom baseURL) may not support it and return 400. Only
	// send it for the official OpenAI endpoint where it disables server-side
	// response storage for privacy.
	if baseURL == "" {
		params.Store = param.NewOpt(false)
	}

	if systemPrompt != "" {
		params.Instructions = param.NewOpt(systemPrompt)
	}

	if req.MaxTokens > 0 {
		params.MaxOutputTokens = param.NewOpt(int64(req.MaxTokens))
	}

	// The Responses API supports temperature and top_p; it has no
	// presence_penalty/top_k/repetition_penalty parameters.
	if req.Temperature != nil {
		params.Temperature = param.NewOpt(*req.Temperature)
	}
	if req.TopP != nil {
		params.TopP = param.NewOpt(*req.TopP)
	}

	// Only send reasoning if the effort value is valid for the OpenAI
	// Responses API. Invalid values (e.g. "On" from Anthropic/GLM families)
	// cause a 400 error. Valid values: minimal, low, medium, high, max.
	if isValidResponsesReasoningEffort(req.ReasoningEffort) {
		params.Reasoning = shared.ReasoningParam{
			Effort:  shared.ReasoningEffort(req.ReasoningEffort),
			Summary: shared.ReasoningSummaryAuto,
		}
	}

	if len(req.Tools) > 0 {
		params.Tools = convertToResponsesTools(req.Tools)
	}

	return params
}

// isValidResponsesReasoningEffort returns true if the effort string is a valid
// value for the OpenAI Responses API reasoning.effort parameter. The Responses
// API accepts: "minimal", "low", "medium", "high", "max" (max for Codex models).
// Other family-specific values like "On"/"Off" (Anthropic, GLM, Qwen) or
// "Max"/"High" (DeepSeek) are not accepted by the Responses API.
func isValidResponsesReasoningEffort(effort string) bool {
	switch effort {
	case "minimal", "low", "medium", "high", "max":
		return true
	default:
		return false
	}
}

// loggerOrDefault returns l if non-nil, otherwise slog.Default().
func loggerOrDefault(l *slog.Logger) *slog.Logger {
	if l != nil {
		return l
	}
	return slog.Default()
}

// convertToResponsesInput converts internal messages to Responses API input items.
// logger is used for debug-level diagnostics when unknown block types are
// encountered (nil = slog.Default()).
func convertToResponsesInput(messages []Message, logger *slog.Logger) responses.ResponseInputParam {
	logger = loggerOrDefault(logger)
	items := make(responses.ResponseInputParam, 0, len(messages))
	for _, msg := range messages {
		switch msg.Role {
		case "user":
			// When ContentBlocks are present, render them as multipart content
			// (input_text and/or input_image parts) instead of the plain
			// Content string. NormalizeContentBlocks prepends Content as a
			// text block when the blocks carry no text, so the task text
			// always reaches the model. Text-only messages without
			// ContentBlocks keep the existing OfString path.
			blocks := NormalizeContentBlocks(msg)
			if blocks != nil {
				contentList := make(responses.ResponseInputMessageContentListParam, 0, len(blocks))
				for _, block := range blocks {
					switch block.Type {
					case "text":
						contentList = append(contentList, responses.ResponseInputContentParamOfInputText(block.Text))
					case "image":
						contentList = append(contentList, responses.ResponseInputContentUnionParam{
							OfInputImage: &responses.ResponseInputImageParam{
								Detail:   responses.ResponseInputImageDetailAuto,
								ImageURL: param.NewOpt("data:" + block.MediaType + ";base64," + block.ImageB64),
							},
						})
					default:
						// Unknown block types are skipped (consistent with other
						// providers); log at debug so misconfigured callers can
						// diagnose silently dropped content.
						logger.Debug("openai responses: skipping unknown content block type",
							"block_type", block.Type)
					}
				}
				items = append(items, responses.ResponseInputItemUnionParam{
					OfMessage: &responses.EasyInputMessageParam{
						Role: responses.EasyInputMessageRoleUser,
						Content: responses.EasyInputMessageContentUnionParam{
							OfInputItemContentList: contentList,
						},
					},
				})
			} else {
				items = append(items, responses.ResponseInputItemUnionParam{
					OfMessage: &responses.EasyInputMessageParam{
						Role: responses.EasyInputMessageRoleUser,
						Content: responses.EasyInputMessageContentUnionParam{
							OfString: param.NewOpt(msg.Content),
						},
					},
				})
			}

		case "assistant":
			// Re-emit reasoning items first (with their original IDs) so the
			// Responses API can maintain the reasoning chain across turns.
			// This is critical for reasoning models (e.g. Codex): without
			// round-tripping reasoning items, the model loses its committed
			// plan between ReAct iterations and reverts to read-only exploration.
			// EncryptedContent (present when the response was generated with
			// include reasoning.encrypted_content) is sent back verbatim so a
			// stateless backend can decrypt the reasoning it did not store.
			for _, ri := range msg.ReasoningItems {
				summaryParams := make([]responses.ResponseReasoningItemSummaryParam, 0, 1)
				if ri.Summary != "" {
					summaryParams = append(summaryParams, responses.ResponseReasoningItemSummaryParam{
						Text: ri.Summary,
					})
				}
				reasoningParam := responses.ResponseReasoningItemParam{
					ID:      ri.ID,
					Summary: summaryParams,
				}
				if ri.EncryptedContent != "" {
					reasoningParam.EncryptedContent = param.NewOpt(ri.EncryptedContent)
				}
				items = append(items, responses.ResponseInputItemUnionParam{
					OfReasoning: &reasoningParam,
				})
			}

			// If assistant has tool calls, add the text message (if any) and then each function_call
			if msg.Content != "" {
				items = append(items, responses.ResponseInputItemUnionParam{
					OfMessage: &responses.EasyInputMessageParam{
						Role: responses.EasyInputMessageRoleAssistant,
						Content: responses.EasyInputMessageContentUnionParam{
							OfString: param.NewOpt(msg.Content),
						},
					},
				})
			}
			for _, tc := range msg.ToolCalls {
				items = append(items, responses.ResponseInputItemUnionParam{
					OfFunctionCall: &responses.ResponseFunctionToolCallParam{
						CallID:    tc.ID,
						Name:      sanitizeResponsesFunctionName(tc.Name),
						Arguments: string(tc.Input),
					},
				})
			}

		case "tool":
			output := msg.Content
			if output == "" {
				output = "(no output)"
			}
			items = append(items, responses.ResponseInputItemUnionParam{
				OfFunctionCallOutput: &responses.ResponseInputItemFunctionCallOutputParam{
					CallID: msg.ToolCallID,
					Output: output,
				},
			})
		}
	}
	return items
}

// sanitizeResponsesFunctionName rewrites a tool-call name so it always
// satisfies the Responses API input pattern ^[a-zA-Z0-9_-]+$ and the 64-char
// cap shared with tool definitions. Model-hallucinated names (observed in
// production: "functions.get_me", "functions.list_projects") round-trip fine
// as model output but are rejected with a retryable=false HTTP 400
// "Invalid 'input[N].name': string does not match pattern" when the history
// is re-sent as input — poisoning every subsequent request of the
// conversation. Mapping offending runes to '_' keeps the wire valid; the
// call_id ↔ function_call_output pairing, the only correlation the backend
// and the host need, is untouched, and a sanitized name can never
// accidentally match a registered tool (a registered tool's name already
// satisfies the pattern, so sanitization is the identity for real calls).
func sanitizeResponsesFunctionName(name string) string {
	if len(name) > 64 {
		name = name[:64]
	}
	var b strings.Builder
	b.Grow(len(name))
	for _, r := range name {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '_', r == '-':
			b.WriteRune(r)
		default:
			b.WriteByte('_')
		}
	}
	if b.Len() == 0 {
		return "unknown"
	}
	return b.String()
}

// convertToResponsesTools converts internal tool definitions to Responses API tools.
func convertToResponsesTools(tools []ToolDefinition) []responses.ToolUnionParam {
	result := make([]responses.ToolUnionParam, 0, len(tools))
	for _, tool := range tools {
		var params map[string]any
		if len(tool.InputSchema) > 0 {
			sanitized := SanitizeSchemaForOpenAI(tool.InputSchema)
			if err := json.Unmarshal(sanitized, &params); err != nil {
				params = map[string]any{
					"type":                 "object",
					"properties":           map[string]any{},
					"required":             []any{},
					"additionalProperties": false,
				}
			}
		}
		result = append(result, responses.ToolParamOfFunction(
			tool.Name,
			params,
			true,
		))
		// Set description on the just-created tool
		if tool.Description != "" {
			result[len(result)-1].OfFunction.Description = param.NewOpt(tool.Description)
		}
	}
	return result
}

// mapResponsesStopReason maps a Responses API response to a standard stop reason.
func mapResponsesStopReason(resp *responses.Response) string {
	// Check if output contains function calls
	for _, item := range resp.Output {
		if item.Type == "function_call" {
			return "tool_use"
		}
	}

	switch resp.Status {
	case responses.ResponseStatusCompleted:
		return "end_turn"
	case responses.ResponseStatusIncomplete:
		if resp.IncompleteDetails.Reason == "max_output_tokens" {
			return "max_tokens"
		}
		return "end_turn"
	case responses.ResponseStatusFailed:
		return "error"
	case responses.ResponseStatusCancelled:
		return "end_turn"
	default:
		return "end_turn"
	}
}

// convertResponsesResponse converts a Responses API response to our ChatResponse.
func convertResponsesResponse(resp *responses.Response) (*ChatResponse, error) {
	if resp == nil {
		return nil, errors.New("responses API: nil response")
	}

	message := Message{
		Role:    "assistant",
		Content: resp.OutputText(),
	}

	// Extract reasoning items and function_call items from the response output.
	var reasoningParts []string
	for _, item := range resp.Output {
		switch item.Type {
		case "reasoning":
			// Collect summary text from the reasoning item.
			summary := responsesReasoningSummary(item)
			if summary != "" {
				reasoningParts = append(reasoningParts, summary)
			}
			if item.ID != "" {
				message.ReasoningItems = append(message.ReasoningItems, ReasoningItem{
					ID:               item.ID,
					Summary:          summary,
					EncryptedContent: item.EncryptedContent,
				})
			}
		case "function_call":
			message.ToolCalls = append(message.ToolCalls, ToolCall{
				ID:    item.CallID,
				Name:  item.Name,
				Input: json.RawMessage(item.Arguments),
			})
		}
	}

	// Populate ReasoningContent and Reasoning from the concatenated reasoning summaries.
	// This makes reasoning visible to the UI and to non-Responses transports.
	if len(reasoningParts) > 0 {
		message.ReasoningContent = strings.Join(reasoningParts, "\n")
	}

	stopReason := mapResponsesStopReason(resp)

	result := &ChatResponse{
		Message:    message,
		StopReason: stopReason,
		Usage: TokenUsage{
			InputTokens:  int(resp.Usage.InputTokens),
			OutputTokens: int(resp.Usage.OutputTokens),
		},
	}
	result.Reasoning = message.ReasoningContent
	return result, nil
}

// responsesReasoningSummary concatenates an output item's reasoning summary
// texts, newline-separated — the single spelling shared by
// convertResponsesResponse and applyStreamedOutputFallback so reasoning
// surfaces identically whichever path produced it.
func responsesReasoningSummary(item responses.ResponseOutputItemUnion) string {
	var b strings.Builder
	for _, s := range item.Summary {
		if s.Text != "" {
			if b.Len() > 0 {
				b.WriteString("\n")
			}
			b.WriteString(s.Text)
		}
	}
	return b.String()
}

// applyStreamedOutputFallback merges the in-call streamed output into
// converted when the terminal event's Response omitted it. Some Responses
// backends (observed on ChatGPT OAuth Codex endpoints) bill and stream the
// full completion via response.output_text.delta /
// response.reasoning_summary_text.delta and response.output_item.done, yet
// deliver response.completed with an empty output list — assembling from the
// terminal alone then yields an empty Message, and hosts fail downstream
// (empty one-shot results, ErrNoJSON on routing extraction). The merge is
// strictly fallback: terminal content, reasoning and tool calls are never
// overridden, and appended items are de-duplicated by ID so a backend that
// emits both shapes does not double them.
func applyStreamedOutputFallback(converted *ChatResponse, text, reasoning string, doneItems []responses.ResponseOutputItemUnion) {
	if converted == nil {
		return
	}

	// Message text: accumulated deltas first, else the message items that
	// arrived via response.output_item.done (no-delta backends).
	if converted.Message.Content == "" {
		if text != "" {
			converted.Message.Content = text
		} else {
			var b strings.Builder
			for _, item := range doneItems {
				if item.Type != "message" {
					continue
				}
				for _, part := range item.Content {
					if part.Type == "output_text" {
						b.WriteString(part.Text)
					}
				}
			}
			converted.Message.Content = b.String()
		}
	}

	// Reasoning: same policy — deltas first, else reasoning-item summaries
	// from output_item.done.
	if converted.Message.ReasoningContent == "" {
		if reasoning != "" {
			converted.Message.ReasoningContent = reasoning
		} else {
			var parts []string
			for _, item := range doneItems {
				if item.Type != "reasoning" {
					continue
				}
				if summary := responsesReasoningSummary(item); summary != "" {
					parts = append(parts, summary)
				}
			}
			if len(parts) > 0 {
				converted.Message.ReasoningContent = strings.Join(parts, "\n")
			}
		}
		if converted.Message.ReasoningContent != "" {
			converted.Reasoning = converted.Message.ReasoningContent
		}
	}

	// Output items the terminal Response omitted: reasoning items (carrying
	// EncryptedContent for the stateless round-trip) and function calls,
	// appended only when their IDs are not already present.
	knownReasoning := make(map[string]bool, len(converted.Message.ReasoningItems))
	for _, ri := range converted.Message.ReasoningItems {
		if ri.ID != "" {
			knownReasoning[ri.ID] = true
		}
	}
	knownCalls := make(map[string]bool, len(converted.Message.ToolCalls))
	for _, tc := range converted.Message.ToolCalls {
		if tc.ID != "" {
			knownCalls[tc.ID] = true
		}
	}
	for _, item := range doneItems {
		switch item.Type {
		case "reasoning":
			if item.ID == "" || knownReasoning[item.ID] {
				continue
			}
			knownReasoning[item.ID] = true
			converted.Message.ReasoningItems = append(converted.Message.ReasoningItems, ReasoningItem{
				ID:               item.ID,
				Summary:          responsesReasoningSummary(item),
				EncryptedContent: item.EncryptedContent,
			})
		case "function_call":
			if item.CallID == "" || knownCalls[item.CallID] {
				continue
			}
			knownCalls[item.CallID] = true
			converted.Message.ToolCalls = append(converted.Message.ToolCalls, ToolCall{
				ID:    item.CallID,
				Name:  item.Name,
				Input: json.RawMessage(item.Arguments),
			})
		}
	}

	// A merged function call means the model asked for a tool; mirror
	// mapResponsesStopReason's precedence (any function_call ⇒ tool_use).
	if len(converted.Message.ToolCalls) > 0 {
		converted.StopReason = "tool_use"
	}
}

// wrapResponsesError wraps errors from the official OpenAI SDK into our error type.
func wrapResponsesError(providerName string, err error) error {
	var apiErr *oai.Error
	if errors.As(err, &apiErr) {
		if msg := enrichOpenAIError(apiErr); msg != "" {
			return WrapProviderError(providerName, apiErr.StatusCode, fmt.Errorf("responses API: %s: %w", msg, err))
		}
		return WrapProviderError(providerName, apiErr.StatusCode, fmt.Errorf("responses API: %w", err))
	}
	return WrapProviderError(providerName, 0, fmt.Errorf("responses API: %w", err))
}
