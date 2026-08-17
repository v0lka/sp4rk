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
func newResponsesClient(apiKey, baseURL string, httpClient *http.Client) *oai.Client {
	opts := []option.RequestOption{
		option.WithAPIKey(apiKey),
	}
	if baseURL != "" {
		opts = append(opts, option.WithBaseURL(baseURL))
	}
	if httpClient != nil {
		opts = append(opts, option.WithHTTPClient(httpClient))
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

	return convertResponsesResponse(resp)
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
			for _, ri := range msg.ReasoningItems {
				summaryParams := make([]responses.ResponseReasoningItemSummaryParam, 0, 1)
				if ri.Summary != "" {
					summaryParams = append(summaryParams, responses.ResponseReasoningItemSummaryParam{
						Text: ri.Summary,
					})
				}
				items = append(items, responses.ResponseInputItemParamOfReasoning(ri.ID, summaryParams))
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
						Name:      tc.Name,
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
			var summaryText strings.Builder
			for _, s := range item.Summary {
				if s.Text != "" {
					if summaryText.Len() > 0 {
						summaryText.WriteString("\n")
					}
					summaryText.WriteString(s.Text)
				}
			}
			summary := summaryText.String()
			if summary != "" {
				reasoningParts = append(reasoningParts, summary)
			}
			if item.ID != "" {
				message.ReasoningItems = append(message.ReasoningItems, ReasoningItem{
					ID:      item.ID,
					Summary: summary,
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
