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
	"net/http/httptest"
	"net/url"
	"os"
	"strings"
	"testing"

	oai "github.com/openai/openai-go"
)

func TestOpenAIProvider_ImplementsInterface(t *testing.T) {
	var _ Provider = (*OpenAIProvider)(nil)
}

func TestOpenAIProvider_CustomBaseURL(t *testing.T) {
	p, err := NewOpenAIProvider(OpenAIProviderConfig{
		Name:    "deepseek",
		APIKey:  "test-key",
		BaseURL: "https://api.deepseek.com/v1",
	})
	if err != nil {
		t.Fatalf("NewOpenAIProvider with custom BaseURL failed: %v", err)
	}
	if p.Name() != "deepseek" {
		t.Errorf("expected name 'deepseek', got %q", p.Name())
	}
}

func TestOpenAIProvider_DefaultBaseURL(t *testing.T) {
	p, err := NewOpenAIProvider(OpenAIProviderConfig{
		Name:   "openai",
		APIKey: "test-key",
	})
	if err != nil {
		t.Fatalf("NewOpenAIProvider with default BaseURL failed: %v", err)
	}
	if p.Name() != "openai" {
		t.Errorf("expected name 'openai', got %q", p.Name())
	}
}

func TestOpenAIProvider_Integration(t *testing.T) {
	t.Skip("integration test disabled: requires valid OPENAI_API_KEY")
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Skip("OPENAI_API_KEY not set")
	}

	p, err := NewOpenAIProvider(OpenAIProviderConfig{
		Name:   "openai",
		APIKey: apiKey,
	})
	if err != nil {
		t.Fatalf("NewOpenAIProvider failed: %v", err)
	}

	ctx := context.Background()
	resp, err := p.ChatCompletion(ctx, ChatRequest{
		Model: "gpt-4o-mini",
		Messages: []Message{
			{Role: "user", Content: "Say hello in exactly one word."},
		},
		MaxTokens: 10,
	})
	if err != nil {
		t.Fatalf("ChatCompletion failed: %v", err)
	}

	if resp.Message.Content == "" {
		t.Error("expected non-empty response content")
	}
	if resp.StopReason == "" {
		t.Error("expected non-empty stop reason")
	}
	if resp.Usage.InputTokens == 0 {
		t.Error("expected non-zero input tokens")
	}
	if resp.Usage.OutputTokens == 0 {
		t.Error("expected non-zero output tokens")
	}
}

func TestOpenAIProvider_BuildChatParams_HoistsNonLeadingSystemMessage(t *testing.T) {
	// Regression: vLLM serving Qwen rejects any system message that is not at
	// the very beginning with HTTP 400 "System message must be at the
	// beginning". sp4rk's prompt assembly can emit a non-leading system message
	// (e.g. the plan as a trailing system message after the task, or a system
	// message carried inside injected conversation history). buildChatParams
	// must hoist every system message to the front so the Chat Completions
	// payload is always system-first, mirroring the Anthropic and OpenAI
	// Responses providers.
	p, _ := NewOpenAIProvider(OpenAIProviderConfig{Name: "vllm", APIKey: "k", BaseURL: "http://localhost:8000/v1"})

	req := ChatRequest{
		Model: "Qwen/Qwen36-35B-A3B-FP8",
		Messages: []Message{
			{Role: "system", Content: "You are a coding agent."},
			{Role: "user", Content: "Do the task."},
			{Role: "system", Content: "Plan:\n1. step one\n2. step two"},
			{Role: "assistant", Content: "Working on it."},
		},
	}

	oaiReq := p.buildChatParams(req)

	jsonBytes, err := json.Marshal(oaiReq)
	if err != nil {
		t.Fatalf("marshal params: %v", err)
	}
	var parsed struct {
		Messages []map[string]any `json:"messages"`
	}
	if err := json.Unmarshal(jsonBytes, &parsed); err != nil {
		t.Fatalf("unmarshal params: %v", err)
	}

	// The very first message must be a system message.
	if got := parsed.Messages[0]["role"]; got != "system" {
		t.Fatalf("expected first message role 'system', got %v", got)
	}
	// No message after the first may be a system message.
	for i, m := range parsed.Messages[1:] {
		if m["role"] == "system" {
			t.Fatalf("message at index %d is a non-leading system message: %v", i+1, m)
		}
	}
	// Both system messages merge into the single leading one (joined with "\n").
	leadingSystem, _ := parsed.Messages[0]["content"].(string)
	if !strings.Contains(leadingSystem, "You are a coding agent.") {
		t.Errorf("leading system message lost the first system content: %q", leadingSystem)
	}
	if !strings.Contains(leadingSystem, "Plan:\n1. step one\n2. step two") {
		t.Errorf("leading system message lost the trailing system (plan) content: %q", leadingSystem)
	}
	// Non-system messages keep their relative order after the system message.
	wantRoles := []string{"system", "user", "assistant"}
	if len(parsed.Messages) != len(wantRoles) {
		t.Fatalf("expected %d messages, got %d", len(wantRoles), len(parsed.Messages))
	}
	for i, want := range wantRoles {
		if got := parsed.Messages[i]["role"]; got != want {
			t.Errorf("message[%d].role = %v, want %q", i, got, want)
		}
	}
}

func TestOpenAIProvider_BuildChatParams(t *testing.T) {
	p, _ := NewOpenAIProvider(OpenAIProviderConfig{Name: "openai", APIKey: "k"})

	temp := 0.5
	req := ChatRequest{
		Model:       "gpt-4o",
		MaxTokens:   1024,
		Temperature: &temp,
		Messages: []Message{
			{Role: "system", Content: "You are helpful."},
			{Role: "user", Content: "Hello"},
		},
		Tools: []ToolDefinition{
			{
				Name:        "search",
				Description: "Search the codebase",
				InputSchema: json.RawMessage(`{"type":"object"}`),
			},
		},
	}

	oaiReq := p.buildChatParams(req)

	if oaiReq.Model != "gpt-4o" {
		t.Errorf("expected model 'gpt-4o', got %q", oaiReq.Model)
	}
	if oaiReq.MaxCompletionTokens.Value != 1024 {
		t.Errorf("expected MaxCompletionTokens 1024, got %d", oaiReq.MaxCompletionTokens.Value)
	}
	if oaiReq.Temperature.Value != 0.5 {
		t.Errorf("expected temperature 0.5, got %f", oaiReq.Temperature.Value)
	}
	if len(oaiReq.Messages) != 2 {
		t.Errorf("expected 2 messages, got %d", len(oaiReq.Messages))
	}
	if len(oaiReq.Tools) != 1 {
		t.Errorf("expected 1 tool, got %d", len(oaiReq.Tools))
	}
	if oaiReq.Tools[0].Function.Name != "search" {
		t.Errorf("expected tool name 'search', got %q", oaiReq.Tools[0].Function.Name)
	}
}

func TestOpenAIProvider_BuildChatParams_WithReasoningContent(t *testing.T) {
	p, _ := NewOpenAIProvider(OpenAIProviderConfig{Name: "deepseek", APIKey: "k"})

	req := ChatRequest{
		Model: "deepseek-reasoner",
		Messages: []Message{
			{Role: "user", Content: "Hello"},
			{
				Role:             "assistant",
				Content:          "Let me think.",
				ReasoningContent: "I need to analyze this.",
				ToolCalls: []ToolCall{
					{ID: "call-1", Name: "search", Input: json.RawMessage(`{"q":"test"}`)},
				},
			},
			{Role: "tool", Content: "result", ToolCallID: "call-1"},
		},
	}

	oaiReq := p.buildChatParams(req)

	if len(oaiReq.Messages) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(oaiReq.Messages))
	}

	// Marshal the entire params to JSON and verify reasoning_content is in the assistant message
	jsonBytes, err := json.Marshal(oaiReq)
	if err != nil {
		t.Fatalf("failed to marshal params: %v", err)
	}

	var parsed struct {
		Messages []map[string]any `json:"messages"`
	}
	if err := json.Unmarshal(jsonBytes, &parsed); err != nil {
		t.Fatalf("failed to unmarshal params JSON: %v", err)
	}

	assistantMsg := parsed.Messages[1]
	if assistantMsg["role"] != "assistant" {
		t.Errorf("expected assistant message at index 1, got %q", assistantMsg["role"])
	}
	if assistantMsg["reasoning_content"] != "I need to analyze this." {
		t.Errorf("reasoning_content = %q, want 'I need to analyze this.'", assistantMsg["reasoning_content"])
	}
}

func TestOpenAIProvider_BuildChatParams_NoTools(t *testing.T) {
	p, _ := NewOpenAIProvider(OpenAIProviderConfig{Name: "openai", APIKey: "k"})

	req := ChatRequest{
		Model:    "gpt-4o",
		Messages: []Message{{Role: "user", Content: "Hi"}},
	}

	oaiReq := p.buildChatParams(req)

	if len(oaiReq.Tools) != 0 {
		t.Errorf("expected 0 tools, got %d", len(oaiReq.Tools))
	}
	// Temperature should be zero value when nil
	if oaiReq.Temperature.Value != 0 {
		t.Errorf("expected temperature 0 (default), got %f", oaiReq.Temperature.Value)
	}
}

func TestOpenAIProvider_ConvertRequestMessage(t *testing.T) {
	p, _ := NewOpenAIProvider(OpenAIProviderConfig{Name: "openai", APIKey: "k"})

	tests := []struct {
		name         string
		msg          Message
		wantRole     string
		wantContent  string
		wantToolCall bool
	}{
		{
			name:        "user message",
			msg:         Message{Role: "user", Content: "Hello"},
			wantRole:    "user",
			wantContent: "Hello",
		},
		{
			name:        "tool message with empty content gets fallback",
			msg:         Message{Role: "tool", Content: "", ToolCallID: "tc-1"},
			wantRole:    "tool",
			wantContent: "(no output)",
		},
		{
			name:        "tool message with content",
			msg:         Message{Role: "tool", Content: "result data", ToolCallID: "tc-2"},
			wantRole:    "tool",
			wantContent: "result data",
		},
		{
			name:        "system message",
			msg:         Message{Role: "system", Content: "Be helpful"},
			wantRole:    "system",
			wantContent: "Be helpful",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := p.convertRequestMessage(tt.msg)
			// The result is a union type, we need to check the actual type
			switch {
			case result.OfUser != nil:
				if tt.wantRole != "user" {
					t.Errorf("expected role %q, got user", tt.wantRole)
				}
				if result.OfUser.Content.OfString.Value != tt.wantContent {
					t.Errorf("content = %q, want %q", result.OfUser.Content.OfString.Value, tt.wantContent)
				}
			case result.OfSystem != nil:
				if tt.wantRole != "system" {
					t.Errorf("expected role %q, got system", tt.wantRole)
				}
				if result.OfSystem.Content.OfString.Value != tt.wantContent {
					t.Errorf("content = %q, want %q", result.OfSystem.Content.OfString.Value, tt.wantContent)
				}
			case result.OfTool != nil:
				if tt.wantRole != "tool" {
					t.Errorf("expected role %q, got tool", tt.wantRole)
				}
				if result.OfTool.Content.OfString.Value != tt.wantContent {
					t.Errorf("content = %q, want %q", result.OfTool.Content.OfString.Value, tt.wantContent)
				}
			default:
				t.Errorf("unexpected message type")
			}
		})
	}
}

func TestOpenAIProvider_ConvertRequestMessage_WithToolCalls(t *testing.T) {
	p, _ := NewOpenAIProvider(OpenAIProviderConfig{Name: "openai", APIKey: "k"})

	msg := Message{
		Role:    "assistant",
		Content: "Let me search.",
		ToolCalls: []ToolCall{
			{ID: "call-1", Name: "search", Input: json.RawMessage(`{"q":"test"}`)},
			{ID: "call-2", Name: "read", Input: json.RawMessage(`{"path":"/tmp"}`)},
		},
	}

	result := p.convertRequestMessage(msg)

	// The result is a union type, check for assistant message with tool calls
	if result.OfAssistant == nil {
		t.Fatalf("expected assistant message, got nil")
	}
	m := result.OfAssistant
	if len(m.ToolCalls) != 2 {
		t.Fatalf("expected 2 tool calls, got %d", len(m.ToolCalls))
	}
	if m.ToolCalls[0].ID != "call-1" {
		t.Errorf("tool call 0 ID = %q, want %q", m.ToolCalls[0].ID, "call-1")
	}
	if m.ToolCalls[0].Function.Name != "search" {
		t.Errorf("tool call 0 name = %q, want %q", m.ToolCalls[0].Function.Name, "search")
	}
	if m.ToolCalls[0].Function.Arguments != `{"q":"test"}` {
		t.Errorf("tool call 0 args = %q, want %q", m.ToolCalls[0].Function.Arguments, `{"q":"test"}`)
	}
}

func TestOpenAIProvider_ConvertRequestMessage_ReasoningContent(t *testing.T) {
	p, _ := NewOpenAIProvider(OpenAIProviderConfig{Name: "deepseek", APIKey: "k"})

	msg := Message{
		Role:             "assistant",
		Content:          "Let me think.",
		ReasoningContent: "This is my internal reasoning.",
	}

	result := p.convertRequestMessage(msg)

	if result.OfAssistant == nil {
		t.Fatalf("expected assistant message, got nil")
	}

	// Marshal the union type to JSON and verify reasoning_content is present
	jsonBytes, err := json.Marshal(result)
	if err != nil {
		t.Fatalf("failed to marshal union: %v", err)
	}

	var parsed map[string]any
	if err := json.Unmarshal(jsonBytes, &parsed); err != nil {
		t.Fatalf("failed to unmarshal JSON: %v", err)
	}

	if parsed["role"] != "assistant" {
		t.Errorf("role = %q, want 'assistant'", parsed["role"])
	}
	if parsed["content"] != "Let me think." {
		t.Errorf("content = %q, want 'Let me think.'", parsed["content"])
	}
	if parsed["reasoning_content"] != "This is my internal reasoning." {
		t.Errorf("reasoning_content = %q, want 'This is my internal reasoning.'", parsed["reasoning_content"])
	}
}

func TestOpenAIProvider_ConvertRequestMessage_ReasoningContentWithToolCalls(t *testing.T) {
	p, _ := NewOpenAIProvider(OpenAIProviderConfig{Name: "deepseek", APIKey: "k"})

	msg := Message{
		Role:             "assistant",
		Content:          "Let me search.",
		ReasoningContent: "I need to find the file.",
		ToolCalls: []ToolCall{
			{ID: "call-1", Name: "search", Input: json.RawMessage(`{"q":"test"}`)},
		},
	}

	result := p.convertRequestMessage(msg)

	if result.OfAssistant == nil {
		t.Fatalf("expected assistant message, got nil")
	}

	jsonBytes, err := json.Marshal(result)
	if err != nil {
		t.Fatalf("failed to marshal union: %v", err)
	}

	var parsed map[string]any
	if err := json.Unmarshal(jsonBytes, &parsed); err != nil {
		t.Fatalf("failed to unmarshal JSON: %v", err)
	}

	if parsed["reasoning_content"] != "I need to find the file." {
		t.Errorf("reasoning_content = %q, want 'I need to find the file.'", parsed["reasoning_content"])
	}

	toolCalls, ok := parsed["tool_calls"].([]any)
	if !ok || len(toolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %v", parsed["tool_calls"])
	}
}

func TestOpenAIProvider_ConvertRequestMessage_ReasoningContentEmptyContent(t *testing.T) {
	p, _ := NewOpenAIProvider(OpenAIProviderConfig{Name: "deepseek", APIKey: "k"})

	// DeepSeek may return assistant message with empty content but with reasoning_content and tool_calls
	msg := Message{
		Role:             "assistant",
		Content:          "",
		ReasoningContent: "I need to search for the file.",
		ToolCalls: []ToolCall{
			{ID: "call-1", Name: "search", Input: json.RawMessage(`{"q":"test"}`)},
		},
	}

	result := p.convertRequestMessage(msg)

	if result.OfAssistant == nil {
		t.Fatalf("expected assistant message, got nil")
	}

	jsonBytes, err := json.Marshal(result)
	if err != nil {
		t.Fatalf("failed to marshal union: %v", err)
	}

	var parsed map[string]any
	if err := json.Unmarshal(jsonBytes, &parsed); err != nil {
		t.Fatalf("failed to unmarshal JSON: %v", err)
	}

	if parsed["reasoning_content"] != "I need to search for the file." {
		t.Errorf("reasoning_content = %q, want 'I need to search for the file.'", parsed["reasoning_content"])
	}
	// content may be omitted if empty; that's OK for OpenAI SDK
}

func TestOpenAIProvider_ConvertRequestMessage_EmptyContentOmitted(t *testing.T) {
	p, _ := NewOpenAIProvider(OpenAIProviderConfig{Name: "deepseek", APIKey: "k"})

	msg := Message{
		Role:             "assistant",
		Content:          "",
		ReasoningContent: "I need to search.",
		ToolCalls: []ToolCall{
			{ID: "call-1", Name: "search", Input: json.RawMessage(`{"q":"test"}`)},
		},
	}

	result := p.convertRequestMessage(msg)
	if result.OfAssistant == nil {
		t.Fatalf("expected assistant message, got nil")
	}

	// Check the underlying assistant param content
	if result.OfAssistant.Content.OfString.Valid() {
		// If content is valid, it will be serialized
		t.Logf("Content.OfString is valid, value=%q", result.OfAssistant.Content.OfString.Value)
	}
}

func TestOpenAIProvider_ConvertRequestMessage_EmptyReasoningContentWithToolCalls(t *testing.T) {
	p, _ := NewOpenAIProvider(OpenAIProviderConfig{Name: "deepseek", APIKey: "k"})

	// DeepSeek V4 requires reasoning_content to be echoed back even when empty
	// for assistant messages that had tool_calls.
	msg := Message{
		Role:             "assistant",
		Content:          "",
		ReasoningContent: "",
		ToolCalls: []ToolCall{
			{ID: "call-1", Name: "search", Input: json.RawMessage(`{"q":"test"}`)},
		},
	}

	result := p.convertRequestMessage(msg)
	if result.OfAssistant == nil {
		t.Fatalf("expected assistant message, got nil")
	}

	jsonBytes, err := json.Marshal(result)
	if err != nil {
		t.Fatalf("failed to marshal union: %v", err)
	}

	var parsed map[string]any
	if err := json.Unmarshal(jsonBytes, &parsed); err != nil {
		t.Fatalf("failed to unmarshal JSON: %v", err)
	}

	// reasoning_content must be present even when empty for tool call messages
	if _, ok := parsed["reasoning_content"]; !ok {
		t.Errorf("reasoning_content field missing for assistant message with tool_calls")
	}
	if parsed["reasoning_content"] != "" {
		t.Errorf("reasoning_content = %q, want empty string", parsed["reasoning_content"])
	}
}

func TestOpenAIProvider_ConvertRequestMessage_EmptyReasoningContentNoToolCalls(t *testing.T) {
	p, _ := NewOpenAIProvider(OpenAIProviderConfig{Name: "deepseek", APIKey: "k"})

	// Constructed assistant messages (e.g., executor nudges) have no tool_calls
	// and empty reasoning_content. DeepSeek V4 still requires the field.
	msg := Message{
		Role:             "assistant",
		Content:          "(proceeding)",
		ReasoningContent: "",
		ToolCalls:        nil,
	}

	result := p.convertRequestMessage(msg)
	if result.OfAssistant == nil {
		t.Fatalf("expected assistant message, got nil")
	}

	jsonBytes, err := json.Marshal(result)
	if err != nil {
		t.Fatalf("failed to marshal union: %v", err)
	}

	var parsed map[string]any
	if err := json.Unmarshal(jsonBytes, &parsed); err != nil {
		t.Fatalf("failed to unmarshal JSON: %v", err)
	}

	// reasoning_content must be present even for constructed messages without tool_calls
	if _, ok := parsed["reasoning_content"]; !ok {
		t.Errorf("reasoning_content field missing for constructed assistant message")
	}
	if parsed["reasoning_content"] != "" {
		t.Errorf("reasoning_content = %q, want empty string", parsed["reasoning_content"])
	}
}

func TestOpenAIProvider_ReasoningContentExtraFieldsPreserved(t *testing.T) {
	p, _ := NewOpenAIProvider(OpenAIProviderConfig{Name: "deepseek", APIKey: "k"})

	msg := Message{
		Role:             "assistant",
		Content:          "hello",
		ReasoningContent: "my reasoning",
	}

	union := p.convertRequestMessage(msg)
	if union.OfAssistant == nil {
		t.Fatal("expected assistant")
	}

	// Re-marshal the union after it was returned from the function
	jsonBytes, err := json.Marshal(union)
	if err != nil {
		t.Fatalf("marshal failed: %v", err)
	}

	var parsed map[string]any
	if err := json.Unmarshal(jsonBytes, &parsed); err != nil {
		t.Fatalf("unmarshal failed: %v", err)
	}

	if parsed["reasoning_content"] != "my reasoning" {
		t.Errorf("reasoning_content lost after function return: got %q", parsed["reasoning_content"])
	}
}

func TestOpenAIProvider_ConvertChatResponseMessage(t *testing.T) {
	p, _ := NewOpenAIProvider(OpenAIProviderConfig{Name: "openai", APIKey: "k"})

	// Simple text message
	t.Run("simple text", func(t *testing.T) {
		oaiMsg := oai.ChatCompletionMessage{
			Role:    "assistant",
			Content: "Hello!",
		}
		result := p.convertChatResponseMessage(oaiMsg)
		if result.Role != "assistant" {
			t.Errorf("role = %q, want 'assistant'", result.Role)
		}
		if result.Content != "Hello!" {
			t.Errorf("content = %q, want 'Hello!'", result.Content)
		}
		if len(result.ToolCalls) != 0 {
			t.Errorf("expected 0 tool calls, got %d", len(result.ToolCalls))
		}
	})

	// Message with tool calls
	t.Run("with tool calls", func(t *testing.T) {
		oaiMsg := oai.ChatCompletionMessage{
			Role: "assistant",
			ToolCalls: []oai.ChatCompletionMessageToolCall{
				{
					ID:   "call-abc",
					Type: "function",
					Function: oai.ChatCompletionMessageToolCallFunction{
						Name:      "get_weather",
						Arguments: `{"city":"NYC"}`,
					},
				},
			},
		}
		result := p.convertChatResponseMessage(oaiMsg)
		if len(result.ToolCalls) != 1 {
			t.Fatalf("expected 1 tool call, got %d", len(result.ToolCalls))
		}
		tc := result.ToolCalls[0]
		if tc.ID != "call-abc" {
			t.Errorf("tool call ID = %q, want 'call-abc'", tc.ID)
		}
		if tc.Name != "get_weather" {
			t.Errorf("tool call Name = %q, want 'get_weather'", tc.Name)
		}
		if string(tc.Input) != `{"city":"NYC"}` {
			t.Errorf("tool call Input = %q, want '{\"city\":\"NYC\"}'", string(tc.Input))
		}
	})
}

func TestConvertChatResponseMessage_WithReasoningContent(t *testing.T) {
	p, _ := NewOpenAIProvider(OpenAIProviderConfig{Name: "deepseek", APIKey: "k"})

	// Simulate a DeepSeek response by unmarshalling JSON that contains reasoning_content
	var oaiMsg oai.ChatCompletionMessage
	rawJSON := `{"role":"assistant","content":"Let me search.","reasoning_content":"I need to find the file.","tool_calls":[{"id":"call-1","type":"function","function":{"name":"search","arguments":"{}"}}]}`
	if err := json.Unmarshal([]byte(rawJSON), &oaiMsg); err != nil {
		t.Fatalf("failed to unmarshal ChatCompletionMessage: %v", err)
	}

	// Verify RawJSON contains reasoning_content
	if !strings.Contains(oaiMsg.RawJSON(), "reasoning_content") {
		t.Fatalf("RawJSON does not contain reasoning_content: %s", oaiMsg.RawJSON())
	}

	result := p.convertChatResponseMessage(oaiMsg)

	if result.ReasoningContent != "I need to find the file." {
		t.Errorf("ReasoningContent = %q, want 'I need to find the file.'", result.ReasoningContent)
	}
	if result.Content != "Let me search." {
		t.Errorf("Content = %q, want 'Let me search.'", result.Content)
	}
	if len(result.ToolCalls) != 1 {
		t.Errorf("expected 1 tool call, got %d", len(result.ToolCalls))
	}
}

func TestExtractReasoningContent(t *testing.T) {
	tests := []struct {
		name string
		json string
		want string
	}{
		{
			name: "empty json",
			json: "",
			want: "",
		},
		{
			name: "no reasoning_content",
			json: `{"role":"assistant","content":"hello"}`,
			want: "",
		},
		{
			name: "with reasoning_content",
			json: `{"role":"assistant","content":"hello","reasoning_content":"Let me think..."}`,
			want: "Let me think...",
		},
		{
			name: "with tool_calls and reasoning_content",
			json: `{"role":"assistant","content":"","reasoning_content":"I need to search","tool_calls":[{"id":"call-1","type":"function","function":{"name":"search","arguments":"{}"}}]}`,
			want: "I need to search",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := extractReasoningContent(tt.json)
			if got != tt.want {
				t.Errorf("extractReasoningContent(%q) = %q, want %q", tt.json, got, tt.want)
			}
		})
	}
}

// TestModelProtocolRouting asserts the model→API-protocol mapping that drives
// OpenAIProvider.ChatCompletion dispatch via DetectProtocol. This replaces the
// legacy needsResponsesAPI gate: GPT-5.x (flagship and codex) now route to the
// Responses API, while gpt-4o / o3 / gpt-4.1 stay on Chat Completions. Claude
// and Gemini are detected as their own (non-OpenAI) protocols.
func TestModelProtocolRouting(t *testing.T) {
	tests := []struct {
		model string
		want  APIProtocol
	}{
		{"gpt-5.3-codex", ProtocolResponses},
		{"gpt-5.6", ProtocolResponses},
		{"codex-mini-latest", ProtocolResponses},
		{"gpt-4o", ProtocolChatCompletions},
		{"o3", ProtocolChatCompletions},
		{"gpt-4.1-mini", ProtocolChatCompletions},
		{"deepseek-chat", ProtocolChatCompletions},
		{"claude-sonnet-4-20250514", ProtocolAnthropic},
		{"gemini-2.5-pro", ProtocolGoogle},
	}

	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			got := DetectProtocol(tt.model)
			if got != tt.want {
				t.Errorf("DetectProtocol(%q) = %q, want %q", tt.model, got, tt.want)
			}
		})
	}
}

// TestOpenAIProvider_ProtocolRouting verifies the DetectProtocol-driven
// dispatch in ChatCompletion at the HTTP level: GPT-5.x models (flagship and
// codex) now route to the Responses API (/responses), while Chat-Completions
// models (gpt-4o, o3, gpt-4.1) stay on /chat/completions. This is the core
// behavior change from the legacy needsResponsesAPI gate, which only covered
// codex-family models — a plain gpt-5.x flagship previously fell through to
// /chat/completions.
func TestOpenAIProvider_ProtocolRouting(t *testing.T) {
	// pathRecorder serves a valid response for either endpoint and records the
	// request path.
	newPathRecorder := func(t *testing.T) (*string, *httptest.Server) {
		var got string
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			got = r.URL.Path
			w.Header().Set("Content-Type", "application/json")
			// Minimal valid response for BOTH endpoints so either path succeeds.
			body := `{"id":"x","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`
			if strings.Contains(r.URL.Path, "responses") {
				body = `{"id":"x","object":"response","created":1,"model":"m","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}],"status":"completed","usage":{"input_tokens":1,"output_tokens":1}}`
			}
			_, _ = w.Write([]byte(body))
		}))
		t.Cleanup(srv.Close)
		return &got, srv
	}

	tests := []struct {
		name    string
		model   string
		wantSub string // path substring that must be present
		notSub  string // path substring that must NOT be present (optional)
	}{
		{name: "gpt-5.6 flagship routes to responses", model: "gpt-5.6", wantSub: "responses"},
		{name: "gpt-5.3-codex routes to responses", model: "gpt-5.3-codex", wantSub: "responses"},
		{name: "gpt-4o stays on chat completions", model: "gpt-4o", wantSub: "chat/completions", notSub: "responses"},
		{name: "o3 stays on chat completions", model: "o3", wantSub: "chat/completions", notSub: "responses"},
		{name: "gpt-4.1-mini stays on chat completions", model: "gpt-4.1-mini", wantSub: "chat/completions", notSub: "responses"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, srv := newPathRecorder(t)
			p, err := NewOpenAIProvider(OpenAIProviderConfig{
				Name: "Zen", APIKey: "k", BaseURL: srv.URL,
				Logger: slog.New(slog.NewTextHandler(io.Discard, nil)),
			})
			if err != nil {
				t.Fatalf("NewOpenAIProvider failed: %v", err)
			}
			if _, err := p.ChatCompletion(context.Background(), ChatRequest{
				Model: tt.model, MaxTokens: 100,
				Messages: []Message{{Role: "user", Content: "hi"}},
			}); err != nil {
				t.Fatalf("ChatCompletion failed: %v", err)
			}
			if !strings.Contains(*got, tt.wantSub) {
				t.Errorf("model %q: expected path to contain %q, got %q", tt.model, tt.wantSub, *got)
			}
			if tt.notSub != "" && strings.Contains(*got, tt.notSub) {
				t.Errorf("model %q: expected path NOT to contain %q, got %q", tt.model, tt.notSub, *got)
			}
		})
	}
}

func TestOpenAIProvider_ResponsesClientInitialized(t *testing.T) {
	p, err := NewOpenAIProvider(OpenAIProviderConfig{
		Name:   "openai",
		APIKey: "test-key",
	})
	if err != nil {
		t.Fatalf("NewOpenAIProvider failed: %v", err)
	}
	if p.responsesClient == nil {
		t.Error("expected responsesClient to be non-nil")
	}
	if p.client == nil {
		t.Error("expected client to be non-nil")
	}
}

func TestOpenAIProvider_ResponsesClientInitialized_CustomBaseURL(t *testing.T) {
	p, err := NewOpenAIProvider(OpenAIProviderConfig{
		Name:    "custom",
		APIKey:  "test-key",
		BaseURL: "https://custom.api.com/v1",
	})
	if err != nil {
		t.Fatalf("NewOpenAIProvider failed: %v", err)
	}
	if p.responsesClient == nil {
		t.Error("expected responsesClient to be non-nil with custom base URL")
	}
}

// TestOpenAIProvider_CodexRoutesToResponsesAPI verifies that codex-family
// models are routed to the Responses API (/v1/responses) regardless of whether
// the provider is the official OpenAI endpoint or a compatible gateway.
//
// This matters because compatible gateways such as OpenCode Zen serve
// gpt-5.x-codex exclusively via /v1/responses; sending these models to
// /v1/chat/completions produces a degenerate HTTP 400 with an empty completion
// body. When a compatible provider does NOT implement /v1/responses (HTTP
// 404/405), the provider falls back to Chat Completions.
func TestOpenAIProvider_CodexRoutesToResponsesAPI(t *testing.T) {
	// pathRecorder returns 200 with a valid completion and records the request path.
	newPathRecorder := func(t *testing.T) (*string, *httptest.Server) {
		var got string
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			got = r.URL.Path
			w.Header().Set("Content-Type", "application/json")
			// Minimal valid response for BOTH endpoints so either path succeeds.
			body := `{"id":"x","object":"chat.completion","created":1,"model":"gpt-5.3-codex","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`
			if strings.Contains(r.URL.Path, "responses") {
				body = `{"id":"x","object":"response","created":1,"model":"gpt-5.3-codex","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}],"status":"completed","usage":{"input_tokens":1,"output_tokens":1}}`
			}
			_, _ = w.Write([]byte(body))
		}))
		t.Cleanup(srv.Close)
		return &got, srv
	}

	codexReq := func() ChatRequest {
		return ChatRequest{
			Model:     "gpt-5.3-codex",
			MaxTokens: 100,
			Messages:  []Message{{Role: "user", Content: "hi"}},
		}
	}

	t.Run("compatible provider (Zen) with codex uses responses API", func(t *testing.T) {
		got, srv := newPathRecorder(t)
		p, err := NewOpenAIProvider(OpenAIProviderConfig{
			Name: "Zen", APIKey: "k", BaseURL: srv.URL,
			Logger: slog.New(slog.NewTextHandler(io.Discard, nil)),
		})
		if err != nil {
			t.Fatalf("NewOpenAIProvider failed: %v", err)
		}
		if _, err := p.ChatCompletion(context.Background(), codexReq()); err != nil {
			t.Fatalf("ChatCompletion failed: %v", err)
		}
		if !strings.Contains(*got, "responses") {
			t.Fatalf("expected codex on compatible provider to hit /responses, got path %q", *got)
		}
	})

	t.Run("official OpenAI with codex uses responses API", func(t *testing.T) {
		got, srv := newPathRecorder(t)
		p, err := NewOpenAIProvider(OpenAIProviderConfig{
			Name: "chatgpt", APIKey: "k", BaseURL: srv.URL,
			Logger: slog.New(slog.NewTextHandler(io.Discard, nil)),
		})
		if err != nil {
			t.Fatalf("NewOpenAIProvider failed: %v", err)
		}
		if _, err := p.ChatCompletion(context.Background(), codexReq()); err != nil {
			t.Fatalf("ChatCompletion failed: %v", err)
		}
		if !strings.Contains(*got, "responses") {
			t.Fatalf("expected codex on official OpenAI to hit /responses, got path %q", *got)
		}
	})

	t.Run("returns clear error when responses endpoint is 404 (no fallback)", func(t *testing.T) {
		var paths []string
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			paths = append(paths, r.URL.Path)
			if strings.Contains(r.URL.Path, "responses") {
				// Responses endpoint not implemented → 404.
				w.WriteHeader(http.StatusNotFound)
				_, _ = w.Write([]byte(`{"error":{"message":"not found","type":"not_found"}}`))
				return
			}
			// Chat Completions is "available" — but must NOT be used as a
			// fallback, because gpt-5.x/codex return a degenerate 400 there.
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"id":"x","object":"chat.completion","created":1,"model":"gpt-5.3-codex","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
		}))
		t.Cleanup(srv.Close)
		p, err := NewOpenAIProvider(OpenAIProviderConfig{
			Name: "chatonly", APIKey: "k", BaseURL: srv.URL,
			Logger: slog.New(slog.NewTextHandler(io.Discard, nil)),
		})
		if err != nil {
			t.Fatalf("NewOpenAIProvider failed: %v", err)
		}
		_, err = p.ChatCompletion(context.Background(), codexReq())
		if err == nil {
			t.Fatal("expected an error when the Responses endpoint is unavailable, got nil")
		}
		// The error must explain the real cause, not be an opaque chat-completions 400.
		if !strings.Contains(err.Error(), "Responses API") {
			t.Errorf("expected error to mention the Responses API, got: %v", err)
		}
		// Must have hit /responses first (404)...
		hitResponses := false
		hitChat := false
		for _, pth := range paths {
			if strings.Contains(pth, "responses") {
				hitResponses = true
			}
			if strings.Contains(pth, "chat/completions") {
				hitChat = true
			}
		}
		if !hitResponses {
			t.Fatalf("expected to attempt /responses first, paths=%v", paths)
		}
		// ...and must NOT have silently fallen back to /chat/completions.
		if hitChat {
			t.Fatalf("should not fall back to /chat/completions when /responses is 404, paths=%v", paths)
		}
	})

	t.Run("does not fall back on responses 400 (endpoint exists, request failed)", func(t *testing.T) {
		var paths []string
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			paths = append(paths, r.URL.Path)
			if strings.Contains(r.URL.Path, "responses") {
				w.WriteHeader(http.StatusBadRequest)
				_, _ = w.Write([]byte(`{"error":{"message":"bad request","type":"invalid_request_error"}}`))
				return
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"id":"x","object":"chat.completion","created":1,"model":"gpt-5.3-codex","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
		}))
		t.Cleanup(srv.Close)
		p, err := NewOpenAIProvider(OpenAIProviderConfig{
			Name: "p", APIKey: "k", BaseURL: srv.URL,
			Logger: slog.New(slog.NewTextHandler(io.Discard, nil)),
		})
		if err != nil {
			t.Fatalf("NewOpenAIProvider failed: %v", err)
		}
		_, err = p.ChatCompletion(context.Background(), codexReq())
		if err == nil {
			t.Fatal("expected an error from the 400 responses call, got nil")
		}
		// Should NOT have fallen back to chat completions on a 400.
		for _, pth := range paths {
			if strings.Contains(pth, "chat/completions") {
				t.Fatalf("should not fall back to chat completions on responses 400, paths=%v", paths)
			}
		}
	})
}

func TestOpenAIProvider_EndToEnd_ReasoningContent(t *testing.T) {
	p, _ := NewOpenAIProvider(OpenAIProviderConfig{Name: "deepseek", APIKey: "k"})

	// Simulate a full cycle: response message -> step -> context window -> request params
	var oaiMsg oai.ChatCompletionMessage
	rawJSON := `{"role":"assistant","content":"Let me search.","reasoning_content":"I need to find the file.","tool_calls":[{"id":"call-1","type":"function","function":{"name":"search","arguments":"{}"}}]}`
	if err := json.Unmarshal([]byte(rawJSON), &oaiMsg); err != nil {
		t.Fatalf("failed to unmarshal ChatCompletionMessage: %v", err)
	}

	// 1. Convert response message
	msg := p.convertChatResponseMessage(oaiMsg)
	if msg.ReasoningContent != "I need to find the file." {
		t.Fatalf("convertChatResponseMessage lost reasoning_content: %q", msg.ReasoningContent)
	}

	// 2. Build a ChatRequest with this message as part of history
	req := ChatRequest{
		Model: "deepseek-reasoner",
		Messages: []Message{
			{Role: "user", Content: "Hello"},
			msg,
			{Role: "tool", Content: "result", ToolCallID: "call-1"},
		},
	}

	// 3. Build OpenAI params
	params := p.buildChatParams(req)

	// 4. Marshal params to JSON
	jsonBytes, err := json.Marshal(params)
	if err != nil {
		t.Fatalf("failed to marshal params: %v", err)
	}

	var parsed struct {
		Messages []map[string]any `json:"messages"`
	}
	if err := json.Unmarshal(jsonBytes, &parsed); err != nil {
		t.Fatalf("failed to unmarshal params JSON: %v", err)
	}

	// Verify reasoning_content is present in the assistant message
	assistantMsg := parsed.Messages[1]
	if assistantMsg["role"] != "assistant" {
		t.Errorf("expected assistant message at index 1, got %q", assistantMsg["role"])
	}
	if assistantMsg["reasoning_content"] != "I need to find the file." {
		t.Errorf("reasoning_content = %q, want 'I need to find the file.'", assistantMsg["reasoning_content"])
	}
}

func TestOpenAIProvider_FullChatCompletionResponse_ReasoningContent(t *testing.T) {
	p, _ := NewOpenAIProvider(OpenAIProviderConfig{Name: "deepseek", APIKey: "k"})

	// Simulate a full DeepSeek chat completion response JSON
	fullResponse := `{
		"id":"chatcmpl-test",
		"object":"chat.completion",
		"created":1234567890,
		"model":"deepseek-reasoner",
		"choices":[{
			"index":0,
			"message":{
				"role":"assistant",
				"content":"Let me search.",
				"reasoning_content":"I need to find the file.",
				"tool_calls":[{"id":"call-1","type":"function","function":{"name":"search","arguments":"{}"}}]
			},
			"finish_reason":"tool_calls"
		}],
		"usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30}
	}`

	var completion oai.ChatCompletion
	if err := json.Unmarshal([]byte(fullResponse), &completion); err != nil {
		t.Fatalf("failed to unmarshal ChatCompletion: %v", err)
	}

	if len(completion.Choices) == 0 {
		t.Fatal("expected at least one choice")
	}

	msg := completion.Choices[0].Message
	if !strings.Contains(msg.RawJSON(), "reasoning_content") {
		t.Fatalf("RawJSON does not contain reasoning_content: %s", msg.RawJSON())
	}

	result := p.convertChatResponseMessage(msg)
	if result.ReasoningContent != "I need to find the file." {
		t.Errorf("ReasoningContent = %q, want 'I need to find the file.'", result.ReasoningContent)
	}
}

func TestOpenAIProvider_WrapError(t *testing.T) {
	p, _ := NewOpenAIProvider(OpenAIProviderConfig{Name: "openai", APIKey: "k"})

	t.Run("APIError", func(t *testing.T) {
		apiErr := &oai.Error{
			StatusCode: 429,
			Message:    "rate limited",
		}
		result := p.wrapError(apiErr)
		var llmErr *Error
		if !errors.As(result, &llmErr) {
			t.Fatal("expected *Error")
		}
		if llmErr.StatusCode != 429 {
			t.Errorf("expected status 429, got %d", llmErr.StatusCode)
		}
		if !llmErr.Retryable {
			t.Error("expected retryable for 429")
		}
	})

	t.Run("plain error", func(t *testing.T) {
		result := p.wrapError(errors.New("connection failed"))
		var llmErr *Error
		if !errors.As(result, &llmErr) {
			t.Fatal("expected *Error")
		}
		if llmErr.StatusCode != 0 {
			t.Errorf("expected status 0, got %d", llmErr.StatusCode)
		}
	})
}

// newOAIErrorFromBody builds a realistic *oai.Error that mimics how the OpenAI
// SDK constructs errors from an HTTP response: the SDK reads the body, extracts
// the nested {"error":{...}} envelope via gjson and unmarshals only that into
// the error (so Message is populated), while re-populating Response.Body with a
// fresh, readable buffer of the original contents.
func newOAIErrorFromBody(t *testing.T, statusCode int, body string) *oai.Error {
	t.Helper()
	bodyBytes := []byte(body)
	u, err := url.Parse("https://api.example.com/v1/chat/completions")
	if err != nil {
		t.Fatalf("parse url: %v", err)
	}
	resp := &http.Response{
		StatusCode: statusCode,
		Status:     fmt.Sprintf("%d %s", statusCode, http.StatusText(statusCode)),
		// The SDK re-populates the body via io.NopCloser(bytes.NewBuffer(contents))
		// so it remains readable for later inspection/dumping.
		Body:   io.NopCloser(bytes.NewReader(bodyBytes)),
		Header: make(http.Header),
	}
	aerr := &oai.Error{
		Request:    &http.Request{Method: http.MethodPost, URL: u, Header: make(http.Header)},
		Response:   resp,
		StatusCode: statusCode,
	}
	// Replicate gjson.Get(body, "error").Raw extraction with encoding/json.
	if unwrapped := extractErrorEnvelope(bodyBytes); unwrapped != "" {
		if err := aerr.UnmarshalJSON([]byte(unwrapped)); err != nil {
			t.Fatalf("UnmarshalJSON: %v", err)
		}
	}
	return aerr
}

// extractErrorEnvelope returns the raw JSON of the nested "error" field, or "".
// This mirrors the SDK's gjson.Get(body, "error").Raw step.
func extractErrorEnvelope(body []byte) string {
	var m map[string]json.RawMessage
	if err := json.Unmarshal(body, &m); err != nil {
		return ""
	}
	if raw, ok := m["error"]; ok {
		return string(raw)
	}
	return ""
}

func TestOpenAIProvider_WrapError_EnrichedMessage(t *testing.T) {
	p, _ := NewOpenAIProvider(OpenAIProviderConfig{Name: "openai", APIKey: "k"})

	// (1) Standard OpenAI envelope: {"error":{"message":"bad image"}}.
	t.Run("standard error message", func(t *testing.T) {
		apiErr := newOAIErrorFromBody(t, 400, `{"error":{"message":"bad image","type":"invalid_request_error"}}`)
		result := p.wrapError(apiErr)
		var llmErr *Error
		if !errors.As(result, &llmErr) {
			t.Fatal("expected *Error")
		}
		if llmErr.StatusCode != 400 {
			t.Errorf("status = %d, want 400", llmErr.StatusCode)
		}
		if llmErr.Retryable {
			t.Error("400 must not be retryable")
		}
		if msg := result.Error(); !strings.Contains(msg, "bad image") {
			t.Errorf("expected message to contain 'bad image', got: %s", msg)
		}
	})

	// (2) Non-standard envelope (e.g. OpenCode Zen): {"detail":"image not allowed"}.
	// Message is empty because there is no nested "error" field; the upstream
	// message survives only in the raw response body.
	t.Run("non-standard detail body", func(t *testing.T) {
		apiErr := newOAIErrorFromBody(t, 400, `{"detail":"image not allowed"}`)
		result := p.wrapError(apiErr)
		var llmErr *Error
		if !errors.As(result, &llmErr) {
			t.Fatal("expected *Error")
		}
		if llmErr.StatusCode != 400 {
			t.Errorf("status = %d, want 400", llmErr.StatusCode)
		}
		if msg := result.Error(); !strings.Contains(msg, "image not allowed") {
			t.Errorf("expected message to contain 'image not allowed' (raw body), got: %s", msg)
		}
	})

	// (3) Empty body: must not panic and must surface the HTTP status.
	t.Run("empty body surfaces status", func(t *testing.T) {
		apiErr := newOAIErrorFromBody(t, 400, "")
		result := p.wrapError(apiErr)
		var llmErr *Error
		if !errors.As(result, &llmErr) {
			t.Fatal("expected *Error")
		}
		if llmErr.StatusCode != 400 {
			t.Errorf("status = %d, want 400", llmErr.StatusCode)
		}
		// Should contain the status code and not panic.
		if msg := result.Error(); !strings.Contains(msg, "400") {
			t.Errorf("expected message to contain status '400', got: %s", msg)
		}
	})
}

// TestOpenAIProvider_EnrichError_RealSDKPath verifies error-body enrichment
// through the real OpenAI SDK HTTP round-trip (not a hand-built *oai.Error).
// A gateway returning a non-standard envelope — {"detail":"..."} with no nested
// "error" field — leaves apiErr.Message empty, so the upstream message survives
// only in the repopulated Response.Body. This test pins the SDK's body-
// repopulation contract: if a future SDK version drains the body once (returning
// EOF on re-read), enrichOpenAIError would silently degrade to a bare "400 Bad
// Request" and this test would catch it.
func TestOpenAIProvider_EnrichError_RealSDKPath(t *testing.T) {
	const detail = "model 'deepseek-fake' not found"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"detail":"` + detail + `"}`))
	}))
	t.Cleanup(srv.Close)

	p, err := NewOpenAIProvider(OpenAIProviderConfig{
		Name: "zen", APIKey: "k", BaseURL: srv.URL,
		Logger: slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	if err != nil {
		t.Fatalf("NewOpenAIProvider failed: %v", err)
	}

	_, err = p.ChatCompletion(context.Background(), ChatRequest{
		Model:    "deepseek-chat",
		Messages: []Message{{Role: "user", Content: "hi"}},
	})
	if err == nil {
		t.Fatal("expected an error from the 400 response, got nil")
	}
	if !strings.Contains(err.Error(), detail) {
		t.Errorf("expected error to surface the non-standard detail body %q, got: %s", detail, err.Error())
	}
}

func TestConvertSchemaToMap(t *testing.T) {
	p, _ := NewOpenAIProvider(OpenAIProviderConfig{Name: "openai", APIKey: "k"})

	t.Run("empty schema returns default object", func(t *testing.T) {
		result := p.convertSchemaToMap(nil)
		if result["type"] != "object" {
			t.Errorf("expected type 'object', got %v", result["type"])
		}
		if result["additionalProperties"] != false {
			t.Error("expected additionalProperties: false")
		}
	})

	t.Run("invalid JSON returns default object", func(t *testing.T) {
		result := p.convertSchemaToMap([]byte("{invalid"))
		if result["type"] != "object" {
			t.Errorf("expected type 'object', got %v", result["type"])
		}
	})

	t.Run("valid JSON is parsed", func(t *testing.T) {
		result := p.convertSchemaToMap([]byte(`{"type":"string"}`))
		if result["type"] != "string" {
			t.Errorf("expected type 'string', got %v", result["type"])
		}
	})
}

func TestOpenAIProvider_WithCustomHTTPClient(t *testing.T) {
	customClient := &http.Client{}
	p, err := NewOpenAIProvider(OpenAIProviderConfig{
		Name:       "custom",
		APIKey:     "test-key",
		HTTPClient: customClient,
	})
	if err != nil {
		t.Fatalf("NewOpenAIProvider with custom HTTP client failed: %v", err)
	}
	if p.Name() != "custom" {
		t.Errorf("expected name 'custom', got %q", p.Name())
	}
}

func TestOpenAIProvider_BuildChatParams_ReasoningEffort(t *testing.T) {
	tests := []struct {
		name    string
		cfgName string
		model   string
		effort  string
	}{
		{"qwen with On", "qwen", "qwen-max", "On"},
		{"glm with On", "glm", "glm-4", "On"},
		{"deepseek with On", "deepseek", "deepseek-reasoner", "On"},
		{"openai with low", "openai", "gpt-5", "low"},
		{"no family in model uses DetectFamily", "custom", "o3-mini", "high"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p, _ := NewOpenAIProvider(OpenAIProviderConfig{Name: tt.cfgName, APIKey: "k"})
			req := ChatRequest{
				Model:           tt.model,
				Messages:        []Message{{Role: "user", Content: "Hi"}},
				ReasoningEffort: tt.effort,
			}
			params := p.buildChatParams(req)
			if params.Model != tt.model {
				t.Errorf("expected model %q, got %q", tt.model, params.Model)
			}
		})
	}
}

func TestOpenAIProvider_BuildChatParams_NoReasoningNoFamily(t *testing.T) {
	p, _ := NewOpenAIProvider(OpenAIProviderConfig{Name: "openai", APIKey: "k"})
	req := ChatRequest{
		Model:           "gpt-4o",
		Messages:        []Message{{Role: "user", Content: "Hi"}},
		ReasoningEffort: "",
	}
	params := p.buildChatParams(req)
	// Verify no panic and model is correct
	if params.Model != "gpt-4o" {
		t.Errorf("expected model 'gpt-4o', got %q", params.Model)
	}
}

func TestOpenAIProvider_BuildChatParams_GLMReasoning(t *testing.T) {
	p, _ := NewOpenAIProvider(OpenAIProviderConfig{Name: "glm", APIKey: "k"})

	// build marshals the generated params to JSON and returns the decoded
	// top-level map so individual reasoning fields can be asserted.
	build := func(t *testing.T, model, effort string) map[string]any {
		req := ChatRequest{
			Model:           model,
			Messages:        []Message{{Role: "user", Content: "Hi"}},
			ReasoningEffort: effort,
		}
		raw, err := json.Marshal(p.buildChatParams(req))
		if err != nil {
			t.Fatalf("marshal params: %v", err)
		}
		var out map[string]any
		if err := json.Unmarshal(raw, &out); err != nil {
			t.Fatalf("unmarshal params: %v", err)
		}
		return out
	}

	t.Run("glm-5.2 none disables thinking and omits reasoning_effort", func(t *testing.T) {
		out := build(t, "glm-5.2", "none")
		thinking, _ := out["thinking"].(map[string]any)
		if thinking["type"] != "disabled" {
			t.Errorf("thinking.type = %v, want disabled", thinking["type"])
		}
		if _, ok := out["reasoning_effort"]; ok {
			t.Errorf("reasoning_effort must be absent for none, got %v", out["reasoning_effort"])
		}
	})

	t.Run("glm-5.2 off (the c0wrk spelling) disables thinking, not the vendor default", func(t *testing.T) {
		// c0wrk stores "off" in its small-LLM config; without the
		// case-insensitive sentinel it fell through the switch, set no
		// fields, and GLM 5.2's thinking-enabled default silently won.
		out := build(t, "glm-5.2", "off")
		thinking, _ := out["thinking"].(map[string]any)
		if thinking["type"] != "disabled" {
			t.Errorf("thinking.type = %v, want disabled", thinking["type"])
		}
		if _, ok := out["reasoning_effort"]; ok {
			t.Errorf("reasoning_effort must be absent for off, got %v", out["reasoning_effort"])
		}
	})

	t.Run("glm-5.2 OFF (uppercase) also disables thinking", func(t *testing.T) {
		out := build(t, "glm-5.2", "OFF")
		thinking, _ := out["thinking"].(map[string]any)
		if thinking["type"] != "disabled" {
			t.Errorf("thinking.type = %v, want disabled", thinking["type"])
		}
	})

	t.Run("glm-5.2 max enables thinking and sets reasoning_effort=max", func(t *testing.T) {
		out := build(t, "glm-5.2", "max")
		thinking, _ := out["thinking"].(map[string]any)
		if thinking["type"] != "enabled" {
			t.Errorf("thinking.type = %v, want enabled", thinking["type"])
		}
		if out["reasoning_effort"] != "max" {
			t.Errorf("reasoning_effort = %v, want max", out["reasoning_effort"])
		}
	})

	t.Run("glm-5.2 high enables thinking and sets reasoning_effort=high", func(t *testing.T) {
		out := build(t, "glm-5.2", "high")
		thinking, _ := out["thinking"].(map[string]any)
		if thinking["type"] != "enabled" {
			t.Errorf("thinking.type = %v, want enabled", thinking["type"])
		}
		if out["reasoning_effort"] != "high" {
			t.Errorf("reasoning_effort = %v, want high", out["reasoning_effort"])
		}
	})

	t.Run("glm-5.2 empty (Auto) sends no reasoning fields", func(t *testing.T) {
		out := build(t, "glm-5.2", "")
		if _, ok := out["thinking"]; ok {
			t.Errorf("thinking must be absent for Auto, got %v", out["thinking"])
		}
		if _, ok := out["reasoning_effort"]; ok {
			t.Errorf("reasoning_effort must be absent for Auto, got %v", out["reasoning_effort"])
		}
	})

	t.Run("glm-5.1 keeps legacy binary thinking, normalized to wire values", func(t *testing.T) {
		// The family options spell the control "On"/"Off" (and c0wrk stores
		// "off"), but the wire values are "enabled"/"disabled" — passing the
		// option spelling through verbatim sends an invalid thinking.type.
		out := build(t, "glm-5.1", "Off")
		thinking, _ := out["thinking"].(map[string]any)
		if thinking["type"] != "disabled" {
			t.Errorf("thinking.type = %v, want disabled (normalized wire value)", thinking["type"])
		}
		if _, ok := out["reasoning_effort"]; ok {
			t.Errorf("reasoning_effort must be absent for legacy GLM, got %v", out["reasoning_effort"])
		}
		out = build(t, "glm-5.1", "On")
		thinking, _ = out["thinking"].(map[string]any)
		if thinking["type"] != "enabled" {
			t.Errorf("thinking.type = %v, want enabled (normalized wire value)", thinking["type"])
		}
	})

	t.Run("glm-5.1 non-canonical effort fails closed to disabled", func(t *testing.T) {
		out := build(t, "glm-5.1", "none")
		thinking, _ := out["thinking"].(map[string]any)
		if thinking["type"] != "disabled" {
			t.Errorf("thinking.type = %v, want disabled", thinking["type"])
		}
	})

	t.Run("glm-5.2 non-canonical effort fails closed to disabled", func(t *testing.T) {
		// A value outside the documented option set must not fall through to
		// GLM 5.2's thinking-enabled default.
		out := build(t, "glm-5.2", "medium")
		thinking, _ := out["thinking"].(map[string]any)
		if thinking["type"] != "disabled" {
			t.Errorf("thinking.type = %v, want disabled (fail closed)", thinking["type"])
		}
		if _, ok := out["reasoning_effort"]; ok {
			t.Errorf("reasoning_effort must be absent, got %v", out["reasoning_effort"])
		}
	})
}

func TestOpenAIProvider_BuildChatParams_DeepSeekReasoning(t *testing.T) {
	p, _ := NewOpenAIProvider(OpenAIProviderConfig{Name: "deepseek", APIKey: "k"})

	// build marshals the generated params to JSON and returns the decoded
	// top-level map so individual reasoning fields can be asserted.
	build := func(t *testing.T, model, effort string) map[string]any {
		req := ChatRequest{
			Model:           model,
			Messages:        []Message{{Role: "user", Content: "Hi"}},
			ReasoningEffort: effort,
		}
		raw, err := json.Marshal(p.buildChatParams(req))
		if err != nil {
			t.Fatalf("marshal params: %v", err)
		}
		var out map[string]any
		if err := json.Unmarshal(raw, &out); err != nil {
			t.Fatalf("unmarshal params: %v", err)
		}
		return out
	}

	thinkingType := func(t *testing.T, effort string) string {
		t.Helper()
		out := build(t, "deepseek-v4-pro", effort)
		thinking, _ := out["thinking"].(map[string]any)
		typ, _ := thinking["type"].(string)
		return typ
	}

	t.Run("canonical Off disables thinking verbatim", func(t *testing.T) {
		if got := thinkingType(t, "Off"); got != "Off" {
			t.Errorf("thinking.type = %v, want Off", got)
		}
	})

	t.Run("lowercase off (the c0wrk spelling) normalizes to canonical Off", func(t *testing.T) {
		if got := thinkingType(t, "off"); got != "Off" {
			t.Errorf("thinking.type = %v, want Off (normalized)", got)
		}
	})

	t.Run("uppercase OFF also normalizes to Off", func(t *testing.T) {
		if got := thinkingType(t, "OFF"); got != "Off" {
			t.Errorf("thinking.type = %v, want Off (normalized)", got)
		}
	})

	t.Run("canonical High and Max pass through verbatim", func(t *testing.T) {
		if got := thinkingType(t, "High"); got != "High" {
			t.Errorf("thinking.type = %v, want High", got)
		}
		if got := thinkingType(t, "Max"); got != "Max" {
			t.Errorf("thinking.type = %v, want Max", got)
		}
	})

	t.Run("non-canonical effort fails closed to Off", func(t *testing.T) {
		for _, effort := range []string{"low", "medium", "On", "xhigh"} {
			if got := thinkingType(t, effort); got != "Off" {
				t.Errorf("thinking.type for effort %q = %v, want Off (fail-closed)", effort, got)
			}
		}
	})
}

func TestOpenAIProvider_BuildChatParams_QwenReasoning(t *testing.T) {
	p, _ := NewOpenAIProvider(OpenAIProviderConfig{Name: "qwen", APIKey: "k"})

	// build marshals the generated params to JSON and returns the decoded
	// top-level map so individual reasoning fields can be asserted.
	build := func(t *testing.T, model, effort string) map[string]any {
		req := ChatRequest{
			Model:           model,
			Messages:        []Message{{Role: "user", Content: "Hi"}},
			ReasoningEffort: effort,
		}
		raw, err := json.Marshal(p.buildChatParams(req))
		if err != nil {
			t.Fatalf("marshal params: %v", err)
		}
		var out map[string]any
		if err := json.Unmarshal(raw, &out); err != nil {
			t.Fatalf("unmarshal params: %v", err)
		}
		return out
	}

	t.Run("legacy On enables thinking without reasoning_effort", func(t *testing.T) {
		out := build(t, "qwen3.8-27b", "On")
		if out["enable_thinking"] != true {
			t.Errorf("enable_thinking = %v, want true", out["enable_thinking"])
		}
		if _, ok := out["reasoning_effort"]; ok {
			t.Errorf("reasoning_effort must be absent for On, got %v", out["reasoning_effort"])
		}
	})

	t.Run("xhigh enables thinking at the native default", func(t *testing.T) {
		out := build(t, "qwen3.8-27b", "xhigh")
		if out["enable_thinking"] != true {
			t.Errorf("enable_thinking = %v, want true", out["enable_thinking"])
		}
		if _, ok := out["reasoning_effort"]; ok {
			t.Errorf("reasoning_effort must be absent for xhigh (native default), got %v", out["reasoning_effort"])
		}
	})

	t.Run("medium sets reasoning_effort and never disables thinking", func(t *testing.T) {
		out := build(t, "qwen3.8-27b", "medium")
		if out["reasoning_effort"] != "medium" {
			t.Errorf("reasoning_effort = %v, want medium", out["reasoning_effort"])
		}
		if out["enable_thinking"] == false {
			t.Errorf("enable_thinking must not be false when reasoning_effort is set, got %v", out["enable_thinking"])
		}
	})

	t.Run("low sets reasoning_effort and never disables thinking", func(t *testing.T) {
		out := build(t, "qwen3.8-27b", "low")
		if out["reasoning_effort"] != "low" {
			t.Errorf("reasoning_effort = %v, want low", out["reasoning_effort"])
		}
		if out["enable_thinking"] == false {
			t.Errorf("enable_thinking must not be false when reasoning_effort is set, got %v", out["enable_thinking"])
		}
	})

	t.Run("Off disables thinking", func(t *testing.T) {
		out := build(t, "qwen3.8-27b", "Off")
		if out["enable_thinking"] != false {
			t.Errorf("enable_thinking = %v, want false", out["enable_thinking"])
		}
		if _, ok := out["reasoning_effort"]; ok {
			t.Errorf("reasoning_effort must be absent for Off, got %v", out["reasoning_effort"])
		}
	})

	t.Run("lower-case off (c0wrk stored value) disables thinking", func(t *testing.T) {
		out := build(t, "qwen3.8-27b", "off")
		if out["enable_thinking"] != false {
			t.Errorf("enable_thinking = %v, want false", out["enable_thinking"])
		}
		if _, ok := out["reasoning_effort"]; ok {
			t.Errorf("reasoning_effort must be absent for off, got %v", out["reasoning_effort"])
		}
	})

	t.Run("any-case OFF disables thinking (case-insensitive sentinel)", func(t *testing.T) {
		out := build(t, "qwen3.8-27b", "OFF")
		if out["enable_thinking"] != false {
			t.Errorf("enable_thinking = %v, want false", out["enable_thinking"])
		}
		if _, ok := out["reasoning_effort"]; ok {
			t.Errorf("reasoning_effort must be absent for OFF, got %v", out["reasoning_effort"])
		}
	})

	t.Run("empty (Auto) sends no reasoning fields", func(t *testing.T) {
		out := build(t, "qwen3.8-27b", "")
		if _, ok := out["enable_thinking"]; ok {
			t.Errorf("enable_thinking must be absent for Auto, got %v", out["enable_thinking"])
		}
		if _, ok := out["reasoning_effort"]; ok {
			t.Errorf("reasoning_effort must be absent for Auto, got %v", out["reasoning_effort"])
		}
	})

	// Pre-3.8 Qwen models keep the legacy binary control: no
	// reasoning_effort is ever sent (at best silently ignored, at worst a
	// 400 from a strict gateway), and a known effort value maps to thinking
	// enabled while anything else fails closed to disabled.
	t.Run("pre-3.8 On enables thinking without reasoning_effort", func(t *testing.T) {
		out := build(t, "qwen3-235b-a22b-instruct", "On")
		if out["enable_thinking"] != true {
			t.Errorf("enable_thinking = %v, want true", out["enable_thinking"])
		}
		if _, ok := out["reasoning_effort"]; ok {
			t.Errorf("reasoning_effort must never be sent to a pre-3.8 model, got %v", out["reasoning_effort"])
		}
	})

	t.Run("pre-3.8 medium maps to binary thinking, not reasoning_effort", func(t *testing.T) {
		out := build(t, "qwen3-235b-a22b-instruct", "medium")
		if out["enable_thinking"] != true {
			t.Errorf("enable_thinking = %v, want true (legacy binary control)", out["enable_thinking"])
		}
		if _, ok := out["reasoning_effort"]; ok {
			t.Errorf("reasoning_effort must never be sent to a pre-3.8 model, got %v", out["reasoning_effort"])
		}
	})

	t.Run("pre-3.8 Off disables thinking", func(t *testing.T) {
		out := build(t, "qwq-32b", "off")
		if out["enable_thinking"] != false {
			t.Errorf("enable_thinking = %v, want false", out["enable_thinking"])
		}
		if _, ok := out["reasoning_effort"]; ok {
			t.Errorf("reasoning_effort must never be sent to a pre-3.8 model, got %v", out["reasoning_effort"])
		}
	})

	t.Run("pre-3.8 unknown value fails closed", func(t *testing.T) {
		out := build(t, "qwen2.5-72b-instruct", "bogus")
		if out["enable_thinking"] != false {
			t.Errorf("enable_thinking = %v, want false (fail-closed default)", out["enable_thinking"])
		}
		if _, ok := out["reasoning_effort"]; ok {
			t.Errorf("reasoning_effort must never be sent to a pre-3.8 model, got %v", out["reasoning_effort"])
		}
	})
}

// TestOpenAIProvider_AnthropicDelegate_RoutesToMessages verifies that a Claude
// model served by an OpenAI-compatible gateway (e.g. Zen) under the
// ProtocolAnthropic dispatch path is delegated to the co-located
// AnthropicProvider: the request POSTs to the gateway's Anthropic /messages
// endpoint (NOT /chat/completions), with an Anthropic Messages-format body, and
// the response is parsed into a ChatResponse. The delegation reuses the
// existing Anthropic Messages implementation — no code is duplicated from
// provider_anthropic.go.
func TestOpenAIProvider_AnthropicDelegate_RoutesToMessages(t *testing.T) {
	var (
		gotPath string
		gotBody []byte
	)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		body, _ := io.ReadAll(r.Body)
		gotBody = body
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"msg_1","type":"message","role":"assistant","model":"claude-3-5-sonnet","content":[{"type":"text","text":"hello world"}],"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":5}}`))
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
		Model:     "claude-3-5-sonnet",
		MaxTokens: 100,
		Messages:  []Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("ChatCompletion failed: %v", err)
	}

	// Must hit the Anthropic Messages endpoint, NOT the OpenAI chat endpoint.
	if !strings.Contains(gotPath, "/messages") {
		t.Errorf("expected Claude model to POST to /messages, got path %q", gotPath)
	}
	if strings.Contains(gotPath, "/chat/completions") {
		t.Errorf("Claude model must NOT POST to /chat/completions, got path %q", gotPath)
	}

	// The request body must be in Anthropic Messages format: a top-level
	// "messages" array, "max_tokens", and the "model" field — the hallmarks of
	// the Messages API (OpenAI Chat Completions would never target /messages).
	var body map[string]any
	if err := json.Unmarshal(gotBody, &body); err != nil {
		t.Fatalf("failed to unmarshal request body: %v\nbody: %s", err, gotBody)
	}
	if _, ok := body["messages"]; !ok {
		t.Errorf("expected Anthropic Messages body to contain \"messages\", got: %s", gotBody)
	}
	if _, ok := body["max_tokens"]; !ok {
		t.Errorf("expected Anthropic Messages body to contain \"max_tokens\", got: %s", gotBody)
	}
	if model, _ := body["model"].(string); model != "claude-3-5-sonnet" {
		t.Errorf("expected body model %q, got %q", "claude-3-5-sonnet", model)
	}

	// The Anthropic Messages response must be parsed into a ChatResponse.
	if resp == nil {
		t.Fatal("expected non-nil ChatResponse")
	}
	if resp.Message.Content != "hello world" {
		t.Errorf("expected parsed content %q, got %q", "hello world", resp.Message.Content)
	}
	if resp.Usage.OutputTokens != 5 {
		t.Errorf("expected output tokens 5, got %d", resp.Usage.OutputTokens)
	}
}

// TestOpenAIProvider_AnthropicDelegate_ErrorWrappedWithProviderName verifies
// that when the delegated Anthropic path fails, the error is wrapped with the
// provider name (inherited from the OpenAIProvider config) so it is observable
// to the caller.
func TestOpenAIProvider_AnthropicDelegate_ErrorWrappedWithProviderName(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"type":"error","error":{"type":"overloaded_error","message":"Internal Server Error"}}`))
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
		Model:    "claude-3-5-sonnet",
		Messages: []Message{{Role: "user", Content: "hi"}},
	})
	if err == nil {
		t.Fatal("expected an error from the failed Anthropic request, got nil")
	}
	// The delegate inherits the provider name from the OpenAIProvider config;
	// the error must surface it so the failure is attributable.
	if !strings.Contains(err.Error(), "Zen") {
		t.Errorf("expected error to be wrapped with provider name \"Zen\", got: %v", err)
	}
}

// TestOpenAIProvider_AllFourProtocolsRouteThroughSingleZenEntry is the
// end-to-end integration of the multi-protocol routing: a SINGLE
// openai_compatible provider entry (the "Zen" style gateway) — one
// OpenAIProvider, one httptest server, one BaseURL — must dispatch each model
// to the API protocol it speaks, verified by the capturing server recording
// the hit path:
//
//   - gpt-5.6        → POST /responses            (ProtocolResponses)
//   - claude-…sonnet → POST /messages             (ProtocolAnthropic, delegated)
//   - gemini-1.5-pro → POST /models/…:generateContent (ProtocolGoogle, delegated)
//   - grok-2         → POST /chat/completions     (ProtocolChatCompletions)
//
// This consolidates the per-protocol routing tests (which each stand up their
// own provider+server) into a single entry-point proof: one Zen provider
// instance correctly fans out to four distinct wire protocols.
func TestOpenAIProvider_AllFourProtocolsRouteThroughSingleZenEntry(t *testing.T) {
	// One capturing server: it records every hit path and serves a valid
	// response in the format matching the requested protocol (so each call
	// succeeds and produces exactly one recorded request).
	var paths []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		paths = append(paths, r.URL.Path)
		w.Header().Set("Content-Type", "application/json")
		switch {
		case strings.Contains(r.URL.Path, "responses"):
			// OpenAI Responses API shape.
			_, _ = w.Write([]byte(`{"id":"x","object":"response","created":1,"model":"m","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}],"status":"completed","usage":{"input_tokens":1,"output_tokens":1}}`))
		case strings.Contains(r.URL.Path, "generateContent"):
			// Google Generative Language shape.
			_, _ = w.Write([]byte(`{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}`))
		case strings.Contains(r.URL.Path, "/messages"):
			// Anthropic Messages shape.
			_, _ = w.Write([]byte(`{"id":"msg_1","type":"message","role":"assistant","model":"claude","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
		default:
			// OpenAI Chat Completions shape.
			_, _ = w.Write([]byte(`{"id":"x","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
		}
	}))
	t.Cleanup(srv.Close)

	// A SINGLE Zen-style openai_compatible provider entry.
	p, err := NewOpenAIProvider(OpenAIProviderConfig{
		Name:    "Zen",
		APIKey:  "k",
		BaseURL: srv.URL,
		Logger:  slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	if err != nil {
		t.Fatalf("NewOpenAIProvider failed: %v", err)
	}

	cases := []struct {
		name    string
		model   string
		wantSub string // path substring that MUST be present
		notSub  string // path substring that MUST NOT be present
	}{
		{"gpt-5.6 routes to /responses", "gpt-5.6", "responses", "chat/completions"},
		{"claude routes to /messages", "claude-3-5-sonnet", "/messages", "chat/completions"},
		{"gemini routes to generateContent", "gemini-1.5-pro", "generateContent", "chat/completions"},
		{"grok routes to /chat/completions", "grok-2", "chat/completions", ""},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			before := len(paths)
			// A uniform request for every protocol; the Anthropic delegate
			// supplies its own default max_tokens, so none is required here.
			resp, err := p.ChatCompletion(context.Background(), ChatRequest{
				Model:    tc.model,
				Messages: []Message{{Role: "user", Content: "hi"}},
			})
			if err != nil {
				t.Fatalf("ChatCompletion for %q failed: %v", tc.model, err)
			}
			if resp == nil {
				t.Fatalf("expected non-nil ChatResponse for %q", tc.model)
			}

			// Each call must produce exactly one backend request.
			if got := len(paths) - before; got != 1 {
				t.Fatalf("expected exactly 1 request for %q, got %d (paths=%v)", tc.model, got, paths)
			}
			gotPath := paths[len(paths)-1]

			if !strings.Contains(gotPath, tc.wantSub) {
				t.Errorf("model %q: expected path to contain %q, got %q", tc.model, tc.wantSub, gotPath)
			}
			if tc.notSub != "" && strings.Contains(gotPath, tc.notSub) {
				t.Errorf("model %q: expected path NOT to contain %q, got %q", tc.model, tc.notSub, gotPath)
			}
		})
	}

	// After all four calls, the captured paths must cover every protocol
	// endpoint exactly once — proving a single provider entry fans out to four
	// distinct wire protocols with no cross-contamination.
	if len(paths) != 4 {
		t.Fatalf("expected exactly 4 total backend requests (one per protocol), got %d: %v", len(paths), paths)
	}
	wantProtocols := map[string]bool{
		"/responses":       false,
		"/messages":        false,
		"generateContent":  false,
		"chat/completions": false,
	}
	for _, pth := range paths {
		for sub := range wantProtocols {
			if strings.Contains(pth, sub) {
				wantProtocols[sub] = true
			}
		}
	}
	for sub, hit := range wantProtocols {
		if !hit {
			t.Errorf("expected a request hitting %q among captured paths, got %v", sub, paths)
		}
	}
}

// TestRouter_RegistryProtocolOverrideRoutesGemmaToChatCompletions proves the
// documented protocol-override escape hatch (protocol.go docstring: "caller
// MUST override ModelMetadata.Protocol") is now functional end to end.
//
// DetectProtocol would route any "gemma"-named model to the Google
// :generateContent endpoint. A tier-1 registry override forcing a different
// APIProtocol must instead steer the request to that protocol's endpoint. The
// override flows through the full chain: the router's prepareRequest resolves
// it from the registry into req.Protocol, and the OpenAI provider honors
// req.Protocol over DetectProtocol.
//
// The table covers an override to each non-default endpoint:
//   - ProtocolChatCompletions → POST /chat/completions
//   - ProtocolResponses       → POST /responses
//   - ProtocolAnthropic       → POST /messages (delegated to AnthropicProvider)
//
// The control case (no override) confirms that, absent the override, the same
// "gemma" model still defaults to :generateContent — i.e. the override is what
// flips routing, not the model name.
func TestRouter_RegistryProtocolOverrideRoutesGemmaToChatCompletions(t *testing.T) {
	const gemmaModel = "gemma-3-27b-it"

	cases := []struct {
		name          string
		overrideProto APIProtocol // "" (zero value) means no tier-1 override
		wantSub       string      // path substring that MUST be present
		notSub        string      // path substring that MUST NOT be present
	}{
		{
			name:          "tier-1 override forces chat_completions",
			overrideProto: ProtocolChatCompletions,
			wantSub:       "chat/completions",
			notSub:        "generateContent",
		},
		{
			name:          "tier-1 override forces responses",
			overrideProto: ProtocolResponses,
			wantSub:       "responses",
			notSub:        "generateContent",
		},
		{
			name:          "tier-1 override forces anthropic",
			overrideProto: ProtocolAnthropic,
			wantSub:       "/messages",
			notSub:        "generateContent",
		},
		{
			name:    "no override defaults to generateContent",
			wantSub: "generateContent",
			notSub:  "chat/completions",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var path string
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				path = r.URL.Path
				w.Header().Set("Content-Type", "application/json")
				switch {
				case strings.Contains(r.URL.Path, "responses"):
					// OpenAI Responses API shape.
					_, _ = w.Write([]byte(`{"id":"x","object":"response","created":1,"model":"m","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}],"status":"completed","usage":{"input_tokens":1,"output_tokens":1}}`))
				case strings.Contains(r.URL.Path, "generateContent"):
					// Google Generative Language shape.
					_, _ = w.Write([]byte(`{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}`))
				case strings.Contains(r.URL.Path, "/messages"):
					// Anthropic Messages shape.
					_, _ = w.Write([]byte(`{"id":"msg_1","type":"message","role":"assistant","model":"claude","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
				default:
					// OpenAI Chat Completions shape.
					_, _ = w.Write([]byte(`{"id":"x","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
				}
			}))
			t.Cleanup(srv.Close)

			// Build a registry whose tier-1 (override) tier is optionally
			// populated for the gemma model. NewModelRegistry lowercases keys,
			// so an exact-case match at Resolve time is not required.
			overrides := map[string]ModelMetadata{}
			if tc.overrideProto != "" {
				overrides[gemmaModel] = ModelMetadata{Protocol: tc.overrideProto}
			}
			registry := NewModelRegistry(overrides)

			p, err := NewOpenAIProvider(OpenAIProviderConfig{
				Name:    "Zen",
				APIKey:  "k",
				BaseURL: srv.URL,
				Logger:  slog.New(slog.NewTextHandler(io.Discard, nil)),
			})
			if err != nil {
				t.Fatalf("NewOpenAIProvider: %v", err)
			}

			// Wire the provider and registry into a Router directly (mirrors
			// newTestRouter). tokenCounter is left nil so validateContextWindow
			// is a no-op; sampling is nil; maxRetries 0 → a single attempt.
			router := &Router{
				activeProvider:  p,
				activeBareModel: gemmaModel,
				registry:        registry,
				logger:          slog.New(slog.NewTextHandler(io.Discard, nil)),
			}

			resp, err := router.Call(context.Background(), ChatRequest{
				Model:    gemmaModel,
				Messages: []Message{{Role: "user", Content: "hi"}},
			})
			if err != nil {
				t.Fatalf("router.Call failed: %v", err)
			}
			if resp == nil {
				t.Fatalf("expected non-nil ChatResponse")
			}

			if !strings.Contains(path, tc.wantSub) {
				t.Errorf("expected path to contain %q, got %q", tc.wantSub, path)
			}
			if tc.notSub != "" && strings.Contains(path, tc.notSub) {
				t.Errorf("expected path NOT to contain %q, got %q", tc.notSub, path)
			}
		})
	}
}

// TestApplyGLMReasoning pins the GLM thinking-control mapping, in particular
// the fail-closed contract: every value outside the documented option set
// disables thinking instead of silently falling through to the model's
// always-on default, and the disable sentinels match case-insensitively so
// a host-stored "off"/"NONE" spelling cannot re-enable thinking either.
func TestApplyGLMReasoning(t *testing.T) {
	tests := []struct {
		name             string
		model            string
		effort           string
		wantThinking     string
		wantReasoningEff string
	}{
		{"glm 5.2 max enables thinking at max", "glm-5.2", "max", "enabled", "max"},
		{"glm 5.2 high enables thinking at high", "glm-5.2", "high", "enabled", "high"},
		{"glm 5.2 none disables thinking", "glm-5.2", "none", "disabled", ""},
		{"glm 5.2 NONE disables thinking case-insensitively", "glm-5.2", "NONE", "disabled", ""},
		{"glm 5.2 None disables thinking case-insensitively", "glm-5.2", "None", "disabled", ""},
		{"glm 5.2 off disables thinking", "glm-5.2", "off", "disabled", ""},
		{"glm 5.2 OFF disables thinking case-insensitively", "glm-5.2", "OFF", "disabled", ""},
		{"glm 5.2 On aliases the enabled default", "glm-5.2", "On", "enabled", "max"},
		{"glm 5.2 ON aliases the enabled default case-insensitively", "glm-5.2", "ON", "enabled", "max"},
		{"glm 5.2 non-canonical effort fails closed to disabled", "glm-5.2", "medium", "disabled", ""},
		{"glm 5.2 unknown effort fails closed to disabled", "glm-5.2", "turbo", "disabled", ""},
		{"glm 5.3 flash max enables thinking", "glm-5.3-flash", "max", "enabled", "max"},
		{"glm 4.7 On maps to the enabled wire value", "glm-4.7", "On", "enabled", ""},
		{"glm 4.7 ON maps to the enabled wire value", "glm-4.7", "ON", "enabled", ""},
		{"glm 4.7 off maps to the disabled wire value", "glm-4.7", "off", "disabled", ""},
		{"glm 4.7 Off maps to the disabled wire value", "glm-4.7", "Off", "disabled", ""},
		{"glm 4.7 none maps to the disabled wire value", "glm-4.7", "none", "disabled", ""},
		{"glm 4.7 max fails closed to the disabled wire value", "glm-4.7", "max", "disabled", ""},
		{"glm 4.7 non-canonical effort fails closed to disabled", "glm-4.7", "medium", "disabled", ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			params := &oai.ChatCompletionNewParams{}
			applyGLMReasoning(params, tt.model, tt.effort)
			thinking, _ := params.ExtraFields()["thinking"].(map[string]string)
			if thinking == nil || thinking["type"] != tt.wantThinking {
				t.Errorf("thinking = %v, want type %q", params.ExtraFields()["thinking"], tt.wantThinking)
			}
			if got, _ := params.ExtraFields()["reasoning_effort"].(string); got != tt.wantReasoningEff {
				t.Errorf("reasoning_effort = %q, want %q", got, tt.wantReasoningEff)
			}
		})
	}
}
