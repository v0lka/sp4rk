package llm

import (
	"context"
	"testing"
)

func TestDetectProtocol(t *testing.T) {
	tests := []struct {
		modelID  string
		expected APIProtocol
	}{
		// Empty model ID → default protocol
		{"", ProtocolChatCompletions},

		// ProtocolResponses — GPT-5 family and Codex
		{"gpt-5", ProtocolResponses},
		{"gpt-5.6", ProtocolResponses},
		{"gpt-5.6-sol", ProtocolResponses},
		{"gpt-5.4-mini", ProtocolResponses},
		{"codex-mini-latest", ProtocolResponses},
		{"gpt-5.3-codex", ProtocolResponses},

		// ProtocolAnthropic — Claude
		{"claude-opus-5", ProtocolAnthropic},
		{"claude-sonnet-4-5", ProtocolAnthropic},
		{"claude-3.5-sonnet", ProtocolAnthropic},
		{"claude-custom", ProtocolAnthropic},

		// ProtocolGoogle — Gemini and Gemma
		{"gemini-3.6-flash", ProtocolGoogle},
		{"gemini-2.5-pro", ProtocolGoogle},
		{"gemini-2.0-flash", ProtocolGoogle},
		{"gemma-4-31b-it", ProtocolGoogle},
		{"gemma-2-27b", ProtocolGoogle},

		// ProtocolChatCompletions — everything else.
		// OpenAI flagship/standard that are NOT gpt-5 (the critical case:
		// FamilyOpenAIFlagship spans both protocols).
		{"gpt-4o", ProtocolChatCompletions},
		{"gpt-4o-mini", ProtocolChatCompletions},
		{"gpt-4-turbo", ProtocolChatCompletions},
		{"gpt-4.1", ProtocolChatCompletions},
		{"gpt-4.1-mini", ProtocolChatCompletions},
		{"o1", ProtocolChatCompletions},
		{"o1-mini", ProtocolChatCompletions},
		{"o3", ProtocolChatCompletions},
		{"o3-mini", ProtocolChatCompletions},
		{"o4-mini", ProtocolChatCompletions},

		// Other families default to Chat Completions
		{"grok-4.20", ProtocolChatCompletions},
		{"deepseek-v4-pro", ProtocolChatCompletions},
		{"mistral-large-latest", ProtocolChatCompletions},
		{"qwen-plus", ProtocolChatCompletions},
		{"qwq-32b", ProtocolChatCompletions},
		{"glm-5.2", ProtocolChatCompletions},
		{"chatglm-4", ProtocolChatCompletions},
		{"kimi-k3", ProtocolChatCompletions},

		// Default (no specific pattern)
		{"llama-3.1-70b", ProtocolChatCompletions},
		{"phi-3-mini", ProtocolChatCompletions},
		{"unknown-model", ProtocolChatCompletions},
	}

	for _, tt := range tests {
		t.Run(tt.modelID, func(t *testing.T) {
			got := DetectProtocol(tt.modelID)
			if got != tt.expected {
				t.Errorf("DetectProtocol(%q) = %q, want %q", tt.modelID, got, tt.expected)
			}
		})
	}
}

func TestDetectProtocol_CaseInsensitive(t *testing.T) {
	// DetectProtocol lowercases the input, so mixed case should still work
	tests := []struct {
		modelID  string
		expected APIProtocol
	}{
		{"GPT-5.6", ProtocolResponses},
		{"CODEX-Mini-Latest", ProtocolResponses},
		{"Claude-Opus-5", ProtocolAnthropic},
		{"GEMINI-2.5-PRO", ProtocolGoogle},
		{"Gemma-4-31B-IT", ProtocolGoogle},
		{"GPT-4O", ProtocolChatCompletions},
		{"O3-MINI", ProtocolChatCompletions},
		{"DeepSeek-V4-Pro", ProtocolChatCompletions},
	}

	for _, tt := range tests {
		t.Run(tt.modelID, func(t *testing.T) {
			got := DetectProtocol(tt.modelID)
			if got != tt.expected {
				t.Errorf("DetectProtocol(%q) = %q, want %q", tt.modelID, got, tt.expected)
			}
		})
	}
}

func TestAPIProtocolConstants(t *testing.T) {
	// Verify the string values of protocol constants
	protocols := map[APIProtocol]string{
		ProtocolChatCompletions: "chat_completions",
		ProtocolResponses:       "responses",
		ProtocolAnthropic:       "anthropic",
		ProtocolGoogle:          "google",
	}

	for protocol, expected := range protocols {
		if string(protocol) != expected {
			t.Errorf("Protocol constant %q != expected %q", string(protocol), expected)
		}
	}
}

// TestResolveProtocol_OverrideEscapesSubstringDetection verifies that an
// explicit ModelMetadata.Protocol override wins over DetectProtocol's
// substring-based detection. This is the documented escape-hatch for custom or
// locally-served models whose name happens to contain a family token but speak a
// different wire protocol (e.g. a vLLM model named "gemini-local" served over
// the OpenAI-compatible /chat/completions endpoint).
func TestResolveProtocol_OverrideEscapesSubstringDetection(t *testing.T) {
	const model = "gemini-local"

	// Sanity: substring detection alone would misroute this model to Google.
	if got := DetectProtocol(model); got != ProtocolGoogle {
		t.Fatalf("DetectProtocol(%q) = %q, want %q (precondition)", model, got, ProtocolGoogle)
	}

	// The caller overrides the protocol via the NewModelRegistry overrides map.
	reg := NewModelRegistry(map[string]ModelMetadata{
		model: {Protocol: ProtocolChatCompletions},
	})

	meta, ok := reg.Resolve(context.Background(), model)
	if !ok {
		t.Fatalf("Resolve(%q) ok=false, want true", model)
	}
	if meta.Protocol != ProtocolChatCompletions {
		t.Errorf("Resolve(%q).Protocol = %q, want %q (override must win over substring detection)",
			model, meta.Protocol, ProtocolChatCompletions)
	}
}
