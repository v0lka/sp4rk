package llm

import "strings"

// APIProtocol represents the wire protocol — i.e. the request/response shape
// and canonical endpoint postfix — a model speaks. It is the protocol-level
// analogue of ModelFamily: DetectFamily adapts prompts and parameters, while
// DetectProtocol adapts which endpoint the request is sent to.
//
// The string values are descriptive identifiers (matching the ModelFamily
// convention); each constant's doc comment records its canonical endpoint
// postfix, which is what callers use to build the request URL.
type APIProtocol string

// API protocol identifiers used by DetectProtocol and ModelMetadata.Protocol.
const (
	// ProtocolChatCompletions is the default OpenAI Chat Completions protocol,
	// served at POST /chat/completions. Used by gpt-4o, gpt-4.1, o1/o3/o4,
	// grok, deepseek, minimax, glm, kimi, qwen, mistral and most compatible
	// gateways.
	ProtocolChatCompletions APIProtocol = "chat_completions"

	// ProtocolResponses is the OpenAI Responses protocol, served at
	// POST /responses. Required by GPT-5 and Codex models — both the official
	// OpenAI endpoint and compatible gateways (e.g. OpenCode Zen) expose these
	// exclusively via /responses; /chat/completions returns a degenerate HTTP
	// 400 with an empty body.
	ProtocolResponses APIProtocol = "responses"

	// ProtocolAnthropic is the Anthropic Messages protocol, served at
	// POST /messages (base URL conventionally includes the API version, e.g.
	// ".../v1/messages"). Used by all Claude models.
	ProtocolAnthropic APIProtocol = "anthropic"

	// ProtocolGoogle is the Google Generative Language protocol, served at
	// POST /models/{model}:generateContent. Used by Gemini and Gemma models.
	ProtocolGoogle APIProtocol = "google"
)

// DetectProtocol determines the API protocol from a model ID string. It is the
// protocol-level analogue of DetectFamily and mirrors its pure, string-based
// detection approach.
//
// Mapping (derived from each model family's canonical native protocol and the
// OpenCode Zen published endpoint table):
//   - contains "gpt-5" or "codex"  → ProtocolResponses
//   - contains "claude"            → ProtocolAnthropic
//   - contains "gemini" or "gemma" → ProtocolGoogle
//   - everything else              → ProtocolChatCompletions (default)
//
// Note that the protocol cannot be derived from ModelFamily alone: the
// FamilyOpenAIFlagship family spans both protocols (gpt-4o/o-series use Chat
// Completions, gpt-5 uses Responses), so independent model-ID detection is
// required.
//
// Because detection is substring-based, a custom or locally-served model whose
// name happens to contain a family token (e.g. a vLLM/Ollama model named
// "gemini-finetune" served over the OpenAI-compatible /chat/completions
// endpoint) would be misrouted to that family's native protocol. For such
// models the caller MUST override the protocol by setting
// ModelMetadata.Protocol (via the NewModelRegistry overrides map, a registered
// source, or a built-in entry): resolveProtocol honors an explicit Protocol
// value and only falls back to this substring detection when it is unset.
func DetectProtocol(modelID string) APIProtocol {
	id := strings.ToLower(modelID)
	if id == "" {
		return ProtocolChatCompletions
	}

	// GPT-5 and Codex → OpenAI Responses API (/responses).
	if strings.Contains(id, "gpt-5") || strings.Contains(id, "codex") {
		return ProtocolResponses
	}

	// Anthropic → /messages.
	if strings.Contains(id, "claude") {
		return ProtocolAnthropic
	}

	// Google (Gemini and Gemma) → /models/{model}:generateContent.
	if strings.Contains(id, "gemini") || strings.Contains(id, "gemma") {
		return ProtocolGoogle
	}

	// Everything else → /chat/completions.
	return ProtocolChatCompletions
}
