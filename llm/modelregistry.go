package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sort"
	"sync"
	"time"
)

// maxHuggingFaceConfigBytes caps the response body when fetching model metadata
// from HuggingFace. config.json is normally small, but the endpoint is
// user/config-controlled and could return an oversized body.
const maxHuggingFaceConfigBytes = 1 << 20 // 1 MiB

// ModelCapabilities describes what a model supports.
type ModelCapabilities struct {
	Attachment  bool // image/PDF support
	Reasoning   bool // reasoning/thinking mode
	Temperature bool // accepts temperature parameter
	ToolCall    bool // function calling support
}

// ModelMetadata holds the capabilities and configuration for a language model.
type ModelMetadata struct {
	ContextWindow int
	OutputLimit   int
	TokenizerType string
	Family        string
	Capabilities  ModelCapabilities
}

// ModelMetadataSource is a function that can resolve model metadata from an external source.
// Returns metadata and true if found, or zero value and false if not found.
type ModelMetadataSource func(model string) (ModelMetadata, bool)

// ModelRegistry provides a 5-tier resolution system for model metadata.
type ModelRegistry struct {
	builtIn    map[string]ModelMetadata
	overrides  map[string]ModelMetadata
	cache      map[string]ModelMetadata
	sources    []ModelMetadataSource // external metadata sources (e.g., LM Studio)
	mu         sync.RWMutex
	httpClient *http.Client
}

// NewModelRegistry creates a new registry with built-in data and optional user overrides.
//
// The overrides map is defensively copied at construction time so that callers
// (e.g. config reloads) can mutate their own map without racing the registry's
// concurrent readers.
func NewModelRegistry(overrides map[string]ModelMetadata) *ModelRegistry {
	copied := make(map[string]ModelMetadata, len(overrides))
	for k, v := range overrides {
		copied[k] = v
	}
	return &ModelRegistry{
		builtIn:    getBuiltInRegistry(),
		overrides:  copied,
		cache:      make(map[string]ModelMetadata),
		httpClient: &http.Client{Timeout: 10 * time.Second},
	}
}

// SetHTTPClient replaces the HTTP client used for metadata lookups (e.g., HuggingFace).
func (r *ModelRegistry) SetHTTPClient(client *http.Client) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if client != nil {
		r.httpClient = client
	}
}

// Resolve returns model metadata using 5-tier lookup:
// 1. User overrides (from config)
// 2. Built-in registry (hardcoded table)
// 3. HuggingFace API lookup (lazy, cached)
// 4. Registered sources (e.g., LM Studio provider)
// 5. Fallback defaults (ok=false)
//
// The second return value indicates whether the model was found in a known source.
// When ok is false, the returned metadata contains usable fallback defaults.
func (r *ModelRegistry) Resolve(ctx context.Context, model string) (ModelMetadata, bool) {
	// Priority 1: Check overrides (no lock needed for read-only map after construction)
	if meta, ok := r.overrides[model]; ok {
		meta.Family = resolveFamily(model, meta)
		return meta, true
	}

	// Priority 2: Check built-in registry (no lock needed for read-only map)
	if meta, ok := r.builtIn[model]; ok {
		meta.Family = resolveFamily(model, meta)
		return meta, true
	}

	// Priority 3: Check cache (needs lock)
	r.mu.RLock()
	if meta, ok := r.cache[model]; ok {
		r.mu.RUnlock()
		meta.Family = resolveFamily(model, meta)
		return meta, true
	}
	r.mu.RUnlock()

	// Priority 3: Fetch from HuggingFace
	meta, err := r.fetchFromHuggingFace(ctx, model)
	if err == nil {
		meta.Family = resolveFamily(model, meta)
		r.mu.Lock()
		r.cache[model] = meta
		r.mu.Unlock()
		return meta, true
	}

	// Priority 4: Try registered sources
	// Copy sources slice under read lock, then call sources without lock
	// (sources may do HTTP calls, so we don't want to hold the lock)
	r.mu.RLock()
	sources := make([]ModelMetadataSource, len(r.sources))
	copy(sources, r.sources)
	r.mu.RUnlock()

	for _, src := range sources {
		m, ok := src(model)
		if !ok {
			continue
		}
		m.Family = resolveFamily(model, m)
		r.mu.Lock()
		r.cache[model] = m
		r.mu.Unlock()
		return m, true
	}

	// Priority 5: Fallback to defaults
	meta = ModelMetadata{
		ContextWindow: 128000,
		OutputLimit:   4096,
		TokenizerType: "approximate",
	}
	meta.Family = resolveFamily(model, meta)
	return meta, false
}

// Invalidate removes an entry from the cache map (for model change mid-session).
func (r *ModelRegistry) Invalidate(model string) {
	r.mu.Lock()
	delete(r.cache, model)
	r.mu.Unlock()
}

// SetCachedMetadata stores a late-learned metadata entry in the registry cache
// (Resolution tier 3). This lets callers that discover a model's real context
// window after construction — e.g. a lazy probe of a local OpenAI-compatible
// server — populate the result so subsequent Resolve calls return it without
// re-querying the network.
//
// Tier-3 priority means the entry takes effect only when no user override
// (tier 1) or built-in entry (tier 2) already exists for the model, so a
// config.yaml override or a well-known model's spec is never silently clobbered.
// Calling this for a model that already resolved from tier 1/2 will simply be
// shadowed at Resolve time.
func (r *ModelRegistry) SetCachedMetadata(model string, meta ModelMetadata) {
	meta.Family = resolveFamily(model, meta)
	r.mu.Lock()
	r.cache[model] = meta
	r.mu.Unlock()
}

// resolveFamily determines the family for a model.
// If already set in metadata, returns it directly; otherwise delegates to DetectFamily.
func resolveFamily(modelID string, meta ModelMetadata) string {
	if meta.Family != "" {
		return meta.Family
	}
	return string(DetectFamily(modelID))
}

// RegisterSource adds a metadata source to the registry.
// Sources are called in order during resolution after HuggingFace lookup fails.
func (r *ModelRegistry) RegisterSource(src ModelMetadataSource) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.sources = append(r.sources, src)
}

// fetchFromHuggingFace queries HuggingFace API for model config.
// HTTP GET to https://huggingface.co/{model}/resolve/main/config.json
// with redirect following. Parses JSON for max_position_embeddings.
func (r *ModelRegistry) fetchFromHuggingFace(ctx context.Context, model string) (ModelMetadata, error) {
	url := fmt.Sprintf("https://huggingface.co/%s/resolve/main/config.json", model)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, http.NoBody)
	if err != nil {
		return ModelMetadata{}, fmt.Errorf("failed to create request: %w", err)
	}

	// Follow redirects automatically (http.Client default behavior).
	// Read httpClient under the lock because SetHTTPClient may replace it
	// concurrently from another goroutine.
	r.mu.RLock()
	client := r.httpClient
	r.mu.RUnlock()
	resp, err := client.Do(req)
	if err != nil {
		return ModelMetadata{}, fmt.Errorf("http request failed: %w", err)
	}
	defer func() {
		_ = resp.Body.Close()
	}()

	if resp.StatusCode != http.StatusOK {
		return ModelMetadata{}, fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, maxHuggingFaceConfigBytes))
	if err != nil {
		return ModelMetadata{}, fmt.Errorf("failed to read response body: %w", err)
	}

	// Parse config.json for max_position_embeddings
	var config struct {
		MaxPositionEmbeddings int `json:"max_position_embeddings"`
	}

	if err := json.Unmarshal(body, &config); err != nil {
		return ModelMetadata{}, fmt.Errorf("failed to parse config.json: %w", err)
	}

	if config.MaxPositionEmbeddings == 0 {
		return ModelMetadata{}, errors.New("max_position_embeddings not found in config")
	}

	return ModelMetadata{
		ContextWindow: config.MaxPositionEmbeddings,
		OutputLimit:   4096,
		TokenizerType: "approximate",
	}, nil
}

// makeBuiltInRegistry creates the hardcoded model metadata table.
//
// Values verified against official provider documentation (August 2026):
//   - OpenAI:    https://platform.openai.com/docs/models
//   - Anthropic: https://platform.claude.com/docs/en/about-claude/models/overview
//   - Google:    https://deepmind.google/models/gemini + https://ai.google.dev/gemini-api/docs/models
//   - Gemma:     https://huggingface.co/google/gemma-4-31B-it (model card)
//   - DeepSeek:  https://api-docs.deepseek.com/quick_start/pricing
//   - Qwen:      https://www.alibabacloud.com/help/en/model-studio/text-generation-model
//   - GLM:       https://docs.z.ai/guides/llm/glm-5.2 + https://docs.bigmodel.cn
//   - Kimi:      https://platform.kimi.ai/docs/models.md
//   - xAI:       https://docs.x.ai/docs/models
func makeBuiltInRegistry() map[string]ModelMetadata {
	return map[string]ModelMetadata{
		// ── OpenAI models ───────────────────────────────────────────────
		// GPT-5.6 (current frontier, Aug 2026): Sol/Terra/Luna all share
		// 1.05M context and 128K max output; reasoning models (no temperature).
		// GPT-5.x flagships: 1.05M context, 128K max output.
		// GPT-5.x mini/nano: 400K context, 128K max output.
		// GPT-5: 400K context, 128K max output.
		"gpt-5.6": {
			ContextWindow: 1050000,
			OutputLimit:   128000,
			TokenizerType: "tiktoken/o200k_base",
			Family:        "openai_flagship",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, ToolCall: true},
		},
		"gpt-5.6-sol": {
			ContextWindow: 1050000,
			OutputLimit:   128000,
			TokenizerType: "tiktoken/o200k_base",
			Family:        "openai_flagship",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, ToolCall: true},
		},
		"gpt-5.6-terra": {
			ContextWindow: 1050000,
			OutputLimit:   128000,
			TokenizerType: "tiktoken/o200k_base",
			Family:        "openai_flagship",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, ToolCall: true},
		},
		"gpt-5.6-luna": {
			ContextWindow: 1050000,
			OutputLimit:   128000,
			TokenizerType: "tiktoken/o200k_base",
			Family:        "openai_flagship",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, ToolCall: true},
		},
		"gpt-5.5": {
			ContextWindow: 1050000,
			OutputLimit:   128000,
			TokenizerType: "tiktoken/o200k_base",
			Family:        "openai_flagship",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, ToolCall: true},
		},
		"gpt-5.4": {
			ContextWindow: 1050000,
			OutputLimit:   128000,
			TokenizerType: "tiktoken/o200k_base",
			Family:        "openai_flagship",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, ToolCall: true},
		},
		"gpt-5.4-mini": {
			ContextWindow: 400000,
			OutputLimit:   128000,
			TokenizerType: "tiktoken/o200k_base",
			Family:        "openai_flagship",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, ToolCall: true},
		},
		"gpt-5.4-nano": {
			ContextWindow: 400000,
			OutputLimit:   128000,
			TokenizerType: "tiktoken/o200k_base",
			Family:        "openai_flagship",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, ToolCall: true},
		},
		"gpt-5": {
			ContextWindow: 400000,
			OutputLimit:   128000,
			TokenizerType: "tiktoken/o200k_base",
			Family:        "openai_flagship",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, ToolCall: true},
		},
		"gpt-4.1": {
			ContextWindow: 1047576,
			OutputLimit:   32768,
			TokenizerType: "tiktoken/o200k_base",
			Family:        "openai_standard",
			Capabilities:  ModelCapabilities{Attachment: true, Temperature: true, ToolCall: true},
		},
		"gpt-4.1-mini": {
			ContextWindow: 1047576,
			OutputLimit:   32768,
			TokenizerType: "tiktoken/o200k_base",
			Family:        "openai_standard",
			Capabilities:  ModelCapabilities{Attachment: true, Temperature: true, ToolCall: true},
		},
		"gpt-4.1-nano": {
			ContextWindow: 1047576,
			OutputLimit:   32768,
			TokenizerType: "tiktoken/o200k_base",
			Family:        "openai_standard",
			Capabilities:  ModelCapabilities{Attachment: true, Temperature: true, ToolCall: true},
		},
		"o4-mini": {
			ContextWindow: 200000,
			OutputLimit:   100000,
			TokenizerType: "tiktoken/o200k_base",
			Family:        "openai_flagship",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, ToolCall: true},
		},
		"o3": {
			ContextWindow: 200000,
			OutputLimit:   100000,
			TokenizerType: "tiktoken/o200k_base",
			Family:        "openai_flagship",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, ToolCall: true},
		},
		"o3-mini": {
			ContextWindow: 200000,
			OutputLimit:   100000,
			TokenizerType: "tiktoken/o200k_base",
			Family:        "openai_flagship",
			Capabilities:  ModelCapabilities{Reasoning: true, ToolCall: true},
		},
		"o1": {
			ContextWindow: 200000,
			OutputLimit:   100000,
			TokenizerType: "tiktoken/o200k_base",
			Family:        "openai_flagship",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, ToolCall: true},
		},
		"o1-mini": {
			ContextWindow: 128000,
			OutputLimit:   65536,
			TokenizerType: "tiktoken/o200k_base",
			Family:        "openai_flagship",
			Capabilities:  ModelCapabilities{Reasoning: true, ToolCall: true},
		},
		"gpt-4o": {
			ContextWindow: 128000,
			OutputLimit:   16384,
			TokenizerType: "tiktoken/o200k_base",
			Family:        "openai_flagship",
			Capabilities:  ModelCapabilities{Attachment: true, Temperature: true, ToolCall: true},
		},
		"gpt-4o-mini": {
			ContextWindow: 128000,
			OutputLimit:   16384,
			TokenizerType: "tiktoken/o200k_base",
			Family:        "openai_flagship",
			Capabilities:  ModelCapabilities{Attachment: true, Temperature: true, ToolCall: true},
		},
		// Codex models — use the OpenAI Responses API (/v1/responses).
		"codex-mini-latest": {
			ContextWindow: 200000,
			OutputLimit:   100000,
			TokenizerType: "tiktoken/o200k_base",
			Family:        "openai_codex",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, ToolCall: true},
		},
		"gpt-5.3-codex": {
			ContextWindow: 400000,
			OutputLimit:   128000,
			TokenizerType: "tiktoken/o200k_base",
			Family:        "openai_codex",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, ToolCall: true},
		},

		// ── Anthropic models ────────────────────────────────────────────
		// All Claude models accept image input (vision) and support function
		// calling. IDs use the canonical dash format from the Anthropic API.
		// Generation 5 / 4.8 / 4.7 / 4.6: 1M context, 128K max output.
		// Generation 4.5: 200K context, 64K max output.
		// Generation 4: 200K context, 32K (Opus) / 64K (Sonnet) max output.
		// Generation 3.5: 200K context, 8K max output.
		"claude-fable-5": {
			ContextWindow: 1000000,
			OutputLimit:   128000,
			TokenizerType: "anthropic-api",
			Family:        "anthropic",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"claude-opus-5": {
			ContextWindow: 1000000,
			OutputLimit:   128000,
			TokenizerType: "anthropic-api",
			Family:        "anthropic",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"claude-sonnet-5": {
			ContextWindow: 1000000,
			OutputLimit:   128000,
			TokenizerType: "anthropic-api",
			Family:        "anthropic",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"claude-opus-4-8": {
			ContextWindow: 1000000,
			OutputLimit:   128000,
			TokenizerType: "anthropic-api",
			Family:        "anthropic",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"claude-opus-4-7": {
			ContextWindow: 1000000,
			OutputLimit:   128000,
			TokenizerType: "anthropic-api",
			Family:        "anthropic",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"claude-opus-4-6": {
			ContextWindow: 1000000,
			OutputLimit:   128000,
			TokenizerType: "anthropic-api",
			Family:        "anthropic",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"claude-sonnet-4-6": {
			ContextWindow: 1000000,
			OutputLimit:   128000,
			TokenizerType: "anthropic-api",
			Family:        "anthropic",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"claude-haiku-4-5": {
			ContextWindow: 200000,
			OutputLimit:   64000,
			TokenizerType: "anthropic-api",
			Family:        "anthropic",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"claude-sonnet-4-5": {
			ContextWindow: 200000,
			OutputLimit:   64000,
			TokenizerType: "anthropic-api",
			Family:        "anthropic",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"claude-opus-4-5": {
			ContextWindow: 200000,
			OutputLimit:   64000,
			TokenizerType: "anthropic-api",
			Family:        "anthropic",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"claude-sonnet-4": {
			ContextWindow: 200000,
			OutputLimit:   64000,
			TokenizerType: "anthropic-api",
			Family:        "anthropic",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"claude-opus-4": {
			ContextWindow: 200000,
			OutputLimit:   32000,
			TokenizerType: "anthropic-api",
			Family:        "anthropic",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"claude-3.5-sonnet": {
			ContextWindow: 200000,
			OutputLimit:   8192,
			TokenizerType: "anthropic-api",
			Family:        "anthropic",
			Capabilities:  ModelCapabilities{Attachment: true, Temperature: true, ToolCall: true},
		},
		"claude-3.5-haiku": {
			ContextWindow: 200000,
			OutputLimit:   8192,
			TokenizerType: "anthropic-api",
			Family:        "anthropic",
			Capabilities:  ModelCapabilities{Attachment: true, Temperature: true, ToolCall: true},
		},

		// ── Google Gemini models ────────────────────────────────────────
		// Accessed via the openai_compatible provider (Gemini API supports
		// the OpenAI /v1/chat/completions protocol).
		// Current (Aug 2026): Gemini 3.6 Flash / 3.5 Flash-Lite / 3.1 Pro.
		// Gemini 3.x and 2.5 Pro/Flash are reasoning (thinking) models.
		// 1M context, 65K max output; 2.0 Flash: 8K output (deprecated).
		"gemini-3.6-flash": {
			ContextWindow: 1048576,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "google",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"gemini-3.5-flash-lite": {
			ContextWindow: 1048576,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "google",
			Capabilities:  ModelCapabilities{Attachment: true, Temperature: true, ToolCall: true},
		},
		"gemini-3.1-pro": {
			ContextWindow: 1048576,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "google",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"gemini-3.1-flash-lite": {
			ContextWindow: 1048576,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "google",
			Capabilities:  ModelCapabilities{Attachment: true, Temperature: true, ToolCall: true},
		},
		"gemini-3-flash": {
			ContextWindow: 1048576,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "google",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"gemini-2.5-pro": {
			ContextWindow: 1048576,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "google",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"gemini-2.5-flash": {
			ContextWindow: 1048576,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "google",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"gemini-2.5-flash-lite": {
			ContextWindow: 1048576,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "google",
			Capabilities:  ModelCapabilities{Attachment: true, Temperature: true, ToolCall: true},
		},
		"gemini-2.0-flash": {
			ContextWindow: 1048576,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "google",
			Capabilities:  ModelCapabilities{Attachment: true, Temperature: true, ToolCall: true},
		},
		// Google Gemma 4 — open-weights (HF: google/gemma-4-31B-it).
		// Multimodal (text+image), configurable thinking, native function
		// calling; the 31B dense variant has a 256K context window.
		"gemma-4-31b-it": {
			ContextWindow: 256000,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "google",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},

		// ── DeepSeek models ─────────────────────────────────────────────
		// Source: https://api-docs.deepseek.com/quick_start/pricing
		// V4: 1M context, max output 384K.
		"deepseek-v4-pro": {
			ContextWindow: 1000000,
			OutputLimit:   384000,
			TokenizerType: "approximate",
			Family:        "deepseek",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		"deepseek-v4-flash": {
			ContextWindow: 1000000,
			OutputLimit:   384000,
			TokenizerType: "approximate",
			Family:        "deepseek",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},

		// ── Kimi models (Moonshot AI) ───────────────────────────────────
		// Source: https://platform.kimi.ai/docs/models.md
		// Current (Aug 2026): kimi-k3 (1M context, native vision, deep
		// reasoning) and the kimi-k2.5/k2.6 series (256K context, multimodal,
		// thinking + non-thinking modes). All support Agent/tool tasks.
		// Note: OutputLimit is a context-window reserve (subtracted from the
		// window), so we store a practical reserve, not the absolute ceiling.
		// kimi-k3: 131K default (configurable up to 1M).
		// kimi-k2.6 / kimi-k2.5: 256K max output; reserve 65K (the 256K
		// ceiling would zero the input budget under EffectiveMax).
		"kimi-k3": {
			ContextWindow: 1000000,
			OutputLimit:   131072,
			TokenizerType: "approximate",
			Family:        "kimi",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"kimi-k2.6": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "kimi",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"kimi-k2.5": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "kimi",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		// kimi-k2 series deprecated May 25 2026; kept for backward compat.
		"kimi-k2": {
			ContextWindow: 131072,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "kimi",
			Capabilities:  ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"kimi-k2-thinking": {
			ContextWindow: 131072,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "kimi",
			Capabilities:  ModelCapabilities{Reasoning: true, ToolCall: true},
		},

		// ── Qwen models (Alibaba) ───────────────────────────────────────
		// Source: https://www.alibabacloud.com/help/en/model-studio/text-generation-model
		// Current (Aug 2026) recommended: qwen3.7-max/plus and qwen3.6-flash
		// (1M context, thinking, function calling). Text-generation Qwen
		// models are text-only — vision lives in the separate qwen-vl line.
		// Qwen3.x 1M-context models support up to 65K max output.
		"qwen3.7-max": {
			ContextWindow: 1000000,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		"qwen3.7-plus": {
			ContextWindow: 1000000,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		"qwen3.6-flash": {
			ContextWindow: 1000000,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		"qwen3-coder-plus": {
			ContextWindow: 1000000,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		// Listed under "Legacy Qwen" in current docs.
		"qwen-plus": {
			ContextWindow: 1000000,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		// qwen-max does NOT support thinking mode per Alibaba's model table.
		"qwen-max": {
			ContextWindow: 128000,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"qwq-plus": {
			ContextWindow: 128000,
			OutputLimit:   16384,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Reasoning: true, ToolCall: true},
		},

		// ── GLM models (Zhipu AI) ───────────────────────────────────────
		// Source: https://docs.z.ai + https://docs.bigmodel.cn
		// GLM 5.x: 1M/200K context, 128K max output.
		// GLM 4.7: 200K context, 128K max output.
		"glm-5.2": {
			ContextWindow: 1000000,
			OutputLimit:   128000,
			TokenizerType: "approximate",
			Family:        "glm",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		"glm-5.1": {
			ContextWindow: 200000,
			OutputLimit:   128000,
			TokenizerType: "approximate",
			Family:        "glm",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		"glm-5": {
			ContextWindow: 200000,
			OutputLimit:   128000,
			TokenizerType: "approximate",
			Family:        "glm",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		"glm-4.7": {
			ContextWindow: 200000,
			OutputLimit:   128000,
			TokenizerType: "approximate",
			Family:        "glm",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		// glm-z1-32b deprecated Nov 2025; kept for backward compat.
		"glm-z1-32b": {
			ContextWindow: 128000,
			OutputLimit:   16384,
			TokenizerType: "approximate",
			Family:        "glm",
			Capabilities:  ModelCapabilities{Reasoning: true, ToolCall: true},
		},

		// ── xAI Grok models ─────────────────────────────────────────────
		// Source: https://docs.x.ai/docs/models
		// Current (Aug 2026): grok-4.5 (500K context) and grok-4.3 / grok-4.20
		// (1M context). Current Grok models accept image input (vision) and
		// support function calling; reasoning ships as a separate variant, so
		// the base aliases are flagged non-reasoning. Older models kept for compat.
		"grok-4.5": {
			ContextWindow: 500000,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  ModelCapabilities{Attachment: true, Temperature: true, ToolCall: true},
		},
		"grok-4.3": {
			ContextWindow: 1000000,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  ModelCapabilities{Attachment: true, Temperature: true, ToolCall: true},
		},
		"grok-4.20": {
			ContextWindow: 1000000,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  ModelCapabilities{Attachment: true, Temperature: true, ToolCall: true},
		},
		"grok-4": {
			ContextWindow: 256000,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"grok-3": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"grok-3-mini": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},

		// ── Local / open-weights models (LM Studio catalog) ────────────
		// Source: https://lmstudio.ai/models (catalog) + HuggingFace model
		// cards / config.json for context windows (Aug 2026).
		// Identifiers are the lowercase org/name form LM Studio serves via its
		// OpenAI-compatible API (e.g. "openai/gpt-oss-20b"). sp4rk strips a
		// provider prefix like "lmstudio/" before lookup, so the bare org/name
		// is what's matched here. TokenizerType is "approximate" for all (no
		// per-model tiktoken); OutputLimit is a conservative context reserve,
		// not the absolute output ceiling. Temperature is accepted by all
		// local backends (llama.cpp/MLX sampling), hence set everywhere.
		//
		// ── Qwen (Alibaba) ──────────────────────────────────────────────
		// Qwen3 (2507 generation): instruct + thinking variants, 256K context.
		"qwen/qwen3-4b-2507": {
			ContextWindow: 262144,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"qwen/qwen3-4b-thinking-2507": {
			ContextWindow: 262144,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		"qwen/qwen3-30b-a3b-2507": {
			ContextWindow: 262144,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"qwen/qwen3-30b-a3b-thinking-2507": {
			ContextWindow: 262144,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		"qwen/qwen3-235b-a22b-2507": {
			ContextWindow: 262144,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"qwen/qwen3-235b-a22b-thinking-2507": {
			ContextWindow: 262144,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		// Qwen3 (1st generation, 2504): hybrid thinking, 128K context.
		"qwen/qwen3-4b": {
			ContextWindow: 131072,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		"qwen/qwen3-8b": {
			ContextWindow: 131072,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		"qwen/qwen3-14b": {
			ContextWindow: 131072,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		"qwen/qwen3-32b": {
			ContextWindow: 131072,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		"qwen/qwen3-30b-a3b": {
			ContextWindow: 131072,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		"qwen/qwen3-235b-a22b": {
			ContextWindow: 131072,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		// Qwen3.5: multimodal (text+image), thinking, 256K context.
		"qwen/qwen3.5-2b": {
			ContextWindow: 262144,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"qwen/qwen3.5-4b": {
			ContextWindow: 262144,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"qwen/qwen3.5-9b": {
			ContextWindow: 262144,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"qwen/qwen3.5-27b": {
			ContextWindow: 262144,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"qwen/qwen3.5-35b-a3b": {
			ContextWindow: 262144,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		// Qwen3.6: multimodal, thinking, 256K context.
		"qwen/qwen3.6-27b": {
			ContextWindow: 262144,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"qwen/qwen3.6-35b-a3b": {
			ContextWindow: 262144,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		// Qwen3-Coder: non-thinking coding MoE, 256K context.
		"qwen/qwen3-coder-30b": {
			ContextWindow: 262144,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"qwen/qwen3-coder-480b": {
			ContextWindow: 262144,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"qwen/qwen3-coder-next": {
			ContextWindow: 262144,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"qwen/qwen3-next-80b": {
			ContextWindow: 262144,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Temperature: true, ToolCall: true},
		},
		// Qwen3-VL: vision-language, thinking, 256K context.
		"qwen/qwen3-vl-2b": {
			ContextWindow: 262144,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"qwen/qwen3-vl-4b": {
			ContextWindow: 262144,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"qwen/qwen3-vl-8b": {
			ContextWindow: 262144,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"qwen/qwen3-vl-30b": {
			ContextWindow: 262144,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"qwen/qwen3-vl-32b": {
			ContextWindow: 262144,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		// Qwen2.5-VL: vision-language, 128K context.
		"qwen/qwen2.5-vl-3b": {
			ContextWindow: 128000,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Attachment: true, Temperature: true, ToolCall: true},
		},
		"qwen/qwen2.5-vl-7b": {
			ContextWindow: 128000,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Attachment: true, Temperature: true, ToolCall: true},
		},
		"qwen/qwen2.5-vl-32b": {
			ContextWindow: 128000,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Attachment: true, Temperature: true, ToolCall: true},
		},
		"qwen/qwen2.5-vl-72b": {
			ContextWindow: 128000,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Attachment: true, Temperature: true, ToolCall: true},
		},

		// ── DeepSeek (open reasoning distills) ─────────────────────────
		// R1 / distills: reasoning-only, not trained for tool calling.
		"deepseek/deepseek-r1-0528-qwen3-8b": {
			ContextWindow: 131072,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "deepseek",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true},
		},
		"deepseek/deepseek-r1-distill-qwen-7b": {
			ContextWindow: 131072,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "deepseek",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true},
		},
		"deepseek/deepseek-r1-distill-llama-8b": {
			ContextWindow: 131072,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "deepseek",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true},
		},
		"deepseek/deepseek-r1-distill-qwen-14b": {
			ContextWindow: 131072,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "deepseek",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true},
		},
		"deepseek/deepseek-r1-distill-qwen-32b": {
			ContextWindow: 131072,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "deepseek",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true},
		},
		"deepseek/deepseek-r1-distill-llama-70b": {
			ContextWindow: 131072,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "deepseek",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true},
		},

		// ── GLM (Z.ai open coding) ─────────────────────────────────────
		"zai-org/glm-4.7-flash": {
			ContextWindow: 202752,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "glm",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		"zai-org/glm-4.6v-flash": {
			ContextWindow: 131072,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "glm",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},

		// ── Google Gemma (open-weights) ────────────────────────────────
		// Gemma 4: multimodal (text+image; +audio on E2B/E4B/12B), thinking,
		// native function calling. E2B/E4B = 128K; 12B/26B-A4B/31B = 256K.
		"google/gemma-4-e2b": {
			ContextWindow: 131072,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "google",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"google/gemma-4-e4b": {
			ContextWindow: 131072,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "google",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"google/gemma-4-12b": {
			ContextWindow: 262144,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "google",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"google/gemma-4-26b-a4b": {
			ContextWindow: 262144,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "google",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"google/gemma-4-31b": {
			ContextWindow: 262144,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "google",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		// Gemma 3: image+text (≥4B), 128K; 270M/1B text-only, 32K.
		"google/gemma-3-270m": {
			ContextWindow: 32768,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "google",
			Capabilities:  ModelCapabilities{Temperature: true},
		},
		"google/gemma-3-1b": {
			ContextWindow: 32768,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "google",
			Capabilities:  ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"google/gemma-3-4b": {
			ContextWindow: 131072,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "google",
			Capabilities:  ModelCapabilities{Attachment: true, Temperature: true, ToolCall: true},
		},
		"google/gemma-3-12b": {
			ContextWindow: 131072,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "google",
			Capabilities:  ModelCapabilities{Attachment: true, Temperature: true, ToolCall: true},
		},
		"google/gemma-3-27b": {
			ContextWindow: 131072,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "google",
			Capabilities:  ModelCapabilities{Attachment: true, Temperature: true, ToolCall: true},
		},
		// Gemma 3n: on-device multimodal, 32K.
		"google/gemma-3n-e2b": {
			ContextWindow: 32768,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "google",
			Capabilities:  ModelCapabilities{Attachment: true, Temperature: true},
		},
		"google/gemma-3n-e4b": {
			ContextWindow: 32768,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "google",
			Capabilities:  ModelCapabilities{Attachment: true, Temperature: true},
		},
		// FunctionGemma: tiny function-calling foundation, 32K.
		"google/functiongemma-270m": {
			ContextWindow: 32768,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "google",
			Capabilities:  ModelCapabilities{Temperature: true, ToolCall: true},
		},

		// ── Mistral (open-weights) ─────────────────────────────────────
		"mistralai/mistral-7b-instruct-v0.3": {
			ContextWindow: 32768,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "mistral",
			Capabilities:  ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"mistralai/mistral-nemo-instruct-2407": {
			ContextWindow: 131072,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "mistral",
			Capabilities:  ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"mistralai/mistral-small-3.2": {
			ContextWindow: 131072,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "mistral",
			Capabilities:  ModelCapabilities{Attachment: true, Temperature: true, ToolCall: true},
		},
		// Magistral: reasoning, 128K. v1.2 (+vision) and v1.1 (text).
		"mistralai/magistral-small-2509": {
			ContextWindow: 131072,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "mistral",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"mistralai/magistral-small": {
			ContextWindow: 131072,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "mistral",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		// Devstral: agentic coding, 128K. Devstral 2 adds vision (123B = 256K).
		"mistralai/devstral-small-2507": {
			ContextWindow: 131072,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "mistral",
			Capabilities:  ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"mistralai/devstral-small-2505": {
			ContextWindow: 131072,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "mistral",
			Capabilities:  ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"mistralai/devstral-small-2-2512": {
			ContextWindow: 131072,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "mistral",
			Capabilities:  ModelCapabilities{Attachment: true, Temperature: true, ToolCall: true},
		},
		"mistralai/devstral-2-2512": {
			ContextWindow: 262144,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "mistral",
			Capabilities:  ModelCapabilities{Attachment: true, Temperature: true, ToolCall: true},
		},
		// Codestral: coding, 32K.
		"mistralai/codestral-22b-v0.1": {
			ContextWindow: 32768,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "mistral",
			Capabilities:  ModelCapabilities{Temperature: true, ToolCall: true},
		},
		// Ministral 3: multimodal, 256K (base instruct variants).
		"mistralai/ministral-3-3b": {
			ContextWindow: 262144,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "mistral",
			Capabilities:  ModelCapabilities{Attachment: true, Temperature: true, ToolCall: true},
		},
		"mistralai/ministral-3-8b": {
			ContextWindow: 262144,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "mistral",
			Capabilities:  ModelCapabilities{Attachment: true, Temperature: true, ToolCall: true},
		},
		"mistralai/ministral-3-14b": {
			ContextWindow: 262144,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "mistral",
			Capabilities:  ModelCapabilities{Attachment: true, Temperature: true, ToolCall: true},
		},

		// ── OpenAI gpt-oss (open-weights) ──────────────────────────────
		// gpt-oss: configurable reasoning effort + tool use, 128K.
		"openai/gpt-oss-20b": {
			ContextWindow: 131072,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		"openai/gpt-oss-120b": {
			ContextWindow: 131072,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		// gpt-oss-safeguard: content classifiers (not chat/agent models).
		"openai/gpt-oss-safeguard-20b": {
			ContextWindow: 131072,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  ModelCapabilities{Temperature: true},
		},
		"openai/gpt-oss-safeguard-120b": {
			ContextWindow: 131072,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  ModelCapabilities{Temperature: true},
		},

		// ── ByteDance / Baidu ──────────────────────────────────────────
		// seed-oss: reasoning with configurable thinking budget, 512K.
		"bytedance/seed-oss-36b": {
			ContextWindow: 524288,
			OutputLimit:   16384,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		// ERNIE 4.5: 21B MoE, 128K.
		"baidu/ernie-4.5-21b-a3b": {
			ContextWindow: 131072,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  ModelCapabilities{Temperature: true, ToolCall: true},
		},

		// ── NVIDIA Nemotron ────────────────────────────────────────────
		// Nemotron 3 / Super: up to 1M context. Omni is multimodal.
		"nvidia/nemotron-3-nano": {
			ContextWindow: 1048576,
			OutputLimit:   16384,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		"nvidia/nemotron-3-nano-omni": {
			ContextWindow: 131072,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"nvidia/nemotron-3-super": {
			ContextWindow: 1048576,
			OutputLimit:   16384,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},

		// ── IBM Granite ────────────────────────────────────────────────
		// Granite 4.x: tool calling + JSON output, 128K.
		"ibm/granite-4-h-micro": {
			ContextWindow: 131072,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"ibm/granite-4-micro": {
			ContextWindow: 131072,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"ibm/granite-4-h-tiny": {
			ContextWindow: 131072,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"ibm/granite-4-h-small": {
			ContextWindow: 131072,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"ibm/granite-4.1-3b": {
			ContextWindow: 131072,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"ibm/granite-4.1-8b": {
			ContextWindow: 131072,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"ibm/granite-4.1-30b": {
			ContextWindow: 131072,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  ModelCapabilities{Temperature: true, ToolCall: true},
		},

		// ── Liquid AI LFM2 (on-device hybrid) ──────────────────────────
		// Small variants: text, 32K, no tool use. 24B-A2B: native function calling.
		"liquid/lfm2-350m": {
			ContextWindow: 32768,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  ModelCapabilities{Temperature: true},
		},
		"liquid/lfm2-700m": {
			ContextWindow: 32768,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  ModelCapabilities{Temperature: true},
		},
		"liquid/lfm2-1.2b": {
			ContextWindow: 32768,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  ModelCapabilities{Temperature: true},
		},
		"liquid/lfm2-24b-a2b": {
			ContextWindow: 32768,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  ModelCapabilities{Temperature: true, ToolCall: true},
		},

		// ── Allen AI (Olmo / olmOCR) ───────────────────────────────────
		// Olmo 3: 64K. olmOCR 2: vision-language OCR, 128K, no tool use.
		"allenai/olmo-3-7b": {
			ContextWindow: 65536,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"allenai/olmo-3-7b-think": {
			ContextWindow: 65536,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		"allenai/olmo-3-32b-think": {
			ContextWindow: 65536,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		"allenai/olmocr-2-7b": {
			ContextWindow: 128000,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  ModelCapabilities{Attachment: true, Temperature: true},
		},

		// ── Essential AI / MiniMax ─────────────────────────────────────
		// Rnj-1: dense reasoning, 32K. MiniMax M2: 230B MoE, ~196K.
		"essentialai/rnj-1": {
			ContextWindow: 32768,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		"minimax/minimax-m2": {
			ContextWindow: 196608,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},

		// ── Microsoft Phi ──────────────────────────────────────────────
		// phi-4: 16K, no tool use. phi-4-mini: 128K. Reasoning variants: 32K/128K.
		"microsoft/phi-4": {
			ContextWindow: 16384,
			OutputLimit:   4096,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  ModelCapabilities{Temperature: true},
		},
		"microsoft/phi-4-mini": {
			ContextWindow: 131072,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  ModelCapabilities{Temperature: true},
		},
		"microsoft/phi-4-mini-reasoning": {
			ContextWindow: 131072,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		"microsoft/phi-4-reasoning": {
			ContextWindow: 32768,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		"microsoft/phi-4-reasoning-plus": {
			ContextWindow: 32768,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
	}
}

// builtInRegistryOnce guards lazy initialization of the cached built-in registry.
var builtInRegistryOnce sync.Once
var builtInRegistryCache map[string]ModelMetadata

// getBuiltInRegistry returns the cached built-in model registry, initializing it
// on first call. The registry is immutable data so sharing is safe.
func getBuiltInRegistry() map[string]ModelMetadata {
	builtInRegistryOnce.Do(func() {
		builtInRegistryCache = makeBuiltInRegistry()
	})
	return builtInRegistryCache
}

// BuiltInModelNames returns model names from the built-in registry filtered by tokenizer type.
// If tokenizerType is empty, returns all model names.
func BuiltInModelNames(tokenizerType string) []string {
	registry := getBuiltInRegistry()
	names := []string{}
	for name, meta := range registry {
		if tokenizerType == "" || meta.TokenizerType == tokenizerType {
			names = append(names, name)
		}
	}
	sort.Strings(names)
	return names
}
