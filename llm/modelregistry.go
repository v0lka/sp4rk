package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"
)

// maxHuggingFaceConfigBytes caps the response body when fetching model metadata
// from HuggingFace. config.json is normally small, but the endpoint is
// user/config-controlled and could return an oversized body.
const maxHuggingFaceConfigBytes = 1 << 20 // 1 MiB

// ModelCapabilities describes what a model supports.
//
// The struct carries json+yaml struct tags so a single canonical type can be
// serialized consistently in both the JSON API contract (GetModelConfig/
// SetModelConfig) and config.yaml (ModelOverride) using snake_case keys,
// without conversion boilerplate at every layer boundary.
type ModelCapabilities struct {
	Attachment  bool `json:"attachment" yaml:"attachment"`   // image/PDF support
	Reasoning   bool `json:"reasoning" yaml:"reasoning"`     // reasoning/thinking mode
	Temperature bool `json:"temperature" yaml:"temperature"` // accepts temperature parameter
	ToolCall    bool `json:"tool_call" yaml:"tool_call"`     // function calling support
}

// ModelMetadata holds the capabilities and configuration for a language model.
type ModelMetadata struct {
	ContextWindow int
	OutputLimit   int
	TokenizerType string
	// Family is the model family (e.g. "openai_flagship", "anthropic"). When
	// set, it is AUTHORITATIVE: resolveFamily returns it as-is and only falls
	// back to DetectFamily when it is empty. Thus the Family value in a built-in
	// or override record is not merely documentary — it wins over substring
	// detection. Omit it only when DetectFamily should derive the family.
	Family string
	// Protocol is the wire protocol / canonical endpoint postfix the model
	// speaks. Populated lazily by resolveProtocol at Resolve time from
	// DetectProtocol (or an explicit override), so callers can route the
	// request to the right endpoint. See DetectProtocol for the mapping.
	Protocol     APIProtocol
	Capabilities ModelCapabilities
}

// ModelMetadataSource is a function that can resolve model metadata from an external source.
// Returns metadata and true if found, or zero value and false if not found.
type ModelMetadataSource func(model string) (ModelMetadata, bool)

// ModelRegistry provides a 5-tier resolution system for model metadata.
type ModelRegistry struct {
	builtIn        map[string]ModelMetadata
	builtInIndex   map[string]ModelMetadata // normalized-ID -> metadata index for fuzzy built-in lookup
	overrides      map[string]ModelMetadata
	overridesIndex map[string]ModelMetadata // normalized-ID -> metadata index for fuzzy override lookup
	cache          map[string]ModelMetadata
	sources        []ModelMetadataSource // external metadata sources (e.g., LM Studio)
	mu             sync.RWMutex
	httpClient     *http.Client
}

// NewModelRegistry creates a new registry with built-in data and optional user overrides.
//
// The overrides map is defensively copied at construction time so that callers
// (e.g. config reloads) can mutate their own map without racing the registry's
// concurrent readers.
func NewModelRegistry(overrides map[string]ModelMetadata) *ModelRegistry {
	copied := make(map[string]ModelMetadata, len(overrides))
	for k, v := range overrides {
		// Normalize override keys to lowercase so they are matched by the
		// case-insensitive Resolve lookup.
		copied[strings.ToLower(k)] = v
	}
	return &ModelRegistry{
		builtIn:        getBuiltInRegistry(),
		builtInIndex:   getBuiltInIndex(),
		overrides:      copied,
		overridesIndex: buildNormalizedIndex(copied),
		cache:          make(map[string]ModelMetadata),
		httpClient:     &http.Client{Timeout: 10 * time.Second},
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

// Resolve returns model metadata using a tiered lookup:
//   - 1. User overrides (from config)
//   - 2. Built-in registry (hardcoded table)
//   - 2b. Fuzzy match across overrides + built-ins, with the vendor prefix
//     stripped and separator punctuation (".", "-", "_") removed — so
//     "qwen3.6" matches "qwen36", "gpt-4" matches "gpt4", and a bare
//     "glm-5.2-fp8" matches the prefixed "zai-org/glm-5.2-fp8". Consulted only
//     on an exact-lookup miss; the result is cached under the query key.
//   - 3. HuggingFace API lookup (lazy, cached)
//   - 4. Registered sources (e.g., LM Studio provider)
//   - 5. Fallback defaults (ok=false)
//
// The second return value indicates whether the model was found in a known source.
// When ok is false, the returned metadata contains usable fallback defaults.
func (r *ModelRegistry) Resolve(ctx context.Context, model string) (ModelMetadata, bool) {
	// Case-insensitive lookup: model IDs arriving from OpenAI-compatible hosts
	// (e.g. vLLM) frequently differ only in casing from their canonical
	// registry keys. Normalize once and use this key for every map lookup;
	// family/protocol detection lowercases independently and HuggingFace
	// fetching preserves the original casing for the URL path.
	key := strings.ToLower(model)

	// Priority 1: Check overrides (no lock needed for read-only map after construction)
	if meta, ok := r.overrides[key]; ok {
		// A partial override (one that pins only some fields, e.g. a
		// protocol-only auto-remap) inherits its unset scalar fields from the
		// lower non-network tiers so it does not collapse the context window
		// or output limit to zero. See enrichPartialOverride.
		meta = r.enrichPartialOverride(model, key, meta)
		meta.Family = resolveFamily(model, meta)
		meta.Protocol = resolveProtocol(model, meta)
		return meta, true
	}

	// Priority 2: Check built-in registry (no lock needed for read-only map)
	if meta, ok := r.builtIn[key]; ok {
		meta.Family = resolveFamily(model, meta)
		meta.Protocol = resolveProtocol(model, meta)
		return meta, true
	}

	// Priority 2b: Fuzzy match — separator-insensitive lookup across overrides
	// and built-ins. Bridges cosmetic naming drift (a host serving
	// "Qwen/Qwen36-35B-A3B-FP8" for the registry key "qwen/qwen3.6-35b-a3b-fp8")
	// without the false-positive risk of edit-distance matching. Cache the hit
	// under the query key so repeat resolves take the fast path.
	if meta, ok := r.fuzzyLookup(model); ok {
		meta.Family = resolveFamily(model, meta)
		meta.Protocol = resolveProtocol(model, meta)
		r.mu.Lock()
		r.cache[key] = meta
		r.mu.Unlock()
		return meta, true
	}

	// Priority 3: Check cache (needs lock)
	r.mu.RLock()
	if meta, ok := r.cache[key]; ok {
		r.mu.RUnlock()
		meta.Family = resolveFamily(model, meta)
		meta.Protocol = resolveProtocol(model, meta)
		return meta, true
	}
	r.mu.RUnlock()

	// Priority 3: Fetch from HuggingFace
	meta, err := r.fetchFromHuggingFace(ctx, model)
	if err == nil {
		meta.Family = resolveFamily(model, meta)
		meta.Protocol = resolveProtocol(model, meta)
		r.mu.Lock()
		r.cache[key] = meta
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
		m.Protocol = resolveProtocol(model, m)
		r.mu.Lock()
		r.cache[key] = m
		r.mu.Unlock()
		return m, true
	}

	// Priority 5: Fallback to defaults.
	// Assume attachment support optimistically: an unknown model may well be
	// multimodal, and it is better to surface a runtime provider error than to
	// silently deny image uploads for a model the registry does not know.
	meta = ModelMetadata{
		ContextWindow: 128000,
		OutputLimit:   32768,
		TokenizerType: "approximate",
		Capabilities:  defaultUnknownCapabilities(),
	}
	meta.Family = resolveFamily(model, meta)
	meta.Protocol = resolveProtocol(model, meta)
	return meta, false
}

// Invalidate removes an entry from the cache map (for model change mid-session).
func (r *ModelRegistry) Invalidate(model string) {
	r.mu.Lock()
	delete(r.cache, strings.ToLower(model))
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
	meta.Protocol = resolveProtocol(model, meta)
	r.mu.Lock()
	r.cache[strings.ToLower(model)] = meta
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

// resolveProtocol determines the API protocol for a model.
// If already set in metadata, returns it directly; otherwise delegates to DetectProtocol.
func resolveProtocol(modelID string, meta ModelMetadata) APIProtocol {
	if meta.Protocol != "" {
		return meta.Protocol
	}
	return DetectProtocol(modelID)
}

// enrichPartialOverride fills the zero/empty scalar fields of a PARTIAL
// override — one that pins only some dimensions, e.g. a protocol-only
// auto-remap that injects {Protocol: ChatCompletions} for a local
// Google-named checkpoint served by LM Studio/vLLM — by inheriting them from
// the lower-priority NON-network tiers (built-in exact → built-in fuzzy →
// cache → fallback defaults). Network tiers (HuggingFace, registered sources)
// are deliberately skipped so an override lookup never performs I/O.
//
// A field already set in the override is authoritative and left untouched, so
// a fully-specified override (the common case: config.yaml entries whose
// ContextWindow/OutputLimit/TokenizerType/Capabilities are all populated at
// registry construction) is returned verbatim. Only the inheritable fields
// participate: Family is derived by resolveFamily, and Protocol is always
// authoritative (that is precisely what a partial override exists to pin).
//
// The inheritable fields are ContextWindow, OutputLimit, TokenizerType, and
// Capabilities. Capabilities joins the scalar set as a first-class inheritable
// field: a zero-valued ModelCapabilities (all bools false) is treated as
// "unset" and inherited from the lower tiers, exactly like a zero
// ContextWindow. This closes the footgun where a minimal protocol-pinning
// override of a well-known model (e.g. {"gpt-4o": {Protocol: ChatCompletions}})
// silently disabled image uploads, reasoning, and tool-calling. The accepted
// tradeoff — mirroring the scalars — is that there is no way to express
// "explicitly disable every capability" via a partial override, since the zero
// value is reserved for "inherit".
//
// The motivating bug: a protocol-only override that also carried a fallback
// ContextWindow (128000) would shadow a lazy local-model probe writing the
// model's REAL window to the cache tier (SetCachedMetadata), so the effective
// window stayed at 128000 forever. Keeping the override's window at zero and
// inheriting it here lets the cache-tier probe result take effect once it
// arrives.
func (r *ModelRegistry) enrichPartialOverride(model, key string, override ModelMetadata) ModelMetadata {
	if override.ContextWindow != 0 && override.OutputLimit != 0 &&
		override.TokenizerType != "" && override.Capabilities != (ModelCapabilities{}) {
		return override
	}
	lower := r.resolveBuiltinOrCache(model, key)
	if override.ContextWindow == 0 {
		override.ContextWindow = lower.ContextWindow
	}
	if override.OutputLimit == 0 {
		override.OutputLimit = lower.OutputLimit
	}
	if override.TokenizerType == "" {
		override.TokenizerType = lower.TokenizerType
	}
	if override.Capabilities == (ModelCapabilities{}) {
		override.Capabilities = lower.Capabilities
	}
	return override
}

// resolveBuiltinOrCache resolves a model using only the non-network,
// already-resolved tiers plus the fallback defaults — built-in exact → built-in
// fuzzy → cache → fallback — with no I/O. It is the "lower-tier" baseline used
// by enrichPartialOverride to fill the unset scalar fields of a partial
// override. r.builtIn/r.builtInIndex are immutable after construction (no lock
// needed); r.cache is guarded by r.mu.
func (r *ModelRegistry) resolveBuiltinOrCache(model, key string) ModelMetadata {
	if meta, ok := r.builtIn[key]; ok {
		return meta
	}
	if want := normalizeModelID(model); want != "" {
		if meta, ok := r.builtInIndex[want]; ok {
			return meta
		}
	}
	r.mu.RLock()
	meta, ok := r.cache[key]
	r.mu.RUnlock()
	if ok {
		return meta
	}
	// Match Resolve's tier-5 fallback exactly, including the optimistic
	// Capabilities. enrichPartialOverride inherits unset scalar fields from
	// this value; if Capabilities were left at the zero value here, a
	// protocol-only partial override for a catalog-MISS model would silently
	// disable Attachment (and every other capability) — the same footgun the
	// built-in inheritance path already guards against for catalog-HIT models.
	return ModelMetadata{
		ContextWindow: 128000,
		OutputLimit:   32768,
		TokenizerType: "approximate",
		Capabilities:  defaultUnknownCapabilities(),
	}
}

// modelIDSeparators strips the punctuation real-world hosts use (or omit)
// inconsistently between tokens of a model identifier.
var modelIDSeparators = strings.NewReplacer(".", "", "-", "", "_", "")

// normalizeModelID normalizes a model identifier for the fuzzy lookup. It
// strips the vendor/org prefix (everything up to and including the first "/"),
// lowercases the result, then removes separator punctuation (".", "-", "_").
//
// The vendor prefix is a routing decoration, not a property of the model: the
// HuggingFace checkpoint "zai-org/glm-5.2-fp8" and the Z.ai API model
// "GLM-5.2-FP8" share the same metadata. Discarding the prefix before
// comparison lets a bare query match a prefixed registry key (and vice versa),
// so vendor spelling never defeats the match.
//
// Only the vendor prefix and punctuation are removed; alphanumerics are never
// altered, so distinct model versions stay distinct: "qwen3.6" and "qwen3.7"
// normalize to "qwen36" and "qwen37". This is the foundation of the fuzzy
// lookup and is deliberately conservative — unlike edit distance it can never
// silently remap one model version onto another.
func normalizeModelID(id string) string {
	return modelIDSeparators.Replace(strings.ToLower(BareModel(id)))
}

// fuzzyLookup performs a vendor-prefix- and separator-insensitive search across
// user overrides and the built-in registry, used when the exact
// (case-insensitive) lookup misses. It bridges the cosmetic naming differences
// real-world model hosts introduce — e.g. a registry key "qwen/qwen3.6-35b-a3b-fp8"
// served by a host as "Qwen/Qwen36-35B-A3B-FP8" (dot dropped) collapses to the
// same normalized form, and a prefixed key "zai-org/glm-5.2-fp8" matches a bare
// query "GLM-5.2-FP8" (vendor prefix discarded). Overrides take priority over
// built-ins, mirroring the exact-lookup tiers. Returns false when nothing
// matches, so callers can fall through to network sources and the final default.
//
// Lookups are O(1) map reads against the normalized-ID indexes
// (r.overridesIndex, r.builtInIndex) built once at registry construction — no
// per-call normalization or full-table scan.
func (r *ModelRegistry) fuzzyLookup(model string) (ModelMetadata, bool) {
	want := normalizeModelID(model)
	if want == "" {
		return ModelMetadata{}, false
	}
	if meta, ok := r.overridesIndex[want]; ok {
		return meta, true
	}
	meta, ok := r.builtInIndex[want]
	return meta, ok
}

// buildNormalizedIndex returns a lookup table mapping each entry's normalized
// identifier (see normalizeModelID) to its metadata. When two keys collapse to
// the same normalized form (rare: usually aliases of one model), the
// lexicographically smallest original key wins, preserving the deterministic
// tie-break previously implemented imperatively by the per-call scan. Building
// the index once — at registry construction for overrides, lazily for the
// shared built-in catalog — turns every fuzzy lookup from an O(n) scan with
// per-key normalization into an O(1) map read.
func buildNormalizedIndex(m map[string]ModelMetadata) map[string]ModelMetadata {
	index := make(map[string]ModelMetadata, len(m))
	winners := make(map[string]string, len(m))
	for k, v := range m {
		norm := normalizeModelID(k)
		if norm == "" {
			continue
		}
		if winner, ok := winners[norm]; !ok || k < winner {
			winners[norm] = k
			index[norm] = v
		}
	}
	return index
}

// builtInIndexOnce guards lazy initialization of the normalized-ID index over
// the shared, immutable built-in catalog.
var (
	builtInIndexOnce  sync.Once
	builtInIndexCache map[string]ModelMetadata
)

// getBuiltInIndex returns the cached normalized-ID index over the built-in
// catalog, initializing it on first call. The underlying catalog is immutable
// data, so the derived index is safe to share.
func getBuiltInIndex() map[string]ModelMetadata {
	builtInIndexOnce.Do(func() {
		builtInIndexCache = buildNormalizedIndex(getBuiltInRegistry())
	})
	return builtInIndexCache
}

// defaultUnknownCapabilities is the capability assumption applied to models for
// which the registry has no authoritative capability data: the final fallback
// (no source recognized the model) and HuggingFace-resolved models (config.json
// yields a context window but no capability information).
//
// Attachment support is assumed optimistically. It is far better to let a user
// attach an image and surface a provider error at runtime than to silently
// disable image uploads for a model the registry simply does not know about.
func defaultUnknownCapabilities() ModelCapabilities {
	return ModelCapabilities{Attachment: true}
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
		OutputLimit:   32768,
		TokenizerType: "approximate",
		// config.json carries no capability information, so assume attachment
		// support optimistically (see defaultUnknownCapabilities).
		Capabilities: defaultUnknownCapabilities(),
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
			OutputLimit:   32768,
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
		// reasoning), the kimi-k2.7-code series (256K, coding-dedicated,
		// thinking always on) and the kimi-k2.5/k2.6 series (256K context,
		// multimodal, thinking + non-thinking modes). All support Agent/tool
		// tasks.
		// The Kimi Code subscription endpoint (api.kimi.com/coding) serves the
		// same models under SHORT, non-prefixed IDs — "k3" (up to 1M, tier
		// dependent), "k3-256k" (fixed 256K), "kimi-for-coding" and
		// "kimi-for-coding-highspeed" (K2.7 Code, 256K). Source:
		// https://www.kimi.com/code/docs/en/kimi-code/models
		// These short IDs contain no "kimi" substring, so DetectFamily would
		// miss them — the explicit Family here is load-bearing.
		// Note: OutputLimit is a context-window reserve (subtracted from the
		// window), so we store a practical reserve, not the absolute ceiling.
		// kimi-k3: 131K default (configurable up to 1M).
		// kimi-k2.7-code / kimi-k2.6 / kimi-k2.5: 256K max output; reserve 65K
		// (the 256K ceiling would zero the input budget under EffectiveMax).
		"kimi-k3": {
			ContextWindow: 1000000,
			OutputLimit:   131072,
			TokenizerType: "approximate",
			Family:        "kimi",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		// Kimi Code endpoint aliases (see the section comment above).
		"k3": {
			ContextWindow: 1000000,
			OutputLimit:   131072,
			TokenizerType: "approximate",
			Family:        "kimi",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"k3-256k": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "kimi",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"kimi-k2.7-code": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "kimi",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"kimi-k2.7-code-highspeed": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "kimi",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"kimi-for-coding": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "kimi",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"kimi-for-coding-highspeed": {
			ContextWindow: 262144,
			OutputLimit:   65536,
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
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "kimi",
			Capabilities:  ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"kimi-k2-thinking": {
			ContextWindow: 131072,
			OutputLimit:   32768,
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
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"qwen/qwen3-4b-thinking-2507": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		"qwen/qwen3-30b-a3b-2507": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"qwen/qwen3-30b-a3b-thinking-2507": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		"qwen/qwen3-235b-a22b-2507": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"qwen/qwen3-235b-a22b-thinking-2507": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		// Qwen3 (1st generation, 2504): hybrid thinking, 128K context.
		"qwen/qwen3-4b": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		"qwen/qwen3-8b": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		"qwen/qwen3-14b": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		"qwen/qwen3-32b": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		"qwen/qwen3-30b-a3b": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		"qwen/qwen3-235b-a22b": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		// Qwen3.5: multimodal (text+image), thinking, 256K context.
		"qwen/qwen3.5-2b": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"qwen/qwen3.5-4b": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"qwen/qwen3.5-9b": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"qwen/qwen3.5-27b": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"qwen/qwen3.5-35b-a3b": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		// Qwen3.5 FP8 (HuggingFace): official FP8 checkpoints of the flagship
		// gated-delta-networks MoE members. Same multimodal (text+image),
		// thinking, and 256K context as the rest of the Qwen3.5 line.
		//   config: max_position_embeddings=262144 (Qwen3_5MoeForConditionalGeneration)
		"qwen/qwen3.5-397b-a17b-fp8": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"qwen/qwen3.5-122b-a10b-fp8": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		// Qwen3.6: multimodal, thinking, 256K context.
		"qwen/qwen3.6-27b": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"qwen/qwen3.6-35b-a3b": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		// Qwen3.6 FP8 (HuggingFace): FP8 checkpoint of the smaller Qwen3.6
		// flagship — same multimodal MoE architecture and 256K context.
		//   config: max_position_embeddings=262144 (Qwen3_5MoeForConditionalGeneration)
		"qwen/qwen3.6-35b-a3b-fp8": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		// Qwen3-Coder: non-thinking coding MoE, 256K context.
		"qwen/qwen3-coder-30b": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"qwen/qwen3-coder-480b": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"qwen/qwen3-coder-next": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"qwen/qwen3-next-80b": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Temperature: true, ToolCall: true},
		},
		// Qwen3-VL: vision-language, thinking, 256K context.
		"qwen/qwen3-vl-2b": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"qwen/qwen3-vl-4b": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"qwen/qwen3-vl-8b": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"qwen/qwen3-vl-30b": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"qwen/qwen3-vl-32b": {
			ContextWindow: 262144,
			OutputLimit:   65536,
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
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "deepseek",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true},
		},
		"deepseek/deepseek-r1-distill-qwen-7b": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "deepseek",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true},
		},
		"deepseek/deepseek-r1-distill-llama-8b": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "deepseek",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true},
		},
		"deepseek/deepseek-r1-distill-qwen-14b": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "deepseek",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true},
		},
		"deepseek/deepseek-r1-distill-qwen-32b": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "deepseek",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true},
		},
		"deepseek/deepseek-r1-distill-llama-70b": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "deepseek",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true},
		},

		// ── GLM (Z.ai open coding) ─────────────────────────────────────
		"zai-org/glm-4.7-flash": {
			ContextWindow: 202752,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "glm",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		"zai-org/glm-4.6v-flash": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "glm",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		// GLM-5.2 FP8 (HuggingFace): official FP8 checkpoint of Z.ai's
		// flagship 753B MoE. Same reasoning/tool-calling model with native
		// 1 MiB context (config max_position_embeddings=1048576,
		// GlmMoeDsaForCausalLM); FP8 only trims the memory footprint.
		"zai-org/glm-5.2-fp8": {
			ContextWindow: 1048576,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "glm",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},

		// ── Google Gemma (open-weights) ────────────────────────────────
		// Gemma 4: multimodal (text+image; +audio on E2B/E4B/12B), thinking,
		// native function calling. E2B/E4B = 128K; 12B/26B-A4B/31B = 256K.
		"google/gemma-4-e2b": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "google",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"google/gemma-4-e4b": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "google",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"google/gemma-4-12b": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "google",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"google/gemma-4-26b-a4b": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "google",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"google/gemma-4-31b": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "google",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		// Gemma 4 QAT (Quantization-Aware Training): GGUF builds of the
		// instruction-tuned 12B/26B-A4B/31B models trained to be robust to
		// low-bit quantization. Same architecture, 256K context, and
		// capabilities as their dense/MoE base models — QAT only reduces the
		// memory footprint, not model behavior.
		//   config: max_position_embeddings=262144 (verified google/gemma-4-31B-it)
		"google/gemma-4-12b-qat": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "google",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"google/gemma-4-26b-a4b-qat": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "google",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"google/gemma-4-31b-qat": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "google",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		// Gemma 4 31B-it FP8: official FP8 checkpoint of the instruction-tuned
		// 31B model (google/gemma-4-31B-it, image-text-to-text). FP8 only trims
		// the memory footprint — same 256K context and multimodal capabilities
		// as the dense 31B model.
		"google/gemma-4-31b-it-fp8": {
			ContextWindow: 262144,
			OutputLimit:   65536,
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
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "mistral",
			Capabilities:  ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"mistralai/mistral-small-3.2": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "mistral",
			Capabilities:  ModelCapabilities{Attachment: true, Temperature: true, ToolCall: true},
		},
		// Magistral: reasoning, 128K. v1.2 (+vision) and v1.1 (text).
		"mistralai/magistral-small-2509": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "mistral",
			Capabilities:  ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"mistralai/magistral-small": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "mistral",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		// Devstral: agentic coding, 128K. Devstral 2 adds vision (123B = 256K).
		"mistralai/devstral-small-2507": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "mistral",
			Capabilities:  ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"mistralai/devstral-small-2505": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "mistral",
			Capabilities:  ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"mistralai/devstral-small-2-2512": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "mistral",
			Capabilities:  ModelCapabilities{Attachment: true, Temperature: true, ToolCall: true},
		},
		"mistralai/devstral-2-2512": {
			ContextWindow: 262144,
			OutputLimit:   65536,
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
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "mistral",
			Capabilities:  ModelCapabilities{Attachment: true, Temperature: true, ToolCall: true},
		},
		"mistralai/ministral-3-8b": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "mistral",
			Capabilities:  ModelCapabilities{Attachment: true, Temperature: true, ToolCall: true},
		},
		"mistralai/ministral-3-14b": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "mistral",
			Capabilities:  ModelCapabilities{Attachment: true, Temperature: true, ToolCall: true},
		},

		// ── OpenAI gpt-oss (open-weights) ──────────────────────────────
		// gpt-oss: configurable reasoning effort + tool use, 128K.
		"openai/gpt-oss-20b": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		"openai/gpt-oss-120b": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		// gpt-oss-safeguard: content classifiers (not chat/agent models).
		"openai/gpt-oss-safeguard-20b": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  ModelCapabilities{Temperature: true},
		},
		"openai/gpt-oss-safeguard-120b": {
			ContextWindow: 131072,
			OutputLimit:   32768,
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
			OutputLimit:   32768,
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
			OutputLimit:   32768,
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
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"ibm/granite-4-micro": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"ibm/granite-4-h-tiny": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"ibm/granite-4-h-small": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"ibm/granite-4.1-3b": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"ibm/granite-4.1-8b": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"ibm/granite-4.1-30b": {
			ContextWindow: 131072,
			OutputLimit:   32768,
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
			OutputLimit:   16384,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"allenai/olmo-3-7b-think": {
			ContextWindow: 65536,
			OutputLimit:   16384,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		"allenai/olmo-3-32b-think": {
			ContextWindow: 65536,
			OutputLimit:   16384,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		"allenai/olmocr-2-7b": {
			ContextWindow: 128000,
			OutputLimit:   16384,
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
			OutputLimit:   32768,
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
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  ModelCapabilities{Temperature: true},
		},
		"microsoft/phi-4-mini-reasoning": {
			ContextWindow: 131072,
			OutputLimit:   32768,
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

// ResolveBuiltInModel resolves a model's metadata using ONLY the built-in
// catalog — an exact case-insensitive lookup followed by the
// separator-insensitive fuzzy match — with no network access and no
// consideration of user overrides, the lazy cache, HuggingFace, or registered
// sources. When the model is absent from the built-in catalog it returns the
// same fallback metadata Resolve uses (ContextWindow=128000, OutputLimit=32768,
// TokenizerType="approximate") with ok=false.
//
// It is intended for callers that need the "factory default" for a model
// independent of any user override — e.g. to decide which fields of a config
// override actually differ from the built-in values, or to merge a partial
// override onto the built-in metadata at registry-construction time. Because it
// never touches the network, it is safe to call from startup/seed paths.
func ResolveBuiltInModel(model string) (ModelMetadata, bool) {
	key := strings.ToLower(model)
	builtIn := getBuiltInRegistry()
	if meta, ok := builtIn[key]; ok {
		meta.Family = resolveFamily(model, meta)
		meta.Protocol = resolveProtocol(model, meta)
		return meta, true
	}
	if want := normalizeModelID(model); want != "" {
		if meta, ok := getBuiltInIndex()[want]; ok {
			meta.Family = resolveFamily(model, meta)
			meta.Protocol = resolveProtocol(model, meta)
			return meta, true
		}
	}
	meta := ModelMetadata{
		ContextWindow: 128000,
		OutputLimit:   32768,
		TokenizerType: "approximate",
		Capabilities:  defaultUnknownCapabilities(),
	}
	meta.Family = resolveFamily(model, meta)
	meta.Protocol = resolveProtocol(model, meta)
	return meta, false
}
