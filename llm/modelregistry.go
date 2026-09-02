package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"
)

// maxHuggingFaceConfigBytes caps the response body when fetching model metadata
// from HuggingFace. config.json is normally small, but the endpoint is
// user/config-controlled and could return an oversized body.
const maxHuggingFaceConfigBytes = 1 << 20 // 1 MiB

// negativeCacheTTL is how long a failed HuggingFace probe suppresses further
// probes of the same model key. Without it, a model unknown to every tier
// would cost one HTTP round-trip per Resolve call, and Resolve sits on hot UI
// paths (model pickers, context-usage estimation) where the same unknown ID is
// resolved over and over. The window bounds a mistyped or genuinely missing
// model to at most one wasted request per TTL while still letting a model
// published in the meantime become visible without an app restart or an
// explicit Invalidate.
const negativeCacheTTL = 10 * time.Minute

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
	Protocol APIProtocol
	// Capabilities is a POINTER so a record can distinguish "capabilities
	// declared" (non-nil — even when every flag is false) from "capabilities
	// unset — inherit from the lower tiers" (nil). Partial records (config
	// overrides, observed runtime entries) leave it nil for the dimensions
	// they did not pin; fully-populated records (built-in catalog, fallback
	// defaults) always carry a non-nil value. Readers must nil-check before
	// dereferencing raw/partial records; Resolve, ResolveLocal, and
	// ResolveBuiltInModel output is always enriched to non-nil AND defensively
	// copied — the caller owns the returned pointer and may mutate it freely
	// (registry state, including the process-global built-in catalog, is never
	// aliased). Symmetrically, records handed TO the registry (overrides maps,
	// SetCachedMetadata, SetRuntimeMetadata) are copied on write, so the
	// caller's later mutations cannot reach the stored entries.
	Capabilities *ModelCapabilities
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
	runtime        map[string]ModelMetadata // observed runtime limits (tier 1.5): above built-ins, below user overrides
	runtimeIndex   map[string]ModelMetadata // normalized-ID -> metadata index for fuzzy runtime lookup; rebuilt under mu on every runtime write
	cache          map[string]ModelMetadata
	negativeCache  map[string]time.Time  // lowercased key → time of the last failed HuggingFace probe
	sources        []ModelMetadataSource // external metadata sources (e.g., LM Studio)
	mu             sync.RWMutex
	httpClient     *http.Client
}

// NewModelRegistry creates a new registry with built-in data and optional user overrides.
//
// The overrides map is defensively copied at construction time so that callers
// (e.g. config reloads) can mutate their own map without racing the registry's
// concurrent readers. The copy is DEEP with respect to Capabilities: each
// entry's pointer is cloned, so the caller mutating its capability structs
// after hand-off cannot reach (or race) the registry's stored entries.
func NewModelRegistry(overrides map[string]ModelMetadata) *ModelRegistry {
	copied := make(map[string]ModelMetadata, len(overrides))
	for k, v := range overrides {
		// Normalize override keys to lowercase so they are matched by the
		// case-insensitive Resolve lookup.
		v.Capabilities = cloneCapabilities(v.Capabilities)
		copied[strings.ToLower(k)] = v
	}
	return &ModelRegistry{
		builtIn:        getBuiltInRegistry(),
		builtInIndex:   getBuiltInIndex(),
		overrides:      copied,
		overridesIndex: buildNormalizedIndex(copied),
		runtime:        make(map[string]ModelMetadata),
		runtimeIndex:   make(map[string]ModelMetadata),
		cache:          make(map[string]ModelMetadata),
		negativeCache:  make(map[string]time.Time),
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
//   - 1.5. Observed runtime entries (SetRuntimeMetadata) — how the model is
//     actually being served; supersedes the built-in spec (tier 2)
//   - 2. Built-in registry (hardcoded table)
//   - 2b. Fuzzy match across overrides, observed runtime entries, and
//     built-ins, with the vendor prefix stripped and separator punctuation
//     (".", "-", "_") removed — so "qwen3.6" matches "qwen36", "gpt-4"
//     matches "gpt4", and a bare "glm-5.2-fp8" matches the prefixed
//     "zai-org/glm-5.2-fp8". Consulted only on an exact-lookup miss; the
//     result is cached under the query key.
//   - 3. Lazy cache (HuggingFace results, source results, SetCachedMetadata)
//   - 4. External sources, in order: HuggingFace API lookup (lazy, cached)
//     and sources registered via RegisterSource (e.g. an LM Studio provider).
//     A FAILED HuggingFace probe is remembered in a negative cache for
//     negativeCacheTTL: repeat Resolve calls inside that window skip the
//     network entirely and fall straight through to registered sources. A
//     subsequent success clears the negative record; termination of the
//     caller's context (cancellation or deadline) is never recorded (see the
//     write site below).
//   - 5. Fallback defaults (ok=false)
//
// The local, in-memory portion of the lookup (tiers 1, 2, 2b, and the cache
// read of tier 3) is shared with ResolveLocal, which Resolve consults first —
// the two methods cannot drift apart on those tiers.
//
// The second return value indicates whether the model was found in a known source.
// When ok is false, the returned metadata contains usable fallback defaults.
// Regardless of tier, the returned Capabilities is always non-nil and
// defensively copied — the caller owns it (see finalizeMeta).
func (r *ModelRegistry) Resolve(ctx context.Context, model string) (ModelMetadata, bool) {
	// Local tiers first: overrides → built-in → fuzzy → lazy cache (see
	// ResolveLocal). A hit needs no I/O, so the network tiers below run only
	// on a miss.
	if meta, ok := r.ResolveLocal(model); ok {
		return meta, true
	}

	// Case-insensitive key for the cache maps; the original casing is
	// preserved for the HuggingFace URL path (same convention as ResolveLocal).
	key := strings.ToLower(model)

	// Priority 4 (first half): Fetch from HuggingFace — unless a recent probe
	// already failed. The negative cache turns "unknown model" from one HTTP
	// round-trip per Resolve into at most one per negativeCacheTTL window.
	if !r.negativeCacheFresh(key) {
		meta, err := r.fetchFromHuggingFace(ctx, model)
		if err == nil {
			meta = finalizeMeta(model, meta)
			// Cache a twin that owns its own Capabilities copy, so caller
			// mutations of the returned metadata can never reach the cached
			// entry (and race later readers). The negative-cache record is
			// cleared under the same lock: a successful probe ends the
			// suppression window atomically with the cache write.
			cached := meta
			cached.Capabilities = cloneCapabilities(meta.Capabilities)
			r.mu.Lock()
			r.cache[key] = cached
			delete(r.negativeCache, key)
			r.mu.Unlock()
			return meta, true
		}
		// Every failure shape counts — 404, timeout, transport error,
		// malformed or empty config — because each means "HuggingFace cannot
		// resolve this model right now". Record it so the next Resolve within
		// the TTL skips the round-trip. Termination of the CALLER's context is
		// the one exception: it describes the caller's state (a step deadline
		// hit, a shutdown, or an explicit cancel), not HuggingFace's, so the
		// next Resolve with a live context re-probes immediately instead of
		// riding out the window on fallback metadata. The registry's own HTTP
		// client timeout still records: it fires while the caller's context is
		// still live (ctx.Err() == nil), and that failure legitimately
		// describes HuggingFace being too slow.
		if ctx.Err() == nil {
			r.mu.Lock()
			r.negativeCache[key] = time.Now()
			r.mu.Unlock()
		}
	}

	// Priority 4 (second half): Try registered sources
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
		m = finalizeMeta(model, m)
		r.cacheResolved(key, m)
		return m, true
	}

	// Priority 5: Fallback to defaults.
	// Assume attachment support optimistically: an unknown model may well be
	// multimodal, and it is better to surface a runtime provider error than to
	// silently deny image uploads for a model the registry does not know.
	return finalizeMeta(model, fallbackMetadata()), false
}

// ResolveLocal returns model metadata using a strictly local, network-free
// tiered lookup — the same tier sequence as Resolve with every tier that can
// perform I/O removed:
//   - 1. User overrides (from config)
//   - 1.5. Observed runtime entries (SetRuntimeMetadata)
//   - 2. Built-in registry (hardcoded table)
//   - 2b. Fuzzy match across overrides, observed runtime entries, and
//     built-ins, cached under the query key
//   - 3. Lazy cache — READ-ONLY: entries previously written by Resolve's
//     HuggingFace/source tiers, by SetCachedMetadata, or by a fuzzy hit
//   - 5. Fallback defaults (ok=false)
//
// GUARANTEE: ResolveLocal never touches the network. It issues no HTTP
// requests, consults no registered metadata sources, and never blocks on I/O —
// for a model no tier knows it returns immediately with the fallback defaults
// and ok=false rather than probing. It is the resolver of choice for UI paths
// (model pickers, context-usage meters, settings screens) where a synchronous
// network lookup would stall rendering or fire on every keystroke, and for
// code paths that hold no context.Context.
//
// Resolve delegates its local tiers to this method, so tiers 1/2/2b and the
// cache read behave identically under both entry points and cannot drift.
//
// The second return value indicates whether the model was found in a known
// local source; when it is false the returned metadata contains the same
// usable fallback defaults Resolve returns. Regardless of tier, the returned
// Capabilities is always non-nil and defensively copied — the caller owns it
// (see finalizeMeta).
func (r *ModelRegistry) ResolveLocal(model string) (ModelMetadata, bool) {
	// Case-insensitive lookup: model IDs arriving from OpenAI-compatible hosts
	// (e.g. vLLM) frequently differ only in casing from their canonical
	// registry keys. Normalize once and use this key for every map lookup;
	// family/protocol detection lowercases independently.
	key := strings.ToLower(model)

	// Priority 1: Check overrides (no lock needed for read-only map after construction)
	if meta, ok := r.overrides[key]; ok {
		// A partial override (one that pins only some fields, e.g. a
		// protocol-only auto-remap) inherits its unset scalar fields from the
		// lower non-network tiers so it does not collapse the context window
		// or output limit to zero. See enrichPartialOverride.
		meta = r.enrichPartialOverride(model, key, meta)
		return finalizeMeta(model, meta), true
	}

	// Priority 1.5: Observed runtime entry (SetRuntimeMetadata). A self-hosted
	// server's runtime window outranks the built-in catalog spec — the catalog
	// describes the checkpoint's maximum capability, while the runtime entry
	// describes the context length the server is actually enforcing. See
	// SetRuntimeMetadata for the precedence rationale. Partial runtime entries
	// enrich against the tiers strictly below runtime (resolveSpecOrCache).
	if meta, ok := r.runtimeLookup(key); ok {
		meta = r.enrichPartialWith(meta, func() ModelMetadata {
			return r.resolveSpecOrCache(model, key)
		})
		return finalizeMeta(model, meta), true
	}

	// Priority 2: Check built-in registry (no lock needed for read-only map)
	if meta, ok := r.builtIn[key]; ok {
		return finalizeMeta(model, meta), true
	}

	// Priority 2b: Fuzzy match — separator-insensitive lookup across
	// overrides, observed runtime entries, and built-ins (in that precedence).
	// Bridges cosmetic naming drift (a host serving
	// "Qwen/Qwen36-35B-A3B-FP8" for the registry key "qwen/qwen3.6-35b-a3b-fp8")
	// without the false-positive risk of edit-distance matching. Cache the hit
	// under the query key so repeat resolves take the fast path.
	if meta, ok := r.fuzzyLookup(model); ok {
		meta = finalizeMeta(model, meta)
		r.cacheResolved(key, meta)
		return meta, true
	}

	// Priority 3: Lazy cache, read-only. Entries arrive from the outside —
	// Resolve's network tiers or SetCachedMetadata — never from this method.
	r.mu.RLock()
	meta, ok := r.cache[key]
	r.mu.RUnlock()
	if ok {
		return finalizeMeta(model, meta), true
	}

	// Priority 5: Fallback to defaults — identical values and rationale as
	// Resolve's tier 5 (see fallbackMetadata).
	return finalizeMeta(model, fallbackMetadata()), false
}

// negativeCacheFresh reports whether a failed HuggingFace probe for key is
// still inside the negativeCacheTTL window, meaning the probe must not be
// repeated yet. It first sweeps every expired entry so the map holds only
// keys inside their current TTL window and cannot grow without bound even for
// models that are never resolved again after their window lapses. An absent
// entry reads as stale, and an EXPIRED entry is deleted on the spot. Either
// way the caller re-probes and, on another failure, overwrites the recorded
// timestamp, restarting the window.
func (r *ModelRegistry) negativeCacheFresh(key string) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	for k, checkedAt := range r.negativeCache {
		if time.Since(checkedAt) >= negativeCacheTTL {
			delete(r.negativeCache, k)
		}
	}
	_, ok := r.negativeCache[key]
	return ok
}

// fallbackMetadata returns the tier-5 defaults used when no source recognizes
// the model. The values are usable in practice — a generous context window, a
// bounded output limit, the approximate tokenizer — plus the optimistic
// capability set (see defaultUnknownCapabilities for why Attachment is
// assumed). Family and Protocol are deliberately left empty for the caller to
// derive from the model ID via resolveFamily/resolveProtocol.
func fallbackMetadata() ModelMetadata {
	return ModelMetadata{
		ContextWindow: 128000,
		OutputLimit:   32768,
		TokenizerType: "approximate",
		Capabilities:  defaultUnknownCapabilities(),
	}
}

// Invalidate removes an entry from the cache map (for model change mid-session).
// It clears both the positive cache entry and any negative-cache record, so the
// next Resolve re-probes HuggingFace from scratch instead of waiting out the
// negativeCacheTTL window. The observed runtime entry (tier 1.5) is cleared
// too, along with its fuzzy-index mirror: it describes the PREVIOUS serving
// arrangement, which a model switch has just invalidated.
func (r *ModelRegistry) Invalidate(model string) {
	r.mu.Lock()
	key := strings.ToLower(model)
	delete(r.cache, key)
	delete(r.negativeCache, key)
	delete(r.runtime, key)
	r.rebuildRuntimeIndex()
	r.mu.Unlock()
}

// SetCachedMetadata stores a late-learned metadata entry in the registry cache
// (Resolution tier 3). This lets callers that discover a model's real context
// window after construction — e.g. a lazy probe of a local OpenAI-compatible
// server — populate the result so subsequent Resolve calls return it without
// re-querying the network.
//
// Tier-3 priority means the entry takes effect only when no user override
// (tier 1), observed runtime entry (tier 1.5), or built-in entry (tier 2)
// already exists for the model, so a config.yaml override, a server probe
// result, or a well-known model's spec is never silently clobbered.
// Calling this for a model that already resolved from tier 1/1.5/2 will simply be
// shadowed at Resolve time.
//
// The entry is defensively copied on write (Capabilities is cloned), so the
// caller may keep mutating its own ModelMetadata afterwards without racing
// the registry's concurrent readers.
func (r *ModelRegistry) SetCachedMetadata(model string, meta ModelMetadata) {
	meta.Family = resolveFamily(model, meta)
	meta.Protocol = resolveProtocol(model, meta)
	meta.Capabilities = cloneCapabilities(meta.Capabilities)
	r.mu.Lock()
	r.cache[strings.ToLower(model)] = meta
	r.mu.Unlock()
}

// SetRuntimeMetadata stores an OBSERVED runtime metadata entry (Resolution
// tier 1.5 — above the built-in catalog, below user overrides). It exists for
// late-learned facts about how a model is ACTUALLY being served, as opposed to
// its published spec: a self-hosted OpenAI-compatible server (LM Studio,
// vLLM, Ollama) frequently serves a well-known checkpoint with a runtime
// context window far below the catalog maximum (LM Studio loads models at a
// user-chosen context length; Ollama defaults num_ctx to 8K; vLLM is started
// with --max-model-len). Budgeting compaction against the catalog window in
// that case leaves the effective budget inflated until the API starts
// rejecting requests, and the status bar displays a maximum the server will
// never honor.
//
// Precedence semantics:
//   - An explicit user override (tier 1) always wins: the user's config.yaml
//     choice is authoritative and is never clobbered by a probe.
//   - The runtime entry supersedes the built-in spec (tier 2) for every field
//     it carries, because the serving runtime — not the vendor datasheet — is
//     the ground truth for what the endpoint will accept.
//   - It also supersedes the lazy cache (tier 3), so a HuggingFace
//     config.json lookup for the checkpoint name cannot pin the spec window
//     over the observed one.
//
// Partial entries are NOT enriched here; zero fields inherit from the tiers
// below runtime (built-in → cache → fallback) via enrichPartialWith against
// resolveSpecOrCache — exactly like a partial user override. Callers that
// probe a server should set the fields they actually observed and leave the
// rest zero.
//
// Unlike the network tiers, this method performs no I/O. The entry is keyed
// by the exact (lowercased) model id the caller resolved, so it is consulted
// by both Resolve and the network-free ResolveLocal; it is also mirrored into
// the normalized-ID fuzzy index, so a later lookup under a cosmetically
// drifted spelling of the same id (e.g. a dropped dot) still finds the
// observed limits.
//
// The entry is defensively copied on write (Capabilities is cloned), so a
// probe that reuses its own ModelMetadata value afterwards cannot race the
// registry's concurrent readers.
func (r *ModelRegistry) SetRuntimeMetadata(model string, meta ModelMetadata) {
	meta.Family = resolveFamily(model, meta)
	meta.Protocol = resolveProtocol(model, meta)
	meta.Capabilities = cloneCapabilities(meta.Capabilities)
	r.mu.Lock()
	r.runtime[strings.ToLower(model)] = meta
	r.rebuildRuntimeIndex()
	r.mu.Unlock()
}

// rebuildRuntimeIndex re-derives the normalized-ID fuzzy index over the
// observed runtime entries. The caller must hold r.mu (exclusive): writes are
// rare (one per probe) and the entry set is small, so a full deterministic
// rebuild is simpler and safer than incremental index maintenance, preserving
// the same lexicographic tie-break as buildNormalizedIndex.
func (r *ModelRegistry) rebuildRuntimeIndex() {
	r.runtimeIndex = buildNormalizedIndex(r.runtime)
}

// RuntimeMetadata returns the observed runtime entry (tier 1.5) for a model,
// when present. It is the read-side companion of SetRuntimeMetadata, exposed
// so callers (e.g. a probe cache validator) can check what a previous probe
// learned without re-probing. The second return is false when no runtime
// entry exists for the model.
//
// The entry is returned AS STORED, without tier enrichment: a partial entry
// (e.g. one carrying only the observed ContextWindow) reports zero for the
// fields the probe did not observe. Use ResolveLocal for the effective,
// enriched values. The Capabilities pointer is cloned so the caller owns its
// copy — mutating the returned metadata never touches the stored entry.
func (r *ModelRegistry) RuntimeMetadata(model string) (ModelMetadata, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	meta, ok := r.runtime[strings.ToLower(model)]
	if meta.Capabilities == nil {
		// A stored entry can carry nil Capabilities (cloneCapabilities(nil)
		// yields nil). Normalize to the optimistic unknown set so callers can
		// dereference Capabilities without a panic, mirroring finalizeMeta's
		// read-side contract.
		meta.Capabilities = defaultUnknownCapabilities()
	} else {
		meta.Capabilities = cloneCapabilities(meta.Capabilities)
	}
	return meta, ok
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
// the lower-priority NON-network tiers (observed runtime → built-in exact →
// built-in fuzzy → cache → fallback defaults). Network tiers (HuggingFace,
// registered sources) are deliberately skipped so an override lookup never
// performs I/O.
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
// field via its nil pointer: nil = "unset, inherit from the lower tiers",
// exactly like a zero ContextWindow, while a non-nil value is AUTHORITATIVE —
// even when every flag is false. This closes the footgun where a minimal
// protocol-pinning override of a well-known model (e.g. {"gpt-4o": {Protocol:
// ChatCompletions}}) silently disabled image uploads, reasoning, and
// tool-calling, without reserving the all-false capability set as "inherit":
// a user who explicitly disables every capability gets exactly that.
//
// The motivating bug: a protocol-only override that also carried a fallback
// ContextWindow (128000) would shadow a lazy local-model probe writing the
// model's REAL window to the cache tier (SetCachedMetadata), so the effective
// window stayed at 128000 forever. Keeping the override's window at zero and
// inheriting it here lets the cache-tier probe result take effect once it
// arrives.
func (r *ModelRegistry) enrichPartialOverride(model, key string, override ModelMetadata) ModelMetadata {
	return r.enrichPartialWith(override, func() ModelMetadata {
		return r.resolveBuiltinOrCache(model, key)
	})
}

// enrichPartialWith is the engine behind enrichPartialOverride: it fills the
// zero/empty scalar fields of `override` from the (memoized-on-call) baseline.
// The baseline is a function so the runtime tier can enrich against the tiers
// strictly below it (see resolveSpecOrCache) without special-casing field
// logic in two places.
func (r *ModelRegistry) enrichPartialWith(override ModelMetadata, baseline func() ModelMetadata) ModelMetadata {
	if override.ContextWindow != 0 && override.OutputLimit != 0 &&
		override.TokenizerType != "" && override.Capabilities != nil {
		return override
	}
	lower := baseline()
	if override.ContextWindow == 0 {
		override.ContextWindow = lower.ContextWindow
	}
	if override.OutputLimit == 0 {
		override.OutputLimit = lower.OutputLimit
	}
	if override.TokenizerType == "" {
		override.TokenizerType = lower.TokenizerType
	}
	if override.Capabilities == nil {
		override.Capabilities = lower.Capabilities
	}
	return override
}

// runtimeLookup reads the observed runtime tier (1.5) under RLock. It returns
// the raw stored entry — enrichment of partial entries is the caller's job,
// mirroring the override tier.
func (r *ModelRegistry) runtimeLookup(key string) (ModelMetadata, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	meta, ok := r.runtime[key]
	return meta, ok
}

// resolveBuiltinOrCache resolves a model using only the non-network,
// already-resolved tiers plus the fallback defaults — runtime observed →
// built-in exact → built-in fuzzy → cache → fallback — with no I/O. It is the
// "lower-tier" baseline used by enrichPartialOverride to fill the unset
// scalar fields of a partial override. r.builtIn/r.builtInIndex are immutable
// after construction (no lock needed); r.cache is guarded by r.mu.
//
// The observed runtime tier leads the chain: when a partial override leaves
// the context window unset and a server probe has observed the real window,
// the observed value — not the catalog spec — must be inherited. This is what
// lets a user override that pins only the output limit still surface the
// self-hosted server's runtime window.
func (r *ModelRegistry) resolveBuiltinOrCache(model, key string) ModelMetadata {
	if meta, ok := r.runtimeLookup(key); ok {
		// A partial runtime entry must enrich against the tiers strictly below
		// runtime before it can serve as the baseline for a partial override.
		// Otherwise a protocol-only override stacked on a partial runtime entry
		// would inherit zero ContextWindow/OutputLimit/TokenizerType instead of
		// falling through to the built-in catalog / cache / fallback tiers.
		return r.enrichPartialWith(meta, func() ModelMetadata {
			return r.resolveSpecOrCache(model, key)
		})
	}
	return r.resolveSpecOrCache(model, key)
}

// resolveSpecOrCache is resolveBuiltinOrCache minus the runtime tier:
// built-in exact → built-in fuzzy → cache → fallback. It is the baseline used
// when enriching a PARTIAL RUNTIME entry — exact or fuzzy hit — whose unset
// fields must inherit from the tiers below it: consulting the runtime tier
// again would return the very entry being enriched and leave its zero fields
// zero.
func (r *ModelRegistry) resolveSpecOrCache(model, key string) ModelMetadata {
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
	// Match Resolve's tier-5 fallback exactly (fallbackMetadata), including
	// the optimistic Capabilities. enrichPartialOverride inherits unset scalar
	// fields from this value; if Capabilities were left nil here, a
	// protocol-only partial override for a catalog-MISS model would
	// silently disable Attachment (and every other capability) — the same
	// footgun the built-in inheritance path already guards against for
	// catalog-HIT models.
	return fallbackMetadata()
}

// modelIDSeparators strips the punctuation real-world hosts use (or omit)
// inconsistently between tokens of a model identifier.
var modelIDSeparators = strings.NewReplacer(".", "", "-", "", "_", "")

// modelIDPostfixes lists delivery/quantization decorations that hosts append to
// a base model name but that are not part of the model's canonical identity.
// They are stripped from the END of the identifier BEFORE its separators are
// removed (see stripModelIDPostfixes), so a GGUF download or an 8-bit quant
// serves the same model as its base: "qwen3-8bit", "qwen3-8-bit-gguf", and
// "qwen3-gguf" all collapse to "qwen3".
//
// Each token carries its leading separator ("-"/"_"), which is what makes
// stripping separator-aware: an instruct suffix is never misread as a
// quantization because "qwen3-8b-it" ends in "-it", not in any postfix token,
// while "qwen3-8bit"/"qwen3-8-bit"/"qwen3_8bit" are all stripped. Internal
// separator spellings ("8-bit"/"8_bit") are expanded explicitly because the
// separators are still present at strip time.
//
// Numeric-parameterized variant families ("mtp", "oq4e", "fp8", "q4", "bf16")
// are NOT in this literal list — their bit-width digits vary, so they are
// matched by modelIDVariantPostfixPattern (see below). int* tokens remain
// deliberately excluded everywhere: they are not an established serving-name
// postfix family.
//
// Order is not significant: stripModelIDPostfixes keeps the longest matching
// suffix across the whole list, so a shorter postfix that is a suffix of a
// longer one ("2bit" inside "32bit", "6bit" inside "16bit") can never win
// first (the leading separator additionally prevents such partial overlaps).
var modelIDPostfixes = buildModelIDPostfixes()

// buildModelIDPostfixes expands the separator-free postfix cores into every
// separator-aware suffix spelling, each prefixed by its leading separator.
func buildModelIDPostfixes() []string {
	cores := []string{
		"safetensors",
		"32bit",
		"16bit",
		"gguf",
		"ggml",
		"onnx",
		"gptq",
		"8bit",
		"6bit",
		"5bit",
		"4bit",
		"3bit",
		"2bit",
		"awq",
		"nf4",
		"mlx",
	}
	var postfixes []string
	for _, core := range cores {
		postfixes = append(postfixes, "-"+core, "_"+core)
		if strings.HasSuffix(core, "bit") {
			prefix := strings.TrimSuffix(core, "bit")
			for _, lead := range []string{"-", "_"} {
				for _, sep := range []string{"-", "_"} {
					postfixes = append(postfixes, lead+prefix+sep+"bit")
				}
			}
		}
	}
	return postfixes
}

// modelIDVariantPostfixPattern matches the numeric-parameterized variant
// families a host may append as a trailing "-"/"_"-separated token:
//
//	mtp          — multi-token-prediction head variant of the base weights
//	oq<N>e       — ordinal quantization encodings ("oq4e", "oq8e")
//	fp<N>[b|bit] — floating-point widths ("fp8", "fp16", "fp8b", "fp8bit")
//	q<N>[b|bit]  — integer quantization widths ("q4", "q8", "q4b", "q4bit")
//	bf<N>[b|bit] — bfloat widths ("bf16", "bf16b", "bf16bit")
//
// These families ARE stripped even though some also appear inside canonical
// catalog keys ("zai-org/glm-5.2-fp8"): exact-tier lookups still reach those
// entries verbatim, while the fuzzy index deliberately collapses a postfixed
// spelling onto its base (see buildNormalizedIndex) — which is what lets a
// host's "Qwen3.8-27B-oQ4e-mtp" resolve to the catalog's "qwen/qwen3.8-27b".
//
// The input token is already lowercased, so the pattern is lowercase-only.
// Anchored: the WHOLE trailing token must be the variant, so "fp8x"/"q4k"
// never match, and a parameter count ("8b", "27b") has no family prefix.
var modelIDVariantPostfixPattern = regexp.MustCompile(`^(?:mtp|oq\d+e|(?:fp|q|bf)\d+(?:bit|b)?)$`)

// stripModelIDPostfixes removes recognized model postfixes from the end of a
// lowercased, still-separated identifier, repeatedly, so chains of postfixes
// collapse to the base name ("qwen3-8bit-gguf" -> "qwen3-8bit" -> "qwen3",
// "qwen3.8-27b-oq4e-mtp" -> "qwen3.8-27b-oq4e" -> "qwen3.8-27b"). Each literal
// token includes its leading separator, so a suffix that is a parameter count
// plus an instruct marker ("qwen3-8b-it") is left intact. At each pass it
// strips the longest matching literal suffix, which keeps a shorter postfix
// that is a suffix of a longer one ("2bit" inside "32bit") from winning
// first; when no literal matches, the trailing "-"/"_"-delimited token is
// tested against the variant pattern (modelIDVariantPostfixPattern).
// Requiring the separator keeps jammed spellings ("qwen3fp8") intact. Returns
// the input unchanged when no postfix matches.
func stripModelIDPostfixes(id string) string {
	for {
		var match string
		for _, p := range modelIDPostfixes {
			if len(p) > len(match) && strings.HasSuffix(id, p) {
				match = p
			}
		}
		if match != "" {
			id = strings.TrimSuffix(id, match)
			continue
		}
		if i := strings.LastIndexAny(id, "-_"); i >= 0 && modelIDVariantPostfixPattern.MatchString(id[i+1:]) {
			id = id[:i]
			continue
		}
		return id
	}
}

// normalizeModelID normalizes a model identifier for the fuzzy lookup. It
// strips the vendor/org prefix (everything up to and including the first "/"),
// lowercases the result, drops an opaque "@"-tagged qualifier together with
// the whole remainder of the name, strips recognized delivery/quantization
// postfixes ("-gguf", "-mlx", "-8bit"/"-8-bit", the variant families "-mtp"/
// "-oq4e"/"-fp8"/"-q4"/"-bf16", and chains of them) from the end, then removes
// separator punctuation (".", "-", "_").
//
// Postfixes are stripped BEFORE the separators are dropped so an instruct
// suffix is never mistaken for a quantization: "qwen3-8b-it" ends in "-it" (not
// a postfix) and survives intact, while "qwen3-8bit", "qwen3-8-bit-gguf", and
// "qwen3.8-27b-oq4e-mtp" collapse to "qwen3" and "qwen3827b" respectively.
//
// The vendor prefix is a routing decoration, not a property of the model: the
// HuggingFace checkpoint "zai-org/glm-5.2-fp8" and the Z.ai API model
// "GLM-5.2-FP8" share the same metadata. Discarding the prefix before
// comparison lets a bare query match a prefixed registry key (and vice versa),
// so vendor spelling never defeats the match. Postfixes are likewise a
// delivery/quantization decoration: "qwen3-8bit" and "qwen3-8-bit-gguf" are the
// same model as "qwen3". An "@" tag is an opaque variant/digest qualifier
// ("qwen3@q4_k_m", "qwen3@sha256:..."): everything from "@" onward is a
// delivery detail, never part of the model identity.
//
// Alphanumerics are otherwise never altered, so distinct model versions stay
// distinct: "qwen3.6" and "qwen3.7" normalize to "qwen36" and "qwen37". This is
// the foundation of the fuzzy lookup and is deliberately conservative — unlike
// edit distance it can never silently remap one model version onto another.
func normalizeModelID(id string) string {
	s := strings.ToLower(BareModel(id))
	if i := strings.IndexByte(s, '@'); i >= 0 {
		s = s[:i]
	}
	return modelIDSeparators.Replace(stripModelIDPostfixes(s))
}

// fuzzyLookup performs a vendor-prefix- and separator-insensitive search across
// user overrides, observed runtime entries, and the built-in registry — in
// that precedence, mirroring the exact-lookup tiers — used when the exact
// (case-insensitive) lookup misses. It bridges the cosmetic naming differences
// real-world model hosts introduce — e.g. a registry key "qwen/qwen3.6-35b-a3b-fp8"
// served by a host as "Qwen/Qwen36-35B-A3B-FP8" (dot dropped) collapses to the
// same normalized form, and a prefixed key "zai-org/glm-5.2-fp8" matches a bare
// query "GLM-5.2-FP8" (vendor prefix discarded). Without the runtime leg, a
// probe that observed a self-hosted server's real window under one spelling
// would be invisible to a later resolve under a drifted spelling of the same
// id, and the catalog maximum would silently win. Returns false when nothing
// matches, so callers can fall through to network sources and the final default.
//
// Lookups are O(1) map reads against the normalized-ID indexes
// (r.overridesIndex, r.builtInIndex, r.runtimeIndex). The first two are
// immutable after construction (no lock); r.runtimeIndex is rebuilt under the
// write lock on every runtime write and read under RLock — the RLock is
// released before any enrichment baseline runs, so no recursive locking.
//
// A partial runtime entry found here enriches against the tiers strictly
// below runtime (resolveSpecOrCache), exactly like the exact tier-1.5 path,
// so a probe that observed only the context window still surfaces the
// catalog output limit under a drifted spelling.
func (r *ModelRegistry) fuzzyLookup(model string) (ModelMetadata, bool) {
	want := normalizeModelID(model)
	if want == "" {
		return ModelMetadata{}, false
	}
	if meta, ok := r.overridesIndex[want]; ok {
		// A partial override hit only via the fuzzy/normalized key must
		// inherit its unset scalar fields exactly like the exact-tier hit
		// (see ResolveLocal), otherwise a protocol-only override surfaced
		// under a drifted spelling collapses ContextWindow/OutputLimit to zero.
		key := strings.ToLower(model)
		meta = r.enrichPartialOverride(model, key, meta)
		return meta, true
	}
	if meta, ok := r.runtimeFuzzyLookup(want); ok {
		key := strings.ToLower(model)
		meta = r.enrichPartialWith(meta, func() ModelMetadata {
			return r.resolveSpecOrCache(model, key)
		})
		return meta, true
	}
	meta, ok := r.builtInIndex[want]
	return meta, ok
}

// runtimeFuzzyLookup reads one normalized-ID entry from the observed runtime
// index under RLock. The lock is held only for the map read; enrichment of a
// partial hit happens in the caller, after release, so its baseline
// (resolveSpecOrCache) can take the same RWMutex without recursive locking.
func (r *ModelRegistry) runtimeFuzzyLookup(want string) (ModelMetadata, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	meta, ok := r.runtimeIndex[want]
	return meta, ok
}

// buildNormalizedIndex returns a lookup table mapping each entry's normalized
// identifier (see normalizeModelID) to its metadata. When two keys collapse to
// the same normalized form — which postfix stripping now makes deliberate, e.g.
// a quantized variant "qwen3-gguf" and its base "qwen3" both normalize to
// "qwen3" — the key with the lexicographically smallest LOWERCASED spelling
// wins, preserving the deterministic tie-break previously implemented
// imperatively by the per-call scan. The comparison is case-insensitive so the
// documented survivor invariant holds across casings: a host's "Qwen3-GGUF"
// override must not beat its base "qwen3" merely because uppercase sorts
// before lowercase. Because the base name is a strict prefix of any postfixed
// spelling, it sorts first and is the survivor: the variant's entry is dropped
// from the fuzzy index (its exact-tier entry is unaffected). Two spellings
// that differ only in case tie on their lowercased form; the raw key then
// breaks the tie so the winner stays deterministic regardless of map
// iteration order. Building the index once — at registry construction for
// overrides, lazily for the shared built-in catalog — turns every fuzzy lookup
// from an O(n) scan with per-key normalization into an O(1) map read.
func buildNormalizedIndex(m map[string]ModelMetadata) map[string]ModelMetadata {
	index := make(map[string]ModelMetadata, len(m))
	winners := make(map[string]string, len(m))
	for k, v := range m {
		norm := normalizeModelID(k)
		if norm == "" {
			continue
		}
		if winner, ok := winners[norm]; !ok {
			winners[norm] = k
			index[norm] = v
		} else {
			lk, lw := strings.ToLower(k), strings.ToLower(winner)
			if lk < lw || (lk == lw && k < winner) {
				winners[norm] = k
				index[norm] = v
			}
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
func defaultUnknownCapabilities() *ModelCapabilities {
	return &ModelCapabilities{Attachment: true}
}

// cloneCapabilities returns a defensive copy of caps. A nil input yields a
// nil output: cloning never invents capability data, it only severs pointer
// aliasing. Callers that hand ModelMetadata across an API boundary — a public
// resolver returning registry state, or a write method storing caller-supplied
// state — route the pointer through this copy so neither side can mutate the
// other's view (or race a concurrent reader on the other side).
func cloneCapabilities(caps *ModelCapabilities) *ModelCapabilities {
	if caps == nil {
		return nil
	}
	copied := *caps
	return &copied
}

// finalizeMeta performs the last normalization step shared by every public
// resolver return path (Resolve, ResolveLocal, ResolveBuiltInModel): it
// derives Family/Protocol from the model ID when the record left them empty,
// and it guarantees the documented Capabilities contract — non-nil AND
// privately owned by the caller:
//
//   - nil Capabilities (no tier declared capabilities, e.g. a partial override
//     enriched against a partial cache entry) falls back to the optimistic
//     unknown set, mirroring the tier-5 fallback rationale: better to surface
//     a runtime provider error than silently deny image uploads for a model
//     the registry knows nothing about.
//   - non-nil Capabilities is cloned so the returned metadata never aliases
//     registry state (built-in catalog entries are process-global; cache and
//     runtime entries are shared across concurrent readers). Callers may
//     mutate the returned struct freely.
//
// A non-nil value is otherwise returned as AUTHORITATIVE — an all-false set
// stays all-false; the nil guard never overrides a declared value.
func finalizeMeta(model string, meta ModelMetadata) ModelMetadata {
	meta.Family = resolveFamily(model, meta)
	meta.Protocol = resolveProtocol(model, meta)
	if meta.Capabilities == nil {
		meta.Capabilities = defaultUnknownCapabilities()
		return meta
	}
	meta.Capabilities = cloneCapabilities(meta.Capabilities)
	return meta
}

// cacheResolved stores a resolved metadata entry in the lazy cache (tier 3)
// under its lowercased key so repeat lookups take the fast path. The stored
// entry owns a private copy of Capabilities: the value about to be handed to
// the resolver's caller is free to mutate its Capabilities, and the cached
// twin must not observe (or race concurrent readers on) those mutations.
func (r *ModelRegistry) cacheResolved(key string, meta ModelMetadata) {
	cached := meta
	cached.Capabilities = cloneCapabilities(meta.Capabilities)
	r.mu.Lock()
	r.cache[key] = cached
	r.mu.Unlock()
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
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, ToolCall: true},
		},
		"gpt-5.6-sol": {
			ContextWindow: 1050000,
			OutputLimit:   128000,
			TokenizerType: "tiktoken/o200k_base",
			Family:        "openai_flagship",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, ToolCall: true},
		},
		"gpt-5.6-terra": {
			ContextWindow: 1050000,
			OutputLimit:   128000,
			TokenizerType: "tiktoken/o200k_base",
			Family:        "openai_flagship",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, ToolCall: true},
		},
		"gpt-5.6-luna": {
			ContextWindow: 1050000,
			OutputLimit:   128000,
			TokenizerType: "tiktoken/o200k_base",
			Family:        "openai_flagship",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, ToolCall: true},
		},
		"gpt-5.5": {
			ContextWindow: 1050000,
			OutputLimit:   128000,
			TokenizerType: "tiktoken/o200k_base",
			Family:        "openai_flagship",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, ToolCall: true},
		},
		"gpt-5.4": {
			ContextWindow: 1050000,
			OutputLimit:   128000,
			TokenizerType: "tiktoken/o200k_base",
			Family:        "openai_flagship",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, ToolCall: true},
		},
		"gpt-5.4-mini": {
			ContextWindow: 400000,
			OutputLimit:   128000,
			TokenizerType: "tiktoken/o200k_base",
			Family:        "openai_flagship",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, ToolCall: true},
		},
		"gpt-5.4-nano": {
			ContextWindow: 400000,
			OutputLimit:   128000,
			TokenizerType: "tiktoken/o200k_base",
			Family:        "openai_flagship",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, ToolCall: true},
		},
		"gpt-5": {
			ContextWindow: 400000,
			OutputLimit:   128000,
			TokenizerType: "tiktoken/o200k_base",
			Family:        "openai_flagship",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, ToolCall: true},
		},
		"gpt-4.1": {
			ContextWindow: 1047576,
			OutputLimit:   32768,
			TokenizerType: "tiktoken/o200k_base",
			Family:        "openai_standard",
			Capabilities:  &ModelCapabilities{Attachment: true, Temperature: true, ToolCall: true},
		},
		"gpt-4.1-mini": {
			ContextWindow: 1047576,
			OutputLimit:   32768,
			TokenizerType: "tiktoken/o200k_base",
			Family:        "openai_standard",
			Capabilities:  &ModelCapabilities{Attachment: true, Temperature: true, ToolCall: true},
		},
		"gpt-4.1-nano": {
			ContextWindow: 1047576,
			OutputLimit:   32768,
			TokenizerType: "tiktoken/o200k_base",
			Family:        "openai_standard",
			Capabilities:  &ModelCapabilities{Attachment: true, Temperature: true, ToolCall: true},
		},
		"o4-mini": {
			ContextWindow: 200000,
			OutputLimit:   100000,
			TokenizerType: "tiktoken/o200k_base",
			Family:        "openai_flagship",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, ToolCall: true},
		},
		"o3": {
			ContextWindow: 200000,
			OutputLimit:   100000,
			TokenizerType: "tiktoken/o200k_base",
			Family:        "openai_flagship",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, ToolCall: true},
		},
		"o3-mini": {
			ContextWindow: 200000,
			OutputLimit:   100000,
			TokenizerType: "tiktoken/o200k_base",
			Family:        "openai_flagship",
			Capabilities:  &ModelCapabilities{Reasoning: true, ToolCall: true},
		},
		"o1": {
			ContextWindow: 200000,
			OutputLimit:   100000,
			TokenizerType: "tiktoken/o200k_base",
			Family:        "openai_flagship",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, ToolCall: true},
		},
		"o1-mini": {
			ContextWindow: 128000,
			OutputLimit:   65536,
			TokenizerType: "tiktoken/o200k_base",
			Family:        "openai_flagship",
			Capabilities:  &ModelCapabilities{Reasoning: true, ToolCall: true},
		},
		"gpt-4o": {
			ContextWindow: 128000,
			OutputLimit:   16384,
			TokenizerType: "tiktoken/o200k_base",
			Family:        "openai_flagship",
			Capabilities:  &ModelCapabilities{Attachment: true, Temperature: true, ToolCall: true},
		},
		"gpt-4o-mini": {
			ContextWindow: 128000,
			OutputLimit:   16384,
			TokenizerType: "tiktoken/o200k_base",
			Family:        "openai_flagship",
			Capabilities:  &ModelCapabilities{Attachment: true, Temperature: true, ToolCall: true},
		},
		// Codex models — use the OpenAI Responses API (/v1/responses).
		"codex-mini-latest": {
			ContextWindow: 200000,
			OutputLimit:   100000,
			TokenizerType: "tiktoken/o200k_base",
			Family:        "openai_codex",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, ToolCall: true},
		},
		"gpt-5.3-codex": {
			ContextWindow: 400000,
			OutputLimit:   128000,
			TokenizerType: "tiktoken/o200k_base",
			Family:        "openai_codex",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, ToolCall: true},
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
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, ToolCall: true},
		},
		"claude-opus-5": {
			ContextWindow: 1000000,
			OutputLimit:   128000,
			TokenizerType: "anthropic-api",
			Family:        "anthropic",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, ToolCall: true},
		},
		"claude-sonnet-5": {
			ContextWindow: 1000000,
			OutputLimit:   128000,
			TokenizerType: "anthropic-api",
			Family:        "anthropic",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, ToolCall: true},
		},
		"claude-opus-4-8": {
			ContextWindow: 1000000,
			OutputLimit:   128000,
			TokenizerType: "anthropic-api",
			Family:        "anthropic",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, ToolCall: true},
		},
		"claude-opus-4-7": {
			ContextWindow: 1000000,
			OutputLimit:   128000,
			TokenizerType: "anthropic-api",
			Family:        "anthropic",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, ToolCall: true},
		},
		"claude-opus-4-6": {
			ContextWindow: 1000000,
			OutputLimit:   128000,
			TokenizerType: "anthropic-api",
			Family:        "anthropic",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"claude-sonnet-4-6": {
			ContextWindow: 1000000,
			OutputLimit:   128000,
			TokenizerType: "anthropic-api",
			Family:        "anthropic",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"claude-haiku-4-5": {
			ContextWindow: 200000,
			OutputLimit:   64000,
			TokenizerType: "anthropic-api",
			Family:        "anthropic",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"claude-sonnet-4-5": {
			ContextWindow: 200000,
			OutputLimit:   64000,
			TokenizerType: "anthropic-api",
			Family:        "anthropic",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"claude-opus-4-5": {
			ContextWindow: 200000,
			OutputLimit:   64000,
			TokenizerType: "anthropic-api",
			Family:        "anthropic",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"claude-sonnet-4": {
			ContextWindow: 200000,
			OutputLimit:   64000,
			TokenizerType: "anthropic-api",
			Family:        "anthropic",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"claude-opus-4": {
			ContextWindow: 200000,
			OutputLimit:   32000,
			TokenizerType: "anthropic-api",
			Family:        "anthropic",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"claude-3.5-sonnet": {
			ContextWindow: 200000,
			OutputLimit:   8192,
			TokenizerType: "anthropic-api",
			Family:        "anthropic",
			Capabilities:  &ModelCapabilities{Attachment: true, Temperature: true, ToolCall: true},
		},
		"claude-3.5-haiku": {
			ContextWindow: 200000,
			OutputLimit:   8192,
			TokenizerType: "anthropic-api",
			Family:        "anthropic",
			Capabilities:  &ModelCapabilities{Attachment: true, Temperature: true, ToolCall: true},
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
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"gemini-3.5-flash-lite": {
			ContextWindow: 1048576,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "google",
			Capabilities:  &ModelCapabilities{Attachment: true, Temperature: true, ToolCall: true},
		},
		"gemini-3.1-pro": {
			ContextWindow: 1048576,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "google",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"gemini-3.1-flash-lite": {
			ContextWindow: 1048576,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "google",
			Capabilities:  &ModelCapabilities{Attachment: true, Temperature: true, ToolCall: true},
		},
		"gemini-3-flash": {
			ContextWindow: 1048576,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "google",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"gemini-2.5-pro": {
			ContextWindow: 1048576,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "google",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"gemini-2.5-flash": {
			ContextWindow: 1048576,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "google",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"gemini-2.5-flash-lite": {
			ContextWindow: 1048576,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "google",
			Capabilities:  &ModelCapabilities{Attachment: true, Temperature: true, ToolCall: true},
		},
		"gemini-2.0-flash": {
			ContextWindow: 1048576,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "google",
			Capabilities:  &ModelCapabilities{Attachment: true, Temperature: true, ToolCall: true},
		},
		// Google Gemma 4 — open-weights (HF: google/gemma-4-31B-it).
		// Multimodal (text+image), configurable thinking, native function
		// calling; the 31B dense variant has a 256K context window.
		"gemma-4-31b-it": {
			ContextWindow: 256000,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "google",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},

		// ── DeepSeek models ─────────────────────────────────────────────
		// Source: https://api-docs.deepseek.com/quick_start/pricing
		// V4: 1M context, max output 384K.
		"deepseek-v4-pro": {
			ContextWindow: 1000000,
			OutputLimit:   384000,
			TokenizerType: "approximate",
			Family:        "deepseek",
			Capabilities:  &ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		"deepseek-v4-flash": {
			ContextWindow: 1000000,
			OutputLimit:   384000,
			TokenizerType: "approximate",
			Family:        "deepseek",
			Capabilities:  &ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		// deepseek-v4-flash-vision-exp — experimental V4 Flash variant that
		// additionally accepts image input (JPEG/PNG/GIF/WebP) alongside the
		// same 1M context / 384K max output as V4 Flash. Images are billed
		// as input tokens, capped at 384 tokens per image after an automatic
		// downscale to roughly 800×800.
		// Source: https://api-docs.deepseek.com/guides/vision/
		"deepseek-v4-flash-vision-exp": {
			ContextWindow: 1000000,
			OutputLimit:   384000,
			TokenizerType: "approximate",
			Family:        "deepseek",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
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
		//
		// Sampling: K3 always has thinking mode enabled — both the platform
		// "kimi-k3" and the endpoint's "k3"/"k3-256k" (source:
		// https://platform.kimi.ai/docs/guide/kimi-k3-quickstart) — and is
		// tuned via the reasoning_effort field (low/high/max) instead of
		// sampling knobs; K2.7 Code likewise runs with thinking always on
		// (disabling thinking routes the request to K2.6). The Kimi Code
		// endpoint additionally enforces temperature=1, rejecting any other
		// value with HTTP 400 "invalid temperature: only 1 is allowed for
		// this model". The K3 and K2.7-code entries therefore carry no
		// Temperature capability (same shape as kimi-k2-thinking and the
		// o-series): no sampling fields are injected for any purpose, and the
		// omitted parameter resolves to the server-managed value — the only
		// one these thinking-locked models accept.
		// Note: OutputLimit is a context-window reserve (subtracted from the
		// window), so we store a practical reserve, not the absolute ceiling.
		// kimi-k3: 131K default (configurable up to 1M).
		// kimi-k2.7-code / kimi-k2.6 / kimi-k2.5: 256K max output; reserve 65K
		// (the 256K ceiling would zero the input budget under EffectiveMax).
		// kimi-k3 is the platform-style ID of the same always-thinking K3
		// served as "k3"/"k3-256k" on the Kimi Code endpoint — see the
		// sampling note in the section comment above.
		"kimi-k3": {
			ContextWindow: 1000000,
			OutputLimit:   131072,
			TokenizerType: "approximate",
			Family:        "kimi",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, ToolCall: true},
		},
		// Kimi Code endpoint aliases (see the section comment above).
		"k3": {
			ContextWindow: 1000000,
			OutputLimit:   131072,
			TokenizerType: "approximate",
			Family:        "kimi",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, ToolCall: true},
		},
		"k3-256k": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "kimi",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, ToolCall: true},
		},
		"kimi-k2.7-code": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "kimi",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, ToolCall: true},
		},
		"kimi-k2.7-code-highspeed": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "kimi",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, ToolCall: true},
		},
		"kimi-for-coding": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "kimi",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, ToolCall: true},
		},
		"kimi-for-coding-highspeed": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "kimi",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, ToolCall: true},
		},
		"kimi-k2.6": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "kimi",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"kimi-k2.5": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "kimi",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		// kimi-k2 series deprecated May 25 2026; kept for backward compat.
		"kimi-k2": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "kimi",
			Capabilities:  &ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"kimi-k2-thinking": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "kimi",
			Capabilities:  &ModelCapabilities{Reasoning: true, ToolCall: true},
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
			Capabilities:  &ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		"qwen3.7-plus": {
			ContextWindow: 1000000,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  &ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		"qwen3.6-flash": {
			ContextWindow: 1000000,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  &ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		"qwen3-coder-plus": {
			ContextWindow: 1000000,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  &ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		// Listed under "Legacy Qwen" in current docs.
		"qwen-plus": {
			ContextWindow: 1000000,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  &ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		// qwen-max does NOT support thinking mode per Alibaba's model table.
		"qwen-max": {
			ContextWindow: 128000,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  &ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"qwq-plus": {
			ContextWindow: 128000,
			OutputLimit:   16384,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  &ModelCapabilities{Reasoning: true, ToolCall: true},
		},

		// ── GLM models (Zhipu AI) ───────────────────────────────────────
		// Source: https://docs.z.ai + https://docs.bigmodel.cn
		// GLM 5.x: 1M/200K context, 128K max output.
		// GLM 4.7: 200K context, 128K max output.
		"glm-5.3": {
			ContextWindow: 1000000,
			OutputLimit:   128000,
			TokenizerType: "approximate",
			Family:        "glm",
			Capabilities:  &ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		"glm-5.2": {
			ContextWindow: 1000000,
			OutputLimit:   128000,
			TokenizerType: "approximate",
			Family:        "glm",
			Capabilities:  &ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		"glm-5.1": {
			ContextWindow: 200000,
			OutputLimit:   128000,
			TokenizerType: "approximate",
			Family:        "glm",
			Capabilities:  &ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		"glm-5": {
			ContextWindow: 200000,
			OutputLimit:   128000,
			TokenizerType: "approximate",
			Family:        "glm",
			Capabilities:  &ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		"glm-4.7": {
			ContextWindow: 200000,
			OutputLimit:   128000,
			TokenizerType: "approximate",
			Family:        "glm",
			Capabilities:  &ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		// glm-z1-32b deprecated Nov 2025; kept for backward compat.
		"glm-z1-32b": {
			ContextWindow: 128000,
			OutputLimit:   16384,
			TokenizerType: "approximate",
			Family:        "glm",
			Capabilities:  &ModelCapabilities{Reasoning: true, ToolCall: true},
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
			Capabilities:  &ModelCapabilities{Attachment: true, Temperature: true, ToolCall: true},
		},
		"grok-4.3": {
			ContextWindow: 1000000,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  &ModelCapabilities{Attachment: true, Temperature: true, ToolCall: true},
		},
		"grok-4.20": {
			ContextWindow: 1000000,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  &ModelCapabilities{Attachment: true, Temperature: true, ToolCall: true},
		},
		"grok-4": {
			ContextWindow: 256000,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  &ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"grok-3": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  &ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"grok-3-mini": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  &ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
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
			Capabilities:  &ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"qwen/qwen3-4b-thinking-2507": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  &ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		"qwen/qwen3-30b-a3b-2507": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  &ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"qwen/qwen3-30b-a3b-thinking-2507": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  &ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		"qwen/qwen3-235b-a22b-2507": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  &ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"qwen/qwen3-235b-a22b-thinking-2507": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  &ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		// Qwen3 (1st generation, 2504): hybrid thinking, 128K context.
		"qwen/qwen3-4b": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  &ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		"qwen/qwen3-8b": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  &ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		"qwen/qwen3-14b": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  &ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		"qwen/qwen3-32b": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  &ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		"qwen/qwen3-30b-a3b": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  &ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		"qwen/qwen3-235b-a22b": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  &ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		// Qwen3.5: multimodal (text+image), thinking, 256K context.
		"qwen/qwen3.5-2b": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"qwen/qwen3.5-4b": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"qwen/qwen3.5-9b": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"qwen/qwen3.5-27b": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"qwen/qwen3.5-35b-a3b": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
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
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"qwen/qwen3.5-122b-a10b-fp8": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		// Qwen3.6: multimodal, thinking, 256K context.
		"qwen/qwen3.6-27b": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"qwen/qwen3.6-35b-a3b": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		// Qwen3.6 FP8 (HuggingFace): FP8 checkpoint of the smaller Qwen3.6
		// flagship — same multimodal MoE architecture and 256K context.
		//   config: max_position_embeddings=262144 (Qwen3_5MoeForConditionalGeneration)
		"qwen/qwen3.6-35b-a3b-fp8": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		// Qwen3.8: dense multimodal (text+image/video), thinking on by default
		// (disabling per request, depth via reasoning_effort), 256K context,
		// extensible to 1M on Qwen Cloud.
		//   config: max_position_embeddings=262144 (Qwen3_5ForConditionalGeneration)
		"qwen/qwen3.8-27b": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		// Qwen3.8-2.4T-A95B: the open flagship — first Qwen-Max-class open
		// release. TEXT-ONLY MoE (no vision tower), unlike the smaller dense
		// 3.8 members. Thinking is mandatory (cannot be disabled); depth via
		// reasoning_effort (xhigh default), history retained via
		// preserve_thinking. 256K native context, extensible to ~1M.
		//   config: max_position_embeddings=262144 (Qwen3_5MoeForCausalLM)
		"qwen/qwen3.8-2.4t-a95b": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  &ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		// Qwen3.8 FP8 (HuggingFace): official FP8 checkpoints of the 3.8
		// lineup — same architectures, thinking, tool support, and 256K
		// context as their base models.
		//   config: max_position_embeddings=262144
		//     2.4T — Qwen3_5MoeForCausalLM (text-only MoE)
		//     27B  — Qwen3_5ForConditionalGeneration (multimodal)
		"qwen/qwen3.8-2.4t-a95b-fp8": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  &ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		"qwen/qwen3.8-27b-fp8": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		// Qwen3.8-Flash-Next (HuggingFace): experimental preview of the
		// Qwen4 architecture — Gated DeltaNet + Qwen Sparse Attention,
		// Gated Residual, and N-gram embedding (125B total / 6B active,
		// plus 51B n-gram embedding params). Multimodal (text+image/video
		// via the vision encoder), thinking on by default, 256K native
		// context extensible to 1M; the Qwen Cloud "Qwen3.8-Flash" API
		// serves this model with 1M context by default.
		//   config: max_position_embeddings=262144 (Qwen4ExpForConditionalGeneration)
		//   source: https://huggingface.co/Qwen/Qwen3.8-Flash-Next
		"qwen/qwen3.8-flash-next": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		// Qwen3.8-Flash: the Qwen Cloud serving name of the Flash-Next
		// preview checkpoint above. The API exposes it with a 1M context
		// window by default (see the Flash-Next entry), so resolving the
		// documented serving name must land here instead of falling back
		// to the generic 128K/32K fallback defaults.
		"qwen/qwen3.8-flash": {
			ContextWindow: 1048576,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		// Qwen3-Coder: non-thinking coding MoE, 256K context.
		"qwen/qwen3-coder-30b": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  &ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"qwen/qwen3-coder-480b": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  &ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"qwen/qwen3-coder-next": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  &ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"qwen/qwen3-next-80b": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  &ModelCapabilities{Temperature: true, ToolCall: true},
		},
		// Qwen3-VL: vision-language, thinking, 256K context.
		"qwen/qwen3-vl-2b": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"qwen/qwen3-vl-4b": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"qwen/qwen3-vl-8b": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"qwen/qwen3-vl-30b": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"qwen/qwen3-vl-32b": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		// Qwen2.5-VL: vision-language, 128K context.
		"qwen/qwen2.5-vl-3b": {
			ContextWindow: 128000,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  &ModelCapabilities{Attachment: true, Temperature: true, ToolCall: true},
		},
		"qwen/qwen2.5-vl-7b": {
			ContextWindow: 128000,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  &ModelCapabilities{Attachment: true, Temperature: true, ToolCall: true},
		},
		"qwen/qwen2.5-vl-32b": {
			ContextWindow: 128000,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  &ModelCapabilities{Attachment: true, Temperature: true, ToolCall: true},
		},
		"qwen/qwen2.5-vl-72b": {
			ContextWindow: 128000,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "qwen",
			Capabilities:  &ModelCapabilities{Attachment: true, Temperature: true, ToolCall: true},
		},

		// ── DeepSeek (open reasoning distills) ─────────────────────────
		// R1 / distills: reasoning-only, not trained for tool calling.
		"deepseek/deepseek-r1-0528-qwen3-8b": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "deepseek",
			Capabilities:  &ModelCapabilities{Reasoning: true, Temperature: true},
		},
		"deepseek/deepseek-r1-distill-qwen-7b": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "deepseek",
			Capabilities:  &ModelCapabilities{Reasoning: true, Temperature: true},
		},
		"deepseek/deepseek-r1-distill-llama-8b": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "deepseek",
			Capabilities:  &ModelCapabilities{Reasoning: true, Temperature: true},
		},
		"deepseek/deepseek-r1-distill-qwen-14b": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "deepseek",
			Capabilities:  &ModelCapabilities{Reasoning: true, Temperature: true},
		},
		"deepseek/deepseek-r1-distill-qwen-32b": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "deepseek",
			Capabilities:  &ModelCapabilities{Reasoning: true, Temperature: true},
		},
		"deepseek/deepseek-r1-distill-llama-70b": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "deepseek",
			Capabilities:  &ModelCapabilities{Reasoning: true, Temperature: true},
		},

		// ── GLM (Z.ai open coding) ─────────────────────────────────────
		"zai-org/glm-4.7-flash": {
			ContextWindow: 202752,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "glm",
			Capabilities:  &ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		"zai-org/glm-4.6v-flash": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "glm",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
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
			Capabilities:  &ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		// GLM-5.3-Flash (HuggingFace): first natively multimodal GLM-5
		// model — 320B total / 18B active hybrid sparse+linear attention
		// MoE with an always-on thinking mode (cannot be disabled). Text
		// parameters are consistent with GLM-5.3 (1M context, 128K max
		// output per https://docs.z.ai/guides/llm/glm-5.3-flash); native
		// image/video input; function calling.
		//   config: max_position_embeddings=1048576 (Glm5NextForConditionalGeneration)
		//   source: https://huggingface.co/zai-org/GLM-5.3-Flash
		"zai-org/glm-5.3-flash": {
			ContextWindow: 1048576,
			OutputLimit:   128000,
			TokenizerType: "approximate",
			Family:        "glm",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},

		// ── Google Gemma (open-weights) ────────────────────────────────
		// Gemma 4: multimodal (text+image; +audio on E2B/E4B/12B), thinking,
		// native function calling. E2B/E4B = 128K; 12B/26B-A4B/31B = 256K.
		"google/gemma-4-e2b": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "google",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"google/gemma-4-e4b": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "google",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"google/gemma-4-12b": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "google",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"google/gemma-4-26b-a4b": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "google",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"google/gemma-4-31b": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "google",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
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
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"google/gemma-4-26b-a4b-qat": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "google",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"google/gemma-4-31b-qat": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "google",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
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
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		// Gemma 3: image+text (≥4B), 128K; 270M/1B text-only, 32K.
		"google/gemma-3-270m": {
			ContextWindow: 32768,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "google",
			Capabilities:  &ModelCapabilities{Temperature: true},
		},
		"google/gemma-3-1b": {
			ContextWindow: 32768,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "google",
			Capabilities:  &ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"google/gemma-3-4b": {
			ContextWindow: 131072,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "google",
			Capabilities:  &ModelCapabilities{Attachment: true, Temperature: true, ToolCall: true},
		},
		"google/gemma-3-12b": {
			ContextWindow: 131072,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "google",
			Capabilities:  &ModelCapabilities{Attachment: true, Temperature: true, ToolCall: true},
		},
		"google/gemma-3-27b": {
			ContextWindow: 131072,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "google",
			Capabilities:  &ModelCapabilities{Attachment: true, Temperature: true, ToolCall: true},
		},
		// Gemma 3n: on-device multimodal, 32K.
		"google/gemma-3n-e2b": {
			ContextWindow: 32768,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "google",
			Capabilities:  &ModelCapabilities{Attachment: true, Temperature: true},
		},
		"google/gemma-3n-e4b": {
			ContextWindow: 32768,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "google",
			Capabilities:  &ModelCapabilities{Attachment: true, Temperature: true},
		},
		// FunctionGemma: tiny function-calling foundation, 32K.
		"google/functiongemma-270m": {
			ContextWindow: 32768,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "google",
			Capabilities:  &ModelCapabilities{Temperature: true, ToolCall: true},
		},

		// ── Mistral (open-weights) ─────────────────────────────────────
		"mistralai/mistral-7b-instruct-v0.3": {
			ContextWindow: 32768,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "mistral",
			Capabilities:  &ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"mistralai/mistral-nemo-instruct-2407": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "mistral",
			Capabilities:  &ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"mistralai/mistral-small-3.2": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "mistral",
			Capabilities:  &ModelCapabilities{Attachment: true, Temperature: true, ToolCall: true},
		},
		// Magistral: reasoning, 128K. v1.2 (+vision) and v1.1 (text).
		"mistralai/magistral-small-2509": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "mistral",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"mistralai/magistral-small": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "mistral",
			Capabilities:  &ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		// Devstral: agentic coding, 128K. Devstral 2 adds vision (123B = 256K).
		"mistralai/devstral-small-2507": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "mistral",
			Capabilities:  &ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"mistralai/devstral-small-2505": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "mistral",
			Capabilities:  &ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"mistralai/devstral-small-2-2512": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "mistral",
			Capabilities:  &ModelCapabilities{Attachment: true, Temperature: true, ToolCall: true},
		},
		"mistralai/devstral-2-2512": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "mistral",
			Capabilities:  &ModelCapabilities{Attachment: true, Temperature: true, ToolCall: true},
		},
		// Codestral: coding, 32K.
		"mistralai/codestral-22b-v0.1": {
			ContextWindow: 32768,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "mistral",
			Capabilities:  &ModelCapabilities{Temperature: true, ToolCall: true},
		},
		// Ministral 3: multimodal, 256K (base instruct variants).
		"mistralai/ministral-3-3b": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "mistral",
			Capabilities:  &ModelCapabilities{Attachment: true, Temperature: true, ToolCall: true},
		},
		"mistralai/ministral-3-8b": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "mistral",
			Capabilities:  &ModelCapabilities{Attachment: true, Temperature: true, ToolCall: true},
		},
		"mistralai/ministral-3-14b": {
			ContextWindow: 262144,
			OutputLimit:   65536,
			TokenizerType: "approximate",
			Family:        "mistral",
			Capabilities:  &ModelCapabilities{Attachment: true, Temperature: true, ToolCall: true},
		},

		// ── OpenAI gpt-oss (open-weights) ──────────────────────────────
		// gpt-oss: configurable reasoning effort + tool use, 128K.
		"openai/gpt-oss-20b": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  &ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		"openai/gpt-oss-120b": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  &ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		// gpt-oss-safeguard: content classifiers (not chat/agent models).
		"openai/gpt-oss-safeguard-20b": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  &ModelCapabilities{Temperature: true},
		},
		"openai/gpt-oss-safeguard-120b": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  &ModelCapabilities{Temperature: true},
		},

		// ── ByteDance / Baidu ──────────────────────────────────────────
		// seed-oss: reasoning with configurable thinking budget, 512K.
		"bytedance/seed-oss-36b": {
			ContextWindow: 524288,
			OutputLimit:   16384,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  &ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		// ERNIE 4.5: 21B MoE, 128K.
		"baidu/ernie-4.5-21b-a3b": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  &ModelCapabilities{Temperature: true, ToolCall: true},
		},

		// ── NVIDIA Nemotron ────────────────────────────────────────────
		// Nemotron 3 / Super: up to 1M context. Omni is multimodal.
		"nvidia/nemotron-3-nano": {
			ContextWindow: 1048576,
			OutputLimit:   16384,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  &ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		"nvidia/nemotron-3-nano-omni": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true, Temperature: true, ToolCall: true},
		},
		"nvidia/nemotron-3-super": {
			ContextWindow: 1048576,
			OutputLimit:   16384,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  &ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},

		// ── IBM Granite ────────────────────────────────────────────────
		// Granite 4.x: tool calling + JSON output, 128K.
		"ibm/granite-4-h-micro": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  &ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"ibm/granite-4-micro": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  &ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"ibm/granite-4-h-tiny": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  &ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"ibm/granite-4-h-small": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  &ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"ibm/granite-4.1-3b": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  &ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"ibm/granite-4.1-8b": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  &ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"ibm/granite-4.1-30b": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  &ModelCapabilities{Temperature: true, ToolCall: true},
		},

		// ── Liquid AI LFM2 (on-device hybrid) ──────────────────────────
		// Small variants: text, 32K, no tool use. 24B-A2B: native function calling.
		"liquid/lfm2-350m": {
			ContextWindow: 32768,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  &ModelCapabilities{Temperature: true},
		},
		"liquid/lfm2-700m": {
			ContextWindow: 32768,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  &ModelCapabilities{Temperature: true},
		},
		"liquid/lfm2-1.2b": {
			ContextWindow: 32768,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  &ModelCapabilities{Temperature: true},
		},
		"liquid/lfm2-24b-a2b": {
			ContextWindow: 32768,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  &ModelCapabilities{Temperature: true, ToolCall: true},
		},

		// ── Allen AI (Olmo / olmOCR) ───────────────────────────────────
		// Olmo 3: 64K. olmOCR 2: vision-language OCR, 128K, no tool use.
		"allenai/olmo-3-7b": {
			ContextWindow: 65536,
			OutputLimit:   16384,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  &ModelCapabilities{Temperature: true, ToolCall: true},
		},
		"allenai/olmo-3-7b-think": {
			ContextWindow: 65536,
			OutputLimit:   16384,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  &ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		"allenai/olmo-3-32b-think": {
			ContextWindow: 65536,
			OutputLimit:   16384,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  &ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		"allenai/olmocr-2-7b": {
			ContextWindow: 128000,
			OutputLimit:   16384,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  &ModelCapabilities{Attachment: true, Temperature: true},
		},

		// ── Essential AI / MiniMax ─────────────────────────────────────
		// Rnj-1: dense reasoning, 32K. MiniMax M2: 230B MoE, ~196K.
		"essentialai/rnj-1": {
			ContextWindow: 32768,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  &ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		"minimax/minimax-m2": {
			ContextWindow: 196608,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  &ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},

		// ── Microsoft Phi ──────────────────────────────────────────────
		// phi-4: 16K, no tool use. phi-4-mini: 128K. Reasoning variants: 32K/128K.
		"microsoft/phi-4": {
			ContextWindow: 16384,
			OutputLimit:   4096,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  &ModelCapabilities{Temperature: true},
		},
		"microsoft/phi-4-mini": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  &ModelCapabilities{Temperature: true},
		},
		"microsoft/phi-4-mini-reasoning": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  &ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		"microsoft/phi-4-reasoning": {
			ContextWindow: 32768,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  &ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
		},
		"microsoft/phi-4-reasoning-plus": {
			ContextWindow: 32768,
			OutputLimit:   8192,
			TokenizerType: "approximate",
			Family:        "default",
			Capabilities:  &ModelCapabilities{Reasoning: true, Temperature: true, ToolCall: true},
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
		return finalizeMeta(model, meta), true
	}
	if want := normalizeModelID(model); want != "" {
		if meta, ok := getBuiltInIndex()[want]; ok {
			return finalizeMeta(model, meta), true
		}
	}
	// Same tier-5 defaults as Resolve (fallbackMetadata); finalizeMeta derives
	// Family/Protocol and guarantees the non-nil, caller-owned Capabilities.
	return finalizeMeta(model, fallbackMetadata()), false
}
