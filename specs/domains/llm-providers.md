# LLM Providers

## Purpose

Provides LLM provider abstractions, a model registry, and routing for multi-provider inference. Configure one or more providers, switch models at runtime, count tokens, track usage, and recover from transient errors with automatic retry and backoff. At the center is the **Router**, which routes every chat request to the currently active (provider, model) pair.

## Key Files

- `github.com/v0lka/sp4rk/llm` — `Router`, `RouterConfig`, `NewRouter`, `SetModel`/`ActiveModel`/`Call`, `Provider` interface, `ProviderEntry`
- `github.com/v0lka/sp4rk/llm` (metadata) — `ModelRegistry`, `ModelMetadata`, `ModelCapabilities`, `DetectFamily`, `FamilyReasoningOptions`, `ResolveBuiltInModel`, `SetCachedMetadata`
- `github.com/v0lka/sp4rk/llm` (protocol) — `APIProtocol`, `DetectProtocol`, and the four protocol constants (`ProtocolChatCompletions`, `ProtocolResponses`, `ProtocolAnthropic`, `ProtocolGoogle`)
- `github.com/v0lka/sp4rk/llm` (token accounting) — `TokenCounter`, `SimpleTokenCounter`, `TiktokenCounter`, `NewTokenCounter`, `ContextTokenTracker`, `UsageTracker`, `TrackingCaller`
- `github.com/v0lka/sp4rk/llm` (request/response) — `ChatRequest`, `ChatResponse`, `Message`, `ContentBlock`, `NormalizeContentBlocks`, `ValidateContentBlocks`, `ToolCall`, `ToolDefinition`, `TokenUsage`
- `github.com/v0lka/sp4rk/llm` (errors) — `Error`, `WrapProviderError`, `IsRetryable`, `ErrContextWindowExceeded`

## Core Types

```go
type ProviderEntry struct {
    Name         string   // logical name ("anthropic", "openai_compatible", …)
    ProviderType string   // "openai" | "anthropic"
    APIKey       string   // pre-expanded key
    BaseURL      string   // pre-expanded base URL
    Models       []string // enabled model names
}

type ModelMetadata struct {
    ContextWindow int
    OutputLimit   int
    TokenizerType string  // "tiktoken/o200k_base", "anthropic-api", "approximate"
    Family        string
    Protocol      APIProtocol // "" ⇒ derived by DetectProtocol; set to force a protocol
    Capabilities  ModelCapabilities
}

type ModelCapabilities struct {
    Attachment, Reasoning, Temperature, ToolCall bool
}

// A message may carry an ordered list of blocks (text and/or images) instead of
// a plain Content string; providers render ContentBlocks when non-empty.
type ContentBlock struct {
    Type      string // "text" | "image"
    Text      string // text content (Type == "text")
    ImageB64  string // base64 image data, WITHOUT the "data:" prefix (Type == "image")
    MediaType string // MIME type, e.g. "image/png" (Type == "image")
}

type TokenCounter interface {
    Count(text string) int
    CountMessages(msgs []Message) int
}
```

## Flow

```
Router.Call(ctx, req)
│
├─ Snapshot active (provider, model) under a read lock; release before the retry loop
│      so SetModel is not blocked by backoff sleeps.
├─ Fill the bare model name when req.Model is empty; auto-detect family if empty.
├─ Resolve model metadata at most once (a single Resolve honoring a tier-1 override); when the
│      caller left req.Protocol empty, fill it from the resolved ModelMetadata.Protocol so a
│      registry override steers routing regardless of the model name (the documented escape hatch).
│      The resolved metadata is reused by applyDefaultTemperature and validateContextWindow so
│      those helpers perform no registry I/O of their own.
├─ Apply family-aware default temperature (via SamplingFunc) unless req.Temperature is set;
│      skip temperature entirely for models whose Capabilities.Temperature is false.
├─ Validate estimated tokens against the effective context window (SafetyMarginPercent +
│      OutputTokenReserve); reject oversized requests with ErrContextWindowExceeded.
├─ Dispatch to the active provider. The OpenAI provider honors req.Protocol when set and otherwise
│      falls back to DetectProtocol(req.Model) (see [Multi-protocol routing](#multi-protocol-routing)):
│      Responses API for gpt-5/codex, /messages (co-located AnthropicProvider) for Claude,
│      generateContent (googleCompletion) for Gemini/Gemma, /chat/completions for everything else.
└─ Retry transient errors (HTTP 408/429/500/502/503/504/520–524/529, transient
      network errors) with exponential backoff + ±20% jitter, up to MaxRetries.
```

## Model identification: composite vs bare IDs

A **composite model identifier** `"providerName/modelName"` is the internal selector that routes a request to a specific (provider, model) pair, disambiguating models that share a bare name across multiple OpenAI-compatible providers. The **bare model name** (part after the first `/`) is what is sent to the API and used for metadata lookups; identifiers are split on the **first** `/` only (model names may themselves contain `/`). Helpers: `CompositeModelID`, `ParseCompositeModelID`, `IsCompositeModelID`, `BareModel`.

### Runtime model switching

`ActiveModel()` returns the composite identifier; `ActiveProviderName()` the logical provider name; `SetModel(ctx, model)` accepts a composite ID (routes directly) or a bare name (resolves to the first matching provider — deterministic, sorted by composite ID — logging a warning on ambiguity). The Router is **safe for concurrent use** (`sync.RWMutex`); a Framework shares one Router across all Conductors.

## ModelRegistry

`NewModelRegistry(overrides)` is thread-safe and lazily fetches external metadata. All map lookups (overrides, built-ins, cache) are **case-insensitive** (`strings.ToLower`), so a model ID differing only in casing from its canonical registry key (common with OpenAI-compatible hosts such as vLLM) still resolves. `Resolve(ctx, model)` uses a 6-tier resolution (first match wins): (1) user overrides; (2) built-in registry of well-known models; (2b) **fuzzy match** — vendor-prefix- and separator-insensitive lookup across overrides then built-ins; (3) cache; (4) external sources — HuggingFace API lookup (lazy, cached), then sources registered via `RegisterSource` (e.g. an LM Studio provider); (5) fallback defaults (`ContextWindow: 128000`, `OutputLimit: 32768`, `ok == false`). `Invalidate(model)` clears a cached entry; `SetHTTPClient` replaces the HF lookup client. `DetectFamily(modelID)` determines a model's family from its ID string, driving prompt and parameter adaptation (families: `anthropic`, `openai_flagship`/`openai_standard`/`openai_codex`, `google`, `mistral`, `deepseek`, `qwen`, `glm`, `kimi`, `default`).

### Partial overrides

A tier-1 override that pins only some fields — a **partial** override, e.g. a protocol-only auto-remap that injects `{Protocol: ChatCompletions}` for a Google-named checkpoint served by LM Studio/vLLM — inherits its unset scalar fields from the lower **non-network** tiers (built-in exact → built-in fuzzy → cache → fallback defaults). The inheritable fields are `ContextWindow`, `OutputLimit`, `TokenizerType`, and `Capabilities` (a zero-valued `ModelCapabilities` is treated as "unset", so a minimal protocol-pinning override no longer silently disables tool-calling, reasoning, or image uploads). A field already set in the override is authoritative; a fully-specified override (the common case: a `config.yaml` entry whose scalars are all populated) is returned verbatim. `Family` is derived by `resolveFamily`, and `Protocol` is always authoritative — pinning the protocol is precisely what a partial override exists to do. Network tiers (HuggingFace, registered sources) are deliberately skipped so an override lookup never performs I/O. The accepted tradeoff — mirroring the scalars — is that the zero value is reserved for "inherit", so there is no way to express "explicitly disable every capability" via a partial override.

### Fuzzy lookup

The fuzzy tier (2b) bridges the cosmetic naming drift real-world model hosts introduce without the false-positive risk of edit-distance matching. `normalizeModelID` strips the org/vendor prefix (everything up to and including the first `/`) and removes separator punctuation (`.`, `-`, `_`), then lowercases; alphanumerics are never altered, so distinct model versions remain distinct (e.g. `qwen3.6` and `qwen3.7` normalize to `qwen36` and `qwen37`). Discarding the vendor prefix lets a bare query like `GLM-5.2-FP8` match a prefixed key such as `zai-org/glm-5.2-fp8`, and vice versa. When several entries normalize to the same form, the lexicographically smallest key wins (deterministic). A fuzzy hit is cached under the query key so repeat resolves take the fast path. Lookups are O(1) map reads against normalized-ID indexes (`builtInIndex`, `overridesIndex`) built once at registry construction — the shared built-in catalog index is initialized lazily via a `sync.Once` — replacing the former per-call O(n) scan.

### ResolveBuiltInModel / SetCachedMetadata

`ResolveBuiltInModel(model)` resolves metadata using **only** the built-in catalog (exact case-insensitive then fuzzy) with no network access and no consideration of overrides, cache, HuggingFace, or registered sources. It returns the fallback defaults with `ok == false` when the model is absent. It is intended for callers needing the "factory default" independent of a user override — e.g. to compare which fields of a config override differ, or to merge a partial override onto built-in metadata at registry-construction time. It never touches the network, so it is safe to call from startup paths.

`SetCachedMetadata(model, meta)` stores a late-learned metadata entry at tier 3 (the cache). It lets a caller that discovers a model's real context window after construction — e.g. a lazy probe of a local OpenAI-compatible server — populate the result so subsequent `Resolve` calls return it without re-querying the network. Because it is tier 3, it takes effect only when no user override (tier 1) or built-in entry (tier 2) exists, so config overrides and well-known models are never silently clobbered. A **partial** override that leaves its `ContextWindow` at zero inherits the cache-tier value once it arrives, so a lazy local-model probe (notably a HuggingFace probe for a custom-served model) is not shadowed by a non-zero fallback window baked into the override.

## Token counting & usage tracking

Two counters: `SimpleTokenCounter` (~4 chars = 1 token approximation) and `TiktokenCounter` (accurate, tiktoken-go, mutex-guarded). `NewTokenCounter(tokenizerType)` selects by metadata type (`tiktoken/*` → Tiktoken; `anthropic-api`/`approximate`/unknown → Simple; always returns a valid counter).

Both counters account for structured content blocks: when a message carries non-empty `ContentBlocks` (after `NormalizeContentBlocks`), text blocks are counted via the counter and image blocks are estimated at a per-image cost; unknown block types are skipped (matching provider rendering). The per-image estimate is provider-specific: the conservative default is **765 tokens** (OpenAI high-detail orientation), but `NewTokenCounter` overrides this to **85 tokens** for `anthropic-api` models so image-heavy Anthropic conversations are not over-counted ~9× and trigger premature context compaction.

`ContextTokenTracker` combines predictive counting with API-corrected actuals (`AddDelta`/`EstimateTotal`/`Correct(apiInputTokens)`/`Reset`). `UsageTracker` accumulates token usage across a session (thread-safe, observer callbacks). `TrackingCaller` wraps a `Caller` to record usage into a `UsageTracker` and correct a `ContextTokenTracker`; `WithContextTracker` returns a step-local caller sharing the same inner caller and session tracker — use it for parallel execution branches.

## Multi-protocol routing

`APIProtocol` (`llm.APIProtocol`) is the wire protocol a model speaks — the protocol-level analogue of `ModelFamily`. `DetectProtocol(modelID)` is a pure, substring-based detector (mirroring `DetectFamily`) with four outcomes:

| Protocol | Endpoint | Models |
| -------- | -------- | ------ |
| `ProtocolResponses` | `POST /responses` | GPT-5.x and Codex (both official OpenAI and compatible gateways) |
| `ProtocolAnthropic` | `POST /messages` | all Claude models |
| `ProtocolGoogle` | `POST /models/{model}:generateContent` | Gemini and Gemma |
| `ProtocolChatCompletions` | `POST /chat/completions` | everything else (default) |

A single OpenAI-compatible `ProviderEntry` (e.g. a multi-model gateway like Zen) dispatches to all four: the OpenAI provider's `Call` honors `req.Protocol` when set (the router fills it from the registry-resolved metadata, which honors an explicit override) and otherwise falls back to `DetectProtocol(req.Model)` — Responses to the Responses API, `ProtocolAnthropic` to a **co-located `AnthropicProvider`** built with the same `baseURL`/`apiKey`/`HTTPClient`/`logger` (`anthropicDelegate`), `ProtocolGoogle` to the `googleCompletion` function (POSTs the Google `contents`/`parts` format), and `ProtocolChatCompletions` to the Chat Completions endpoint. A 404/405 on `/responses` surfaces a clear error rather than silently falling back to Chat Completions. No new `ProviderType` is introduced — Google and Anthropic models behind an OpenAI-compatible host are reached by delegation, not by a new registry type.

The protocol cannot be derived from `ModelFamily` alone: `FamilyOpenAIFlagship` spans both protocols (gpt-4o/o-series use Chat Completions, gpt-5 uses Responses), so independent model-ID detection is required. Because detection is substring-based, a locally-served model whose name contains a family token but speaks a different protocol (e.g. a vLLM model named `gemini-finetune` served over `/chat/completions`) would be misrouted. For such models the caller sets `ModelMetadata.Protocol` (via the overrides map, a registered source, or a built-in entry): `resolveProtocol` honors an explicit `Protocol` and only falls back to `DetectProtocol` when it is unset. The router's `prepareRequest` threads the resolved protocol into `ChatRequest.Protocol`, and the OpenAI provider honors `req.Protocol` over its own `DetectProtocol` — so a tier-1 override takes effect even when the caller invokes the provider through the router. A partial (protocol-only) override now inherits the model's real `ContextWindow`/`OutputLimit`/`TokenizerType`/`Capabilities` from the lower tiers (see [Partial overrides](#partial-overrides)), so pinning the protocol no longer collapses the context window or disables capabilities.

## Multimodal content blocks

`Message.ContentBlocks` (`[]ContentBlock`) carries an ordered list of structured content elements (text and/or images) alongside `Content`. When non-empty, providers (Anthropic, OpenAI, Responses) render the blocks; when empty, `Content` is used as before (backward compatible for text-only messages). Two helpers govern the boundary:

- `NormalizeContentBlocks(msg)` returns `nil` when `ContentBlocks` is empty (caller takes the legacy `Content` path). When the blocks contain no `text` block and `Content` is non-empty, it **prepends** a text block carrying `Content` — so a caller using `SetTaskWithBlocks` with an image-only block list still gets its instruction text to the model.
- `ValidateContentBlocks(blocks)` enforces that `text` blocks carry non-empty `Text` and `image` blocks carry both `MediaType` and `ImageB64`. Providers call it before rendering so a caller that omits a required field gets a clear local error instead of an opaque API 400. Image `ImageB64` is base64 **without** the `data:` prefix.

The Conductor accepts a multimodal task via `ConductorConfig.ContentBlocks`: when non-empty, it type-asserts the `ContextManager` against the `BlockTaskAware` capability (`SetTaskWithBlocks`) — implemented by `memory.ContextWindow` — so `BuildPrompt` emits the user message carrying both the task text and the blocks (see [orchestration/conductor.md](orchestration/conductor.md)). Conversation summarization replaces image blocks with the placeholder `[image attached]` and concatenates text blocks, so a multimodal user turn survives compaction in text form.

## Invariants

- The Router is safe for concurrent use; `SetModel` takes a write lock, `Call` snapshots under a read lock and releases before backoff.
- `Resolve` always returns usable metadata, even for unknown models (fallback defaults with optimistic `Attachment` capability).
- A partial override (one that leaves some scalar fields unset) inherits its unset fields from the lower non-network tiers; a fully-specified override is returned verbatim.
- `prepareRequest` resolves model metadata at most once per call and reuses it for protocol resolution, default temperature, and context-window validation.
- Model-registry lookups (overrides, built-ins, cache) are case-insensitive.
- `NewTokenCounter` always returns a non-nil counter.
- Pre-call validation rejects oversized requests with `ErrContextWindowExceeded` (detectable via `errors.Is`); this is independent of the agent loop's ongoing context-fill tracking.
- Retryable errors are classified by `WrapProviderError` (HTTP 408/429/500/502/503/504/520–524/529 — request timeout, rate limit, upstream server/gateway faults, Cloudflare edge errors, Anthropic overload — plus transient network errors); `IsRetryable` reports whether a chain contains a retryable `*Error`.
- An explicit `ModelMetadata.Protocol` is always honored over substring `DetectProtocol` detection; the router threads the resolved protocol into `ChatRequest.Protocol`, and the provider honors `req.Protocol` when set.

## Configuration

`RouterConfig`:

| Field | Default | Description |
| ----- | ------- | ----------- |
| `MaxRetries` | `3` (when unset/zero) | Retry attempts for transient errors. Defaults to 3 when unset or zero; a **negative** value disables retries entirely (0 retries). |
| `InitialBackoff` / `MaxBackoff` | `1s` / `30s` | Exponential backoff bounds (doubles each attempt, ±20% jitter). |
| `SafetyMarginPercent` | `5` | Effective context window fraction reserved for counting inaccuracy. |
| `OutputTokenReserve` | `4096` | Default output reserve when metadata lacks an `OutputLimit`. |
| `HTTPClient` | optional | Proxy-configured client. |
| `SamplingFunc` | optional | `func(family string) *float64`; return `nil` to use the provider default. |
| `Logger` | optional | Logs ambiguity warnings on bare-name resolution. |

`APIKey`/`BaseURL`/`Models` must be pre-resolved by the caller (env vars expanded, durations parsed) before `NewRouter`.

## Extension Points

- **Add a provider**: a new `ProviderType` backend implementing the `Provider` interface; the Router dispatches by `ProviderType` (`"anthropic"` / `"openai"`). The `"openai"` type covers any OpenAI-compatible endpoint (set `BaseURL` to a proxy, LM Studio, vLLM, …) and dispatches each model to its native wire protocol via [Multi-protocol routing](#multi-protocol-routing) — including co-located Anthropic delegation for Claude and the Google `generateContent` path for Gemini/Gemma. There is no separate `"google"` `ProviderType`.
- **Anthropic-compatible endpoints**: the built-in Anthropic provider normalizes `BaseURL` to end with `/v1` (the go-anthropic SDK expects the version segment in the base URL, unlike the official Anthropic SDK convention which omits it). A URL already ending in `/v1` is left untouched; otherwise `/v1` is appended. The provider also installs a response-body-capturing transport and surfaces errors that the SDK would otherwise swallow — a `{"type":"error",...}` object or a degenerate empty response returned with HTTP 200 is reported as an error rather than a silent empty reply (common when a misconfigured base URL misses the `/v1` path segment).
- **Custom metadata source**: `ModelRegistry.RegisterSource(src)` for a source consulted after the HuggingFace lookup (e.g. a local model server). `SetCachedMetadata` stores a late-learned entry at tier 3 (cache).
- **Force a wire protocol**: set `ModelMetadata.Protocol` (via the overrides map, a registered source, or a built-in entry) to override substring `DetectProtocol` detection for a model whose name matches a family token but speaks a different protocol.
- **Family-aware sampling**: supply `RouterConfig.SamplingFunc` (see [prompt-building.md](prompt-building.md) for the SDK's `DefaultSampling` defaults).
- **Step-local callers**: `TrackingCaller.WithContextTracker` for independent context trackers in parallel branches.

## Related Specs

- [prompt-building.md](prompt-building.md) — family-aware sampling defaults
- [orchestration/README.md](orchestration/README.md) — the Router is the LLM caller for the Conductor, planner, router, and reflector
- [memory/README.md](memory/README.md) — `ModelMetadata` drives `ContextWindow` sizing
- [embedding.md](embedding.md) — local embeddings (no LLM provider required)
