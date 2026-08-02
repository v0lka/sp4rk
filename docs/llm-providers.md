# LLM Providers

The `llm` package provides LLM provider abstractions, a model registry, and routing for multi-provider inference. It lets you configure one or more providers, switch models at runtime, count tokens, track usage, and recover from transient errors with automatic retry and backoff.

```go
import "github.com/v0lka/sp4rk/llm"
```

## Overview

At the center of the package is the **Router**, which holds one or more **providers** and routes every chat request to the currently active (provider, model) pair. A **ModelRegistry** supplies metadata (context window, capabilities, costs) used for pre-call validation and parameter adaptation. Token counting and usage tracking are layered on top so the agent loop can budget context and account for cost.

Key types:

| Type | Purpose |
| --- | --- |
| `ProviderEntry` | Declarative description of a single provider and its models |
| `Router` / `RouterConfig` | Routes calls to the active provider; holds retry config |
| `Provider` | Unified interface implemented by every provider backend |
| `ModelRegistry` / `ModelMetadata` | 6-tier metadata resolution for any model |
| `APIProtocol` / `DetectProtocol` | Wire-protocol detection; routes a request to the right endpoint |
| `TokenCounter` / `ContextTokenTracker` | Token estimation and API-corrected tracking |
| `UsageTracker` / `TrackingCaller` | Per-session and per-step token accounting |
| `ChatRequest` / `ChatResponse` / `Message` | Request/response types sent to providers |
| `ContentBlock` | Structured content element (text/image) within a `Message` |

## ProviderEntry

`ProviderEntry` describes a single LLM provider with its enabled models. It is the declarative input you pass into `RouterConfig.Providers`.

```go
type ProviderEntry struct {
    Name         string   // logical name ("anthropic", "openai_compatible", …)
    ProviderType string   // provider type: "openai" or "anthropic"
    APIKey       string   // already-expanded API key
    BaseURL      string   // already-expanded base URL
    Models       []string // enabled model names for this provider
}
```

- **`Name`** — a logical, caller-chosen identifier used in logging, error reporting, and composite model IDs. It is *not* a hardcoded family name, so you can name compatible proxies anything you like (e.g. `"lmstudio"`, `"my-anthropic-proxy"`).
- **`ProviderType`** — selects the backend implementation. Supported values: `"anthropic"` and `"openai"`. The `"openai"` type covers any OpenAI-compatible endpoint (set `BaseURL` to point at a proxy, LM Studio, vLLM, etc.), and a single `"openai"`-typed provider can serve *all four* API protocols (Chat Completions, Responses, Anthropic Messages, Google generateContent) — Google and Anthropic models reached through an `"openai"` provider are handled by delegation *inside* the OpenAI provider, not by a new `ProviderType`.
- **`APIKey`** / **`BaseURL`** — must be pre-resolved by the caller (environment variables expanded, etc.) before constructing the router.
- **`Models`** — the bare model names enabled for this provider. The first provider's first model becomes the initial active model.

## Anthropic-compatible endpoints

An `anthropic`-typed provider can target the official Anthropic API (`BaseURL` empty) or any Anthropic-compatible proxy (`BaseURL` set). Two provider-level behaviours make compatible proxies work reliably:

**Base URL normalization.** The go-anthropic SDK treats `BaseURL` as *already including the API version path* — it appends only `/messages`, so its built-in default is `https://api.anthropic.com/v1`. Anthropic-compatible endpoints are conventionally documented with a base URL that **excludes** `/v1` (e.g. `https://api.z.ai/api/anthropic`, matching the `ANTHROPIC_BASE_URL` convention of the official SDK). To bridge this, the provider runs `BaseURL` through `normalizeAnthropicBaseURL`, which appends `/v1` unless the URL already ends with `/v1` (with or without a trailing slash). Pass the convention-style URL without `/v1`; URLs that already follow the go-anthropic convention are left untouched.

**Response-body capture for swallowed errors.** The go-anthropic SDK surfaces errors only for non-2xx HTTP status codes. Some compatible endpoints return an error object (or a degenerate empty body) with HTTP 200, which the SDK then silently decodes into an empty `MessagesResponse` — the provider would see a silent empty reply. The provider installs a `capturingTransport` that reads the raw response body on every call and stashes it so a degenerate 200 response is surfaced as a descriptive error instead of an empty reply. The provided `HTTPClient` (if any) is *cloned*, not mutated, so shared proxy/TLS/timeout configuration is preserved and other consumers of the same client are unaffected.

## Router

`Router` routes LLM calls to the active provider. It is created with `NewRouter`:

```go
func NewRouter(ctx context.Context, cfg RouterConfig, registry *ModelRegistry) (*Router, error)
```

`NewRouter` builds a provider for every `ProviderEntry`, constructs a reverse index from composite model IDs to providers, and selects the first provider's first model as the initial active model. If `registry` is non-nil, providers may register their own metadata sources with it.

> **Concurrency:** `Router` is **safe for concurrent use** from multiple goroutines. It is protected by a `sync.RWMutex`: `SetModel` takes a write lock; `Call` snapshots the active provider and model under a read lock, then releases it before the retry loop so `SetModel` is not blocked by backoff sleeps. The Framework shares one Router across all Conductors created via `NewConductor`.

### RouterConfig

```go
type RouterConfig struct {
    Providers           []ProviderEntry
    MaxRetries          int           // default 3 when unset/zero; negative = 0 (disabled)
    InitialBackoff      time.Duration // default 1s; negative = 0
    MaxBackoff          time.Duration // default 30s; negative = 0
    SafetyMarginPercent int           // default 5
    OutputTokenReserve  int           // default 4096
    HTTPClient          *http.Client  // optional proxy-configured client
    SamplingFunc        SamplingFunc  // optional family-aware temperature defaults
    Logger              *slog.Logger  // optional logger for ambiguity warnings
}
```

All values must be pre-resolved by the caller (env vars expanded, durations parsed) before calling `NewRouter`.

## Model identification: composite vs bare IDs

A **composite model identifier** has the form `"providerName/modelName"` and is the internal *selector* used to route a request to a specific (provider, model) pair. This disambiguates models that share the same bare name across multiple OpenAI-compatible providers (e.g. `"openai/gpt-4"` vs `"lmstudio/gpt-4"`).

The **bare model name** (the part after the first `/`) is what is sent to the LLM API and used for model metadata lookups. Model names may themselves contain a `/` (e.g. `"meta-llama/Llama-3-70b"`), so identifiers are always split on the **first** `/` only.

Helper functions in the package:

```go
// Build a composite identifier from a provider name and a bare model name.
id := llm.CompositeModelID("openai", "gpt-4o") // "openai/gpt-4o"

// Split a composite identifier. ok is false when id has no "/".
provider, model, ok := llm.ParseCompositeModelID(id)

// Report whether id carries a provider prefix.
llm.IsCompositeModelID(id) // true

// Return the bare model name portion (unchanged if already bare).
llm.BareModel("openai/gpt-4o") // "gpt-4o"

// Parse a composite identifier into its parts.
provider, model, ok := llm.ParseCompositeModelID("openai/gpt-4o") // "openai", "gpt-4o", true
```

## Runtime model switching

The router exposes the active selection and lets you switch it at runtime:

```go
// ActiveModel returns the composite identifier ("provider/model").
func (r *Router) ActiveModel() string

// ActiveProviderName returns the logical name of the active provider.
func (r *Router) ActiveProviderName() string

// SetModel switches the active provider and model.
func (r *Router) SetModel(ctx context.Context, model string) error
```

`SetModel` accepts either a composite `"provider/model"` identifier (which routes directly to the named provider) or a bare model name (resolved to the first matching provider for backward compatibility). When a bare name matches multiple providers, the first match — deterministic, sorted by composite ID — is selected and a warning is logged when a logger is configured. Use a composite identifier to disambiguate explicitly.

```go
// Switch to a specific provider's model.
if err := router.SetModel(ctx, "openai/gpt-4o"); err != nil {
    log.Fatal(err)
}
fmt.Printf("active: %s (provider: %s)\n",
    router.ActiveModel(), router.ActiveProviderName())

// Bare name resolves to the first matching provider.
if err := router.SetModel(ctx, "claude-sonnet-4-5"); err != nil {
    log.Fatal(err)
}
```

### Multi-protocol routing

`DetectProtocol(modelID)` determines which wire protocol (and therefore which endpoint) a model speaks — the protocol-level analogue of `DetectFamily`, which adapts prompts and parameters. It is pure substring detection:

| Model ID contains | Protocol | Endpoint |
| --- | --- | --- |
| `gpt-5` or `codex` | `ProtocolResponses` | `POST /responses` |
| `claude` | `ProtocolAnthropic` | `POST /messages` |
| `gemini` or `gemma` | `ProtocolGoogle` | `POST /models/{model}:generateContent` |
| anything else | `ProtocolChatCompletions` (default) | `POST /chat/completions` |

```go
protocol := llm.DetectProtocol("gpt-5.6") // ProtocolResponses
```

The protocol cannot be derived from `ModelFamily` alone: `FamilyOpenAIFlagship` spans both Chat Completions (gpt-4o / o-series) and Responses (gpt-5), so independent model-ID detection is required.

A single OpenAI-compatible `ProviderEntry` dispatches to all four protocols. The OpenAI provider's `Call` honors `req.Protocol` when set (the router fills it from the registry-resolved metadata, which honors an explicit `ModelMetadata.Protocol` override) and otherwise falls back to `DetectProtocol(req.Model)`: Responses → the Responses API; Anthropic → a co-located `AnthropicProvider` (built from the same `BaseURL`/`APIKey`/`HTTPClient`/`Logger`); Google → the `googleCompletion` function (POSTs `{baseURL}/models/{model}:generateContent` with the Google contents/parts format); otherwise Chat Completions. GPT-5.x flagships route to the Responses API (alongside Codex); when `/responses` is genuinely missing (HTTP 404/405) a clear error is surfaced rather than a silent fallback to Chat Completions.

`ModelMetadata.Protocol` is an override escape-hatch. `resolveProtocol` honors an explicit `Protocol` value and only falls back to `DetectProtocol` when it is unset — needed for locally-served models whose name contains a family token but speak a different protocol (e.g. a vLLM model named `gemini-finetune` served over `/chat/completions`). Set it via the `NewModelRegistry` overrides map, a registered source, or a built-in entry. The router's `prepareRequest` resolves model metadata once and threads the resolved protocol into `ChatRequest.Protocol`; the OpenAI provider honors `req.Protocol` over its own `DetectProtocol`, so a tier-1 override takes effect for router-driven calls (direct provider use without a router falls back to name-based detection). A protocol-only partial override inherits the model's real `ContextWindow`/`OutputLimit`/`TokenizerType`/`Capabilities` from the lower tiers (see [Partial overrides](#partial-overrides)), so pinning the protocol no longer collapses the context window or disables capabilities.

## Multi-provider configuration

Configure multiple providers in a single `RouterConfig` to enable runtime model switching — for example, using a strong reasoning model for planning and a faster model for execution:

```go
package main

import (
    "context"
    "log"
    "os"

    "github.com/v0lka/sp4rk/llm"
)

func main() {
    registry := llm.NewModelRegistry(nil)

    providers := []llm.ProviderEntry{
        {
            Name:         "anthropic",
            ProviderType: "anthropic",
            APIKey:       os.Getenv("ANTHROPIC_API_KEY"),
            Models:       []string{"claude-sonnet-4-5"},
        },
        {
            Name:         "openai",
            ProviderType: "openai",
            APIKey:       os.Getenv("OPENAI_API_KEY"),
            Models:       []string{"gpt-4o", "gpt-4o-mini"},
        },
    }

    router, err := llm.NewRouter(context.Background(), llm.RouterConfig{
        Providers:           providers,
        MaxRetries:          3,
        InitialBackoff:      1 * 1000_000_000, // 1s
        MaxBackoff:          30 * 1000_000_000, // 30s
        SafetyMarginPercent: 5,
        OutputTokenReserve:  4096,
    }, registry)
    if err != nil {
        log.Fatal(err)
    }

    // Initial active model is the first provider's first model.
    log.Printf("active: %s (%s)",
        router.ActiveModel(), router.ActiveProviderName())

    // Switch to an OpenAI model for execution.
    if err := router.SetModel(context.Background(), "openai/gpt-4o"); err != nil {
        log.Fatal(err)
    }

    resp, err := router.Call(context.Background(), llm.ChatRequest{
        Messages: []llm.Message{
            {Role: "user", Content: "Say hello in one sentence."},
        },
        MaxTokens: 64,
    })
    if err != nil {
        log.Fatal(err)
    }
    log.Print(resp.Message.Content)
}
```

## ModelRegistry

`ModelRegistry` provides a 6-tier resolution system for model metadata. It is thread-safe and lazily fetches from external sources. All lookups are keyed case-insensitively (`strings.ToLower`) so model IDs that differ only in casing from their canonical registry keys resolve correctly.

```go
registry := llm.NewModelRegistry(nil) // nil = no user overrides
```

`NewModelRegistry` accepts an optional `overrides` map that is defensively copied at construction time, so callers (e.g. config reloads) can mutate their own map without racing the registry's concurrent readers.

### 6-tier resolution

`Resolve(ctx, model)` returns `ModelMetadata` and a boolean indicating whether the model was found in a known source. When `ok` is false, the returned metadata contains usable fallback defaults.

```go
func (r *ModelRegistry) Resolve(ctx context.Context, model string) (ModelMetadata, bool)
```

Resolution order (first match wins; every map lookup uses the lowercased key):

1. **User overrides** — from the `overrides` map passed to `NewModelRegistry`.
2. **Built-in registry** — a hardcoded table of well-known models (OpenAI, Anthropic, Google, DeepSeek, Qwen, GLM, Kimi, xAI Grok, …).
2b. **Fuzzy match** — a vendor-prefix- and separator-insensitive lookup across overrides then built-ins. `normalizeModelID` strips the org/vendor prefix up to the first `/`, lowercases the result, and removes `.`/`-`/`_` punctuation (alphanumerics are never altered, so distinct versions stay distinct — unlike edit-distance it cannot collapse versions); on a multi-match the lexicographically smallest key wins; the hit is cached under the query key. Lookups are O(1) reads against normalized-ID indexes built once at registry construction. Bridges naming drift between hosts and the registry, e.g. `"gpt4o"` matches `"gpt-4o"`, a bare `"glm-5.2-fp8"` matches the prefixed `"zai-org/glm-5.2-fp8"`.
3. **Cache** — results from previous external lookups.
4. **External sources** — HuggingFace API lookup (lazy, cached), then any sources registered via `RegisterSource` (e.g. an LM Studio provider).
5. **Fallback defaults** — `ContextWindow: 128000`, `OutputLimit: 4096`, `TokenizerType: "approximate"`; `ok` is false. Unknown and HuggingFace-resolved models default to attachment-capable (`defaultUnknownCapabilities` sets `Attachment: true`) — a runtime provider error is preferred over silently denying image uploads for a model the registry does not know.

Additional methods:

- `RegisterSource(src ModelMetadataSource)` — add a custom metadata source. Sources are called in order after the HuggingFace lookup fails.
- `Invalidate(model)` — remove a cached entry (e.g. on a mid-session model change).
- `SetCachedMetadata(model, meta)` — store a late-learned entry at tier 3 (the cache). Lets a caller that discovers a model's real context window after construction (e.g. a lazy probe of a local server) populate it. Takes effect only when no tier-1 override or tier-2 built-in exists, so overrides and well-known specs are never silently clobbered. A partial override that leaves its `ContextWindow` at zero inherits the cache-tier value once it arrives, so a lazy local-model probe is not shadowed by a non-zero fallback window baked into the override.
- `SetHTTPClient(client)` — replace the HTTP client used for HuggingFace lookups.
- `ResolveBuiltInModel(model)` — a package-level function that resolves using **only** the built-in catalog (exact case-insensitive, then fuzzy), with no network access and no overrides/cache/HF/sources; returns the fallback defaults with `ok=false` when absent. Safe for startup paths, e.g. to compare a config override against the built-in values or to merge a partial override onto built-in metadata at construction time.

#### Partial overrides

A tier-1 override that pins only some fields — a **partial** override — inherits its unset scalar fields from the lower non-network tiers (built-in → cache → fallback). The inheritable fields are `ContextWindow`, `OutputLimit`, `TokenizerType`, and `Capabilities`. This closes two footguns:

- A protocol-only override (e.g. `{Protocol: ChatCompletions}` for a Google-named checkpoint served locally) previously left `ContextWindow`/`OutputLimit` at zero, collapsing the context window and rejecting requests, and left `Capabilities` all-false, silently disabling tool-calling, reasoning, and image uploads. A zero-valued `ModelCapabilities` is now treated as "unset" and inherited, exactly like a zero `ContextWindow`.
- A protocol-only override that also carried a fallback `ContextWindow: 128000` shadowed a lazy local-model probe writing the model's real window to the cache tier via `SetCachedMetadata`. Keeping the override's window at zero lets the cache-tier probe result take effect once it arrives.

A field already set in the override is authoritative; a fully-specified override (the common case: a `config.yaml` entry whose scalars are all populated) is returned verbatim. `Family` is derived, and `Protocol` is always authoritative — pinning it is precisely what a partial override exists to do. The accepted tradeoff is that the zero value is reserved for "inherit", so there is no way to express "explicitly disable every capability" via a partial override.

```go
registry := llm.NewModelRegistry(map[string]llm.ModelMetadata{
    // Protocol-only partial override: ContextWindow/OutputLimit/Capabilities
    // are inherited from the built-in catalog, so this local server is treated
    // like the real model except for the forced wire protocol.
    "gemma-finetune": {Protocol: llm.ProtocolChatCompletions},
})
```

### ModelMetadata

```go
type ModelMetadata struct {
    ContextWindow int
    OutputLimit   int
    TokenizerType string
    Family        string
    Protocol      APIProtocol
    Capabilities  ModelCapabilities
}
```

- **`ContextWindow`** — maximum input context size in tokens.
- **`OutputLimit`** — maximum output tokens the model can produce.
- **`TokenizerType`** — e.g. `"tiktoken/o200k_base"`, `"anthropic-api"`, `"approximate"`. Drives token counter selection.
- **`Family`** — model family string used for prompt and parameter adaptation. Resolved from metadata or auto-detected via `DetectFamily`.
- **`Protocol`** — wire protocol / canonical endpoint postfix the model speaks. Populated lazily by `resolveProtocol` at `Resolve` time from `DetectProtocol` (or an explicit override). See [Multi-protocol routing](#multi-protocol-routing).
- **`Capabilities`** — see below.

### ModelCapabilities

```go
type ModelCapabilities struct {
    Attachment  bool // image/PDF support
    Reasoning   bool // reasoning/thinking mode
    Temperature bool // accepts the temperature parameter
    ToolCall    bool // function calling support
}
```

`ModelCapabilities` carries `json` + `yaml` struct tags (snake_case) so a single canonical type serializes consistently in both the JSON API contract and `config.yaml` without conversion boilerplate at every layer boundary.

The router uses `Capabilities.Temperature` to decide whether to send a temperature parameter at all — reasoning models that reject temperature are skipped automatically.

### Model families

`DetectFamily(modelID)` determines a model's family from its ID string. Families drive prompt and parameter adaptation. Recognized families include `anthropic`, `openai_flagship`, `openai_standard`, `openai_codex`, `google`, `mistral`, `deepseek`, `qwen`, `glm`, `kimi`, and `default`.

```go
family := llm.DetectFamily("claude-sonnet-4-5") // FamilyAnthropic
```

`FamilyReasoningOptions(family)` returns the native reasoning/thinking options available for a family, the recommended default (always the maximum available effort), and whether the family supports reasoning at all.

## Token counting

```go
type TokenCounter interface {
    Count(text string) int
    CountMessages(msgs []Message) int
}
```

Two implementations are provided:

- **`SimpleTokenCounter`** — fast approximation using the ~4 chars = 1 token rule. Created with `NewSimpleTokenCounter()`.
- **`TiktokenCounter`** — accurate counter using `tiktoken-go` for OpenAI models. Created with `NewTiktokenCounter(encoding)`. Its `Encode` method is not safe for concurrent use, so it is guarded by an internal mutex.

`NewTokenCounter(tokenizerType)` creates a `TokenCounter` based on the tokenizer type from model metadata. Supported types:

| Tokenizer type | Counter |
| --- | --- |
| `tiktoken/o200k_base`, `tiktoken/cl100k_base`, … | `TiktokenCounter` |
| `anthropic-api` | `SimpleTokenCounter` (relies on API correction) |
| `approximate`, `""`, or unknown | `SimpleTokenCounter` |

The returned counter is always valid (never nil). The error indicates that a fallback counter was used instead of the requested type.

```go
counter, err := llm.NewTokenCounter("tiktoken/o200k_base")
// err is nil on success; non-nil means a fallback SimpleTokenCounter was used.
n := counter.CountMessages(msgs)
```

When a message carries image content blocks (see [Multimodal content blocks](#multimodal-content-blocks)), counters apply a per-image estimate: 765 tokens (`estimatedTokensPerImage`, the OpenAI-oriented conservative default — roughly what OpenAI's high-detail processing costs for a typical screenshot) and 85 tokens (`estimatedAnthropicTokensPerImage`) for Anthropic-family models. `NewTokenCounter` overrides the default to the Anthropic value for `tokenizerType == "anthropic-api"`, so image-heavy Anthropic conversations are not over-counted ~9× and do not trigger premature context compaction.

### ContextTokenTracker

`ContextTokenTracker` is a hybrid coordinator that combines predictive counting with API-corrected actuals. It uses a predictive counter for estimates between API calls, then corrects with the real `input_tokens` from API responses.

```go
tracker := llm.NewContextTokenTracker(counter)
tracker.AddDelta("some text to estimate") // add to pending estimate
total := tracker.EstimateTotal()          // lastKnownUsed + pendingDelta
tracker.Correct(resp.Usage.InputTokens)   // reconcile with API actuals
tracker.Reset()                           // zero both counters
```

## Usage tracking

`UsageTracker` accumulates token usage across all LLM calls in a session. It is thread-safe and supports observer callbacks.

```go
tracker := llm.NewUsageTracker()
tracker.AddObserver(func(usage llm.TokenUsage, totalIn, totalOut int, model, family string) {
    log.Printf("call: in=%d out=%d | session total: in=%d out=%d (%s)",
        usage.InputTokens, usage.OutputTokens, totalIn, totalOut, model)
})
tracker.Record(resp.Usage, resp.Model, resp.Family)
in, out := tracker.Totals()
```

### TrackingCaller

`TrackingCaller` wraps a `Caller` and automatically records usage from each response into a `UsageTracker` and corrects an optional `ContextTokenTracker`. It implements the `Caller` interface, so it can be dropped in anywhere a caller is expected.

```go
type TrackingCaller struct { /* ... */ }

func NewTrackingCaller(inner Caller, tracker *UsageTracker) *TrackingCaller
func (tc *TrackingCaller) Call(ctx context.Context, req ChatRequest) (*ChatResponse, error)
func (tc *TrackingCaller) WithContextTracker(t *ContextTokenTracker) *TrackingCaller
```

`WithContextTracker` returns a new `TrackingCaller` that shares the same inner caller and session-level `UsageTracker` but corrects a per-step `ContextTokenTracker`. Use this to create step-local callers for parallel execution.

```go
sessionTracker := llm.NewUsageTracker()
caller := llm.NewTrackingCaller(router, sessionTracker)

// Per-step context tracker for a single execution branch.
stepTracker := llm.NewContextTokenTracker(counter)
stepCaller := caller.WithContextTracker(stepTracker)
resp, err := stepCaller.Call(ctx, req)
```

## Retry & backoff

The router retries transient errors with exponential backoff plus ±20% jitter. Retryable errors are classified by `WrapProviderError` and include:

- **HTTP 408, 429, 500, 502, 503, 504, 529** — request timeout, rate limits, transient server/gateway errors, and Anthropic-specific overload.
- **HTTP 520–524** — Cloudflare edge errors (unknown error, web server down, connection timed out, origin unreachable, origin timeout); 524 is Cloudflare's equivalent of a gateway timeout.
- **Transient network errors** — timeouts, connection refused/reset, EHOSTUNREACH, DNS errors, unexpected EOF.

`IsRetryable(err)` reports whether an error chain contains a retryable `*Error`.

Backoff doubles after each attempt and is capped at `MaxBackoff`:

| Attempt | Backoff (default config) |
| --- | --- |
| 1 → 2 | 1s (±20% jitter) |
| 2 → 3 | 2s (±20% jitter) |
| 3 → 4 | 4s (±20% jitter) |

With the default `MaxRetries: 3`, the worst-case retry path adds up to ~7s of latency. Callers that rely on error propagation for compaction timing, circuit-breaker resets, or budget control should account for this.

### Disabling retries

`MaxRetries` follows the same sentinel convention as the backoff durations: `0` (unset) resolves to the default of **3**; a **negative value** resolves to **0**, disabling retries entirely. To turn retries off, pass a negative `MaxRetries`:

```go
router, err := llm.NewRouter(ctx, llm.RouterConfig{
    Providers:      providers,
    MaxRetries:     -1,   // 0 retries — retries disabled
}, registry)
```

To reduce retry latency without removing retries entirely, keep retries on with a small `MaxRetries` (e.g. `1`) and a short `MaxBackoff`:

```go
router, err := llm.NewRouter(ctx, llm.RouterConfig{
    Providers:      providers,
    MaxRetries:     1,    // one retry attempt
    MaxBackoff:     1 * time.Second,
}, registry)
```

## ChatRequest / ChatResponse

```go
type ChatRequest struct {
    Model           string           // bare model name (filled from active model only when empty)
    ModelFamily     string           // family hint; auto-detected if empty
    Protocol        APIProtocol      // wire protocol; auto-resolved from the registry when empty, else DetectProtocol
    Messages        []Message
    Tools           []ToolDefinition
    MaxTokens       int
    Temperature     *float64         // nil = use provider/sampling default
    ReasoningEffort string           // native reasoning value (e.g. "On", "high")
}

type ChatResponse struct {
    Model      string
    Family     string
    Message    Message
    Reasoning  string     // extended thinking/reasoning (if supported)
    Usage      TokenUsage
    StopReason string     // "end_turn" | "tool_use" | "max_tokens"
}

type TokenUsage struct {
    InputTokens  int
    OutputTokens int
}
```

`Router.Call` fills in the active bare model when `Model` is empty, resolves the model metadata once (threading the resolved protocol into `Protocol` and reusing the metadata for default temperature and context-window validation), validates the context window, then sends the request. On success it ensures `Model` and `Family` are set on the response and trims trailing whitespace from content and reasoning fields.

### Forcing a specific model without switching providers

The "fill only when empty" rule for `Model` is part of the router contract: a pre-set `req.Model` is preserved and sent to the active provider (`r.activeProvider`) unchanged, while the provider that handles the call is always the active one. This lets a caller force a specific model string without switching providers.

`agent.NewModelOverrideCaller(inner LLMCaller, model string)` exploits this: it wraps an `LLMCaller` so every `Call` sets `req.Model` to `model` before delegating to `inner`, relying on the rule above to bypass the router's active-model selection for that caller. It returns `inner` unchanged when `model` is empty, so the override applies conditionally — used to apply a per-agent `Model` from a [Subagent Profile](agents.md). It is a `Caller` wrapper in the same spirit as the usage-tracking wrapper described in [Usage tracking](#usage-tracking).

## Message types

```go
type Message struct {
    Role             string          // "system" | "user" | "assistant" | "tool"
    Content          string          // text fallback when ContentBlocks has no text block
    ContentBlocks    []ContentBlock  // structured content; when non-empty, providers render the blocks
    ReasoningContent string          // chain-of-thought (e.g. DeepSeek)
    ToolCalls        []ToolCall      // tool calls (for assistant)
    ToolCallID       string          // call ID (for tool responses)
    ReasoningItems   []ReasoningItem // Responses API reasoning output items
}

type ToolCall struct {
    ID    string
    Name  string
    Input json.RawMessage
}

type ReasoningItem struct {
    ID      string // required to round-trip to the Responses API
    Summary string
}
```

- **`system` / `user` / `assistant` / `tool`** — the four roles. `tool` messages carry a `ToolCallID` correlating them to the originating call.
- **`ToolCalls`** — emitted by assistant messages when the model decides to call a tool.
- **`ReasoningContent`** — chain-of-thought text (e.g. DeepSeek).
- **`ReasoningItems`** — reasoning output items from the OpenAI Responses API. Each item's `ID` is required when sending the item back to the API in subsequent requests to maintain the reasoning chain across turns.

`ToolDefinition` describes a tool to the LLM as a JSON Schema:

```go
type ToolDefinition struct {
    Name        string
    Description string
    InputSchema json.RawMessage
}
```

### Multimodal content blocks

`Message.ContentBlocks` carries an ordered list of structured content elements (text and/or images) instead of a plain `Content` string. When non-empty, providers render the blocks; when empty, `Content` is used as before (backward compatible for text-only messages).

```go
type ContentBlock struct {
    Type      string // "text" | "image"
    Text      string // text content (Type == "text")
    ImageB64  string // base64 image data, WITHOUT the "data:" prefix (Type == "image")
    MediaType string // MIME type, e.g. "image/png" (Type == "image")
}
```

- **`NormalizeContentBlocks(msg)`** returns `nil` when `ContentBlocks` is empty (the legacy `Content` path). When the blocks lack a `"text"` block and `Content` is non-empty, it **prepends** a text block carrying `Content` — so a caller using image-only blocks still gets its instruction text to the model. The returned slice shares storage with `msg.ContentBlocks` when no prepend is needed.
- **`ValidateContentBlocks(blocks)`** enforces that text blocks carry `Text`, and image blocks carry both `MediaType` and `ImageB64`. Providers call it before rendering so a missing field yields a clear local error instead of an opaque API 400.

The memory layer attaches images via `ContextWindow.SetTaskWithBlocks(task, blocks)` (in the `memory` package); the Conductor invokes it when a task carries content blocks. Conversation summaries replace each image block with the literal `[image attached]` (text blocks are concatenated). Providers (Anthropic, OpenAI, Responses) render blocks and log unknown block types for diagnostics.

## Safety margin and output token reserve

Before every call, the router validates that the estimated token count of the messages fits within the model's context window minus the output reserve.

- **`SafetyMarginPercent`** (default 5) — a percentage of the effective context window reserved to account for counting inaccuracy. The effective maximum is reduced by this fraction before comparison.
- **`OutputTokenReserve`** (default 4096) — the default output token reserve used when model metadata does not specify an `OutputLimit`.

If the estimated count exceeds the effective maximum, the router returns a non-retryable `*Error` wrapping the `ErrContextWindowExceeded` sentinel. Detect it with `errors.Is`:

```go
if errors.Is(err, llm.ErrContextWindowExceeded) {
    // request was too large — compact or trim before retrying
}
```

> This pre-submission guard rejects obviously oversized requests. It is intentionally independent from ongoing context-fill tracking in the agent loop.

## SamplingFunc for family-aware temperature defaults

```go
type SamplingFunc func(family string) *float64
```

When `ChatRequest.Temperature` is nil, the router applies a default temperature based on the active model's family. Set `RouterConfig.SamplingFunc` to control this. Return `nil` to use the provider's built-in default (no temperature parameter sent).

The router skips temperature application entirely for models whose `Capabilities.Temperature` is false (e.g. reasoning models like o1/o3). When no sampling function is configured, the fallback default is deterministic (temperature 0.0).

```go
router, err := llm.NewRouter(ctx, llm.RouterConfig{
    Providers: providers,
    SamplingFunc: func(family string) *float64 {
        switch family {
        case "anthropic":
            t := 1.0
            return &t
        case "openai_flagship":
            t := 0.7
            return &t
        default:
            return nil // use provider default
        }
    },
}, registry)
```

## Error types

Provider errors are wrapped in a classified `*Error`:

```go
type Error struct {
    Provider   string // e.g. "openai", "anthropic"
    StatusCode int    // HTTP status code (0 if not applicable)
    Retryable  bool   // whether this error is safe to retry
    Err        error  // the original underlying error
}
```

- `NewError(provider, statusCode, retryable, err)` — construct directly.
- `WrapProviderError(provider, statusCode, err)` — classify by HTTP status and network error type.
- `IsRetryable(err)` — true when the chain contains a `*Error` with `Retryable == true`.
- `NewContextWindowError(...)` — non-retryable error for context window overflow, wrapping `ErrContextWindowExceeded`.
