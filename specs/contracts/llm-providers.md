# Contract: LLM Providers & Routing

> This contract documents the public LLM interface an embedding application implements and consumes for multi-provider inference. It is the boundary between the generic provider/routing layer (`github.com/v0lka/sp4rk/llm`) and the host application (and the agent executor) that drives LLM calls.

## Boundary Rule

The host application consumes the LLM types from `github.com/v0lka/sp4rk/llm` directly. The `llm` package depends only on third-party SDK clients (`go-anthropic`, `openai-go`, `tiktoken-go`) and on sibling sp4rk packages where needed; it never imports host-application code. An embedder plugs in inference by **providing** a `Provider` (or using the built-in OpenAI/Anthropic providers), composing them through a `Router`, and resolving model metadata via a `ModelRegistry`. The agent executor consumes LLM capability exclusively through `LLMCaller`, which `Router` satisfies.

## Interfaces

| Interface / Type | Package | Implemented / Consumed By | Purpose |
| --- | --- | --- | --- |
| `Provider` | llm | Implemented by built-ins / host | Unified provider interface: `ChatCompletion(ctx, ChatRequest)(*ChatResponse, error)` and `Name() string` |
| `Caller` | llm | Minimal alias | Single-call interface (`Call`); higher layers may define compatible interfaces without importing `llm` |
| `Router` | llm | Constructed by host | Routes calls to the active provider; retry/backoff, context-window validation, family-aware sampling; satisfies `LLMCaller` |
| `RouterConfig` | llm | Built by host | Pre-resolved router config: providers, retries, backoffs, safety margin, HTTP client, purpose-aware `SamplingFunc`, logger |
| `SamplingDefaults` / `SamplingFunc` | llm | Supplied by host | Five-field family preset (`Temperature`, `TopP`, `TopK`, `RepetitionPenalty`, `PresencePenalty`); `func(family string) SamplingDefaults` |
| `CallPurpose` | llm | Set by callers | Classifies executor/default versus routing/compaction/summarization calls so the router selects vendor or deterministic sampling |
| `ProviderEntry` | llm | Built by host | One enabled provider: logical name, type, expanded API key/base URL, enabled models |
| `ModelRegistry` | llm | Constructed by host | 7-tier model metadata resolution (overrides → observed runtime → built-in → fuzzy → cache → HuggingFace/registered sources → fallback) |
| `ModelMetadata` | llm | Consumed by host/executor | Context window, output limit, tokenizer type, family, protocol, capabilities |
| `ModelCapabilities` | llm | Consumed by host | `Attachment`, `Reasoning`, `Temperature`, `ToolCall` flags (JSON+YAML-tagged; serialized in the JSON API contract and config.yaml) |
| `APIProtocol` / `DetectProtocol` | llm | Consumed by Router/provider | Wire-protocol selection per model ID (`ProtocolChatCompletions`/`ProtocolResponses`/`ProtocolAnthropic`/`ProtocolGoogle`); the protocol-level analogue of `DetectFamily`. `prepareRequest` resolves it once via the registry (honoring an explicit `ModelMetadata.Protocol` override) and threads it into `ChatRequest.Protocol`; `DetectProtocol` is the name-based fallback used when no registry/override is in play |
| `Message` | llm | Consumed by host | LLM message unit (role, content, content blocks, reasoning, tool calls, reasoning items) |
| `ContentBlock` | llm | Consumed by host | Structured content element: `text` or `image` block; rendered by providers when `Message.ContentBlocks` is non-empty |
| `ChatRequest` | llm | Consumed by host/executor | Request: model, family/protocol hints, messages, tools, max tokens, `CallPurpose`, five optional sampling fields, reasoning effort |
| `ChatResponse` | llm | Consumed by host/executor | Response: model, family, message, reasoning, `TokenUsage`, stop reason |
| `TokenCounter` | llm | Implemented by built-ins / host | Token counting for context budget: `Count(string)int`, `CountMessages([]Message)int` |
| `TokenUsage` | llm | Consumed by host | Input/output token consumption from a response |
| `ReasoningEffort` | llm | Consumed by host | Native reasoning level (plain `string`, not a custom type) |

## Initialization

At startup the host assembles the LLM surface in this order:

1. Resolve provider configuration (expand environment variables, parse durations) into a `RouterConfig` whose `Providers` slice contains at least one `ProviderEntry`.
2. Optionally build a `ModelRegistry` with user `overrides` and register any `ModelMetadataSource` functions (e.g. a local-model server).
3. Call `NewRouter(ctx, cfg, registry)` to construct the `Router`. The router creates the underlying `Provider` instances for each entry, builds a `model → provider` reverse index, and selects the first provider's first model as active.
4. Build a `TokenCounter` via `NewTokenCounter(tokenizerType)` and (if hosting the router's metadata/validation) attach it; the registry's context-window validation relies on a counter.
5. The host passes the `Router` to the agent executor as its `LLMCaller`. The active model can be switched later via `SetModel`.

## Data Flow Across Boundary

- **Host → Router (in):** `RouterConfig` (providers, HTTP client, retry/backoff, sampling) and a `*ModelRegistry`.
- **Host → Router (runtime):** `SetModel(ctx, compositeModelID)` selects the active provider+model; `ActiveModel()` returns the composite ID (`"provider/model"`).
- **executor → Router:** `Call(ctx, ChatRequest)` — messages, tools, max tokens, `CallPurpose`, optional sampling fields, and reasoning effort. When `ChatRequest.Model` is empty the router fills it with the active bare model snapshot (`prepareRequest` sets `req.Model = bareModel` only when `req.Model == ""`); a pre-set `req.Model` is preserved and sent to the active provider unchanged. This lets a caller force a specific model without switching providers — `agent.NewModelOverrideCaller` wraps an `LLMCaller` to set `req.Model` on every call, relying on this "fill only when empty" rule. The provider that handles the call is always the active one (`r.activeProvider`); the override changes the model string, not provider routing.
- **Router sampling policy:** explicit per-request sampling fields win field-by-field. `CallPurposeExecutor` and the zero-value purpose receive the configured family `SamplingDefaults`; routing, compaction, and summarization receive only a deterministic temperature (`0`, with family-safe floors `google=1.0`, `qwen=0.6`) and never inherit preset top-p/top-k/penalties. A model whose resolved capabilities reject temperature receives no injected sampling fields. Without a `SamplingFunc`, an executor/default request falls back to temperature `0`; an all-nil returned preset leaves provider defaults untouched.
- **Router → Provider:** an internally-built provider request derived from the `ChatRequest` (provider-specific mapping). `prepareRequest` resolves the model metadata once (a single `Resolve`, honoring an explicit `ModelMetadata.Protocol` override) and threads the resolved protocol into `ChatRequest.Protocol` when the caller left it empty; this also lets `applyDefaultSampling` and `validateContextWindow` reuse the resolved metadata without per-call registry I/O. Providers serialize only supported sampling parameters: OpenAI-compatible Chat accepts all five on custom endpoints but official OpenAI drops `top_k`/`repetition_penalty`; Responses accepts temperature/top-p; Anthropic accepts temperature/top-p/top-k and drops all sampling while extended thinking is enabled; Google accepts temperature/top-p/top-k. Nil and unsupported fields are omitted. The OpenAI provider honors `req.Protocol` when set and otherwise falls back to `DetectProtocol(req.Model)` — dispatching to the Responses API, a co-located Anthropic provider, or the Google `generateContent` path as appropriate (see [../domains/llm-providers.md](../domains/llm-providers.md#multi-protocol-routing)). When `Message.ContentBlocks` is non-empty, the provider renders the structured blocks (text + images) instead of the plain `Content` string.
- **Provider → Router:** `*ChatResponse` (message, reasoning, usage, stop reason).
- **Router → executor:** `*ChatResponse`, or a retryable-error path that re-attempts with backoff.
- **ModelRegistry → Router/host:** `Resolve(ctx, model)` returns `ModelMetadata` + an `ok` flag indicating a known source (vs. fallback defaults). A **partial** override (one that pins only some fields, e.g. a protocol-only auto-remap) inherits its unset scalar fields — `ContextWindow`, `OutputLimit`, `TokenizerType`, `Capabilities` — from the lower non-network tiers (observed runtime → built-in → cache → fallback), so it does not collapse the context window or silently disable capabilities; a fully-specified override is returned verbatim. `SetRuntimeMetadata(model, meta)` stores an OBSERVED runtime entry (tier 1.5: above the built-in catalog, below user overrides) describing how a model is actually served — e.g. a self-hosted server's runtime context window; partial runtime entries enrich from the tiers below runtime, and `RuntimeMetadata(model)` reads an entry back as stored (without enrichment). `ResolveBuiltInModel(model)` resolves factory defaults from the catalog only (no network/overrides); `SetCachedMetadata(model, meta)` stores a late-learned entry at the cache tier.
- **TokenCounter → Router/host:** message token counts used for pre-submission context-window validation and ongoing fill tracking.

Data is plain Go values. The composite model-ID convention (`"provider/model"`) disambiguates models that share a bare name across multiple providers; `BareModel(ActiveModel())` yields the bare name sent to the LLM API.

## Error Propagation

- **Transient errors** (HTTP 408, 429, 500, 502, 503, 504, 520–524, 529, network blips) are retried inside `Router.Call` with exponential backoff (1s → 2s → 4s, capped at `MaxBackoff`, ±20% jitter) before the error reaches the caller. `MaxRetries` defaults to 3 when unset; a negative value disables retries.
- **Non-retryable errors** (auth, 4xx other than 429, malformed response) propagate out of `Call` to the executor.
- **Context-window overflow** is detected pre-submission: `Router.validateContextWindow` rejects requests whose estimated token count exceeds the model's effective window (context window minus output reserve minus safety margin) by returning a context-window error before the provider is called.
- **Model switching errors** from `SetModel` (unknown composite ID, ambiguous bare name) propagate to the host. When a bare model name is ambiguous across providers, the router logs a warning (if a logger is configured) and selects deterministically.
- **Metadata fallback** is **not** an error: `Resolve` returns usable fallback `ModelMetadata` with `ok=false` when no known source matches; callers treat this as "use defaults, not found" rather than a failure.
- **Token-counter errors**: `NewTokenCounter` always returns a valid (never nil) counter, falling back to `SimpleTokenCounter` and returning an `error` only to signal that a fallback was used.

## Breaking Change Checklist

- If you change the `Provider` interface, you MUST update the built-in OpenAI and Anthropic providers and any host-supplied provider.
- If you change `Router.Call`, `SetModel`, or `ActiveModel` signatures, you MUST verify the router still satisfies `agent.LLMCaller` and update every host call site.
- If you change `ChatRequest`/`ChatResponse`/`Message`, you MUST update the agent executor (prompt building, response parsing), every provider mapping, and serialization.
- If you change `SamplingDefaults`, `SamplingFunc`, `CallPurpose`, or deterministic floors, you MUST update the root conversion from `prompt.SamplingConfig`, every structured-call site (router/planner/reflector/judge/compaction/summarization), and the provider filtering matrix.
- If you change the composite model-ID convention, you MUST update `SetModel` callers, `ActiveModel`/`BareModel` consumers, and persisted model selectors.
- If you change `ModelMetadata`/`ModelCapabilities`, you MUST update metadata sources, context-window validation, prompt/parameter adaptation, and any host capability gating. (`ModelCapabilities` is serialized in both the JSON API contract and config.yaml via its struct tags.)
- If you change the 7-tier `ModelRegistry.Resolve` ordering (including the tier-1.5 observed-runtime entries written by `SetRuntimeMetadata`) or the fuzzy `normalizeModelID` rule, you MUST document the new precedence and update host overrides expectations.
- If you change `DetectProtocol` or the four `APIProtocol` constants, you MUST update the OpenAI provider dispatch, the Google `generateContent` path, the co-located Anthropic delegate, and any host relying on `ModelMetadata.Protocol` to force a protocol.
- If you change `ContentBlock`, `NormalizeContentBlocks`, or `ValidateContentBlocks`, you MUST update every provider's block rendering, the token counters' per-image estimation, conversation-summarization block handling, and `memory.ContextWindow.SetTaskWithBlocks`.
- If you change `TokenCounter`, you MUST update `SimpleTokenCounter`, `TiktokenCounter`, `NewTokenCounter`, and every counter consumer (router validation, context manager fill tracking).
- If you alter retry/backoff defaults or the retryable-status-code set, you MUST document the latency impact on callers that rely on error propagation for compaction timing, circuit-breaker resets, or budget control.
- If you change `prepareRequest`'s "fill `req.Model` only when empty" rule, you MUST update `agent.NewModelOverrideCaller` (which relies on it to force a per-agent model) and document the new contract for callers that pre-set `ChatRequest.Model`.
