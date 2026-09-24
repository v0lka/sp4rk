package llm

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"math/rand/v2"
	"net/http"
	"sort"
	"sync"
	"time"
)

// SamplingDefaults holds family-aware sampling defaults the router may inject
// into a ChatRequest. It mirrors the vendor matrix in prompt/sampling.go (the
// llm package cannot import prompt, so the shape is duplicated here and
// converted by the host application). nil fields = no override (provider
// default).
type SamplingDefaults struct {
	Temperature       *float64
	TopP              *float64
	TopK              *int
	RepetitionPenalty *float64
	PresencePenalty   *float64
}

// SamplingFunc returns family-aware sampling defaults for the given model
// family. Return a zero-value (all nil) SamplingDefaults — or a nil field — to
// use the provider's built-in default (parameter not sent).
type SamplingFunc func(family string) SamplingDefaults

// RouterConfig configures the LLM router.
// All values must be pre-resolved by the caller (env vars expanded, durations parsed).
//
// MaxRetries defaults to 3 when unset (zero); a negative value means
// explicitly 0 (retries disabled). With retries enabled, transient errors
// (HTTP 429, 502, 503, 529, network blips) recover automatically with
// exponential backoff (1s → 2s → 4s, capped at MaxBackoff). This adds up to
// ~7s of latency on the worst-case retry path. Callers that rely on error
// propagation for compaction timing, circuit-breaker resets, or budget
// control should account for this default, or disable retries with a
// negative MaxRetries. InitialBackoff and MaxBackoff follow the same
// convention: 0 → default (1s / 30s), negative → explicitly 0.
type RouterConfig struct {
	Providers           []ProviderEntry // all enabled providers (at least one required)
	MaxRetries          int             // Max retry attempts on retryable errors
	InitialBackoff      time.Duration   // Already parsed initial backoff duration
	MaxBackoff          time.Duration   // Already parsed max backoff duration
	SafetyMarginPercent int             // Percentage of context window reserved as safety margin (default: 5)
	OutputTokenReserve  int             // Default output token reserve when model metadata doesn't specify (default: 4096)
	HTTPClient          *http.Client    // Optional proxy-configured HTTP client (nil = default)
	SamplingFunc        SamplingFunc    // Optional family-aware temperature defaults; nil = no default (provider decides)
	Logger              *slog.Logger    // Optional logger for ambiguity warnings (nil = silent)
}

// ReasoningWire selects the JSON spelling a provider expects for Qwen-family
// reasoning controls (enable_thinking / reasoning_effort). It is an explicit,
// operator-chosen per-provider switch — sp4rk deliberately does NOT guess it
// from the base URL, because a loopback heuristic mislabels every non-llama.cpp
// local server (LM Studio, vLLM, Ollama, KoboldCpp) and silently changes the
// request body of providers that were working.
//
// ReasoningWireVendorDefault keeps the historical spelling: both controls as
// TOP-LEVEL request fields. That is what vLLM, LM Studio, SGLang, Ollama and
// DashScope read.
//
// llama.cpp — and forks of it, e.g. the PrismML-Eng/llama.cpp build that serves
// c0wrk's embedded Bonsai 2 27B model — parses the same payload differently, in
// oaicompat_chat_params_parse (tools/server/server-common.cpp):
//
//   - A TOP-LEVEL "enable_thinking" is never read. The flag is picked up only
//     from "chat_template_kwargs", whose values are JSON-dumped into a
//     key→string map and later re-parsed (common/chat.cpp) into the template's
//     extra context. The dumped form is compared against the strings
//     "true"/"false", so a JSON BOOLEAN works, while a JSON *string* dumps with
//     surrounding quotes and makes the server throw
//     `invalid type for "enable_thinking" (expected boolean, got string)`.
//     Sending enable_thinking at the top level is therefore a silent no-op
//     there: "Off" would not turn thinking off.
//   - A TOP-LEVEL "reasoning_effort" IS read: "none" sets enable_thinking=false
//     and erases the kwarg; any other non-empty value is copied into
//     chat_template_kwargs["reasoning_effort"]. This is why the native
//     xhigh/medium/low levels already reach a llama.cpp server today.
//   - Unknown top-level fields are silently ignored (there is no strict schema
//     validation), so the vendor-default spelling is harmless — it just does
//     not express "thinking off".
//
// ReasoningWireChatTemplateKwargs emits the llama.cpp spelling: one top-level
// "chat_template_kwargs" object carrying enable_thinking (as a JSON boolean)
// and, for the native effort levels only, reasoning_effort.
type ReasoningWire string

const (
	// ReasoningWireVendorDefault emits Qwen reasoning controls as top-level
	// request fields (enable_thinking, reasoning_effort) — the historical
	// sp4rk behavior, and the correct spelling for every OpenAI-compatible
	// server that reads them there. Zero value; used by every provider that
	// does not opt in.
	ReasoningWireVendorDefault ReasoningWire = ""

	// ReasoningWireChatTemplateKwargs emits Qwen reasoning controls inside a
	// single top-level "chat_template_kwargs" object, the spelling llama.cpp
	// (and its forks) parses. See ReasoningWire for the server-side contract.
	ReasoningWireChatTemplateKwargs ReasoningWire = "chat_template_kwargs"
)

// ProviderEntry describes a single LLM provider with its enabled models.
type ProviderEntry struct {
	Name         string   // logical name ("anthropic", "openai_compatible", …)
	ProviderType string   // provider type: "openai", "anthropic"
	APIKey       string   // already-expanded API key
	BaseURL      string   // already-expanded base URL
	Models       []string // enabled model names for this provider
	// HTTPClient optionally overrides the router-level client for this
	// provider only (e.g. a per-provider TLS configuration). nil = use
	// RouterConfig.HTTPClient (which may itself be nil → SDK default).
	HTTPClient *http.Client
	// ReasoningWire optionally selects a non-default JSON spelling for
	// Qwen-family reasoning controls on THIS provider only (see
	// ReasoningWire). Zero value = ReasoningWireVendorDefault. Set it to
	// ReasoningWireChatTemplateKwargs for a llama.cpp-served endpoint.
	ReasoningWire ReasoningWire
}

// Router routes LLM calls to the active provider.
//
// Concurrency: Router is safe for concurrent use from multiple goroutines.
// SetModel takes a write lock; all other methods take read locks. Call
// snapshots the active provider and model under a read lock, then releases it
// before the retry loop so SetModel is not blocked by backoff sleeps.
type Router struct {
	mu                 sync.RWMutex
	providers          map[string]Provider
	modelToProvider    map[string]string // composite model ID ("provider/model") → provider name
	activeProvider     Provider
	activeModel        string // composite model ID ("provider/model") — the selector
	activeBareModel    string // bare model name sent to the LLM API / used for metadata
	activeProviderName string
	// Retry configuration
	maxRetries     int
	initialBackoff time.Duration
	maxBackoff     time.Duration
	// Pre-call context window validation
	registry            *ModelRegistry
	tokenCounter        TokenCounter
	sampling            SamplingFunc
	safetyMarginPercent int // percentage of context window reserved as safety margin (default: 5)
	outputTokenReserve  int // default output token reserve when model metadata doesn't specify (default: 4096)
	logger              *slog.Logger
}

// NewRouter creates a new Router from the given configuration.
// The caller is responsible for resolving provider config, expanding env vars,
// and parsing durations before calling this function.
// If registry is provided, providers may register their metadata sources.
func NewRouter(ctx context.Context, cfg RouterConfig, registry *ModelRegistry) (*Router, error) {
	if len(cfg.Providers) == 0 {
		return nil, errors.New("no providers configured")
	}

	providers := make(map[string]Provider, len(cfg.Providers))
	modelToProvider := make(map[string]string)

	for _, entry := range cfg.Providers {
		if entry.ProviderType == "" {
			return nil, fmt.Errorf("provider %q has no type", entry.Name)
		}
		// Per-provider client override: an entry may carry its own HTTP client
		// (e.g. host-app TLS pinning for one self-signed endpoint); otherwise
		// every provider shares the router-level client.
		providerClient := entry.HTTPClient
		if providerClient == nil {
			providerClient = cfg.HTTPClient
		}
		provider, err := createProviderFromConfig(ctx, entry.Name, entry.ProviderType, entry.APIKey, entry.BaseURL, providerClient, cfg.Logger, entry.ReasoningWire)
		if err != nil {
			return nil, fmt.Errorf("failed to create provider %q: %w", entry.Name, err)
		}
		providers[entry.Name] = provider

		// Build reverse index: composite model ID ("provider/model") → provider
		// name. Composite keys disambiguate models that share the same bare name
		// across multiple providers.
		for _, m := range entry.Models {
			modelToProvider[CompositeModelID(entry.Name, m)] = entry.Name
		}
	}

	// Set initial active provider+model = first provider's first model
	first := cfg.Providers[0]
	if len(first.Models) == 0 {
		return nil, fmt.Errorf("provider %q has no enabled models", first.Name)
	}
	activeProvider := providers[first.Name]
	if activeProvider == nil {
		return nil, fmt.Errorf("provider %q not found", first.Name)
	}

	// Sentinel convention (see RouterConfig doc): 0 → default, negative →
	// explicitly 0/disabled.
	maxRetries := cfg.MaxRetries
	switch {
	case maxRetries == 0:
		// Default to 3 retries so transient errors (HTTP 429/502/503/529,
		// network blips) recover automatically.
		maxRetries = 3
	case maxRetries < 0:
		maxRetries = 0 // retries explicitly disabled
	}
	initialBackoff := cfg.InitialBackoff
	switch {
	case initialBackoff == 0:
		initialBackoff = 1 * time.Second
	case initialBackoff < 0:
		initialBackoff = 0
	}
	maxBackoff := cfg.MaxBackoff
	switch {
	case maxBackoff == 0:
		maxBackoff = 30 * time.Second
	case maxBackoff < 0:
		maxBackoff = 0
	}
	safetyMarginPercent := cfg.SafetyMarginPercent
	if safetyMarginPercent <= 0 {
		safetyMarginPercent = 5
	}
	outputTokenReserve := cfg.OutputTokenReserve
	if outputTokenReserve <= 0 {
		outputTokenReserve = 4096
	}

	return &Router{
		providers:           providers,
		modelToProvider:     modelToProvider,
		activeProvider:      activeProvider,
		activeModel:         CompositeModelID(first.Name, first.Models[0]),
		activeBareModel:     first.Models[0],
		activeProviderName:  first.Name,
		maxRetries:          maxRetries,
		initialBackoff:      initialBackoff,
		maxBackoff:          maxBackoff,
		registry:            registry,
		tokenCounter:        NewSimpleTokenCounter(),
		sampling:            cfg.SamplingFunc,
		safetyMarginPercent: safetyMarginPercent,
		outputTokenReserve:  outputTokenReserve,
		logger:              cfg.Logger,
	}, nil
}

// createProviderFromConfig creates a Provider based on the provider type.
// The caller must have already expanded environment variables.
// name is the logical provider name (config key) used for logging and error
// reporting; it is forwarded to the provider so named compatible providers
// (e.g. "lmstudio", "my-anthropic-proxy") report their real name, not a
// hardcoded family name. logger is forwarded to the provider for debug-level
// diagnostics (nil = slog.Default()). reasoningWire selects the JSON spelling
// of Qwen-family reasoning controls for this provider (zero value =
// vendor-default top-level fields; see ReasoningWire).
func createProviderFromConfig(ctx context.Context, name, provType, apiKey, baseURL string, httpClient *http.Client, logger *slog.Logger, reasoningWire ReasoningWire) (Provider, error) {
	switch provType {
	case "openai":
		return NewOpenAIProvider(OpenAIProviderConfig{
			Name:          name,
			APIKey:        apiKey,
			BaseURL:       baseURL,
			HTTPClient:    httpClient,
			Logger:        logger,
			ReasoningWire: reasoningWire,
		})

	case "anthropic":
		return NewAnthropicProvider(AnthropicProviderConfig{
			Name:       name,
			APIKey:     apiKey,
			BaseURL:    baseURL,
			HTTPClient: httpClient,
			Logger:     logger,
		})

	default:
		return nil, fmt.Errorf("unknown provider type: %s", provType)
	}
}

// retryBackoff sleeps for the given duration with +/- 20% jitter, respecting context cancellation.
// Returns false if the context was cancelled during sleep.
func retryBackoff(ctx context.Context, backoff time.Duration) bool {
	// Add jitter: +/- 20%
	jitterFactor := 0.8 + 0.4*rand.Float64() // random factor between 0.8 and 1.2
	jitteredBackoff := float64(backoff) * jitterFactor
	jitter := time.Duration(jitteredBackoff)
	select {
	case <-time.After(jitter):
		return true
	case <-ctx.Done():
		return false
	}
}

// validateContextWindow checks whether the estimated token count of msgs fits
// within the model's context window minus output reserve. Returns nil when
// validation passes or should be skipped (unknown model, zero context window,
// nil registry).
//
// The caller resolves metadata once and passes it in (meta) so this method
// performs no registry I/O of its own.
//
// NOTE: This is a pre-submission guard to reject obviously oversized requests.
// It differs from ContextWindow.EffectiveMax() (which tracks ongoing fill during
// the agent loop). The two calculations are intentionally independent.
func (r *Router) validateContextWindow(model string, msgs []Message, meta ModelMetadata) error {
	if r.registry == nil || r.tokenCounter == nil {
		return nil
	}

	// Skip validation when metadata is a fallback or context window is 0
	if meta.ContextWindow == 0 {
		return nil
	}

	outputReserve := meta.OutputLimit
	if outputReserve <= 0 {
		outputReserve = r.outputTokenReserve
	}

	effectiveMax := meta.ContextWindow - outputReserve
	if effectiveMax <= 0 {
		return nil
	}

	// Apply safety margin to account for counting inaccuracy
	effectiveMax = int(float64(effectiveMax) * (1 - float64(r.safetyMarginPercent)/100.0))

	estimated := r.tokenCounter.CountMessages(msgs)
	if estimated > effectiveMax {
		return NewContextWindowError(model, estimated, effectiveMax, meta.ContextWindow, outputReserve)
	}

	return nil
}

// applyDefaultSampling fills family-aware sampling defaults on the request for
// every parameter the caller left unset. Per-field priority: an explicit
// ChatRequest value always wins, then the preset from the sampling func, then
// the provider's default (field left nil) — except for models that AUTHORITATIVELY
// declare they don't support the temperature parameter (e.g. reasoning models
// like o1, o3, thinking-locked Kimi endpoints, adaptive-thinking Claude): those
// models equally reject top_p/top_k and penalty overrides, so no sampling
// field is injected and explicitly-set values are stripped as well — on the
// wire they are guaranteed HTTP 400s ("temperature is deprecated for this
// model"), hard-failing any caller that sets its own profile.
//
// "Authoritatively" is the operative word: the declaration must come from the
// built-in catalog, a user override, or an observed runtime entry. Caps filled
// by registry guesswork (meta.GuessedCapabilities — every Resolve path fills
// unknown/local models with defaultUnknownCapabilities, whose Temperature is
// the zero value false) are NOT trusted to strip anything: they mirror the
// nil-Capabilities passthrough below, so a host's explicitly-configured
// sampling for its local models (LM Studio, Ollama, vLLM, config-only entries)
// survives, and no preset is injected into fields the host left unset — the
// provider's own defaults govern them, leaving the host in control of models
// the registry cannot vouch for.
//
// The caller resolves metadata once and passes it in (meta) so this method
// performs no registry I/O of its own.
//
// Purpose-aware policy (CallPurpose): requests marked as routing /
// compaction / summarization get a deterministic profile instead of the
// vendor preset — their output is parsed as structured data (or merged into
// persisted summaries) and must not drift with creative sampling settings.
// Only temperature is injected (0.0, or the family-safe floor for families
// whose vendors document instability at low temperature); top_p/top_k and
// penalties are never preset on deterministic calls. Executor requests and
// requests with no declared purpose keep the full vendor preset.
func (r *Router) applyDefaultSampling(req *ChatRequest, meta ModelMetadata) {
	if r.registry != nil {
		if meta.Capabilities != nil && !meta.Capabilities.Temperature && !meta.GuessedCapabilities {
			// Authoritatively declared incapable of sampling parameters:
			// strip explicitly-set values instead of passing them through.
			// Mirrors the provider-level strip Anthropic requests get in
			// extended-thinking mode, but at the router so every provider and
			// every call purpose (including hosts' own service calls with a
			// fixed profile) is covered.
			req.Temperature = nil
			req.TopP = nil
			req.TopK = nil
			req.RepetitionPenalty = nil
			req.PresencePenalty = nil
			return
		}
		if meta.Capabilities == nil || meta.GuessedCapabilities {
			// meta may be a raw/partial record for a model absent from every
			// registry tier (e.g. a locally-served custom model), or a record
			// whose capabilities are the registry's optimistic guess
			// (GuessedCapabilities — which is what every Resolve path returns
			// for models unknown to the catalog and to the user's config):
			// inject no preset, but let explicitly-set values pass through
			// unchanged so the host keeps full control over models the
			// registry cannot vouch for. The guessed branch mirrors the nil
			// branch exactly — a guessed Temperature=false must not zero a
			// host-configured sampling profile.
			return
		}
	}

	if req.CallPurpose.Deterministic() {
		// Deterministic profile: temperature only, no preset injection of
		// top_p/top_k/penalties. An explicit caller temperature still wins.
		if req.Temperature == nil {
			req.Temperature = DeterministicTemperature(meta.Family)
		}
		return
	}

	// Apply the sampling function regardless of registry presence.
	// The sampling func is responsible for handling empty family
	// (e.g. falling back to a default family or passing through).
	var preset SamplingDefaults
	if r.sampling != nil {
		preset = r.sampling(meta.Family)
	}

	if req.Temperature == nil {
		switch {
		case preset.Temperature != nil:
			req.Temperature = preset.Temperature
		case r.sampling == nil:
			// Fallback: no sampling func — default to deterministic (0.0)
			temp := 0.0
			req.Temperature = &temp
		}
	}
	// Remaining parameters have no fallback without a sampling func: nil stays
	// nil (provider default).
	if req.TopP == nil {
		req.TopP = preset.TopP
	}
	if req.TopK == nil {
		req.TopK = preset.TopK
	}
	if req.RepetitionPenalty == nil {
		req.RepetitionPenalty = preset.RepetitionPenalty
	}
	if req.PresencePenalty == nil {
		req.PresencePenalty = preset.PresencePenalty
	}
}

// prepareRequest fills defaults (model, temperature, wire protocol) and
// validates the context window. Returns an error if the request would exceed
// the context window.
//
// The model registry is resolved at most once here; the resolved ModelMetadata
// is threaded into applyDefaultSampling and validateContextWindow so those
// helpers never trigger their own Resolve (and thus no extra registry I/O).
//
// The bare model name (without provider prefix) is what is sent to the LLM API
// and used for metadata lookups; the composite identifier is only the internal
// selector stored in activeModel.
//
// bareModel is the snapshot of the active bare model taken under the read lock
// by the caller (Call). prepareRequest does not read r.active* fields itself,
// so it does not require the caller to hold r.mu.
func (r *Router) prepareRequest(ctx context.Context, req *ChatRequest, bareModel string) error {
	if req.Model == "" {
		req.Model = bareModel
	}

	// Resolve the model metadata at most once. Resolve honors an explicit
	// tier-1 override and only falls back to DetectProtocol when none is set,
	// so a registry override can steer routing regardless of the model name
	// (the documented escape hatch in protocol.go). When ok is false the
	// returned metadata contains usable fallback defaults, so a single Resolve
	// suffices for protocol, family/capabilities, and context-window lookups
	// alike. For a model absent from every tier (notably a locally-served
	// custom model), this also avoids repeated HuggingFace probes — previously
	// each of protocol/family/context-window resolution issued its own Resolve.
	var meta ModelMetadata
	var ok bool
	if r.registry != nil {
		meta, ok = r.registry.Resolve(ctx, req.Model)
		if !ok && r.logger != nil {
			r.logger.Debug("router: model not found in registry", "model", req.Model)
		}
		if req.Protocol == "" {
			req.Protocol = meta.Protocol
		}
		if req.ModelFamily == "" {
			req.ModelFamily = meta.Family
		}
	}

	r.applyDefaultSampling(req, meta)
	return r.validateContextWindow(req.Model, req.Messages, meta)
}

// Call sends a chat request to the active provider.
func (r *Router) Call(ctx context.Context, req ChatRequest) (*ChatResponse, error) {
	// Snapshot the active provider and bare model under the read lock, then
	// release the lock before the retry loop. The retry loop includes
	// exponential backoff sleeps (up to ~7s); holding the read lock for that
	// duration would block SetModel (write lock) and freeze model switching
	// during retry storms. applyDefaultSampling and validateContextWindow
	// only read immutable fields (registry, tokenCounter, sampling, etc.), so
	// they are safe to call without the lock after the snapshot.
	r.mu.RLock()
	provider := r.activeProvider
	bareModel := r.activeBareModel
	r.mu.RUnlock()

	if err := r.prepareRequest(ctx, &req, bareModel); err != nil {
		return nil, err
	}

	var lastErr error
	backoff := r.initialBackoff

	for attempt := 0; attempt <= r.maxRetries; attempt++ {
		resp, err := provider.ChatCompletion(ctx, req)
		// Defense in depth: a Provider must never return (nil, nil), but a
		// buggy or non-conforming provider can (e.g. a JSON `null` body decoded
		// into a nil response). Surface it as a retryable error rather than
		// dereferencing nil below.
		if err == nil && resp == nil {
			err = NewError(provider.Name(), 0, true,
				errors.New("llm: provider returned a nil response with no error"))
		}
		if err == nil {
			// Ensure model is set in response
			if resp.Model == "" {
				resp.Model = req.Model
			}
			// Resolve family from model registry
			if r.registry != nil && resp.Family == "" {
				meta, _ := r.registry.Resolve(ctx, resp.Model)
				resp.Family = meta.Family
			}
			normalizeResponse(resp)
			return resp, nil
		}

		lastErr = err

		// Don't retry if not retryable or this was the last attempt
		if !IsRetryable(err) || attempt == r.maxRetries {
			return nil, err
		}

		// Sleep with jitter, respecting context cancellation
		if !retryBackoff(ctx, backoff) {
			return nil, lastErr
		}

		// Exponential backoff: double, capped at max
		backoff *= 2
		if backoff > r.maxBackoff {
			backoff = r.maxBackoff
		}
	}

	return nil, lastErr
}

// DefaultProvider returns the active provider.
// Returns nil if no provider is configured.
func (r *Router) DefaultProvider() Provider {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.activeProvider
}

// ActiveProviderName returns the logical name of the active provider.
func (r *Router) ActiveProviderName() string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.activeProviderName
}

// ActiveModel returns the composite model identifier ("provider/model") of the
// currently active model. Use BareModel(ActiveModel()) to obtain the bare model
// name shown to users / sent to the LLM API.
func (r *Router) ActiveModel() string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.activeModel
}

// SetModel switches the active provider and model to the given model identifier.
//
// The identifier may be a composite "provider/model" (routes to the named
// provider, disambiguating models that share a bare name across providers) or a
// bare model name (resolved to the first matching provider for backward
// compatibility). When a bare name matches multiple providers, the first match
// (deterministic, sorted by composite ID) is selected and a warning is logged
// when a logger is configured.
func (r *Router) SetModel(ctx context.Context, model string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Composite identifier: direct lookup.
	if IsCompositeModelID(model) {
		providerName, ok := r.modelToProvider[model]
		if !ok {
			return fmt.Errorf("model %q is not enabled in any provider", model)
		}
		provider, ok := r.providers[providerName]
		if !ok {
			return fmt.Errorf("provider %q not found", providerName)
		}
		r.activeProvider = provider
		r.activeModel = model
		r.activeBareModel = BareModel(model)
		r.activeProviderName = providerName
		return nil
	}

	// Bare identifier: resolve to a composite ID via reverse scan of the index.
	var matches []string
	for id := range r.modelToProvider {
		if BareModel(id) == model {
			matches = append(matches, id)
		}
	}
	if len(matches) == 0 {
		return fmt.Errorf("model %q is not enabled in any provider", model)
	}
	sort.Strings(matches) // deterministic first-match when ambiguous
	compositeID := matches[0]
	if len(matches) > 1 && r.logger != nil {
		r.logger.Warn("model name is ambiguous across providers; selecting first match — use a composite \"provider/model\" identifier to disambiguate",
			"model", model, "selected", compositeID, "candidates", matches)
	}
	providerName := r.modelToProvider[compositeID]
	provider, ok := r.providers[providerName]
	if !ok {
		return fmt.Errorf("provider %q not found", providerName)
	}
	r.activeProvider = provider
	r.activeModel = compositeID
	r.activeBareModel = model
	r.activeProviderName = providerName
	return nil
}
