package llm

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"
)

// Sampling-parameter serialization tests: every provider must serialize the
// parameters its API supports, omit the ones it does not, and never emit
// nil fields into the wire payload.

// jsonMap marshals v and decodes it back into a generic object so tests can
// assert on the wire-level key set (proving omitempty behavior).
func jsonMap(t *testing.T, v any) map[string]any {
	t.Helper()
	raw, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatalf("unmarshal payload %s: %v", raw, err)
	}
	return m
}

func wantNum(t *testing.T, m map[string]any, key string, want float64) {
	t.Helper()
	got, ok := m[key]
	if !ok {
		t.Fatalf("expected %q in payload, got keys of %v", key, m)
	}
	n, ok := got.(float64)
	if !ok {
		t.Fatalf("expected %q to be a number, got %T (%v)", key, got, got)
	}
	if n != want {
		t.Errorf("expected %q=%v, got %v", key, want, n)
	}
}

func wantAbsent(t *testing.T, m map[string]any, keys ...string) {
	t.Helper()
	for _, key := range keys {
		if _, ok := m[key]; ok {
			t.Errorf("expected %q NOT in payload, got %v", key, m[key])
		}
	}
}

func fullSamplingRequest() ChatRequest {
	topK := 20
	return ChatRequest{
		Model:             "qwen3-coder",
		Messages:          []Message{{Role: "user", Content: "hi"}},
		Temperature:       floatPtr(0.6),
		TopP:              floatPtr(0.95),
		TopK:              &topK,
		RepetitionPenalty: floatPtr(1.05),
		PresencePenalty:   floatPtr(0.1),
	}
}

// chatBodyRecorder spins up an OpenAI-compatible chat completions endpoint
// that records the raw request body and replies with a minimal valid
// completion.
func chatBodyRecorder(t *testing.T) (*[]byte, *httptest.Server) {
	t.Helper()
	var body []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read request body: %v", err)
		}
		body = raw
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"ok"}}]}`))
	}))
	t.Cleanup(srv.Close)
	return &body, srv
}

func openAIProvider(t *testing.T, baseURL string) *OpenAIProvider {
	t.Helper()
	p, err := NewOpenAIProvider(OpenAIProviderConfig{
		Name:    "Zen",
		APIKey:  "k",
		BaseURL: baseURL,
		Logger:  slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	if err != nil {
		t.Fatalf("NewOpenAIProvider: %v", err)
	}
	return p
}

// --- OpenAI Chat Completions (compatible endpoints: LM Studio/vLLM/llama.cpp/Ollama) ---

func TestOpenAIChatSampling_CompatibleEndpointSendsAllParams(t *testing.T) {
	body, srv := chatBodyRecorder(t)
	p := openAIProvider(t, srv.URL)

	if _, err := p.ChatCompletion(context.Background(), fullSamplingRequest()); err != nil {
		t.Fatalf("ChatCompletion: %v", err)
	}

	var payload map[string]any
	if err := json.Unmarshal(*body, &payload); err != nil {
		t.Fatalf("unmarshal request body: %v", err)
	}
	wantNum(t, payload, "temperature", 0.6)
	wantNum(t, payload, "top_p", 0.95)
	wantNum(t, payload, "top_k", 20)
	wantNum(t, payload, "repetition_penalty", 1.05)
	wantNum(t, payload, "presence_penalty", 0.1)
}

// The strict official api.openai.com endpoint rejects unknown request fields,
// so top_k/repetition_penalty (not part of the OpenAI schema) must never be
// serialized when no custom baseURL is configured.
func TestOpenAIChatSampling_StrictOfficialEndpointDropsNonSchemaParams(t *testing.T) {
	p := openAIProvider(t, "") // no baseURL → official endpoint
	if p.baseURL != "" {
		t.Fatalf("expected empty baseURL for strict endpoint, got %q", p.baseURL)
	}

	params := p.buildChatParams(fullSamplingRequest())
	payload := jsonMap(t, params)

	wantNum(t, payload, "temperature", 0.6)
	wantNum(t, payload, "top_p", 0.95)
	wantNum(t, payload, "presence_penalty", 0.1)
	wantAbsent(t, payload, "top_k", "repetition_penalty")
}

func TestOpenAIChatSampling_NilFieldsOmitted(t *testing.T) {
	body, srv := chatBodyRecorder(t)
	p := openAIProvider(t, srv.URL)

	if _, err := p.ChatCompletion(context.Background(), ChatRequest{
		Model:    "qwen3-coder",
		Messages: []Message{{Role: "user", Content: "hi"}},
	}); err != nil {
		t.Fatalf("ChatCompletion: %v", err)
	}

	var payload map[string]any
	if err := json.Unmarshal(*body, &payload); err != nil {
		t.Fatalf("unmarshal request body: %v", err)
	}
	wantAbsent(t, payload, "temperature", "top_p", "top_k", "repetition_penalty", "presence_penalty")
}

// Regression guard: the OpenAI SDK's SetExtraFields replaces the whole map,
// so the sampling extras must not clobber the reasoning extras (or vice
// versa) — both "enable_thinking" and "top_k" have to survive in one payload.
func TestOpenAIChatSampling_MergesWithReasoningExtras(t *testing.T) {
	body, srv := chatBodyRecorder(t)
	p := openAIProvider(t, srv.URL)

	req := fullSamplingRequest()
	req.ReasoningEffort = "On"
	if _, err := p.ChatCompletion(context.Background(), req); err != nil {
		t.Fatalf("ChatCompletion: %v", err)
	}

	var payload map[string]any
	if err := json.Unmarshal(*body, &payload); err != nil {
		t.Fatalf("unmarshal request body: %v", err)
	}
	if _, ok := payload["enable_thinking"]; !ok {
		t.Errorf("expected reasoning extra \"enable_thinking\" in payload, got keys of %v", payload)
	}
	wantNum(t, payload, "top_k", 20)
}

// --- OpenAI Responses API ---

// The Responses API supports temperature and top_p only; penalties and top_k
// are not part of its schema and must never be serialized.
func TestResponsesSamplingParams_SerializedAndUnsupportedOmitted(t *testing.T) {
	params := buildResponsesParams(fullSamplingRequest(), "https://api.openai.com/v1", nil)
	payload := jsonMap(t, params)

	wantNum(t, payload, "temperature", 0.6)
	wantNum(t, payload, "top_p", 0.95)
	wantAbsent(t, payload, "top_k", "repetition_penalty", "presence_penalty")
}

func TestResponsesSamplingParams_NilFieldsOmitted(t *testing.T) {
	params := buildResponsesParams(ChatRequest{
		Model:    "gpt-5.1",
		Messages: []Message{{Role: "user", Content: "hi"}},
	}, "https://api.openai.com/v1", nil)
	payload := jsonMap(t, params)

	wantAbsent(t, payload, "temperature", "top_p", "top_k", "repetition_penalty", "presence_penalty")
}

// --- Anthropic Messages API ---

func anthropicProvider(t *testing.T) *AnthropicProvider {
	t.Helper()
	p, err := NewAnthropicProvider(AnthropicProviderConfig{Name: "anthropic", APIKey: "k"})
	if err != nil {
		t.Fatalf("NewAnthropicProvider: %v", err)
	}
	return p
}

// Anthropic accepts temperature/top_p/top_k; OpenAI-style penalty knobs are
// not part of the Messages API and must never be serialized.
func TestAnthropicSamplingParams_SerializedAndUnsupportedOmitted(t *testing.T) {
	req := fullSamplingRequest()
	req.Model = "claude-sonnet-4-5"
	anthropicReq, err := anthropicProvider(t).buildRequest(req)
	if err != nil {
		t.Fatalf("buildRequest: %v", err)
	}
	payload := jsonMap(t, anthropicReq)

	wantNum(t, payload, "temperature", 0.6)
	wantNum(t, payload, "top_p", 0.95)
	wantNum(t, payload, "top_k", 20)
	wantAbsent(t, payload, "repetition_penalty", "presence_penalty")
}

// With extended thinking, Anthropic requires temperature to be unset and
// equally rejects top_p/top_k — all sampling knobs are dropped.
func TestAnthropicSamplingParams_ThinkingDropsSampling(t *testing.T) {
	req := fullSamplingRequest()
	req.Model = "claude-sonnet-4-5"
	req.ReasoningEffort = "On"
	req.MaxTokens = 8192 // thinking budget must fit (half of max tokens ≥ 1024)
	anthropicReq, err := anthropicProvider(t).buildRequest(req)
	if err != nil {
		t.Fatalf("buildRequest: %v", err)
	}
	payload := jsonMap(t, anthropicReq)

	wantAbsent(t, payload, "temperature", "top_p", "top_k", "repetition_penalty", "presence_penalty")
}

func TestAnthropicSamplingParams_NilFieldsOmitted(t *testing.T) {
	anthropicReq, err := anthropicProvider(t).buildRequest(ChatRequest{
		Model:    "claude-sonnet-4-5",
		Messages: []Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("buildRequest: %v", err)
	}
	payload := jsonMap(t, anthropicReq)

	wantAbsent(t, payload, "temperature", "top_p", "top_k", "repetition_penalty", "presence_penalty")
}

// --- Google Gemini API ---

// Gemini accepts temperature/topP/topK in generationConfig; it has no
// penalty knobs, which must never be serialized.
func TestGoogleSamplingParams_SerializedAndUnsupportedOmitted(t *testing.T) {
	req := fullSamplingRequest()
	req.Model = "gemini-2.5-pro"
	out := buildGoogleRequest(req)
	if out.GenerationConfig == nil {
		t.Fatal("expected generationConfig to be set")
	}
	payload := jsonMap(t, out.GenerationConfig)

	wantNum(t, payload, "temperature", 0.6)
	wantNum(t, payload, "topP", 0.95)
	wantNum(t, payload, "topK", 20)
	wantAbsent(t, payload, "repetition_penalty", "presence_penalty", "presencePenalty")
}

func TestGoogleSamplingParams_NilFieldsOmitted(t *testing.T) {
	out := buildGoogleRequest(ChatRequest{
		Model:    "gemini-2.5-pro",
		Messages: []Message{{Role: "user", Content: "hi"}},
	})
	if out.GenerationConfig != nil {
		t.Fatalf("expected no generationConfig without sampling params/max tokens, got %v", out.GenerationConfig)
	}
}

// --- Router preset application ---

func TestRouterApplyDefaultSampling_Priority(t *testing.T) {
	topK := 40
	preset := SamplingDefaults{
		Temperature:       floatPtr(0.7),
		TopP:              floatPtr(0.9),
		TopK:              &topK,
		RepetitionPenalty: floatPtr(1.1),
		PresencePenalty:   floatPtr(0.2),
	}
	r := &Router{
		sampling: func(family string) SamplingDefaults { return preset },
	}
	meta := ModelMetadata{Family: "qwen"}

	req := ChatRequest{
		Temperature: floatPtr(0.1), // explicit — beats preset
		TopP:        floatPtr(0.5), // explicit — beats preset
	}
	r.applyDefaultSampling(&req, meta)

	if *req.Temperature != 0.1 {
		t.Errorf("explicit temperature must win, got %v", *req.Temperature)
	}
	if *req.TopP != 0.5 {
		t.Errorf("explicit top_p must win, got %v", *req.TopP)
	}
	if req.TopK == nil || *req.TopK != 40 {
		t.Errorf("expected preset top_k 40, got %v", req.TopK)
	}
	if req.RepetitionPenalty == nil || *req.RepetitionPenalty != 1.1 {
		t.Errorf("expected preset repetition_penalty 1.1, got %v", req.RepetitionPenalty)
	}
	if req.PresencePenalty == nil || *req.PresencePenalty != 0.2 {
		t.Errorf("expected preset presence_penalty 0.2, got %v", req.PresencePenalty)
	}
}

func TestRouterApplyDefaultSampling_NoSamplingFuncFallsBackToDeterministic(t *testing.T) {
	r := &Router{} // no sampling func
	req := ChatRequest{}
	r.applyDefaultSampling(&req, ModelMetadata{Family: "openai"})

	if req.Temperature == nil || *req.Temperature != 0.0 {
		t.Errorf("expected deterministic fallback temperature 0.0, got %v", req.Temperature)
	}
	if req.TopP != nil {
		t.Errorf("expected top_p to stay nil without sampling func, got %v", *req.TopP)
	}
	if req.TopK != nil {
		t.Errorf("expected top_k to stay nil without sampling func, got %v", *req.TopK)
	}
	if req.RepetitionPenalty != nil {
		t.Errorf("expected repetition_penalty to stay nil without sampling func, got %v", *req.RepetitionPenalty)
	}
	if req.PresencePenalty != nil {
		t.Errorf("expected presence_penalty to stay nil without sampling func, got %v", *req.PresencePenalty)
	}
}

func TestRouterApplyDefaultSampling_AllNilPresetKeepsProviderDefaults(t *testing.T) {
	r := &Router{
		sampling: func(family string) SamplingDefaults { return SamplingDefaults{} },
	}
	req := ChatRequest{}
	r.applyDefaultSampling(&req, ModelMetadata{Family: "anthropic"})

	if req.Temperature != nil || req.TopP != nil || req.TopK != nil ||
		req.RepetitionPenalty != nil || req.PresencePenalty != nil {
		t.Errorf("expected all sampling fields to stay nil for all-nil preset, got %+v", req)
	}
}

// --- Call-purpose policy ---

// fullVendorPreset returns a preset with every knob set, so any accidental
// leak into a deterministic call is detectable.
func fullVendorPreset() SamplingDefaults {
	topK := 20
	return SamplingDefaults{
		Temperature:       floatPtr(0.8),
		TopP:              floatPtr(0.95),
		TopK:              &topK,
		RepetitionPenalty: floatPtr(1.15),
		PresencePenalty:   floatPtr(0.3),
	}
}

func presetRouter() *Router {
	return &Router{
		sampling: func(family string) SamplingDefaults { return fullVendorPreset() },
	}
}

// A routing call must stay deterministic regardless of how creative the
// vendor preset is: temperature 0.0 and no top_p/top_k/penalty injection.
func TestRouterApplyDefaultSampling_RoutingPurposeDeterministicDespitePreset(t *testing.T) {
	r := presetRouter()
	req := ChatRequest{CallPurpose: CallPurposeRouting}
	r.applyDefaultSampling(&req, ModelMetadata{Family: "deepseek"})

	if req.Temperature == nil || *req.Temperature != 0.0 {
		t.Errorf("routing call must get deterministic temperature 0.0, got %v", req.Temperature)
	}
	if req.TopP != nil {
		t.Errorf("routing call must not inherit preset top_p, got %v", *req.TopP)
	}
	if req.TopK != nil {
		t.Errorf("routing call must not inherit preset top_k, got %v", *req.TopK)
	}
	if req.RepetitionPenalty != nil {
		t.Errorf("routing call must not inherit preset repetition_penalty, got %v", *req.RepetitionPenalty)
	}
	if req.PresencePenalty != nil {
		t.Errorf("routing call must not inherit preset presence_penalty, got %v", *req.PresencePenalty)
	}
}

// Compaction and summarization calls follow the same deterministic policy.
func TestRouterApplyDefaultSampling_CompactionAndSummarizationDeterministic(t *testing.T) {
	r := presetRouter()

	compaction := ChatRequest{CallPurpose: CallPurposeCompaction}
	r.applyDefaultSampling(&compaction, ModelMetadata{Family: "openai"})
	if compaction.Temperature == nil || *compaction.Temperature != 0.0 {
		t.Errorf("compaction call must get deterministic temperature 0.0, got %v", compaction.Temperature)
	}
	if compaction.TopP != nil || compaction.TopK != nil {
		t.Errorf("compaction call must not inherit preset top_p/top_k, got %v/%v", compaction.TopP, compaction.TopK)
	}

	summarization := ChatRequest{CallPurpose: CallPurposeSummarization}
	r.applyDefaultSampling(&summarization, ModelMetadata{Family: "mistral"})
	if summarization.Temperature == nil || *summarization.Temperature != 0.0 {
		t.Errorf("summarization call must get deterministic temperature 0.0, got %v", summarization.Temperature)
	}
	if summarization.RepetitionPenalty != nil || summarization.PresencePenalty != nil {
		t.Errorf("summarization call must not inherit preset penalties, got %+v", summarization)
	}
}

// The executor call is the one class entitled to the full vendor preset.
func TestRouterApplyDefaultSampling_ExecutorPurposeGetsFullVendorPreset(t *testing.T) {
	r := presetRouter()
	req := ChatRequest{CallPurpose: CallPurposeExecutor}
	r.applyDefaultSampling(&req, ModelMetadata{Family: "qwen"})

	if req.Temperature == nil || *req.Temperature != 0.8 {
		t.Errorf("executor call must get preset temperature 0.8, got %v", req.Temperature)
	}
	if req.TopP == nil || *req.TopP != 0.95 {
		t.Errorf("executor call must get preset top_p 0.95, got %v", req.TopP)
	}
	if req.TopK == nil || *req.TopK != 20 {
		t.Errorf("executor call must get preset top_k 20, got %v", req.TopK)
	}
	if req.RepetitionPenalty == nil || *req.RepetitionPenalty != 1.15 {
		t.Errorf("executor call must get preset repetition_penalty 1.15, got %v", req.RepetitionPenalty)
	}
	if req.PresencePenalty == nil || *req.PresencePenalty != 0.3 {
		t.Errorf("executor call must get preset presence_penalty 0.3, got %v", req.PresencePenalty)
	}
}

// Requests with no declared purpose keep the vendor preset — backward
// compatibility for hosts that never set CallPurpose.
func TestRouterApplyDefaultSampling_DefaultPurposeKeepsVendorPreset(t *testing.T) {
	r := presetRouter()
	req := ChatRequest{}
	r.applyDefaultSampling(&req, ModelMetadata{Family: "qwen"})

	if req.Temperature == nil || *req.Temperature != 0.8 {
		t.Errorf("purpose-less call must keep preset temperature 0.8, got %v", req.Temperature)
	}
	if req.TopP == nil || *req.TopP != 0.95 {
		t.Errorf("purpose-less call must keep preset top_p 0.95, got %v", req.TopP)
	}
}

// Families whose vendors document instability at low temperature get their
// documented floor instead of 0.0 on deterministic calls.
func TestRouterApplyDefaultSampling_FamilySafeFloor(t *testing.T) {
	cases := []struct {
		family string
		want   float64
	}{
		{"deepseek", 0.0},
		{"openai", 0.0},
		{"anthropic", 0.0},
		{"google", 1.0}, // Gemini loops at low temperature
		{"qwen", 0.6},   // Qwen3 thinking floor
	}
	for _, tc := range cases {
		req := ChatRequest{CallPurpose: CallPurposeRouting}
		presetRouter().applyDefaultSampling(&req, ModelMetadata{Family: tc.family})
		if req.Temperature == nil || *req.Temperature != tc.want {
			t.Errorf("family %s: want deterministic temperature %v, got %v", tc.family, tc.want, req.Temperature)
		}
	}
}

// An explicit caller temperature still wins over the deterministic profile.
func TestRouterApplyDefaultSampling_RoutingPurposeExplicitTemperatureWins(t *testing.T) {
	r := presetRouter()
	req := ChatRequest{CallPurpose: CallPurposeRouting, Temperature: floatPtr(0.3)}
	r.applyDefaultSampling(&req, ModelMetadata{Family: "google"})

	if req.Temperature == nil || *req.Temperature != 0.3 {
		t.Errorf("explicit temperature must beat deterministic profile, got %v", req.Temperature)
	}
}

// Models that reject the temperature parameter are skipped entirely — even
// for deterministic purposes, no sampling field is injected.
func TestRouterApplyDefaultSampling_RoutingPurposeReasoningModelSkipped(t *testing.T) {
	r := presetRouter()
	r.registry = &ModelRegistry{}
	req := ChatRequest{CallPurpose: CallPurposeRouting}
	r.applyDefaultSampling(&req, ModelMetadata{Family: "openai_flagship", Capabilities: &ModelCapabilities{Temperature: false}})

	if req.Temperature != nil || req.TopP != nil || req.TopK != nil ||
		req.RepetitionPenalty != nil || req.PresencePenalty != nil {
		t.Errorf("reasoning model must receive no sampling fields on routing calls, got %+v", req)
	}
}

// DeterministicTemperature pins the family floors the router relies on.
func TestDeterministicTemperature(t *testing.T) {
	cases := map[string]float64{
		"":            0.0,
		"deepseek":    0.0,
		"openai":      0.0,
		"anthropic":   0.0,
		"google":      1.0,
		"qwen":        0.6,
		"made-up-fam": 0.0,
	}
	for family, want := range cases {
		got := DeterministicTemperature(family)
		if got == nil || *got != want {
			t.Errorf("DeterministicTemperature(%q): want %v, got %v", family, want, got)
		}
	}
}
