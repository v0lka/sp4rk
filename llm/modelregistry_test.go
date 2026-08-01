package llm

import (
	"context"
	"net/http"
	"net/http/httptest"
	"net/url"
	"sync"
	"testing"
)

func TestModelRegistry_OverridePriority(t *testing.T) {
	// Create registry with override for a built-in model
	overrides := map[string]ModelMetadata{
		"gpt-4o": {
			ContextWindow: 999999,
			OutputLimit:   8888,
			TokenizerType: "custom-tokenizer",
		},
	}

	registry := NewModelRegistry(overrides)

	// Override should take priority over built-in
	meta, ok := registry.Resolve(context.Background(), "gpt-4o")
	if !ok {
		t.Fatal("expected ok=true for override model")
	}

	if meta.ContextWindow != 999999 {
		t.Errorf("expected ContextWindow 999999, got %d", meta.ContextWindow)
	}
	if meta.OutputLimit != 8888 {
		t.Errorf("expected OutputLimit 8888, got %d", meta.OutputLimit)
	}
	if meta.TokenizerType != "custom-tokenizer" {
		t.Errorf("expected TokenizerType 'custom-tokenizer', got %s", meta.TokenizerType)
	}
}

func TestModelRegistry_BuiltInResolution(t *testing.T) {
	registry := NewModelRegistry(nil)

	tests := []struct {
		model                 string
		expectedContextWindow int
		expectedOutputLimit   int
		expectedTokenizer     string
	}{
		// OpenAI models — verified July 2026 from platform.openai.com/docs/models
		{"gpt-5.4", 1050000, 128000, "tiktoken/o200k_base"},
		{"gpt-4o", 128000, 16384, "tiktoken/o200k_base"},
		{"o3-mini", 200000, 100000, "tiktoken/o200k_base"},

		// OpenAI Codex models
		{"codex-mini-latest", 200000, 100000, "tiktoken/o200k_base"},

		// Anthropic models — verified July 2026 from platform.claude.com/docs
		{"claude-opus-4-6", 1000000, 128000, "anthropic-api"},
		{"claude-3.5-sonnet", 200000, 8192, "anthropic-api"},

		// Gemini models
		{"gemini-2.5-pro", 1048576, 65536, "approximate"},
		{"gemini-2.0-flash", 1048576, 8192, "approximate"},

		// DeepSeek V4 models
		{"deepseek-v4-pro", 1000000, 384000, "approximate"},
		{"deepseek-v4-flash", 1000000, 384000, "approximate"},

		// Grok models — verified from docs.x.ai
		{"grok-4.20", 1000000, 32768, "approximate"},
		{"grok-3-mini", 131072, 32768, "approximate"},

		// GLM models (Zhipu AI) — verified from docs.z.ai
		{"glm-5.2", 1000000, 128000, "approximate"},
		{"glm-5.1", 200000, 128000, "approximate"},
		{"glm-5", 200000, 128000, "approximate"},
	}

	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			meta, _ := registry.Resolve(context.Background(), tt.model)

			if meta.ContextWindow != tt.expectedContextWindow {
				t.Errorf("expected ContextWindow %d, got %d", tt.expectedContextWindow, meta.ContextWindow)
			}
			if meta.OutputLimit != tt.expectedOutputLimit {
				t.Errorf("expected OutputLimit %d, got %d", tt.expectedOutputLimit, meta.OutputLimit)
			}
			if meta.TokenizerType != tt.expectedTokenizer {
				t.Errorf("expected TokenizerType %s, got %s", tt.expectedTokenizer, meta.TokenizerType)
			}
		})
	}
}

func TestModelRegistry_FallbackForUnknownModel(t *testing.T) {
	registry := NewModelRegistry(nil)

	// Unknown model should return fallback defaults
	meta, ok := registry.Resolve(context.Background(), "unknown-model-v123")
	if ok {
		t.Fatal("expected ok=false for unknown model")
	}

	expected := ModelMetadata{
		ContextWindow: 128000,
		OutputLimit:   4096,
		TokenizerType: "approximate",
		// Unknown models default to optimistic attachment support: better to
		// surface a runtime provider error than to deny image uploads.
		Capabilities: ModelCapabilities{Attachment: true},
	}

	if meta.ContextWindow != expected.ContextWindow {
		t.Errorf("expected ContextWindow %d, got %d", expected.ContextWindow, meta.ContextWindow)
	}
	if meta.OutputLimit != expected.OutputLimit {
		t.Errorf("expected OutputLimit %d, got %d", expected.OutputLimit, meta.OutputLimit)
	}
	if meta.TokenizerType != expected.TokenizerType {
		t.Errorf("expected TokenizerType %s, got %s", expected.TokenizerType, meta.TokenizerType)
	}
	if meta.Capabilities != expected.Capabilities {
		t.Errorf("expected Capabilities %+v, got %+v", expected.Capabilities, meta.Capabilities)
	}
}

func TestModelRegistry_Invalidate(t *testing.T) {
	registry := NewModelRegistry(nil)

	// Manually add an entry to the cache
	registry.mu.Lock()
	registry.cache["cached-model"] = ModelMetadata{
		ContextWindow: 50000,
		OutputLimit:   2000,
		TokenizerType: "cached-tokenizer",
	}
	registry.mu.Unlock()

	// Verify it's in cache
	registry.mu.RLock()
	_, exists := registry.cache["cached-model"]
	registry.mu.RUnlock()

	if !exists {
		t.Fatal("cached model should exist before invalidation")
	}

	// Invalidate the cache entry
	registry.Invalidate("cached-model")

	// Verify it's removed from cache
	registry.mu.RLock()
	_, exists = registry.cache["cached-model"]
	registry.mu.RUnlock()

	if exists {
		t.Error("cached model should not exist after invalidation")
	}
}

func TestModelRegistry_CaseInsensitiveResolve(t *testing.T) {
	registry := NewModelRegistry(nil)

	// Built-in keys are stored lowercase (e.g. "gpt-5.4"). A model ID arriving
	// from an OpenAI-compatible host may differ only in casing. All casings
	// must resolve to the same metadata as the canonical lowercase key.
	//
	// This mirrors the real-world vLLM scenario where models are registered
	// with mixed-case HF identifiers (e.g. "Qwen/Qwen35-397B-A17B-FP8").
	tests := []struct {
		name  string
		model string
	}{
		{"lowercase canonical", "gpt-5.4"},
		{"uppercase", "GPT-5.4"},
		{"mixed case", "Gpt-5.4"},
	}

	var canonical ModelMetadata
	for i, tt := range tests {
		meta, ok := registry.Resolve(context.Background(), tt.model)
		if !ok {
			t.Fatalf("%s: expected ok=true for %q", tt.name, tt.model)
		}
		if i == 0 {
			canonical = meta
			continue
		}
		if meta.ContextWindow != canonical.ContextWindow {
			t.Errorf("%s: ContextWindow mismatch: got %d, want %d", tt.name, meta.ContextWindow, canonical.ContextWindow)
		}
		if meta.OutputLimit != canonical.OutputLimit {
			t.Errorf("%s: OutputLimit mismatch: got %d, want %d", tt.name, meta.OutputLimit, canonical.OutputLimit)
		}
		if meta.Family != canonical.Family {
			t.Errorf("%s: Family mismatch: got %q, want %q", tt.name, meta.Family, canonical.Family)
		}
	}
}

func TestModelRegistry_CaseInsensitiveOverride(t *testing.T) {
	// An override registered with a mixed-case key must be found regardless of
	// the casing used at Resolve time.
	overrides := map[string]ModelMetadata{
		"Zai-Org/GLM-5.2-FP8": {
			ContextWindow: 1048576,
			OutputLimit:   8192,
		},
	}
	registry := NewModelRegistry(overrides)

	// Resolve with the exact mixed-case, then with a different casing.
	for _, model := range []string{"zai-org/GLM-5.2-FP8", "ZAI-ORG/GLM-5.2-FP8", "Zai-Org/GLM-5.2-FP8"} {
		meta, ok := registry.Resolve(context.Background(), model)
		if !ok {
			t.Fatalf("expected ok=true for %q", model)
		}
		if meta.ContextWindow != 1048576 {
			t.Errorf("Resolve(%q): ContextWindow = %d, want 1048576", model, meta.ContextWindow)
		}
	}
}

func TestModelRegistry_CaseInsensitiveSetCachedInvalidate(t *testing.T) {
	registry := NewModelRegistry(nil)

	// SetCachedMetadata with mixed case, then Resolve with a different case.
	registry.SetCachedMetadata("Qwen/Qwen35-397B-A17B-FP8", ModelMetadata{
		ContextWindow: 262144,
		OutputLimit:   8192,
	})

	meta, ok := registry.Resolve(context.Background(), "qwen/qwen35-397b-a17b-fp8")
	if !ok {
		t.Fatal("expected ok=true for case-insensitive cache lookup")
	}
	if meta.ContextWindow != 262144 {
		t.Errorf("ContextWindow = %d, want 262144", meta.ContextWindow)
	}

	// Invalidate with yet another casing must remove the entry.
	registry.Invalidate("QWEN/Qwen35-397B-A17B-FP8")

	// After invalidation, the lowercased key should no longer hit the cache.
	registry.mu.RLock()
	_, exists := registry.cache["qwen/qwen35-397b-a17b-fp8"]
	registry.mu.RUnlock()
	if exists {
		t.Error("cache entry should be removed after case-insensitive Invalidate")
	}
}

func TestModelRegistry_FuzzyMatch_BuiltInDottedToNoDot(t *testing.T) {
	// Real-world scenario: a host serves "Qwen/Qwen36-35B-A3B-FP8" for the
	// registry key "qwen/qwen3.6-35b-a3b-fp8" (dot dropped). Case is already
	// handled; this exercises separator-insensitive matching.
	registry := NewModelRegistry(nil)

	meta, ok := registry.Resolve(context.Background(), "Qwen/Qwen36-35B-A3B-FP8")
	if !ok {
		t.Fatal("expected ok=true for fuzzy (separator-insensitive) built-in match")
	}
	if meta.ContextWindow != 262144 {
		t.Errorf("ContextWindow = %d, want 262144", meta.ContextWindow)
	}
	if !meta.Capabilities.Attachment {
		t.Error("fuzzy-matched built-in should keep its declared Attachment capability")
	}
}

func TestModelRegistry_FuzzyMatch_Table(t *testing.T) {
	// Covers separator variations that should collapse to the same model, plus
	// negative cases that must NOT spuriously match (different version, org,
	// or capacity).
	registry := NewModelRegistry(nil)

	tests := []struct {
		name    string
		model   string
		wantOK  bool
		wantCtx int
	}{
		{"dotted key, no-dot query", "qwen/qwen36-35b-a3b-fp8", true, 262144},
		{"underscores instead of dots/dashes", "qwen/qwen3_6_35b_a3b_fp8", true, 262144},
		{"plain qwen3.6 built-in via dashes", "qwen/qwen36-35b-a3b", true, 262144},
		{"slashes kept, dots/dashes removed", "qwen/qwen3635ba3bfp8", true, 262144},
		// Negatives: org boundary removed entirely -> must NOT match.
		{"org boundary dropped does not match", "qwenqwen3635ba3bfp8", false, 0},
		// Negatives: a different minor version must NOT match qwen3.6.
		{"different version does not match", "qwen/qwen3.7-35b-a3b-fp8", false, 0},
		{"different capacity does not match", "qwen/qwen3.6-27b-a3b-fp8", false, 0},
		// "gpt4o" should fuzzy-match the built-in "gpt-4o" (dash removed).
		{"gpt4o matches gpt-4o", "gpt4o", true, 128000},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			meta, ok := registry.Resolve(context.Background(), tt.model)
			if ok != tt.wantOK {
				t.Fatalf("ok = %v, want %v", ok, tt.wantOK)
			}
			if ok && meta.ContextWindow != tt.wantCtx {
				t.Errorf("ContextWindow = %d, want %d", meta.ContextWindow, tt.wantCtx)
			}
		})
	}
}

func TestModelRegistry_FuzzyMatch_CachesUnderQueryKey(t *testing.T) {
	// A fuzzy hit must be cached under the (lowercased) query key so repeat
	// resolves take the fast cache path and don't re-run the fuzzy scan.
	registry := NewModelRegistry(nil)

	if _, ok := registry.Resolve(context.Background(), "qwen/qwen36-35b-a3b-fp8"); !ok {
		t.Fatal("expected fuzzy match")
	}

	registry.mu.RLock()
	_, cached := registry.cache["qwen/qwen36-35b-a3b-fp8"]
	registry.mu.RUnlock()
	if !cached {
		t.Error("fuzzy match should be cached under the lowercased query key")
	}
}

func TestModelRegistry_FuzzyMatch_OverrideStillWins(t *testing.T) {
	// A user override that only fuzzy-matches (separators differ from the
	// query) must still win over a closer-built-in fuzzy match.
	registry := NewModelRegistry(map[string]ModelMetadata{
		"qwen/qwen3.6-35b-a3b-fp8": {
			ContextWindow: 999999, // sentinel: proves the override, not built-in, won
		},
	})

	meta, ok := registry.Resolve(context.Background(), "qwen/qwen36-35b-a3b-fp8")
	if !ok {
		t.Fatal("expected ok=true")
	}
	if meta.ContextWindow != 999999 {
		t.Errorf("ContextWindow = %d, want override sentinel 999999", meta.ContextWindow)
	}
}

func TestModelRegistry_FuzzyMatch_DoesNotCollideVersions(t *testing.T) {
	// Guard against the core risk of fuzzy matching: collapsing distinct model
	// versions together. "qwen3.6" (-> qwen36) and "qwen3.7" (-> qwen37) must
	// resolve to different metadata. Both are in the built-in registry.
	registry := NewModelRegistry(nil)

	m36, ok36 := registry.Resolve(context.Background(), "qwen/qwen3.6-35b-a3b-fp8")
	m37, ok37 := registry.Resolve(context.Background(), "qwen3.7-max")

	if !ok36 || !ok37 {
		t.Fatalf("both should resolve; got ok36=%v ok37=%v", ok36, ok37)
	}
	// Built-in qwen3.6-35b-a3b-fp8 has OutputLimit 8192; qwen3.7-max has 65536.
	// They must not have been swapped or merged.
	if m36.OutputLimit == m37.OutputLimit {
		t.Errorf("distinct versions resolved to identical OutputLimit %d — possible collision",
			m36.OutputLimit)
	}
}

func TestModelRegistry_HuggingFace_AttachmentDefault(t *testing.T) {
	// A model resolved from HuggingFace config.json has no capability data,
	// so it should get the optimistic Attachment default.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"max_position_embeddings": 8192}`))
	}))
	defer server.Close()

	registry := NewModelRegistry(nil)
	registry.httpClient = &http.Client{
		Transport: &rewriteTransport{base: http.DefaultTransport, serverURL: server.URL},
	}

	meta, ok := registry.Resolve(context.Background(), "hf-unknown-model")
	if !ok {
		t.Fatal("expected ok=true for HuggingFace-resolved model")
	}
	if !meta.Capabilities.Attachment {
		t.Error("HuggingFace-resolved model should default to Attachment=true")
	}
	if meta.ContextWindow != 8192 {
		t.Errorf("ContextWindow = %d, want 8192", meta.ContextWindow)
	}
}

func TestModelRegistry_ThreadSafe(t *testing.T) {
	registry := NewModelRegistry(nil)

	// Run multiple goroutines concurrently accessing Resolve
	var wg sync.WaitGroup
	numGoroutines := 100
	numIterations := 50

	// Test concurrent reads of built-in models
	wg.Add(numGoroutines)
	for i := 0; i < numGoroutines; i++ {
		go func() {
			defer wg.Done()
			for j := 0; j < numIterations; j++ {
				_, _ = registry.Resolve(context.Background(), "gpt-4o")
				_, _ = registry.Resolve(context.Background(), "claude-opus-4-6")
				_, _ = registry.Resolve(context.Background(), "unknown-model")
			}
		}()
	}

	// Test concurrent cache invalidations
	wg.Add(numGoroutines / 2)
	for i := 0; i < numGoroutines/2; i++ {
		go func(id int) {
			defer wg.Done()
			for j := 0; j < numIterations; j++ {
				registry.Invalidate("nonexistent-model")
			}
		}(i)
	}

	wg.Wait()

	// If we get here without panic or data race, the test passes
}

func TestModelRegistry_OverrideUnknownModel(t *testing.T) {
	// Create registry with override for a model not in built-in
	overrides := map[string]ModelMetadata{
		"custom-model": {
			ContextWindow: 50000,
			OutputLimit:   2000,
			TokenizerType: "custom",
		},
	}

	registry := NewModelRegistry(overrides)

	meta, _ := registry.Resolve(context.Background(), "custom-model")

	if meta.ContextWindow != 50000 {
		t.Errorf("expected ContextWindow 50000, got %d", meta.ContextWindow)
	}
	if meta.OutputLimit != 2000 {
		t.Errorf("expected OutputLimit 2000, got %d", meta.OutputLimit)
	}
	if meta.TokenizerType != "custom" {
		t.Errorf("expected TokenizerType 'custom', got %s", meta.TokenizerType)
	}
}

func TestModelRegistry_NilOverrides(t *testing.T) {
	// Test that nil overrides doesn't cause panic
	registry := NewModelRegistry(nil)

	meta, _ := registry.Resolve(context.Background(), "gpt-4o")

	if meta.ContextWindow != 128000 {
		t.Errorf("expected ContextWindow 128000, got %d", meta.ContextWindow)
	}
}

func TestModelRegistry_EmptyOverrides(t *testing.T) {
	// Test that empty overrides map works correctly
	registry := NewModelRegistry(map[string]ModelMetadata{})

	meta, _ := registry.Resolve(context.Background(), "gpt-4o")

	if meta.ContextWindow != 128000 {
		t.Errorf("expected ContextWindow 128000, got %d", meta.ContextWindow)
	}
}

func TestModelRegistry_RegisteredSource(t *testing.T) {
	// Create registry with no overrides and no built-in match for test model
	registry := NewModelRegistry(nil)

	// Register a source that returns known metadata for a test model
	testModel := "test-source-model-v1"
	expectedMeta := ModelMetadata{
		ContextWindow: 65536,
		OutputLimit:   2048,
		TokenizerType: "test-tokenizer",
	}

	registry.RegisterSource(func(model string) (ModelMetadata, bool) {
		if model == testModel {
			return expectedMeta, true
		}
		return ModelMetadata{}, false
	})

	// Resolve should use the registered source
	meta, ok := registry.Resolve(context.Background(), testModel)
	if !ok {
		t.Fatal("expected ok=true for registered source model")
	}

	if meta.ContextWindow != expectedMeta.ContextWindow {
		t.Errorf("expected ContextWindow %d, got %d", expectedMeta.ContextWindow, meta.ContextWindow)
	}
	if meta.OutputLimit != expectedMeta.OutputLimit {
		t.Errorf("expected OutputLimit %d, got %d", expectedMeta.OutputLimit, meta.OutputLimit)
	}
	if meta.TokenizerType != expectedMeta.TokenizerType {
		t.Errorf("expected TokenizerType %q, got %q", expectedMeta.TokenizerType, meta.TokenizerType)
	}
}

// TestResolveFamily_BuiltinModels verifies that known built-in models get the correct family.
func TestResolveFamily_BuiltinModels(t *testing.T) {
	registry := NewModelRegistry(nil)

	tests := []struct {
		model          string
		expectedFamily string
	}{
		// OpenAI flagship models
		{"gpt-5.4", "openai_flagship"},
		{"gpt-5.4-mini", "openai_flagship"},
		{"gpt-5", "openai_flagship"},
		{"gpt-4o", "openai_flagship"},
		{"gpt-4o-mini", "openai_flagship"},
		{"o4-mini", "openai_flagship"},
		{"o3", "openai_flagship"},
		{"o3-mini", "openai_flagship"},
		{"o1", "openai_flagship"},
		{"o1-mini", "openai_flagship"},

		// OpenAI standard models
		{"gpt-4.1", "openai_standard"},

		// OpenAI Codex models
		{"codex-mini-latest", "openai_codex"},

		// Anthropic models
		{"claude-opus-4-6", "anthropic"},
		{"claude-sonnet-4-5", "anthropic"},
		{"claude-haiku-4-5", "anthropic"},
		{"claude-3.5-sonnet", "anthropic"},
		{"claude-3.5-haiku", "anthropic"},

		// Gemini models
		{"gemini-3.1-pro", "google"},
		{"gemini-3.1-flash-lite", "google"},
		{"gemini-2.5-pro", "google"},
		{"gemini-2.5-flash", "google"},
		{"gemini-2.0-flash", "google"},

		// DeepSeek V4 models
		{"deepseek-v4-pro", "deepseek"},
		{"deepseek-v4-flash", "deepseek"},

		// Grok models → default family
		{"grok-4.20", "default"},
		{"grok-4", "default"},
		{"grok-3", "default"},
		{"grok-3-mini", "default"},
	}

	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			meta, _ := registry.Resolve(context.Background(), tt.model)
			if meta.Family != tt.expectedFamily {
				t.Errorf("expected Family %q, got %q", tt.expectedFamily, meta.Family)
			}
		})
	}
}

// TestResolveFamily_PatternMatching verifies DetectFamily-based detection for unknown model IDs.
func TestResolveFamily_PatternMatching(t *testing.T) {
	registry := NewModelRegistry(nil)

	tests := []struct {
		model          string
		expectedFamily string
	}{
		// OpenAI flagship patterns
		{"gpt-4-turbo-custom", "openai_flagship"},
		{"gpt-5-preview", "openai_flagship"},
		{"o1-preview-custom", "openai_flagship"},
		{"o3-mini-custom", "openai_flagship"},
		{"o4-model", "openai_flagship"},

		// OpenAI standard patterns
		{"gpt-4.1-turbo", "openai_standard"},

		// Anthropic patterns
		{"claude-custom-model", "anthropic"},

		// Gemini patterns
		{"gemini-custom-pro", "google"},

		// DeepSeek patterns
		{"deepseek-v3-custom", "deepseek"},
		{"deepseek-reasoner-v2", "deepseek"},

		// Mistral patterns
		{"mistral-small-latest", "mistral"},
		{"mistral-7b-instruct", "mistral"},
		{"devstral-custom", "mistral"},
		{"codestral-latest", "mistral"},

		// Kimi patterns
		{"kimi-k2", "kimi"},

		// Qwen patterns
		{"qwen-2.5-72b", "qwen"},
		{"qwq-plus", "qwen"},

		// GLM patterns
		{"glm-z1-32b", "glm"},

		// Default family (no specific pattern)
		{"grok-custom-model", "default"},
		{"llama-3.1-70b", "default"},
		{"phi-3-mini", "default"},
		{"codellama-34b", "default"},
	}

	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			// These are unknown models resolved via DetectFamily
			meta, _ := registry.Resolve(context.Background(), tt.model)
			if meta.Family != tt.expectedFamily {
				t.Errorf("expected Family %q for model %q, got %q", tt.expectedFamily, tt.model, meta.Family)
			}
		})
	}
}

// TestResolveFamily_SourceWithoutFamily verifies that when a source returns metadata
// without Family set, resolveFamily delegates to DetectFamily.
func TestResolveFamily_SourceWithoutFamily(t *testing.T) {
	tests := []struct {
		name           string
		model          string
		expectedFamily string
	}{
		{
			name:           "claude model from source gets anthropic family",
			model:          "claude-custom-v2",
			expectedFamily: "anthropic",
		},
		{
			name:           "gemini model from source gets google family",
			model:          "gemini-custom-pro",
			expectedFamily: "google",
		},
		{
			name:           "unknown model from source gets default family",
			model:          "custom-llm-v1",
			expectedFamily: "default",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			registry := NewModelRegistry(nil)

			// Register a source that returns metadata without Family set
			registry.RegisterSource(func(model string) (ModelMetadata, bool) {
				if model == tt.model {
					return ModelMetadata{
						ContextWindow: 128000,
						OutputLimit:   8192,
						TokenizerType: "test",
					}, true
				}
				return ModelMetadata{}, false
			})

			meta, _ := registry.Resolve(context.Background(), tt.model)
			if meta.Family != tt.expectedFamily {
				t.Errorf("expected Family %q, got %q", tt.expectedFamily, meta.Family)
			}
		})
	}
}

// TestResolveFamily_UserOverride verifies that user override Family takes precedence.
func TestResolveFamily_UserOverride(t *testing.T) {
	tests := []struct {
		name           string
		model          string
		overrideFamily string
		expectedFamily string
	}{
		{
			name:           "override with explicit anthropic family",
			model:          "custom-model",
			overrideFamily: "anthropic",
			expectedFamily: "anthropic",
		},
		{
			name:           "override with explicit google family",
			model:          "custom-model",
			overrideFamily: "google",
			expectedFamily: "google",
		},
		{
			name:           "override without family should get DetectFamily result",
			model:          "claude-custom", // matches anthropic pattern
			overrideFamily: "",
			expectedFamily: "anthropic",
		},
		{
			name:           "override builtin model with different family",
			model:          "gpt-4o",    // normally openai_flagship
			overrideFamily: "anthropic", // overridden
			expectedFamily: "anthropic",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			overrideMeta := ModelMetadata{
				ContextWindow: 100000,
				OutputLimit:   5000,
				TokenizerType: "override",
			}
			if tt.overrideFamily != "" {
				overrideMeta.Family = tt.overrideFamily
			}

			registry := NewModelRegistry(map[string]ModelMetadata{
				tt.model: overrideMeta,
			})

			meta, _ := registry.Resolve(context.Background(), tt.model)
			if meta.Family != tt.expectedFamily {
				t.Errorf("expected Family %q, got %q", tt.expectedFamily, meta.Family)
			}
		})
	}
}

// TestResolveFamily_EmptyModelID verifies that Resolve("") returns "default" family.
// This is important because core components call Resolve("") when model ID
// isn't threaded through, and we want predictable fallback behavior.
func TestResolveFamily_EmptyModelID(t *testing.T) {
	reg := NewModelRegistry(nil)
	meta, _ := reg.Resolve(context.Background(), "")
	if meta.Family != "default" {
		t.Errorf("expected Family 'default' for empty model ID, got %q", meta.Family)
	}
}

func TestModelRegistry_SourcePriority(t *testing.T) {
	// Create registry with both a source and an override for the same model
	testModel := "priority-test-model"

	// Source returns these values
	sourceMeta := ModelMetadata{
		ContextWindow: 50000,
		OutputLimit:   2000,
		TokenizerType: "source-tokenizer",
	}

	// Override has different values (should win)
	overrideMeta := ModelMetadata{
		ContextWindow: 99999,
		OutputLimit:   9999,
		TokenizerType: "override-tokenizer",
	}

	registry := NewModelRegistry(map[string]ModelMetadata{
		testModel: overrideMeta,
	})

	registry.RegisterSource(func(model string) (ModelMetadata, bool) {
		if model == testModel {
			return sourceMeta, true
		}
		return ModelMetadata{}, false
	})

	// Override (tier 1) should take priority over source (tier 4)
	meta, _ := registry.Resolve(context.Background(), testModel)

	if meta.ContextWindow != overrideMeta.ContextWindow {
		t.Errorf("expected override ContextWindow %d, got %d", overrideMeta.ContextWindow, meta.ContextWindow)
	}
	if meta.OutputLimit != overrideMeta.OutputLimit {
		t.Errorf("expected override OutputLimit %d, got %d", overrideMeta.OutputLimit, meta.OutputLimit)
	}
	if meta.TokenizerType != overrideMeta.TokenizerType {
		t.Errorf("expected override TokenizerType %q, got %q", overrideMeta.TokenizerType, meta.TokenizerType)
	}
}

func TestModelRegistry_SourceFallback(t *testing.T) {
	// Register a source that returns false for a model
	registry := NewModelRegistry(nil)

	testModel := "fallback-test-model"

	registry.RegisterSource(func(model string) (ModelMetadata, bool) {
		// Source doesn't know about this model
		return ModelMetadata{}, false
	})

	// Resolve should use fallback defaults
	meta, ok := registry.Resolve(context.Background(), testModel)
	if ok {
		t.Fatal("expected ok=false when source returns false")
	}

	// Fallback defaults: ContextWindow: 128000, OutputLimit: 4096, TokenizerType: "approximate"
	if meta.ContextWindow != 128000 {
		t.Errorf("expected fallback ContextWindow 128000, got %d", meta.ContextWindow)
	}
	if meta.OutputLimit != 4096 {
		t.Errorf("expected fallback OutputLimit 4096, got %d", meta.OutputLimit)
	}
	if meta.TokenizerType != "approximate" {
		t.Errorf("expected fallback TokenizerType %q, got %q", "approximate", meta.TokenizerType)
	}
}

func TestBuiltInModelNames_AllModels(t *testing.T) {
	names := BuiltInModelNames("")
	if len(names) == 0 {
		t.Fatal("expected non-empty model names list")
	}
	// Should be sorted
	for i := 1; i < len(names); i++ {
		if names[i] < names[i-1] {
			t.Errorf("names not sorted: %q comes after %q", names[i], names[i-1])
		}
	}
}

func TestBuiltInModelNames_ByTokenizer(t *testing.T) {
	tests := []struct {
		tokenizer string
		wantMin   int
	}{
		{"tiktoken/o200k_base", 5},
		{"anthropic-api", 3},
		{"approximate", 5},
		{"nonexistent", 0},
	}

	for _, tt := range tests {
		t.Run(tt.tokenizer, func(t *testing.T) {
			names := BuiltInModelNames(tt.tokenizer)
			if len(names) < tt.wantMin {
				t.Errorf("expected at least %d models for tokenizer %q, got %d", tt.wantMin, tt.tokenizer, len(names))
			}
			// Verify all returned models actually have the correct tokenizer
			registry := makeBuiltInRegistry()
			for _, name := range names {
				if meta, ok := registry[name]; ok {
					if meta.TokenizerType != tt.tokenizer {
						t.Errorf("model %q has tokenizer %q, expected %q", name, meta.TokenizerType, tt.tokenizer)
					}
				}
			}
		})
	}
}

func TestModelRegistry_FetchFromHuggingFace(t *testing.T) {
	// Test successful fetch
	t.Run("success", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"max_position_embeddings": 8192}`))
		}))
		defer server.Close()

		registry := NewModelRegistry(nil)
		registry.httpClient = server.Client()
		// Override the URL by adjusting the httpClient transport
		registry.httpClient = &http.Client{
			Transport: &rewriteTransport{base: http.DefaultTransport, serverURL: server.URL},
		}

		meta, err := registry.fetchFromHuggingFace(context.Background(), "test-model")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if meta.ContextWindow != 8192 {
			t.Errorf("expected ContextWindow 8192, got %d", meta.ContextWindow)
		}
		if meta.OutputLimit != 4096 {
			t.Errorf("expected OutputLimit 4096, got %d", meta.OutputLimit)
		}
		if meta.TokenizerType != "approximate" {
			t.Errorf("expected TokenizerType 'approximate', got %q", meta.TokenizerType)
		}
	})

	// Test HTTP error
	t.Run("http_error", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusNotFound)
		}))
		defer server.Close()

		registry := NewModelRegistry(nil)
		registry.httpClient = &http.Client{
			Transport: &rewriteTransport{base: http.DefaultTransport, serverURL: server.URL},
		}

		_, err := registry.fetchFromHuggingFace(context.Background(), "test-model")
		if err == nil {
			t.Error("expected error for 404 response")
		}
	})

	// Test invalid JSON
	t.Run("invalid_json", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`not json`))
		}))
		defer server.Close()

		registry := NewModelRegistry(nil)
		registry.httpClient = &http.Client{
			Transport: &rewriteTransport{base: http.DefaultTransport, serverURL: server.URL},
		}

		_, err := registry.fetchFromHuggingFace(context.Background(), "test-model")
		if err == nil {
			t.Error("expected error for invalid JSON")
		}
	})

	// Test zero max_position_embeddings
	t.Run("zero_embeddings", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"max_position_embeddings": 0}`))
		}))
		defer server.Close()

		registry := NewModelRegistry(nil)
		registry.httpClient = &http.Client{
			Transport: &rewriteTransport{base: http.DefaultTransport, serverURL: server.URL},
		}

		_, err := registry.fetchFromHuggingFace(context.Background(), "test-model")
		if err == nil {
			t.Error("expected error for zero max_position_embeddings")
		}
	})
}

// rewriteTransport rewrites the request URL to the test server.
type rewriteTransport struct {
	base      http.RoundTripper
	serverURL string
}

func (t *rewriteTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	req = req.Clone(req.Context())
	req.URL.Scheme = "http"
	// Parse the server URL to extract host
	parsed, _ := url.Parse(t.serverURL)
	req.URL.Host = parsed.Host
	return t.base.RoundTrip(req)
}

func TestModelRegistry_CacheAfterFetchFromHuggingFace(t *testing.T) {
	callCount := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"max_position_embeddings": 4096}`))
	}))
	defer server.Close()

	registry := NewModelRegistry(nil)
	registry.httpClient = &http.Client{
		Transport: &rewriteTransport{base: http.DefaultTransport, serverURL: server.URL},
	}

	// First resolve should fetch from HuggingFace
	meta, _ := registry.Resolve(context.Background(), "hf-test-model")
	if meta.ContextWindow != 4096 {
		t.Errorf("expected ContextWindow 4096, got %d", meta.ContextWindow)
	}

	// Second resolve should use cache (no additional HTTP call)
	meta2, _ := registry.Resolve(context.Background(), "hf-test-model")
	if meta2.ContextWindow != 4096 {
		t.Errorf("expected cached ContextWindow 4096, got %d", meta2.ContextWindow)
	}

	if callCount != 1 {
		t.Errorf("expected 1 HTTP call (cached on second), got %d", callCount)
	}
}

func TestModelRegistry_MultipleSources(t *testing.T) {
	// Register two sources: first returns false, second returns metadata
	registry := NewModelRegistry(nil)

	testModel := "multi-source-model"
	expectedMeta := ModelMetadata{
		ContextWindow: 32768,
		OutputLimit:   1024,
		TokenizerType: "second-source-tokenizer",
	}

	// First source doesn't know the model
	registry.RegisterSource(func(model string) (ModelMetadata, bool) {
		return ModelMetadata{}, false
	})

	// Second source knows the model
	registry.RegisterSource(func(model string) (ModelMetadata, bool) {
		if model == testModel {
			return expectedMeta, true
		}
		return ModelMetadata{}, false
	})

	// Resolve should use the second source's metadata
	meta, _ := registry.Resolve(context.Background(), testModel)

	if meta.ContextWindow != expectedMeta.ContextWindow {
		t.Errorf("expected ContextWindow %d, got %d", expectedMeta.ContextWindow, meta.ContextWindow)
	}
	if meta.OutputLimit != expectedMeta.OutputLimit {
		t.Errorf("expected OutputLimit %d, got %d", expectedMeta.OutputLimit, meta.OutputLimit)
	}
	if meta.TokenizerType != expectedMeta.TokenizerType {
		t.Errorf("expected TokenizerType %q, got %q", expectedMeta.TokenizerType, meta.TokenizerType)
	}
}

// TestModelRegistry_SetCachedMetadata_PopulatesCache verifies that a late-learned
// entry written via SetCachedMetadata is returned by Resolve and is preferred
// over a HuggingFace fetch (tier 3 is consulted before the network lookup at
// the same tier).
func TestModelRegistry_SetCachedMetadata_PopulatesCache(t *testing.T) {
	registry := NewModelRegistry(nil)
	model := "local-only-model"
	expected := ModelMetadata{
		ContextWindow: 262144,
		OutputLimit:   4096,
		TokenizerType: "approximate",
	}

	registry.SetCachedMetadata(model, expected)

	meta, ok := registry.Resolve(context.Background(), model)
	if !ok {
		t.Fatal("expected ok=true for cached model")
	}
	if meta.ContextWindow != expected.ContextWindow {
		t.Errorf("ContextWindow: got %d, want %d", meta.ContextWindow, expected.ContextWindow)
	}
	if meta.OutputLimit != expected.OutputLimit {
		t.Errorf("OutputLimit: got %d, want %d", meta.OutputLimit, expected.OutputLimit)
	}
	if meta.Family == "" {
		t.Error("expected non-empty Family resolved by SetCachedMetadata")
	}
}

// TestModelRegistry_SetCachedMetadata_DoesNotOverrideBuiltIn verifies the
// tier priority: an entry written via SetCachedMetadata is shadowed when the
// model has a built-in (tier 2) entry. Built-in specs always win.
func TestModelRegistry_SetCachedMetadata_DoesNotOverrideBuiltIn(t *testing.T) {
	registry := NewModelRegistry(nil)

	// "gpt-4o" is a built-in model. Seed the cache with a bogus window.
	registry.SetCachedMetadata("gpt-4o", ModelMetadata{
		ContextWindow: 999999,
		OutputLimit:   1,
		TokenizerType: "approximate",
	})

	meta, _ := registry.Resolve(context.Background(), "gpt-4o")
	// Built-in gpt-4o window is 128000; the cached bogus value must NOT surface.
	if meta.ContextWindow != 128000 {
		t.Errorf("built-in spec clobbered by cache: got %d, want 128000", meta.ContextWindow)
	}
}

// TestModelRegistry_SetCachedMetadata_DoesNotOverrideUserOverride verifies that
// a user override (tier 1) always wins over a late-learned cache entry (tier 3).
func TestModelRegistry_SetCachedMetadata_DoesNotOverrideUserOverride(t *testing.T) {
	overrides := map[string]ModelMetadata{
		"custom-model": {
			ContextWindow: 32768,
			OutputLimit:   2048,
			TokenizerType: "custom",
		},
	}
	registry := NewModelRegistry(overrides)

	registry.SetCachedMetadata("custom-model", ModelMetadata{
		ContextWindow: 262144,
		OutputLimit:   4096,
		TokenizerType: "approximate",
	})

	meta, _ := registry.Resolve(context.Background(), "custom-model")
	if meta.ContextWindow != 32768 {
		t.Errorf("user override clobbered by cache: got %d, want 32768", meta.ContextWindow)
	}
	if meta.OutputLimit != 2048 {
		t.Errorf("user override clobbered by cache: got %d, want 2048", meta.OutputLimit)
	}
}

// TestResolveProtocol_BuiltinAndPattern verifies that ModelMetadata.Protocol is
// populated by Resolve across the four protocols, for both built-in models and
// DetectProtocol-based pattern matching.
func TestResolveProtocol_BuiltinAndPattern(t *testing.T) {
	registry := NewModelRegistry(nil)

	tests := []struct {
		model            string
		expectedProtocol APIProtocol
	}{
		// GPT-5 / Codex → Responses (built-in)
		{"gpt-5", ProtocolResponses},
		{"gpt-5.6", ProtocolResponses},
		{"codex-mini-latest", ProtocolResponses},

		// GPT-5 pattern matching → Responses (the family-vs-protocol split:
		// FamilyOpenAIFlagship, but protocol Responses)
		{"gpt-5-preview", ProtocolResponses},

		// Claude → Anthropic (built-in + pattern)
		{"claude-opus-4-6", ProtocolAnthropic},
		{"claude-sonnet-4-5", ProtocolAnthropic},
		{"claude-custom-model", ProtocolAnthropic},

		// Gemini / Gemma → Google (built-in + pattern)
		{"gemini-2.5-pro", ProtocolGoogle},
		{"gemini-2.0-flash", ProtocolGoogle},
		{"gemini-custom-pro", ProtocolGoogle},
		{"gemma-4-31b-it", ProtocolGoogle},

		// Chat Completions — OpenAI non-gpt-5 models (built-in)
		{"gpt-4o", ProtocolChatCompletions},
		{"gpt-4o-mini", ProtocolChatCompletions},
		{"gpt-4.1", ProtocolChatCompletions},
		{"o1", ProtocolChatCompletions},
		{"o3", ProtocolChatCompletions},
		{"o4-mini", ProtocolChatCompletions},

		// Chat Completions — other families / default
		{"grok-4", ProtocolChatCompletions},
		{"deepseek-v4-pro", ProtocolChatCompletions},
		{"llama-3.1-70b", ProtocolChatCompletions},
		{"unknown-model", ProtocolChatCompletions},
	}

	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			meta, _ := registry.Resolve(context.Background(), tt.model)
			if meta.Protocol != tt.expectedProtocol {
				t.Errorf("expected Protocol %q for model %q, got %q",
					tt.expectedProtocol, tt.model, meta.Protocol)
			}
		})
	}
}

// TestResolveProtocol_SourceWithoutProtocol verifies that when a source returns
// metadata without Protocol set, resolveProtocol delegates to DetectProtocol.
func TestResolveProtocol_SourceWithoutProtocol(t *testing.T) {
	tests := []struct {
		name             string
		model            string
		expectedProtocol APIProtocol
	}{
		{
			name:             "gpt-5 model from source gets responses protocol",
			model:            "gpt-5-custom-v2",
			expectedProtocol: ProtocolResponses,
		},
		{
			name:             "claude model from source gets anthropic protocol",
			model:            "claude-custom-v2",
			expectedProtocol: ProtocolAnthropic,
		},
		{
			name:             "gemini model from source gets google protocol",
			model:            "gemini-custom-pro",
			expectedProtocol: ProtocolGoogle,
		},
		{
			name:             "unknown model from source gets chat_completions protocol",
			model:            "custom-llm-v1",
			expectedProtocol: ProtocolChatCompletions,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			registry := NewModelRegistry(nil)

			// Register a source that returns metadata without Protocol set
			registry.RegisterSource(func(model string) (ModelMetadata, bool) {
				if model == tt.model {
					return ModelMetadata{
						ContextWindow: 128000,
						OutputLimit:   8192,
						TokenizerType: "test",
					}, true
				}
				return ModelMetadata{}, false
			})

			meta, _ := registry.Resolve(context.Background(), tt.model)
			if meta.Protocol != tt.expectedProtocol {
				t.Errorf("expected Protocol %q, got %q", tt.expectedProtocol, meta.Protocol)
			}
		})
	}
}

// TestResolveProtocol_UserOverride verifies that an explicit user-override
// Protocol takes precedence over DetectProtocol.
func TestResolveProtocol_UserOverride(t *testing.T) {
	tests := []struct {
		name             string
		model            string
		overrideProtocol APIProtocol
		expectedProtocol APIProtocol
	}{
		{
			name:             "override with explicit responses protocol",
			model:            "custom-model",
			overrideProtocol: ProtocolResponses,
			expectedProtocol: ProtocolResponses,
		},
		{
			name:             "override without protocol should get DetectProtocol result",
			model:            "claude-custom", // matches anthropic pattern
			overrideProtocol: "",
			expectedProtocol: ProtocolAnthropic,
		},
		{
			name:             "override builtin model with different protocol",
			model:            "gpt-4o", // normally chat_completions
			overrideProtocol: ProtocolResponses,
			expectedProtocol: ProtocolResponses,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			overrideMeta := ModelMetadata{
				ContextWindow: 100000,
				OutputLimit:   5000,
				TokenizerType: "override",
			}
			if tt.overrideProtocol != "" {
				overrideMeta.Protocol = tt.overrideProtocol
			}

			registry := NewModelRegistry(map[string]ModelMetadata{
				tt.model: overrideMeta,
			})

			meta, _ := registry.Resolve(context.Background(), tt.model)
			if meta.Protocol != tt.expectedProtocol {
				t.Errorf("expected Protocol %q, got %q", tt.expectedProtocol, meta.Protocol)
			}
		})
	}
}

func TestResolveBuiltInModel_KnownModel(t *testing.T) {
	// A model present in the built-in catalog must return its built-in values
	// with ok=true, regardless of any override that would be active in a real
	// registry. gpt-4o is a stable, well-known built-in entry.
	meta, ok := ResolveBuiltInModel("gpt-4o")
	if !ok {
		t.Fatal("expected ok=true for a known built-in model")
	}
	if meta.ContextWindow != 128000 {
		t.Errorf("expected ContextWindow 128000, got %d", meta.ContextWindow)
	}
	if meta.OutputLimit != 16384 {
		t.Errorf("expected OutputLimit 16384, got %d", meta.OutputLimit)
	}
	if meta.TokenizerType != "tiktoken/o200k_base" {
		t.Errorf("expected TokenizerType tiktoken/o200k_base, got %s", meta.TokenizerType)
	}
	// Family and Protocol must be resolved, not left empty.
	if meta.Family == "" {
		t.Error("expected non-empty Family for a known built-in model")
	}
	if meta.Protocol == "" {
		t.Error("expected non-empty Protocol for a known built-in model")
	}
}

func TestResolveBuiltInModel_CaseInsensitive(t *testing.T) {
	// Built-in lookup is case-insensitive, mirroring Resolve.
	upper, okUpper := ResolveBuiltInModel("GPT-4O")
	lower, okLower := ResolveBuiltInModel("gpt-4o")
	if !okUpper || !okLower {
		t.Fatal("expected both lookups to succeed")
	}
	if upper != lower {
		t.Errorf("case-insensitive lookup mismatch: %+v vs %+v", upper, lower)
	}
}

func TestResolveBuiltInModel_FuzzyMatch(t *testing.T) {
	// The separator-insensitive fuzzy lookup should bridge a host that drops
	// the dot ("gpt4o" vs "gpt-4o"). The result must equal the built-in entry.
	meta, ok := ResolveBuiltInModel("gpt4o")
	if !ok {
		t.Fatal("expected fuzzy match to succeed for 'gpt4o'")
	}
	if meta.ContextWindow != 128000 {
		t.Errorf("expected fuzzy-matched ContextWindow 128000, got %d", meta.ContextWindow)
	}
}

func TestResolveBuiltInModel_UnknownModel(t *testing.T) {
	// An unknown model must return the fallback defaults with ok=false and
	// never touch the network.
	meta, ok := ResolveBuiltInModel("definitely-not-a-real-model-xyz")
	if ok {
		t.Fatal("expected ok=false for an unknown model")
	}
	if meta.ContextWindow != 128000 {
		t.Errorf("expected fallback ContextWindow 128000, got %d", meta.ContextWindow)
	}
	if meta.OutputLimit != 4096 {
		t.Errorf("expected fallback OutputLimit 4096, got %d", meta.OutputLimit)
	}
	if meta.TokenizerType != "approximate" {
		t.Errorf("expected fallback TokenizerType approximate, got %s", meta.TokenizerType)
	}
	if !meta.Capabilities.Attachment {
		t.Error("expected optimistic attachment capability in fallback")
	}
}

func TestResolveBuiltInModel_UnaffectedByOverride(t *testing.T) {
	// Constructing a registry with an override for gpt-4o must NOT change what
	// ResolveBuiltInModel returns — it consults only the built-in catalog.
	_ = NewModelRegistry(map[string]ModelMetadata{
		"gpt-4o": {ContextWindow: 999999, OutputLimit: 999999},
	})
	meta, ok := ResolveBuiltInModel("gpt-4o")
	if !ok {
		t.Fatal("expected ok=true for gpt-4o")
	}
	if meta.ContextWindow != 128000 {
		t.Errorf("override leaked into built-in lookup: got ContextWindow %d, want 128000", meta.ContextWindow)
	}
	if meta.OutputLimit != 16384 {
		t.Errorf("override leaked into built-in lookup: got OutputLimit %d, want 16384", meta.OutputLimit)
	}
}
