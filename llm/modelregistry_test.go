package llm

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"sync"
	"testing"
	"time"
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

		// Kimi models (Moonshot AI) — verified from platform.kimi.ai/docs and
		// kimi.com/code/docs (the Kimi Code endpoint serves short, non-prefixed
		// IDs that contain no "kimi" substring).
		{"kimi-k3", 1000000, 131072, "approximate"},
		{"k3", 1000000, 131072, "approximate"},
		{"k3-256k", 262144, 65536, "approximate"},
		{"kimi-k2.7-code", 262144, 65536, "approximate"},
		{"kimi-for-coding", 262144, 65536, "approximate"},
		{"kimi-for-coding-highspeed", 262144, 65536, "approximate"},
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
		OutputLimit:   32768,
		TokenizerType: "approximate",
		// Unknown models default to optimistic attachment support: better to
		// surface a runtime provider error than to deny image uploads.
		Capabilities: &ModelCapabilities{Attachment: true},
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
	if meta.Capabilities == nil || *meta.Capabilities != *expected.Capabilities {
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
		// Negative: the query has no "/", so its "qwen" org fragment is NOT
		// stripped and becomes part of the bare name — it must NOT match a
		// prefixed key whose org prefix WAS stripped.
		{"org jammed into name with no slash does not match", "qwenqwen3635ba3bfp8", false, 0},
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

func TestNormalizeModelID_StripsPostfixes(t *testing.T) {
	// Delivery/quantization postfixes must be stripped from the end — including
	// chains of them — while parameter counts ("-8b") and canonical
	// quantizations like "fp8" must survive untouched.
	tests := []struct {
		name string
		in   string
		want string
	}{
		{"no postfix", "qwen3", "qwen3"},
		{"gguf postfix", "qwen3-gguf", "qwen3"},
		{"mlx postfix", "qwen3-mlx", "qwen3"},
		{"safetensors postfix", "qwen3-safetensors", "qwen3"},
		{"8bit postfix", "qwen3-8bit", "qwen3"},
		{"8-bit postfix", "qwen3-8-bit", "qwen3"},
		{"8_bit postfix", "qwen3-8_bit", "qwen3"},
		{"16bit postfix (not reduced via 6bit)", "qwen3-16bit", "qwen3"},
		{"32bit postfix (not reduced via 2bit)", "qwen3-32bit", "qwen3"},
		{"postfix chain", "qwen3-8bit-gguf", "qwen3"},
		{"postfix chain reversed", "qwen3-gguf-8bit", "qwen3"},
		{"longer chain", "qwen3-8-bit-gguf-mlx", "qwen3"},
		{"mixed case postfix", "QWEN3-8BIT-GGUF", "qwen3"},
		{"vendor prefix still stripped", "qwen/qwen3-8bit", "qwen3"},
		{"parameter count 8b not stripped", "qwen3-8b", "qwen38b"},
		{"parameter count 4b not stripped", "qwen3-4b", "qwen34b"},
		{"fp8 not stripped", "qwen3.6-35b-a3b-fp8", "qwen3635ba3bfp8"},
		{"base suffix not stripped", "qwen3-base", "qwen3base"},
		{"instruct 8b-it not stripped", "qwen3-8b-it", "qwen38bit"},
		{"instruct 32b-it not stripped", "qwen3-32b-it", "qwen332bit"},
		{"instruct gemma-2-2b-it not stripped", "gemma-2-2b-it", "gemma22bit"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := normalizeModelID(tt.in); got != tt.want {
				t.Errorf("normalizeModelID(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}

func TestModelRegistry_FuzzyMatch_IgnoresPostfixes(t *testing.T) {
	// A host that appends delivery/quantization postfixes to a base model name
	// must resolve to the same metadata as the base. Chains of postfixes are
	// stripped too.
	registry := NewModelRegistry(map[string]ModelMetadata{
		"my-model": {ContextWindow: 424242},
	})

	for _, model := range []string{
		"my-model",
		"my-model-gguf",
		"my-model-mlx",
		"my-model-8bit",
		"my-model-8-bit",
		"my-model-8bit-gguf",
		"my-model-gguf-8bit-mlx",
	} {
		meta, ok := registry.ResolveLocal(model)
		if !ok {
			t.Errorf("ResolveLocal(%q): expected ok=true", model)
			continue
		}
		if meta.ContextWindow != 424242 {
			t.Errorf("ResolveLocal(%q): ContextWindow = %d, want 424242", model, meta.ContextWindow)
		}
	}

	// A parameter-count suffix ("-8b") is not a quantization postfix and must
	// NOT collapse onto the base model.
	if _, ok := registry.ResolveLocal("my-model-8b"); ok {
		t.Error("ResolveLocal(my-model-8b): expected ok=false (parameter count is not a postfix)")
	}
}

func TestModelRegistry_FuzzyIndex_DeterministicWinner(t *testing.T) {
	// When two override keys collapse to the same normalized form, the
	// lexicographically smallest original key deterministically wins the fuzzy
	// slot (see buildNormalizedIndex). A postfixed spelling sorts after its
	// base, so the base is the survivor and a drifted query resolves to the
	// base metadata, not the variant's.
	registry := NewModelRegistry(map[string]ModelMetadata{
		"qwen3-gguf": {ContextWindow: 200},
		"qwen3":      {ContextWindow: 100},
	})

	meta, ok := registry.ResolveLocal("qwen3_gguf")
	if !ok {
		t.Fatal("ResolveLocal(qwen3_gguf): expected ok=true")
	}
	if meta.ContextWindow != 100 {
		t.Errorf("ResolveLocal(qwen3_gguf): ContextWindow = %d, want 100 (base key wins)", meta.ContextWindow)
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
	// They must not have been swapped or merged. qwen3.6-35b-a3b-fp8 (262144)
	// and qwen3.7-max (1000000) have distinct context windows, so compare those
	// — both legitimately share OutputLimit 65536 now.
	if m36.ContextWindow == m37.ContextWindow {
		t.Errorf("distinct versions resolved to identical ContextWindow %d — possible collision",
			m36.ContextWindow)
	}
}

func TestModelRegistry_FuzzyMatch_IgnoresVendorPrefix(t *testing.T) {
	// The vendor/org prefix is a routing decoration, not a property of the
	// model. A prefixed built-in key ("zai-org/glm-5.2-fp8", the HuggingFace
	// checkpoint) and a bare query ("GLM-5.2-FP8", the Z.ai API name) must
	// resolve to the same metadata. Both directions of the asymmetry are
	// exercised: prefixed-key-exact/bare-query-fuzzy and bare-exact-does-not-
	// exist/prefixed-key-fuzzy.
	registry := NewModelRegistry(nil)

	// zai-org/glm-5.2-fp8 is an exact built-in key (the HuggingFace FP8
	// checkpoint, 1 MiB context). GLM-5.2-FP8 is NOT a built-in key — only the
	// fuzzy lookup, which now discards the vendor prefix, can match it.
	prefixed, okP := registry.Resolve(context.Background(), "zai-org/glm-5.2-fp8")
	bare, okB := registry.Resolve(context.Background(), "GLM-5.2-FP8")

	if !okP || !okB {
		t.Fatalf("both should resolve; got okP=%v okB=%v", okP, okB)
	}
	// 1048576 is unique to the zai-org/glm-5.2-fp8 entry; the bare "glm-5.2"
	// built-in has a 1000000 context window, so this also guards against the
	// bare query accidentally matching the shorter-context sibling.
	const wantCtx = 1048576
	if prefixed.ContextWindow != wantCtx {
		t.Errorf("prefixed ContextWindow = %d, want %d", prefixed.ContextWindow, wantCtx)
	}
	if bare.ContextWindow != wantCtx {
		t.Errorf("bare ContextWindow = %d, want %d", bare.ContextWindow, wantCtx)
	}
	if prefixed.ContextWindow != bare.ContextWindow {
		t.Errorf("prefixed (%d) and bare (%d) resolved to different metadata",
			prefixed.ContextWindow, bare.ContextWindow)
	}
}

func TestModelRegistry_FuzzyMatch_VendorPrefixCaseInsensitive(t *testing.T) {
	// A vendor prefix spelled with different casing must not block the match:
	// "ZAI-Org/glm-5.2-fp8" (uppercased org) and "zai-org/glm-5.2-fp8" (the
	// canonical key) differ only in the prefix's casing, which the prefix
	// stripping discards entirely.
	registry := NewModelRegistry(nil)

	meta, ok := registry.Resolve(context.Background(), "ZAI-Org/glm-5.2-fp8")
	if !ok {
		t.Fatal("expected ok=true for case-variant vendor prefix")
	}
	if meta.ContextWindow != 1048576 {
		t.Errorf("ContextWindow = %d, want 1048576", meta.ContextWindow)
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

// TestModelRegistry_PartialOverrideInheritsProbeWindow is the regression test
// for the lazy-probe shadowing bug. A protocol-only partial override (the kind
// c0wrk's local Google-protocol remap injects for Gemma checkpoints served by
// LM Studio) pins the protocol but leaves the scalar fields (context window,
// output limit, tokenizer) at zero. The lazy local-model probe then discovers
// the model's REAL context window and stores it in the cache tier via
// SetCachedMetadata. Resolve must inherit the probed window (8192) from the
// cache tier rather than collapsing to zero or to the 128000 fallback —
// otherwise the context-fill accounting and compaction thresholds would be
// computed against an inflated window forever.
func TestModelRegistry_PartialOverrideInheritsProbeWindow(t *testing.T) {
	registry := NewModelRegistry(map[string]ModelMetadata{
		// A catalog-miss local model (no built-in entry). The override pins
		// only the protocol, mirroring the auto-remap; its scalar fields are
		// zero so they can be inherited.
		"gemma-3-27b-it": {Protocol: ProtocolChatCompletions},
	})

	// Simulate the lazy probe landing its result in the cache tier.
	registry.SetCachedMetadata("gemma-3-27b-it", ModelMetadata{
		ContextWindow: 8192,
		OutputLimit:   4096,
		TokenizerType: "approximate",
	})

	meta, ok := registry.Resolve(context.Background(), "gemma-3-27b-it")
	if !ok {
		t.Fatal("expected ok=true for overridden model")
	}
	if meta.ContextWindow != 8192 {
		t.Errorf("ContextWindow = %d, want 8192 (inherited from cache/probe, not 128000 fallback)", meta.ContextWindow)
	}
	if meta.Protocol != ProtocolChatCompletions {
		t.Errorf("Protocol = %q, want %q (override must remain authoritative)", meta.Protocol, ProtocolChatCompletions)
	}
}

// TestModelRegistry_PartialOverrideInheritsBuiltinScalar ensures a partial
// override for a catalog-hit model inherits the built-in scalar fields it left
// unset, while keeping its pinned protocol. gpt-4o has a built-in context
// window; the override pins the protocol only and must inherit that window.
func TestModelRegistry_PartialOverrideInheritsBuiltinScalar(t *testing.T) {
	registry := NewModelRegistry(map[string]ModelMetadata{
		"gpt-4o": {Protocol: ProtocolChatCompletions},
	})

	meta, ok := registry.Resolve(context.Background(), "gpt-4o")
	if !ok {
		t.Fatal("expected ok=true for overridden built-in model")
	}
	// Built-in gpt-4o context window (128000) must be inherited, not zero.
	if meta.ContextWindow == 0 {
		t.Errorf("ContextWindow = 0, want inherited built-in value (non-zero)")
	}
	if meta.Protocol != ProtocolChatCompletions {
		t.Errorf("Protocol = %q, want %q (override authoritative)", meta.Protocol, ProtocolChatCompletions)
	}
}

// TestModelRegistry_PartialOverrideInheritsBuiltinCapabilities ensures a partial
// override for a catalog-hit model inherits the built-in Capabilities it left
// unset (zero value), not just the scalar fields. gpt-4o declares
// Attachment/Temperature/ToolCall; a protocol-only override must inherit those
// rather than silently disabling image uploads, temperature, and tool calling.
// This is the regression test for the Issue-1 footgun.
func TestModelRegistry_PartialOverrideInheritsBuiltinCapabilities(t *testing.T) {
	registry := NewModelRegistry(map[string]ModelMetadata{
		"gpt-4o": {Protocol: ProtocolChatCompletions},
	})

	meta, ok := registry.Resolve(context.Background(), "gpt-4o")
	if !ok {
		t.Fatal("expected ok=true for overridden built-in model")
	}
	// Built-in gpt-4o capabilities must be inherited, not left all-false.
	if !meta.Capabilities.Attachment || !meta.Capabilities.Temperature || !meta.Capabilities.ToolCall {
		t.Errorf("expected inherited gpt-4o capabilities (Attachment, Temperature, ToolCall), got %+v", meta.Capabilities)
	}
	if meta.Protocol != ProtocolChatCompletions {
		t.Errorf("Protocol = %q, want %q (override authoritative)", meta.Protocol, ProtocolChatCompletions)
	}
}

// TestModelRegistry_PartialOverrideCatalogMissKeepsOptimisticCapabilities is the
// regression test for the catalog-MISS counterpart of the capabilities footgun.
// A protocol-only partial override for a model absent from every non-network
// tier (not in the built-in catalog, not in the cache) must still inherit the
// optimistic fallback Capabilities (Attachment=true), exactly as a no-override
// Resolve of the same unknown model would (tier-5 fallback). Previously
// resolveBuiltinOrCache's fallback returned a zero-value Capabilities, so the
// override silently disabled image uploads for local models that c0wrk's
// auto-remap targets (e.g. Gemma checkpoints served by LM Studio/vLLM).
func TestModelRegistry_PartialOverrideCatalogMissKeepsOptimisticCapabilities(t *testing.T) {
	registry := NewModelRegistry(map[string]ModelMetadata{
		// Catalog-miss: no built-in entry, no seeded cache.
		"totally-unknown-local-checkpoint": {Protocol: ProtocolChatCompletions},
	})

	meta, ok := registry.Resolve(context.Background(), "totally-unknown-local-checkpoint")
	if !ok {
		t.Fatal("expected ok=true for overridden model")
	}
	if !meta.Capabilities.Attachment {
		t.Errorf("catalog-miss partial override disabled Attachment; got %+v (want optimistic Attachment=true like the no-override fallback)", meta.Capabilities)
	}
	if meta.Protocol != ProtocolChatCompletions {
		t.Errorf("Protocol = %q, want %q (override authoritative)", meta.Protocol, ProtocolChatCompletions)
	}
}

// TestModelRegistry_FullySpecifiedOverrideNotEnriched guards that a fully
// specified override (all three scalar fields AND Capabilities set) is returned
// verbatim and is never silently merged with lower-tier data — the prior
// behaviour relied on by TestModelRegistry_OverridePriority and real config.yaml
// entries. Capabilities is a first-class inheritable field, so it must be
// non-nil for the override to count as fully specified; a nil Capabilities
// would otherwise be inherited from the built-in entry.
func TestModelRegistry_FullySpecifiedOverrideNotEnriched(t *testing.T) {
	registry := NewModelRegistry(map[string]ModelMetadata{
		"gpt-4o": {
			ContextWindow: 999999,
			OutputLimit:   8888,
			TokenizerType: "custom-tokenizer",
			Protocol:      ProtocolChatCompletions,
			Capabilities:  &ModelCapabilities{ToolCall: true},
		},
	})
	// Seed a cache entry whose values differ; it must NOT bleed through.
	registry.SetCachedMetadata("gpt-4o", ModelMetadata{
		ContextWindow: 1,
		OutputLimit:   1,
		TokenizerType: "should-not-leak",
		Capabilities:  &ModelCapabilities{Attachment: true, Reasoning: true},
	})

	meta, ok := registry.Resolve(context.Background(), "gpt-4o")
	if !ok {
		t.Fatal("expected ok=true for overridden model")
	}
	if meta.ContextWindow != 999999 || meta.OutputLimit != 8888 || meta.TokenizerType != "custom-tokenizer" {
		t.Errorf("fully-specified override was enriched: got %+v", meta)
	}
	if meta.Capabilities == nil || *meta.Capabilities != (ModelCapabilities{ToolCall: true}) {
		t.Errorf("fully-specified override capabilities were enriched: got %+v", meta.Capabilities)
	}
}

// TestModelRegistry_AllFalseCapabilitiesOverrideIsAuthoritative guards the
// pointer semantics of ModelMetadata.Capabilities: a non-nil override with
// every flag false must WIN over the built-in catalog's capabilities instead
// of being treated as "unset" and silently re-inherited. Under the old
// value-struct semantics the all-false set was indistinguishable from
// "inherit", so a user who disabled every capability in the settings dialog
// silently got the catalog defaults back at runtime.
func TestModelRegistry_AllFalseCapabilitiesOverrideIsAuthoritative(t *testing.T) {
	registry := NewModelRegistry(map[string]ModelMetadata{
		"gpt-4o": {
			// Only capabilities pinned (all false); scalars left unset so they
			// inherit from the built-in entry.
			Capabilities: &ModelCapabilities{},
		},
	})

	meta, ok := registry.Resolve(context.Background(), "gpt-4o")
	if !ok {
		t.Fatal("expected ok=true for overridden model")
	}
	if meta.Capabilities == nil {
		t.Fatal("expected non-nil capabilities after enrichment")
	}
	if *meta.Capabilities != (ModelCapabilities{}) {
		t.Errorf("all-false override was re-inherited from catalog: got %+v", *meta.Capabilities)
	}
	// The unset scalars still inherit from the built-in entry.
	if meta.ContextWindow == 0 {
		t.Errorf("expected inherited context window from built-in catalog, got 0")
	}
}

// The three tests below are regression tests for the resolver contract
// "Resolve/ResolveLocal output is always enriched to non-nil Capabilities".
// Before finalizeMeta, each scenario inherited a nil pointer from a PARTIAL
// lower-tier record: no tier declared capabilities, and the nil sailed
// through to the caller, where the natural `meta.Capabilities.Attachment`
// dereference panicked. The guard restores the optimistic unknown set — the
// same assumption tier 5 makes for wholly unknown models.

// TestModelRegistry_NilCapabilitiesGuard_PartialOverridePartialCache: a
// protocol-only override enriched against a cache entry whose probe observed
// only scalars (enrichPartialWith inherited the cache entry's nil).
func TestModelRegistry_NilCapabilitiesGuard_PartialOverridePartialCache(t *testing.T) {
	registry := NewModelRegistry(map[string]ModelMetadata{
		// Catalog-miss model; the override pins only the protocol.
		"gemma-3-27b-it": {Protocol: ProtocolChatCompletions},
	})
	// The probe observed scalars only; Capabilities stays nil.
	registry.SetCachedMetadata("gemma-3-27b-it", ModelMetadata{
		ContextWindow: 8192,
		OutputLimit:   4096,
		TokenizerType: "approximate",
	})

	resolvers := []struct {
		name string
		call func() (ModelMetadata, bool)
	}{
		{"Resolve", func() (ModelMetadata, bool) {
			return registry.Resolve(context.Background(), "gemma-3-27b-it")
		}},
		{"ResolveLocal", func() (ModelMetadata, bool) {
			return registry.ResolveLocal("gemma-3-27b-it")
		}},
	}
	for _, r := range resolvers {
		meta, ok := r.call()
		if !ok {
			t.Fatalf("%s: expected ok=true for overridden model", r.name)
		}
		if meta.Capabilities == nil {
			t.Fatalf("%s: Capabilities = nil, want non-nil (optimistic default)", r.name)
		}
		if want := (ModelCapabilities{Attachment: true}); *meta.Capabilities != want {
			t.Errorf("%s: Capabilities = %+v, want %+v", r.name, *meta.Capabilities, want)
		}
		if meta.ContextWindow != 8192 {
			t.Errorf("%s: ContextWindow = %d, want 8192 (inherited from partial cache entry)", r.name, meta.ContextWindow)
		}
	}
}

// TestModelRegistry_NilCapabilitiesGuard_PartialCacheEntry: the tier-3 direct
// read of a cache entry written by a probe that observed only the window.
func TestModelRegistry_NilCapabilitiesGuard_PartialCacheEntry(t *testing.T) {
	registry := NewModelRegistry(nil)
	registry.SetCachedMetadata("local-probe-only-window", ModelMetadata{ContextWindow: 32768})

	meta, ok := registry.ResolveLocal("local-probe-only-window")
	if !ok {
		t.Fatal("expected ok=true for cached model")
	}
	if meta.Capabilities == nil {
		t.Fatal("Capabilities = nil, want non-nil (optimistic default)")
	}
	if !meta.Capabilities.Attachment {
		t.Error("Capabilities.Attachment = false, want true (optimistic default)")
	}
	if meta.ContextWindow != 32768 {
		t.Errorf("ContextWindow = %d, want 32768", meta.ContextWindow)
	}
}

// TestModelRegistry_NilCapabilitiesGuard_PartialOverridePartialRuntime: a
// partial override enriched against a PARTIAL runtime entry —
// resolveBuiltinOrCache returned the raw entry whose nil was inherited.
func TestModelRegistry_NilCapabilitiesGuard_PartialOverridePartialRuntime(t *testing.T) {
	registry := NewModelRegistry(map[string]ModelMetadata{
		// Catalog-miss model; the override pins only the protocol.
		"local-checkpoint": {Protocol: ProtocolChatCompletions},
	})
	// The probe observed the window only; Capabilities stays nil.
	registry.SetRuntimeMetadata("local-checkpoint", ModelMetadata{ContextWindow: 65536})

	meta, ok := registry.Resolve(context.Background(), "local-checkpoint")
	if !ok {
		t.Fatal("expected ok=true for overridden model")
	}
	if meta.Capabilities == nil {
		t.Fatal("Capabilities = nil, want non-nil (optimistic default)")
	}
	if !meta.Capabilities.Attachment {
		t.Error("Capabilities.Attachment = false, want true (optimistic default)")
	}
	if meta.ContextWindow != 65536 {
		t.Errorf("ContextWindow = %d, want 65536 (inherited from partial runtime entry)", meta.ContextWindow)
	}
}

// TestModelRegistry_NilCapabilitiesGuard_PartialRuntimeOverPartialCache: the
// tier-1.5 path — a partial runtime entry enriched against a partial cache
// entry inherited the cache's nil via resolveSpecOrCache.
func TestModelRegistry_NilCapabilitiesGuard_PartialRuntimeOverPartialCache(t *testing.T) {
	registry := NewModelRegistry(nil)
	registry.SetCachedMetadata("local-checkpoint", ModelMetadata{ContextWindow: 4096})
	registry.SetRuntimeMetadata("local-checkpoint", ModelMetadata{ContextWindow: 65536})

	meta, ok := registry.ResolveLocal("local-checkpoint")
	if !ok {
		t.Fatal("expected ok=true for runtime-observed model")
	}
	if meta.Capabilities == nil {
		t.Fatal("Capabilities = nil, want non-nil (optimistic default)")
	}
	if meta.ContextWindow != 65536 {
		t.Errorf("ContextWindow = %d, want 65536 (runtime entry wins over cache)", meta.ContextWindow)
	}
}

// The tests below guard the OWNERSHIP half of the contract: metadata returned
// by the public API must never alias registry state, and records handed to
// the registry are copied on write. Built-in catalog entries are process-wide
// and shared by every ModelRegistry, so a caller mutating a resolved
// Capabilities pointer must not poison later resolutions — here or in any
// other registry — and must not race concurrent readers.

// TestModelRegistry_Resolve_CapabilitiesDefensivelyCopied covers the built-in
// tier, the fuzzy path (whose cache write goes through cacheResolved), and
// ResolveBuiltInModel, which reads the same global catalog.
func TestModelRegistry_Resolve_CapabilitiesDefensivelyCopied(t *testing.T) {
	registry := NewModelRegistry(nil)

	first, ok := registry.ResolveLocal("gpt-4o")
	if !ok {
		t.Fatal("expected ok=true for built-in model")
	}
	want := *first.Capabilities

	// Simulate a careless host mutating the returned metadata in place.
	*first.Capabilities = ModelCapabilities{}

	if second, _ := registry.ResolveLocal("gpt-4o"); *second.Capabilities != want {
		t.Errorf("mutation of returned Capabilities leaked into the registry: got %+v, want %+v", *second.Capabilities, want)
	}

	// The catalog is global: a second registry must be unaffected too.
	other := NewModelRegistry(nil)
	if meta, _ := other.Resolve(context.Background(), "gpt-4o"); *meta.Capabilities != want {
		t.Errorf("mutation leaked into the global built-in catalog: got %+v, want %+v", *meta.Capabilities, want)
	}

	// Fuzzy hit ("gpt4o" → "gpt-4o") caches a twin under the query key; the
	// returned metadata must not alias the cached twin.
	fuzzy, ok := registry.ResolveLocal("gpt4o")
	if !ok {
		t.Fatal("expected ok=true for fuzzy match")
	}
	*fuzzy.Capabilities = ModelCapabilities{}
	if again, _ := registry.ResolveLocal("gpt4o"); *again.Capabilities != want {
		t.Errorf("mutation of fuzzy-resolved Capabilities leaked into the cache: got %+v, want %+v", *again.Capabilities, want)
	}

	// ResolveBuiltInModel reads the same global map and must hand out its own copy.
	builtin, _ := ResolveBuiltInModel("gpt-4o")
	*builtin.Capabilities = ModelCapabilities{}
	if fresh, _ := ResolveBuiltInModel("gpt-4o"); *fresh.Capabilities != want {
		t.Errorf("ResolveBuiltInModel mutation leaked into the global catalog: got %+v, want %+v", *fresh.Capabilities, want)
	}
}

// TestModelRegistry_OverridesCopiedOnConstruction: NewModelRegistry deep-copies
// each override entry's Capabilities pointer, so a caller mutating its own
// structs after hand-off cannot race the registry's concurrent readers.
func TestModelRegistry_OverridesCopiedOnConstruction(t *testing.T) {
	caps := &ModelCapabilities{Attachment: true, ToolCall: true}
	registry := NewModelRegistry(map[string]ModelMetadata{
		"my-local-model": {
			ContextWindow: 1000,
			OutputLimit:   500,
			TokenizerType: "approximate",
			Capabilities:  caps,
		},
	})

	*caps = ModelCapabilities{} // caller mutates its own struct afterwards

	meta, ok := registry.ResolveLocal("my-local-model")
	if !ok {
		t.Fatal("expected ok=true for overridden model")
	}
	if want := (ModelCapabilities{Attachment: true, ToolCall: true}); *meta.Capabilities != want {
		t.Errorf("override Capabilities = %+v, want %+v (constructor must deep-copy)", *meta.Capabilities, want)
	}
}

// TestModelRegistry_SetCachedMetadata_CopiesCapabilities: a caller reusing its
// ModelMetadata value (e.g. a probe loop) must not reach the stored entry.
func TestModelRegistry_SetCachedMetadata_CopiesCapabilities(t *testing.T) {
	registry := NewModelRegistry(nil)
	entry := ModelMetadata{ContextWindow: 1000, Capabilities: &ModelCapabilities{Reasoning: true}}
	registry.SetCachedMetadata("cached-model", entry)
	*entry.Capabilities = ModelCapabilities{}

	meta, ok := registry.ResolveLocal("cached-model")
	if !ok {
		t.Fatal("expected ok=true for cached model")
	}
	if !meta.Capabilities.Reasoning {
		t.Errorf("Capabilities.Reasoning = false, want true (setter must copy)")
	}
}

// TestModelRegistry_SetRuntimeMetadata_CopiesCapabilities: same ownership rule
// for the runtime tier, both via the enriched resolver and via the raw
// RuntimeMetadata read — which hands out its own copy as well.
func TestModelRegistry_SetRuntimeMetadata_CopiesCapabilities(t *testing.T) {
	registry := NewModelRegistry(nil)
	entry := ModelMetadata{ContextWindow: 1000, Capabilities: &ModelCapabilities{Reasoning: true}}
	registry.SetRuntimeMetadata("runtime-model", entry)
	*entry.Capabilities = ModelCapabilities{}

	meta, ok := registry.ResolveLocal("runtime-model")
	if !ok {
		t.Fatal("expected ok=true for runtime model")
	}
	if !meta.Capabilities.Reasoning {
		t.Errorf("ResolveLocal: Capabilities.Reasoning = false, want true (setter must copy)")
	}

	raw, ok := registry.RuntimeMetadata("runtime-model")
	if !ok {
		t.Fatal("expected ok=true from RuntimeMetadata")
	}
	if !raw.Capabilities.Reasoning {
		t.Errorf("RuntimeMetadata: Capabilities.Reasoning = false, want true (setter must copy)")
	}

	// The raw read must not alias the stored entry either.
	*raw.Capabilities = ModelCapabilities{}
	if raw2, _ := registry.RuntimeMetadata("runtime-model"); !raw2.Capabilities.Reasoning {
		t.Error("RuntimeMetadata returned an alias of the stored entry: caller mutation leaked back")
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

		// Kimi models — the Kimi Code endpoint's short IDs ("k3", "k3-256k")
		// contain no "kimi" substring, so these pin the explicit built-in
		// Family rather than DetectFamily substring matching.
		{"kimi-k3", "kimi"},
		{"k3", "kimi"},
		{"k3-256k", "kimi"},
		{"kimi-k2.7-code", "kimi"},
		{"kimi-for-coding", "kimi"},

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

	// Fallback defaults: ContextWindow: 128000, OutputLimit: 32768, TokenizerType: "approximate"
	if meta.ContextWindow != 128000 {
		t.Errorf("expected fallback ContextWindow 128000, got %d", meta.ContextWindow)
	}
	if meta.OutputLimit != 32768 {
		t.Errorf("expected fallback OutputLimit 32768, got %d", meta.OutputLimit)
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
		if meta.OutputLimit != 32768 {
			t.Errorf("expected OutputLimit 32768, got %d", meta.OutputLimit)
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

// countingTransport counts RoundTrip calls and fails every one of them. Wiring
// it into a registry's HTTP client turns the client into a network-usage
// detector: any code path that is supposed to be network-free leaves the
// counter at zero, and one that misbehaves both increments it and surfaces a
// transport error instead of silently succeeding.
type countingTransport struct {
	calls int
}

func (t *countingTransport) RoundTrip(_ *http.Request) (*http.Response, error) {
	t.calls++
	return nil, errors.New("unexpected network call")
}

func TestModelRegistry_ResolveLocal_NeverTouchesNetwork(t *testing.T) {
	// ResolveLocal must serve every local tier — override, built-in, fuzzy,
	// lazy cache — and the offline fallback, without a single HTTP attempt.
	// The counting transport enforces the guarantee: it fails every call and
	// leaves calls == 0 only if the network was never reached.
	registry := NewModelRegistry(map[string]ModelMetadata{
		"my-override-model": {
			ContextWindow: 111111,
			OutputLimit:   2222,
			TokenizerType: "override-tok",
		},
	})
	transport := &countingTransport{}
	registry.httpClient = &http.Client{Transport: transport}

	t.Run("override tier", func(t *testing.T) {
		meta, ok := registry.ResolveLocal("My-Override-Model")
		if !ok {
			t.Fatal("expected ok=true for override model")
		}
		if meta.ContextWindow != 111111 {
			t.Errorf("ContextWindow = %d, want 111111", meta.ContextWindow)
		}
		if meta.OutputLimit != 2222 {
			t.Errorf("OutputLimit = %d, want 2222", meta.OutputLimit)
		}
		if meta.TokenizerType != "override-tok" {
			t.Errorf("TokenizerType = %q, want override-tok", meta.TokenizerType)
		}
	})

	t.Run("built-in tier", func(t *testing.T) {
		meta, ok := registry.ResolveLocal("GPT-4O")
		if !ok {
			t.Fatal("expected ok=true for built-in model")
		}
		if meta.ContextWindow != 128000 {
			t.Errorf("ContextWindow = %d, want 128000", meta.ContextWindow)
		}
		if meta.OutputLimit != 16384 {
			t.Errorf("OutputLimit = %d, want 16384", meta.OutputLimit)
		}
		if meta.Family == "" || meta.Protocol == "" {
			t.Error("built-in tier must resolve Family and Protocol postfixes")
		}
	})

	t.Run("fuzzy tier", func(t *testing.T) {
		// "gpt4o" fuzzy-matches the built-in "gpt-4o" (dash dropped by host).
		meta, ok := registry.ResolveLocal("gpt4o")
		if !ok {
			t.Fatal("expected ok=true for fuzzy match")
		}
		if meta.ContextWindow != 128000 {
			t.Errorf("ContextWindow = %d, want 128000", meta.ContextWindow)
		}
		// Like Resolve, a fuzzy hit is cached under the (lowercased) query key.
		registry.mu.RLock()
		_, cached := registry.cache["gpt4o"]
		registry.mu.RUnlock()
		if !cached {
			t.Error("fuzzy hit should be cached under the query key")
		}
	})

	t.Run("lazy cache tier", func(t *testing.T) {
		// Populate the cache the way the network tiers would, then confirm
		// ResolveLocal serves the entry read-only, across casings.
		registry.mu.Lock()
		registry.cache["hf-cached-model"] = ModelMetadata{
			ContextWindow: 7777,
			OutputLimit:   1234,
			TokenizerType: "hf-tok",
		}
		registry.mu.Unlock()

		meta, ok := registry.ResolveLocal("HF-CACHED-MODEL")
		if !ok {
			t.Fatal("expected ok=true for cached model")
		}
		if meta.ContextWindow != 7777 {
			t.Errorf("ContextWindow = %d, want 7777", meta.ContextWindow)
		}
		if meta.OutputLimit != 1234 {
			t.Errorf("OutputLimit = %d, want 1234", meta.OutputLimit)
		}
	})

	t.Run("unknown model falls back offline", func(t *testing.T) {
		meta, ok := registry.ResolveLocal("definitely-not-a-real-model-xyz")
		if ok {
			t.Fatal("expected ok=false for unknown model")
		}
		if meta.ContextWindow != 128000 {
			t.Errorf("fallback ContextWindow = %d, want 128000", meta.ContextWindow)
		}
		if meta.OutputLimit != 32768 {
			t.Errorf("fallback OutputLimit = %d, want 32768", meta.OutputLimit)
		}
		if meta.TokenizerType != "approximate" {
			t.Errorf("fallback TokenizerType = %q, want approximate", meta.TokenizerType)
		}
		if !meta.Capabilities.Attachment {
			t.Error("fallback should keep the optimistic attachment capability")
		}
		if meta.Family == "" || meta.Protocol == "" {
			t.Error("fallback must still resolve Family and Protocol postfixes")
		}
	})

	if transport.calls != 0 {
		t.Fatalf("ResolveLocal must never touch the network; got %d HTTP attempts", transport.calls)
	}
}

func TestModelRegistry_ResolveLocal_UnknownSkipsRegisteredSources(t *testing.T) {
	// Registered sources are a network-capable tier: ResolveLocal must not
	// consult them even when they know the model — that is Resolve's job.
	registry := NewModelRegistry(nil)
	transport := &countingTransport{}
	registry.httpClient = &http.Client{Transport: transport}
	registry.RegisterSource(func(model string) (ModelMetadata, bool) {
		return ModelMetadata{ContextWindow: 55555}, true
	})

	meta, ok := registry.ResolveLocal("source-only-model")
	if ok {
		t.Fatal("expected ok=false: ResolveLocal must not consult registered sources")
	}
	if meta.ContextWindow != 128000 {
		t.Errorf("ContextWindow = %d, want fallback 128000", meta.ContextWindow)
	}
	if transport.calls != 0 {
		t.Fatalf("ResolveLocal must never touch the network; got %d HTTP attempts", transport.calls)
	}
}

func TestModelRegistry_NegativeCache_NoRetryWithinTTL(t *testing.T) {
	// A 404 from HuggingFace is cached as a negative result: repeated Resolve
	// calls inside negativeCacheTTL skip the HTTP round-trip entirely and
	// return the offline fallback.
	callCount := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		callCount++
		w.WriteHeader(http.StatusNotFound)
	}))
	defer server.Close()

	registry := NewModelRegistry(nil)
	registry.httpClient = &http.Client{
		Transport: &rewriteTransport{base: http.DefaultTransport, serverURL: server.URL},
	}

	// First Resolve probes and fails.
	if _, ok := registry.Resolve(context.Background(), "hf-negative-cache-model"); ok {
		t.Fatal("expected ok=false when HuggingFace returns 404")
	}
	if callCount != 1 {
		t.Fatalf("expected 1 HTTP call after first Resolve, got %d", callCount)
	}

	// Repeated resolves within the TTL must not re-probe.
	for i := 0; i < 3; i++ {
		meta, ok := registry.Resolve(context.Background(), "hf-negative-cache-model")
		if ok {
			t.Fatal("expected ok=false while the negative entry is fresh")
		}
		if meta.ContextWindow != 128000 {
			t.Errorf("fallback ContextWindow = %d, want 128000", meta.ContextWindow)
		}
	}
	if callCount != 1 {
		t.Fatalf("negative cache should suppress re-probes within TTL; got %d HTTP calls", callCount)
	}

	// The failure must be recorded under the lowercased key.
	registry.mu.RLock()
	_, recorded := registry.negativeCache["hf-negative-cache-model"]
	registry.mu.RUnlock()
	if !recorded {
		t.Error("expected a negative cache entry after the failed probe")
	}
}

func TestModelRegistry_NegativeCache_TransportErrorAlsoCached(t *testing.T) {
	// Timeouts, DNS failures and other transport errors are negative results
	// too: the counting transport fails every call, so a second attempt would
	// both increment the counter and be observable.
	registry := NewModelRegistry(nil)
	transport := &countingTransport{}
	registry.httpClient = &http.Client{Transport: transport}

	if _, ok := registry.Resolve(context.Background(), "hf-transport-error-model"); ok {
		t.Fatal("expected ok=false when the probe errors")
	}
	if transport.calls != 1 {
		t.Fatalf("expected exactly 1 probe, got %d", transport.calls)
	}

	if _, ok := registry.Resolve(context.Background(), "hf-transport-error-model"); ok {
		t.Fatal("expected ok=false on repeat resolve")
	}
	if transport.calls != 1 {
		t.Fatalf("transport error should be negatively cached within TTL; got %d probes", transport.calls)
	}
}

func TestModelRegistry_NegativeCache_ExpiresAndRecovers(t *testing.T) {
	// After the TTL elapses the probe runs again; a successful re-probe
	// populates the positive cache, clears the negative record, and repeat
	// resolves are served without further HTTP calls.
	fail := true
	callCount := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		callCount++
		if fail {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"max_position_embeddings": 6543}`))
	}))
	defer server.Close()

	registry := NewModelRegistry(nil)
	registry.httpClient = &http.Client{
		Transport: &rewriteTransport{base: http.DefaultTransport, serverURL: server.URL},
	}

	const model = "hf-ttl-expiry-model"
	if _, ok := registry.Resolve(context.Background(), model); ok {
		t.Fatal("expected ok=false while the probe fails")
	}
	if callCount != 1 {
		t.Fatalf("expected 1 HTTP call, got %d", callCount)
	}

	// Simulate TTL expiry by rewinding the recorded failure time, and let the
	// server start answering successfully.
	registry.mu.Lock()
	registry.negativeCache[model] = time.Now().Add(-negativeCacheTTL - time.Second)
	registry.mu.Unlock()
	fail = false

	meta, ok := registry.Resolve(context.Background(), model)
	if !ok {
		t.Fatal("expected ok=true after successful re-probe past the TTL")
	}
	if meta.ContextWindow != 6543 {
		t.Errorf("ContextWindow = %d, want 6543", meta.ContextWindow)
	}
	if callCount != 2 {
		t.Fatalf("expected a re-probe after TTL expiry, got %d HTTP calls", callCount)
	}

	// Success must clear the negative record…
	registry.mu.RLock()
	_, negative := registry.negativeCache[model]
	registry.mu.RUnlock()
	if negative {
		t.Error("negative cache entry should be cleared after a successful probe")
	}

	// …and the positive cache must serve repeats without further calls.
	if _, ok := registry.Resolve(context.Background(), model); !ok {
		t.Fatal("expected ok=true from the positive cache")
	}
	if callCount != 2 {
		t.Errorf("positive cache should serve repeats without HTTP calls; got %d calls", callCount)
	}
}

func TestModelRegistry_Invalidate_ClearsNegativeCache(t *testing.T) {
	// Invalidate means "forget everything you know about this model" — both
	// the positive entry and the failed-probe record, so the next Resolve
	// re-probes instead of waiting out the TTL window.
	callCount := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		callCount++
		w.WriteHeader(http.StatusNotFound)
	}))
	defer server.Close()

	registry := NewModelRegistry(nil)
	registry.httpClient = &http.Client{
		Transport: &rewriteTransport{base: http.DefaultTransport, serverURL: server.URL},
	}

	const model = "hf-invalidate-negative-model"
	if _, ok := registry.Resolve(context.Background(), model); ok {
		t.Fatal("expected ok=false when the probe fails")
	}
	if callCount != 1 {
		t.Fatalf("expected 1 HTTP call, got %d", callCount)
	}

	registry.Invalidate(model)

	if _, ok := registry.Resolve(context.Background(), model); ok {
		t.Fatal("expected ok=false on re-probe")
	}
	if callCount != 2 {
		t.Errorf("Invalidate must clear the negative entry and force a re-probe; got %d HTTP calls", callCount)
	}
}

// blockUntilCanceledTransport holds every request open until its context is
// canceled, then surfaces the context error — an in-flight HuggingFace probe
// aborted by the caller, with no real-network timing and no shared state.
type blockUntilCanceledTransport struct{}

func (blockUntilCanceledTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	<-req.Context().Done()
	return nil, req.Context().Err()
}

func TestModelRegistry_NegativeCache_CallerCancellationNotCached(t *testing.T) {
	// A probe aborted by the CALLER's context cancellation describes the
	// caller (a step deadline hit, a shutdown), not HuggingFace — it must
	// not poison the next ten minutes of resolves. A real 404 afterwards is
	// still a genuine negative result and is cached as before.
	registry := NewModelRegistry(nil)
	registry.SetHTTPClient(&http.Client{Transport: blockUntilCanceledTransport{}})

	const model = "hf-canceled-probe-model"
	ctx, cancel := context.WithCancel(context.Background())
	time.AfterFunc(25*time.Millisecond, cancel)
	if _, ok := registry.Resolve(ctx, model); ok {
		t.Fatal("expected ok=false when the probe is canceled mid-flight")
	}

	// The canceled probe must leave no negative record…
	registry.mu.RLock()
	_, negative := registry.negativeCache[model]
	registry.mu.RUnlock()
	if negative {
		t.Fatal("caller cancellation must not be written to the negative cache")
	}

	// …so the very next Resolve re-probes instead of riding out the TTL. If
	// the canceled probe HAD been recorded, this Resolve would be suppressed
	// and no negative entry could ever appear; a genuine 404 writes one.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	defer server.Close()
	registry.SetHTTPClient(&http.Client{
		Transport: &rewriteTransport{base: http.DefaultTransport, serverURL: server.URL},
	})
	if _, ok := registry.Resolve(context.Background(), model); ok {
		t.Fatal("expected ok=false when the re-probe 404s")
	}

	registry.mu.RLock()
	_, negative = registry.negativeCache[model]
	registry.mu.RUnlock()
	if !negative {
		t.Fatal("the post-cancellation re-probe must have run and 404'd (a fresh negative entry proves no suppression)")
	}
}

func TestModelRegistry_NegativeCache_CallerDeadlineNotCached(t *testing.T) {
	// A probe aborted by the CALLER's deadline describes the caller's step
	// deadline, not HuggingFace, so it must not poison the next ten minutes
	// of resolves — exactly like caller cancellation. The registry's OWN HTTP
	// client timeout still records (see the client-timeout test below).
	server := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
		<-r.Context().Done()
	}))
	defer server.Close()

	registry := NewModelRegistry(nil)
	registry.httpClient = &http.Client{
		Transport: &rewriteTransport{base: http.DefaultTransport, serverURL: server.URL},
	}

	const model = "hf-deadline-probe-model"
	ctx, cancel := context.WithTimeout(context.Background(), 25*time.Millisecond)
	defer cancel()
	if _, ok := registry.Resolve(ctx, model); ok {
		t.Fatal("expected ok=false when the probe misses its deadline")
	}

	registry.mu.RLock()
	_, negative := registry.negativeCache[model]
	registry.mu.RUnlock()
	if negative {
		t.Fatal("caller deadline must not be written to the negative cache")
	}
}

func TestModelRegistry_NegativeCache_ClientTimeoutStillCached(t *testing.T) {
	// The registry's OWN HTTP client timeout is a genuine HuggingFace failure:
	// the caller's context is still live (ctx.Err() == nil), so the negative
	// record is written and suppresses re-probes for the TTL. This is the
	// distinction the caller-deadline exemption must preserve.
	server := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
		<-r.Context().Done()
	}))
	defer server.Close()

	registry := NewModelRegistry(nil)
	registry.httpClient = &http.Client{
		Timeout:   25 * time.Millisecond,
		Transport: &rewriteTransport{base: http.DefaultTransport, serverURL: server.URL},
	}

	const model = "hf-client-timeout-model"
	if _, ok := registry.Resolve(context.Background(), model); ok {
		t.Fatal("expected ok=false when the client times out")
	}

	registry.mu.RLock()
	_, negative := registry.negativeCache[model]
	registry.mu.RUnlock()
	if !negative {
		t.Fatal("the registry's own client timeout must be negatively cached")
	}
}

func TestModelRegistry_NegativeCache_StaleEntryEvictedOnRead(t *testing.T) {
	// A negative record past its TTL window is not merely ignored — it is
	// deleted at read time, so keys that are never resolved again cannot
	// accumulate in the map forever.
	registry := NewModelRegistry(nil)
	transport := &countingTransport{}
	registry.httpClient = &http.Client{Transport: transport}

	const model = "hf-stale-negative-model"
	if _, ok := registry.Resolve(context.Background(), model); ok {
		t.Fatal("expected ok=false when the probe errors")
	}

	registry.mu.Lock()
	registry.negativeCache[model] = time.Now().Add(-negativeCacheTTL - time.Second)
	registry.mu.Unlock()

	if registry.negativeCacheFresh(model) {
		t.Fatal("an entry past the TTL window must read as stale")
	}
	registry.mu.RLock()
	_, present := registry.negativeCache[model]
	registry.mu.RUnlock()
	if present {
		t.Error("a stale negative entry must be evicted on read, not merely skipped")
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

// TestModelRegistry_SetRuntimeMetadata_BeatsBuiltIn verifies the tier-1.5
// semantics: an OBSERVED runtime entry supersedes the built-in catalog spec.
// This is the self-hosted scenario the tier exists for — LM Studio/Ollama/
// vLLM serve a well-known checkpoint at a runtime context length far below
// the catalog maximum ("qwen/qwen3.6-35b-a3b" catalogs at 262144 but the
// server may enforce e.g. 32768).
func TestModelRegistry_SetRuntimeMetadata_BeatsBuiltIn(t *testing.T) {
	registry := NewModelRegistry(nil)

	// Built-in entry exists with ContextWindow 262144.
	registry.SetRuntimeMetadata("qwen/qwen3.6-35b-a3b", ModelMetadata{
		ContextWindow: 32768,
		OutputLimit:   8192,
		TokenizerType: "approximate",
	})

	meta, ok := registry.ResolveLocal("qwen/qwen3.6-35b-a3b")
	if !ok {
		t.Fatal("expected runtime entry to resolve")
	}
	if meta.ContextWindow != 32768 {
		t.Errorf("runtime entry did not supersede built-in: got %d, want 32768", meta.ContextWindow)
	}
	if meta.OutputLimit != 8192 {
		t.Errorf("runtime OutputLimit not honored: got %d, want 8192", meta.OutputLimit)
	}
}

// TestModelRegistry_SetRuntimeMetadata_UserOverrideStillWins verifies tier 1
// beats tier 1.5: an explicit config.yaml override is never clobbered by an
// observed runtime entry.
func TestModelRegistry_SetRuntimeMetadata_UserOverrideStillWins(t *testing.T) {
	registry := NewModelRegistry(map[string]ModelMetadata{
		"qwen/qwen3.6-35b-a3b": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
		},
	})

	registry.SetRuntimeMetadata("qwen/qwen3.6-35b-a3b", ModelMetadata{
		ContextWindow: 32768,
		OutputLimit:   8192,
		TokenizerType: "approximate",
	})

	meta, _ := registry.ResolveLocal("qwen/qwen3.6-35b-a3b")
	if meta.ContextWindow != 131072 {
		t.Errorf("user override clobbered by runtime: got %d, want 131072", meta.ContextWindow)
	}
}

// TestModelRegistry_SetRuntimeMetadata_BeatsCache verifies the runtime tier
// also shadows the lazy cache (tier 3): a HuggingFace config.json lookup
// cached under the checkpoint name cannot pin the spec window over the
// observed runtime window.
func TestModelRegistry_SetRuntimeMetadata_BeatsCache(t *testing.T) {
	registry := NewModelRegistry(nil)

	// Simulate an earlier HuggingFace fetch having cached the checkpoint spec.
	registry.SetCachedMetadata("qwen/qwen3.6-35b-a3b", ModelMetadata{
		ContextWindow: 262144,
		OutputLimit:   65536,
		TokenizerType: "approximate",
	})
	// The local-server probe then observes the runtime window.
	registry.SetRuntimeMetadata("qwen/qwen3.6-35b-a3b", ModelMetadata{
		ContextWindow: 32768,
		OutputLimit:   8192,
		TokenizerType: "approximate",
	})

	meta, _ := registry.ResolveLocal("qwen/qwen3.6-35b-a3b")
	if meta.ContextWindow != 32768 {
		t.Errorf("cache shadowed runtime: got %d, want 32768", meta.ContextWindow)
	}
}

// TestModelRegistry_SetRuntimeMetadata_PartialUserOverrideInheritsWindow is
// the motivating regression for the c0wrk self-hosted bug: the user pins ONLY
// the output limit in config.yaml (context window left unset = inherit), and
// the self-hosted server's runtime window must surface through the partial
// override via enrichPartialOverride — not the built-in catalog spec.
func TestModelRegistry_SetRuntimeMetadata_PartialUserOverrideInheritsWindow(t *testing.T) {
	// Partial override: only OutputLimit pinned (ContextWindow 0 = inherit).
	registry := NewModelRegistry(map[string]ModelMetadata{
		"qwen/qwen3.6-35b-a3b": {OutputLimit: 65536},
	})

	registry.SetRuntimeMetadata("qwen/qwen3.6-35b-a3b", ModelMetadata{
		ContextWindow: 32768,
		OutputLimit:   8192,
		TokenizerType: "approximate",
	})

	meta, _ := registry.ResolveLocal("qwen/qwen3.6-35b-a3b")
	if meta.ContextWindow != 32768 {
		t.Errorf("partial override did not inherit runtime window: got %d, want 32768", meta.ContextWindow)
	}
	if meta.OutputLimit != 65536 {
		t.Errorf("user-pinned OutputLimit not authoritative: got %d, want 65536", meta.OutputLimit)
	}
}

// TestModelRegistry_SetRuntimeMetadata_PartialRuntimeEntryEnriches verifies a
// PARTIAL runtime entry inherits its unset scalars from the tiers BELOW it
// (built-in → cache → fallback) — not from itself.
func TestModelRegistry_SetRuntimeMetadata_PartialRuntimeEntryEnriches(t *testing.T) {
	registry := NewModelRegistry(nil)

	// Runtime entry observes only the window; output limit/tokenizer inherit
	// from the built-in catalog entry (qwen/qwen3.6-35b-a3b: 65536 output).
	registry.SetRuntimeMetadata("qwen/qwen3.6-35b-a3b", ModelMetadata{ContextWindow: 32768})

	meta, _ := registry.ResolveLocal("qwen/qwen3.6-35b-a3b")
	if meta.ContextWindow != 32768 {
		t.Errorf("runtime window lost during enrichment: got %d, want 32768", meta.ContextWindow)
	}
	if meta.OutputLimit != 65536 {
		t.Errorf("unset runtime OutputLimit did not inherit built-in 65536: got %d", meta.OutputLimit)
	}
	if meta.TokenizerType == "" {
		t.Error("unset runtime TokenizerType did not inherit from lower tiers")
	}
}

// TestModelRegistry_Invalidate_ClearsRuntime verifies Invalidate drops the
// observed runtime entry along with the cache, so a model switch re-probes
// instead of pinning the previous serving arrangement's window.
func TestModelRegistry_Invalidate_ClearsRuntime(t *testing.T) {
	registry := NewModelRegistry(nil)

	registry.SetRuntimeMetadata("qwen/qwen3.6-35b-a3b", ModelMetadata{
		ContextWindow: 32768,
		OutputLimit:   8192,
		TokenizerType: "approximate",
	})
	registry.Invalidate("qwen/qwen3.6-35b-a3b")

	if _, ok := registry.RuntimeMetadata("qwen/qwen3.6-35b-a3b"); ok {
		t.Error("Invalidate did not clear the runtime entry")
	}
	// Post-invalidate resolution falls back to the built-in spec.
	meta, _ := registry.ResolveLocal("qwen/qwen3.6-35b-a3b")
	if meta.ContextWindow != 262144 {
		t.Errorf("post-invalidate window = %d, want built-in 262144", meta.ContextWindow)
	}
}

// TestModelRegistry_SetRuntimeMetadata_FuzzyLookupFindsRuntimeEntry is the
// drifted-spelling scenario the runtime fuzzy mirror exists for: a probe
// stored the observed window under the canonical id
// "qwen/qwen3.6-35b-a3b", and a later resolve arrives under a host spelling
// with the dot dropped ("Qwen/Qwen36-35B-A3B" — the exact case the fuzzy tier
// exists to bridge). Without the runtime leg in the fuzzy tier, the lookup
// falls through to the built-in catalog maximum (262144) and silently ignores
// the server's observed 32768.
func TestModelRegistry_SetRuntimeMetadata_FuzzyLookupFindsRuntimeEntry(t *testing.T) {
	registry := NewModelRegistry(nil)

	registry.SetRuntimeMetadata("qwen/qwen3.6-35b-a3b", ModelMetadata{
		ContextWindow: 32768,
		OutputLimit:   8192,
		TokenizerType: "approximate",
	})

	meta, ok := registry.ResolveLocal("Qwen/Qwen36-35B-A3B")
	if !ok {
		t.Fatal("expected fuzzy lookup to find the runtime entry")
	}
	if meta.ContextWindow != 32768 {
		t.Errorf("fuzzy lookup lost the observed runtime window: got %d, want 32768", meta.ContextWindow)
	}
}

// TestModelRegistry_SetRuntimeMetadata_FuzzyPartialRuntimeEnriches verifies a
// PARTIAL runtime entry found via the fuzzy tier enriches its unset scalars
// from the tiers below runtime, mirroring the exact tier-1.5 path: the probe
// observed only the window, and the catalog output limit (65536 for
// qwen/qwen3.6-35b-a3b) must still surface under the drifted spelling.
func TestModelRegistry_SetRuntimeMetadata_FuzzyPartialRuntimeEnriches(t *testing.T) {
	registry := NewModelRegistry(nil)

	registry.SetRuntimeMetadata("qwen/qwen3.6-35b-a3b", ModelMetadata{ContextWindow: 32768})

	meta, ok := registry.ResolveLocal("Qwen/Qwen36-35B-A3B")
	if !ok {
		t.Fatal("expected fuzzy lookup to find the runtime entry")
	}
	if meta.ContextWindow != 32768 {
		t.Errorf("fuzzy runtime window lost during enrichment: got %d, want 32768", meta.ContextWindow)
	}
	if meta.OutputLimit != 65536 {
		t.Errorf("unset fuzzy runtime OutputLimit did not inherit built-in 65536: got %d", meta.OutputLimit)
	}
}

// TestModelRegistry_SetRuntimeMetadata_FuzzyOverrideBeatsRuntime verifies the
// fuzzy tier mirrors the exact-tier precedence override > runtime > built-in:
// when a user override and a runtime entry normalize to the same form, the
// override wins even under a drifted query spelling.
func TestModelRegistry_SetRuntimeMetadata_FuzzyOverrideBeatsRuntime(t *testing.T) {
	registry := NewModelRegistry(map[string]ModelMetadata{
		"qwen/qwen3.6-35b-a3b": {
			ContextWindow: 131072,
			OutputLimit:   32768,
			TokenizerType: "approximate",
		},
	})

	registry.SetRuntimeMetadata("qwen/qwen3.6-35b-a3b", ModelMetadata{
		ContextWindow: 32768,
		OutputLimit:   8192,
		TokenizerType: "approximate",
	})

	meta, ok := registry.ResolveLocal("Qwen/Qwen36-35B-A3B")
	if !ok {
		t.Fatal("expected fuzzy lookup to resolve")
	}
	if meta.ContextWindow != 131072 {
		t.Errorf("fuzzy lookup clobbered user override with runtime entry: got %d, want 131072", meta.ContextWindow)
	}
}

// TestModelRegistry_Invalidate_ClearsRuntimeFuzzyIndex verifies Invalidate
// drops the normalized-ID mirror along with the runtime entry, so a resolve
// under a drifted spelling falls back to the built-in spec instead of the
// previous serving arrangement's window.
func TestModelRegistry_Invalidate_ClearsRuntimeFuzzyIndex(t *testing.T) {
	registry := NewModelRegistry(nil)

	registry.SetRuntimeMetadata("qwen/qwen3.6-35b-a3b", ModelMetadata{
		ContextWindow: 32768,
		OutputLimit:   8192,
		TokenizerType: "approximate",
	})
	registry.Invalidate("qwen/qwen3.6-35b-a3b")

	meta, ok := registry.ResolveLocal("Qwen/Qwen36-35B-A3B")
	if !ok {
		t.Fatal("expected fuzzy lookup to find the built-in entry")
	}
	if meta.ContextWindow != 262144 {
		t.Errorf("post-invalidate fuzzy window = %d, want built-in 262144", meta.ContextWindow)
	}
}

// TestModelRegistry_SetRuntimeMetadata_ConcurrentFuzzyReads hammers the
// runtime fuzzy index from concurrent readers and writers (run under -race):
// the index is rebuilt under the write lock on every store, so readers must
// always observe a consistent map and the observed window.
func TestModelRegistry_SetRuntimeMetadata_ConcurrentFuzzyReads(t *testing.T) {
	registry := NewModelRegistry(nil)
	registry.SetRuntimeMetadata("qwen/qwen3.6-35b-a3b", ModelMetadata{ContextWindow: 32768})

	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(2)
		go func() {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				registry.SetRuntimeMetadata("qwen/qwen3.6-35b-a3b", ModelMetadata{
					ContextWindow: 32768,
					OutputLimit:   8192,
					TokenizerType: "approximate",
				})
			}
		}()
		go func() {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				meta, ok := registry.ResolveLocal("Qwen/Qwen36-35B-A3B")
				if !ok || meta.ContextWindow != 32768 {
					t.Errorf("ResolveLocal(Qwen/Qwen36-35B-A3B) = (window %d, ok %t), want (32768, true)", meta.ContextWindow, ok)
					return
				}
			}
		}()
	}
	wg.Wait()
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
	// Built-in lookup is case-insensitive, mirroring Resolve. Capabilities is
	// compared by VALUE: each call hands the caller its own defensive copy
	// (see finalizeMeta), so pointer identity between two calls is
	// deliberately not guaranteed.
	upper, okUpper := ResolveBuiltInModel("GPT-4O")
	lower, okLower := ResolveBuiltInModel("gpt-4o")
	if !okUpper || !okLower {
		t.Fatal("expected both lookups to succeed")
	}
	upperCaps, lowerCaps := upper.Capabilities, lower.Capabilities
	upper.Capabilities, lower.Capabilities = nil, nil
	if upper != lower {
		t.Errorf("case-insensitive lookup mismatch: %+v vs %+v", upper, lower)
	}
	if upperCaps == nil || lowerCaps == nil || *upperCaps != *lowerCaps {
		t.Errorf("case-insensitive capabilities mismatch: %+v vs %+v", upperCaps, lowerCaps)
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
	if meta.OutputLimit != 32768 {
		t.Errorf("expected fallback OutputLimit 32768, got %d", meta.OutputLimit)
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
