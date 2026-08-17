package prompt

import (
	"testing"
)

// expectF64 asserts a *float64 field matches the expectation (nil-safe).
func expectF64(t *testing.T, label string, got, want *float64) {
	t.Helper()
	switch {
	case got == nil && want == nil:
		return
	case got == nil && want != nil:
		t.Errorf("%s: expected %v, got nil", label, *want)
	case got != nil && want == nil:
		t.Errorf("%s: expected nil, got %v", label, *got)
	case *got != *want:
		t.Errorf("%s: expected %v, got %v", label, *want, *got)
	}
}

// expectInt asserts a *int field matches the expectation (nil-safe).
func expectInt(t *testing.T, label string, got, want *int) {
	t.Helper()
	switch {
	case got == nil && want == nil:
		return
	case got == nil && want != nil:
		t.Errorf("%s: expected %d, got nil", label, *want)
	case got != nil && want == nil:
		t.Errorf("%s: expected nil, got %d", label, *got)
	case *got != *want:
		t.Errorf("%s: expected %d, got %d", label, *want, *got)
	}
}

func TestDefaultSampling(t *testing.T) {
	tests := []struct {
		name     string
		family   string
		wantTemp *float64
		wantTopP *float64
		wantTopK *int
		wantRep  *float64
	}{
		{
			name:     "anthropic returns all nil (model self-selects)",
			family:   "anthropic",
			wantTemp: nil,
			wantTopP: nil,
		},
		{
			// GPT-5 / o-series reasoning models reject temperature and top_p
			// overrides — everything must stay nil.
			name:     "openai_flagship returns all nil (reasoning models)",
			family:   "openai_flagship",
			wantTemp: nil,
			wantTopP: nil,
		},
		{
			name:     "openai_standard returns 0.3 temperature",
			family:   "openai_standard",
			wantTemp: fp(0.3),
			wantTopP: nil,
		},
		{
			name:     "google returns 1.0 temperature",
			family:   "google",
			wantTemp: fp(1.0),
			wantTopP: nil,
		},
		{
			// Server-side default is already 0.7 — no override sent.
			name:     "mistral returns all nil (server default 0.7)",
			family:   "mistral",
			wantTemp: nil,
			wantTopP: nil,
		},
		{
			name:     "deepseek returns 0.0 temperature",
			family:   "deepseek",
			wantTemp: fp(0.0),
			wantTopP: nil,
		},
		{
			// qwen.readthedocs.io quickstart, thinking-mode default.
			name:     "qwen returns thinking default 0.6/0.95/20",
			family:   "qwen",
			wantTemp: fp(0.6),
			wantTopP: fp(0.95),
			wantTopK: ip(20),
		},
		{
			// docs.z.ai "Migrate to GLM-4.6": temperature 1.0, top_p 0.95,
			// top_k 40.
			name:     "glm returns 1.0/0.95/40",
			family:   "glm",
			wantTemp: fp(1.0),
			wantTopP: fp(0.95),
			wantTopK: ip(40),
		},
		{
			name:     "kimi returns all nil (server-managed)",
			family:   "kimi",
			wantTemp: nil,
			wantTopP: nil,
		},
		{
			name:     "default family returns 0.5 temp and 0.95 topP",
			family:   "default",
			wantTemp: fp(0.5),
			wantTopP: fp(0.95),
		},
		{
			name:     "unknown family falls back to default",
			family:   "unknown_provider",
			wantTemp: fp(0.5),
			wantTopP: fp(0.95),
		},
		{
			name:     "empty family falls back to default",
			family:   "",
			wantTemp: fp(0.5),
			wantTopP: fp(0.95),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := DefaultSampling(tt.family)
			expectF64(t, "Temperature", got.Temperature, tt.wantTemp)
			expectF64(t, "TopP", got.TopP, tt.wantTopP)
			expectInt(t, "TopK", got.TopK, tt.wantTopK)
			expectF64(t, "RepetitionPenalty", got.RepetitionPenalty, tt.wantRep)
			if got.PresencePenalty != nil {
				t.Errorf("PresencePenalty: expected nil, got %v", *got.PresencePenalty)
			}
			if got.MaxTokens != nil {
				t.Errorf("MaxTokens: expected nil, got %d", *got.MaxTokens)
			}
		})
	}
}

// TestDefaultSamplingCodexIsNil verifies openai_codex no longer falls through
// to the generic default (0.5 / 0.95): Codex models are reasoning models on
// the Responses API and must get an all-nil preset.
func TestDefaultSamplingCodexIsNil(t *testing.T) {
	got := DefaultSampling("openai_codex")

	if got.Temperature != nil {
		t.Errorf("Temperature: expected nil, got %v", *got.Temperature)
	}
	if got.TopP != nil {
		t.Errorf("TopP: expected nil, got %v", *got.TopP)
	}
	if got.TopK != nil {
		t.Errorf("TopK: expected nil, got %d", *got.TopK)
	}
	if got.RepetitionPenalty != nil {
		t.Errorf("RepetitionPenalty: expected nil, got %v", *got.RepetitionPenalty)
	}
	if got.PresencePenalty != nil {
		t.Errorf("PresencePenalty: expected nil, got %v", *got.PresencePenalty)
	}
	if got.MaxTokens != nil {
		t.Errorf("MaxTokens: expected nil, got %d", *got.MaxTokens)
	}
}
