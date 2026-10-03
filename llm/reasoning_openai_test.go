package llm

import (
	"slices"
	"testing"
)

func TestOpenAIModelReasoningOptions(t *testing.T) {
	tests := []struct {
		model string
		opts  []string
		off   string
		min   string
	}{
		{"gpt-5", []string{"minimal", "low", "medium", "high"}, "minimal", "minimal"},
		{"gpt-5.1", []string{"none", "low", "medium", "high"}, "none", "low"},
		{"OpenAI/GPT-5.2-2025-12-11", []string{"none", "low", "medium", "high", "xhigh"}, "none", "low"},
		{"gpt-5.4", []string{"none", "low", "medium", "high", "xhigh"}, "none", "low"},
		{"gpt-5.6-luna", []string{"none", "low", "medium", "high", "xhigh", "max"}, "none", "low"},
		{"gpt-6-astra", []string{"low", "medium", "high", "xhigh", "max"}, "low", "low"},
		{"gpt-6.1-sol", []string{"low", "medium", "high", "xhigh", "max"}, "low", "low"},
		{"gpt-5-pro", []string{"high"}, "high", "high"},
		{"gpt-5.2-pro", []string{"medium", "high", "xhigh"}, "medium", "medium"},
		{"gpt-5.4-pro", []string{"medium", "high", "xhigh"}, "medium", "medium"},
		{"gpt-5.5-pro", []string{"medium", "high", "xhigh"}, "medium", "medium"},
		{"gpt-5.3-codex", []string{"low", "medium", "high", "xhigh"}, "low", "low"},
		{"gpt-5.2-codex", []string{"low", "medium", "high", "xhigh"}, "low", "low"},
		{"o3", []string{"low", "medium", "high"}, "low", "low"},
	}
	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			family := string(DetectFamily(tt.model))
			opts, _, ok := ModelReasoningOptions(family, tt.model)
			if !ok || !slices.Equal(opts, tt.opts) {
				t.Errorf("ModelReasoningOptions(%q, %q) = %v, %t, want %v, true", family, tt.model, opts, ok, tt.opts)
			}
			for _, tier := range []struct {
				input ReasoningTier
				want  string
			}{{ReasoningTierOff, tt.off}, {ReasoningTierMinimal, tt.min}} {
				got := ReasoningForCall(family, tt.model, tier.input)
				if got != tier.want {
					t.Errorf("ReasoningForCall(%q, %q, %q) = %q, want %q", family, tt.model, tier.input, got, tier.want)
				}
				params := buildResponsesParams(ChatRequest{Model: tt.model, ReasoningEffort: got}, "", nil)
				if effort := string(params.Reasoning.Effort); effort != tier.want {
					t.Errorf("buildResponsesParams(%q, %q).effort = %q, want %q", tt.model, tier.input, effort, tier.want)
				}
			}
		})
	}
}
