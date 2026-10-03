package llm

import (
	"testing"
)

// TestGLMAlwaysThinkingWire guards both service tiers and stale host settings.
// The model option set and the JSON encoder must agree on the allowed floor.
func TestGLMAlwaysThinkingWire(t *testing.T) {
	p, err := NewOpenAIProvider(OpenAIProviderConfig{Name: "compatible", APIKey: "test"})
	if err != nil {
		t.Fatalf("NewOpenAIProvider(compatible) error = %v, want nil", err)
	}
	models := []string{"glm-5.3", "glm-5.3-flash", "Z-ai-API/GLM-5.3", "zai-org/glm-5.3-flash"}
	efforts := []struct {
		input string
		want  string
	}{
		{"Off", "low"}, {"off", "low"}, {"NONE", "low"}, {"none", "low"},
		{"low", "low"}, {"LOW", "low"}, {"high", "high"}, {"max", "max"},
		{"On", "max"}, {"turbo", "low"},
	}
	for _, model := range models {
		t.Run(model, func(t *testing.T) {
			for _, tier := range []ReasoningTier{ReasoningTierOff, ReasoningTierMinimal} {
				if got := ReasoningForCall("glm", model, tier); got != "low" {
					t.Errorf("ReasoningForCall(glm, %q, %q) = %q, want low", model, tier, got)
				}
			}
			for _, effort := range efforts {
				t.Run(effort.input, func(t *testing.T) {
					req := ChatRequest{Model: model, ReasoningEffort: effort.input}
					body := jsonMap(t, p.buildChatParams(req))
					thinking, ok := body["thinking"].(map[string]any)
					if !ok || thinking["type"] != "enabled" {
						t.Errorf("buildChatParams(%q, %q).thinking = %v, want type enabled", model, effort.input, body["thinking"])
					}
					if got := body["reasoning_effort"]; got != effort.want {
						t.Errorf("buildChatParams(%q, %q).reasoning_effort = %v, want %q", model, effort.input, got, effort.want)
					}
				})
			}
		})
	}
}
