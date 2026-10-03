package llm

import "strings"

// openAIModelReasoningOptions keeps model restrictions ahead of family defaults.
// In particular, minimal is a GPT-5 option, not a universal OpenAI floor.
// Sources: developers.openai.com/api/docs/models/{gpt-5.1,gpt-5.2,
// gpt-5.3-codex,gpt-5.4,gpt-5-pro,gpt-5.2-pro,gpt-5.6-luna} and
// developers.openai.com/api/docs/guides/reasoning.
func openAIModelReasoningOptions(model string) (options []string, preferred string, ok bool) {
	bare := strings.ToLower(strings.TrimSpace(BareModel(model)))
	variant := func(base string) bool {
		return bare == base || strings.HasPrefix(bare, base+"-")
	}
	switch {
	case variant("gpt-5-pro"), variant("o1-pro"):
		return []string{"high"}, "high", true
	case strings.Contains(bare, "codex"):
		// Codex models do not accept minimal or none. Older Codex models
		// share low/medium/high; the 5.3 generation also documents xhigh.
		if variant("gpt-5.3-codex") || variant("gpt-5.2-codex") || variant("gpt-5.1-codex-max") {
			return []string{"low", "medium", "high", "xhigh"}, "xhigh", true
		}
		return []string{"low", "medium", "high"}, "high", true
	case variant("gpt-5.2-pro"), variant("gpt-5.4-pro"), variant("gpt-5.5-pro"):
		return []string{"medium", "high", "xhigh"}, "xhigh", true
	case variant("gpt-5.1"):
		return []string{"none", "low", "medium", "high"}, "high", true
	case variant("gpt-5.2"), variant("gpt-5.4"), variant("gpt-5.5"):
		return []string{"none", "low", "medium", "high", "xhigh"}, "xhigh", true
	case variant("gpt-5.6"):
		return []string{"none", "low", "medium", "high", "xhigh", "max"}, "max", true
	case variant("gpt-6"), variant("gpt-6.1"):
		// The GPT-6 reasoning contract rejects none (6.1 Sol also rejects
		// minimal); low is the documented reasoning floor.
		return []string{"low", "medium", "high", "xhigh", "max"}, "max", true
	case variant("o1"), variant("o3"), variant("o4"):
		return []string{"low", "medium", "high"}, "high", true
	default:
		return nil, "", false
	}
}
