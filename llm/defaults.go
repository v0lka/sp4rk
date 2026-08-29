package llm

import "time"

// DefaultOutputTokenReserve is the number of tokens reserved for model output
// when no explicit OutputTokenReserve is configured. 4096 is generous enough
// for most assistant responses (including multi-tool-call replies) without
// wasting context on an unnecessarily large reserve.
const DefaultOutputTokenReserve = 4096

// DefaultRouterConfig returns sensible defaults for Router configuration.
// The Providers field must be set by the caller since there is no sensible
// default for provider selection. Use this as a starting point and override as needed.
func DefaultRouterConfig() RouterConfig {
	return RouterConfig{
		MaxRetries:          3,
		InitialBackoff:      1 * time.Second,
		MaxBackoff:          30 * time.Second,
		SafetyMarginPercent: 5,
		OutputTokenReserve:  DefaultOutputTokenReserve,
	}
}

// DeterministicTemperature returns the temperature the router injects into
// routing / compaction / summarization calls (see CallPurpose). Most families
// accept fully greedy decoding (0.0). Families whose vendors document
// instability or degraded output at low temperature get their documented
// floor instead — a family-safe minimum that keeps the call as deterministic
// as the family allows:
//
//   - google: Gemini degrades into repetition loops at low temperature; the
//     vendor default of 1.0 is the documented safe point.
//   - kimi: the Kimi Code subscription endpoint (api.kimi.com/coding) rejects
//     any temperature other than 1 (HTTP 400 "invalid temperature: only 1 is
//     allowed for this model"), and Kimi thinking modes document 1.0 as the
//     only stable value. The coding-endpoint models are capability-gated out
//     of sampling injection entirely (registry Temperature=false); 1.0 keeps
//     deterministic calls accepted for kimi-family entries that still declare
//     Temperature (platform kimi-k3 / k2.5 / k2.6 / k2).
//   - qwen: Qwen3 thinking mode is documented to misbehave at low
//     temperature (the superseded Qwen3 quickstart recommended a 0.6 minimum;
//     the Qwen3.8 card now recommends 1.0 for thinking). 0.6 is retained as
//     a conservative family-safe floor for deterministic (routing/compaction)
//     calls, which the Qwen3.8 guidance does not contradict as a minimum.
//
// Reasoning models that reject the temperature parameter entirely never reach
// this function — applyDefaultSampling gates on Capabilities.Temperature
// first. Families not listed here return 0.0 (exact determinism).
func DeterministicTemperature(family string) *float64 {
	var t float64
	switch family {
	case "google", "kimi":
		t = 1.0
	case "qwen":
		t = 0.6
	default:
		t = 0.0
	}
	return &t
}
