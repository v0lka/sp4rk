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
//   - qwen: Qwen3 thinking mode misbehaves below 0.6; the quickstart pins
//     0.6 as the recommended minimum for both modes.
//
// Reasoning models that reject the temperature parameter entirely never reach
// this function — applyDefaultSampling gates on Capabilities.Temperature
// first. Families not listed here return 0.0 (exact determinism).
func DeterministicTemperature(family string) *float64 {
	var t float64
	switch family {
	case "google":
		t = 1.0
	case "qwen":
		t = 0.6
	default:
		t = 0.0
	}
	return &t
}
