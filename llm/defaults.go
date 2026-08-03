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
