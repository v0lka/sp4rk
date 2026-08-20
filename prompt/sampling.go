package prompt

// SamplingConfig holds family-aware generation parameter defaults.
// Pointer fields indicate "set" vs "unset" — nil means no override.
type SamplingConfig struct {
	Temperature       *float64
	TopP              *float64
	TopK              *int
	RepetitionPenalty *float64
	PresencePenalty   *float64
	MaxTokens         *int
}

// DefaultSampling returns recommended generation parameters for the given model family.
// These are advisory defaults — providers should use them only when no explicit
// user overrides are set.
func DefaultSampling(family string) SamplingConfig {
	switch family {
	case "anthropic":
		// Anthropic recommends letting model self-select temperature
		return SamplingConfig{} // all nil
	case "openai_flagship":
		// This family mixes reasoning (o-series) and non-reasoning (gpt-4o /
		// gpt-4o-mini) members. Reasoning models reject temperature/top_p
		// overrides, but applyDefaultSampling already skips any model whose
		// Capabilities do not declare Temperature=true, so returning the
		// non-reasoning preset (0.3) here only affects gpt-4o/gpt-4o-mini.
		// gpt-4.1-class models live in openai_standard.
		return SamplingConfig{Temperature: fp(0.3)}
	case "openai_codex":
		// Codex models (e.g. gpt-5.x-codex) run on the Responses API and share
		// the reasoning-model restriction of openai_flagship: temperature and
		// top_p overrides are rejected. Previously this family fell through to
		// the generic default (0.5 / 0.95), which is wrong for reasoning models.
		// Source: platform.openai.com/docs/guides/reasoning.
		return SamplingConfig{} // all nil
	case "openai_standard":
		// Pre-existing repo recommendation for gpt-4o/gpt-4.1-class models
		// (unchanged in this update; OpenAI publishes no stricter per-model value).
		return SamplingConfig{Temperature: fp(0.3)}
	case "google":
		// Source: ai.google.dev Gemini API reference — temperature "defaults to 1.0".
		return SamplingConfig{Temperature: fp(1.0)} // low values cause looping
	case "mistral":
		// La Plateforme applies a server-side default temperature of 0.7 when
		// the parameter is omitted — no override needed.
		// Source: docs.mistral.ai API reference (chat completions, temperature default 0.7).
		return SamplingConfig{} // all nil — let the server default apply
	case "deepseek":
		// Pre-existing repo recommendation for coding/math tasks
		// (unchanged in this update; ignored when thinking mode is enabled).
		return SamplingConfig{Temperature: fp(0.0)}
	case "qwen":
		// Qwen3 general / thinking-mode default.
		// Source: qwen.readthedocs.io quickstart — "We recommend
		// temperature=0.6, top_p=0.95, top_k=20, and min_p=0".
		return SamplingConfig{
			Temperature: fp(0.6),
			TopP:        fp(0.95),
			TopK:        ip(20),
		}
	case "glm":
		// Source: docs.z.ai "Migrate to GLM-4.6" — "temperature default value
		// 1.0, top_p default value 0.95" (recommend tuning only one of them);
		// top_k=40 for coding/agentic workloads per Z.AI recommended settings.
		return SamplingConfig{
			Temperature: fp(1.0),
			TopP:        fp(0.95),
			TopK:        ip(40),
		}
	case "kimi":
		// Kimi server enforces temperature per model (0.6 standard, 1.0 thinking)
		return SamplingConfig{} // all nil — let server decide
	default:
		// Generic fallback for families without vendor guidance — pre-existing
		// repo default (unchanged in this update).
		return SamplingConfig{
			Temperature: fp(0.5),
			TopP:        fp(0.95),
		}
	}
}

// fp is a helper that returns a pointer to a float64 value.
func fp(v float64) *float64 { return &v }

// ip is a helper that returns a pointer to an int value.
func ip(v int) *int { return &v }
