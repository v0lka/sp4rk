package llm

import (
	"regexp"
	"strconv"
	"strings"
)

// FamilyReasoningOptions returns the native reasoning/thinking options available
// for a given model family. It also returns the recommended default (always the
// maximum available effort) and whether the family supports reasoning at all.
//
// Note: this is the family-level view. When the specific model version matters
// (e.g. GLM 5.2+ introduced reasoning_effort), use ModelReasoningOptions.
func FamilyReasoningOptions(family string) (options []string, preferred string, ok bool) {
	switch family {
	case "anthropic":
		return []string{"On", "Off"}, "On", true
	case "openai_flagship", "openai_standard":
		return []string{"minimal", "low", "medium", "high"}, "high", true
	case "openai_codex":
		return []string{"minimal", "low", "medium", "high", "max"}, "max", true
	case "google":
		return []string{"MINIMAL", "LOW", "MEDIUM", "HIGH"}, "HIGH", true
	case "deepseek":
		return []string{"Off", "High", "Max"}, "Max", true
	case "qwen":
		// Qwen3.8+ exposes native per-request reasoning_effort (xhigh
		// default, medium, low) with thinking on by default; "Off" disables
		// it. The legacy "On" value is still accepted by the provider as an
		// alias of the native default "xhigh". Pre-3.8 Qwen models do not
		// know reasoning_effort and keep the binary On/Off control — use
		// ModelReasoningOptions for the per-model view.
		return []string{"xhigh", "medium", "low", "Off"}, "xhigh", true
	case "glm":
		return []string{"On", "Off"}, "On", true
	default:
		return nil, "", false
	}
}

// glmVersionRe captures the GLM major (and optional minor) version from a bare
// model name such as "glm-5.2" or "glm-4.7". It anchors on the "glm-" prefix so
// older/unversioned names (e.g. "glm-z1-32b", "chatglm-4") do not match.
var glmVersionRe = regexp.MustCompile(`^glm-(\d+)(?:\.(\d+))?`)

// IsGLM52OrLater reports whether model is a GLM model version 5.2 or later.
// GLM 5.2+ introduced the reasoning_effort parameter (values "max"/"high"),
// which is honored when thinking is enabled. The model argument may be a bare
// name ("glm-5.2") or a composite "provider/name" identifier.
func IsGLM52OrLater(model string) bool {
	bare := strings.ToLower(strings.TrimSpace(BareModel(model)))
	m := glmVersionRe.FindStringSubmatch(bare)
	if m == nil {
		return false
	}
	major, err := strconv.Atoi(m[1])
	if err != nil {
		return false
	}
	minor := 0
	if m[2] != "" {
		if minor, err = strconv.Atoi(m[2]); err != nil {
			minor = 0
		}
	}
	return major > 5 || (major == 5 && minor >= 2)
}

// qwenVersionRe captures the Qwen major (and optional minor) version from a
// bare model name such as "qwen3.8" or "qwen2.5". It anchors on the "qwen"
// prefix immediately followed by a digit so unversioned or differently-named
// Qwen-family models (e.g. "qwq-32b", "qwen3-coder-480b" — which carries no
// minor version and is not a 3.8 deployment) match only their actual number.
var qwenVersionRe = regexp.MustCompile(`^qwen(\d+)(?:\.(\d+))?`)

// IsQwen38OrLater reports whether model is a Qwen model version 3.8 or later.
// Qwen 3.8 introduced the per-request reasoning_effort parameter (values
// "xhigh"/"medium"/"low"); older Qwen models support only the binary
// enable_thinking switch and reject or ignore reasoning_effort. The model
// argument may be a bare name ("qwen3.8-27b") or a composite
// "provider/name" identifier ("qwen/qwen3.8-flash-next").
func IsQwen38OrLater(model string) bool {
	bare := strings.ToLower(strings.TrimSpace(BareModel(model)))
	m := qwenVersionRe.FindStringSubmatch(bare)
	if m == nil {
		return false
	}
	major, err := strconv.Atoi(m[1])
	if err != nil {
		return false
	}
	minor := 0
	if m[2] != "" {
		if minor, err = strconv.Atoi(m[2]); err != nil {
			minor = 0
		}
	}
	return major > 3 || (major == 3 && minor >= 8)
}

// ModelReasoningOptions is the model-aware counterpart of
// FamilyReasoningOptions. It returns the reasoning options for a specific model
// when the model version matters (e.g. GLM 5.2+), and falls back to the
// family-level options otherwise.
//
// GLM 5.2+ exposes three options:
//
//   - "none": thinking disabled
//   - "max":  thinking enabled with reasoning_effort=max (the GLM default)
//   - "high": thinking enabled with reasoning_effort=high
//
// GLM-5.3-Flash and later flash variants are always-thinking — the API
// offers no disable spelling — so they expose only the enable tiers
// "max"/"high". Older GLM models keep the family-level "On"/"Off" options.
//
// Qwen follows the same version split: Qwen 3.8+ exposes the native
// "xhigh"/"medium"/"low"/"Off" set, while pre-3.8 Qwen models (qwen3-*,
// qwen2.5-*, qwq-*, …) only understand the binary "On"/"Off" thinking switch —
// offering them the native efforts would silently no-op (or be rejected) on
// serving stacks that predate the parameter.
func ModelReasoningOptions(family, model string) (options []string, preferred string, ok bool) {
	if family == "glm" && IsGLM52OrLater(model) {
		if glmFlashAlwaysThinking(model) {
			return []string{"max", "high"}, "max", true
		}
		return []string{"none", "max", "high"}, "max", true
	}
	if family == "qwen" && !IsQwen38OrLater(model) {
		return []string{"On", "Off"}, "On", true
	}
	return FamilyReasoningOptions(family)
}

// glmFlashAlwaysThinking reports whether model is a GLM "flash" variant whose
// thinking mode is always on and cannot be disabled — GLM-5.3-Flash and later
// flash releases (see the registry entry). Such a model must not be offered a
// disable option: the API has no spelling for it.
func glmFlashAlwaysThinking(model string) bool {
	bare := strings.ToLower(strings.TrimSpace(BareModel(model)))
	m := glmVersionRe.FindStringSubmatch(bare)
	if m == nil {
		return false
	}
	major, err := strconv.Atoi(m[1])
	if err != nil {
		return false
	}
	minor := 0
	if m[2] != "" {
		if minor, err = strconv.Atoi(m[2]); err != nil {
			minor = 0
		}
	}
	if major < 5 || (major == 5 && minor < 3) {
		return false
	}
	return strings.Contains(bare, "flash")
}
