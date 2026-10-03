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
// GLM 5.2+ introduced the reasoning_effort parameter. GLM 5.2 offers
// max/high and permits disabling; GLM 5.3+ is always-thinking with low/high/max.
// The model argument may be a bare
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

// qwen38ArchitectureAliases lists models that ARE built on the Qwen 3.8
// architecture — and therefore speak its native per-request reasoning_effort
// protocol — but whose identifiers carry no "qwen<version>" token, so
// qwenVersionRe cannot read a version out of them. Keys are normalizeModelID
// forms, which makes the match insensitive to casing, to a vendor prefix, and
// to delivery postfixes: a single entry covers the bare serving name
// "Bonsai 2 27B", the composite selector "embedded/Bonsai 2 27B", and the
// checkpoint spelling "Ternary-Bonsai-2-27B-gguf".
//
// This table is the ONE funnel for such models. IsQwen38OrLater is consulted
// both by the user-facing option set (ModelReasoningOptions) and by the
// wire-level encoder (applyQwenReasoning), so registering an alias here fixes
// the picker and the request encoding together — teaching only one of the two
// would either offer native efforts the provider path then refuses to send, or
// send efforts the picker never offered. The catalog counterpart (Family
// "qwen" for the same identifiers) lives in makeBuiltInRegistry: the family
// gates ModelReasoningOptions' qwen branch, this table gates the version.
var qwen38ArchitectureAliases = map[string]struct{}{
	// Ternary-Bonsai-2-27B (PrismML): a Qwen3.8-architecture checkpoint
	// published and served under its own name. See the catalog entries
	// "prism-ml/ternary-bonsai-2-27b" and "bonsai 2 27b".
	normalizeModelID("prism-ml/ternary-bonsai-2-27b"): {},
	normalizeModelID("bonsai 2 27b"):                  {},
}

// IsQwen38OrLater reports whether model is a Qwen model version 3.8 or later.
// Qwen 3.8 introduced the per-request reasoning_effort parameter (values
// "xhigh"/"medium"/"low"); older Qwen models support only the binary
// enable_thinking switch and reject or ignore reasoning_effort. The model
// argument may be a bare name ("qwen3.8-27b") or a composite
// "provider/name" identifier ("qwen/qwen3.8-flash-next").
//
// A model whose name carries no readable version but which is built on the
// 3.8 architecture is recognized through qwen38ArchitectureAliases, which is
// consulted first and matches on the normalized identifier.
func IsQwen38OrLater(model string) bool {
	if _, ok := qwen38ArchitectureAliases[normalizeModelID(model)]; ok {
		return true
	}
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
// GLM 5.2 exposes three options:
//
//   - "none": thinking disabled
//   - "max":  thinking enabled with reasoning_effort=max (the GLM default)
//   - "high": thinking enabled with reasoning_effort=high
//
// GLM 5.3 and its Flash variant are always-thinking and expose only
// "max"/"high"/"low". GLM 5.2 retains "none"/"max"/"high"; older
// GLM models keep the family-level "On"/"Off" options.
//
// Qwen follows the same version split: Qwen 3.8+ exposes the native
// "xhigh"/"medium"/"low"/"Off" set, while pre-3.8 Qwen models (qwen3-*,
// qwen2.5-*, qwq-*, …) only understand the binary "On"/"Off" thinking switch —
// offering them the native efforts would silently no-op (or be rejected) on
// serving stacks that predate the parameter.
func ModelReasoningOptions(family, model string) (options []string, preferred string, ok bool) {
	if strings.HasPrefix(family, "openai_") {
		if options, preferred, ok := openAIModelReasoningOptions(model); ok {
			return options, preferred, true
		}
	}
	if family == "glm" && IsGLM52OrLater(model) {
		if glmAlwaysThinking(model) {
			return []string{"max", "high", "low"}, "max", true
		}
		return []string{"none", "max", "high"}, "max", true
	}
	if family == "qwen" && !IsQwen38OrLater(model) {
		return []string{"On", "Off"}, "On", true
	}
	return FamilyReasoningOptions(family)
}

// glmAlwaysThinking reports whether model uses the GLM 5.3+ reasoning
// contract. Both flagship and Flash variants require thinking to stay enabled.
// Source: https://docs.z.ai/guides/llm/glm-5.3 (Feature Changes).
func glmAlwaysThinking(model string) bool {
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
	return major > 5 || (major == 5 && minor >= 3)
}

// ReasoningTier is the family-agnostic reasoning level a caller wants for a
// single LLM call. It abstracts over the per-family native spellings so call
// sites can request a reasoning BUDGET ("as cheap as this model allows") or
// an OFF without knowing which family serves the request:
//
//   - ReasoningTierOff: reasoning disabled — maps to the model's disable
//     spelling ("Off" for binary models, "none" for GLM 5.2 and newer
//     non-Pro GPT models). When the option set carries no disable spelling
//     (GLM 5.3+, Kimi, Codex, Pro and other thinking-locked models), the tier
//     degrades to the minimal available effort — the closest the provider
//     can get.
//   - ReasoningTierMinimal: the cheapest non-disabled effort of the model
//     ("minimal" for original GPT-5, "low" for GLM 5.3+/Codex/Qwen 3.8+,
//     "High" for DeepSeek, "On" for binary models, …).
//
// The zero value and any unrecognized tier fail closed to "" (no field).
type ReasoningTier string

const (
	// ReasoningTierOff asks for reasoning to be disabled on the call,
	// degrading to the minimal effort where no disable spelling exists.
	ReasoningTierOff ReasoningTier = "off"
	// ReasoningTierMinimal asks for the cheapest non-disabled reasoning
	// effort the model's family offers.
	ReasoningTierMinimal ReasoningTier = "minimal"
)

// reasoningEffortRank orders the known native effort spellings from cheapest
// to strongest so minimalReasoningEffort can pick the floor of an arbitrary
// family's option set. The declared option lists are NOT consistently
// ordered (OpenAI/Google list weakest-first, Qwen strongest-first, GLM 5.2+
// carries the disable spelling first), so no positional rule can find the
// floor — the rank table can. Keys are lowercased spellings; callers must
// lowercase before lookup and keep the ORIGINAL spelling of the winning
// option (the wire value is case-sensitive per family — Google's "MINIMAL",
// DeepSeek's "High").
//
// "on" is the single non-disabled value of the binary families, so ranking it
// cheapest is harmless: it can only win when it is the only candidate.
// Unrecognized spellings rank strongest — never chosen over a known effort,
// but still returned when they are the only non-disabled option (they are,
// by definition, the minimal available one then).
var reasoningEffortRank = map[string]int{
	"minimal": 0,
	"on":      0,
	"low":     1,
	"medium":  2,
	"high":    3,
	"xhigh":   4,
	"max":     5,
}

// reasoningUnknownEffortRank ranks spellings missing from reasoningEffortRank.
const reasoningUnknownEffortRank = 1 << 30

// reasoningDisableSpelling returns the disable spelling of a family's option
// set — "Off" for the binary families, "none" for GLM 5.2+ — matched
// case-insensitively, or "" when thinking cannot be disabled. The returned
// spelling preserves the option set's native casing.
func reasoningDisableSpelling(options []string) string {
	for _, opt := range options {
		if strings.EqualFold(opt, "off") || strings.EqualFold(opt, "none") {
			return opt
		}
	}
	return ""
}

// minimalReasoningEffort returns the cheapest non-disabled effort of a
// family's option set (see reasoningEffortRank), preserving its native
// spelling, or "" when every option is a disable spelling.
func minimalReasoningEffort(options []string) string {
	best := ""
	bestRank := reasoningUnknownEffortRank + 1
	for _, opt := range options {
		if strings.EqualFold(opt, "off") || strings.EqualFold(opt, "none") {
			continue
		}
		rank, ok := reasoningEffortRank[strings.ToLower(opt)]
		if !ok {
			rank = reasoningUnknownEffortRank
		}
		if rank < bestRank {
			best, bestRank = opt, rank
		}
	}
	return best
}

// kimiMinimalReasoningEffort is the floor of the Kimi reasoning_effort range
// (low/high/max — see the kimi-k3 catalog entry). The kimi family declares no
// option set in FamilyReasoningOptions — its models are thinking-locked with
// heterogeneous, partly undocumented controls — but the effort-tuned K3 /
// K2.7-code models document this floor, so it is the minimal available
// effort a tier can target. Whether the field is actually emitted for a given
// Kimi model remains the wire layer's decision (see the provider family
// switch): only the K3 series documents the field, so the wire emits it for
// K3-series models and drops it (fail-closed) for every other kimi model,
// same as for every other spelling this helper returns.
const kimiMinimalReasoningEffort = "low"

// ReasoningForCall maps an abstract reasoning tier to the native
// reasoning_effort spelling to send for ONE call to the given model. It is
// the per-call counterpart of the option pickers (FamilyReasoningOptions /
// ModelReasoningOptions): those describe what a family CAN express, this
// resolves what to actually PUT ON THE WIRE for a desired budget. Version
// awareness rides on ModelReasoningOptions, so GLM 5.2+ (reasoning_effort
// with the "none" disable spelling), GLM 5.3+ (always thinking, floor low)
// and Qwen 3.8+ / qwen38ArchitectureAliases (native efforts;
// pre-3.8 models binary "On"/"Off") are handled without extra logic here.
//
// The empty-string contract: "" means "send no reasoning field at all" and is
// returned when (a) the tier is unrecognized, (b) the model AUTHORITATIVELY
// declares no reasoning capability (built-in catalog Capabilities.Reasoning ==
// false — e.g. kimi-k2; guessed capabilities of unknown models do NOT gate,
// mirroring Router.applyDefaultSampling), or (c) the family has no known
// reasoning control (mistral, unknown names). Callers merge "" by simply not
// setting ChatRequest.ReasoningEffort.
func ReasoningForCall(family, model string, tier ReasoningTier) string {
	switch tier {
	case ReasoningTierOff, ReasoningTierMinimal:
	default:
		return ""
	}
	// Capability gate — declared-only. ResolveBuiltInModel reports known=false
	// for models outside the built-in catalog, whose filled capabilities are
	// registry guesswork; trusting those would silently strip the reasoning
	// field from unknown/local models the host knows better.
	if meta, known := ResolveBuiltInModel(model); known && meta.Capabilities != nil && !meta.Capabilities.Reasoning {
		return ""
	}
	options, _, ok := ModelReasoningOptions(family, model)
	if !ok {
		// Families with no declared option set. Kimi is thinking-locked:
		// no disable spelling exists, so both tiers land on the documented
		// effort floor of the effort-tuned models. Every other
		// option-less family fails closed — no field.
		if family == "kimi" {
			return kimiMinimalReasoningEffort
		}
		return ""
	}
	if tier == ReasoningTierOff {
		if off := reasoningDisableSpelling(options); off != "" {
			return off
		}
		// No disable spelling (GLM-flash, Kimi-style thinking-locked
		// variants of families that do declare efforts, OpenAI, Google):
		// the closest possible is the cheapest effort — fall through.
	}
	return minimalReasoningEffort(options)
}
