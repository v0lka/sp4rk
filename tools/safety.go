package tools

import (
	"context"
	"encoding/json"
)

// JudgeSeverity classifies how hard the reason behind a judge escalation is.
// Both severities escalate identically today (user confirmation); the
// distinction exists so downstream policy layers can treat hard reasons as
// never-overridable while soft ones may be weighed by other evidence.
//
// JudgeSeverityHard is the zero value: an escalation that arrives without an
// explicit classification is treated as hard (fail-closed).
type JudgeSeverity int

const (
	// JudgeSeverityHard marks security-control triggers: blacklist pattern
	// matches and SSRF protection (private/reserved targets, degraded SSRF
	// checks), including fail-closed cases where the input could not be
	// assessed at all. These reasons must never be weakened or auto-overridden.
	JudgeSeverityHard JudgeSeverity = iota
	// JudgeSeveritySoft marks advisory escalations: path-containment and
	// locality concerns (a path outside session roots, an unresolvable
	// target). The operation itself may be legitimate — only its scope is in
	// question.
	JudgeSeveritySoft
)

// String returns a human-readable severity name ("hard"/"soft").
func (s JudgeSeverity) String() string {
	switch s {
	case JudgeSeveritySoft:
		return "soft"
	default:
		return "hard"
	}
}

// JudgeOutcome is the result of a tool-local safety judge: whether the call is
// allowed, the reason when it is not, and how severe that reason is.
// Allow=false with an empty Reason means "no tool-specific concern" — the
// registry proceeds without escalating.
type JudgeOutcome struct {
	Allow    bool
	Reason   string
	Severity JudgeSeverity
}

// ToolJudger is an optional interface that tools can implement to provide
// tool-specific safety heuristics. When a tool with PolicyAlwaysAllow implements
// this interface, the registry calls Judge before execution. If the judge returns
// Allow=false with non-empty Reason, the call is escalated to user confirmation.
type ToolJudger interface {
	Judge(ctx context.Context, input json.RawMessage) JudgeOutcome
}

// ConfirmationRequest describes a tool execution that needs user confirmation.
type ConfirmationRequest struct {
	ToolName       string          `json:"tool_name"`
	Input          json.RawMessage `json:"input"`
	JudgeReasoning string          `json:"judge_reasoning,omitempty"`
	// JudgeSeverity classifies the escalation so hosts can decide whether it
	// may be auto-resolved (soft: a scope question a strict judge may settle)
	// or must stay interactive (hard: a fired security control, never
	// auto-overridable). It is set for judge-escalated calls from the judge
	// outcome's Severity; plain PolicyUserConfirm gates escalate as hard (no
	// judge classified them). The zero value is hard — fail-closed.
	JudgeSeverity JudgeSeverity `json:"judge_severity"`
	// DisableJudge prevents a confirmation surfaced by the strict automatic
	// judge from being sent through the advisory on-demand judge a second time.
	// The zero value preserves the existing Ask Agent flow for ordinary gates.
	DisableJudge bool `json:"disable_judge,omitempty"`
}

// ConfirmationResponse represents the user's confirmation decision.
type ConfirmationResponse int

const (
	// ConfirmAllowOnce allows this single execution.
	ConfirmAllowOnce ConfirmationResponse = iota
	// ConfirmDeny denies this execution.
	ConfirmDeny
	// ConfirmDenyAndStop denies the execution and cancels the entire task.
	ConfirmDenyAndStop
)

// ConfirmFunc is called before executing a tool whose effective policy is
// PolicyUserConfirm. If nil, such calls are DENIED (fail-closed) — set one
// via ToolRegistry.SetConfirmFunc or use explicit PolicyAlwaysAllow overrides.
type ConfirmFunc func(ctx context.Context, req ConfirmationRequest) (ConfirmationResponse, error)
