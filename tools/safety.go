package tools

import (
	"context"
	"encoding/json"
	"fmt"
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
	// locality concerns — a path that was fully assessed and resolved outside
	// the session roots. The operation itself may be legitimate — only its
	// scope is in question. An input that could NOT be assessed at all is not
	// soft; it escalates as hard (see JudgeSeverityHard).
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

// MarshalJSON renders the severity as its String() name ("hard"/"soft"), the
// same vocabulary String() exposes, so serialized ConfirmationRequests (and
// JSON logs) stay legible and stable across enum reordering — a bare int
// would leak iota positions onto the wire. Out-of-range values marshal as
// "hard", mirroring String()'s fail-closed default.
func (s JudgeSeverity) MarshalJSON() ([]byte, error) {
	return json.Marshal(s.String())
}

// UnmarshalJSON parses a severity name ("hard"/"soft"). An unknown name is an
// error rather than a silent fallback to a numeric or zero value — callers
// must notice malformed input; where a value is absent, the type's zero value
// (hard) applies. JSON null is a no-op per the encoding/json convention: the
// receiver keeps its current value (hard for a fresh variable), so a null
// field behaves exactly like an omitted one — fail-closed. A bare int is
// rejected: the canonical wire form is the name.
func (s *JudgeSeverity) UnmarshalJSON(data []byte) error {
	if string(data) == "null" {
		return nil
	}
	var name string
	if err := json.Unmarshal(data, &name); err != nil {
		return fmt.Errorf("judge severity: %w", err)
	}
	switch name {
	case "hard":
		*s = JudgeSeverityHard
	case "soft":
		*s = JudgeSeveritySoft
	default:
		return fmt.Errorf("judge severity: unknown name %q", name)
	}
	return nil
}

// JudgeReasonCode is a stable, machine-checkable classification of the reason
// behind a judge escalation. Unlike Reason — human-readable prose that may be
// reworded freely — codes are a cross-repository contract: hosts downstream
// (for example c0wrk's canonical hard-reason backstop) key deterministic
// policy decisions off the code instead of matching prose. A published code
// must never be renamed or reused; add new codes instead. The empty value
// means "unclassified" — hosts decide unclassified outcomes by their own
// fail-closed policy, never by matching the prose.
type JudgeReasonCode string

const (
	// ReasonCodeCommandBlacklist marks a shell command that matched a
	// configured blacklist pattern — a fired security control.
	ReasonCodeCommandBlacklist JudgeReasonCode = "command_blacklist"
	// ReasonCodeUnresolvablePathToken marks a command containing path-like
	// tokens the resolver cannot assess ("~user", "${VAR:-/etc/passwd}").
	// Hard but scope-shaped: a strict judge may positively clear it.
	ReasonCodeUnresolvablePathToken JudgeReasonCode = "unresolvable_path_token"
	// ReasonCodeOutsideSessionRoots marks a fully assessed path (or shell
	// path reference) that resolved outside the session roots — an advisory
	// scope question (soft).
	ReasonCodeOutsideSessionRoots JudgeReasonCode = "outside_session_roots"
	// ReasonCodeSSRFPrivateAddress marks a fetch target that resolves to a
	// private/reserved address — an SSRF escape attempt (fired control).
	ReasonCodeSSRFPrivateAddress JudgeReasonCode = "ssrf_private_address"
	// ReasonCodeSSRFDegraded marks an unavailable SSRF check (the CIDR list
	// failed to initialize): the posture is unassessable, fail-closed.
	ReasonCodeSSRFDegraded JudgeReasonCode = "ssrf_protection_degraded"
	// ReasonCodeUnassessableURL marks an input whose target URL could not be
	// determined at all — unassessable, fail-closed.
	ReasonCodeUnassessableURL JudgeReasonCode = "unassessable_url"
	// ReasonCodeUnassessablePath marks an input whose target path could not
	// be determined at all — unassessable, fail-closed.
	ReasonCodeUnassessablePath JudgeReasonCode = "unassessable_path"
	// ReasonCodeSymlinkEscape marks an input that traverses symlinks
	// resolving outside the session roots. Set by hosts that run symlink
	// detection over tool input (e.g. c0wrk's registry gate), since the
	// sp4rk walker reports traversals rather than a JudgeOutcome.
	ReasonCodeSymlinkEscape JudgeReasonCode = "symlink_escape"
	// ReasonCodeSymlinkSuspicious marks symlink input that could not be
	// fully resolved (target unknown) without a confirmed escape. Set by
	// hosts alongside ReasonCodeSymlinkEscape; unassessable-shaped but
	// without a fired control, so hosts may let a strict judge clear it.
	ReasonCodeSymlinkSuspicious JudgeReasonCode = "symlink_suspicious"
	// ReasonCodeGitInternal marks a mutating operation whose target path
	// contains a ".git" path component at or below the workspace root — the
	// repository's object database, refs, config, and hooks (including nested
	// repos, submodules, and worktrees, where ".git" may be a gitdir-pointer
	// file rather than a directory). Writes there can rewrite history, forge
	// refs, or plant executable hooks, so this is a fired security control
	// (hard severity, mirroring ReasonCodeSymlinkEscape): the call escalates
	// to user confirmation and hosts must never auto-override it.
	ReasonCodeGitInternal JudgeReasonCode = "git_internal_path"
)

// JudgeOutcome is the result of a tool-local safety judge: whether the call is
// allowed, the reason when it is not, and how severe that reason is.
// Allow=false with an empty Reason means "no tool-specific concern" — the
// registry proceeds without escalating.
type JudgeOutcome struct {
	Allow    bool
	Reason   string
	Severity JudgeSeverity
	// ReasonCode is the typed classification of Reason (see JudgeReasonCode).
	// It is the stable contract consumers key off; Reason remains the
	// human-readable prose. The zero value means unclassified.
	ReasonCode JudgeReasonCode
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
	// JudgeReasonCode is the typed classification of the escalation (see
	// JudgeReasonCode): the machine-checkable contract paired with the
	// JudgeReasoning prose. It is set for judge-escalated calls from the judge
	// outcome's ReasonCode; plain PolicyUserConfirm gates escalate with no
	// code (no judge classified them). The zero value means unclassified —
	// hosts decide unclassified escalations by their own fail-closed policy.
	JudgeReasonCode JudgeReasonCode `json:"judge_reason_code,omitempty"`
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
