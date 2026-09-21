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
// fail-closed policy, never by matching the prose. See [IsCanonicalReasonCode]
// for the subset of codes a host must never auto-override.
type JudgeReasonCode string

const (
	// ReasonCodeCommandBlacklist marks a shell command that matched a
	// configured blacklist pattern — a fired security control.
	ReasonCodeCommandBlacklist JudgeReasonCode = "command_blacklist"
	// ReasonCodeUnresolvablePathToken is retained for contract stability but is
	// no longer fired: it classified a command containing path-like tokens the
	// resolver cannot assess ("~user", "${VAR:-/etc/passwd}") as hard but
	// scope-shaped. The static shell-path checks were removed — dynamic
	// constructs are the deterministic flowsh analysis's domain — and an
	// unresolvable construct now surfaces through C6
	// (command_unbounded_analysis). Published codes are never renamed or
	// reused; a host that still maps it treats it as an unassessable-shaped
	// hard-but-non-canonical reason (clearable by a strict judge).
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
	// ReasonCodeSymlinkSuspicious is retained for contract stability but is
	// no longer fired: it classified symlink input that could not be fully
	// resolved (unresolved shell expansions) without a confirmed escape.
	// The expansion-suspicion checks were removed from the symlink walk —
	// dynamic constructs are the deterministic flowsh analysis's domain —
	// so no built-in path sets this code anymore. Published codes are never
	// renamed or reused; a host that still maps it treats it as
	// unassessable-shaped without a fired control (clearable by a strict
	// judge).
	ReasonCodeSymlinkSuspicious JudgeReasonCode = "symlink_suspicious"
	// ReasonCodeGitInternal marks a mutating operation whose target path
	// contains a ".git" path component (case-insensitively) at or below the
	// workspace root or an additional allowed root — the repository's object
	// database, refs, config, and hooks (including nested repos, submodules,
	// and worktrees, where ".git" may be a gitdir-pointer file rather than a
	// directory, and roots that themselves end inside a ".git" tree). Writes there can rewrite history, forge
	// refs, or plant executable hooks, so this is a fired security control
	// (hard severity, mirroring ReasonCodeSymlinkEscape): the call escalates
	// to user confirmation and hosts must never auto-override it.
	ReasonCodeGitInternal JudgeReasonCode = "git_internal_path"
	// ReasonCodeCommandExfilFlow marks a shell command whose flowsh analysis
	// found a credential-exfiltration pairing — a secret read (CredAccess or
	// FSRead of a secret-bearing path) reaching a tainted network egress
	// (criterion C1). Fired control, hard and canonical: never weaken it.
	ReasonCodeCommandExfilFlow JudgeReasonCode = "command_exfil_flow"
	// ReasonCodeCommandPrivilegeEscalation marks a shell command whose flowsh
	// analysis found a privilege-escalation effect — sudo, setuid installs,
	// and their kin (criterion C2). Fired control, hard and canonical.
	ReasonCodeCommandPrivilegeEscalation JudgeReasonCode = "command_privilege_escalation"
	// ReasonCodeCommandSystemWrite marks a shell command whose flowsh analysis
	// found a filesystem write/metadata effect landing on a system path
	// (/etc, /usr, /boot, /bin, /sbin, Windows System32 / Program Files) or a
	// non-harmless raw device (criterion C3). Fired control, hard and
	// canonical.
	ReasonCodeCommandSystemWrite JudgeReasonCode = "command_system_write"
	// ReasonCodeCommandDestructiveOutsideRoots marks a shell command combining
	// a destructive knowledge-base flag of class D/E, irreversibility, and a
	// concrete filesystem-write target outside the session roots (criterion
	// C4). Fired control, hard and canonical.
	ReasonCodeCommandDestructiveOutsideRoots JudgeReasonCode = "command_destructive_outside_roots"
	// ReasonCodeCommandDownloadCradle marks a shell command whose flowsh
	// analysis established a network→code-execution FLOW — fetched network
	// content reaching a shell/interpreter (a pipe to sh, a sourced or
	// process-substituted fetch, a command-substitution sink): the
	// download-cradle shape (criterion C5). Fired control, hard and canonical —
	// keyed on the proven flow, so a canonical verdict is always backed by the
	// flow it ships in the digest.
	ReasonCodeCommandDownloadCradle JudgeReasonCode = "command_download_cradle"
	// ReasonCodeCommandUnboundedAnalysis marks a shell command the analyser
	// could not bound (⊤/conservative) without a cradle flow, or an
	// irreversible write whose target it could not resolve (⊤ target, e.g.
	// abbreviated PowerShell parameters) — criterion C6. Hard but
	// NON-canonical: it is an analysis limitation, not a confirmed control — an
	// advisory judge may clear it on closer reading.
	ReasonCodeCommandUnboundedAnalysis JudgeReasonCode = "command_unbounded_analysis"
	// ReasonCodeCommandExternalContentIngest marks a shell command whose flowsh
	// analysis established a network→filesystem FLOW — a download client
	// (curl -o/-O, wget -O/default) wrote content it fetched over the network to
	// a file (criterion C7). Hard but NON-canonical: persistent external-content
	// ingest is a real attack shape (staging a payload for later use, or
	// writing out data), but it is also routine benign behaviour (fetching a
	// document or an archive), so the advisory judge may clear it on closer
	// reading. A fetch to stdout is not an ingest and never fires it.
	ReasonCodeCommandExternalContentIngest JudgeReasonCode = "command_external_content_ingest"
	// ReasonCodeCredentialAccess marks a shell command whose flowsh analysis
	// found credential/secret material accessed without a paired egress
	// (criterion C8). Advisory scope concern, soft.
	ReasonCodeCredentialAccess JudgeReasonCode = "credential_access"
	// ReasonCodeCommandExecOutsideRoots marks a shell command whose flowsh
	// analysis produced a DIRECT code-execution or process-spawn effect whose
	// path-shaped target resolves outside every session root — a driver
	// (test runner, interpreter, build tool) pointed at code that lives
	// outside the trusted roots (criterion C10). It is the exec sibling of C4
	// (destructive write outside the roots) and C9 (FS* outside the roots):
	// the analysis bounds WHAT runs, the roots decide WHERE it may point.
	// Hard but NON-canonical: executing out-of-root code is a judgment shape,
	// not a confirmed control — the file may be a scratch script the session
	// itself wrote to the host temp dir — so the advisory judge may clear it
	// on closer reading (and the workspace-scoped verification marker stays
	// off for it by its own containment condition). It only ever fires on a
	// BOUNDED report: an unbounded call is C6's territory.
	ReasonCodeCommandExecOutsideRoots JudgeReasonCode = "command_exec_outside_roots"
	// ReasonCodeCommandAnalysisUnavailable marks a shell command whose
	// deterministic analysis could not be produced at all — the flowsh
	// analyzer or its embedded knowledge base failed to initialise. The
	// deterministic floor (criteria C1–C10) is unavailable for the call, so it
	// fails CLOSED: a fired control-like reason, hard and canonical, never
	// auto-overridable, so the call still escalates under an `allow` policy
	// and blocks under verify-on-edit's unattended path.
	ReasonCodeCommandAnalysisUnavailable JudgeReasonCode = "command_analysis_unavailable"
)

// canonicalReasonCodes is the set of published codes whose fired reason a host
// must never auto-override — the deterministic backstop a host consults before
// letting an advisory or strict judge waive an escalation. It covers two
// classes:
//
//   - a hard fired security control: a shell blocklist match, the flowsh
//     shell-analysis controls the digest marks canonical (exfiltration flow,
//     privilege escalation, a system-path/raw-device write, an irreversible
//     destructive write outside the session roots, and a download cradle),
//     SSRF protection, a symlink escape out of the session roots, and a write
//     into git internals;
//   - an input whose safety the judge is structurally unable to assess:
//     degraded SSRF protection, an undeterminable URL/path, and a
//     deterministic shell analysis that could not run at all.
//
// The two hard-but-clearable shell-analysis codes are deliberately absent:
// command_unbounded_analysis (the analyzer's ⊤ limitation) and
// command_external_content_ingest (a download-client ingest flow) are hard but
// non-canonical, so a strict judge may positively clear them. The soft scope
// codes are likewise not canonical. This mirrors the set hosts such as c0wrk
// (its "canonical hard reason" backstop) key deterministic policy off.
var canonicalReasonCodes = map[JudgeReasonCode]bool{
	ReasonCodeCommandBlacklist:               true,
	ReasonCodeCommandExfilFlow:               true,
	ReasonCodeCommandPrivilegeEscalation:     true,
	ReasonCodeCommandSystemWrite:             true,
	ReasonCodeCommandDestructiveOutsideRoots: true,
	ReasonCodeCommandDownloadCradle:          true,
	ReasonCodeCommandAnalysisUnavailable:     true,
	ReasonCodeSSRFPrivateAddress:             true,
	ReasonCodeSSRFDegraded:                   true,
	ReasonCodeUnassessableURL:                true,
	ReasonCodeUnassessablePath:               true,
	ReasonCodeSymlinkEscape:                  true,
	ReasonCodeGitInternal:                    true,
}

// IsCanonicalReasonCode reports whether a JudgeReasonCode is canonical — a hard
// fired security control (or an unassessable input) that a host must NEVER
// auto-override, even when an advisory or strict judge returns allow. The
// remaining hard codes (command_unbounded_analysis, command_external_content_ingest)
// and the soft scope codes are not canonical: a strict judge may positively
// clear them. An empty or unknown code reports false.
//
// It is a convenience over the per-criterion canonicality the shell-analysis
// digest carries (ShellCriterion.Canonical and ShellAnalysis.Canonical): a host
// that keys deterministic policy off ConfirmationRequest.JudgeReasonCode alone
// can consult this instead of hard-coding the canonical set. Published codes
// are a cross-repository contract, so this set is part of it — extend it when a
// new canonical code is added, never by renaming an existing one.
func IsCanonicalReasonCode(code JudgeReasonCode) bool {
	return canonicalReasonCodes[code]
}

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
