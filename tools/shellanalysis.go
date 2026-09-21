// SPDX-License-Identifier: Apache-2.0
//
// shellanalysis.go — the deterministic shell-command analysis engine layered
// on top of github.com/v0lka/flowsh. flowsh parses a bash/PowerShell command
// into a frozen effect IR (filesystem/network/process/credential effects, an
// exfiltration-pairing taint analysis, and a destructive-flags knowledge
// base); this file turns that report into the fixed-priority criteria C1–C9
// that shell-exec judges (and the advisory on-demand judge) consume.
//
// Design contract:
//
//   - The blocklist match stays in the tool's own Judge (bash_exec/posh_exec);
//     it is NOT re-implemented here. This engine covers everything the
//     effect IR can see.
//   - Criteria are evaluated in the fixed priority C1…C9; every fired
//     criterion is recorded in the digest, and the highest-priority one
//     becomes the JudgeOutcome.
//   - Canonicality: C1–C5 are hard AND canonical (hosts must never
//     auto-override them). C5 fires only on an established cradle FLOW —
//     the network→code-execution flow the analysis proved (fetched content
//     reaching a shell/interpreter) — never a bare NetEgress/CodeExec
//     co-occurrence, so a canonical verdict is always backed by a real flow.
//     C6 (unbounded analysis) and C7 (external-content ingest) are hard but
//     NON-canonical — an analysis limitation and a flow the advisory judge
//     may clear. C8/C9 are soft scope questions.
//   - The flowsh score/grade are carried in the digest for context but are
//     deliberately NOT used as decision thresholds — the criteria fire off
//     structural facts (effect kinds, targets, KB classes), not scores.
//   - Containment (C4/C9) consults only FS* effects whose concrete targets
//     are path-shaped; CLI noise ("-30", "s/foo/bar/g", "+x") is discarded by
//     a path-shape check before resolution. Empty session roots disable
//     containment entirely, as the former shell-path containment check did.

package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"path"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"

	"github.com/v0lka/flowsh/api"
	"github.com/v0lka/flowsh/engine"
)

// ShellDigestSchemaVersion tags the shell-analysis digest contract
// ("sp4rk-shell-analysis/v3"). Bump it whenever the digest gains, loses or
// reshapes a field. v2 added the WorkspaceScopedVerification marker; v3 adds
// the network data-flow keys (cradleFlows/ingestFlows) and the flow-based
// criteria (C5 download-cradle flow, C7 external-content ingest).
const ShellDigestSchemaVersion = "sp4rk-shell-analysis/v3"

// ─────────────────────────────────────────────────────────────────────────
// Process-global analyzer
// ─────────────────────────────────────────────────────────────────────────

// shellAnalyzerOnce lazily initialises the process-global flowsh analyzer.
// api.Analyzer holds the loaded command knowledge base in memory and is safe
// for concurrent use, so one instance is shared by every analysis call.
var (
	shellAnalyzerOnce sync.Once
	shellAnalyzer     *api.Analyzer
	shellAnalyzerErr  error
)

// shellFlowshAnalyzer returns the process-global analyzer, loading the
// embedded knowledge base on first use. The error is sticky: a failed load is
// reported for every subsequent call rather than retried (fail-closed).
func shellFlowshAnalyzer() (*api.Analyzer, error) {
	shellAnalyzerOnce.Do(func() {
		shellAnalyzer, shellAnalyzerErr = api.NewAnalyzer()
	})
	return shellAnalyzer, shellAnalyzerErr
}

// shellToolLangs maps a tool name onto the flowsh dialect its command string
// is written in. Only the shell-exec tools are analysable; anything else is a
// caller bug and fails loudly.
var shellToolLangs = map[string]api.Lang{
	"bash_exec": api.LangBash,
	"posh_exec": api.LangPowerShell,
}

// ─────────────────────────────────────────────────────────────────────────
// Digest types (the compact JSON contract for judges)
// ─────────────────────────────────────────────────────────────────────────

// ShellEffectDigest is one flowsh effect, trimmed to what a judge needs: no
// taint detail, no why-trace, no input echo.
type ShellEffectDigest struct {
	Kind       string   `json:"kind"`
	Mode       string   `json:"mode"`
	Certainty  string   `json:"certainty"`
	Targets    []string `json:"targets"`
	Arbitrary  bool     `json:"arbitrary"`
	Reversible bool     `json:"reversible"`
}

// ShellScoreDigest is the flowsh composite score with the embedded exfil
// pairs stripped (they are surfaced separately as source/sink keys).
type ShellScoreDigest struct {
	Destructiveness string `json:"destructiveness"`
	Irreversibility string `json:"irreversibility"`
	Breadth         string `json:"breadth"`
	Influence       string `json:"influence"`
	Exfil           string `json:"exfil"`
	Confidence      int    `json:"confidence"`
	Reversible      bool   `json:"reversible"`
	Grade           string `json:"grade"`
}

// ShellExfilPairDigest identifies one credential-exfiltration pairing by the
// stable keys of its source and sink effects (Effect.Key() form,
// "Kind|Mode|[targets]").
type ShellExfilPairDigest struct {
	Source string `json:"source"`
	Sink   string `json:"sink"`
}

// ShellFlowPairDigest identifies one established network data-flow — a cradle
// (network→code-execution) or an ingest (network→filesystem) — by the stable
// keys of its source and sink effects (Effect.Key() form,
// "Kind|Mode|[targets]"). It is the flow evidence behind criteria C5/C7, and
// surfaces the flows themselves so a judge can see what the analyzer proved
// rather than a bare NetEgress/CodeExec co-occurrence.
type ShellFlowPairDigest struct {
	Source string `json:"source"`
	Sink   string `json:"sink"`
}

// ShellDestructiveDigest is one matched destructive-flags KB entry, without
// the reason prose.
type ShellDestructiveDigest struct {
	Command string `json:"command"`
	Spec    string `json:"spec"`
	Class   string `json:"class"`
}

// ShellResolutionDigest names the command the analysis actually resolved to.
type ShellResolutionDigest struct {
	Invoked string `json:"invoked,omitempty"`
	Kind    string `json:"kind"`
	Name    string `json:"name,omitempty"`
}

// ShellCriterion is one fired criterion: its stable reason code, severity and
// canonicality. Canonical marks reasons a host must never auto-override.
type ShellCriterion struct {
	Fired     JudgeReasonCode `json:"fired"`
	Severity  JudgeSeverity   `json:"severity"`
	Canonical bool            `json:"canonical"`
}

// ShellAnalysisDigest is the compact, stable JSON document handed to judges:
// the bounded facts of the flowsh report plus the fired criteria — no
// why-traces and no echo of the raw input command.
type ShellAnalysisDigest struct {
	SchemaVersion string                 `json:"schemaVersion"`
	Lang          string                 `json:"lang"`
	Top           bool                   `json:"top"`
	Conservative  bool                   `json:"conservative"`
	Reason        string                 `json:"reason,omitempty"`
	Commands      int                    `json:"commands"`
	Resolution    ShellResolutionDigest  `json:"resolution"`
	Effects       []ShellEffectDigest    `json:"effects"`
	Score         ShellScoreDigest       `json:"score"`
	ExfilPairs    []ShellExfilPairDigest `json:"exfilPairs"`
	// CradleFlows lists the established network→code-execution flows (the
	// download-cradle shape behind criterion C5): fetched content the analysis
	// proved reaches a shell/interpreter. Empty when no such flow exists.
	CradleFlows []ShellFlowPairDigest `json:"cradleFlows"`
	// IngestFlows lists the established network→filesystem flows (behind
	// criterion C7): a download client wrote content it fetched over the
	// network to a file (curl -o/-O, wget default/-O). A stdout fetch is not
	// an ingest and yields none. Empty when no such flow exists.
	IngestFlows []ShellFlowPairDigest    `json:"ingestFlows"`
	Destructive []ShellDestructiveDigest `json:"destructive"`
	Criteria    []ShellCriterion         `json:"criteria"`
	// WorkspaceScopedVerification is the deterministic workspace-scoped
	// verification marker (v2): every resolved binary in the command is a
	// catalogued verification driver or benign plumbing utility, at least one
	// is a driver, every file operand and write redirection resolves inside
	// the session roots, environment prefixes come from a safe set, there is
	// no network effect, no dependency-manifest write and no unresolved
	// expansion. It is positive EVIDENCE for the judges — it never suppresses
	// a fired criterion (C6 keeps escalating so the judge stays in the loop;
	// see the judge prompt's Static Analysis Report rules).
	WorkspaceScopedVerification bool `json:"workspaceScopedVerification"`
	// Signature is the deterministic effect signature of the analysed
	// command: the catalogued verification drivers it invokes, its canonical
	// effect set, the codes of the fired criteria and the verification
	// marker, rendered as one stable string (see [shellEffectSignature] for
	// the exact components and format). Two commands that differ in form but
	// not in effect — a blocked call retried through an equivalent spelling —
	// carry the same signature, so a host can recognize a re-escalation of an
	// already-adjudicated effect instead of re-trying it from scratch. The
	// signature is EVIDENCE for memoizing verdicts; it never suppresses a
	// criterion and never overrides canonicality.
	Signature string `json:"signature"`
}

// ShellAnalysis is the full result of the deterministic shell analysis: the
// judge-facing digest, the winning criterion as a JudgeOutcome (Allow=true
// when nothing fired), and the canonicality of that winning reason.
type ShellAnalysis struct {
	// Digest is the compact JSON document for judges (see ShellAnalysisDigest).
	Digest ShellAnalysisDigest
	// Outcome is the highest-priority fired criterion expressed as a
	// JudgeOutcome. Allow=true (zero value otherwise) when no criterion fired.
	Outcome JudgeOutcome
	// Canonical reports whether the winning reason is canonical — a hard
	// security control a host must never auto-override. False for soft
	// reasons and for the non-canonical hard reasons
	// ReasonCodeCommandUnboundedAnalysis and
	// ReasonCodeCommandExternalContentIngest.
	Canonical bool
}

// ─────────────────────────────────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────────────────────────────────

// AnalyzeShellCommandForJudge runs the flowsh-based deterministic analysis of
// one shell-exec tool input and maps it onto the C1–C9 criteria. It is the
// advisory-path helper: callers feed it the tool name ("bash_exec" or
// "posh_exec", which selects the dialect) and the tool's raw JSON input
// ({command, working_directory}); the returned ShellAnalysis carries the
// digest plus the winning judge outcome. Session roots for the containment
// criteria (C4/C9) come from ctx exactly as they did for the former shell-path
// containment check; with no roots attached those criteria cannot fire.
// Host-known variable bindings attached via [WithShellVarBindings] are
// forwarded to the analyzer (flowsh Options.Vars): every binding behaves as
// though the script had assigned it a literal value before its first
// statement, so a $name read resolves to the concrete value instead of
// degrading its word — and every path derived from it — to ⊤. An in-script
// assignment overrides the seeded binding; an empty table (or none) analyses
// exactly as before.
//
// A tool name outside the shell-exec pair, an unparsable input, or a failed
// knowledge-base load is an error — callers fail closed on it.
//
// For hosts whose shell tool carries an operator-configured invocation
// override, use [ShellAnalysisLangForTool] (or [ShellKindToAnalysisLang]) to
// resolve the dialect from the DECLARED shell kind and call
// [AnalyzeShellCommandForJudgeWithDialect] instead — the tool name no longer
// implies the syntax when the launch wrapper is user-configured.
func AnalyzeShellCommandForJudge(ctx context.Context, toolName string, input json.RawMessage) (*ShellAnalysis, error) {
	lang, ok := shellToolLangs[toolName]
	if !ok {
		return nil, fmt.Errorf("shell analysis: unsupported tool %q (want bash_exec or posh_exec)", toolName)
	}
	return AnalyzeShellCommandForJudgeWithDialect(ctx, lang, input)
}

// AnalyzeShellCommandForJudgeWithDialect is [AnalyzeShellCommandForJudge] with
// an explicit flowsh dialect. Hosts whose shell-exec tool accepts an
// operator-configured invocation override derive the dialect from the
// DECLARED shell kind (ShellKindToAnalysisLang) rather than the tool
// name, because the tool name no longer implies the command syntax. An
// unsupported lang is an error — callers fail closed on it.
func AnalyzeShellCommandForJudgeWithDialect(ctx context.Context, lang api.Lang, input json.RawMessage) (*ShellAnalysis, error) {
	if lang != api.LangBash && lang != api.LangPowerShell {
		return nil, fmt.Errorf("shell analysis: unsupported dialect %q (want bash or powershell)", string(lang))
	}
	var params struct {
		Command          string `json:"command"`
		WorkingDirectory string `json:"working_directory"`
	}
	if err := json.Unmarshal(input, &params); err != nil {
		return nil, fmt.Errorf("shell analysis: parse shell tool input: %w", err)
	}
	analyzer, err := shellFlowshAnalyzer()
	if err != nil {
		return nil, fmt.Errorf("shell analysis: flowsh analyzer init: %w", err)
	}
	report := analyzer.AnalyzeWith(lang, params.Command, api.Options{Vars: ShellVarBindingsFrom(ctx)})
	return evaluateShellReport(ctx, params.WorkingDirectory, params.Command, lang, report), nil
}

// DeclaredShellKinder is implemented by shell-exec tools that carry a
// (default or operator-overridden) declared shell kind. Both built-in shell
// tools implement it; hosts use it to resolve the analysis dialect of a
// registered tool via [ShellAnalysisLangForTool].
type DeclaredShellKinder interface {
	DeclaredShellKind() ShellKind
}

// ShellAnalysisLangForTool resolves the flowsh analysis dialect for a
// registered shell-exec tool instance: the dialect of its DECLARED shell kind
// when the tool implements [DeclaredShellKinder], else the legacy tool-name
// mapping (bash_exec → bash, posh_exec → powershell). ok is false for tools
// that are not shell-exec tools (callers must not attach an analysis for
// them). The tool is typed `any` on purpose — resolution only needs
// DeclaredShellKind, and hosts hold the registered instance behind varying
// interfaces.
func ShellAnalysisLangForTool(tool any, toolName string) (api.Lang, bool) {
	if k, ok := tool.(DeclaredShellKinder); ok {
		if lang, ok2 := ShellKindToAnalysisLang(k.DeclaredShellKind()); ok2 {
			return lang, true
		}
	}
	lang, ok := shellToolLangs[toolName]
	return lang, ok
}

// ─────────────────────────────────────────────────────────────────────────
// Context attachment (host-precomputed analysis for the shell Judges)
// ─────────────────────────────────────────────────────────────────────────

// shellAnalysisAttachment is the value stored under shellAnalysisKey: exactly
// one of analysis/err is non-nil (the return shape of
// [AnalyzeShellCommandForJudge]).
type shellAnalysisAttachment struct {
	analysis *ShellAnalysis
	err      error
}

// shellAnalysisKey is the context key carrying the host-precomputed shell
// analysis for the imminent bash_exec/posh_exec call.
type shellAnalysisKey struct{}

// WithShellAnalysis attaches a host-precomputed [ShellAnalysis] — or the
// error that prevented computing one — to ctx for the imminent
// bash_exec/posh_exec tool call. Hosts run [AnalyzeShellCommandForJudge]
// once per call and attach the result; the deterministic shell Judges read
// it through [ShellJudgeOutcome] without re-running the engine. analysis is
// non-nil iff err is nil. The attachment is per-call: a host must attach the
// analysis computed for the very input it is about to submit to the Judge.
func WithShellAnalysis(ctx context.Context, analysis *ShellAnalysis, err error) context.Context {
	return context.WithValue(ctx, shellAnalysisKey{}, shellAnalysisAttachment{analysis: analysis, err: err})
}

// ShellAnalysisFrom extracts the host-attached shell analysis, if any. Both
// return values are nil when nothing is attached.
func ShellAnalysisFrom(ctx context.Context) (*ShellAnalysis, error) {
	if v, ok := ctx.Value(shellAnalysisKey{}).(shellAnalysisAttachment); ok {
		return v.analysis, v.err
	}
	return nil, nil
}

// shellVarBindingsKey is the context key carrying the host-known shell
// variable bindings for the imminent bash_exec/posh_exec analysis.
type shellVarBindingsKey struct{}

// WithShellVarBindings attaches the variable bindings the HOST knows from its
// own session context — a session temp directory, a workspace path, a run
// identifier — to ctx for the imminent bash_exec/posh_exec analysis.
// [AnalyzeShellCommandForJudge] forwards them into the analyzer (flowsh
// Options.Vars), where each binding behaves exactly as though the script had
// assigned it a literal value before its first statement: a later $name read
// resolves to the concrete value instead of degrading its word (and every
// path derived from it) to ⊤. The bindings model host-known intent, not a
// persistent shell: they resolve the cross-command expansions the script text
// alone does not determine. The map is copied defensively; nil and empty maps
// are equivalent to no attachment (analysis unchanged). The attachment is
// per-call, read by the analysis entry point only — it does not reach the
// executed process environment.
func WithShellVarBindings(ctx context.Context, vars map[string]string) context.Context {
	if len(vars) == 0 {
		return ctx
	}
	cp := make(map[string]string, len(vars))
	for k, v := range vars {
		cp[k] = v
	}
	return context.WithValue(ctx, shellVarBindingsKey{}, cp)
}

// ShellVarBindingsFrom extracts the host-attached variable bindings, if any.
// Returns nil when nothing is attached. Callers must treat the returned map
// as read-only.
func ShellVarBindingsFrom(ctx context.Context) map[string]string {
	if v, ok := ctx.Value(shellVarBindingsKey{}).(map[string]string); ok {
		return v
	}
	return nil
}

// ShellJudgeOutcome is the deterministic outcome the bash_exec/posh_exec
// Judges report for an analysis attached via [WithShellAnalysis]: the
// analysis's winning [ShellAnalysis.Outcome] verbatim (Allow=true when no
// criterion fired). When nothing is attached it returns the empty outcome,
// deferring the call to the advisory judges. When the attachment carries an
// error — the analyzer or its embedded knowledge base failed to initialise,
// so the deterministic floor (C1–C9) is unavailable for this call — it FAILS
// CLOSED: a hard canonical [ReasonCodeCommandAnalysisUnavailable] outcome, so
// the call still escalates under an `allow` policy and blocks under
// verify-on-edit's unattended path, instead of running with no floor at all.
// The error is logged to [slog.Default] as well.
//
// IMPORTANT for hosts upgrading from a version whose built-in shell tools ran
// their own static containment/unresolvable-token checks: those checks moved
// into the flowsh analysis (criteria C4/C6/C9), and the SDK never attaches the
// analysis on its own. A host that does not call [AnalyzeShellCommandForJudge]
// and [WithShellAnalysis] therefore gets NO deterministic shell escalation for
// a shell call — only the advisory LLM judge (ToolJudge.Judge) remains — so
// wiring the analysis is required to preserve the previous deterministic floor.
func ShellJudgeOutcome(ctx context.Context, toolName string) JudgeOutcome {
	analysis, err := ShellAnalysisFrom(ctx)
	if err != nil {
		slog.Default().Warn("shell judge: pre-computed analysis failed; failing closed",
			"tool", toolName, "error", err)
		return JudgeOutcome{
			Allow:      false,
			Reason:     "deterministic shell analysis is unavailable: " + err.Error(),
			Severity:   JudgeSeverityHard,
			ReasonCode: ReasonCodeCommandAnalysisUnavailable,
		}
	}
	if analysis == nil {
		return JudgeOutcome{}
	}
	return analysis.Outcome
}

// ─────────────────────────────────────────────────────────────────────────
// Criteria engine
// ─────────────────────────────────────────────────────────────────────────

// evaluateShellReport applies the fixed-priority criteria C1–C9 to a flowsh
// report and assembles the digest + winning outcome. It never fails: a report
// is always assessable (flowsh itself degrades to ⊤/conservative, which C6/C7
// handle). command is the raw tool-input command text — used only to verify
// environment-prefix values for the workspace-scoped verification marker; it
// never reaches the digest (the no-input-echo contract). lang is the dialect
// the command was analyzed in: the system-path classifier is dialect-aware,
// because a forward-slash "/Windows/…" is a system path under PowerShell but
// an ordinary path under bash.
func evaluateShellReport(ctx context.Context, workDir, command string, lang api.Lang, report *api.Report) *ShellAnalysis {
	var criteria []ShellCriterion
	fire := func(code JudgeReasonCode, severity JudgeSeverity, canonical bool) {
		criteria = append(criteria, ShellCriterion{Fired: code, Severity: severity, Canonical: canonical})
	}

	hasPrivEsc := shellHasEffectKind(report, engine.KindPrivEsc)
	hasCredAccess := shellHasEffectKind(report, engine.KindCredAccess)
	hasExfilPair := len(report.Score.ExfilPairs) > 0
	unbounded := report.Top || report.Conservative
	unboundedWrite := shellHasUnboundedWrite(report)
	// The network data-flows the analyzer established: a cradle is a proven
	// network→code-execution flow (fetched content reaching a shell or
	// interpreter), an ingest a proven network→filesystem flow (a download
	// client writing fetched content to a file). C5 keys on the cradle FLOW —
	// not on a NetEgress/CodeExec co-occurrence — so a canonical cradle verdict
	// is always backed by the evidence it ships.
	hasCradleFlow := len(report.Score.CradleFlows) > 0
	hasIngestFlow := len(report.Score.IngestFlows) > 0

	// C1 — exfiltration flow: a secret read paired with tainted egress.
	if hasExfilPair {
		fire(ReasonCodeCommandExfilFlow, JudgeSeverityHard, true)
	}
	// C2 — privilege escalation effect.
	if hasPrivEsc {
		fire(ReasonCodeCommandPrivilegeEscalation, JudgeSeverityHard, true)
	}
	// C3 — direct write/metadata effect on a system path or raw device.
	if len(shellSystemOrDeviceTargets(report, lang)) > 0 {
		fire(ReasonCodeCommandSystemWrite, JudgeSeverityHard, true)
	}
	// C4 — irreversible destructive command (KB class D/E) writing outside
	// the session roots.
	if shellHasDestructiveClassDE(report) && !report.Score.Reversible {
		if len(shellOutsideRoots(ctx, report, workDir, shellFSWriteKinds, false)) > 0 {
			fire(ReasonCodeCommandDestructiveOutsideRoots, JudgeSeverityHard, true)
		}
	}
	// C5 — a download cradle: the analysis established a network→code-execution
	// flow (fetched content reaching a shell/interpreter). Keyed on the FLOW,
	// not on a NetEgress/CodeExec co-occurrence: a canonical cradle verdict is
	// therefore always backed by the flow it ships in the digest, whether or
	// not the egress target itself resolved to a literal host. Hard canonical.
	if hasCradleFlow {
		fire(ReasonCodeCommandDownloadCradle, JudgeSeverityHard, true)
	}
	// C6 — the analyzer could not bound the command (⊤/conservative) without a
	// cradle flow, OR an irreversible write's target could not be resolved
	// (⊤): hard but NON-canonical, an analysis limitation the advisory judge
	// may clear. The unboundedWrite disjunct is deliberately independent of
	// egress: an unresolved irreversible write is destructive wherever it
	// lands, so it fires even when the command also carries network egress.
	// Keying the unbounded disjunct on !hasCradleFlow (rather than on a
	// resolved egress target) keeps a genuinely unbounded egress escalating —
	// an unresolved download, or a dialect whose pipe the analysis could not
	// follow into a cradle — instead of passing silently; C5 already owns the
	// established-cradle shape.
	if unboundedWrite || (unbounded && !hasCradleFlow) {
		fire(ReasonCodeCommandUnboundedAnalysis, JudgeSeverityHard, false)
	}
	// C7 — persistent external-content ingest: the analysis established a
	// network→filesystem flow (a download client wrote content it fetched over
	// the network to a file). A fetch to stdout is not an ingest. Hard but
	// NON-canonical: the download may be a legitimate document or archive, so
	// the advisory judge may clear it on closer reading.
	if hasIngestFlow {
		fire(ReasonCodeCommandExternalContentIngest, JudgeSeverityHard, false)
	}
	// C8 — credential access without an exfil pairing.
	if hasCredAccess && !hasExfilPair {
		fire(ReasonCodeCredentialAccess, JudgeSeveritySoft, false)
	}
	// C9 — direct FS* effect outside the session roots on a non-system
	// path: the scope question, reusing the outside_session_roots code.
	if len(shellOutsideRootDirectNonSystemTargets(ctx, report, workDir, lang)) > 0 {
		fire(ReasonCodeOutsideSessionRoots, JudgeSeveritySoft, false)
	}

	result := &ShellAnalysis{Digest: newShellAnalysisDigest(ctx, workDir, command, report, criteria)}
	if len(criteria) > 0 {
		winner := criteria[0]
		result.Outcome = JudgeOutcome{
			Allow:      false,
			Reason:     shellCriterionReason(winner.Fired),
			Severity:   winner.Severity,
			ReasonCode: winner.Fired,
		}
		result.Canonical = winner.Canonical
	} else {
		// No criterion fired: allow. Note JudgeOutcome's zero value has
		// Allow=false, so the explicit true matters here.
		result.Outcome = JudgeOutcome{Allow: true}
	}
	return result
}

// shellCriterionReason returns the human-readable prose for a fired
// criterion. Prose may be reworded freely; the code is the contract.
func shellCriterionReason(code JudgeReasonCode) string {
	switch code {
	case ReasonCodeCommandExfilFlow:
		return "Shell analysis: secret/credential read paired with tainted network egress (exfiltration flow)"
	case ReasonCodeCommandPrivilegeEscalation:
		return "Shell analysis: privilege escalation effect detected"
	case ReasonCodeCommandSystemWrite:
		return "Shell analysis: filesystem write/metadata effect on a system path or raw device"
	case ReasonCodeCommandDestructiveOutsideRoots:
		return "Shell analysis: irreversible destructive command writing outside the session roots"
	case ReasonCodeCommandDownloadCradle:
		return "Shell analysis: downloaded network content reaches code execution (download cradle)"
	case ReasonCodeCommandUnboundedAnalysis:
		return "Shell analysis: command could not be bounded by static analysis — top/conservative without a download-cradle flow, or an irreversible write whose target could not be resolved"
	case ReasonCodeCommandExternalContentIngest:
		return "Shell analysis: download client wrote fetched external content to a file (external-content ingest)"
	case ReasonCodeCredentialAccess:
		return "Shell analysis: credential/secret material accessed without a paired egress"
	case ReasonCodeOutsideSessionRoots:
		return "Shell analysis: filesystem effect outside the session roots"
	default:
		return "Shell analysis: criterion " + string(code) + " fired"
	}
}

// shellHasEffectKind reports whether any effect has the given kind.
func shellHasEffectKind(report *api.Report, kind engine.EffectKind) bool {
	for _, e := range report.Effects {
		if e.Kind == kind {
			return true
		}
	}
	return false
}

// shellHasDestructiveClassDE reports whether any matched destructive-flags
// entry carries class D or E (hard/critical).
func shellHasDestructiveClassDE(report *api.Report) bool {
	for _, d := range report.Destructive {
		if d.Class == "D" || d.Class == "E" {
			return true
		}
	}
	return false
}

// shellHasUnboundedWrite reports whether the report contains a directly
// performed, irreversible filesystem write whose target the analyzer could
// not resolve (⊤). Such a write is an analysis limitation with destructive
// potential — the target may be anywhere, so no containment criterion can
// reason about it. PowerShell parameter abbreviations are a known producer of
// this shape: flowsh does not expand "-r"/"-f" to "-Recurse"/"-Force", loses
// the positional target, and emits an arbitrary FSWrite ("Remove-Item -r -f
// C:\Windows\System32" would otherwise pass silently). Folding it into C6
// restores a fired reason for the abbreviated destructive-delete vocabulary
// without shipping a dialect-specific pattern list.
func shellHasUnboundedWrite(report *api.Report) bool {
	for _, e := range report.Effects {
		if e.Kind == engine.KindFSWrite && e.Mode == engine.ModeDirect && e.Target.IsTop() && !e.Reversible {
			return true
		}
	}
	return false
}

// shellFSWriteKinds / shellFSAllKinds / shellFSDirectKinds name the effect-kind
// sets the containment criteria consult.
var (
	shellFSWriteKinds = []engine.EffectKind{engine.KindFSWrite}
	shellFSAllKinds   = []engine.EffectKind{engine.KindFSRead, engine.KindFSWrite, engine.KindFSMeta}
)

// shellEffectKindsHas reports whether kinds contains e.Kind.
func shellEffectKindsHas(kinds []engine.EffectKind, e engine.Effect) bool {
	for _, k := range kinds {
		if e.Kind == k {
			return true
		}
	}
	return false
}

// shellSystemOrDeviceTargets returns the C3 targets: write/metadata effects
// performed directly whose concrete, path-shaped targets land on a system
// path or a non-harmless raw device. Relative targets are skipped (an empty
// resolution base leaves them unanchored; system paths are absolute by
// nature).
func shellSystemOrDeviceTargets(report *api.Report, lang api.Lang) []string {
	var out []string
	for _, e := range report.Effects {
		if !shellEffectKindsHas([]engine.EffectKind{engine.KindFSWrite, engine.KindFSMeta}, e) || e.Mode != engine.ModeDirect || e.Target.IsTop() {
			continue
		}
		for _, t := range e.Target.Targets() {
			if !shellPathLikeTarget(t) {
				continue
			}
			if abs, ok := shellResolveTarget(t, ""); ok && shellIsSystemOrRawDevicePath(abs, lang) {
				out = append(out, abs)
			}
		}
	}
	return out
}

// shellOutsideTarget is one out-of-root containment hit: the resolved
// absolute path plus the effect kind that produced it. C9 needs the kind to
// tell C3-owned writes/metadata (system paths) apart from reads, which no
// higher-priority criterion covers.
type shellOutsideTarget struct {
	path string
	kind engine.EffectKind
}

// shellOutsideRoots resolves the path-shaped concrete targets of matching
// effects against the session roots and returns those that fall outside
// every root, each tagged with its effect kind. Matching effects must carry
// one of kinds; when directOnly is set they must additionally be performed
// directly (Mode == Direct), as the C9 scope criterion requires. With no
// session roots attached — or no resolvable base for relative targets — it
// returns nil, as the former shell-path containment check did.
func shellOutsideRoots(ctx context.Context, report *api.Report, workDir string, kinds []engine.EffectKind, directOnly bool) []shellOutsideTarget {
	roots := SessionRoots(ctx)
	if len(roots) == 0 {
		return nil
	}
	base := workDir
	if base == "" {
		base = WorkspacePathFrom(ctx)
	}
	var outside []shellOutsideTarget
	for _, e := range report.Effects {
		if !shellEffectKindsHas(kinds, e) || e.Target.IsTop() {
			continue
		}
		if directOnly && e.Mode != engine.ModeDirect {
			continue
		}
		for _, t := range e.Target.Targets() {
			if !shellPathLikeTarget(t) {
				continue
			}
			abs, ok := shellResolveTarget(t, base)
			if !ok || shellIsHarmlessDevicePath(abs) {
				continue
			}
			inside := false
			for _, root := range roots {
				if IsWithinRoot(ctx, root, abs) {
					inside = true
					break
				}
			}
			if !inside {
				outside = append(outside, shellOutsideTarget{path: abs, kind: e.Kind})
			}
		}
	}
	return outside
}

// shellOutsideRootDirectNonSystemTargets returns the C9 targets: directly
// performed FS* effects whose path-shaped, resolvable targets fall outside
// every session root. Direct writes/metadata on a system path or raw device
// are excluded — C3 owns those, at a higher priority and as a hard canonical
// control. Reads of system *files* are deliberately NOT excluded: no
// higher-priority criterion covers them, and an out-of-root read of a system
// (or credential) file is exactly the scope question C9 exists to raise —
// leaving it silent would reopen the gap the former shell-path containment
// check used to close.
// Raw-device reads (/dev/urandom, /dev/zero, …) ARE excluded — they are
// routine inputs, not a scope concern, and a raw device is "system" only as
// a write target. Harmless devices (/dev/null, /dev/full) are exempt via
// [shellIsHarmlessDevicePath] inside the containment walk.
func shellOutsideRootDirectNonSystemTargets(ctx context.Context, report *api.Report, workDir string, lang api.Lang) []string {
	var out []string
	for _, t := range shellOutsideRoots(ctx, report, workDir, shellFSAllKinds, true) {
		if t.kind == engine.KindFSWrite || t.kind == engine.KindFSMeta {
			// C3 owns writes/metadata on system paths and raw devices.
			if shellIsSystemOrRawDevicePath(t.path, lang) {
				continue
			}
		} else if shellIsRawDeviceTarget(t.path) {
			// A raw-device read is a routine input source, not a scope question.
			continue
		}
		out = append(out, t.path)
	}
	return out
}

// shellIsRawDeviceTarget reports whether an absolute path names a raw device —
// the POSIX /dev tree — rather than a system file. Reads from it (/dev/urandom,
// /dev/zero, /dev/random) are routine; [shellIsHarmlessDevicePath] already
// exempts the bit buckets before this is consulted.
func shellIsRawDeviceTarget(absPath string) bool {
	if absPath == "" {
		return false
	}
	posix := path.Clean(filepath.ToSlash(absPath))
	return posix == "/dev" || strings.HasPrefix(posix, "/dev/")
}

// ─────────────────────────────────────────────────────────────────────────
// Path-shape and system-path classification
// ─────────────────────────────────────────────────────────────────────────

// shellPathLikeTarget reports whether an effect target string is shaped like a
// filesystem path rather than CLI operand noise. Rejected shapes: flags
// ("-30"), mode strings ("+x"), the "-" stdin/stdout marker, and sed
// substitution scripts ("s/foo/bar/g"). Accepted shapes: absolute POSIX
// paths, Windows drive/UNC paths, and any other relative reference — a bare
// relative name resolves under the working directory, so treating it as a
// path is the conservative choice (in-root resolutions stay benign).
func shellPathLikeTarget(target string) bool {
	if target == "" || target == "-" {
		return false
	}
	if strings.HasPrefix(target, "-") || strings.HasPrefix(target, "+") {
		return false
	}
	if shellIsSedScript(target) {
		return false
	}
	return true
}

// shellIsSedScript reports whether t has the shape of a sed substitution
// script, "s/…/…[/flags]" — the canonical non-path operand that still
// contains path separators and would otherwise survive a separator check.
func shellIsSedScript(t string) bool {
	if !strings.HasPrefix(t, "s/") {
		return false
	}
	// After the leading "s/", a further "/" separates the pattern from the
	// replacement ("s/foo/bar/g"); a bare "s/foo" has none.
	return strings.Contains(t[2:], "/")
}

// shellResolveTarget turns a path-shaped effect target into an absolute path.
// Absolute POSIX and Windows forms are cleaned as-is — with the slash (not
// filepath) cleaner for POSIX forms, so "/root" stays "/root" on a Windows
// host too (filepath.Clean would rewrite it to "\root"); relative targets
// are anchored to base (the working directory). Unanchored relative targets
// with an empty base are unresolvable (ok=false) and skipped by containment.
func shellResolveTarget(target, base string) (string, bool) {
	switch {
	case shellIsWindowsAbsPath(target):
		return filepath.Clean(target), true
	case strings.HasPrefix(target, "/"):
		return path.Clean(target), true
	case filepath.IsAbs(target):
		return filepath.Clean(target), true
	case base != "":
		return filepath.Join(base, target), true
	default:
		return "", false
	}
}

// isASCIILetter reports whether b is an ASCII letter: the drive run of a
// Windows drive-letter path is a single letter, case-insensitively.
func isASCIILetter(b byte) bool {
	return 'a' <= b && b <= 'z' || 'A' <= b && b <= 'Z'
}

// shellIsWindowsAbsPath reports whether s is an absolute Windows path: a
// drive-letter form ("C:\x", "C:/x") or a UNC path ("\\server\share").
func shellIsWindowsAbsPath(s string) bool {
	if strings.HasPrefix(s, `\\`) {
		return true
	}
	return len(s) >= 3 && isASCIILetter(s[0]) && s[1] == ':' && (s[2] == '\\' || s[2] == '/')
}

// posixSystemPathPrefixes lists the POSIX locations whose mutation is a
// fired security control: system configuration, system binaries, boot and
// the raw-device tree.
var posixSystemPathPrefixes = []string{"/etc", "/usr", "/boot", "/bin", "/sbin", "/dev"}

// windowsSystemPathPrefixes lists the lower-cased, back-slashed Windows
// locations with the same meaning: the OS installation directory and the
// program-files trees (including the 32-bit "Program Files (x86)" sibling,
// which is a distinct directory, not a child of "Program Files"). Drive-less
// forms cover targets written relative to the current drive
// ("\Windows\System32"). Prefixes are matched at a component boundary (see
// [shellIsSystemOrRawDevicePath]), so "c:\windows" covers the whole tree
// beneath it without matching "c:\windowsx".
var windowsSystemPathPrefixes = []string{
	`c:\windows`, `\windows`,
	`c:\program files`, `\program files`,
	`c:\program files (x86)`, `\program files (x86)`,
}

// shellIsHarmlessDevicePath reports whether an absolute shell-analysis target
// is a harmless bit-bucket device (/dev/null, /dev/full; Windows NUL), evaluated
// independently of the host OS.
//
// Unlike the host-gated [IsHarmlessDevicePath] — correct for filesystem tools,
// which act on the real host filesystem, where "/dev/null" on a Windows host is
// an ordinary out-of-root path — this is the shell-analysis view: it classifies
// command semantics, not host files. A POSIX-rooted target stays "/dev/null" on
// every host ([shellResolveTarget] keeps it forward-slashed and absolute), and
// in the dialect that produced it the path IS the bit bucket, so the criteria
// must exempt it regardless of where the analyzer itself runs — otherwise the
// same command reaches a different verdict on a Windows CI host than on POSIX.
// Windows-shaped NUL targets keep the host-gated treatment by delegating to
// [IsHarmlessDevicePath] (the posh dialect only ever executes on Windows, so
// its device names match exactly where the tool runs).
func shellIsHarmlessDevicePath(absPath string) bool {
	if absPath == "" {
		return false
	}
	if harmlessPOSIXDevices[path.Clean(filepath.ToSlash(absPath))] {
		return true
	}
	return IsHarmlessDevicePath(absPath)
}

// shellIsSystemOrRawDevicePath reports whether an absolute target is a
// system path (POSIX prefixes, Windows System32 / Program Files) or a
// non-harmless raw device under /dev. Harmless bit-bucket devices
// (/dev/null, /dev/full, and Windows NUL via [shellIsHarmlessDevicePath])
// are exempt so routine redirections never fire.
//
// lang is the dialect the command was analyzed in. It makes the Windows
// branch dialect-aware: a forward-slash "/windows/x" is a POSIX path under
// bash (where "/" is the only separator, so it must NOT be re-read as a
// drive-less Windows path) but a system path under PowerShell (where "/" is
// also a separator, so "Set-Content /Windows/System32/…" is a real write into
// the OS tree). Only the bash family takes the early return; the PowerShell
// family falls through to the Windows table.
func shellIsSystemOrRawDevicePath(absPath string, lang api.Lang) bool {
	if absPath == "" {
		return false
	}
	// POSIX side: slash-normalised prefix check.
	posix := path.Clean(filepath.ToSlash(absPath))
	for _, prefix := range posixSystemPathPrefixes {
		if posix == prefix || strings.HasPrefix(posix, prefix+"/") {
			// /dev is a system tree, but harmless devices are exempt.
			if prefix == "/dev" && shellIsHarmlessDevicePath(absPath) {
				continue
			}
			return true
		}
	}
	// Windows side: case-folded, back-slashed prefix check at a component
	// boundary (verbatim long-path prefixes are stripped first) — the same
	// guard the POSIX branch applies, so a prefix-adjacent non-system name
	// ("c:\program filesold") is not misclassified. Skipped for a bash
	// POSIX-rooted target: under bash a POSIX "/windows/x" is an ordinary
	// path and must not be re-read as a Windows system path by the drive-less
	// prefix table. The PowerShell family keeps the table, because there "/"
	// is a valid separator and "/Windows/System32" is the OS tree.
	if lang == api.LangBash && strings.HasPrefix(absPath, "/") {
		return false
	}
	win := strings.ToLower(filepath.ToSlash(absPath))
	win = strings.ReplaceAll(win, "/", `\`)
	win = strings.TrimPrefix(win, `\\?\`)
	for _, prefix := range windowsSystemPathPrefixes {
		if win == prefix || strings.HasPrefix(win, prefix+`\`) {
			return true
		}
	}
	return false
}

// ─────────────────────────────────────────────────────────────────────────
// Digest assembly
// ─────────────────────────────────────────────────────────────────────────

// newShellAnalysisDigest projects a flowsh report plus the fired criteria
// onto the compact digest: bounded facts only, no input echo, no why-traces.
// The workspace-scoped verification marker is computed here (it shares the
// criteria engine's session roots and resolution base) and stamped into the
// v3 digest.
func newShellAnalysisDigest(ctx context.Context, workDir, command string, report *api.Report, criteria []ShellCriterion) ShellAnalysisDigest {
	effects := make([]ShellEffectDigest, 0, len(report.Effects))
	for _, e := range report.Effects {
		targets := e.Target.Targets()
		if targets == nil {
			targets = []string{}
		}
		effects = append(effects, ShellEffectDigest{
			Kind:       string(e.Kind),
			Mode:       string(e.Mode),
			Certainty:  e.Certainty.String(),
			Targets:    targets,
			Arbitrary:  e.Target.IsTop(),
			Reversible: e.Reversible,
		})
	}
	pairs := make([]ShellExfilPairDigest, 0, len(report.Score.ExfilPairs))
	for _, p := range report.Score.ExfilPairs {
		pairs = append(pairs, ShellExfilPairDigest{Source: p.Source.Key(), Sink: p.Sink.Key()})
	}
	cradles := make([]ShellFlowPairDigest, 0, len(report.Score.CradleFlows))
	for _, f := range report.Score.CradleFlows {
		cradles = append(cradles, ShellFlowPairDigest{Source: f.Source.Key(), Sink: f.Sink.Key()})
	}
	ingests := make([]ShellFlowPairDigest, 0, len(report.Score.IngestFlows))
	for _, f := range report.Score.IngestFlows {
		ingests = append(ingests, ShellFlowPairDigest{Source: f.Source.Key(), Sink: f.Sink.Key()})
	}
	destructive := make([]ShellDestructiveDigest, 0, len(report.Destructive))
	for _, d := range report.Destructive {
		destructive = append(destructive, ShellDestructiveDigest{Command: d.Command, Spec: d.Spec, Class: d.Class})
	}
	if criteria == nil {
		criteria = []ShellCriterion{}
	}
	marker := shellWorkspaceScopedVerification(ctx, workDir, command, report)
	return ShellAnalysisDigest{
		SchemaVersion: ShellDigestSchemaVersion,
		Lang:          report.Lang,
		Top:           report.Top,
		Conservative:  report.Conservative,
		Reason:        report.Reason,
		Commands:      report.Commands,
		Resolution: ShellResolutionDigest{
			Invoked: report.Resolution.Invoked,
			Kind:    string(report.Resolution.Kind),
			Name:    report.Resolution.Name,
		},
		Effects: effects,
		Score: ShellScoreDigest{
			Destructiveness: report.Score.Destructiveness.String(),
			Irreversibility: report.Score.Irreversibility.String(),
			Breadth:         report.Score.Breadth.String(),
			Influence:       report.Score.Influence.String(),
			Exfil:           report.Score.Exfil.String(),
			Confidence:      report.Score.Confidence,
			Reversible:      report.Score.Reversible,
			Grade:           report.Score.Grade.String(),
		},
		ExfilPairs:                  pairs,
		CradleFlows:                 cradles,
		IngestFlows:                 ingests,
		Destructive:                 destructive,
		Criteria:                    criteria,
		WorkspaceScopedVerification: marker,
		Signature:                   shellEffectSignature(report, criteria, marker),
	}
}

// ─────────────────────────────────────────────────────────────────────────
// Effect signature (digest v3)
// ─────────────────────────────────────────────────────────────────────────
//
// The signature is the comparable identity of a command's EFFECT, not of its
// text (silent-mode recommendations §3, Track D): verdict memoization keyed on
// it must survive a retry through an equivalent spelling while still
// separating genuinely different commands. It is a union of four components:
//
//   - the catalogued verification DRIVERS the command invokes (binary plus
//     subcommand for subcommand-scoped drivers). Drivers are the one binary
//     family whose canonical form collapses to the same target-less effect
//     regardless of identity (a ⊤ CodeExec / bare ProcSpawn), so their names
//     are the only signal separating e.g. tsc from vitest. Conversely, the
//     transport and inspection utilities that legitimately differ between
//     equivalent retry forms (mv folding a staged write, wc versus grep on a
//     trailing check) are deliberately NOT part of the binary component —
//     their contribution is already carried, concretely, by the canonical
//     effects. This is what keeps the audited retry pairs identical:
//     963134/963140 (npx tsc versus ./node_modules/.bin/tsc) and
//     968120/968126 (sed > staging && mv versus sed -i).
//   - the CANONICAL effect set (flowsh v2 Canonical): each effect rendered as
//     its stable key "Kind|Mode|[targets]" plus reversibility ("|R"/"|I"),
//     staging-folded and stripped of non-path operand targets.
//   - the codes of the fired criteria (C1–C9), sorted.
//   - the workspace-scoped verification marker (B).
//
// Every component is sorted and deduplicated, so the rendering is a pure
// function of the analysis: identical input yields the identical signature.
// "sig1" is the format tag; bump it when the component set changes.

// shellEffectSignature renders the deterministic effect signature for one
// flowsh report plus its fired criteria and verification marker.
func shellEffectSignature(report *api.Report, criteria []ShellCriterion, marker bool) string {
	var bins []string
	seenBins := make(map[string]bool, len(report.CommandCalls))
	for _, call := range report.CommandCalls {
		allowedSubs, driver := shellVerificationDrivers[call.Resolved]
		if !driver || call.Resolved == "" {
			continue
		}
		token := call.Resolved
		// Subcommand-scoped drivers (go test/build/get, npm test/run, …)
		// share one binary name with materially different behaviours, so the
		// first non-flag operand disambiguates them the same way the marker
		// catalog's allowlists do.
		if allowedSubs != nil {
			if sub, ok := shellFirstFlagFreeOperand(call.Args); ok {
				token += ":" + sub
			}
		}
		if seenBins[token] {
			continue
		}
		seenBins[token] = true
		bins = append(bins, token)
	}
	sort.Strings(bins)

	var fx []string
	seenFx := make(map[string]bool)
	if report.Canonical != nil {
		for _, e := range report.Canonical.Effects {
			k := e.Key()
			if e.Reversible {
				k += "|R"
			} else {
				k += "|I"
			}
			if seenFx[k] {
				continue
			}
			seenFx[k] = true
			fx = append(fx, k)
		}
	}
	sort.Strings(fx)

	seenCodes := make(map[string]bool, len(criteria))
	var codes []string
	for _, c := range criteria {
		code := string(c.Fired)
		if seenCodes[code] {
			continue
		}
		seenCodes[code] = true
		codes = append(codes, code)
	}
	sort.Strings(codes)

	return strings.Join([]string{
		"sig1",
		"bins=" + strings.Join(bins, ","),
		"fx=" + strings.Join(fx, ";"),
		"crit=" + strings.Join(codes, ","),
		fmt.Sprintf("B=%t", marker),
	}, "|")
}

// shellFirstFlagFreeOperand returns the first non-empty, non-flag operand of
// args — the subcommand position for subcommand-scoped drivers.
func shellFirstFlagFreeOperand(args []string) (string, bool) {
	for _, a := range args {
		if a == "" || strings.HasPrefix(a, "-") {
			continue
		}
		return a, true
	}
	return "", false
}

// ─────────────────────────────────────────────────────────────────────────
// Workspace-scoped verification marker (digest v3, Track B)
// ─────────────────────────────────────────────────────────────────────────
//
// The marker is the deterministic evidence the strict judge needs for its
// "positive establishment" doctrine on the C6 (command_unbounded_analysis)
// escalation: a verification driver run over the session's own roots is the
// one command family that is unbounded ONLY because the analyzer cannot see
// inside the driver (vitest/gofmt/… are unknown binaries whose sole report
// contribution is a ⊤ CodeExec effect), never because it does anything the
// criteria could point at. The marker is computed from the same flowsh
// report the criteria use; it NEVER suppresses a criterion — C6 keeps firing
// (non-canonical, hard) so the call still escalates to the judge, which then
// clears it on the marker (defense in depth; the judge stays in the loop).
//
// Conditions (ALL must hold — any miss keeps the marker off, fail-closed):
//
//   - session roots are attached (an empty root set establishes nothing);
//   - the report is not ⊤ and every command statement is accounted for:
//     len(CommandCalls) == Commands. Sinks (bash -c, python3, node, docker,
//     awk, …), code-executing builtins (eval/source), shell-only builtins
//     (true/:/exit), user functions and dynamically-named commands never
//     reach the binder, so any hidden statement breaks the count — those are
//     exactly the forms whose effects the analyzer cannot see at all;
//   - every resolved binary is catalogued: at least one VERIFICATION DRIVER
//     (go test/vet/build/fmt…, gofmt, golangci-lint, tsc, vitest, eslint,
//     jest, rg, npm test/run) plus any number of benign PLUMBING utilities
//     (cd/echo/tail/grep/…), each passing its argument screens (find may not
//     -exec/-delete, rg may not --pre, go rejects -mod=mod and non-verify
//     subcommands, …). npx and node_modules/.bin wrappers normalize to the
//     driven binary (flowsh's CommandCall.Resolved);
//   - every path-shaped file operand — effect targets, call arguments and
//     write-redirection targets — resolves inside a session root (the raw
//     device tree is exempt via the harmless-device check);
//   - every environment assignment (EnvWrite) names a safe variable and
//     carries a safe value from the command text (CI=1, NO_COLOR,
//     GOFLAGS=-mod=readonly, GOPROXY=off, GOWORK=off, TERM=dumb);
//   - no network effect of any kind (NetEgress/NetIngress) — which also
//     excludes `go get`-style fetch subcommands via their KB effects;
//   - no write (effect or redirection) names a dependency-manifest file
//     (go.mod/go.work/package.json/…);
//   - no unresolved expansion survives: a ⊤ target is tolerated ONLY on a
//     CodeExec effect — the unknown-driver signature. Any other ⊤ effect
//     (a $VAR the binding layer could not resolve) keeps the marker off.
//
// The marker encodes the operator-trust premise "the session roots are
// trusted" (the same premise as workspace auto-approval); see ADR-052.

// shellVerificationDrivers catalogues the verification-driver binaries. The
// map value is the allowed first-operand (subcommand) allowlist; nil means
// every subcommand is accepted (the binary's own semantics are the trust
// premise). Resolution is by the NORMALIZED binary (flowsh CommandCall.
// Resolved): basename with node_modules/.bin stripped and package runners
// (npx) consumed, so npx tsc, ./node_modules/.bin/tsc and /usr/bin/tsc all
// resolve to tsc. Extend this catalog (and the plumbing set below) when a
// new driver earns its place — never widen an existing entry's screens.
var shellVerificationDrivers = map[string][]string{
	"go":            {"test", "vet", "build", "fmt", "list", "env", "version", "doc"},
	"gofmt":         nil,
	"gofumpt":       nil,
	"golangci-lint": {"run", "fmt", "version"},
	"tsc":           nil,
	"vitest":        nil,
	"jest":          nil,
	"eslint":        nil,
	"prettier":      nil,
	"rg":            nil,
	"ripgrep":       nil,
	"npm":           {"test", "run"},
}

// shellVerificationPlumbing catalogues the benign utilities that may appear
// alongside a driver without breaking the marker: directory navigation,
// output shaping and read-only inspection. None can execute code by itself
// (the code executors — bash/python/node/awk/… — are flowsh sinks and fail
// the call-coverage condition long before this table is consulted); the
// argument screens below close their remaining edges (find -exec, …).
var shellVerificationPlumbing = map[string]struct{}{
	"cd": {}, "pwd": {}, "echo": {}, "printf": {}, "ls": {}, "cat": {},
	"head": {}, "tail": {}, "grep": {}, "egrep": {}, "fgrep": {}, "sed": {},
	"find": {}, "wc": {}, "file": {}, "which": {}, "jq": {}, "sort": {},
	"uniq": {}, "cut": {}, "tr": {}, "column": {}, "basename": {},
	"dirname": {}, "realpath": {}, "date": {}, "sleep": {}, "true": {},
	"false": {}, "test": {}, "[": {},
}

// shellDriverForbiddenFlags maps a resolved binary onto argument tokens that
// break its driver/plumbing status outright. Screened as exact tokens or
// --long= prefixes ("--pre" covers "--pre cmd" and "--pre=cmd").
var shellDriverForbiddenFlags = map[string][]string{
	"rg":      {"--pre", "--pre-exec"}, // --pre executes a preprocessor command
	"ripgrep": {"--pre", "--pre-exec"},
	// find's action flags execute commands or delete; a verification find is
	// a pure search (-name/-type/…).
	"find": {"-exec", "-execdir", "-ok", "-okdir", "-delete", "-fprint", "-fprintf", "-fls"},
	// go's -mod=mod rewrites go.mod/go.sum during the run (a manifest write).
	"go": {"-mod=mod"},
}

// shellVerificationSafeEnv is the safe environment table: variable name →
// allowed values. An EnvWrite to any other name keeps the marker off; an
// allowed name must carry one of these values in EVERY textual
// NAME=VALUE occurrence of the command (the value must be statically
// verifiable — a dynamically built value fails the lookup, fail-closed).
var shellVerificationSafeEnv = map[string]map[string]bool{
	"CI":       {"1": true, "true": true, "false": true, "": true},
	"NO_COLOR": {"1": true, "true": true, "": true},
	"TERM":     {"dumb": true},
	"GOFLAGS":  {"-mod=readonly": true, "-mod=vendor": true},
	"GOPROXY":  {"off": true},
	"GOWORK":   {"off": true},
}

// shellDependencyManifests lists dependency/module-manifest basenames whose
// write (effect target or redirection) breaks the marker: rewriting the
// module graph is a supply-chain control, not verification plumbing.
var shellDependencyManifests = map[string]struct{}{
	"go.mod": {}, "go.sum": {}, "go.work": {}, "go.work.sum": {},
	"package.json": {}, "package-lock.json": {}, "npm-shrinkwrap.json": {},
	"yarn.lock": {}, "pnpm-lock.yaml": {}, "bun.lockb": {},
	"Cargo.toml": {}, "Cargo.lock": {},
	"requirements.txt": {}, "pyproject.toml": {}, "Pipfile": {}, "Pipfile.lock": {}, "poetry.lock": {},
	"composer.json": {}, "composer.lock": {}, "Gemfile": {}, "Gemfile.lock": {},
}

// shellEnvAssignRe matches a NAME=VALUE assignment word in the raw command
// text (prefix assignments, export statements, bare assignments). Values may
// be quoted; surrounding double quotes are trimmed before the safe-value
// lookup.
var shellEnvAssignRe = regexp.MustCompile(`(?:^|[^A-Za-z0-9_])([A-Za-z_][A-Za-z0-9_]*)=([^\s;&|()<>]+)`)

// shellWorkspaceScopedVerification computes the workspace-scoped verification
// marker for one report (see the section comment for the full conditions).
func shellWorkspaceScopedVerification(ctx context.Context, workDir, command string, report *api.Report) bool {
	if report == nil || report.Top || len(report.CommandCalls) == 0 {
		return false
	}
	roots := SessionRoots(ctx)
	if len(roots) == 0 {
		return false
	}
	// Every command statement must be a visible, catalogued call. Sinks,
	// eval-family builtins, shell-only builtins, functions and dynamic names
	// never produce a binder call, so a count mismatch means something ran
	// that the report cannot account for.
	if len(report.CommandCalls) != report.Commands {
		return false
	}
	if !shellMarkerEffectsSafe(ctx, workDir, report) {
		return false
	}
	return shellMarkerCallsSafe(ctx, workDir, report) &&
		shellMarkerEnvSafe(command, report)
}

// shellMarkerEffectsSafe enforces the effect-level conditions: no network,
// no non-CodeExec ⊤ (unresolved expansion), every path-shaped filesystem/
// process target inside a session root, and no dependency-manifest write.
func shellMarkerEffectsSafe(ctx context.Context, workDir string, report *api.Report) bool {
	for _, e := range report.Effects {
		if e.Kind == engine.KindNetEgress || e.Kind == engine.KindNetIngress {
			return false
		}
		if e.Target.IsTop() {
			// The unknown-driver signature is a target-less ⊤ CodeExec.
			// Any other ⊤ is an unresolved expansion (or an unbounded
			// write) the marker must not paper over.
			if e.Kind != engine.KindCodeExec {
				return false
			}
			continue
		}
		var pathKinds bool
		switch e.Kind {
		case engine.KindFSRead, engine.KindFSWrite, engine.KindFSMeta, engine.KindProcSpawn, engine.KindCodeExec:
			pathKinds = true
		default:
			// Stdio/Env* targets are literal words/variable names, not file
			// operands; no containment signal to enforce.
		}
		if !pathKinds {
			continue
		}
		manifestWrite := e.Kind == engine.KindFSWrite || e.Kind == engine.KindFSMeta
		for _, t := range e.Target.Targets() {
			if !shellPathLikeTarget(t) {
				continue
			}
			if manifestWrite && shellIsDependencyManifest(t) {
				return false
			}
			if !shellMarkerTargetInRoots(ctx, workDir, t) {
				return false
			}
		}
	}
	return true
}

// shellMarkerTargetInRoots resolves one path-shaped marker operand against
// the resolution base and reports whether it lands inside a session root.
// Harmless bit-bucket devices (/dev/null, /dev/full) are always in scope.
func shellMarkerTargetInRoots(ctx context.Context, base, target string) bool {
	abs, ok := shellResolveTarget(target, base)
	if !ok {
		// Relative with no base: the root set is non-empty by construction,
		// so an unanchorable operand cannot be established as in-root.
		return false
	}
	if shellIsHarmlessDevicePath(abs) {
		return true
	}
	for _, root := range SessionRoots(ctx) {
		if IsWithinRoot(ctx, root, abs) {
			return true
		}
	}
	return false
}

// shellIsDependencyManifest reports whether a path-shaped operand names a
// dependency/module-manifest file (basename match, case-insensitive).
func shellIsDependencyManifest(target string) bool {
	if target == "" {
		return false
	}
	name := strings.ToLower(path.Base(filepath.ToSlash(strings.TrimPrefix(strings.TrimPrefix(target, "\\"), "/"))))
	// The raw basename suffices for every catalogued form; a trailing slash
	// (directory operand) is not a manifest.
	_, ok := shellDependencyManifests[name]
	return ok
}

// shellMarkerCallsSafe enforces the per-call conditions: catalogued binaries
// (≥1 driver), per-driver argument screens, operand containment (arguments
// and redirection targets), and the manifest/network-device redirect rules.
func shellMarkerCallsSafe(ctx context.Context, workDir string, report *api.Report) bool {
	base := workDir
	if base == "" {
		base = WorkspacePathFrom(ctx)
	}
	hasDriver := false
	for _, call := range report.CommandCalls {
		allowedSubs, driver := shellVerificationDrivers[call.Resolved]
		if !driver {
			if _, plumbing := shellVerificationPlumbing[call.Resolved]; !plumbing {
				return false
			}
		} else {
			// A driver outside its subcommand allowlist (go get, npm install)
			// is not the catalogued verification form.
			if !shellFirstOperand(call.Args, allowedSubs) {
				return false
			}
			hasDriver = true
		}
		if !shellDriverArgsScreened(call.Resolved, call.Args) {
			return false
		}
		if !shellCallOperandsInRoots(ctx, base, call) {
			return false
		}
	}
	return hasDriver
}

// shellDriverArgsScreened applies the per-binary forbidden-argument tokens
// (exact or --long=value form).
func shellDriverArgsScreened(resolved string, args []string) bool {
	forbidden := shellDriverForbiddenFlags[resolved]
	for _, a := range args {
		for _, f := range forbidden {
			if a == f || (strings.HasPrefix(f, "--") && strings.HasPrefix(a, f+"=")) {
				return false
			}
		}
	}
	return true
}

// shellFirstOperand reports whether the first non-flag operand of args is in
// the allowlist (nil allowlist accepts everything).
func shellFirstOperand(args, allowed []string) bool {
	if allowed == nil {
		return true
	}
	for _, a := range args {
		if a == "" || strings.HasPrefix(a, "-") {
			continue
		}
		for _, ok := range allowed {
			if a == ok {
				return true
			}
		}
		return false
	}
	// No operand at all (e.g. bare `go`): nothing to screen against.
	return true
}

// shellCallOperandsInRoots containment-checks one call's path-shaped
// operands: positional arguments and redirection targets. Flag values glued
// with "=" are checked for their absolute-path value part; bare flags are
// skipped. fd-duplication targets ("1", "2", "&1") are not paths.
func shellCallOperandsInRoots(ctx context.Context, base string, call api.CommandCall) bool {
	for _, a := range call.Args {
		operand := a
		if strings.HasPrefix(a, "-") {
			if eq := strings.Index(a, "="); eq >= 0 {
				operand = a[eq+1:]
				if operand == "" || !strings.HasPrefix(operand, "/") && !shellIsWindowsAbsPath(operand) {
					continue // --flag=value with a non-path value
				}
			} else {
				continue // bare flag
			}
		}
		if shellIsFdTarget(operand) || !shellPathLikeTarget(operand) {
			continue
		}
		if !shellMarkerTargetInRoots(ctx, base, operand) {
			return false
		}
	}
	for _, r := range call.Redirs {
		if !shellRedirectMarkerSafe(ctx, base, r) {
			return false
		}
	}
	return true
}

// shellIsFdTarget reports whether a redirection target names a file
// descriptor ("1", "2", "&1") rather than a path.
func shellIsFdTarget(t string) bool {
	if t == "" {
		return false
	}
	rest := strings.TrimPrefix(t, "&")
	for _, c := range rest {
		if c < '0' || c > '9' {
			return false
		}
	}
	return true
}

// shellRedirectMarkerSafe checks one resolved redirection: /dev/tcp and
// /dev/udp are network pseudo-devices (marker off), write redirections must
// land inside a session root and must not target a dependency manifest.
func shellRedirectMarkerSafe(ctx context.Context, base string, r api.CallRedirect) bool {
	if !r.Known || r.Target == "" {
		return false // an unknown redirect target is an unresolved operand
	}
	posix := path.Clean(filepath.ToSlash(r.Target))
	if posix == "/dev/tcp" || posix == "/dev/udp" || strings.HasPrefix(posix, "/dev/tcp/") || strings.HasPrefix(posix, "/dev/udp/") {
		return false
	}
	if shellIsFdTarget(r.Target) || !shellPathLikeTarget(r.Target) {
		return true
	}
	write := strings.Contains(r.Op, ">") && r.Op != "<"
	if write && shellIsDependencyManifest(r.Target) {
		return false
	}
	if !shellMarkerTargetInRoots(ctx, base, r.Target) {
		return false
	}
	return true
}

// shellMarkerEnvSafe enforces the environment conditions: every EnvWrite
// name is in the safe table and every textual NAME=VALUE occurrence of that
// name carries a safe value (dynamically-built or decoy-unsafe values fail;
// an EnvWrite with no textual occurrence — value produced at run time —
// fails closed too).
func shellMarkerEnvSafe(command string, report *api.Report) bool {
	var envNames []string
	for _, e := range report.Effects {
		if e.Kind != engine.KindEnvWrite || e.Target.IsTop() {
			continue
		}
		envNames = append(envNames, e.Target.Targets()...)
	}
	if len(envNames) == 0 {
		return true
	}
	values := map[string][]string{}
	for _, m := range shellEnvAssignRe.FindAllStringSubmatch(command, -1) {
		name, value := m[1], strings.Trim(m[2], `"`)
		values[name] = append(values[name], value)
	}
	for _, name := range envNames {
		allowed, ok := shellVerificationSafeEnv[name]
		if !ok {
			return false
		}
		seen, ok := values[name]
		if !ok || len(seen) == 0 {
			// The value is not statically visible (read/export -n tricks):
			// cannot establish safety — fail closed.
			return false
		}
		for _, v := range seen {
			if !allowed[v] {
				return false
			}
		}
	}
	return true
}
