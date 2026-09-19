// SPDX-License-Identifier: Apache-2.0
//
// shellanalysis.go — the deterministic shell-command analysis engine layered
// on top of github.com/v0lka/flowsh. flowsh parses a bash/PowerShell command
// into a frozen effect IR (filesystem/network/process/credential effects, an
// exfiltration-pairing taint analysis, and a destructive-flags knowledge
// base); this file turns that report into the fixed-priority criteria C1–C8
// that shell-exec judges (and the advisory on-demand judge) consume.
//
// Design contract:
//
//   - The blocklist match stays in the tool's own Judge (bash_exec/posh_exec);
//     it is NOT re-implemented here. This engine covers everything the
//     effect IR can see.
//   - Criteria are evaluated in the fixed priority C1…C8; every fired
//     criterion is recorded in the digest, and the highest-priority one
//     becomes the JudgeOutcome.
//   - Canonicality: C1–C5 are hard AND canonical (hosts must never
//     auto-override them). C6 is hard but NON-canonical — an analysis
//     limitation the advisory judge may clear. C7/C8 are soft scope
//     questions.
//   - The flowsh score/grade are carried in the digest for context but are
//     deliberately NOT used as decision thresholds — the criteria fire off
//     structural facts (effect kinds, targets, KB classes), not scores.
//   - Containment (C4/C8) consults only FS* effects whose concrete targets
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
	"strings"
	"sync"

	"github.com/v0lka/flowsh/api"
	"github.com/v0lka/flowsh/engine"
)

// ShellDigestSchemaVersion tags the shell-analysis digest contract ("sp4rk-shell-analysis/v1").
// Bump it whenever the digest gains, loses or reshapes a field.
const ShellDigestSchemaVersion = "sp4rk-shell-analysis/v1"

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
	SchemaVersion string                   `json:"schemaVersion"`
	Lang          string                   `json:"lang"`
	Top           bool                     `json:"top"`
	Conservative  bool                     `json:"conservative"`
	Reason        string                   `json:"reason,omitempty"`
	Commands      int                      `json:"commands"`
	Resolution    ShellResolutionDigest    `json:"resolution"`
	Effects       []ShellEffectDigest      `json:"effects"`
	Score         ShellScoreDigest         `json:"score"`
	ExfilPairs    []ShellExfilPairDigest   `json:"exfilPairs"`
	Destructive   []ShellDestructiveDigest `json:"destructive"`
	Criteria      []ShellCriterion         `json:"criteria"`
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
	// reasons and for the non-canonical hard reason
	// ReasonCodeCommandUnboundedAnalysis.
	Canonical bool
}

// ─────────────────────────────────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────────────────────────────────

// AnalyzeShellCommandForJudge runs the flowsh-based deterministic analysis of
// one shell-exec tool input and maps it onto the C1–C8 criteria. It is the
// advisory-path helper: callers feed it the tool name ("bash_exec" or
// "posh_exec", which selects the dialect) and the tool's raw JSON input
// ({command, working_directory}); the returned ShellAnalysis carries the
// digest plus the winning judge outcome. Session roots for the containment
// criteria (C4/C8) come from ctx exactly as they did for the former shell-path
// containment check; with no roots attached those criteria cannot fire.
//
// A tool name outside the shell-exec pair, an unparsable input, or a failed
// knowledge-base load is an error — callers fail closed on it.
func AnalyzeShellCommandForJudge(ctx context.Context, toolName string, input json.RawMessage) (*ShellAnalysis, error) {
	lang, ok := shellToolLangs[toolName]
	if !ok {
		return nil, fmt.Errorf("shell analysis: unsupported tool %q (want bash_exec or posh_exec)", toolName)
	}
	var params struct {
		Command          string `json:"command"`
		WorkingDirectory string `json:"working_directory"`
	}
	if err := json.Unmarshal(input, &params); err != nil {
		return nil, fmt.Errorf("shell analysis: parse %s input: %w", toolName, err)
	}
	analyzer, err := shellFlowshAnalyzer()
	if err != nil {
		return nil, fmt.Errorf("shell analysis: flowsh analyzer init: %w", err)
	}
	report := analyzer.Analyze(lang, params.Command)
	return evaluateShellReport(ctx, params.WorkingDirectory, report), nil
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

// ShellJudgeOutcome is the deterministic outcome the bash_exec/posh_exec
// Judges report for an analysis attached via [WithShellAnalysis]: the
// analysis's winning [ShellAnalysis.Outcome] verbatim (Allow=true when no
// criterion fired). When nothing is attached it returns the empty outcome,
// deferring the call to the advisory judges. When the attachment carries an
// error — the analyzer or its embedded knowledge base failed to initialise,
// so the deterministic floor (C1–C8) is unavailable for this call — it FAILS
// CLOSED: a hard canonical [ReasonCodeCommandAnalysisUnavailable] outcome, so
// the call still escalates under an `allow` policy and blocks under
// verify-on-edit's unattended path, instead of running with no floor at all.
// The error is logged to [slog.Default] as well.
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

// evaluateShellReport applies the fixed-priority criteria C1–C8 to a flowsh
// report and assembles the digest + winning outcome. It never fails: a report
// is always assessable (flowsh itself degrades to ⊤/conservative, which C5/C6
// handle).
func evaluateShellReport(ctx context.Context, workDir string, report *api.Report) *ShellAnalysis {
	var criteria []ShellCriterion
	fire := func(code JudgeReasonCode, severity JudgeSeverity, canonical bool) {
		criteria = append(criteria, ShellCriterion{Fired: code, Severity: severity, Canonical: canonical})
	}

	hasNetEgress := shellHasEffectKind(report, engine.KindNetEgress)
	hasPrivEsc := shellHasEffectKind(report, engine.KindPrivEsc)
	hasCredAccess := shellHasEffectKind(report, engine.KindCredAccess)
	hasExfilPair := len(report.Score.ExfilPairs) > 0
	unbounded := report.Top || report.Conservative
	unboundedWrite := shellHasUnboundedWrite(report)

	// C1 — exfiltration flow: a secret read paired with tainted egress.
	if hasExfilPair {
		fire(ReasonCodeCommandExfilFlow, JudgeSeverityHard, true)
	}
	// C2 — privilege escalation effect.
	if hasPrivEsc {
		fire(ReasonCodeCommandPrivilegeEscalation, JudgeSeverityHard, true)
	}
	// C3 — direct write/metadata effect on a system path or raw device.
	if len(shellSystemOrDeviceTargets(report)) > 0 {
		fire(ReasonCodeCommandSystemWrite, JudgeSeverityHard, true)
	}
	// C4 — irreversible destructive command (KB class D/E) writing outside
	// the session roots.
	if shellHasDestructiveClassDE(report) && !report.Score.Reversible {
		if len(shellOutsideRoots(ctx, report, workDir, shellFSWriteKinds, false)) > 0 {
			fire(ReasonCodeCommandDestructiveOutsideRoots, JudgeSeverityHard, true)
		}
	}
	// C5 — unbounded analysis with network egress: download-cradle shape.
	if unbounded && hasNetEgress {
		fire(ReasonCodeCommandDownloadCradle, JudgeSeverityHard, true)
	}
	// C6 — the analyzer could not bound the command (⊤/conservative) without
	// network egress, OR an irreversible write's target could not be resolved
	// (⊤): hard but NON-canonical, an analysis limitation the advisory judge
	// may clear.
	if (unbounded || unboundedWrite) && !hasNetEgress {
		fire(ReasonCodeCommandUnboundedAnalysis, JudgeSeverityHard, false)
	}
	// C7 — credential access without an exfil pairing.
	if hasCredAccess && !hasExfilPair {
		fire(ReasonCodeCredentialAccess, JudgeSeveritySoft, false)
	}
	// C8 — direct FS* effect outside the session roots on a non-system
	// path: the scope question, reusing the outside_session_roots code.
	if len(shellOutsideRootDirectNonSystemTargets(ctx, report, workDir)) > 0 {
		fire(ReasonCodeOutsideSessionRoots, JudgeSeveritySoft, false)
	}

	result := &ShellAnalysis{Digest: newShellAnalysisDigest(report, criteria)}
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
		return "Shell analysis: unbounded command with network egress (possible download cradle)"
	case ReasonCodeCommandUnboundedAnalysis:
		return "Shell analysis: command could not be bounded by static analysis (top/conservative), no network egress found"
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
func shellSystemOrDeviceTargets(report *api.Report) []string {
	var out []string
	for _, e := range report.Effects {
		if !shellEffectKindsHas([]engine.EffectKind{engine.KindFSWrite, engine.KindFSMeta}, e) || e.Mode != engine.ModeDirect || e.Target.IsTop() {
			continue
		}
		for _, t := range e.Target.Targets() {
			if !shellPathLikeTarget(t) {
				continue
			}
			if abs, ok := shellResolveTarget(t, ""); ok && shellIsSystemOrRawDevicePath(abs) {
				out = append(out, abs)
			}
		}
	}
	return out
}

// shellOutsideTarget is one out-of-root containment hit: the resolved
// absolute path plus the effect kind that produced it. C8 needs the kind to
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
// directly (Mode == Direct), as the C8 scope criterion requires. With no
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

// shellOutsideRootDirectNonSystemTargets returns the C8 targets: directly
// performed FS* effects whose path-shaped, resolvable targets fall outside
// every session root. Direct writes/metadata on a system path or raw device
// are excluded — C3 owns those, at a higher priority and as a hard canonical
// control. Reads of system *files* are deliberately NOT excluded: no
// higher-priority criterion covers them, and an out-of-root read of a system
// (or credential) file is exactly the scope question C8 exists to raise —
// leaving it silent would reopen the gap the former shell-path containment
// check used to close.
// Raw-device reads (/dev/urandom, /dev/zero, …) ARE excluded — they are
// routine inputs, not a scope concern, and a raw device is "system" only as
// a write target. Harmless devices (/dev/null, /dev/full) are exempt via
// [shellIsHarmlessDevicePath] inside the containment walk.
func shellOutsideRootDirectNonSystemTargets(ctx context.Context, report *api.Report, workDir string) []string {
	var out []string
	for _, t := range shellOutsideRoots(ctx, report, workDir, shellFSAllKinds, true) {
		if t.kind == engine.KindFSWrite || t.kind == engine.KindFSMeta {
			// C3 owns writes/metadata on system paths and raw devices.
			if shellIsSystemOrRawDevicePath(t.path) {
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
// every host ([resolveShellToken] keeps it forward-slashed and absolute), and
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
func shellIsSystemOrRawDevicePath(absPath string) bool {
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
	// ("c:\program filesold") is not misclassified.
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
func newShellAnalysisDigest(report *api.Report, criteria []ShellCriterion) ShellAnalysisDigest {
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
	destructive := make([]ShellDestructiveDigest, 0, len(report.Destructive))
	for _, d := range report.Destructive {
		destructive = append(destructive, ShellDestructiveDigest{Command: d.Command, Spec: d.Spec, Class: d.Class})
	}
	if criteria == nil {
		criteria = []ShellCriterion{}
	}
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
		ExfilPairs:  pairs,
		Destructive: destructive,
		Criteria:    criteria,
	}
}
