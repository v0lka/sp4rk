// SPDX-License-Identifier: Apache-2.0
//
// shellpaths.go — extract path-like tokens from a shell command string and
// resolve shell idioms (tilde, environment variables, relative "..") to
// absolute paths. It mirrors the cross-platform, build-tag-free contract of
// tools/builtins/workdir.go: it must compile on linux/darwin/windows.
//
// The resolution is deliberately conservative: unresolvable tokens (unknown
// user home "~user", unset/empty environment variables, relative tokens
// without a base workDir) are SKIPPED rather than emitted. This favours
// false negatives over false positives; the per-tool blacklist and the
// user_confirm security policy remain the authoritative backstops.
//
// Absolute-path detection reuses [pathRegex] (the same pattern the judge
// fast-path consults), so the shell extractor and the JSON-input extractor
// agree on what counts as a path. Tilde, env-var and ".." tokens are matched
// by additional RE2-compatible patterns (no lookarounds) and dispatched by
// the token's leading character in [resolveShellToken].

package tools

import (
	"context"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"strings"
)

// ShellKind identifies the shell dialect whose idioms a token string uses.
type ShellKind string

const (
	// ShellBash is the POSIX/bash dialect: "~", "~user", "$VAR", "${VAR}".
	ShellBash ShellKind = "bash"
	// ShellPosh is the PowerShell dialect: "~", "$env:VAR", "${env:VAR}".
	ShellPosh ShellKind = "posh"
)

// Regex fragments. All are RE2-compatible (no lookarounds). Within character
// classes, "." and "-" are escaped for clarity; "~" is literal.

// tildeBashRe matches "~", "~user", "~/path" and "~user/path" in the bash
// dialect. The (possibly empty) username run is followed by an optional
// slash-separated remainder.
const tildeBashRe = `~[a-zA-Z0-9_.\-]*(?:/[a-zA-Z0-9/_.\-~]+)?`

// tildePoshRe matches a bare "~" optionally followed by a backslash- or
// slash-separated remainder (PowerShell home expansion is "~" only — no
// "~user" form).
const tildePoshRe = `~(?:[\\/][A-Za-z0-9\\/_\.\-~]*)?`

// relDotDotRe matches a ".." parent-reference optionally followed by a
// separator and a path remainder, e.g. "..", "../x", "..\..\etc".
const relDotDotRe = `\.\.(?:[\\/][A-Za-z0-9\\/_\.\-~]*)?`

// bashEnvBraceRe matches "${VAR}" optionally followed by a slash-remainder.
const bashEnvBraceRe = `\$\{[A-Za-z_][A-Za-z0-9_]*\}(?:/[a-zA-Z0-9/_.\-~]+)?`

// bashEnvRe matches "$VAR" optionally followed by a slash-remainder.
const bashEnvRe = `\$[A-Za-z_][A-Za-z0-9_]*(?:/[a-zA-Z0-9/_.\-~]+)?`

// poshEnvBraceRe matches "${env:VAR}" optionally followed by a path remainder.
const poshEnvBraceRe = `\$\{env:[A-Za-z_][A-Za-z0-9_]*\}(?:[\\/][A-Za-z0-9\\/_\.\-~]+)?`

// poshEnvRe matches "$env:VAR" optionally followed by a path remainder.
const poshEnvRe = `\$env:[A-Za-z_][A-Za-z0-9_]*(?:[\\/][A-Za-z0-9\\/_\.\-~]+)?`

// shellTokenRe maps a ShellKind to a single combined regex that finds every
// candidate token in one pass. Alternatives are ordered so the longest / most
// specific match at a given position wins under Go's leftmost-first semantics:
// the brace/env forms ("${env:..", "${..", "$env:..") precede the bare "$VAR"
// form, tilde precedes the absolute alternatives (so "~/a" is not split into
// "~" plus an absolute "/a"), and the reused [pathRegex] alternatives precede
// the relative ".." form.
var shellTokenRe = map[ShellKind]*regexp.Regexp{
	ShellBash: regexp.MustCompile(`(?:` + bashEnvBraceRe + `|` + bashEnvRe + `|` + tildeBashRe + `|` + pathRegex.String() + `|` + relDotDotRe + `)`),
	ShellPosh: regexp.MustCompile(`(?:` + poshEnvBraceRe + `|` + poshEnvRe + `|` + tildePoshRe + `|` + pathRegex.String() + `|` + relDotDotRe + `)`),
}

// tildeUserRe matches a bash "~user" form: a tilde followed by one or more
// username characters. Unlike "~" and "~/…" (which resolve to the current
// user's home), "~user" expands to the named user's home directory and cannot
// be resolved without a user-database lookup. The word-boundary requirement is
// enforced in [UnresolvablePathTokens], not in the regex, because RE2 lacks
// lookbehind.
const tildeUserRe = `~[a-zA-Z0-9_.\-]+`

// bashParamExpansionRe matches any bash "${...}" brace form. The plain
// "${VAR}" and "${VAR}/suffix" forms are handled by [resolveEnvToken]; a
// parameter-expansion operator form (e.g. "${VAR:-/etc/passwd}") is not
// assessed and, when path-bearing, must be escalated (see
// [isPathBearingParamExpansion]).
const bashParamExpansionRe = `\$\{[^{}]*\}`

var unresolvableTokenRe = map[ShellKind]*regexp.Regexp{
	ShellBash: regexp.MustCompile(`(?:` + tildeUserRe + `|` + bashParamExpansionRe + `)`),
}

// bashPlainVarRe matches a plain "$NAME" or "${NAME}" reference — no operator
// forms, which [bashParamExpansionRe] covers separately. A leading "=" or
// "~" sigil is tolerated so zsh word-splitting/pattern spellings ("$=D")
// are still found. It feeds the dynamically-rebound-variable pass of
// [UnresolvablePathTokens].
var bashPlainVarRe = regexp.MustCompile(`\$[=~]?[A-Za-z_][A-Za-z0-9_]*|\$\{[=~]?[A-Za-z_][A-Za-z0-9_]*\}`)

// bashBracedVarRe is the compiled form of [bashParamExpansionRe], matching
// any "${...}" braced parameter form (plain or operator). It feeds the
// dynamically-rebound-variable pass of [UnresolvablePathTokens].
var bashBracedVarRe = regexp.MustCompile(bashParamExpansionRe)

// bashPositionalVarRe matches a positional parameter reference ("$1",
// "${12}", "$@", "$*") plus an optional path suffix. Positional values come
// from the invocation context or a "set --" rebind, never from this
// command's static bindings.
var bashPositionalVarRe = regexp.MustCompile(`\$\{?[0-9@*]+\}?(?:/[a-zA-Z0-9/_.\-~]+)?`)

// bashArithExpRe matches a word-level arithmetic expansion — "$((…))", with
// one level of nested parentheses so subshell and command-substitution
// operands stay whole — plus bash's legacy "$[…]" spelling. A bare "((…))"
// command is an assignment or condition, not a value producer, and is
// folded by the binding walker's arithmetic handling instead.
var bashArithExpRe = regexp.MustCompile(`\$\(\((?:[^()]|\([^()]*\))*\)\)|\$[[][^[\]]*[\]]`)

// arithExpNameRe extracts the bare identifier references an arithmetic
// expression evaluates — inside arithmetic a variable needs no "$" sigil,
// which is exactly why the plain/braced variable passes cannot see them.
var arithExpNameRe = regexp.MustCompile(`[A-Za-z_][A-Za-z0-9_]*`)

// arithExpReferencesUnassessable reports whether the arithmetic expansion
// tok evaluates any variable the binding walker marks unassessable (dynamic
// or whole-command opaque): the expansion's runtime value is then unknown —
// often attacker-controlled file content — exactly like a plain "$D" to a
// rebound name. Assignment targets ("D=5") are skipped: the binding walker's
// arithmetic handling already marks the assigned name, while the target
// itself is not a reference whose unknown value flows out.
func arithExpReferencesUnassessable(tok string, bindings *envBindings) bool {
	for _, m := range arithExpNameRe.FindAllStringIndex(tok, -1) {
		if arithNameIsAssignmentTarget(tok, m[1]) {
			continue
		}
		if bindings.unassessable(tok[m[0]:m[1]]) {
			return true
		}
	}
	return false
}

// arithNameIsAssignmentTarget reports whether the identifier ending at index
// end inside an arithmetic expression is the TARGET of a plain assignment
// ("D=5", "D = 5"). A comparison ("D==5") is not an assignment, and a
// compound operator ("D+=1", "D<<=2") reads the name as well as writing it,
// so those occurrences still count as references.
func arithNameIsAssignmentTarget(tok string, end int) bool {
	i := end
	for i < len(tok) && (tok[i] == ' ' || tok[i] == '\t') {
		i++
	}
	if i >= len(tok) || tok[i] != '=' {
		return false
	}
	if i+1 < len(tok) && tok[i+1] == '=' {
		return false // "==": comparison, not assignment
	}
	return true
}

// bracedOperatorCarriesPath reports whether a braced "${...}" OPERATOR form
// (non-plain, non-indirect, non-length) is used where an absolute path can be
// assembled: the enclosing word is absolute-shaped (a "/" precedes the token
// within it, as in "/e${X:1}") or continues with a path suffix after the
// token ("${HOME:-safe}/.ssh/id_rsa"). Such a reference routes the base
// name's value through an operator the resolver cannot compose, so it must
// fail closed instead of resolving to a benign candidate.
func bracedOperatorCarriesPath(tok, command string, start, end int) bool {
	inner := tok[2 : len(tok)-1]
	if strings.HasPrefix(inner, "!") || strings.HasPrefix(inner, "#") {
		return false // indirect and length forms have their own rules
	}
	name, rest := splitEnvName(inner)
	if name == "" || rest == "" {
		return false // plain "${NAME}" — resolved through valueCandidates
	}
	wordStart, wordEnd := shellWordBounds(command, start, end)
	if wordStart < start && wordStartsAbsolute(command[wordStart:]) {
		return true
	}
	tail := strings.Map(func(r rune) rune {
		if r == '"' || r == '\'' {
			return -1
		}
		return r
	}, command[end:wordEnd])
	return strings.Contains(tail, "/")
}

// isIndirectParamExpansion reports whether a "${...}" token is an indirect
// expansion "${!name}" — excluding the "${!prefix@}"/"${!prefix*}" name
// listings, whose values are identifier lists and cannot carry a path. The
// indirect form resolves the referenced variable's NAME from the base
// name's runtime VALUE, so no static candidate list can assess it.
func isIndirectParamExpansion(tok string) bool {
	inner := tok[2 : len(tok)-1]
	if !strings.HasPrefix(inner, "!") {
		return false
	}
	return !strings.HasSuffix(inner, "@") && !strings.HasSuffix(inner, "*")
}

// isPathBearingParamExpansion reports whether a "${...}" token is a parameter
// expansion whose operand references a path — i.e. the part after the variable
// name contains "/", "~", or "$". Plain "${VAR}" and benign defaults such as
// "${VAR:-hello}" or "${count:=5}" are not path-bearing and are NOT escalated;
// only an expansion that could hide an out-of-root path (e.g.
// "${VAR:-/etc/passwd}") is.
func isPathBearingParamExpansion(tok string) bool {
	inner := tok[2 : len(tok)-1] // strip "${" and "}"
	i := 0
	for i < len(inner) {
		c := inner[i]
		isName := (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_'
		if !isName {
			break
		}
		i++
	}
	if i == 0 || i == len(inner) {
		return false // malformed or plain "${VAR}"
	}
	return strings.ContainsAny(inner[i:], "/~$")
}

// UnresolvablePathTokens returns the path-like tokens in command that
// [ResolveShellPathTokens] cannot resolve to an absolute path, and that
// therefore represent an out-of-root reference that was NOT assessed. Callers
// (the shell-exec Judges) escalate these as HARD — an input that cannot be
// assessed at all must never be silently let through (see JudgeSeverityHard).
//
// It is the conservative counterpart to [ResolveShellPathTokens], which favours
// false negatives (skipping unresolvable tokens) so its output stays clean of
// fabricated paths. For a security containment check the dangerous direction is
// a false negative, so the skipped tokens are recovered here.
//
// Three bash forms are detected:
//
//   - "~user": a "~" at a word boundary followed by username characters. The
//     word-boundary check skips "~" that continues a path-component word (e.g.
//     the "~3" in "git log HEAD~3", which is a revision suffix, not a tilde
//     expansion). PowerShell has no "~user" form.
//   - a path-bearing "${VAR<operator>…}" parameter expansion (see
//     [isPathBearingParamExpansion]). PowerShell has no equivalent operator
//     form (its "~" and "$env:"/"${env:}" idioms are all resolved by
//     [resolveEnvToken] and [resolveTildeToken]), so the PowerShell dialect
//     reports nothing.
//   - a plain "$NAME"/"${NAME}" reference — or a braced form derived from it
//     ("${A[0]}", "${D:0}", "${D@Q}", "${D^}", or an indirect "${!…}") — to
//     a name the binding walker cannot assess: rebound by a construct that
//     never produces a plain assignment node ("read D", "printf -v D",
//     "mapfile D", "unset D", "getopts", for/select loop iteration, an
//     arithmetic assignment "((D=5))", an attribute-changing
//     "declare -a/-n/…", a dynamic right-hand side), or to ANY name at all
//     once the command contains an opaque "source"/"."/"eval"/"let"
//     (however spelled), a dynamically-built command word, or a non-literal
//     "bash -c" script. The runtime value of such a reference is unknown —
//     often attacker-controlled file content — so no candidate list can
//     assess it. A reference to a name with only plain literal bindings is
//     NOT reported: [resolveEnvToken] assesses it (unioning the empty
//     expansion).
func UnresolvablePathTokens(command string, shell ShellKind) []string {
	re := unresolvableTokenRe[shell]
	if re == nil {
		return nil
	}
	seen := make(map[string]struct{})
	var out []string
	for _, m := range re.FindAllStringIndex(command, -1) {
		start, end := m[0], m[1]
		tok := command[start:end]
		// "~user" only: require a word start. A "~" glued to a preceding
		// path-component character (e.g. "HEAD~3") is not a tilde expansion.
		switch tok[0] {
		case '~':
			if start > 0 && isPathComponentChar(command[start-1]) {
				continue
			}
		case '$':
			// Only flag parameter expansions whose operand could hide a path,
			// so benign "${VAR:-hello}" defaults are not escalated; every
			// indirect "${!name}" form (except the name listings); and braced
			// OPERATOR forms used where a path is assembled — a "/" before
			// the token in the same word ("/e${X:1}") or a path suffix after
			// it ("${HOME:-safe}/.ssh/id_rsa"): the operator routes the
			// variable's value through a transformation no candidate list can
			// compose while the runtime word still lands on an absolute path.
			if !isPathBearingParamExpansion(tok) && !isIndirectParamExpansion(tok) &&
				!bracedOperatorCarriesPath(tok, command, start, end) {
				continue
			}
		}
		if _, dup := seen[tok]; dup {
			continue
		}
		seen[tok] = struct{}{}
		out = append(out, tok)
	}
	if shell == ShellBash {
		out = appendUnassessableVarTokens(command, seen, out)
	}
	return out
}

// appendUnassessableVarTokens appends the variable references whose name the
// bash binding walker marks unassessable (see [envBindings.unassessable]):
// the name is rebound in-command by a construct whose runtime value is
// statically unknown, or the whole command is opaque ("source"/"."/"eval"/
// "let", a dynamically-built command word, or a non-literal "bash -c"
// script). Both plain "$NAME"/"${NAME}" references and braced operator forms
// derived from a name ("${A[0]}", "${D:0}", "${D@Q}", "${D^}") fail closed
// at the same HARD escalation as "~user" and path-bearing parameter
// expansions, as does every indirect "${!…}" form once any rebinding is
// present — the referenced variable's NAME is itself runtime data.
// Positional references ("$1", "$@") escalate when they carry a path suffix
// or when the command rebinds them ("set -- …", "shift"). When the command
// contains no dynamic or opaque rebinding at all, nothing else is appended
// — plain literal bindings and environment-only names stay assessable
// through [resolveEnvToken].
func appendUnassessableVarTokens(command string, seen map[string]struct{}, out []string) []string {
	bindings := collectCommandEnvBindings(command)
	// Positional parameters are never statically assessable: their values
	// come from the invocation context or an in-command "set --"/"shift"
	// rebind. A reference carrying a PATH SUFFIX ("cat \"$1/etc/passwd\"")
	// always escalates — the empty/unknown expansion concatenated with an
	// absolute suffix is exactly the hidden out-of-root shape; so does a
	// positional COMPOSED with more word content ("$1$2", "$1frag"), which
	// no candidate list can express — while a bare positional
	// ("awk '{print $1}'") escalates only once the command rebinds the
	// positionals, keeping ordinary awk/sed one-liners quiet.
	positionalRebound := bindings != nil && bindings.positionalRebound
	for _, m := range bashPositionalVarRe.FindAllStringIndex(command, -1) {
		tok := command[m[0]:m[1]]
		// Single-quoted content is literal (awk/sed scripts) — unless the
		// command rebinds positionals, in which case even a nested -c
		// script's "$1" references hold unknown values.
		if inSingleQuotes(command, m[0]) && !positionalRebound {
			continue
		}
		ws, we := shellWordBounds(command, m[0], m[1])
		composed := (ws < m[0] || we > m[1]) && (!inSingleQuotes(command, ws) || positionalRebound)
		if !positionalRebound && !strings.Contains(tok, "/") && !composed {
			continue
		}
		if _, dup := seen[tok]; dup {
			continue
		}
		seen[tok] = struct{}{}
		out = append(out, tok)
	}
	// Words whose candidate product cannot be composed — an operator form,
	// a positional, or an over-cap product — are unassessable as a whole:
	// the true runtime combination may be any of the dropped ones, so the
	// word itself escalates. The word qualifies when it starts with "/"
	// ("/$P$Q/x") or when any referenced variable holds an absolute-valued
	// candidate ("$P$Q/passwd" with P="/e" — the composition lands outside
	// the roots).
	for _, m := range bashPlainVarRe.FindAllStringIndex(command, -1) {
		if inSingleQuotes(command, m[0]) {
			continue // literal text — no composed runtime value
		}
		ws, we := shellWordBounds(command, m[0], m[1])
		if ws >= m[0] && we <= m[1] {
			continue // single-token word — per-token rules apply
		}
		word := command[ws:we]
		if _, dup := seen[word]; dup {
			continue
		}
		if !wordStartsAbsolute(word) && !wordHasAbsoluteCandidate(word, bindings) {
			continue // a purely relative composition stays in-root
		}
		if _, ok := absoluteWordExpansions(word, bindings); ok {
			continue
		}
		seen[word] = struct{}{}
		out = append(out, word)
	}
	if bindings == nil || (!bindings.opaque && len(bindings.dynamic) == 0) {
		return out
	}
	sawVarToken := false
	seenName := make(map[string]struct{})
	for _, m := range bashPlainVarRe.FindAllStringIndex(command, -1) {
		tok := command[m[0]:m[1]]
		body := strings.TrimSuffix(strings.TrimPrefix(strings.TrimPrefix(tok, "$"), "{"), "}")
		// A leading "="/"~" is a zsh word-splitting/pattern sigil ("$=D"),
		// not part of the name.
		name := strings.TrimLeft(body, "=~")
		if _, dup := seenName[name]; dup {
			continue
		}
		if !bindings.unassessable(name) {
			continue
		}
		seenName[name] = struct{}{}
		sawVarToken = true
		if _, dup := seen[tok]; dup {
			continue
		}
		seen[tok] = struct{}{}
		out = append(out, tok)
	}
	// Braced operator forms: bashPlainVarRe skips them, and the main
	// operator pass (isPathBearingParamExpansion) escalates only operands
	// carrying a literal path marker — a reference whose VALUE is the
	// variable's runtime content carries none. Escalate any braced form
	// whose BASE name is unassessable, and every indirect ("${!…}") form.
	for _, m := range bashBracedVarRe.FindAllStringIndex(command, -1) {
		tok := command[m[0]:m[1]]
		if _, dup := seen[tok]; dup {
			continue
		}
		inner := tok[2 : len(tok)-1]
		indirect := strings.HasPrefix(inner, "!")
		base := strings.TrimPrefix(inner, "!")
		// A zsh parenthesized expansion-flag group ("${(z)D}") prefixes the
		// name.
		if strings.HasPrefix(base, "(") {
			if end := strings.IndexByte(base, ')'); end >= 0 {
				base = base[end+1:]
			}
		}
		name, _ := splitEnvName(base)
		if name == "" || (name[0] >= '0' && name[0] <= '9') {
			continue // positional parameter or malformed — no base name
		}
		if !indirect && !bindings.unassessable(name) {
			continue
		}
		sawVarToken = true
		seen[tok] = struct{}{}
		out = append(out, tok)
	}
	// Arithmetic expansions: inside "$((…))" (and the legacy "$[…]"
	// spelling) a variable is referenced WITHOUT the "$" sigil, so neither
	// the plain nor the braced pass can see it — a reference to an
	// unassessable name evaluated in arithmetic would otherwise escape
	// every pass while its runtime value flows into the word exactly like
	// "$D" would. Escalate the whole expansion when any name it evaluates
	// is unassessable; a single-quoted occurrence is literal text.
	for _, m := range bashArithExpRe.FindAllStringIndex(command, -1) {
		if inSingleQuotes(command, m[0]) {
			continue // literal text — no runtime reference
		}
		tok := command[m[0]:m[1]]
		if _, dup := seen[tok]; dup {
			continue
		}
		if !arithExpReferencesUnassessable(tok, bindings) {
			continue
		}
		sawVarToken = true
		seen[tok] = struct{}{}
		out = append(out, tok)
	}
	// An OPAQUE construct (source/eval/let, a dynamically-built command
	// word, a pipe- or stdin-fed interpreter, a non-POSIX dialect script)
	// executes hidden shell text even when the command carries NO variable
	// reference — without this sentinel such commands would slip both
	// passes silently. It is appended only when no variable reference was
	// found to carry the escalation (a reference-bearing command already
	// escalates through its tokens).
	if bindings.opaque && !sawVarToken {
		out = append(out, opaqueConstructToken)
	}
	return out
}

// opaqueConstructToken is the synthetic token reported for a command whose
// opaque constructs (source/eval-style hidden shell text) carry no variable
// reference of their own to escalate.
const opaqueConstructToken = "<opaque-rebinding-construct>"

// ResolveShellPathTokens extracts path-like tokens from a shell command string
// and resolves shell idioms to absolute paths:
//
//   - Absolute paths (matched by [pathRegex]) are passed through as-is, but
//     only when their leading separator ("/", or a drive letter's separator)
//     STARTS a token. A "/" that follows a path-component character is a
//     separator inside a relative path (e.g. the "/src" in
//     "frontend/src/main.tsx"): rather than treat the fragment as an absolute
//     path, the ENTIRE enclosing relative-path word is resolved against
//     workDir. This avoids both the false positive of flagging an in-workspace
//     relative path and the false negative of silently dropping a genuine
//     parent-ref escape hidden behind a relative prefix (e.g.
//     "./../etc/passwd", "a/../../etc/passwd", "subdir/../..").
//   - Tokens consisting entirely of separators — a bare "//" run (POSIX) or a
//     drive prefix followed by only separators ("C:\\") — are SKIPPED: they
//     are shell-language artifacts (the "//" of a sed address
//     "sed 's/.*function //'", a comment marker, an integer-division
//     "$(( total // count ))", an escaped drive root) that carry no path
//     component and name no out-of-root location; resolving them anyway would
//     clean a bare "//" to the filesystem root and produce a false-positive
//     escalation. See [isPureSeparatorRunToken].
//   - "~" and "~user" (bash) and bare "~" (posh) expand to the current user's
//     home directory via [os.UserHomeDir] (falling back to $HOME then
//     $USERPROFILE). "~user" is best-effort and skipped when unresolvable.
//   - "$VAR"/"${VAR}" (bash) and "$env:VAR"/"${env:VAR}" (posh) expand via
//     [os.Getenv] and, for bash, the command's own literal "VAR=value"
//     bindings ([collectCommandEnvBindings]); when several expansions are
//     possible — a binding alongside a non-empty env value, a name assigned
//     more than once / dynamically, or the EMPTY expansion of a possibly
//     unset or cleared variable (see [envBindings.emptyExpansionPossible]) —
//     ALL of them are reported: a binding adds candidates, it never masks an
//     environment value, an earlier binding, or the empty expansion
//     (fail-closed). Tokens whose every expansion is empty AND carry no path
//     suffix are SKIPPED (a bare empty expansion names no path). A
//     word-initial "/" before the env token ("/$D/passwd") additionally
//     resolves the absolutized variants, because the runtime expansion of
//     such a word is an absolute path even though the token itself resolves
//     to a relative join.
//   - Relative tokens containing ".." resolve against workDir; only tokens
//     with ".." are analyzed (plain relative names already resolve inside the
//     validated workDir and are ignored). Relative tokens are skipped when
//     workDir is empty (cannot resolve without a base).
//
// Every resolved path is [filepath.Clean]-ed and the result is deduplicated.
// Only absolute resolved paths are returned.
func ResolveShellPathTokens(command string, shell ShellKind, workDir string) []string {
	re := shellTokenRe[shell]
	if re == nil {
		// Unknown dialect: fall back to the bash grammar.
		re = shellTokenRe[ShellBash]
	}

	seen := make(map[string]struct{})
	var out []string
	// Collect in-command literal variable bindings (bash only) so "$VAR" can
	// resolve to the command's own assignment value ("VAR=value",
	// "export VAR=value", "VAR=value cmd ...") in ADDITION to the process
	// environment — a binding never masks an env value, and a re-bound or
	// dynamically assigned name contributes ALL of its distinct literal
	// values as candidates (see [envBindings]). PowerShell has no mvdan
	// parser here, so it gets no bindings.
	var bindings *envBindings
	if shell == ShellBash {
		bindings = collectCommandEnvBindings(command)
	}
	for _, m := range re.FindAllStringIndex(command, -1) {
		start, end := m[0], m[1]
		tok := command[start:end]
		// A token consisting entirely of separators — a bare "//" run (the
		// trailing address of "sed 's/.*function //'", a comment marker, an
		// integer-division "$(( a // b ))") or a drive prefix with only
		// separators ("C:\\") — is a shell-language artifact that names no
		// location. Skip it before dispatch: resolving a bare "//" cleans to
		// the filesystem root and would produce a false-positive escalation.
		if isPureSeparatorRunToken(tok) {
			continue
		}
		// Boundary checks for tokens whose leading character could continue a
		// preceding relative-path word (".." parent-refs and POSIX "/abs"
		// paths). Env ("$VAR") and tilde ("~") tokens start with non-path
		// characters and are inherently bounded, so they are exempt — the
		// '$' case below handles absolute-shaped assembled words.
		switch tok[0] {
		case '.':
			// A ".." glued to a preceding path-component char — e.g. the ".." in
			// the filename "x.." inside "x../../etc/passwd" — is part of a
			// filename component, never a standalone parent-directory reference,
			// so it cannot on its own be the start of an out-of-root escape.
			if start > 0 && isPathComponentChar(command[start-1]) {
				continue
			}
			// Trailing boundary: a run of 3+ dots ("...", "....") is an ellipsis
			// (e.g. the go-test recursive pattern "go test ..."), not a
			// parent-directory reference. Only treat ".." as a parent ref when it
			// is not immediately followed by another dot.
			if after := start + 2; after < len(command) && command[after] == '.' {
				continue
			}
		case '/':
			// A "/" that follows a path-component char is the separator INSIDE a
			// relative path (e.g. the "/src" in "frontend/src/main.tsx"), not the
			// start of an absolute path. Resolving just the "/..." fragment
			// against the filesystem root would misclassify in-workspace relative
			// paths — but a plain skip would also swallow a genuine parent-ref
			// escape hidden behind a relative prefix, such as "./../etc/passwd"
			// or "a/../../etc/passwd". Resolve the WHOLE relative-path word
			// against workDir instead: in-root words resolve inside workDir (and
			// are dropped by PathsOutsideRoots), while escaping words surface as
			// a real out-of-root path. Windows drive-letter alternatives start
			// with a letter, never "/", so they never reach this branch.
			if start > 0 && isPathComponentChar(command[start-1]) {
				wordStart := relativePathWordStart(command, start)
				resolved, ok := resolveRelativeToken(command[wordStart:end], workDir)
				if !ok || resolved == "" || !isAbsResolved(resolved) {
					continue
				}
				if _, dup := seen[resolved]; dup {
					continue
				}
				seen[resolved] = struct{}{}
				out = append(out, resolved)
				continue
			}
		case '$':
			// An env token inside a word whose runtime expansion may be
			// ABSOLUTE — the word starts with "/" ("/$D/passwd",
			// "/et$B/passwd") or one of its referenced variables holds an
			// absolute-valued candidate ("$P$Q/passwd" with P="/e") —
			// assembles at runtime from the literal fragments and every
			// referenced variable's candidates. Resolving the token alone
			// cannot see the literal prefix or the sibling tokens, so the
			// whole word is expanded across the candidate product (see
			// absoluteWordExpansions). Words that cannot be composed
			// (operator forms, over-cap products) are left to the
			// unresolvable pass, which escalates them hard.
			if shell == ShellBash && bindings != nil && !inSingleQuotes(command, start) {
				if ws, we := shellWordBounds(command, start, end); (ws < start || we > end) &&
					(wordStartsAbsolute(command[ws:]) || wordHasAbsoluteCandidate(command[ws:we], bindings)) {
					word := command[ws:we]
					if cands, ok := absoluteWordExpansions(word, bindings); ok {
						for _, c := range cands {
							if c == "" || !isAbsResolved(c) {
								continue
							}
							if _, dup := seen[c]; dup {
								continue
							}
							seen[c] = struct{}{}
							out = append(out, c)
						}
						continue
					}
				}
			}
		}
		for _, resolved := range resolveShellToken(tok, shell, workDir, bindings) {
			if resolved == "" || !isAbsResolved(resolved) {
				continue
			}
			if _, dup := seen[resolved]; dup {
				continue
			}
			seen[resolved] = struct{}{}
			out = append(out, resolved)
		}
	}
	return out
}

// pathComponentChars lists the characters that may occur inside a filesystem
// path component (filename). pathRegex's absolute-path alternative can match a
// "/" that follows such a character — the separator inside a relative path —
// so a preceding path-component character marks a "/" as part of a relative
// path rather than the start of an absolute one.
const pathComponentChars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-~"

// isPathComponentChar reports whether b may occur inside a filesystem path
// component (filename), i.e. it is one of [pathComponentChars].
func isPathComponentChar(b byte) bool {
	return strings.IndexByte(pathComponentChars, b) >= 0
}

// isPureSeparatorRunToken reports whether tok consists entirely of separator
// characters: a POSIX run of two or more slashes ("//", "///", ...) or, after
// a two-character drive prefix, a run of two or more separators with no path
// component ("C:\\"). Such tokens are artifacts of the shell language, not
// filesystem paths — the trailing "//" of a sed address ("sed 's/.*function
// //'"), a comment marker ("echo \"// TODO fix\" >> notes.md"), an
// integer-division operator ("echo $(( total // count ))") or an escaped
// PowerShell drive root ("C:\\"). They carry no path component and therefore
// name no out-of-root location; resolving them anyway (a bare "//" cleans to
// the filesystem root "/") only produced false-positive escalations.
//
// Detection is not weakened by the skip: a bare "/" never matches [pathRegex]
// (it requires at least one character after the leading separator), so the
// POSIX form only skips runs of TWO or more slashes, and "cat //etc/passwd" —
// whose token still carries the "etc/passwd" components — keeps being
// reported. On the drive form a single trailing separator is a drive root
// ("C:\") and a token with a component ("C:\\Windows") is a real path; both
// remain tokens — only the pure separator run of two or more ("C:\\") is
// skipped.
func isPureSeparatorRunToken(tok string) bool {
	// Drive form: two-character drive prefix ("X:"), then only separators.
	if len(tok) > 2 && tok[1] == ':' {
		rest := tok[2:]
		return len(rest) >= 2 && strings.Trim(rest, "/\\") == ""
	}
	// POSIX form: a run of two or more slashes and nothing else.
	return len(tok) >= 2 && strings.TrimLeft(tok, "/") == ""
}

// relativePathWordStart finds the byte offset at which the relative-path word
// containing position start begins. It scans backwards over characters that may
// form a relative path component (letters, digits and the separator/idiom
// characters '.', '_', '-', '~', '/', '\') until it reaches a character that
// cannot be part of a path word (whitespace, a shell operator, a quote, etc.).
// The result is then trimmed back to the first non-separator character so the
// returned offset points at the start of the word (e.g. for "a/../b" with
// start at the "/b", the result is the offset of "a").
//
// This is used to resolve a "/"-leading fragment that pathRegex matches inside
// a relative path (e.g. the "/../etc/passwd" in "./../etc/passwd") by resolving
// the entire relative word against workDir rather than treating the fragment as
// an absolute path.
func relativePathWordStart(command string, start int) int {
	const relativeWordChars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-~/\\"
	i := start
	for i > 0 && strings.IndexByte(relativeWordChars, command[i-1]) >= 0 {
		i--
	}
	// Trim a leading separator run ("../", "./", "/" or "\") back to the first
	// component character so the word starts at an actual name/idiom character.
	for i < start && (command[i] == '/' || command[i] == '\\') {
		i++
	}
	return i
}

// PathsOutsideRoots resolves the shell tokens in command and reports those
// resolved absolute paths that fall outside every session root. The session
// roots are the canonical union of workspace, temp directory and additional
// allowed roots returned by [SessionRoots], and containment is tested with the
// symlink-aware, case-sensitivity-aware [IsWithinRoot].
//
// When workDir is empty it defaults to the workspace path from context
// ([WorkspacePathFrom]), mirroring the Execute fallback (cmd.Dir = workspace).
// This keeps the Judge and Execute in agreement about the base directory for
// relative ".." tokens: without it the Judge would skip relative tokens
// (seeing workDir="") while Execute runs them against the workspace, so a
// command like "cat ../../etc/passwd" with no working_directory would bypass
// containment in the Judge.
//
// When [SessionRoots] returns an empty list the function returns nil: with no
// roots configured containment cannot be enforced, mirroring
// tools/builtins/workdir.go's no-roots contract.
//
// Harmless special-device paths (/dev/null, /dev/full; NUL on Windows) are
// excluded via [IsHarmlessDevicePath] so shell idioms like `cmd > /dev/null`
// do not escalate to user confirmation when all other referenced paths are
// in-root.
func PathsOutsideRoots(ctx context.Context, command string, shell ShellKind, workDir string) []string {
	roots := SessionRoots(ctx)
	if len(roots) == 0 {
		return nil
	}

	if workDir == "" {
		workDir = WorkspacePathFrom(ctx)
	}

	var outside []string
	for _, p := range ResolveShellPathTokens(command, shell, workDir) {
		if IsHarmlessDevicePath(p) {
			continue
		}
		inside := false
		for _, root := range roots {
			if IsWithinRoot(ctx, root, p) {
				inside = true
				break
			}
		}
		if !inside {
			outside = append(outside, p)
		}
	}
	return outside
}

// ExistingPaths returns the subset of paths that exist on the host filesystem.
//
// This is the pure on-disk existence primitive. The shell-exec Judges
// (bash_exec, posh_exec) use the anchored variant [ExistingOrAnchoredPaths]
// instead, which also retains write/create targets whose nearest existing
// ancestor directory exists — so that creating a new file in an existing
// out-of-root directory still escalates to confirmation. ExistingPaths is kept
// for callers that need strict on-disk existence only.
//
// Fail-safe: a path whose existence cannot be determined for a reason other
// than "does not exist" (e.g. a permission error on a parent directory) is
// kept — the entry may exist, so it remains subject to the check rather than
// being silently let through.
func ExistingPaths(paths []string) []string {
	out := make([]string, 0, len(paths))
	for _, p := range paths {
		_, err := os.Stat(p)
		if err == nil || !os.IsNotExist(err) {
			out = append(out, p)
		}
	}
	return out
}

// ExistingOrAnchoredPaths returns the subset of paths that are anchored to a
// real on-disk location: the path itself exists, or it is a write/create
// target whose nearest existing ancestor directory exists (below the volume
// root). The shell-exec Judges (bash_exec, posh_exec) use it to decide whether
// a command referencing an out-of-root path should escalate to user
// confirmation.
//
// Unlike [ExistingPaths], which only keeps paths that already exist, this also
// keeps paths that live under an existing out-of-root directory — the
// security-sensitive case of a write that CREATES a new file in a real system
// directory (e.g. "/etc/cron.d/newjob", "~/.ssh/newkey"). Dropping such
// targets would let a prompt-injection-driven write bypass the confirmation
// gate under auto-approval, so they are retained. This is consistent with the
// file-tool Judges, which consult [AllPathsInSessionRoots] for pure
// containment with no existence filter: bash must not become a weaker route
// to an out-of-root write than write_file.
//
// A wholly non-existent subtree — a path whose only existing ancestor is the
// filesystem/volume root itself (e.g. a fabricated token "/zzz/qqq" where
// "/zzz" does not exist) — is dropped: no real directory anchors it, so a
// write there cannot succeed without a prior mkdir (which the Judge assesses
// as a separate command). This keeps the false-positive reduction for
// path-like tokens that do not correspond to any real location.
//
// Fail-safe: a path whose existence, or whose ancestor's existence, cannot be
// determined for a reason other than "does not exist" (e.g. a permission error
// on a parent directory) is kept — it may be real, so it remains subject to
// the check rather than being silently let through.
func ExistingOrAnchoredPaths(paths []string) []string {
	out := make([]string, 0, len(paths))
	for _, p := range paths {
		if pathHasExistingAnchor(p) {
			out = append(out, p)
		}
	}
	return out
}

// pathHasExistingAnchor reports whether p is anchored to a real on-disk
// location: p itself exists, or its nearest existing ancestor directory exists
// below the volume root. See [ExistingOrAnchoredPaths] for the policy rationale.
func pathHasExistingAnchor(p string) bool {
	_, err := os.Stat(p)
	if err == nil {
		return true // p exists on disk.
	}
	if !os.IsNotExist(err) {
		return true // fail-safe: undetermined existence → keep.
	}
	// p does not exist; keep it as a write target when its nearest existing
	// ancestor directory exists (a real directory would receive the write).
	_, ok := nearestExistingAncestorDir(p)
	return ok
}

// nearestExistingAncestorDir returns the nearest existing ancestor directory of
// p, walking up from filepath.Dir(p). It does NOT accept the filesystem/volume
// root as an anchor: a path whose only existing ancestor is the root points
// into a wholly non-existent subtree and returns ok=false. The walk is
// fail-safe — an ancestor whose existence cannot be determined (an error other
// than "does not exist") is treated as existing.
func nearestExistingAncestorDir(p string) (string, bool) {
	dir := filepath.Dir(filepath.Clean(p))
	for {
		// The volume/filesystem root is not a meaningful anchor: filepath.Dir
		// of a root returns the root itself, so this identifies it on every
		// platform ("/" on POSIX, "C:\" on Windows).
		if filepath.Dir(dir) == dir {
			return "", false
		}
		if _, err := os.Stat(dir); err == nil {
			return dir, true
		} else if !os.IsNotExist(err) {
			return dir, true // fail-safe: undetermined → assume existing.
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "", false
		}
		dir = parent
	}
}

// resolveShellToken dispatches a single matched token to the appropriate
// resolver based on its leading character(s). It returns the resolved
// absolute-path candidates; an empty slice means the token cannot be resolved
// and the caller drops it. Env tokens may yield several candidates — see
// [resolveEnvToken].
func resolveShellToken(token string, shell ShellKind, workDir string, bindings *envBindings) []string {
	switch {
	case token == "":
		return nil
	case isURLSchemeFragment(token):
		// A single letter + "://" is the drive-letter tail of a URL scheme
		// (e.g. the "s://" extracted from "https://") caught by pathRegex's
		// [A-Za-z]:[\\/] alternative. It is not a filesystem path and must
		// be skipped to avoid false positives — notably on Windows, where
		// "s://host/path" looks like a drive path.
		return nil
	case strings.HasPrefix(token, "~"):
		resolved, ok := resolveTildeToken(token)
		if !ok {
			return nil
		}
		return []string{resolved}
	case strings.HasPrefix(token, "$"):
		return resolveEnvToken(token, shell, bindings)
	case strings.HasPrefix(token, ".."):
		resolved, ok := resolveRelativeToken(token, workDir)
		if !ok {
			return nil
		}
		return []string{resolved}
	default:
		// Anything else matched by the combined regex is an absolute path
		// coming from [pathRegex]: a POSIX root ("/...") or a Windows drive
		// ("C:\..."). A POSIX-rooted path is absolute by definition on every
		// host, but filepath.IsAbs returns false for it on Windows and
		// filepath.Clean rewrites its separators to backslashes — so clean it
		// with the path package to keep it forward-slashed and POSIX-absolute
		// regardless of host OS. Windows drive paths use filepath as usual.
		if strings.HasPrefix(token, "/") {
			return []string{path.Clean(token)}
		}
		return []string{filepath.Clean(token)}
	}
}

// urlSchemeFragmentRe matches the drive-letter tail of a URL scheme: a single
// letter followed by "://". pathRegex's [A-Za-z]:[\\/] alternative captures
// this from schemes like "https://", "ftp://", "git+ssh://", etc. A genuine
// Windows drive path uses a single separator ("C:\" or "C:/"), never a double
// slash, so this check does not reject real paths.
var urlSchemeFragmentRe = regexp.MustCompile(`^[A-Za-z]://`)

// isURLSchemeFragment reports whether token is the drive-letter tail of a URL
// scheme (e.g. "s://github.com/..." extracted from "https://github.com/...").
func isURLSchemeFragment(token string) bool {
	return urlSchemeFragmentRe.MatchString(token)
}

// isAbsResolved reports whether resolved is an absolute path on any host. It
// treats a POSIX root ("/...") — which a shell command can carry regardless of
// the host OS — as absolute on every platform, in addition to what the host's
// [filepath.IsAbs] considers absolute (a Windows drive "C:\..." on Windows).
//
// On Windows, filepath.IsAbs("/etc/passwd") returns false and the resolved
// POSIX path would be dropped from the output; this helper keeps it, so a
// command like "cat /etc/passwd" is still flagged as referencing an out-of-root
// path on a Windows CI host. Windows drive paths remain absolute everywhere
// because they are detected by prefix, not by filepath.IsAbs.
func isAbsResolved(resolved string) bool {
	return strings.HasPrefix(resolved, "/") || filepath.IsAbs(resolved)
}

// resolveTildeToken expands "~" (and "~<sep>...") to the home directory.
// "~user" forms are best-effort and skipped: resolving another user's home
// requires a user-database lookup that is intentionally out of scope here.
func resolveTildeToken(token string) (string, bool) {
	home := userHomeDir()
	if home == "" {
		return "", false
	}
	rest := token[1:] // strip leading "~"
	// "~" or "~/..." / "~\..." → home + remainder.
	if rest == "" || rest[0] == '/' || rest[0] == '\\' {
		rest = normalizeSeparators(rest)
		return filepath.Clean(filepath.Join(home, rest)), true
	}
	// "~user..." — cannot resolve without a user lookup; skip (best-effort).
	return "", false
}

// inSingleQuotes reports whether position pos in command sits inside a
// single-quoted span, tracking quoting as the shell lexes it: a single
// quote toggles the span only OUTSIDE double quotes (inside double quotes
// an apostrophe is a literal character), a double quote toggles its context
// only outside single quotes, and a backslash escapes the next character
// only outside single quotes (inside '...' backslashes are literal).
// Single-quoted content is literal — no expansion, no composition — so
// positional and word-assembly rules must not fire inside it (awk/sed
// scripts: "awk '{print $1}'"), while an apostrophe inside double quotes
// ("it's") must not flip that decision.
func inSingleQuotes(command string, pos int) bool {
	single, double := false, false
	for i := 0; i < pos; i++ {
		switch command[i] {
		case '\\':
			if !single {
				i++ // skip the escaped character (literal inside '...')
			}
		case '\'':
			if !double {
				single = !single
			}
		case '"':
			if !single {
				double = !double
			}
		}
	}
	return single
}

// wordStartsAbsolute reports whether a word's first non-quoting character is
// the POSIX root separator ('"/et$B/passwd"' is an absolute-shaped word).
func wordStartsAbsolute(word string) bool {
	return strings.HasPrefix(strings.TrimLeft(word, `"'`), "/")
}

// shellWordBounds returns the [start, end) bounds of the shell word
// enclosing the token at command[tokStart:tokEnd]: the run of word-ish
// characters around it — path components, separators, quotes, and
// reference sigils — bounded by whitespace or any other shell
// metacharacter. A shell word continues through embedded quotes
// ("$D"c/passwd) and adjacent references ("$P$Q", "${D}${E}").
func shellWordBounds(command string, tokStart, tokEnd int) (wordStart, wordEnd int) {
	isWordChar := func(c byte) bool {
		return isPathComponentChar(c) || c == '/' || c == '"' || c == '\'' ||
			c == '$' || c == '{' || c == '}'
	}
	start := tokStart
	for start > 0 && isWordChar(command[start-1]) {
		start--
	}
	end := tokEnd
	for end < len(command) && isWordChar(command[end]) {
		end++
	}
	return start, end
}

// bashPlainWordVarRe matches a plain "$NAME"/"${NAME}"/"$=NAME" reference
// with NO suffix, for splitting a word into literal and variable segments.
var bashPlainWordVarRe = regexp.MustCompile(`\$[=~]?\{?[A-Za-z_][A-Za-z0-9_]*\}?`)

// absoluteWordExpansions expands an absolute-shaped shell word (one starting
// with "/") across the candidate product of its plain variable references:
// every literal fragment concatenates with every candidate value of every
// referenced name, so "/et$B/passwd" with B="c" surfaces "/etc/passwd" and
// "/$P$Q/passwd" surfaces the pairwise joins. assessable reports whether
// the word could be composed at all: an operator-form braced reference
// ("${X:1}") or a positional ("$1") derives a value no candidate list can
// express, so the caller must escalate the word instead of resolving it.
// The product is capped; an over-cap word is UNASSESSABLE (assessable=false)
// so the caller escalates it hard — a truncated emission could silently
// omit the one true runtime combination (fail-open).
// quoteSep temporarily replaces embedded quoting inside an assembled word
// ("$D"c/passwd): the quote is a zero-width word separator at expansion
// time — segments around it join without it, but a reference name must not
// bleed across it.
const quoteSep = "\x00"

var wordQuoteReplacer = strings.NewReplacer(`"`, quoteSep, "'", quoteSep)

func absoluteWordExpansions(word string, bindings *envBindings) (cands []string, assessable bool) {
	// Embedded quotes are word separators, not content.
	word = wordQuoteReplacer.Replace(word)
	for _, m := range bashBracedVarRe.FindAllString(word, -1) {
		inner := m[2 : len(m)-1]
		if strings.HasPrefix(inner, "!") {
			continue // indirect forms are escalated by their own pass
		}
		name, rest := splitEnvName(inner)
		if name == "" || rest != "" {
			return nil, false // an operator form — cannot be composed
		}
	}
	if bashPositionalVarRe.MatchString(word) {
		return nil, false // positional references — never statically assessable
	}
	parts := bashPlainWordVarRe.Split(word, -1)
	matches := bashPlainWordVarRe.FindAllString(word, -1)
	// parts[0] is the literal prefix before the first reference ("/et" in
	// "/et$B/passwd") — the word is absolute-shaped, so it starts with "/".
	products := []string{parts[0]}
	const maxProducts = 32
	for i, name := range matches {
		body := strings.TrimSuffix(strings.TrimPrefix(strings.TrimPrefix(name, "$"), "{"), "}")
		varName := strings.TrimLeft(body, "=~")
		vals := bindings.valueCandidates(varName)
		if len(vals) == 0 {
			vals = []string{""}
		}
		var next []string
		for _, p := range products {
			for _, v := range vals {
				if len(next) >= maxProducts {
					// The full product does not fit: the true runtime
					// combination may be the one dropped — fail closed.
					return nil, false
				}
				next = append(next, p+v)
			}
		}
		products = next
		if i+1 < len(parts) {
			var withSuffix []string
			for _, p := range products {
				if len(withSuffix) >= maxProducts {
					return nil, false
				}
				withSuffix = append(withSuffix, p+parts[i+1])
			}
			products = withSuffix
		}
	}
	out := make([]string, 0, len(products))
	for _, p := range products {
		out = append(out, cleanJoined(strings.ReplaceAll(p, quoteSep, "")))
	}
	return out, true
}

// wordHasAbsoluteCandidate reports whether any plain variable reference in
// the word has a "/"-prefixed resolution candidate — the word's runtime
// composition can therefore land on an absolute path even though the word
// itself does not start with "/" ("$P$Q/passwd" with P="/e").
func wordHasAbsoluteCandidate(word string, bindings *envBindings) bool {
	for _, name := range bashPlainWordVarRe.FindAllString(word, -1) {
		body := strings.TrimSuffix(strings.TrimPrefix(strings.TrimPrefix(name, "$"), "{"), "}")
		varName := strings.TrimLeft(body, "=~")
		for _, v := range bindings.valueCandidates(varName) {
			if strings.HasPrefix(v, "/") {
				return true
			}
		}
	}
	return false
}

// resolveEnvToken expands a "$VAR"/"${VAR}" (bash) or
// "$env:VAR"/"${env:VAR}" (posh) reference plus an optional path remainder,
// returning every absolute-path candidate.
//
// In-command literal bindings never shadow the process environment: the
// binding pre-pass is position- and control-flow-unaware (an assignment inside
// a branch not taken still lands in the summary), so a name with BOTH a
// binding and a non-empty env value is ambiguous and BOTH expansions are
// returned — fail-closed over-report. A name assigned more than once (or
// dynamically) likewise contributes ALL of its distinct literal values as
// candidates (see [envBindings.valueCandidates]); a binding can add
// candidates, it can never mask an environment value or an earlier binding.
// The EMPTY expansion is a candidate of equal standing: whenever the variable
// may be unset, empty or cleared at the reference (see
// [envBindings.emptyExpansionPossible]), the token also resolves its path
// suffix alone — "$D/etc/passwd" with a possibly-unset D (a decoy in-command
// binding included) surfaces "/etc/passwd" exactly like the historical
// unbound-variable fallback. A candidate whose expansion is bare and empty
// names no path and is skipped.
//
// Note: an env var that resolves to an absolute path outside the session roots
// (e.g. "$HOME", "$USERPROFILE") is reported by PathsOutsideRoots. This is
// intentional fail-safe behavior — such a token may dereference an out-of-root
// path — and the user-confirmation policy gates the command. The cost is a
// false positive on benign commands like "echo $HOME"; the conservative
// stance is preferable to silently allowing an out-of-root dereference.
func resolveEnvToken(token string, shell ShellKind, bindings *envBindings) []string {
	var name, rest string
	switch {
	case strings.HasPrefix(token, "${env:"):
		name, rest = splitBrace(token, "${env:")
	case strings.HasPrefix(token, "${"):
		name, rest = splitBrace(token, "${")
	case strings.HasPrefix(token, "$env:"):
		name, rest = splitEnvPrefix(token, "$env:")
	case strings.HasPrefix(token, "$"):
		name, rest = splitEnvName(token[1:])
	default:
		return nil
	}
	if name == "" {
		return nil
	}
	// The per-shell combined regex already guarantees that bash tokens are the
	// "$VAR"/"${VAR}" forms and posh tokens are the "$env:"/"${env:" forms, so
	// no cross-dialect mismatch can reach here.
	rest = normalizeSeparators(rest)
	vals := bindings.valueCandidates(name)
	out := make([]string, 0, len(vals))
	for _, val := range vals {
		// The empty expansion concatenated with a literal suffix resolves in
		// the shell to just the suffix (e.g. "$UNSET/etc/passwd" →
		// "/etc/passwd"), so a prompt-injection command that hides an absolute
		// path behind a possibly-unset env var still surfaces as an
		// out-of-root reference. A bare empty expansion (no suffix) names no
		// path and remains skipped.
		if val == "" {
			if rest != "" {
				out = append(out, cleanJoined(rest))
			}
			continue
		}
		if rest != "" {
			out = append(out, cleanJoined(val, rest))
			continue
		}
		out = append(out, cleanJoined(val))
	}
	return out
}

// splitBrace extracts the variable name inside "${...}" / "${env:...}" and the
// remainder after the closing brace. prefix is the literal opener already
// stripped from token's start, including for the env form.
func splitBrace(token, prefix string) (name, rest string) {
	open := len(prefix)
	end := strings.Index(token[open:], "}")
	if end < 0 {
		return "", ""
	}
	name = token[open : open+end]
	rest = token[open+end+1:]
	return name, rest
}

// splitEnvPrefix handles the "$env:VAR<rest>" form: the name is the run of
// [A-Za-z0-9_] after the prefix, and the remainder is whatever follows.
func splitEnvPrefix(token, prefix string) (name, rest string) {
	return splitEnvName(token[len(prefix):])
}

// splitEnvName splits body into a leading [A-Za-z_][A-Za-z0-9_]* name and the
// remainder. Returns ("", body) when body does not start with a name char.
func splitEnvName(body string) (name, rest string) {
	i := 0
	for i < len(body) {
		c := body[i]
		isName := (c >= 'a' && c <= 'z') ||
			(c >= 'A' && c <= 'Z') ||
			(c >= '0' && c <= '9') ||
			c == '_'
		if !isName {
			break
		}
		i++
	}
	return body[:i], body[i:]
}

// resolveRelativeToken resolves a ".."-bearing token against workDir. A plain
// relative name never reaches here (only the relDotDotRe alternative produces
// tokens starting with ".."). When workDir is empty the token cannot be
// resolved and is skipped.
func resolveRelativeToken(token, workDir string) (string, bool) {
	if workDir == "" {
		return "", false
	}
	return filepath.Clean(filepath.Join(workDir, normalizeSeparators(token))), true
}

// normalizeSeparators converts backslashes to forward slashes so resolution is
// uniform across platforms (a Windows-style remainder "$env:X\.ssh" resolves
// the same on a POSIX test host as on Windows). [filepath.Join] re-applies the
// platform separator on the final, cleaned path.
func normalizeSeparators(s string) string {
	return strings.ReplaceAll(s, "\\", "/")
}

// cleanJoined cleans the join of a resolved env-var value with an optional path
// remainder, preserving POSIX separators when the value is a POSIX-rooted
// absolute path. [filepath.Clean]/[filepath.Join] on Windows rewrite the
// leading "/" of a POSIX path to "\" (collapsing "/var/log/syslog" to a
// relative "var\log\syslog"), which would make a POSIX-absolute env path
// resolve as relative on a Windows host. POSIX-rooted values are therefore
// joined and cleaned with the path package; Windows-style (drive/UNC) and host
// paths use filepath as usual.
//
// Every part is normalized to forward slashes before joining. Joining with
// [filepath.Separator] ('\' on Windows) would mix a backslash into a
// POSIX-rooted value — and [path.Clean] (used for POSIX roots) treats '\' as a
// literal character, not a separator, so "/opt/x" + '\' + "/y/.." would clean
// to "/opt/x\y" rather than "/opt/y". Normalizing first keeps a single,
// consistent separator so the POSIX branch's path.Clean behaves correctly on
// every host.
func cleanJoined(parts ...string) string {
	normalized := make([]string, len(parts))
	for i, p := range parts {
		normalized[i] = normalizeSeparators(p)
	}
	joined := strings.Join(normalized, "/")
	if strings.HasPrefix(joined, "/") {
		return path.Clean(joined)
	}
	return filepath.Clean(joined)
}

// userHomeDir returns the current user's home directory, preferring $HOME
// (the override the bash/posh "tilde expansion tests rely on via t.Setenv and
// the variable os.UserHomeDir consults last on Windows), then $USERPROFILE,
// then [os.UserHomeDir]. On Windows os.UserHomeDir resolves via USERPROFILE
// and does not consult $HOME, so a test that t.Setenv("HOME", fakeHome) to
// pin tilde expansion would be silently bypassed without this precedence.
func userHomeDir() string {
	if h := os.Getenv("HOME"); h != "" {
		return h
	}
	if h := os.Getenv("USERPROFILE"); h != "" {
		return h
	}
	if h, err := os.UserHomeDir(); err == nil && h != "" {
		return h
	}
	return ""
}
