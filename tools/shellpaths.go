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
//   - "~" and "~user" (bash) and bare "~" (posh) expand to the current user's
//     home directory via [os.UserHomeDir] (falling back to $HOME then
//     $USERPROFILE). "~user" is best-effort and skipped when unresolvable.
//   - "$VAR"/"${VAR}" (bash) and "$env:VAR"/"${env:VAR}" (posh) expand via
//     [os.Getenv]; tokens whose expansion is empty are SKIPPED.
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
	for _, m := range re.FindAllStringIndex(command, -1) {
		start, end := m[0], m[1]
		tok := command[start:end]
		// Boundary checks for tokens whose leading character could continue a
		// preceding relative-path word (".." parent-refs and POSIX "/abs"
		// paths). Env ("$VAR") and tilde ("~") tokens start with non-path
		// characters and are inherently bounded, so they are exempt.
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
		}
		resolved, ok := resolveShellToken(tok, shell, workDir)
		if !ok || resolved == "" || !isAbsResolved(resolved) {
			continue
		}
		if _, dup := seen[resolved]; dup {
			continue
		}
		seen[resolved] = struct{}{}
		out = append(out, resolved)
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

// resolveShellToken dispatches a single matched token to the appropriate
// resolver based on its leading character(s). It returns (resolved, ok); ok is
// false when the token cannot be resolved (the caller drops it).
func resolveShellToken(token string, shell ShellKind, workDir string) (string, bool) {
	switch {
	case token == "":
		return "", false
	case isURLSchemeFragment(token):
		// A single letter + "://" is the drive-letter tail of a URL scheme
		// (e.g. the "s://" extracted from "https://") caught by pathRegex's
		// [A-Za-z]:[\\/] alternative. It is not a filesystem path and must
		// be skipped to avoid false positives — notably on Windows, where
		// "s://host/path" looks like a drive path.
		return "", false
	case strings.HasPrefix(token, "~"):
		return resolveTildeToken(token)
	case strings.HasPrefix(token, "$"):
		return resolveEnvToken(token, shell)
	case strings.HasPrefix(token, ".."):
		return resolveRelativeToken(token, workDir)
	default:
		// Anything else matched by the combined regex is an absolute path
		// coming from [pathRegex]: a POSIX root ("/...") or a Windows drive
		// ("C:\..."). A POSIX-rooted path is absolute by definition on every
		// host, but filepath.IsAbs returns false for it on Windows and
		// filepath.Clean rewrites its separators to backslashes — so clean it
		// with the path package to keep it forward-slashed and POSIX-absolute
		// regardless of host OS. Windows drive paths use filepath as usual.
		if strings.HasPrefix(token, "/") {
			return path.Clean(token), true
		}
		return filepath.Clean(token), true
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

// resolveEnvToken expands a "$VAR"/"${VAR}" (bash) or
// "$env:VAR"/"${env:VAR}" (posh) reference plus an optional path remainder.
// Tokens whose expansion is empty are skipped.
//
// Note: an env var that resolves to an absolute path outside the session roots
// (e.g. "$HOME", "$USERPROFILE") is reported by PathsOutsideRoots. This is
// intentional fail-safe behavior — such a token may dereference an out-of-root
// path — and the user-confirmation policy gates the command. The cost is a
// false positive on benign commands like "echo $HOME"; the conservative
// stance is preferable to silently allowing an out-of-root dereference.
func resolveEnvToken(token string, shell ShellKind) (string, bool) {
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
		return "", false
	}
	if name == "" {
		return "", false
	}
	// The per-shell combined regex already guarantees that bash tokens are the
	// "$VAR"/"${VAR}" forms and posh tokens are the "$env:"/"${env:" forms, so
	// no cross-dialect mismatch can reach here.
	val := os.Getenv(name)
	if val == "" {
		return "", false
	}
	rest = normalizeSeparators(rest)
	if rest != "" {
		return cleanJoined(val, rest), true
	}
	return cleanJoined(val), true
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
