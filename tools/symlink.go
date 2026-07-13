package tools

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"syscall"

	"mvdan.cc/sh/v3/syntax"

	"github.com/v0lka/sp4rk/pathutil"
)

// SymlinkTraversal describes a path that traverses a symlink.
type SymlinkTraversal struct {
	OriginalPath     string // user-visible path from tool input
	SymlinkAt        string // component where the symlink was detected
	ResolvesTo       string // what the symlink points to (readlink result)
	FullResolved     string // fully resolved absolute path after symlink chain
	OutsideWorkspace bool   // does the fully resolved path fall outside the workspace?
	Unresolvable     bool   // component could not be inspected (Lstat/Readlink failure) — escalate
}

// DetectSymlinksInToolInput extracts all path-like values from a tool input
// and checks each for symlinks. Returns traversals partitioned by whether
// the resolved target is inside or outside the workspace, plus a suspicious
// flag for bash_exec commands with unexpandable tokens.
//
// schema is the tool's JSON input schema, used to identify which properties
// carry filesystem paths. When path fields are recognizable, only those fields
// are scanned for symlinks, so content payloads (edit_file old_string/new_string,
// write_file content) are never mistaken for paths. When no schema is available
// or no path-like field is recognized, detection falls back to scanning all
// strings to preserve coverage for unconventional tools.
//
// Path fields are recognized by naming convention (see pathFieldExactNames and
// pathFieldSuffixes): exact names such as path, file, dir, cwd, root,
// working_directory, dest; or suffixes _path, _dir, _file, _filepath, _root.
// When a schema declares a recognized path field alongside other string-typed
// fields whose names do NOT follow the convention, those non-path fields are
// excluded from scanning and the omission is logged (slog.Default, Warn level)
// so it is observable rather than silent. To ensure a path-carrying parameter
// is scanned, name it with one of the recognized names or suffixes.
func DetectSymlinksInToolInput(ctx context.Context, toolName string, input, schema json.RawMessage) (
	inside []SymlinkTraversal,
	outside []SymlinkTraversal,
	suspicious bool,
) {
	workspace := WorkspacePathFrom(ctx)
	roots := SessionRoots(ctx)

	if toolName == "bash_exec" {
		paths, unexpandable := extractBashPathsFromInput(input, workspace)
		inside, outside = checkPathsForSymlinks(paths, roots)
		return inside, outside, unexpandable
	}

	pathFields := pathFieldNamesFromSchema(schema)
	var paths []string
	if len(pathFields) > 0 {
		// Field-aware: scan only declared path fields. Content fields are
		// structurally excluded from symlink scanning.
		if others := unrecognizedStringFieldNames(schema, pathFields); len(others) > 0 {
			// The allowlist narrows scanning to recognized path-field names.
			// String-typed fields that are NOT recognized (e.g. an MCP tool
			// parameter named "target" or "source") may carry paths under a
			// non-conventional name and will NOT be scanned. Log the omission
			// so it is observable rather than silent; detection does not
			// escalate on those fields.
			slog.Warn("symlink detection narrowed by path-field allowlist; non-path string fields not scanned",
				"tool", toolName,
				"scanned_path_fields", sortedFieldNames(pathFields),
				"unscanned_string_fields", others)
		}
		paths = extractPathsFromFields(input, pathFields, workspace)
	} else {
		// No schema or no recognizable path field — fall back to scanning all
		// strings so detection still works for tools whose schema does not
		// follow the path-naming convention.
		paths = extractAllPathsFromJSON(input, workspace)
	}
	inside, outside = checkPathsForSymlinks(paths, roots)
	return inside, outside, false
}

// extractAllPathsFromJSON extracts all path-like strings from a JSON tool input.
// Each string value containing "/" is treated as a candidate path.
// Relative paths are resolved against workspace.
func extractAllPathsFromJSON(input json.RawMessage, workspace string) []string {
	var parsed any
	if err := json.Unmarshal(input, &parsed); err != nil {
		return nil
	}

	strValues := ExtractJSONStrings(parsed)
	seen := make(map[string]struct{})
	var paths []string

	for _, s := range strValues {
		if !looksLikePath(s) {
			continue
		}
		resolved := resolvePathCandidate(s, workspace)
		if resolved == "" {
			continue
		}
		cleaned := filepath.Clean(resolved)
		if _, ok := seen[cleaned]; ok {
			continue
		}
		seen[cleaned] = struct{}{}
		paths = append(paths, cleaned)
	}

	return paths
}

// pathFieldNamesFromSchema inspects a tool's JSON Schema and returns the set
// of property names that carry filesystem paths (as opposed to content,
// patterns, or other string payloads). Scanning is then confined to these
// fields, so content fields (edit_file old_string/new_string, write_file
// content, ...) are never mistaken for paths.
//
// A property is a path field when it is string-typed (or has no explicit type)
// and its lowercased name exactly matches a known name (path, file, dir,
// directory, filepath, filename, cwd, root, working_directory, workdir, dest,
// destination) or ends with one of (_path, _dir, _directory, _file, _filepath,
// _root). This deliberately excludes content/pattern fields like content,
// old_string, new_string, command, pattern, file_pattern.
func pathFieldNamesFromSchema(schema json.RawMessage) map[string]bool {
	fields := make(map[string]bool)
	if len(schema) == 0 {
		return fields
	}
	var s struct {
		Properties map[string]json.RawMessage `json:"properties"`
	}
	if err := json.Unmarshal(schema, &s); err != nil {
		return fields
	}
	for name, raw := range s.Properties {
		if isPathFieldProperty(name, raw) {
			fields[name] = true
		}
	}
	return fields
}

// isStringTypedProperty reports whether a schema property is string-typed, or
// has no explicit type (treated as string for path-field purposes since some
// schemas omit "type"). looksLikePath still filters non-path values at
// extraction time.
func isStringTypedProperty(raw json.RawMessage) bool {
	var prop struct {
		Type string `json:"type"`
	}
	_ = json.Unmarshal(raw, &prop)
	return prop.Type == "" || prop.Type == "string"
}

// isPathFieldProperty reports whether a schema property is a path field: a
// string-typed (or untyped) property whose name follows path-naming
// conventions. Non-string types are never path fields.
func isPathFieldProperty(name string, raw json.RawMessage) bool {
	return isStringTypedProperty(raw) && isPathFieldName(name)
}

// pathFieldExactNames and pathFieldSuffixes define the naming convention for
// path-carrying properties. They are intentionally conservative to avoid
// matching content fields.
var (
	pathFieldExactNames = map[string]bool{
		"path": true, "file": true, "dir": true, "directory": true,
		"filepath": true, "filename": true, "cwd": true, "root": true,
		"working_directory": true, "workdir": true,
		"dest": true, "destination": true,
	}
	pathFieldSuffixes = []string{"_path", "_dir", "_directory", "_file", "_filepath", "_root"}
)

func isPathFieldName(name string) bool {
	lower := strings.ToLower(name)
	if pathFieldExactNames[lower] {
		return true
	}
	for _, suf := range pathFieldSuffixes {
		if strings.HasSuffix(lower, suf) {
			return true
		}
	}
	return false
}

// unrecognizedStringFieldNames returns the string-typed property names in the
// schema that are NOT recognized as path fields. When a schema declares path
// fields alongside such fields, the latter are excluded from symlink scanning
// (the allowlist confines scanning to recognized path-field names). Callers
// log the result so the omission is observable rather than silent. The result
// is sorted for deterministic logging.
func unrecognizedStringFieldNames(schema json.RawMessage, recognized map[string]bool) []string {
	if len(schema) == 0 {
		return nil
	}
	var s struct {
		Properties map[string]json.RawMessage `json:"properties"`
	}
	if err := json.Unmarshal(schema, &s); err != nil {
		return nil
	}
	var others []string
	for name, raw := range s.Properties {
		if recognized[name] {
			continue
		}
		if isStringTypedProperty(raw) && !isPathFieldName(name) {
			others = append(others, name)
		}
	}
	sort.Strings(others)
	return others
}

// sortedFieldNames returns the keys of m sorted, for deterministic log output.
func sortedFieldNames(m map[string]bool) []string {
	names := make([]string, 0, len(m))
	for name := range m {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// extractPathsFromFields extracts path-like strings only from the named fields
// of a JSON tool input. Unlike extractAllPathsFromJSON (which scans every
// string), this confines scanning to declared path fields so content payloads
// are never mistaken for paths. Each field's value is recursed (handling a
// string or an array of paths); relative paths are resolved against workspace.
func extractPathsFromFields(input json.RawMessage, fields map[string]bool, workspace string) []string {
	if len(fields) == 0 {
		return nil
	}
	var parsed map[string]any
	if err := json.Unmarshal(input, &parsed); err != nil {
		return nil
	}
	seen := make(map[string]struct{})
	var paths []string
	for name := range fields {
		val, ok := parsed[name]
		if !ok {
			continue
		}
		for _, s := range ExtractJSONStrings(val) {
			if !looksLikePath(s) {
				continue
			}
			resolved := resolvePathCandidate(s, workspace)
			if resolved == "" {
				continue
			}
			cleaned := filepath.Clean(resolved)
			if _, ok := seen[cleaned]; ok {
				continue
			}
			seen[cleaned] = struct{}{}
			paths = append(paths, cleaned)
		}
	}
	return paths
}

// looksLikePath returns true if a string resembles a filesystem path
// (contains a path separator and is not a URL). Recognizes both POSIX-style
// ("/") and Windows-style paths (drive letters like "C:\" or "D:/", and
// UNC paths starting with "\\\\").
//
// To avoid false positives, strings that are obviously content rather than
// paths are rejected: those containing control characters (newlines, tabs, or
// any byte below 0x20 / 0x7f) and those exceeding maxPathCandidateLen.
// Legitimate filesystem paths never contain control characters.
// maxPathCandidateLen is a conservative upper bound for a path candidate.
// Real filesystem paths are bounded by PATH_MAX (4096 on Linux, 1024 on
// macOS). Strings longer than this are almost certainly content (e.g. a code
// blob) mistakenly treated as a path, not a real path.
const maxPathCandidateLen = 4096

func looksLikePath(s string) bool {
	if s == "" || s == "." || s == ".." {
		return false
	}
	if len(s) > maxPathCandidateLen {
		return false
	}
	if hasControlChar(s) {
		// Newlines, tabs, and other control characters never appear in a real
		// filesystem path; their presence signals content (e.g. a code
		// snippet from edit_file old_string/new_string) rather than a path.
		return false
	}
	hasSeparator := strings.Contains(s, "/") ||
		strings.ContainsRune(s, filepath.Separator) ||
		looksLikeWindowsDriveLetter(s) ||
		strings.HasPrefix(s, `\\`)
	if !hasSeparator {
		return false
	}
	lower := strings.ToLower(s)
	for _, scheme := range []string{"http://", "https://", "ftp://", "ssh://", "file://", "git://"} {
		if strings.HasPrefix(lower, scheme) {
			return false
		}
	}
	return true
}

// hasControlChar reports whether s contains an ASCII control character
// (bytes < 0x20, including newline and tab, or 0x7f DEL). Legitimate
// filesystem paths never contain control characters. Iterating bytes is safe:
// UTF-8 continuation and leading bytes are all >= 0x80, so no multibyte rune
// is misclassified as a control character.
func hasControlChar(s string) bool {
	for i := 0; i < len(s); i++ {
		if c := s[i]; c < 0x20 || c == 0x7f {
			return true
		}
	}
	return false
}

// looksLikeWindowsDriveLetter reports whether s starts with a drive-letter
// prefix like "C:\" or "D:/".
func looksLikeWindowsDriveLetter(s string) bool {
	if len(s) < 3 {
		return false
	}
	c := s[0]
	isLetter := (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')
	if !isLetter {
		return false
	}
	if s[1] != ':' {
		return false
	}
	return s[2] == '\\' || s[2] == '/'
}

// resolvePathCandidate resolves a potential path to an absolute form.
// Absolute paths are returned cleaned. Relative paths are joined with workspace.
func resolvePathCandidate(s, workspace string) string {
	if s == "" {
		return ""
	}
	if filepath.IsAbs(s) {
		return s
	}
	if workspace == "" {
		return ""
	}
	return filepath.Join(workspace, s)
}

// extractBashPathsFromInput extracts paths from a bash_exec tool input.
// Returns resolved paths and a flag indicating whether the command contains
// unexpandable shell constructs ($var, $(cmd), `cmd`, process substitution).
func extractBashPathsFromInput(input json.RawMessage, workspace string) (paths []string, hasUnexpandable bool) {
	var params struct {
		Command          string `json:"command"`
		WorkingDirectory string `json:"working_directory"`
	}
	if err := json.Unmarshal(input, &params); err != nil || params.Command == "" {
		return nil, false
	}

	wd := params.WorkingDirectory
	if wd == "" {
		wd = workspace
	}

	return extractBashPaths(params.Command, wd, workspace)
}

// extractBashPaths parses a bash command using mvdan.cc/sh and extracts
// path-like literals. Working directory is used for relative path resolution.
func extractBashPaths(command, workingDirectory, workspace string) (paths []string, hasUnexpandable bool) {
	parser := syntax.NewParser()
	file, err := parser.Parse(strings.NewReader(command), "")
	if err != nil {
		return nil, true // unparseable — suspicious
	}

	seen := make(map[string]struct{})
	syntax.Walk(file, func(node syntax.Node) bool {
		switch n := node.(type) {
		case *syntax.CmdSubst:
			hasUnexpandable = true
			return false // don't recurse into $(...) or `...`
		case *syntax.ParamExp:
			hasUnexpandable = true
			return false
		case *syntax.ProcSubst:
			hasUnexpandable = true
			return false
		case *syntax.Word:
			// Check for unexpandable shell constructs within the word
			// (e.g., $HOME/file parses as one Word with Parts [ParamExp, Lit]).
			for _, part := range n.Parts {
				switch part.(type) {
				case *syntax.ParamExp, *syntax.CmdSubst, *syntax.ProcSubst:
					hasUnexpandable = true
				}
			}
			lit := wordLiteral(n)
			if lit == "" || !looksLikePath(lit) {
				return false
			}
			resolved := resolvePathCandidate(lit, workingDirectory)
			if resolved == "" {
				resolved = resolvePathCandidate(lit, workspace)
			}
			if resolved == "" {
				return false
			}
			cleaned := filepath.Clean(resolved)
			if _, ok := seen[cleaned]; !ok {
				seen[cleaned] = struct{}{}
				paths = append(paths, cleaned)
			}
			return false // already collected — don't recurse into Parts
		}
		return true
	})

	return paths, hasUnexpandable
}

// wordLiteral extracts the literal (unquoted) value from a shell Word node.
// Handles Lit, SglQuoted, and DblQuoted parts — but not variable expansions,
// command substitutions, or process substitutions.
// Escaped characters (e.g., "\ ") are preserved as-is in Lit values
// by the parser, so they appear as part of the extracted path.
func wordLiteral(w *syntax.Word) string {
	var parts []string
	for _, part := range w.Parts {
		switch p := part.(type) {
		case *syntax.Lit:
			parts = append(parts, p.Value)
		case *syntax.SglQuoted:
			parts = append(parts, p.Value)
		case *syntax.DblQuoted:
			for _, dp := range p.Parts {
				if lit, ok := dp.(*syntax.Lit); ok {
					parts = append(parts, lit.Value)
				}
			}
		}
	}
	return strings.Join(parts, "")
}

// checkPathsForSymlinks walks each path component-by-component looking for
// symlinks. Returns traversals partitioned by session-root containment: a
// traversal whose fully resolved target lies within ANY root is classified as
// 'inside'. OS-level symlinks (those above or at the workspace root, e.g.
// macOS /var → /private/var) are skipped — they are mapping infrastructure,
// not security-relevant traversals. roots is the union of workspace, temp dir
// and allowed roots (from tools.SessionRoots); the primary root (first element,
// the workspace) is passed to walkSymlinkComponents for OS-level symlink
// detection and fail-closed escalation scope.
func checkPathsForSymlinks(paths, roots []string) (inside, outside []SymlinkTraversal) {
	// The workspace (first session root, when present) is used for OS-level
	// symlink detection (IsOSLevelSymlink) and for the fail-closed escalation
	// scope in walkSymlinkComponents. Relative-path resolution is unaffected:
	// paths arriving here are already absolute.
	var workspace string
	if len(roots) > 0 {
		workspace = roots[0]
	}
	for _, p := range paths {
		for _, t := range walkSymlinkComponents(p, workspace) {
			if t.Unresolvable {
				// Cannot determine where the component resolves — fail closed
				// and escalate as an outside-workspace traversal.
				t.OutsideWorkspace = true
				outside = append(outside, t)
				continue
			}
			ok := false
			for _, root := range roots {
				if root == "" {
					continue
				}
				if within, _ := pathutil.IsWithinPath(root, t.FullResolved); within {
					ok = true
					break
				}
			}
			t.OutsideWorkspace = !ok
			if t.OutsideWorkspace {
				outside = append(outside, t)
			} else {
				inside = append(inside, t)
			}
		}
	}
	return inside, outside
}

// walkSymlinkComponents walks a path from root to leaf, checking each component
// for symlinks. Returns nil if no symlinks are found anywhere in the chain.
// All symlinked components of the original path are collected, not just the
// first one (os.Lstat resolves intermediate symlinks itself, so the walk
// continues naturally after a symlink is found). If the final component
// doesn't exist but a parent component is a symlink, the symlink is still
// detected.
//
// Fail-closed behavior: when a component cannot be inspected (Lstat
// permission error, ELOOP, or Readlink failure on a confirmed symlink), a
// traversal record with Unresolvable=true is returned so callers escalate to
// confirmation. To avoid over-prompting, Lstat failures are only escalated
// when the unreadable component lies within the workspace — unreadable
// paths outside the workspace are already gated by containment/judge checks.
//
// Invalid-path errors (ENAMETOOLONG, ENOTDIR, EINVAL) are NOT escalated: they
// mean the candidate string is not a valid filesystem path (e.g. a code blob
// longer than NAME_MAX that was mistakenly treated as a path), so there is
// definitely no symlink there. Escalating on them caused false prompts.
func walkSymlinkComponents(absPath, workspace string) []SymlinkTraversal {
	if absPath == "" {
		return nil
	}

	cleaned := filepath.Clean(absPath)
	parts := pathutil.SplitPathComponents(cleaned)
	if len(parts) == 0 {
		return nil
	}

	var traversals []SymlinkTraversal
	// Start reconstruction from the volume prefix so Windows drive-letter
	// paths (e.g. "C:") and UNC roots are rebuilt correctly. SplitPathComponents
	// already stripped the volume, so the first part is a real component.
	//
	// On Windows filepath.VolumeName returns "C:" — the *relative* current-dir-
	// on-C form, NOT the absolute "C:\". Without a separator, filepath.Join
	// would produce "C:Users" instead of "C:\Users". Append the separator to
	// make the drive path absolute. On Unix filepath.VolumeName returns "" and
	// we fall back to the root "/".
	vol := filepath.VolumeName(cleaned)
	var current string
	if vol == "" {
		current = string(filepath.Separator)
	} else {
		current = vol + string(filepath.Separator)
	}
	for i, part := range parts {
		if part == "" {
			continue
		}
		current = filepath.Join(current, part)

		fi, err := os.Lstat(current)
		if err != nil {
			if os.IsNotExist(err) {
				break // path doesn't exist — no further components to check
			}
			if isInvalidPathError(err) {
				// The component is not a valid filesystem path — e.g.
				// ENAMETOOLONG when a string mistaken for a path (a code
				// blob) was joined onto the workspace, or ENOTDIR/EINVAL for
				// a malformed component. Such a candidate cannot be a symlink,
				// so stop the walk without escalating. Escalating here caused
				// false "symlink escapes workspace" prompts on edit_file
				// inputs whose old_string/new_string contain "/".
				break
			}
			// Permission error (EACCES) or ELOOP — cannot determine whether
			// this component is a symlink. Escalate only when the component
			// lies within the workspace; unreadable paths outside the
			// workspace are gated by containment checks elsewhere.
			if workspace != "" {
				if within, werr := pathutil.IsWithinPath(workspace, current); werr == nil && within {
					traversals = append(traversals, SymlinkTraversal{
						OriginalPath: absPath,
						SymlinkAt:    current,
						Unresolvable: true,
					})
				}
			}
			break
		}

		if fi.Mode()&os.ModeSymlink == 0 {
			continue
		}

		// Skip OS-level infrastructure symlinks: a well-known OS symlink
		// (macOS /var → /private/var, Linux /bin → /usr/bin, Windows
		// junctions) or a symlink that is an ancestor of the workspace root
		// (the workspace itself is reached through it). These are mapping
		// infrastructure, not user-created security-relevant traversals.
		// IsOSLevelSymlink is the shared chokepoint; the well-known list lives
		// only in os_symlinks.go.
		if IsOSLevelSymlink(current, workspace) {
			continue
		}

		// Symlink found at current component
		target, err := os.Readlink(current)
		if err != nil {
			// Confirmed symlink whose target cannot be read — escalate.
			traversals = append(traversals, SymlinkTraversal{
				OriginalPath: absPath,
				SymlinkAt:    current,
				Unresolvable: true,
			})
			break
		}

		// Resolve the target relative to the symlink's directory
		resolvedTarget := target
		if !filepath.IsAbs(target) {
			resolvedTarget = filepath.Join(filepath.Dir(current), target)
		}

		// Build the full resolved path: resolvedTarget + remaining components
		tail := filepath.Join(parts[i+1:]...)
		fullResolved := filepath.Join(resolvedTarget, tail)

		// Resolve any remaining symlinks in the chain component-by-component.
		// Without this, chained symlinks (e.g., /ws/link1 → /real/link2 → /outside)
		// would only report the first symlink and miss subsequent ones that could
		// escape the workspace.
		if evaled, err := filepath.EvalSymlinks(fullResolved); err == nil {
			fullResolved = evaled
		}

		// Get absolute form
		fullAbs, err := filepath.Abs(fullResolved)
		if err != nil {
			fullAbs = filepath.Clean(fullResolved)
		}

		traversals = append(traversals, SymlinkTraversal{
			OriginalPath: absPath,
			SymlinkAt:    current,
			ResolvesTo:   target,
			FullResolved: fullAbs,
		})
		// Continue the walk: os.Lstat on deeper components follows this
		// (already recorded) symlink transparently, so subsequent symlinked
		// components of the original path are also collected.
	}

	return traversals
}

// isInvalidPathError reports whether err indicates the candidate is not a
// valid filesystem path, rather than a genuine unreadable symlink. These
// errors arise when a string mistakenly treated as a path violates filesystem
// naming limits — e.g. ENAMETOOLONG when a code snippet longer than NAME_MAX
// is joined onto the workspace, or ENOTDIR/EINVAL for a malformed component.
// Such a candidate cannot be a symlink, so callers stop the walk instead of
// escalating (which would otherwise produce a false confirmation prompt).
//
// Only these specific "the path itself is invalid" errors are recognized.
// Permission errors (EACCES) and symlink loops (ELOOP) still escalate.
func isInvalidPathError(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, syscall.ENAMETOOLONG) {
		return true
	}
	if errors.Is(err, syscall.ENOTDIR) {
		return true
	}
	if errors.Is(err, syscall.EINVAL) {
		return true
	}
	return false
}

// FormatSymlinkReasoning formats symlink traversals into a human-readable
// message for the confirmation dialog. Outside-workspace traversals are
// highlighted as more dangerous.
func FormatSymlinkReasoning(inside, outside []SymlinkTraversal, suspicious bool) string {
	var sb strings.Builder

	if len(outside) > 0 {
		sb.WriteString("This tool call traverses symlinks that resolve OUTSIDE the workspace:\n\n")
		for i, t := range outside {
			if i >= 10 {
				fmt.Fprintf(&sb, "  ... and %d more symlink(s)\n", len(outside)-10)
				break
			}
			if t.Unresolvable {
				fmt.Fprintf(&sb, "  %s\n    └─ component at: %s (could not be resolved — target unknown)\n\n",
					t.OriginalPath, t.SymlinkAt)
				continue
			}
			fmt.Fprintf(&sb, "  %s\n    └─ symlink at: %s → %s (outside workspace)\n\n",
				t.OriginalPath, t.SymlinkAt, t.FullResolved)
		}
	}

	if len(inside) > 0 {
		sb.WriteString("This tool call traverses symlinks (target is within workspace):\n\n")
		for i, t := range inside {
			if i >= 10 {
				fmt.Fprintf(&sb, "  ... and %d more symlink(s)\n", len(inside)-10)
				break
			}
			fmt.Fprintf(&sb, "  %s\n    └─ symlink at: %s → %s (inside workspace)\n\n",
				t.OriginalPath, t.SymlinkAt, t.FullResolved)
		}
	}

	if len(outside) > 0 {
		sb.WriteString("The agent will follow the symlink and operate on the actual target outside the workspace.\n")
	} else if len(inside) > 0 {
		sb.WriteString("The agent will follow the symlink and operate on the resolved target within the workspace.\n")
	}

	if suspicious {
		sb.WriteString("\n⚠ Best-effort check: the command contains unresolved shell expansions ($var, $(cmd), `cmd`) that may hide additional paths.\n")
	}

	return strings.TrimSpace(sb.String())
}
