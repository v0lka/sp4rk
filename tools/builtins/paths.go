package builtins

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/v0lka/sp4rk/pathutil"
	"github.com/v0lka/sp4rk/tools"
)

// resolvePath resolves a file path against the session roots (workspace and
// temp directory, treated as equal peers).
//
// Relative paths are joined with the workspace root and MUST stay within it —
// escaping via ".." components is rejected (returns ""). Relative paths cannot
// target the temp directory; callers must use absolute paths for temp access.
//
// Absolute paths are symlink-resolved via pathutil.ResolveExistingPrefix and
// returned regardless of whether they fall inside or outside the session
// roots. Containment is NOT enforced here: operations outside the session
// roots are allowed after user confirmation (gated by the Judge layer and
// registry confirmation flow). Callers that need to know whether the resolved
// path is inside the session roots should use isPathInSessionRoots.
//
// If no workspace is available in the context, the path is returned as-is
// (callers validate on their own).
func resolvePath(ctx context.Context, path string) string {
	ws := tools.WorkspacePathFrom(ctx)
	if ws == "" {
		return path
	}

	// Resolve symlinks on the workspace root (fall back to unresolved path
	// if the directory doesn't exist yet — common for No Project sessions).
	realWS, err := resolveWorkspaceRoot(ws)
	if err != nil {
		return "" // unresolvable workspace — reject
	}

	if filepath.IsAbs(path) {
		// Resolve symlinks on the longest existing prefix (the file may not
		// exist yet). Return the resolved path regardless of containment;
		// the Judge layer and registry confirmation flow handle access
		// control for paths outside the session roots.
		return pathutil.ResolveExistingPrefix(path)
	}

	// Relative path: join with workspace then resolve to absolute for
	// containment validation. filepath.Join resolves ".." components,
	// so the result may escape the workspace — reject if it does.
	joined := filepath.Join(ws, path)
	absJoined, absErr := filepath.Abs(joined)
	if absErr != nil {
		return ""
	}
	resolved := pathutil.ResolveExistingPrefix(absJoined)
	if !tools.IsWithinRoot(ctx, realWS, resolved) {
		return ""
	}
	return resolved
}

// ResolvePath is the exported form of resolvePath. It resolves a file path
// against the session roots (workspace and temp directory, treated as equal
// peers), replicating exactly the logic used by the built-in
// read_file/write_file/edit_file tools — including symlink resolution of the
// workspace root (so OS-level symlinks such as macOS /tmp → /private/tmp do
// not cause false negatives on containment).
//
// External packages that wrap a built-in tool (e.g. c0wrk's
// document-converting read_file wrapper) should call this instead of
// re-implementing the resolution, so path handling and containment checks stay
// consistent with the inner tools. See resolvePath for the full contract.
func ResolvePath(ctx context.Context, path string) string {
	return resolvePath(ctx, path)
}

// resolveWorkspaceRoot resolves symlinks on a session root path (workspace or
// temp directory). Falls back to the unresolved clean path when the directory
// doesn't exist yet (e.g., brand-new No Project session workspace).
func resolveWorkspaceRoot(ws string) (string, error) {
	resolved, err := filepath.EvalSymlinks(filepath.Clean(ws))
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return filepath.Clean(ws), nil
		}
		return "", err
	}
	return resolved, nil
}

// validateResolvedPath checks that the resolved path is non-empty. A non-empty
// result from resolvePath indicates a usable path; an empty result means the
// input was a relative path that escaped the workspace (rejected by resolvePath).
//
// Containment within the session roots is NOT enforced here — operations
// outside workspace/temp are allowed after user confirmation. Use
// isPathInSessionRoots when containment must be known.
func validateResolvedPath(resolved string) error {
	if resolved == "" {
		return errors.New("path is outside the session workspace")
	}
	return nil
}

// isPathInSessionRoots reports whether absPath is contained within any of the
// session roots: the workspace, the temp directory, and any additional allowed
// roots. All roots are treated as equal peers: any operation permitted inside
// the workspace is permitted inside the temp directory and allowed roots, and
// vice versa. Symlinks are resolved through the longest existing prefix so
// that OS-level symlinks (e.g., macOS /tmp → /private/tmp) do not cause false
// negatives.
//
// Harmless special-device paths (/dev/null, /dev/full; NUL on Windows) are
// treated as local via [tools.IsHarmlessDevicePath] so file operations
// targeting them do not force a user-confirmation prompt.
func isPathInSessionRoots(ctx context.Context, absPath string) bool {
	if tools.IsHarmlessDevicePath(absPath) {
		return true
	}
	for _, root := range tools.SessionRoots(ctx) {
		if isPathInRootStr(ctx, absPath, root) {
			return true
		}
	}
	return false
}

// isPathInRootStr reports whether absPath is contained within the single root
// string. Returns false when the root is empty or unresolvable, or when
// absPath is not contained within it. Uses [tools.IsWithinRoot] so containment
// respects the session case-sensitivity flag: case-insensitive filesystems
// (macOS APFS / Windows NTFS) fold letter case, case-sensitive ones (Linux
// ext4) do not.
func isPathInRootStr(ctx context.Context, absPath, root string) bool {
	if root == "" {
		return false
	}
	rootAbs, err := resolveWorkspaceRoot(root)
	if err != nil {
		return false
	}
	return tools.IsWithinRoot(ctx, rootAbs, absPath)
}

// formatOutsideRootsError returns a descriptive error for a path that falls
// outside all session roots. Used by Judge helpers when escalating to user
// confirmation.
func formatOutsideRootsError(absPath string) error {
	return fmt.Errorf("path is outside the session roots: %s", absPath)
}

// isPathInGitDir reports whether absPath lies inside the workspace's git
// internals — the path is contained within the workspace root and its
// workspace-relative part contains a ".git" path component. This covers the
// repository root's .git directory, nested repositories (submodules and
// worktrees, where ".git" may be a gitdir-pointer file rather than a
// directory), and any deeper path such as .git/objects or
// .git/hooks/pre-commit. Regular dotfiles and dot-directories (.gitignore,
// .github, .golangci) are NOT matched: only the exact ".git" component is.
//
// Scope is decided by [tools.IsWithinRoot] — the canonical containment check
// — so it folds letter case exactly when locality auto-approval does. This
// pairing is load-bearing: on a case-insensitive filesystem (macOS APFS,
// Windows NTFS) a target spelled with a case-mismatched workspace prefix
// ("/WS/.git/config" for workspace "/ws") IS the on-disk git directory, so a
// lexical scope test here (filepath.Rel returns "../WS/…") would let that
// spelling bypass the guard while the very same path auto-approves as local.
// The workspace-relative remainder is still computed lexically with
// filepath.Rel, so under folding it may carry ".." segments; the component
// scan below sees every component regardless.
//
// Component matching respects the session case-sensitivity flag
// ([tools.CaseInsensitivePathsFrom]) the same way [tools.IsWithinRoot] does:
// on case-insensitive filesystems (macOS APFS, Windows NTFS) a ".GIT" or
// ".Git" component IS the git directory and is flagged, while on
// case-sensitive filesystems (Linux ext4) only the literal ".git" is — a
// distinct-cased sibling there is an ordinary directory, not git internals.
// Note that resolution runs before this predicate: on Windows,
// filepath.EvalSymlinks canonicalizes existing components to their on-disk
// spelling, so a ".GIT" that aliases the real ".git" directory arrives here
// already spelled ".git" and is flagged regardless of the session flag — the
// write genuinely targets git internals. The flag governs only components
// that exist on disk in no spelling: their spelling survives resolution on
// every platform, so there — and only there — case sensitivity applies.
//
// Paths outside the workspace root (including the session temp directory,
// which is a separate root) return false: they are not "inside a workspace"
// and are already handled by the outside-session-roots escalation. The
// predicate fails open (false) when no workspace is attached or the workspace
// root cannot be resolved — in those situations the surrounding judge flow
// fails closed through its own containment checks instead.
func isPathInGitDir(ctx context.Context, absPath string) bool {
	ws := tools.WorkspacePathFrom(ctx)
	if ws == "" {
		return false
	}
	realWS, err := resolveWorkspaceRoot(ws)
	if err != nil {
		return false
	}
	// Scope: only paths the canonical containment check places inside the
	// workspace subtree. IsWithinRoot folds letter case when the session flag
	// says the filesystem does, so a case-mismatched spelling of a workspace
	// path stays in scope — matching how locality auto-approval treats it.
	if !tools.IsWithinRoot(ctx, realWS, absPath) {
		return false
	}
	rel, relErr := filepath.Rel(realWS, absPath)
	if relErr != nil {
		return false
	}
	caseFold := tools.CaseInsensitivePathsFrom(ctx)
	for _, component := range pathutil.SplitPathComponents(rel) {
		if component == ".git" || (caseFold && strings.EqualFold(component, ".git")) {
			return true
		}
	}
	return false
}
