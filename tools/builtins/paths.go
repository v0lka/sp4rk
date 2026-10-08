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
// Relative paths are resolved in OS component order against the workspace
// root (symlinks expanded as encountered, ".." climbing from the resolved
// location) and MUST stay within it — a spelling that POSIX-resolves outside
// the workspace (a plain ".." escape or a "symlink/.." climbing from an
// out-of-root link target) is rejected (returns ""). Relative paths cannot
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

	// Relative path: resolve in OS component order from the (already
	// symlink-resolved) workspace root. The raw concatenation — no Clean, no
	// filepath.Join — is load-bearing: a "symlink/.." spelling must expand
	// the link first and climb ".." from the link TARGET's directory, exactly
	// as the kernel does. A lexical Join would erase the link before any
	// symlink resolution and silently re-point the path inside the
	// workspace; the containment check below then refuses what POSIX
	// resolution places outside it.
	resolved := pathutil.ResolveExistingPrefix(rawJoin(realWS, path))
	if !tools.IsWithinRoot(ctx, realWS, resolved) {
		return ""
	}
	return resolved
}

// resolvePathUnresolved resolves a file path against the session roots like
// resolvePath, but WITHOUT following a symlink in the final path component.
// It exists for the DESTRUCTIVE tools (delete_file, delete_directory): POSIX
// rm semantics remove the link itself, never the target, so the path that is
// unlinked must stay as spelled in its last component while everything above
// it is canonicalized (longest-existing-prefix resolution of the parent —
// never a full EvalSymlinks of the target).
//
// The parent prefix is resolved in OS component order — symlinks are expanded
// as encountered and ".." climbs from the resolved location — so
// "symlink/../name" addresses the link TARGET's parent, exactly as the kernel
// resolves the spelling. filepath.Clean and filepath.Join must never run on
// the input first: they collapse "symlink/.." lexically, erasing the link
// before any symlink resolution and silently re-pointing the spelling at the
// link's lexical parent.
//
// The contract otherwise mirrors resolvePath: relative paths MUST stay within
// the workspace after OS-order resolution ("" is returned otherwise — a
// relative symlink pointing outside, or a "symlink/.." spelling that climbs
// out of the link target, is therefore refused, the fail-closed direction);
// absolute paths are returned regardless of containment, leaving access
// control to the Judge layer and the registry confirmation flow. When no
// workspace is available the path is returned as-is.
func resolvePathUnresolved(ctx context.Context, path string) string {
	ws := tools.WorkspacePathFrom(ctx)
	if ws == "" {
		return path
	}

	realWS, err := resolveWorkspaceRoot(ws)
	if err != nil {
		return "" // unresolvable workspace — reject
	}

	if filepath.IsAbs(path) {
		// Resolve the parent prefix in OS component order; the final
		// component stays exactly as spelled so a symlink there is never
		// followed. No Clean: it would erase "symlink/.." before any
		// symlink resolution (see resolveKeepFinal).
		return resolveKeepFinal(path)
	}

	// Relative path: resolve in OS component order from the resolved
	// workspace root, keeping the final component as spelled. Containment is
	// checked on the RESULT: a spelling that POSIX-resolves outside the
	// workspace (a "symlink/.." climbing from an out-of-root target, or a
	// plain ".." escape) is refused, and [tools.IsWithinRoot] internally
	// resolves symlinks, so a final-component symlink pointing outside is
	// refused as before — the fail-closed direction.
	resolved := resolveKeepFinal(rawJoin(realWS, path))
	if !tools.IsWithinRoot(ctx, realWS, resolved) {
		return ""
	}
	return resolved
}

// rawJoin concatenates base and rel at the raw byte level WITHOUT cleaning:
// "symlink/.." sequences in rel must survive to
// [pathutil.ResolveExistingPrefix], which evaluates them in OS component
// order. filepath.Join would Clean the concatenation first and lexically
// erase the link before any symlink resolution runs.
func rawJoin(base, rel string) string {
	trimmed := strings.TrimRight(base, string(filepath.Separator))
	if trimmed == "" {
		// base is the filesystem root itself ("/" or a Windows volume root).
		trimmed = string(filepath.Separator)
	}
	if rel == "" {
		return trimmed
	}
	return trimmed + string(filepath.Separator) + rel
}

// resolveKeepFinal resolves the directory prefix of rawAbs in OS component
// order — symlinks are expanded as encountered and ".." climbs from the
// resolved location ([pathutil.ResolveExistingPrefix] on the un-cleaned
// parent) — while the FINAL component is kept exactly as spelled: a symlink
// in the last position is never followed (POSIX rm semantics for the
// destructive tools). The input must be absolute; callers building it from a
// relative path must go through [rawJoin] so no Clean collapses
// "symlink/.." beforehand.
//
// "." and ".." as the final component are folded back into the directory and
// fully resolved: neither can be a symlink, so resolving them is safe and
// matches what the kernel does with such spellings.
func resolveKeepFinal(rawAbs string) string {
	trimmed := strings.TrimRight(rawAbs, string(filepath.Separator))
	if trimmed == "" {
		// The path is the filesystem root itself ("/" or a volume root):
		// there is no final component to keep unresolved.
		return pathutil.ResolveExistingPrefix(rawAbs)
	}
	dir, base := filepath.Split(trimmed)
	dir = strings.TrimRight(dir, string(filepath.Separator))
	switch {
	case dir == "" && strings.HasPrefix(trimmed, string(filepath.Separator)):
		// Only the leading root separator preceded the final component and
		// TrimRight removed it — restore the root.
		dir = string(filepath.Separator)
	case dir != "" && dir == filepath.VolumeName(trimmed):
		// A volume root ("C:\", "\\server\share"): the separator after the
		// volume is structural, so restore it — a bare volume ("C:") is a
		// cwd-relative spelling on Windows, not the volume root.
		dir += string(filepath.Separator)
	case dir == "":
		// A single component with no root at all — nothing to keep
		// unresolved, resolve fully.
		return pathutil.ResolveExistingPrefix(trimmed)
	}
	if base == "." || base == ".." {
		return pathutil.ResolveExistingPrefix(rawJoin(dir, base))
	}
	return filepath.Join(pathutil.ResolveExistingPrefix(dir), base)
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
// Harmless special-device paths (/dev/null; NUL on Windows) are
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

// isGitPathComponent reports whether a single path component denotes git
// internals. The match folds letter case UNCONDITIONALLY (".git", ".GIT",
// ".Git" all match) — the fail-safe direction: case-variant dot-git
// spellings can denote live git internals (mechanism confirmed empirically
// against git 2.50.1), so the guard never bets on filesystem case
// sensitivity to decide whether a write targets a repository. A false
// positive costs one interactive confirmation for an exotic case-distinct
// sibling; a false negative is a write into real git internals. This is
// deliberately stricter than the per-root scope check in isPathInGitDir,
// which still folds only when the session case-sensitivity flag says the
// filesystem does.
func isGitPathComponent(component string) bool {
	return strings.EqualFold(component, ".git")
}

// pathContainsGitComponent reports whether any component of the given
// (absolute or relative) path is a dot-git component per
// [isGitPathComponent]. Volume prefixes and empty segments are stripped by
// [pathutil.SplitPathComponents].
func pathContainsGitComponent(path string) bool {
	for _, component := range pathutil.SplitPathComponents(path) {
		if isGitPathComponent(component) {
			return true
		}
	}
	return false
}

// isPathInGitDir reports whether absPath lies inside the git internals of
// the workspace or any additional allowed root (work directory): the path is
// contained within one of those roots and either its root-relative part OR
// the root's own components contains a ".git" path component. This covers
// the repository root's .git directory, nested repositories (submodules and
// worktrees, where ".git" may be a gitdir-pointer file rather than a
// directory), any deeper path such as .git/objects or
// .git/hooks/pre-commit — and a root that itself ends inside a ".git" tree
// (e.g. a workspace opened at <repo>/.git/worktrees/<name>), where every
// target under the root is git internals even though the root-relative
// remainder never mentions ".git". Regular dotfiles and dot-directories
// (.gitignore, .github, .golangci) are NOT matched: only the exact ".git"
// component is.
//
// Per-root scope is decided by [tools.IsWithinRoot] — the canonical
// containment check — so it folds letter case exactly when locality
// auto-approval does. This pairing is load-bearing: on a case-insensitive
// filesystem (macOS APFS, Windows NTFS) a target spelled with a
// case-mismatched root prefix ("/WS/.git/config" for workspace "/ws") IS
// the on-disk git directory, so a lexical scope test here (filepath.Rel
// returns "../WS/…") would let that spelling bypass the guard while the very
// same path auto-approves as local. The root-relative remainder is still
// computed lexically with filepath.Rel, so under folding it may carry ".."
// segments; the component scan sees every component regardless.
//
// Component matching folds letter case unconditionally
// ([isGitPathComponent]) — see its doc for the fail-safe rationale. Note
// that resolution runs before this predicate: on Windows,
// filepath.EvalSymlinks canonicalizes existing components to their on-disk
// spelling, so a ".GIT" that aliases the real ".git" directory arrives here
// already spelled ".git" and is flagged regardless of any flag.
//
// The session temp directory ([tools.TempDirFrom]) is deliberately NOT
// guarded: it is host-managed per-session scratch space, so creating a
// throwaway repository there is legitimate. Hosts that want OS-level temp
// trees guarded route them through the allowed-roots channel instead
// (c0wrk's implicit temp roots do exactly that), and those roots ARE
// guarded here. The exemption is a documented accepted-risk decision, not
// an oversight.
//
// Windows 8.3 short-name (SFN) caveat: the standard library resolves
// existing path components by querying the filesystem, and Windows returns
// the long-name form for 8.3 short names there (FindFirstFile semantics),
// so an SFN spelling of an existing component is normalized to its long
// form before reaching this predicate. Whether an SFN alias for the literal
// ".git" component (itself short enough to never require one) can exist on
// a real volume is UNVERIFIED — before changing any code in this area,
// empirically verify SFN ".git" behavior on Windows first. Documented
// deliberately without a code change.
//
// Paths outside every guarded root return false: they are handled by the
// outside-session-roots escalation. The predicate fails open (false) when
// no root is attached or a root cannot be resolved — in those situations
// the surrounding judge flow fails closed through its own containment
// checks instead.
func isPathInGitDir(ctx context.Context, absPath string) bool {
	roots := append([]string{tools.WorkspacePathFrom(ctx)}, tools.AllowedRootsFrom(ctx)...)
	for _, root := range roots {
		if root == "" {
			continue
		}
		realRoot, err := resolveWorkspaceRoot(root)
		if err != nil {
			continue
		}
		// Scope: only paths the canonical containment check places inside
		// this root's subtree. IsWithinRoot folds letter case when the
		// session flag says the filesystem does, so a case-mismatched
		// spelling of a root path stays in scope — matching how locality
		// auto-approval treats it.
		if !tools.IsWithinRoot(ctx, realRoot, absPath) {
			continue
		}
		rel, relErr := filepath.Rel(realRoot, absPath)
		if relErr != nil {
			continue
		}
		// The target's own components…
		if pathContainsGitComponent(rel) {
			return true
		}
		// …and the root's own components: a root ending inside a /.git/…
		// tree makes every target under it git internals.
		if pathContainsGitComponent(realRoot) {
			return true
		}
	}
	return false
}
