// Package pathutil provides reusable filesystem-path algorithms with no
// knowledge of any application's directory layout. Safe for use from any layer.
package pathutil

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

// IsWithinPath returns true if child is equal to or a descendant of parent.
// Both paths are symlink-resolved through their longest existing prefix
// (ResolveExistingPrefix) to handle OS-level symlinks like macOS
// /var → /private/var even when paths don't exist on disk.
//
// Both parent and child must be absolute paths. Passing a relative path
// produces undefined results; callers must resolve paths to absolute form
// before calling this function.
//
// The comparison is case-SENSITIVE. Use [IsWithinPathFold] for the
// case-insensitive variant required by path-locality decisions on
// case-insensitive filesystems (the default on macOS and Windows), where
// "/Users/Foo" and "/Users/foo" denote the same location.
//
// Returns an error when parent is empty (containment cannot be determined —
// callers must guard empty roots explicitly) or when parent and child live on
// different volumes (e.g. different Windows drives).
func IsWithinPath(parent, child string) (bool, error) {
	return isWithinPath(parent, child, false)
}

// IsWithinPathFold is the case-insensitive variant of [IsWithinPath]: two
// path components that differ only by Unicode case (via strings.ToLower) are
// treated as equal. This mirrors the semantics of case-insensitive
// filesystems — the default on macOS (APFS) and Windows (NTFS) — where a path
// written with different casing than the session root still resolves to the
// same on-disk location.
//
// Both parent and child must be absolute paths. Passing a relative path
// produces undefined results; callers must resolve paths to absolute form
// before calling this function.
//
// This is the primitive consulted by tool-argument path-locality checks
// (judge fast-path auto-approval, file-tool judges, symlink inside/outside
// classification, working-directory validation): a tool path that is local
// must be recognized as such regardless of letter case. Existing paths are
// already case-normalized by EvalSymlinks inside ResolveExistingPrefix, so
// the folding only matters for not-yet-existing paths (e.g. write_file
// targets) whose longest existing prefix cannot carry the canonical casing.
//
// Case folding never weakens escape prevention: it can only cause an
// otherwise-mismatched path to be considered "within" a root; the ".."
// traversal and symlink-escape logic is unaffected. Returns an error when
// parent is empty, matching IsWithinPath.
func IsWithinPathFold(parent, child string) (bool, error) {
	return isWithinPath(parent, child, true)
}

// isWithinPath is the shared implementation of [IsWithinPath] (case-sensitive)
// and [IsWithinPathFold] (case-insensitive). When fold is true, both resolved
// paths are lowercased so the final containment check is case-insensitive.
// Lowercasing both sides preserves volume-prefix agreement (e.g. "C:" / "c:").
//
// Containment is decided by an explicit prefix check rather than filepath.Rel,
// which compares path components case-insensitively on Windows
// (strings.EqualFold) and would treat a differing-case sibling as a genuine
// descendant. Two safeguards surround the prefix check:
//   - a volume mismatch (filepath.VolumeName differs) means parent and child
//     live in different roots and cannot nest — it returns an error;
//     VolumeName is empty on Unix, so this is a no-op there.
//   - the parent is terminated with a separator before the prefix test, but
//     never doubled (a root parent already ends in one), so a filesystem root
//     or a Windows volume root is correctly recognized as containing its
//     children. That separator also rejects a prefix-only sibling whose name
//     merely starts with the parent's last component (e.g. "/usr" vs "/us").
func isWithinPath(parent, child string, fold bool) (bool, error) {
	// Empty parent means containment cannot be determined — fail closed.
	if parent == "" {
		return false, errors.New("pathutil: empty parent path — containment cannot be determined")
	}
	parentResolved := ResolveExistingPrefix(filepath.Clean(parent))
	childResolved := ResolveExistingPrefix(filepath.Clean(child))
	if fold {
		parentResolved = strings.ToLower(parentResolved)
		childResolved = strings.ToLower(childResolved)
	}

	// A volume mismatch (e.g. different Windows drives) means parent and child
	// live in different roots and cannot nest. filepath.VolumeName is empty on
	// Unix, so this is a no-op there.
	if filepath.VolumeName(parentResolved) != filepath.VolumeName(childResolved) {
		return false, errors.New("pathutil: parent and child on different volumes")
	}

	// Decide containment with an explicit prefix check: child is within parent
	// only when, after the optional case folding applied above, it is parent
	// followed by a separator (or identical to it). Terminate the parent with a
	// separator so the prefix test cannot match a prefix-only sibling whose
	// last component merely starts with the parent's (e.g. "/usr" vs "/us"), but
	// never double it — a root parent already ends in one, and "//" (or
	// "C:\\") would match no real child. The fold variant lowercased both
	// sides, so this same check is case-insensitive for it — the desired split
	// between IsWithinPath and IsWithinPathFold.
	sep := string(filepath.Separator)
	if !strings.HasSuffix(parentResolved, sep) {
		parentResolved += sep
	}
	return childResolved == strings.TrimSuffix(parentResolved, sep) ||
		strings.HasPrefix(childResolved, parentResolved), nil
}

// SplitPathComponents splits a cleaned absolute path into non-empty components,
// stripping the root separator and any volume prefix. For example:
//
//	"/home/user/file.txt"           → ["home", "user", "file.txt"]
//	"C:\Users\file.txt"             → ["Users", "file.txt"]   (Windows)
//	"/"                             → []
func SplitPathComponents(absPath string) []string {
	// Strip the volume prefix (e.g. "C:" on Windows, "\\server\share" for a
	// UNC path) so it is not mistaken for a path component. On Unix the volume
	// is always empty, so this is a no-op there.
	vol := filepath.VolumeName(absPath)
	rest := absPath[len(vol):]
	parts := strings.Split(rest, string(filepath.Separator))
	var result []string
	for _, part := range parts {
		if part != "" {
			result = append(result, part)
		}
	}
	return result
}

// DetectCaseInsensitive reports whether the filesystem at dir treats names
// that differ only by Unicode case as the same file. It probes by creating a
// temporary file whose name carries an upper-case letter, then checking
// whether the lower-cased name resolves to it. This mirrors the default
// behaviour of macOS APFS and Windows NTFS (case-insensitive) versus Linux
// ext4/tmpfs/btrfs (case-sensitive).
//
// Results are memoized per probed directory for the lifetime of the process:
// case-sensitivity is a property of the filesystem mount and cannot change
// while the process runs, so the probe file is created and deleted at most
// ONCE per directory no matter how often the function is called. This keeps
// incidental re-invocations (e.g. hosts that rebuild ignore resolvers or
// workspace contexts per request) from churning probe files in the workspace
// root. A failed probe (unwritable directory → false, the fail-safe default)
// is cached too, so a permanently read-only root is not re-probed forever.
//
// When dir does not exist (e.g. a brand-new session workspace) or is not
// writable, the probe climbs to the nearest existing ancestor directory:
// case-sensitivity is a filesystem property shared by the whole mount, so an
// ancestor on the same volume yields the correct answer for the target. If no
// usable ancestor is found, it returns false — the fail-safe default, since
// assuming case-insensitivity on an actually case-sensitive filesystem can
// turn a non-local path into a "local" one (an authorization-bypass risk for
// path-locality checks such as [IsWithinPathFold]).
//
// Hosts that already resolve the flag once at session-root resolution time
// and pass it down explicitly (via [tools.WithCaseInsensitivePaths]) keep
// working unchanged; the memoization is transparent to them.
func DetectCaseInsensitive(dir string) bool {
	probeDir, ok := existingAncestorDir(filepath.Clean(dir))
	if !ok {
		return false
	}
	caseInsensitiveMu.Lock()
	defer caseInsensitiveMu.Unlock()
	if ci, known := caseInsensitiveKnown[probeDir]; known {
		return ci
	}
	ci := probeCaseInsensitive(probeDir)
	caseInsensitiveKnown[probeDir] = ci
	return ci
}

// caseInsensitiveKnown memoizes probeCaseInsensitive results, keyed by the
// existing-ancestor directory the probe ran in. caseInsensitiveMu guards both
// the map and the probe itself, so concurrent first-time callers of the same
// directory perform exactly one probe between them (losers block briefly on
// the mutex and then read the stored result) rather than racing to create
// duplicate probe files.
var (
	caseInsensitiveMu    sync.Mutex
	caseInsensitiveKnown = make(map[string]bool)
)

// existingAncestorDir returns the deepest existing directory at or above dir,
// climbing toward the filesystem root. It returns ("", false) when no existing
// directory is found (e.g. an unreachable root). Writability is confirmed by
// the probe itself: a non-writable ancestor simply fails the probe.
func existingAncestorDir(dir string) (string, bool) {
	candidate := dir
	for {
		if info, err := os.Stat(candidate); err == nil && info.IsDir() {
			return candidate, true
		}
		parent := filepath.Dir(candidate)
		if parent == candidate {
			// Reached the filesystem root without an existing directory.
			return "", false
		}
		candidate = parent
	}
}

// probeCaseInsensitive creates a uniquely-named temporary file with a
// mixed-case name in dir and reports whether the opposite-case name resolves
// to it. All created entries are removed before returning.
func probeCaseInsensitive(dir string) bool {
	f, err := os.CreateTemp(dir, "CaseSense-*.probe")
	if err != nil {
		return false
	}
	defer func() { _ = os.Remove(f.Name()) }()
	if err := f.Close(); err != nil {
		return false
	}

	// The "CaseSense-" prefix guarantees an upper-case letter, so the
	// lower-cased name always differs and the test is meaningful.
	name := filepath.Base(f.Name())
	flipped := strings.ToLower(name)
	if flipped == name {
		flipped = strings.ToUpper(name)
	}
	if flipped == name {
		return false
	}
	if _, err := os.Stat(filepath.Join(dir, flipped)); err == nil {
		// The differently-cased name resolves — the filesystem is
		// case-insensitive. On such a filesystem this is the same file, so
		// the deferred Remove of the original name already cleaned it up.
		return true
	}
	return false
}

// ResolveExistingPrefix resolves symlinks on the longest existing prefix of
// path, joining the non-existent suffix back. Used when validating paths for
// files or directories that may not exist yet (e.g., write/mkdir tool targets).
//
// Example: if "/ws/link" is a symlink but "/ws/link/newfile.txt" doesn't exist,
// returns the symlink-resolved prefix + "/newfile.txt".
func ResolveExistingPrefix(path string) string {
	candidate := path
	for {
		resolved, err := filepath.EvalSymlinks(candidate)
		if err == nil {
			if candidate == path {
				return resolved
			}
			rel, relErr := filepath.Rel(candidate, path)
			if relErr != nil {
				// Paths on different volumes — fall back to unresolved.
				return path
			}
			return filepath.Join(resolved, rel)
		}
		if errors.Is(err, os.ErrNotExist) {
			parent := filepath.Dir(candidate)
			if parent == candidate {
				// Reached root — nothing exists, return as-is.
				return path
			}
			candidate = parent
			continue
		}
		// Permission or other error — return as-is.
		return path
	}
}
