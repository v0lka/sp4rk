// Package ignore is a multi-root ignore resolver that loads .gitignore and
// .aiignore files (at the root and in every nested directory) for each root
// and answers whether an arbitrary path is ignored by the patterns of the root
// that contains it.
//
// It is a pure algorithmic building block: it performs no hidden-dotfile or
// binary-file filtering. Those universal guards are caller-side concerns that
// layer on top of this resolver.
//
// Negation patterns (lines beginning with '!') are intentionally unsupported —
// they are silently skipped, matching the behaviour this resolver replaces.
package ignore

import (
	"bufio"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	"github.com/bmatcuk/doublestar/v4"
	"github.com/v0lka/sp4rk/pathutil"
)

// ignoreFileNames is the set of files, in any directory, whose patterns are
// honoured. A directory's ignore file applies to that directory and below.
var ignoreFileNames = map[string]bool{
	".gitignore": true,
	".aiignore":  true,
}

// pattern is a single compiled ignore rule. glob is a doublestar-compatible
// pattern expressed relative to the resolver root. dirOnly marks rules that
// originated from a trailing-slash form and therefore match directories only.
type pattern struct {
	glob    string
	dirOnly bool
}

// Resolver is the ignore resolver for a single root directory. It walks the
// root once at construction, collecting every .gitignore and .aiignore file
// (root plus nested directories) and compiling their patterns into globs
// anchored relative to the root.
type Resolver struct {
	root     string
	patterns []pattern
}

// NewResolver walks root once, collecting and compiling every ignore file
// found beneath it. root may be absolute or relative; it is canonicalized to
// an absolute, symlink-resolved form so that ignore queries work regardless of
// the path form callers supply. Root resolution uses pathutil's
// longest-existing-prefix resolution (rather than a strict EvalSymlinks on the
// whole path) so a root that does not fully exist yet still loads cleanly.
// A failure to resolve, read, or walk returns a wrapping error.
func NewResolver(root string) (*Resolver, error) {
	absRoot, err := filepath.Abs(filepath.Clean(root))
	if err != nil {
		return nil, fmt.Errorf("ignore: resolve root %q: %w", root, err)
	}
	// Canonicalize via symlink resolution of the longest existing prefix.
	// This makes the stored root match the form IsWithinPath and the tools
	// produce (the tools' resolvePath runs ResolveExistingPrefix whenever a
	// workspace is in context, yielding e.g. /private/tmp on macOS rather
	// than the raw /tmp). Without this, a raw root (/tmp) queried with a
	// resolved path (/private/tmp/...) would yield a Rel of "../../private/..."
	// and every ignored file would leak as "not ignored".
	absRoot = pathutil.ResolveExistingPrefix(absRoot)
	r := &Resolver{root: absRoot}
	if err := r.load(); err != nil {
		return nil, fmt.Errorf("ignore: load root %q: %w", absRoot, err)
	}
	return r, nil
}

// Root returns the cleaned absolute root this resolver was built for.
func (r *Resolver) Root() string {
	return r.root
}

// Match reports whether relPath (relative to the resolver root, slash or
// OS separators accepted) is ignored by any compiled rule.
//
// isDir indicates whether the path itself is a directory. To honour directory
// semantics correctly, Match also considers every ancestor directory of
// relPath as a directory: if any ancestor is ignored, the path is ignored too
// (the standard gitignore "once a directory is ignored, so are its contents"
// behaviour).
func (r *Resolver) Match(relPath string, isDir bool) bool {
	relPath = filepath.ToSlash(filepath.Clean(relPath))
	// The root itself, an empty path, or a path escaping the root cannot be
	// ignored by a rule.
	if relPath == "" || relPath == "." || strings.HasPrefix(relPath, "..") {
		return false
	}
	segments := strings.Split(relPath, "/")
	cum := ""
	for i, seg := range segments {
		if cum == "" {
			cum = seg
		} else {
			cum = cum + "/" + seg
		}
		segIsDir := i < len(segments)-1 || isDir
		if r.matchOne(cum, segIsDir) {
			return true
		}
	}
	return false
}

// Ignored reports whether absPath (an absolute path) is ignored. absPath is
// canonicalized via longest-existing-prefix symlink resolution and then
// converted to a root-relative path; this makes the resolver robust to either
// path form callers supply (raw /tmp/... or resolved /private/tmp/...), which
// matters because the tools emit different forms depending on whether a
// workspace is in context. Paths that cannot be made relative to this
// resolver's root are not ignored.
func (r *Resolver) Ignored(absPath string, isDir bool) bool {
	rel, err := filepath.Rel(r.root, pathutil.ResolveExistingPrefix(absPath))
	if err != nil {
		return false
	}
	return r.Match(rel, isDir)
}

// matchOne tests a single root-relative slash path (no ancestor walking)
// against every compiled pattern. dirOnly rules are skipped when isDir is
// false.
func (r *Resolver) matchOne(path string, isDir bool) bool {
	for _, p := range r.patterns {
		matched, err := doublestar.Match(p.glob, path)
		if err != nil || !matched {
			continue
		}
		if p.dirOnly && !isDir {
			continue
		}
		return true
	}
	return false
}

// load walks the root collecting patterns from every ignore file. It prunes
// the walk for efficiency: the .git directory is always skipped (it is never
// source we want to honour ignore files for, and it can be enormous), and any
// directory that is itself ignored by the patterns collected so far is pruned
// too — once a directory is ignored, ignore files beneath it are irrelevant.
func (r *Resolver) load() error {
	return filepath.WalkDir(r.root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() {
			if !ignoreFileNames[d.Name()] {
				return nil
			}
			relDir, relErr := filepath.Rel(r.root, filepath.Dir(path))
			if relErr != nil {
				return nil //nolint:nilerr // skip ignore file whose dir is unresolvable; continue the walk
			}
			relDir = filepath.ToSlash(relDir)
			if relDir == "." {
				relDir = ""
			}
			pats, err := readIgnoreFile(path, relDir)
			if err != nil {
				return fmt.Errorf("read %s: %w", path, err)
			}
			r.patterns = append(r.patterns, pats...)
			return nil
		}
		// Directory pruning.
		name := d.Name()
		// Always skip .git: it is never meaningful source and can be huge.
		if name == ".git" && path != r.root {
			return fs.SkipDir
		}
		// Prune directories that are already ignored by the patterns gathered
		// so far: ignore files beneath an ignored directory have no effect.
		if path != r.root {
			relDir, relErr := filepath.Rel(r.root, path)
			if relErr == nil {
				if r.Match(filepath.ToSlash(relDir), true) {
					return fs.SkipDir
				}
			}
		}
		return nil
	})
}

// readIgnoreFile parses a single ignore file at absPath and compiles its
// non-empty, non-comment, non-negation lines into patterns scoped to relDir
// (the slash-relative directory containing the file, "" for the root).
func readIgnoreFile(absPath, relDir string) ([]pattern, error) {
	f, err := os.Open(absPath)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()

	var pats []pattern
	sc := bufio.NewScanner(f)
	// Allow generously long lines (gitignore patterns can be wide globs).
	sc.Buffer(make([]byte, 64*1024), 1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		// Negation patterns are unsupported and silently skipped.
		if strings.HasPrefix(line, "!") {
			continue
		}
		// Strip unescaped trailing spaces (a simplification: escaped literal
		// trailing spaces are not honoured, matching prior behaviour).
		line = strings.TrimRight(line, " ")
		if line == "" {
			continue
		}
		p := compile(relDir, line)
		if p.glob != "" {
			pats = append(pats, p)
		}
	}
	return pats, sc.Err()
}

// compile converts a raw gitignore-style pattern line into a doublestar glob
// anchored relative to relDir (the directory of the file the line came from,
// "" for the root).
//
// Semantics:
//   - A leading slash anchors the pattern to relDir (root-relative only).
//   - Any internal slash (after trimming leading/trailing slashes) also
//     anchors the pattern to relDir, per gitignore rules.
//   - A bare name with no slash matches at any depth beneath relDir.
//   - A trailing slash marks the rule as directory-only.
func compile(relDir, line string) pattern {
	anchored, dirOnly, core := analyze(line)
	var glob string
	switch {
	case core == "":
		glob = ""
	case !anchored && relDir == "":
		glob = "**/" + core
	case !anchored:
		glob = relDir + "/**/" + core
	case relDir == "":
		glob = core
	default:
		glob = relDir + "/" + core
	}
	return pattern{glob: glob, dirOnly: dirOnly}
}

// analyze splits a trimmed pattern line into its anchoring, directory-only
// flag, and core glob body (leading and trailing slashes stripped).
func analyze(line string) (anchored, dirOnly bool, core string) {
	core = line
	if strings.HasPrefix(core, "/") {
		anchored = true
		core = strings.TrimPrefix(core, "/")
	}
	if strings.HasSuffix(core, "/") {
		dirOnly = true
		core = strings.TrimSuffix(core, "/")
	}
	// A pattern containing a slash anywhere besides a trailing slash is
	// implicitly anchored to its file's directory.
	if strings.Contains(core, "/") {
		anchored = true
	}
	return anchored, dirOnly, core
}

// Multi is the multi-root ignore resolver. It holds a Resolver per root and
// answers ignore queries by delegating to whichever root contains the path.
type Multi struct {
	resolvers []*Resolver
}

// NewMulti builds a Resolver for each root and returns a Multi over them.
// Roots may be absolute or relative; each is resolved to an absolute path.
// A failure building any resolver returns a wrapping error.
func NewMulti(roots ...string) (*Multi, error) {
	if len(roots) == 0 {
		return nil, errors.New("ignore: NewMulti requires at least one root")
	}
	resolvers := make([]*Resolver, 0, len(roots))
	for _, root := range roots {
		r, err := NewResolver(root)
		if err != nil {
			return nil, fmt.Errorf("ignore: NewMulti root %q: %w", root, err)
		}
		resolvers = append(resolvers, r)
	}
	return &Multi{resolvers: resolvers}, nil
}

// Resolvers returns the per-root resolvers. The returned slice is a copy; it
// is safe for callers to retain or reorder without affecting the Multi.
func (m *Multi) Resolvers() []*Resolver {
	out := make([]*Resolver, len(m.resolvers))
	copy(out, m.resolvers)
	return out
}

// RootFor returns the Resolver whose root contains absPath, or nil when no
// known root contains it. Containment is symlink-aware via pathutil.
func (m *Multi) RootFor(absPath string) *Resolver {
	for _, r := range m.resolvers {
		ok, err := pathutil.IsWithinPath(r.root, absPath)
		if err != nil {
			continue
		}
		if ok {
			return r
		}
	}
	return nil
}

// Ignored reports whether absPath is ignored by the root that contains it.
// Returns false when no known root contains the path, mirroring the
// IgnoreChecker contract that paths outside all roots are never ignored.
func (m *Multi) Ignored(absPath string, isDir bool) bool {
	r := m.RootFor(absPath)
	if r == nil {
		return false
	}
	return r.Ignored(absPath, isDir)
}

// IgnoreChecker is the abstraction tools depend on: given an absolute path and
// whether it is a directory, report whether it is ignored. Both Resolver and
// Multi satisfy this interface.
type IgnoreChecker interface { //nolint:revive // name mandated by spec; stutter with package name is acceptable for the canonical interface name
	Ignored(absPath string, isDir bool) bool
}
