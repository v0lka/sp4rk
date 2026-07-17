package ignore

import (
	"os"
	"path/filepath"
	"testing"
)

// writeFile is a small helper that writes content to a path under dir,
// creating parent directories as needed.
func writeFile(t *testing.T, dir, rel, content string) {
	t.Helper()
	full := joinRel(dir, rel)
	if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", full, err)
	}
	if err := os.WriteFile(full, []byte(content), 0o644); err != nil {
		t.Fatalf("write %s: %v", full, err)
	}
}

// touchFile creates an empty regular file at dir/rel (parents created).
func touchFile(t *testing.T, dir, rel string) {
	t.Helper()
	writeFile(t, dir, rel, "")
}

// joinRel joins a slash-separated relative path to dir using filepath.Join so
// that no path separator appears inside any single argument (avoids the
// gocritic filepathJoin check while keeping test paths readable).
func joinRel(dir, rel string) string {
	parts := []string{dir}
	for _, seg := range splitSlash(rel) {
		if seg != "" {
			parts = append(parts, seg)
		}
	}
	return filepath.Join(parts...)
}

// splitSlash splits a slash-separated string into segments.
func splitSlash(s string) []string {
	var out []string
	start := 0
	for i := 0; i < len(s); i++ {
		if s[i] == '/' {
			out = append(out, s[start:i])
			start = i + 1
		}
	}
	out = append(out, s[start:])
	return out
}

// assertIgnored fails the test if path is not ignored by the checker.
func assertIgnored(t *testing.T, c IgnoreChecker, absPath string, isDir bool) {
	t.Helper()
	if !c.Ignored(absPath, isDir) {
		t.Errorf("expected %q (isDir=%v) to be ignored, but it was not", absPath, isDir)
	}
}

// assertNotIgnored fails the test if path is ignored by the checker.
func assertNotIgnored(t *testing.T, c IgnoreChecker, absPath string, isDir bool) {
	t.Helper()
	if c.Ignored(absPath, isDir) {
		t.Errorf("expected %q (isDir=%v) NOT to be ignored, but it was", absPath, isDir)
	}
}

// ---------------------------------------------------------------------------
// compile / analyze unit checks
// ---------------------------------------------------------------------------

func TestCompile_BareNameMatchesAnyDepth(t *testing.T) {
	// Bare name (no slash) at root -> matches at any depth, including root.
	p := compile("", "node_modules")
	if p.glob != "**/node_modules" {
		t.Fatalf("bare name root glob = %q, want **/node_modules", p.glob)
	}
	if p.dirOnly {
		t.Fatalf("bare name should not be dirOnly")
	}
}

func TestCompile_LeadingSlashIsAnchored(t *testing.T) {
	// A leading slash anchors the rule to the root (no **/ prefix).
	p := compile("", "/secret")
	if p.glob != "secret" {
		t.Fatalf("anchored root glob = %q, want secret", p.glob)
	}
}

func TestCompile_TrailingSlashIsDirOnly(t *testing.T) {
	p := compile("", "build/")
	if !p.dirOnly {
		t.Fatalf("trailing slash should set dirOnly")
	}
}

func TestCompile_InternalSlashAnchors(t *testing.T) {
	// A slash inside the body anchors to the root even without a leading slash.
	p := compile("", "src/temp")
	if p.glob != "src/temp" {
		t.Fatalf("internal-slash glob = %q, want src/temp", p.glob)
	}
}

func TestCompile_ScopedToNestedDir(t *testing.T) {
	// Bare name from a nested ignore file is scoped beneath that directory.
	p := compile("sub", "local")
	if p.glob != "sub/**/local" {
		t.Fatalf("nested bare glob = %q, want sub/**/local", p.glob)
	}
	// Anchored rule from a nested ignore file is prefixed with its directory.
	a := compile("sub", "/local")
	if a.glob != "sub/local" {
		t.Fatalf("nested anchored glob = %q, want sub/local", a.glob)
	}
}

// ---------------------------------------------------------------------------
// Resolver: root .gitignore + .aiignore, nested files, anchoring, dirOnly
// ---------------------------------------------------------------------------

func TestResolver_RootGitignoreAndAiignore(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, ".gitignore", "*.log\n")
	writeFile(t, root, ".aiignore", "*.secret\n")
	touchFile(t, root, "app.log")
	touchFile(t, root, "creds.secret")
	touchFile(t, root, "keep.txt")

	r, err := NewResolver(root)
	if err != nil {
		t.Fatalf("NewResolver: %v", err)
	}

	assertIgnored(t, r, joinRel(root, "app.log"), false)      // root .gitignore
	assertIgnored(t, r, joinRel(root, "creds.secret"), false) // root .aiignore
	assertNotIgnored(t, r, joinRel(root, "keep.txt"), false)  // not ignored
}

func TestResolver_NestedIgnoreFiles(t *testing.T) {
	root := t.TempDir()
	// Nested .gitignore deep in the tree.
	writeFile(t, root, "src/pkg/.gitignore", "*.gen.go\n")
	// Nested .aiignore.
	writeFile(t, root, "src/pkg/.aiignore", "drafts/\n")
	touchFile(t, root, "src/pkg/thing.gen.go")
	touchFile(t, root, "src/pkg/drafts/old.txt")
	touchFile(t, root, "src/pkg/keep.go")
	// A file with the same name outside the nested scope.
	touchFile(t, root, "other.gen.go")

	r, err := NewResolver(root)
	if err != nil {
		t.Fatalf("NewResolver: %v", err)
	}

	// Nested .gitignore matches the file under that directory.
	assertIgnored(t, r, joinRel(root, "src/pkg/thing.gen.go"), false)
	// Nested .aiignore dir pattern: the dir and its contents are ignored.
	assertIgnored(t, r, joinRel(root, "src/pkg/drafts"), true)
	assertIgnored(t, r, joinRel(root, "src/pkg/drafts/old.txt"), false)
	// Same name outside the nested scope is NOT ignored.
	assertNotIgnored(t, r, joinRel(root, "other.gen.go"), false)
	assertNotIgnored(t, r, joinRel(root, "src/pkg/keep.go"), false)
}

func TestResolver_BareNameMatchesAnyDepth(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, ".gitignore", "node_modules\n")
	touchFile(t, root, "node_modules/dummy")
	touchFile(t, root, "deep/node_modules/x")

	r, err := NewResolver(root)
	if err != nil {
		t.Fatalf("NewResolver: %v", err)
	}

	// Bare name matches the directory at the root...
	assertIgnored(t, r, joinRel(root, "node_modules"), true)
	// ...and at any depth.
	assertIgnored(t, r, joinRel(root, "deep/node_modules"), true)
	assertIgnored(t, r, joinRel(root, "deep/node_modules/x"), false)
	// A file within the ignored dir is matched too (ancestor is ignored).
	assertIgnored(t, r, joinRel(root, "node_modules/dummy"), false)
}

func TestResolver_AnchoredMatchesRootRelativeOnly(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, ".gitignore", "/secret\n")
	touchFile(t, root, "secret")
	touchFile(t, root, "deep/secret")

	r, err := NewResolver(root)
	if err != nil {
		t.Fatalf("NewResolver: %v", err)
	}

	// Anchored pattern matches only the root-relative path.
	assertIgnored(t, r, joinRel(root, "secret"), false)
	assertNotIgnored(t, r, joinRel(root, "deep/secret"), false)
}

func TestResolver_DirOnlyPatternsOnlyMatchDirs(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, ".gitignore", "build/\n")

	r, err := NewResolver(root)
	if err != nil {
		t.Fatalf("NewResolver: %v", err)
	}

	// "build/" matches a directory named build...
	assertIgnored(t, r, joinRel(root, "build"), true)
	// ...and its contents (because the ancestor dir is ignored).
	touchFile(t, root, "build/out.txt")
	assertIgnored(t, r, joinRel(root, "build/out.txt"), false)
	// ...but a plain FILE named build is NOT matched by a dirOnly pattern.
	_ = os.RemoveAll(joinRel(root, "build"))
	writeFile(t, root, "build", "not a dir")
	assertNotIgnored(t, r, joinRel(root, "build"), false)
}

func TestResolver_NonMatchingPathNotIgnored(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, ".gitignore", "*.log\n/dist/\n")
	touchFile(t, root, "src/main.go")
	touchFile(t, root, "README.md")

	r, err := NewResolver(root)
	if err != nil {
		t.Fatalf("NewResolver: %v", err)
	}

	assertNotIgnored(t, r, joinRel(root, "src/main.go"), false)
	assertNotIgnored(t, r, joinRel(root, "README.md"), false)
	// Path outside the root entirely.
	assertNotIgnored(t, r, joinRel(t.TempDir(), "x.log"), false)
}

func TestResolver_NegationSkipped(t *testing.T) {
	root := t.TempDir()
	// Negation unsupported: everything ignored, then un-ignored via '!'.
	// Since negation is a no-op, the '*.tmp' rule still ignores tmp files.
	writeFile(t, root, ".gitignore", "*.tmp\n!keep.tmp\n")
	touchFile(t, root, "keep.tmp")

	r, err := NewResolver(root)
	if err != nil {
		t.Fatalf("NewResolver: %v", err)
	}
	assertIgnored(t, r, joinRel(root, "keep.tmp"), false)
}

func TestResolver_UnionOfGitignoreAndAiignore(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, ".gitignore", "*.log\n")
	writeFile(t, root, ".aiignore", "*.bin\n")
	touchFile(t, root, "a.log")
	touchFile(t, root, "b.bin")
	touchFile(t, root, "c.txt")

	r, err := NewResolver(root)
	if err != nil {
		t.Fatalf("NewResolver: %v", err)
	}
	assertIgnored(t, r, joinRel(root, "a.log"), false)
	assertIgnored(t, r, joinRel(root, "b.bin"), false)
	assertNotIgnored(t, r, joinRel(root, "c.txt"), false)
}

// ---------------------------------------------------------------------------
// Resolver.Match direct API (root-relative paths)
// ---------------------------------------------------------------------------

func TestResolver_MatchRootRelativeAPI(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, ".gitignore", "/dist\n*.log\nbuild/\n")
	r, err := NewResolver(root)
	if err != nil {
		t.Fatalf("NewResolver: %v", err)
	}
	cases := []struct {
		rel  string
		dir  bool
		want bool
	}{
		{"dist", false, true},       // anchored root file
		{"sub/dist", false, false},  // anchored -> not nested
		{"app.log", false, true},    // bare name any depth
		{"deep/app.log", false, true},
		{"build", true, true},       // dirOnly matches dir
		{"build", false, false},     // dirOnly not a file
		{"build/x", false, true},    // contents under ignored dir
		{"src/main.go", false, false},
		{".", false, false}, // root itself
		{"", false, false},
	}
	for _, c := range cases {
		if got := r.Match(c.rel, c.dir); got != c.want {
			t.Errorf("Match(%q, dir=%v) = %v, want %v", c.rel, c.dir, got, c.want)
		}
	}
}

// ---------------------------------------------------------------------------
// Multi: root selection, containment, IgnoreChecker
// ---------------------------------------------------------------------------

func TestMulti_RootForSelectsContainingRoot(t *testing.T) {
	alpha := t.TempDir()
	beta := t.TempDir()
	writeFile(t, alpha, ".gitignore", "*.alpha\n")
	writeFile(t, beta, ".gitignore", "*.beta\n")
	touchFile(t, alpha, "x.alpha")
	touchFile(t, beta, "y.beta")

	m, err := NewMulti(alpha, beta)
	if err != nil {
		t.Fatalf("NewMulti: %v", err)
	}

	// Each root's own patterns apply only within that root.
	assertIgnored(t, m, joinRel(alpha, "x.alpha"), false)
	assertNotIgnored(t, m, joinRel(alpha, "y.beta"), false)
	assertIgnored(t, m, joinRel(beta, "y.beta"), false)
	assertNotIgnored(t, m, joinRel(beta, "x.alpha"), false)

	// RootFor returns the matching resolver.
	if got := m.RootFor(joinRel(alpha, "x.alpha")); got == nil || got.Root() != filepath.Clean(alpha) {
		t.Errorf("RootFor(alpha path) = %v, want alpha root", got)
	}
	if got := m.RootFor(joinRel(beta, "y.beta")); got == nil || got.Root() != filepath.Clean(beta) {
		t.Errorf("RootFor(beta path) = %v, want beta root", got)
	}
}

func TestMulti_IgnoredFalseOutsideAllRoots(t *testing.T) {
	alpha := t.TempDir()
	writeFile(t, alpha, ".gitignore", "*.log\n")
	m, err := NewMulti(alpha)
	if err != nil {
		t.Fatalf("NewMulti: %v", err)
	}

	// A path under no known root is never ignored (resolver is path-based, no
	// stat required).
	outside := joinRel(t.TempDir(), "sneaky.log")
	if m.RootFor(outside) != nil {
		t.Error("RootFor should be nil for a path under no known root")
	}
	assertNotIgnored(t, m, outside, false)
}

func TestMulti_BothImplementIgnoreChecker(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, ".gitignore", "*.log\n")
	r, err := NewResolver(root)
	if err != nil {
		t.Fatalf("NewResolver: %v", err)
	}
	m, err := NewMulti(root)
	if err != nil {
		t.Fatalf("NewMulti: %v", err)
	}

	// Compile-time assertion: both concrete types satisfy IgnoreChecker.
	var _ IgnoreChecker = r
	var _ IgnoreChecker = m

	touchFile(t, root, "a.log")
	assertIgnored(t, r, joinRel(root, "a.log"), false)
	assertIgnored(t, m, joinRel(root, "a.log"), false)
}

func TestNewMulti_RequiresAtLeastOneRoot(t *testing.T) {
	if _, err := NewMulti(); err == nil {
		t.Fatal("NewMulti() should error without roots")
	}
}
