package specslint

// Guard against ADR numbering conflicts in specs/decisions/.
//
// META.md fixes the ADR file contract: `NNN-slug.md` — a three-digit number
// and a kebab-case slug. Two files claiming the same number silently fork the
// identifier space: every "ADR-NNN" citation in specs, AGENTS.md and code
// comments becomes ambiguous, and the later file loses its cross references.
// This guard fails the suite on any collision, on a malformed file name, and
// on an H1 title whose number disagrees with the file name.

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strings"
	"testing"
)

var (
	adrFileNameRe = regexp.MustCompile(`^(\d{3})-([a-z0-9-]+)\.md$`)
	adrTitleRe    = regexp.MustCompile(`^# ADR-(\d{3}):`)
)

func repoRoot(t *testing.T) string {
	t.Helper()
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("cannot locate the specslint guard source")
	}
	return filepath.Clean(filepath.Join(filepath.Dir(file), "..", ".."))
}

func firstLine(data []byte) string {
	if i := strings.IndexByte(string(data), '\n'); i >= 0 {
		return string(data[:i])
	}
	return string(data)
}

func TestADRNumberingContract(t *testing.T) {
	dir := filepath.Join(repoRoot(t), "specs", "decisions")
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("read specs/decisions: %v", err)
	}

	hasTemplate := false
	numbers := map[string][]string{} // ADR number -> file names claiming it
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasSuffix(name, ".md") {
			t.Errorf("specs/decisions/%s: only ADR markdown files (NNN-slug.md per specs/META.md) and _template.md belong here", name)
			continue
		}
		if name == "_template.md" {
			hasTemplate = true
			continue
		}
		m := adrFileNameRe.FindStringSubmatch(name)
		if m == nil {
			t.Errorf("specs/decisions/%s: file name violates the META.md contract `NNN-slug.md` (three-digit number, kebab-case slug)", name)
			continue
		}
		numbers[m[1]] = append(numbers[m[1]], name)

		data, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			t.Fatalf("read specs/decisions/%s: %v", name, err)
		}
		tm := adrTitleRe.FindStringSubmatch(strings.TrimSpace(firstLine(data)))
		if tm == nil {
			t.Errorf("specs/decisions/%s: first line must be an `# ADR-%s:` heading", name, m[1])
			continue
		}
		if tm[1] != m[1] {
			t.Errorf("specs/decisions/%s: H1 says ADR-%s but the file number is %s — one of them is stale", name, tm[1], m[1])
		}
	}
	if !hasTemplate {
		t.Fatalf("specs/decisions/_template.md is missing — the guard resolved the wrong root or the template was removed")
	}
	if len(numbers) == 0 {
		t.Fatalf("no ADR files parsed from specs/decisions — the guard's path resolution is broken")
	}

	var conflicts []string
	for number, files := range numbers {
		if len(files) > 1 {
			sort.Strings(files)
			conflicts = append(conflicts, fmt.Sprintf("  ADR-%s claimed by: %s", number, strings.Join(files, ", ")))
		}
	}
	if len(conflicts) > 0 {
		sort.Strings(conflicts)
		t.Fatalf("ADR numbering conflict — several decisions claim the same number; keep the number for the file that landed first, give the later one the next free number, and update every reference to it (path links and bare ADR-NNN mentions):\n%s",
			strings.Join(conflicts, "\n"))
	}
}
