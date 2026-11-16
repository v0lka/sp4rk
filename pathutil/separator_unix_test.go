//go:build !windows

package pathutil

import (
	"slices"
	"testing"
)

// TestSeparators_BackslashIsOrdinaryByte pins the unix half of the platform's
// separator rules: only "/" separates components, so a backslash inside a name
// — legal in a unix filename — must survive both the component split and the
// duplicate-collapse repair. Windows additionally accepts "\" as a separator;
// that half lives in collapse_windows_test.go.
func TestSeparators_BackslashIsOrdinaryByte(t *testing.T) {
	// A directory literally named `a\b` is ONE component.
	if got, want := SplitPathComponents(`/dir/a\b/c`), []string{"dir", `a\b`, "c"}; !slices.Equal(got, want) {
		t.Errorf("SplitPathComponents(%q) = %q, want %q", `/dir/a\b/c`, got, want)
	}

	// Two identical `a\b` components ARE an adjacent repeat, so the repair
	// applies — and it must compare whole components, treating the backslash
	// as content: were "\" a separator here, the components would be
	// ["dir","a","b","a","b"], hold no repeat, and the spelling would come
	// back unchanged.
	if got, want := CollapseDuplicateComponents(`/dir/a\b/a\b`), `/dir/a\b`; got != want {
		t.Errorf("CollapseDuplicateComponents(%q) = %q, want %q", `/dir/a\b/a\b`, got, want)
	}

	// A Windows spelling is a plain relative path on unix: it is neither
	// rooted nor duplicated, so it must be returned byte-for-byte unchanged.
	const windowsSpelling = `C:\ws\ws\f.txt`
	if got := SplitPathComponents(windowsSpelling); !slices.Equal(got, []string{windowsSpelling}) {
		t.Errorf("SplitPathComponents(%q) = %q, want the whole string as one component", windowsSpelling, got)
	}
	if got := CollapseDuplicateComponents(windowsSpelling); got != windowsSpelling {
		t.Errorf("CollapseDuplicateComponents(%q) = %q, want unchanged", windowsSpelling, got)
	}
}
