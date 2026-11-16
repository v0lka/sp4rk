package pathutil

import (
	"path/filepath"
	"testing"
)

// collapseCase is one repair vector: the spelling in, the expected spelling out.
type collapseCase struct {
	name string
	in   string
	want string
}

func TestCollapseDuplicateComponents(t *testing.T) {
	// These vectors are written with "/" roots because the repair preserves
	// the separator style it is given: on Windows a "/"-rooted spelling is a
	// current-drive-rooted path, and its repair must come back "/"-rooted
	// too. Windows-only spellings (drive letters, UNC shares, mixed
	// separators) are covered in collapse_windows_test.go.
	tests := []collapseCase{
		{
			name: "doubled leading run",
			in:   "/a/b/a/b/c",
			want: "/a/b/c",
		},
		{
			name: "adjacent repeat",
			in:   "/x/x/f.txt",
			want: "/x/f.txt",
		},
		{
			name: "triple run collapses in one pass",
			in:   "/a/a/a",
			want: "/a",
		},
		{
			name: "deep doubled prefix",
			in:   "/r1/r2/r1/r2/deep/file.txt",
			want: "/r1/r2/deep/file.txt",
		},
		{
			name: "interior adjacent repeat",
			in:   "/a/b/b/a/c",
			want: "/a/b/a/c",
		},
		{
			name: "no duplication keeps spelling byte-for-byte",
			in:   "/a/b/c",
			want: "/a/b/c",
		},
		{
			name: "no duplication keeps odd separator spellings untouched",
			in:   "//weird//spelling",
			want: "//weird//spelling",
		},
		{
			name: "filesystem root unchanged",
			in:   "/",
			want: "/",
		},
		{
			name: "empty path unchanged",
			in:   "",
			want: "",
		},
		{
			name: "relative path unchanged (would change meaning)",
			in:   "rel/x/x/f",
			want: "rel/x/x/f",
		},
		{
			name: "single component unchanged",
			in:   "/only",
			want: "/only",
		},
		{
			name: "doubled leaf directories",
			in:   "/ws/project/project/src/main.go",
			want: "/ws/project/src/main.go",
		},
		{
			name: "dot-suppressed: adjacent repeat before a .. climb keeps spelling",
			in:   "/a/a/../a",
			want: "/a/a/../a",
		},
		{
			name: "dot-suppressed: adjacent repeat after a .. climb keeps spelling",
			in:   "/a/b/../b/b/f",
			want: "/a/b/../b/b/f",
		},
		{
			name: "dot-suppressed: single dot also suppresses collapsing",
			in:   "/x/./x/x/f.txt",
			want: "/x/./x/x/f.txt",
		},
	}
	runCollapseCases(t, tests)
}

// TestCollapseDuplicateComponents_NativeSeparator repeats the headline shapes
// with the OS's own separator, so the native spelling of a doubled prefix is
// exercised everywhere rather than only on the leg whose separator happens to
// be "/" (on Windows these are backslash paths; on unix they coincide with the
// "/" vectors above).
func TestCollapseDuplicateComponents_NativeSeparator(t *testing.T) {
	sep := string(filepath.Separator)
	tests := []collapseCase{
		{
			name: "doubled native prefix",
			in:   sep + "a" + sep + "b" + sep + "a" + sep + "b" + sep + "c",
			want: sep + "a" + sep + "b" + sep + "c",
		},
		{
			name: "adjacent native repeat",
			in:   sep + "ws" + sep + "project" + sep + "project" + sep + "main.go",
			want: sep + "ws" + sep + "project" + sep + "main.go",
		},
		{
			name: "native spelling without duplication unchanged",
			in:   sep + "ws" + sep + "project" + sep + "main.go",
			want: sep + "ws" + sep + "project" + sep + "main.go",
		},
	}
	runCollapseCases(t, tests)
}

// runCollapseCases asserts each repair and its single-pass idempotence:
// collapsing the result again must not change it further.
func runCollapseCases(t *testing.T, tests []collapseCase) {
	t.Helper()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := CollapseDuplicateComponents(tt.in)
			if got != tt.want {
				t.Errorf("CollapseDuplicateComponents(%q) = %q, want %q", tt.in, got, tt.want)
			}
			if again := CollapseDuplicateComponents(got); again != got {
				t.Errorf("repair not stable: CollapseDuplicateComponents(%q) = %q", got, again)
			}
		})
	}
}
