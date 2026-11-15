package pathutil

import "testing"

func TestCollapseDuplicateComponents(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
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
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := CollapseDuplicateComponents(tt.in)
			if got != tt.want {
				t.Errorf("CollapseDuplicateComponents(%q) = %q, want %q", tt.in, got, tt.want)
			}
			// The repair is single-pass idempotent: collapsing the result
			// again must not change it further.
			if again := CollapseDuplicateComponents(got); again != got {
				t.Errorf("repair not stable: CollapseDuplicateComponents(%q) = %q", got, again)
			}
		})
	}
}
