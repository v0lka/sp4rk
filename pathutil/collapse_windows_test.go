//go:build windows

package pathutil

import (
	"slices"
	"testing"
)

// TestCollapseDuplicateComponents_WindowsSeparators covers the Windows-only
// spelling rules the shared vectors cannot reach: "\" and "/" are BOTH
// separators, the volume prefix is preserved verbatim (drive letter or UNC
// share), a repaired spelling keeps the separator style it was written with,
// and a volume-relative spelling is never rebuilt (its components do not carry
// the meaning the caller wrote).
//
// The regression this pins: splitting on filepath.Separator alone left a
// forward-slash spelling as one opaque component, so "/a/b/a/b/c" and
// "C:/ws/project/project/src/main.go" came back unrepaired on Windows while
// their backslash forms collapsed.
func TestCollapseDuplicateComponents_WindowsSeparators(t *testing.T) {
	tests := []collapseCase{
		{
			name: "backslash doubled prefix",
			in:   `C:\ws\project\project\src\main.go`,
			want: `C:\ws\project\src\main.go`,
		},
		{
			name: "forward-slash doubled prefix keeps forward slashes",
			in:   `C:/ws/project/project/src/main.go`,
			want: `C:/ws/project/src/main.go`,
		},
		{
			name: "mixed separators collapse and keep the root separator",
			in:   `C:\ws\project/project\src/main.go`,
			want: `C:\ws\project\src\main.go`,
		},
		{
			name: "current-drive rooted forward-slash spelling",
			in:   `/a/b/a/b/c`,
			want: `/a/b/c`,
		},
		{
			name: "current-drive rooted backslash spelling",
			in:   `\a\b\a\b\c`,
			want: `\a\b\c`,
		},
		{
			name: "UNC share preserved",
			in:   `\\server\share\ws\ws\f.txt`,
			want: `\\server\share\ws\f.txt`,
		},
		{
			name: "UNC share with forward slashes preserved",
			in:   `//server/share/ws/ws/f.txt`,
			want: `//server/share/ws/f.txt`,
		},
		{
			name: "UNC root unchanged",
			in:   `\\server\share`,
			want: `\\server\share`,
		},
		{
			name: "volume-relative spelling unchanged",
			in:   `C:ws\ws\f.txt`,
			want: `C:ws\ws\f.txt`,
		},
		{
			name: "drive root unchanged",
			in:   `C:\`,
			want: `C:\`,
		},
		{
			name: "no duplication keeps spelling byte-for-byte",
			in:   `C:\ws\project\src\main.go`,
			want: `C:\ws\project\src\main.go`,
		},
		{
			name: "dot component suppresses collapsing",
			in:   `C:\ws\..\ws\ws\f.txt`,
			want: `C:\ws\..\ws\ws\f.txt`,
		},
	}
	runCollapseCases(t, tests)
}

// TestSplitPathComponents_WindowsSeparators pins the split rule that the repair
// above depends on: Windows accepts "/" as well as "\", so a forward-slash
// spelling must yield exactly the components its backslash form does instead
// of one opaque component.
func TestSplitPathComponents_WindowsSeparators(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want []string
	}{
		{
			name: "backslash drive path",
			in:   `C:\Users\alice\file.txt`,
			want: []string{"Users", "alice", "file.txt"},
		},
		{
			name: "forward-slash drive path",
			in:   `C:/Users/alice/file.txt`,
			want: []string{"Users", "alice", "file.txt"},
		},
		{
			name: "mixed separators",
			in:   `C:\Users/alice\file.txt`,
			want: []string{"Users", "alice", "file.txt"},
		},
		{
			name: "current-drive rooted",
			in:   `/a/b`,
			want: []string{"a", "b"},
		},
		{
			name: "UNC path",
			in:   `\\server\share\a\b`,
			want: []string{"a", "b"},
		},
		{
			name: "forward-slash dot-git component stays its own component",
			in:   `C:/ws/repo/.git/config`,
			want: []string{"ws", "repo", ".git", "config"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := SplitPathComponents(tt.in)
			if !slices.Equal(got, tt.want) {
				t.Errorf("SplitPathComponents(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}
