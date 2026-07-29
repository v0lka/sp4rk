package tools

import "testing"

func TestIsShellExecTool(t *testing.T) {
	tests := []struct {
		name string
		want bool
	}{
		{ToolBashExec, true},
		{ToolPoshExec, true},
		{ToolBatch, false},      // meta-tool, not a shell executor
		{ToolReadFile, false},   // file tool
		{ToolWebFetch, false},   // web tool
		{"unknown_tool", false}, // unregistered
		{"", false},             // empty
	}
	for _, tt := range tests {
		if got := IsShellExecTool(tt.name); got != tt.want {
			t.Errorf("IsShellExecTool(%q) = %v, want %v", tt.name, got, tt.want)
		}
	}
}
