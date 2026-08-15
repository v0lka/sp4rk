package tools

import "testing"

// TestAllToolGroups_DeclaredExactlyOnce verifies the declared taxonomy: eight
// groups, no duplicates, no "unknown" value (every tool must declare a real
// group).
func TestAllToolGroups_DeclaredExactlyOnce(t *testing.T) {
	all := AllToolGroups()
	if len(all) != 8 {
		t.Fatalf("expected exactly 8 declared groups, got %d: %v", len(all), all)
	}
	seen := make(map[ToolGroup]bool, len(all))
	for _, g := range all {
		if g == "" {
			t.Fatalf("empty group value in AllToolGroups: %v", all)
		}
		if seen[g] {
			t.Fatalf("duplicate group %q in AllToolGroups", g)
		}
		seen[g] = true
		if !IsValidToolGroup(g) {
			t.Fatalf("group %q returned by AllToolGroups but rejected by IsValidToolGroup", g)
		}
	}
}

func TestIsValidToolGroup(t *testing.T) {
	for _, g := range []ToolGroup{
		GroupExecute,
		GroupLocalRead,
		GroupLocalWrite,
		GroupRemoteRead,
		GroupRemoteWrite,
		GroupSystem,
		GroupLocalMCP,
		GroupRemoteMCP,
	} {
		if !IsValidToolGroup(g) {
			t.Errorf("IsValidToolGroup(%q) = false, want true", g)
		}
	}
	for _, g := range []ToolGroup{"", "unknown", "Unknown", "mcp", "read", "web"} {
		if IsValidToolGroup(g) {
			t.Errorf("IsValidToolGroup(%q) = true, want false", g)
		}
	}
}

// TestMCPToolGroup_ByTransport pins the transport → group mapping used by the
// MCP gateway: stdio servers run locally, http servers are remote, and an
// unspecified transport defaults to stdio/local (mirroring Server.Connect).
func TestMCPToolGroup_ByTransport(t *testing.T) {
	tests := []struct {
		transport string
		want      ToolGroup
	}{
		{"stdio", GroupLocalMCP},
		{"", GroupLocalMCP},
		{"http", GroupRemoteMCP},
	}
	for _, tt := range tests {
		if got := MCPToolGroup(tt.transport); got != tt.want {
			t.Errorf("MCPToolGroup(%q) = %q, want %q", tt.transport, got, tt.want)
		}
	}
}

// TestJudgeSeverity_String verifies severity naming used in logs/UI.
func TestJudgeSeverity_String(t *testing.T) {
	if JudgeSeveritySoft.String() != "soft" {
		t.Errorf("JudgeSeveritySoft.String() = %q, want %q", JudgeSeveritySoft.String(), "soft")
	}
	if JudgeSeverityHard.String() != "hard" {
		t.Errorf("JudgeSeverityHard.String() = %q, want %q", JudgeSeverityHard.String(), "hard")
	}
}

// TestBaseTool_Group_ZeroValueUndeclared documents the fail-closed behaviour
// for a tool that forgot to declare its group: the zero value is not a valid
// group and matches no group-based allow-list. (All shipped tools declare one;
// this keeps the invariant checkable.)
func TestBaseTool_Group_ZeroValueUndeclared(t *testing.T) {
	bt := &BaseTool{ToolName: "sloppy"}
	if got := bt.Group(); got != "" || IsValidToolGroup(got) {
		t.Fatalf("undeclared BaseTool group = %q (valid=%v), want empty/invalid", got, IsValidToolGroup(got))
	}
	bt = &BaseTool{ToolName: "declared", ToolGroup: GroupExecute}
	if got := bt.Group(); got != GroupExecute {
		t.Fatalf("BaseTool.Group() = %q, want %q", got, GroupExecute)
	}
}
