package builtins

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/v0lka/sp4rk/tools"
)

// Severity taxonomy (tools.JudgeOutcome):
//   - hard: security-control triggers — shell blacklist matches, SSRF
//     (private/reserved targets, degraded SSRF protection, unassessable URLs)
//   - soft: advisory path-containment escalations (file tools, shell
//     out-of-root path references)

func TestWebFetchTool_JudgeSeverity_SSRFIsHard(t *testing.T) {
	tool := NewWebFetchTool(WebFetchLimits{})
	input, _ := json.Marshal(map[string]string{"url": "http://127.0.0.1/secret"})
	outcome := tool.Judge(context.Background(), input)
	if outcome.Allow {
		t.Fatal("expected private-address fetch to be denied")
	}
	if outcome.Severity != tools.JudgeSeverityHard {
		t.Fatalf("SSRF severity = %v, want hard", outcome.Severity)
	}
}

func TestWebFetchTool_JudgeSeverity_UnassessableURLIsHard(t *testing.T) {
	tool := NewWebFetchTool(WebFetchLimits{})
	outcome := tool.Judge(context.Background(), json.RawMessage(`{invalid`))
	if outcome.Allow {
		t.Fatal("expected unparseable input to be denied (fail closed)")
	}
	if outcome.Severity != tools.JudgeSeverityHard {
		t.Fatalf("unassessable URL severity = %v, want hard", outcome.Severity)
	}
}

func TestWebFetchTool_JudgeSeverity_PublicAllow(t *testing.T) {
	tool := NewWebFetchTool(WebFetchLimits{})
	input, _ := json.Marshal(map[string]string{"url": "https://example.com/"})
	outcome := tool.Judge(context.Background(), input)
	if !outcome.Allow {
		t.Fatalf("expected public fetch to be allowed, got reason %q", outcome.Reason)
	}
}

func TestWriteFileTool_JudgeSeverity_ContainmentIsSoft(t *testing.T) {
	tool := NewWriteFileTool()
	ws := t.TempDir()
	other := t.TempDir()
	ctx := tools.WithWorkspacePath(context.Background(), ws)
	input, _ := json.Marshal(WriteFileInput{Path: other + "/out.txt", Content: "x"})
	outcome := tool.Judge(ctx, input)
	if outcome.Allow {
		t.Fatal("expected out-of-roots write to be denied")
	}
	if outcome.Severity != tools.JudgeSeveritySoft {
		t.Fatalf("containment severity = %v, want soft", outcome.Severity)
	}
}

func TestReadFileTool_JudgeSeverity_ContainmentIsSoft(t *testing.T) {
	tool := NewReadFileTool()
	ws := t.TempDir()
	other := t.TempDir()
	ctx := tools.WithWorkspacePath(context.Background(), ws)
	input, _ := json.Marshal(map[string]string{"path": other + "/file.txt"})
	outcome := tool.Judge(ctx, input)
	if outcome.Allow {
		t.Fatal("expected out-of-roots read to be denied")
	}
	if outcome.Severity != tools.JudgeSeveritySoft {
		t.Fatalf("containment severity = %v, want soft", outcome.Severity)
	}
}

func TestDeleteFileTool_JudgeSeverity_ContainmentIsSoft(t *testing.T) {
	tool := NewDeleteFileTool()
	ws := t.TempDir()
	other := t.TempDir()
	ctx := tools.WithWorkspacePath(context.Background(), ws)
	input, _ := json.Marshal(DeleteFileInput{Path: other + "/gone.txt"})
	outcome := tool.Judge(ctx, input)
	if outcome.Allow {
		t.Fatal("expected out-of-roots delete to be denied")
	}
	if outcome.Severity != tools.JudgeSeveritySoft {
		t.Fatalf("containment severity = %v, want soft", outcome.Severity)
	}
}

// TestFileToolGroupsByCapability verifies the group taxonomy of the file
// builtins directly (the full all-builtins iteration lives in the root
// sp4rk package; the shell tools are covered by platform-tagged tests).
func TestFileToolGroupsByCapability(t *testing.T) {
	tests := []struct {
		tool tools.Tool
		want tools.ToolGroup
	}{
		{NewReadFileTool(), tools.GroupLocalRead},
		{NewListDirectoryTool(), tools.GroupLocalRead},
		{NewGlobTool(), tools.GroupLocalRead},
		{NewRipgrepTool(), tools.GroupLocalRead},
		{NewWriteFileTool(), tools.GroupLocalWrite},
		{NewEditFileTool(), tools.GroupLocalWrite},
		{NewCreateDirectoryTool(), tools.GroupLocalWrite},
		{NewDeleteFileTool(), tools.GroupLocalWrite},
		{NewDeleteDirectoryTool(), tools.GroupLocalWrite},
	}
	for _, tt := range tests {
		if got := tools.ToolGroupOf(tt.tool); got != tt.want {
			t.Errorf("tool %q group = %q, want %q", tt.tool.Name(), got, tt.want)
		}
	}
}
