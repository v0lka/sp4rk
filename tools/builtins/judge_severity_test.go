package builtins

import (
	"context"
	"encoding/json"
	"fmt"
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

// TestJudgeReasonCodes_ClassificationTable pins the severity↔reason-code
// pairing for every deterministic escalation branch of the cross-platform
// built-in judges. Reason codes are a cross-repository contract (see
// tools.JudgeReasonCode): prose may be reworded freely, these pairs may not
// drift. The shell judges are covered by platform-tagged counterparts
// (bash_judge_severity_test.go / posh_judge_severity_windows_test.go); the
// SSRF-degraded branch (tools.ReasonCodeSSRFDegraded) needs a failed CIDR
// initialization and is not deterministically triggerable here.
func TestJudgeReasonCodes_ClassificationTable(t *testing.T) {
	ws := t.TempDir()
	other := t.TempDir()
	ctx := tools.WithWorkspacePath(context.Background(), ws)

	tests := []struct {
		name     string
		judger   tools.ToolJudger
		input    string
		wantSev  tools.JudgeSeverity
		wantCode tools.JudgeReasonCode
	}{
		{
			name:     "webfetch private address is hard ssrf_private_address",
			judger:   NewWebFetchTool(WebFetchLimits{}),
			input:    `{"url":"http://127.0.0.1/secret"}`,
			wantSev:  tools.JudgeSeverityHard,
			wantCode: tools.ReasonCodeSSRFPrivateAddress,
		},
		{
			name:     "webfetch unparseable input is hard unassessable_url",
			judger:   NewWebFetchTool(WebFetchLimits{}),
			input:    `{invalid`,
			wantSev:  tools.JudgeSeverityHard,
			wantCode: tools.ReasonCodeUnassessableURL,
		},
		{
			name:     "read_file missing path is hard unassessable_path",
			judger:   NewReadFileTool(),
			input:    `{}`,
			wantSev:  tools.JudgeSeverityHard,
			wantCode: tools.ReasonCodeUnassessablePath,
		},
		{
			name:     "read_file out-of-roots absolute is soft outside_session_roots",
			judger:   NewReadFileTool(),
			input:    fmt.Sprintf(`{"path":%q}`, other+"/file.txt"),
			wantSev:  tools.JudgeSeveritySoft,
			wantCode: tools.ReasonCodeOutsideSessionRoots,
		},
		{
			name:     "read_file relative escape is soft outside_session_roots",
			judger:   NewReadFileTool(),
			input:    `{"path":"../escape.txt"}`,
			wantSev:  tools.JudgeSeveritySoft,
			wantCode: tools.ReasonCodeOutsideSessionRoots,
		},
		{
			name:     "write_file out-of-roots absolute is soft outside_session_roots",
			judger:   NewWriteFileTool(),
			input:    fmt.Sprintf(`{"path":%q}`, other+"/out.txt"),
			wantSev:  tools.JudgeSeveritySoft,
			wantCode: tools.ReasonCodeOutsideSessionRoots,
		},
		{
			name:     "write_file relative escape is soft outside_session_roots",
			judger:   NewWriteFileTool(),
			input:    `{"path":"../escape.txt"}`,
			wantSev:  tools.JudgeSeveritySoft,
			wantCode: tools.ReasonCodeOutsideSessionRoots,
		},
		{
			name:     "edit_file relative escape is soft outside_session_roots",
			judger:   NewEditFileTool(),
			input:    `{"path":"../escape.txt"}`,
			wantSev:  tools.JudgeSeveritySoft,
			wantCode: tools.ReasonCodeOutsideSessionRoots,
		},
		{
			name:     "create_directory relative escape is soft outside_session_roots",
			judger:   NewCreateDirectoryTool(),
			input:    `{"path":"../escape"}`,
			wantSev:  tools.JudgeSeveritySoft,
			wantCode: tools.ReasonCodeOutsideSessionRoots,
		},
		{
			name:     "delete_file relative escape is soft outside_session_roots",
			judger:   NewDeleteFileTool(),
			input:    `{"path":"../escape.txt"}`,
			wantSev:  tools.JudgeSeveritySoft,
			wantCode: tools.ReasonCodeOutsideSessionRoots,
		},
		{
			name:     "delete_directory relative escape is soft outside_session_roots",
			judger:   NewDeleteDirectoryTool(),
			input:    `{"path":"../escape"}`,
			wantSev:  tools.JudgeSeveritySoft,
			wantCode: tools.ReasonCodeOutsideSessionRoots,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outcome := tt.judger.Judge(ctx, json.RawMessage(tt.input))
			if outcome.Allow {
				t.Fatalf("expected escalation (allow=false), got %+v", outcome)
			}
			if outcome.Reason == "" {
				t.Fatal("expected non-empty escalation reason")
			}
			if outcome.Severity != tt.wantSev {
				t.Errorf("severity = %v, want %v", outcome.Severity, tt.wantSev)
			}
			if outcome.ReasonCode != tt.wantCode {
				t.Errorf("reason code = %q, want %q", outcome.ReasonCode, tt.wantCode)
			}
		})
	}
}
