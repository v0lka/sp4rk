package main

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/v0lka/sp4rk/tools"
)

func TestCustomToolsDeclareCapabilityGroups(t *testing.T) {
	if got := tools.ToolGroupOf(newFetchWebpageTool()); got != tools.GroupRemoteRead {
		t.Fatalf("fetch_webpage group = %q, want %q", got, tools.GroupRemoteRead)
	}

	if got := tools.ToolGroupOf(newAppendLogTool(t.TempDir())); got != tools.GroupLocalWrite {
		t.Fatalf("append_log group = %q, want %q", got, tools.GroupLocalWrite)
	}
}

func TestStrictJudgeWithoutProviderFailsSafeWithoutNetwork(t *testing.T) {
	judge := tools.NewToolJudge(nil, "", 0, nil)
	verdict, reason, err := judge.JudgeStrict(context.Background(), tools.StrictJudgeRequest{
		ToolName:    "append_log",
		Input:       json.RawMessage(`{"path":"/outside","line":"blocked"}`),
		TaskContext: "append inside the workspace",
		ToolSource:  "core",
	})
	if err != nil {
		t.Fatalf("JudgeStrict: %v", err)
	}
	if verdict != tools.VerdictConfirm {
		t.Fatalf("verdict = %v, want VerdictConfirm", verdict)
	}
	if !strings.Contains(strings.ToLower(reason), "manual confirmation") {
		t.Fatalf("reason = %q, want manual-confirmation fail-safe", reason)
	}
}
