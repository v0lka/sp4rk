package builtins

import (
	"strings"
	"testing"

	sdktools "github.com/v0lka/sp4rk/tools"
)

// maxDescriptionLength is the generous guard for builtin tool descriptions.
// Descriptions follow the rubric purpose -> when-to-use -> inputs -> outputs ->
// example -> anti-example (~150-250 tokens); anything beyond this limit means
// marketing prose crept back in.
const maxDescriptionLength = 1200

// builtinTools returns every builtin tool registered by this package on the
// current platform, so the guard cannot drift out of sync with registrations.
// The platform shell tool (bash_exec on Unix, posh_exec on Windows) is
// contributed by platformShellTools in the build-tagged files.
func builtinTools(t *testing.T) []sdktools.Tool {
	t.Helper()

	vecTool := NewVectorSearchTool(nil, nil)

	return append([]sdktools.Tool{
		NewReadFileTool(),
		NewWriteFileTool(),
		NewEditFileTool(),
		NewListDirectoryTool(),
		NewCreateDirectoryTool(),
		NewDeleteFileTool(),
		NewDeleteDirectoryTool(),
		NewGlobTool(),
		NewRipgrepTool(),
		NewWebFetchTool(WebFetchLimits{}),
		vecTool,
		NewBatchTool(),
		NewToolResultReadTool(),
		NewReadStepOutputTool(),
		NewListStepOutputsTool(),
		NewReadFinalResultTool(),
		NewUpdateChecklistTool(),
		NewStoreFactTool(),
		NewSearchFactsTool(),
		NewReadAttachmentTool(),
	}, platformShellTools()...)
}

func TestBuiltinDescriptionsWithinGuardLimit(t *testing.T) {
	tools := builtinTools(t)
	if len(tools) < 20 {
		t.Fatalf("expected at least 20 builtin tools in the guard, got %d — registration drift?", len(tools))
	}
	for _, tool := range tools {
		desc := tool.Description()
		if strings.TrimSpace(desc) == "" {
			t.Errorf("tool %s: description is empty", tool.Name())
		}
		if len(desc) > maxDescriptionLength {
			t.Errorf("tool %s: description is %d chars, exceeds guard limit %d — trim it to the rubric (purpose/when-to-use/inputs/outputs/example/anti-example)",
				tool.Name(), len(desc), maxDescriptionLength)
		}
	}
}

// TestBuiltinDescriptionsFollowRubric guards the structural rubric: every
// description must state its purpose and steer away from neighboring tools
// (anti-example), which is what disambiguates conflicting tool pairs.
func TestBuiltinDescriptionsFollowRubric(t *testing.T) {
	for _, tool := range builtinTools(t) {
		desc := tool.Description()
		if !strings.Contains(desc, "Purpose:") {
			t.Errorf("tool %s: description lacks a 'Purpose:' section", tool.Name())
		}
		if !strings.Contains(desc, "Use when:") {
			t.Errorf("tool %s: description lacks a 'Use when:' section", tool.Name())
		}
		if !strings.Contains(desc, "Inputs:") {
			t.Errorf("tool %s: description lacks an 'Inputs:' section", tool.Name())
		}
		if !strings.Contains(desc, "Anti-example:") {
			t.Errorf("tool %s: description lacks an 'Anti-example:' section", tool.Name())
		}
	}
}
