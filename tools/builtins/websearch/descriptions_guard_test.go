package websearch

import (
	"strings"
	"testing"
)

// maxDescriptionLength mirrors the guard limit used by tools/builtins.
const maxDescriptionLength = 1200

// rubricSections is the shared description rubric
// (purpose -> when-to-use -> inputs -> outputs -> example -> anti-example).
// Every built-in tool must carry all six labels so host UIs that render tool
// descriptions format them identically instead of falling back to a single
// unlabelled paragraph.
var rubricSections = []string{"Purpose:", "Use when:", "Inputs:", "Outputs:", "Example:", "Anti-example:"}

func TestWebSearchDescriptionWithinGuardLimit(t *testing.T) {
	tool := NewTool(nil, Limits{})
	desc := tool.Description()
	if strings.TrimSpace(desc) == "" {
		t.Fatal("web_search description is empty")
	}
	if len(desc) > maxDescriptionLength {
		t.Fatalf("web_search description is %d chars, exceeds guard limit %d", len(desc), maxDescriptionLength)
	}
	for _, section := range rubricSections {
		if !strings.Contains(desc, section) {
			t.Errorf("web_search description lacks %q section", section)
		}
	}
	// Every rubric section must begin its own line, otherwise a line-based
	// markdown reformatter cannot split it into bold-labelled paragraphs.
	for _, line := range rubricSections {
		if !strings.Contains(desc, "\n"+line) && !strings.HasPrefix(desc, line) {
			t.Errorf("web_search description: %q does not start a line — the markdown reformatter needs one label per line", line)
		}
	}
}
