package websearch

import (
	"strings"
	"testing"
)

// maxDescriptionLength mirrors the guard limit used by tools/builtins.
const maxDescriptionLength = 1200

func TestWebSearchDescriptionWithinGuardLimit(t *testing.T) {
	tool := NewTool(nil, Limits{})
	desc := tool.Description()
	if strings.TrimSpace(desc) == "" {
		t.Fatal("web_search description is empty")
	}
	if len(desc) > maxDescriptionLength {
		t.Fatalf("web_search description is %d chars, exceeds guard limit %d", len(desc), maxDescriptionLength)
	}
	for _, section := range []string{"Purpose:", "Use when:", "Inputs:", "Anti-example:"} {
		if !strings.Contains(desc, section) {
			t.Errorf("web_search description lacks %q section", section)
		}
	}
}
