package builtins

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/v0lka/sp4rk/tools"
)

// TestFactsTools_RegistryAcceptsCommaSeparatedKeywords verifies that the
// default execution path (ToolRegistry.Execute, which now validates input
// before dispatch) accepts the comma-separated *string* form of "keywords".
// The tools deliberately coerce that form in their own decoders, so the
// schema must not reject it as a type error.
func TestFactsTools_RegistryAcceptsCommaSeparatedKeywords(t *testing.T) {
	tests := []struct {
		name string
		tool tools.Tool
		in   string
	}{
		{
			name: "store_fact",
			tool: NewStoreFactTool(),
			in:   `{"keywords":"auth, middleware, policy","content":"x"}`,
		},
		{
			name: "search_facts",
			tool: NewSearchFactsTool(),
			in:   `{"keywords":"auth, config"}`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reg := tools.NewToolRegistry()
			reg.Register(tt.tool)

			res, err := reg.Execute(context.Background(), tt.tool.Name(), json.RawMessage(tt.in))
			if err != nil {
				t.Fatalf("unexpected Go error: %v", err)
			}
			if strings.Contains(res.Content, "must be of type") {
				t.Fatalf("string keywords form must pass validation, got %q", res.Content)
			}
			// No fact store is attached in this context, so the tool's own
			// "not available" path proves validation let the call through.
			if !strings.Contains(res.Content, "Fact store not available") {
				t.Fatalf("expected the call to reach the tool body, got %q", res.Content)
			}
		})
	}
}
