package tools

import (
	"fmt"
	"strings"
)

// ToolNotFoundResult builds an error result for a name that failed exact lookup.
// It suggests removing only the functions. prefix, and only when available
// confirms that the candidate is available to the caller. It never dispatches
// the candidate; callers must perform exact lookup before using this helper.
func ToolNotFoundResult(name string, available func(string) bool) ToolResult {
	result := ToolResult{Content: "tool not found: " + name, IsError: true}
	candidate, prefixed := strings.CutPrefix(name, "functions.")
	if prefixed && candidate != "" && available != nil && available(candidate) {
		result.Content += fmt.Sprintf("; use the exact catalog name %q in batch.calls[].tool or as the direct tool call name, without adding the functions. namespace", candidate)
	}
	return result
}
