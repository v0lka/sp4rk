// SPDX-License-Identifier: Apache-2.0

package tools

import (
	"encoding/json"
	"strings"
	"testing"
)

// ValidateToolInputForCategory exposes the category-dependent key-admission
// default the registry applies: built-in (SourceCategoryCore) schemas are
// CLOSED by default, MCP-proxied (SourceCategoryMCP) schemas are OPEN by
// default. The plain ValidateToolInput always applies the strict built-in
// (closed) default.
const categoryTestSchema = `{
	"type": "object",
	"properties": {"path": {"type": "string"}},
	"required": ["path"]
}`

func TestValidateToolInputForCategory_MCPIsOpenByDefault(t *testing.T) {
	input := json.RawMessage(`{"path":"/ws/file.txt","server_extra":true}`)
	if err := ValidateToolInputForCategory("mcp_fetch", json.RawMessage(categoryTestSchema), input, SourceCategoryMCP); err != nil {
		t.Fatalf("MCP-proxied schema must admit extra keys (open default): %v", err)
	}
	// The strict exported helper stays closed regardless of the caller.
	if err := ValidateToolInput("mcp_fetch", json.RawMessage(categoryTestSchema), input); err == nil {
		t.Fatal("ValidateToolInput must stay closed (built-in default)")
	}
}

func TestValidateToolInputForCategory_BuiltinIsClosedByDefault(t *testing.T) {
	input := json.RawMessage(`{"path":"/ws/file.txt","bogus":1}`)
	err := ValidateToolInputForCategory("core_reader", json.RawMessage(categoryTestSchema), input, SourceCategoryCore)
	if err == nil {
		t.Fatal("built-in schema must remain a closed set")
	}
	if want := `unknown parameter "bogus"`; !strings.Contains(err.Error(), want) {
		t.Errorf("error = %q, want it to contain %q", err.Error(), want)
	}
}
