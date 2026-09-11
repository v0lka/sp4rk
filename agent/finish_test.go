package agent

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/v0lka/sp4rk/tools"
)

func TestFinishTool_Metadata(t *testing.T) {
	ft := NewFinishTool()

	if ft.Name() != "finish" {
		t.Errorf("Name() = %q, want %q", ft.Name(), "finish")
	}
	if ft.Description() == "" {
		t.Error("Description() should not be empty")
	}
	if ft.DefaultPolicy() != tools.PolicyAlwaysAllow {
		t.Errorf("DefaultPolicy() = %v, want PolicyAlwaysAllow", ft.DefaultPolicy())
	}

	schema := ft.InputSchema()
	if !json.Valid(schema) {
		t.Error("InputSchema() is not valid JSON")
	}

	var s map[string]interface{}
	if err := json.Unmarshal(schema, &s); err != nil {
		t.Fatalf("unmarshal schema: %v", err)
	}
	if s["type"] != "object" {
		t.Errorf("schema type = %v, want object", s["type"])
	}
}

func TestFinishTool_Execute_Success(t *testing.T) {
	ft := NewFinishTool()
	input := json.RawMessage(`{"answer": "42"}`)
	result, err := ft.Execute(context.Background(), input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Errorf("result.IsError = true, want false")
	}
	if result.Content != "42" {
		t.Errorf("result.Content = %q, want %q", result.Content, "42")
	}
}

func TestFinishTool_Execute_InvalidJSON(t *testing.T) {
	ft := NewFinishTool()
	input := json.RawMessage(`not json`)
	result, err := ft.Execute(context.Background(), input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Error("expected IsError=true for invalid JSON input")
	}
	if result.Content == "" {
		t.Error("expected non-empty error content")
	}
}

func TestFinishTool_IsUntrusted(t *testing.T) {
	ft := NewFinishTool()
	if ft.IsUntrusted() {
		t.Error("FinishTool.IsUntrusted() should return false — finish is a trusted internal tool")
	}
}

// maxDescriptionLength mirrors the guard in tools/builtins and skills:
// descriptions follow the purpose -> when-to-use -> inputs -> outputs ->
// example -> anti-example rubric; anything beyond this limit means marketing
// prose crept back in.
const maxDescriptionLength = 1200

// finishRubricSections is the ordered set of labels every tool description must
// carry — each on its own line.
var finishRubricSections = []string{"Purpose:", "Use when:", "Inputs:", "Outputs:", "Example:", "Anti-example:"}

// TestFinishDescriptionFollowsRubric guards the finish tool's description so the
// rubric cannot regress. finish is special — its description is declared as a
// raw constant in this package rather than sourced from tools/builtins — so it
// needs its own guard mirroring descriptions_guard_test.go. Host UIs (e.g.
// c0wrk's tool picker tooltip) render tool descriptions with a line-based
// markdown reformatter, so every label must begin its own line, and the Inputs
// section must name the `answer` field so the model knows what to pass.
func TestFinishDescriptionFollowsRubric(t *testing.T) {
	desc := NewFinishTool().Description()

	if strings.TrimSpace(desc) == "" {
		t.Fatal("finish: description is empty")
	}
	for _, section := range finishRubricSections {
		if !strings.Contains(desc, section) {
			t.Errorf("finish: description lacks %q section", section)
		}
		// Every rubric section must begin its own line, otherwise a line-based
		// markdown reformatter cannot split it into bold-labelled paragraphs.
		if !strings.Contains(desc, "\n"+section) && !strings.HasPrefix(desc, section) {
			t.Errorf("finish: %q does not start a line — the markdown reformatter needs one label per line", section)
		}
	}
	if len(desc) > maxDescriptionLength {
		t.Errorf("finish: description is %d chars, exceeds guard limit %d", len(desc), maxDescriptionLength)
	}
	// The Inputs section must name the `answer` field, matching the tool's JSON
	// schema, so the model knows what to pass.
	var inputsLine string
	for _, line := range strings.Split(desc, "\n") {
		if strings.HasPrefix(line, "Inputs:") {
			inputsLine = line
			break
		}
	}
	if inputsLine == "" {
		t.Fatal("finish: description has no Inputs: line to check")
	}
	if !strings.Contains(inputsLine, "answer") {
		t.Errorf("finish: Inputs line does not name the `answer` field: %q", inputsLine)
	}
}
