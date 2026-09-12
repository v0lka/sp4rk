package tools

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

// read_file-like schema: closed property set with one required key — the
// exact shape whose silent-misparse failure motivated the original e2s
// validator (offset/limit instead of start_line/end_line).
const readFileTestSchema = `{
	"type": "object",
	"properties": {
		"path": {"type": "string", "description": "file path"},
		"start_line": {"type": "integer"},
		"end_line": {"type": "integer"}
	},
	"required": ["path"]
}`

// declare_plan-like schema: a required array of task objects — exercises
// items-recursion with paths like tasks[2].id.
const declarePlanLikeSchema = `{
	"type": "object",
	"properties": {
		"plan_title": {"type": "string"},
		"tasks": {
			"type": "array",
			"items": {
				"type": "object",
				"properties": {
					"id": {"type": "string"},
					"title": {"type": "string"},
					"description": {"type": "string"}
				},
				"required": ["id", "title"]
			}
		}
	},
	"required": ["tasks"]
}`

func TestValidateToolInput_UnknownParameterRejected(t *testing.T) {
	// Both offset and limit are unknown here; the validator reports the
	// first in sorted order — the assertion covers whichever surfaces.
	err := ValidateToolInput("read_file", json.RawMessage(readFileTestSchema), json.RawMessage(`{"path":"a.txt","offset":1,"limit":100}`))
	if err == nil {
		t.Fatal("unknown parameter offset must be rejected")
	}
	msg := err.Error()
	if !strings.Contains(msg, "unknown parameter") ||
		(!strings.Contains(msg, `"offset"`) && !strings.Contains(msg, `"limit"`)) {
		t.Errorf("error must name an unknown parameter: %q", msg)
	}
	for _, valid := range []string{"path", "start_line", "end_line"} {
		if !strings.Contains(msg, valid) {
			t.Errorf("error must list valid parameter %q: %q", valid, msg)
		}
	}
}

func TestValidateToolInput_ValidCallPasses(t *testing.T) {
	if err := ValidateToolInput("read_file", json.RawMessage(readFileTestSchema), json.RawMessage(`{"path":"a.txt","start_line":321,"end_line":520}`)); err != nil {
		t.Fatalf("valid ranged read must pass: %v", err)
	}
	if err := ValidateToolInput("read_file", json.RawMessage(readFileTestSchema), json.RawMessage(`{"path":"a.txt"}`)); err != nil {
		t.Fatalf("minimal valid read must pass: %v", err)
	}
	if err := ValidateToolInput("declare_plan", json.RawMessage(declarePlanLikeSchema), json.RawMessage(`{"plan_title":"Roadmap","tasks":[{"id":"t1","title":"First","description":"d"},{"id":"t2","title":"Second"}]}`)); err != nil {
		t.Fatalf("valid declare_plan-like call must pass: %v", err)
	}
}

func TestValidateToolInput_MissingRequired(t *testing.T) {
	err := ValidateToolInput("read_file", json.RawMessage(readFileTestSchema), json.RawMessage(`{"start_line":1}`))
	if err == nil || !strings.Contains(err.Error(), `missing required parameter "path"`) {
		t.Fatalf("missing required must be rejected: %v", err)
	}
	var verr *InputValidationError
	if !errors.As(err, &verr) {
		t.Fatalf("error must be *InputValidationError, got %T", err)
	}
	if verr.Path != "" || verr.Reason != `missing required parameter "path"` {
		t.Errorf("unexpected typed error fields: %+v", verr)
	}
	if strings.Join(verr.ValidParams, ",") != "end_line,path,start_line" {
		t.Errorf("valid params must be sorted schema properties: %v", verr.ValidParams)
	}
	if !strings.Contains(err.Error(), "valid parameters: end_line, path, start_line") {
		t.Errorf("message must list valid parameters: %q", err.Error())
	}
}

func TestValidateToolInput_TypeMismatch(t *testing.T) {
	err := ValidateToolInput("read_file", json.RawMessage(readFileTestSchema), json.RawMessage(`{"path":42}`))
	if err == nil || !strings.Contains(err.Error(), `parameter "path" must be of type string, got number`) {
		t.Fatalf("type mismatch must be rejected verbatim: %v", err)
	}
	var verr *InputValidationError
	if !errors.As(err, &verr) || verr.Path != "path" {
		t.Fatalf("type error must carry the parameter path, got %v", err)
	}
	// Integer-typed parameter accepts a JSON number (encoding/json decodes
	// all numbers as float64; the validator maps integer→number).
	if err := ValidateToolInput("read_file", json.RawMessage(readFileTestSchema), json.RawMessage(`{"path":"a","start_line":5}`)); err != nil {
		t.Fatalf("integer parameter with number value must pass: %v", err)
	}
}

func TestValidateToolInput_TypeArrayAcceptsNullAndMember(t *testing.T) {
	schema := json.RawMessage(`{"type":"object","properties":{"a":{"type":["string","null"]}}}`)
	if err := ValidateToolInput("t", schema, json.RawMessage(`{"a":null}`)); err != nil {
		t.Fatalf("type array must accept null member: %v", err)
	}
	if err := ValidateToolInput("t", schema, json.RawMessage(`{"a":"s"}`)); err != nil {
		t.Fatalf("type array must accept string member: %v", err)
	}
	err := ValidateToolInput("t", schema, json.RawMessage(`{"a":5}`))
	if err == nil || !strings.Contains(err.Error(), `parameter "a" must be of type string|null, got number`) {
		t.Fatalf("type array must reject a non-member type: %v", err)
	}
}

func TestValidateToolInput_OpenSchemasFailOpen(t *testing.T) {
	cases := []struct {
		name   string
		schema string
		args   string
	}{
		{"no properties", `{"type":"object","additionalProperties":true}`, `{"whatever":"x"}`},
		{"empty schema", `{}`, `{"whatever":"x"}`},
		{"malformed schema", `{"properties":`, `{"whatever":"x"}`},
		{"additionalProperties true", `{"type":"object","properties":{"a":{"type":"string"}},"additionalProperties":true}`, `{"extra":"x","a":"s"}`},
		{"additionalProperties false-ish absent at nested level", `{"type":"object","properties":{"a":{"type":"object"}}}`, `{"a":{"extra":"x"}}`},
		{"additionalProperties object form", `{"type":"object","properties":{"a":{"type":"string"}},"additionalProperties":{"type":"string"}}`, `{"extra":"x","a":"s"}`},
	}
	for _, tc := range cases {
		if err := ValidateToolInput("t", json.RawMessage(tc.schema), json.RawMessage(tc.args)); err != nil {
			t.Errorf("%s: must fail open, got %v", tc.name, err)
		}
	}
}

func TestValidateToolInput_NonObjectArgsRejected(t *testing.T) {
	for _, args := range []string{`[1,2]`, `"str"`, `42`} {
		err := ValidateToolInput("read_file", json.RawMessage(readFileTestSchema), json.RawMessage(args))
		if err == nil {
			t.Fatalf("non-object args %s must be rejected", args)
		}
		msg := err.Error()
		if !strings.Contains(msg, "input must be a JSON object") {
			t.Errorf("rejection must explain the object requirement: %q", msg)
		}
		if !strings.Contains(msg, "valid parameters: end_line, path, start_line") {
			t.Errorf("rejection must list valid parameters: %q", msg)
		}
	}
}

func TestValidateToolInput_EmptyArgsWithRequiredRejected(t *testing.T) {
	if err := ValidateToolInput("read_file", json.RawMessage(readFileTestSchema), json.RawMessage(`{}`)); err == nil {
		t.Fatal("empty args against required path must be rejected")
	}
	if err := ValidateToolInput("read_file", json.RawMessage(readFileTestSchema), nil); err == nil {
		t.Fatal("nil args against required path must be rejected")
	}
}

func TestValidateToolInput_NestedObjectProperty(t *testing.T) {
	schema := json.RawMessage(`{
		"type": "object",
		"properties": {
			"path": {"type": "string"},
			"options": {
				"type": "object",
				"properties": {"mode": {"type": "string"}, "depth": {"type": "integer"}},
				"required": ["mode"]
			}
		},
		"required": ["path"]
	}`)
	if err := ValidateToolInput("t", schema, json.RawMessage(`{"path":"a.txt","options":{"mode":"fast","depth":3}}`)); err != nil {
		t.Fatalf("valid nested object must pass: %v", err)
	}
	err := ValidateToolInput("t", schema, json.RawMessage(`{"path":"a.txt","options":{"depth":3}}`))
	if err == nil {
		t.Fatal("missing required in a nested object must be rejected")
	}
	msg := err.Error()
	if !strings.Contains(msg, "options") || !strings.Contains(msg, `missing required parameter "mode"`) {
		t.Fatalf("nested missing-required must carry the nested path and key: %q", msg)
	}
	var verr *InputValidationError
	if !errors.As(err, &verr) || verr.Path != "options" {
		t.Fatalf("nested error path must be the property key path, got %v", err)
	}
}

func TestValidateToolInput_DeclarePlanLikeArrayRecursion(t *testing.T) {
	// tasks[2] is an object missing the required "id": the error must carry
	// the array path (tasks[2]) and the parameter name (id).
	err := ValidateToolInput("declare_plan", json.RawMessage(declarePlanLikeSchema), json.RawMessage(`{"plan_title":"Roadmap","tasks":[{"id":"t1","title":"First"},{"id":"t2","title":"Second"},{"title":"Third"}]}`))
	if err == nil {
		t.Fatal("missing required inside an array element must be rejected")
	}
	var verr *InputValidationError
	if !errors.As(err, &verr) {
		t.Fatalf("error must be *InputValidationError, got %T", err)
	}
	if verr.Path != "tasks[2]" || verr.Reason != `missing required parameter "id"` {
		t.Errorf("unexpected typed error fields: %+v", verr)
	}
	if strings.Join(verr.ValidParams, ",") != "description,id,title" {
		t.Errorf("valid params must be the items-level properties: %v", verr.ValidParams)
	}
	msg := err.Error()
	if !strings.Contains(msg, "tasks[2]") || !strings.Contains(msg, `"id"`) {
		t.Errorf("message must contain the array path and parameter name: %q", msg)
	}

	// Type mismatch on a nested element field: full dotted path.
	err = ValidateToolInput("declare_plan", json.RawMessage(declarePlanLikeSchema), json.RawMessage(`{"tasks":[{"id":"t1","title":"First"},{"id":7,"title":"Second"}]}`))
	if err == nil || !strings.Contains(err.Error(), `parameter "id" must be of type string, got number`) {
		t.Fatalf("nested type mismatch must be rejected verbatim: %v", err)
	}
	if !errors.As(err, &verr) || verr.Path != "tasks[1].id" {
		t.Fatalf("nested type error path must be tasks[1].id, got %v", err)
	}

	// Unknown key inside an array element: rejected against the items set.
	err = ValidateToolInput("declare_plan", json.RawMessage(declarePlanLikeSchema), json.RawMessage(`{"tasks":[{"id":"t1","title":"First","bogus":1}]}`))
	if err == nil || !strings.Contains(err.Error(), `unknown parameter "bogus"`) {
		t.Fatalf("unknown key inside an array element must be rejected: %v", err)
	}
	if !errors.As(err, &verr) || verr.Path != "tasks[0]" {
		t.Fatalf("nested unknown-key error path must be tasks[0], got %v", err)
	}

	// A non-object array element is a type error carrying the element path.
	err = ValidateToolInput("declare_plan", json.RawMessage(declarePlanLikeSchema), json.RawMessage(`{"tasks":[{"id":"t1","title":"First"},"oops"]}`))
	if err == nil || !strings.Contains(err.Error(), `parameter "tasks[1]" must be of type object, got string`) {
		t.Fatalf("non-object element must be a type error with the element path: %v", err)
	}
}

func TestValidateToolInput_NestedAdditionalPropertiesVariants(t *testing.T) {
	closed := json.RawMessage(`{
		"type": "object",
		"properties": {
			"tasks": {
				"type": "array",
				"items": {
					"type": "object",
					"properties": {"id": {"type": "string"}},
					"required": ["id"]
				}
			}
		},
		"required": ["tasks"]
	}`)
	err := ValidateToolInput("t", closed, json.RawMessage(`{"tasks":[{"id":"a","bogus":1}]}`))
	if err == nil || !strings.Contains(err.Error(), `unknown parameter "bogus"`) {
		t.Fatalf("closed items schema must reject unknown element keys: %v", err)
	}

	open := json.RawMessage(`{
		"type": "object",
		"properties": {
			"tasks": {
				"type": "array",
				"items": {
					"type": "object",
					"properties": {"id": {"type": "string"}},
					"required": ["id"],
					"additionalProperties": true
				}
			}
		},
		"required": ["tasks"]
	}`)
	if err := ValidateToolInput("t", open, json.RawMessage(`{"tasks":[{"id":"a","extra":1}]}`)); err != nil {
		t.Fatalf("items additionalProperties:true must accept extra keys: %v", err)
	}
}

func TestValidateToolInput_RefSubtreesFailOpen(t *testing.T) {
	cases := []struct {
		name   string
		schema string
		args   string
	}{
		{
			"top-level $ref schema",
			`{"$ref":"#/definitions/tool"}`,
			`{"whatever":"x"}`,
		},
		{
			"$ref property subtree",
			`{"type":"object","properties":{"a":{"$ref":"#/definitions/a"}},"required":["a"]}`,
			`{"a":{"bogus":1}}`,
		},
		{
			"$ref items subtree",
			`{"type":"object","properties":{"tasks":{"type":"array","items":{"$ref":"#/definitions/task"}}},"required":["tasks"]}`,
			`{"tasks":[{"bogus":1}]}`,
		},
	}
	for _, tc := range cases {
		if err := ValidateToolInput("t", json.RawMessage(tc.schema), json.RawMessage(tc.args)); err != nil {
			t.Errorf("%s: must fail open, got %v", tc.name, err)
		}
	}
}

func TestValidateToolInput_MalformedNestedSchemaFailsOpen(t *testing.T) {
	// The top-level document parses, but a property's schema has the wrong
	// JSON shape: the subtree is skipped fail-open.
	cases := []struct {
		name   string
		schema string
		args   string
	}{
		{"non-object property schema", `{"type":"object","properties":{"a":5}}`, `{"a":{"x":1}}`},
		{"properties not an object", `{"type":"object","properties":{"a":{"properties":5}}}`, `{"a":{"x":1}}`},
		{"required not an array", `{"type":"object","properties":{"a":{"required":5}}}`, `{"a":{"x":1}}`},
		{"items not an object", `{"type":"object","properties":{"a":{"type":"array","items":5}}}`, `{"a":[1,2]}`},
	}
	for _, tc := range cases {
		if err := ValidateToolInput("t", json.RawMessage(tc.schema), json.RawMessage(tc.args)); err != nil {
			t.Errorf("%s: must fail open, got %v", tc.name, err)
		}
	}
}

// deepNested builds a schema/input pair whose innermost object (missing its
// required "deepest_prop") sits at the given nesting depth below the root.
func deepNested(depth int) (schema, input string) {
	schema = `{"type":"object","properties":{"deepest_prop":{"type":"string"}},"required":["deepest_prop"]}`
	input = `{}`
	for range depth {
		schema = `{"type":"object","properties":{"a":` + schema + `}}`
		input = `{"a":` + input + `}`
	}
	return schema, input
}

func TestValidateToolInput_DepthCapFailsOpen(t *testing.T) {
	// Depth 17 (beyond maxValidationDepth=16): the deepest violation is
	// skipped fail-open.
	schema, input := deepNested(17)
	if err := ValidateToolInput("t", json.RawMessage(schema), json.RawMessage(input)); err != nil {
		t.Fatalf("depth 17 must fail open, got %v", err)
	}

	// Depth 16 (at the cap): the deepest violation is still caught, with
	// the full 16-segment path.
	schema, input = deepNested(16)
	err := ValidateToolInput("t", json.RawMessage(schema), json.RawMessage(input))
	if err == nil {
		t.Fatal("depth 16 must still be validated")
	}
	var verr *InputValidationError
	if !errors.As(err, &verr) {
		t.Fatalf("error must be *InputValidationError, got %T", err)
	}
	if want := strings.Repeat("a.", 15) + "a"; verr.Path != want {
		t.Errorf("path must be %d nested segments %q, got %q", 16, want, verr.Path)
	}
	if !strings.Contains(err.Error(), `missing required parameter "deepest_prop"`) {
		t.Errorf("message must name the missing deepest parameter: %q", err.Error())
	}
}
