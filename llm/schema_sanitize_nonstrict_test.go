package llm

import (
	"encoding/json"
	"reflect"
	"testing"
)

// requiredOf unmarshals a sanitized schema and returns its top-level required
// array as a []string, plus whether the key is present at all.
func requiredOf(t *testing.T, raw json.RawMessage) ([]string, bool) {
	t.Helper()
	var parsed map[string]any
	if err := json.Unmarshal(raw, &parsed); err != nil {
		t.Fatalf("failed to parse result: %v", err)
	}
	reqAny, exists := parsed["required"]
	if !exists {
		return nil, false
	}
	arr, ok := reqAny.([]any)
	if !ok {
		t.Fatalf("required is not an array: %T", reqAny)
	}
	out := make([]string, 0, len(arr))
	for _, v := range arr {
		s, ok := v.(string)
		if !ok {
			t.Fatalf("required entry is not a string: %T", v)
		}
		out = append(out, s)
	}
	return out, true
}

// TestSanitizeSchemaForOpenAINonStrict_PreservesOptionalParams verifies the core
// contract of recommendation 1: when strict mode is off, properties that the
// caller did not declare as required are NOT promoted to required. This keeps
// optional parameters (e.g. bash_exec's "timeout") genuinely optional so the
// model is not pressured to emit every field on every call.
func TestSanitizeSchemaForOpenAINonStrict_PreservesOptionalParams(t *testing.T) {
	schema := json.RawMessage(`{
		"type": "object",
		"properties": {
			"command": {"type": "string"},
			"timeout": {"type": "string"},
			"working_directory": {"type": "string"}
		},
		"required": ["command"]
	}`)

	got, _ := requiredOf(t, SanitizeSchemaForOpenAINonStrict(schema))
	want := []string{"command"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("required = %v, want %v (optional params must stay optional)", got, want)
	}

	// Sanity: the strict counterpart DOES promote every property.
	strict, _ := requiredOf(t, SanitizeSchemaForOpenAI(schema))
	wantStrict := []string{"command", "timeout", "working_directory"}
	if !reflect.DeepEqual(strict, wantStrict) {
		t.Errorf("strict required = %v, want %v", strict, wantStrict)
	}
}

// TestSanitizeSchemaForOpenAINonStrict_DropsDanglingRequired verifies that
// required entries referencing non-existent properties are still removed (a
// compatibility fix independent of strictness).
func TestSanitizeSchemaForOpenAINonStrict_DropsDanglingRequired(t *testing.T) {
	schema := json.RawMessage(`{
		"type": "object",
		"properties": {"a": {"type": "string"}},
		"required": ["a", "ghost"]
	}`)

	got, present := requiredOf(t, SanitizeSchemaForOpenAINonStrict(schema))
	if !present {
		t.Fatal("required key should be preserved")
	}
	if !reflect.DeepEqual(got, []string{"a"}) {
		t.Errorf("required = %v, want [a]", got)
	}
}

// TestSanitizeSchemaForOpenAINonStrict_NoRequiredKeyAdded verifies that when the
// caller declares no required list, none is synthesized. An absent required key
// means "everything optional", which is exactly what non-strict mode should
// preserve.
func TestSanitizeSchemaForOpenAINonStrict_NoRequiredKeyAdded(t *testing.T) {
	schema := json.RawMessage(`{
		"type": "object",
		"properties": {"a": {"type": "string"}, "b": {"type": "integer"}}
	}`)

	var parsed map[string]any
	if err := json.Unmarshal(SanitizeSchemaForOpenAINonStrict(schema), &parsed); err != nil {
		t.Fatalf("failed to parse result: %v", err)
	}
	if _, exists := parsed["required"]; exists {
		t.Errorf("required key should not be synthesized, got %v", parsed["required"])
	}
	if parsed["additionalProperties"] != false {
		t.Errorf("additionalProperties = %v, want false", parsed["additionalProperties"])
	}

	// Sanity: strict mode does synthesize required with all properties.
	if _, exists := mustParse(t, SanitizeSchemaForOpenAI(schema))["required"]; !exists {
		t.Error("strict mode should synthesize required")
	}
}

// TestSanitizeSchemaForOpenAINonStrict_RecursesNestedObjects verifies that the
// non-strict flag is threaded through nested object properties: a nested
// property keeps only its declared required entries.
func TestSanitizeSchemaForOpenAINonStrict_RecursesNestedObjects(t *testing.T) {
	schema := json.RawMessage(`{
		"type": "object",
		"properties": {
			"outer": {
				"type": "object",
				"properties": {
					"req": {"type": "string"},
					"opt": {"type": "string"}
				},
				"required": ["req"]
			}
		},
		"required": ["outer"]
	}`)

	parsed := mustParse(t, SanitizeSchemaForOpenAINonStrict(schema))
	outer, ok := parsed["properties"].(map[string]any)["outer"].(map[string]any)
	if !ok {
		t.Fatal("outer property missing")
	}
	reqAny, ok := outer["required"].([]any)
	if !ok {
		t.Fatalf("nested required missing/wrong type: %T", outer["required"])
	}
	if len(reqAny) != 1 || reqAny[0] != "req" {
		t.Errorf("nested required = %v, want [req]", reqAny)
	}
	if outer["additionalProperties"] != false {
		t.Errorf("nested additionalProperties = %v, want false", outer["additionalProperties"])
	}
}

// TestSanitizeSchemaForOpenAINonStrict_ResolvesRefAndKeepsOptionality verifies
// that $ref resolution still happens (definitions removed, ref inlined) while
// optionality inside the referenced definition is preserved.
func TestSanitizeSchemaForOpenAINonStrict_ResolvesRefAndKeepsOptionality(t *testing.T) {
	schema := json.RawMessage(`{
		"$defs": {
			"Thing": {
				"type": "object",
				"properties": {"x": {"type": "string"}, "y": {"type": "string"}},
				"required": ["x"]
			}
		},
		"type": "object",
		"properties": {"thing": {"$ref": "#/$defs/Thing"}},
		"required": ["thing"]
	}`)

	parsed := mustParse(t, SanitizeSchemaForOpenAINonStrict(schema))
	if _, exists := parsed["$defs"]; exists {
		t.Error("$defs should be removed")
	}
	thing, ok := parsed["properties"].(map[string]any)["thing"].(map[string]any)
	if !ok {
		t.Fatal("thing property missing")
	}
	if _, exists := thing["$ref"]; exists {
		t.Error("$ref should be resolved")
	}
	reqAny, ok := thing["required"].([]any)
	if !ok {
		t.Fatalf("resolved required missing: %T", thing["required"])
	}
	if len(reqAny) != 1 || reqAny[0] != "x" {
		t.Errorf("resolved required = %v, want [x]", reqAny)
	}
}

// TestSanitizeSchemaForOpenAINonStrict_ArrayItems exercises the items recursion
// under non-strict mode.
func TestSanitizeSchemaForOpenAINonStrict_ArrayItems(t *testing.T) {
	schema := json.RawMessage(`{
		"type": "array",
		"items": {
			"type": "object",
			"properties": {"a": {"type": "string"}, "b": {"type": "string"}},
			"required": ["a"]
		}
	}`)

	parsed := mustParse(t, SanitizeSchemaForOpenAINonStrict(schema))
	items, ok := parsed["items"].(map[string]any)
	if !ok {
		t.Fatal("items missing")
	}
	reqAny, ok := items["required"].([]any)
	if !ok {
		t.Fatalf("items required missing: %T", items["required"])
	}
	if len(reqAny) != 1 || reqAny[0] != "a" {
		t.Errorf("items required = %v, want [a]", reqAny)
	}
}

func mustParse(t *testing.T, raw json.RawMessage) map[string]any {
	t.Helper()
	var parsed map[string]any
	if err := json.Unmarshal(raw, &parsed); err != nil {
		t.Fatalf("failed to parse result: %v", err)
	}
	return parsed
}
