package llm

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestToolSchemaNullableObjectsByProtocol(t *testing.T) {
	input := json.RawMessage(`{
		"type":"object",
		"properties":{
			"csvFormat":{"type":["object","null"],"properties":{
				"quote":{"type":"string"},
				"quoteEscape":{"type":["string","null"]}
			},"required":["quote"]},
			"columnMappings":{"type":"array","items":{"type":"object","properties":{
				"mismatch":{"$ref":"#/$defs/Mismatch"}
			}}}
		},
		"$defs":{"Mismatch":{"type":["null","object"],"properties":{"kind":{"type":"string"}}}}
	}`)
	tests := []struct {
		name     string
		sanitize func(json.RawMessage) json.RawMessage
	}{
		{"Responses", nil},
		{"ChatCompletions", SanitizeSchemaForOpenAINonStrict},
		{"Anthropic", SanitizeSchemaForAnthropic},
		{"Google", SanitizeSchemaForGoogle},
	}
	type node struct {
		Type                 any             `json:"type"`
		Properties           map[string]node `json:"properties"`
		Items                *node           `json:"items"`
		Required             []string        `json:"required"`
		AdditionalProperties *bool           `json:"additionalProperties"`
		Nullable             bool            `json:"nullable"`
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var root node
			var raw json.RawMessage
			if tc.sanitize == nil {
				tools := convertToResponsesTools([]ToolDefinition{{Name: "import_data", InputSchema: input}}, nil)
				if len(tools) != 1 || tools[0].OfFunction == nil || !tools[0].OfFunction.Strict.Value {
					t.Fatalf("expected one strict Responses function, got %+v", tools)
				}
				var err error
				raw, err = json.Marshal(tools[0].OfFunction.Parameters)
				if err != nil {
					t.Fatal(err)
				}
			} else {
				raw = tc.sanitize(input)
			}
			if err := json.Unmarshal(raw, &root); err != nil {
				t.Fatal(err)
			}
			items := root.Properties["columnMappings"].Items
			if items == nil {
				t.Fatal("missing columnMappings.items")
			}
			format := root.Properties["csvFormat"]
			mismatch := items.Properties["mismatch"]
			if tc.name == "Google" {
				if format.Type != "object" || !format.Nullable || mismatch.Type != "object" || !mismatch.Nullable {
					t.Fatalf("expected nullable Google objects, got %+v and %+v", format, mismatch)
				}
			} else if !reflect.DeepEqual(format.Type, []any{"object", "null"}) ||
				!reflect.DeepEqual(mismatch.Type, []any{"null", "object"}) {
				t.Fatalf("expected original type unions, got %v and %v", format.Type, mismatch.Type)
			}
			if tc.name == "Responses" {
				if format.AdditionalProperties == nil || *format.AdditionalProperties ||
					mismatch.AdditionalProperties == nil || *mismatch.AdditionalProperties ||
					!reflect.DeepEqual(format.Required, []string{"quote", "quoteEscape"}) ||
					!reflect.DeepEqual(mismatch.Required, []string{"kind"}) {
					t.Fatalf("expected closed objects with all properties required, got %+v and %+v", format, mismatch)
				}
				return
			}
			if format.AdditionalProperties != nil || mismatch.AdditionalProperties != nil ||
				!reflect.DeepEqual(format.Required, []string{"quote"}) || mismatch.Required != nil {
				t.Fatalf("expected original optionality and openness, got %+v and %+v", format, mismatch)
			}
		})
	}
}
