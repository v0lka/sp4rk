package llm

import (
	"encoding/json"
	"slices"
	"strings"
	"testing"
)

// storeFactLikeSchema mirrors the shape that regressed: an object whose author
// declared "keywords" before "content" and listed required in that same order.
const storeFactLikeSchema = `{
	"type": "object",
	"properties": {
		"keywords": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 10},
		"content": {"type": "string"}
	},
	"required": ["keywords", "content"]
}`

func TestSanitizeSchemaForOpenAINonStrict_PreservesDeclaredOrder(t *testing.T) {
	out := SanitizeSchemaForOpenAINonStrict(json.RawMessage(storeFactLikeSchema))

	var got struct {
		Required []string `json:"required"`
	}
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatalf("sanitized output is not valid JSON: %v (%s)", err, out)
	}
	if want := []string{"keywords", "content"}; !slices.Equal(got.Required, want) {
		t.Errorf("required = %v, want %v", got.Required, want)
	}

	// The serialized document must keep "keywords" before "content" inside
	// properties, not the alphabetical {content, keywords}.
	s := string(out)
	kw, ct := strings.Index(s, `"keywords"`), strings.Index(s, `"content"`)
	if kw < 0 || ct < 0 || kw > ct {
		t.Errorf("property order wrong: keywords@%d content@%d, want keywords first (%s)", kw, ct, s)
	}
	if ti, pi := strings.Index(s, `"type"`), strings.Index(s, `"properties"`); ti < 0 || ti > pi {
		t.Errorf("top-level order wrong: type@%d properties@%d, want type first (%s)", ti, pi, s)
	}
}

func TestSanitizeSchemaForOpenAI_StrictKeepsDeclaredRequiredPrefix(t *testing.T) {
	// Strict mode adds every property to required; the caller-declared order
	// must be kept as a prefix, with the promoted optional appended after.
	raw := json.RawMessage(`{
		"type": "object",
		"properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
		"required": ["name"]
	}`)
	out := SanitizeSchemaForOpenAI(raw)

	var got struct {
		Required []string `json:"required"`
	}
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatalf("sanitized output is not valid JSON: %v (%s)", err, out)
	}
	if want := []string{"name", "age"}; !slices.Equal(got.Required, want) {
		t.Errorf("required = %v, want %v", got.Required, want)
	}
}

func TestSanitizeSchemaForAnthropic_PreservesDeclaredOrder(t *testing.T) {
	out := SanitizeSchemaForAnthropic(json.RawMessage(storeFactLikeSchema))

	s := string(out)
	kw, ct := strings.Index(s, `"keywords"`), strings.Index(s, `"content"`)
	if kw < 0 || ct < 0 || kw > ct {
		t.Errorf("property order wrong: keywords@%d content@%d, want keywords first (%s)", kw, ct, s)
	}
	if !json.Valid(out) {
		t.Errorf("sanitized Anthropic schema is not valid JSON: %s", s)
	}
}

func TestMarshalPreservingKeyOrder_FallsBackOnUnparseableRaw(t *testing.T) {
	got := marshalPreservingKeyOrder(json.RawMessage(`{`), map[string]any{"b": 1, "a": 2})
	if !json.Valid(got) {
		t.Fatalf("fallback output is not valid JSON: %s", got)
	}
	if string(got) != `{"a":2,"b":1}` {
		t.Errorf("fallback = %s, want alphabetical {\"a\":2,\"b\":1}", got)
	}
}
