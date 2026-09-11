package llm

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
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

// assertNoDuplicateMembers fails the test if any object in out carries a
// duplicate member name. Go's json.Unmarshal collapses duplicates (last wins),
// so they can only be detected on the raw token stream: the walker tracks a
// frame per open container; inside an object, scalar tokens alternate between
// member key and member value, and closing a nested container completes the
// enclosing member's value, making the next token a key again.
func assertNoDuplicateMembers(t *testing.T, out json.RawMessage) {
	t.Helper()
	type frame struct {
		isObj     bool
		seen      map[string]struct{}
		expectKey bool
	}
	var stack []frame
	dec := json.NewDecoder(bytes.NewReader(out))
	dec.UseNumber()
	for {
		tok, err := dec.Token()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatalf("tokenizing sanitized output: %v (%s)", err, out)
		}
		delim, ok := tok.(json.Delim)
		if !ok {
			if len(stack) == 0 || !stack[len(stack)-1].isObj {
				continue // top-level scalar or array element: no key semantics
			}
			top := &stack[len(stack)-1]
			if top.expectKey {
				key, ok := tok.(string)
				if !ok {
					t.Fatalf("non-string member key %v in %s", tok, out)
				}
				if _, dup := top.seen[key]; dup {
					t.Errorf("duplicate member %q in sanitized output: %s", key, out)
					return
				}
				top.seen[key] = struct{}{}
				top.expectKey = false
			} else {
				top.expectKey = true
			}
			continue
		}
		switch delim {
		case '{':
			stack = append(stack, frame{isObj: true, seen: map[string]struct{}{}, expectKey: true})
		case '[':
			stack = append(stack, frame{})
		case '}', ']':
			if len(stack) == 0 {
				t.Fatalf("unbalanced JSON in sanitized output: %s", out)
			}
			stack = stack[:len(stack)-1]
			// A closed container completed the value of the enclosing
			// object's current member; the next token there is a key.
			if len(stack) > 0 && stack[len(stack)-1].isObj {
				stack[len(stack)-1].expectKey = true
			}
		}
	}
}

func TestSanitizers_EmitDuplicateSourceMembersOnce(t *testing.T) {
	// MCP-gateway tools pass through third-party schemas where copy-paste
	// editing can produce duplicated members; every mainstream JSON decoder
	// accepts them silently (last wins). The sanitizer is the defensive
	// boundary and must normalize such input to one member per key: duplicate
	// members are ambiguous JSON, and strict validators (e.g. OpenAI
	// strict:true schemas) may reject the tool outright. Regression guard for
	// the ordered marshaller, which feeds all three sanitizer variants.
	const dupTopLevel = `{
		"type": "object",
		"properties": {"a": {"type": "string"}},
		"properties": {"b": {"type": "integer"}},
		"required": ["b"]
	}`
	const dupNested = `{
		"type": "object",
		"properties": {
			"x": {"type": "string", "type": "integer"},
			"y": {"type": "string"}
		},
		"required": ["x"]
	}`
	sanitizers := []struct {
		name     string
		sanitize func(json.RawMessage) json.RawMessage
	}{
		{"openai_strict", SanitizeSchemaForOpenAI},
		{"openai_nonstrict", SanitizeSchemaForOpenAINonStrict},
		{"anthropic", SanitizeSchemaForAnthropic},
	}
	for _, tc := range []struct {
		name      string
		raw       string
		topLevel  bool
		wantKey   string
		wantValue string
	}{
		{name: "top-level duplicate properties", raw: dupTopLevel, topLevel: true, wantKey: "b", wantValue: "integer"},
		{name: "nested duplicate type", raw: dupNested, wantKey: "x", wantValue: "integer"},
	} {
		for _, san := range sanitizers {
			t.Run(tc.name+"/"+san.name, func(t *testing.T) {
				out := san.sanitize(json.RawMessage(tc.raw))
				if !json.Valid(out) {
					t.Fatalf("sanitized output is not valid JSON: %s", out)
				}
				assertNoDuplicateMembers(t, out)

				// The last-wins merge must survive as the single emitted
				// member: the shadowed first declaration is gone and the
				// surviving member carries the last declared value.
				var got struct {
					Properties map[string]struct {
						Type string `json:"type"`
					} `json:"properties"`
				}
				if err := json.Unmarshal(out, &got); err != nil {
					t.Fatalf("unmarshaling sanitized output: %v (%s)", err, out)
				}
				if tc.topLevel {
					if _, ok := got.Properties["a"]; ok {
						t.Errorf("properties retains \"a\" from the shadowed first declaration: %s", out)
					}
				}
				if prop, ok := got.Properties[tc.wantKey]; !ok || prop.Type != tc.wantValue {
					t.Errorf("properties[%q].type = %q, ok=%v, want the last-wins declaration with type %q: %s",
						tc.wantKey, got.Properties[tc.wantKey].Type, ok, tc.wantValue, out)
				}
			})
		}
	}
}
