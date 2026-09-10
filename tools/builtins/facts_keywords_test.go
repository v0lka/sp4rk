package builtins

import (
	"encoding/json"
	"slices"
	"strings"
	"testing"
)

func TestStoreFactInput_UnmarshalJSON_KeywordsShapes(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want []string
	}{
		{
			name: "array form",
			in:   `{"keywords":["auth","middleware","policy"],"content":"x"}`,
			want: []string{"auth", "middleware", "policy"},
		},
		{
			name: "comma-separated string form",
			in:   `{"keywords":"auth, middleware, policy","content":"x"}`,
			want: []string{"auth", "middleware", "policy"},
		},
		{
			name: "single keyword string form",
			in:   `{"keywords":"postgres","content":"x"}`,
			want: []string{"postgres"},
		},
		{
			name: "missing keywords yields nil (validation reports it)",
			in:   `{"content":"x"}`,
			want: nil,
		},
		{
			name: "non-string array element yields nil",
			in:   `{"keywords":[1,2,3],"content":"x"}`,
			want: nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var in StoreFactInput
			if err := json.Unmarshal([]byte(tt.in), &in); err != nil {
				t.Fatalf("StoreFactInput.UnmarshalJSON(%s) error = %v, want nil", tt.in, err)
			}
			if !slices.Equal(in.Keywords, tt.want) {
				t.Errorf("StoreFactInput.UnmarshalJSON(%s).Keywords = %v, want %v", tt.in, in.Keywords, tt.want)
			}
		})
	}
}

func TestStoreFactInput_UnmarshalJSON_MalformedJSONReturnsError(t *testing.T) {
	var in StoreFactInput
	// Unquoted keywords are not valid JSON: the error must still surface so the
	// caller can report it, exactly as before this change.
	err := json.Unmarshal([]byte(`{"keywords": [git, cache]}`), &in)
	if err == nil {
		t.Fatalf("StoreFactInput.UnmarshalJSON(%q) error = nil, want non-nil", `{"keywords": [git, cache]}`)
	}
}

func TestSearchFactsInput_UnmarshalJSON_KeywordsShapes(t *testing.T) {
	var in SearchFactsInput
	if err := json.Unmarshal([]byte(`{"keywords":"auth, config"}`), &in); err != nil {
		t.Fatalf("SearchFactsInput.UnmarshalJSON() error = %v, want nil", err)
	}
	if want := []string{"auth", "config"}; !slices.Equal(in.Keywords, want) {
		t.Errorf("SearchFactsInput.Keywords = %v, want %v", in.Keywords, want)
	}
}

// TestFactDescriptionsShowQuotedKeywordExamples guards against regressing to the
// unquoted "[auth, config]" examples that nudge models into emitting invalid
// JSON arrays (the reported "invalid character 'g'" parse failure).
func TestFactDescriptionsShowQuotedKeywordExamples(t *testing.T) {
	for _, desc := range []string{toolStoreFactDescription, toolSearchFactsDescription} {
		if strings.Contains(desc, "[auth,") {
			t.Errorf("description contains an unquoted array example:\n%s", desc)
		}
		if !strings.Contains(desc, `["auth",`) {
			t.Errorf("description is missing a quoted array example:\n%s", desc)
		}
	}
}
