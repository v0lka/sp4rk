package agents

import (
	"reflect"
	"strings"
	"testing"
)

func TestToolPreferenceWithError(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		tools   string
		want    any
		wantErr bool
	}{
		{name: "empty is nil (default all)", tools: "", want: nil},
		{name: "all is nil", tools: "all", want: nil},
		{name: "padded all is nil", tools: " all ", want: nil},
		{name: "padded read-only", tools: " read-only ", want: "read-only"},
		{name: "read-only string", tools: "read-only", want: "read-only"},
		{name: "single group", tools: "execute", want: []string{"execute"}},
		{name: "comma list trimmed", tools: "local-read, execute , local-mcp", want: []string{"local-read", "execute", "local-mcp"}},
		{name: "underscore spelling canonicalized to kebab", tools: "local_read,remote_write", want: []string{"local-read", "remote-write"}},
		// Fail-closed: an unknown token errors instead of being dropped
		// (dropping every token would silently widen nil → full toolset).
		{name: "unknown token errors", tools: "edit_file", wantErr: true},
		{name: "unknown among valid errors", tools: "execute,edit_file", wantErr: true},
		{name: "empty items error", tools: " , , ", wantErr: true},
		{name: "duplicate group errors", tools: "execute,execute", wantErr: true},
		{name: "duplicate via mixed spelling errors", tools: "local-read,local_read", wantErr: true},
		// Presets cannot be mixed into a group list (same rule the parser
		// enforces via validateToolsField, which ToolPreferenceWithError delegates to).
		{name: "all mixed with groups errors", tools: "all,execute", wantErr: true},
		{name: "read-only mixed with groups errors", tools: "read-only,local-read", wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			a := &Agent{Metadata: AgentMetadata{Tools: tt.tools}}
			got, err := a.ToolPreferenceWithError()
			if gotErr := err != nil; gotErr != tt.wantErr {
				t.Fatalf("ToolPreference(%q) error = %v, want error presence = %t", tt.tools, err, tt.wantErr)
			}
			if tt.wantErr {
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("ToolPreference(%q) = %#v (%T), want %#v", tt.tools, got, got, tt.want)
			}
		})
	}
}

func TestToolPreferenceWithError_MixedPresetErrorsWithParseMessage(t *testing.T) {
	t.Parallel()

	// ToolPreferenceWithError delegates to validateToolsField, so a mixed
	// "all"/"read-only" list must fail with the parser's clear "cannot be
	// combined" message — not the self-contradictory `unknown tool group
	// "all" (… also accepted: "all", "read-only")` wording it used to emit.
	for _, tools := range []string{"all,execute", "read-only,local-read"} {
		a := &Agent{Metadata: AgentMetadata{Tools: tools}}
		_, err := a.ToolPreferenceWithError()
		if err == nil {
			t.Fatalf("ToolPreference(%q) = nil error, want error", tools)
		}
		if !strings.Contains(err.Error(), "cannot be combined") {
			t.Errorf("ToolPreference(%q) error = %q, want it to contain %q", tools, err, "cannot be combined")
		}
	}
}

func TestDescriptor_HiddenPropagated(t *testing.T) {
	t.Parallel()

	a := &Agent{Metadata: AgentMetadata{Name: "n", Description: "d", Hidden: true}}
	got := a.Descriptor()
	want := AgentDescriptor{Name: "n", Description: "d", Hidden: true}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Descriptor() = %#v, want %#v", got, want)
	}
}

func TestNormalizeToolGroupToken(t *testing.T) {
	t.Parallel()

	valid := map[string]string{
		"execute":    "execute",
		"local-read": "local-read",
		"local_read": "local-read",
		"local-mcp":  "local-mcp",
	}
	for in, want := range valid {
		got, ok := NormalizeToolGroupToken(in)
		if !ok || got != want {
			t.Errorf("NormalizeToolGroupToken(%q) = %q,%v; want %q,true", in, got, ok, want)
		}
	}
	// Whitespace is trimmed by the canonicalizer.
	if got, ok := NormalizeToolGroupToken(" local-mcp "); !ok || got != "local-mcp" {
		t.Errorf("NormalizeToolGroupToken(%q) = %q,%v; want %q,true", " local-mcp ", got, ok, "local-mcp")
	}

	for _, in := range []string{"", "all", "read-only", "unknown", "local--read"} {
		if got, ok := NormalizeToolGroupToken(in); ok {
			t.Errorf("NormalizeToolGroupToken(%q) = %q, want rejected", in, got)
		}
	}
}

func TestToolGroupTokens_Canonical(t *testing.T) {
	t.Parallel()

	got := ToolGroupTokens()
	want := []string{"execute", "local-read", "local-write", "remote-read", "remote-write", "system", "local-mcp", "remote-mcp"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("ToolGroupTokens() = %v, want %v", got, want)
	}
	// The returned slice must be a copy — callers must not be able to mutate
	// the package-level set through it.
	got[0] = "mutated"
	if ToolGroupTokens()[0] != "execute" {
		t.Error("ToolGroupTokens() returned slice aliases the package-level set")
	}
}
