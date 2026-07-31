package agents

import (
	"reflect"
	"testing"
)

func TestToolPreference(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		tools string
		want  any
	}{
		{name: "empty is nil (default all)", tools: "", want: nil},
		{name: "all is nil", tools: "all", want: nil},
		{name: "read-only string", tools: "read-only", want: "read-only"},
		{name: "single tool", tools: "edit_file", want: []string{"edit_file"}},
		{name: "comma list trimmed", tools: "edit_file, bash_exec , write_file", want: []string{"edit_file", "bash_exec", "write_file"}},
		{name: "comma list no spaces", tools: "a,b,c", want: []string{"a", "b", "c"}},
		{name: "only commas collapses to nil", tools: " , , ", want: nil},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			a := &Agent{Metadata: AgentMetadata{Tools: tt.tools}}
			got := a.ToolPreference()
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("ToolPreference() = %#v (%T), want %#v", got, got, tt.want)
			}
		})
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
