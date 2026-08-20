package main

import (
	"reflect"
	"testing"

	"github.com/v0lka/sp4rk/agents"
)

func TestSeededAgentProfileUsesValidatedToolGroups(t *testing.T) {
	dir := t.TempDir()
	seedAgentProfile(dir)

	manager := agents.NewAgentManager([]string{dir}, nil)
	if err := manager.Scan(); err != nil {
		t.Fatalf("Scan: %v", err)
	}
	profile, ok := manager.Get("code-researcher")
	if !ok {
		t.Fatal("seeded code-researcher profile was not discovered")
	}

	preference, err := profile.ToolPreferenceWithError()
	if err != nil {
		t.Fatalf("ToolPreferenceWithError: %v", err)
	}
	want := []string{"local-read", "execute"}
	if !reflect.DeepEqual(preference, want) {
		t.Fatalf("tool preference = %#v, want %#v", preference, want)
	}
}
