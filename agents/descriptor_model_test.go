package agents

import (
	"encoding/json"
	"strings"
	"testing"
)

// TestDescriptor_CarriesModel verifies that the lightweight descriptor carries
// the profile's model override at discovery time: set on the descriptor when
// the frontmatter declares `model:`, empty when absent.
func TestDescriptor_CarriesModel(t *testing.T) {
	a := &Agent{Metadata: AgentMetadata{Name: "n", Description: "d", Model: "qwen3:32b"}}
	got := a.Descriptor()
	if got.Model != "qwen3:32b" {
		t.Errorf("Descriptor().Model = %q, want %q", got.Model, "qwen3:32b")
	}

	a = &Agent{Metadata: AgentMetadata{Name: "n", Description: "d"}}
	got = a.Descriptor()
	if got.Model != "" {
		t.Errorf("Descriptor().Model = %q, want \"\" (no frontmatter model)", got.Model)
	}
}

// TestAgentDescriptor_ModelJSONVisibility verifies the JSON contract of the
// descriptor's Model field: present when set, omitted when empty (omitempty —
// existing descriptor consumers see byte-identical JSON for model-less
// profiles).
func TestAgentDescriptor_ModelJSONVisibility(t *testing.T) {
	set := AgentDescriptor{Name: "n", Description: "d", Model: "qwen3:32b"}
	b, err := json.Marshal(set)
	if err != nil {
		t.Fatalf("marshal: unexpected error: %v", err)
	}
	if !strings.Contains(string(b), `"model":"qwen3:32b"`) {
		t.Errorf("set Model must serialize, got %s", b)
	}

	empty := AgentDescriptor{Name: "n", Description: "d"}
	b, err = json.Marshal(empty)
	if err != nil {
		t.Fatalf("marshal: unexpected error: %v", err)
	}
	if strings.Contains(string(b), `"model"`) {
		t.Errorf("empty Model must be omitted, got %s", b)
	}
}
