package llm

import "testing"

func TestFamilyReasoningOptions(t *testing.T) {
	tests := []struct {
		family      string
		wantOptions []string
		wantDefault string
		wantOK      bool
	}{
		{"anthropic", []string{"On", "Off"}, "On", true},
		{"openai_flagship", []string{"minimal", "low", "medium", "high"}, "high", true},
		{"openai_standard", []string{"minimal", "low", "medium", "high"}, "high", true},
		{"openai_codex", []string{"minimal", "low", "medium", "high", "max"}, "max", true},
		{"google", []string{"MINIMAL", "LOW", "MEDIUM", "HIGH"}, "HIGH", true},
		{"deepseek", []string{"Off", "High", "Max"}, "Max", true},
		{"qwen", []string{"xhigh", "medium", "low", "Off"}, "xhigh", true},
		{"glm", []string{"On", "Off"}, "On", true},
		// Unsupported families
		{"mistral", nil, "", false},
		{"kimi", nil, "", false},
		{"default", nil, "", false},
		{"", nil, "", false},
	}

	for _, tt := range tests {
		t.Run(tt.family, func(t *testing.T) {
			opts, def, ok := FamilyReasoningOptions(tt.family)
			if ok != tt.wantOK {
				t.Errorf("ok = %v, want %v", ok, tt.wantOK)
			}
			if def != tt.wantDefault {
				t.Errorf("default = %q, want %q", def, tt.wantDefault)
			}
			if len(opts) != len(tt.wantOptions) {
				t.Errorf("options len = %d, want %d", len(opts), len(tt.wantOptions))
				return
			}
			for i, opt := range opts {
				if opt != tt.wantOptions[i] {
					t.Errorf("options[%d] = %q, want %q", i, opt, tt.wantOptions[i])
				}
			}
		})
	}
}

func TestIsGLM52OrLater(t *testing.T) {
	tests := []struct {
		model string
		want  bool
	}{
		{"glm-5.2", true},
		{"GLM-5.2", true}, // case-insensitive
		{"glm-5.3", true}, // future version
		{"glm-5.2-turbo", true},
		{"glm-5.1", false},
		{"glm-5", false},
		{"glm-4.7", false},
		{"glm-z1-32b", false}, // unversioned legacy
		{"chatglm-4", false},
		{"Zen/glm-5.2", true}, // composite id
		{"Zen/glm-5.1", false},
		{"", false},
		{"claude-sonnet-4", false},
	}
	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			if got := IsGLM52OrLater(tt.model); got != tt.want {
				t.Errorf("IsGLM52OrLater(%q) = %v, want %v", tt.model, got, tt.want)
			}
		})
	}
}

func TestIsQwen38OrLater(t *testing.T) {
	tests := []struct {
		model string
		want  bool
	}{
		{"qwen3.8-27b", true},
		{"Qwen3.8", true},   // case-insensitive
		{"qwen4-72b", true}, // future major
		{"qwen/qwen3.8-flash-next", true},
		{"qwen3.9-plus", true},
		{"qwen3-235b-a22b-instruct", false}, // version 3 without a minor
		{"qwen3-coder-480b-a35b", false},
		{"qwen2.5-72b-instruct", false},
		{"qwq-32b", false},    // unversioned Qwen-family reasoning model
		{"qvq-72b", false},    // unversioned non-reasoning variant
		{"qwen-turbo", false}, // unversioned legacy commercial name
		{"", false},
		{"claude-sonnet-4", false},
	}
	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			if got := IsQwen38OrLater(tt.model); got != tt.want {
				t.Errorf("IsQwen38OrLater(%q) = %v, want %v", tt.model, got, tt.want)
			}
		})
	}
}

func TestModelReasoningOptions(t *testing.T) {
	tests := []struct {
		name        string
		family      string
		model       string
		wantOptions []string
		wantDefault string
		wantOK      bool
	}{
		{
			name:        "glm 5.2 reasoning_effort",
			family:      "glm",
			model:       "glm-5.2",
			wantOptions: []string{"none", "max", "high"},
			wantDefault: "max",
			wantOK:      true,
		},
		{
			name:        "glm 5.1 legacy on/off",
			family:      "glm",
			model:       "glm-5.1",
			wantOptions: []string{"On", "Off"},
			wantDefault: "On",
			wantOK:      true,
		},
		{
			name:        "glm 4.7 legacy on/off",
			family:      "glm",
			model:       "glm-4.7",
			wantOptions: []string{"On", "Off"},
			wantDefault: "On",
			wantOK:      true,
		},
		{
			name:        "glm 5.2 composite id",
			family:      "glm",
			model:       "Zen/glm-5.2",
			wantOptions: []string{"none", "max", "high"},
			wantDefault: "max",
			wantOK:      true,
		},
		{
			name:        "glm 5.3 flash is always thinking",
			family:      "glm",
			model:       "zai-org/glm-5.3-flash",
			wantOptions: []string{"max", "high"},
			wantDefault: "max",
			wantOK:      true,
		},
		{
			name:        "glm 5.3 non-flash offers disable",
			family:      "glm",
			model:       "glm-5.3",
			wantOptions: []string{"none", "max", "high"},
			wantDefault: "max",
			wantOK:      true,
		},
		{
			name:        "qwen 3.8 native reasoning_effort",
			family:      "qwen",
			model:       "qwen3.8-27b",
			wantOptions: []string{"xhigh", "medium", "low", "Off"},
			wantDefault: "xhigh",
			wantOK:      true,
		},
		{
			name:        "qwen 3.8 composite id",
			family:      "qwen",
			model:       "qwen/qwen3.8-flash-next",
			wantOptions: []string{"xhigh", "medium", "low", "Off"},
			wantDefault: "xhigh",
			wantOK:      true,
		},
		{
			name:        "qwen 3 legacy on/off",
			family:      "qwen",
			model:       "qwen3-235b-a22b-instruct",
			wantOptions: []string{"On", "Off"},
			wantDefault: "On",
			wantOK:      true,
		},
		{
			name:        "qwen 2.5 legacy on/off",
			family:      "qwen",
			model:       "qwen2.5-72b-instruct",
			wantOptions: []string{"On", "Off"},
			wantDefault: "On",
			wantOK:      true,
		},
		{
			name:        "qwq unversioned legacy on/off",
			family:      "qwen",
			model:       "qwq-32b",
			wantOptions: []string{"On", "Off"},
			wantDefault: "On",
			wantOK:      true,
		},
		// Non-GLM families delegate to FamilyReasoningOptions unchanged.
		{
			name:        "anthropic delegates",
			family:      "anthropic",
			model:       "claude-sonnet-4",
			wantOptions: []string{"On", "Off"},
			wantDefault: "On",
			wantOK:      true,
		},
		{
			name:        "unsupported family",
			family:      "mistral",
			model:       "mistral-large",
			wantOptions: nil,
			wantDefault: "",
			wantOK:      false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			opts, def, ok := ModelReasoningOptions(tt.family, tt.model)
			if ok != tt.wantOK {
				t.Errorf("ok = %v, want %v", ok, tt.wantOK)
			}
			if def != tt.wantDefault {
				t.Errorf("default = %q, want %q", def, tt.wantDefault)
			}
			if len(opts) != len(tt.wantOptions) {
				t.Errorf("options len = %d, want %d", len(opts), len(tt.wantOptions))
				return
			}
			for i, opt := range opts {
				if opt != tt.wantOptions[i] {
					t.Errorf("options[%d] = %q, want %q", i, opt, tt.wantOptions[i])
				}
			}
		})
	}
}
