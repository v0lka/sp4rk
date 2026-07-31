package agents

import (
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// writeAgent writes an AGENT.md with the given content into a directory named
// after the agent and returns the agent directory path.
func writeAgent(t *testing.T, parent, name, content string) string {
	t.Helper()
	dir := filepath.Join(parent, name)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", dir, err)
	}
	if err := os.WriteFile(filepath.Join(dir, "AGENT.md"), []byte(content), 0o644); err != nil {
		t.Fatalf("write AGENT.md: %v", err)
	}
	return dir
}

func TestParseAgent_AllFields(t *testing.T) {
	t.Parallel()

	const content = `---
name: code-reviewer
description: Reviews pull requests for correctness and style.
tools: edit_file,bash_exec,write_file
max-steps: 25
model: gpt-4o
allow-redelegate: true
hidden: false
color: "#e06c75"
---

You are a meticulous code reviewer. Always read the full diff before commenting.
`
	dir := writeAgent(t, t.TempDir(), "code-reviewer", content)

	agent, err := ParseAgent(filepath.Join(dir, "AGENT.md"), dir)
	if err != nil {
		t.Fatalf("ParseAgent: unexpected error: %v", err)
	}

	wantMeta := AgentMetadata{
		Name:            "code-reviewer",
		Description:     "Reviews pull requests for correctness and style.",
		Tools:           "edit_file,bash_exec,write_file",
		MaxSteps:        25,
		Model:           "gpt-4o",
		AllowRedelegate: true,
		Hidden:          false,
		Color:           "#e06c75",
	}
	if !reflect.DeepEqual(agent.Metadata, wantMeta) {
		t.Errorf("metadata = %#v, want %#v", agent.Metadata, wantMeta)
	}
	if agent.DirPath != dir {
		t.Errorf("DirPath = %q, want %q", agent.DirPath, dir)
	}
	if agent.Body != "You are a meticulous code reviewer. Always read the full diff before commenting.\n" {
		t.Errorf("Body = %q", agent.Body)
	}
}

func TestParseAgent_Minimal(t *testing.T) {
	t.Parallel()

	const content = "---\nname: researcher\ndescription: Investigates topics.\n---\nDo the research.\n"
	dir := writeAgent(t, t.TempDir(), "researcher", content)

	agent, err := ParseAgent(filepath.Join(dir, "AGENT.md"), dir)
	if err != nil {
		t.Fatalf("ParseAgent: unexpected error: %v", err)
	}
	if agent.Metadata.Tools != "" {
		t.Errorf("Tools = %q, want empty", agent.Metadata.Tools)
	}
	if agent.Metadata.MaxSteps != 0 {
		t.Errorf("MaxSteps = %d, want 0", agent.Metadata.MaxSteps)
	}
	if agent.Metadata.AllowRedelegate {
		t.Errorf("AllowRedelegate = true, want false")
	}
}

func TestParseAgent_UnknownFieldsIgnored(t *testing.T) {
	t.Parallel()

	// temperature/top-p, per-tool policy, mode, resources are OUT of v1 and
	// must be silently ignored (not a parse error).
	const content = "---\n" +
		"name: scientist\n" +
		"description: A scientist.\n" +
		"temperature: 0.3\n" +
		"top-p: 0.9\n" +
		"mode: blocking\n" +
		"resources:\n" +
		"  cpu: 2\n" +
		"policies:\n" +
		"  - edit_file: confirm\n" +
		"some-future-field: value\n" +
		"---\nBody.\n"
	dir := writeAgent(t, t.TempDir(), "scientist", content)

	agent, err := ParseAgent(filepath.Join(dir, "AGENT.md"), dir)
	if err != nil {
		t.Fatalf("ParseAgent: unknown fields must not error, got: %v", err)
	}
	if agent.Metadata.Name != "scientist" {
		t.Errorf("Name = %q", agent.Metadata.Name)
	}
}

func TestParseAgent_ValidationErrors(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		dirName string
		content string
	}{
		{
			name:    "missing name",
			dirName: "no-name",
			content: "---\ndescription: No name.\n---\nBody.\n",
		},
		{
			name:    "name mismatch dir",
			dirName: "mismatched",
			content: "---\nname: different\ndescription: Mismatch.\n---\nBody.\n",
		},
		{
			name:    "missing description",
			dirName: "no-desc",
			content: "---\nname: no-desc\n---\nBody.\n",
		},
		{
			name:    "bad regex uppercase",
			dirName: "Bad-Name",
			content: "---\nname: Bad-Name\ndescription: Upper.\n---\nBody.\n",
		},
		{
			name:    "bad regex leading hyphen",
			dirName: "-leading",
			content: "---\nname: -leading\ndescription: Leading hyphen.\n---\nBody.\n",
		},
		{
			name:    "bad regex trailing hyphen",
			dirName: "trailing-",
			content: "---\nname: trailing-\ndescription: Trailing hyphen.\n---\nBody.\n",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			dir := writeAgent(t, t.TempDir(), tt.dirName, tt.content)

			_, err := ParseAgent(filepath.Join(dir, "AGENT.md"), dir)
			if err == nil {
				t.Fatal("expected ParseError, got nil")
			}
			// Must be a *ParseError.
			var pe *ParseError
			if !errors.As(err, &pe) {
				t.Errorf("expected *ParseError, got %T: %v", err, err)
			}
		})
	}
}

func TestParseAgent_MissingFile(t *testing.T) {
	t.Parallel()

	_, err := ParseAgent(filepath.Join(t.TempDir(), "AGENT.md"), t.TempDir())
	if err == nil {
		t.Fatal("expected error for missing AGENT.md, got nil")
	}
}

func TestParseAgent_MissingFrontmatter(t *testing.T) {
	t.Parallel()

	dir := writeAgent(t, t.TempDir(), "nofm", "Just markdown, no frontmatter at all.")
	_, err := ParseAgent(filepath.Join(dir, "AGENT.md"), dir)
	if err == nil {
		t.Fatal("expected error for missing frontmatter, got nil")
	}
}

// TestParseAgent_ValidNameShapes exercises the name regex with boundary values.
func TestParseAgent_ValidNameShapes(t *testing.T) {
	t.Parallel()

	validNames := []string{"a", "ab", "a-b", "code-reviewer", "r2d2", "a1-b2"}
	for _, name := range validNames {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			dir := writeAgent(t, t.TempDir(), name,
				"---\nname: "+name+"\ndescription: d\n---\nBody.\n")
			if _, err := ParseAgent(filepath.Join(dir, "AGENT.md"), dir); err != nil {
				t.Errorf("name %q should be valid, got: %v", name, err)
			}
		})
	}
}
