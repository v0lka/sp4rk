package agents

import (
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
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
tools: local-read,local-write,execute
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
		Tools:           "local-read,local-write,execute",
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

// TestParseAgent_InvalidTools verifies the `tools` frontmatter field is
// validated at parse time: unknown groups, empty items (stray commas), and
// mixing "all"/"read-only" with group tokens all fail the profile with a
// *ParseError carrying a comprehensible message (fail-closed — a typo must
// never silently widen or narrow a subagent's toolset).
func TestParseAgent_InvalidTools(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		tools   string
		wantMsg string
	}{
		{name: "unknown group", tools: "local-read,edit_file", wantMsg: `unknown tool group "edit_file"`},
		{name: "fully unknown single", tools: "everything", wantMsg: `unknown tool group "everything"`},
		{name: "empty item from stray comma", tools: "local-read,,execute", wantMsg: "empty item in group list"},
		{name: "all mixed with groups", tools: "all,execute", wantMsg: `"all" cannot be combined`},
		{name: "read-only mixed with groups", tools: "read-only,local-read", wantMsg: `"read-only" cannot be combined`},
		{name: "duplicate group", tools: "execute,execute", wantMsg: `duplicate group "execute"`},
		{name: "duplicate via mixed spelling", tools: "local-read,local_read", wantMsg: `duplicate group "local_read"`},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			dir := writeAgent(t, t.TempDir(), "tooler",
				"---\nname: tooler\ndescription: d\ntools: "+tt.tools+"\n---\nBody.\n")

			_, err := ParseAgent(filepath.Join(dir, "AGENT.md"), dir)
			if err == nil {
				t.Fatal("expected ParseError, got nil")
			}
			var pe *ParseError
			if !errors.As(err, &pe) {
				t.Fatalf("expected *ParseError, got %T: %v", err, err)
			}
			if !strings.Contains(pe.Message, tt.wantMsg) {
				t.Errorf("error message %q does not contain %q", pe.Message, tt.wantMsg)
			}
			// The unknown-group message must teach the valid alternatives
			// (comprehensible fail-closed error, not a bare rejection).
			if strings.Contains(tt.wantMsg, "unknown tool group") && !strings.Contains(pe.Message, "local-mcp") {
				t.Errorf("error message %q should list valid groups", pe.Message)
			}
		})
	}
}

// TestParseAgent_ValidTools accepted `tools` spellings round-trip cleanly.
func TestParseAgent_ValidTools(t *testing.T) {
	t.Parallel()

	valid := []string{"", "all", " all ", "read-only", "execute", "local-read", "remote-read,remote-write", "system,local-mcp,remote-mcp", "local_read", " execute , remote-write "}
	for _, tools := range valid {
		t.Run(tools, func(t *testing.T) {
			t.Parallel()
			dir := writeAgent(t, t.TempDir(), "tooler",
				"---\nname: tooler\ndescription: d\ntools: "+tools+"\n---\nBody.\n")
			if _, err := ParseAgent(filepath.Join(dir, "AGENT.md"), dir); err != nil {
				t.Errorf("tools %q should be valid, got: %v", tools, err)
			}
		})
	}
}
