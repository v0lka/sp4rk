package skills

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/v0lka/sp4rk/safeio"
)

// TestParseSkillFileTooLarge is the OOM regression: an oversized SKILL.md
// planted in a scanned skill directory is refused with the explicit safeio
// error instead of being read into memory. The failure is a read error, not a
// *ParseError, so discovery keeps treating the directory as ordinary noise
// (debug log) rather than warning about an invalid skill.
func TestParseSkillFileTooLarge(t *testing.T) {
	t.Parallel()
	skillDir := filepath.Join(t.TempDir(), "big-skill")
	if err := os.MkdirAll(skillDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(skillDir, "SKILL.md"), make([]byte, safeio.DefaultMaxFileSize+1), 0o644); err != nil {
		t.Fatal(err)
	}

	_, err := ParseSkill(filepath.Join(skillDir, "SKILL.md"), skillDir)
	if err == nil {
		t.Fatal("expected error for oversized SKILL.md, got nil")
	}
	if !errors.Is(err, safeio.ErrTooLarge) {
		t.Errorf("err = %v, want errors.Is(err, safeio.ErrTooLarge)", err)
	}
	var parseErr *ParseError
	if errors.As(err, &parseErr) {
		t.Errorf("oversized SKILL.md must be a read failure, not a *ParseError: %v", err)
	}
}

// TestParseSkillJustUnderCapParses pins the other side of the cap: a large
// but legitimate SKILL.md (exactly DefaultMaxFileSize bytes) still parses.
func TestParseSkillJustUnderCapParses(t *testing.T) {
	t.Parallel()
	prefix := "---\nname: big-skill\ndescription: A skill with a large body.\n---\n"
	content := prefix + strings.Repeat("x", int(safeio.DefaultMaxFileSize)-len(prefix))
	skillDir := filepath.Join(t.TempDir(), "big-skill")
	if err := os.MkdirAll(skillDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(skillDir, "SKILL.md"), []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	skill, err := ParseSkill(filepath.Join(skillDir, "SKILL.md"), skillDir)
	if err != nil {
		t.Fatalf("ParseSkill(large but within cap): %v", err)
	}
	if skill.Metadata.Name != "big-skill" {
		t.Errorf("Name = %q, want %q", skill.Metadata.Name, "big-skill")
	}
}

// TestReadSkillResourceToolExecuteTooLarge is the OOM regression for the
// resource tool: an oversized resource planted in a skill directory is
// reported as an explicit error result instead of being read into memory.
func TestReadSkillResourceToolExecuteTooLarge(t *testing.T) {
	t.Parallel()
	skillDir := filepath.Join(t.TempDir(), "my-skill")
	if err := os.MkdirAll(skillDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(skillDir, "huge.md"), make([]byte, safeio.DefaultMaxFileSize+1), 0o644); err != nil {
		t.Fatal(err)
	}

	resolver := func(_ context.Context, name string) (string, bool) {
		if name == "my-skill" {
			return skillDir, true
		}
		return "", false
	}
	tool := NewReadSkillResourceTool(resolver)

	input := json.RawMessage(`{"skill": "my-skill", "path": "huge.md"}`)
	result, err := tool.Execute(context.Background(), input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Fatalf("Execute succeeded, want a too-large error result (content: %q)", result.Content)
	}
	if !strings.Contains(result.Content, "read limit") {
		t.Errorf("Content = %q, want the explicit read-limit message", result.Content)
	}
}
