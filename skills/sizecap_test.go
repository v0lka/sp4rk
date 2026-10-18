package skills

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/v0lka/sp4rk/safeio"
)

// sizecapRecordingHandler captures log records emitted during a scan so the
// oversized-skip test can assert on log levels without string-matching
// stderr.
type sizecapRecordingHandler struct {
	records []slog.Record
}

func (h *sizecapRecordingHandler) Enabled(_ context.Context, _ slog.Level) bool { return true }

func (h *sizecapRecordingHandler) Handle(_ context.Context, r slog.Record) error {
	h.records = append(h.records, r)
	return nil
}

func (h *sizecapRecordingHandler) WithAttrs(_ []slog.Attr) slog.Handler { return h }

func (h *sizecapRecordingHandler) WithGroup(_ string) slog.Handler { return h }

func (h *sizecapRecordingHandler) hasRecord(level slog.Level, msg string) bool {
	for _, r := range h.records {
		if r.Level == level && r.Message == msg {
			return true
		}
	}
	return false
}

// TestParseSkillFileTooLarge is the OOM regression: an oversized SKILL.md
// planted in a scanned skill directory is refused with the explicit safeio
// error instead of being read into memory. The failure is a read error, not
// a *ParseError, and discovery classifies it as an oversized skip (Warn —
// the operator loses a skill that exists but cannot be read) while discovery
// of the remaining skills continues; see
// TestSkillManager_OversizedSkillSkippedWithWarn.
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

// TestSkillManager_OversizedSkillSkippedWithWarn pins the discovery-side
// handling of a too-large SKILL.md (review finding 33): the skill is skipped
// and discovery of the remaining skills continues, and the skip is logged at
// Warn — the operator loses a skill that exists but could not be read (the
// safeio read cap refused it), which is the same visibility class as the
// unreadable-directory Warn already emitted by scanDir.
func TestSkillManager_OversizedSkillSkippedWithWarn(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	// A valid skill the scan must still find.
	goodDir := filepath.Join(dir, "good-skill")
	if err := os.MkdirAll(goodDir, 0o755); err != nil {
		t.Fatal(err)
	}
	goodMD := "---\nname: good-skill\ndescription: ok\n---\nBody.\n"
	if err := os.WriteFile(filepath.Join(goodDir, "SKILL.md"), []byte(goodMD), 0o644); err != nil {
		t.Fatal(err)
	}
	// An oversized SKILL.md: exists, but exceeds the safeio read cap.
	bigDir := filepath.Join(dir, "big-skill")
	if err := os.MkdirAll(bigDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(bigDir, "SKILL.md"), make([]byte, safeio.DefaultMaxFileSize+1), 0o644); err != nil {
		t.Fatal(err)
	}

	h := &sizecapRecordingHandler{}
	mgr := NewSkillManager([]string{dir}, slog.New(h))
	if err := mgr.Scan(); err != nil {
		t.Fatalf("Scan: %v", err)
	}

	if !h.hasRecord(slog.LevelWarn, "skipped oversized skill") {
		t.Errorf("expected a Warn record %q for the oversized skill, got records: %+v", "skipped oversized skill", h.records)
	}
	if _, ok := mgr.Get("good-skill"); !ok {
		t.Error("expected 'good-skill' to be loaded: an oversized skill must not break discovery of the others")
	}
	if _, ok := mgr.Get("big-skill"); ok {
		t.Error("oversized skill must not be in the catalog")
	}
}
