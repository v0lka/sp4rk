package agents

import (
	"errors"
	"path/filepath"
	"strings"
	"testing"

	"github.com/v0lka/sp4rk/safeio"
)

// TestParseAgent_TooLargeRefused is the OOM regression: an oversized
// AGENT.md planted in a scanned profile directory is refused with the
// explicit safeio error instead of being read into memory. The failure is a
// read error, not a *ParseError, so discovery keeps classifying the directory
// as ordinary noise (debug log) rather than warning about an invalid profile.
func TestParseAgent_TooLargeRefused(t *testing.T) {
	t.Parallel()

	dir := writeAgent(t, t.TempDir(), "oversized", strings.Repeat("a", int(safeio.DefaultMaxFileSize)+1))

	_, err := ParseAgent(filepath.Join(dir, "AGENT.md"), dir)
	if err == nil {
		t.Fatal("expected error for oversized AGENT.md, got nil")
	}
	if !errors.Is(err, safeio.ErrTooLarge) {
		t.Errorf("err = %v, want errors.Is(err, safeio.ErrTooLarge)", err)
	}
	var parseErr *ParseError
	if errors.As(err, &parseErr) {
		t.Errorf("oversized AGENT.md must be a read failure, not a *ParseError: %v", err)
	}
}

// TestParseAgent_JustUnderCapParses pins the other side of the cap: a large
// but legitimate AGENT.md (exactly DefaultMaxFileSize bytes) still parses.
func TestParseAgent_JustUnderCapParses(t *testing.T) {
	t.Parallel()

	const prefix = "---\nname: big-profile\ndescription: A profile with a large body.\n---\n"
	content := prefix + strings.Repeat("x", int(safeio.DefaultMaxFileSize)-len(prefix))
	dir := writeAgent(t, t.TempDir(), "big-profile", content)

	agent, err := ParseAgent(filepath.Join(dir, "AGENT.md"), dir)
	if err != nil {
		t.Fatalf("ParseAgent(large but within cap): %v", err)
	}
	if agent.Metadata.Name != "big-profile" {
		t.Errorf("Name = %q, want %q", agent.Metadata.Name, "big-profile")
	}
}
