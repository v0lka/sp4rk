package agents

import (
	"context"
	"log/slog"
	"os"
	"path/filepath"
	"testing"
)

func TestAgentManager_PriorityFirstWins(t *testing.T) {
	t.Parallel()

	high := t.TempDir() // higher priority
	low := t.TempDir()  // lower priority

	// Same agent name in both dirs; descriptions differ.
	writeAgent(t, low, "shared", "---\nname: shared\ndescription: low priority copy\n---\nLow body.\n")
	writeAgent(t, high, "shared", "---\nname: shared\ndescription: high priority copy\n---\nHigh body.\n")

	// dirs order: [high, low] (highest first). High must win.
	mgr := NewAgentManager([]string{high, low}, nil)
	if err := mgr.Scan(); err != nil {
		t.Fatalf("Scan: %v", err)
	}

	got, ok := mgr.Get("shared")
	if !ok {
		t.Fatal("expected agent 'shared' to be found")
	}
	if got.Metadata.Description != "high priority copy" {
		t.Errorf("Description = %q, want high priority copy (first/highest must win)", got.Metadata.Description)
	}
	if got.Body != "High body.\n" {
		t.Errorf("Body = %q, want high body", got.Body)
	}
}

func TestAgentManager_GetMissing(t *testing.T) {
	t.Parallel()

	mgr := NewAgentManager([]string{t.TempDir()}, nil)
	if err := mgr.Scan(); err != nil {
		t.Fatalf("Scan: %v", err)
	}
	if _, ok := mgr.Get("does-not-exist"); ok {
		t.Error("expected Get to return false for missing agent")
	}
}

func TestAgentManager_List(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	writeAgent(t, dir, "alpha", "---\nname: alpha\ndescription: Alpha agent\n---\nA.\n")
	writeAgent(t, dir, "beta", "---\nname: beta\ndescription: Beta agent\n---\nB.\n")

	mgr := NewAgentManager([]string{dir}, nil)
	if err := mgr.Scan(); err != nil {
		t.Fatalf("Scan: %v", err)
	}

	descs := mgr.List()
	if len(descs) != 2 {
		t.Fatalf("List() returned %d descriptors, want 2", len(descs))
	}
	byName := map[string]AgentDescriptor{}
	for _, d := range descs {
		byName[d.Name] = d
	}
	if d, ok := byName["alpha"]; !ok || d.Description != "Alpha agent" {
		t.Errorf("alpha descriptor = %#v", d)
	}
	if d, ok := byName["beta"]; !ok || d.Description != "Beta agent" {
		t.Errorf("beta descriptor = %#v", d)
	}
}

func TestAgentManager_HiddenInDescriptorAndGet(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	writeAgent(t, dir, "visible", "---\nname: visible\ndescription: shown\n---\nV.\n")
	writeAgent(t, dir, "secret", "---\nname: secret\ndescription: hidden one\nhidden: true\n---\nS.\n")

	mgr := NewAgentManager([]string{dir}, nil)
	if err := mgr.Scan(); err != nil {
		t.Fatalf("Scan: %v", err)
	}

	// Hidden agent is still retrievable via Get (hiding affects discovery only).
	secret, ok := mgr.Get("secret")
	if !ok {
		t.Fatal("expected hidden agent to be retrievable via Get")
	}
	if !secret.Metadata.Hidden {
		t.Error("expected secret agent Hidden=true")
	}

	// List returns the hidden flag so the consumer can filter.
	descs := mgr.List()
	hiddenSeen := false
	for _, d := range descs {
		if d.Name == "secret" {
			if !d.Hidden {
				t.Error("expected secret descriptor Hidden=true")
			}
			hiddenSeen = true
		}
	}
	if !hiddenSeen {
		t.Error("List() did not include hidden agent")
	}
}

func TestAgentManager_InvalidAgentSkipped(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	// A valid agent.
	writeAgent(t, dir, "good", "---\nname: good\ndescription: ok\n---\nG.\n")
	// An invalid agent (name != dir).
	writeAgent(t, dir, "bad", "---\nname: not-bad\ndescription: mismatch\n---\nB.\n")

	mgr := NewAgentManager([]string{dir}, nil)
	if err := mgr.Scan(); err != nil {
		t.Fatalf("Scan: %v", err)
	}

	if _, ok := mgr.Get("good"); !ok {
		t.Error("expected 'good' to be loaded")
	}
	if _, ok := mgr.Get("bad"); ok {
		t.Error("expected 'bad' to be skipped (invalid)")
	}
	if _, ok := mgr.Get("not-bad"); ok {
		t.Error("expected 'not-bad' to be absent (dir was 'bad')")
	}
}

func TestAgentManager_RescanClears(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	writeAgent(t, dir, "first", "---\nname: first\ndescription: one\n---\n1.\n")

	mgr := NewAgentManager([]string{dir}, nil)
	if err := mgr.Scan(); err != nil {
		t.Fatalf("Scan: %v", err)
	}
	if _, ok := mgr.Get("first"); !ok {
		t.Fatal("expected 'first' after initial scan")
	}

	// Rescan with an empty dir set: previous agents must be cleared.
	mgr2 := NewAgentManager([]string{t.TempDir()}, nil)
	if err := mgr2.Scan(); err != nil {
		t.Fatalf("Scan on empty dir: %v", err)
	}
	if _, ok := mgr2.Get("first"); ok {
		t.Error("expected 'first' to be gone after rescan on different (empty) dir")
	}
}

func TestAgentManager_SymlinkDir(t *testing.T) {
	t.Parallel()

	// Real agent directory lives under "real"; we expose it via a symlink under
	// "scanroot" to verify symlinked agent directories are dereferenced.
	scanRoot := t.TempDir()
	realRoot := t.TempDir()
	writeAgent(t, realRoot, "symlinked", "---\nname: symlinked\ndescription: via symlink\n---\nS.\n")

	linkTarget := filepath.Join(realRoot, "symlinked")
	linkPath := filepath.Join(scanRoot, "symlinked")
	if err := os.Symlink(linkTarget, linkPath); err != nil {
		// Some CI sandboxes disallow symlinks; skip rather than fail.
		t.Skipf("cannot create symlink: %v", err)
	}

	mgr := NewAgentManager([]string{scanRoot}, nil)
	if err := mgr.Scan(); err != nil {
		t.Fatalf("Scan: %v", err)
	}

	got, ok := mgr.Get("symlinked")
	if !ok {
		t.Fatal("expected symlinked agent to be discovered via symlink")
	}
	if got.Metadata.Description != "via symlink" {
		t.Errorf("Description = %q", got.Metadata.Description)
	}
}

func TestAgentManager_NonexistentDirSkipped(t *testing.T) {
	t.Parallel()

	mgr := NewAgentManager([]string{filepath.Join(t.TempDir(), "does-not-exist")}, nil)
	if err := mgr.Scan(); err != nil {
		t.Fatalf("Scan on missing dir must not error, got: %v", err)
	}
	if len(mgr.List()) != 0 {
		t.Errorf("expected no agents from missing dir, got %d", len(mgr.List()))
	}
}

// recordingHandler captures log records emitted during a scan so tests can
// assert on levels without string-matching stderr.
type recordingHandler struct {
	records []slog.Record
}

func (h *recordingHandler) Enabled(_ context.Context, _ slog.Level) bool { return true }

func (h *recordingHandler) Handle(_ context.Context, r slog.Record) error {
	h.records = append(h.records, r)
	return nil
}

func (h *recordingHandler) WithAttrs(_ []slog.Attr) slog.Handler { return h }

func (h *recordingHandler) WithGroup(_ string) slog.Handler { return h }

// hasRecord reports whether any captured record has the given level and
// message.
func (h *recordingHandler) hasRecord(level slog.Level, msg string) bool {
	for _, r := range h.records {
		if r.Level == level && r.Message == msg {
			return true
		}
	}
	return false
}

// TestAgentManager_ScanSkipLogging verifies the visibility of skipped
// entries: an AGENT.md that exists but fails validation (e.g. a `tools:`
// list still using pre-group tool names) is logged at Warn — the operator
// silently loses that profile otherwise — while a directory that simply has
// no AGENT.md stays at Debug.
func TestAgentManager_ScanSkipLogging(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	// Invalid profile: tools uses tool names, not group tokens.
	writeAgent(t, dir, "reviewer",
		"---\nname: reviewer\ndescription: reviews\ntools: edit_file,bash_exec\n---\nBody.\n")
	// Not an agent at all: no AGENT.md inside.
	if err := os.MkdirAll(filepath.Join(dir, "not-an-agent"), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}

	h := &recordingHandler{}
	mgr := NewAgentManager([]string{dir}, slog.New(h))
	if err := mgr.Scan(); err != nil {
		t.Fatalf("Scan: %v", err)
	}

	if !h.hasRecord(slog.LevelWarn, "skipped invalid agent") {
		t.Errorf("expected a Warn record %q for the invalid profile, got records: %+v", "skipped invalid agent", h.records)
	}
	if h.hasRecord(slog.LevelWarn, "skipped non-agent directory") {
		t.Error("non-agent directories must not be logged at Warn")
	}
	if !h.hasRecord(slog.LevelDebug, "skipped non-agent directory") {
		t.Errorf("expected a Debug record %q for the plain directory, got records: %+v", "skipped non-agent directory", h.records)
	}
	if _, ok := mgr.Get("reviewer"); ok {
		t.Error("invalid profile must not be in the catalog")
	}
}
