package agent

import (
	"encoding/json"
	"strings"
	"testing"
)

// --- mergeFragmentWindow ---

func TestMergeFragmentWindow_OverlaysFragmentParams(t *testing.T) {
	orig := `{"path":"/abs/main.go","start_line":1,"end_line":500}`
	frag := `{"hash":"ab12","start_line":501,"num_lines":100}`
	got := mergeFragmentWindow(orig, frag)

	var m map[string]any
	if err := json.Unmarshal([]byte(got), &m); err != nil {
		t.Fatalf("merged args not valid JSON: %v (got %s)", err, got)
	}
	if m["path"] != "/abs/main.go" {
		t.Errorf("path = %v, want /abs/main.go (original args preserved)", m["path"])
	}
	if m["start_line"] != float64(501) {
		t.Errorf("start_line = %v, want 501 (fragment window wins)", m["start_line"])
	}
	if m["num_lines"] != float64(100) {
		t.Errorf("num_lines = %v, want 100", m["num_lines"])
	}
	if m["hash"] != "ab12" {
		t.Errorf("hash = %v, want ab12", m["hash"])
	}
	if _, ok := m["end_line"]; ok {
		t.Error("end_line must be dropped: it describes the original window, not the fragment")
	}
}

func TestMergeFragmentWindow_SingleLineDropsStaleStart(t *testing.T) {
	// orig includes num_lines to model a cached tool_result_read result being
	// re-read via the `line` escape hatch: line takes precedence over both
	// start_line and num_lines, so both must be dropped.
	orig := `{"path":"/a.txt","start_line":1,"num_lines":300,"end_line":500}`
	frag := `{"hash":"cd34","line":42}`
	got := mergeFragmentWindow(orig, frag)

	var m map[string]any
	if err := json.Unmarshal([]byte(got), &m); err != nil {
		t.Fatalf("merged args not valid JSON: %v (got %s)", err, got)
	}
	if m["line"] != float64(42) {
		t.Errorf("line = %v, want 42", m["line"])
	}
	if _, ok := m["start_line"]; ok {
		t.Error("stale start_line must be dropped for single-line fragments")
	}
	if _, ok := m["num_lines"]; ok {
		t.Error("stale num_lines must be dropped for single-line fragments")
	}
	if _, ok := m["end_line"]; ok {
		t.Error("end_line must be dropped")
	}
}

func TestMergeFragmentWindow_HashOnlyKeepsEndLine(t *testing.T) {
	orig := `{"path":"/a.go","start_line":1,"end_line":500}`
	frag := `{"hash":"ef56"}`
	got := mergeFragmentWindow(orig, frag)

	var m map[string]any
	if err := json.Unmarshal([]byte(got), &m); err != nil {
		t.Fatalf("merged args not valid JSON: %v (got %s)", err, got)
	}
	if m["hash"] != "ef56" {
		t.Errorf("hash = %v, want ef56", m["hash"])
	}
	if m["start_line"] != float64(1) {
		t.Errorf("start_line = %v, want 1 preserved (hash-only fragment has no window)", m["start_line"])
	}
	if m["end_line"] != float64(500) {
		t.Errorf("end_line = %v, want 500 preserved (hash-only fragment has no window)", m["end_line"])
	}
}

func TestMergeFragmentWindow_InvalidJSONReturnsOrig(t *testing.T) {
	orig := `{"path":"/a.go"}`
	if got := mergeFragmentWindow(orig, `{bad`); got != orig {
		t.Errorf("unparseable fragment args: got %q, want orig unchanged", got)
	}
	if got := mergeFragmentWindow(`{bad`, `{}`); got != "{bad" {
		t.Errorf("unparseable original args: got %q, want orig unchanged", got)
	}
	// Non-object JSON (e.g. an array) is equally unusable as a base map.
	if got := mergeFragmentWindow(`[1,2]`, `{"hash":"x"}`); got != "[1,2]" {
		t.Errorf("non-object original args: got %q, want orig unchanged", got)
	}
}

// --- resolveToolResultReadDisplay ---

func TestResolveToolResultReadDisplay(t *testing.T) {
	cache := NewToolResultCache(0)
	hash := cache.Store("read_file", "cached content", ToolCacheMeta{
		Input: `{"path":"/abs/main.go","start_line":1,"end_line":500}`,
	})

	exec := newExecutorDefaultHITL(&mockLLMCaller{}, newMockToolExecutor(), &mockTokenCounter{}, 5, nil, false, ToolResultBudget{}, defaultCircuitBreakerConfig)
	exec.SetToolCache(cache)

	// Resolved hash: original base name + merged original args.
	base, input, ok := exec.resolveToolResultReadDisplay("tool_result_read",
		json.RawMessage(`{"hash":"`+hash+`","start_line":501,"num_lines":100}`))
	if !ok {
		t.Fatal("expected hash to resolve")
	}
	if base != "read_file" {
		t.Errorf("baseName = %q, want read_file", base)
	}
	var m map[string]any
	if err := json.Unmarshal([]byte(input), &m); err != nil {
		t.Fatalf("display input not valid JSON: %v (got %s)", err, input)
	}
	if m["path"] != "/abs/main.go" {
		t.Errorf("path = %v, want /abs/main.go (original args for the card title)", m["path"])
	}
	if m["start_line"] != float64(501) {
		t.Errorf("start_line = %v, want 501 (fragment window)", m["start_line"])
	}

	// Unknown hash: unchanged passthrough.
	base, input, ok = exec.resolveToolResultReadDisplay("tool_result_read", json.RawMessage(`{"hash":"zzzz"}`))
	if ok || base != "tool_result_read" || input != `{"hash":"zzzz"}` {
		t.Errorf("unresolved hash must pass through unchanged, got (%q, %q, %v)", base, input, ok)
	}

	// Other tools: unchanged passthrough.
	base, input, ok = exec.resolveToolResultReadDisplay("read_file", json.RawMessage(`{"path":"/x"}`))
	if ok || base != "read_file" || input != `{"path":"/x"}` {
		t.Errorf("non-tool_result_read must pass through unchanged, got (%q, %q, %v)", base, input, ok)
	}

	// Entry without retained input (oversized or stored by an older build):
	// the name still resolves, the args stay fragment-only.
	hash2 := cache.Store("web_fetch", "<html>…</html>", ToolCacheMeta{})
	base, input, ok = exec.resolveToolResultReadDisplay("tool_result_read", json.RawMessage(`{"hash":"`+hash2+`"}`))
	if !ok || base != "web_fetch" {
		t.Fatalf("expected web_fetch resolution, got (%q, %v)", base, ok)
	}
	if input != `{"hash":"`+hash2+`"}` {
		t.Errorf("display input = %q, want fragment args unchanged", input)
	}
}

// --- Store input retention ---

func TestToolResultCache_Store_RetainsCappedInput(t *testing.T) {
	c := NewToolResultCache(0)

	small := c.Store("read_file", "content-a", ToolCacheMeta{Input: `{"path":"/a.go"}`})
	entry, ok := c.Get(small)
	if !ok {
		t.Fatal("entry not found")
	}
	if entry.Input != `{"path":"/a.go"}` {
		t.Errorf("Input = %q, want retained", entry.Input)
	}

	oversized := strings.Repeat("x", maxCacheInputBytes+1)
	large := c.Store("write_file", "content-b", ToolCacheMeta{Input: oversized})
	entry2, ok := c.Get(large)
	if !ok {
		t.Fatal("entry not found")
	}
	if entry2.Input != "" {
		t.Errorf("oversized Input must be dropped, got %d bytes", len(entry2.Input))
	}
}
