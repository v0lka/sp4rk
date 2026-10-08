package builtins

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// TestRipgrepTool_ZeroLimitsConstructor is the regression test for review
// finding 31: a zero-value RipgrepLimits used to be stored verbatim, so the
// per-call context deadline was zero and every search aborted immediately
// with "context deadline exceeded".
func TestRipgrepTool_ZeroLimitsConstructor(t *testing.T) {
	requireRipgrep(t)
	base := setupRipgrepTestDir(t)

	for name, tool := range map[string]*RipgrepTool{
		"WithLimits": NewRipgrepToolWithLimits(RipgrepLimits{}),
		"WithPath":   NewRipgrepToolWithPath(RipgrepLimits{}, ""),
	} {
		t.Run(name, func(t *testing.T) {
			if tool.limits.Timeout != DefaultRipgrepLimits().Timeout {
				t.Fatalf("normalized limits.Timeout = %v, want %v", tool.limits.Timeout, DefaultRipgrepLimits().Timeout)
			}

			input, err := json.Marshal(RipgrepInput{Pattern: "Hello", Path: base})
			if err != nil {
				t.Fatalf("marshal input: %v", err)
			}

			result, err := tool.Execute(context.Background(), input)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if result.IsError {
				t.Fatalf("zero-value limits made the tool non-functional: %s", result.Content)
			}
			if !strings.Contains(result.Content, "Hello") {
				t.Errorf("expected a match on 'Hello', got %q", result.Content)
			}
		})
	}
}

// TestRipgrepTool_EffectiveTimeout covers the Execute-side fallback for a
// directly constructed tool carrying zero-value limits.
func TestRipgrepTool_EffectiveTimeout(t *testing.T) {
	zero := &RipgrepTool{} // effectiveTimeout must not touch the embedded BaseTool
	if got := zero.effectiveTimeout(); got != DefaultRipgrepLimits().Timeout {
		t.Errorf("effectiveTimeout() on zero limits = %v, want %v", got, DefaultRipgrepLimits().Timeout)
	}
	populated := &RipgrepTool{}
	populated.limits = RipgrepLimits{Timeout: 3 * time.Second}
	if got := populated.effectiveTimeout(); got != 3*time.Second {
		t.Errorf("effectiveTimeout() on configured limits = %v, want 3s", got)
	}
}

// TestRipgrepTool_ResultCap is the regression test for review finding 25
// (ripgrep part): the tool used to accumulate its entire result in an
// unbounded strings.Builder — a search over a tree with more matching output
// than RAM could OOM the host, since the executor's truncation layer runs
// only after Execute returns. Output past maxRipgrepResultBytes must be
// dropped, the child must still exit promptly, and the result must carry an
// incompleteness marker.
func TestRipgrepTool_ResultCap(t *testing.T) {
	requireRipgrep(t)

	base := t.TempDir()
	f, err := os.Create(filepath.Join(base, "big.txt"))
	if err != nil {
		t.Fatalf("create test file: %v", err)
	}
	// ~12,000 lines of ~1 KiB, every line matching: ~12 MiB of match output,
	// 2 MiB past the 10 MiB result cap.
	pad := strings.Repeat("a", 1000)
	var sb strings.Builder
	for i := 0; i < 12000; i++ {
		sb.WriteString("needle ")
		sb.WriteString(pad)
		sb.WriteByte('\n')
	}
	if _, err := f.WriteString(sb.String()); err != nil {
		t.Fatalf("write test file: %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("close test file: %v", err)
	}

	tool := NewRipgrepTool()
	input, err := json.Marshal(RipgrepInput{Pattern: "needle", Path: base})
	if err != nil {
		t.Fatalf("marshal input: %v", err)
	}

	result, err := tool.Execute(context.Background(), input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Fatalf("unexpected error result: %s", result.Content)
	}
	if !strings.Contains(result.Content, "result cap") {
		t.Errorf("expected the result-cap marker in the result, got %d bytes without one", len(result.Content))
	}
	if len(result.Content) > maxRipgrepResultBytes+64*1024 {
		t.Errorf("result length = %d, want <= cap (%d) + marker overhead", len(result.Content), maxRipgrepResultBytes)
	}
	if !strings.Contains(result.Content, "Found ") {
		t.Errorf("expected the 'Found N matches' summary, got %q", result.Content)
	}
}
