package builtins

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	doublestar "github.com/bmatcuk/doublestar/v4"
)

// TestNewGlobToolWithLimits_ZeroStructFallback proves the constructor is
// fail-safe for an all-zero GlobLimits (so a zero-value BuiltinToolsConfig
// cannot register an unbounded walk) while preserving an explicit per-field
// zero as "disabled" on an otherwise-populated struct, and passing an explicit
// non-zero value through verbatim.
func TestNewGlobToolWithLimits_ZeroStructFallback(t *testing.T) {
	def := DefaultGlobLimits()

	t.Run("all-zero struct falls back to defaults", func(t *testing.T) {
		tool := NewGlobToolWithLimits(GlobLimits{})
		if tool.limits != def {
			t.Fatalf("zero GlobLimits: got %+v, want %+v", tool.limits, def)
		}
	})

	t.Run("individual zero fields are honored as disabled", func(t *testing.T) {
		// A per-field fill would silently turn these zeros into the defaults and
		// make "0 = disabled" unreachable; the fallback must be all-or-nothing.
		tool := NewGlobToolWithLimits(GlobLimits{MaxResults: 3})
		want := GlobLimits{MaxEntries: 0, MaxResults: 3, Timeout: 0}
		if tool.limits != want {
			t.Fatalf("partial GlobLimits: got %+v, want %+v", tool.limits, want)
		}
	})

	t.Run("explicit values are preserved", func(t *testing.T) {
		explicit := GlobLimits{MaxEntries: 7, MaxResults: 11, Timeout: 90 * time.Second}
		if got := NewGlobToolWithLimits(explicit).limits; got != explicit {
			t.Fatalf("explicit GlobLimits: got %+v, want %+v", got, explicit)
		}
	})

	t.Run("NewGlobTool uses the defaults", func(t *testing.T) {
		if got := NewGlobTool().limits; got != def {
			t.Fatalf("NewGlobTool: got %+v, want %+v", got, def)
		}
	})
}

// TestGlobTool_MaxResultsCapReturnsPartialResults proves the result cap does not
// discard the matches already collected: the walk returns the capped matches
// plus a warning suffix as a non-error result.
func TestGlobTool_MaxResultsCapReturnsPartialResults(t *testing.T) {
	base := t.TempDir()
	for i := 0; i < 6; i++ {
		if err := os.WriteFile(filepath.Join(base, fmt.Sprintf("f%d.txt", i)), []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	tool := NewGlobToolWithLimits(GlobLimits{MaxResults: 3})
	input, _ := json.Marshal(GlobInput{Pattern: "*.txt", Path: base, Type: "files"})

	result, err := tool.Execute(context.Background(), input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Fatalf("bound-abort must not be an error result, got: %s", result.Content)
	}
	if lines := globResultLines(result.Content); len(lines) != 3 {
		t.Fatalf("expected the 3 collected matches to be returned, got %d: %q", len(lines), result.Content)
	}
	if !strings.Contains(result.Content, "results limited to 3") {
		t.Errorf("expected a result-cap warning, got: %q", result.Content)
	}
}

// TestGlobTool_EntryBudgetReturnsPartialResults proves the entry budget does not
// discard the matches already collected and that the walk abort is reported as a
// non-error warning rather than a hard error.
func TestGlobTool_EntryBudgetReturnsPartialResults(t *testing.T) {
	base := t.TempDir()
	if err := os.WriteFile(filepath.Join(base, "a.txt"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(base, "sub"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(base, "sub", "b.txt"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(base, "sub", "c.txt"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}

	input, _ := json.Marshal(GlobInput{Pattern: "**/*.txt", Path: base, Type: "files"})

	// A budget large enough for the root read and the first match, but not for
	// the nested directory read: the walk keeps `a.txt` and warns.
	tool := NewGlobToolWithLimits(GlobLimits{MaxEntries: 3})
	result, err := tool.Execute(context.Background(), input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Fatalf("entry-budget abort must not be an error result, got: %s", result.Content)
	}
	if !strings.Contains(result.Content, "filesystem entries") {
		t.Errorf("expected an entry-budget warning, got: %q", result.Content)
	}
	if !strings.Contains(result.Content, "a.txt") {
		t.Errorf("expected the match collected before the abort to be kept, got: %q", result.Content)
	}

	// A budget of one aborts during the root read: no match is returned, but the
	// bound is still reported as a non-error warning, never "no matching files".
	tiny := NewGlobToolWithLimits(GlobLimits{MaxEntries: 1})
	result, err = tiny.Execute(context.Background(), input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Fatalf("entry-budget abort must not be an error result, got: %s", result.Content)
	}
	if !strings.Contains(result.Content, "filesystem entries") {
		t.Errorf("expected an entry-budget warning, got: %q", result.Content)
	}
}

// TestBoundedFS_Charge unit-tests the two guards charge() enforces.
func TestBoundedFS_Charge(t *testing.T) {
	ctx := context.Background()

	t.Run("entry budget", func(t *testing.T) {
		b := newBoundedFS(ctx, os.DirFS(t.TempDir()), GlobLimits{MaxEntries: 2})
		if err := b.charge(1); err != nil {
			t.Fatalf("charge 1/2: %v", err)
		}
		if err := b.charge(1); err != nil {
			t.Fatalf("charge 2/2: %v", err)
		}
		if err := b.charge(1); !errors.Is(err, errGlobEntryBudget) {
			t.Fatalf("charge over budget: got %v, want errGlobEntryBudget", err)
		}
	})

	t.Run("non-positive budget is disabled", func(t *testing.T) {
		b := newBoundedFS(ctx, os.DirFS(t.TempDir()), GlobLimits{MaxEntries: 0})
		for i := 0; i < 100; i++ {
			if err := b.charge(1); err != nil {
				t.Fatalf("disabled budget charged %d: %v", i, err)
			}
		}
	})

	t.Run("done context wins over the budget", func(t *testing.T) {
		cctx, cancel := context.WithCancel(context.Background())
		cancel()
		b := newBoundedFS(cctx, os.DirFS(t.TempDir()), GlobLimits{MaxEntries: 100})
		if err := b.charge(1); !errors.Is(err, errGlobCanceled) {
			t.Fatalf("charge with done ctx: got %v, want errGlobCanceled", err)
		}
	})
}

// TestGlobTool_WalkWarning covers every branch of walkWarning and the
// error/result split it drives: the glob's own bounds and benign I/O errors
// yield a warning (partial results kept), while a caller cancellation/deadline
// and a malformed pattern do not.
func TestGlobTool_WalkWarning(t *testing.T) {
	tool := NewGlobTool()
	bg := context.Background()

	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	deadline, cancelDeadline := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
	defer cancelDeadline()

	cases := []struct {
		name    string
		ctx     context.Context
		err     error
		wantSub string
		wantAny bool // true: any non-empty warning is acceptable
	}{
		{"results limited", bg, errGlobResultsLimited, "results limited to", true},
		{"entry budget", bg, errGlobEntryBudget, "filesystem entries", true},
		{"own timeout (live caller ctx)", bg, errGlobCanceled, "timed out after", true},
		{"caller cancellation is not a warning", cancelled, errGlobCanceled, "", false},
		{"caller deadline is not a warning", deadline, errGlobCanceled, "", false},
		{"permission I/O error is swallowed", bg, &fs.PathError{Op: "open", Path: "x", Err: fs.ErrPermission}, "some entries could not be read", true},
		{"bad pattern is not a warning", bg, doublestar.ErrBadPattern, "", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := tool.walkWarning(tc.ctx, tc.err)
			if !tc.wantAny {
				if got != "" {
					t.Fatalf("got warning %q, want none (hard-error path)", got)
				}
				return
			}
			if !strings.Contains(got, tc.wantSub) {
				t.Fatalf("got warning %q, want substring %q", got, tc.wantSub)
			}
		})
	}
}

// TestGlobTool_WalkErrorMessage covers the hard-error branch, including the
// generic (duration-free) context-cancellation messages.
func TestGlobTool_WalkErrorMessage(t *testing.T) {
	tool := NewGlobTool()

	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	deadline, cancelDeadline := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
	defer cancelDeadline()

	if got := tool.walkErrorMessage(cancelled, errGlobCanceled); got != "glob canceled" {
		t.Errorf("cancelled ctx: got %q", got)
	}
	if got := tool.walkErrorMessage(deadline, errGlobCanceled); got != "glob interrupted: context deadline exceeded" {
		t.Errorf("deadline ctx: got %q", got)
	}
	if got := tool.walkErrorMessage(context.Background(), doublestar.ErrBadPattern); !strings.Contains(got, "glob error:") {
		t.Errorf("bad pattern: got %q", got)
	}
}

// TestIsSkippableWalkError pins the per-entry-error classifier.
func TestIsSkippableWalkError(t *testing.T) {
	if !isSkippableWalkError(&fs.PathError{Op: "open", Path: "x", Err: fs.ErrPermission}) {
		t.Error("permission PathError must be skippable")
	}
	if !isSkippableWalkError(&fs.PathError{Op: "open", Path: "x", Err: fs.ErrNotExist}) {
		t.Error("not-exist PathError must be skippable")
	}
	if !isSkippableWalkError(fs.ErrPermission) {
		t.Error("unwrapped permission error must be skippable")
	}
	if isSkippableWalkError(doublestar.ErrBadPattern) {
		t.Error("ErrBadPattern must not be skippable")
	}
	if isSkippableWalkError(errGlobCanceled) || isSkippableWalkError(errGlobEntryBudget) || isSkippableWalkError(errGlobResultsLimited) {
		t.Error("boundedFS sentinels must not be classified as skippable I/O errors")
	}
}
