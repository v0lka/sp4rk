//go:build unix

package ignore

import (
	"context"
	"os"
	"path/filepath"
	"syscall"
	"testing"
	"time"
)

// TestResolver_SkipsNonRegularIgnoreFile is the regression test for a
// shutdown freeze: an ignore-named entry that is not a regular file (here a
// FIFO) beneath a walked root. A read-open (O_RDONLY) of a FIFO blocks until
// a writer appears, and load's per-entry context check cannot interrupt a
// goroutine blocked inside the open(2) syscall — so before the guard the walk
// never returned and any shutdown join waiting on it (the session manager's
// stopBackground, on the main thread) burned its whole budget in a visible
// freeze. The walk must instead skip the entry and finish.
//
// A bound (5s) turns the bug into a deterministic failure rather than a test
// that hangs forever.
func TestResolver_SkipsNonRegularIgnoreFile(t *testing.T) {
	root := t.TempDir()

	// A regular root ignore file whose rule MUST still be honoured — proving
	// the guard skips only the non-regular entry, not ignore files in general.
	if err := os.WriteFile(filepath.Join(root, ".gitignore"), []byte("root-ignored.txt\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	// A subdirectory holding a FIFO named ".gitignore": reading it would block
	// forever.
	sub := filepath.Join(root, "sub")
	if err := os.Mkdir(sub, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := syscall.Mkfifo(filepath.Join(sub, ".gitignore"), 0o644); err != nil {
		t.Fatalf("mkfifo: %v", err)
	}

	type result struct {
		r   *Resolver
		err error
	}
	done := make(chan result, 1)
	start := time.Now()
	go func() {
		r, err := NewResolverContext(context.Background(), root)
		done <- result{r: r, err: err}
	}()

	select {
	case res := <-done:
		if res.err != nil {
			t.Fatalf("NewResolverContext: %v", res.err)
		}
		if !res.r.Match("root-ignored.txt", false) {
			t.Error("regular .gitignore rule was not honoured after skipping the FIFO")
		}
	case <-time.After(5 * time.Second):
		t.Fatalf("NewResolverContext hung on a FIFO named .gitignore (no return in %s)", time.Since(start).Round(time.Millisecond))
	}
}

// TestResolver_FollowsSymlinkedIgnoreFile guards the behaviour the FIFO guard
// must preserve: a symlink to a regular ignore file is still followed and its
// rules honoured (the guard resolves the link via os.Stat rather than refusing
// every non-regular dirent).
func TestResolver_FollowsSymlinkedIgnoreFile(t *testing.T) {
	root := t.TempDir()
	target := filepath.Join(root, "real-ignore-rules")
	if err := os.WriteFile(target, []byte("linked-ignored.txt\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(target, filepath.Join(root, ".gitignore")); err != nil {
		t.Skipf("symlinks unsupported: %v", err)
	}

	r, err := NewResolver(root)
	if err != nil {
		t.Fatalf("NewResolver: %v", err)
	}
	if !r.Match("linked-ignored.txt", false) {
		t.Error("rule from a symlinked .gitignore was not honoured")
	}
}
