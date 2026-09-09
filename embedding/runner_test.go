package embedding

import (
	"bytes"
	"runtime"
	"strconv"
	"sync"
	"testing"
)

// goroutineID returns the calling goroutine's id, parsed from its stack
// header. The runner's goroutine calls runtime.LockOSThread and never
// unlocks, so "same goroutine" and "same OS thread" are equivalent for it —
// which is the property these tests actually care about and cannot observe
// directly in portable Go.
func goroutineID(t *testing.T) uint64 {
	t.Helper()
	var buf [64]byte
	n := runtime.Stack(buf[:], false)
	line := buf[:n]
	const prefix = "goroutine "
	if !bytes.HasPrefix(line, []byte(prefix)) {
		t.Fatalf("unexpected stack header %q", line)
	}
	rest := line[len(prefix):]
	sp := bytes.IndexByte(rest, ' ')
	if sp < 0 {
		t.Fatalf("unexpected stack header %q", line)
	}
	id, err := strconv.ParseUint(string(rest[:sp]), 10, 64)
	if err != nil {
		t.Fatalf("parsing goroutine id from %q: %v", line, err)
	}
	return id
}

func TestORTRunner_SingleThreadAcrossCallers(t *testing.T) {
	// The whole point of the runner: no matter how many different goroutines
	// (and therefore OS threads) call in, ONNX work lands on exactly one.
	r := newORTRunner()
	defer r.stop()

	const callers = 20
	ids := make([]uint64, callers)
	var wg sync.WaitGroup
	for i := range callers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			runtime.LockOSThread() // caller sits on its own OS thread
			defer runtime.UnlockOSThread()
			r.do(func() { ids[i] = goroutineID(t) })
		}()
	}
	wg.Wait()

	for i, id := range ids {
		if id == 0 {
			t.Fatalf("job %d did not run", i)
		}
		if id != ids[0] {
			t.Errorf("job %d ran on goroutine %d, want %d (all jobs must share one thread)", i, id, ids[0])
		}
	}
	if ids[0] == goroutineID(t) {
		t.Error("jobs ran on the calling goroutine, want the runner's dedicated one")
	}
}

func TestORTRunner_Serializes(t *testing.T) {
	// Jobs must never overlap: the ONNX sessions they drive are not
	// concurrency-safe.
	r := newORTRunner()
	defer r.stop()

	var (
		mu      sync.Mutex
		active  int
		maxSeen int
	)
	var wg sync.WaitGroup
	for range 50 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			r.do(func() {
				mu.Lock()
				active++
				if active > maxSeen {
					maxSeen = active
				}
				mu.Unlock()
				runtime.Gosched()
				mu.Lock()
				active--
				mu.Unlock()
			})
		}()
	}
	wg.Wait()
	if maxSeen != 1 {
		t.Errorf("max concurrent jobs = %d, want 1", maxSeen)
	}
}

func TestORTRunner_PanicReachesCallerAndRunnerSurvives(t *testing.T) {
	// A panic inside a job must not kill the runner goroutine: that would
	// leave every later do() blocked forever on a channel nobody reads.
	r := newORTRunner()
	defer r.stop()

	func() {
		defer func() {
			if recover() == nil {
				t.Error("do() did not propagate the job's panic to the caller")
			}
		}()
		r.do(func() { panic("boom") })
	}()

	// The runner must still serve work — and from the same thread as before.
	done := make(chan uint64, 1)
	r.do(func() { done <- goroutineID(t) })
	if id := <-done; id == 0 {
		t.Error("runner stopped serving jobs after a panic")
	}
}

func TestRunOnORTThread_NilRunsInline(t *testing.T) {
	// The CPU provider passes a nil runner and must keep calling ONNX
	// directly, with no channel hop and no extra thread.
	want := goroutineID(t)
	var got uint64
	runOnORTThread(nil, func() { got = goroutineID(t) })
	if got != want {
		t.Errorf("fn ran on goroutine %d, want the caller's %d", got, want)
	}
}
