package embedding

import (
	"bytes"
	"log/slog"
	"strings"
	"sync"
	"testing"
	"time"
)

// fakeClock is a mutex-guarded virtual clock: the test advances it between
// deterministic barriers, and the runner reads it from whichever goroutine
// runs beginJob/endJob/wedged. No real time passes in these tests — every
// duration is fabricated by Advance, so the wedged bound is exercised at
// exact nanosecond boundaries without a single sleep.
type fakeClock struct {
	mu  sync.Mutex
	now time.Time
}

func newFakeClock() *fakeClock {
	return &fakeClock{now: time.Unix(0, 0)}
}

func (c *fakeClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *fakeClock) Advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = c.now.Add(d)
}

// newTestRunner builds a runner on the fake clock with an optional evidence
// logger. Fields are set before the first job is submitted; the channel send
// of that job establishes the happens-before the worker needs.
func newTestRunner(clock *fakeClock, logger *slog.Logger) *ortRunner {
	r := newORTRunner(logger)
	r.now = clock.Now
	return r
}

// holdJob launches a job on r that signals entered (after beginJob has
// stamped the heartbeat — beginJob runs before fn) and then blocks until
// hold is closed. It returns the completion signal. Cleanup discipline: the
// caller must close hold before the test ends (the runner's stop waits for
// the worker to reach the next job boundary, so a still-held job must be
// released first).
func holdJob(r *ortRunner, hold chan struct{}) (entered, done chan struct{}) {
	entered = make(chan struct{})
	done = make(chan struct{})
	go func() {
		defer close(done)
		r.do(func() {
			close(entered)
			<-hold
		})
	}()
	return entered, done
}

// TestORTRunner_Wedged_IdleRunnerNeverWedged pins the idle half of the
// verdict: no job has ever run, so there is no heartbeat to age — the
// runner must report healthy no matter how far the clock has drifted. A
// non-positive bound means "no opinion" and is never wedged either.
func TestORTRunner_Wedged_IdleRunnerNeverWedged(t *testing.T) {
	clock := newFakeClock()
	r := newTestRunner(clock, nil)
	defer r.stop()

	clock.Advance(24 * time.Hour)
	if r.wedged(time.Second) {
		t.Error("an idle runner reported wedged: no job has ever run")
	}
	if r.wedged(0) || r.wedged(-time.Second) {
		t.Error("a non-positive bound must mean no opinion, never wedged")
	}
}

// TestORTRunner_Wedged_TracksRunningJob pins the running half: the verdict
// fires only when the worker's own heartbeat for the CURRENT job reaches
// the bound, and it stays fired while the job is still stuck. The instant
// the job finishes the runner is idle again — even hours later on the
// virtual clock — because between-jobs proves nothing about inference.
func TestORTRunner_Wedged_TracksRunningJob(t *testing.T) {
	clock := newFakeClock()
	r := newTestRunner(clock, nil)
	defer r.stop()

	const bound = 5 * time.Second
	hold := make(chan struct{})
	entered, done := holdJob(r, hold)
	<-entered

	if r.wedged(bound) {
		t.Error("a job that just started reported wedged")
	}
	clock.Advance(bound - 1)
	if r.wedged(bound) {
		t.Error("wedged fired 1ns before the bound")
	}
	clock.Advance(1) // heartbeat is now exactly bound old
	if !r.wedged(bound) {
		t.Error("wedged must fire when the running job's heartbeat reaches the bound")
	}
	clock.Advance(time.Hour)
	if !r.wedged(bound) {
		t.Error("wedged stopped firing while the job is still stuck")
	}

	close(hold)
	<-done
	if r.wedged(bound) {
		t.Error("an idle runner (between jobs) reported wedged despite the clock sitting hours past the last job")
	}
}

// TestORTRunner_Wedged_FreshHeartbeatPerJob pins that every job re-stamps
// the heartbeat: a completed long job must not poison the verdict for the
// next one, whose own stamp starts at its beginJob.
func TestORTRunner_Wedged_FreshHeartbeatPerJob(t *testing.T) {
	clock := newFakeClock()
	r := newTestRunner(clock, nil)
	defer r.stop()

	const bound = time.Second
	// Job 1 runs long (on the virtual clock) and finishes.
	hold1 := make(chan struct{})
	entered1, done1 := holdJob(r, hold1)
	<-entered1
	clock.Advance(10 * bound)
	close(hold1)
	<-done1

	// Job 2 starts while the clock still sits far past job 1's stamp: its
	// fresh heartbeat must keep the verdict healthy until IT ages.
	hold2 := make(chan struct{})
	entered2, done2 := holdJob(r, hold2)
	<-entered2
	if r.wedged(bound) {
		t.Error("job 2 must start with a fresh heartbeat despite the clock sitting far past job 1's stamp")
	}
	clock.Advance(bound)
	if !r.wedged(bound) {
		t.Error("job 2's own heartbeat did not age into the verdict")
	}
	close(hold2)
	<-done2
	if r.wedged(bound) {
		t.Error("runner must be healthy again once job 2 finished")
	}
}

// TestEmbedder_Wedged_NoRunnerReportsFalse pins the CPU-provider shape: no
// dedicated thread exists to wedge, so the exported query is always false.
func TestEmbedder_Wedged_NoRunnerReportsFalse(t *testing.T) {
	e := &Embedder{}
	if e.Wedged(time.Hour) {
		t.Error("a CPU-provider (nil runner) embedder must never report wedged")
	}
}

// TestEmbedder_Wedged_DelegatesToRunnerHeartbeat pins the exported surface
// end to end: Embedder.Wedged reflects the runner's per-job heartbeat —
// fired while the job is stuck past the bound, false again once it finished.
func TestEmbedder_Wedged_DelegatesToRunnerHeartbeat(t *testing.T) {
	clock := newFakeClock()
	e := &Embedder{}
	e.runner.Store(newTestRunner(clock, nil))
	defer e.runner.Load().stop()

	const bound = time.Second
	hold := make(chan struct{})
	entered, done := holdJob(e.runner.Load(), hold)
	<-entered
	clock.Advance(2 * bound)
	if !e.Wedged(bound) {
		t.Error("Embedder.Wedged must reflect the runner's aged heartbeat")
	}
	close(hold)
	<-done
	if e.Wedged(bound) {
		t.Error("Embedder.Wedged must be false once the job finished")
	}
}

// TestEmbedder_Wedged_AnswersWhileTheWriteLockIsHeld pins the deadlock the
// lock-free runner read exists for: EmbedDocuments holds e.mu WRITE-locked
// across all ONNX inference, so a mutex-synchronized runner read inside
// Wedged would park until the inference returned — forever under the very
// wedge Wedged exists to diagnose (the restart ladder would never fire,
// Shutdown's drain would never end, and app quit could freeze). The test
// reproduces the wedged-EmbedDocuments shape exactly — the write lock held
// across a stuck inference job — and requires the verdict to arrive anyway.
// The real-time select is a bounded hang guard, not a scheduler sleep: the
// verdict channel is buffered, so a healthy Wedged answers immediately.
func TestEmbedder_Wedged_AnswersWhileTheWriteLockIsHeld(t *testing.T) {
	clock := newFakeClock()
	e := &Embedder{}
	e.runner.Store(newTestRunner(clock, nil))
	defer e.runner.Load().stop()

	const bound = time.Second
	hold := make(chan struct{})
	entered, done := holdJob(e.runner.Load(), hold)
	<-entered
	clock.Advance(2 * bound)

	// The wedged-EmbedDocuments shape: the caller's write lock stays held
	// across the whole (stuck) inference, exactly as a real call would
	// hold it until the C call returned.
	e.mu.Lock()
	verdict := make(chan bool, 1)
	go func() { verdict <- e.Wedged(bound) }()
	select {
	case got := <-verdict:
		if !got {
			t.Error("the stuck job's aged heartbeat must report wedged")
		}
	case <-time.After(10 * time.Second):
		t.Fatal("Wedged deadlocked behind e.mu while a stuck inference holds the write lock — the authoritative wedge verdict is unobservable")
	}
	e.mu.Unlock()

	close(hold)
	<-done
}

// TestORTRunner_EvidenceLog pins the worker's evidence trail at the source:
// every job gets a Debug start/finish pair carrying its sequence and
// duration; exactly the jobs whose virtual duration reached
// slowJobEvidenceThreshold add ONE Warn (severity, message, payload and
// count asserted); a panicking job still records its finish line because
// endJob runs in the panic path before the value is re-raised on the caller.
func TestORTRunner_EvidenceLog(t *testing.T) {
	clock := newFakeClock()
	var buf bytes.Buffer
	r := newTestRunner(clock, slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug})))
	defer r.stop()

	// Fast job: Debug pair only, no Warn.
	r.do(func() {})

	// Slow job: held while the virtual clock passes the evidence threshold.
	hold := make(chan struct{})
	entered, done := holdJob(r, hold)
	<-entered
	clock.Advance(slowJobEvidenceThreshold + time.Minute)
	close(hold)
	<-done

	// Panicking job: the evidence trail must survive the panic.
	finishes := func() {
		defer func() { _ = recover() }() // the re-raised value; asserted by TestORTRunner_PanicReachesCallerAndRunnerSurvives
		r.do(func() { panic("evidence-boom") })
	}
	finishes()

	logs := buf.String()
	started := strings.Count(logs, "ort runner job started")
	finished := strings.Count(logs, "ort runner job finished")
	warned := strings.Count(logs, "level=WARN")
	if started != 3 || finished != 3 {
		t.Errorf("evidence trail incomplete: started=%d finished=%d, want 3/3 (panicking jobs included)", started, finished)
	}
	if warned != 1 {
		t.Errorf("slow-job evidence: %d WARN lines, want exactly 1 (only the job past the threshold)", warned)
	}
	if !strings.Contains(logs, "ort runner job took abnormally long") {
		t.Error("the slow job's WARN is missing the diagnosis message")
	}
	if !strings.Contains(logs, "duration=1m30s") {
		t.Errorf("the WARN's duration payload is missing or wrong (want the virtual 1m30s), got:\n%s", logs)
	}
	if !strings.Contains(logs, "duration=0s") {
		t.Errorf("the fast job's finish trace lacks its (zero) duration payload, got:\n%s", logs)
	}
}
