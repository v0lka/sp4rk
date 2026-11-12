package embedding

import (
	"log/slog"
	"runtime"
	"sync"
	"time"
)

// ortRunner executes ONNX Runtime work on one dedicated, locked OS thread.
//
// WHY THIS EXISTS: the CUDA execution provider keeps per-thread state — a
// cuBLAS handle plus its workspace — for every OS thread that enters
// Session.Run. Go's scheduler moves goroutines between OS threads freely, so
// a sequence of inference calls that looks perfectly serial at the Go level
// (Embedder.mu serializes them) still arrives on a steadily growing set of
// threads, and the provider allocates a fresh per-thread context for each one.
// Measured on an RTX 5060 Ti with jina-v2-small at batch 32: ~1 GiB of device
// memory per distinct calling thread, exhausting a 16 GiB card within seconds
// of indexing and failing with "CUBLAS failure 3: the resource allocation
// failed" inside cublasCreate. Funnelling every ONNX call through a single
// locked thread holds the footprint flat.
//
// The CPU provider has no such per-thread cost, so Embedder only creates a
// runner for hardware providers and calls ONNX directly otherwise. The
// OpenVINO provider gets the same pinning as CUDA: its runtime builds device
// contexts and thread pools around the threads that enter Session.Run, and
// holding every ORT touch to one thread keeps that bookkeeping flat no matter
// how Go's scheduler migrates the calling goroutines.
type ortRunner struct {
	jobs chan func()

	// now is the clock Wedged and the evidence log read. Fixed at
	// construction (time.Now); tests replace it with a virtual clock.
	now func() time.Time
	// logger carries the worker's evidence log (job start/finish traces and
	// the slow-job warning). NewEmbedder wires its resolved config logger in
	// here; nil (a bare runner built by a test without one) disables logging
	// entirely.
	logger *slog.Logger

	// mu guards the per-job heartbeat below. The worker goroutine is the
	// only writer; Wedged readers on arbitrary caller goroutines take the
	// same lock, which is uncontended outside the begin/end windows.
	mu           sync.Mutex
	jobSeq       uint64
	jobRunning   bool
	jobStartedAt time.Time
}

// slowJobEvidenceThreshold is the per-job duration at which the worker stops
// trusting "slow machine" as the explanation. A jina-v2-small batch on the
// CPU provider takes hundreds of milliseconds; hardware providers land in
// the same order of magnitude for the session shapes the embedder builds.
// A single job at or beyond this threshold is the near-wedge evidence the
// log exists to keep: the batch before a wedge usually shows up here first.
const slowJobEvidenceThreshold = 30 * time.Second

// providerUsesDedicatedThread reports whether inference under the named
// execution provider must run on the ortRunner's single locked OS thread.
// Everything except the CPU provider (including "auto", whose ladder starts
// on a hardware provider) is pinned.
func providerUsesDedicatedThread(provider string) bool {
	switch provider {
	case ExecutionProviderCPU:
		return false
	case ExecutionProviderCUDA, ExecutionProviderOpenVINO, ExecutionProviderAuto:
		return true
	default:
		return false
	}
}

// newORTRunner starts the dedicated thread and returns a runner bound to it.
// logger receives the worker's evidence log (job traces and the slow-job
// warning); nil keeps the runner silent. The caller must call stop exactly
// once, after the last do.
func newORTRunner(logger *slog.Logger) *ortRunner {
	r := &ortRunner{jobs: make(chan func()), now: time.Now, logger: logger}
	go func() {
		// Never unlocked: the thread must die with the goroutine so the
		// runtime cannot hand it to another goroutine, and so ONNX never sees
		// a second thread.
		runtime.LockOSThread()
		for job := range r.jobs {
			job()
		}
	}()
	return r
}

// beginJob stamps the heartbeat for a job about to run and returns its
// sequence number. Called on the worker goroutine only.
func (r *ortRunner) beginJob() uint64 {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.jobSeq++
	r.jobRunning = true
	r.jobStartedAt = r.clock()
	if r.logger != nil {
		r.logger.Debug("ort runner job started", "seq", r.jobSeq)
	}
	return r.jobSeq
}

// endJob clears the running mark and records the job's duration as evidence:
// a Debug trace always (the worker's own start/finish trail — for a job that
// never finishes it is the start line above that remains), plus a Warn when
// the duration reached slowJobEvidenceThreshold. The panic value, if any, is
// passed through untouched; endJob never recovers. Called on the worker
// goroutine only, from do's deferred cleanup.
func (r *ortRunner) endJob(seq uint64, panicValue any) {
	r.mu.Lock()
	finishedAt := r.clock()
	startedAt := r.jobStartedAt
	r.jobRunning = false
	logger := r.logger
	r.mu.Unlock()

	if logger == nil {
		return
	}
	duration := finishedAt.Sub(startedAt)
	logger.Debug("ort runner job finished", "seq", seq, "duration", duration.Round(time.Millisecond))
	if duration >= slowJobEvidenceThreshold {
		logger.Warn("ort runner job took abnormally long — evidence the inference thread is close to wedging",
			"seq", seq, "duration", duration.Round(time.Millisecond),
			"threshold", slowJobEvidenceThreshold)
	}
}

// clock reads the injectable clock, defaulting to time.Now for a zero-value
// runner (tests may build one directly).
func (r *ortRunner) clock() time.Time {
	if r.now == nil {
		return time.Now()
	}
	return r.now()
}

// wedged is the authoritative liveness verdict for the dedicated thread:
// true only while a job is running AND its heartbeat is at least bound old.
// Idle (between jobs, before the first job, after stop) is never wedged —
// an empty jobs channel cannot prove anything about the next inference —
// and a non-positive bound means "no opinion". The heartbeat can only age
// past any bound while the worker is stuck INSIDE fn: a C call that never
// returns holds the worker past beginJob, so no timer on any caller side
// can produce this answer — only the worker's own stamp can.
func (r *ortRunner) wedged(bound time.Duration) bool {
	if bound <= 0 {
		return false
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if !r.jobRunning {
		return false
	}
	return r.clock().Sub(r.jobStartedAt) >= bound
}

// do runs fn on the runner's thread and returns once fn has completed.
//
// A panic inside fn is recovered on the runner's thread and re-raised on the
// caller's, UNCHANGED: the caller's recover() sees the original panic value
// with its original type. Letting it escape would kill the runner goroutine
// and leave every later do blocked forever on a channel nobody reads;
// re-raising keeps the failure visible to the caller's own recover (the
// desktop app wraps embedder work in one) while the runner survives.
//
// Every job is bracketed by beginJob/endJob: the heartbeat is stamped by the
// worker itself, so Wedged stays authoritative even when the caller has
// already given up — a caller timeout (or an abandoned context) says nothing
// about the thread, which may still be inside fn.
func (r *ortRunner) do(fn func()) {
	done := make(chan any, 1)
	r.jobs <- func() {
		seq := r.beginJob()
		defer func() {
			p := recover()
			r.endJob(seq, p)
			done <- p
		}()
		fn()
	}
	if p := <-done; p != nil {
		panic(p)
	}
}

// stop shuts the runner's thread down. Calling do afterwards panics on a send
// to a closed channel, which is why Embedder.Close stops the runner last.
func (r *ortRunner) stop() {
	close(r.jobs)
}
