package embedding

import (
	"runtime"
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
}

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
// The caller must call stop exactly once, after the last do.
func newORTRunner() *ortRunner {
	r := &ortRunner{jobs: make(chan func())}
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

// do runs fn on the runner's thread and returns once fn has completed.
//
// A panic inside fn is recovered on the runner's thread and re-raised on the
// caller's, UNCHANGED: the caller's recover() sees the original panic value
// with its original type. Letting it escape would kill the runner goroutine
// and leave every later do blocked forever on a channel nobody reads;
// re-raising keeps the failure visible to the caller's own recover (the
// desktop app wraps embedder work in one) while the runner survives.
func (r *ortRunner) do(fn func()) {
	done := make(chan any, 1)
	r.jobs <- func() {
		defer func() { done <- recover() }()
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
