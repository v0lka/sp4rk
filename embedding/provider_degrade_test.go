// SPDX-License-Identifier: Apache-2.0

package embedding

import (
	"bytes"
	"errors"
	"log/slog"
	"strings"
	"testing"
)

// errHardwareTestFailure is the sentinel the degradation tests' run closures
// fail with. errors.Is matches it both on the raw inference error (ineligible
// or below-threshold failures are returned unchanged) and through the loud
// chained errors the degradation path wraps it in with %w.
var errHardwareTestFailure = errors.New("CUBLAS failure 3: the resource allocation failed")

// newDegradeTestEmbedder returns an embedder configured the way the runtime
// degradation tests need it: an "auto" request parked on a hardware provider
// (fallbackPending), a logger that captures WARN records, and — when cb is
// non-nil — the OnProviderDegrade callback wired through the same field
// NewEmbedder fills. All ONNX-touching state is left at its zero value so the
// degradation path runs without an ONNX Runtime environment: with
// IntraOpThreads 0 the CPU options build returns nil without any ORT call,
// and onnxSession.destroy on an empty session is a no-op.
func newDegradeTestEmbedder(cb func(from, to, cause string)) (*Embedder, *bytes.Buffer) {
	var buf bytes.Buffer
	e := &Embedder{
		logger:            slog.New(slog.NewTextHandler(&buf, nil)),
		bucketSessions:    make(map[sessionKey]*onnxSession),
		executionProvider: ExecutionProviderCUDA,
		fallbackPending:   true,
		degradeCallback:   cb,
	}
	return e, &buf
}

// TestEmbedder_embedInference_DegradesOnThresholdRetriesOnce pins the core
// runtime-degradation scenario: an "auto" request on a hardware provider
// fails one inference (loud error, embedder unchanged), fails a second
// consecutive one (threshold reached) — the embedder degrades to the CPU
// provider (live sessions destroyed, WARN, callback), recreates the session
// it needs, and retries the inference exactly once there. The successful CPU
// retry completes the call and resets the streak.
func TestEmbedder_embedInference_DegradesOnThresholdRetriesOnce(t *testing.T) {
	var (
		callbackFrom  string
		callbackTo    string
		callbackCause string
		callbackCalls int
	)
	e, buf := newDegradeTestEmbedder(func(from, to, cause string) {
		callbackCalls++
		callbackFrom, callbackTo, callbackCause = from, to, cause
	})
	e.sess = &onnxSession{}
	e.batchSess = &onnxSession{}
	e.bucketSessions[sessionKey{batchSize: 32, seqLen: 64}] = &onnxSession{}

	var (
		runCalls   int
		createCall int
	)
	// run fails on the hardware provider and succeeds on the CPU one, so the
	// post-degradation retry is the only passing attempt. It reads the field
	// directly: the test holds e.mu for writing, so the locking getter would
	// deadlock.
	run := func() error {
		runCalls++
		if e.executionProvider == ExecutionProviderCPU {
			return nil
		}
		return errHardwareTestFailure
	}
	recreate := func() error {
		createCall++
		if e.executionProvider != ExecutionProviderCPU {
			t.Error("recreate ran before the degradation committed the CPU provider")
		}
		return nil
	}

	e.mu.Lock()
	// First failure: below the threshold, returned unchanged, no degradation.
	err1 := e.embedInference(run, recreate)
	if err1 == nil {
		e.mu.Unlock()
		t.Fatal("first failure returned nil error")
	}
	// Second consecutive failure: threshold reached, degrade + recreate + retry.
	err2 := e.embedInference(run, recreate)
	streak := e.inferenceErrStreak
	provider := e.executionProvider
	pending := e.fallbackPending
	sessionsGone := e.sess == nil && e.batchSess == nil && len(e.bucketSessions) == 0
	e.mu.Unlock()

	if !errors.Is(err1, errHardwareTestFailure) {
		t.Errorf("first failure err = %v, want the raw inference error", err1)
	}
	if err2 != nil {
		t.Errorf("second failure after degradation+retry err = %v, want nil", err2)
	}
	if runCalls != 3 {
		t.Errorf("run calls = %d, want 3 (fail, fail, CPU retry)", runCalls)
	}
	if createCall != 1 {
		t.Errorf("recreate calls = %d, want 1", createCall)
	}
	if streak != 0 {
		t.Errorf("streak after the successful CPU retry = %d, want 0", streak)
	}
	if provider != ExecutionProviderCPU {
		t.Errorf("provider after degradation = %q, want %q", provider, ExecutionProviderCPU)
	}
	if pending {
		t.Error("fallbackPending still true after the degradation; a second degrade must be impossible")
	}
	if !sessionsGone {
		t.Error("live sessions survived the degradation")
	}
	if callbackCalls != 1 {
		t.Fatalf("OnProviderDegrade calls = %d, want 1", callbackCalls)
	}
	if callbackFrom != ExecutionProviderCUDA || callbackTo != ExecutionProviderCPU {
		t.Errorf("callback args = (%q, %q), want (%q, %q)",
			callbackFrom, callbackTo, ExecutionProviderCUDA, ExecutionProviderCPU)
	}
	if !strings.Contains(callbackCause, "CUBLAS failure 3") {
		t.Errorf("callback cause = %q, want it to name the original inference error", callbackCause)
	}
	out := buf.String()
	if !strings.Contains(out, "execution provider unavailable, falling back to CPU") ||
		!strings.Contains(out, "CUBLAS failure 3") {
		t.Errorf("WARN log missing message or cause; got:\n%s", out)
	}
}

// TestEmbedder_embedInference_StreakResetsOnSuccess pins the reset rule: a
// successful inference clears the consecutive-failure streak, so an isolated
// failure never accumulates into a degradation — fail, succeed, fail leaves
// the embedder on its hardware provider.
func TestEmbedder_embedInference_StreakResetsOnSuccess(t *testing.T) {
	e, _ := newDegradeTestEmbedder(func(from, to, cause string) {
		t.Errorf("OnProviderDegrade fired without a degradation (from=%q to=%q cause=%q)", from, to, cause)
	})
	var (
		runCalls   int
		createCall int
	)
	run := func() error {
		runCalls++
		if runCalls == 2 {
			return nil // the second attempt succeeds
		}
		return errHardwareTestFailure
	}
	recreate := func() error {
		createCall++
		return nil
	}

	e.mu.Lock()
	err1 := e.embedInference(run, recreate) // fail: streak 1
	err2 := e.embedInference(run, recreate) // success: streak 0
	err3 := e.embedInference(run, recreate) // fail: streak 1, below the threshold
	streak := e.inferenceErrStreak
	provider := e.executionProvider
	e.mu.Unlock()

	if !errors.Is(err1, errHardwareTestFailure) || err2 != nil || !errors.Is(err3, errHardwareTestFailure) {
		t.Fatalf("errors = (%v, %v, %v), want (sentinel, nil, sentinel)", err1, err2, err3)
	}
	if streak != 1 {
		t.Errorf("streak = %d, want 1 (reset by the success)", streak)
	}
	if provider != ExecutionProviderCUDA {
		t.Errorf("provider = %q, want unchanged %q", provider, ExecutionProviderCUDA)
	}
	if createCall != 0 {
		t.Errorf("recreate calls = %d, want 0 (no degradation happened)", createCall)
	}
	if runCalls != 3 {
		t.Errorf("run calls = %d, want 3 (no retry without degradation)", runCalls)
	}
}

// TestEmbedder_embedInference_ExplicitProviderNeverDegrades pins the loud
// contract for explicit cuda/openvino requests: fallbackPending is false for
// them, so even threshold-reached consecutive inference failures never
// degrade — the error surfaces to the caller unchanged every time.
func TestEmbedder_embedInference_ExplicitProviderNeverDegrades(t *testing.T) {
	e, _ := newDegradeTestEmbedder(func(from, to, cause string) {
		t.Errorf("OnProviderDegrade fired for an explicit provider (from=%q to=%q cause=%q)", from, to, cause)
	})
	e.fallbackPending = false // an explicit "cuda" request

	var createCall int
	run := func() error { return errHardwareTestFailure }
	recreate := func() error {
		createCall++
		return nil
	}

	e.mu.Lock()
	err1 := e.embedInference(run, recreate)
	err2 := e.embedInference(run, recreate)
	err3 := e.embedInference(run, recreate)
	streak := e.inferenceErrStreak
	provider := e.executionProvider
	e.mu.Unlock()

	for i, err := range []error{err1, err2, err3} {
		if !errors.Is(err, errHardwareTestFailure) {
			t.Errorf("failure %d err = %v, want the raw inference error", i+1, err)
		}
	}
	if streak != 3 {
		t.Errorf("streak = %d, want 3 (counting continues even without degradation)", streak)
	}
	if provider != ExecutionProviderCUDA {
		t.Errorf("provider = %q, want unchanged %q", provider, ExecutionProviderCUDA)
	}
	if createCall != 0 {
		t.Errorf("recreate calls = %d, want 0", createCall)
	}
}

// TestEmbedder_embedInference_NoDegradeAfterClose pins the closed-embedder
// guard: a closed embedder never degrades, whatever the streak and the
// pending flag say.
func TestEmbedder_embedInference_NoDegradeAfterClose(t *testing.T) {
	e, _ := newDegradeTestEmbedder(func(from, to, cause string) {
		t.Errorf("OnProviderDegrade fired on a closed embedder (from=%q to=%q cause=%q)", from, to, cause)
	})
	e.closed = true
	e.sess = &onnxSession{}

	var createCall int
	run := func() error { return errHardwareTestFailure }
	recreate := func() error {
		createCall++
		return nil
	}

	e.mu.Lock()
	err1 := e.embedInference(run, recreate)
	err2 := e.embedInference(run, recreate)
	provider := e.executionProvider
	sessIntact := e.sess != nil
	e.mu.Unlock()

	if !errors.Is(err1, errHardwareTestFailure) || !errors.Is(err2, errHardwareTestFailure) {
		t.Fatalf("errors = (%v, %v), want the raw inference errors", err1, err2)
	}
	if provider != ExecutionProviderCUDA {
		t.Errorf("provider = %q, want unchanged %q", provider, ExecutionProviderCUDA)
	}
	if !sessIntact {
		t.Error("the live session was destroyed on a closed embedder without degradation")
	}
	if createCall != 0 {
		t.Errorf("recreate calls = %d, want 0", createCall)
	}
}

// TestEmbedder_embedInference_LoudErrorWhenCPURetryFails pins the one-loud-
// error contract: when the degradation succeeds but the CPU retry fails too,
// the caller gets a single error chaining both the original hardware failure
// and the CPU failure — never a silent loss of either.
func TestEmbedder_embedInference_LoudErrorWhenCPURetryFails(t *testing.T) {
	e, _ := newDegradeTestEmbedder(nil)
	e.batchSess = &onnxSession{}

	var (
		runCalls   int
		cpuFailure = errors.New("cpu retry also failed")
	)
	run := func() error {
		runCalls++
		if runCalls == 1 {
			return errHardwareTestFailure
		}
		if runCalls == 2 {
			return errHardwareTestFailure // threshold reached on this attempt
		}
		return cpuFailure // the post-degradation retry
	}
	recreate := func() error { return nil }

	e.mu.Lock()
	_ = e.embedInference(run, recreate)    // first failure: streak 1, below the threshold
	err := e.embedInference(run, recreate) // second failure: degrade, recreate, retry fails
	provider := e.executionProvider
	e.mu.Unlock()

	if !errors.Is(err, errHardwareTestFailure) {
		t.Errorf("err = %v, want it to chain the original hardware failure", err)
	}
	if !errors.Is(err, cpuFailure) {
		t.Errorf("err = %v, want it to chain the CPU retry failure", err)
	}
	for _, want := range []string{ExecutionProviderCUDA, "runtime degradation", cpuFailure.Error()} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("err text %q does not mention %q", err.Error(), want)
		}
	}
	if provider != ExecutionProviderCPU {
		t.Errorf("provider = %q, want %q (the degradation committed before the failed retry)", provider, ExecutionProviderCPU)
	}
	if runCalls != 3 {
		t.Errorf("run calls = %d, want 3 (exactly one retry)", runCalls)
	}
}

// TestEmbedder_embedInference_LoudErrorWhenRecreateFails pins the loud error
// for a failing post-degradation session recreation: the caller gets one
// error chaining the original hardware failure and the recreation failure.
func TestEmbedder_embedInference_LoudErrorWhenRecreateFails(t *testing.T) {
	e, _ := newDegradeTestEmbedder(nil)
	recreateFailure := errors.New("out of memory loading the model")
	var (
		runCalls   int
		createCall int
	)
	run := func() error {
		runCalls++
		return errHardwareTestFailure
	}
	recreate := func() error {
		createCall++
		return recreateFailure
	}

	e.mu.Lock()
	_ = e.embedInference(run, recreate)    // first failure: streak 1, below the threshold
	err := e.embedInference(run, recreate) // second failure: degrade, recreate fails
	provider := e.executionProvider
	e.mu.Unlock()

	if !errors.Is(err, errHardwareTestFailure) || !errors.Is(err, recreateFailure) {
		t.Errorf("err = %v, want it to chain both the hardware failure and the recreation failure", err)
	}
	if provider != ExecutionProviderCPU {
		t.Errorf("provider = %q, want %q (the degradation committed before recreate ran)", provider, ExecutionProviderCPU)
	}
	if runCalls != 2 {
		t.Errorf("run calls = %d, want 2 (no inference retry after a failed recreation)", runCalls)
	}
	if createCall != 1 {
		t.Errorf("recreate calls = %d, want 1", createCall)
	}
}

// TestEmbedder_degradeToCPU_DestroysLiveSessionsOnORTThread exercises the
// degradation with a real ortRunner: the live sessions are destroyed as one
// job on the dedicated thread (the thread inference used) before the runner
// stops, the provider flips to CPU, and the callback fires once with the
// degraded-from provider and the cause.
func TestEmbedder_degradeToCPU_DestroysLiveSessionsOnORTThread(t *testing.T) {
	var (
		callbackFrom string
		callbackTo   string
		callbackCall int
	)
	e, _ := newDegradeTestEmbedder(func(from, to, cause string) {
		callbackCall++
		callbackFrom, callbackTo = from, to
	})
	e.runner = newORTRunner()
	e.sess = &onnxSession{}
	e.batchSess = &onnxSession{}
	e.bucketSessions[sessionKey{batchSize: 32, seqLen: 128}] = &onnxSession{}

	e.mu.Lock()
	dErr := e.degradeToCPU(errHardwareTestFailure)
	provider := e.executionProvider
	pending := e.fallbackPending
	sessionsGone := e.sess == nil && e.batchSess == nil && len(e.bucketSessions) == 0
	e.mu.Unlock()

	if dErr != nil {
		t.Fatalf("degradeToCPU err = %v, want nil", dErr)
	}
	if provider != ExecutionProviderCPU {
		t.Errorf("provider = %q, want %q", provider, ExecutionProviderCPU)
	}
	if pending {
		t.Error("fallbackPending = true, want false")
	}
	if !sessionsGone {
		t.Error("live sessions survived the degradation")
	}
	if callbackCall != 1 || callbackFrom != ExecutionProviderCUDA || callbackTo != ExecutionProviderCPU {
		t.Errorf("callback = %d call(s) with (%q, %q), want one with (%q, %q)",
			callbackCall, callbackFrom, callbackTo, ExecutionProviderCUDA, ExecutionProviderCPU)
	}
}

// TestNewEmbedder_OnProviderDegradeWiring verifies that EmbedderConfig
// carries the callback onto the constructed embedder (the field
// NewEmbedder fills), next to the other degrade state.
func TestNewEmbedder_OnProviderDegradeWiring(t *testing.T) {
	var (
		libPath  = testLibraryPath(t)
		modelPth = testModelPath(t)
		tokPath  = testTokenizerPath(t)
	)
	e, err := NewEmbedder(EmbedderConfig{
		ModelPath:         modelPth,
		TokenizerPath:     tokPath,
		LibraryPath:       libPath,
		ExecutionProvider: ExecutionProviderCPU,
		OnProviderDegrade: func(from, to, cause string) {},
	})
	if err != nil {
		t.Fatalf("NewEmbedder err = %v", err)
	}
	defer closeSessionOnly(e)
	if e.degradeCallback == nil {
		t.Error("degradeCallback not wired from EmbedderConfig.OnProviderDegrade")
	}
	if e.fallbackPending {
		t.Error("fallbackPending = true on a plain cpu embedder, want false")
	}
}
