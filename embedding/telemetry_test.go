package embedding

import (
	"sync"
	"testing"
	"time"
)

func TestTelemetrySnapshotAggregatesContentFreeMetrics(t *testing.T) {
	t.Parallel()
	telemetry := &Telemetry{}
	telemetry.observe(StageTokenization, 2*time.Millisecond)
	telemetry.observeTokens([]int64{1, 1, 0, 0, 1, 1, 1, 0}, 2, 4)
	telemetry.observeInference(2, 4, 3*time.Millisecond)
	telemetry.observeSession(4 * time.Millisecond)

	s := telemetry.Snapshot()
	if got, want := len(s.Stages), 3; got != want {
		t.Fatalf("stage cardinality = %d, want %d", got, want)
	}
	allowed := map[TelemetryStage]bool{
		StageTokenization: true, StageONNXInference: true, StageSessionCreate: true,
	}
	for stage := range s.Stages {
		if !allowed[stage] {
			t.Fatalf("unexpected free-form stage %q", stage)
		}
	}
	if s.Texts != 2 || s.Tokens != 5 || s.MinTokens != 2 || s.MaxTokens != 3 {
		t.Fatalf("token aggregate = texts:%d tokens:%d min:%d max:%d", s.Texts, s.Tokens, s.MinTokens, s.MaxTokens)
	}
	if got := s.AverageTokens(); got != 2.5 {
		t.Fatalf("AverageTokens = %v, want 2.5", got)
	}
	if got := s.BatchFill(); got != 0.5 {
		t.Fatalf("BatchFill = %v, want 0.5", got)
	}
	if s.InferenceCount != 1 || s.SessionCount != 1 {
		t.Fatalf("counts = inference:%d session:%d, want 1/1", s.InferenceCount, s.SessionCount)
	}
}

func TestTelemetryConcurrentObservationAndReset(t *testing.T) {
	t.Parallel()
	telemetry := &Telemetry{}
	var wg sync.WaitGroup
	for range 16 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 100 {
				telemetry.observeInference(8, 16, time.Microsecond)
				_ = telemetry.Snapshot()
			}
		}()
	}
	wg.Wait()
	if got := telemetry.Snapshot().InferenceCount; got != 1600 {
		t.Fatalf("InferenceCount = %d, want 1600", got)
	}
	telemetry.Reset()
	if got := telemetry.Snapshot().InferenceCount; got != 0 {
		t.Fatalf("InferenceCount after Reset = %d, want 0", got)
	}
}

// TestTelemetryInferenceCountMatchesStageCalls pins the accounting invariant
// of counting inference ATTEMPTS (not only successes) through
// observeInference: onnxSession.run/runBatch call it on both the success and
// the failure path, so InferenceCount always equals the ONNX-inference stage
// call count and a derived average inference latency stays meaningful after
// transient failures.
func TestTelemetryInferenceCountMatchesStageCalls(t *testing.T) {
	t.Parallel()
	telemetry := &Telemetry{}
	// Two successful attempts and one failed one, in the call shape used by
	// run (1x1) and runBatch (nxcapacity).
	telemetry.observeInference(1, 1, time.Millisecond)
	telemetry.observeInference(8, 16, 2*time.Millisecond)
	telemetry.observeInference(8, 16, 3*time.Millisecond) // failed attempt, still counted
	s := telemetry.Snapshot()
	if s.InferenceCount != 3 {
		t.Fatalf("InferenceCount = %d, want 3 (attempts)", s.InferenceCount)
	}
	if stage := s.Stages[StageONNXInference]; stage.Calls != 3 {
		t.Fatalf("Stages[StageONNXInference].Calls = %d, want 3", stage.Calls)
	}
}
