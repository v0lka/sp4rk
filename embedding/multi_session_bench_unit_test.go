package embedding

import (
	"context"
	"os"
	"testing"
	"time"
)

func TestBenchmarkOwnedTensorBytes(t *testing.T) {
	// batch=2, seq=4: three int64 inputs (192 bytes) plus output [2,4,3]
	// float32 (96 bytes).
	if got, want := benchmarkOwnedTensorBytes(2, 4, 3), uint64(288); got != want {
		t.Fatalf("benchmarkOwnedTensorBytes() = %d, want %d", got, want)
	}
}

func TestBenchmarkP95(t *testing.T) {
	values := make([]time.Duration, 100)
	for i := range values {
		values[i] = time.Duration(100-i) * time.Millisecond
	}
	if got := benchmarkP95(values); got != 95*time.Millisecond {
		t.Fatalf("benchmarkP95() = %s, want 95ms", got)
	}
}

func TestBenchmarkSessionPoolSubmitCancellationWhenQueueFull(t *testing.T) {
	pool := &benchmarkSessionPool{
		queries: make(chan benchmarkSessionJob, 1),
		docs:    make(chan benchmarkSessionJob, 1),
		stop:    make(chan struct{}),
	}
	pool.docs <- benchmarkSessionJob{}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := pool.submit(benchmarkSessionJob{ctx: ctx}); err == nil {
		t.Fatal("submit to full queue with cancelled context succeeded")
	}
}

func TestBenchmarkSessionPoolCloseIdempotentWithoutWorkers(t *testing.T) {
	pool := &benchmarkSessionPool{
		queries: make(chan benchmarkSessionJob, 1),
		docs:    make(chan benchmarkSessionJob, 1),
		stop:    make(chan struct{}),
	}
	pool.close()
	pool.close()
}

func TestNewBenchmarkSessionPoolMemoryCapRejectsBeforeSessionCreation(t *testing.T) {
	model := t.TempDir() + "/model.onnx"
	if err := os.WriteFile(model, []byte("stub"), 0o600); err != nil {
		t.Fatal(err)
	}
	before := onnxSessionsCreated.Load()
	if _, err := newBenchmarkSessionPool(model, 1, 1, 1, 8, 2, 1, 1, nil); err == nil {
		t.Fatal("memory admission accepted a 1-byte cap")
	}
	if onnxSessionsCreated.Load() != before {
		t.Error("rejected pool created ONNX sessions")
	}
	if _, err := newBenchmarkSessionPool(model, 0, 1, 1, 8, 2, 1, 1<<30, nil); err == nil {
		t.Fatal("zero workers accepted")
	}
}
