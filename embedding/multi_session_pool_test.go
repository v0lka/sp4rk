package embedding

import (
	"context"
	"runtime"
	"sync"
	"testing"
	"time"
)

// TestBenchmarkSessionPool_OwnershipDeterminismQueryPriority exercises the
// grid pool with real ONNX assets: every submitted job produces exactly one
// result, results are attributed by sequence (deterministic scatter), query
// jobs are answered, and pool close joins every worker exactly once without
// goroutine leaks.
func TestBenchmarkSessionPool_OwnershipDeterminismQueryPriority(t *testing.T) {
	modelPath, tokenizerPath, libraryPath := benchAssets(t)
	if err := initONNXRuntime(libraryPath); err != nil {
		t.Fatalf("initONNXRuntime: %v", err)
	}
	tok, err := NewTokenizer(tokenizerPath)
	if err != nil {
		t.Fatalf("NewTokenizer: %v", err)
	}

	const (
		workers   = 2
		batchSize = 4
		seqLen    = 64
		docCount  = 16
	)
	pool, err := newBenchmarkSessionPool(modelPath, workers, 1, batchSize, seqLen, DefaultHiddenDim, 2, 8<<30, nil)
	if err != nil {
		t.Fatalf("newBenchmarkSessionPool: %v", err)
	}

	docs := benchChunks("short", docCount)
	ids, mask, types, _, err := tok.EncodeBatchWithLengths(docs, seqLen)
	if err != nil {
		t.Fatalf("EncodeBatchWithLengths: %v", err)
	}
	queryIDs, queryMask, queryTypes, err := tok.Encode("priority probe query", seqLen)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}

	runtime.GC()
	before := runtime.NumGoroutine()
	for pass := range 3 {
		if err := runGridPass(context.Background(), pool, ids, mask, types, queryIDs, queryMask, queryTypes, docCount, batchSize, seqLen, nil); err != nil {
			t.Fatalf("pass %d: %v", pass, err)
		}
	}
	pool.close()
	pool.close() // idempotent

	// Workers must have joined: goroutine count returns to the pre-pool level
	// once the scheduler settles.
	deadline := time.Now().Add(5 * time.Second)
	for runtime.NumGoroutine() > before {
		if time.Now().After(deadline) {
			t.Fatalf("goroutine leak: before=%d after=%d", before, runtime.NumGoroutine())
		}
		runtime.Gosched()
		time.Sleep(10 * time.Millisecond)
	}

	// Submitting after close must fail instead of deadlocking.
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := pool.submit(benchmarkSessionJob{ctx: ctx, rows: 1, ids: queryIDs, mask: queryMask, types: queryTypes}); err == nil {
		t.Fatal("submit after close succeeded")
	}
}

// TestBenchmarkSessionPool_CancelStopsPendingJobs proves a cancelled caller
// unblocks from a full queue: submits fill both queues, then the cancelled
// context aborts the next submit and the pool still closes cleanly.
func TestBenchmarkSessionPool_CancelStopsPendingJobs(t *testing.T) {
	pool := &benchmarkSessionPool{
		queries: make(chan benchmarkSessionJob, 1),
		docs:    make(chan benchmarkSessionJob, 1),
		stop:    make(chan struct{}),
		workers: 0,
	}
	pool.docs <- benchmarkSessionJob{}
	pool.queries <- benchmarkSessionJob{query: true}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := pool.submit(benchmarkSessionJob{ctx: ctx}); err == nil {
		t.Fatal("submit with cancelled context succeeded against full queues")
	}
	close(pool.stop)
}

// TestBenchmarkSessionPool_ConcurrentSubmitters runs several goroutines
// through the real pool concurrently to expose races between submit, worker
// delivery, and close under the race detector.
func TestBenchmarkSessionPool_ConcurrentSubmitters(t *testing.T) {
	modelPath, tokenizerPath, libraryPath := benchAssets(t)
	if err := initONNXRuntime(libraryPath); err != nil {
		t.Fatalf("initONNXRuntime: %v", err)
	}
	tok, err := NewTokenizer(tokenizerPath)
	if err != nil {
		t.Fatalf("NewTokenizer: %v", err)
	}
	const (
		batchSize = 4
		seqLen    = 64
		docCount  = 8
	)
	pool, err := newBenchmarkSessionPool(modelPath, 2, 1, batchSize, seqLen, DefaultHiddenDim, 2, 8<<30, nil)
	if err != nil {
		t.Fatalf("newBenchmarkSessionPool: %v", err)
	}
	defer pool.close()

	ids, mask, types, _, err := tok.EncodeBatchWithLengths(benchChunks("short", docCount), seqLen)
	if err != nil {
		t.Fatalf("EncodeBatchWithLengths: %v", err)
	}
	queryIDs, queryMask, queryTypes, err := tok.Encode("concurrent query", seqLen)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}

	var wg sync.WaitGroup
	errs := make(chan error, 4)
	for range 4 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			errs <- runGridPass(context.Background(), pool, ids, mask, types, queryIDs, queryMask, queryTypes, docCount, batchSize, seqLen, nil)
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Error(err)
		}
	}
}
