package embedding

import (
	"context"
	"errors"
	"reflect"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestOneAheadPipeline_DepthAndOrder(t *testing.T) {
	var started atomic.Int64
	pipeline := startOneAheadPipeline(context.Background(), 4, func(_ context.Context, i int) (int, error) {
		started.Add(1)
		return i, nil
	})

	first := <-pipeline.items
	if first.value != 0 {
		t.Fatalf("first item = %d, want 0", first.value)
	}
	waitForAtomic(t, &started, 2)
	time.Sleep(10 * time.Millisecond)
	if got := started.Load(); got != 2 {
		t.Fatalf("producer started %d items while item 1 awaited handoff, want 2", got)
	}

	got := []int{first.value}
	for item := range pipeline.items {
		got = append(got, item.value)
	}
	if err := pipeline.stop(); err != nil {
		t.Fatal(err)
	}
	if want := []int{0, 1, 2, 3}; !reflect.DeepEqual(got, want) {
		t.Fatalf("pipeline order = %v, want %v", got, want)
	}
}

func TestOneAheadPipeline_ErrorCancelAndWaitPaths(t *testing.T) {
	t.Run("producer error is joined", func(t *testing.T) {
		wantErr := errors.New("tokenization failed")
		pipeline := startOneAheadPipeline(context.Background(), 3, func(_ context.Context, i int) (int, error) {
			if i == 1 {
				return 0, wantErr
			}
			return i, nil
		})
		if item := <-pipeline.items; item.value != 0 {
			t.Fatalf("first item = %d, want 0", item.value)
		}
		for range pipeline.items {
		}
		if err := pipeline.stop(); !errors.Is(err, wantErr) {
			t.Fatalf("stop error = %v, want %v", err, wantErr)
		}
		if err := pipeline.stop(); !errors.Is(err, wantErr) {
			t.Fatalf("second stop error = %v, want %v", err, wantErr)
		}
	})

	t.Run("consumer abort cancels blocked handoff and joins", func(t *testing.T) {
		pipeline := startOneAheadPipeline(context.Background(), 3, func(_ context.Context, i int) (int, error) {
			return i, nil
		})
		waitForPipelineProduction(t, pipeline)
		if err := pipeline.stop(); !errors.Is(err, context.Canceled) {
			t.Fatalf("stop error = %v, want context.Canceled", err)
		}
	})

	t.Run("cancel releases blocked handoff and joins", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		pipeline := startOneAheadPipeline(ctx, 3, func(_ context.Context, i int) (int, error) {
			return i, nil
		})
		waitForPipelineProduction(t, pipeline)
		cancel()
		if err := pipeline.stop(); !errors.Is(err, context.Canceled) {
			t.Fatalf("stop error = %v, want context.Canceled", err)
		}
	})
}

func TestTokenizer_ConcurrentEncodeRaceAndDeterminism(t *testing.T) {
	tok, err := NewTokenizer(testTokenizerPath(t))
	if err != nil {
		t.Fatal(err)
	}
	texts := []string{"hello world", "deterministic concurrent tokenizer", "Привет, мир", "invalid \xff input"}
	type encoded struct {
		ids, mask, types []int64
		lengths          []int
	}
	want := make([]encoded, len(texts))
	for i, text := range texts {
		ids, mask, types, lengths, encodeErr := tok.EncodeBatchWithLengths([]string{text}, 128)
		if encodeErr != nil {
			t.Fatal(encodeErr)
		}
		want[i] = encoded{ids: ids, mask: mask, types: types, lengths: lengths}
	}

	const workers = 8
	var wg sync.WaitGroup
	errs := make(chan error, workers)
	for worker := 0; worker < workers; worker++ {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			for round := 0; round < 50; round++ {
				i := (worker + round) % len(texts)
				ids, mask, types, lengths, encodeErr := tok.EncodeBatchWithLengths([]string{texts[i]}, 128)
				if encodeErr != nil {
					errs <- encodeErr
					return
				}
				got := encoded{ids: ids, mask: mask, types: types, lengths: lengths}
				if !reflect.DeepEqual(got, want[i]) {
					errs <- errors.New("concurrent tokenizer output differs from serial baseline")
					return
				}
			}
		}(worker)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
}

func TestEmbedder_BatchPipelineParityAndOrder(t *testing.T) {
	e, err := NewEmbedder(EmbedderConfig{
		ModelPath: testModelPath(t), TokenizerPath: testTokenizerPath(t), LibraryPath: testLibraryPath(t),
		BatchSize: 4,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer closeSessionOnly(e)

	texts := []string{
		"zero unique document", "one unique document", "two unique document", "three unique document",
		"four unique document", "five unique document", "six unique document", "seven unique document",
		"eight unique partial tail",
	}
	serial, err := e.EmbedDocuments(context.Background(), texts)
	if err != nil {
		t.Fatal(err)
	}
	e.batchPipeline = true
	pipelined, err := e.EmbedDocuments(context.Background(), texts)
	if err != nil {
		t.Fatal(err)
	}
	assertEmbeddingParity(t, pipelined, serial, 0)
}

func waitForAtomic(t *testing.T, value *atomic.Int64, want int64) {
	t.Helper()
	deadline := time.Now().Add(time.Second)
	for value.Load() < want && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if got := value.Load(); got < want {
		t.Fatalf("counter = %d, want at least %d", got, want)
	}
}

func waitForPipelineProduction[T any](t *testing.T, pipeline *oneAheadPipeline[T]) {
	t.Helper()
	select {
	case <-pipeline.done:
		t.Fatal("producer exited before cancellation while handoff was blocked")
	case <-time.After(10 * time.Millisecond):
	}
}
