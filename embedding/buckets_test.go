package embedding

import (
	"context"
	"fmt"
	"log/slog"
	"math"
	"strings"
	"sync"
	"testing"
)

func TestTokenizer_EncodeBatchWithLengths(t *testing.T) {
	tok, err := NewTokenizer(testTokenizerPath(t))
	if err != nil {
		t.Fatal(err)
	}

	texts := []string{"hello", strings.Repeat("deterministic token ", 90)}
	ids, mask, types, lengths, err := tok.EncodeBatchWithLengths(texts, 128)
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) != len(texts)*128 || len(mask) != len(ids) || len(types) != len(ids) {
		t.Fatalf("unexpected tensor lengths: ids=%d mask=%d types=%d", len(ids), len(mask), len(types))
	}
	if len(lengths) != len(texts) {
		t.Fatalf("length count = %d, want %d", len(lengths), len(texts))
	}
	for row, length := range lengths {
		var maskSum int
		for _, value := range mask[row*128 : (row+1)*128] {
			maskSum += int(value)
		}
		if length != maskSum {
			t.Errorf("length[%d] = %d, attention-mask sum = %d", row, length, maskSum)
		}
	}
	if lengths[1] != 128 {
		t.Errorf("truncated length = %d, want 128", lengths[1])
	}
}

func TestEmbedder_SequenceBucket(t *testing.T) {
	e := &Embedder{maxSeqLen: 512}
	for _, tc := range []struct{ length, want int }{
		{2, 64}, {64, 64}, {65, 128}, {128, 128}, {129, 256}, {256, 256}, {257, 512}, {512, 512},
	} {
		if got := e.sequenceBucket(tc.length); got != tc.want {
			t.Errorf("sequenceBucket(%d) = %d, want %d", tc.length, got, tc.want)
		}
	}

	e.maxSeqLen = 100
	if got := e.sequenceBucket(65); got != 100 {
		t.Errorf("sequenceBucket with max=100 = %d, want 100", got)
	}
}

func TestONNXSession_DestroyIdempotentAndUseAfterClose(t *testing.T) {
	s := &onnxSession{batchSize: 1, seqLen: 8, hiddenDim: 2}
	s.destroy()
	s.destroy()
	if !s.closed.Load() {
		t.Fatal("session not marked closed")
	}
	inputs := make([]int64, 8)
	if _, err := s.run(inputs, inputs, inputs); err == nil {
		t.Error("run after destroy succeeded")
	}
	if _, err := s.runBatch(1, inputs, inputs, inputs); err == nil {
		t.Error("runBatch after destroy succeeded")
	}
}

func TestEmbedder_CloseIsIdempotent(t *testing.T) {
	skipIfEnvManaged(t)
	e := &Embedder{
		tokenizer:      &Tokenizer{},
		logger:         slog.New(slog.DiscardHandler),
		bucketSessions: make(map[sessionKey]*onnxSession),
	}
	_ = e.Close() // missing runtime may report an error, but resources are now closed
	if !e.closed {
		t.Fatal("first Close did not mark embedder closed")
	}
	if err := e.Close(); err != nil {
		t.Fatalf("second Close() error = %v, want nil", err)
	}
}

func TestEmbedder_LengthBuckets_CancelledDoesNotCreateSession(t *testing.T) {
	e := &Embedder{
		tokenizer:      &Tokenizer{},
		maxSeqLen:      512,
		hiddenDim:      512,
		batchSize:      4,
		lengthBuckets:  true,
		bucketSessions: make(map[sessionKey]*onnxSession),
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	before := onnxSessionsCreated.Load()
	if _, err := e.EmbedDocuments(ctx, []string{"cancelled"}); err == nil {
		t.Fatal("cancelled bucket embedding succeeded")
	}
	if got := onnxSessionsCreated.Load() - before; got != 0 {
		t.Errorf("cancelled call created %d sessions, want 0", got)
	}
}

func TestEmbedder_LengthBuckets_ParityOrderLazyReuse(t *testing.T) {
	modelPath := testModelPath(t)
	tokenizerPath := testTokenizerPath(t)
	libraryPath := testLibraryPath(t)
	texts := bucketTestTexts()

	fixed, err := NewEmbedder(EmbedderConfig{
		ModelPath: modelPath, TokenizerPath: tokenizerPath, LibraryPath: libraryPath,
		BatchSize: 4,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer closeSessionOnly(fixed)
	baseline, err := fixed.EmbedDocuments(context.Background(), texts)
	if err != nil {
		t.Fatal(err)
	}

	bucketed, err := NewEmbedder(EmbedderConfig{
		ModelPath: modelPath, TokenizerPath: tokenizerPath, LibraryPath: libraryPath,
		BatchSize: 4, EnableLengthBuckets: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer closeSessionOnly(bucketed)
	if len(bucketed.bucketSessions) != 0 || bucketed.sess != nil {
		t.Fatal("bucket sessions must be fully lazy")
	}

	before := onnxSessionsCreated.Load()
	got, err := bucketed.EmbedDocuments(context.Background(), texts)
	if err != nil {
		t.Fatal(err)
	}
	created := onnxSessionsCreated.Load() - before
	if created != 4 {
		t.Fatalf("created sessions = %d, want one for each of four used buckets", created)
	}
	if len(bucketed.bucketSessions) != 4 {
		t.Fatalf("cached sessions = %d, want 4", len(bucketed.bucketSessions))
	}
	assertEmbeddingParity(t, got, baseline, 2e-4)

	before = onnxSessionsCreated.Load()
	gotAgain, err := bucketed.EmbedDocuments(context.Background(), texts)
	if err != nil {
		t.Fatal(err)
	}
	if created := onnxSessionsCreated.Load() - before; created != 0 {
		t.Errorf("steady-state call created %d sessions, want 0", created)
	}
	assertEmbeddingParity(t, gotAgain, baseline, 2e-4)
}

func TestEmbedder_LengthBuckets_ConcurrentUse(t *testing.T) {
	e, err := NewEmbedder(EmbedderConfig{
		ModelPath: testModelPath(t), TokenizerPath: testTokenizerPath(t), LibraryPath: testLibraryPath(t),
		BatchSize: 4, EnableLengthBuckets: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer closeSessionOnly(e)

	texts := bucketTestTexts()
	const workers = 4
	var wg sync.WaitGroup
	errs := make(chan error, workers)
	for worker := range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			got, embedErr := e.EmbedDocuments(context.Background(), texts)
			if embedErr != nil {
				errs <- embedErr
				return
			}
			if len(got) != len(texts) {
				errs <- fmt.Errorf("worker %d: got %d vectors, want %d", worker, len(got), len(texts))
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
	if len(e.bucketSessions) != 4 {
		t.Errorf("concurrent calls cached %d sessions, want 4", len(e.bucketSessions))
	}
}

func TestEmbedder_LengthBuckets_ConcurrentClose(t *testing.T) {
	e, err := NewEmbedder(EmbedderConfig{
		ModelPath: testModelPath(t), TokenizerPath: testTokenizerPath(t), LibraryPath: testLibraryPath(t),
		BatchSize: 4, EnableLengthBuckets: true,
	})
	if err != nil {
		t.Fatal(err)
	}

	started := make(chan struct{})
	done := make(chan error, 1)
	go func() {
		close(started)
		_, embedErr := e.EmbedDocuments(context.Background(), bucketTestTexts())
		done <- embedErr
	}()
	<-started
	closeSessionOnly(e) // serializes with any in-flight call before destroying sessions
	<-done              // the raced call may finish or observe the closed state; both are safe
	if _, err := e.EmbedDocuments(context.Background(), []string{"after close"}); err == nil {
		t.Fatal("use after concurrent close succeeded")
	}
	for key, sess := range e.bucketSessions {
		t.Errorf("bucket session %v retained after close: %p", key, sess)
	}
}

func bucketTestTexts() []string {
	// Deliberately interleave lengths so grouped inference must scatter results
	// back to the original positions.
	return []string{
		strings.Repeat("long deterministic token ", 300),
		"tiny document",
		strings.Repeat("medium token ", 90),
		strings.Repeat("short token ", 45),
		"another tiny document",
	}
}

func assertEmbeddingParity(t *testing.T, got, want [][]float32, tolerance float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("vector count = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if len(got[i]) != len(want[i]) {
			t.Fatalf("vector %d dim = %d, want %d", i, len(got[i]), len(want[i]))
		}
		var maxDiff float64
		for dim := range want[i] {
			maxDiff = max(maxDiff, math.Abs(float64(got[i][dim]-want[i][dim])))
		}
		if maxDiff > tolerance {
			t.Errorf("vector %d max absolute difference = %g, tolerance %g", i, maxDiff, tolerance)
		}
	}
}
