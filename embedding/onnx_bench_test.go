package embedding

import (
	"context"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"
)

// Reproducible ONNX benchmarks. Assets are supplied explicitly and the entire
// benchmark suite skips cleanly when any path is absent:
//
//	EMBEDDING_TEST_MODEL_PATH=... EMBEDDING_TEST_TOKENIZER_PATH=... \
//	EMBEDDING_TEST_LIBRARY_PATH=... go test -bench=BenchmarkEmbedder -benchmem -benchtime=1x ./embedding
//
// The corpus is generated from fixed templates; it never reads repository
// source files. Telemetry reports only aggregate counts/timings and cannot
// contain corpus text or asset paths.

const benchDocCount = 256

func benchAssets(tb testing.TB) (modelPath, tokenizerPath, libraryPath string) {
	tb.Helper()
	modelPath = os.Getenv("EMBEDDING_TEST_MODEL_PATH")
	tokenizerPath = os.Getenv("EMBEDDING_TEST_TOKENIZER_PATH")
	libraryPath = os.Getenv("EMBEDDING_TEST_LIBRARY_PATH")
	if modelPath == "" || tokenizerPath == "" || libraryPath == "" {
		tb.Skip("ONNX assets are not configured; set EMBEDDING_TEST_MODEL_PATH, EMBEDDING_TEST_TOKENIZER_PATH, and EMBEDDING_TEST_LIBRARY_PATH")
	}
	for label, path := range map[string]string{"model": modelPath, "tokenizer": tokenizerPath, "library": libraryPath} {
		if _, err := os.Stat(path); err != nil {
			tb.Skipf("ONNX %s asset unavailable; skipping benchmark: %v", label, err)
		}
	}
	return modelPath, tokenizerPath, libraryPath
}

func benchChunks(kind string, count int) []string {
	docs := make([]string, count)
	for i := range docs {
		short := fmt.Sprintf("package corpus\nfunc item%d(value int) int { return value + %d }", i, i)
		if kind == "short" || i%2 == 0 {
			docs[i] = short
			continue
		}
		tail := strings.Repeat(fmt.Sprintf(" deterministic worker %d validates telemetry batching and indexing latency.", i), 22)
		docs[i] = short + tail
	}
	return docs
}

func newBenchEmbedder(tb testing.TB, batchSize int, lengthBuckets bool, telemetry *Telemetry) *Embedder {
	tb.Helper()
	modelPath, tokenizerPath, libraryPath := benchAssets(tb)
	e, err := NewEmbedder(EmbedderConfig{
		ModelPath: modelPath, TokenizerPath: tokenizerPath, LibraryPath: libraryPath,
		BatchSize: batchSize, EnableLengthBuckets: lengthBuckets, Telemetry: telemetry,
	})
	if err != nil {
		tb.Fatalf("NewEmbedder(batchSize=%d): %v", batchSize, err)
	}
	tb.Cleanup(func() { closeSessionOnly(e) })
	return e
}

func reportBenchMetrics(b *testing.B, s, setup TelemetrySnapshot, docs int, elapsed time.Duration, peakBefore uint64) {
	b.Helper()
	b.ReportMetric(float64(docs)*float64(b.N)/elapsed.Seconds(), "docs/sec")
	b.ReportMetric(float64(elapsed)/float64(time.Millisecond)/float64(b.N), "ms/op")
	b.ReportMetric(float64(s.Stages[StageTokenization].Duration)/float64(time.Millisecond)/float64(b.N), "ms/tokenization")
	b.ReportMetric(float64(s.Stages[StageONNXInference].Duration)/float64(time.Millisecond)/float64(b.N), "ms/inference-total")
	b.ReportMetric(float64(s.InferenceCount)/float64(b.N), "inferences/op")
	b.ReportMetric(float64(setup.SessionCount), "sessions")
	b.ReportMetric(float64(setup.Stages[StageSessionCreate].Duration)/float64(time.Millisecond), "ms/session-create")
	b.ReportMetric(s.BatchFill(), "batch-fill")
	b.ReportMetric(s.AverageTokens(), "tokens/avg")
	b.ReportMetric(float64(s.MinTokens), "tokens/min")
	b.ReportMetric(float64(s.MaxTokens), "tokens/max")
	peakAfter := benchmarkPeakRSSBytes()
	b.ReportMetric(float64(peakAfter)/(1024*1024), "peak-RSS-MiB")
	if peakAfter >= peakBefore {
		b.ReportMetric(float64(peakAfter-peakBefore)/(1024*1024), "peak-RSS-delta-MiB")
	}
}

func BenchmarkEmbedderPerItem(b *testing.B) {
	for _, corpus := range []string{"short", "mixed"} {
		b.Run("corpus="+corpus, func(b *testing.B) {
			telemetry := &Telemetry{}
			e := newBenchEmbedder(b, DefaultBatchSize, false, telemetry)
			docs := benchChunks(corpus, benchDocCount)
			ctx := context.Background()
			if _, err := e.EmbedQuery(ctx, docs[0]); err != nil {
				b.Fatal(err)
			}
			setup := telemetry.Snapshot()
			telemetry.Reset()
			peakBefore := benchmarkPeakRSSBytes()
			b.ReportAllocs()
			b.ResetTimer()
			started := time.Now()
			for range b.N {
				for _, doc := range docs {
					if _, err := e.EmbedQuery(ctx, doc); err != nil {
						b.Fatal(err)
					}
				}
			}
			elapsed := time.Since(started)
			b.StopTimer()
			s := telemetry.Snapshot()
			if want := int64(len(docs) * b.N); s.InferenceCount != want {
				b.Fatalf("inferences = %d, want %d", s.InferenceCount, want)
			}
			reportBenchMetrics(b, s, setup, len(docs), elapsed, peakBefore)
		})
	}
}

func BenchmarkEmbedderBatch(b *testing.B) {
	for _, mode := range []struct {
		name    string
		buckets bool
	}{
		{name: "fixed"},
		{name: "buckets", buckets: true},
	} {
		for _, corpus := range []string{"short", "mixed"} {
			for _, size := range []int{8, 16, 32, 64} {
				b.Run(fmt.Sprintf("mode=%s/corpus=%s/batch=%d", mode.name, corpus, size), func(b *testing.B) {
					runBatchBenchmark(b, corpus, size, mode.buckets)
				})
			}
		}
	}
}

func BenchmarkEmbedderBatchPipeline(b *testing.B) {
	for _, pipeline := range []bool{false, true} {
		mode := "serial"
		if pipeline {
			mode = "pipeline"
		}
		for _, corpus := range []string{"short", "mixed"} {
			for _, size := range []int{8, 16, 32} {
				b.Run(fmt.Sprintf("mode=%s/corpus=%s/batch=%d", mode, corpus, size), func(b *testing.B) {
					runPipelineBenchmark(b, corpus, size, pipeline)
				})
			}
		}
	}
}

func runPipelineBenchmark(b *testing.B, corpus string, batchSize int, pipeline bool) {
	telemetry := &Telemetry{}
	modelPath, tokenizerPath, libraryPath := benchAssets(b)
	e, err := NewEmbedder(EmbedderConfig{
		ModelPath: modelPath, TokenizerPath: tokenizerPath, LibraryPath: libraryPath,
		BatchSize: batchSize, EnableBatchPipeline: pipeline, Telemetry: telemetry,
	})
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { closeSessionOnly(e) })
	docs := benchChunks(corpus, benchDocCount)
	ctx := context.Background()
	if _, err := e.EmbedDocuments(ctx, docs); err != nil {
		b.Fatal(err)
	}
	setup := telemetry.Snapshot()
	telemetry.Reset()
	peakBefore := benchmarkPeakRSSBytes()
	b.ReportAllocs()
	b.ResetTimer()
	started := time.Now()
	for range b.N {
		if _, err := e.EmbedDocuments(ctx, docs); err != nil {
			b.Fatal(err)
		}
	}
	elapsed := time.Since(started)
	b.StopTimer()

	s := telemetry.Snapshot()
	wantInferences := int64((len(docs)+batchSize-1)/batchSize) * int64(b.N)
	if s.InferenceCount != wantInferences {
		b.Fatalf("inferences = %d, want %d", s.InferenceCount, wantInferences)
	}
	if s.SessionCount != 0 {
		b.Fatalf("steady-state benchmark created %d sessions, want 0", s.SessionCount)
	}
	reportBenchMetrics(b, s, setup, len(docs), elapsed, peakBefore)
}

func BenchmarkEmbedderDynamicShape(b *testing.B) {
	for _, corpus := range []string{"short", "mixed"} {
		b.Run("corpus="+corpus, func(b *testing.B) {
			modelPath, tokenizerPath, libraryPath := benchAssets(b)
			if err := initONNXRuntime(libraryPath); err != nil {
				b.Fatal(err)
			}
			tok, err := NewTokenizer(tokenizerPath)
			if err != nil {
				b.Fatal(err)
			}
			telemetry := &Telemetry{}
			session, err := newDynamicBenchSession(modelPath, DefaultHiddenDim, nil, telemetry)
			if err != nil {
				b.Fatal(err)
			}
			b.Cleanup(session.destroy)
			docs := benchChunks(corpus, benchDocCount)
			ctx := context.Background()
			if _, err := benchmarkDynamicDocuments(ctx, tok, session, docs, DefaultMaxSeqLength); err != nil {
				b.Fatal(err)
			}
			setup := telemetry.Snapshot()
			telemetry.Reset()
			peakBefore := benchmarkPeakRSSBytes()
			b.ReportAllocs()
			b.ResetTimer()
			started := time.Now()
			for range b.N {
				if _, err := benchmarkDynamicDocuments(ctx, tok, session, docs, DefaultMaxSeqLength); err != nil {
					b.Fatal(err)
				}
			}
			elapsed := time.Since(started)
			b.StopTimer()
			reportBenchMetrics(b, telemetry.Snapshot(), setup, len(docs), elapsed, peakBefore)
		})
	}
}

func runBatchBenchmark(b *testing.B, corpus string, batchSize int, lengthBuckets bool) {
	telemetry := &Telemetry{}
	e := newBenchEmbedder(b, batchSize, lengthBuckets, telemetry)
	docs := benchChunks(corpus, benchDocCount)
	ctx := context.Background()
	if _, err := e.EmbedDocuments(ctx, docs); err != nil {
		b.Fatal(err)
	}
	setup := telemetry.Snapshot()
	telemetry.Reset()
	peakBefore := benchmarkPeakRSSBytes()
	b.ReportAllocs()
	b.ResetTimer()
	started := time.Now()
	for range b.N {
		if _, err := e.EmbedDocuments(ctx, docs); err != nil {
			b.Fatal(err)
		}
	}
	elapsed := time.Since(started)
	b.StopTimer()

	s := telemetry.Snapshot()
	if s.InferenceCount == 0 {
		b.Fatal("benchmark performed zero inferences")
	}
	if s.SessionCount != 0 {
		b.Fatalf("steady-state benchmark created %d sessions, want 0", s.SessionCount)
	}
	reportBenchMetrics(b, s, setup, len(docs), elapsed, peakBefore)
}
