package embedding

import (
	"context"
	"fmt"
	"os"
	"sort"
	"strings"
	"testing"
	"time"
)

// Benchmarks comparing per-item embedding (one EmbedQuery call per document,
// the pre-batching baseline) against batched EmbedDocuments at several fixed
// batch-session capacities. They justify DefaultBatchSize with data.
//
// All benchmarks are env-gated: they need the same variables as the env-gated
// tests in embedder_test.go and skip cleanly when any of them is unset:
//
//	EMBEDDING_TEST_MODEL_PATH     path to jina-v2-small.onnx
//	EMBEDDING_TEST_TOKENIZER_PATH path to tokenizer.json
//	EMBEDDING_TEST_LIBRARY_PATH   path to libonnxruntime.{dylib,so,dll}
//
// Run (single iteration per benchmark; a full corpus pass takes seconds):
//
//	EMBEDDING_TEST_MODEL_PATH=... EMBEDDING_TEST_TOKENIZER_PATH=... \
//	EMBEDDING_TEST_LIBRARY_PATH=... go test -bench=. -benchtime=1x ./embedding
//
// The benchmarks go through the public EmbedDocuments/EmbedQuery API and reuse
// the ONNX environment managed by TestMain, releasing only their own sessions
// via closeSessionOnly. IntraOpThreads is left at 0 (all cores), matching the
// c0wrk production default (vector_index.embedding_threads: 0).

const (
	// benchDocCount is the number of documents embedded per benchmark
	// iteration — large enough for several inferences at every batch size
	// under test (128 docs/batch → 2 runs, 8 → 32 runs).
	benchDocCount = 256

	// benchChunkChars is the target chunk length in characters, matching the
	// c0wrk vector-index default max_chunk_size (1500).
	benchChunkChars = 1500
)

// benchAssets returns the model, tokenizer and ONNX library paths from the
// environment, skipping the benchmark when any of them is unset.
func benchAssets(tb testing.TB) (modelPath, tokenizerPath, libraryPath string) {
	tb.Helper()
	modelPath = os.Getenv("EMBEDDING_TEST_MODEL_PATH")
	tokenizerPath = os.Getenv("EMBEDDING_TEST_TOKENIZER_PATH")
	libraryPath = os.Getenv("EMBEDDING_TEST_LIBRARY_PATH")
	if modelPath == "" || tokenizerPath == "" || libraryPath == "" {
		tb.Skip("EMBEDDING_TEST_MODEL_PATH, EMBEDDING_TEST_TOKENIZER_PATH or EMBEDDING_TEST_LIBRARY_PATH not set; skipping ONNX benchmark")
	}
	return modelPath, tokenizerPath, libraryPath
}

// benchChunks synthesizes docCount realistic chunk texts of ~chunkChars
// characters from the package's own Go sources (read from the working
// directory, which `go test` sets to the package dir). Source bytes are
// concatenated and cut into chunks at whitespace boundaries; if the sources
// yield fewer chunks than requested, the corpus wraps around (repeated
// chunks cost the same tokenization and inference work, so throughput
// numbers are unaffected).
func benchChunks(tb testing.TB, docCount, chunkChars int) []string {
	tb.Helper()

	entries, err := os.ReadDir(".")
	if err != nil {
		tb.Skipf("cannot read package sources: %v; skipping corpus-dependent benchmark", err)
	}
	var names []string
	for _, e := range entries {
		if !e.IsDir() && strings.HasSuffix(e.Name(), ".go") {
			names = append(names, e.Name())
		}
	}
	sort.Strings(names)
	if len(names) == 0 {
		tb.Skip("no .go sources in working directory; skipping corpus-dependent benchmark")
	}

	var src strings.Builder
	for _, name := range names {
		data, err := os.ReadFile(name)
		if err != nil {
			tb.Fatalf("reading %s: %v", name, err)
		}
		src.WriteString("\n\n// ---- ")
		src.WriteString(name)
		src.WriteString(" ----\n\n")
		src.Write(data)
	}

	// Materialize the corpus once: Builder.String() returns a copy, so
	// calling it per chunk would allocate the whole corpus on every iteration.
	corpus := src.String()
	chunks := make([]string, 0, len(corpus)/chunkChars+1)
	for i := 0; i < len(corpus); i += chunkChars {
		end := i + chunkChars
		if end > len(corpus) {
			end = len(corpus)
		}
		chunk := corpus[i:end]
		// Back off to the last whitespace so chunks do not cut words in half;
		// the skipped tail bytes are simply not reused.
		if end < len(corpus) {
			if sp := strings.LastIndexAny(chunk, " \t\n"); sp > chunkChars/2 {
				chunk = chunk[:sp]
			}
		}
		chunks = append(chunks, chunk)
	}
	if len(chunks) == 0 {
		tb.Fatalf("synthesized 0 chunks from %d source files", len(names))
	}

	docs := make([]string, docCount)
	for i := range docs {
		docs[i] = chunks[i%len(chunks)]
	}
	tb.Logf("corpus: %d docs of ~%d chars from %d unique chunks", docCount, chunkChars, len(chunks))
	return docs
}

// newBenchEmbedder builds an Embedder on the env-provided assets with the
// given batch capacity, releasing its sessions (but not the shared ONNX
// environment) on cleanup.
func newBenchEmbedder(tb testing.TB, batchSize int) *Embedder {
	tb.Helper()
	modelPath, tokenizerPath, libraryPath := benchAssets(tb)
	e, err := NewEmbedder(EmbedderConfig{
		ModelPath:     modelPath,
		TokenizerPath: tokenizerPath,
		LibraryPath:   libraryPath,
		BatchSize:     batchSize,
	})
	if err != nil {
		tb.Fatalf("NewEmbedder(batchSize=%d): %v", batchSize, err)
	}
	tb.Cleanup(func() { closeSessionOnly(e) })
	return e
}

// reportBenchMetrics reports throughput (docs/sec) and per-inference-call
// latency from the measured elapsed time and the exact ONNX inference count.
func reportBenchMetrics(b *testing.B, docs int, elapsed time.Duration, runs int64) {
	b.Helper()
	b.ReportMetric(float64(docs)*float64(b.N)/elapsed.Seconds(), "docs/sec")
	b.ReportMetric(float64(elapsed)/float64(runs)/float64(time.Millisecond), "ms/inference")
	b.ReportMetric(float64(runs)/float64(b.N), "inferences/op")
}

// BenchmarkEmbedderPerItem is the baseline: documents embedded one at a time
// through the single-text fast path, as chromem-go does when indexing without
// batching. One inference per document.
func BenchmarkEmbedderPerItem(b *testing.B) {
	e := newBenchEmbedder(b, DefaultBatchSize)
	docs := benchChunks(b, benchDocCount, benchChunkChars)
	ctx := context.Background()

	// Warm up the single-text session, tokenizer and first-run allocations.
	for i := range 4 {
		if _, err := e.EmbedQuery(ctx, docs[i]); err != nil {
			b.Fatalf("warmup EmbedQuery: %v", err)
		}
	}
	b.ResetTimer()

	runsBefore := onnxInferenceRuns.Load()
	started := time.Now()
	for i := 0; i < b.N; i++ {
		for _, doc := range docs {
			if _, err := e.EmbedQuery(ctx, doc); err != nil {
				b.Fatalf("EmbedQuery: %v", err)
			}
		}
	}
	elapsed := time.Since(started)
	runs := onnxInferenceRuns.Load() - runsBefore

	if runs != int64(len(docs))*int64(b.N) {
		b.Fatalf("inference runs = %d, want %d (one per document)", runs, int64(len(docs))*int64(b.N))
	}
	reportBenchMetrics(b, len(docs), elapsed, runs)
}

// BenchmarkEmbedderBatch measures batched EmbedDocuments throughput at the
// batch capacities considered for DefaultBatchSize. Each sub-benchmark embeds
// the full 256-doc corpus with a persistent batch session of the given
// capacity, so per iteration it performs ceil(256/capacity) inferences.
//
// Memory note: the output tensor of a capacity-B session is
// B×512×512×4 bytes = B MiB, so larger capacities trade peak memory for
// throughput (B=128 → a 128 MiB output tensor per inference).
func BenchmarkEmbedderBatch(b *testing.B) {
	for _, size := range []int{8, 16, 32, 64, 128} {
		b.Run(fmt.Sprintf("batch%d", size), func(b *testing.B) {
			runBatchBenchmark(b, size)
		})
	}
}

func runBatchBenchmark(b *testing.B, batchSize int) {
	e := newBenchEmbedder(b, batchSize)
	docs := benchChunks(b, benchDocCount, benchChunkChars)
	ctx := context.Background()

	// Warm up outside the measurement: one full EmbedDocuments call on a
	// smaller slice triggers the lazy batch-session creation (~2s) plus
	// first-run allocations for a full chunk of batchSize rows.
	warm := docs[:min(batchSize, len(docs))]
	if _, err := e.EmbedDocuments(ctx, warm); err != nil {
		b.Fatalf("warmup EmbedDocuments(batch=%d): %v", batchSize, err)
	}
	b.ResetTimer()

	runsBefore := onnxInferenceRuns.Load()
	started := time.Now()
	for i := 0; i < b.N; i++ {
		if _, err := e.EmbedDocuments(ctx, docs); err != nil {
			b.Fatalf("EmbedDocuments(batch=%d): %v", batchSize, err)
		}
	}
	elapsed := time.Since(started)
	runs := onnxInferenceRuns.Load() - runsBefore

	wantRuns := int64(0)
	for start := 0; start < len(docs); start += batchSize {
		wantRuns++
	}
	if runs != wantRuns*int64(b.N) {
		b.Fatalf("inference runs = %d, want %d (ceil(%d/%d) per iteration)", runs, wantRuns*int64(b.N), len(docs), batchSize)
	}
	reportBenchMetrics(b, len(docs), elapsed, runs)
}
