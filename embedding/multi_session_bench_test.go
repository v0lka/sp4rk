package embedding

import (
	"context"
	"errors"
	"fmt"
	"os"
	"runtime"
	"sort"
	"strconv"
	"sync"
	"testing"
	"time"

	ort "github.com/yalue/onnxruntime_go"
)

const defaultMultiSessionQueueDepth = 2

// benchmarkSessionOverheadBytes is a conservative constant bound for the
// per-worker ONNX Runtime session cost beyond the model file and the owned
// fixed-capacity tensors (graph state, arena, thread pools). Calibrated from
// an isolated workers=1 measurement (Apple M4 Max, ORT 1.28.1, batch 8,
// seqLen 64): peak RSS 392.6 MiB against a 124.9 MiB model+tensor estimate —
// roughly 268 MiB of overhead including the process baseline. 384 MiB keeps
// admission an upper bound rather than a tight fit; the measured peak-RSS
// metric remains the real enforcement signal.
const benchmarkSessionOverheadBytes = 384 * 1024 * 1024

type benchmarkSessionJob struct {
	ctx         context.Context
	sequence    uint64
	query       bool
	rows        int
	ids         []int64
	mask        []int64
	types       []int64
	submittedAt time.Time
	result      chan benchmarkSessionResult
}

type benchmarkSessionResult struct {
	sequence uint64
	vectors  [][]float32
	wait     time.Duration
	err      error
}

type benchmarkSessionOwner struct {
	session *onnxSession
	opts    *ort.SessionOptions
}

func (w *benchmarkSessionOwner) run(job benchmarkSessionJob) benchmarkSessionResult {
	wait := time.Since(job.submittedAt)
	if err := job.ctx.Err(); err != nil {
		return benchmarkSessionResult{sequence: job.sequence, wait: wait, err: err}
	}
	vectors, err := w.session.runBatch(job.rows, job.ids, job.mask, job.types)
	return benchmarkSessionResult{sequence: job.sequence, vectors: vectors, wait: wait, err: err}
}

func (w *benchmarkSessionOwner) close() {
	w.session.destroy()
	if w.opts != nil {
		_ = w.opts.Destroy()
		w.opts = nil
	}
}

type benchmarkSessionPool struct {
	queries chan benchmarkSessionJob
	docs    chan benchmarkSessionJob
	stop    chan struct{}
	wg      sync.WaitGroup
	once    sync.Once
	workers int
}

func newBenchmarkSessionPool(modelPath string, workers, threads, batchSize, seqLen, hiddenDim, queueDepth int, memoryCap uint64, telemetry *Telemetry) (*benchmarkSessionPool, error) {
	if workers < 1 || queueDepth < 1 {
		return nil, errors.New("workers and queue depth must be positive")
	}
	modelInfo, err := os.Stat(modelPath)
	if err != nil {
		return nil, fmt.Errorf("stat model for memory admission: %w", err)
	}
	perWorker := uint64(modelInfo.Size()) + benchmarkOwnedTensorBytes(batchSize, seqLen, hiddenDim) + benchmarkSessionOverheadBytes
	if perWorker > memoryCap || uint64(workers) > memoryCap/perWorker {
		return nil, fmt.Errorf("worker memory admission rejected: workers=%d estimated=%d cap=%d", workers, perWorker*uint64(workers), memoryCap)
	}

	pool := &benchmarkSessionPool{
		queries: make(chan benchmarkSessionJob, queueDepth),
		docs:    make(chan benchmarkSessionJob, queueDepth),
		stop:    make(chan struct{}),
		workers: workers,
	}
	owners := make([]*benchmarkSessionOwner, 0, workers)
	for range workers {
		opts, optsErr := buildSessionOptions(threads)
		if optsErr != nil {
			for _, owner := range owners {
				owner.close()
			}
			return nil, optsErr
		}
		session, sessionErr := newONNXSession(modelPath, batchSize, seqLen, hiddenDim, opts, telemetry)
		if sessionErr != nil {
			if opts != nil {
				_ = opts.Destroy()
			}
			for _, owner := range owners {
				owner.close()
			}
			return nil, sessionErr
		}
		owners = append(owners, &benchmarkSessionOwner{session: session, opts: opts})
	}
	for _, owner := range owners {
		pool.wg.Add(1)
		go pool.worker(owner)
	}
	return pool, nil
}

func benchmarkOwnedTensorBytes(batchSize, seqLen, hiddenDim int) uint64 {
	// Three int64 inputs plus one float32 output, all fixed-capacity and owned
	// by exactly one worker. The model bytes are accounted separately.
	return uint64(batchSize) * uint64(seqLen) * uint64(3*8+hiddenDim*4)
}

func (p *benchmarkSessionPool) worker(owner *benchmarkSessionOwner) {
	defer p.wg.Done()
	defer owner.close()
	for {
		var job benchmarkSessionJob
		var ok bool
		// Strict query preference whenever a query is already queued.
		select {
		case job, ok = <-p.queries:
		default:
			select {
			case job, ok = <-p.queries:
			case job, ok = <-p.docs:
			case <-p.stop:
				return
			}
		}
		if !ok {
			return
		}
		result := owner.run(job)
		select {
		case job.result <- result:
		case <-job.ctx.Done():
		case <-p.stop:
			return
		}
	}
}

func (p *benchmarkSessionPool) submit(job benchmarkSessionJob) error {
	// Deterministic fast-path for post-close submits: without this check the
	// select below races between a ready buffered send and the closed stop
	// channel, randomly succeeding after close.
	select {
	case <-p.stop:
		return errors.New("session pool is closed")
	default:
	}
	job.submittedAt = time.Now()
	queue := p.docs
	if job.query {
		queue = p.queries
	}
	select {
	case queue <- job:
		return nil
	case <-job.ctx.Done():
		return job.ctx.Err()
	case <-p.stop:
		return errors.New("session pool is closed")
	}
}

func (p *benchmarkSessionPool) close() {
	p.once.Do(func() {
		close(p.stop)
		p.wg.Wait()
	})
}

func BenchmarkEmbedderMultiSessionGrid(b *testing.B) {
	workers := benchmarkEnvInt(b, "EMBEDDING_GRID_WORKERS", 1)
	// threads=0 preserves the production configuration: buildSessionOptions
	// returns nil and ONNX Runtime picks the default intra-op thread count.
	threads := benchmarkEnvIntAllowZero(b, "EMBEDDING_GRID_THREADS", 0, 0)
	batchSize := benchmarkEnvInt(b, "EMBEDDING_GRID_BATCH", 8)
	seqLen := benchmarkEnvInt(b, "EMBEDDING_GRID_SEQUENCE", 64)
	memoryCapMiB := benchmarkEnvInt(b, "EMBEDDING_GRID_MEMORY_CAP_MIB", 4096)
	modelPath, tokenizerPath, libraryPath := benchAssets(b)
	if err := initONNXRuntime(libraryPath); err != nil {
		b.Fatal(err)
	}
	tokenizer, err := NewTokenizer(tokenizerPath)
	if err != nil {
		b.Fatal(err)
	}
	telemetry := &Telemetry{}
	pool, err := newBenchmarkSessionPool(modelPath, workers, threads, batchSize, seqLen, DefaultHiddenDim, defaultMultiSessionQueueDepth*workers, uint64(memoryCapMiB)*1024*1024, telemetry)
	if err != nil {
		b.Skipf("grid case rejected by memory cap: %v", err)
	}
	b.Cleanup(pool.close)

	docs := benchChunks("mixed", benchDocCount)
	ids, mask, types, _, err := tokenizer.EncodeBatchWithLengths(docs, seqLen)
	if err != nil {
		b.Fatal(err)
	}
	queryIDs, queryMask, queryTypes, err := tokenizer.Encode("query side latency probe", seqLen)
	if err != nil {
		b.Fatal(err)
	}

	// Sessions and tensors are warm before metrics begin.
	if err := runGridPass(context.Background(), pool, ids, mask, types, queryIDs, queryMask, queryTypes, len(docs), batchSize, seqLen, nil); err != nil {
		b.Fatal(err)
	}
	telemetry.Reset()
	peakBefore := benchmarkPeakRSSBytes()
	cpuBefore := benchmarkCPUTime()
	b.ReportAllocs()
	b.ResetTimer()
	started := time.Now()
	var queryLatencies []time.Duration
	for range b.N {
		if err := runGridPass(context.Background(), pool, ids, mask, types, queryIDs, queryMask, queryTypes, len(docs), batchSize, seqLen, &queryLatencies); err != nil {
			b.Fatal(err)
		}
	}
	elapsed := time.Since(started)
	b.StopTimer()
	cpu := benchmarkCPUTime() - cpuBefore
	p95 := benchmarkP95(queryLatencies)
	b.ReportMetric(float64(len(docs)*b.N)/elapsed.Seconds(), "docs/sec")
	b.ReportMetric(float64(p95)/float64(time.Millisecond), "query-p95-ms")
	b.ReportMetric(cpu.Seconds()/elapsed.Seconds(), "cpu-cores")
	b.ReportMetric(cpu.Seconds()/elapsed.Seconds()/float64(runtime.NumCPU())*100, "cpu-util-percent")
	b.ReportMetric(float64(benchmarkPeakRSSBytes())/(1024*1024), "peak-RSS-MiB")
	if peak := benchmarkPeakRSSBytes(); peak >= peakBefore {
		b.ReportMetric(float64(peak-peakBefore)/(1024*1024), "peak-RSS-delta-MiB")
	}
	b.ReportMetric(float64(workers), "workers")
	b.ReportMetric(float64(threads), "intra-op-threads")
	b.ReportMetric(float64(batchSize), "batch")
	b.ReportMetric(float64(seqLen), "sequence")
}

func runGridPass(ctx context.Context, pool *benchmarkSessionPool, ids, mask, types, queryIDs, queryMask, queryTypes []int64, docCount, batchSize, seqLen int, queryLatencies *[]time.Duration) error {
	jobs := (docCount + batchSize - 1) / batchSize
	results := make(chan benchmarkSessionResult, jobs+1)
	for i := range jobs {
		start := i * batchSize
		end := min(start+batchSize, docCount)
		lo, hi := start*seqLen, end*seqLen
		job := benchmarkSessionJob{ctx: ctx, sequence: uint64(i), rows: end - start, ids: ids[lo:hi], mask: mask[lo:hi], types: types[lo:hi], result: results}
		if err := pool.submit(job); err != nil {
			return err
		}
		if i == jobs/2 {
			query := benchmarkSessionJob{ctx: ctx, sequence: uint64(jobs), query: true, rows: 1, ids: queryIDs, mask: queryMask, types: queryTypes, result: results}
			if err := pool.submit(query); err != nil {
				return err
			}
		}
	}
	seen := make([]bool, jobs+1)
	for range jobs + 1 {
		select {
		case result := <-results:
			if result.err != nil {
				return result.err
			}
			if result.sequence >= uint64(len(seen)) || seen[result.sequence] {
				return errors.New("non-deterministic duplicate or invalid result sequence")
			}
			seen[result.sequence] = true
			if result.sequence == uint64(jobs) && queryLatencies != nil {
				// Worker-side wait (queue + inference), independent of when the
				// receiver drains the results channel.
				*queryLatencies = append(*queryLatencies, result.wait)
			}
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	return nil
}

func benchmarkP95(values []time.Duration) time.Duration {
	if len(values) == 0 {
		return 0
	}
	ordered := append([]time.Duration(nil), values...)
	sort.Slice(ordered, func(i, j int) bool { return ordered[i] < ordered[j] })
	return ordered[(len(ordered)*95+99)/100-1]
}

func benchmarkEnvInt(tb testing.TB, name string, fallback int) int {
	tb.Helper()
	return benchmarkEnvIntAllowZero(tb, name, fallback, 1)
}

// benchmarkEnvIntAllowZero parses a non-negative integer env override. Zero is
// meaningful for the intra-op axis (0 = ONNX Runtime default thread count, the
// production configuration); minimum bounds every other axis.
func benchmarkEnvIntAllowZero(tb testing.TB, name string, fallback, minimum int) int {
	tb.Helper()
	value := os.Getenv(name)
	if value == "" {
		return fallback
	}
	parsed, err := strconv.Atoi(value)
	if err != nil || parsed < minimum {
		tb.Fatalf("%s must be an integer >= %d, got %q", name, minimum, value)
	}
	return parsed
}
