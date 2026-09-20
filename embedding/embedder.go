package embedding

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sync"
	"time"

	chromem "github.com/philippgille/chromem-go"
	ort "github.com/yalue/onnxruntime_go"
)

const (
	// DefaultMaxSeqLength is the default maximum sequence length for tokenization.
	// jina-v2-small supports up to 8192, but 512 is practical for most use cases.
	DefaultMaxSeqLength = 512

	// DefaultHiddenDim is the embedding dimension for jina-embeddings-v2-small-en.
	DefaultHiddenDim = 512

	// DefaultBatchSize is the default fixed row capacity of the persistent
	// batch ONNX session used for multi-text EmbedDocuments calls.
	//
	// The value is justified by measurement (see onnx_bench_test.go,
	// BenchmarkEmbedderPerItem vs BenchmarkEmbedderBatch; jina-v2-small,
	// seqLen=512, 256 realistic ~1500-char chunks, 16-core arm64, intra-op
	// threads = all cores): per-item embedding ~35.8 docs/sec; batched
	// embedding plateaus at ~40-42 docs/sec — B=8: 39.2, B=16: 40.9,
	// B=32: 41.7-42.4, B=64: 41.0, B=128: 41.8. 32 sits at the knee of the
	// curve: within noise of the throughput plateau while larger capacities
	// linearly increase both the per-inference output tensor (B x 512 x 512
	// x 4 bytes = B MiB) and single-call latency (B=32 ~0.75s vs B=128
	// ~3.1s) with no additional throughput.
	DefaultBatchSize = 32
)

// EmbedderConfig holds configuration for creating an Embedder.
type EmbedderConfig struct {
	// ModelPath is the path to the ONNX model file (.onnx).
	ModelPath string

	// TokenizerPath is the path to the HuggingFace tokenizer.json file.
	TokenizerPath string

	// LibraryPath is the path to the ONNX Runtime shared library
	// (e.g., libonnxruntime.dylib, libonnxruntime.so, onnxruntime.dll).
	LibraryPath string

	// MaxSeqLength is the maximum token sequence length. Defaults to 512.
	MaxSeqLength int

	// HiddenDim is the embedding dimension of the model. Defaults to 512 for jina-v2-small.
	HiddenDim int

	// BatchSize is the fixed row capacity of the persistent batch ONNX session
	// used when EmbedDocuments receives more than one text. Multi-text batches
	// are processed in chunks of at most BatchSize rows with a single ONNX
	// inference per full/padded chunk; smaller chunks are zero-padded up to the
	// capacity (padded rows are masked out during pooling and discarded).
	// In fixed mode, a single text uses the dedicated batchSize=1 fast-path
	// session. In bucket mode, every call uses the lazily cached session for its
	// smallest fitting sequence bucket.
	// A value of 0 (the default) selects DefaultBatchSize (32). Negative values
	// are treated like 0 rather than rejected: the field is not validated, so
	// callers should pass a non-negative value.
	BatchSize int

	// EnableBatchPipeline overlaps tokenization of fixed-mode batch N+1 with
	// ONNX inference of batch N. The producer is bounded to one completed batch
	// ahead, while ONNX execution remains serialized by the Embedder mutex.
	// It is opt-in: false preserves the serial baseline until benchmarks show a
	// throughput improvement above the documented noise threshold.
	// It has no effect while EnableLengthBuckets is enabled — bucket mode
	// tokenizes serially — and NewEmbedder logs a warning about that.
	EnableBatchPipeline bool

	// EnableLengthBuckets groups tokenized documents into the smallest fitting
	// fixed sequence length (64, 128, 256, or MaxSeqLength) and lazily creates
	// one persistent ONNX session per used bucket. It is opt-in: false preserves
	// the fixed-MaxSeqLength baseline and its single batch session. When also
	// EnableBatchPipeline is set, the pipeline is inactive (bucket mode wins)
	// and NewEmbedder logs a warning about that.
	EnableLengthBuckets bool

	// IntraOpThreads limits the number of ONNX Runtime intra-op threads used
	// during inference. A value of 0 (the default) preserves the legacy
	// behavior of letting ONNX Runtime choose the thread count (the session is
	// created with a nil *SessionOptions). A positive value N constrains
	// intra-op parallelism to exactly N threads, which is useful for bounding
	// CPU usage in resource-constrained environments such as the desktop app.
	// Negative values are treated as 0 (legacy behavior) rather than rejected:
	// the field is not validated, so callers should pass a non-negative value.
	IntraOpThreads int

	// ExecutionProvider selects the ONNX Runtime execution provider used for
	// inference: ExecutionProviderCPU ("cpu"), ExecutionProviderCUDA ("cuda"),
	// or ExecutionProviderAuto ("auto"). The empty string (the default) means
	// "cpu" and preserves the legacy behavior exactly.
	//
	// ONNX Runtime does not pick a GPU on its own — pointing LibraryPath at a
	// GPU build changes nothing unless the provider is requested here. "cuda"
	// additionally requires the CUDA provider shared libraries next to
	// LibraryPath and a working driver/toolkit; when any of that is missing,
	// NewEmbedder fails loudly rather than silently falling back to the CPU.
	//
	// "auto" prefers CUDA and degrades gracefully: NewEmbedder attempts the
	// CUDA provider and, when that attempt fails (CPU-only ONNX Runtime build,
	// missing driver, invalid device id, ...), logs a WARN and continues on
	// the CPU provider instead of failing. Which side won is observable via
	// Embedder.ExecutionProvider. DeviceID participates only in the CUDA
	// attempt, never in the CPU fallback.
	//
	// Unknown values are rejected.
	ExecutionProvider string

	// DeviceID is the GPU device index used when ExecutionProvider resolves
	// to "cuda" (explicitly or via "auto"'s first attempt). Defaults to 0,
	// which is the right choice on single-GPU machines and selects the first
	// device reported by the driver on multi-GPU ones. It is ignored by the
	// CPU provider. Negative values are rejected by NewEmbedder for every
	// provider including cpu, so one config struct validates uniformly.
	DeviceID int

	// Logger for structured logging. If nil, a discard logger is used.
	Logger *slog.Logger

	// Telemetry optionally collects bounded, content-free stage aggregates.
	// Nil disables collection with no background goroutines or exporters.
	Telemetry *Telemetry
}

// sessionKey identifies one persistent fixed-shape ONNX session.
type sessionKey struct {
	batchSize int
	seqLen    int
}

// bucketItem retains the original result position while documents are grouped
// by sequence length.
type bucketItem struct {
	index int
}

// Embedder provides ONNX-based text embedding using jina-embeddings-v2-small-en.
// It is safe for concurrent use.
type Embedder struct {
	tokenizer     *Tokenizer
	modelPath     string
	maxSeqLen     int
	hiddenDim     int
	batchSize     int
	batchPipeline bool
	lengthBuckets bool
	logger        *slog.Logger
	telemetry     *Telemetry
	mu            sync.Mutex
	closed        bool
	sess          *onnxSession // legacy fixed-max session for batchSize=1
	// batchSess is the legacy fixed-max session for multi-text calls. It remains
	// the default until measured bucket benchmarks justify changing the default.
	batchSess *onnxSession
	// bucketSessions contains only buckets used by successful inference calls.
	// Creation, use, and destruction are serialized by mu.
	bucketSessions map[sessionKey]*onnxSession
	sessOpts       *ort.SessionOptions
	// executionProvider is the provider inference actually runs on: always
	// ExecutionProviderCPU or ExecutionProviderCUDA, never "auto" — the auto
	// request is resolved during NewEmbedder, and ExecutionProvider() reports
	// the winner.
	executionProvider string
	// runner pins every ONNX Runtime call to one OS thread; nil for the CPU
	// provider, which does not need the pinning. See ortRunner.
	runner *ortRunner
	// deviceID and intraOpThreads retain the config NewEmbedder used to build
	// session options, so a later session-creation failure can rebuild them on
	// the CPU provider (see degradeToCPU).
	deviceID       int
	intraOpThreads int
	// cudaFallbackPending is true while inference runs on a CUDA provider that
	// an "auto" request selected. A lazily created session that cannot be built
	// on the GPU then degrades the embedder to the CPU provider once and retries
	// (see newSessionWithFallback), mirroring the eager-session fallback in
	// NewEmbedder.
	cudaFallbackPending bool
}

// runOnORTThread executes fn on r's dedicated thread, or inline when r is nil.
func runOnORTThread(r *ortRunner, fn func()) {
	if r == nil {
		fn()
		return
	}
	r.do(fn)
}

// onORTThread executes fn on the embedder's dedicated ONNX thread, or inline
// when no runner is in use. The caller must hold e.mu, which is what keeps the
// ONNX sessions single-threaded for the CPU provider too.
func (e *Embedder) onORTThread(fn func()) {
	runOnORTThread(e.runner, fn)
}

// NewEmbedder creates a new Embedder by loading the tokenizer and initializing
// the ONNX Runtime environment.
//
// DESIGN NOTE: The ONNX Runtime is a process-global singleton — only one Embedder
// can exist at a time and it lives for the process lifetime. This is now ENFORCED
// by sync.Once in initONNXRuntime: the first successful initialization is final
// and cannot be repeated in the same process, even after Close/destroy. There is
// no reference counting; desktop.App is the single owner responsible for calling
// Close() at shutdown. This is a known limitation for library-reuse scenarios but
// sufficient for the single-process desktop app architecture.
func NewEmbedder(cfg EmbedderConfig) (*Embedder, error) {
	if cfg.ModelPath == "" {
		return nil, errors.New("ModelPath is required")
	}
	if cfg.TokenizerPath == "" {
		return nil, errors.New("TokenizerPath is required")
	}
	if cfg.LibraryPath == "" {
		return nil, errors.New("LibraryPath is required")
	}

	// An unknown provider is a configuration error: it must fail loudly, not
	// silently degrade to the CPU (which would look like a working setup).
	requested, err := normalizeExecutionProvider(cfg.ExecutionProvider)
	if err != nil {
		return nil, err
	}

	// A negative DeviceID is a configuration error in the same spirit: it must
	// fail loudly for every provider, not silently degrade. On cuda it would
	// sink into the CUDA provider options and surface as a confusing runtime
	// rejection; on auto it would silently trigger the CPU fallback, hiding
	// the misconfiguration behind a WARN; on cpu it would be silently ignored.
	// All three are silent-failure shapes the loud-validation rule exists to
	// prevent — one config struct validates uniformly for every provider.
	if cfg.DeviceID < 0 {
		return nil, fmt.Errorf("DeviceID must be a non-negative GPU ordinal, got %d", cfg.DeviceID)
	}

	maxSeqLen := cfg.MaxSeqLength
	if maxSeqLen <= 0 {
		maxSeqLen = DefaultMaxSeqLength
	}

	hiddenDim := cfg.HiddenDim
	if hiddenDim <= 0 {
		hiddenDim = DefaultHiddenDim
	}

	batchSize := cfg.BatchSize
	if batchSize <= 0 {
		batchSize = DefaultBatchSize
	}

	logger := cfg.Logger
	if logger == nil {
		logger = slog.New(slog.DiscardHandler)
	}

	// The batch pipeline only engages in fixed mode; a host enabling both
	// flags gets bucket mode with serial tokenization, which is easy to
	// mistake for a pipeline that is silently underperforming — surface the
	// interaction once instead.
	if cfg.EnableBatchPipeline && cfg.EnableLengthBuckets {
		logger.Warn("EnableBatchPipeline is ignored while EnableLengthBuckets is enabled; running in bucket mode")
	}

	// GPU providers keep per-thread state that must not be duplicated across
	// Go's scheduler threads (see ortRunner). Every ONNX call below — and
	// every one made later through the returned Embedder — is funnelled onto
	// this single thread. The CPU provider needs none of it and keeps calling
	// ONNX inline, exactly as before. "auto" starts on the runner because its
	// first attempt is CUDA; the runner is stopped when that attempt fails
	// and the CPU fallback continues inline.
	effective := requested
	var runner *ortRunner
	if effective == ExecutionProviderCUDA || effective == ExecutionProviderAuto {
		runner = newORTRunner()
	}
	// attempt is the provider buildSessionOptions is asked for: "auto" is not
	// a constructible provider, its first attempt is CUDA. Only used for the
	// build call; `effective` tracks what inference will run on.
	attempt := effective
	if attempt == ExecutionProviderAuto {
		attempt = ExecutionProviderCUDA
	}

	// cleanup releases whatever the aborted initialization already acquired
	// and shuts the ONNX thread down. It must NEVER destroy the ONNX Runtime
	// environment: initONNXRuntime is sync.Once-guarded per process, so a
	// destroyed environment could never be reinitialized — not for a later
	// NewEmbedder retry, and not for the "auto" CPU fallback below. Only
	// Embedder.Close performs the full teardown, as the embedder's single
	// owner at shutdown.
	cleanup := func(sessOpts *ort.SessionOptions) {
		runOnORTThread(runner, func() {
			if sessOpts != nil {
				_ = sessOpts.Destroy()
			}
		})
		if runner != nil {
			runner.stop()
		}
	}

	logger.Info("initializing ONNX Runtime", "library", cfg.LibraryPath)
	var initErr error
	runOnORTThread(runner, func() { initErr = initONNXRuntime(cfg.LibraryPath) })
	if initErr != nil {
		// A failed environment initialization is fatal for every provider:
		// the CPU fallback needs the same environment, so there is nothing to
		// fall back to. (No env teardown here — see the cleanup note above.)
		if runner != nil {
			runner.stop()
		}
		return nil, fmt.Errorf("initializing ONNX Runtime: %w", initErr)
	}

	// Build session options (selects the execution provider and limits
	// intra-op threads when configured). Must run after initONNXRuntime,
	// because ort.NewSessionOptions requires the ONNX environment to be
	// initialized. A nil result preserves the legacy behavior (session created
	// with nil *SessionOptions).
	//
	// For "auto" this is the CUDA attempt: a CPU-only ONNX Runtime build, a
	// missing driver or an invalid device id all fail here, and the request
	// degrades to the CPU provider with a WARN instead of an error.
	buildOpts := func(provider string) (*ort.SessionOptions, error) {
		var (
			opts *ort.SessionOptions
			err  error
		)
		runOnORTThread(runner, func() {
			opts, err = buildSessionOptions(provider, cfg.DeviceID, cfg.IntraOpThreads)
		})
		return opts, err
	}
	sessOpts, err := buildOpts(attempt)
	if err != nil {
		if effective != ExecutionProviderAuto {
			cleanup(nil)
			return nil, fmt.Errorf("building ONNX session options: %w", err)
		}
		// auto: WARN and continue on CPU. The CUDA attempt may already have
		// allocated nothing beyond options internals (buildSessionOptions
		// destroys its own partial options on failure), so the fallback just
		// abandons the dedicated thread and proceeds inline like a plain CPU
		// embedder.
		logger.Warn("CUDA execution provider unavailable, falling back to CPU",
			"error", err.Error(), "deviceID", cfg.DeviceID)
		runner.stop()
		runner = nil
		effective = ExecutionProviderCPU
		sessOpts, err = buildOpts(effective)
		if err != nil {
			cleanup(sessOpts)
			return nil, fmt.Errorf("building ONNX session options: %w", err)
		}
	}
	// From here on sessOpts is owned by the embedder-to-be; cleanup(sessOpts)
	// releases it on failure. A successful CUDA attempt also resolves "auto"
	// to its winner — but the eager-session block below still needs to know
	// whether this was an "auto" request that landed on CUDA, so it can retry
	// on the CPU provider when the GPU session itself cannot be built. Once
	// resolved, `effective` is never "auto" again, so capture the auto-ness here.
	autoOnCUDA := effective == ExecutionProviderAuto
	if effective == ExecutionProviderAuto {
		effective = ExecutionProviderCUDA
	}

	logger.Info("loading tokenizer", "path", cfg.TokenizerPath)
	tok, err := NewTokenizer(cfg.TokenizerPath)
	if err != nil {
		cleanup(sessOpts)
		return nil, fmt.Errorf("loading tokenizer: %w", err)
	}

	var sess *onnxSession
	// sessionStarted is reset by each eager/lazy creation site right before
	// its newONNXSession call, so observeSession measures that attempt only.
	var sessionStarted time.Time
	// newSess creates the eager batchSize=1 session through the runner (a
	// no-op indirection for the CPU provider) and observes the session-create
	// stage in telemetry, mirroring the lazy batch/bucket session paths.
	newSess := func() (*onnxSession, error) {
		var (
			s    *onnxSession
			sErr error
		)
		runOnORTThread(runner, func() {
			s, sErr = newONNXSession(cfg.ModelPath, 1, maxSeqLen, hiddenDim, sessOpts, cfg.Telemetry)
		})
		if sErr == nil {
			cfg.Telemetry.observeSession(time.Since(sessionStarted))
		}
		return s, sErr
	}
	if !cfg.EnableLengthBuckets {
		logger.Info("creating persistent ONNX session", "model", cfg.ModelPath)
		sessionStarted = time.Now()
		sess, err = newSess()
		if err != nil {
			if !autoOnCUDA {
				cleanup(sessOpts)
				return nil, fmt.Errorf("creating persistent ONNX session: %w", err)
			}
			// auto + CUDA options: the provider accepted the options but the
			// session itself could not be built on the GPU (driver/device trouble,
			// out of device memory at graph allocation). Retry once on the CPU
			// provider before giving up — a failed attempt has already released
			// its own tensors, and the CPU retry cannot make things worse.
			cudaSessErr := err
			logger.Warn("CUDA session creation failed, falling back to CPU",
				"error", err.Error(), "deviceID", cfg.DeviceID)
			cleanup(sessOpts)
			runner = nil
			effective = ExecutionProviderCPU
			sessOpts, err = buildOpts(effective)
			if err != nil {
				cleanup(sessOpts)
				return nil, fmt.Errorf("building ONNX session options: %w", err)
			}
			// Measure the CPU retry from its own start; the failed CUDA
			// attempt above is not part of this session-create observation.
			sessionStarted = time.Now()
			sess, err = newSess()
			if err != nil {
				cleanup(sessOpts)
				return nil, fmt.Errorf("creating persistent ONNX session after CUDA->CPU fallback (original CUDA error: %w): %w", cudaSessErr, err)
			}
		}
	}

	initialized := []any{
		"model", cfg.ModelPath,
		"maxSeqLen", maxSeqLen,
		"hiddenDim", hiddenDim,
		"batchSize", batchSize,
		"batchPipeline", cfg.EnableBatchPipeline,
		"lengthBuckets", cfg.EnableLengthBuckets,
		"executionProvider", effective,
		"deviceID", cfg.DeviceID,
	}
	if requested != effective {
		// An auto request that landed on the CPU must never be silent: the
		// WARN above names the reason, and this line keeps the mismatch
		// greppable in the single "embedder initialized" entry.
		initialized = append(initialized, "requestedExecutionProvider", requested)
	}
	logger.Info("embedder initialized", initialized...)

	return &Embedder{
		tokenizer:         tok,
		modelPath:         cfg.ModelPath,
		maxSeqLen:         maxSeqLen,
		hiddenDim:         hiddenDim,
		batchSize:         batchSize,
		batchPipeline:     cfg.EnableBatchPipeline,
		lengthBuckets:     cfg.EnableLengthBuckets,
		logger:            logger,
		telemetry:         cfg.Telemetry,
		sess:              sess,
		bucketSessions:    make(map[sessionKey]*onnxSession, 4),
		sessOpts:          sessOpts,
		executionProvider: effective,
		runner:            runner,
		deviceID:          cfg.DeviceID,
		intraOpThreads:    cfg.IntraOpThreads,
		// Only an auto request that is still (or again) on CUDA can degrade to
		// the CPU from a later lazy session-creation failure. If the eager-session
		// fallback above already ran, effective is CPU and this stays false.
		cudaFallbackPending: autoOnCUDA && effective == ExecutionProviderCUDA,
	}, nil
}

// ExecutionProvider reports the execution provider the embedder actually runs
// on: ExecutionProviderCPU ("cpu") or ExecutionProviderCUDA ("cuda"). The
// "auto" request is resolved inside NewEmbedder, so this never returns "auto";
// comparing the return value against the requested one is how callers detect
// a silent CUDA->CPU slide — including the zero-value embedder, which reports
// "cpu" (the legacy default).
func (e *Embedder) ExecutionProvider() string {
	if e.executionProvider == "" {
		return ExecutionProviderCPU
	}
	return e.executionProvider
}

// EmbedDocuments embeds a batch of text documents and returns their embedding vectors.
func (e *Embedder) EmbedDocuments(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	// Guard against use-after-close.
	if e.closed || e.tokenizer == nil {
		return nil, errors.New("embedder is closed")
	}

	// Honor cancellation that arrived while waiting for the lock BEFORE the
	// unbounded preparatory work: EncodeBatch tokenizes every text, and the
	// batch path's lazy session creation loads the model (~2s), all under
	// e.mu — a cancelled caller must bail out instead of stalling concurrent
	// EmbedQuery calls first. The per-chunk re-checks below additionally
	// bound wasted inference on long batches.
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	numTexts := len(texts)
	if e.batchPipeline && !e.lengthBuckets && numTexts > e.batchSize {
		if err := e.ensureBatchSession(); err != nil {
			return nil, err
		}
		return e.embedFixedPipeline(ctx, texts)
	}

	inputIDs, attentionMask, tokenTypeIDs, lengths, err := e.tokenizeDocuments(texts)
	if err != nil {
		return nil, err
	}

	if e.lengthBuckets {
		return e.embedLengthBuckets(ctx, inputIDs, attentionMask, tokenTypeIDs, lengths)
	}

	// Fast path: use the persistent session for single-text embedding.
	// This is the common case when chromem-go calls EmbeddingFunc one text at a time.
	if numTexts == 1 && e.sess != nil {
		e.logger.Debug("running inference (persistent session)", "seqLen", e.maxSeqLen)
		var (
			vec []float32
			err error
		)
		e.onORTThread(func() { vec, err = e.sess.run(inputIDs, attentionMask, tokenTypeIDs) })
		if err != nil {
			return nil, fmt.Errorf("embedding document: %w", err)
		}
		return [][]float32{vec}, nil
	}

	// Batch path: reuse the persistent batch session. It is created lazily on
	// the first multi-text call (creating an ONNX session costs ~2s, so
	// single-text-only embedders never pay for it); every later call reuses it
	// with zero session-creation overhead. The batch is processed in chunks of
	// at most e.batchSize rows; a partial final chunk is zero-padded up to the
	// session capacity, and padded rows are masked out during pooling.
	if err := e.ensureBatchSession(); err != nil {
		return nil, err
	}

	e.logger.Debug("running batch inference (persistent batch session)",
		"texts", numTexts, "chunkSize", e.batchSize, "seqLen", e.maxSeqLen)

	results := make([][]float32, 0, numTexts)
	for start := 0; start < numTexts; start += e.batchSize {
		// Re-check the context before every chunk: ONNX inference blocks and
		// is uninterruptible, so a long multi-chunk batch must remain
		// cancellable between chunks (cancellation that arrived while waiting
		// for the lock is already honored above, before any preparatory work).
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}
		end := start + e.batchSize
		if end > numTexts {
			end = numTexts
		}
		lo, hi := start*e.maxSeqLen, end*e.maxSeqLen
		var (
			vecs [][]float32
			err  error
		)
		e.onORTThread(func() {
			vecs, err = e.batchSess.runBatch(end-start,
				inputIDs[lo:hi], attentionMask[lo:hi], tokenTypeIDs[lo:hi])
		})
		if err != nil {
			return nil, fmt.Errorf("embedding batch chunk [%d:%d] of %d documents: %w",
				start, end, numTexts, err)
		}
		results = append(results, vecs...)
	}

	return results, nil
}

// sequenceBucket returns the smallest supported fixed sequence length that can
// hold actualLen. MaxSeqLength remains the hard truncation ceiling.
func (e *Embedder) sequenceBucket(actualLen int) int {
	for _, bucket := range [...]int{64, 128, 256, 512} {
		if bucket >= e.maxSeqLen {
			return e.maxSeqLen
		}
		if actualLen <= bucket {
			return bucket
		}
	}
	return e.maxSeqLen
}

// embedLengthBuckets groups rows stably by sequence length and scatters each
// embedding back to its original index. The caller holds e.mu.
func (e *Embedder) embedLengthBuckets(ctx context.Context, inputIDs, attentionMask, tokenTypeIDs []int64, lengths []int) ([][]float32, error) {
	groups := make(map[int][]bucketItem, 4)
	bucketOrder := make([]int, 0, 4)
	for i, length := range lengths {
		bucket := e.sequenceBucket(length)
		if _, ok := groups[bucket]; !ok {
			bucketOrder = append(bucketOrder, bucket)
		}
		groups[bucket] = append(groups[bucket], bucketItem{index: i})
	}

	results := make([][]float32, len(lengths))
	for _, bucket := range bucketOrder {
		items := groups[bucket]
		for start := 0; start < len(items); start += e.batchSize {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			end := min(start+e.batchSize, len(items))
			rows := end - start
			ids := make([]int64, rows*bucket)
			mask := make([]int64, rows*bucket)
			types := make([]int64, rows*bucket)
			for row, item := range items[start:end] {
				src := item.index * e.maxSeqLen
				dst := row * bucket
				copy(ids[dst:dst+bucket], inputIDs[src:src+bucket])
				copy(mask[dst:dst+bucket], attentionMask[src:src+bucket])
				copy(types[dst:dst+bucket], tokenTypeIDs[src:src+bucket])
			}

			sess, err := e.ensureBucketSession(bucket)
			if err != nil {
				return nil, err
			}
			var (
				vecs [][]float32
				rErr error
			)
			e.onORTThread(func() {
				vecs, rErr = sess.runBatch(rows, ids, mask, types)
			})
			if rErr != nil {
				return nil, fmt.Errorf("embedding length bucket %d rows [%d:%d]: %w", bucket, start, end, rErr)
			}
			for row, item := range items[start:end] {
				results[item.index] = vecs[row]
			}
		}
	}
	return results, nil
}

// newSession builds one persistent ONNX session on the embedder's current
// provider. The caller must hold e.mu.
func (e *Embedder) newSession(batchSize, seqLen int) (*onnxSession, error) {
	var (
		sess *onnxSession
		err  error
	)
	e.onORTThread(func() {
		sess, err = newONNXSession(e.modelPath, batchSize, seqLen, e.hiddenDim, e.sessOpts, e.telemetry)
	})
	return sess, err
}

// newSessionWithFallback builds a persistent ONNX session, honouring the "auto"
// contract for the lazily created sessions (length buckets and the batch
// session), not just the eager one: when the session cannot be built on the
// CUDA provider an "auto" request selected, the embedder degrades to the CPU
// provider once and retries — the same CUDA->CPU degrade the eager session
// performs inside NewEmbedder. An explicit "cuda" request still fails loudly
// (cudaFallbackPending is false for it). The caller must hold e.mu.
func (e *Embedder) newSessionWithFallback(batchSize, seqLen int) (*onnxSession, error) {
	sess, err := e.newSession(batchSize, seqLen)
	if err == nil || !e.canDegradeToCPU() {
		return sess, err
	}
	cudaErr := err
	if fbErr := e.degradeToCPU(cudaErr); fbErr != nil {
		return nil, fmt.Errorf("creating persistent ONNX session on CUDA failed (%w); CPU fallback also failed: %w", cudaErr, fbErr)
	}
	sess, err = e.newSession(batchSize, seqLen)
	if err != nil {
		return nil, fmt.Errorf("creating persistent ONNX session after CUDA->CPU fallback (original CUDA error: %w): %w", cudaErr, err)
	}
	return sess, nil
}

// canDegradeToCPU reports whether the embedder can safely retry session creation
// on the CPU provider: the request was an "auto" one now running on CUDA, and no
// CUDA-backed session is live yet — a live session needs the dedicated ONNX
// thread the fallback stops, so degrading mid-flight would strand it. In
// practice this is the first session creation in length-bucket mode, which is
// exactly the path the eager fallback does not cover.
func (e *Embedder) canDegradeToCPU() bool {
	return e.cudaFallbackPending && e.sess == nil && e.batchSess == nil && len(e.bucketSessions) == 0
}

// degradeToCPU rebuilds the session options on the CPU provider, stops the
// dedicated ONNX thread, and records that inference now runs on the CPU. It is
// called when a session could not be built on a CUDA provider chosen by an
// "auto" request. The caller must hold e.mu, must have verified
// canDegradeToCPU, and must retry session creation afterwards.
func (e *Embedder) degradeToCPU(cause error) error {
	e.logger.Warn("CUDA execution provider unavailable, falling back to CPU",
		"error", cause.Error(), "deviceID", e.deviceID)
	var (
		opts *ort.SessionOptions
		err  error
	)
	e.onORTThread(func() {
		if e.sessOpts != nil {
			_ = e.sessOpts.Destroy()
			e.sessOpts = nil
		}
		opts, err = buildSessionOptions(ExecutionProviderCPU, e.deviceID, e.intraOpThreads)
	})
	if err != nil {
		return fmt.Errorf("building ONNX session options: %w", err)
	}
	if e.runner != nil {
		e.runner.stop()
		e.runner = nil
	}
	e.sessOpts = opts
	e.executionProvider = ExecutionProviderCPU
	e.cudaFallbackPending = false
	return nil
}

// ensureBucketSession lazily creates and caches one session per sequence bucket.
// The caller must hold e.mu.
func (e *Embedder) ensureBucketSession(seqLen int) (*onnxSession, error) {
	key := sessionKey{batchSize: e.batchSize, seqLen: seqLen}
	if sess := e.bucketSessions[key]; sess != nil {
		return sess, nil
	}
	started := time.Now()
	sess, err := e.newSessionWithFallback(e.batchSize, seqLen)
	if err != nil {
		return nil, fmt.Errorf("creating persistent ONNX session for length bucket %d: %w", seqLen, err)
	}
	e.bucketSessions[key] = sess
	e.telemetry.observeSession(time.Since(started))
	return sess, nil
}

// ensureBatchSession lazily creates the persistent batch ONNX session on its
// first invocation. The caller must hold e.mu.
func (e *Embedder) ensureBatchSession() error {
	if e.batchSess != nil {
		return nil
	}
	e.logger.Info("creating persistent batch ONNX session (one-time init; loading the model takes a few seconds)",
		"batchSize", e.batchSize, "seqLen", e.maxSeqLen)
	started := time.Now()
	sess, err := e.newSessionWithFallback(e.batchSize, e.maxSeqLen)
	if err != nil {
		return fmt.Errorf("creating persistent batch ONNX session: %w", err)
	}
	elapsed := time.Since(started)
	e.batchSess = sess
	e.telemetry.observeSession(elapsed)
	e.logger.Info("persistent batch ONNX session ready",
		"batchSize", e.batchSize, "elapsed", elapsed.Round(time.Millisecond))
	return nil
}

// EmbedQuery embeds a single text query and returns its embedding vector.
func (e *Embedder) EmbedQuery(ctx context.Context, text string) ([]float32, error) {
	results, err := e.EmbedDocuments(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	if len(results) == 0 {
		return nil, errors.New("no embedding returned")
	}
	return results[0], nil
}

// EmbeddingFunc returns a chromem-go compatible embedding function that can be
// passed to chromem.NewCollection as the embedding function parameter.
func (e *Embedder) EmbeddingFunc() chromem.EmbeddingFunc {
	return func(ctx context.Context, text string) ([]float32, error) {
		return e.EmbedQuery(ctx, text)
	}
}

// Close releases the ONNX Runtime environment and associated resources.
func (e *Embedder) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.closed {
		return nil
	}
	e.closed = true
	e.logger.Info("closing embedder, destroying ONNX Runtime environment")
	// Teardown touches the same CUDA context the sessions were built on, so
	// it runs on the ONNX thread too. One job for the whole sequence — sess,
	// batchSess, every bucket session, sessOpts, and the environment destroy
	// — because the runner is stopped right after and no ONNX call may
	// outlive it.
	var destroyErr error
	e.onORTThread(func() {
		if e.sess != nil {
			e.sess.destroy()
			e.sess = nil
		}
		if e.batchSess != nil {
			e.batchSess.destroy()
			e.batchSess = nil
		}
		for key, sess := range e.bucketSessions {
			sess.destroy()
			delete(e.bucketSessions, key)
		}
		if e.sessOpts != nil {
			_ = e.sessOpts.Destroy()
			e.sessOpts = nil
		}
		destroyErr = destroyONNXRuntime()
	})
	if e.runner != nil {
		e.runner.stop()
		e.runner = nil
	}
	// Mark the embedder closed so EmbedDocuments/EmbedQuery return an error
	// instead of touching the destroyed ONNX environment.
	e.tokenizer = nil
	if destroyErr != nil {
		return fmt.Errorf("destroying ONNX Runtime environment: %w", destroyErr)
	}
	return nil
}
