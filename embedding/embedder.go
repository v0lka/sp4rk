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
	// A single text always uses the dedicated batchSize=1 fast-path session and
	// never touches the batch session.
	// A value of 0 (the default) selects DefaultBatchSize (32). Negative values
	// are treated like 0 rather than rejected: the field is not validated, so
	// callers should pass a non-negative value.
	BatchSize int

	// IntraOpThreads limits the number of ONNX Runtime intra-op threads used
	// during inference. A value of 0 (the default) preserves the legacy
	// behavior of letting ONNX Runtime choose the thread count (the session is
	// created with a nil *SessionOptions). A positive value N constrains
	// intra-op parallelism to exactly N threads, which is useful for bounding
	// CPU usage in resource-constrained environments such as the desktop app.
	// Negative values are treated as 0 (legacy behavior) rather than rejected:
	// the field is not validated, so callers should pass a non-negative value.
	IntraOpThreads int

	// Logger for structured logging. If nil, a discard logger is used.
	Logger *slog.Logger
}

// Embedder provides ONNX-based text embedding using jina-embeddings-v2-small-en.
// It is safe for concurrent use.
type Embedder struct {
	tokenizer *Tokenizer
	modelPath string
	maxSeqLen int
	hiddenDim int
	batchSize int
	logger    *slog.Logger
	mu        sync.Mutex
	sess      *onnxSession // persistent session for batchSize=1 (fast path)
	// batchSess is the persistent session for multi-text EmbedDocuments calls,
	// with input tensors shaped [batchSize, maxSeqLen]. It is created lazily on
	// the first multi-text call (session creation costs ~2s, so embedders that
	// only ever embed single texts never pay for it). Guarded by mu.
	batchSess *onnxSession
	sessOpts  *ort.SessionOptions
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

	logger.Info("initializing ONNX Runtime", "library", cfg.LibraryPath)
	if err := initONNXRuntime(cfg.LibraryPath); err != nil {
		return nil, fmt.Errorf("initializing ONNX Runtime: %w", err)
	}

	// Build session options (limits intra-op threads when configured). Must
	// run after initONNXRuntime, because ort.NewSessionOptions requires the
	// ONNX environment to be initialized. A nil result preserves the legacy
	// behavior (session created with nil *SessionOptions).
	sessOpts, err := buildSessionOptions(cfg.IntraOpThreads)
	if err != nil {
		_ = destroyONNXRuntime()
		return nil, fmt.Errorf("building ONNX session options: %w", err)
	}

	logger.Info("loading tokenizer", "path", cfg.TokenizerPath)
	tok, err := NewTokenizer(cfg.TokenizerPath)
	if err != nil {
		// Clean up ONNX env on failure.
		if sessOpts != nil {
			_ = sessOpts.Destroy()
		}
		_ = destroyONNXRuntime()
		return nil, fmt.Errorf("loading tokenizer: %w", err)
	}

	logger.Info("creating persistent ONNX session", "model", cfg.ModelPath)
	sess, err := newONNXSession(cfg.ModelPath, 1, maxSeqLen, hiddenDim, sessOpts)
	if err != nil {
		if sessOpts != nil {
			_ = sessOpts.Destroy()
		}
		_ = destroyONNXRuntime()
		return nil, fmt.Errorf("creating persistent ONNX session: %w", err)
	}

	logger.Info("embedder initialized",
		"model", cfg.ModelPath,
		"maxSeqLen", maxSeqLen,
		"hiddenDim", hiddenDim,
		"batchSize", batchSize,
	)

	return &Embedder{
		tokenizer: tok,
		modelPath: cfg.ModelPath,
		maxSeqLen: maxSeqLen,
		hiddenDim: hiddenDim,
		batchSize: batchSize,
		logger:    logger,
		sess:      sess,
		sessOpts:  sessOpts,
	}, nil
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
	if e.tokenizer == nil {
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
	inputIDs, attentionMask, tokenTypeIDs, err := e.tokenizer.EncodeBatch(texts, e.maxSeqLen)
	if err != nil {
		return nil, fmt.Errorf("embedding documents: tokenizer encode: %w", err)
	}

	// Fast path: use the persistent session for single-text embedding.
	// This is the common case when chromem-go calls EmbeddingFunc one text at a time.
	if numTexts == 1 && e.sess != nil {
		e.logger.Debug("running inference (persistent session)", "seqLen", e.maxSeqLen)
		vec, err := e.sess.run(inputIDs, attentionMask, tokenTypeIDs)
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
		vecs, err := e.batchSess.runBatch(end-start,
			inputIDs[lo:hi], attentionMask[lo:hi], tokenTypeIDs[lo:hi])
		if err != nil {
			return nil, fmt.Errorf("embedding batch chunk [%d:%d] of %d documents: %w",
				start, end, numTexts, err)
		}
		results = append(results, vecs...)
	}

	return results, nil
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
	sess, err := newONNXSession(e.modelPath, e.batchSize, e.maxSeqLen, e.hiddenDim, e.sessOpts)
	if err != nil {
		return fmt.Errorf("creating persistent batch ONNX session: %w", err)
	}
	e.batchSess = sess
	e.logger.Info("persistent batch ONNX session ready",
		"batchSize", e.batchSize, "elapsed", time.Since(started).Round(time.Millisecond))
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

	e.logger.Info("closing embedder, destroying ONNX Runtime environment")
	if e.sess != nil {
		e.sess.destroy()
		e.sess = nil
	}
	if e.batchSess != nil {
		e.batchSess.destroy()
		e.batchSess = nil
	}
	if e.sessOpts != nil {
		_ = e.sessOpts.Destroy()
		e.sessOpts = nil
	}
	// Mark the embedder closed so EmbedDocuments/EmbedQuery return an error
	// instead of touching the destroyed ONNX environment.
	e.tokenizer = nil
	if err := destroyONNXRuntime(); err != nil {
		return fmt.Errorf("destroying ONNX Runtime environment: %w", err)
	}
	return nil
}
