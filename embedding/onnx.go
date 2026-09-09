package embedding

import (
	"errors"
	"fmt"
	"math"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	ort "github.com/yalue/onnxruntime_go"
)

// onnxSessionsCreated counts ONNX sessions successfully created via
// newONNXSession. Production code never reads it; it exists so tests can
// assert that steady-state inference performs zero session creations
// (session creation costs ~2s, inference ~50ms).
var onnxSessionsCreated atomic.Int64

// onnxInferenceRuns counts successful session.Run invocations across all
// sessions (failed attempts are tracked only through Telemetry).
// Production code never reads it; it exists so tests can assert that a batch
// of n <= capacity texts performs exactly one inference per call.
var onnxInferenceRuns atomic.Int64

// onnxInitOnce ensures initONNXRuntime performs its work at most once per process.
// onnxInitErr caches the outcome of that single initialization attempt.
var (
	onnxInitOnce sync.Once
	onnxInitErr  error
)

// initONNXRuntime initializes the global ONNX Runtime environment.
// Must be called once before creating any sessions. libraryPath is the path to
// the ONNX Runtime shared library.
//
// Initialization is guarded by sync.Once, so only the first call performs the
// actual work; all subsequent calls return the cached result. Consequences:
//   - Once initialized, the ONNX Runtime cannot be reinitialized in the same
//     process, even after destroyONNXRuntime has been called. The first
//     libraryPath supplied is final.
//   - If the first initialization fails, every later call returns the same
//     error without retrying.
func initONNXRuntime(libraryPath string) error {
	onnxInitOnce.Do(func() {
		ort.SetSharedLibraryPath(libraryPath)
		if err := ort.InitializeEnvironment(); err != nil {
			onnxInitErr = fmt.Errorf("initializing ONNX Runtime environment: %w", err)
			return
		}
	})
	return onnxInitErr
}

// destroyONNXRuntime cleans up the global ONNX Runtime environment.
func destroyONNXRuntime() error {
	return ort.DestroyEnvironment()
}

// Execution provider identifiers accepted by EmbedderConfig.ExecutionProvider
// and buildSessionOptions. The empty string is a synonym for
// ExecutionProviderCPU and is the zero-value default.
const (
	// ExecutionProviderCPU runs inference on the ONNX Runtime CPU execution
	// provider — the runtime's own default when no provider is appended.
	ExecutionProviderCPU = "cpu"

	// ExecutionProviderCUDA runs inference on an NVIDIA GPU via the CUDA
	// execution provider. It requires a CUDA-enabled ONNX Runtime build:
	// LibraryPath must point at a libonnxruntime built with GPU support, with
	// libonnxruntime_providers_shared and libonnxruntime_providers_cuda sitting
	// beside it (the runtime dlopens them from the main library's directory).
	ExecutionProviderCUDA = "cuda"
)

// buildSessionOptions constructs ONNX Runtime session options for the
// requested execution provider, limiting intra-op parallelism to
// intraOpThreads.
//
// It returns nil when the CPU provider is requested and intraOpThreads <= 0,
// preserving the legacy behavior in which NewAdvancedSession is called with a
// nil *SessionOptions (byte-identical to the pre-existing code path).
// Otherwise a fresh *SessionOptions is allocated via ort.NewSessionOptions and
// configured: SetIntraOpNumThreads for a positive intraOpThreads, and an
// appended CUDA execution provider for provider == ExecutionProviderCUDA.
//
// ONNX Runtime never selects a GPU on its own: without an explicit
// AppendExecutionProviderCUDA call it executes the graph on the CPU provider
// even when the shared library is a GPU build. That is why a non-nil
// *SessionOptions is mandatory for CUDA regardless of the thread count.
//
// The ONNX Runtime environment must be initialized before calling this with
// anything but the legacy nil-returning combination, because
// ort.NewSessionOptions requires it. The caller owns the returned options and
// must call Destroy() exactly once when they are no longer needed (typically
// in Embedder.Close). The CUDA provider options are owned by this function:
// AppendExecutionProviderCUDA copies what it needs, so they are destroyed
// before returning.
func buildSessionOptions(provider string, deviceID, intraOpThreads int) (*ort.SessionOptions, error) {
	switch provider {
	case "", ExecutionProviderCPU, ExecutionProviderCUDA:
	default:
		return nil, fmt.Errorf("unknown execution provider %q (want %q or %q)",
			provider, ExecutionProviderCPU, ExecutionProviderCUDA)
	}

	cuda := provider == ExecutionProviderCUDA
	if !cuda && intraOpThreads <= 0 {
		return nil, nil
	}

	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("creating ONNX session options: %w", err)
	}
	if intraOpThreads > 0 {
		if err := opts.SetIntraOpNumThreads(intraOpThreads); err != nil {
			_ = opts.Destroy()
			return nil, fmt.Errorf("setting intra-op thread count: %w", err)
		}
	}
	if cuda {
		if err := appendCUDAProvider(opts, deviceID); err != nil {
			_ = opts.Destroy()
			return nil, err
		}
	}
	return opts, nil
}

// appendCUDAProvider configures and appends the CUDA execution provider to
// opts, pinning it to deviceID. Everything beyond the device id is left at the
// runtime's defaults. On failure opts is left untouched for the caller to
// destroy.
func appendCUDAProvider(opts *ort.SessionOptions, deviceID int) error {
	cudaOpts, err := ort.NewCUDAProviderOptions()
	if err != nil {
		return fmt.Errorf("creating CUDA provider options: %w", err)
	}
	defer func() { _ = cudaOpts.Destroy() }()

	if err := cudaOpts.Update(map[string]string{"device_id": strconv.Itoa(deviceID)}); err != nil {
		return fmt.Errorf("setting CUDA device_id=%d: %w", deviceID, err)
	}
	if err := opts.AppendExecutionProviderCUDA(cudaOpts); err != nil {
		return fmt.Errorf("appending CUDA execution provider: %w", err)
	}
	return nil
}

// onnxSession holds a reusable ONNX Runtime session with pre-allocated tensors
// for fixed-capacity batch inference. Creating an ONNX session is expensive
// (~2s for model loading + graph optimization), while inference is fast
// (~50ms). Reusing the session eliminates per-call overhead: one session with
// capacity 1 serves the single-text fast path, and one session with the
// configured batch capacity serves all multi-text EmbedDocuments calls
// (smaller batches are zero-padded up to the capacity).
type onnxSession struct {
	session    *ort.AdvancedSession
	inputIDs   *ort.Tensor[int64]
	attMask    *ort.Tensor[int64]
	tokenTypes *ort.Tensor[int64]
	output     *ort.Tensor[float32]
	batchSize  int // fixed row capacity of the input/output tensors
	seqLen     int
	hiddenDim  int
	telemetry  *Telemetry
	closed     atomic.Bool
}

// newONNXSession creates a persistent ONNX session whose input tensors are
// shaped [batchSize, seqLen] and output tensor [batchSize, seqLen, hiddenDim].
// batchSize is the fixed capacity of the session; individual runs may feed
// fewer rows (see runBatch). The session and its tensors are kept alive for
// reuse across multiple calls.
// opts may be nil (legacy behavior) or a *SessionOptions produced by
// buildSessionOptions to limit intra-op parallelism. opts ownership is NOT
// transferred: the caller keeps the handle and must destroy it separately.
func newONNXSession(modelPath string, batchSize, seqLen, hiddenDim int, opts *ort.SessionOptions, telemetry *Telemetry) (*onnxSession, error) {
	if batchSize < 1 {
		return nil, fmt.Errorf("batchSize must be >= 1, got %d", batchSize)
	}
	inputShape := ort.NewShape(int64(batchSize), int64(seqLen))

	inputIDsData := make([]int64, batchSize*seqLen)
	inputIDsTensor, err := ort.NewTensor(inputShape, inputIDsData)
	if err != nil {
		return nil, fmt.Errorf("creating input_ids tensor: %w", err)
	}

	attMaskData := make([]int64, batchSize*seqLen)
	attMaskTensor, err := ort.NewTensor(inputShape, attMaskData)
	if err != nil {
		_ = inputIDsTensor.Destroy()
		return nil, fmt.Errorf("creating attention_mask tensor: %w", err)
	}

	tokenTypesData := make([]int64, batchSize*seqLen)
	tokenTypesTensor, err := ort.NewTensor(inputShape, tokenTypesData)
	if err != nil {
		_ = inputIDsTensor.Destroy()
		_ = attMaskTensor.Destroy()
		return nil, fmt.Errorf("creating token_type_ids tensor: %w", err)
	}

	outputShape := ort.NewShape(int64(batchSize), int64(seqLen), int64(hiddenDim))
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		_ = inputIDsTensor.Destroy()
		_ = attMaskTensor.Destroy()
		_ = tokenTypesTensor.Destroy()
		return nil, fmt.Errorf("creating output tensor: %w", err)
	}

	session, err := ort.NewAdvancedSession(
		modelPath,
		[]string{"input_ids", "attention_mask", "token_type_ids"},
		[]string{"last_hidden_state"},
		[]ort.Value{inputIDsTensor, attMaskTensor, tokenTypesTensor},
		[]ort.Value{outputTensor},
		opts,
	)
	if err != nil {
		_ = inputIDsTensor.Destroy()
		_ = attMaskTensor.Destroy()
		_ = tokenTypesTensor.Destroy()
		_ = outputTensor.Destroy()
		return nil, fmt.Errorf("creating ONNX session: %w", err)
	}

	onnxSessionsCreated.Add(1)

	return &onnxSession{
		session:    session,
		inputIDs:   inputIDsTensor,
		attMask:    attMaskTensor,
		tokenTypes: tokenTypesTensor,
		output:     outputTensor,
		batchSize:  batchSize,
		seqLen:     seqLen,
		hiddenDim:  hiddenDim,
		telemetry:  telemetry,
	}, nil
}

// run executes inference using the persistent session with the given input data.
// The caller must ensure len(inputIDs) == len(attMask) == len(tokenTypes) == seqLen.
// Returns a single pooled, L2-normalized embedding vector.
func (s *onnxSession) run(inputIDs, attMask, tokenTypes []int64) ([]float32, error) {
	if s == nil || s.closed.Load() {
		return nil, errors.New("ONNX session is closed")
	}
	// Validate input lengths to prevent stale data from prior inferences.
	if len(inputIDs) != s.seqLen || len(attMask) != s.seqLen || len(tokenTypes) != s.seqLen {
		return nil, fmt.Errorf("input length mismatch: got (%d,%d,%d), want seqLen=%d",
			len(inputIDs), len(attMask), len(tokenTypes), s.seqLen)
	}
	copy(s.inputIDs.GetData(), inputIDs)
	copy(s.attMask.GetData(), attMask)
	copy(s.tokenTypes.GetData(), tokenTypes)

	inferenceStarted := time.Now()
	runErr := s.session.Run()
	// Count every attempt (not only successes) so InferenceCount always
	// equals Stages[StageONNXInference].Calls and a derived average
	// inference latency stays meaningful after transient failures.
	s.telemetry.observeInference(1, 1, time.Since(inferenceStarted))
	if runErr != nil {
		return nil, fmt.Errorf("running ONNX inference: %w", runErr)
	}
	onnxInferenceRuns.Add(1)

	results := meanPoolAndNormalize(s.output.GetData(), attMask, 1, s.seqLen, s.hiddenDim)
	return results[0], nil
}

// runBatch executes ONE inference on the persistent batch session for the
// first n rows of the flattened inputs. Each input slice must hold exactly
// n*seqLen elements in row-major order ([n, seqLen]). Rows n..batchSize-1 of
// the session tensors are zero-padded so they cannot influence the returned
// embeddings; pooling covers only the first n rows, and padded rows are
// never pooled.
func (s *onnxSession) runBatch(n int, inputIDs, attMask, tokenTypes []int64) ([][]float32, error) {
	if s == nil || s.closed.Load() {
		return nil, errors.New("ONNX session is closed")
	}
	if n < 1 || n > s.batchSize {
		return nil, fmt.Errorf("batch rows out of range: got %d, capacity %d", n, s.batchSize)
	}
	want := n * s.seqLen
	if len(inputIDs) != want || len(attMask) != want || len(tokenTypes) != want {
		return nil, fmt.Errorf("input length mismatch: got (%d,%d,%d), want n*seqLen=%d",
			len(inputIDs), len(attMask), len(tokenTypes), want)
	}

	// Zero the full tensors first so any rows beyond n are zero-padded, then
	// copy the n real rows over the front. This also clears stale data from
	// prior inferences for rows the current call no longer fills.
	idsData := s.inputIDs.GetData()
	maskData := s.attMask.GetData()
	typesData := s.tokenTypes.GetData()
	clear(idsData)
	clear(maskData)
	clear(typesData)
	copy(idsData, inputIDs)
	copy(maskData, attMask)
	copy(typesData, tokenTypes)

	inferenceStarted := time.Now()
	runErr := s.session.Run()
	// Count every attempt (not only successes) — see run for the rationale.
	s.telemetry.observeInference(n, s.batchSize, time.Since(inferenceStarted))
	if runErr != nil {
		return nil, fmt.Errorf("running ONNX inference: %w", runErr)
	}
	onnxInferenceRuns.Add(1)

	// Rows are independent, so pool over the n real rows only; padded rows
	// are never touched.
	pooled := meanPoolAndNormalize(s.output.GetData(), maskData, n, s.seqLen, s.hiddenDim)
	return pooled, nil
}

// destroy releases the session and all associated tensors exactly once.
func (s *onnxSession) destroy() {
	if s == nil || !s.closed.CompareAndSwap(false, true) {
		return
	}
	if s.session != nil {
		_ = s.session.Destroy()
		s.session = nil
	}
	if s.inputIDs != nil {
		_ = s.inputIDs.Destroy()
		s.inputIDs = nil
	}
	if s.attMask != nil {
		_ = s.attMask.Destroy()
		s.attMask = nil
	}
	if s.tokenTypes != nil {
		_ = s.tokenTypes.Destroy()
		s.tokenTypes = nil
	}
	if s.output != nil {
		_ = s.output.Destroy()
		s.output = nil
	}
}

// meanPoolAndNormalize performs masked mean pooling across the sequence dimension
// and L2-normalizes the resulting vectors.
func meanPoolAndNormalize(hiddenStates []float32, attentionMask []int64, batchSize, seqLen, hiddenDim int) [][]float32 {
	embeddings := make([][]float32, batchSize)

	for b := range batchSize {
		embedding := make([]float32, hiddenDim)
		var maskSum float32

		for s := range seqLen {
			mask := float32(attentionMask[b*seqLen+s])
			if mask == 0 {
				continue
			}
			maskSum += mask
			baseIdx := (b*seqLen + s) * hiddenDim
			for d := range hiddenDim {
				embedding[d] += hiddenStates[baseIdx+d] * mask
			}
		}

		// Average by mask sum.
		if maskSum > 0 {
			for d := range hiddenDim {
				embedding[d] /= maskSum
			}
		}

		// L2 normalization.
		var norm float64
		for d := range hiddenDim {
			norm += float64(embedding[d]) * float64(embedding[d])
		}
		norm = math.Sqrt(norm)
		if norm > 0 {
			invNorm := float32(1.0 / norm)
			for d := range hiddenDim {
				embedding[d] *= invNorm
			}
		}

		embeddings[b] = embedding
	}

	return embeddings
}
