package embedding

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"sync/atomic"
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

	// inferenceDegradeThreshold is the number of consecutive failed ONNX
	// inference attempts that triggers the runtime hardware->CPU degradation
	// for an "auto" request (see Embedder.embedInference). One failure stays
	// a loud error as before — a transient CUDA/OpenVINO hiccup must not
	// demote the embedder on its own; the second consecutive failure is the
	// evidence that the provider stopped working mid-flight (device lost,
	// exhausted device memory), which is exactly the situation the one-shot
	// degradation exists to survive.
	//
	// The counter only observes attempts that RETURN: a driver wedged inside
	// the ONNX Runtime C call never returns, so nothing is counted, the
	// threshold is never reached and no degradation fires for that failure
	// mode. Embedder.Wedged is the authoritative diagnosis there — consult it
	// instead of inferring a wedge from a streak of timeouts, and keep the
	// caller-side timeout as the backstop that bounds the wait.
	inferenceDegradeThreshold = 2
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
	// ExecutionProviderOpenVINO ("openvino"), or ExecutionProviderAuto
	// ("auto"). The empty string (the default) means "cpu" and preserves the
	// legacy behavior exactly.
	//
	// ONNX Runtime does not pick a GPU on its own — pointing LibraryPath at a
	// GPU build changes nothing unless the provider is requested here. "cuda"
	// additionally requires the CUDA provider shared libraries next to
	// LibraryPath and a working driver/toolkit; "openvino" requires the
	// OpenVINO plugin library next to LibraryPath (see openvino.go). When any
	// of that is missing, NewEmbedder fails loudly rather than silently
	// falling back to the CPU.
	//
	// "auto" prefers hardware providers over the CPU and degrades gracefully:
	// NewEmbedder attempts the ladder CUDA → OpenVINO and, when every attempt
	// fails (CPU-only ONNX Runtime build, missing driver, missing OpenVINO
	// plugin, invalid device id, ...), logs a WARN naming each failure and
	// continues on the CPU provider instead of failing. Which side won is
	// observable via Embedder.ExecutionProvider. DeviceID participates only
	// in the hardware attempts, never in the CPU fallback.
	//
	// Unknown values are rejected.
	ExecutionProvider string

	// DeviceID is the GPU device index used when ExecutionProvider resolves
	// to a hardware provider. On "cuda" — explicitly or via "auto" — it
	// selects the device reported by the driver. Under an explicit
	// "openvino" request a non-zero value is a hard configuration error (the
	// wrapper exposes no per-device selection for OpenVINO devices), while 0
	// lets OpenVINO pick its best device; under "auto" a non-zero DeviceID
	// fails the OpenVINO rung the same way. Defaults to 0, which is the
	// right choice on single-GPU machines and selects the first device
	// reported by the driver on multi-GPU ones. It is ignored by the CPU
	// provider. Negative values are rejected by NewEmbedder for every
	// provider including cpu, so one config struct validates uniformly.
	DeviceID int

	// Logger for structured logging. If nil, a discard logger is used.
	Logger *slog.Logger

	// Telemetry optionally collects bounded, content-free stage aggregates.
	// Nil disables collection with no background goroutines or exporters.
	Telemetry *Telemetry

	// OnProviderDegrade optionally observes a runtime execution-provider
	// degradation — the embedder stopping inference on one provider and
	// continuing on another after construction. Today that is exactly the
	// post-construction hardware->CPU degradation of an "auto" request: the
	// lazy session-creation fallback and the consecutive-inference-failure
	// degradation (see Embedder.embedInference), both of which call back from
	// degradeToCPU. Two related slides do NOT fire it: the initial provider
	// resolution inside NewEmbedder (an "auto" request whose whole ladder
	// fails before any session exists), and the eager hardware->CPU session
	// fallback, which runs in-constructor before this callback can be wired.
	// Both of those are reported through the WARN log and the
	// requestedExecutionProvider field of the "embedder initialized" entry
	// instead.
	//
	// from and to are provider identifiers ("cuda" -> "cpu"); cause is the
	// error text that triggered the degradation (the same text the WARN log
	// carries). The callback fires synchronously under the embedder's
	// internal write lock: it must be quick and must not call back into the
	// Embedder (including ExecutionProvider). Nil disables notification.
	OnProviderDegrade func(from, to, cause string)
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
	mu            sync.RWMutex
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
	// ExecutionProviderCPU, ExecutionProviderCUDA or
	// ExecutionProviderOpenVINO, never "auto" — the auto request is resolved
	// during NewEmbedder, and ExecutionProvider() reports the winner.
	executionProvider string
	// runner pins every ONNX Runtime call to one OS thread; nil for the CPU
	// provider, which does not need the pinning. See ortRunner.
	//
	// It is an atomic.Pointer, NOT an e.mu-guarded plain field, because Wedged
	// must read it without taking e.mu: EmbedDocuments holds e.mu write-locked
	// across all ONNX inference, so a mutex-synchronized read would park until
	// the inference returns — and a wedged inference never returns, which is
	// exactly the condition Wedged exists to report. Writers (degradeToCPU,
	// Close) already run under e.mu; the atomic store keeps that discipline
	// while letting the lock-free reader observe the pointer instantly.
	runner atomic.Pointer[ortRunner]
	// deviceID and intraOpThreads retain the config NewEmbedder used to build
	// session options, so a later session-creation failure can rebuild them on
	// the CPU provider or on the ladder's next rung (see degradeToCPU and
	// advanceAutoLadderSession).
	deviceID       int
	intraOpThreads int
	// libraryPath retains the config NewEmbedder used, so a later session-
	// creation failure can rebuild options for another hardware ladder rung:
	// the OpenVINO provider locates its plugin beside it, while the CPU and
	// CUDA providers never read it (degradeToCPU passes an empty path).
	libraryPath string
	// fallbackPending is true while inference runs on a hardware provider
	// (CUDA or OpenVINO) that an "auto" request selected. A lazily created
	// session that cannot be built on that provider then degrades the embedder
	// to the CPU provider once and retries (see newSessionWithFallback),
	// mirroring the eager-session fallback in NewEmbedder. The same state
	// gates the runtime degradation of embedInference: consecutive inference
	// failures destroy the live sessions and move the embedder to the CPU
	// provider (see degradeToCPU). degradeToCPU clears it, so both
	// degradations are one-shot per embedder.
	fallbackPending bool
	// inferenceErrStreak counts consecutive failed ONNX inference attempts
	// across all sessions; any successful inference resets it to zero. When
	// the streak reaches inferenceDegradeThreshold on an "auto" request still
	// running on a hardware provider, embedInference degrades the embedder to
	// the CPU provider once and retries the failed inference there. It is
	// always read and written under e.mu.
	inferenceErrStreak int
	// degradeCallback carries EmbedderConfig.OnProviderDegrade; nil when the
	// host did not configure one. Invoked by degradeToCPU under e.mu.
	degradeCallback func(from, to, cause string)
	// newSessionFn, when non-nil, replaces newSession. It exists solely for the
	// tests that pin the lazy session-creation gates (ensureBatchSession,
	// ensureBucketSession, ensureQuerySession) without an ONNX Runtime
	// environment; production embedders never set it.
	newSessionFn func(batchSize, seqLen int) (*onnxSession, error)
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
	runOnORTThread(e.runner.Load(), fn)
}

// autoAttemptLadder is the ordered hardware providers an "auto" request tries
// before the terminal CPU fallback: CUDA first (the measured ~20x winner on
// the reference hardware, ADR-045), then the OpenVINO plugin for Intel
// hardware. Each rung is attempted through buildSessionOptions, which fails
// fast and side-effect-free for a missing driver or plugin — in particular an
// absent OpenVINO plugin is detected from the filesystem before any ORT call,
// so auto neither registers nor enumerates anything when the plugin is not
// installed, and the recorded reason says exactly that. The ladder is walked
// at the session stage too: when the winning rung's options build but its
// session cannot be created, the remaining rungs are attempted before the
// terminal CPU fallback (advanceAutoLadderSession), so a CUDA session failure
// cannot silently skip a working OpenVINO plugin.
var autoAttemptLadder = [...]string{ExecutionProviderCUDA, ExecutionProviderOpenVINO}

// ladderIndex returns the index of provider in autoAttemptLadder, or -1 when
// the provider is not a ladder rung (the CPU fallback).
func ladderIndex(provider string) int {
	for i, rung := range autoAttemptLadder {
		if rung == provider {
			return i
		}
	}
	return -1
}

// advanceAutoLadderSession continues the "auto" ladder at the session stage:
// every autoAttemptLadder rung after index from gets its session options
// built, hands them to install — which must destroy the failed rung's
// options, commit the new ones, and record the rung as the active provider —
// and retries session creation there through createSess. The first rung whose
// session is created wins: the session is returned with ok == true and the
// caller's state already describes that rung. When every remaining rung
// fails, ok == false and the caller falls through to the terminal CPU
// fallback; each failed attempt has been appended to reasons verbatim and
// warned through warn, so no rung is skipped silently. This is the
// session-stage counterpart of the options-stage ladder in NewEmbedder: a
// provider whose options build but whose session cannot be created (wedged
// driver, out-of-device memory at graph allocation, a model the provider
// cannot compile) must still let the remaining rungs try.
func advanceAutoLadderSession(
	from int,
	buildOpts func(string) (*ort.SessionOptions, error),
	install func(rung string, opts *ort.SessionOptions),
	createSess func() (*onnxSession, error),
	warn func(msg, provider, cause string),
	reasons *[]string,
) (sess *onnxSession, ok bool, lastErr error) {
	for _, rung := range autoAttemptLadder[from:] {
		opts, optsErr := buildOpts(rung)
		if optsErr != nil {
			*reasons = append(*reasons, rung+": "+optsErr.Error())
			warn("hardware provider options failed, trying next rung", rung, optsErr.Error())
			continue
		}
		install(rung, opts)
		s, sErr := createSess()
		if sErr == nil {
			return s, true, nil
		}
		lastErr = sErr
		*reasons = append(*reasons, rung+": "+sErr.Error())
		warn("hardware provider session creation failed, trying next rung", rung, sErr.Error())
	}
	return nil, false, lastErr
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

	// Hardware providers keep per-thread state that must not be duplicated
	// across Go's scheduler threads (see ortRunner and
	// providerUsesDedicatedThread). Every ONNX call below — and every one
	// made later through the returned Embedder — is funnelled onto this
	// single thread. The CPU provider needs none of it and keeps calling ONNX
	// inline, exactly as before. "auto" starts on the runner because its
	// ladder begins on a hardware provider; the runner is stopped only when
	// the whole ladder fails and the CPU fallback continues inline.
	effective := requested
	var runner *ortRunner
	if providerUsesDedicatedThread(effective) {
		runner = newORTRunner(logger)
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
	// For "auto" this walks autoAttemptLadder: each failed rung's reason is
	// kept for the fallback WARN, and the first rung whose options build
	// becomes the winner.
	buildOpts := func(provider string) (*ort.SessionOptions, error) {
		var (
			opts *ort.SessionOptions
			err  error
		)
		runOnORTThread(runner, func() {
			opts, err = buildSessionOptions(provider, cfg.LibraryPath, cfg.DeviceID, cfg.IntraOpThreads)
		})
		return opts, err
	}
	var sessOpts *ort.SessionOptions
	// reasons accumulates every failed "auto" ladder attempt — options-stage
	// and session-stage alike — for the terminal fallback WARN. ladderIdx is
	// the index in autoAttemptLadder of the rung whose options won; -1 means
	// the options stage already degraded to the CPU.
	var reasons []string
	ladderIdx := -1
	if effective != ExecutionProviderAuto {
		sessOpts, err = buildOpts(effective)
		if err != nil {
			cleanup(nil)
			return nil, fmt.Errorf("building ONNX session options: %w", err)
		}
	} else {
		for i, candidate := range autoAttemptLadder {
			var attemptErr error
			sessOpts, attemptErr = buildOpts(candidate)
			if attemptErr == nil {
				effective = candidate
				ladderIdx = i
				break
			}
			// The failed rung's own error already names the concrete cause
			// (CPU-only build, missing driver, missing plugin file, ...);
			// keep it verbatim for the fallback WARN.
			reasons = append(reasons, candidate+": "+attemptErr.Error())
		}
		if ladderIdx == -1 {
			// Every hardware rung failed: degrade to the CPU loudly, with
			// each rung's reason in the WARN (ADR-045: a fallback is tracked
			// and warned, never silent). A failed attempt allocated nothing
			// beyond options internals (buildSessionOptions destroys its own
			// partial options on failure), so the fallback just abandons the
			// dedicated thread and proceeds inline like a plain CPU embedder.
			logger.Warn("no hardware execution provider available, falling back to CPU",
				"reasons", strings.Join(reasons, "; "), "deviceID", cfg.DeviceID)
			runner.stop()
			runner = nil
			effective = ExecutionProviderCPU
			sessOpts, err = buildOpts(effective)
			if err != nil {
				cleanup(sessOpts)
				return nil, fmt.Errorf("building ONNX session options: %w", err)
			}
		}
	}
	// From here on sessOpts is owned by the embedder-to-be; cleanup(sessOpts)
	// releases it on failure. `effective` is resolved now — never "auto"
	// again — but the eager-session block below still needs to know whether
	// this was an "auto" request that landed on a hardware provider, so it
	// can retry on the CPU provider when the session itself cannot be built.
	requestedAuto := requested == ExecutionProviderAuto

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
	// replaceOpts swaps the live session options while the OpenVINO device
	// retry rebuilds them per candidate device; the previous handle is released
	// on the dedicated ONNX thread, like every other options mutation.
	replaceOpts := func(opts *ort.SessionOptions) {
		runOnORTThread(runner, func() {
			if sessOpts != nil {
				_ = sessOpts.Destroy()
			}
		})
		sessOpts = opts
	}
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
		if err != nil && effective == ExecutionProviderOpenVINO {
			// The OpenVINO plugin takes the first of the devices it was given;
			// when that one cannot compile this model (an Intel NPU rejects
			// its dynamic shapes) try the plugin's other devices before
			// letting the failure sink the rung. An explicit "openvino"
			// request stays loud when no device can run the model; an "auto"
			// request still reaches its ladder below.
			sess, err = retryOpenVINOSession(buildOpts, replaceOpts, func() (*onnxSession, error) {
				sessionStarted = time.Now()
				return newSess()
			}, func(msg, provider, cause string) {
				logger.Warn(msg, "provider", provider, "error", cause, "deviceID", cfg.DeviceID)
			}, &reasons, err)
			if err == nil {
				logger.Info("OpenVINO session created on a fallback device", "deviceIndex", openVINOAttemptedDevice())
			}
		}
		if err != nil {
			if !requestedAuto {
				cleanup(sessOpts)
				return nil, fmt.Errorf("creating persistent ONNX session: %w", err)
			}
			if ladderIdx < 0 {
				// The options stage already degraded to the CPU and even the
				// CPU session cannot be created — there is nothing left to
				// fall back to.
				cleanup(sessOpts)
				return nil, fmt.Errorf("creating persistent ONNX session: %w", err)
			}
			// auto + hardware options: the provider accepted the options but
			// the session itself could not be built on the device (driver
			// trouble, out of device memory at graph allocation, a model the
			// provider cannot compile). Continue the ladder at the session
			// stage before the terminal CPU fallback — each remaining rung
			// gets its options built and the session retried there, so a
			// CUDA session failure can still land on a working OpenVINO
			// plugin. A failed attempt has already released its own tensors,
			// and a later rung cannot make things worse.
			hardwareSessErr := err
			reasons = append(reasons, effective+": "+err.Error())
			logger.Warn("hardware session creation failed, trying remaining providers",
				"provider", effective, "error", err.Error(), "deviceID", cfg.DeviceID)
			installOpts := func(rung string, opts *ort.SessionOptions) {
				runOnORTThread(runner, func() {
					if sessOpts != nil {
						_ = sessOpts.Destroy()
					}
				})
				sessOpts = opts
				effective = rung
			}
			attemptSess := func() (*onnxSession, error) {
				// Measure each retry from its own start; the failed attempts
				// are not part of a session-create observation.
				sessionStarted = time.Now()
				return newSess()
			}
			warn := func(msg, provider, cause string) {
				logger.Warn(msg, "provider", provider, "error", cause, "deviceID", cfg.DeviceID)
			}
			var advanced bool
			sess, advanced, _ = advanceAutoLadderSession(ladderIdx+1, buildOpts, installOpts, attemptSess, warn, &reasons)
			if !advanced {
				// Every hardware rung failed at the session stage too:
				// degrade to the CPU loudly, with each failed attempt's
				// reason in the WARN (ADR-045: a fallback is tracked and
				// warned, never silent). The abandoned hardware options are
				// released and the dedicated thread retired; the fallback
				// proceeds inline like a plain CPU embedder.
				logger.Warn("no hardware execution provider available, falling back to CPU",
					"reasons", strings.Join(reasons, "; "), "deviceID", cfg.DeviceID)
				cleanup(sessOpts)
				runner = nil
				effective = ExecutionProviderCPU
				sessOpts, err = buildOpts(effective)
				if err != nil {
					cleanup(sessOpts)
					return nil, fmt.Errorf("building ONNX session options: %w", err)
				}
				sess, err = attemptSess()
				if err != nil {
					cleanup(sessOpts)
					return nil, fmt.Errorf("creating persistent ONNX session after hardware->CPU fallback (original error: %w): %w", hardwareSessErr, err)
				}
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

	e := &Embedder{
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
		deviceID:          cfg.DeviceID,
		intraOpThreads:    cfg.IntraOpThreads,
		libraryPath:       cfg.LibraryPath,
		// Only an auto request that is still (or again) on a hardware provider
		// can degrade to the CPU from a later lazy session-creation failure. If
		// the eager-session fallback above already ran, effective is CPU and
		// this stays false.
		fallbackPending: requestedAuto && effective != ExecutionProviderCPU,
		degradeCallback: cfg.OnProviderDegrade,
	}
	// The runner pointer is published atomically (nil for the CPU provider):
	// Wedged reads it without e.mu — see the field comment.
	e.runner.Store(runner)
	return e, nil
}

// ExecutionProvider reports the execution provider the embedder actually runs
// on: ExecutionProviderCPU ("cpu"), ExecutionProviderCUDA ("cuda") or
// ExecutionProviderOpenVINO ("openvino"). The "auto" request is resolved
// inside NewEmbedder, so this never returns "auto"; comparing the return value
// against the requested one is how callers detect a silent hardware->CPU
// slide — including the zero-value embedder, which reports
// "cpu" (the legacy default).
func (e *Embedder) ExecutionProvider() string {
	// The provider is written at runtime by the lazy hardware->CPU fallback
	// (degradeToCPU, under e.mu), so take the read lock: a concurrent getter
	// must not race the writer. Embedder is documented "safe for concurrent
	// use", and the getter is a consumed surface (hosts poll it to detect a
	// silent slide), so the read is synchronized rather than left unlocked.
	e.mu.RLock()
	provider := e.executionProvider
	e.mu.RUnlock()
	if provider == "" {
		return ExecutionProviderCPU
	}
	return provider
}

// Wedged reports whether the embedder's dedicated ONNX inference thread is
// wedged: a job has been running on that thread for at least bound. The
// verdict comes from the worker's own per-job heartbeat (ortRunner), so it is
// authoritative — a caller whose embed call timed out or was cancelled
// cannot tell the difference between "slow inference" and "the thread is
// stuck inside a C call and will never return"; this query can. False on the
// CPU provider (no dedicated thread exists to wedge — an inline call blocks
// only its own goroutine), while idle between jobs, and for a non-positive
// bound. The intended consumer contract (c0wrk's vector-index manager):
// consult Wedged for the honest diagnosis instead of inferring a wedge from
// a streak of consecutive timeouts; keep the caller-side timeout bound
// solely as the backstop that bounds the wait itself.
//
// The runner pointer is read WITHOUT e.mu, deliberately: EmbedDocuments
// holds e.mu write-locked across all ONNX inference, so an RLock here would
// park until the inference returns — forever under the very wedge this
// method diagnoses, turning the authoritative probe into a deadlock (the
// runner pointer is therefore an atomic.Pointer; see the field comment).
// The runner's own mutex guards only the per-job heartbeat stamps, never
// the job body, so wedged below answers even while the worker is stuck
// inside a C call.
func (e *Embedder) Wedged(bound time.Duration) bool {
	r := e.runner.Load()
	if r == nil {
		return false
	}
	return r.wedged(bound)
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

	// The query session can be missing on an open embedder: the runtime
	// hardware->CPU degradation destroys every live session (degradeToCPU) and
	// only the sites that pass a recreate closure rebuild what they need.
	// Re-create it here, before the fast-path gate, so a degradation triggered
	// by a multi-text call cannot silently downgrade every later single-text
	// query to the [BatchSize x MaxSeqLen] batch session below (runBatch
	// computes the full session capacity, ~15-25x the single-text cost).
	// A failed re-create is logged and swallowed: the call then falls through to
	// the batch path exactly as it did before this gate existed, so the attempt
	// can only ever restore throughput, never turn a working query into an
	// error.
	if numTexts == 1 {
		e.ensureQuerySession()
	}

	// Fast path: use the persistent session for single-text embedding.
	// This is the common case when chromem-go calls EmbeddingFunc one text at a time.
	if numTexts == 1 && e.sess != nil {
		e.logger.Debug("running inference (persistent session)", "seqLen", e.maxSeqLen)
		var vec []float32
		runQuery := func() error {
			var runErr error
			e.onORTThread(func() { vec, runErr = e.sess.run(inputIDs, attentionMask, tokenTypeIDs) })
			return runErr
		}
		recreate := func() error {
			sess, sErr := e.newSession(1, e.maxSeqLen)
			if sErr != nil {
				return sErr
			}
			e.sess = sess
			return nil
		}
		if err := e.embedInference(runQuery, recreate); err != nil {
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
		var vecs [][]float32
		// runChunk re-reads e.batchSess on every attempt: the first attempt
		// uses the live session, the post-degradation retry the freshly
		// recreated one (ensureBatchSession below).
		runChunk := func() error {
			var chunkErr error
			e.onORTThread(func() {
				vecs, chunkErr = e.batchSess.runBatch(end-start,
					inputIDs[lo:hi], attentionMask[lo:hi], tokenTypeIDs[lo:hi])
			})
			return chunkErr
		}
		if err := e.embedInference(runChunk, e.ensureBatchSession); err != nil {
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

			if _, err := e.ensureBucketSession(bucket); err != nil {
				return nil, err
			}
			var vecs [][]float32
			// key re-resolves the bucket session on every attempt: the first
			// attempt uses the live session, the post-degradation retry the
			// freshly recreated one (ensureBucketSession below).
			key := sessionKey{batchSize: e.batchSize, seqLen: bucket}
			runRows := func() error {
				sess := e.bucketSessions[key]
				var runErr error
				e.onORTThread(func() {
					vecs, runErr = sess.runBatch(rows, ids, mask, types)
				})
				return runErr
			}
			recreate := func() error {
				_, sErr := e.ensureBucketSession(bucket)
				return sErr
			}
			if rErr := e.embedInference(runRows, recreate); rErr != nil {
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
	if e.newSessionFn != nil {
		return e.newSessionFn(batchSize, seqLen)
	}
	var (
		sess *onnxSession
		err  error
	)
	e.onORTThread(func() {
		sess, err = newONNXSession(e.modelPath, batchSize, seqLen, e.hiddenDim, e.sessOpts, e.telemetry)
	})
	return sess, err
}

// retryOpenVINODeviceSession attempts the OpenVINO plugin's remaining devices
// when a lazily created session could not be built on the one in use — the
// lazy counterpart of the eager-session retry in NewEmbedder, and a no-op for
// every other provider. The caller must hold e.mu.
func (e *Embedder) retryOpenVINODeviceSession(batchSize, seqLen int) (*onnxSession, bool) {
	if e.executionProvider != ExecutionProviderOpenVINO {
		return nil, false
	}
	deviceCount, err := openVINODeviceCount()
	if err != nil || deviceCount <= 1 {
		return nil, false
	}
	buildOpts := func(provider string) (*ort.SessionOptions, error) {
		var (
			opts    *ort.SessionOptions
			optsErr error
		)
		e.onORTThread(func() {
			opts, optsErr = buildSessionOptions(provider, e.libraryPath, e.deviceID, e.intraOpThreads)
		})
		return opts, optsErr
	}
	replaceOpts := func(opts *ort.SessionOptions) {
		e.onORTThread(func() {
			if e.sessOpts != nil {
				_ = e.sessOpts.Destroy()
			}
		})
		e.sessOpts = opts
	}
	warn := func(msg, provider, cause string) {
		e.logger.Warn(msg, "provider", provider, "error", cause, "deviceID", e.deviceID)
	}
	var reasons []string
	sess, ok := retryOpenVINODevices(deviceCount, openVINOAttemptedDevice(), buildOpts, replaceOpts,
		func() (*onnxSession, error) { return e.newSession(batchSize, seqLen) }, warn, &reasons)
	if ok {
		e.logger.Info("OpenVINO session created on a fallback device", "deviceIndex", openVINOAttemptedDevice())
	}
	return sess, ok
}

// newSessionWithFallback builds a persistent ONNX session, honouring the "auto"
// contract for the lazily created sessions (length buckets and the batch
// session), not just the eager one: when the session cannot be built on the
// hardware provider an "auto" request selected, the remaining ladder rungs are
// attempted at the session stage (advanceAutoLadderSession) before the embedder
// degrades to the CPU provider once and retries — the same session-stage ladder
// walk and degrade the eager session performs inside NewEmbedder. An explicit
// "cuda" or "openvino" request still fails loudly (fallbackPending is false
// for it). The caller must hold e.mu.
func (e *Embedder) newSessionWithFallback(batchSize, seqLen int) (*onnxSession, error) {
	sess, err := e.newSession(batchSize, seqLen)
	if err == nil {
		return sess, nil
	}
	if s, ok := e.retryOpenVINODeviceSession(batchSize, seqLen); ok {
		return s, nil
	}
	if !e.canDegradeToCPU() {
		return sess, err
	}
	hardwareProvider := e.executionProvider
	hardwareErr := err
	var reasons []string
	buildOpts := func(provider string) (*ort.SessionOptions, error) {
		var (
			opts    *ort.SessionOptions
			optsErr error
		)
		e.onORTThread(func() {
			opts, optsErr = buildSessionOptions(provider, e.libraryPath, e.deviceID, e.intraOpThreads)
		})
		return opts, optsErr
	}
	install := func(rung string, opts *ort.SessionOptions) {
		e.onORTThread(func() {
			if e.sessOpts != nil {
				_ = e.sessOpts.Destroy()
			}
		})
		e.sessOpts = opts
		e.executionProvider = rung
	}
	warn := func(msg, provider, cause string) {
		e.logger.Warn(msg, "provider", provider, "error", cause, "deviceID", e.deviceID)
	}
	s, ok, lastErr := advanceAutoLadderSession(ladderIndex(hardwareProvider)+1, buildOpts, install,
		func() (*onnxSession, error) { return e.newSession(batchSize, seqLen) }, warn, &reasons)
	if ok {
		return s, nil
	}
	// Every remaining rung failed: degrade to the CPU provider (the terminal
	// fallback). The WARN names the provider actually being degraded from —
	// the last attempted rung — whose own failure the ladder walk above
	// already reported.
	cause := hardwareErr
	if lastErr != nil {
		cause = lastErr
	}
	if fbErr := e.degradeToCPU(cause); fbErr != nil {
		return nil, fmt.Errorf("creating persistent ONNX session on %s failed (%w); CPU fallback also failed: %w", hardwareProvider, hardwareErr, fbErr)
	}
	sess, err = e.newSession(batchSize, seqLen)
	if err != nil {
		return nil, fmt.Errorf("creating persistent ONNX session after %s->CPU fallback (original error: %w): %w", hardwareProvider, hardwareErr, err)
	}
	return sess, nil
}

// canDegradeToCPU reports whether the embedder can safely retry session creation
// on the CPU provider: the request was an "auto" one now running on a hardware
// provider (CUDA or OpenVINO), and no hardware-backed session is live yet. The
// session-stage degradation walks the remaining provider ladder first, and
// every rung rebuilds the session options — replacing options a live session
// was built from (and that degradeToCPU would destroy under it) is only safe
// while no session references them. In practice this is the first session
// creation in length-bucket mode, which is exactly the path the eager fallback
// does not cover. The runtime degradation of embedInference does not go
// through this gate: it destroys the live sessions (inside degradeToCPU)
// before rebuilding anything.
func (e *Embedder) canDegradeToCPU() bool {
	return e.fallbackPending && e.sess == nil && e.batchSess == nil && len(e.bucketSessions) == 0
}

// canDegradeAtRuntime reports whether the runtime hardware->CPU degradation
// of embedInference is available for this embedder: it is open, and the
// request was an "auto" one still running on a hardware provider
// (fallbackPending). An explicit "cuda" or "openvino" request never
// degrades at runtime — its inference failures stay loud errors — and a
// closed embedder has nothing left to degrade. degradeToCPU clears
// fallbackPending, which makes the degradation structurally one-shot.
// The caller must hold e.mu.
func (e *Embedder) canDegradeAtRuntime() bool {
	return !e.closed && e.fallbackPending
}

// embedInference executes one ONNX inference attempt through run and
// implements the runtime provider-degradation contract: consecutive inference
// failures are counted (any success resets the streak), and when the streak
// reaches inferenceDegradeThreshold while the embedder is an "auto" request
// still running on a hardware provider (canDegradeAtRuntime), the embedder
// destroys its live sessions, rebuilds its options on the CPU provider
// (degradeToCPU — WARN log and OnProviderDegrade callback included), lets
// recreate lazily rebuild the session this call needs, and retries the
// inference exactly once on the CPU provider. The retry is structurally
// once: after degradeToCPU the embedder reports "cpu" and fallbackPending is
// cleared, so a later failure can never degrade again.
//
// An eligible call therefore either completes on the CPU provider (nil
// error) or returns ONE loud error chaining the original hardware failure
// with whichever step of the degradation or retry failed. Ineligible calls —
// explicit cuda/openvino requests, an already-CPU embedder, a closed
// embedder, or a streak below the threshold — return the inference error
// unchanged (the site wraps it with its own context, as before).
//
// The caller must hold e.mu, which also keeps the dedicated ONNX thread
// quiescent: the destruction and options rebuild run on it via
// e.onORTThread while nothing else can reach the sessions.
func (e *Embedder) embedInference(run, recreate func() error) error {
	err := run()
	if err == nil {
		e.inferenceErrStreak = 0
		return nil
	}
	e.inferenceErrStreak++
	if e.inferenceErrStreak < inferenceDegradeThreshold || !e.canDegradeAtRuntime() {
		return err
	}
	hardwareProvider := e.executionProvider
	if dErr := e.degradeToCPU(err); dErr != nil {
		return fmt.Errorf("inference on %s failed (%w) and the runtime degradation to CPU failed: %w", hardwareProvider, err, dErr)
	}
	if rErr := recreate(); rErr != nil {
		return fmt.Errorf("inference on %s failed (%w); recreating sessions after the runtime %s->CPU degradation also failed: %w", hardwareProvider, err, hardwareProvider, rErr)
	}
	if rErr := run(); rErr != nil {
		return fmt.Errorf("inference failed on %s (%w) and again on the CPU provider after the runtime degradation: %w", hardwareProvider, err, rErr)
	}
	e.inferenceErrStreak = 0
	return nil
}

// degradeToCPU rebuilds the session options on the CPU provider, destroys
// every live session on the dedicated ONNX thread, stops that thread, and
// records that inference now runs on the CPU. It serves both degradation
// paths: the session-creation fallback (called via newSessionWithFallback
// after canDegradeToCPU verified no session is live — the destruction loop
// below is a no-op there) and the runtime degradation of embedInference,
// where live sessions exist and are destroyed here, on the dedicated thread
// they belong to, right before the thread stops. The caller must hold e.mu
// and must retry session creation (or inference) afterwards.
//
// On failure the embedder is left exactly as it was — still on the hardware
// provider with its options and sessions intact — never in a half-degraded
// state. On success the host's OnProviderDegrade callback (when configured)
// fires with the provider that was degraded from, the CPU provider, and the
// cause, synchronously under e.mu.
func (e *Embedder) degradeToCPU(cause error) error {
	e.logger.Warn("execution provider unavailable, falling back to CPU",
		"provider", e.executionProvider, "error", cause.Error(), "deviceID", e.deviceID)
	var (
		opts *ort.SessionOptions
		err  error
	)
	// Build the CPU options FIRST, before tearing down the hardware handle
	// or any session: if this fails (ort.NewSessionOptions error) the
	// embedder must be left exactly as it was — still on the hardware
	// provider with its options and live sessions intact. The library path
	// is empty because the CPU provider never reads it (no provider lookup
	// happens on the CPU path).
	e.onORTThread(func() {
		opts, err = buildSessionOptions(ExecutionProviderCPU, "", e.deviceID, e.intraOpThreads)
	})
	if err != nil {
		return fmt.Errorf("building ONNX session options: %w", err)
	}
	// The CPU options are in hand; now destroy the hardware state and commit
	// the new state, so every mutation that follows is unconditional. Live
	// sessions are destroyed as one job on the dedicated thread they were
	// created and run on — the same single-job teardown Close uses — because
	// they touch the same provider context the thread pins; after this job
	// the thread stops, so no ONNX call can outlive its session.
	from := e.executionProvider
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
		}
	})
	if r := e.runner.Load(); r != nil {
		r.stop()
		e.runner.Store(nil)
	}
	e.sessOpts = opts
	e.executionProvider = ExecutionProviderCPU
	e.fallbackPending = false
	if e.degradeCallback != nil {
		e.degradeCallback(from, ExecutionProviderCPU, cause.Error())
	}
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

// ensureQuerySession lazily (re-)creates the persistent batchSize=1 query
// session the single-text fast path runs on. NewEmbedder creates it eagerly in
// every non-bucket mode, so a nil e.sess on an open embedder means the runtime
// hardware->CPU degradation destroyed it (degradeToCPU nils every live
// session). Without this gate the fast path would stay disabled for the rest
// of the process and each single-text query would run on the batch session's
// [BatchSize x MaxSeqLen] shape — correct results, but the full-capacity
// compute of a multi-row call.
//
// Session creation goes through newSessionWithFallback for uniformity with the
// other lazy sites and because it is the one place that honours the provider
// state (`sessOpts`/`executionProvider`, already CPU after a degradation). Its
// own session-stage degradation branch cannot engage here: this gate returns
// early while e.sess is live, in every non-bucket mode a pending fallback
// implies a live eager session, and after a real degradation fallbackPending is
// already false. Bucket mode has no query session by design (the constructor
// skips the eager one). The caller must hold e.mu.
//
// A failed re-create is logged at WARN and deliberately not reported to the
// caller, which is why this method returns nothing: the gate is a throughput
// recovery, not a correctness step. The call then falls through to the batch
// path, which serves the text on the batch session exactly as it did before
// this gate existed — so a failing re-create can never turn a query that would
// have succeeded into an error, while a successful one re-enables the fast
// path for the rest of the process. No failure is memoized either: while
// e.sess stays nil each single-text call makes one bounded creation attempt,
// which is what lets a transient failure heal on its own (the host's Wedged
// probe and caller-side timeout remain the backstop for a slow attempt).
func (e *Embedder) ensureQuerySession() {
	if e.sess != nil || e.lengthBuckets {
		return
	}
	e.logger.Info("re-creating persistent ONNX session (no live query session; loading the model takes a few seconds)",
		"seqLen", e.maxSeqLen)
	started := time.Now()
	sess, err := e.newSessionWithFallback(1, e.maxSeqLen)
	if err != nil {
		e.logger.Warn("re-creating the persistent query session failed; single-text embedding falls back to the batch session",
			"seqLen", e.maxSeqLen, "error", err.Error())
		return
	}
	elapsed := time.Since(started)
	e.sess = sess
	e.telemetry.observeSession(elapsed)
	e.logger.Info("persistent ONNX session ready",
		"seqLen", e.maxSeqLen, "elapsed", elapsed.Round(time.Millisecond))
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
	if r := e.runner.Load(); r != nil {
		r.stop()
		e.runner.Store(nil)
	}
	// Mark the embedder closed so EmbedDocuments/EmbedQuery return an error
	// instead of touching the destroyed ONNX environment.
	e.tokenizer = nil
	if destroyErr != nil {
		return fmt.Errorf("destroying ONNX Runtime environment: %w", destroyErr)
	}
	return nil
}
