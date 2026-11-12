# Embedding & Vector Search

The `embedding` package provides ONNX-based text embedding for local semantic search, plus a document chunker. Everything runs in-process — no external API calls are required for embedding. The built-in vector search tool that consumes these embeddings lives in `tools/builtins`, not in `embedding` — see [VectorSearchTool](#vectorsearchtool).

```go
import "github.com/v0lka/sp4rk/embedding"
```

## Embedder

`Embedder` provides ONNX-based text embedding using `jina-embeddings-v2-small-en`. It is safe for concurrent use.

### EmbedderConfig

```go
type EmbedderConfig struct {
    // ModelPath is the path to the ONNX model file (.onnx).
    ModelPath string
    // TokenizerPath is the path to the HuggingFace tokenizer.json file.
    TokenizerPath string
    // LibraryPath is the path to the ONNX Runtime shared library
    // (e.g. libonnxruntime.dylib, libonnxruntime.so, onnxruntime.dll).
    LibraryPath string
    // MaxSeqLength is the maximum token sequence length. Defaults to 512.
    MaxSeqLength int
    // HiddenDim is the embedding dimension of the model. Defaults to 512.
    HiddenDim int
    // BatchSize is the fixed row capacity of the persistent batch ONNX
    // session used when EmbedDocuments receives more than one text.
    // Multi-text batches are processed in chunks of at most BatchSize rows
    // (smaller chunks are zero-padded; padded rows are masked out during
    // pooling). 0 selects DefaultBatchSize (32).
    BatchSize int
    // EnableBatchPipeline overlaps tokenization of fixed-mode batch N+1 with
    // ONNX inference of batch N. Opt-in; has no effect while
    // EnableLengthBuckets is enabled (bucket mode wins, and NewEmbedder
    // logs a warning about that).
    EnableBatchPipeline bool
    // EnableLengthBuckets groups tokenized documents into the smallest
    // fitting fixed sequence length (64, 128, 256, or MaxSeqLength) and
    // lazily creates one persistent ONNX session per used bucket. Opt-in.
    EnableLengthBuckets bool
    // IntraOpThreads limits ONNX Runtime intra-op threads during inference.
    // 0 (the default) preserves legacy behavior: the session is created with a
    // nil *SessionOptions, letting ONNX Runtime choose the thread count. A
    // positive value N constrains intra-op parallelism to N threads, bounding
    // CPU usage in resource-constrained environments. Negative values are
    // treated as 0.
    IntraOpThreads int
    // ExecutionProvider selects the ONNX Runtime execution provider:
    // "cpu" (also the empty-string default), "cuda", "openvino", or "auto"
    // (the ladder CUDA → OpenVINO → CPU with a WARN naming each failure).
    // "cuda" and "openvino" require the provider libraries beside LibraryPath
    // and fail loudly when they are missing; unknown values are rejected.
    // See the initialization sequence below.
    ExecutionProvider string
    // DeviceID is the GPU device index used when ExecutionProvider resolves
    // to a hardware provider: the CUDA device on "cuda"/"auto"; under an
    // explicit "openvino" request a non-zero value is a hard configuration
    // error, while 0 lets the OpenVINO rung resolve the device itself (the
    // plugin's first device, then the remaining ones when it cannot compile
    // the model). Under "auto" a non-zero DeviceID fails the OpenVINO rung
    // the same way — but only as one rung of the ladder, so NewEmbedder logs
    // a WARN naming the failure and continues, degrading to the CPU provider
    // when no hardware rung succeeds, instead of failing.
    // Ignored by the CPU provider; negative values are rejected
    // for every provider.
    DeviceID int
    // Logger for structured logging. If nil, a discard logger is used
    // (slog.New(slog.DiscardHandler)).
    Logger *slog.Logger
    // Telemetry optionally collects bounded, content-free stage aggregates.
    // Nil disables collection.
    Telemetry *Telemetry
    // OnProviderDegrade optionally observes a runtime execution-provider
    // degradation — the embedder stopping inference on one provider and
    // continuing on another after construction (the post-construction
    // hardware→CPU fallbacks of an "auto" request, including the
    // consecutive-inference-failure degradation). from/to are provider
    // identifiers ("cuda" -> "cpu"); cause is the triggering error text.
    // Fires synchronously under the embedder's internal write lock: must be
    // quick and must not call back into the Embedder. The initial provider
    // resolution inside NewEmbedder does not fire it. Nil disables
    // notification.
    OnProviderDegrade func(from, to, cause string)
}
```

### Constants

```go
const (
    DefaultMaxSeqLength = 512 // jina-v2-small supports up to 8192; 512 is practical
    DefaultHiddenDim    = 512 // embedding dimension for jina-embeddings-v2-small-en
    DefaultBatchSize    = 32  // fixed row capacity of the persistent batch session

    ExecutionProviderCPU      = "cpu"      // default; the empty string means "cpu"
    ExecutionProviderCUDA     = "cuda"     // NVIDIA GPU via the CUDA execution provider
    ExecutionProviderOpenVINO = "openvino" // Intel hardware via the OpenVINO plugin
    ExecutionProviderAuto     = "auto"     // ladder CUDA → OpenVINO → CPU with a WARN
)
```

### NewEmbedder

```go
func NewEmbedder(cfg EmbedderConfig) (*Embedder, error)
```

`NewEmbedder` creates a new `Embedder` by loading the tokenizer and initializing the ONNX Runtime environment. `ModelPath`, `TokenizerPath`, and `LibraryPath` are all required. `MaxSeqLength` and `HiddenDim` default to `512` when zero or negative.

The initialization sequence is:

1. `initONNXRuntime(libraryPath)` — sets the shared library path and initializes the global ONNX Runtime environment.
2. `buildSessionOptions(provider, ortLibraryPath, deviceID, intraOpThreads)` — builds `*ort.SessionOptions` that select the execution provider (CPU, CUDA, or OpenVINO; `deviceID` applies to the CUDA provider) and limit intra-op threads when `intraOpThreads` is positive; returns `nil` for the legacy CPU default (session created with nil options). Must run after `initONNXRuntime`.
3. `NewTokenizer(tokenizerPath)` — loads the HuggingFace tokenizer.
4. `newONNXSession(modelPath, batchSize, seqLen, hiddenDim, sessOpts, telemetry)` — creates a persistent ONNX session with pre-allocated tensors for the fast path.

The resulting `sessOpts` is reused for the persistent single-text session, the batch session, and the per-bucket sessions. `EmbedderConfig.ExecutionProvider` / `DeviceID` select the provider: `"cuda"` and `"openvino"` request a specific hardware provider (an `"openvino"` request rejects a non-zero `DeviceID` loudly — `0` lets the rung resolve the device itself). `"auto"` walks the ladder CUDA → OpenVINO; when a rung fails at the options stage — or when the winning provider accepts the *options* but the *session* itself cannot be built on it, the remaining rungs are retried at the session stage first — and only when every attempt has failed, a WARN naming each failure is logged and initialization continues on the CPU (the eager session, and, in length-bucket mode, the first lazily created bucket session). `Embedder.ExecutionProvider()` reports which provider inference actually runs on (`cpu`, `cuda`, or `openvino`). A hardware provider that dies *after* construction is recovered by the runtime degradation — see [Execution provider degradation](#execution-provider-degradation).

Inside the OpenVINO rung the *device* is resolved the same empirical way, because the plugin applies its own priority to an appended device list but then uses only the first device, and the wrapper exposes no hardware-device class to filter with. The first session attempt appends every device the plugin reports; if that session cannot be created (on Intel Core Ultra hosts the first device is the NPU, whose compiler rejects this model's dynamic shapes), the remaining devices are tried one at a time through the same session creation, and the winner is remembered process-wide (`openVINOPreferredDevice`) so later sessions append only it. See `specs/domains/embedding.md` → *OpenVINO device resolution*.

On any failure the session options are cleaned up before returning the error, but the ONNX Runtime **environment is deliberately NOT destroyed**: `initONNXRuntime` is `sync.Once`-guarded per process, so a destroyed environment could never be reinitialized — not for an `"auto"` CPU-fallback retry, and not for a later `NewEmbedder`. Only `Embedder.Close` performs the full teardown. `sessOpts` is released by `Close()`.

### Process-global singleton limitation

The ONNX Runtime is a **process-global singleton** — only one `Embedder` can exist at a time, and it lives for the process lifetime. There is no reference counting. The single owner is responsible for calling `Close()` at shutdown. This is a known limitation for library-reuse scenarios but sufficient for a single-process application.

### Methods

| Method | Description |
| --- | --- |
| `EmbedDocuments(ctx, texts []string) ([][]float32, error)` | Embeds a batch of text documents and returns their embedding vectors. |
| `EmbedQuery(ctx, text string) ([]float32, error)` | Embeds a single text query and returns its embedding vector. |
| `EmbeddingFunc() chromem.EmbeddingFunc` | Returns a chromem-go compatible embedding function for use with `chromem.NewCollection`. |
| `Close() error` | Releases the ONNX Runtime environment and associated resources. |
| `ExecutionProvider() string` | Reports which execution provider inference actually runs on (`cpu`, `cuda`, or `openvino`). |

`EmbedDocuments` uses a **fast path** for single-text embedding: a persistent ONNX session with pre-allocated tensors is reused, eliminating per-call session creation overhead. Creating an ONNX session is expensive (~2s for model loading and graph optimization), while inference is fast (~50ms). For multi-text batches, a persistent batch ONNX session (fixed row capacity `BatchSize`, default 32) is created lazily on the first multi-text call — a one-time ~2s init — and reused for every later call; batches larger than the capacity are processed in zero-padded chunks with a single inference per chunk.

All embeddings are **mean-pooled** (using the attention mask) and **L2-normalized** before being returned.

```go
emb, err := embedding.NewEmbedder(embedding.EmbedderConfig{
    ModelPath:     "/path/to/model.onnx",
    TokenizerPath: "/path/to/tokenizer.json",
    LibraryPath:   "/path/to/libonnxruntime.dylib",
    MaxSeqLength:  512,
    HiddenDim:     512,
    Logger:        logger,
})
if err != nil {
    log.Fatal(err)
}
defer emb.Close()

vec, err := emb.EmbedQuery(context.Background(), "authentication middleware")
```

To bound CPU usage in a resource-constrained environment, set `IntraOpThreads` to a positive value (omitting it or setting `0` keeps the legacy ONNX default):

```go
emb, err := embedding.NewEmbedder(embedding.EmbedderConfig{
    ModelPath:      "/path/to/model.onnx",
    TokenizerPath:  "/path/to/tokenizer.json",
    LibraryPath:    "/path/to/libonnxruntime.dylib",
    IntraOpThreads: 2, // cap ONNX intra-op parallelism at 2 threads
    Logger:         logger,
})
```

## ONNX Runtime

The `onnx.go` file manages the ONNX Runtime lifecycle:

- `initONNXRuntime(libraryPath)` — sets the shared library path via `ort.SetSharedLibraryPath` and initializes the global environment. Must be called once before creating any sessions.
- `destroyONNXRuntime()` — cleans up the global ONNX Runtime environment.
- `newONNXSession(modelPath, batchSize, seqLen, hiddenDim, sessOpts, telemetry)` — builds a reusable `onnxSession` with pre-allocated tensors for a fixed `[batchSize, seqLen]` shape; the session and its tensors are kept alive for reuse across calls (the single-text session uses `batchSize=1`; the batch and bucket sessions use their own capacities).
- `(*onnxSession).run(...)` / `(*onnxSession).runBatch(n, ...)` — run the model for a single row or the first `n` rows of the pre-allocated batch, then mean-pool (with the attention mask) and L2-normalize.

For `jina-embeddings-v2-small-en`:

- **Inputs:** `input_ids`, `attention_mask`, `token_type_ids` (all `int64`, shape `[batch, seq]`).
- **Output:** `last_hidden_state` (`float32`, shape `[batch, seq, hiddenDim]`).
- **Post-processing:** mean pooling with the attention mask, then L2 normalization.

## Execution provider degradation

A hardware-backed execution provider can die *after* a successful initialization: a wedged driver following a driver upgrade, a device lost across a suspend/resume cycle, device memory exhausted by another process. The embedder survives this with a degradation ladder of **three stages**, each covering the failure window the previous one cannot see:

1. **Initialization ladder** — an `"auto"` request walks CUDA → OpenVINO at the session-options stage and again at the session-creation stage (see [NewEmbedder](#newembedder)); when every hardware rung has failed, a WARN naming each reason is logged and initialization continues on the CPU provider.
2. **Session-creation fallback** — after initialization, a persistent session that cannot be created on the hardware provider (for example the first lazy batch or length-bucket session, after the hardware went bad between calls) is retried on the remaining rungs and finally on the CPU provider. This path fires only while **no hardware-backed session is live yet**.
3. **Runtime degradation** — inference itself failing repeatedly on a **live** hardware session moves the embedder to the CPU provider inside the running process. Before this stage existed, such a failure bricked the embedder until the process was restarted: every call kept failing even though the CPU provider remained perfectly usable.

Stages 2 and 3 are one-shot per embedder and funnel through the same `degradeToCPU` choke point.

### Runtime degradation

Every failed ONNX inference attempt bumps a consecutive-failure counter; any successful inference resets it to zero. The counter spans all sessions — the eager single-text session, the batch session, the pipeline and the length buckets. When the counter reaches `inferenceDegradeThreshold` (a compile-time constant, `2`) on an `"auto"` request still running on a hardware provider, the embedder degrades to the CPU provider **once**:

- `degradeToCPU` builds the CPU session options **first** — if that build fails, the embedder is left exactly as it was, still on the hardware provider with its options and sessions intact, never half-degraded. It then destroys every live session on the dedicated ONNX thread (the thread those sessions belong to), stops the thread, and commits the CPU state. A WARN log names the provider, the causing error and the device ID.
- The call that hit the threshold lazily re-creates the session it needs and retries the failed inference **exactly once** on the CPU provider. The retry is structurally once: the degradation clears the pending flag, so a later failure can never trigger a second degradation.
- An eligible call therefore either completes on the CPU provider (a `nil` error) or returns **one loud error** chaining the original hardware failure with whichever degradation or retry step failed. Ineligible calls — explicit `cuda`/`openvino` requests, an already-CPU embedder, a closed embedder, or a streak below the threshold — return the inference error unchanged, exactly as before.

One failure stays a loud error: a transient hiccup must not demote the embedder on its own. The second consecutive failure is the evidence that the provider stopped working mid-flight. An explicit `"cuda"` or `"openvino"` request **never** degrades at runtime — the caller asked for that specific hardware, so its failures keep surfacing as loud errors. After a successful degradation `ExecutionProvider()` reports `cpu`.

### Observing a degradation

Besides the WARN log, hosts can observe the flip programmatically through `EmbedderConfig.OnProviderDegrade`:

```go
cfg.OnProviderDegrade = func(from, to, cause string) {
    logger.Warn("embedding provider degraded", "from", from, "to", to, "cause", cause)
}
```

The callback fires **once per degradation**, synchronously under the embedder's internal write lock, right next to the WARN log. It covers both post-construction degradation paths (the session-creation fallback and the runtime degradation); the initial provider resolution inside `NewEmbedder` does **not** fire it — that resolution is already reported through the WARN log and the `requestedExecutionProvider` field of the "embedder initialized" log entry. `from` and `to` are provider identifiers (`"cuda"` → `"cpu"`) and `cause` is the same error text the WARN log carries. Because the callback runs under the write lock, it **must be quick and must not call back into the `Embedder`** (including `ExecutionProvider`); a nil callback disables notification.

See `specs/domains/embedding.md` → *Execution providers* (the three-stage degradation walk) and `specs/decisions/011-runtime-hardware-cpu-degradation.md` for the full contract, including the rejected alternatives.

## GPU diagnostics

Three helpers answer "what GPUs exist", "is this process actually on the GPU" and "did OpenVINO really run on the Intel GPU" from *outside* the process, rather than trusting ONNX Runtime to self-report which execution provider it picked.

```go
func ListGPUDevices(ctx context.Context) ([]GPUDevice, error)
func GPUInUse(ctx context.Context) (bool, error)
func VerifyOpenVINOExecution(ctx context.Context, emb *Embedder) (bool, error)

type GPUDevice struct {
    Index    int    // driver-assigned ordinal, as used by CUDA_VISIBLE_DEVICES ("0" = primary GPU)
    Name     string // product name as reported by the driver; informational only
    Provider string // ExecutionProviderCUDA or ExecutionProviderOpenVINO
}
```

Every probe takes a caller-supplied `context.Context` first and derives its working context with `context.WithTimeout`, so the effective budget is **min(caller deadline, internal cap)** — an earlier caller deadline always wins, and the cap is the upper bound for callers without one. The cap is 2 s (`gpuProbeTimeout`) for `GPUInUse` and `ListGPUDevices`, and 30 s for `VerifyOpenVINOExecution`. A wedged driver therefore never stalls the caller.

The result semantics are shared, per probe source:

- Probe tool **absent from `PATH`** → a no-error, no-signal result (`false, nil` or an empty list). Absence of a vendor's userspace means "no GPUs from that vendor", not a failure, and that fast path returns without consulting `ctx`.
- Probe tool **present but the invocation fails** (driver down, non-zero exit, timeout, or a cancelled/expired `ctx` once the probe is actually running) → an error.
- Callers **must treat any error as "unverified", never as "not on GPU"**, and must not turn it into a startup failure.

`GPUInUse` runs `nvidia-smi --query-compute-apps=pid --format=csv,noheader`, parses the listed compute-application PIDs and compares them with `os.Getpid()`. An empty list means no process on the machine is using the GPU; a non-empty list without our PID means some *other* process is. A CUDA context is created lazily — on the first inference on the CUDA provider — and lives until the ONNX Runtime environment is destroyed, so the correct moment to probe is *after* at least one real inference, not immediately after session creation. Known limitation: in PID-namespaced environments (containers, WSL) NVML may report host-namespace PIDs that never match the in-namespace `os.Getpid()`.

`ListGPUDevices` enumerates two sources, each following the semantics above independently: NVIDIA GPUs via `nvidia-smi --query-gpu=index,name --format=csv,noheader` (tagged `Provider: "cuda"`) and Intel GPUs via `vulkaninfo --summary` filtered to the Intel PCI vendor (tagged `Provider: "openvino"`) — the hardware the OpenVINO provider runs on. Devices are concatenated NVIDIA-first. **Partial answers win**: the merged list is returned with a `nil` error as long as at least one source produced a device, so one broken driver stack cannot hide the GPUs another still reports; only when *no* source produced a device **and** at least one present source failed does the call fail, with the errors joined — the caller then sees "hardware present, probe failed" instead of a bare "no GPUs". For `cuda` entries, `Index` is the driver ordinal that matches `EmbedderConfig.DeviceID` and `CUDA_VISIBLE_DEVICES`; the ordinal is only meaningful within its `Provider`, so a caller selects a (provider, index) pair, never a bare index. Intel entries always carry `0`, because the plugin-based OpenVINO surface exposes no per-device selection.

`VerifyOpenVINOExecution` is the OpenVINO counterpart of `GPUInUse`: it drives sustained inference on `emb` while sampling `xpu-smi`'s device-level compute-utilization counter, and reports whether any sample reached the 25 % threshold — external proof that the OpenVINO provider really executes on the Intel GPU rather than sliding to a CPU device inside the plugin. At most 12 samples are taken and sampling stops early once the threshold is crossed; the cadence comes from the spawns themselves, so there is deliberately no sleep between samples. A sample reaching the threshold yields `(true, nil)`; sampling concluding without ever reaching it yields `(false, nil)` — inference demonstrably never drove the GPU's compute engine. The verdict is **device-level by nature**: `xpu-smi` exposes no per-process attribution on integrated Intel GPUs, so unlike `GPUInUse` this probe cannot name the owning PID, and callers must present the result as device-level rather than process-level. It is diagnostics-only by contract — it never blocks or fails embedder startup, and callers run it in a background/diagnostic context.

The output parsers are deliberately forgiving: prose placeholders (`No running processes found`, `[N/A]`), CRLF endings and extra comma-separated columns degrade to "no GPUs / no PIDs" instead of an error.

## Telemetry

`Telemetry` aggregates low-cardinality, **content-free** embedding metrics — a snapshot contains no text, no model paths and no caller-provided labels. It is safe for concurrent use, and a nil `*Telemetry` is intentionally a no-op, which is why leaving `EmbedderConfig.Telemetry` unset disables collection with no background goroutines or exporters.

```go
type Telemetry struct { /* ... */ }

type TelemetryStage string // a bounded label for an embedding pipeline stage

const (
    StageTokenization  TelemetryStage = "tokenization"   // tokenizer encoding work
    StageONNXInference TelemetryStage = "onnx_inference" // blocking ONNX session runs
    StageSessionCreate TelemetryStage = "session_create" // successful ONNX session construction
)

type StageMetrics struct {
    Calls    int64
    Duration time.Duration
}

type TelemetrySnapshot struct {
    Stages    map[TelemetryStage]StageMetrics
    Texts     int64
    Tokens    int64
    MinTokens int
    MaxTokens int
    // InferenceCount counts ONNX inference ATTEMPTS, including ones that
    // failed — it always equals Stages[StageONNXInference].Calls.
    InferenceCount int64
    SessionCount   int64
    BatchRows      int64
    BatchCapacity  int64
}

func (t *Telemetry) Snapshot() TelemetrySnapshot // a stable copy of the current aggregates
func (t *Telemetry) Reset()                      // clears all aggregates, keeps the collector instance

func (s TelemetrySnapshot) AverageTokens() float64 // mean non-padding tokens per text
func (s TelemetrySnapshot) BatchFill() float64     // fraction of batch capacity holding real rows
```

`TelemetryStage` is a **bounded** vocabulary — the three constants above are the whole set — so metric cardinality cannot grow with the workload. `AverageTokens()` is `Tokens / Texts`, and `0` when no text has been embedded; `BatchFill()` is `BatchRows / BatchCapacity`, and `0` when nothing has been batched yet.

The stages and counters map directly onto the pipeline features described above:

- `StageTokenization` times the tokenizer encoding step that `EnableBatchPipeline` overlaps with inference, while `StageONNXInference` times the blocking session runs that overlap hides — comparing the two `Duration` totals is how you tell whether the pipeline is actually buying overlap.
- `StageSessionCreate` counts successful ONNX session construction. In fixed mode that is the persistent batch session plus the dedicated `batchSize=1` fast-path session; under `EnableLengthBuckets` a session is created lazily per *used* bucket, so `SessionCount` reveals how many of the 64/128/256/`MaxSeqLength` buckets the corpus actually populated.
- `Texts`/`Tokens` count embedded texts and their **non-padding** tokens (derived from the attention mask), with `MinTokens`/`MaxTokens` tracking the spread. Together with `AverageTokens()` this is the signal for sizing `MaxSeqLength` and for judging whether the length buckets are well placed.
- `BatchFill()` is the direct read on how much of the ONNX batch capacity is padding waste — and therefore on whether a smaller `BatchSize`, or bucket mode, would pay off.

`InferenceCount` counts inference **attempts**, including ones that failed, and always equals `Stages[StageONNXInference].Calls`, so a derived average inference latency (`Duration / InferenceCount`) stays meaningful after transient failures.

## Tokenizer

`Tokenizer` wraps a HuggingFace-compatible WordPiece tokenizer loaded from a `tokenizer.json` file. It produces `input_ids`, `attention_mask`, and `token_type_ids` suitable for BERT-family models like `jina-embeddings-v2-small-en`.

```go
type Tokenizer struct { /* ... */ }

func NewTokenizer(path string) (*Tokenizer, error)
func (t *Tokenizer) Encode(text string, maxLen int) (inputIDs, attentionMask, tokenTypeIDs []int64, err error)
func (t *Tokenizer) EncodeWithLength(text string, maxLen int) (inputIDs, attentionMask, tokenTypeIDs []int64, actualLen int, err error)
func (t *Tokenizer) EncodeBatch(texts []string, maxLen int) (inputIDs, attentionMask, tokenTypeIDs []int64, err error)
func (t *Tokenizer) EncodeBatchWithLengths(texts []string, maxLen int) (inputIDs, attentionMask, tokenTypeIDs []int64, lengths []int, err error)
```

`Encode` tokenizes a single text and returns padded/truncated tensors ready for ONNX inference. `maxLen` controls the maximum sequence length (including the `[CLS]` and `[SEP]` special tokens). `EncodeBatch` tokenizes multiple texts and returns flattened, row-major tensors of shape `[batch_size * maxLen]`.

`EncodeWithLength` is `Encode` plus `actualLen`, the actual non-padding sequence length after truncation; `EncodeBatchWithLengths` is `EncodeBatch` plus one such length per input text. The lengths let a caller select a smaller fixed-shape ONNX session without rescanning the attention mask — which is exactly what the length-bucket mode does. `Encode` and `EncodeBatch` are thin wrappers: they call the `…WithLength`/`…WithLengths` variant and discard the length result. Both variants reject a `maxLen` below 2 with an error rather than panicking on an out-of-range index — the guard lives in `EncodeWithLength`, which the batch path calls per text, so there it surfaces wrapped as `batch encode text N: …`.

## Chunker

The chunker splits file contents into semantically meaningful chunks for embedding. Each chunk carries location metadata so search results can point back to the original file and line range.

### Chunk

```go
type Chunk struct {
    Content   string // the actual text content
    FilePath  string // absolute path to source file
    FileName  string // basename of the file
    StartLine int    // 1-based start line in original file
    EndLine   int    // 1-based end line in original file
    Language  string // detected language/type (e.g. "go", "typescript", "markdown")
}
```

### ChunkerConfig

```go
type ChunkerConfig struct {
    MaxChunkSize int // max characters per chunk (default: 1500)
    Overlap      int // character overlap for fixed splits (default: 200)
}
```

If `Overlap` is greater than or equal to `MaxChunkSize`, it is reduced to `MaxChunkSize / 5`.

### ChunkFile

```go
func ChunkFile(filePath string, content []byte, cfg ChunkerConfig) ([]Chunk, error)
```

`ChunkFile` splits a file's content into chunks using a strategy chosen by file type:

| File type | Strategy |
| --- | --- |
| **Code** (`.go`, `.ts`, `.py`, `.rs`, `.java`, …) | Split by blank lines, then fixed-size if a section is still oversized. |
| **Markdown** (`.md`, `.mdx`) | Split by `## ` (H2) headers, then by blank lines, then fixed-size. |
| **Config** (`.json`, `.yaml`, `.yml`, `.toml`, …) | Split by top-level keys; falls back to fixed-size. |
| **Other** | Fixed-size split with overlap. |

**Binary detection:** files with null bytes in the first 512 bytes are treated as binary and return `nil` (no chunks).

```go
chunks, err := embedding.ChunkFile("/path/to/main.go", content, embedding.ChunkerConfig{
    MaxChunkSize: 1500,
    Overlap:      200,
})
```

### ComputeFileHash

```go
func ComputeFileHash(content []byte) string
```

Returns the SHA-256 hex digest of the content, useful for change detection (re-embed only when a file's hash changes).

## VectorSearchTool

`VectorSearchTool` is a built-in tool (registered under the name `semantic_search`) that searches the project codebase using hybrid (vector + BM25) similarity matching. It finds code by meaning and intent as well as by literal symbol/keyword match. The tool and its supporting types live in the `tools/builtins` package, not `embedding`:

```go
import "github.com/v0lka/sp4rk/tools/builtins"
```

```go
func NewVectorSearchTool(searchFunc VectorSearchFunc, waitFunc VectorSearchWaitFunc) *VectorSearchTool
```

- `searchFunc` performs the actual search (provided by the backend layer at wiring time).
- `waitFunc` blocks until the vector index is ready (the embedder loads asynchronously; searches return empty results until ready).

### Search modes

| Mode | Description |
| --- | --- |
| `hybrid` (default) | Fuses vector and BM25 results via Reciprocal Rank Fusion. Auto-falls-back to `vector` when the lexical index is empty. |
| `vector` | Embedding similarity only. |
| `lexical` | BM25 only. |

### Parameters

| Parameter | Description |
| --- | --- |
| `query` | Natural language description of the code concept, functionality, or pattern. Tokens prefixed with `+` (e.g. `+MatcherFactory`) are treated as must-match substrings. |
| `top_k` | Number of results. Default `10`, max `50`. |
| `file_pattern` | Optional glob to narrow results (e.g. `**/*.go`). |
| `must_match` | Optional list of literal substrings that must all appear in a chunk's content. |
| `mode` | Retrieval strategy: `hybrid`, `vector`, or `lexical`. |

### VectorSearchResult

```go
type VectorSearchResult struct {
    FilePath    string
    FileName    string
    Content     string
    Score       float32  // fused RRF score (hybrid), cosine similarity (vector), or BM25 (lexical)
    StartLine   int
    EndLine     int
    Language    string
    VectorRank  int // 1-based rank from the vector retriever; 0 if not returned
    LexicalRank int // 1-based rank from the lexical retriever; 0 if not returned
}
```

## External Dependencies

The embedding subsystem requires three external assets, fetched separately:

1. **ONNX Runtime shared library** — the platform-specific shared library (`libonnxruntime.dylib` / `.so` / `.dll`). Its path is passed as `EmbedderConfig.LibraryPath`.
2. **Embedding model** — the quantized `jina-embeddings-v2-small-en` ONNX model file. Its path is passed as `EmbedderConfig.ModelPath`.
3. **Tokenizer** — the HuggingFace `tokenizer.json` file. Its path is passed as `EmbedderConfig.TokenizerPath`.

**ONNX Runtime is optional for the framework.** It is required only by the `embedding` subsystem: the `onnxruntime_go` bindings are imported by the `embedding` package alone, so no other package in the module has ONNX Runtime anywhere in its dependency graph. Every other sp4rk package — the executor, planner, tool registry, LLM providers, MCP gateway — builds and runs with no ONNX Runtime present, so the dependency is opt-in.

Nothing is loaded eagerly either. The shared library is opened by `initONNXRuntime(LibraryPath)`, whose only production caller is `NewEmbedder`; the `embedding` package has no `init` that touches ONNX Runtime. A host that never constructs an `Embedder` therefore never needs `LibraryPath`, `ModelPath` or `TokenizerPath`, and never loads the library. The [GPU diagnostics](#gpu-diagnostics), the chunker and the tokenizer are the parts of `embedding` usable without it — `GPUInUse` and `ListGPUDevices` shell out to driver tooling and take no embedder, though `VerifyOpenVINOExecution` does need one to drive inference; `ChunkFile` and `ComputeFileHash` (`embedding/chunker.go`) are pure stdlib; and `NewTokenizer` (`embedding/tokenizer.go`) needs only `sugarme/tokenizer`, never an `Embedder`.

## Complete Example

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/v0lka/sp4rk/embedding"
)

func main() {
	emb, err := embedding.NewEmbedder(embedding.EmbedderConfig{
		ModelPath:     "models/jina-v2-small.onnx",
		TokenizerPath: "models/tokenizer.json",
		LibraryPath:   "lib/libonnxruntime.dylib",
		MaxSeqLength:  embedding.DefaultMaxSeqLength,
		HiddenDim:     embedding.DefaultHiddenDim,
	})
	if err != nil {
		log.Fatal(err)
	}
	defer emb.Close()

	// Chunk a source file.
	content := []byte("package main\n\nfunc main() {}\n")
	chunks, err := embedding.ChunkFile("main.go", content, embedding.ChunkerConfig{
		MaxChunkSize: 1500,
		Overlap:      200,
	})
	if err != nil {
		log.Fatal(err)
	}

	// Embed each chunk.
	ctx := context.Background()
	for _, c := range chunks {
		vec, err := emb.EmbedQuery(ctx, c.Content)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("%s:%d-%d → %d-dim vector\n", c.FileName, c.StartLine, c.EndLine, len(vec))
	}

	// Embed a search query and compare (cosine similarity, since vectors are L2-normalized).
	queryVec, _ := emb.EmbedQuery(ctx, "entry point function")
	_ = queryVec
}
```
