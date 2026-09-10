# Embedding

## Purpose

ONNX-based local text embedding for semantic search, plus a document chunker and a built-in vector search tool. Everything runs in-process — no external API calls are required for embedding. Inference runs on the ONNX Runtime CPU provider by default or, when configured, on an NVIDIA GPU via the CUDA execution provider, with an `auto` mode that prefers CUDA and degrades gracefully to CPU. Independent of the provider choice, document batching can additionally be tuned with opt-in fixed-mode pipelining and length buckets. The package provides the embedder, tokenizer, and chunker primitives; the persistent vector index/store itself is a host-application concern (the SDK supplies the `semantic_search` tool that delegates to a host-provided search function).

## Key Files

- `github.com/v0lka/sp4rk/embedding` — `Embedder`, `EmbedderConfig`, `NewEmbedder`, `EmbedDocuments`/`EmbedQuery`/`EmbeddingFunc`/`ExecutionProvider`/`Close`
- `github.com/v0lka/sp4rk/embedding` (runtime) — ONNX Runtime lifecycle (`initONNXRuntime`, `destroyONNXRuntime`, reusable sessions with pre-allocated tensors), execution-provider selection (`ExecutionProviderCPU`/`CUDA`/`Auto`, `buildSessionOptions`), and the single-thread ONNX runner (`ortRunner`); per-shape persistent sessions: the fixed-mode query session (batch 1, eager unless buckets are enabled), the fixed-mode persistent batch session (`[BatchSize, MaxSeqLength]`, lazy), and the opt-in length-bucket sessions (lazy, keyed by `(BatchSize, sequence bucket)`); `onnxSession` teardown is idempotent and safe after close
- `github.com/v0lka/sp4rk/embedding` (pipeline) — `embedFixedPipeline`, the depth-one tokenization/inference overlap for fixed mode (`EnableBatchPipeline`)
- `github.com/v0lka/sp4rk/embedding` (GPU probes) — `GPUInUse`, `ListGPUDevices`, `GPUDevice` (bounded `nvidia-smi` diagnostics)
- `github.com/v0lka/sp4rk/embedding` (tokenizer) — `Tokenizer`, `NewTokenizer`, `Encode`/`EncodeBatch`, length-aware variants
- `github.com/v0lka/sp4rk/embedding` (telemetry) — optional bounded aggregates for tokenization, ONNX inference, sessions, token lengths, and batch fill
- `github.com/v0lka/sp4rk/embedding/onnx_bench_test.go` — fixed/bucketed short/mixed batch matrix plus separate dynamic-shape evaluation
- `github.com/v0lka/sp4rk/embedding/multi_session_bench_test.go` — benchmark-only multi-session grid harness (workers × threads × batch × sequence; no production path)
- `github.com/v0lka/sp4rk/embedding` (chunker) — `Chunk`, `ChunkerConfig`, `ChunkFile`, `ComputeFileHash`
- `github.com/v0lka/sp4rk/tools/builtins` — `VectorSearchTool` (the `semantic_search` tool), `VectorSearchFunc`, `VectorSearchResult`

## Core Types

```go
type EmbedderConfig struct {
    ModelPath            string       // .onnx model file
    TokenizerPath        string       // HuggingFace tokenizer.json
    LibraryPath          string       // ONNX Runtime shared library
    MaxSeqLength         int          // default 512
    HiddenDim            int          // default 512
    BatchSize            int          // default 32; fixed row capacity per persistent session
    EnableBatchPipeline  bool         // default false; overlap fixed batch N+1 tokenization with inference N
    EnableLengthBuckets  bool         // default false; opt-in 64/128/256/512 sequence buckets
    IntraOpThreads       int          // default 0 (ONNX Runtime chooses); >0 bounds intra-op threads
    ExecutionProvider    string       // "cpu" (default) | "cuda" | "auto"
    DeviceID             int          // GPU index for the CUDA provider; default 0
    Logger               *slog.Logger
    Telemetry            *Telemetry   // optional bounded, content-free aggregates
}

const (
    ExecutionProviderCPU  = "cpu"  // CPU provider; the runtime's own default
    ExecutionProviderCUDA = "cuda" // NVIDIA GPU; requires a CUDA-enabled ONNX Runtime build
    ExecutionProviderAuto = "auto" // prefer CUDA, fall back to CPU with a WARN
)

type GPUDevice struct {
    Index int    // driver-assigned ordinal ("0" = primary GPU)
    Name  string // product name, e.g. "NVIDIA GeForce RTX 4070"
}

type Chunk struct {
    Content   string
    FilePath  string // absolute path to source file
    FileName  string // basename
    StartLine int    // 1-based
    EndLine   int    // 1-based
    Language  string // detected type ("go", "typescript", "markdown", …)
}

type ChunkerConfig struct {
    MaxChunkSize int // default 1500
    Overlap      int // default 200 (reduced to MaxChunkSize/5 if >= MaxChunkSize)
}
```

## Flow

```
Indexing (host):
  ChunkFile(path, content, cfg) → []Chunk  (with location metadata)
       │
       ├─ ComputeFileHash(content) → SHA-256 (change detection)
       └─ Embedder.EmbedDocuments(chunks) → [][]float32  (mean-pooled, L2-normalized)
            stored in the host's vector index keyed by file hash

Querying:
  Embedder.EmbedQuery(text) → []float32
       └─ host index performs hybrid (vector + BM25) search
       └─ VectorSearchTool surfaces results to the agent
```

## Embedder

`NewEmbedder(cfg)` loads the tokenizer and initializes the ONNX Runtime environment. `ModelPath`, `TokenizerPath`, and `LibraryPath` are required; `MaxSeqLength`/`HiddenDim` default to `512`; `BatchSize` defaults to `DefaultBatchSize` (32). The config's `ExecutionProvider` is validated up front: the empty string normalizes to `"cpu"`, and an unknown value is rejected before anything is initialized, so a config typo fails loudly instead of silently degrading to CPU.

The init sequence is `initONNXRuntime(libraryPath)` → `buildSessionOptions(provider, cfg.DeviceID, cfg.IntraOpThreads)` → `NewTokenizer(tokenizerPath)` → mode-specific session initialization. On any failure the acquired resources are released (session options destroyed, runner thread stopped) and the error returned — the ONNX Runtime **environment is never destroyed** on a failure path: initialization is `sync.Once`-guarded per process, so a destroyed environment could never be reinitialized, neither for a later `NewEmbedder` retry nor for the `auto` CPU fallback. Only `Embedder.Close` performs the environment teardown.

The fixed mode is the default. It eagerly creates the batch-size-1 query session during `NewEmbedder` and lazily creates one `[BatchSize, MaxSeqLength]` document session on the first multi-text call. When `EnableLengthBuckets` is true, no session is created during `NewEmbedder` (`sess = nil`); all model sessions are lazy and keyed by `(BatchSize, sequence bucket)`, with the supported sequence buckets `64`, `128`, `256`, and `512` capped by `MaxSeqLength`.

### Execution providers

`ExecutionProvider` selects the ONNX Runtime execution provider. ONNX Runtime never selects a GPU on its own — pointing `LibraryPath` at a GPU build changes nothing unless the provider is requested here.

| Provider | Meaning | On failure |
| -------- | ------- | ---------- |
| `cpu` (default; `""` ≡ `cpu`) | ONNX Runtime CPU provider; session created with `nil` options when `IntraOpThreads ≤ 0` (legacy byte-identical path) | `NewEmbedder` returns the error |
| `cuda` | NVIDIA GPU via the CUDA execution provider. Requires a CUDA-enabled build: `LibraryPath` points at a GPU `libonnxruntime`, with `libonnxruntime_providers_shared` and `libonnxruntime_providers_cuda` beside it (dlopened from that directory), plus a working driver. `DeviceID` selects the GPU (default 0 = first device). `buildSessionOptions` always allocates `*SessionOptions` for CUDA — an explicit `AppendExecutionProviderCUDA` call is mandatory regardless of the thread count | Fails loudly; the error names the failing stage |
| `auto` | Prefer CUDA, degrade gracefully. `NewEmbedder` attempts CUDA (on the dedicated runner thread); when the attempt fails — CPU-only build, missing driver, invalid device id, or a session that cannot be created on the GPU — it logs a WARN and continues on the CPU provider | Falls back to CPU; only an environment-init or CPU-path failure returns an error |

The resolved provider is observable via `Embedder.ExecutionProvider()`, which always reports `cpu` or `cuda` — `auto` is resolved inside `NewEmbedder` and never leaks out. Comparing the returned value against the requested one is how callers detect a silent CUDA→CPU slide. The `embedder initialized` log records the effective `executionProvider`/`deviceID`, plus `requestedExecutionProvider` whenever the request and the winner differ. `DeviceID` participates only in the CUDA attempt; the CPU fallback ignores it.

The `auto` fallback happens at **two stages**, and the modes differ in what they cover:

- **Options stage (all modes).** `buildSessionOptions` is the first CUDA attempt — a CPU-only ONNX Runtime build, a missing driver, or an invalid device id all fail there. This fallback applies identically in fixed mode and in bucket mode: in both, the request degrades to CPU options with a WARN and initialization continues inline.
- **Session stage (fixed mode only).** In fixed mode, when the CUDA options were accepted but the eager query session itself could not be built on the GPU, `NewEmbedder` retries once on the CPU provider (options rebuilt, session re-created) before giving up; the final error chains both the original CUDA error and the CPU retry error. In bucket mode there is **no session-stage fallback**: bucket sessions are created lazily at first use, so a GPU session failure surfaces at that first `EmbedDocuments` call as a loud error (`creating persistent ONNX session for length bucket %d`) — the options-stage `auto` degradation is the only CUDA→CPU fallback bucket mode performs.

`IntraOpThreads` bounds ONNX Runtime intra-op parallelism. `0` (or any non-positive value) preserves the legacy behavior for the CPU provider: `buildSessionOptions` returns `nil` and the session is created with a nil `*SessionOptions`, letting ONNX Runtime choose the thread count. A positive value `N` allocates a `*SessionOptions` configured with `SetIntraOpNumThreads(N)`, constraining inference to `N` threads — useful for bounding CPU usage in resource-constrained environments. `buildSessionOptions` runs *after* `initONNXRuntime` because ONNX session-option construction requires the environment. The `Embedder` owns the options handle and destroys it in `Close`.

### Dedicated ONNX thread (GPU)

The CUDA execution provider keeps per-thread state — a cuBLAS handle plus its workspace — for every OS thread that enters `Session.Run`. Go's scheduler migrates goroutines between OS threads, so even serialized Go-level calls arrive on a growing set of threads, and the provider allocates a fresh multi-hundred-MiB context per thread (measured ~1 GiB per distinct calling thread), quickly exhausting device memory. For the CUDA and `auto` providers, `NewEmbedder` therefore starts an `ortRunner`: a goroutine pinned to one locked OS thread, and every ONNX Runtime call the embedder makes — init, session construction (eager, batch, bucket), inference, teardown — is funnelled through it. The CPU provider has no such per-thread cost and calls ONNX inline on the caller's goroutine (embedder mutex serialization is what keeps it safe).

### Embedding modes

In the default fixed mode, `EmbedDocuments` uses a persistent batch-size-1 session for a single text and a lazy persistent `[BatchSize, MaxSeqLength]` session for larger calls. Partial row batches are zero-padded and discarded after pooling.

`EnableBatchPipeline` is an opt-in fixed-mode optimization for calls spanning more than one batch. After the lazy document session is ready, one producer goroutine tokenizes chunk N+1 while the caller goroutine executes ONNX inference for chunk N. The handoff channel is unbuffered: one completed chunk may wait for the consumer, but the producer cannot begin N+2 until N+1 is accepted, so pipeline depth is at most one batch ahead. The caller retains the embedder mutex for the entire pipeline, keeping ONNX execution, session creation, and `Close` serialized. Every exit path cancels a derived context and joins the producer; tokenization error, caller cancellation, or inference error therefore cannot leave a goroutine accessing tokenizer data or delay session teardown. Single-text, single-batch, and length-bucket calls remain serial. The producer is the only tokenizer user while the mutex is held, so no tokenizer-instance pool is needed for this depth-one design.

With `EnableLengthBuckets`, tokenization returns each text's actual post-truncation length. Documents are stably grouped into the smallest fitting sequence bucket (`64`, `128`, `256`, `512`), processed in `BatchSize` chunks, and scattered back into a pre-sized result slice by original index. Every used bucket session is created lazily and reused; unused buckets allocate no ONNX session or tensors. A single embedder mutex serializes tokenization, session creation/inference, and `Close`, preventing duplicate session creation and native-resource destruction during in-flight inference. Cancellation is checked before lock acquisition, after waiting for the lock, and before each inference chunk. `Close` is idempotent, destroys every cached session and tensor once, then destroys options and the process-global environment; later calls fail with `embedder is closed`.

All embeddings are **mean-pooled** (attention mask) and **L2-normalized**. `EmbeddingFunc()` returns a chromem-go-compatible embedding function.

### Process-global singleton limitation

The ONNX Runtime is a **process-global singleton** — only one `Embedder` can exist at a time, and it lives for the process lifetime. Initialization is enforced by `sync.Once`: the first successful initialization is final and cannot be repeated in the same process, even after `Close`/destroy; the first `LibraryPath` is final. There is no reference counting; the single owner must call `Close()` at shutdown. Sufficient for a single-process application; a known limitation for library-reuse scenarios.

For `jina-embeddings-v2-small-en`: inputs are `input_ids`/`attention_mask`/`token_type_ids` (`int64`, `[batch, seq]`); output is `last_hidden_state` (`float32`, `[batch, seq, hiddenDim]`); post-processing is mean pooling + L2 normalization.

### GPU diagnostics

Two diagnostic helpers answer "what GPUs exist" and "is this process actually on the GPU" without depending on ONNX Runtime self-reporting. Both shell out to `nvidia-smi` bounded by the same 2s budget, so a wedged driver stalls neither. Result semantics mirror each other: a missing `nvidia-smi` yields a no-error empty result (absence of the tool is not an error), while a failed invocation returns an error.

- **`GPUInUse()`** runs `nvidia-smi --query-compute-apps=pid --format=csv,noheader` and reports whether the driver registers *this* process (by PID) as a CUDA compute application. Callers treat an error as "unverified", never as "not on GPU". A CUDA context materializes lazily (first inference on the CUDA provider) and lives until the ONNX Runtime environment is destroyed, so the correct moment to probe is after at least one real inference. In PID-namespaced environments (containers, WSL) NVML may report host-namespace PIDs that never match `os.Getpid()` — a known limitation.
- **`ListGPUDevices()`** runs `nvidia-smi --query-gpu=index,name --format=csv,noheader` and returns one `GPUDevice{Index, Name}` per parseable line; `Index` is the driver-assigned ordinal that matches `EmbedderConfig.DeviceID` and `CUDA_VISIBLE_DEVICES`. Empty or prose-only output yields an empty, non-nil slice.

Both parsers are deliberately forgiving: prose placeholders (`No running processes found`, `[N/A]`), CRLF endings, and extra comma-separated columns degrade to "no GPUs / no PIDs" instead of an error.

## Tokenizer

`Tokenizer` wraps a HuggingFace-compatible WordPiece tokenizer loaded from `tokenizer.json`. `Encode(text, maxLen)`/`EncodeBatch(texts, maxLen)` produce `input_ids`/`attention_mask`/`token_type_ids` suitable for BERT-family models, padded/truncated to `maxLen` (including `[CLS]`/`[SEP]` special tokens). `EncodeWithLength`/`EncodeBatchWithLengths` additionally return the actual non-padding lengths after safe truncation; `EncodeBatch` returns flattened row-major tensors of shape `[batch_size * maxLen]`.

The production jina-v2-small WordPiece tokenizer is covered by a concurrent determinism test under Go's race detector. The batch pipeline does not depend on that implementation-specific result: its single producer owns tokenizer access while the embedder mutex is held. This also keeps the design safe for other tokenizer models whose internals may contain mutable caches, without loading or pooling duplicate tokenizer instances.

## Batch-Pipeline Benchmark Decision

`BenchmarkEmbedderBatchPipeline` compares steady-state fixed-mode serial and pipelined execution on identical deterministic short/mixed corpora and batch sizes `8/16/32`. Session creation is excluded from timed iterations. The enable-by-default threshold is a repeatable **5% docs/sec improvement** over the matching serial case; lower differences are treated as benchmark noise and `EnableBatchPipeline` remains false by default.

On Apple M4 Max / Darwin arm64 / Go 1.27.1 / ONNX Runtime 1.28.1, 256 documents, batch 32, three one-iteration repetitions (2026-09-09), median throughput was:

| Corpus | Serial | Pipeline | Delta |
|---|---:|---:|---:|
| short (~20 tokens) | 78.23 docs/s | 78.73 docs/s | +0.64% |
| mixed (19–416 tokens) | 76.70 docs/s | 77.66 docs/s | +1.25% |

Both deltas are below the 5% threshold and the serial ranges overlap pipeline results. `EnableBatchPipeline` therefore remains **false by default**. The opt-in implementation and benchmark remain available for workloads where tokenization is a larger share of wall time.

## Length-Bucket Benchmark Decision

The production default remains fixed-512. The opt-in bucket path and a benchmark-only dynamic-shape path are compared by `onnx_bench_test.go` in isolated processes so peak RSS high-water marks do not leak between cases. On Apple M4 Max / Darwin arm64 / Go 1.27.1 / ONNX Runtime 1.28.1, 256 documents, batch 32, one measured iteration (2026-09-09):

| Corpus | Mode | Wall time | Docs/sec | Peak RSS | RSS delta | Sessions |
|---|---|---:|---:|---:|---:|---:|
| short (~20 tokens) | fixed-512 | 3.208 s | 79.8 | 2795 MiB | 30.3 MiB | 2 |
| short (~20 tokens) | buckets | 0.346 s | 739.7 | 515 MiB | 24.3 MiB | 1 |
| short (~20 tokens) | dynamic | 0.148 s | 1734 | 759 MiB | 192.5 MiB | 1 |
| mixed (19–416 tokens) | fixed-512 | 3.308 s | 77.4 | 3135 MiB | 355 MiB | 2 |
| mixed (19–416 tokens) | buckets | 1.961 s | 130.6 | 3404 MiB | 391 MiB | 2 |
| mixed (19–416 tokens) | dynamic | 2.740 s | 93.4 | 16787 MiB | 7639 MiB | 1 |

Buckets materially reduce both wall time and tensor RSS for short-heavy corpora. Mixed input is faster but retains multiple model sessions (`64` and `512`) and therefore increases RSS, so it does not meet the default-switch condition. Dynamic shape is evaluated separately and is not the base solution: whole-batch output allocation and conversion produce unacceptable mixed-corpus RSS. These figures are a checked-in directional measurement; use multiple isolated repetitions for statistical performance claims.

## Multi-Session Grid Decision

`BenchmarkEmbedderMultiSessionGrid` (in `embedding/multi_session_bench_test.go`) evaluates running several ONNX sessions in parallel — the only unexplored throughput axis after buckets and the batch pipeline. It is a **benchmark-only harness**: the production `Embedder` keeps its single mutex-serialized session path, and no production API or c0wrk wiring exposes multiple sessions.

Harness invariants (covered by `multi_session_pool_test.go` under the race detector):

- Every worker exclusively owns one `onnxSession` plus its own `*ort.SessionOptions`; each owner is closed exactly once by a single `defer` in its worker goroutine. A typed-nil options pointer (intra-op `0`) must never reach `Destroy` — the wrapper stores the concrete `*ort.SessionOptions`, not an interface.
- A bounded dispatcher holds two queues (documents and queries) with strict query preference; a query jumps ahead of already-queued document jobs but never preempts a running inference.
- Memory admission is fail-closed before any session is created: `model bytes + owned tensor bytes + 384 MiB conservative ONNX session overhead` per worker must fit the configured cap, otherwise the case is rejected with `SKIP`.
- Query p95 is measured worker-side (submit → inference complete), independent of when the collector drains results; results are attributed by submission sequence, so delivery order is nondeterministic while the scatter is deterministic and exactly-once.

The grid axes are `workers × intra-op threads × batch × sequence bucket`, driven by `EMBEDDING_GRID_*` env vars and the `GRID_*` knobs of the host `scripts/benchmark-vector-index.sh`; every case runs in an isolated process because peak RSS is a process high-water mark. `threads=0` is the production configuration (ONNX Runtime default thread count). Reported metrics: `docs/sec`, `cpu-cores`/`cpu-util-percent` (user+system CPU over wall time), `query-p95-ms`, `peak-RSS-MiB`.

On Apple M4 Max / Darwin arm64 / Go 1.27.1 / ONNX Runtime 1.28.1, 256 deterministic mixed-corpus documents, one query per pass, `-benchtime=1x -count=3` medians for the production shape (batch 32, sequence 512):

| Config | docs/sec | Δ docs/s | Query p95 | Δ p95 | Peak RSS | CPU util |
|---|---:|---:|---:|---:|---:|---:|
| workers=1, threads=0 (production) | 69.3 | — | 411 ms | — | ~3.3 GiB | 49% |
| workers=2, threads=0 | 93.5 | +35% | 570 ms | **+39% worse** | ~6.4 GiB | 87% |

The full 32-case grid (`workers {1,2} × threads {0,1,2,4} × batch {8,32} × sequence {64,512}`, single iteration) confirms the pattern: no workers>1 configuration improves query p95 at the production sequence length, and every workers=2 case at sequence 512 roughly doubles peak RSS (duplicate model, arena, and thread pools per session). Capping intra-op threads to make room for a second worker forfeits more throughput than the second session recovers (workers=2/threads=2 is *slower* than production at sequence 512). At sequence 64 workers=2 does add +37–47% docs/s with p95 parity — but that regime is already dominated by the opt-in length buckets, which deliver a ~9x throughput win without extra sessions.

**Decision: no-go for production.** Throughput alone (+35%) would clear the 5% threshold used for earlier opt-ins, but the acceptance criteria weigh docs/sec, CPU utilization, query p95, and peak RSS together. Doubling RSS to ~6.4 GiB, degrading query p95 by ~39%, and nearly doubling CPU draw are unacceptable for a desktop application that shares cores with the agent runtime, the UI, and user processes. `workers=1` remains the production path; the grid harness stays checked in for re-evaluation on different hardware (e.g. many-core servers with capacity headroom).

## Chunker

`ChunkFile(filePath, content, cfg)` splits a file's content into semantically meaningful chunks using a strategy chosen by file type:

| File type | Strategy |
| --------- | -------- |
| Code (`.go`, `.ts`, `.py`, `.rs`, `.java`, …) | Split by blank lines, then fixed-size if a section is still oversized. |
| Markdown (`.md`, `.mdx`) | Split by `## ` (H2) headers, then blank lines, then fixed-size. |
| Config (`.json`, `.yaml`, `.yml`, `.toml`, …) | Split by top-level keys; fall back to fixed-size. |
| Other | Fixed-size split with overlap. |

Files with null bytes in the first 512 bytes are treated as binary and return `nil` (no chunks). `ComputeFileHash(content)` returns the SHA-256 hex digest for change detection (re-embed only when a file's hash changes).

## VectorSearchTool

`semantic_search` is a built-in tool (in `tools/builtins`) that searches the codebase using hybrid (vector + BM25) similarity matching. It is constructed with a host-provided search function and an optional wait function (the embedder loads asynchronously; searches return empty results until ready).

```go
func NewVectorSearchTool(searchFunc VectorSearchFunc, waitFunc VectorSearchWaitFunc) *VectorSearchTool
```

### Search modes

| Mode | Description |
| ---- | ----------- |
| `hybrid` (default) | Fuses vector and BM25 results via Reciprocal Rank Fusion; auto-falls-back to `vector` when the lexical index is empty. |
| `vector` | Embedding similarity only. |
| `lexical` | BM25 only. |

Parameters: `query` (natural-language description; tokens prefixed with `+` are must-match substrings), `top_k` (default 10, max 50), `file_pattern` (optional glob), `must_match` (literal substrings that must all appear), `mode`. Results carry file path/name, content, and line range.

## Invariants

- Only one `Embedder` exists per process (ONNX Runtime singleton, `sync.Once`-enforced); idempotent `Close()` releases the persistent query session, the persistent batch session, and every created bucket session exactly once, then the session options (if allocated), then the ONNX environment — that whole sequence runs as a single job on the ONNX thread when a runner is in use, and the runner thread is stopped only after it — and marks the embedder closed so later `EmbedDocuments`/`EmbedQuery` calls fail with `embedder is closed`.
- `NewEmbedder` failure paths release acquired resources and leave the ONNX Runtime environment intact (`sync.Once` makes a destroyed environment unrecoverable); only `Embedder.Close` performs the environment teardown.
- An unknown `ExecutionProvider` value fails `NewEmbedder` before initialization; an explicit `cuda` failure fails loudly; only `auto` may degrade to CPU, always with a WARN log.
- `Embedder.ExecutionProvider()` always reports `cpu` or `cuda` — the `auto` request is resolved during `NewEmbedder` — so callers can always detect a CUDA→CPU slide by comparing against the request.
- The `auto` CUDA→CPU fallback at the `buildSessionOptions` stage applies in every mode (fixed and bucket); the additional session-stage retry-on-CPU exists only in fixed mode. In bucket mode a GPU session failure at first use is a loud error, never a silent CPU retry.
- Every ONNX Runtime call on the CUDA path — including teardown, lazy batch-session creation, bucket-session creation, and bucket/pipeline inference — runs on the single locked runner thread; the CPU path calls ONNX inline under the embedder mutex.
- A positive `IntraOpThreads` constrains inference in every persistent fixed or bucket session, because the same `sessOpts` is passed to each eager and lazy session creation.
- Fixed-512 and serial batching remain the defaults; `EnableBatchPipeline` is explicitly enabled only after repeatable throughput gains exceed the documented 5% noise threshold.
- A running batch pipeline has exactly one tokenizer producer, an unbuffered handoff, and one ONNX consumer; producer cancellation and join complete before the embedder mutex is released or sessions can be destroyed.
- Fixed-512 remains the default; length buckets are explicitly enabled with `EnableLengthBuckets` because mixed-corpus RSS does not improve.
- Multiple parallel ONNX sessions are benchmark-only: each grid worker exclusively owns and closes its session/options once, the dispatcher is bounded with strict query preference, and memory admission fails closed before session creation. The production embedder stays single-session (`workers=1`); adopting parallel sessions requires a new grid decision that improves docs/sec, query p95, CPU utilization, and peak RSS together.
- Bucket assignment uses the smallest fitting length in `64/128/256/512`, never exceeds `MaxSeqLength`, and result vectors retain the caller's original text order.
- Bucket sessions are lazy and persistent: unused buckets create no session, used buckets are reused, and `Close` serializes with in-flight inference.
- All embeddings are mean-pooled and L2-normalized before being returned.
- `GPUInUse`/`ListGPUDevices` treat a missing `nvidia-smi` as "no signal" (empty result, nil error) and a failed invocation as an error; both complete within the shared 2s probe budget.
- Optional telemetry has a fixed three-stage enum and numeric snapshots only; it never records text, model/tokenizer/library paths, or caller-defined labels.

## Configuration

`EmbedderConfig` and `ChunkerConfig` are the configuration surfaces. Defaults: `MaxSeqLength`/`HiddenDim` = `512`; `BatchSize` = `32`; `EnableBatchPipeline` = `false` (serial batching remains default until a repeatable gain exceeds 5%); `EnableLengthBuckets` = `false` (fixed-512 default); `IntraOpThreads` = `0` (ONNX Runtime chooses the thread count; a positive value bounds intra-op parallelism); `ExecutionProvider` = `""` ≡ `cpu` (valid values: `cpu`, `cuda`, `auto`; anything else fails validation); `DeviceID` = `0` (first GPU; used only by the CUDA attempt); `Telemetry` = `nil` (collection disabled, no background goroutines or exporters). `MaxChunkSize` = `1500`; `Overlap` = `200` (reduced to `MaxChunkSize/5` if it would exceed `MaxChunkSize`). Model/tokenizer/runtime library paths are host-resolved at wiring time; for `cuda`, the CUDA provider shared libraries must sit beside `LibraryPath`. Because the embedder loads asynchronously, the host gates `semantic_search` with a wait function and surfaces readiness separately.

Host integration policy: the zero-value configuration is the complete production setup — the reference desktop host constructs the embedder with defaults plus resolved paths and does not surface `EnableBatchPipeline`/`EnableLengthBuckets` (or any multi-session variant) as host configuration knobs. High-risk inference paths stay behind their benchmark gates in the SDK; flipping a default requires a new checked-in benchmark decision, not a host flag.

## Extension Points

- **Custom embedding model**: provide a different ONNX model + tokenizer; adjust `MaxSeqLength`/`HiddenDim` to match. The runtime + chosen `ModelPath`/`LibraryPath` are **final for the entire process lifetime** — `Close()` releases resources but does not allow swapping to a different model in the same process (the underlying `sync.Once` guard is never reset). Choose the custom model before the first `NewEmbedder` in the process.
- **GPU acceleration**: set `ExecutionProvider` to `auto` (safe default when hardware varies) or `cuda` (loud failure when CUDA is expected), pair it with a CUDA-enabled ONNX Runtime build, and pick the device with `DeviceID` (enumerate candidates via `ListGPUDevices`, whose `GPUDevice.Index` matches). Verify a running setup externally with `GPUInUse()` after the first inference.
- **Custom chunking**: implement an alternative splitter producing `[]Chunk` (each with location metadata) and feed chunks to `EmbedDocuments`.
- **Vector index backend**: the host owns the index/store; `EmbeddingFunc()` returns a chromem-go-compatible function for `chromem.NewCollection`, but any store consuming `[][]float32` works.
- **Custom search**: supply a `VectorSearchFunc` to `NewVectorSearchTool` implementing the host's retrieval (hybrid/vector/lexical).

## Related Specs

- [tool-system/builtins.md](tool-system/builtins.md) — `semantic_search` tool catalog entry
- [tool-system/README.md](tool-system/README.md) — tool registration and execution pipeline
- [llm-providers.md](llm-providers.md) — embeddings run fully in-process (no LLM provider required)
