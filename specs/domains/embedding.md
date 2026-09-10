# Embedding

## Purpose

ONNX-based local text embedding for semantic search, plus a document chunker and a built-in vector search tool. Everything runs in-process — no external API calls are required for embedding. Inference runs on the ONNX Runtime CPU provider by default or, when configured, on an NVIDIA GPU via the CUDA execution provider, with an `auto` mode that prefers CUDA and degrades gracefully to CPU. The package provides the embedder, tokenizer, and chunker primitives; the persistent vector index/store itself is a host-application concern (the SDK supplies the `semantic_search` tool that delegates to a host-provided search function).

## Key Files

- `github.com/v0lka/sp4rk/embedding` — `Embedder`, `EmbedderConfig`, `NewEmbedder`, `EmbedDocuments`/`EmbedQuery`/`EmbeddingFunc`/`ExecutionProvider`/`Close`
- `github.com/v0lka/sp4rk/embedding` (runtime) — ONNX Runtime lifecycle (`initONNXRuntime`, `destroyONNXRuntime`, reusable sessions with pre-allocated tensors), execution-provider selection (`ExecutionProviderCPU`/`CUDA`/`Auto`, `buildSessionOptions`), and the single-thread ONNX runner (`ortRunner`)
- `github.com/v0lka/sp4rk/embedding` (GPU probes) — `GPUInUse`, `ListGPUDevices`, `GPUDevice` (bounded `nvidia-smi` diagnostics)
- `github.com/v0lka/sp4rk/embedding` (tokenizer) — `Tokenizer`, `NewTokenizer`, `Encode`/`EncodeBatch`
- `github.com/v0lka/sp4rk/embedding` (chunker) — `Chunk`, `ChunkerConfig`, `ChunkFile`, `ComputeFileHash`
- `github.com/v0lka/sp4rk/tools/builtins` — `VectorSearchTool` (the `semantic_search` tool), `VectorSearchFunc`, `VectorSearchResult`

## Core Types

```go
type EmbedderConfig struct {
    ModelPath         string       // .onnx model file
    TokenizerPath     string       // HuggingFace tokenizer.json
    LibraryPath       string       // ONNX Runtime shared library
    MaxSeqLength      int          // default 512
    HiddenDim         int          // default 512
    BatchSize         int          // default 32; row capacity of the persistent batch session
    IntraOpThreads    int          // default 0 (ONNX Runtime chooses); >0 bounds intra-op threads
    ExecutionProvider string       // "cpu" (default) | "cuda" | "auto"
    DeviceID          int          // GPU index for the CUDA provider; default 0
    Logger            *slog.Logger
}

const (
    ExecutionProviderCPU  = "cpu"  // CPU provider; the runtime's own default
    ExecutionProviderCUDA = "cuda" // NVIDIA GPU; requires a CUDA-enabled ONNX Runtime build
    ExecutionProviderAuto = "auto" // prefer CUDA, fall back to CPU with a WARN
}

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

`NewEmbedder(cfg)` loads the tokenizer and initializes the ONNX Runtime environment. `ModelPath`, `TokenizerPath`, and `LibraryPath` are required; `MaxSeqLength`/`HiddenDim` default to `512`; `BatchSize` defaults to `DefaultBatchSize` (32). The config's `ExecutionProvider` is validated up front: the empty string normalizes to `"cpu"`, and an unknown value is rejected before anything is initialized, so a config typo fails loudly instead of silently degrading to CPU. The init sequence is `initONNXRuntime(libraryPath)` → `buildSessionOptions(provider, cfg.DeviceID, cfg.IntraOpThreads)` → `NewTokenizer(tokenizerPath)` → `newONNXSession(modelPath, 1, maxSeqLen, hiddenDim, sessOpts)`. On any failure the acquired resources are released (session options destroyed, runner thread stopped) and the error returned — the ONNX Runtime **environment is never destroyed** on a failure path: initialization is `sync.Once`-guarded per process, so a destroyed environment could never be reinitialized, neither for a later `NewEmbedder` retry nor for the `auto` CPU fallback. Only `Embedder.Close` performs the environment teardown.

### Execution providers

`ExecutionProvider` selects the ONNX Runtime execution provider. ONNX Runtime never selects a GPU on its own — pointing `LibraryPath` at a GPU build changes nothing unless the provider is requested here.

| Provider | Meaning | On failure |
| -------- | ------- | ---------- |
| `cpu` (default; `""` ≡ `cpu`) | ONNX Runtime CPU provider; session created with `nil` options when `IntraOpThreads ≤ 0` (legacy byte-identical path) | `NewEmbedder` returns the error |
| `cuda` | NVIDIA GPU via the CUDA execution provider. Requires a CUDA-enabled build: `LibraryPath` points at a GPU `libonnxruntime`, with `libonnxruntime_providers_shared` and `libonnxruntime_providers_cuda` beside it (dlopened from that directory), plus a working driver. `DeviceID` selects the GPU (default 0 = first device). `buildSessionOptions` always allocates `*SessionOptions` for CUDA — an explicit `AppendExecutionProviderCUDA` call is mandatory regardless of the thread count | Fails loudly; the error names the failing stage |
| `auto` | Prefer CUDA, degrade gracefully. `NewEmbedder` attempts CUDA (on the dedicated runner thread); when the attempt fails — CPU-only build, missing driver, invalid device id, or a session that cannot be created on the GPU — it logs a WARN and continues on the CPU provider | Falls back to CPU; only an environment-init or CPU-path failure returns an error |

The resolved provider is observable via `Embedder.ExecutionProvider()`, which always reports `cpu` or `cuda` — `auto` is resolved inside `NewEmbedder` and never leaks out. Comparing the returned value against the requested one is how callers detect a silent CUDA→CPU slide. The `embedder initialized` log records the effective `executionProvider`/`deviceID`, plus `requestedExecutionProvider` whenever the request and the winner differ. `DeviceID` participates only in the CUDA attempt; the CPU fallback ignores it.

`IntraOpThreads` bounds ONNX Runtime intra-op parallelism. `0` (or any non-positive value) preserves the legacy behavior for the CPU provider: `buildSessionOptions` returns `nil` and the session is created with a nil `*SessionOptions`, letting ONNX Runtime choose the thread count. A positive value `N` allocates a `*SessionOptions` configured with `SetIntraOpNumThreads(N)`, constraining inference to `N` threads — useful for bounding CPU usage in resource-constrained environments. `buildSessionOptions` runs *after* `initONNXRuntime` because ONNX session-option construction requires the environment. The `Embedder` owns the options handle and destroys it in `Close`.

### Dedicated ONNX thread (GPU)

The CUDA execution provider keeps per-thread state — a cuBLAS handle plus its workspace — for every OS thread that enters `Session.Run`. Go's scheduler migrates goroutines between OS threads, so even serialized Go-level calls arrive on a growing set of threads, and the provider allocates a fresh multi-hundred-MiB context per thread (measured ~1 GiB per distinct calling thread), quickly exhausting device memory. For the CUDA and `auto` providers, `NewEmbedder` therefore starts an `ortRunner`: a goroutine pinned to one locked OS thread, and every ONNX Runtime call the embedder makes — init, session construction, inference, teardown — is funnelled through it. The CPU provider has no such per-thread cost and calls ONNX inline on the caller's goroutine (embedder mutex serialization is what keeps it safe).

`EmbedDocuments` uses a **fast path** for single-text embedding: a persistent ONNX session with pre-allocated tensors is reused (session creation is ~2s; inference ~50ms). Multi-text calls use a second persistent **batch session** with row capacity `BatchSize`, created lazily on the first multi-text call (single-text-only embedders never pay for it); batches are processed in chunks of at most `BatchSize` rows, partial final chunks are zero-padded up to capacity, and padded rows are masked out during pooling. Both sessions receive the same `sessOpts`, so the thread limit and the CUDA provider apply to batch inference too. All embeddings are **mean-pooled** (attention mask) and **L2-normalized**. `EmbeddingFunc()` returns a chromem-go-compatible embedding function.

### Process-global singleton limitation

The ONNX Runtime is a **process-global singleton** — only one `Embedder` can exist at a time, and it lives for the process lifetime. Initialization is enforced by `sync.Once`: the first successful initialization is final and cannot be repeated in the same process, even after `Close`/destroy; the first `LibraryPath` is final. There is no reference counting; the single owner must call `Close()` at shutdown. Sufficient for a single-process application; a known limitation for library-reuse scenarios.

For `jina-embeddings-v2-small-en`: inputs are `input_ids`/`attention_mask`/`token_type_ids` (`int64`, `[batch, seq]`); output is `last_hidden_state` (`float32`, `[batch, seq, hiddenDim]`); post-processing is mean pooling + L2 normalization.

### GPU diagnostics

Two diagnostic helpers answer "what GPUs exist" and "is this process actually on the GPU" without depending on ONNX Runtime self-reporting. Both shell out to `nvidia-smi` bounded by the same 2s budget, so a wedged driver stalls neither. Result semantics mirror each other: a missing `nvidia-smi` yields a no-error empty result (absence of the tool is not an error), while a failed invocation returns an error.

- **`GPUInUse()`** runs `nvidia-smi --query-compute-apps=pid --format=csv,noheader` and reports whether the driver registers *this* process (by PID) as a CUDA compute application. Callers treat an error as "unverified", never as "not on GPU". A CUDA context materializes lazily (first inference on the CUDA provider) and lives until the ONNX Runtime environment is destroyed, so the correct moment to probe is after at least one real inference. In PID-namespaced environments (containers, WSL) NVML may report host-namespace PIDs that never match `os.Getpid()` — a known limitation.
- **`ListGPUDevices()`** runs `nvidia-smi --query-gpu=index,name --format=csv,noheader` and returns one `GPUDevice{Index, Name}` per parseable line; `Index` is the driver-assigned ordinal that matches `EmbedderConfig.DeviceID` and `CUDA_VISIBLE_DEVICES`. Empty or prose-only output yields an empty, non-nil slice.

Both parsers are deliberately forgiving: prose placeholders (`No running processes found`, `[N/A]`), CRLF endings, and extra comma-separated columns degrade to "no GPUs / no PIDs" instead of an error.

## Tokenizer

`Tokenizer` wraps a HuggingFace-compatible WordPiece tokenizer loaded from `tokenizer.json`. `Encode(text, maxLen)`/`EncodeBatch(texts, maxLen)` produce `input_ids`/`attention_mask`/`token_type_ids` suitable for BERT-family models, padded/truncated to `maxLen` (including `[CLS]`/`[SEP]` special tokens). `EncodeBatch` returns flattened row-major tensors of shape `[batch_size * maxLen]`.

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

- Only one `Embedder` exists per process (ONNX Runtime singleton, `sync.Once`-enforced); `Close()` releases the persistent and batch sessions, the session options (if allocated), the ONNX environment, and the runner thread — in that order, all on the ONNX thread when a runner is in use — and marks the embedder closed so later `EmbedDocuments`/`EmbedQuery` calls return an error.
- `NewEmbedder` failure paths release acquired resources and leave the ONNX Runtime environment intact (`sync.Once` makes a destroyed environment unrecoverable); only `Embedder.Close` performs the environment teardown.
- An unknown `ExecutionProvider` value fails `NewEmbedder` before initialization; an explicit `cuda` failure fails loudly; only `auto` may degrade to CPU, always with a WARN log.
- `Embedder.ExecutionProvider()` always reports `cpu` or `cuda` — the `auto` request is resolved during `NewEmbedder` — so callers can always detect a CUDA→CPU slide by comparing against the request.
- Every ONNX Runtime call on the CUDA path (including teardown) runs on the single locked runner thread; the CPU path calls ONNX inline under the embedder mutex.
- A positive `IntraOpThreads` constrains inference in both the persistent (batch=1) and batch sessions, because the same `sessOpts` is passed to both.
- All embeddings are mean-pooled and L2-normalized before being returned.
- `GPUInUse`/`ListGPUDevices` treat a missing `nvidia-smi` as "no signal" (empty result, nil error) and a failed invocation as an error; both complete within the shared 2s probe budget.
- `ChunkFile` returns `nil` for binary files; chunks always carry location metadata (file path + 1-based line range).
- The persistent vector index/store is host-side; the SDK provides only the embedder, tokenizer, chunker, and the search tool that delegates to a host-provided function.
- `semantic_search` returns empty results until the embedder/index is ready (the wait function gates it).

## Configuration

`EmbedderConfig` and `ChunkerConfig` are the configuration surfaces. Defaults: `MaxSeqLength`/`HiddenDim` = `512`; `BatchSize` = `32`; `IntraOpThreads` = `0` (ONNX Runtime chooses the thread count; a positive value bounds intra-op parallelism); `ExecutionProvider` = `""` ≡ `cpu` (valid values: `cpu`, `cuda`, `auto`; anything else fails validation); `DeviceID` = `0` (first GPU; used only by the CUDA attempt). `MaxChunkSize` = `1500`; `Overlap` = `200` (reduced to `MaxChunkSize/5` if it would exceed `MaxChunkSize`). Model/tokenizer/runtime library paths are host-resolved at wiring time; for `cuda`, the CUDA provider shared libraries must sit beside `LibraryPath`. Because the embedder loads asynchronously, the host gates `semantic_search` with a wait function and surfaces readiness separately.

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
