# Embedding & Vector Search

The `embedding` package provides ONNX-based text embedding for local semantic search, plus a document chunker and a built-in vector search tool. Everything runs in-process — no external API calls are required for embedding.

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
    // IntraOpThreads limits ONNX Runtime intra-op threads during inference.
    // 0 (the default) preserves legacy behavior: the session is created with a
    // nil *SessionOptions, letting ONNX Runtime choose the thread count. A
    // positive value N constrains intra-op parallelism to N threads, bounding
    // CPU usage in resource-constrained environments. Negative values are
    // treated as 0.
    IntraOpThreads int
    // Logger for structured logging. If nil, a no-op logger is used.
    Logger *slog.Logger
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
    // the model). Ignored by the CPU provider; negative values are rejected
    // for every provider.
    DeviceID int
    // Telemetry optionally collects bounded, content-free stage aggregates.
    // Nil disables collection.
    Telemetry *Telemetry
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

The resulting `sessOpts` is reused for the persistent single-text session, the batch session, and the per-bucket sessions. `EmbedderConfig.ExecutionProvider` / `DeviceID` select the provider: `"cuda"` and `"openvino"` request a specific hardware provider (an `"openvino"` request rejects a non-zero `DeviceID` loudly — `0` lets the rung resolve the device itself). `"auto"` walks the ladder CUDA → OpenVINO; when a rung fails at the options stage — or when the winning provider accepts the *options* but the *session* itself cannot be built on it, the remaining rungs are retried at the session stage first — and only when every attempt has failed, a WARN naming each failure is logged and initialization continues on the CPU (the eager session, and, in length-bucket mode, the first lazily created bucket session). `Embedder.ExecutionProvider()` reports which provider inference actually runs on (`cpu`, `cuda`, or `openvino`).

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

## Tokenizer

`Tokenizer` wraps a HuggingFace-compatible WordPiece tokenizer loaded from a `tokenizer.json` file. It produces `input_ids`, `attention_mask`, and `token_type_ids` suitable for BERT-family models like `jina-embeddings-v2-small-en`.

```go
type Tokenizer struct { /* ... */ }

func NewTokenizer(path string) (*Tokenizer, error)
func (t *Tokenizer) Encode(text string, maxLen int) (inputIDs, attentionMask, tokenTypeIDs []int64, err error)
func (t *Tokenizer) EncodeBatch(texts []string, maxLen int) (inputIDs, attentionMask, tokenTypeIDs []int64, err error)
```

`Encode` tokenizes a single text and returns padded/truncated tensors ready for ONNX inference. `maxLen` controls the maximum sequence length (including the `[CLS]` and `[SEP]` special tokens). `EncodeBatch` tokenizes multiple texts and returns flattened, row-major tensors of shape `[batch_size * maxLen]`.

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
