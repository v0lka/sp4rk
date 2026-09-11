package embedding

import (
	"bytes"
	"context"
	"fmt"
	"log/slog"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	ort "github.com/yalue/onnxruntime_go"
)

// testTokenizerPath returns the path to a test tokenizer.json if available.
// Tests that require a real tokenizer file are skipped if it's not present.
func testTokenizerPath(t *testing.T) string {
	t.Helper()
	path := os.Getenv("EMBEDDING_TEST_TOKENIZER_PATH")
	if path == "" {
		t.Skip("EMBEDDING_TEST_TOKENIZER_PATH not set; skipping tokenizer-dependent test")
	}
	return path
}

// testModelPath returns the path to a test ONNX model if available.
func testModelPath(t *testing.T) string {
	t.Helper()
	path := os.Getenv("EMBEDDING_TEST_MODEL_PATH")
	if path == "" {
		t.Skip("EMBEDDING_TEST_MODEL_PATH not set; skipping model-dependent test")
	}
	return path
}

// testLibraryPath returns the path to the ONNX Runtime shared library if available.
func testLibraryPath(t *testing.T) string {
	t.Helper()
	path := os.Getenv("EMBEDDING_TEST_LIBRARY_PATH")
	if path == "" {
		t.Skip("EMBEDDING_TEST_LIBRARY_PATH not set; skipping ONNX-dependent test")
	}
	return path
}

func TestNewTokenizer(t *testing.T) {
	tokPath := testTokenizerPath(t)

	tok, err := NewTokenizer(tokPath)
	if err != nil {
		t.Fatalf("NewTokenizer() error = %v", err)
	}
	if tok == nil {
		t.Fatal("NewTokenizer() returned nil")
	}
}

func TestTokenizer_Encode(t *testing.T) {
	tokPath := testTokenizerPath(t)

	tok, err := NewTokenizer(tokPath)
	if err != nil {
		t.Fatalf("NewTokenizer() error = %v", err)
	}

	ids, mask, typeIDs, err := tok.Encode("hello world", 16)
	if err != nil {
		t.Fatal(err)
	}

	// Should have maxLen elements.
	if len(ids) != 16 {
		t.Errorf("inputIDs length = %d, want 16", len(ids))
	}
	if len(mask) != 16 {
		t.Errorf("attentionMask length = %d, want 16", len(mask))
	}
	if len(typeIDs) != 16 {
		t.Errorf("tokenTypeIDs length = %d, want 16", len(typeIDs))
	}

	// First token should be [CLS] (101) for BERT-style tokenizers.
	if ids[0] != clsTokenID {
		t.Errorf("first token = %d, want %d ([CLS])", ids[0], clsTokenID)
	}

	// Attention mask should be 1 for real tokens, 0 for padding.
	if mask[0] != 1 {
		t.Errorf("first attention mask = %d, want 1", mask[0])
	}

	// Last elements should be padding (0).
	if ids[15] != 0 {
		t.Errorf("last token = %d, want 0 (padding)", ids[15])
	}
	if mask[15] != 0 {
		t.Errorf("last attention mask = %d, want 0 (padding)", mask[15])
	}
}

func TestTokenizer_EncodeBatch(t *testing.T) {
	tokPath := testTokenizerPath(t)

	tok, err := NewTokenizer(tokPath)
	if err != nil {
		t.Fatalf("NewTokenizer() error = %v", err)
	}

	texts := []string{"hello", "world"}
	ids, mask, typeIDs, err := tok.EncodeBatch(texts, 8)
	if err != nil {
		t.Fatalf("EncodeBatch error: %v", err)
	}

	// Should be flattened: 2 * 8 = 16 elements.
	if len(ids) != 16 {
		t.Errorf("batched inputIDs length = %d, want 16", len(ids))
	}
	if len(mask) != 16 {
		t.Errorf("batched attentionMask length = %d, want 16", len(mask))
	}
	if len(typeIDs) != 16 {
		t.Errorf("batched tokenTypeIDs length = %d, want 16", len(typeIDs))
	}
}

func TestMeanPoolAndNormalize(t *testing.T) {
	// Test with a simple 1-sample, 3-token, 2-dim example.
	batchSize := 1
	seqLen := 3
	hiddenDim := 2

	// Hidden states: [[1,2], [3,4], [5,6]]
	hiddenStates := []float32{1, 2, 3, 4, 5, 6}
	// Attention mask: [1, 1, 0] (only first 2 tokens are real)
	attentionMask := []int64{1, 1, 0}

	result := meanPoolAndNormalize(hiddenStates, attentionMask, batchSize, seqLen, hiddenDim)

	if len(result) != 1 {
		t.Fatalf("result length = %d, want 1", len(result))
	}
	if len(result[0]) != 2 {
		t.Fatalf("embedding dim = %d, want 2", len(result[0]))
	}

	// Mean of [1,2] and [3,4] = [2, 3].
	// L2 norm = sqrt(4+9) = sqrt(13).
	// Normalized: [2/sqrt(13), 3/sqrt(13)].
	expectedNorm := math.Sqrt(13)
	expected0 := float32(2.0 / expectedNorm)
	expected1 := float32(3.0 / expectedNorm)

	const tolerance = 1e-6
	if diff := math.Abs(float64(result[0][0] - expected0)); diff > tolerance {
		t.Errorf("result[0][0] = %f, want %f (diff=%e)", result[0][0], expected0, diff)
	}
	if diff := math.Abs(float64(result[0][1] - expected1)); diff > tolerance {
		t.Errorf("result[0][1] = %f, want %f (diff=%e)", result[0][1], expected1, diff)
	}

	// Verify unit norm.
	norm := math.Sqrt(float64(result[0][0])*float64(result[0][0]) + float64(result[0][1])*float64(result[0][1]))
	if diff := math.Abs(norm - 1.0); diff > tolerance {
		t.Errorf("embedding norm = %f, want 1.0", norm)
	}
}

func TestNewEmbedder_ValidationErrors(t *testing.T) {
	tests := []struct {
		name string
		cfg  EmbedderConfig
	}{
		{"missing model path", EmbedderConfig{TokenizerPath: "t.json", LibraryPath: "lib.so"}},
		{"missing tokenizer path", EmbedderConfig{ModelPath: "m.onnx", LibraryPath: "lib.so"}},
		{"missing library path", EmbedderConfig{ModelPath: "m.onnx", TokenizerPath: "t.json"}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewEmbedder(tt.cfg)
			if err == nil {
				t.Error("NewEmbedder() expected error, got nil")
			}
		})
	}
}

func TestEmbedder_EmbedDocuments_EmptyInput(t *testing.T) {
	// EmbedDocuments with empty input should return nil without error,
	// even without a real embedder (we test the early-return path).
	// We can't fully construct an Embedder without real files, so test the logic directly.
	e := &Embedder{
		maxSeqLen: DefaultMaxSeqLength,
		hiddenDim: DefaultHiddenDim,
	}

	result, err := e.EmbedDocuments(context.Background(), nil)
	if err != nil {
		t.Errorf("EmbedDocuments(nil) error = %v", err)
	}
	if result != nil {
		t.Errorf("EmbedDocuments(nil) = %v, want nil", result)
	}

	result, err = e.EmbedDocuments(context.Background(), []string{})
	if err != nil {
		t.Errorf("EmbedDocuments([]) error = %v", err)
	}
	if result != nil {
		t.Errorf("EmbedDocuments([]) = %v, want nil", result)
	}
}

func TestEmbedder_EmbedDocuments_CancelledContext(t *testing.T) {
	e := &Embedder{
		maxSeqLen: DefaultMaxSeqLength,
		hiddenDim: DefaultHiddenDim,
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := e.EmbedDocuments(ctx, []string{"hello"})
	if err == nil {
		t.Error("EmbedDocuments with cancelled context expected error")
	}
}

func TestEmbedder_EmbeddingFunc(t *testing.T) {
	libPath := testLibraryPath(t)
	tokPath := testTokenizerPath(t)
	modelPath := testModelPath(t)

	emb, err := NewEmbedder(EmbedderConfig{
		ModelPath:     modelPath,
		TokenizerPath: tokPath,
		LibraryPath:   libPath,
	})
	if err != nil {
		t.Fatalf("NewEmbedder() error = %v", err)
	}
	// Close the session only; the process-global ONNX environment is owned by
	// TestMain and shared across all tests, so Embedder.Close (which destroys
	// it) must not be used here.
	defer closeSessionOnly(emb)

	fn := emb.EmbeddingFunc()
	if fn == nil {
		t.Fatal("EmbeddingFunc() returned nil")
	}

	vec, err := fn(context.Background(), "test embedding")
	if err != nil {
		t.Fatalf("EmbeddingFunc()() error = %v", err)
	}
	if len(vec) != DefaultHiddenDim {
		t.Errorf("embedding dim = %d, want %d", len(vec), DefaultHiddenDim)
	}
}

func TestEmbedder_EndToEnd(t *testing.T) {
	libPath := testLibraryPath(t)
	tokPath := testTokenizerPath(t)
	modelPath := testModelPath(t)

	emb, err := NewEmbedder(EmbedderConfig{
		ModelPath:     modelPath,
		TokenizerPath: tokPath,
		LibraryPath:   libPath,
		MaxSeqLength:  128,
	})
	if err != nil {
		t.Fatalf("NewEmbedder() error = %v", err)
	}
	// See TestEmbedder_EmbeddingFunc: close the session without destroying the
	// shared ONNX environment owned by TestMain.
	defer closeSessionOnly(emb)

	ctx := context.Background()

	// Single query embedding.
	vec, err := emb.EmbedQuery(ctx, "The quick brown fox jumps over the lazy dog")
	if err != nil {
		t.Fatalf("EmbedQuery() error = %v", err)
	}
	if len(vec) != DefaultHiddenDim {
		t.Errorf("EmbedQuery() dim = %d, want %d", len(vec), DefaultHiddenDim)
	}

	// Verify unit norm.
	var norm float64
	for _, v := range vec {
		norm += float64(v) * float64(v)
	}
	norm = math.Sqrt(norm)
	if diff := math.Abs(norm - 1.0); diff > 1e-5 {
		t.Errorf("embedding norm = %f, want 1.0", norm)
	}

	// Batch embedding.
	vecs, err := emb.EmbedDocuments(ctx, []string{"hello", "world"})
	if err != nil {
		t.Fatalf("EmbedDocuments() error = %v", err)
	}
	if len(vecs) != 2 {
		t.Errorf("EmbedDocuments() count = %d, want 2", len(vecs))
	}
	for i, v := range vecs {
		if len(v) != DefaultHiddenDim {
			t.Errorf("vecs[%d] dim = %d, want %d", i, len(v), DefaultHiddenDim)
		}
	}
}

// --- Tokenizer edge cases (no tokenizer file needed) ---

func TestTokenizer_Encode_MaxLenZero(t *testing.T) {
	// maxLen=0 triggers the error guard (maxLen < 2) before
	// accessing the inner tokenizer, so nil inner is safe.
	tok := &Tokenizer{inner: nil}
	_, _, _, err := tok.Encode("hello world", 0)
	if err == nil {
		t.Error("expected error for maxLen=0")
	}
}

func TestTokenizer_Encode_MaxLenOne(t *testing.T) {
	tok := &Tokenizer{inner: nil}
	_, _, _, err := tok.Encode("hello world", 1)
	if err == nil {
		t.Error("expected error for maxLen=1")
	}
}

func TestTokenizer_EncodeBatch_EmptyTexts(t *testing.T) {
	tok := &Tokenizer{inner: nil}
	ids, mask, typeIDs, _ := tok.EncodeBatch([]string{}, 16)

	if len(ids) != 0 {
		t.Errorf("inputIDs length = %d, want 0 (empty texts should produce empty batch)", len(ids))
	}
	if len(mask) != 0 {
		t.Errorf("attentionMask length = %d, want 0", len(mask))
	}
	if len(typeIDs) != 0 {
		t.Errorf("tokenTypeIDs length = %d, want 0", len(typeIDs))
	}
}

func TestTokenizer_EncodeBatch_EmptyTexts_MaxLenZero(t *testing.T) {
	tok := &Tokenizer{inner: nil}
	ids, mask, typeIDs, _ := tok.EncodeBatch([]string{}, 0)

	if len(ids) != 0 {
		t.Errorf("inputIDs length = %d, want 0", len(ids))
	}
	if len(mask) != 0 {
		t.Errorf("attentionMask length = %d, want 0", len(mask))
	}
	if len(typeIDs) != 0 {
		t.Errorf("tokenTypeIDs length = %d, want 0", len(typeIDs))
	}
}

// --- buildSessionOptions ---

func TestBuildSessionOptions_Zero(t *testing.T) {
	// intraOpThreads == 0 on the CPU provider must return (nil, nil) without
	// touching the ONNX runtime. This is the legacy / default code path and
	// must not require the shared library to be initialized. The empty
	// provider string is the zero value and a synonym for "cpu".
	for _, provider := range []string{"", ExecutionProviderCPU} {
		opts, err := buildSessionOptions(provider, 0, 0)
		if err != nil {
			t.Fatalf("buildSessionOptions(%q, 0, 0) error = %v, want nil", provider, err)
		}
		if opts != nil {
			t.Errorf("buildSessionOptions(%q, 0, 0) opts = %p, want nil", provider, opts)
		}
	}
}

func TestBuildSessionOptions_Negative(t *testing.T) {
	// Negative values are treated like 0: return (nil, nil).
	for _, n := range []int{-1, -42} {
		opts, err := buildSessionOptions("", 0, n)
		if err != nil {
			t.Fatalf("buildSessionOptions(\"\", 0, %d) error = %v, want nil", n, err)
		}
		if opts != nil {
			t.Errorf("buildSessionOptions(\"\", 0, %d) opts = %p, want nil", n, opts)
		}
	}
}

func TestBuildSessionOptions_UnknownProvider(t *testing.T) {
	// An unrecognized provider is rejected rather than silently downgraded to
	// the CPU — a typo in the caller's configuration must not look like a
	// working GPU setup. "auto" is likewise refused here: it is a NewEmbedder
	// directive, not a constructible provider. Rejection happens before any
	// ONNX call, so this test needs no shared library.
	for _, provider := range []string{"CUDA", "gpu", "cuda11", "coreml", ExecutionProviderAuto} {
		opts, err := buildSessionOptions(provider, 0, 0)
		if err == nil {
			t.Errorf("buildSessionOptions(%q, 0, 0) error = nil, want non-nil", provider)
		}
		if opts != nil {
			t.Errorf("buildSessionOptions(%q, 0, 0) opts = %p, want nil", provider, opts)
		}
	}
}

// --- Execution provider: normalization, auto fallback, effective getter ---

// processCreatedCUDAContext records whether any test in this process has
// driven a real CUDA inference. A CUDA context, once created, lives until the
// ONNX Runtime environment is destroyed (end of the test binary), so
// afterwards the driver legitimately lists our PID among its compute apps.
// Tests that assert "own PID absent" must consult this flag first — the
// shared-process suite cannot assume CUDA was never touched.
var processCreatedCUDAContext atomic.Bool

// providerLogs is a slog sink for assertions about WARN/INFO lines emitted by
// NewEmbedder. Only WARN and above are captured: the fallback reason is a
// WARN, and INFO-level noise (initializing, loading tokenizer, ...) would
// only bloat the buffer.
type providerLogs struct {
	buf bytes.Buffer
}

func newProviderLogs() *providerLogs {
	return &providerLogs{}
}

func (p *providerLogs) logger() *slog.Logger {
	return slog.New(slog.NewTextHandler(&p.buf, &slog.HandlerOptions{Level: slog.LevelWarn}))
}

func (p *providerLogs) contains(substr string) bool {
	return strings.Contains(p.buf.String(), substr)
}

// TestNormalizeExecutionProvider covers every accepted spelling plus the
// loud-rejection requirement: a typo must produce an error, never a silent
// CPU degradation.
func TestNormalizeExecutionProvider(t *testing.T) {
	for _, provider := range []string{"", ExecutionProviderCPU, ExecutionProviderCUDA, ExecutionProviderAuto} {
		got, err := normalizeExecutionProvider(provider)
		if err != nil {
			t.Errorf("normalizeExecutionProvider(%q) error = %v, want nil", provider, err)
			continue
		}
		want := provider
		if want == "" {
			want = ExecutionProviderCPU
		}
		if got != want {
			t.Errorf("normalizeExecutionProvider(%q) = %q, want %q", provider, got, want)
		}
	}
	for _, provider := range []string{"CUDA", "gpu", "cuda11", "coreml", "Auto"} {
		if _, err := normalizeExecutionProvider(provider); err == nil {
			t.Errorf("normalizeExecutionProvider(%q) error = nil, want non-nil", provider)
		}
	}
}

// TestNewEmbedder_UnknownExecutionProvider verifies the loud rejection of an
// unknown provider value end to end: NewEmbedder must fail with an error that
// names the unknown provider, without touching the ONNX env lifecycle.
func TestNewEmbedder_UnknownExecutionProvider(t *testing.T) {
	destroys := onnxEnvDestroys.Load()
	for _, provider := range []string{"CUDA", "gpu", "tensorrt", "AUTO"} {
		_, err := NewEmbedder(EmbedderConfig{
			ModelPath:         "m.onnx",
			TokenizerPath:     "t.json",
			LibraryPath:       "lib.so",
			ExecutionProvider: provider,
		})
		if err == nil {
			t.Errorf("NewEmbedder(provider=%q) error = nil, want non-nil", provider)
			continue
		}
		if !strings.Contains(err.Error(), "unknown execution provider") {
			t.Errorf("NewEmbedder(provider=%q) error = %v, want it to name the unknown provider", provider, err)
		}
	}
	if got := onnxEnvDestroys.Load() - destroys; got != 0 {
		t.Errorf("destroyONNXRuntime called %d times during rejected configs, want 0", got)
	}
}

// TestNewEmbedder_FailurePaths_NeverDestroyEnv asserts the core safety
// invariant behind the auto fallback: no NewEmbedder failure path may call
// destroyONNXRuntime. initONNXRuntime is sync.Once-guarded per process, so a
// destroyed environment could never be reinitialized — not for a later
// NewEmbedder retry, and not for the CPU fallback itself. Only Embedder.Close
// tears the environment down, exactly once, as the process's single owner.
//
// The cases below drive every early-return error path reachable in one
// process: path validation failures (before ONNX is touched), an unknown
// provider (validation), a tokenizer load failure (after env init and options
// build — the deepest cleanup(sessOpts) path), and an explicit "cuda" that
// fails (the loudest historical failure).
func TestNewEmbedder_FailurePaths_NeverDestroyEnv(t *testing.T) {
	destroys := onnxEnvDestroys.Load()

	// Path validation failures (validated before ONNX is touched).
	for _, cfg := range []EmbedderConfig{
		{TokenizerPath: "t.json", LibraryPath: "lib.so"},
		{ModelPath: "m.onnx", LibraryPath: "lib.so"},
		{ModelPath: "m.onnx", TokenizerPath: "t.json"},
	} {
		if _, err := NewEmbedder(cfg); err == nil {
			t.Errorf("NewEmbedder(%+v) expected error, got nil", cfg)
		}
	}

	// Unknown provider.
	if _, err := NewEmbedder(EmbedderConfig{
		ModelPath: "m.onnx", TokenizerPath: "t.json", LibraryPath: "lib.so",
		ExecutionProvider: "gpu",
	}); err == nil {
		t.Error("NewEmbedder(unknown provider) expected error, got nil")
	}

	if got := onnxEnvDestroys.Load() - destroys; got != 0 {
		t.Errorf("destroyONNXRuntime called %d times across early failure paths, want 0", got)
	}

	// Tokenizer load failure — after env init, options build: exercises
	// cleanup(sessOpts) with the process env still shared.
	libPath := testLibraryPath(t)
	if _, err := NewEmbedder(EmbedderConfig{
		ModelPath:     "m.onnx",
		TokenizerPath: filepath.Join(t.TempDir(), "nonexistent-tokenizer.json"),
		LibraryPath:   libPath,
	}); err == nil {
		t.Error("NewEmbedder(missing tokenizer) expected error, got nil")
	}
	if got := onnxEnvDestroys.Load() - destroys; got != 0 {
		t.Errorf("destroyONNXRuntime called %d times after tokenizer failure, want 0", got)
	}

	// Explicit "cuda" with a CUDA failure must fail loudly (not fall back),
	// and still must not destroy the env on its way out.
	if librarySupportsCUDA(t) {
		if _, err := NewEmbedder(EmbedderConfig{
			ModelPath:         testModelPath(t),
			TokenizerPath:     testTokenizerPath(t),
			LibraryPath:       libPath,
			ExecutionProvider: ExecutionProviderCUDA,
			DeviceID:          999, // rejected by the CUDA provider: no such device
		}); err == nil {
			t.Error("NewEmbedder(explicit cuda, invalid device) expected error, got nil")
		}
		if got := onnxEnvDestroys.Load() - destroys; got != 0 {
			t.Errorf("destroyONNXRuntime called %d times after explicit-cuda failure, want 0", got)
		}
	}
}

// TestNewEmbedder_ExecutionProviderAuto_CPUOnlyBuild covers "auto" on a
// CPU-only ONNX Runtime build: the CUDA attempt fails at provider-options
// creation, a WARN naming the reason is logged, and the caller still receives
// a fully working CPU embedder whose ExecutionProvider() reports "cpu".
//
// Which library the process loaded is decided once by TestMain, so this test
// is driven by probing the loaded environment: it runs (rather than skips)
// exactly when the initialized library is a CPU-only build.
func TestNewEmbedder_ExecutionProviderAuto_CPUOnlyBuild(t *testing.T) {
	libPath := testLibraryPath(t)
	tokPath := testTokenizerPath(t)
	modelPath := testModelPath(t)
	if librarySupportsCUDA(t) {
		t.Skip("ONNX Runtime library is a CUDA build; run with a CPU-only EMBEDDING_TEST_LIBRARY_PATH to exercise the CPU fallback")
	}

	logs := newProviderLogs()
	destroys := onnxEnvDestroys.Load()
	emb, err := NewEmbedder(EmbedderConfig{
		ModelPath:         modelPath,
		TokenizerPath:     tokPath,
		LibraryPath:       libPath,
		ExecutionProvider: ExecutionProviderAuto,
		Logger:            logs.logger(),
	})
	if err != nil {
		t.Fatalf("auto on CPU-only build: NewEmbedder() error = %v, want working CPU fallback", err)
	}
	t.Cleanup(func() { closeSessionOnly(emb) })

	if got := emb.ExecutionProvider(); got != ExecutionProviderCPU {
		t.Errorf("ExecutionProvider() = %q, want %q (CPU fallback after failed CUDA attempt)", got, ExecutionProviderCPU)
	}
	if !logs.contains("falling back to CPU") {
		t.Errorf("missing WARN about the CUDA->CPU fallback in logs:\n%s", logs.buf.String())
	}
	if got := onnxEnvDestroys.Load() - destroys; got != 0 {
		t.Errorf("destroyONNXRuntime called %d times during auto fallback, want 0", got)
	}

	// The fallback embedder must be fully functional, not half-initialized.
	vec, err := emb.EmbedQuery(context.Background(), "auto fallback still embeds")
	if err != nil {
		t.Fatalf("EmbedQuery() error = %v", err)
	}
	if len(vec) != DefaultHiddenDim {
		t.Errorf("embedding dim = %d, want %d", len(vec), DefaultHiddenDim)
	}
}

// TestNewEmbedder_ExecutionProviderAuto_InvalidDevice covers "auto" on a CUDA
// build when CUDA init fails for a non-build reason — here an invalid device
// id, rejected by the CUDA provider while parsing provider options. The
// embedder must still come up on CPU with a WARN.
func TestNewEmbedder_ExecutionProviderAuto_InvalidDevice(t *testing.T) {
	libPath := testLibraryPath(t)
	tokPath := testTokenizerPath(t)
	modelPath := testModelPath(t)
	if !librarySupportsCUDA(t) {
		t.Skip("ONNX Runtime library is not a CUDA build; skipping auto-with-failed-CUDA test")
	}

	logs := newProviderLogs()
	destroys := onnxEnvDestroys.Load()
	emb, err := NewEmbedder(EmbedderConfig{
		ModelPath:         modelPath,
		TokenizerPath:     tokPath,
		LibraryPath:       libPath,
		ExecutionProvider: ExecutionProviderAuto,
		DeviceID:          999, // rejected by the CUDA provider: no such device
		Logger:            logs.logger(),
	})
	if err != nil {
		t.Fatalf("auto with invalid device: NewEmbedder() error = %v, want working CPU fallback", err)
	}
	t.Cleanup(func() { closeSessionOnly(emb) })

	if got := emb.ExecutionProvider(); got != ExecutionProviderCPU {
		t.Errorf("ExecutionProvider() = %q, want %q", got, ExecutionProviderCPU)
	}
	if !logs.contains("falling back to CPU") {
		t.Errorf("missing WARN about the CUDA->CPU fallback in logs:\n%s", logs.buf.String())
	}
	if got := onnxEnvDestroys.Load() - destroys; got != 0 {
		t.Errorf("destroyONNXRuntime called %d times during auto fallback, want 0", got)
	}

	if _, err := emb.EmbedQuery(context.Background(), "cpu fallback embeds"); err != nil {
		t.Fatalf("EmbedQuery() error = %v", err)
	}
}

// TestNewEmbedder_ExecutionProviderCUDA_InvalidDeviceFails covers the
// complement of the auto fallback: an explicit "cuda" under the same CUDA
// failure must return an error, not a silently degraded embedder.
func TestNewEmbedder_ExecutionProviderCUDA_InvalidDeviceFails(t *testing.T) {
	libPath := testLibraryPath(t)
	tokPath := testTokenizerPath(t)
	modelPath := testModelPath(t)
	if !librarySupportsCUDA(t) {
		t.Skip("ONNX Runtime library is not a CUDA build; skipping explicit-cuda failure test")
	}

	_, err := NewEmbedder(EmbedderConfig{
		ModelPath:         modelPath,
		TokenizerPath:     tokPath,
		LibraryPath:       libPath,
		ExecutionProvider: ExecutionProviderCUDA,
		DeviceID:          999,
		Logger:            slog.New(slog.DiscardHandler),
	})
	if err == nil {
		t.Fatal("NewEmbedder(explicit cuda, DeviceID=999) error = nil, want non-nil (loud failure)")
	}
	if !strings.Contains(err.Error(), "building ONNX session options") {
		t.Errorf("error should originate from session-options build, got: %v", err)
	}
}

// TestNewEmbedder_ExecutionProviderAuto_GPU asserts the happy path on a real
// GPU machine: "auto" resolves to CUDA and — the core requirement — the test
// process's PID shows up in nvidia-smi's compute-apps list after a real
// inference, so a silent CPU slide cannot hide behind a green test.
func TestNewEmbedder_ExecutionProviderAuto_GPU(t *testing.T) {
	libPath := testLibraryPath(t)
	tokPath := testTokenizerPath(t)
	modelPath := testModelPath(t)
	if !librarySupportsCUDA(t) {
		t.Skip("ONNX Runtime library is not a CUDA build; skipping GPU auto test")
	}
	nvidiaSMI := requireNvidiaSMI(t)

	emb, err := NewEmbedder(EmbedderConfig{
		ModelPath:         modelPath,
		TokenizerPath:     tokPath,
		LibraryPath:       libPath,
		ExecutionProvider: ExecutionProviderAuto,
		Logger:            slog.New(slog.DiscardHandler),
	})
	if err != nil {
		t.Fatalf("NewEmbedder(auto on GPU) error = %v", err)
	}
	t.Cleanup(func() { closeSessionOnly(emb) })

	if got := emb.ExecutionProvider(); got != ExecutionProviderCUDA {
		t.Fatalf("ExecutionProvider() = %q, want %q on a CUDA-capable machine", got, ExecutionProviderCUDA)
	}

	// Real inference: CUDA contexts materialize lazily, so the PID appears in
	// nvidia-smi's compute-apps only after Session.Run has executed on GPU.
	if _, err := emb.EmbedQuery(context.Background(), "real inference to materialize the CUDA context"); err != nil {
		t.Fatalf("EmbedQuery() error = %v", err)
	}
	// The CUDA context now exists for the rest of the process lifetime; let
	// PID-absent assertions elsewhere in the suite know.
	processCreatedCUDAContext.Store(true)

	if !computeAppsContainsPID(nvidiaSMI, os.Getpid()) {
		t.Errorf("test PID %d not found in `nvidia-smi --query-compute-apps=pid` after CUDA inference — inference did not reach the GPU", os.Getpid())
	}
}

// TestEmbedder_ExecutionProviderGetter_ZeroValue pins the getter's contract
// for embedders not built by NewEmbedder (tests, partial states): the
// zero-value field must report the legacy CPU default, never "" or "auto".
func TestEmbedder_ExecutionProviderGetter_ZeroValue(t *testing.T) {
	e := &Embedder{}
	if got := e.ExecutionProvider(); got != ExecutionProviderCPU {
		t.Errorf("zero-value ExecutionProvider() = %q, want %q", got, ExecutionProviderCPU)
	}
}

// librarySupportsCUDA reports whether the ONNX Runtime library this test run
// initialized is a CUDA-capable build. ort.NewCUDAProviderOptions fails with
// a distinctive error on CPU-only builds ("CUDA execution provider is not
// enabled in this build"), which is a cheap, non-mutating probe. Requires the
// ONNX env to be live; when TestMain did not initialize it the answer is
// conservatively "no CUDA".
func librarySupportsCUDA(t *testing.T) bool {
	t.Helper()
	if !onnxEnvManagedForTests || !ort.IsInitialized() {
		return false
	}
	cudaOpts, err := ort.NewCUDAProviderOptions()
	if err != nil {
		// CPU-only build (or any other failure): treat as no CUDA.
		return false
	}
	_ = cudaOpts.Destroy()
	return true
}

// nvidiaSMIProbeTimeout bounds the nvidia-smi invocation used by test
// assertions: a healthy nvidia-smi answers in tens of milliseconds, and a
// wedged driver must never hang a test run.
const nvidiaSMIProbeTimeout = 10 * time.Second

// requireNvidiaSMI skips the test when nvidia-smi is unavailable (no NVIDIA
// driver on this machine) and otherwise returns its path.
func requireNvidiaSMI(t *testing.T) string {
	t.Helper()
	path, err := exec.LookPath("nvidia-smi")
	if err != nil {
		t.Skip("nvidia-smi not found; skipping GPU-visibility assertion")
	}
	return path
}

// computeAppsContainsPID runs nvidia-smi and reports whether pid appears
// among the compute-app PIDs it lists.
func computeAppsContainsPID(nvidiaSMI string, pid int) bool {
	ctx, cancel := context.WithTimeout(context.Background(), nvidiaSMIProbeTimeout)
	defer cancel()
	out, err := exec.CommandContext(ctx, nvidiaSMI,
		"--query-compute-apps=pid", "--format=csv,noheader").Output()
	if err != nil {
		return false
	}
	want := strconv.Itoa(pid)
	for _, line := range strings.Split(string(out), "\n") {
		if strings.TrimSpace(line) == want {
			return true
		}
	}
	return false
}

func TestBuildSessionOptions_Positive(t *testing.T) {
	// A positive thread count requires the ONNX runtime environment to be
	// initialized (ort.NewSessionOptions checks IsInitialized). The environment
	// is initialized once for the whole suite by TestMain; this test is gated
	// on the shared library being available via testLibraryPath, which skips
	// when EMBEDDING_TEST_LIBRARY_PATH is unset.
	_ = testLibraryPath(t)

	opts, err := buildSessionOptions("", 0, 4)
	if err != nil {
		t.Fatalf("buildSessionOptions(\"\", 0, 4) error = %v, want nil", err)
	}
	if opts == nil {
		t.Fatal("buildSessionOptions(\"\", 0, 4) opts = nil, want non-nil")
	}
	t.Cleanup(func() {
		if err := opts.Destroy(); err != nil {
			t.Errorf("opts.Destroy() error = %v", err)
		}
	})
}

// --- meanPoolAndNormalize edge cases ---

func TestMeanPoolAndNormalize_ZeroMaskSum(t *testing.T) {
	// When all attention masks are 0, no tokens contribute to pooling.
	// The embedding should remain all zeros (maskSum stays 0, no averaging).
	hiddenStates := []float32{1, 2, 3, 4, 5, 6}
	attentionMask := []int64{0, 0, 0}

	result := meanPoolAndNormalize(hiddenStates, attentionMask, 1, 3, 2)

	if len(result) != 1 {
		t.Fatalf("result length = %d, want 1", len(result))
	}
	if len(result[0]) != 2 {
		t.Fatalf("embedding dim = %d, want 2", len(result[0]))
	}
	if result[0][0] != 0 || result[0][1] != 0 {
		t.Errorf("zero mask sum should produce zero embedding, got [%f, %f]", result[0][0], result[0][1])
	}
}

func TestMeanPoolAndNormalize_MixedMaskSum(t *testing.T) {
	// Some attention mask positions are 1, some 0.
	// hiddenStates: [1,2,  3,4,  5,6]  (batch=1, seq=3, dim=2)
	// attentionMask: [1,    0,   1]
	// maskSum = 2.0, averaged = ((1,2)+(5,6)) / 2 = (3,4)
	// norm = sqrt(9+16) = 5, result = (0.6, 0.8)
	hiddenStates := []float32{1, 2, 3, 4, 5, 6}
	attentionMask := []int64{1, 0, 1}

	result := meanPoolAndNormalize(hiddenStates, attentionMask, 1, 3, 2)

	if len(result) != 1 || len(result[0]) != 2 {
		t.Fatalf("unexpected result shape: %d x %d", len(result), len(result[0]))
	}

	expected0 := float32(0.6)
	expected1 := float32(0.8)
	const eps = 1e-6
	if diff := math.Abs(float64(result[0][0] - expected0)); diff > eps {
		t.Errorf("result[0][0] = %f, want %f", result[0][0], expected0)
	}
	if diff := math.Abs(float64(result[0][1] - expected1)); diff > eps {
		t.Errorf("result[0][1] = %f, want %f", result[0][1], expected1)
	}

	// Verify unit norm.
	norm := math.Sqrt(float64(result[0][0])*float64(result[0][0]) + float64(result[0][1])*float64(result[0][1]))
	if diff := math.Abs(norm - 1.0); diff > eps {
		t.Errorf("embedding norm = %f, want 1.0", norm)
	}
}

func TestMeanPoolAndNormalize_BatchSizeTwo(t *testing.T) {
	// Batch of 2, each with 2 tokens, dim=2.
	// Batch 0: hiddenStates[0..3] = [1,2, 3,4], mask [1,1]
	// Batch 1: hiddenStates[4..7] = [5,6, 7,8], mask [1,0]
	hiddenStates := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	attentionMask := []int64{1, 1, 1, 0}

	result := meanPoolAndNormalize(hiddenStates, attentionMask, 2, 2, 2)

	if len(result) != 2 {
		t.Fatalf("result length = %d, want 2", len(result))
	}
	for i := range result {
		if len(result[i]) != 2 {
			t.Fatalf("embedding[%d] dim = %d, want 2", i, len(result[i]))
		}
	}

	// Batch 0: mean of [1,2] and [3,4] = [2,3]. norm = sqrt(4+9) = sqrt(13).
	expectedNorm0 := math.Sqrt(13)
	expected00 := float32(2.0 / expectedNorm0)
	expected01 := float32(3.0 / expectedNorm0)
	// Batch 1: mean of [5,6] only = [5,6]. norm = sqrt(25+36) = sqrt(61).
	expectedNorm1 := math.Sqrt(61)
	expected10 := float32(5.0 / expectedNorm1)
	expected11 := float32(6.0 / expectedNorm1)

	const eps = 1e-6
	if diff := math.Abs(float64(result[0][0] - expected00)); diff > eps {
		t.Errorf("result[0][0] = %f, want %f", result[0][0], expected00)
	}
	if diff := math.Abs(float64(result[0][1] - expected01)); diff > eps {
		t.Errorf("result[0][1] = %f, want %f", result[0][1], expected01)
	}
	if diff := math.Abs(float64(result[1][0] - expected10)); diff > eps {
		t.Errorf("result[1][0] = %f, want %f", result[1][0], expected10)
	}
	if diff := math.Abs(float64(result[1][1] - expected11)); diff > eps {
		t.Errorf("result[1][1] = %f, want %f", result[1][1], expected11)
	}
}

// --- Persistent batch session (multi-text EmbedDocuments) ---

// cosineSimilarity returns the cosine similarity between two equal-length
// vectors. Both inputs are assumed non-zero.
func cosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) {
		panic("cosineSimilarity: length mismatch")
	}
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

// batchTestTexts returns n pairwise-distinct texts for batch embedding tests.
func batchTestTexts(n int) []string {
	base := []string{
		"The quick brown fox jumps over the lazy dog",
		"Machine learning models transform tokens into vectors",
		"Persistent sessions avoid repeated model loading",
		"Zero-padded rows must never leak into results",
		"Chunked inference processes batches in fixed-capacity slices",
		"Jina embeddings are normalized to unit length",
		"Attention masks exclude padding from mean pooling",
		"The cosine similarity between equivalent vectors is one",
	}
	if n <= len(base) {
		return base[:n]
	}
	texts := make([]string, n)
	for i := range texts {
		texts[i] = base[i%len(base)] + fmt.Sprintf(" (variant %d)", i/len(base))
	}
	return texts
}

// embedBatchRefs embeds each text individually via EmbedQuery (the
// batchSize=1 fast path) and returns the per-item reference embeddings.
func embedBatchRefs(t *testing.T, e *Embedder, texts []string) [][]float32 {
	t.Helper()
	refs := make([][]float32, len(texts))
	for i, text := range texts {
		ref, err := e.EmbedQuery(context.Background(), text)
		if err != nil {
			t.Fatalf("EmbedQuery(%d) error = %v", i, err)
		}
		refs[i] = ref
	}
	return refs
}

// TestEmbedder_BatchEmbedDocuments_LazyInitAndDefaultBatchSize verifies that
// the batch session is created lazily and exactly once, that single-text
// calls never create it, and that BatchSize normalization falls back to
// DefaultBatchSize for zero and negative values.
func TestEmbedder_BatchEmbedDocuments_LazyInitAndDefaultBatchSize(t *testing.T) {
	libPath := testLibraryPath(t)
	tokPath := testTokenizerPath(t)
	modelPath := testModelPath(t)

	emb, err := NewEmbedder(EmbedderConfig{
		ModelPath:     modelPath,
		TokenizerPath: tokPath,
		LibraryPath:   libPath,
	})
	if err != nil {
		t.Fatalf("NewEmbedder() error = %v", err)
	}
	// See TestEmbedder_EmbeddingFunc: close sessions without destroying the
	// shared ONNX environment owned by TestMain.
	defer closeSessionOnly(emb)

	if emb.batchSize != DefaultBatchSize {
		t.Errorf("batchSize = %d, want DefaultBatchSize (%d) for zero config", emb.batchSize, DefaultBatchSize)
	}
	if emb.batchSess != nil {
		t.Fatal("batchSess must be nil right after NewEmbedder (lazy init)")
	}

	// Single-text calls must not create the batch session.
	before := onnxSessionsCreated.Load()
	if _, err := emb.EmbedQuery(context.Background(), "single text keeps the fast path"); err != nil {
		t.Fatalf("EmbedQuery() error = %v", err)
	}
	if onnxSessionsCreated.Load() != before {
		t.Error("single-text embedding must not create a session")
	}
	if emb.batchSess != nil {
		t.Error("single-text embedding must not create the batch session")
	}

	// First multi-text call creates exactly one batch session.
	before = onnxSessionsCreated.Load()
	if _, err := emb.EmbedDocuments(context.Background(), []string{"first", "second"}); err != nil {
		t.Fatalf("EmbedDocuments() error = %v", err)
	}
	if got := onnxSessionsCreated.Load() - before; got != 1 {
		t.Errorf("first multi-text call created %d sessions, want 1", got)
	}
	first := emb.batchSess
	if first == nil {
		t.Fatal("batchSess must be non-nil after the first multi-text call")
	}

	// Subsequent multi-text calls create zero sessions and reuse the same one.
	before = onnxSessionsCreated.Load()
	if _, err := emb.EmbedDocuments(context.Background(), []string{"third", "fourth", "fifth"}); err != nil {
		t.Fatalf("EmbedDocuments() error = %v", err)
	}
	if onnxSessionsCreated.Load() != before {
		t.Error("second multi-text call must not create a session")
	}
	if emb.batchSess != first {
		t.Error("second multi-text call must reuse the same batch session")
	}

	// Negative BatchSize falls back to the default.
	neg, err := NewEmbedder(EmbedderConfig{
		ModelPath:     modelPath,
		TokenizerPath: tokPath,
		LibraryPath:   libPath,
		BatchSize:     -3,
	})
	if err != nil {
		t.Fatalf("NewEmbedder(BatchSize=-3) error = %v", err)
	}
	defer closeSessionOnly(neg)
	if neg.batchSize != DefaultBatchSize {
		t.Errorf("batchSize = %d for negative config, want DefaultBatchSize (%d)", neg.batchSize, DefaultBatchSize)
	}
}

// TestEmbedder_BatchEmbedDocuments_PaddedChunk verifies that a batch smaller
// than the session capacity performs exactly one inference, returns exactly n
// results, and matches per-item embeddings — proving zero-padded rows never
// leak into results.
func TestEmbedder_BatchEmbedDocuments_PaddedChunk(t *testing.T) {
	libPath := testLibraryPath(t)
	tokPath := testTokenizerPath(t)
	modelPath := testModelPath(t)

	const capacity = 4
	emb, err := NewEmbedder(EmbedderConfig{
		ModelPath:     modelPath,
		TokenizerPath: tokPath,
		LibraryPath:   libPath,
		BatchSize:     capacity,
	})
	if err != nil {
		t.Fatalf("NewEmbedder() error = %v", err)
	}
	defer closeSessionOnly(emb)

	texts := batchTestTexts(2) // n=2 < capacity=4 -> rows 2..3 zero-padded
	ctx := context.Background()

	refs := embedBatchRefs(t, emb, texts)

	beforeRuns := onnxInferenceRuns.Load()
	beforeSessions := onnxSessionsCreated.Load()
	got, err := emb.EmbedDocuments(ctx, texts)
	if err != nil {
		t.Fatalf("EmbedDocuments() error = %v", err)
	}

	// n <= capacity -> exactly one inference and one lazy session creation.
	if d := onnxInferenceRuns.Load() - beforeRuns; d != 1 {
		t.Errorf("n<=capacity performed %d inferences, want 1", d)
	}
	if d := onnxSessionsCreated.Load() - beforeSessions; d != 1 {
		t.Errorf("first batch call created %d sessions, want 1 (lazy init)", d)
	}

	// Padded rows never leak: exactly n unit-norm results, each matching its
	// per-item reference. A leaked padded row would be an all-zero vector or
	// change the result count.
	if len(got) != len(texts) {
		t.Fatalf("len(got) = %d, want %d (padded rows must not leak)", len(got), len(texts))
	}
	for i := range texts {
		if len(got[i]) != DefaultHiddenDim {
			t.Errorf("got[%d] dim = %d, want %d", i, len(got[i]), DefaultHiddenDim)
		}
		if c := cosineSimilarity(got[i], refs[i]); c < 0.999 {
			t.Errorf("cosine(got[%d], ref[%d]) = %f, want >= 0.999", i, i, c)
		}
		var sqNorm float64
		for _, v := range got[i] {
			sqNorm += float64(v) * float64(v)
		}
		if diff := math.Abs(math.Sqrt(sqNorm) - 1.0); diff > 1e-5 {
			t.Errorf("got[%d] norm = %f, want 1.0", i, math.Sqrt(sqNorm))
		}
	}
}

// TestEmbedder_BatchEmbedDocuments_Chunking verifies that n > capacity is
// processed in ceil(n/capacity) inferences, returns exactly n vectors, and
// every vector matches its per-item embedding (order preserved across chunks).
func TestEmbedder_BatchEmbedDocuments_Chunking(t *testing.T) {
	libPath := testLibraryPath(t)
	tokPath := testTokenizerPath(t)
	modelPath := testModelPath(t)

	const capacity = 3
	emb, err := NewEmbedder(EmbedderConfig{
		ModelPath:     modelPath,
		TokenizerPath: tokPath,
		LibraryPath:   libPath,
		BatchSize:     capacity,
	})
	if err != nil {
		t.Fatalf("NewEmbedder() error = %v", err)
	}
	defer closeSessionOnly(emb)

	texts := batchTestTexts(7) // chunks: 3 + 3 + 1
	ctx := context.Background()

	refs := embedBatchRefs(t, emb, texts)

	beforeRuns := onnxInferenceRuns.Load()
	beforeSessions := onnxSessionsCreated.Load()
	got, err := emb.EmbedDocuments(ctx, texts)
	if err != nil {
		t.Fatalf("EmbedDocuments() error = %v", err)
	}

	if d := onnxSessionsCreated.Load() - beforeSessions; d != 1 {
		t.Errorf("first batch call created %d sessions, want 1 (lazy init)", d)
	}
	if d := onnxInferenceRuns.Load() - beforeRuns; d != 3 {
		t.Errorf("chunked batch performed %d inferences, want 3 (ceil(7/3))", d)
	}
	if len(got) != len(texts) {
		t.Fatalf("len(got) = %d, want %d", len(got), len(texts))
	}
	for i := range texts {
		if c := cosineSimilarity(got[i], refs[i]); c < 0.999 {
			t.Errorf("cosine(got[%d], ref[%d]) = %f, want >= 0.999", i, i, c)
		}
	}
}

// --- runBatch / newONNXSession validation guards (no ONNX runtime needed) ---

func TestNewONNXSession_ZeroBatchSize(t *testing.T) {
	// The batchSize guard rejects the request before any ONNX call.
	_, err := newONNXSession("model.onnx", 0, 8, 2, nil, nil)
	if err == nil {
		t.Error("newONNXSession(batchSize=0) expected error, got nil")
	}
}

func TestOnnxSession_RunBatch_Validation(t *testing.T) {
	s := &onnxSession{batchSize: 4, seqLen: 8, hiddenDim: 2}

	ids := make([]int64, 2*8) // correct length for n=2, seqLen=8

	// n out of range.
	if _, err := s.runBatch(0, ids, ids, ids); err == nil {
		t.Error("runBatch(n=0) expected error, got nil")
	}
	if _, err := s.runBatch(5, ids, ids, ids); err == nil {
		t.Error("runBatch(n>capacity) expected error, got nil")
	}

	// Length mismatch (want n*seqLen = 16).
	short := make([]int64, 8)
	if _, err := s.runBatch(2, short, ids, ids); err == nil {
		t.Error("runBatch(length mismatch) expected error, got nil")
	}
}
