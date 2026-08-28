package embedding

import (
	"context"
	"fmt"
	"math"
	"os"
	"testing"
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
	// intraOpThreads == 0 must return (nil, nil) without touching the ONNX
	// runtime. This is the legacy / default code path and must not require
	// the shared library to be initialized.
	opts, err := buildSessionOptions(0)
	if err != nil {
		t.Fatalf("buildSessionOptions(0) error = %v, want nil", err)
	}
	if opts != nil {
		t.Errorf("buildSessionOptions(0) opts = %p, want nil", opts)
	}
}

func TestBuildSessionOptions_Negative(t *testing.T) {
	// Negative values are treated like 0: return (nil, nil).
	for _, n := range []int{-1, -42} {
		opts, err := buildSessionOptions(n)
		if err != nil {
			t.Fatalf("buildSessionOptions(%d) error = %v, want nil", n, err)
		}
		if opts != nil {
			t.Errorf("buildSessionOptions(%d) opts = %p, want nil", n, opts)
		}
	}
}

func TestBuildSessionOptions_Positive(t *testing.T) {
	// A positive thread count requires the ONNX runtime environment to be
	// initialized (ort.NewSessionOptions checks IsInitialized). The environment
	// is initialized once for the whole suite by TestMain; this test is gated
	// on the shared library being available via testLibraryPath, which skips
	// when EMBEDDING_TEST_LIBRARY_PATH is unset.
	_ = testLibraryPath(t)

	opts, err := buildSessionOptions(4)
	if err != nil {
		t.Fatalf("buildSessionOptions(4) error = %v, want nil", err)
	}
	if opts == nil {
		t.Fatal("buildSessionOptions(4) opts = nil, want non-nil")
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

	texts := batchTestTexts(2) // n=2 < capacity=4 → rows 2..3 zero-padded
	ctx := context.Background()

	refs := embedBatchRefs(t, emb, texts)

	beforeRuns := onnxInferenceRuns.Load()
	beforeSessions := onnxSessionsCreated.Load()
	got, err := emb.EmbedDocuments(ctx, texts)
	if err != nil {
		t.Fatalf("EmbedDocuments() error = %v", err)
	}

	// n <= capacity → exactly one inference and one lazy session creation.
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
	_, err := newONNXSession("model.onnx", 0, 8, 2, nil)
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
