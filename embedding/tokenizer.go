package embedding

import (
	"fmt"
	"strings"
	"unicode/utf8"

	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/pretrained"
)

// CLS token ID (101) and SEP token ID (102) for BERT-family tokenizers.
const (
	clsTokenID = 101
	sepTokenID = 102
)

// Tokenizer wraps a HuggingFace-compatible WordPiece tokenizer loaded from
// a tokenizer.json file. It produces input_ids, attention_mask, and
// token_type_ids suitable for BERT-family models like jina-embeddings-v2-small-en.
type Tokenizer struct {
	inner *tokenizer.Tokenizer
}

// NewTokenizer loads a tokenizer from a HuggingFace tokenizer.json file.
func NewTokenizer(path string) (*Tokenizer, error) {
	tk, err := pretrained.FromFile(path)
	if err != nil {
		return nil, fmt.Errorf("loading tokenizer from %s: %w", path, err)
	}
	return &Tokenizer{inner: tk}, nil
}

// Encode tokenizes a single text string and returns padded/truncated tensors
// ready for ONNX inference. maxLen controls the maximum sequence length
// (including [CLS] and [SEP] special tokens).
// Returns an error when encoding fails, e.g. due to a corrupted tokenizer.
//
// Robustness: sugarme/tokenizer v0.3.0 panics on input that is not valid
// UTF-8 — its NormalizedString alignment bookkeeping breaks on undecodable
// byte sequences (slice-bounds and nil-pointer panics inside
// AddedVocabulary.splitWithIndices). Such input occurs in practice when
// indexing files in legacy single-byte encodings (Windows-1251, Latin-1)
// or corrupted files that slip past NUL-based binary sniffing. Encode
// replaces invalid sequences with U+FFFD up front (the same lossy
// conversion the Rust HF tokenizers library applies at its input
// boundary), and additionally converts any residual library panic into an
// ordinary error so a tokenizer bug can never crash the host process.
func (t *Tokenizer) Encode(text string, maxLen int) (inputIDs, attentionMask, tokenTypeIDs []int64, err error) {
	// Guard against zero or too-small maxLen which would cause index-out-of-range.
	if maxLen < 2 {
		return nil, nil, nil, fmt.Errorf("maxLen must be >= 2, got %d", maxLen)
	}

	if !utf8.ValidString(text) {
		text = strings.ToValidUTF8(text, "�")
	}

	// Second line of defense: turn any library panic into an error.
	defer func() {
		if r := recover(); r != nil {
			inputIDs, attentionMask, tokenTypeIDs = nil, nil, nil
			if rErr, ok := r.(error); ok {
				err = fmt.Errorf("tokenizer encode panic: %w", rErr)
			} else {
				err = fmt.Errorf("tokenizer encode panic: %v", r)
			}
		}
	}()

	en, err := t.inner.EncodeSingle(text, true)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("tokenizer encode: %w", err)
	}

	ids := en.GetIds()
	mask := en.GetAttentionMask()
	typeIDs := en.GetTypeIds()

	// Truncate if needed, with bounds safety for mask/typeIDs slices.
	seqLen := len(ids)
	if seqLen > maxLen {
		seqLen = maxLen
	}
	// Ensure mask and typeIDs are at least seqLen to avoid panics.
	if len(mask) < seqLen {
		seqLen = len(mask)
	}
	if len(typeIDs) < seqLen {
		seqLen = len(typeIDs)
	}
	ids = ids[:seqLen]
	mask = mask[:seqLen]
	typeIDs = typeIDs[:seqLen]

	// Convert to int64 and pad to maxLen.
	inputIDs = make([]int64, maxLen)
	attentionMask = make([]int64, maxLen)
	tokenTypeIDs = make([]int64, maxLen)

	for i := 0; i < seqLen; i++ {
		inputIDs[i] = int64(ids[i])
		attentionMask[i] = int64(mask[i])
		tokenTypeIDs[i] = int64(typeIDs[i])
	}

	return inputIDs, attentionMask, tokenTypeIDs, nil
}

// EncodeBatch tokenizes multiple texts and returns batched tensors.
// Each returned slice is flattened in row-major order: [batch_size * maxLen].
// Returns an error if any text fails to encode.
func (t *Tokenizer) EncodeBatch(texts []string, maxLen int) (inputIDs, attentionMask, tokenTypeIDs []int64, err error) {
	batchSize := len(texts)
	totalLen := batchSize * maxLen

	inputIDs = make([]int64, totalLen)
	attentionMask = make([]int64, totalLen)
	tokenTypeIDs = make([]int64, totalLen)

	for i, text := range texts {
		ids, mask, types, e := t.Encode(text, maxLen)
		if e != nil {
			return nil, nil, nil, fmt.Errorf("batch encode text %d: %w", i, e)
		}
		offset := i * maxLen
		copy(inputIDs[offset:offset+maxLen], ids)
		copy(attentionMask[offset:offset+maxLen], mask)
		copy(tokenTypeIDs[offset:offset+maxLen], types)
	}

	return inputIDs, attentionMask, tokenTypeIDs, nil
}
