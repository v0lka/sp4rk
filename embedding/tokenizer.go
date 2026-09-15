package embedding

import (
	"fmt"
	"strings"
	"unicode"

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
func (t *Tokenizer) Encode(text string, maxLen int) (inputIDs, attentionMask, tokenTypeIDs []int64, err error) {
	inputIDs, attentionMask, tokenTypeIDs, _, err = t.EncodeWithLength(text, maxLen)
	return inputIDs, attentionMask, tokenTypeIDs, err
}

// EncodeWithLength is Encode plus the actual non-padding sequence length after
// truncation. The length lets callers select a smaller fixed-shape ONNX session
// without rescanning the attention mask.
//
// Robustness: sugarme/tokenizer v0.3.0 panics whenever BERT normalization
// changes the byte layout of the text it is about to split. Its NormalizedString
// alignment bookkeeping assumes the normalized string still maps byte-for-byte
// onto the original runes; once a normalization step breaks that assumption the
// offsets desynchronize and the library panics (slice-bounds in TransformRange,
// nil-pointer in Slice reached through AddedVocabulary.splitWithIndices). Two
// steps of the jina-v2-small normalizer do this:
//
//   - cleanText (the library's normalizer/bert.go doCleanText) removes NUL
//     (U+0000), the replacement character (U+FFFD) and control runes
//     (unicode.Cc / unicode.Cf, with tab/newline/carriage-return excepted);
//   - lowercase changes a rune's byte length (U+023A -> U+2C65 and U+023E ->
//     U+2C66, both expanding 2 -> 3 bytes).
//
// Crucially, both fire on VALID UTF-8 — NUL and control runes are legal UTF-8,
// and U+023A is an ordinary letter — so this is not an "invalid UTF-8" bug.
// Malformed byte sequences (legacy single-byte files such as Windows-1251 /
// Latin-1, or corrupted bytes that slip past NUL-based binary sniffing) are
// only one route into the cleanText class: ranging over a string yields U+FFFD
// for every undecodable byte, which cleanText then strips.
//
// EncodeWithLength defends in two layers. It first sanitizes the input:
// sanitizeForTokenizer rewrites every rune cleanText would strip — and every
// undecodable byte, which ranges as U+FFFD — to a plain space, a character
// cleanText never removes, so the alignment bookkeeping stays consistent.
// Sanitization cannot fix the lowercase class, though, because those runes are
// not in cleanText's deletion set; for that EncodeWithLength falls back once to
// an ASCII-folded copy (see asciiFold), indexing the text with a coarsened
// embedding rather than dropping the document. Any panic that survives both
// layers is converted into an ordinary error so a tokenizer bug can never crash
// the host process.
func (t *Tokenizer) EncodeWithLength(text string, maxLen int) (inputIDs, attentionMask, tokenTypeIDs []int64, actualLen int, err error) {
	// Guard against zero or too-small maxLen which would cause index-out-of-range.
	if maxLen < 2 {
		return nil, nil, nil, 0, fmt.Errorf("maxLen must be >= 2, got %d", maxLen)
	}

	text = sanitizeForTokenizer(text)

	// encodeOnce performs one tokenize-and-pad pass. It owns the recover guard:
	// a library panic becomes an ordinary error, so a tokenizer bug can never
	// crash the host process. The non-padding length is written to the caller's
	// named return so the tensors and their length always describe the same
	// encode.
	encodeOnce := func(text string) (inputIDs, attentionMask, tokenTypeIDs []int64, panicErr error) {
		defer func() {
			if r := recover(); r != nil {
				inputIDs, attentionMask, tokenTypeIDs = nil, nil, nil
				actualLen = 0
				if rErr, ok := r.(error); ok {
					panicErr = fmt.Errorf("tokenizer encode panic: %w", rErr)
				} else {
					panicErr = fmt.Errorf("tokenizer encode panic: %v", r)
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

		actualLen = seqLen
		return inputIDs, attentionMask, tokenTypeIDs, nil
	}

	inputIDs, attentionMask, tokenTypeIDs, err = encodeOnce(text)
	if err != nil {
		// Sanitization is not sufficient on its own: some valid UTF-8 runes
		// whose lowercase mapping changes the byte length (e.g. U+023A ->
		// U+2C65 and U+023E -> U+2C66, both expanding 2 -> 3 bytes) corrupt the
		// library's normalization alignment bookkeeping and yield a nil-deref
		// panic deep in NormalizedString.Slice, never entering cleanText's
		// deletion set. Rather than dropping the document, retry once on an
		// ASCII-folded copy so a coarsened embedding still reaches the index.
		if folded := asciiFold(text); folded != text {
			inputIDs, attentionMask, tokenTypeIDs, err = encodeOnce(folded)
		}
	}
	if err != nil {
		return nil, nil, nil, 0, err
	}

	return inputIDs, attentionMask, tokenTypeIDs, actualLen, nil
}

// asciiFold returns text reduced to printable ASCII (0x20..0x7E) plus '\n'.
// Every other rune — control characters, non-ASCII scripts and any multi-byte
// sequence — is replaced by a single space. It is the tokenizer's last-resort
// input: text can be valid UTF-8, survive sanitizeForTokenizer, and still trip
// the library's alignment bookkeeping, so folding to ASCII guarantees an input
// the normalizer cannot choke on. The result is lossy by design — the document
// is indexed with a coarsened embedding instead of being discarded.
func asciiFold(text string) string {
	ascii := true
	for i := 0; i < len(text); i++ {
		if c := text[i]; c > 0x7E || (c < 0x20 && c != '\n') {
			ascii = false
			break
		}
	}
	if ascii {
		return text
	}

	var b strings.Builder
	b.Grow(len(text))
	for _, r := range text {
		switch {
		case r == '\n':
			b.WriteByte('\n')
		case r >= 0x20 && r <= 0x7E:
			b.WriteByte(byte(r))
		default:
			b.WriteByte(' ')
		}
	}
	return b.String()
}

// EncodeBatch tokenizes multiple texts and returns batched tensors.
// Each returned slice is flattened in row-major order: [batch_size * maxLen].
// Returns an error if any text fails to encode.
func (t *Tokenizer) EncodeBatch(texts []string, maxLen int) (inputIDs, attentionMask, tokenTypeIDs []int64, err error) {
	inputIDs, attentionMask, tokenTypeIDs, _, err = t.EncodeBatchWithLengths(texts, maxLen)
	return inputIDs, attentionMask, tokenTypeIDs, err
}

// EncodeBatchWithLengths is EncodeBatch plus one actual non-padding sequence
// length per input text.
func (t *Tokenizer) EncodeBatchWithLengths(texts []string, maxLen int) (inputIDs, attentionMask, tokenTypeIDs []int64, lengths []int, err error) {
	batchSize := len(texts)
	totalLen := batchSize * maxLen

	inputIDs = make([]int64, totalLen)
	attentionMask = make([]int64, totalLen)
	tokenTypeIDs = make([]int64, totalLen)
	lengths = make([]int, batchSize)

	for i, text := range texts {
		ids, mask, types, actualLen, e := t.EncodeWithLength(text, maxLen)
		if e != nil {
			return nil, nil, nil, nil, fmt.Errorf("batch encode text %d: %w", i, e)
		}
		offset := i * maxLen
		copy(inputIDs[offset:offset+maxLen], ids)
		copy(attentionMask[offset:offset+maxLen], mask)
		copy(tokenTypeIDs[offset:offset+maxLen], types)
		lengths[i] = actualLen
	}

	return inputIDs, attentionMask, tokenTypeIDs, lengths, nil
}

// sanitizeForTokenizer rewrites to a single ASCII space every rune that
// sugarme's BERT cleanText would strip from the normalized string — NUL
// (U+0000), the replacement character (U+FFFD) and control runes
// (unicode.Cc / unicode.Cf, with tab, newline and carriage return preserved
// because BERT keeps them). Because cleanText never removes a space, its
// alignment bookkeeping stays consistent; feeding it those runes directly
// previously drove it to an "index out of range" panic.
//
// Ranging over a string yields U+FFFD for every invalid UTF-8 byte, so this
// single pass also replaces undecodable bytes (legacy single-byte encodings,
// corrupted files) with a space instead of handing the library invalid input.
// Text that needs no rewriting is returned unchanged, without allocating.
func sanitizeForTokenizer(text string) string {
	if strings.IndexFunc(text, needsSpaceReplacement) < 0 {
		return text
	}
	var b strings.Builder
	b.Grow(len(text))
	for _, r := range text {
		if needsSpaceReplacement(r) {
			b.WriteByte(' ')
		} else {
			b.WriteRune(r)
		}
	}
	return b.String()
}

// needsSpaceReplacement reports whether r must be rewritten to a space before
// tokenization. It mirrors sugarme's isControl (unicode.Cc / unicode.Cf with
// tab, newline and carriage return excepted) and additionally flags NUL and
// U+FFFD — the latter being the rune produced for every invalid UTF-8 byte
// sequence when ranging over a string.
func needsSpaceReplacement(r rune) bool {
	switch r {
	case '\t', '\n', '\r':
		return false
	case 0, '\uFFFD':
		return true
	}
	return unicode.In(r, unicode.Cc, unicode.Cf)
}
