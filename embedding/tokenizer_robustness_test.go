package embedding

import (
	"strings"
	"testing"
)

// TestTokenizer_Encode_InvalidUTF8 is a regression test for production
// crashes: sugarme/tokenizer v0.3.0 panics (slice-bounds / nil-pointer
// inside NormalizedString.Slice via AddedVocabulary.splitWithIndices) on
// any input that is not valid UTF-8. c0wrk's vector indexer hits this when
// a file in a legacy single-byte encoding (e.g. Windows-1251) or with
// corrupted bytes passes the NUL-header binary sniff and its raw content
// reaches the tokenizer.
//
// Every input below was verified to panic the library before the fix.
// Encode must now sanitize such input (U+FFFD replacement) and return a
// normal result — neither panic nor error.
func TestTokenizer_Encode_InvalidUTF8(t *testing.T) {
	tokPath := testTokenizerPath(t)
	tok, err := NewTokenizer(tokPath)
	if err != nil {
		t.Fatalf("NewTokenizer() error = %v", err)
	}

	inputs := map[string]string{
		"lone-continuation-bytes":  "abc\x80\x81def",
		"0xff-byte":                "hello\xffworld",
		"truncated-multibyte":      "foo\xc3",
		"overlong-encoding":        "bar\xc0\xafbaz",
		"invalid-with-added-token": "\xff\xfe[MASK]",
		"invalid-cjk-mix":          "abc\x80你好",
		"surrogate-half":           "foo\xed\xa0\x80bar",
	}

	for name, input := range inputs {
		t.Run(name, func(t *testing.T) {
			// A panic here crashes the test binary and fails the run:
			// that IS the regression signal.
			ids, mask, typeIDs, err := tok.Encode(input, 64)
			if err != nil {
				t.Fatalf("Encode(%q) error = %v; want sanitized success", input, err)
			}
			if len(ids) != 64 || len(mask) != 64 || len(typeIDs) != 64 {
				t.Errorf("Encode(%q) lengths = %d/%d/%d, want 64 each", input, len(ids), len(mask), len(typeIDs))
			}
			if ids[0] != clsTokenID {
				t.Errorf("Encode(%q) first token = %d, want %d ([CLS])", input, ids[0], clsTokenID)
			}
		})
	}
}

// TestTokenizer_Encode_ValidUTF8Unaffected guards the sanitizer against
// overreach: text that is already valid UTF-8 — including multi-byte
// scripts, emoji, control characters and literal added-token strings —
// must keep encoding successfully.
func TestTokenizer_Encode_ValidUTF8Unaffected(t *testing.T) {
	tokPath := testTokenizerPath(t)
	tok, err := NewTokenizer(tokPath)
	if err != nil {
		t.Fatalf("NewTokenizer() error = %v", err)
	}

	inputs := map[string]string{
		"cjk":                 "你好世界，这是一段中文文本。",
		"russian":             "Привет, мир! Это русский текст.",
		"emoji-zwj":           "👨‍👩‍👧‍👦 family 🏳️‍🌈",
		"combining-marks":     "é á ö",
		"turkish-dotted-I":    "İstanbul",
		"arabic-rtl":          "مرحبا بالعالم",
		"added-token-strings": "the [MASK] and [CLS] tokens [SEP]",
		"control-chars":       "foo\x00bar\x07baz\x1f",
	}

	for name, input := range inputs {
		t.Run(name, func(t *testing.T) {
			ids, _, _, err := tok.Encode(input, 64)
			if err != nil {
				t.Fatalf("Encode(%q) error = %v; want success", input, err)
			}
			if ids[0] != clsTokenID {
				t.Errorf("Encode(%q) first token = %d, want %d ([CLS])", input, ids[0], clsTokenID)
			}
		})
	}
}

// TestTokenizer_Encode_PanicConvertedToError exercises the recover guard
// itself, without needing a tokenizer.json: a Tokenizer with a nil inner
// tokenizer panics inside EncodeSingle, and Encode must surface that as an
// ordinary error instead of crashing the caller.
//
// NOTE: this test deliberately relies on the pinned sugarme/tokenizer v0.3.0
// (see go.mod) panicking on a nil receiver — Encode -> EncodeSingleSequence
// dereferences the nil *tokenizer.Tokenizer. If the library is ever upgraded
// to a version that nil-guards its methods (returning an error instead of
// panicking), this test will fail explicitly: Encode will return a non-nil
// error whose message no longer mentions "panic". That failure is the signal
// to revisit the recover guard — e.g. trigger a genuine panic differently or
// drop the guard if the library made it unreachable — rather than a silent
// behavior change.
func TestTokenizer_Encode_PanicConvertedToError(t *testing.T) {
	tok := &Tokenizer{} // inner == nil: EncodeSingle dereferences it and panics

	_, _, _, err := tok.Encode("hello", 8)
	if err == nil {
		t.Fatal("Encode with nil inner tokenizer: expected recovered-panic error, got nil")
	}
	if !strings.Contains(err.Error(), "panic") {
		t.Errorf("error = %q; want it to mention the recovered panic", err)
	}
}
