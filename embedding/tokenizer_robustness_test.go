package embedding

import (
	"strings"
	"testing"
	"unicode/utf8"
)

// TestTokenizer_Encode_MalformedBytesRegression is a regression test for
// production crashes on text whose raw bytes are not valid UTF-8. Such input is
// not itself special — it is one route into sugarme/tokenizer v0.3.0's
// normalization panic: ranging over a string yields U+FFFD for every undecodable
// byte, and cleanText then removes U+FFFD, desynchronizing the library's
// NormalizedString alignment bookkeeping (slice-bounds / nil-pointer inside
// AddedVocabulary.splitWithIndices). c0wrk's vector indexer hits this when a file
// in a legacy single-byte encoding (e.g. Windows-1251) or with corrupted bytes
// passes the NUL-header binary sniff and its raw content reaches the tokenizer.
//
// The panic is NOT specific to invalid UTF-8: see
// TestTokenizer_Encode_ValidUTF8CleanTextTriggers and
// TestTokenizer_Encode_SanitizedPanicFallbackToASCII for the valid-UTF-8
// triggers.
//
// Every input below was verified to panic the library before the fix. Encode
// must now sanitize such input (rewriting the offending bytes to spaces) and
// return a normal result — neither panic nor error.
func TestTokenizer_Encode_MalformedBytesRegression(t *testing.T) {
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

// TestTokenizer_Encode_ValidUTF8CleanTextTriggers covers the other valid-UTF-8
// trigger class: runes that cleanText removes. NUL, the replacement character
// (U+FFFD) and control runes are all legal UTF-8, so ordinary text can contain
// them; feeding them straight to the library desynchronizes its alignment
// bookkeeping and panics. Defense A — sanitizeForTokenizer — rewrites each one
// to a space, so every input must now encode successfully.
func TestTokenizer_Encode_ValidUTF8CleanTextTriggers(t *testing.T) {
	tok, err := NewTokenizer(testTokenizerPath(t))
	if err != nil {
		t.Fatalf("NewTokenizer() error = %v", err)
	}

	inputs := map[string]string{
		"control-and-replacement-mix": "\uFFFD\t\u0080b\x01",
		"nul-and-replacement":         "\x00\uFFFD\x00",
		"control-run":                 "foo\x00bar\x07baz\x1f",
		"repeated-mix":                strings.Repeat("\uFFFD\t\u0080b\x01", 20),
	}

	for name, input := range inputs {
		t.Run(name, func(t *testing.T) {
			// These inputs are valid UTF-8 by construction — that is the whole
			// point — so assert it rather than assume it.
			if !utf8.ValidString(input) {
				t.Fatalf("test input %q is not valid UTF-8", input)
			}
			// Defense A must engage: without sanitization this input panics.
			if sanitized := sanitizeForTokenizer(input); sanitized == input {
				t.Fatalf("sanitizeForTokenizer(%q) == input; defense A would be skipped", input)
			}

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

// TestTokenizer_Encode_SanitizedPanicFallbackToASCII is a regression test for a
// panic the cleanText sanitizer cannot prevent: some valid UTF-8 runes whose
// lowercase mapping changes the byte length (U+023A -> U+2C65 and U+023E ->
// U+2C66, both expanding 2 -> 3 bytes) corrupt sugarme/tokenizer v0.3.0's
// normalization alignment bookkeeping and produce a nil-pointer dereference in
// NormalizedString.Slice (normalizer/normalized.go:391). Neither rune is in
// cleanText's deletion set, so sanitizeForTokenizer leaves them untouched and
// the panic is reachable. Every input below was verified to fail the sanitized
// encode before the ASCII-fold fallback existed.
//
// Encode must not drop the document: it retries on an ASCII-folded copy and
// returns a normal result — neither panic nor error.
func TestTokenizer_Encode_SanitizedPanicFallbackToASCII(t *testing.T) {
	tok, err := NewTokenizer(testTokenizerPath(t))
	if err != nil {
		t.Fatalf("NewTokenizer() error = %v", err)
	}

	inputs := map[string]string{
		"u023a-stroke-a":     "a\u023A",
		"u023e-stroke-t":     "prefix \u023E suffix",
		"cjk-around-trigger": "\u4f60a\u023A\u597d",
	}

	for name, input := range inputs {
		t.Run(name, func(t *testing.T) {
			// The fallback only fires when folding changes the text; assert that
			// precondition so the test cannot silently stop exercising it.
			if folded := asciiFold(input); folded == input {
				t.Fatalf("asciiFold(%q) == input; the fallback path would be skipped", input)
			}

			ids, mask, typeIDs, err := tok.Encode(input, 64)
			if err != nil {
				t.Fatalf("Encode(%q) error = %v; want ASCII-fallback success", input, err)
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

// TestASCIFFold pins the ASCII-fold contract used as the tokenizer's
// last-resort input: printable ASCII (0x20..0x7E) and '\n' survive verbatim,
// while everything else — other control characters, multi-byte runes and the
// U+FFFD produced for invalid UTF-8 — collapses to a single space.
func TestASCIFFold(t *testing.T) {
	cases := map[string]struct {
		in   string
		want string
	}{
		"printable-passthrough":  {"hello world", "hello world"},
		"newline-kept":           {"a\nb", "a\nb"},
		"tab-to-space":           {"a\tb", "a b"},
		"cr-to-space":            {"a\r\nb", "a \nb"},
		"del-to-space":           {"a\x7fb", "a b"},
		"latin1-to-space":        {"caf\u00e9", "caf "},
		"stroke-a-to-space":      {"a\u023A", "a "},
		"cjk-to-space":           {"\u4f60\u597d", "  "},
		"replacement-char":       {"a\uFFFDb", "a b"},
		"invalid-utf8-lone-byte": {"a\x80b", "a b"},
		"empty":                  {"", ""},
	}

	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			if got := asciiFold(tc.in); got != tc.want {
				t.Errorf("asciiFold(%q) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}
