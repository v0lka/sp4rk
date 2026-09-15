package embedding

import (
	"strings"
	"testing"
)

// TestSanitizeForTokenizer covers the pure single-pass sanitizer: every rune
// that sugarme's BERT cleanText (normalizer/bert.go doCleanText) would strip —
// NUL (U+0000), the replacement character (U+FFFD) and control runes other
// than tab/newline/carriage-return — plus every invalid UTF-8 byte must become
// a single space.
func TestSanitizeForTokenizer(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want string
	}{
		{"nul", "a\x00b", "a b"},
		{"replacement-char", "a\uFFFDb", "a b"},
		{"control-c0", "a\x01\x07b", "a  b"},
		{"control-c1", "a\u0080\u009fb", "a  b"},
		{"del", "a\x7fb", "a b"},
		{"zero-width-space-cf", "a\u200bb", "a b"},
		{"zero-width-joiner-cf", "a\u200db", "a b"},
		{"bom-cf", "a\uFEFFb", "a b"},
		{"invalid-lone-continuation", "abc\x80\x81def", "abc  def"},
		{"invalid-overlong", "bar\xc0\xafbaz", "bar  baz"},
		{"invalid-truncated-multibyte", "foo\xc3", "foo "},
		{"invalid-surrogate-half", "foo\xed\xa0\x80bar", "foo   bar"},
		// The regression reproducer: U+FFFD, C1 control and C0 control mix.
		{"reproducer", "\uFFFD\t\u0080b\x01", " \t b "},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := sanitizeForTokenizer(tc.in); got != tc.want {
				t.Errorf("sanitizeForTokenizer(%q) = %q; want %q", tc.in, got, tc.want)
			}
		})
	}
}

// TestSanitizeForTokenizerPreservesCleanInput guards the sanitizer against
// overreach: input with no strippable rune — including multi-byte scripts,
// emoji, tab/newline/CR and literal added-token strings — is returned
// byte-for-byte unchanged (and, via the fast path, without allocating).
//
// Note: emoji here must not contain a ZWJ (U+200D, category Cf) — that rune is
// deliberately sanitized (it is in the mirrored Cc/Cf set; sugarme's cleanText
// would strip it anyway), so a ZWJ sequence is covered by the replacement cases
// above rather than here.
func TestSanitizeForTokenizerPreservesCleanInput(t *testing.T) {
	kept := []string{
		"hello world",
		"tab\tnewline\ncarriage\rreturn",
		"你好世界，这是一段中文文本。",
		"Привет, мир! Это русский текст.",
		"😀🎉🚀",
		"İstanbul é á ö",
		"مرحبا بالعالم",
		"the [MASK] and [CLS] tokens [SEP]",
	}
	for _, in := range kept {
		if got := sanitizeForTokenizer(in); got != in {
			t.Errorf("sanitizeForTokenizer(%q) = %q; want unchanged", in, got)
		}
	}
}

// TestNeedsSpaceReplacement pins the rune classification to the exact contract
// mirrored from sugarme's isControl (unicode.Cc / unicode.Cf, with tab, newline
// and carriage return preserved) plus NUL and U+FFFD.
func TestNeedsSpaceReplacement(t *testing.T) {
	replaced := []rune{0, '\uFFFD', 0x01, 0x07, 0x1f, 0x7f, 0x80, 0x9f, 0x200b, 0x200c, 0xad, 0xFEFF}
	for _, r := range replaced {
		if !needsSpaceReplacement(r) {
			t.Errorf("needsSpaceReplacement(%U) = false; want true", r)
		}
	}
	preserved := []rune{'\t', '\n', '\r', ' ', 'a', 'Z', '0', '你', 'я', '🙂'}
	for _, r := range preserved {
		if needsSpaceReplacement(r) {
			t.Errorf("needsSpaceReplacement(%U) = true; want false", r)
		}
	}
}

// TestTokenizer_Encode_SanitizesStrippedRunes is the end-to-end regression test
// for the cleanText alignment panic: input containing runes the BERT normalizer
// would remove (U+FFFD, C1/C0 controls, NUL) — alone and repeated well past the
// truncation boundary — must encode successfully (no panic, no error).
func TestTokenizer_Encode_SanitizesStrippedRunes(t *testing.T) {
	tok, err := NewTokenizer(testTokenizerPath(t))
	if err != nil {
		t.Fatalf("NewTokenizer() error = %v", err)
	}

	base := "\uFFFD\t\u0080b\x01"
	inputs := map[string]string{
		"reproducer":                 base,
		"repeated":                   strings.Repeat(base, 20),
		"repeated-long":              strings.Repeat(base, 200),
		"nul-and-replacement":        "\x00\uFFFD\x00",
		"mixed-invalid-and-stripped": "abc\x80\u0080\x01你好\uFFFD",
	}

	for name, input := range inputs {
		t.Run(name, func(t *testing.T) {
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

// TestTokenizer_Encode_ValidUTF8StillTokenizes is the companion check that the
// sanitizer does not overreach end to end: untouched valid UTF-8 (CJK, Russian,
// emoji, tab/newline and literal added-token strings) still encodes.
func TestTokenizer_Encode_ValidUTF8StillTokenizes(t *testing.T) {
	tok, err := NewTokenizer(testTokenizerPath(t))
	if err != nil {
		t.Fatalf("NewTokenizer() error = %v", err)
	}

	inputs := map[string]string{
		"cjk":                 "你好世界，这是一段中文文本。",
		"russian":             "Привет, мир! Это русский текст.",
		"emoji":               "👨‍👩‍👧‍👦 family 🏳️‍🌈",
		"tab-and-newline":     "line1\nline2\ttabbed\r",
		"added-token-strings": "the [MASK] and [CLS] tokens [SEP]",
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
