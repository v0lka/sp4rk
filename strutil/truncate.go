// Package strutil provides string utilities such as UTF-8-safe truncation.
package strutil

import "unicode/utf8"

// InvisibleTrimSet is the cutset of trailing invisible characters common in LLM output.
// Includes spaces, tabs, newlines, carriage returns, null, form feed, vertical tab,
// zero-width space (U+200B), zero-width non-joiner (U+200C), zero-width joiner (U+200D),
// and byte-order mark (U+FEFF).
const InvisibleTrimSet = " \t\n\r\x00\f\v\u200b\u200c\u200d\ufeff"

// TruncateUTF8 truncates s to at most maxChars runes, appending "…" if truncated.
func TruncateUTF8(s string, maxChars int) string {
	if maxChars <= 0 {
		return ""
	}
	runes := []rune(s)
	runesLen := utf8.RuneCountInString(s)
	if runesLen <= maxChars {
		return s
	}
	if maxChars <= 1 {
		return string(runes[:maxChars]) + "…"
	}
	return string(runes[:maxChars-1]) + "…"
}

// TruncateUTF8AtLineBoundary truncates s to at most maxChars runes,
// then rewinds to the last newline to avoid splitting mid-line.
// It never splits multi-byte runes; the result is always a valid prefix
// of the input ending at a '\n' boundary (or the full string if no
// newline exists within maxChars).
func TruncateUTF8AtLineBoundary(s string, maxChars int) string {
	if maxChars <= 0 {
		return ""
	}
	runes := []rune(s)
	if len(runes) <= maxChars {
		return s
	}
	// Rewind from maxChars to the last newline.
	for i := maxChars - 1; i >= 0; i-- {
		if runes[i] == '\n' {
			return string(runes[:i+1])
		}
	}
	// No newline found within maxChars — return exactly maxChars runes.
	return string(runes[:maxChars])
}
