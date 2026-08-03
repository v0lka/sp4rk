package strutil

import "testing"

func TestTruncateUTF8(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		maxChars int
		want     string
	}{
		{name: "empty", input: "", maxChars: 10, want: ""},
		{name: "shorter than max", input: "hello", maxChars: 10, want: "hello"},
		{name: "equal to max", input: "hello", maxChars: 5, want: "hello"},
		{name: "truncate with ellipsis", input: "hello world", maxChars: 8, want: "hello w…"},
		{name: "truncate small max fits ellipsis", input: "hello world", maxChars: 2, want: "h…"},
		{name: "truncate max=1", input: "hello world", maxChars: 1, want: "h…"},
		{name: "negative max", input: "hello", maxChars: -1, want: ""},
		{name: "zero max", input: "hello", maxChars: 0, want: ""},
		// Multibyte: "café" = 4 runes (c,a,f,é). maxChars=3 → "ca…"
		{name: "multibyte truncate", input: "café", maxChars: 3, want: "ca…"},
		{name: "multibyte no truncation", input: "café", maxChars: 4, want: "café"},
		{name: "multibyte max=1", input: "café", maxChars: 1, want: "c…"},
		// "你好" = 2 runes (你,好). maxChars=1 → "你…"
		{name: "chinese max=1", input: "你好", maxChars: 1, want: "你…"},
		{name: "chinese no truncation", input: "你好", maxChars: 2, want: "你好"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := TruncateUTF8(tt.input, tt.maxChars)
			if got != tt.want {
				t.Errorf("TruncateUTF8(%q, %d) = %q, want %q", tt.input, tt.maxChars, got, tt.want)
			}
		})
	}
}

func TestTruncateUTF8AtLineBoundary(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		maxChars int
		want     string
	}{
		{name: "empty", input: "", maxChars: 10, want: ""},
		{name: "short string no truncation", input: "hello", maxChars: 10, want: "hello"},
		{name: "exact fit", input: "hello", maxChars: 5, want: "hello"},
		// "line1\nline2\nline3" — maxChars=14, snaps to "line1\nline2\n".
		{name: "mid-line truncation snaps to newline", input: "line1\nline2\nline3", maxChars: 14, want: "line1\nline2\n"},
		{name: "no newline returns truncated", input: "hello world", maxChars: 5, want: "hello"},
		{name: "newline at idx 0 returns newline only", input: "\nhello", maxChars: 3, want: "\n"},
		// Multibyte: "你好\nabc" — maxChars=2 → only "你好" fits, no newline found
		// so returns first 2 runes.
		{name: "multibyte no newline in range", input: "你好\nabc", maxChars: 2, want: "你好"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := TruncateUTF8AtLineBoundary(tt.input, tt.maxChars)
			if got != tt.want {
				t.Errorf("TruncateUTF8AtLineBoundary(%q, %d) = %q, want %q", tt.input, tt.maxChars, got, tt.want)
			}
		})
	}
}
