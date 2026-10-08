package builtins

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func writeTempFile(t *testing.T, name, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("failed to write temp file: %v", err)
	}
	return path
}

// countContentLines counts lines in ReadFileRange output, accounting for
// trailing newlines the same way ReadFileRange counts file lines.
func countContentLines(content string) int {
	if content == "" {
		return 0
	}
	n := strings.Count(content, "\n")
	if !strings.HasSuffix(content, "\n") {
		n++
	}
	return n
}

func TestReadFileRange_DefaultWindow(t *testing.T) {
	var sb strings.Builder
	for i := 1; i <= 10000; i++ {
		if i > 1 {
			sb.WriteByte('\n')
		}
		fmt.Fprintf(&sb, "Line %d", i)
	}
	path := writeTempFile(t, "large.txt", sb.String())

	result, err := ReadFileRange(FileReadParams{
		Path:         path,
		DefaultLines: 2000,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.TotalLines != 10000 {
		t.Errorf("TotalLines = %d, want 10000", result.TotalLines)
	}
	if result.StartLine != 1 {
		t.Errorf("StartLine = %d, want 1", result.StartLine)
	}
	if result.EndLine != 2000 {
		t.Errorf("EndLine = %d, want 2000", result.EndLine)
	}
	if n := countContentLines(result.Content); n != 2000 {
		t.Errorf("content has %d lines, want 2000", n)
	}
	if !strings.HasPrefix(result.Content, "Line 1\n") {
		t.Errorf("expected content to start with 'Line 1\\n', got: %q", result.Content[:min(80, len(result.Content))])
	}
	if !strings.Contains(result.Content, "Line 2000") {
		t.Errorf("expected 'Line 2000' in content")
	}
	if result.WindowCapped {
		t.Error("WindowCapped should be false for default window within MaxWindowLines")
	}
}

func TestReadFileRange_ExplicitRange(t *testing.T) {
	var sb strings.Builder
	for i := 1; i <= 100; i++ {
		if i > 1 {
			sb.WriteByte('\n')
		}
		fmt.Fprintf(&sb, "Line %d", i)
	}
	path := writeTempFile(t, "medium.txt", sb.String())

	result, err := ReadFileRange(FileReadParams{
		Path:      path,
		StartLine: 10,
		EndLine:   20,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.TotalLines != 100 {
		t.Errorf("TotalLines = %d, want 100", result.TotalLines)
	}
	if n := countContentLines(result.Content); n != 11 {
		t.Errorf("content has %d lines, want 11", n)
	}
	if !strings.HasPrefix(result.Content, "Line 10\n") {
		t.Errorf("expected content to start with 'Line 10\\n'")
	}
	if !strings.Contains(result.Content, "Line 20") {
		t.Errorf("expected 'Line 20' in content")
	}
}

func TestReadFileRange_TrailingNewline(t *testing.T) {
	content := "Line 1\nLine 2\nLine 3\n"
	path := writeTempFile(t, "trailing.txt", content)

	result, err := ReadFileRange(FileReadParams{
		Path:         path,
		EndLine:      3,
		DefaultLines: 100,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.TotalLines != 3 {
		t.Errorf("TotalLines = %d, want 3 (trailing newline should not add a line)", result.TotalLines)
	}
}

func TestReadFileRange_NoTrailingNewline(t *testing.T) {
	content := "Line 1\nLine 2\nLine 3"
	path := writeTempFile(t, "notrailing.txt", content)

	result, err := ReadFileRange(FileReadParams{
		Path:         path,
		EndLine:      3,
		DefaultLines: 100,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.TotalLines != 3 {
		t.Errorf("TotalLines = %d, want 3", result.TotalLines)
	}
}

func TestReadFileRange_EmptyFile(t *testing.T) {
	path := writeTempFile(t, "empty.txt", "")

	result, err := ReadFileRange(FileReadParams{
		Path:         path,
		DefaultLines: 2000,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.TotalLines != 0 {
		t.Errorf("TotalLines = %d, want 0 for empty file", result.TotalLines)
	}
	if result.Content != "" {
		t.Errorf("Content = %q, want empty", result.Content)
	}
}

func TestReadFileRange_SingleNewline(t *testing.T) {
	path := writeTempFile(t, "nl.txt", "\n")

	result, err := ReadFileRange(FileReadParams{
		Path:         path,
		DefaultLines: 2000,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.TotalLines != 1 {
		t.Errorf("TotalLines = %d, want 1", result.TotalLines)
	}
	if result.Content != "\n" {
		t.Errorf("Content = %q, want %q", result.Content, "\n")
	}
}

func TestReadFileRange_StartLineBeyondEOF(t *testing.T) {
	var sb strings.Builder
	for i := 1; i <= 50; i++ {
		if i > 1 {
			sb.WriteByte('\n')
		}
		fmt.Fprintf(&sb, "Line %d", i)
	}
	path := writeTempFile(t, "fifty.txt", sb.String())

	result, err := ReadFileRange(FileReadParams{
		Path:      path,
		StartLine: 200,
		EndLine:   300,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.TotalLines != 50 {
		t.Errorf("TotalLines = %d, want 50", result.TotalLines)
	}
	if result.Content != "" {
		t.Errorf("Content = %q, want empty (start beyond EOF)", result.Content)
	}
}

func TestReadFileRange_MaxLineBytes(t *testing.T) {
	longLine := strings.Repeat("a", 5000)
	path := writeTempFile(t, "longline.txt", longLine)

	result, err := ReadFileRange(FileReadParams{
		Path:         path,
		DefaultLines: 2000,
		MaxLineBytes: 100,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(result.Content, "[...line 1 truncated at 100 bytes. Use tool_result_read(hash, line=1) to read the full line...]") {
		t.Errorf("expected line truncation marker, got: %s", result.Content)
	}
}

func TestReadFileRange_MaxLineBytesPreservesNewline(t *testing.T) {
	longLine := strings.Repeat("a", 5000)
	content := longLine + "\nshort\n"
	path := writeTempFile(t, "longwithnl.txt", content)

	result, err := ReadFileRange(FileReadParams{
		Path:         path,
		DefaultLines: 2000,
		MaxLineBytes: 100,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(result.Content, "[...line 1 truncated at 100 bytes. Use tool_result_read(hash, line=1) to read the full line...]") {
		t.Errorf("expected truncation marker in content")
	}
	if !strings.Contains(result.Content, "\nshort\n") {
		t.Errorf("expected 'short' line after truncated line, got: %q", result.Content)
	}
}

func TestReadFileRange_MaxWindowLines(t *testing.T) {
	var sb strings.Builder
	for i := 1; i <= 100000; i++ {
		if i > 1 {
			sb.WriteByte('\n')
		}
		fmt.Fprintf(&sb, "Line %d", i)
	}
	path := writeTempFile(t, "huge.txt", sb.String())

	result, err := ReadFileRange(FileReadParams{
		Path:           path,
		StartLine:      1,
		EndLine:        100000,
		MaxWindowLines: 1000,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.WindowCapped {
		t.Error("expected WindowCapped = true")
	}
	if result.EndLine != 1000 {
		t.Errorf("EndLine = %d, want 1000 (capped)", result.EndLine)
	}
	if n := countContentLines(result.Content); n != 1000 {
		t.Errorf("content has %d lines, want 1000", n)
	}
	if result.TotalLines != 100000 {
		t.Errorf("TotalLines = %d, want 100000", result.TotalLines)
	}
}

func TestReadFileRange_LargeFileNoOOM(t *testing.T) {
	var sb strings.Builder
	for i := 1; i <= 100000; i++ {
		if i > 1 {
			sb.WriteByte('\n')
		}
		fmt.Fprintf(&sb, "Line %d with some content to make it realistic", i)
	}
	path := writeTempFile(t, "big.txt", sb.String())

	result, err := ReadFileRange(FileReadParams{
		Path:         path,
		DefaultLines: 2000,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.TotalLines != 100000 {
		t.Errorf("TotalLines = %d, want 100000", result.TotalLines)
	}
	if n := countContentLines(result.Content); n != 2000 {
		t.Errorf("content has %d lines, want 2000", n)
	}
}

func TestReadFileRange_FastCountAfterWindow(t *testing.T) {
	var sb strings.Builder
	for i := 1; i <= 50000; i++ {
		if i > 1 {
			sb.WriteByte('\n')
		}
		fmt.Fprintf(&sb, "Line %d", i)
	}
	path := writeTempFile(t, "fiftyk.txt", sb.String())

	result, err := ReadFileRange(FileReadParams{
		Path:         path,
		DefaultLines: 100,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.TotalLines != 50000 {
		t.Errorf("TotalLines = %d, want 50000 (fast count after window)", result.TotalLines)
	}
	if n := countContentLines(result.Content); n != 100 {
		t.Errorf("content has %d lines, want 100", n)
	}
}

func TestReadFileRange_NonExistentFile(t *testing.T) {
	_, err := ReadFileRange(FileReadParams{
		Path:         "/nonexistent/path/file.txt",
		DefaultLines: 2000,
	})
	if err == nil {
		t.Error("expected error for non-existent file")
	}
}

// ReadSingleLine is the escape-hatch behind tool_result_read's line parameter.
// It must return the FULL raw line (bypassing MaxLineBytes), report the total
// line count, and error when the requested line is past the end of the file.
func TestReadSingleLine(t *testing.T) {
	longLine := strings.Repeat("a", 5000) // well above a typical MaxLineBytes cap
	content := "header\n" + longLine + "\nshort\n"
	path := writeTempFile(t, "escape.txt", content)

	// Line 2 is the pathological 5000-byte line.
	line, totalLines, err := ReadSingleLine(path, 2)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if totalLines != 3 {
		t.Errorf("totalLines = %d, want 3", totalLines)
	}
	if len(line) != 5000 {
		t.Errorf("len(line) = %d, want full 5000 bytes (escape-hatch bypasses MaxLineBytes)", len(line))
	}
	if line != longLine {
		t.Error("line content mismatch: expected the full long line")
	}

	// A normal line is returned verbatim (no trailing newline).
	got, _, err := ReadSingleLine(path, 3)
	if err != nil {
		t.Fatalf("unexpected error reading line 3: %v", err)
	}
	if got != "short" {
		t.Errorf("line 3 = %q, want %q", got, "short")
	}

	// Past end of file returns an error naming the bounds.
	if _, _, err := ReadSingleLine(path, 99); err == nil {
		t.Error("expected error for line past end of file")
	}

	// Invalid line number.
	if _, _, err := ReadSingleLine(path, 0); err == nil {
		t.Error("expected error for lineNum <= 0")
	}
}

// TestReadFileRange_SingleHugeLineBounded is the regression repro for review
// finding #19: a file whose entire content is ONE line far above the per-line
// cap must still read successfully with the truncation marker, and the
// emitted output must be bounded by the cap — the whole line must never be
// materialized just to be truncated afterwards.
func TestReadFileRange_SingleHugeLineBounded(t *testing.T) {
	const lineCap = 64 << 10           // 64 KiB
	huge := strings.Repeat("x", 8<<20) // single 8 MiB line, no newline
	path := writeTempFile(t, "hugeline.bin", huge)

	result, err := ReadFileRange(FileReadParams{
		Path:         path,
		DefaultLines: 2000,
		MaxLineBytes: lineCap,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.TotalLines != 1 {
		t.Errorf("TotalLines = %d, want 1", result.TotalLines)
	}
	if !strings.Contains(result.Content, "[...line 1 truncated at 65536 bytes. Use tool_result_read(hash, line=1) to read the full line...]") {
		t.Errorf("expected truncation marker, got suffix: %q", tail(result.Content, 200))
	}
	if result.BytesRead > lineCap+200 {
		t.Errorf("emitted content is %d bytes, want bounded by cap %d (+marker)", result.BytesRead, lineCap)
	}
}

// TestReadFileRange_HugeLineBeforeWindow proves the bounded scan keeps line
// accounting correct when an over-cap line sits BEFORE the requested window:
// the huge line is drained (not accumulated) and line 2 is returned intact.
func TestReadFileRange_HugeLineBeforeWindow(t *testing.T) {
	const lineCap = 16 << 10
	content := strings.Repeat("y", 4<<20) + "\n" + "second line\n"
	path := writeTempFile(t, "hugelinefirst.txt", content)

	result, err := ReadFileRange(FileReadParams{
		Path:         path,
		StartLine:    2,
		EndLine:      2,
		DefaultLines: 1,
		MaxLineBytes: lineCap,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.TotalLines != 2 {
		t.Errorf("TotalLines = %d, want 2", result.TotalLines)
	}
	if result.Content != "second line\n" {
		t.Errorf("window content = %q, want %q", result.Content, "second line\n")
	}
}

// TestReadSingleLine_LargeLineRecoverable proves the tool_result_read escape
// hatch still recovers a multi-megabyte line IN FULL through the bounded
// scanner (the line exceeds the 1 MiB default read cap but stays far below
// the 64 MiB absolute ceiling).
func TestReadSingleLine_LargeLineRecoverable(t *testing.T) {
	payload := strings.Repeat("z", 4<<20)
	path := writeTempFile(t, "recoverable.txt", payload+"\nlast\n")

	line, total, err := ReadSingleLine(path, 1)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if line != payload {
		t.Errorf("recovered line length = %d, want %d (content mismatch)", len(line), len(payload))
	}
	if total != 2 {
		t.Errorf("total = %d, want 2", total)
	}
}

// tail returns at most n trailing bytes of s — helper for bounded-output
// assertions that check the marker at the end of the window.
func tail(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[len(s)-n:]
}

// TestReadLineBounded_BoundedRetention pins the scan-level contract of
// readLineBounded directly: with a reader buffer far smaller than the line,
// a line far above the cap is retained only up to the cap, flagged truncated,
// and the delimiter is consumed so the next line stays in sync — a revert to
// a buffer-the-whole-line ReadString would fail the retention and sync
// assertions.
func TestReadLineBounded_BoundedRetention(t *testing.T) {
	const capBytes = 8 << 10
	// Reader buffer (512 B) ≪ cap ≪ line: the scan must fragment, retain at
	// most the cap, and drain the rest fragment-by-fragment.
	input := strings.Repeat("a", 1<<20) + "\nsecond\n"
	r := bufio.NewReaderSize(strings.NewReader(input), 512)

	ln, err := readLineBounded(r, capBytes)
	if err != nil {
		t.Fatalf("readLineBounded: %v", err)
	}
	if !ln.truncated {
		t.Error("expected truncated=true for a line above the cap")
	}
	if !ln.complete {
		t.Error("expected complete=true: the delimiter was consumed")
	}
	if len(ln.text) != capBytes {
		t.Errorf("retained %d bytes, want exactly the cap %d", len(ln.text), capBytes)
	}
	if got := strings.TrimSuffix(ln.text, "\n"); got != strings.Repeat("a", capBytes) {
		t.Error("retained text is not the line's first cap bytes")
	}

	// Delimiter sync: the next read must return the SECOND line, not the
	// undrained remainder of the first.
	ln2, err := readLineBounded(r, capBytes)
	if err != nil {
		t.Fatalf("readLineBounded (second line): %v", err)
	}
	if ln2.truncated || ln2.text != "second\n" {
		t.Errorf("second line = %q (truncated=%v), want %q", ln2.text, ln2.truncated, "second\n")
	}
}

// TestReadLineBounded_NoCapAccumulatesFully pins the maxBytes<=0 contract:
// the line is accumulated in full and never flagged truncated.
func TestReadLineBounded_NoCapAccumulatesFully(t *testing.T) {
	line := strings.Repeat("b", 256<<10) + "\n"
	r := bufio.NewReaderSize(strings.NewReader(line), 512)
	ln, err := readLineBounded(r, 0)
	if err != nil {
		t.Fatalf("readLineBounded: %v", err)
	}
	if ln.truncated || !ln.complete || ln.text != line {
		t.Errorf("no-cap read: truncated=%v complete=%v len=%d, want false/true/full", ln.truncated, ln.complete, len(ln.text))
	}
}

// TestReadFileRange_SingleHugeLineBoundedMemory is the allocation-side guard
// for review finding #19: reading a single 16 MiB line with a 64 KiB cap must
// allocate on the order of the cap, not the line. The pre-fix
// buffer-the-whole-line ReadString materialized the full line (with append
// growth) before truncating — an order of magnitude above this budget — so a
// silent revert fails here even though the emitted-output assertions alone
// would still pass.
func TestReadFileRange_SingleHugeLineBoundedMemory(t *testing.T) {
	const lineCap = 64 << 10
	huge := strings.Repeat("x", 16<<20)
	path := writeTempFile(t, "hugeline16m.bin", huge)

	runtime.GC()
	var before runtime.MemStats
	runtime.ReadMemStats(&before)

	result, err := ReadFileRange(FileReadParams{
		Path:         path,
		DefaultLines: 2000,
		MaxLineBytes: lineCap,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(result.Content, "truncated at 65536 bytes") {
		t.Fatalf("expected truncation marker, got suffix %q", tail(result.Content, 200))
	}

	runtime.GC()
	var after runtime.MemStats
	runtime.ReadMemStats(&after)
	// Budget: cap-sized retention, fixed reader buffers, the result string,
	// and OS read buffers — 64× the cap, yet ~8× below what materializing
	// the 16 MiB line alone would allocate.
	const budget = 4 << 20
	if delta := after.TotalAlloc - before.TotalAlloc; delta > budget {
		t.Errorf("cumulative allocations during bounded read = %d bytes, want <= %d (a buffer-the-whole-line implementation allocates the full 16 MiB line)", delta, budget)
	}
}
