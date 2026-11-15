package builtins

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"strings"
	"syscall"

	"github.com/v0lka/sp4rk/pathutil"
	"github.com/v0lka/sp4rk/safeio"
	"github.com/v0lka/sp4rk/tools"
)

// FileReadParams controls a streaming line-range read from a file.
// All line numbers are 1-based.
type FileReadParams struct {
	Path           string // absolute or relative path to the file
	StartLine      int    // 1-based; <= 0 means 1
	EndLine        int    // 1-based; <= 0 means StartLine + DefaultLines - 1
	DefaultLines   int    // window size when EndLine is not specified
	MaxLineBytes   int    // per-line byte cap; 0 = no cap
	MaxWindowLines int    // hard cap on (EndLine - StartLine + 1); 0 = no cap
}

// FileReadResult holds the output of ReadFileRange.
type FileReadResult struct {
	Content      string // extracted window (lines joined with \n)
	TotalLines   int    // total lines in the file
	StartLine    int    // resolved start line (after defaults)
	EndLine      int    // resolved end line (after defaults + hard cap, before TotalLines clamping)
	WindowCapped bool   // true if MaxWindowLines truncated the requested range
	BytesRead    int    // byte length of Content
}

// ReadFileRange reads a line range from a file using O(1) memory — only the
// requested window is buffered. TotalLines is computed in the same pass by
// counting newlines; after the window, scanning switches to a fast byte-count
// mode that does not allocate per line.
//
// Line counting convention:
//   - "a\nb\nc\n" → 3 lines
//   - "a\nb\nc"   → 3 lines (last line without trailing \n is still counted)
//   - ""          → 0 lines
//   - "\n"        → 1 line (one empty line)
//
// The caller is responsible for clamping StartLine/EndLine to TotalLines for
// display purposes; ReadFileRange returns the resolved (but unclamped) values.
func ReadFileRange(params FileReadParams) (*FileReadResult, error) {
	startLine := params.StartLine
	if startLine <= 0 {
		startLine = 1
	}

	endLine := params.EndLine
	if endLine <= 0 {
		defaultLines := params.DefaultLines
		if defaultLines <= 0 {
			defaultLines = 1
		}
		endLine = startLine + defaultLines - 1
	}

	windowCapped := false
	if params.MaxWindowLines > 0 && endLine-startLine+1 > params.MaxWindowLines {
		endLine = startLine + params.MaxWindowLines - 1
		windowCapped = true
	}

	// safeio.Open refuses a non-regular target (FIFO/socket/device) via fstat
	// on the already-open, O_NONBLOCK descriptor: a FIFO planted at the path
	// can never block the open(2) syscall, so a read_file call cannot hang.
	f, err := safeio.Open(params.Path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()

	reader := bufio.NewReaderSize(f, 64*1024)

	var builder strings.Builder
	lineNum := 0
	totalLines := 0
	pastWindow := false

	for {
		ln, rerr := readLineBounded(reader, params.MaxLineBytes)

		if ln.text != "" {
			lineNum++
			totalLines = lineNum

			if !pastWindow && lineNum >= startLine && lineNum <= endLine {
				writeLine(&builder, ln, lineNum, params.MaxLineBytes)
			}

			if lineNum > endLine {
				pastWindow = true
			}
		}

		if errors.Is(rerr, io.EOF) {
			break
		}
		if rerr != nil {
			return nil, rerr
		}

		if pastWindow {
			remaining, countErr := countRemainingNewlines(reader)
			if countErr != nil {
				return nil, countErr
			}
			totalLines += remaining
			break
		}
	}

	return &FileReadResult{
		Content:      builder.String(),
		TotalLines:   totalLines,
		StartLine:    startLine,
		EndLine:      endLine,
		WindowCapped: windowCapped,
		BytesRead:    builder.Len(),
	}, nil
}

// isPathShapeError reports whether err is an open failure caused by the path
// spelling itself rather than by permissions or file content: a missing target
// (ENOENT) or a regular file used as a directory (ENOTDIR) — exactly the two
// failures a duplicated-component spelling produces ("<root>/<root>/rest"
// misses its target; "<file>/<file>/rest" walks through a regular file).
// Permission errors (EACCES) and mid-read I/O failures are not shape errors:
// a repair attempt cannot help them, so none is made.
//
// Portability note: on Windows the POSIX ENOTDIR value never matches the raw
// Windows system codes (see tools.isInvalidPathError for the same caveat), so
// there only the missing-target case is recognized — a deliberate, safe
// degradation: the repair is skipped and the original error is returned.
func isPathShapeError(err error) bool {
	return errors.Is(err, fs.ErrNotExist) || errors.Is(err, syscall.ENOTDIR)
}

// autoCorrectPath attempts ONE containment-safe repair of a path spelling with
// duplicated components (via [pathutil.CollapseDuplicateComponents]): a
// doubled workspace prefix ("<root>/<root>/rest", typically the result of a
// model concatenating an absolute path onto the workspace root it already
// contains) or a repeated directory ("<dir>/<dir>/file") fails to open, while
// the de-duplicated spelling often addresses the intended file.
//
// The repair is containment-gated: every session root that contains the
// ORIGINAL spelling must also contain the repaired one ([pathutil.IsWithinPath]
// resolves symlinks on both sides, so a ".." climb whose buffer directory was
// removed by the collapse, or a symlink re-resolution, cannot smuggle the
// repair outside the root). A repair that would escape containment is
// forbidden — the caller fails on the original error, exactly as before. A
// path contained in no session root (an absolute path read with user
// confirmation) is never auto-corrected either: re-pointing an explicitly
// chosen outside path would be a silent authorization change. Only ONE repair
// attempt is made per failed read; there is no iterative or cascading repair.
func autoCorrectPath(ctx context.Context, original string) (string, bool) {
	repaired := pathutil.CollapseDuplicateComponents(original)
	if repaired == "" || repaired == original {
		return "", false
	}
	contained := false
	for _, root := range tools.SessionRoots(ctx) {
		inOrig, err := pathutil.IsWithinPath(root, original)
		if err != nil || !inOrig {
			continue // not this root (or undecidable — skip, fail-safe)
		}
		inRep, err := pathutil.IsWithinPath(root, repaired)
		if err != nil || !inRep {
			return "", false // escape — forbidden, fail as before
		}
		contained = true
	}
	if !contained {
		return "", false // outside every session root — never auto-corrected
	}
	return repaired, true
}

// boundedLine is one physical line as produced by readLineBounded: the
// retained text (trailing '\n' included only when that byte fit within the
// cap), whether content past the cap was drained but dropped, and whether the
// trailing '\n' was actually consumed (false for a final line without one —
// the distinction the truncation marker needs to reproduce the old line
// structure).
type boundedLine struct {
	text      string
	truncated bool
	complete  bool
}

// readLineBounded reads one '\n'-terminated line from r while accumulating at
// most maxBytes bytes of it. Peak memory for the line is therefore bounded by
// maxBytes (plus the reader's fixed buffer) no matter how long the physical
// line is: once the cap is hit the remaining fragments of the line are still
// drained — the delimiter is always consumed so the caller stays in sync —
// but no longer retained, and truncated reports that content was dropped.
// maxBytes <= 0 means no cap (the line is accumulated in full, the caller's
// documented contract).
//
// The returned error is the terminal read error: nil when the delimiter was
// found, io.EOF at end of input (with any final partial line still returned),
// or any other reader error.
func readLineBounded(r *bufio.Reader, maxBytes int) (boundedLine, error) {
	var ln boundedLine
	var buf []byte
	for {
		frag, err := r.ReadSlice('\n')
		if !ln.truncated {
			if maxBytes <= 0 || len(buf)+len(frag) <= maxBytes {
				buf = append(buf, frag...)
			} else {
				buf = append(buf, frag[:maxBytes-len(buf)]...)
				ln.truncated = true
			}
		}
		switch {
		case err == nil:
			// Delimiter found: the line is complete.
			ln.text, ln.complete = string(buf), true
			return ln, nil
		case errors.Is(err, bufio.ErrBufferFull):
			// The reader's buffer filled before the delimiter; keep
			// scanning (and draining) the next fragment.
			continue
		case errors.Is(err, io.EOF):
			ln.text = string(buf)
			return ln, io.EOF
		default:
			ln.text = string(buf)
			return ln, err
		}
	}
}

// writeLine appends a line to the builder, applying an optional per-line byte
// cap. When the line exceeded MaxLineBytes (truncated from readLineBounded),
// a marker is appended. The marker names the line number and points at the
// recovery path: tool_result_read(hash, line=lineNum) reads the full raw line
// via ReadSingleLine, bypassing the cap. The trailing newline (when the line
// was complete) is preserved after the marker.
func writeLine(builder *strings.Builder, ln boundedLine, lineNum, maxLineBytes int) {
	if maxLineBytes <= 0 || !ln.truncated {
		builder.WriteString(ln.text)
		return
	}

	builder.WriteString(strings.TrimSuffix(ln.text, "\n"))
	fmt.Fprintf(builder, "[...line %d truncated at %d bytes. Use tool_result_read(hash, line=%d) to read the full line...]",
		lineNum, maxLineBytes, lineNum)
	if ln.complete {
		builder.WriteString("\n")
	}
}

// maxRawLineBytes is the absolute ceiling for ReadSingleLine, the escape-hatch
// behind tool_result_read's line parameter. It is far above the default
// MaxLineBytes DoS guard (1 MiB) so that any realistic pathological-but-useful
// single line (e.g. a multi-megabyte minified bundle) is fully recoverable,
// while a truly adversarial multi-gigabyte line cannot exhaust memory. Default
// reads never approach this ceiling; only an explicit tool_result_read(line=N)
// request can reach it.
const maxRawLineBytes = 64 << 20 // 64 MiB

// ReadSingleLine reads exactly one line (1-based lineNum) from a file in full,
// bypassing the per-line MaxLineBytes DoS guard. It is the escape-hatch backing
// tool_result_read's line parameter: default reads (read_file and
// tool_result_read window reads) always apply MaxLineBytes, so a line longer
// than that cap is otherwise inaccessible. ReadSingleLine makes such a line
// recoverable by an explicit, limited request — one line at a time.
//
// It scans to the requested line in O(1) memory (only lineNum is buffered),
// then returns the line's full content — without the trailing newline — capped
// only by maxRawLineBytes as a last-resort memory bound. totalLines is the
// total line count of the file (computed in the same pass). An error is
// returned when lineNum is past the end of the file.
func ReadSingleLine(path string, lineNum int) (line string, totalLines int, err error) {
	if lineNum <= 0 {
		return "", 0, fmt.Errorf("lineNum must be >= 1, got %d", lineNum)
	}

	// safeio.Open refuses a non-regular target without blocking, so the
	// tool_result_read escape hatch cannot hang on a FIFO either.
	f, openErr := safeio.Open(path)
	if openErr != nil {
		return "", 0, openErr
	}
	defer func() { _ = f.Close() }()

	reader := bufio.NewReaderSize(f, 64*1024)
	current := 0

	for {
		ln, rerr := readLineBounded(reader, maxRawLineBytes)
		if ln.text != "" {
			current++
			if current == lineNum {
				full := strings.TrimSuffix(ln.text, "\n")
				if ln.truncated {
					full += fmt.Sprintf("[...raw line capped at %d bytes (absolute memory bound)...]", maxRawLineBytes)
				}
				// Continue scanning only to compute totalLines.
				remaining, countErr := countRemainingNewlines(reader)
				if countErr != nil {
					return "", 0, countErr
				}
				return full, current + remaining, nil
			}
		}
		if errors.Is(rerr, io.EOF) {
			break
		}
		if rerr != nil {
			return "", 0, rerr
		}
	}

	return "", current, fmt.Errorf("line %d is past end of file (%d lines)", lineNum, current)
}

// countRemainingNewlines counts lines in the remaining data from reader without
// allocating per line. Each \n counts as one line; if data remains after the
// last \n, it counts as one additional line (the trailing-no-newline case).
func countRemainingNewlines(r *bufio.Reader) (int, error) {
	count := 0
	sawNonNewline := false
	buf := make([]byte, 64*1024)
	for {
		n, err := r.Read(buf)
		for i := 0; i < n; i++ {
			if buf[i] == '\n' {
				count++
				sawNonNewline = false
			} else {
				sawNonNewline = true
			}
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			return 0, err
		}
	}
	if sawNonNewline {
		count++
	}
	return count, nil
}
