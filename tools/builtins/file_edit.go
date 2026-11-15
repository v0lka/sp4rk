package builtins

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"unicode/utf8"

	"github.com/v0lka/sp4rk/safeio"
	"github.com/v0lka/sp4rk/tools"
)

const toolEditFileDescription = `Purpose: find-and-replace exactly one occurrence of a substring in an existing file — the surgical alternative to rewriting the whole file.
Use when: modifying a specific fragment of an existing file — read (or re-read) the file immediately before editing and copy old_string verbatim (whitespace, tabs) from that read. old_string must occur exactly once — include surrounding lines for uniqueness; zero or multiple matches fail loudly instead of guessing.
Inputs: path — required; old_string — required (the exact text to replace, with enough context to be unique); new_string — required (the replacement; empty string deletes the matched text).
Outputs: confirmation of the edit, or an error stating the match was not found / matched more than once.
Example: to rename a function, put its signature plus one neighboring line into old_string and the renamed signature into new_string.
Anti-example: not for creating files or full rewrites (write_file); one call fixes one occurrence — for bulk renames across many files ask the user or use a scripted change via the shell tool; on a "file changed since read" conflict error, re-read the file and rebuild old_string before retrying.`

// EditFileTool performs find-and-replace edits on files.
type EditFileTool struct {
	*tools.BaseTool
	limits FileLimits
}

// NewEditFileTool creates a new EditFileTool instance with default limits.
func NewEditFileTool() *EditFileTool {
	return NewEditFileToolWithLimits(DefaultFileLimits())
}

// NewEditFileToolWithLimits creates a new EditFileTool instance with the
// given limits. MaxEditFileBytes caps the size of a file the tool will read
// and rewrite: edit_file slurps the entire target into memory (and makes two
// to three more transient copies of its content while replacing), so the cap
// keeps an edit's peak memory proportional to a known, configured file
// ceiling — in-workspace writes are auto-executed, before the executor's
// central truncation layer can intervene. safeio.ReadFile already bounds
// whole-file reads at its own DefaultMaxFileSize (10 MiB) and refuses
// non-regular targets; the edit-specific cap (default 64 MiB) deliberately
// raises that read ceiling for edits while adding a cheap stat guard that
// refuses an oversized target with a clear error before any slurp. The cap
// bounds both the pre-read stat guard and the read itself
// (safeio.ReadFileLimited), so a target that grows between the stat and the
// read is still refused. 0 disables the edit-specific cap; safeio.ReadFile
// then still enforces its own DefaultMaxFileSize read bound.
func NewEditFileToolWithLimits(limits FileLimits) *EditFileTool {
	return &EditFileTool{
		BaseTool: &tools.BaseTool{
			ToolName:        "edit_file",
			ToolGroup:       tools.GroupLocalWrite,
			ToolDescription: toolEditFileDescription,
			Schema: json.RawMessage(`{
				"type": "object",
				"properties": {
					"path": {
						"type": "string",
						"description": "Absolute or relative path to the file to edit. The file must already exist."
					},
					"old_string": {
						"type": "string",
						"description": "The exact substring to find in the file. Must appear exactly once; the operation fails if zero or multiple matches are found. Include sufficient surrounding context to guarantee uniqueness."
					},
					"new_string": {
						"type": "string",
						"description": "The replacement text that will replace old_string. Can be empty to delete the matched text."
					}
				},
				"required": ["path", "old_string", "new_string"]
			}`),
			Policy: tools.PolicyUserConfirm,
		},
		limits: limits,
	}
}

// EditFileInput represents the input parameters for edit_file.
type EditFileInput struct {
	Path      string `json:"path"`
	OldString string `json:"old_string"`
	NewString string `json:"new_string"`
}

// Judge uses session roots check for write operations.
func (t *EditFileTool) Judge(ctx context.Context, input json.RawMessage) tools.JudgeOutcome {
	var params EditFileInput
	if err := json.Unmarshal(input, &params); err != nil {
		return tools.JudgeOutcome{}
	}
	params.Path = resolvePath(ctx, params.Path)
	if err := validateResolvedPath(params.Path); err != nil {
		return softOutcome(false, err.Error(), tools.ReasonCodeOutsideSessionRoots)
	}
	return judgeWriteInSessionRoots(ctx, params.Path)
}

// atomicWriteFile writes data to path atomically: the content is written to a
// temporary file in the same directory, then renamed over the target. This
// guarantees readers never observe a partially-written file. The original
// file's permission bits are preserved (falling back to 0o644 when the file
// cannot be stat'ed).
//
// It is FIFO-safe by construction: the target at path is never open()ed — a
// fresh regular temporary file is created next to it (os.CreateTemp) and
// renamed over the target (os.Rename replaces the directory entry without
// opening it). A FIFO planted at path therefore cannot block this write; the
// rename simply replaces it with the new regular file. The os.Stat above is
// used only to copy permission bits and never blocks on a FIFO.
func atomicWriteFile(path string, data []byte) error {
	perm := os.FileMode(0o644)
	if info, err := os.Stat(path); err == nil {
		perm = info.Mode().Perm()
	}

	dir := filepath.Dir(path)
	tmp, err := os.CreateTemp(dir, "."+filepath.Base(path)+".tmp-*")
	if err != nil {
		return fmt.Errorf("create temp file: %w", err)
	}
	tmpName := tmp.Name()
	defer func() {
		// Best-effort cleanup on failure; no-op after successful rename.
		_ = os.Remove(tmpName)
	}()

	if _, err := tmp.Write(data); err != nil {
		_ = tmp.Close()
		return fmt.Errorf("write temp file: %w", err)
	}
	if err := tmp.Chmod(perm); err != nil {
		_ = tmp.Close()
		return fmt.Errorf("chmod temp file: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("close temp file: %w", err)
	}
	if err := os.Rename(tmpName, path); err != nil {
		return fmt.Errorf("rename temp file: %w", err)
	}
	return nil
}

// Execute performs ACI-style find-and-replace in a file.
func (t *EditFileTool) Execute(ctx context.Context, input json.RawMessage) (tools.ToolResult, error) {
	var params EditFileInput
	if err := json.Unmarshal(input, &params); err != nil {
		return tools.ParseInputError(err)
	}

	if params.Path == "" {
		return tools.ToolResult{Content: "validation error: path is required", IsError: true}, nil
	}
	if params.OldString == "" {
		return tools.ToolResult{Content: "validation error: old_string is required and must not be empty", IsError: true}, nil
	}

	params.Path = resolvePath(ctx, params.Path)
	if err := validateResolvedPath(params.Path); err != nil {
		return tools.ToolResult{Content: err.Error(), IsError: true}, nil
	}

	// If the target is a symlink, resolve it and write to the resolved path.
	// This preserves write-through-symlink semantics: an atomic rename on the
	// symlink path itself would replace the symlink instead of its target.
	// Symlink safety is gated above this layer (Judge / registry symlink checks).
	writePath := params.Path
	if resolved, evalErr := filepath.EvalSymlinks(params.Path); evalErr == nil {
		writePath = resolved
	}

	// Coherence check: block edit if file was modified since this session's last read.
	checker := tools.CoherenceFrom(ctx)
	if checker != nil {
		checker.Lock(writePath)
		if conflict := checker.CheckWrite(ctx, writePath); conflict != nil {
			checker.Unlock(writePath)
			return tools.ToolResult{Content: formatWriteConflict(conflict), IsError: true}, nil
		}
		defer checker.Unlock(writePath)
	}

	// Size guard before the slurping read: edit_file holds the whole file
	// (plus transient copies) in memory, and in-workspace writes are
	// auto-executed before the executor's central truncation layer can
	// intervene. A stat failure here is deliberately ignored — the read
	// below reports a missing target with its own error.
	var data []byte
	var readErr error
	if t.limits.MaxEditFileBytes > 0 {
		if info, statErr := os.Stat(writePath); statErr == nil && info.Size() > int64(t.limits.MaxEditFileBytes) {
			return tools.ToolResult{
				Content: fmt.Sprintf("refusing to edit: file is %d bytes, exceeding the %d byte edit limit: %s",
					info.Size(), t.limits.MaxEditFileBytes, writePath),
				IsError: true,
			}, nil
		}
		// The read enforces the same bound (safeio.ReadFileLimited), so a
		// target that grows between the stat and the read is still refused.
		data, readErr = safeio.ReadFileLimited(writePath, int64(t.limits.MaxEditFileBytes))
	} else {
		// Cap disabled: safeio.ReadFile still refuses non-regular targets
		// (FIFO/socket/device) via fstat on the O_NONBLOCK descriptor and
		// enforces its own DefaultMaxFileSize read bound.
		data, readErr = safeio.ReadFile(writePath)
	}
	if readErr != nil {
		return tools.ToolResult{Content: fmt.Sprintf("failed to read file: %v", readErr), IsError: true}, nil
	}

	content := string(data)
	count := strings.Count(content, params.OldString)

	if count == 0 {
		msg := "old_string not found in file"
		if hint := nearestMatchHint(content, params.OldString); hint != "" {
			msg += "\n" + hint
		}
		return tools.ToolResult{Content: msg, IsError: true}, nil
	}

	if count > 1 {
		return tools.ToolResult{Content: fmt.Sprintf("old_string is not unique, found %d occurrences", count), IsError: true}, nil
	}

	newContent := strings.Replace(content, params.OldString, params.NewString, 1)

	if err := atomicWriteFile(writePath, []byte(newContent)); err != nil {
		return tools.ToolResult{Content: fmt.Sprintf("failed to write file: %v", err), IsError: true}, nil
	}

	if checker != nil {
		checker.RecordWrite(ctx, writePath)
	}

	return tools.ToolResult{Content: "successfully edited file", IsError: false}, nil
}

// Bounds for the nearest-match hint on an old_string-not-found error. Every
// value exists to keep the hint a bounded, single-fragment suggestion — never
// a full-file dump — regardless of how large the edited file is (the edit cap
// allows tens of MiB).
const (
	hintAnchorMaxBytes   = 200  // cap on the anchor (first meaningful old_string line)
	hintFragmentMaxBytes = 160  // cap on the rendered fragment
	hintLineScanBytes    = 4096 // per-line examination cap during the scan
	hintMaxTokens        = 16   // tokens considered per side
	hintTokenMaxBytes    = 64   // per-token cap
	hintMinScore         = 0.5  // minimum token overlap for a hint to be emitted
)

// nearestMatchHint builds a bounded nearest-match fragment hint for an
// old_string-not-found error. The anchor is old_string's first meaningful
// line, whitespace-normalized and capped; the file is scanned line by line
// (each line capped at hintLineScanBytes) for the line whose whitespace-
// normalized token set overlaps the anchor's the most. When the best overlap
// reaches hintMinScore, the winning line — whitespace-collapsed and capped at
// hintFragmentMaxBytes — is returned as a one-line hint naming its 1-based
// line number. The failure mode this targets is a near miss: an indentation,
// whitespace or small drift difference between the read text and the copied
// old_string. A dissimilar old_string (score below the threshold) yields ""
// and the error stays exactly as it was.
//
// The scan is O(file size): tokens are extracted into a reused scratch slice,
// compared with a bounded nested loop, and no per-line string is materialized
// except when the running best improves.
func nearestMatchHint(content, oldString string) string {
	anchor := truncateRunes(collapseASCIISpace(firstNonEmptyLine(oldString)), hintAnchorMaxBytes)
	if anchor == "" {
		return ""
	}
	anchorTokens := appendTokens(nil, anchor, hintMaxTokens, hintTokenMaxBytes)
	if len(anchorTokens) == 0 {
		return ""
	}

	bestScore := 0.0
	bestLine := 0
	bestFragment := ""
	scratch := make([]string, 0, hintMaxTokens)
	lineNum := 0
	for line := range strings.Lines(content) {
		lineNum++
		capped := truncateRunes(line, hintLineScanBytes)
		scratch = appendTokens(scratch[:0], capped, hintMaxTokens, hintTokenMaxBytes)
		score := tokenOverlapScore(anchorTokens, scratch)
		if score > bestScore {
			bestScore = score
			bestLine = lineNum
			bestFragment = truncateRunes(collapseASCIISpace(capped), hintFragmentMaxBytes)
		}
	}
	if bestLine == 0 || bestScore < hintMinScore {
		return ""
	}
	return fmt.Sprintf("[nearest match: line %d: %q]", bestLine, bestFragment)
}

// firstNonEmptyLine returns the first line of s containing at least one
// non-whitespace character (trimmed), or "" when there is none.
func firstNonEmptyLine(s string) string {
	for line := range strings.Lines(s) {
		if trimmed := strings.TrimSpace(line); trimmed != "" {
			return trimmed
		}
	}
	return ""
}

// collapseASCIISpace collapses every run of ASCII whitespace into a single
// space and trims the ends. Non-ASCII bytes pass through untouched.
func collapseASCIISpace(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	pendingSpace := false
	for i := 0; i < len(s); i++ {
		c := s[i]
		if isASCIISpaceByte(c) {
			pendingSpace = true
			continue
		}
		if pendingSpace && b.Len() > 0 {
			b.WriteByte(' ')
		}
		pendingSpace = false
		b.WriteByte(c)
	}
	return b.String()
}

// isASCIISpaceByte reports whether c is an ASCII whitespace byte.
func isASCIISpaceByte(c byte) bool {
	switch c {
	case ' ', '\t', '\n', '\r', '\v', '\f':
		return true
	}
	return false
}

// appendTokens appends up to max whitespace-separated tokens of s to dst,
// each truncated to tokenMax bytes (a mid-token cut may split a UTF-8
// sequence; tokens are only ever compared against equally capped tokens, so
// the heuristic stays deterministic).
func appendTokens(dst []string, s string, maxTokens, tokenMax int) []string {
	for len(dst) < maxTokens {
		i := 0
		for i < len(s) && isASCIISpaceByte(s[i]) {
			i++
		}
		s = s[i:]
		if s == "" {
			break
		}
		j := 0
		for j < len(s) && !isASCIISpaceByte(s[j]) {
			j++
		}
		tok := s[:j]
		if len(tok) > tokenMax {
			tok = tok[:tokenMax]
		}
		dst = append(dst, tok)
		s = s[j:]
	}
	return dst
}

// tokenOverlapScore scores two token slices as their overlap fraction:
// (tokens of a present in b) / (union size), 0..1. Duplicates are not
// deduplicated — the heuristic only needs a stable ordering signal.
func tokenOverlapScore(a, b []string) float64 {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}
	inter := 0
	for _, t := range a {
		if containsToken(b, t) {
			inter++
		}
	}
	union := len(a) + len(b) - inter
	if union == 0 {
		return 0
	}
	return float64(inter) / float64(union)
}

// containsToken reports whether toks contains t.
func containsToken(toks []string, t string) bool {
	for _, x := range toks {
		if x == t {
			return true
		}
	}
	return false
}

// truncateRunes cuts s to at most maxBytes without splitting a UTF-8
// sequence: the cut point backs up to a rune boundary when needed.
func truncateRunes(s string, maxBytes int) string {
	if len(s) <= maxBytes {
		return s
	}
	cut := s[:maxBytes]
	for cut != "" && !utf8.ValidString(cut) {
		cut = cut[:len(cut)-1]
	}
	return cut
}
