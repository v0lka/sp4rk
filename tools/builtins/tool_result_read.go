package builtins

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/v0lka/sp4rk/agent"
	"github.com/v0lka/sp4rk/tools"
)

const toolResultReadDescription = `Purpose: re-read fragments of a previously truncated tool result from the output cache — without re-executing the original tool.
Use when: a tool result was cut off and ended with a truncation nudge like "[This output was truncated ... hash: abc123 ...]". Pass that hash with a line range to retrieve exactly the fragment you need.
Inputs: hash (from the truncation nudge); start_line (1-based); num_lines (bounded, e.g. <=2000); optional line (single-line escape hatch: returns the FULL raw content of one line, bypassing the per-line size cap — the only way to recover a line shown as "[...line N truncated...]"; takes precedence over the range).
Outputs: the requested fragment of the cached result.
Example: hash "abc123", start_line 1, num_lines 100 — the first hundred lines of a truncated listing.
Anti-example: never re-run the original tool just to see more of a truncated output — the cache already holds it; do not invent a hash — only nudges carry valid ones.`

const defaultResultReadLines = 500

// ToolResultReadTool reads fragments of cached tool results by hash.
type ToolResultReadTool struct {
	*tools.BaseTool
}

// NewToolResultReadTool creates a new ToolResultReadTool instance.
func NewToolResultReadTool() *ToolResultReadTool {
	return &ToolResultReadTool{BaseTool: &tools.BaseTool{
		ToolName:        "tool_result_read",
		ToolGroup:       tools.GroupSystem,
		ToolDescription: toolResultReadDescription,
		Schema: json.RawMessage(`{
		"type": "object",
		"properties": {
			"hash": {
				"type": "string",
				"description": "The short cache hash from the truncation nudge message, e.g. \"abc123\". A unique prefix of the full hash is also accepted."
			},
			"start_line": {
				"type": "integer",
				"description": "1-based line number to start reading from. Defaults to 1."
			},
			"num_lines": {
				"type": "integer",
				"description": "Maximum number of lines to return. Defaults to 500."
			},
			"line": {
				"type": "integer",
				"description": "Escape-hatch: return the FULL raw content of a single line (1-based line number), bypassing the per-line MaxLineBytes cap. Use this to recover a line truncated with a '[...line N truncated...]' marker. Takes precedence over start_line/num_lines; only one line is returned."
			}
		},
		"required": ["hash"]
	}`),
		// ASI01: re-surfaces previously cached tool output, which may be
		// untrusted data from an earlier external read. Preserve the wrap.
		Untrusted: true,
		Policy:    tools.PolicyAlwaysAllow,
	}}
}

// toolResultReadInput represents the input parameters for tool_result_read.
type toolResultReadInput struct {
	Hash      string `json:"hash"`
	StartLine int    `json:"start_line"`
	NumLines  int    `json:"num_lines"`
	Line      int    `json:"line"`
}

// Execute retrieves the cached result fragment. For file-backed entries
// (read_file), fragments are streamed from disk via ReadFileRange — O(1)
// memory, no content stored in the cache. For content-backed entries,
// fragments are extracted from the cached string.
func (t *ToolResultReadTool) Execute(ctx context.Context, input json.RawMessage) (tools.ToolResult, error) {
	var params toolResultReadInput
	if err := json.Unmarshal(input, &params); err != nil {
		return tools.ParseInputError(err)
	}

	if params.Hash == "" {
		return tools.ToolResult{Content: "validation error: hash is required", IsError: true}, nil
	}

	cache := agent.ToolResultCacheFromContext(ctx)
	if cache == nil {
		return tools.ErrorResult("Tool result cache not available"), nil
	}

	entry, ok := cache.Get(params.Hash)
	if !ok {
		return tools.ErrorResult("No cached result found for hash: %s. The cache entry may have expired or the hash is incorrect.", params.Hash), nil
	}

	// Coherence check for file-based tools.
	if entry.FilePath != "" {
		valid, reason := cache.CheckCoherence(params.Hash)
		if !valid {
			return tools.ErrorResult("Cached result for '%s' is stale: %s. Re-run the original tool to obtain fresh data.", entry.ToolName, reason), nil
		}
	}

	// Apply defaults.
	if params.StartLine <= 0 {
		params.StartLine = 1
	}
	if params.NumLines <= 0 {
		params.NumLines = defaultResultReadLines
	}

	// Enforce num_lines upper bound from per-tool truncation config.
	// The LLM is told in the nudge to keep num_lines <= MaxLines, but we enforce it server-side.
	// When per-tool config exists, its MaxLines IS the ceiling; the default
	// hard cap applies only as a fallback when no per-tool config is available.
	capped := false
	if perToolCfg := agent.PerToolTruncationFromContext(ctx); perToolCfg != nil {
		if cfg, ok := perToolCfg[entry.ToolName]; ok && cfg.MaxLines > 0 {
			if params.NumLines > cfg.MaxLines {
				params.NumLines = cfg.MaxLines
			}
			capped = true
		}
	}
	if !capped && params.NumLines > defaultResultReadLines {
		params.NumLines = defaultResultReadLines
	}

	// Escape-hatch: return the FULL raw content of a single line, bypassing the
	// per-line MaxLineBytes DoS guard. The default read path (read_file and this
	// tool's own window reads) always applies MaxLineBytes, so a line truncated
	// by that guard is otherwise inaccessible. This is the only way to recover
	// it — reachable solely via an explicit line=N request.
	if params.Line > 0 {
		return t.readSingleLine(entry, params)
	}

	// File-backed: stream fragments from disk.
	if entry.FileBacked {
		return t.readFromFileBacked(entry, params)
	}

	// Content-backed: extract from cached string.
	allLines := strings.Split(entry.Content, "\n")
	totalLines := len(allLines)

	// Clamp start_line.
	if params.StartLine > totalLines {
		params.StartLine = totalLines
	}

	endLine := params.StartLine + params.NumLines - 1
	if endLine > totalLines {
		endLine = totalLines
	}

	selectedLines := allLines[params.StartLine-1 : endLine]
	fragment := strings.Join(selectedLines, "\n")

	var sb strings.Builder
	fmt.Fprintf(&sb, "[Lines %d-%d of %d from cached %s result | hash: %s]\n",
		params.StartLine, endLine, totalLines, entry.ToolName, params.Hash)
	sb.WriteString(fragment)

	// Add continuation nudge if more lines are available.
	if endLine < totalLines {
		fmt.Fprintf(&sb, "\n\n[Use tool_result_read(hash=\"%s\", start_line=%d, num_lines=%d) to continue reading]",
			params.Hash, endLine+1, params.NumLines)
	}

	return tools.ToolResult{Content: sb.String()}, nil
}

// readFromFileBacked streams a fragment from the file on disk using
// ReadFileRange. O(1) memory — only the requested window is buffered.
func (t *ToolResultReadTool) readFromFileBacked(entry *agent.ToolResultCacheEntry, params toolResultReadInput) (tools.ToolResult, error) {
	defaults := DefaultFileLimits()
	result, err := ReadFileRange(FileReadParams{
		Path:           entry.FilePath,
		StartLine:      params.StartLine,
		EndLine:        params.StartLine + params.NumLines - 1,
		MaxLineBytes:   defaults.MaxLineBytes,
		MaxWindowLines: defaults.MaxWindowLines,
	})
	if err != nil {
		return tools.ToolResult{Content: fmt.Sprintf("failed to read cached file: %v", err), IsError: true}, nil
	}

	totalLines := result.TotalLines

	if totalLines == 0 {
		return tools.ToolResult{Content: fmt.Sprintf("[Cached %s result is an empty file | hash: %s]", entry.ToolName, params.Hash)}, nil
	}

	// Clamp for display.
	startLine := params.StartLine
	endLine := params.StartLine + params.NumLines - 1
	if startLine > totalLines {
		startLine = totalLines
	}
	if endLine > totalLines {
		endLine = totalLines
	}

	var sb strings.Builder
	fmt.Fprintf(&sb, "[Lines %d-%d of %d from cached %s result | hash: %s]\n",
		startLine, endLine, totalLines, entry.ToolName, params.Hash)
	sb.WriteString(result.Content)

	if endLine < totalLines {
		fmt.Fprintf(&sb, "\n\n[Use tool_result_read(hash=\"%s\", start_line=%d, num_lines=%d) to continue reading]",
			params.Hash, endLine+1, params.NumLines)
	}

	return tools.ToolResult{Content: sb.String()}, nil
}

// readSingleLine returns the full raw content of one line, bypassing the
// per-line MaxLineBytes cap. For file-backed entries it reads the line directly
// from disk via ReadSingleLine (true recovery of the raw line); for
// content-backed entries it indexes into the cached string (the cached
// representation, which for converted documents may already reflect an earlier
// window/truncation). This is the escape-hatch for lines truncated by the
// default read path and is only reachable via an explicit line=N request —
// default window reads always apply MaxLineBytes.
func (t *ToolResultReadTool) readSingleLine(entry *agent.ToolResultCacheEntry, params toolResultReadInput) (tools.ToolResult, error) {
	// File-backed: stream the single full line straight from disk.
	if entry.FileBacked {
		line, totalLines, err := ReadSingleLine(entry.FilePath, params.Line)
		if err != nil {
			return tools.ToolResult{Content: fmt.Sprintf(
				"[Line %d not available from cached %s result: %v | hash: %s]",
				params.Line, entry.ToolName, err, params.Hash)}, nil
		}
		return tools.ToolResult{Content: fmt.Sprintf(
			"[Full line %d of %d from cached %s result | hash: %s]\n%s",
			params.Line, totalLines, entry.ToolName, params.Hash, line)}, nil
	}

	// Content-backed: index into the cached string (uncapped relative to the
	// cache — the cached content is returned verbatim, no MaxLineBytes applied).
	allLines := strings.Split(entry.Content, "\n")
	totalLines := len(allLines)
	if params.Line > totalLines {
		return tools.ToolResult{Content: fmt.Sprintf(
			"[Line %d is past end of cached %s result (%d lines) | hash: %s]",
			params.Line, entry.ToolName, totalLines, params.Hash)}, nil
	}
	return tools.ToolResult{Content: fmt.Sprintf(
		"[Line %d of %d from cached %s result | hash: %s]\n%s",
		params.Line, totalLines, entry.ToolName, params.Hash, allLines[params.Line-1])}, nil
}
