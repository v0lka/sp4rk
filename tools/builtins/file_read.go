package builtins

import (
	"context"
	"encoding/json"
	"fmt"
	"path/filepath"

	"github.com/v0lka/sp4rk/tools"
)

const toolReadFileDescription = `Purpose: read the contents of a text file at a given path, optionally a line range.
Use when: you already know the exact path — discover paths first with list_directory, glob or ripgrep.
Inputs: path (absolute or workspace-relative); optional start_line (1-based, inclusive) and end_line (1-based, inclusive) for pagination of large files.
Outputs: the file content with a metadata header (file name, returned range, total line count); when more content remains beyond the range, a continuation hint names the next start_line — follow it with another read_file call.
Example: start_line=100, end_line=250 reads the second chunk of a long file.
Anti-example: not for searching content by pattern (ripgrep), by meaning (semantic_search) or by file name (glob); do not read a huge file whole — page through ranges; binary and document formats (e.g. pdf, docx) are returned as-is, not converted.`

// ReadFileTool reads file contents with pagination support.
type ReadFileTool struct {
	*tools.BaseTool
	limits FileLimits
}

// NewReadFileTool creates a new ReadFileTool instance with default limits.
func NewReadFileTool() *ReadFileTool {
	return NewReadFileToolWithLimits(DefaultFileLimits())
}

// NewReadFileToolWithLimits creates a new ReadFileTool instance with specified limits.
func NewReadFileToolWithLimits(limits FileLimits) *ReadFileTool {
	return &ReadFileTool{
		BaseTool: &tools.BaseTool{
			ToolName:        "read_file",
			ToolGroup:       tools.GroupLocalRead,
			ToolDescription: toolReadFileDescription,
			Schema: json.RawMessage(`{
				"type": "object",
				"properties": {
					"path": {
						"type": "string",
						"description": "Absolute or relative path to the file to read."
					},
					"start_line": {
						"type": "integer",
						"description": "1-based line number to start reading from. Defaults to the beginning of the file."
					},
					"end_line": {
						"type": "integer",
						"description": "1-based line number to stop reading at (inclusive). Defaults to the end of file. Values beyond the file length are clamped automatically."
					}
				},
				"required": ["path"]
			}`),
			Policy:    tools.PolicyAlwaysAllow,
			Untrusted: true,
		},
		limits: limits,
	}
}

// ReadFileInput represents the input parameters for read_file.
type ReadFileInput struct {
	Path      string `json:"path"`
	StartLine int    `json:"start_line"`
	EndLine   int    `json:"end_line"`
}

// Judge checks whether the read targets a path inside the session roots.
// Reads outside workspace/temp require user confirmation.
func (t *ReadFileTool) Judge(ctx context.Context, input json.RawMessage) tools.JudgeOutcome {
	return judgeReadInSessionRoots(ctx, input)
}

// Execute reads and returns the content of a file with streaming pagination.
// Uses ReadFileRange for O(1) memory — only the requested window is buffered,
// never the full file. When no line range is specified, a default window of
// ReadDefaultLines lines is returned from the beginning of the file.
func (t *ReadFileTool) Execute(ctx context.Context, input json.RawMessage) (tools.ToolResult, error) {
	var params ReadFileInput
	if err := json.Unmarshal(input, &params); err != nil {
		return tools.ParseInputError(err)
	}

	if params.Path == "" {
		return tools.ToolResult{Content: "validation error: path is required", IsError: true}, nil
	}
	if params.StartLine < 0 {
		return tools.ToolResult{Content: fmt.Sprintf("validation error: start_line must be >= 1, got %d", params.StartLine), IsError: true}, nil
	}
	if params.EndLine < 0 {
		return tools.ToolResult{Content: fmt.Sprintf("validation error: end_line must be >= 1, got %d", params.EndLine), IsError: true}, nil
	}
	if params.StartLine > 0 && params.EndLine > 0 && params.StartLine > params.EndLine {
		return tools.ToolResult{Content: fmt.Sprintf("validation error: start_line (%d) must not exceed end_line (%d)", params.StartLine, params.EndLine), IsError: true}, nil
	}

	params.Path = resolvePath(ctx, params.Path)
	if err := validateResolvedPath(params.Path); err != nil {
		return tools.ToolResult{Content: err.Error(), IsError: true}, nil //nolint:nilerr // error embedded in ToolResult by design
	}

	// Coherence check: detect if file was modified by another session since last read.
	var coherenceWarning string
	if checker := tools.CoherenceFrom(ctx); checker != nil {
		checker.Lock(params.Path)
		if conflict := checker.CheckRead(ctx, params.Path); conflict != nil {
			coherenceWarning = formatReadConflict(conflict)
		}
		checker.Unlock(params.Path)
	}

	result, err := ReadFileRange(FileReadParams{
		Path:           params.Path,
		StartLine:      params.StartLine,
		EndLine:        params.EndLine,
		DefaultLines:   t.limits.ReadDefaultLines,
		MaxLineBytes:   t.limits.MaxLineBytes,
		MaxWindowLines: t.limits.MaxWindowLines,
	})
	if err != nil {
		return tools.ToolResult{Content: fmt.Sprintf("failed to read file: %v", err), IsError: true}, nil
	}

	filename := filepath.Base(params.Path)

	// Handle empty file.
	if result.TotalLines == 0 {
		content := fmt.Sprintf("[File: %s | 0 lines | empty file]\n", filename)
		if coherenceWarning != "" {
			content = coherenceWarning + "\n" + content
		}
		return tools.ToolResult{Content: content, IsError: false}, nil
	}

	// Clamp resolved range to file bounds for display.
	startLine := result.StartLine
	endLine := result.EndLine
	if startLine > result.TotalLines {
		startLine = result.TotalLines
	}
	if endLine > result.TotalLines {
		endLine = result.TotalLines
	}
	if startLine < 1 {
		startLine = 1
	}

	header := fmt.Sprintf("[File: %s | Lines %d-%d of %d | %d bytes]\n", filename, startLine, endLine, result.TotalLines, len(result.Content))

	content := result.Content

	// Add continuation hint if more lines remain.
	if endLine < result.TotalLines {
		content = header + content + fmt.Sprintf("\n[Use start_line=%d to continue reading]", endLine+1)
	} else {
		content = header + content
	}

	if coherenceWarning != "" {
		content = coherenceWarning + "\n" + content
	}

	return tools.ToolResult{Content: content, IsError: false}, nil
}
