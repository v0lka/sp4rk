package builtins

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/v0lka/sp4rk/tools"
)

const toolWriteFileDescription = `Purpose: create a new file or fully replace an existing one (parent directories are created automatically, like mkdir -p).
Use when: emitting a new artifact, or replacing a file's entire content deliberately. For surgical changes to an existing file prefer edit_file — it anchors on an exact match and cannot silently clobber unexpected content.
Inputs: path (target location); content (the complete new file body — this replaces everything).
Outputs: confirmation of the write, or an error (invalid path, policy denial).
Example: writing a generated report to reports/summary.md.
Anti-example: not for partial edits (edit_file); read the current file first if it exists — overwrite is silent and destructive; do not use for scratch outside the workspace.`

// WriteFileTool creates or overwrites files.
type WriteFileTool struct {
	*tools.BaseTool
}

// NewWriteFileTool creates a new WriteFileTool instance.
func NewWriteFileTool() *WriteFileTool {
	return &WriteFileTool{
		BaseTool: &tools.BaseTool{
			ToolName:        "write_file",
			ToolGroup:       tools.GroupLocalWrite,
			ToolDescription: toolWriteFileDescription,
			Schema: json.RawMessage(`{
				"type": "object",
				"properties": {
					"path": {
						"type": "string",
						"description": "Absolute or relative path to the file to create or overwrite."
					},
					"content": {
						"type": "string",
						"description": "The full text content to write to the file. Replaces any existing content entirely."
					}
				},
				"required": ["path", "content"]
			}`),
			Policy: tools.PolicyUserConfirm,
		},
	}
}

// WriteFileInput represents the input parameters for write_file.
type WriteFileInput struct {
	Path    string `json:"path"`
	Content string `json:"content"`
}

// Judge uses session roots check for write operations.
func (t *WriteFileTool) Judge(ctx context.Context, input json.RawMessage) tools.JudgeOutcome {
	var params WriteFileInput
	if err := json.Unmarshal(input, &params); err != nil {
		return tools.JudgeOutcome{}
	}
	params.Path = resolvePath(ctx, params.Path)
	if err := validateResolvedPath(params.Path); err != nil {
		return softOutcome(false, err.Error(), tools.ReasonCodeOutsideSessionRoots)
	}
	return judgeWriteInSessionRoots(ctx, params.Path)
}

// Execute writes content to a file, creating parent directories if needed.
func (t *WriteFileTool) Execute(ctx context.Context, input json.RawMessage) (tools.ToolResult, error) {
	var params WriteFileInput
	if err := json.Unmarshal(input, &params); err != nil {
		return tools.ParseInputError(err)
	}

	if params.Path == "" {
		return tools.ToolResult{Content: "validation error: path is required", IsError: true}, nil
	}

	params.Path = resolvePath(ctx, params.Path)
	if err := validateResolvedPath(params.Path); err != nil {
		return tools.ToolResult{Content: err.Error(), IsError: true}, nil
	}

	// Refuse to REPLACE a non-regular target. The atomic rename below
	// replaces any existing non-directory destination — including device
	// nodes (/dev/null, NUL — which the harmless-device exemption keeps
	// judge-local), FIFOs and sockets. Creating a NEW file at such a path is
	// unaffected: a missing target falls through to the write. os.Stat
	// (follow) is deliberate so write-through-symlink semantics are kept: a
	// link to a regular file is written through, a link to a device is
	// refused just like the device itself.
	if info, statErr := os.Stat(params.Path); statErr == nil && !info.Mode().IsRegular() {
		return tools.ToolResult{
			Content: "refusing to write: target exists and is not a regular file (directory, device node, FIFO, or socket): " + params.Path,
			IsError: true,
		}, nil
	}

	// Coherence check: block write if file was modified since this session's last read.
	checker := tools.CoherenceFrom(ctx)
	if checker != nil {
		checker.Lock(params.Path)
		if conflict := checker.CheckWrite(ctx, params.Path); conflict != nil {
			checker.Unlock(params.Path)
			return tools.ToolResult{Content: formatWriteConflict(conflict), IsError: true}, nil
		}
		defer checker.Unlock(params.Path)
	}

	dir := filepath.Dir(params.Path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return tools.ToolResult{Content: fmt.Sprintf("failed to create directories: %v", err), IsError: true}, nil
	}

	if err := atomicWriteFile(params.Path, []byte(params.Content)); err != nil {
		return tools.ToolResult{Content: fmt.Sprintf("failed to write file: %v", err), IsError: true}, nil
	}

	if checker != nil {
		checker.RecordWrite(ctx, params.Path)
	}

	return tools.ToolResult{Content: fmt.Sprintf("successfully wrote %d bytes to %s", len(params.Content), params.Path), IsError: false}, nil
}
