package builtins

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	"github.com/v0lka/sp4rk/tools"
)

const toolDeleteFileDescription = `Purpose: delete a single regular file.
Use when: removing an obsolete file you have verified is inside the workspace and truly disposable — deletion is destructive and there is no undo. Fails if the path is a directory, and refuses a non-regular existing target (device node, FIFO, socket).
Inputs: path (must point to a regular file, not a directory).
Outputs: confirmation, or an error (missing file, path is a directory, policy denial).
Example: delete a scratch script after its purpose is served.
Anti-example: not for directories (delete_directory); do not batch-delete via the shell tool — per-tool policy checks would be bypassed; prefer leaving cleanup to the user unless asked.
Note: a symlink is unlinked itself (POSIX rm semantics) — never the file it points to.`

// DeleteFileTool deletes files.
type DeleteFileTool struct {
	*tools.BaseTool
}

// NewDeleteFileTool creates a new DeleteFileTool instance.
func NewDeleteFileTool() *DeleteFileTool {
	return &DeleteFileTool{
		BaseTool: &tools.BaseTool{
			ToolName:        "delete_file",
			ToolGroup:       tools.GroupLocalWrite,
			ToolDescription: toolDeleteFileDescription,
			Schema: json.RawMessage(`{
				"type": "object",
				"properties": {
					"path": {
						"type": "string",
						"description": "Absolute or relative path of the file to delete. Must point to a regular file, not a directory."
					}
				},
				"required": ["path"]
			}`),
			Policy: tools.PolicyUserConfirm,
		},
	}
}

// DeleteFileInput represents the input parameters for delete_file.
type DeleteFileInput struct {
	Path string `json:"path"`
}

// Judge uses session roots check for write operations. The judge evaluates
// the path as spelled in its final component (no symlink resolution of the
// target): delete_file unlinks the path itself, POSIX rm style, so the LINK
// path is what must be assessed — an in-root link is auto-approved and its
// target is never touched, while a link pointing outside the roots escalates
// to user confirmation.
func (t *DeleteFileTool) Judge(ctx context.Context, input json.RawMessage) tools.JudgeOutcome {
	var params DeleteFileInput
	if err := json.Unmarshal(input, &params); err != nil {
		return tools.JudgeOutcome{}
	}
	params.Path = resolvePathUnresolved(ctx, params.Path)
	if err := validateResolvedPath(params.Path); err != nil {
		return softOutcome(false, err.Error(), tools.ReasonCodeOutsideSessionRoots)
	}
	return judgeWriteInSessionRootsUnresolved(ctx, params.Path)
}

// Execute deletes a single file. Returns an error if the path is a directory.
//
// Symlink semantics are POSIX rm semantics: a symlink is unlinked ITSELF (the
// link path is containment-checked, the target is never resolved), so
// deleting an in-workspace link can never destroy the file or tree it points
// to. Non-symlink targets keep the historical behavior, except that a target
// which exists but is not a regular file (device node, FIFO, socket) is
// refused rather than unlinked.
func (t *DeleteFileTool) Execute(ctx context.Context, input json.RawMessage) (tools.ToolResult, error) {
	var params DeleteFileInput
	if err := json.Unmarshal(input, &params); err != nil {
		return tools.ParseInputError(err)
	}

	if params.Path == "" {
		return tools.ToolResult{Content: "validation error: path is required", IsError: true}, nil
	}

	params.Path = resolvePathUnresolved(ctx, params.Path)
	if err := validateResolvedPath(params.Path); err != nil {
		return tools.ToolResult{Content: err.Error(), IsError: true}, nil
	}

	// Coherence check: block delete if file was modified since this session's last read.
	checker := tools.CoherenceFrom(ctx)
	if checker != nil {
		checker.Lock(params.Path)
		if conflict := checker.CheckWrite(ctx, params.Path); conflict != nil {
			checker.Unlock(params.Path)
			return tools.ToolResult{Content: formatWriteConflict(conflict), IsError: true}, nil
		}
		defer checker.Unlock(params.Path)
	}

	// Lstat, never Stat: the destructive act must be decided on the link
	// itself, not on whatever the link points at.
	info, err := os.Lstat(params.Path)
	if err != nil {
		return tools.ToolResult{Content: fmt.Sprintf("failed to stat path: %v", err), IsError: true}, nil
	}

	if info.Mode()&os.ModeSymlink != 0 {
		// rm semantics: remove the link, never its target. This holds for a
		// link to a file, a link to a directory, a link to a device, and a
		// dangling link alike.
		if err := os.Remove(params.Path); err != nil {
			return tools.ToolResult{Content: fmt.Sprintf("failed to delete file: %v", err), IsError: true}, nil
		}
		if checker != nil {
			checker.RecordDelete(ctx, params.Path)
		}
		return tools.ToolResult{Content: "successfully deleted file: " + params.Path, IsError: false}, nil
	}

	if info.IsDir() {
		return tools.ToolResult{Content: "path is a directory, use delete_directory instead", IsError: true}, nil
	}
	if !info.Mode().IsRegular() {
		return tools.ToolResult{
			Content: "refusing to delete: path exists but is not a regular file (device node, FIFO, or socket): " + params.Path,
			IsError: true,
		}, nil
	}

	if err := os.Remove(params.Path); err != nil {
		return tools.ToolResult{Content: fmt.Sprintf("failed to delete file: %v", err), IsError: true}, nil
	}

	if checker != nil {
		checker.RecordDelete(ctx, params.Path)
	}

	return tools.ToolResult{Content: "successfully deleted file: " + params.Path, IsError: false}, nil
}
