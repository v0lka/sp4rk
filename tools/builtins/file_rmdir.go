package builtins

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	"github.com/v0lka/sp4rk/tools"
)

const toolDeleteDirectoryDescription = `Purpose: delete a directory — empty-only by default, or the whole subtree with recursive=true (like rm -rf).
Use when: removing a generated or obsolete directory tree. Deletion is destructive: prefer the default non-recursive mode (it fails loudly on non-empty), and escalate to recursive only after verifying the contents are disposable.
Inputs: path (must be a directory); recursive (bool — true removes the directory and all contents).
Outputs: confirmation, or an error (not a directory, non-empty without recursive, policy denial).
Example: delete build/tmp after verifying its contents.
Anti-example: never recursive-delete unreviewed content "to clean up"; for a single file use delete_file.
Note: a symlink is unlinked itself (POSIX rm -rf semantics) — never the tree it points to.`

// DeleteDirectoryTool deletes directories.
type DeleteDirectoryTool struct {
	*tools.BaseTool
}

// NewDeleteDirectoryTool creates a new DeleteDirectoryTool instance.
func NewDeleteDirectoryTool() *DeleteDirectoryTool {
	return &DeleteDirectoryTool{
		BaseTool: &tools.BaseTool{
			ToolName:        "delete_directory",
			ToolGroup:       tools.GroupLocalWrite,
			ToolDescription: toolDeleteDirectoryDescription,
			Schema: json.RawMessage(`{
				"type": "object",
				"properties": {
					"path": {
						"type": "string",
						"description": "Absolute or relative path of the directory to delete."
					},
					"recursive": {
						"type": "boolean",
						"description": "When true, deletes the directory and all files and subdirectories within it. When false, the operation fails if the directory is not empty."
					}
				},
				"required": ["path", "recursive"]
			}`),
			Policy: tools.PolicyUserConfirm,
		},
	}
}

// DeleteDirectoryInput represents the input parameters for delete_directory.
type DeleteDirectoryInput struct {
	Path      string `json:"path"`
	Recursive bool   `json:"recursive"`
}

// Judge uses session roots check for write operations. Like delete_file, the
// judge evaluates the path as spelled in its final component (no symlink
// resolution of the target): delete_directory unlinks the path itself, POSIX
// rm -rf style, so a link is judged — and later removed — as the link, never
// as the tree it points to.
func (t *DeleteDirectoryTool) Judge(ctx context.Context, input json.RawMessage) tools.JudgeOutcome {
	var params DeleteDirectoryInput
	if err := json.Unmarshal(input, &params); err != nil {
		return tools.JudgeOutcome{}
	}
	params.Path = resolvePathUnresolved(ctx, params.Path)
	if err := validateResolvedPath(params.Path); err != nil {
		return softOutcome(false, err.Error(), tools.ReasonCodeOutsideSessionRoots)
	}
	return judgeWriteInSessionRootsUnresolved(ctx, params.Path)
}

// Execute deletes a directory. If recursive is true, it removes all contents.
//
// Symlink semantics are POSIX rm -rf semantics: a symlink is removed ITSELF
// (os.RemoveAll on a symlink unlinks just the link, never the target tree),
// so deleting an in-workspace directory link can never wipe the tree it
// points to. Real directories keep the historical behavior.
func (t *DeleteDirectoryTool) Execute(ctx context.Context, input json.RawMessage) (tools.ToolResult, error) {
	var params DeleteDirectoryInput
	if err := json.Unmarshal(input, &params); err != nil {
		return tools.ParseInputError(err)
	}

	if params.Path == "" {
		return tools.ToolResult{Content: "validation error: path is required", IsError: true}, nil
	}

	params.Path = resolvePathUnresolved(ctx, params.Path)
	if err := validateResolvedPath(params.Path); err != nil {
		return tools.ToolResult{Content: err.Error(), IsError: true}, nil //nolint:nilerr // error embedded in ToolResult by design
	}

	// Lstat, never Stat: the destructive act must be decided on the link
	// itself, not on whatever the link points at.
	info, err := os.Lstat(params.Path)
	if err != nil {
		return tools.ToolResult{Content: fmt.Sprintf("failed to stat path: %v", err), IsError: true}, nil
	}

	if info.Mode()&os.ModeSymlink != 0 {
		// rm -rf semantics: os.RemoveAll on a symlink removes just the link.
		if err := os.RemoveAll(params.Path); err != nil {
			return tools.ToolResult{Content: fmt.Sprintf("failed to delete directory: %v", err), IsError: true}, nil
		}
		return tools.ToolResult{Content: "successfully deleted directory: " + params.Path, IsError: false}, nil
	}

	if !info.IsDir() {
		return tools.ToolResult{Content: "path is not a directory", IsError: true}, nil
	}

	if params.Recursive {
		if err := os.RemoveAll(params.Path); err != nil {
			return tools.ToolResult{Content: fmt.Sprintf("failed to delete directory: %v", err), IsError: true}, nil
		}
	} else {
		if err := os.Remove(params.Path); err != nil {
			return tools.ToolResult{Content: fmt.Sprintf("failed to delete directory: %v", err), IsError: true}, nil
		}
	}

	return tools.ToolResult{Content: "successfully deleted directory: " + params.Path, IsError: false}, nil
}
