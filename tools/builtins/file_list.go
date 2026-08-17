package builtins

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/v0lka/sp4rk/tools"
)

const toolListDirectoryDescription = `Purpose: list the immediate contents of one directory — each entry's name, type (file or dir) and size in bytes. Non-recursive.
Use when: exploring an unfamiliar tree, verifying a path exists, or deciding which file to open next. For recursive name-pattern search use glob; for content search use ripgrep; once you know the exact file, open it with read_file.
Inputs: path (absolute or workspace-relative directory).
Outputs: one line per entry with name, type and size; an error if the path is missing or not a directory.
Example: list src/ to see the packages before reading any file.
Anti-example: not for finding files by pattern across subtrees (glob '**' does that); not for searching inside files (ripgrep).`

// ListDirectoryTool lists directory contents.
type ListDirectoryTool struct {
	*tools.BaseTool
}

// NewListDirectoryTool creates a new ListDirectoryTool instance.
func NewListDirectoryTool() *ListDirectoryTool {
	return &ListDirectoryTool{
		BaseTool: &tools.BaseTool{
			ToolName:        "list_directory",
			ToolGroup:       tools.GroupLocalRead,
			ToolDescription: toolListDirectoryDescription,
			Schema: json.RawMessage(`{
				"type": "object",
				"properties": {
					"path": {
						"type": "string",
						"description": "Absolute or relative path to the directory to list."
					}
				},
				"required": ["path"]
			}`),
			// ASI01: filenames may carry prompt-injection payloads.
			Untrusted: true,
			Policy:    tools.PolicyAlwaysAllow,
		},
	}
}

// ListDirectoryInput represents the input parameters for list_directory.
type ListDirectoryInput struct {
	Path string `json:"path"`
}

// Judge checks whether the list targets a path inside the session roots.
// Directories outside workspace/temp require user confirmation.
func (t *ListDirectoryTool) Judge(ctx context.Context, input json.RawMessage) tools.JudgeOutcome {
	return judgeReadInSessionRoots(ctx, input)
}

// Execute lists the contents of a directory.
func (t *ListDirectoryTool) Execute(ctx context.Context, input json.RawMessage) (tools.ToolResult, error) {
	var params ListDirectoryInput
	if err := json.Unmarshal(input, &params); err != nil {
		return tools.ParseInputError(err)
	}

	if params.Path == "" {
		return tools.ToolResult{Content: "validation error: path is required", IsError: true}, nil
	}

	params.Path = resolvePath(ctx, params.Path)
	if err := validateResolvedPath(params.Path); err != nil {
		return tools.ToolResult{Content: err.Error(), IsError: true}, nil //nolint:nilerr // error embedded in ToolResult by design
	}

	entries, err := os.ReadDir(params.Path)
	if err != nil {
		return tools.ToolResult{Content: fmt.Sprintf("failed to read directory: %v", err), IsError: true}, nil
	}

	var sb strings.Builder
	for _, entry := range entries {
		info, err := entry.Info()
		if err != nil {
			continue
		}

		entryType := "file"
		if entry.IsDir() {
			entryType = "dir"
		}

		fmt.Fprintf(&sb, "%s\t%s\t%d\n", entry.Name(), entryType, info.Size())
	}

	return tools.ToolResult{Content: sb.String(), IsError: false}, nil
}
