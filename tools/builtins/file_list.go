package builtins

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"sort"
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
	maxEntries int // cap on rendered entries; <= 0 resolves to maxListDirectoryEntries
	maxBytes   int // cap on rendered output bytes; <= 0 resolves to maxListDirectoryBytes
}

// maxListDirectoryEntries and maxListDirectoryBytes bound a single
// list_directory call (defined locally — limits.go's structs model other
// tools): a huge directory can neither flood the model context nor grow the
// result without bound, mirroring the glob tool's result cap. Past either
// cap the tool stops reading the directory and appends a truncation marker.
const (
	maxListDirectoryEntries = 10_000
	maxListDirectoryBytes   = 1 << 20 // 1 MiB
)

// listDirReadBatch is the number of directory entries read per ReadDir call
// while streaming the directory, so the entry cap stops the read mid-directory
// instead of materializing the whole listing first.
const listDirReadBatch = 256

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
		maxEntries: maxListDirectoryEntries,
		maxBytes:   maxListDirectoryBytes,
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

// listEntry is one rendered directory entry: the fields the tool reports per
// line, collected while streaming so the directory read can stop at the cap.
type listEntry struct {
	name string
	kind string
	size int64
}

// Execute lists the contents of a directory. The listing is bounded: entries
// are streamed in batches and rendering stops at maxListDirectoryEntries
// entries or maxListDirectoryBytes bytes (whichever hits first), with a
// truncation marker appended when the cap fired.
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

	maxEntries := t.maxEntries
	if maxEntries <= 0 {
		maxEntries = maxListDirectoryEntries
	}
	maxBytes := t.maxBytes
	if maxBytes <= 0 {
		maxBytes = maxListDirectoryBytes
	}

	// Stream the directory in bounded batches instead of os.ReadDir, which
	// materializes the whole listing before the caps can be applied.
	dir, err := os.Open(params.Path)
	if err != nil {
		return tools.ToolResult{Content: fmt.Sprintf("failed to read directory: %v", err), IsError: true}, nil
	}
	defer func() { _ = dir.Close() }()

	entries := make([]listEntry, 0, 64)
	truncated := false
	for !truncated {
		batch, readErr := dir.ReadDir(listDirReadBatch)
		for _, entry := range batch {
			if len(entries) >= maxEntries {
				truncated = true
				break
			}
			info, infoErr := entry.Info()
			if infoErr != nil {
				continue
			}
			kind := "file"
			if entry.IsDir() {
				kind = "dir"
			}
			entries = append(entries, listEntry{name: entry.Name(), kind: kind, size: info.Size()})
		}
		if readErr != nil {
			if !errors.Is(readErr, io.EOF) {
				return tools.ToolResult{Content: fmt.Sprintf("failed to read directory: %v", readErr), IsError: true}, nil
			}
			break
		}
	}

	// os.ReadDir sorted its output; the streaming path reads in filesystem
	// order, so the collected subset (the read-order prefix up to the entry
	// cap) is sorted here to preserve the documented name-sorted rendering.
	// The KEPT SET is therefore the read-order prefix, not the alphabetical
	// prefix — determining the latter would require materializing the whole
	// directory, defeating the bound.
	sort.Slice(entries, func(i, j int) bool { return entries[i].name < entries[j].name })

	var sb strings.Builder
	for _, entry := range entries {
		line := fmt.Sprintf("%s\t%s\t%d\n", entry.name, entry.kind, entry.size)
		if sb.Len()+len(line) > maxBytes {
			truncated = true
			break
		}
		sb.WriteString(line)
	}

	if truncated {
		fmt.Fprintf(&sb, "[... truncated — directory listing capped at %d entries / %d bytes ...]\n", maxEntries, maxBytes)
	}

	return tools.ToolResult{Content: sb.String(), IsError: false}, nil
}
