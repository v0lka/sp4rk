package builtins

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	doublestar "github.com/bmatcuk/doublestar/v4"
	"github.com/v0lka/sp4rk/tools"
)

const toolGlobDescription = `Purpose: find files and directories by name pattern — supports * within a level and ** across levels (e.g. **/*.go, src/**/*.ts, **/*.test.ts).
Use when: you know the name/extension shape but not the exact path. For a single directory's contents use list_directory; to search file contents use ripgrep; to open a found path use read_file.
Inputs: pattern (glob expression); optional path (base directory, defaults to the workspace); optional type filter (files | dirs | all; default files).
Outputs: the list of matching paths.
Example: pattern "**/*.go" with path "core/" finds every Go file under core/.
Anti-example: not for content search (ripgrep matches what is inside files); not for listing one directory's entries (list_directory); '*' does not cross directory boundaries — use '**' for recursion.`

// GlobTool finds files and directories matching doublestar glob patterns.
type GlobTool struct {
	*tools.BaseTool
	limits GlobLimits
}

// NewGlobTool creates a new GlobTool instance with default limits.
func NewGlobTool() *GlobTool {
	return NewGlobToolWithLimits(DefaultGlobLimits())
}

// NewGlobToolWithLimits creates a new GlobTool instance with the given limits.
func NewGlobToolWithLimits(limits GlobLimits) *GlobTool {
	return &GlobTool{
		BaseTool: &tools.BaseTool{
			ToolName:        "glob",
			ToolGroup:       tools.GroupLocalRead,
			ToolDescription: toolGlobDescription,
			Schema: json.RawMessage(`{
		"type": "object",
		"properties": {
			"pattern": {
				"type": "string",
				"description": "Glob pattern to match against file paths, e.g. **/*.java, src/**/*.ts, **/*.cs, *.json"
			},
			"path": {
				"type": "string",
				"description": "Base directory to search from. Defaults to the project workspace when omitted."
			},
			"type": {
				"type": "string",
				"enum": ["files", "dirs", "all"],
				"description": "Filter results: \"files\" (default), \"dirs\", or \"all\""
			}
		},
		"required": ["pattern"]
	}`),
			Policy:    tools.PolicyAlwaysAllow,
			Untrusted: true,
		},
		limits: limits,
	}
}

// GlobInput represents the input parameters for glob search.
type GlobInput struct {
	Pattern string `json:"pattern"`
	Path    string `json:"path"`
	Type    string `json:"type"`
}

// Judge checks whether the glob targets a path inside the session roots.
// The `path` parameter is optional and defaults to the workspace root, so an
// omitted `path` is the safest case and auto-approves. Paths outside
// workspace/temp require user confirmation.
func (t *GlobTool) Judge(ctx context.Context, input json.RawMessage) tools.JudgeOutcome {
	return judgeReadInSessionRootsOptionalPath(ctx, input)
}

// Execute runs the glob pattern search and returns matching file paths.
func (t *GlobTool) Execute(ctx context.Context, input json.RawMessage) (tools.ToolResult, error) {
	var params GlobInput
	if err := json.Unmarshal(input, &params); err != nil {
		return tools.ParseInputError(err)
	}

	if params.Pattern == "" {
		return tools.ToolResult{Content: "validation error: pattern is required", IsError: true}, nil
	}

	if params.Path == "" {
		params.Path = tools.WorkspacePathFrom(ctx)
		if params.Path == "" {
			return tools.ToolResult{Content: "path is required when no workspace is available", IsError: true}, nil
		}
	} else {
		params.Path = resolvePath(ctx, params.Path)
		if err := validateResolvedPath(params.Path); err != nil {
			return tools.ToolResult{Content: err.Error(), IsError: true}, nil //nolint:nilerr // error embedded in ToolResult by design
		}
	}

	// Apply defaults
	if params.Type == "" {
		params.Type = "files"
	}

	// Validate path exists and is a directory
	info, err := os.Stat(params.Path)
	if err != nil {
		return tools.ToolResult{Content: fmt.Sprintf("path error: %v", err), IsError: true}, nil
	}
	if !info.IsDir() {
		return tools.ToolResult{Content: "path is not a directory: " + params.Path, IsError: true}, nil
	}

	// Bound the walk. A wall-clock timeout plus the entry/result budgets
	// enforced by boundedFS and the callback below guarantee that a filesystem
	// runaway (e.g. a symlink loop or an enormous tree) can neither hang the
	// tool nor exhaust memory.
	walkCtx := ctx
	if t.limits.Timeout > 0 {
		var cancel context.CancelFunc
		walkCtx, cancel = context.WithTimeout(ctx, t.limits.Timeout)
		defer cancel()
	}

	var results []string

	walkErr := doublestar.GlobWalk(
		newBoundedFS(walkCtx, os.DirFS(params.Path), t.limits),
		params.Pattern,
		func(p string, d fs.DirEntry) error {
			// Filter by type
			switch params.Type {
			case "files":
				if d.IsDir() {
					return nil
				}
			case "dirs":
				if !d.IsDir() {
					return nil
				}
				// "all": no filtering
			}

			// Honour .gitignore/.aiignore when a checker is plumbed through the
			// context. p is relative to params.Path (the search root), so resolve
			// it to an absolute path that the multi-root checker can map to its
			// containing root. No checker in context => today's behaviour (no
			// filtering). Ignored directories are skipped here, and their file
			// children are skipped too because the checker considers ancestor
			// directories when deciding whether a path is ignored.
			if checker := tools.IgnoreCheckerFrom(walkCtx); checker != nil {
				absEntry := filepath.Join(params.Path, p)
				if checker.Ignored(absEntry, d.IsDir()) {
					return nil
				}
			}

			results = append(results, p)
			if t.limits.MaxResults > 0 && len(results) >= t.limits.MaxResults {
				return errGlobResultsLimited
			}
			return nil
		},
		// Never follow symbolic links: closes the classic symlink-loop hang.
		doublestar.WithNoFollow(),
		// Surface the boundedFS sentinels (and any real I/O error) as a walk
		// error instead of silently ignoring them.
		doublestar.WithFailOnIOErrors(),
	)

	if walkErr != nil {
		return tools.ToolResult{Content: t.walkErrorMessage(walkCtx, walkErr), IsError: true}, nil
	}

	if len(results) == 0 {
		return tools.ToolResult{Content: "no matching files found"}, nil
	}

	output := strings.Join(results, "\n")

	return tools.ToolResult{Content: output}, nil
}

// walkErrorMessage translates a GlobWalk error into a clear, model-facing
// message. The boundedFS sentinels are reported with actionable guidance; any
// other error is the walk's own (e.g. doublestar.ErrBadPattern or an I/O error
// surfaced by WithFailOnIOErrors).
func (t *GlobTool) walkErrorMessage(ctx context.Context, err error) string {
	switch {
	case errors.Is(err, errGlobCanceled):
		if errors.Is(ctx.Err(), context.DeadlineExceeded) {
			return fmt.Sprintf("glob timed out after %s; narrow the pattern or path", t.limits.Timeout)
		}
		return "glob canceled"
	case errors.Is(err, errGlobEntryBudget):
		return fmt.Sprintf("glob aborted after visiting %d filesystem entries (possible symlink loop or runaway directory); narrow the pattern or path", t.limits.MaxEntries)
	case errors.Is(err, errGlobResultsLimited):
		return fmt.Sprintf("glob matched %d or more results; narrow the pattern or path", t.limits.MaxResults)
	default:
		return fmt.Sprintf("glob error: %v", err)
	}
}
