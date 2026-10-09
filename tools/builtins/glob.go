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

const toolGlobDescription = `Purpose: find files and directories by name pattern — supports * within a level and ** across levels (e.g. **/*.go, src/**/*.ts, **/*.test.ts). Symlinked directories are not traversed, results outside the session roots are filtered out (a symlink escaping them is never listed), and a single walk is bounded by a result cap, an entry budget and a wall-clock timeout.
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
// A completely zero GlobLimits (e.g. from a zero-value BuiltinToolsConfig) is
// replaced by DefaultGlobLimits() before storing, so a caller that forgets to
// populate the struct cannot register an unbounded walk: the runaway protection
// is a property of the tool, not of every caller remembering to populate it.
//
// An individual zero field on an otherwise-populated struct is honored as
// "disabled" — the documented config contract ("0 = no entry budget / no result
// cap / no timeout") — so only an all-zero struct falls back to defaults. This
// is why the fallback is all-or-nothing rather than per-field: a per-field fill
// would silently convert a deliberate `0` (disable this budget) into the default
// and make the published "0 disables" contract unreachable.
//
// A host whose config resolution already distinguishes "never populated" from
// "explicitly zeroed on every knob" — and wants the all-zero combination to
// mean "all bounds disabled", not "the defaults" — uses
// NewGlobToolWithLimitsOverride instead, which stores the values verbatim.
func NewGlobToolWithLimits(limits GlobLimits) *GlobTool {
	if limits == (GlobLimits{}) {
		limits = DefaultGlobLimits()
	}
	return newGlobTool(limits)
}

// NewGlobToolWithLimitsOverride creates a new GlobTool instance that uses the
// given limits EXACTLY as provided: unlike NewGlobToolWithLimits, an all-zero
// GlobLimits is NOT replaced by DefaultGlobLimits(). It exists for hosts whose
// config layer has already separated "unset" (take the defaults) from
// "explicitly zero" (the documented "0 = no budget" per field), so an
// operator's explicit zero on every knob — "disable the runaway-walk
// protection entirely" — is honored instead of being silently re-armed.
//
// Callers that cannot make that distinction (a zero-value struct may merely
// mean "never populated") must keep using NewGlobToolWithLimits, whose
// fallback guarantees the walk can never register unbounded by accident.
func NewGlobToolWithLimitsOverride(limits GlobLimits) *GlobTool {
	return newGlobTool(limits)
}

// newGlobTool is the shared constructor; the limits are stored verbatim.
func newGlobTool(limits GlobLimits) *GlobTool {
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

// Limits returns the walk bounds the tool was constructed with. Hosts use it
// to verify that explicitly configured limits survived registration verbatim —
// e.g. that an all-zero override (NewGlobToolWithLimitsOverride) was not
// silently replaced by the defaults.
func (t *GlobTool) Limits() GlobLimits { return t.limits }

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

	// Per-entry containment is enforced only when session roots are attached:
	// with no workspace/temp/allowed roots in the context there is nothing to
	// contain within, and the walk keeps today's fail-open behavior (the same
	// convention as the ignore checker below).
	enforceContainment := len(tools.SessionRoots(ctx)) > 0

	walkErr := doublestar.GlobWalk(
		newBoundedFS(walkCtx, os.DirFS(params.Path), t.limits),
		params.Pattern,
		func(p string, d fs.DirEntry) error {
			// Containment comes first: doublestar.WithNoFollow does not
			// suppress a symlink named in the pattern's literal prefix, so
			// entries under such a prefix (and any other symlink escaping
			// the roots) can resolve outside the walk's boundary. Resolve
			// each entry against the session roots AND the search root
			// itself — the search-root arm keeps glob usable on a
			// work-directory root that is not a session root, while the
			// resolution inside IsWithinRoot still catches an in-tree
			// symlink whose target leaves both. A directory leaves via
			// fs.SkipDir, which the walker handles as a benign subtree skip
			// rather than a walk failure.
			if enforceContainment {
				absEntry := filepath.Join(params.Path, p)
				if !isPathInSessionRoots(walkCtx, absEntry) && !tools.IsWithinRoot(walkCtx, params.Path, absEntry) {
					if d.IsDir() {
						return fs.SkipDir
					}
					return nil
				}
			}

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
		// One of the glob's own bounds (result cap, entry budget, wall-clock
		// timeout) or a benign per-entry filesystem error (a permission-denied
		// directory, a path deleted mid-walk) must not discard the matches
		// already collected: return them with a warning suffix (#5, #14). Only a
		// genuinely fatal walk error — a caller cancellation/deadline or a
		// malformed pattern — is surfaced as an error result below.
		if warning := t.walkWarning(ctx, walkErr); warning != "" {
			return tools.ToolResult{Content: formatGlobResults(results, warning)}, nil
		}
		return tools.ToolResult{Content: t.walkErrorMessage(ctx, walkErr), IsError: true}, nil
	}

	if len(results) == 0 {
		return tools.ToolResult{Content: "no matching files found"}, nil
	}

	return tools.ToolResult{Content: strings.Join(results, "\n")}, nil
}

// walkWarning returns a non-fatal warning suffix for a walk that stopped early
// on one of the glob's own bounds (result cap, entry budget, wall-clock
// timeout) or on a benign per-entry filesystem error, and "" for a walk error
// that must be surfaced as-is (a caller cancellation/deadline, or a malformed
// pattern). The matches collected before the stop are always kept, so the
// caller gets the partial results plus this warning instead of nothing.
func (t *GlobTool) walkWarning(ctx context.Context, err error) string {
	switch {
	case errors.Is(err, errGlobResultsLimited):
		return fmt.Sprintf("warning: results limited to %d matches; narrow the pattern or path", t.limits.MaxResults)
	case errors.Is(err, errGlobEntryBudget):
		return fmt.Sprintf("warning: walk stopped after visiting %d filesystem entries; results may be incomplete (narrow the pattern or path)", t.limits.MaxEntries)
	case errors.Is(err, errGlobCanceled) && ctx.Err() == nil:
		// errGlobCanceled with a live caller context means the glob's own
		// wall-clock timer fired; a caller cancellation has ctx.Err() != nil
		// and is reported as a hard error by walkErrorMessage instead.
		return fmt.Sprintf("warning: glob timed out after %s; results may be incomplete", t.limits.Timeout)
	case isSkippableWalkError(err):
		return "warning: some entries could not be read; results may be incomplete"
	}
	return ""
}

// isSkippableWalkError reports whether err is a per-entry filesystem I/O error
// (permission denied, a path deleted mid-walk, an unreadable mount) that the
// pre-WithFailOnIOErrors walk silently skipped. The os filesystem wraps such
// errors in *fs.PathError; errors.Is(err, fs.ErrPermission) additionally covers
// an unwrapped permission error should a filesystem surface one. A genuine walk
// error such as doublestar.ErrBadPattern is neither and is still surfaced as an
// error result.
func isSkippableWalkError(err error) bool {
	var pathErr *fs.PathError
	return errors.As(err, &pathErr) || errors.Is(err, fs.ErrPermission)
}

// formatGlobResults renders the collected matches with a warning suffix. An
// empty result set renders the warning alone, so a bound that fired before any
// match still yields an actionable message.
func formatGlobResults(results []string, warning string) string {
	if len(results) == 0 {
		return warning
	}
	return strings.Join(results, "\n") + "\n\n" + warning
}

// walkErrorMessage translates a walk error that must surface as an error result
// into a clear, model-facing message. Only a caller cancellation/deadline and a
// malformed pattern reach here — the glob's own bounds and benign I/O errors are
// reported as partial results by walkWarning. No specific duration is quoted for
// the context case (#16): whether the caller's context or the glob's own timer
// produced the cancellation cannot be told apart reliably, and the glob's own
// timeout is reported by walkWarning when it fires.
func (t *GlobTool) walkErrorMessage(ctx context.Context, err error) string {
	switch {
	case errors.Is(err, errGlobCanceled):
		if errors.Is(ctx.Err(), context.DeadlineExceeded) {
			return "glob interrupted: context deadline exceeded"
		}
		return "glob canceled"
	default:
		return fmt.Sprintf("glob error: %v", err)
	}
}
