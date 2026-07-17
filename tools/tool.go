// Package tools provides the tool abstraction, registry, and core types for agent tool execution.
package tools

import (
	"context"
	"encoding/json"
	"fmt"
)

// ToolPolicy defines the security policy for a tool.
type ToolPolicy int

const (
	// PolicyAlwaysAllow executes the tool without any confirmation or judge check.
	PolicyAlwaysAllow ToolPolicy = iota
	// PolicyAlwaysDeny blocks the tool from executing.
	PolicyAlwaysDeny
	// PolicyUserConfirm always requires user confirmation before executing.
	PolicyUserConfirm
)

// Tool is the unified interface for all tools (Core and MCP).
type Tool interface {
	Name() string
	Description() string
	InputSchema() json.RawMessage
	Execute(ctx context.Context, input json.RawMessage) (ToolResult, error)
	DefaultPolicy() ToolPolicy
	// IsUntrusted reports whether this tool returns external/untrusted data
	// (web, MCP, filesystem) that should be sanitized before LLM context.
	IsUntrusted() bool
}

// ToolResult is the result of tool execution.
type ToolResult struct {
	Content string
	IsError bool
}

// ToolSourceCategory classifies a tool's origin for routing and display.
type ToolSourceCategory string

const (
	// SourceCategoryCore marks tools built into sp4rk.
	SourceCategoryCore ToolSourceCategory = "core"
	// SourceCategoryMCP marks tools provided by MCP servers.
	SourceCategoryMCP ToolSourceCategory = "mcp"
)

// ToolDescriptor — describes a tool for Planner/Executor (metadata only, no execution).
type ToolDescriptor struct {
	Name           string             `json:"name"`
	Description    string             `json:"description"`
	InputSchema    json.RawMessage    `json:"input_schema"`
	Source         string             `json:"source"` // "core" or the MCP server name for MCP-sourced tools
	SourceCategory ToolSourceCategory `json:"-"`      // cached category for fast checks
}

// ParseToolPolicy converts a policy string to a ToolPolicy constant.
func ParseToolPolicy(s string) ToolPolicy {
	switch s {
	case "always_allow":
		return PolicyAlwaysAllow
	case "always_deny":
		return PolicyAlwaysDeny
	case "user_confirm":
		return PolicyUserConfirm
	default:
		return PolicyUserConfirm
	}
}

// ErrorResult creates a ToolResult with IsError=true.
func ErrorResult(format string, args ...any) ToolResult {
	return ToolResult{Content: fmt.Sprintf(format, args...), IsError: true}
}

// ParseInputError returns a standard parse-error ToolResult.
func ParseInputError(err error) (ToolResult, error) {
	return ErrorResult("failed to parse input: %v", err), nil
}

// BaseTool provides default implementations of Name, Description, InputSchema,
// DefaultPolicy, and IsUntrusted so concrete tools only need to implement Execute.
type BaseTool struct {
	ToolName        string
	ToolDescription string
	Schema          json.RawMessage
	Policy          ToolPolicy
	Untrusted       bool // marks tool output as external/untrusted data for prompt injection defense
}

// Name returns the tool name.
func (b *BaseTool) Name() string { return b.ToolName }

// Description returns the tool description.
func (b *BaseTool) Description() string { return b.ToolDescription }

// InputSchema returns the tool's JSON input schema.
func (b *BaseTool) InputSchema() json.RawMessage { return b.Schema }

// DefaultPolicy returns the tool's default security policy.
func (b *BaseTool) DefaultPolicy() ToolPolicy { return b.Policy }

// IsUntrusted returns whether this tool produces external/untrusted data.
func (b *BaseTool) IsUntrusted() bool { return b.Untrusted }

// workspacePathKey is the context key for the session workspace path.
type workspacePathKey struct{}

// WithWorkspacePath returns a new context with the session workspace path attached.
func WithWorkspacePath(ctx context.Context, path string) context.Context {
	return context.WithValue(ctx, workspacePathKey{}, path)
}

// WorkspacePathFrom extracts the session workspace path from the context.
// Returns an empty string if not found.
func WorkspacePathFrom(ctx context.Context) string {
	if v, ok := ctx.Value(workspacePathKey{}).(string); ok {
		return v
	}
	return ""
}

// tempDirKey is the context key for the temporary directory path.
type tempDirKey struct{}

// WithTempDir returns a new context with the temporary directory path attached.
func WithTempDir(ctx context.Context, path string) context.Context {
	return context.WithValue(ctx, tempDirKey{}, path)
}

// TempDirFrom extracts the temporary directory path from the context.
// Returns an empty string if not found.
func TempDirFrom(ctx context.Context) string {
	if v, ok := ctx.Value(tempDirKey{}).(string); ok {
		return v
	}
	return ""
}

// allowedRootsKey is the context key for additional containment roots
// (auxiliary working directories) that are treated as peers of the workspace
// and temp directory by every path-containment check. Roots must be absolute.
type allowedRootsKey struct{}

// WithAllowedRoots attaches additional containment roots (auxiliary working
// directories) to the context. Operations inside any of these roots are
// treated the same as the workspace and temp directory (auto-allow in the
// judge fast-path, no confirmation for path-local AlwaysAllow tools). Roots
// must be absolute paths.
func WithAllowedRoots(ctx context.Context, roots []string) context.Context {
	return context.WithValue(ctx, allowedRootsKey{}, roots)
}

// AllowedRootsFrom extracts the additional containment roots. Returns nil if none set.
func AllowedRootsFrom(ctx context.Context) []string {
	if v, ok := ctx.Value(allowedRootsKey{}).([]string); ok {
		return v
	}
	return nil
}

// IgnoreChecker reports whether an absolute path is ignored by the ignore
// rules (.gitignore / .aiignore) of the root that contains it. It is satisfied
// structurally by github.com/v0lka/sp4rk/ignore.Resolver and ignore.Multi, so
// it is plumbed through tool context without the tools package taking a direct
// dependency on the ignore package. Read-style tools (glob, ripgrep) consult
// it to honour ignore rules for both the workspace and any work-directory root.
type IgnoreChecker interface {
	Ignored(absPath string, isDir bool) bool
}

// ignoreCheckerKey is the context key for the session ignore checker.
type ignoreCheckerKey struct{}

// WithIgnoreChecker returns a new context with the ignore checker attached.
// Passing a nil checker is equivalent to not attaching one: downstream tools
// treat a missing checker as "no ignore filtering" and preserve their default
// behaviour.
func WithIgnoreChecker(ctx context.Context, checker IgnoreChecker) context.Context {
	return context.WithValue(ctx, ignoreCheckerKey{}, checker)
}

// IgnoreCheckerFrom extracts the ignore checker from the context. Returns nil
// when none is attached; callers MUST then skip ignore filtering and keep
// their pre-ignore behaviour (graceful, no regression).
func IgnoreCheckerFrom(ctx context.Context) IgnoreChecker {
	if v, ok := ctx.Value(ignoreCheckerKey{}).(IgnoreChecker); ok {
		return v
	}
	return nil
}

// SessionRoots returns the deduplicated, non-empty union of the workspace
// path, the temp directory, and any additional allowed roots. This is the
// canonical list consulted by ALL path-containment checks.
func SessionRoots(ctx context.Context) []string {
	seen := make(map[string]struct{})
	var roots []string
	add := func(r string) {
		if r == "" {
			return
		}
		if _, ok := seen[r]; ok {
			return
		}
		seen[r] = struct{}{}
		roots = append(roots, r)
	}
	add(WorkspacePathFrom(ctx))
	add(TempDirFrom(ctx))
	for _, r := range AllowedRootsFrom(ctx) {
		add(r)
	}
	return roots
}

// taskContextKey is the context key for passing task context through Go's context.Context.
type taskContextKey struct{}

// WithTaskContext returns a new context with the task description attached.
func WithTaskContext(ctx context.Context, desc string) context.Context {
	return context.WithValue(ctx, taskContextKey{}, desc)
}

// TaskContextFrom extracts the task description from the context.
// Returns an empty string if not found.
func TaskContextFrom(ctx context.Context) string {
	if v, ok := ctx.Value(taskContextKey{}).(string); ok {
		return v
	}
	return ""
}
