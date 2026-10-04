package tools

import (
	"context"
	"fmt"
	"strings"
)

// ToolNotFoundResult builds an error result for a name that failed exact lookup.
// It suggests removing only the functions. prefix, and only when the candidate
// is BOTH confirmed available by the available predicate AND within the
// effective catalog scope attached to ctx (see WithCatalogScope). It never
// dispatches the candidate; callers must perform exact lookup before using
// this helper. Without a scope attached (an unrestricted caller), availability
// alone decides.
func ToolNotFoundResult(ctx context.Context, name string, available func(string) bool) ToolResult {
	result := ToolResult{Content: "tool not found: " + name, IsError: true}
	candidate, prefixed := strings.CutPrefix(name, "functions.")
	if prefixed && candidate != "" && available != nil && catalogContains(ctx, candidate) && available(candidate) {
		result.Content += fmt.Sprintf("; use the exact catalog name %q in batch.calls[].tool or as the direct tool call name, without adding the functions. namespace", candidate)
	}
	return result
}

// catalogScopeKey carries the effective tool catalog of the current run.
type catalogScopeKey struct{}

// WithCatalogScope attaches the effective tool catalog of the current run —
// typically a delegation's TaskTools — to ctx. Tool-not-found diagnostics
// scoped by it recommend only candidates the current run can actually call: a
// name that is registered globally but excluded from this catalog is never
// recommended, so error recovery never points outside the task-scoped grant
// (see c0wrk SECURITY.md ASI07). The scope is diagnostic-only: it never gates
// dispatch, validation, policy, or confirmation. An empty names slice is a
// valid empty catalog — nothing is recommended.
func WithCatalogScope(ctx context.Context, names []string) context.Context {
	scope := make(map[string]struct{}, len(names))
	for _, name := range names {
		scope[name] = struct{}{}
	}
	return context.WithValue(ctx, catalogScopeKey{}, scope)
}

// catalogContains reports whether name belongs to the effective catalog scope
// attached to ctx. Without a scope it reports true: an unrestricted caller
// falls back to registry-level availability.
func catalogContains(ctx context.Context, name string) bool {
	scope, ok := ctx.Value(catalogScopeKey{}).(map[string]struct{})
	if !ok {
		return true
	}
	_, contains := scope[name]
	return contains
}
