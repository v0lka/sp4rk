package builtins

import (
	"context"
	"encoding/json"
	"path/filepath"

	"github.com/v0lka/sp4rk/tools"
)

// judgeWriteInSessionRoots checks whether a write operation targets a path
// inside the session workspace or the session temp directory. Both roots are
// treated as equal peers: writes inside either are auto-approved (when
// auto_approve_workspace_writes is enabled); writes outside both escalate to
// user confirmation. Returns (true, reason) if allowed, or (false, reason) to
// defer to the confirmation flow with an explanatory reason.
func judgeWriteInSessionRoots(ctx context.Context, path string) (allowed bool, reason string) {
	absPath, err := filepath.Abs(path)
	if err != nil {
		return false, ""
	}
	absPath = filepath.Clean(absPath)
	if resolved, evalErr := filepath.EvalSymlinks(absPath); evalErr == nil {
		absPath = resolved
	} else {
		parentDir := filepath.Dir(absPath)
		if resolvedParent, parentErr := filepath.EvalSymlinks(parentDir); parentErr == nil {
			absPath = filepath.Join(resolvedParent, filepath.Base(absPath))
		}
	}

	if isPathInSessionRoots(ctx, absPath) {
		return true, "target is within session workspace or temp directory"
	}
	return false, formatOutsideRootsError(absPath).Error()
}

// judgeReadInSessionRootsForPath is the shared containment core for read-side
// judges: it makes the path absolute, resolves symlinks (falling back to the
// resolved parent for not-yet-existing paths), and checks session-root
// membership. Callers supply a non-empty path already resolved according to
// the tool's own semantics (e.g. glob/ripgrep defaulting an empty `path` to
// the workspace root); path resolution and validation are the caller's
// responsibility so each tool can apply its own defaults.
//
// Returns (true, reason) to auto-execute, or (false, reason) to escalate to
// user confirmation.
func judgeReadInSessionRootsForPath(ctx context.Context, resolvedPath string) (allowed bool, reason string) {
	absPath, err := filepath.Abs(resolvedPath)
	if err != nil {
		// Cannot determine path — fail closed and escalate to confirmation.
		return false, "cannot determine target path"
	}
	absPath = filepath.Clean(absPath)
	if evaled, evalErr := filepath.EvalSymlinks(absPath); evalErr == nil {
		absPath = evaled
	} else {
		parentDir := filepath.Dir(absPath)
		if resolvedParent, parentErr := filepath.EvalSymlinks(parentDir); parentErr == nil {
			absPath = filepath.Join(resolvedParent, filepath.Base(absPath))
		}
	}

	if isPathInSessionRoots(ctx, absPath) {
		return true, "read-only file operation within session workspace or temp directory"
	}

	return false, formatOutsideRootsError(absPath).Error()
}

// judgeReadInSessionRoots checks whether a read operation targets a path inside
// the session workspace or the session temp directory, for tools whose `path`
// parameter is REQUIRED (read_file, list_directory). Both roots are treated as
// equal peers: reads inside either auto-execute without confirmation; reads
// outside escalate to user confirmation. A missing `path` fails closed.
//
// Returns (true, reason) to auto-execute, or (false, reason) to escalate to
// user confirmation.
func judgeReadInSessionRoots(ctx context.Context, input json.RawMessage) (allowed bool, reason string) {
	return judgeReadWithPathDefault(ctx, input, false)
}

// judgeReadInSessionRootsOptionalPath is the read-side judge for tools whose
// `path` parameter is OPTIONAL and defaults to the workspace root when omitted
// (glob, ripgrep). An omitted `path` resolves to the workspace — the safest
// case — and is auto-approved; an explicit path is validated and checked for
// session-root membership like the required variant. If no workspace is
// available, the call fails closed.
//
// Returns (true, reason) to auto-execute, or (false, reason) to escalate to
// user confirmation.
func judgeReadInSessionRootsOptionalPath(ctx context.Context, input json.RawMessage) (allowed bool, reason string) {
	return judgeReadWithPathDefault(ctx, input, true)
}

// judgeReadWithPathDefault is the shared implementation for the read-side
// judges. When pathOptional is true, an empty `path` is resolved to the
// workspace root (matching the Execute semantics of glob/ripgrep); otherwise
// an empty `path` fails closed.
func judgeReadWithPathDefault(ctx context.Context, input json.RawMessage, pathOptional bool) (allowed bool, reason string) {
	var params struct {
		Path string `json:"path"`
	}
	if err := json.Unmarshal(input, &params); err != nil {
		// Cannot determine path — fail closed and escalate to confirmation.
		return false, "cannot determine target path"
	}

	if params.Path == "" {
		if !pathOptional {
			return false, "cannot determine target path"
		}
		ws := tools.WorkspacePathFrom(ctx)
		if ws == "" {
			return false, "cannot determine target path"
		}
		return judgeReadInSessionRootsForPath(ctx, ws)
	}

	resolved := resolvePath(ctx, params.Path)
	if err := validateResolvedPath(resolved); err != nil {
		return false, err.Error()
	}
	return judgeReadInSessionRootsForPath(ctx, resolved)
}
