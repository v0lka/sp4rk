package builtins

import (
	"context"
	"encoding/json"
	"path/filepath"

	"github.com/v0lka/sp4rk/tools"
)

// softOutcome wraps an (allowed, reason) pair from a containment check into a
// JudgeOutcome with soft severity and the passed reason code: path-locality
// escalations are advisory — the operation may be legitimate, only its scope
// is in question. Allowed outcomes pass an empty code (nothing to classify);
// denied outcomes carry the code matching their escalation branch.
func softOutcome(allowed bool, reason string, code tools.JudgeReasonCode) tools.JudgeOutcome {
	return tools.JudgeOutcome{Allow: allowed, Reason: reason, Severity: tools.JudgeSeveritySoft, ReasonCode: code}
}

// hardOutcome wraps an (allowed, reason) pair into a JudgeOutcome with hard
// severity and the passed reason code. Used for unassessable inputs ("cannot
// determine target path") — mirroring the web_fetch judge's "cannot determine
// target URL" — because an input that cannot be assessed at all must never be
// auto-resolved by Smart Approve or weakened by the advisory judge
// (JudgeSeverityHard contract in tools/safety.go).
func hardOutcome(allowed bool, reason string, code tools.JudgeReasonCode) tools.JudgeOutcome {
	return tools.JudgeOutcome{Allow: allowed, Reason: reason, Severity: tools.JudgeSeverityHard, ReasonCode: code}
}

// judgeWriteInSessionRoots checks whether a write operation targets a path
// inside the session workspace or the session temp directory. Both roots are
// treated as equal peers: writes inside either are auto-approved (when
// auto_approve_workspace_writes is enabled); writes outside both escalate to
// user confirmation. Returns an allowed outcome if permitted, or a denied
// outcome with an explanatory reason to defer to the confirmation flow.
func judgeWriteInSessionRoots(ctx context.Context, path string) tools.JudgeOutcome {
	absPath, err := filepath.Abs(path)
	if err != nil {
		// Cannot determine path — fail closed and escalate to confirmation
		// (mirrors the read-side judge). Unassessable input is hard: it must
		// not be auto-resolved by Smart Approve.
		return hardOutcome(false, "cannot determine target path", tools.ReasonCodeUnassessablePath)
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
		return softOutcome(true, "target is within session workspace or temp directory", "")
	}
	return softOutcome(false, formatOutsideRootsError(absPath).Error(), tools.ReasonCodeOutsideSessionRoots)
}

// judgeReadInSessionRootsForPath is the shared containment core for read-side
// judges: it makes the path absolute, resolves symlinks (falling back to the
// resolved parent for not-yet-existing paths), and checks session-root
// membership. Callers supply a non-empty path already resolved according to
// the tool's own semantics (e.g. glob/ripgrep defaulting an empty `path` to
// the workspace root); path resolution and validation are the caller's
// responsibility so each tool can apply its own defaults.
//
// Returns an allowed outcome to auto-execute, or a denied outcome to escalate
// to user confirmation.
func judgeReadInSessionRootsForPath(ctx context.Context, resolvedPath string) tools.JudgeOutcome {
	absPath, err := filepath.Abs(resolvedPath)
	if err != nil {
		// Cannot determine path — fail closed and escalate to confirmation.
		// Unassessable input is hard (see hardOutcome).
		return hardOutcome(false, "cannot determine target path", tools.ReasonCodeUnassessablePath)
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
		return softOutcome(true, "read-only file operation within session workspace or temp directory", "")
	}

	return softOutcome(false, formatOutsideRootsError(absPath).Error(), tools.ReasonCodeOutsideSessionRoots)
}

// judgeReadInSessionRoots checks whether a read operation targets a path inside
// the session workspace or the session temp directory, for tools whose `path`
// parameter is REQUIRED (read_file, list_directory). Both roots are treated as
// equal peers: reads inside either auto-execute without confirmation; reads
// outside escalate to user confirmation. A missing `path` fails closed.
//
// Returns an allowed outcome to auto-execute, or a denied outcome to escalate
// to user confirmation.
func judgeReadInSessionRoots(ctx context.Context, input json.RawMessage) tools.JudgeOutcome {
	return judgeReadWithPathDefault(ctx, input, false)
}

// judgeReadInSessionRootsOptionalPath is the read-side judge for tools whose
// `path` parameter is OPTIONAL and defaults to the workspace root when omitted
// (glob, ripgrep). An omitted `path` resolves to the workspace — the safest
// case — and is auto-approved; an explicit path is validated and checked for
// session-root membership like the required variant. If no workspace is
// available, the call fails closed.
//
// Returns an allowed outcome to auto-execute, or a denied outcome to escalate
// to user confirmation.
func judgeReadInSessionRootsOptionalPath(ctx context.Context, input json.RawMessage) tools.JudgeOutcome {
	return judgeReadWithPathDefault(ctx, input, true)
}

// judgeReadWithPathDefault is the shared implementation for the read-side
// judges. When pathOptional is true, an empty `path` is resolved to the
// workspace root (matching the Execute semantics of glob/ripgrep); otherwise
// an empty `path` fails closed.
func judgeReadWithPathDefault(ctx context.Context, input json.RawMessage, pathOptional bool) tools.JudgeOutcome {
	var params struct {
		Path string `json:"path"`
	}
	if err := json.Unmarshal(input, &params); err != nil {
		// Cannot determine path — fail closed and escalate to confirmation.
		// Unassessable input is hard (see hardOutcome).
		return hardOutcome(false, "cannot determine target path", tools.ReasonCodeUnassessablePath)
	}

	if params.Path == "" {
		if !pathOptional {
			return hardOutcome(false, "cannot determine target path", tools.ReasonCodeUnassessablePath)
		}
		ws := tools.WorkspacePathFrom(ctx)
		if ws == "" {
			return hardOutcome(false, "cannot determine target path", tools.ReasonCodeUnassessablePath)
		}
		return judgeReadInSessionRootsForPath(ctx, ws)
	}

	resolved := resolvePath(ctx, params.Path)
	if err := validateResolvedPath(resolved); err != nil {
		return softOutcome(false, err.Error(), tools.ReasonCodeOutsideSessionRoots)
	}
	return judgeReadInSessionRootsForPath(ctx, resolved)
}
