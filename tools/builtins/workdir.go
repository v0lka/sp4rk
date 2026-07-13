package builtins

import (
	"fmt"
	"os"

	"github.com/v0lka/sp4rk/pathutil"
)

// validateWorkDir checks that dir is within a session root (workspace, temp
// directory, or any additional allowed root) or the system temp directory.
// If roots is empty, any directory is accepted (no workspace context means no
// containment is possible). Callers should build roots via
// tools.SessionRoots(ctx), which already includes the workspace and temp dir.
//
// Containment is evaluated through pathutil.IsWithinPath (symlink-resolved)
// rather than inline prefix checks, per the project's centralized path API.
//
// Shared by all shell-execution tools (bash_exec on Unix, posh_exec on
// Windows) so it must build on every platform.
func validateWorkDir(dir string, roots []string) error {
	if len(roots) == 0 {
		return nil // no workspace context — cannot enforce containment
	}

	// Allow any session root (workspace, temp directory, allowed roots).
	for _, root := range roots {
		if root == "" {
			continue
		}
		if ok, _ := pathutil.IsWithinPath(root, dir); ok {
			return nil
		}
	}

	// Allow system temp directory.
	if ok, _ := pathutil.IsWithinPath(os.TempDir(), dir); ok {
		return nil
	}

	return fmt.Errorf("path %q is outside the session roots and temp directory", dir)
}
