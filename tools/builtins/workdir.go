package builtins

import (
	"context"
	"fmt"
	"os"

	"github.com/v0lka/sp4rk/tools"
)

// validateWorkDir checks that dir is within a session root (workspace, temp
// directory, or any additional allowed root) or the system temp directory.
// If roots is empty, any directory is accepted (no workspace context means no
// containment is possible). Callers should build roots via
// tools.SessionRoots(ctx), which already includes the workspace and temp dir.
//
// Containment is evaluated through [tools.IsWithinRoot] (symlink-resolved,
// case-sensitivity per the session flag) rather than inline prefix checks, per
// the project's centralized path API. The flag matches the detected filesystem
// case-sensitivity (macOS APFS / Windows NTFS fold case; Linux ext4 does not)
// so a working directory written with different casing than a session root is
// recognized as local on case-insensitive filesystems only.
//
// Shared by all shell-execution tools (bash_exec on Unix, posh_exec on
// Windows) so it must build on every platform.
func validateWorkDir(ctx context.Context, dir string, roots []string) error {
	if len(roots) == 0 {
		return nil // no workspace context — cannot enforce containment
	}

	// Allow any session root (workspace, temp directory, allowed roots).
	for _, root := range roots {
		if root == "" {
			continue
		}
		if tools.IsWithinRoot(ctx, root, dir) {
			return nil
		}
	}

	// Allow system temp directory.
	if tools.IsWithinRoot(ctx, os.TempDir(), dir) {
		return nil
	}

	return fmt.Errorf("path %q is outside the session roots and temp directory", dir)
}
