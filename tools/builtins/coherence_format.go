package builtins

import (
	"fmt"
	"time"

	"github.com/v0lka/sp4rk/tools"
)

// formatWriteConflict formats a coherence write-check conflict as an error
// message for the mutating file tools (write_file, edit_file, delete_file).
func formatWriteConflict(c *tools.CoherenceConflict) string {
	age := time.Since(c.ModifiedAt).Truncate(time.Second)
	return fmt.Sprintf(
		"File conflict: %s was modified since your last read.\n"+
			"Modified by: %q (%s ago).\n"+
			"Your snapshot: mtime=%s, size=%d. Current: mtime=%s, size=%d.\n"+
			"Action required: re-read the file with read_file before retrying the operation.\n"+
			"Hint: if the modification came from a shell command YOU ran earlier in this "+
			"session (build, formatter, code generation), it is your own change — re-read "+
			"the file and retry the operation; if anything else made it (another session, "+
			"the user, or an external process), re-read the file, review that change, and "+
			"reconcile your operation with it before retrying.",
		c.Path,
		c.ModifiedBy,
		age,
		c.LastReadSig.ModTime.Format(time.RFC3339),
		c.LastReadSig.Size,
		c.CurrentSig.ModTime.Format(time.RFC3339),
		c.CurrentSig.Size,
	)
}

// formatReadConflict formats a coherence conflict as a warning annotation for read operations.
func formatReadConflict(c *tools.CoherenceConflict) string {
	return fmt.Sprintf(
		"[!] This file was modified by %q since your last read. Review carefully.\n",
		c.ModifiedBy,
	)
}
