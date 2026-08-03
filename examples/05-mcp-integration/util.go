package main

import "github.com/v0lka/sp4rk/strutil"

// truncate shortens a string to maxLen bytes. The cut is UTF-8-safe
// (delegates to strutil.TruncateUTF8, which appends "…" when truncated).
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return strutil.TruncateUTF8(s, maxLen)
}
