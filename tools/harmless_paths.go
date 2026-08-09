package tools

import (
	"path"
	"path/filepath"
	"runtime"
	"strings"
)

// harmlessPOSIXDevices lists POSIX special-device paths whose access cannot
// leak data outside the workspace nor persist unwanted changes. Both entries
// are real character devices (never symlinks) wherever they exist:
//   - /dev/null — reads return EOF; writes are discarded (the bit bucket).
//   - /dev/full — reads return EOF; writes fail with ENOSPC (nothing stored).
//     /dev/full is Linux-only; on macOS it does not exist, so an operation
//     targeting it simply fails at open(2) — still harmless.
//
// Deliberately EXCLUDED:
//   - /dev/stdin, /dev/stdout, /dev/stderr — the process's standard streams.
//     Their target is host-defined (whatever fd 0/1/2 are bound to: a pipe to
//     another process, a redirected file outside the workspace, a socket), so
//     writing to /dev/stdout can exfiltrate data and reading /dev/stdin
//     consumes the agent's input. They are NOT provably safe. On macOS they are
//     symlinks into /dev/fd/*, which the symlink gate must therefore walk and
//     resolve (see [checkPathsForSymlinks]); on Linux they are character
//     devices and produce no symlink traversal. On Windows there is no /dev.
//   - /dev/zero and /dev/urandom (infinite/unbounded read), /dev/random,
//     /dev/tty (interactive), real block/character devices, and named pipes.
//
// The list is conservative on purpose — every entry must be provably safe to
// both read from and write to.
//
// This is the single source of truth consulted by IsHarmlessDevicePath, which
// the path-locality checks (file-operation judge fast-path, file-tool read/
// write judges, and shell-exec containment) consult to exempt harmless device
// paths from the "outside workspace" determination. The symlink gate does NOT
// consult it — it resolves every path via Lstat (see [checkPathsForSymlinks]).
var harmlessPOSIXDevices = map[string]bool{
	"/dev/null": true,
	"/dev/full": true,
}

// IsHarmlessDevicePath reports whether absPath is a harmless special device —
// a bit-bucket or error-sink pseudo-device — whose access cannot leak data
// outside the workspace nor persist unwanted changes.
//
// Such paths are excluded from path-locality determination (the file-operation
// judge fast-path, the file-tool read/write judges, and the shell-exec
// containment check [PathsOutsideRoots]) so they never force a
// user-confirmation prompt on their own. For example `read_file /dev/null`,
// `write_file /dev/null`, and `cat file > /dev/null` are not treated as
// out-of-workspace operations.
//
// Recognized harmless devices:
//   - POSIX: /dev/null and /dev/full (matched as cleaned absolute paths).
//   - Windows: NUL — the /dev/null equivalent — matched case-insensitively as
//     the final path component, with or without a drive/path prefix or verbatim
//     long-path prefix (e.g. "NUL", "nul", "C:\\NUL", "\\\\?\\C:\\NUL"). A file
//     extension is NOT stripped, so "nul.go" or "NUL.txt" are treated as real
//     files, not the device. NUL is a reserved device name only on Windows, so
//     the check is host-gated.
//
// absPath is cleaned and separator-normalized before the lookup. Returns false
// for the empty string and for any path that is not one of the listed devices.
func IsHarmlessDevicePath(absPath string) bool {
	if absPath == "" {
		return false
	}
	// POSIX device paths are forward-slash-rooted on every platform (shell-
	// token resolution keeps them so). Normalize separators and clean with the
	// POSIX path cleaner: drive letters / UNC prefixes never occur in /dev
	// paths, so path.Clean is exact for this lookup, and ToSlash guarantees the
	// forward-slash form on Windows hosts too.
	normalized := path.Clean(filepath.ToSlash(absPath))
	if harmlessPOSIXDevices[normalized] {
		return true
	}
	return isNullDeviceForOS(absPath, runtime.GOOS)
}

// isNullDeviceForOS reports whether p is the Windows null device (NUL), the
// /dev/null equivalent, evaluated as if the host OS were goos. NUL is a
// reserved name recognized in the final path component regardless of drive/path
// prefix, case-insensitively. The check is OS-gated: on a POSIX host a file
// literally named "nul" is an ordinary file and must NOT be treated as the null
// device. Separated from the runtime call so the matching logic is unit-testable
// on any platform.
//
// A file extension is deliberately NOT stripped: a name like "nul.go" or
// "NUL.txt" is far more likely a real source file than a device reference, and
// exempting it would skip the containment check (a security gap). Prompting on a
// genuine "NUL.txt" device reference is the safe, conservative trade-off.
//
// Examples that match when goos == "windows": "NUL", "nul", "C:\\NUL", "D:/nul",
// "\\\\?\\C:\\NUL", "\\\\.\\NUL", "C:\\dir\\nul" (reserved in the last component).
func isNullDeviceForOS(p, goos string) bool {
	if goos != "windows" {
		return false
	}
	if p == "" {
		return false
	}
	// Strip a Windows long-path verbatim prefix (\\?\ or \\.\) if present.
	s := p
	for _, pre := range []string{`\\?\`, `\\.\`} {
		if strings.HasPrefix(p, pre) {
			s = strings.TrimPrefix(p, pre)
			break
		}
	}
	// Isolate the final path component.
	base := s
	if idx := strings.LastIndexAny(s, `\/`); idx >= 0 {
		base = s[idx+1:]
	}
	// Match the bare reserved name exactly (case-insensitive). No extension
	// stripping — see the function doc.
	return strings.EqualFold(base, "NUL")
}
