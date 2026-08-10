package tools

import (
	"runtime"
	"testing"
)

func TestIsHarmlessDevicePath(t *testing.T) {
	tests := []struct {
		name string
		path string
		want bool
	}{
		{"empty", "", false},
		{"null", "/dev/null", true},
		{"full", "/dev/full", true},
		// Process streams are deliberately NOT harmless: their targets are
		// host-defined file descriptors, so writing can exfiltrate and reading
		// consumes the agent's input. On macOS these are symlinks into /dev/fd/*.
		{"stdin", "/dev/stdin", false},
		{"stdout", "/dev/stdout", false},
		{"stderr", "/dev/stderr", false},
		// Deliberately NOT harmless — must not match.
		{"zero", "/dev/zero", false},
		{"urandom", "/dev/urandom", false},
		{"random", "/dev/random", false},
		{"tty", "/dev/tty", false},
		{"disk", "/dev/sda1", false},
		{"normal file", "/tmp/foo.txt", false},
		{"normal relative", "dev/null", false}, // not absolute → not a device
		// A bare "nul" is an ordinary file on a POSIX host (not a device), so
		// it is NOT harmless there. On Windows it is the reserved NUL device —
		// the /dev/null equivalent — and therefore harmless. The expectation
		// is thus host-dependent, matching IsHarmlessDevicePath's OS-gating.
		{"bare nul", "nul", runtime.GOOS == "windows"},
		{"ordinary file named devnull", "/home/me/devnull", false},
		// Variations that should still match after normalization.
		{"trailing slash cleaned", "/dev/null/", true},
		{"double slash cleaned", "/dev//null", true},
		{"dot resolved", "/dev/./null", true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := IsHarmlessDevicePath(tt.path); got != tt.want {
				t.Errorf("IsHarmlessDevicePath(%q) = %v, want %v", tt.path, got, tt.want)
			}
		})
	}
}

// TestIsNullDeviceForOS exercises the Windows NUL matcher directly for every
// target OS (Linux, macOS, Windows), so the branch-heavy matching logic has
// coverage on POSIX CI runners where isNullDeviceForOS(_, runtime.GOOS) always
// returns false.
func TestIsNullDeviceForOS(t *testing.T) {
	tests := []struct {
		name string
		path string
		goos string
		want bool
	}{
		// Windows host: NUL matches in the final component, case-insensitively.
		{"win bare upper", "NUL", "windows", true},
		{"win bare lower", "nul", "windows", true},
		{"win drive upper", `C:\NUL`, "windows", true},
		{"win drive slash lower", `D:/nul`, "windows", true},
		{"win verbatim prefix", `\\?\C:\NUL`, "windows", true},
		{"win device prefix", `\\.\NUL`, "windows", true},
		// "nul" is reserved in the LAST component on Windows, so a path like
		// C:\dir\nul is still the device (Windows forbids a real file named
		// "nul" with no extension).
		{"win in subdir", `C:\dir\nul`, "windows", true},
		// Extensions are NOT stripped — these are real files, not the device.
		{"win nul.go", `src\nul.go`, "windows", false},
		{"win NUL.txt", `NUL.txt`, "windows", false},
		{"win nul_test.go", `pkg\nul_test.go`, "windows", false},
		// Windows negatives.
		{"win null", `null`, "windows", false},
		{"win nully", `nully`, "windows", false},
		{"win empty", "", "windows", false},
		{"win ordinary path", `C:\repo\main.go`, "windows", false},
		// POSIX hosts: "nul" is an ordinary file — never the device.
		{"linux bare nul", "nul", "linux", false},
		{"linux drive path", `C:\NUL`, "linux", false},
		{"darwin bare nul", "nul", "darwin", false},
		{"darwin drive path", `C:\NUL`, "darwin", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := isNullDeviceForOS(tt.path, tt.goos); got != tt.want {
				t.Errorf("isNullDeviceForOS(%q, %q) = %v, want %v", tt.path, tt.goos, got, tt.want)
			}
		})
	}
}
