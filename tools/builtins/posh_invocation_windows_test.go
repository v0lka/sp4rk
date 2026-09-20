//go:build windows

// SPDX-License-Identifier: Apache-2.0

package builtins

import (
	"strings"
	"testing"

	"github.com/v0lka/sp4rk/tools"
)

// TestPoshExecTool_DefaultInvocationDescriptionIsLegacy pins the byte-for-byte
// contract: without an override the posh description is exactly the legacy
// const.
func TestPoshExecTool_DefaultInvocationDescriptionIsLegacy(t *testing.T) {
	tool, err := NewPoshExecToolWithInvocation(nil, DefaultBashTimeouts(), tools.DefaultPoshInvocation())
	if err != nil {
		t.Fatalf("constructor: %v", err)
	}
	if tool.Description() != toolPoshDescription {
		t.Error("default invocation must keep the legacy description byte-for-byte")
	}
}

// TestPoshExecTool_CommandText_BootstrapGating verifies the PowerShell UTF-8
// bootstrap is applied only when the declared shell kind really belongs to
// the PowerShell family — an operator-configured non-PowerShell wrapper must
// receive the command verbatim, never PowerShell statements.
func TestPoshExecTool_CommandText_BootstrapGating(t *testing.T) {
	defaultTool, err := NewPoshExecTool(nil)
	if err != nil {
		t.Fatalf("default constructor: %v", err)
	}
	got := defaultTool.commandText("Get-Date")
	if !strings.HasPrefix(got, "try { [Console]::OutputEncoding") || !strings.HasSuffix(got, "; Get-Date") {
		t.Errorf("PowerShell-kind command must carry the UTF-8 bootstrap, got %q", got)
	}

	pwshTool, err := NewPoshExecToolWithInvocation(nil, DefaultBashTimeouts(), tools.ShellInvocation{
		Binary: "pwsh.exe",
		Args:   []string{"-NoProfile", "-Command", tools.ShellCommandPlaceholder},
		Kind:   tools.ShellKindPwsh,
	})
	if err != nil {
		t.Fatalf("pwsh constructor: %v", err)
	}
	got = pwshTool.commandText("Get-Date")
	if !strings.HasPrefix(got, "try { [Console]::OutputEncoding") {
		t.Errorf("pwsh-kind command must still carry the bootstrap (same family), got %q", got)
	}

	zshTool, err := NewPoshExecToolWithInvocation(nil, DefaultBashTimeouts(), tools.ShellInvocation{
		Binary: `C:\msys64\usr\bin\zsh.exe`,
		Args:   []string{"-c", tools.ShellCommandPlaceholder},
		Kind:   tools.ShellKindZsh,
	})
	if err != nil {
		t.Fatalf("zsh constructor: %v", err)
	}
	got = zshTool.commandText("uname -a")
	if got != "uname -a" {
		t.Errorf("non-PowerShell-kind command must pass through verbatim, got %q", got)
	}
	if zshTool.DeclaredShellKind() != tools.ShellKindZsh {
		t.Errorf("DeclaredShellKind = %q, want zsh", zshTool.DeclaredShellKind())
	}
	desc := zshTool.Description()
	if !strings.Contains(desc, `C:\msys64\usr\bin\zsh.exe -c <command>`) {
		t.Errorf("override description must name the actual invocation, got: %s", desc)
	}
	if !strings.Contains(desc, "zsh-compatible") {
		t.Errorf("override description must declare the shell compatibility, got: %s", desc)
	}
}

// TestPoshExecTool_OverrideInvalidInvocationRejected mirrors the bash-side
// constructor validation on the Windows tool.
func TestPoshExecTool_OverrideInvalidInvocationRejected(t *testing.T) {
	if _, err := NewPoshExecToolWithInvocation(nil, DefaultBashTimeouts(), tools.ShellInvocation{
		Binary: "nu.exe",
		Args:   []string{tools.ShellCommandPlaceholder},
		Kind:   tools.ShellKind("nushell"),
	}); err == nil {
		t.Error("unsupported shell kind must fail construction")
	}
	if _, err := NewPoshExecToolWithInvocation(nil, DefaultBashTimeouts(), tools.ShellInvocation{
		Binary: "pwsh.exe",
		Args:   []string{"-Command"},
		Kind:   tools.ShellKindPwsh,
	}); err == nil {
		t.Error("missing {command} placeholder must fail construction")
	}
}
