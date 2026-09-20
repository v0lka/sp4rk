// SPDX-License-Identifier: Apache-2.0

package tools

import (
	"reflect"
	"strings"
	"testing"

	"github.com/v0lka/flowsh/api"
)

func TestParseShellKind_Supported(t *testing.T) {
	for _, kind := range []string{"bash", "sh", "zsh", "ksh", "dash", "powershell", "pwsh"} {
		got, err := ParseShellKind(kind)
		if err != nil {
			t.Errorf("ParseShellKind(%q): unexpected error %v", kind, err)
			continue
		}
		if string(got) != kind {
			t.Errorf("ParseShellKind(%q) = %q", kind, got)
		}
	}
}

func TestParseShellKind_RejectsUnanalyzableShells(t *testing.T) {
	// Shells flowsh cannot parse must be rejected: accepting them would
	// silently drop the deterministic safety floor (or fail closed on every
	// call). The closed list is the contract.
	for _, kind := range []string{"fish", "nushell", "nu", "csh", "tcsh", "cmd", "cmd.exe", "", "BASH", "Bash"} {
		if got, err := ParseShellKind(kind); err == nil {
			t.Errorf("ParseShellKind(%q) = %q, want error", kind, got)
		}
	}
}

func TestShellKind_Family(t *testing.T) {
	bashFamily := map[ShellKind]bool{
		ShellKindBash: true, ShellKindSh: true, ShellKindZsh: true,
		ShellKindKsh: true, ShellKindDash: true,
	}
	poshFamily := map[ShellKind]bool{
		ShellKindPowerShell: true, ShellKindPwsh: true,
	}
	for kind := range bashFamily {
		if got := kind.Family(); got != ShellFamilyBash {
			t.Errorf("%v.Family() = %q, want bash", kind, got)
		}
	}
	for kind := range poshFamily {
		if got := kind.Family(); got != ShellFamilyPowerShell {
			t.Errorf("%v.Family() = %q, want powershell", kind, got)
		}
	}
	if got := ShellKind("nushell").Family(); got != "" {
		t.Errorf("unknown kind Family() = %q, want empty", got)
	}
}

func TestShellKindToAnalysisLang(t *testing.T) {
	if lang, ok := ShellKindToAnalysisLang(ShellKindZsh); !ok || lang != api.LangBash {
		t.Errorf("zsh → %q,%v want bash,true", lang, ok)
	}
	if lang, ok := ShellKindToAnalysisLang(ShellKindPwsh); !ok || lang != api.LangPowerShell {
		t.Errorf("pwsh → %q,%v want powershell,true", lang, ok)
	}
	if _, ok := ShellKindToAnalysisLang(ShellKind("fish")); ok {
		t.Error("fish must not map to any analysis dialect")
	}
}

func TestShellInvocation_Validate(t *testing.T) {
	valid := ShellInvocation{
		Binary: "/opt/homebrew/bin/zsh",
		Args:   []string{"-l", "-c", ShellCommandPlaceholder},
		Kind:   ShellKindZsh,
	}
	if err := valid.Validate(); err != nil {
		t.Fatalf("valid invocation rejected: %v", err)
	}

	cases := []struct {
		name string
		inv  ShellInvocation
	}{
		{"empty binary", ShellInvocation{Args: []string{ShellCommandPlaceholder}, Kind: ShellKindBash}},
		{"whitespace binary", ShellInvocation{Binary: "   ", Args: []string{ShellCommandPlaceholder}, Kind: ShellKindBash}},
		{"binary is placeholder", ShellInvocation{Binary: ShellCommandPlaceholder, Args: []string{ShellCommandPlaceholder}, Kind: ShellKindBash}},
		{"no args", ShellInvocation{Binary: "zsh", Kind: ShellKindZsh}},
		{"no placeholder", ShellInvocation{Binary: "zsh", Args: []string{"-c"}, Kind: ShellKindZsh}},
		{"two placeholders", ShellInvocation{Binary: "zsh", Args: []string{ShellCommandPlaceholder, ShellCommandPlaceholder}, Kind: ShellKindZsh}},
		{"embedded placeholder", ShellInvocation{Binary: "zsh", Args: []string{"-c" + ShellCommandPlaceholder}, Kind: ShellKindZsh}},
		{"unsupported kind", ShellInvocation{Binary: "fish", Args: []string{ShellCommandPlaceholder}, Kind: ShellKind("fish")}},
		{"empty kind", ShellInvocation{Binary: "zsh", Args: []string{ShellCommandPlaceholder}}},
	}
	for _, tc := range cases {
		if err := tc.inv.Validate(); err == nil {
			t.Errorf("%s: expected error, got nil", tc.name)
		}
	}
}

func TestShellInvocation_Argv(t *testing.T) {
	inv := ShellInvocation{
		Binary: "/bin/sh",
		Args:   []string{"-c", ShellCommandPlaceholder},
		Kind:   ShellKindSh,
	}
	argv := inv.Argv("echo hello")
	want := []string{"-c", "echo hello"}
	if !reflect.DeepEqual(argv, want) {
		t.Fatalf("Argv = %v, want %v", argv, want)
	}
	// The command must be a SINGLE argv element, verbatim — no re-splitting.
	argv = inv.Argv("echo a  b | grep x")
	if len(argv) != 2 || argv[1] != "echo a  b | grep x" {
		t.Fatalf("command not preserved as one argv element: %v", argv)
	}
	// Argv must not mutate the template.
	if !reflect.DeepEqual(inv.Args, []string{"-c", ShellCommandPlaceholder}) {
		t.Fatalf("template mutated: %v", inv.Args)
	}
}

func TestShellInvocation_DisplayCommand(t *testing.T) {
	inv := DefaultBashInvocation()
	if got, want := inv.DisplayCommand(), "bash -c <command>"; got != want {
		t.Fatalf("DisplayCommand = %q, want %q", got, want)
	}
	inv = DefaultPoshInvocation()
	if got, want := inv.DisplayCommand(), "powershell.exe -NoProfile -NonInteractive -Command <command>"; got != want {
		t.Fatalf("DisplayCommand = %q, want %q", got, want)
	}
	inv = ShellInvocation{Binary: "/usr/local/bin/my shell", Args: []string{"-x", ShellCommandPlaceholder}, Kind: ShellKindSh}
	if got := inv.DisplayCommand(); !strings.Contains(got, `"/usr/local/bin/my shell"`) {
		t.Fatalf("DisplayCommand should quote whitespace-bearing binary, got %q", got)
	}
}

func TestShellInvocation_DefaultsValidAndEqual(t *testing.T) {
	for name, inv := range map[string]ShellInvocation{
		"bash": DefaultBashInvocation(),
		"posh": DefaultPoshInvocation(),
	} {
		if err := inv.Validate(); err != nil {
			t.Fatalf("default %s invocation invalid: %v", name, err)
		}
		if inv.IsZero() {
			t.Fatalf("default %s invocation reports zero", name)
		}
	}
	if !DefaultBashInvocation().Equal(DefaultBashInvocation()) {
		t.Error("Equal must hold for identical invocations")
	}
	a := DefaultBashInvocation()
	a.Args[0] = "-lc"
	if a.Equal(DefaultBashInvocation()) {
		t.Error("Equal must detect arg differences")
	}
}
