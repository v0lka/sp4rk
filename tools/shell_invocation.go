// SPDX-License-Identifier: Apache-2.0
//
// shell_invocation.go — the shell-exec invocation model shared by the
// bash_exec (Unix) and posh_exec (Windows) built-ins. A host may let its
// operator override HOW the agent's command is launched (the classic
// `bash -c <command>` / `powershell.exe -NoProfile -NonInteractive -Command
// <command>` wrappers become arbitrary argv templates) and MUST then declare
// which shell the command text is written in, because three things follow
// from that declaration:
//
//  1. the tool description the model sees names the actual invocation and the
//     declared shell (the description travels with every tool catalog, so it
//     reaches every agent that can call the tool);
//  2. the PowerShell UTF-8 output bootstrap is applied only when the declared
//     shell really is a PowerShell host;
//  3. the deterministic flowsh analysis (tools/shellanalysis.go) runs with the
//     dialect of the declared shell family — flowsh understands exactly two
//     syntax families, bash-like and PowerShell-like, so the supported shell
//     list is closed: anything else has no analyzer and is rejected here
//     rather than silently losing the deterministic safety floor.

package tools

import (
	"errors"
	"fmt"
	"strings"

	"github.com/v0lka/flowsh/api"
)

// ShellKind names a shell whose syntax the agent's command text is written
// in. The set is closed on purpose: every member maps onto one of the two
// syntax families the flowsh analyzer can reason about, so a declared kind
// always keeps the deterministic shell-safety floor intact. Shells outside
// this set (fish, nushell, csh, cmd.exe, …) have no analyzer; hosts must not
// accept them — a command the analyzer cannot parse would either run without
// the deterministic criteria or fail closed on every single call.
type ShellKind string

// Supported shell kinds. The bash family (Bash, Sh, Zsh, Ksh, Dash) shares
// the POSIX-derived syntax flowsh parses with its bash grammar; the
// PowerShell family (PowerShell, Pwsh) shares the PowerShell grammar.
const (
	ShellKindBash       ShellKind = "bash"
	ShellKindSh         ShellKind = "sh"
	ShellKindZsh        ShellKind = "zsh"
	ShellKindKsh        ShellKind = "ksh"
	ShellKindDash       ShellKind = "dash"
	ShellKindPowerShell ShellKind = "powershell"
	ShellKindPwsh       ShellKind = "pwsh"
)

// ShellFamily is the syntax family a ShellKind belongs to — the coarser
// notion the flowsh analyzer (and the PowerShell-specific bootstrap) key on.
type ShellFamily string

const (
	// ShellFamilyBash is the POSIX/bash-derived syntax family.
	ShellFamilyBash ShellFamily = "bash"
	// ShellFamilyPowerShell is the PowerShell syntax family.
	ShellFamilyPowerShell ShellFamily = "powershell"
)

// ShellCommandPlaceholder marks the single argv element of a ShellInvocation
// that is replaced by the agent's command text at execution time. The
// placeholder must be an ENTIRE argv element — the command is passed as one
// argv element (exactly like `bash -c <command>` passes the command), never
// spliced into another argument string, so no extra quoting layer is
// introduced between the agent and the shell.
const ShellCommandPlaceholder = "{command}"

// ParseShellKind resolves a shell-kind string against the closed supported
// set. Matching is exact (lowercase), so a config layer can surface a precise
// "one of …" error instead of a fuzzy guess.
func ParseShellKind(s string) (ShellKind, error) {
	switch ShellKind(s) {
	case ShellKindBash, ShellKindSh, ShellKindZsh, ShellKindKsh, ShellKindDash,
		ShellKindPowerShell, ShellKindPwsh:
		return ShellKind(s), nil
	default:
		return "", fmt.Errorf("unsupported shell kind %q (want one of: bash, sh, zsh, ksh, dash, powershell, pwsh)", s)
	}
}

// Family reports the syntax family of the kind.
func (k ShellKind) Family() ShellFamily {
	switch k {
	case ShellKindBash, ShellKindSh, ShellKindZsh, ShellKindKsh, ShellKindDash:
		return ShellFamilyBash
	case ShellKindPowerShell, ShellKindPwsh:
		return ShellFamilyPowerShell
	default:
		return ""
	}
}

// ShellKindToAnalysisLang maps a declared shell kind onto the flowsh dialect
// used by the deterministic shell analysis. Unknown kinds report false —
// callers fail closed (see tools/shellanalysis.go).
func ShellKindToAnalysisLang(k ShellKind) (api.Lang, bool) {
	switch k.Family() {
	case ShellFamilyBash:
		return api.LangBash, true
	case ShellFamilyPowerShell:
		return api.LangPowerShell, true
	default:
		return "", false
	}
}

// ShellInvocation describes HOW the shell-exec tool launches the agent's
// command: a binary plus an argv template in which exactly one element equals
// ShellCommandPlaceholder. At execution time the placeholder element is
// replaced by the command text as a single argv element, so the command
// crosses the process boundary verbatim — no shell, no quoting, no injection
// surface beyond what the declared shell itself parses.
type ShellInvocation struct {
	// Binary is the executable path or name resolved via PATH (exec semantics).
	Binary string
	// Args is the argv template; exactly one element must equal
	// ShellCommandPlaceholder. It must contain at least the placeholder.
	Args []string
	// Kind declares the shell the command text is written in. It drives the
	// tool description, the PowerShell-only UTF-8 bootstrap, and the analysis
	// dialect.
	Kind ShellKind
}

// Validate reports whether the invocation is well-formed: a non-empty binary
// (never the placeholder itself), a non-empty argv template containing
// exactly one full-element placeholder, and a supported shell kind. Any other
// element CONTAINING the placeholder as a substring is rejected too — a
// partial splice would reintroduce the quoting layer the element model
// exists to avoid.
func (inv ShellInvocation) Validate() error {
	if strings.TrimSpace(inv.Binary) == "" {
		return errors.New("shell invocation: binary is required")
	}
	if inv.Binary == ShellCommandPlaceholder {
		return fmt.Errorf("shell invocation: binary must not be the %s placeholder", ShellCommandPlaceholder)
	}
	if len(inv.Args) == 0 {
		return fmt.Errorf("shell invocation: args must contain the %s placeholder", ShellCommandPlaceholder)
	}
	placeholders := 0
	for _, arg := range inv.Args {
		if arg == ShellCommandPlaceholder {
			placeholders++
			continue
		}
		if strings.Contains(arg, ShellCommandPlaceholder) {
			return fmt.Errorf("shell invocation: %s must be a standalone argv element, not embedded in %q", ShellCommandPlaceholder, arg)
		}
	}
	if placeholders != 1 {
		return fmt.Errorf("shell invocation: args must contain exactly one %s element, got %d", ShellCommandPlaceholder, placeholders)
	}
	if _, err := ParseShellKind(string(inv.Kind)); err != nil {
		return fmt.Errorf("shell invocation: %w", err)
	}
	return nil
}

// IsZero reports whether the invocation is unset (all fields empty).
func (inv ShellInvocation) IsZero() bool {
	return inv.Binary == "" && len(inv.Args) == 0 && inv.Kind == ""
}

// Equal reports whether two invocations are identical (field-wise; Args is
// compared element-wise, not by reference).
func (inv ShellInvocation) Equal(other ShellInvocation) bool {
	if inv.Binary != other.Binary || inv.Kind != other.Kind || len(inv.Args) != len(other.Args) {
		return false
	}
	for i := range inv.Args {
		if inv.Args[i] != other.Args[i] {
			return false
		}
	}
	return true
}

// Argv renders the process argv for command: the placeholder element is
// replaced by the command text verbatim. The invocation must have passed
// Validate; as a defensive fallback (never on validated input) a commandless
// template appends the command as the final element so a mis-shaped
// invocation cannot silently drop the command.
func (inv ShellInvocation) Argv(command string) []string {
	argv := make([]string, 0, len(inv.Args)+1)
	replaced := false
	for _, arg := range inv.Args {
		if arg == ShellCommandPlaceholder && !replaced {
			argv = append(argv, command)
			replaced = true
			continue
		}
		argv = append(argv, arg)
	}
	if !replaced {
		argv = append(argv, command)
	}
	return argv
}

// DisplayCommand renders the invocation for prompts: binary plus template
// arguments with the placeholder shown as <command>. Whitespace-bearing
// tokens (a binary path like "/usr/local/bin/my shell") are quoted so the
// rendering round-trips visually.
func (inv ShellInvocation) DisplayCommand() string {
	quote := func(s string) string {
		if strings.ContainsAny(s, " \t") {
			return fmt.Sprintf("%q", s)
		}
		return s
	}
	parts := make([]string, 0, len(inv.Args)+1)
	parts = append(parts, quote(inv.Binary))
	for _, arg := range inv.Args {
		if arg == ShellCommandPlaceholder {
			parts = append(parts, "<command>")
			continue
		}
		parts = append(parts, quote(arg))
	}
	return strings.Join(parts, " ")
}

// DefaultBashInvocation is the built-in bash_exec launch shape:
// `bash -c <command>` with bash-family command syntax.
func DefaultBashInvocation() ShellInvocation {
	return ShellInvocation{
		Binary: "bash",
		Args:   []string{"-c", ShellCommandPlaceholder},
		Kind:   ShellKindBash,
	}
}

// DefaultPoshInvocation is the built-in posh_exec launch shape:
// `powershell.exe -NoProfile -NonInteractive -Command <command>` with
// PowerShell-family command syntax.
func DefaultPoshInvocation() ShellInvocation {
	return ShellInvocation{
		Binary: "powershell.exe",
		Args:   []string{"-NoProfile", "-NonInteractive", "-Command", ShellCommandPlaceholder},
		Kind:   ShellKindPowerShell,
	}
}
