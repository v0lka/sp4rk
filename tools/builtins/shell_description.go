// SPDX-License-Identifier: Apache-2.0
//
// shell_description.go — description composition for the shell-exec tools
// (bash_exec on Unix, posh_exec on Windows). The tool description is the
// channel that reaches EVERY agent able to call the tool: it is part of the
// tools catalog attached to each request (Conductor, subagents, verifiers,
// E2S loops alike). When the host configures an operator shell-invocation
// override, the description must therefore name the actual launch command and
// the declared shell compatibility — otherwise the model keeps writing bash
// for a zsh wrapper (or PowerShell for a pwsh wrapper) and every command
// fails on syntax.

package builtins

import (
	"strings"

	"github.com/v0lka/sp4rk/tools"
)

// shellDescriptionFallbackMark is the fixed tail both legacy descriptions
// share. From it onward the text is the tool's usage policy, inputs, outputs
// and examples — written in the tool's own syntax family (PowerShell on
// posh_exec, bash on bash_exec), NOT dialect-neutral. The composed header
// replaces only the legacy "Purpose: … via <default invocation>" prefix, and
// the tail is reused only when the override keeps the same syntax family.
const shellDescriptionFallbackMark = " — the fallback for what"

// shellDescriptionNeutralTail is the dialect-neutral guidance appended after
// the invocation header when an operator override changes the tool's SYNTAX
// FAMILY (a bash-family binary on posh_exec, e.g. a zsh wrapper, or a
// PowerShell-family binary on bash_exec). The tool's own legacy tail is
// written in the other family's syntax, so reusing it would contradict the
// header (PowerShell examples under a zsh declaration, or vice versa). This
// tail names the declared shell generically and carries no family-specific
// examples.
const shellDescriptionNeutralTail = "— the fallback for what no dedicated tool covers: builds, test runs, package managers, git operations, system tasks.\n" +
	"Use when: reading (read_file), editing (edit_file), listing (list_directory) and searching (ripgrep, glob) all have dedicated tools — reach for the shell only when they cannot do the job.\n" +
	"Inputs: command (a statement in the declared shell above; pipelines and redirections as that shell supports them); optional working_directory (absolute path for the execution context); optional timeout, a JSON string with a unit suffix, e.g. \"30s\" or \"2m\" (default \"60s\", capped at the configured max).\n" +
	"Outputs: combined stdout and stderr; failing exit codes surface as errors carrying the output. Keep output minimal to avoid flooding context."

// composeShellDescription builds the tool description for a possibly
// overridden shell invocation. The default (built-in) invocation returns the
// legacy description byte-for-byte; a same-family override replaces only the
// header with the real launch shape plus the declared shell compatibility,
// keeping the tool's dialect-appropriate tail intact. A CROSS-family override
// (the declared kind belongs to the other syntax family, so the legacy tail
// speaks the wrong dialect) replaces the tail too with the dialect-neutral
// tail, so the description never mixes syntax families.
func composeShellDescription(legacy string, invocation, defaultInvocation tools.ShellInvocation) string {
	if invocation.Equal(defaultInvocation) {
		return legacy
	}

	header := shellInvocationHeader(invocation)
	if invocation.Kind.Family() != defaultInvocation.Kind.Family() {
		return header + " " + shellDescriptionNeutralTail
	}

	idx := strings.Index(legacy, shellDescriptionFallbackMark)
	if idx < 0 {
		// Defensive: an unexpected legacy shape falls back to prepending the
		// invocation facts before the full legacy text rather than dropping
		// the shell declaration.
		return header + "\n\n" + legacy
	}

	tail := legacy[idx+1:] // keep the leading "—" separator onward
	return header + " " + tail
}

// shellInvocationHeader renders the "Purpose:" header line for an overridden
// invocation: the actual launch command plus the declared shell the agent's
// command text must be written in.
func shellInvocationHeader(invocation tools.ShellInvocation) string {
	kind := string(invocation.Kind)
	var sb strings.Builder
	sb.WriteString("Purpose: execute a shell command via operator-configured invocation `")
	sb.WriteString(invocation.DisplayCommand())
	sb.WriteString("` — the command text you pass MUST be ")
	sb.WriteString(kind)
	sb.WriteString("-compatible (declared shell: ")
	sb.WriteString(kind)
	sb.WriteString(")")
	return sb.String()
}
