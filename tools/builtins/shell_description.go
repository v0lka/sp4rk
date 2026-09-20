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
// share; everything from it onward is invocation-independent guidance
// (usage policy, inputs, outputs, examples). The composed header replaces
// only the legacy "Purpose: … via <default invocation>" prefix.
const shellDescriptionFallbackMark = " — the fallback for what"

// composeShellDescription builds the tool description for a possibly
// overridden shell invocation. The default (built-in) invocation returns the
// legacy description byte-for-byte; an override replaces only the header with
// the real launch shape plus the declared shell compatibility, keeping the
// invocation-independent tail intact.
func composeShellDescription(legacy string, invocation, defaultInvocation tools.ShellInvocation) string {
	if invocation.Equal(defaultInvocation) {
		return legacy
	}

	idx := strings.Index(legacy, shellDescriptionFallbackMark)
	if idx < 0 {
		// Defensive: an unexpected legacy shape falls back to prepending the
		// invocation facts before the full legacy text rather than dropping
		// the shell declaration.
		return shellInvocationHeader(invocation) + "\n\n" + legacy
	}

	tail := legacy[idx+1:] // keep the leading "—" separator onward
	return shellInvocationHeader(invocation) + " " + tail
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
