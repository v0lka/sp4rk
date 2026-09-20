//go:build !windows

// SPDX-License-Identifier: Apache-2.0

package builtins

import (
	"context"
	"strings"
	"testing"

	"github.com/v0lka/sp4rk/tools"
)

// TestBashExecTool_DefaultInvocationDescriptionIsLegacy pins the
// byte-for-byte contract: without an override the tool description is exactly
// the legacy bash description, so hosts that never configure an invocation
// see zero prompt drift.
func TestBashExecTool_DefaultInvocationDescriptionIsLegacy(t *testing.T) {
	tool, err := NewBashExecToolWithInvocation(nil, DefaultBashTimeouts(), tools.DefaultBashInvocation())
	if err != nil {
		t.Fatalf("constructor: %v", err)
	}
	if tool.Description() != toolBashDescription {
		t.Errorf("default invocation must keep the legacy description byte-for-byte")
	}
	plain, err := NewBashExecTool(nil)
	if err != nil {
		t.Fatalf("plain constructor: %v", err)
	}
	if plain.Description() != toolBashDescription {
		t.Errorf("NewBashExecTool must keep the legacy description byte-for-byte")
	}
}

// TestBashExecTool_OverrideDescriptionNamesInvocationAndShell verifies the
// prompt channel: an overridden description names the actual launch command
// and the declared shell compatibility, and keeps the invocation-independent
// guidance tail.
func TestBashExecTool_OverrideDescriptionNamesInvocationAndShell(t *testing.T) {
	inv := tools.ShellInvocation{
		Binary: "/opt/homebrew/bin/zsh",
		Args:   []string{"-l", "-c", tools.ShellCommandPlaceholder},
		Kind:   tools.ShellKindZsh,
	}
	tool, err := NewBashExecToolWithInvocation(nil, DefaultBashTimeouts(), inv)
	if err != nil {
		t.Fatalf("constructor: %v", err)
	}
	desc := tool.Description()
	if !strings.Contains(desc, "/opt/homebrew/bin/zsh -l -c <command>") {
		t.Errorf("description must name the actual invocation, got: %s", desc)
	}
	if !strings.Contains(desc, "zsh-compatible") {
		t.Errorf("description must declare the shell compatibility, got: %s", desc)
	}
	// The legacy guidance tail survives.
	if !strings.Contains(desc, "the fallback for what no dedicated tool covers") {
		t.Errorf("description must keep the legacy guidance tail, got: %s", desc)
	}
	if !strings.Contains(desc, `Example: "go test ./core/... 2>&1 | tail -30"`) {
		t.Errorf("description must keep the usage examples, got: %s", desc)
	}
	if tool.DeclaredShellKind() != tools.ShellKindZsh {
		t.Errorf("DeclaredShellKind = %q, want zsh", tool.DeclaredShellKind())
	}
}

// TestBashExecTool_OverrideInvalidInvocationRejected covers constructor
// validation: a mis-shaped invocation (no/many placeholders, unsupported
// shell) must fail construction — never silently run.
func TestBashExecTool_OverrideInvalidInvocationRejected(t *testing.T) {
	cases := map[string]tools.ShellInvocation{
		"no placeholder":       {Binary: "/bin/sh", Args: []string{"-c"}, Kind: tools.ShellKindSh},
		"two placeholders":     {Binary: "/bin/sh", Args: []string{tools.ShellCommandPlaceholder, tools.ShellCommandPlaceholder}, Kind: tools.ShellKindSh},
		"embedded placeholder": {Binary: "/bin/sh", Args: []string{"-c" + tools.ShellCommandPlaceholder}, Kind: tools.ShellKindSh},
		"empty binary":         {Args: []string{tools.ShellCommandPlaceholder}, Kind: tools.ShellKindSh},
		"unsupported shell":    {Binary: "fish", Args: []string{tools.ShellCommandPlaceholder}, Kind: tools.ShellKind("fish")},
	}
	for name, inv := range cases {
		if _, err := NewBashExecToolWithInvocation(nil, DefaultBashTimeouts(), inv); err == nil {
			t.Errorf("%s: expected constructor error, got nil", name)
		}
	}
}

// TestBashExecTool_OverrideExecute runs a real command through an overridden
// invocation: /bin/sh replaces bash, the command still crosses as one argv
// element, and output handling is unchanged.
func TestBashExecTool_OverrideExecute(t *testing.T) {
	inv := tools.ShellInvocation{
		Binary: "/bin/sh",
		Args:   []string{"-c", tools.ShellCommandPlaceholder},
		Kind:   tools.ShellKindSh,
	}
	tool, err := NewBashExecToolWithInvocation(nil, DefaultBashTimeouts(), inv)
	if err != nil {
		t.Fatalf("constructor: %v", err)
	}
	result, err := tool.Execute(context.Background(), []byte(`{"command": "echo override-hello"}`))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Fatalf("unexpected error result: %s", result.Content)
	}
	if !strings.Contains(result.Content, "override-hello") {
		t.Errorf("expected output to contain 'override-hello', got %q", result.Content)
	}
}

// TestBashExecTool_OverrideBlacklistStillEnforced pins that the operator
// blocklist is evaluated on the COMMAND TEXT regardless of the launch
// wrapper — the override changes how, never what is checked.
func TestBashExecTool_OverrideBlacklistStillEnforced(t *testing.T) {
	inv := tools.ShellInvocation{
		Binary: "/bin/sh",
		Args:   []string{"-c", tools.ShellCommandPlaceholder},
		Kind:   tools.ShellKindSh,
	}
	tool, err := NewBashExecToolWithInvocation([]string{"forbidden-cmd"}, DefaultBashTimeouts(), inv)
	if err != nil {
		t.Fatalf("constructor: %v", err)
	}
	outcome := tool.Judge(context.Background(), []byte(`{"command": "forbidden-cmd --now"}`))
	if outcome.Allow {
		t.Fatalf("blocklisted command allowed under override: %+v", outcome)
	}
	if outcome.Severity != tools.JudgeSeverityHard || outcome.ReasonCode != tools.ReasonCodeCommandBlacklist {
		t.Errorf("unexpected outcome: %+v", outcome)
	}
}
