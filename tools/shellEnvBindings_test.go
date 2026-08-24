// SPDX-License-Identifier: Apache-2.0

package tools

import (
	"context"
	"path/filepath"
	"testing"
)

// bindingFixture builds a portable OS-absolute fixture path (forward-slashed
// for embedding in bash command strings) so these tests behave identically on
// POSIX and Windows hosts.
func bindingFixture(parts ...string) string {
	return filepath.ToSlash(osAbsPath(parts...))
}

// TestExtractBashPaths_BoundVarResolvesLiteral covers the core binding case:
// "D=/tmp/build" binds D to a literal value so "$D/a" resolves to
// /tmp/build/a and the command is NOT flagged unexpandable/suspicious. The env
// is pinned empty so the binding is the only possible runtime value.
func TestExtractBashPaths_BoundVarResolvesLiteral(t *testing.T) {
	t.Setenv("D", "")
	bind := bindingFixture("tmp", "build")
	paths, suspicious := extractBashPaths(`D=`+bind+`; cat "$D/a"`, "", "")
	if suspicious {
		t.Fatalf("expected not suspicious for a bound literal var, got suspicious=true")
	}
	want := filepath.Clean(osAbsPath("tmp", "build", "a"))
	if !sliceContains(paths, want) {
		t.Fatalf("expected %q in paths, got %v", want, paths)
	}
}

// TestExtractBashPaths_BoundVarOutOfRootReported covers a bound var whose value
// points outside the roots: "$D/passwd" resolves to the bound value, which is
// both extracted as a path and reported by PathsOutsideRoots.
func TestExtractBashPaths_BoundVarOutOfRootReported(t *testing.T) {
	t.Setenv("D", "")
	target := osAbsPath("etc", "passwd")
	paths, suspicious := extractBashPaths(`D=`+bindingFixture("etc", "passwd")+`; cat "$D/passwd"`, "", "")
	if suspicious {
		t.Fatalf("expected not suspicious for a bound literal var, got suspicious=true")
	}
	want := filepath.Clean(target)
	if !sliceContains(paths, want) {
		t.Fatalf("expected %q in paths, got %v", want, paths)
	}

	ws := t.TempDir()
	ctx := WithWorkspacePath(context.Background(), ws)
	got := PathsOutsideRoots(ctx, `D=`+bindingFixture("etc", "passwd")+`; cat "$D/passwd"`, ShellBash, ws)
	if !sliceContains(got, want) {
		t.Fatalf("expected %q reported outside roots, got %v", want, got)
	}
}

// TestExtractBashPaths_ExportAndPrefixBindings covers the "export VAR=value"
// and the command-prefix "VAR=value cmd ..." forms.
func TestExtractBashPaths_ExportAndPrefixBindings(t *testing.T) {
	t.Setenv("D", "")
	bind := bindingFixture("tmp", "build")

	paths, suspicious := extractBashPaths(`export D=`+bind+`; cat "$D/a"`, "", "")
	want := filepath.Clean(osAbsPath("tmp", "build", "a"))
	if suspicious || !sliceContains(paths, want) {
		t.Fatalf("export binding: paths=%v suspicious=%v", paths, suspicious)
	}

	// The prefix form's binding is not visible to the argument expansion of
	// the same command in real bash (arguments expand first), but with no env
	// value set the only statically known value is the binding — extracted,
	// not suspicious. With a differing env value the union applies instead: see
	// TestExtractBashPaths_PrefixAssignmentWithEnvUnions below.
	paths, suspicious = extractBashPaths(`D=`+bind+` mkdir "$D/x"`, "", "")
	want = filepath.Clean(osAbsPath("tmp", "build", "x"))
	if suspicious || !sliceContains(paths, want) {
		t.Fatalf("prefix binding: paths=%v suspicious=%v", paths, suspicious)
	}
}

// TestExtractBashPaths_BindingWithDifferingEnvStaysSuspicious pins the union
// semantics at the parser layer: a binding never masks the process env, so
// when the referenced name also has a non-empty, differing env value the word
// stays unexpandable (command suspicious) while the bound literal is still
// extracted as a path candidate. A prompt-injection command can otherwise hide
// an out-of-root env dereference behind a decoy in-command assignment that may
// never execute (e.g. inside a branch not taken).
func TestExtractBashPaths_BindingWithDifferingEnvStaysSuspicious(t *testing.T) {
	envVal := osAbsPath("from", "env")
	t.Setenv("D", filepath.ToSlash(envVal))
	bind := bindingFixture("tmp", "build")

	paths, suspicious := extractBashPaths(`D=`+bind+`; cat "$D/a"`, "", "")
	if !suspicious {
		t.Fatalf("expected suspicious for a bound var with a differing env value, paths=%v", paths)
	}
	want := filepath.Clean(osAbsPath("tmp", "build", "a"))
	if !sliceContains(paths, want) {
		t.Fatalf("expected bound literal %q in paths, got %v", want, paths)
	}

	// A decoy binding in a branch that never executes must not clear the env
	// suspicion either — the binding pre-pass is control-flow-unaware.
	paths, suspicious = extractBashPaths(`if false; then D=`+bind+`; fi; cat "$D/a"`, "", "")
	if !suspicious {
		t.Fatalf("expected suspicious for a decoy binding in a dead branch, paths=%v", paths)
	}
	if !sliceContains(paths, want) {
		t.Fatalf("expected bound literal %q in paths, got %v", want, paths)
	}
}

// TestExtractBashPaths_BindingMatchingEnvNotSuspicious covers the equality
// exception of the union: when the env value equals the binding, both possible
// runtime expansions coincide, so the word is not ambiguous and the command
// stays non-suspicious.
func TestExtractBashPaths_BindingMatchingEnvNotSuspicious(t *testing.T) {
	bind := bindingFixture("tmp", "build")
	t.Setenv("D", bind)

	paths, suspicious := extractBashPaths(`D=`+bind+`; cat "$D/a"`, "", "")
	if suspicious {
		t.Fatalf("expected not suspicious when binding equals env value, paths=%v", paths)
	}
	want := filepath.Clean(osAbsPath("tmp", "build", "a"))
	if !sliceContains(paths, want) {
		t.Fatalf("expected %q in paths, got %v", want, paths)
	}
}

// TestExtractBashPaths_DynamicRHSNotBound covers a dynamic RHS: "$(echo ...)"
// cannot be bound, so "$D/a" stays suspended/unexpandable and only the literal
// suffix resolves.
func TestExtractBashPaths_DynamicRHSNotBound(t *testing.T) {
	t.Setenv("D", "")
	suffix := osAbsPath("a")
	paths, suspicious := extractBashPaths(`D=$(echo x); cat "${D}`+bindingFixture("a")+`"`, "", "")
	if !suspicious {
		t.Fatalf("expected suspicious for a dynamic (unbindable) RHS, got suspicious=false, paths=%v", paths)
	}
	want := filepath.Clean(suffix)
	if !sliceContains(paths, want) {
		t.Fatalf("expected literal suffix %q in paths, got %v", want, paths)
	}
}

// TestExtractBashPaths_LastAssignmentWins covers left-to-right semantics: the
// last literal assignment to a name wins.
func TestExtractBashPaths_LastAssignmentWins(t *testing.T) {
	t.Setenv("D", "")
	first := bindingFixture("etc")
	last := bindingFixture("tmp")
	paths, suspicious := extractBashPaths(`D=`+first+`; D=`+last+`; echo "$D"`, "", "")
	if suspicious {
		t.Fatalf("expected not suspicious, got suspicious=true")
	}
	want := filepath.Clean(osAbsPath("tmp"))
	if !sliceContains(paths, want) {
		t.Fatalf("expected %q (last assignment) in paths, got %v", want, paths)
	}
}

// TestExtractBashPaths_CommandSubstRemainsSuspicious covers "cat $(echo ...)",
// a top-level command substitution: stays suspicious and yields no path.
func TestExtractBashPaths_CommandSubstRemainsSuspicious(t *testing.T) {
	paths, suspicious := extractBashPaths(`cat $(echo x)`, "", "")
	if !suspicious {
		t.Fatalf("expected suspicious for command substitution, got suspicious=false")
	}
	if len(paths) != 0 {
		t.Fatalf("expected no paths, got %v", paths)
	}
}

// TestResolveShellPathTokens_BoundVar covers the regex-based resolver: an
// in-command literal binding resolves "$D/passwd" to the bound value in
// addition to (not instead of) any env value.
func TestResolveShellPathTokens_BoundVar(t *testing.T) {
	t.Setenv("D", "")
	got := ResolveShellPathTokens(`D=`+bindingFixture("etc")+`; cat "$D/passwd"`, ShellBash, "")
	want := filepath.Clean(osAbsPath("etc", "passwd"))
	if !sliceContains(got, want) {
		t.Fatalf("expected %q, got %v", want, got)
	}
}

// TestResolveShellPathTokens_BindingUnionsEnv pins the union semantics at the
// resolver layer: a name with BOTH an in-command binding and a non-empty,
// differing env value reports BOTH expansions — the binding pre-pass cannot
// prove the assignment is in effect where the reference executes, and a
// binding must never mask an environment dereference.
func TestResolveShellPathTokens_BindingUnionsEnv(t *testing.T) {
	envVal := bindingFixture("from", "env")
	t.Setenv("D", envVal)
	bind := bindingFixture("tmp", "build")

	got := ResolveShellPathTokens(`D=`+bind+`; cat "$D/a"`, ShellBash, "")
	for _, want := range []string{
		filepath.Clean(osAbsPath("tmp", "build")),
		filepath.Clean(osAbsPath("tmp", "build", "a")),
		filepath.Clean(osAbsPath("from", "env", "a")),
	} {
		if !sliceContains(got, want) {
			t.Errorf("expected %q in union of binding and env expansions, got %v", want, got)
		}
	}

	// Unbound $D (no in-command assignment) falls back to the process env.
	back := ResolveShellPathTokens(`cat "$D/a"`, ShellBash, "")
	wantBack := filepath.Clean(osAbsPath("from", "env", "a"))
	if !sliceContains(back, wantBack) {
		t.Fatalf("env fallback: expected %q, got %v", wantBack, back)
	}
}

// TestResolveShellPathTokens_PrefixAssignmentUnionsEnv covers the command-prefix
// form ("VAR=value cmd ...") against a set env value: in real bash the
// argument words expand BEFORE the prefix assignment takes effect, so the
// runtime may read the env value — the resolver reports both (fail-closed).
func TestResolveShellPathTokens_PrefixAssignmentUnionsEnv(t *testing.T) {
	envVal := bindingFixture("from", "env")
	t.Setenv("D", envVal)
	bind := bindingFixture("tmp", "build")

	got := ResolveShellPathTokens(`D=`+bind+` mkdir "$D/x"`, ShellBash, "")
	for _, want := range []string{
		filepath.Clean(osAbsPath("tmp", "build")),
		filepath.Clean(osAbsPath("tmp", "build", "x")),
		filepath.Clean(osAbsPath("from", "env", "x")),
	} {
		if !sliceContains(got, want) {
			t.Errorf("expected %q in union of prefix binding and env expansions, got %v", want, got)
		}
	}
}

// TestResolveShellPathTokens_LastAssignmentWins verifies the shellpaths.go
// resolver honors the last in-command literal binding.
func TestResolveShellPathTokens_LastAssignmentWins(t *testing.T) {
	t.Setenv("D", "")
	got := ResolveShellPathTokens(
		`D=`+bindingFixture("etc")+`; D=`+bindingFixture("tmp")+`; echo "$D"`, ShellBash, "")
	want := filepath.Clean(osAbsPath("tmp"))
	if !sliceContains(got, want) {
		t.Fatalf("expected %q (last assignment), got %v", want, got)
	}
}

// TestResolveShellPathTokens_UnboundVarStillEnvFallback ensures an in-command
// assignment is NOT applied to variables that are never assigned in-command,
// preserving the historical environment-only behavior.
func TestResolveShellPathTokens_UnboundVarStillEnvFallback(t *testing.T) {
	varName := "C0WRK_TEST_UNBOUND_VAR"
	t.Setenv(varName, bindingFixture("etc"))
	got := ResolveShellPathTokens(`cat "$`+varName+`/passwd"`, ShellBash, "")
	want := filepath.Clean(osAbsPath("etc", "passwd"))
	if !sliceContains(got, want) {
		t.Fatalf("expected %q from env, got %v", want, got)
	}
}
