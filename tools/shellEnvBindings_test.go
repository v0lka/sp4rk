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
// /tmp/build/a, which is still extracted as a path candidate. The env is
// pinned EMPTY, so the binding can never be proven live at the reference
// (the pre-pass is control-flow-unaware — the assignment may sit in a branch
// not taken), and the EMPTY expansion stays possible: the command therefore
// stays flagged suspicious (fail-closed) exactly like an unbound "$UNSET"
// reference. The non-suspicious case is pinned by
// TestExtractBashPaths_BindingMatchingEnvNotSuspicious (env set and equal).
func TestExtractBashPaths_BoundVarResolvesLiteral(t *testing.T) {
	t.Setenv("D", "")
	bind := bindingFixture("tmp", "build")
	paths, suspicious := extractBashPaths(`D=`+bind+`; cat "$D/a"`, "", "")
	if !suspicious {
		t.Fatalf("expected suspicious for a bound literal var with an empty env value (empty expansion possible), got suspicious=false")
	}
	want := filepath.Clean(osAbsPath("tmp", "build", "a"))
	if !sliceContains(paths, want) {
		t.Fatalf("expected %q in paths, got %v", want, paths)
	}
}

// TestExtractBashPaths_BoundVarOutOfRootReported covers a bound var whose value
// points outside the roots: "$D/passwd" resolves to the bound value, which is
// both extracted as a path and reported by PathsOutsideRoots. The empty env
// value keeps the word unexpandable/suspicious (see
// TestExtractBashPaths_BoundVarResolvesLiteral).
func TestExtractBashPaths_BoundVarOutOfRootReported(t *testing.T) {
	t.Setenv("D", "")
	target := osAbsPath("etc", "passwd")
	paths, suspicious := extractBashPaths(`D=`+bindingFixture("etc", "passwd")+`; cat "$D/passwd"`, "", "")
	if !suspicious {
		t.Fatalf("expected suspicious for a bound literal var with an empty env value, got suspicious=false")
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
// and the command-prefix "VAR=value cmd ..." forms: the bound literal stays a
// path candidate, while an empty env value keeps the reference ambiguous
// (suspicious) — the same fail-closed treatment as a plain binding.
func TestExtractBashPaths_ExportAndPrefixBindings(t *testing.T) {
	t.Setenv("D", "")
	bind := bindingFixture("tmp", "build")

	paths, suspicious := extractBashPaths(`export D=`+bind+`; cat "$D/a"`, "", "")
	want := filepath.Clean(osAbsPath("tmp", "build", "a"))
	if !suspicious || !sliceContains(paths, want) {
		t.Fatalf("export binding: paths=%v suspicious=%v", paths, suspicious)
	}

	// The prefix form's binding is not visible to the argument expansion of
	// the same command in real bash (arguments expand first), so the word is
	// unexpandable regardless; the bound literal is still extracted as the
	// statically known candidate. With a differing env value the union adds
	// that too: see TestExtractBashPaths_PrefixAssignmentWithEnvUnions below.
	paths, suspicious = extractBashPaths(`D=`+bind+` mkdir "$D/x"`, "", "")
	want = filepath.Clean(osAbsPath("tmp", "build", "x"))
	if !suspicious || !sliceContains(paths, want) {
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

// TestExtractBashPaths_ModifierParamExpStaysSuspicious pins the fail-closed
// treatment of modifier parameter expansions: "${D:+…}", "${#D}", "${D:0:1}",
// "${D/x/y}", and "${!D}" all derive a value OTHER than D's own value, so
// substituting D's binding would resolve the word to a confidently wrong,
// possibly benign-looking path (e.g. "D=/ws/safe; cat ${D:+/etc/passwd}"
// really reads /etc/passwd while the substituted literal /ws/safe looks
// contained). Every modifier form must therefore keep the word unexpandable
// (command suspicious) regardless of any in-command binding, mirroring the
// historical wordLiteral treatment that flagged all parameter expansions.
func TestExtractBashPaths_ModifierParamExpStaysSuspicious(t *testing.T) {
	t.Setenv("D", "")
	bind := bindingFixture("tmp", "build")
	evil := bindingFixture("etc", "passwd")
	// osAbsPath is host-absolute; embed with forward slashes inside ${…}.
	evilSlash := filepath.ToSlash(evil)

	for _, cmd := range []string{
		`D=` + bind + `; cat ${D:+` + evilSlash + `}`,
		`D=` + bind + `; cat ${D-` + evilSlash + `}`,
		`D=` + bind + `; cat ${D:0:1}`,
		`D=` + bind + `; cat ${#D}`,
		`D=` + bind + `; cat ${!D}`,
		`D=` + bind + `; cat ${D/tmp/etc}`,
	} {
		paths, suspicious := extractBashPaths(cmd, "", "")
		if !suspicious {
			t.Errorf("expected suspicious for modifier parameter expansion: %q (paths=%v)", cmd, paths)
		}
	}
}

// TestExtractBashPaths_AnsiCQuotingStaysSuspicious pins the fail-closed
// treatment of ANSI-C $'…' quoting: the parser keeps the RAW, undecoded value
// (bash decodes \xHH/\nnn/\e… at runtime), so "D=$'/ws/\x2e\x2e/etc/passwd';
// cat \"$D\"" expands at runtime to an out-of-root ../etc/passwd path while
// every static candidate built from the raw text stays in-root — the word and
// the binding must therefore stay unexpandable/suspicious. The locale-translated
// $"…" form gets the same treatment. Plain '…' single quotes stay literal.
func TestExtractBashPaths_AnsiCQuotingStaysSuspicious(t *testing.T) {
	t.Setenv("D", "")
	for _, cmd := range []string{
		`D=$'/ws/\x2e\x2e/etc/passwd'; cat "$D"`,
		`cat $'/etc/passwd'`,
		`echo $"x"`,
	} {
		if _, suspicious := extractBashPaths(cmd, "", ""); !suspicious {
			t.Errorf("expected suspicious for dollar-quoted word: %q", cmd)
		}
	}
	// Plain single quotes remain literal and non-suspicious.
	if _, suspicious := extractBashPaths(`cat '/etc/passwd'`, "", ""); suspicious {
		t.Error("expected plain single-quoted literal to stay non-suspicious")
	}
}

// TestExtractBashPaths_PlainBracedBindingStillResolves guards the other
// direction: the plain "${D}" form (no modifier) keeps resolving to the
// in-command binding as a path candidate. With an empty env value the empty
// expansion stays possible, so the word remains unexpandable (suspicious) —
// same as the unbraced form.
func TestExtractBashPaths_PlainBracedBindingStillResolves(t *testing.T) {
	t.Setenv("D", "")
	bind := bindingFixture("tmp", "build")
	paths, suspicious := extractBashPaths(`D=`+bind+`; cat "${D}/a"`, "", "")
	if !suspicious {
		t.Fatalf("expected suspicious for plain braced bound var with an empty env value, paths=%v", paths)
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

// TestExtractBashPaths_RebindingStaysSuspicious pins the fail-closed treatment
// of re-bound names: "D=etc; D=tmp; echo \"$D\"" has TWO assignments to D, and
// the pre-pass is position- and control-flow-unaware — it cannot prove which
// value the reference sees (a decoy re-binding after the reference, or branch
// divergence, must not mask an earlier out-of-root binding). The final literal
// is still contributed as the path candidate (resolve-and-stay-suspicious,
// mirroring the env union), and the command stays flagged suspicious.
func TestExtractBashPaths_RebindingStaysSuspicious(t *testing.T) {
	t.Setenv("D", "")
	first := bindingFixture("etc")
	last := bindingFixture("tmp")
	paths, suspicious := extractBashPaths(`D=`+first+`; D=`+last+`; echo "$D"`, "", "")
	if !suspicious {
		t.Fatalf("expected suspicious for a re-bound var (assignment ambiguity), got suspicious=false")
	}
	want := filepath.Clean(osAbsPath("tmp"))
	if !sliceContains(paths, want) {
		t.Fatalf("expected final literal %q in paths, got %v", want, paths)
	}
}

// TestExtractBashPaths_ReferenceBetweenAssignmentsStaysSuspicious covers the
// masking exploit the rebinding rule closes: a decoy re-binding AFTER the
// reference ("D=etc; cat \"/$D/passwd\"; D=safe") must not make the reference
// resolve to the decoy — runtime reads the FIRST value, statically the name is
// ambiguous, so the command stays suspicious while the final literal is still
// extracted as a candidate. The out-of-root /etc/passwd reference itself is
// reported by the regex layer's multi-value union (see
// TestResolveShellPathTokens_RebindingReportsAllValues).
func TestExtractBashPaths_ReferenceBetweenAssignmentsStaysSuspicious(t *testing.T) {
	t.Setenv("D", "")
	first := bindingFixture("etc", "passwd")
	decoy := bindingFixture("ws", "safe")
	paths, suspicious := extractBashPaths(`D=`+first+`; cat "/$D/passwd"; D=`+decoy, "", "")
	if !suspicious {
		t.Fatalf("expected suspicious for a reference between two assignments, paths=%v", paths)
	}
	want := filepath.Clean(osAbsPath("ws", "safe"))
	if !sliceContains(paths, want) {
		t.Fatalf("expected final literal %q in paths, got %v", want, paths)
	}
}

// TestExtractBashPaths_BranchDivergentBindingsStaySuspicious covers
// branch-blindness: then/else branches assigning different literals leave the
// name ambiguous at any later reference — the command stays suspicious rather
// than resolving to whichever branch the walk happened to visit last.
func TestExtractBashPaths_BranchDivergentBindingsStaySuspicious(t *testing.T) {
	t.Setenv("D", "")
	then := bindingFixture("etc", "passwd")
	els := bindingFixture("ws", "safe")
	cmd := `if true; then D=` + then + `; else D=` + els + `; fi; cat "$D/x"`
	paths, suspicious := extractBashPaths(cmd, "", "")
	if !suspicious {
		t.Fatalf("expected suspicious for branch-divergent bindings, paths=%v", paths)
	}
	// Both branch values remain visible as literal RHS words.
	for _, want := range []string{
		filepath.Clean(osAbsPath("etc", "passwd")),
		filepath.Clean(osAbsPath("ws", "safe")),
	} {
		if !sliceContains(paths, want) {
			t.Fatalf("expected branch literal %q in paths, got %v", want, paths)
		}
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

// TestPathsOutsideRoots_CdLiteralTargetSurfacesPWDCandidate pins the
// cd-rebinding containment: "cd /" rebinds PWD to "/", so "$PWD/etc/passwd"
// must surface /etc/passwd through the PWD value-candidate union (soft
// escalation), while "cd sub" keeps the joined candidate relative and
// in-root (quiet).
func TestPathsOutsideRoots_CdLiteralTargetSurfacesPWDCandidate(t *testing.T) {
	ws := t.TempDir()
	ctx := WithWorkspacePath(context.Background(), ws)
	if out := PathsOutsideRoots(ctx, `cd /; cat "$PWD/etc/passwd"`, ShellBash, ws); !sliceContains(out, "/etc/passwd") {
		t.Fatalf("expected /etc/passwd reported outside roots, got %v", out)
	}
	// Pin the process PWD into the roots so only the cd target can surface:
	// a relative target joins to a relative (in-root) candidate.
	t.Setenv("PWD", ws)
	if out := PathsOutsideRoots(ctx, `cd sub && cat "$PWD/x"`, ShellBash, ws); len(out) != 0 {
		t.Fatalf("expected quiet for a relative cd target, got %v", out)
	}
}

// TestPathsOutsideRoots_AbsoluteShapedWordAssembly pins the word-level
// candidate reconstruction: an env reference inside a word that starts with
// "/" ("/et$B/passwd", "/$P$Q/passwd") assembles an absolute path at runtime
// from literal fragments and every referenced variable's candidates — the
// joined product must surface the out-of-root path the plain per-token
// resolution cannot see.
func TestPathsOutsideRoots_AbsoluteShapedWordAssembly(t *testing.T) {
	ws := t.TempDir()
	ctx := WithWorkspacePath(context.Background(), ws)
	t.Setenv("B", "c")
	if out := PathsOutsideRoots(ctx, `B=c; cat /et$B/passwd`, ShellBash, ws); !sliceContains(out, "/etc/passwd") {
		t.Fatalf("expected /etc/passwd from the assembled word, got %v", out)
	}
	t.Setenv("P", "e")
	t.Setenv("Q", "tc")
	if out := PathsOutsideRoots(ctx, `P=e; Q=tc; cat /$P$Q/passwd`, ShellBash, ws); !sliceContains(out, "/etc/passwd") {
		t.Fatalf("expected /etc/passwd from the multi-token product, got %v", out)
	}
}

// TestPathsOutsideRoots_VariableLedAbsoluteAssembly pins the assembly
// defense when the leading slash lives INSIDE a variable: "$P$Q/passwd"
// with P="/e", Q="tc" composes /etc/passwd at runtime — the whole-word
// candidate product must surface it, in every quoting/concatenation
// spelling.
func TestPathsOutsideRoots_VariableLedAbsoluteAssembly(t *testing.T) {
	ws := t.TempDir()
	ctx := WithWorkspacePath(context.Background(), ws)
	for _, c := range []string{
		`P=/e; Q=tc; cat "$P$Q/passwd"`,
		`D=/et; cat "$D"c/passwd`,
		`D=/e; cat "$D"'tc/passwd'`,
		`D=/e; E=tc; cat "${D}${E}/passwd"`,
		`D=/e; cat "$D"tc/passwd`,
	} {
		if out := PathsOutsideRoots(ctx, c, ShellBash, ws); !sliceContains(out, "/etc/passwd") {
			t.Errorf("expected /etc/passwd from the assembled word %q, got %v", c, out)
		}
	}
	// A positional composed with more word content cannot be assessed.
	if tok := UnresolvablePathTokens(`bash -c 'cat "$1$2"' _ /e tc/passwd`, ShellBash); len(tok) == 0 {
		t.Error("expected composed positional reference to escalate")
	}
}

// TestPathsOutsideRoots_SelfReferenceComposition pins the union for pure
// self-reference assignments: "D=/et; D=$D\"c\"" composes "/etc" from the
// prior candidate and the literal fragment — the composed runtime value
// must stay visible as a candidate ("$D/passwd" surfaces /etc/passwd).
func TestPathsOutsideRoots_SelfReferenceComposition(t *testing.T) {
	ws := t.TempDir()
	ctx := WithWorkspacePath(context.Background(), ws)
	t.Setenv("D", "")
	out := PathsOutsideRoots(ctx, `D=/et; D=$D"c"; cat "$D/passwd"`, ShellBash, ws)
	if !sliceContains(out, "/etc/passwd") {
		t.Fatalf("expected /etc/passwd from the composed candidate, got %v", out)
	}
}

// ── Fail-closed regressions: empty-expansion masking and invisible
// rebinding constructs ──────────────────────────────────────────────────────

// TestResolveShellPathTokens_DeadBranchDecoyUnionsEmptyExpansion is the
// regression test for the empty-expansion masking hole: a decoy in-command
// binding inside a branch that never executes ("if false; then D=safe; fi")
// must not mask the EMPTY expansion of the possibly-unset variable —
// "$D/etc/passwd" with D unset at runtime reads /etc/passwd, so the
// suffix-alone candidate must be reported exactly like the unbound-variable
// fallback, at BOTH the resolver and the PathsOutsideRoots escalation layer.
func TestResolveShellPathTokens_DeadBranchDecoyUnionsEmptyExpansion(t *testing.T) {
	t.Setenv("D", "")
	safe := bindingFixture("ws", "safe")
	cmd := `if false; then D=` + safe + `; fi; cat "$D/etc/passwd"`

	got := ResolveShellPathTokens(cmd, ShellBash, "")
	if !sliceContains(got, "/etc/passwd") {
		t.Fatalf("expected the suffix-alone /etc/passwd candidate (empty expansion), got %v", got)
	}

	ws := t.TempDir()
	ctx := WithWorkspacePath(context.Background(), ws)
	if out := PathsOutsideRoots(ctx, cmd, ShellBash, ws); !sliceContains(out, "/etc/passwd") {
		t.Fatalf("expected /etc/passwd reported outside roots, got %v", out)
	}

	// The parser layer keeps the reference ambiguous (suspicious) for the
	// same reason: the binding cannot be proven live at the reference.
	if _, suspicious := extractBashPaths(cmd, "", ""); !suspicious {
		t.Fatal("expected suspicious for a dead-branch decoy binding")
	}
}

// TestUnresolvablePathTokens_RebindingConstructsFailClosed is the regression
// test for the invisible-rebinding hole: rebinding constructs that never
// produce *syntax.Assign nodes ("read", "source"/".", "eval", "printf -v",
// "mapfile", "unset", "getopts", for/select loops, arithmetic assignment,
// attribute-changing declares) must make references to the affected names
// unassessable — the runtime value is unknown (often attacker-controlled
// file content), so the Judges escalate the token HARD instead of letting a
// decoy literal binding auto-approve the command.
func TestUnresolvablePathTokens_RebindingConstructsFailClosed(t *testing.T) {
	cases := []struct {
		name    string
		command string
		want    []string
	}{
		{"read target", `D=safe; read -r D < cfg; cat "$D/x"`, []string{"$D"}},
		{"read default target REPLY", `read < cfg; cat "$REPLY/x"`, []string{"$REPLY"}},
		{"read with option arguments", `read -u 3 -p "go:" DD; cat "$DD/x"`, []string{"$DD"}},
		{"source dot form", `. ./cfg; cat "$D/x"`, []string{"$D"}},
		{"source word form", `source ./cfg; cat "$D/x"`, []string{"$D"}},
		{"source behind command prefix", `command source ./cfg; cat "$D/x"`, []string{"$D"}},
		{"eval", `eval "D=/etc"; cat "$D/x"`, []string{"$D"}},
		{"let", `let "D=5"; cat "$D/x"`, []string{"$D"}},
		{"unset clears", `D=safe; unset D; cat "$D/x"`, []string{"$D"}},
		{"printf -v assigns", `D=safe; printf -v D '%s' /etc; cat "$D/x"`, []string{"$D"}},
		{"mapfile assigns", `D=safe; mapfile -t D < cfg; cat "$D/x"`, []string{"$D"}},
		{"readarray synonym", `readarray -t D < cfg; cat "$D/x"`, []string{"$D"}},
		{"getopts assigns", `getopts "d:" OPT; cat "$OPT/x"`, []string{"$OPT"}},
		{"for-loop rebinds", `for D in $(cat cfg); do cat "$D"; done`, []string{"$D"}},
		{"select-loop rebinds", `select D in a b; do echo "$D"; done`, []string{"$D"}},
		{"c-style for-loop arithmetic", `for ((i=0; i<3; i++)); do echo "$i"; done`, []string{"$i"}},
		{"arithmetic command assigns", `D=safe; ((D=5)); echo "$D"`, []string{"$D"}},
		{"arithmetic increment assigns", `i=0; ((i++)); echo "$i"`, []string{"$i"}},
		{"declare nameref redirects references", `declare -n R=D; cat "$R/x"`, []string{"$R"}},
		{"declare array attribute clears", `declare -a A; cat "$A/x"`, []string{"$A"}},
		{"local integer attribute converts", `local -i N; echo "$N"`, []string{"$N"}},
		{"declare export attribute keeps literal binding assessable", `declare -x D=/ws/safe; cat "$D/a"`, nil},
		{"declare readonly attribute keeps literal binding assessable", `readonly D=/ws/safe; cat "$D/a"`, nil},
		{"opaque source makes every reference unassessable", `. ./cfg; cat "$OTHER/x"`, []string{"$OTHER"}},
		{"escaped source is still opaque", `\source cfg; cat "$D/x"`, []string{"$D"}},
		{"escaped dot is still opaque", `\. cfg; cat "$D/x"`, []string{"$D"}},
		{"source behind command wrapper with option", `command -p source cfg; cat "$D/x"`, []string{"$D"}},
		{"source behind builtin wrapper with --", `builtin -- source cfg; cat "$D/x"`, []string{"$D"}},
		{"eval behind command wrapper with option", `command -p eval "D=/etc"; cat "$D/x"`, []string{"$D"}},
		{"dynamically-built command word is opaque", `E=eval; $E "$(cat cfg)"; cat "$D/x"`, []string{"$E", "$D"}},
		{"dynamically-built external command word is opaque", `"$TOOL" --flag; cat "$D/x"`, []string{"$TOOL", "$D"}},
		{"braced indexed reference to rebound array", `mapfile -t A < cfg; cat ${A[0]}`, []string{"${A[0]}"}},
		{"braced slice reference to rebound name", `D=$(cat cfg); cat ${D:0}`, []string{"${D:0}"}},
		{"indirect reference form is unassessable", `D=$(cat cfg); X=D; cat ${!X}`, []string{"${!X}"}},
		{"braced quote operator on rebound name", `read D < cfg; cat ${D@Q}`, []string{"${D@Q}"}},
		{"braced case operator on rebound name", `D=$(cat cfg); cat ${D^}`, []string{"${D^}"}},
		{"nested bash -c literal script is parsed", `bash -c 'read D < cfg; cat "$D/x"'`, []string{"$D"}},
		{"nested sh -c expanded script is opaque", `sh -c "$(cat cfg)"; echo "$D"`, []string{"$D"}},
		{"nested bash -c benign literal stays assessable", `bash -c 'echo hi'; echo "$D"`, nil},
		{"bash script file without -c is a child process", `bash cfg.sh; cat "$D/a"`, nil},
		{"wrapper chain command command source", `command command source cfg; cat "$D/x"`, []string{"$D"}},
		{"wrapper chain builtin builtin source", `builtin builtin source cfg; cat "$D/x"`, []string{"$D"}},
		{"wrapper chain mixed with options", `command -p command source cfg; cat "$D/x"`, []string{"$D"}},
		{"wrapper chain to eval", `command builtin eval "D=/etc"; cat "$D/x"`, []string{"$D"}},
		{"builtin declare nameref behind wrapper", `builtin declare -n R=D; cat "$R/x"`, []string{"$R"}},
		{"command export behind wrapper is dynamic", `command export D="$(cat cfg)"; cat "$D/x"`, []string{"$D"}},
		{"escaped export is dynamic", `\export D="$(cat cfg)"; cat "$D/x"`, []string{"$D"}},
		{"escaped declare array is dynamic", `\declare -a A; cat "$A/x"`, []string{"$A"}},
		{"alias definition is opaque", `shopt -s expand_aliases; alias r='read -r D < val'; r; cat "$D/x"`, []string{"$D"}},
		{"bare alias listing is quiet", `alias; cat "$D/a"`, nil},
		{"shopt expand_aliases alone is opaque", `shopt -s expand_aliases; cat "$D/x"`, []string{"$D"}},
		{"shopt unrelated option is quiet", `shopt -s nullglob; cat "$D/a"`, nil},
		{"BASH_ENV assignment makes bash -c opaque", `BASH_ENV=cfg bash -c 'cat "$D/hosts"'`, []string{"$D"}},
		{"bash --rcfile makes command opaque", `bash --rcfile cfg -i -c 'cat "$D/hosts"'`, []string{"$D"}},
		{"env argument assignments are dynamic", `env D="$(cat val)" bash -c 'cat "$D/hosts"'`, []string{"$D"}},
		{"env -u marks the removed name (child-side unset)", `env -u D bash cfg.sh; cat "$D/a"`, []string{"$D"}},
		{"set with arguments rebinds positionals", `set -- "$(cat val)"; cat "$1/hosts"`, []string{"$1/hosts"}},
		{"shift rebinds positionals", `shift; cat "$1/hosts"`, []string{"$1/hosts"}},
		{"positional with absolute suffix escalates without set", `cat "$1/etc/passwd"`, []string{"$1/etc/passwd"}},
		{"bare positional without rebinding stays quiet", `awk '{print $1}' file.txt`, nil},
		{"set options only stay quiet", `set -euo pipefail; cat "$D/a"`, nil},
		{"exec-wrapped interpreter script is parsed", `exec bash -c 'read D < cfg; cat "$D/x"'`, []string{"$D"}},
		{"timeout-wrapped interpreter script is parsed", `timeout 30 bash -c 'read D < cfg; cat "$D/x"'`, []string{"$D"}},
		{"nohup-wrapped interpreter script is parsed", `nohup bash -c 'read D < cfg; cat "$D/x"'`, []string{"$D"}},
		{"env-wrapped interpreter script is parsed", `env X=1 bash -c 'read D < cfg; cat "$D/x"'`, []string{"$D"}},
		{"exec-wrapped source is opaque", `exec source cfg; cat "$D/x"`, []string{"$D"}},
		{"dynamic trap handler is opaque", `trap "D=$(cat cfg)" EXIT; cat "$D/x"`, []string{"$D"}},
		{"plain trap without handler stays quiet", `trap - EXIT; echo "$D"`, nil},
		{"benign trap handler stays quiet", `trap 'rm -f /tmp/f' EXIT; echo "$D"`, nil},
		{"bundled -lc script is parsed", `bash -lc 'read D < cfg; cat "$D/passwd"'`, []string{"$D"}},
		{"bundled -ec script is parsed", `bash -ec 'read D < cfg; cat "$D/passwd"'`, []string{"$D"}},
		{"bundled -uc script is parsed", `sh -uc 'read D < cfg; cat "$D/passwd"'`, []string{"$D"}},
		{"wrapped bundled flags are parsed", `timeout 5 bash -lc 'read D < cfg; cat "$D/passwd"'`, []string{"$D"}},
		{"dynamic read target is opaque", `N=D; read -r "$N" < cfg; cat "$D/passwd"`, []string{"$N", "$D"}},
		{"dynamic printf -v target is opaque", `N=D; printf -v "$N" '%s' "$(cat cfg)"; cat "$D/passwd"`, []string{"$N", "$D"}},
		{"dynamic mapfile target is opaque", `N=D; mapfile -t "$N" < cfg; cat "$D/passwd"`, []string{"$N", "$D"}},
		{"nameref target is marked", `declare -n R=D; read -r R < cfg; cat "$D/passwd"`, []string{"$D"}},
		{"wrapped nameref target is marked", `builtin declare -n R=D; read -r R < cfg; cat "$D/passwd"`, []string{"$D"}},
		{"local nameref in function marks target", `f() { local -n R=D; read -r R < cfg; cat "$D/passwd"; }; f`, []string{"$D"}},
		{"indirect literal-bound base is unassessable", `D=HOME; cat ${!D}/.zshrc`, []string{"${!D}"}},
		{"indirect chained base is unassessable", `X=HOME; D=X; cat ${!D}/.zshrc`, []string{"${!D}"}},
		{"heredoc-fed interpreter script is parsed", "bash <<'EOF'\nread D < cfg\ncat \"$D/passwd\"\nEOF", []string{"$D"}},
		{"unquoted heredoc script is opaque", "bash <<EOF\nD=$(cat cfg)\ncat \"$D/passwd\"\nEOF", []string{"$D"}},
		{"here-string script is parsed", `bash <<< 'read D < cfg; cat "$D/passwd"'`, []string{"$D"}},
		{"pipe-fed interpreter is opaque", `printf 'read D < cfg; cat "$D/passwd"' | bash`, []string{"$D"}},
		{"env -S embedded script is parsed", `env -S 'read D < cfg; cat "$D/passwd"'`, []string{"$D"}},
		{"qualified interpreter path is parsed", `./bash -c 'read D < cfg; cat "$D/passwd"'`, []string{"$D"}},
		{"absolute interpreter path is parsed", `/bin/bash -c 'read D < cfg; cat "$D/passwd"'`, []string{"$D"}},
		{"PATH self-append stays assessable", `PATH=$PATH:/usr/local/bin go vet ./...`, nil},
		{"CFLAGS self-append stays assessable", `CFLAGS="$CFLAGS -O2" make`, nil},
		{"wrapped let is opaque", `command let "D=5"; cat "$D/passwd"`, []string{"$D"}},
		{"word-level arithmetic assignment is marked", `echo $((D=5)); cat "$D/passwd"`, []string{"$D"}},
		{"inner-escaped read is detected", `r\ead D < cfg; cat "$D"`, []string{"$D"}},
		{"inner-escaped source is detected", `D=safe; s\ource cfg2; cat "$D"`, []string{"$D"}},
		{"inner-escaped eval is detected", `e\v\al 'read D < cfg'; cat "$D"`, []string{"$D"}},
		{"inner-escaped interpreter is detected", `b\ash -c 'read D < cfg; cat "$D"'`, []string{"$D"}},
		{"wrapped inner-escaped source is detected", `command sou\rce cfg2; cat "$D"`, []string{"$D"}},
		{"env-set BASH_ENV is opaque", `env BASH_ENV=cfg3 bash -c 'cat "$D"'`, []string{"$D"}},
		{"env-set ZDOTDIR is opaque", `env ZDOTDIR=zd zsh -c 'cat "$D"'`, []string{"$D"}},
		{"read-marked BASH_ENV is opaque", `read BASH_ENV < b; export BASH_ENV; bash -c 'cat "$D"'`, []string{"$D"}},
		{"getopts implicit OPTARG is marked", `set -- "-d/etc"; getopts "d:" o; cat "$OPTARG/hosts"`, []string{"$OPTARG"}},
		{"array-element read target is marked", `read 'A[0]' < cfg; cat "${A[0]}"`, []string{"${A[0]}"}},
		{"array-element printf target is marked", `printf -v 'A[0]' '%s' /etc; cat "${A[0]}"`, []string{"${A[0]}"}},
		{"set -o with operand stays quiet", `set -o pipefail; awk '{print $1}' log.txt`, nil},
		{"set -o errexit stays quiet", `set -o errexit; go build ./...`, nil},
		{"exec -a operand skipped to real command", `exec -a x source cfg2; cat "$D"`, []string{"$D"}},
		{"mksh nested script is parsed", `mksh -c 'read D < cfg; cat "$D"'`, []string{"$D"}},
		{"ash nested script is parsed", `ash -c 'read D < cfg; cat "$D"'`, []string{"$D"}},
		{"zsh dialect is opaque not parsed", `zsh -c 'read D < c; cat $=D'`, []string{"$=D"}},
		{"zsh braced flag form escalates", `zsh -c 'read D < c; cat ${(z)D}'`, []string{"${(z)D}"}},
		{"ksh dialect is opaque", `ksh -c 'read D < c; cat "$D"'`, []string{"$D"}},
		{"cd literal target surfaces PWD candidate", `cd /; cat "$PWD/etc/passwd"`, nil},
		{"cd dynamic target makes PWD unassessable", `cd $X; cat "$PWD/etc/passwd"`, []string{"$PWD"}},
		{"cd with no target makes PWD unassessable", `cd; cat "$PWD/etc/passwd"`, []string{"$PWD"}},
		{"cd dash target makes PWD unassessable", `cd -; cat "$PWD/etc/passwd"`, []string{"$PWD"}},
		{"pushd makes DIRSTACK unassessable", `pushd /; cat "$DIRSTACK/x"`, []string{"$DIRSTACK"}},
		{"popd makes PWD unassessable", `popd; cat "$PWD/etc/passwd"`, []string{"$PWD"}},
		{"cd sub stays quiet for relative joins", `cd sub && cat "$PWD/x"`, nil},
		{"env -u marks the removed name", `env -u D bash -c 'cat "$D/etc/passwd"'`, []string{"$D"}},
		{"env --unset= marks the removed name", `env --unset=D bash -c 'cat "$D/etc/passwd"'`, []string{"$D"}},
		{"env --unset marks the removed name", `env --unset D bash -c 'cat "$D/etc/passwd"'`, []string{"$D"}},
		{"env -u with dynamic name fails closed", `env -u "$N" bash -c 'cat "$D/etc/passwd"'`, []string{"$N", "$D"}},
		{"env --unset with dynamic name fails closed", `env --unset "$N" true; cat "$D/etc/passwd"`, []string{"$N", "$D"}},
		{"env -u without operand fails closed", `env -u; cat "$D/etc/passwd"`, []string{"$D"}},
		{"env -u with non-name operand fails closed", `env -u 'A[0]' true; cat "$D"`, []string{"$D"}},
		{"env --unset= with malformed operand fails closed", `env --unset= true; cat "$D"`, []string{"$D"}},
		{"env -u with attached operand marks the name", `env -uD bash -c 'cat "$D/etc/passwd"'`, []string{"$D"}},
		{"env flag bundle hiding -u fails closed", `env -iu cmd; cat "$D/etc/passwd"`, []string{"$D"}},
		{"env -i clears the child environment", `env -i bash -c 'cat "$D/etc/passwd"'`, []string{"$D"}},
		{"env --ignore-environment clears the child environment", `env --ignore-environment bash -c 'cat "$D/etc/passwd"'`, []string{"$D"}},
		{"env -i without a reference escalates via the opaque sentinel", `env -i cat cfg`, []string{opaqueConstructToken}},
		{"exec -c clears the environment", `exec -c bash -c 'cat "$D/etc/passwd"'`, []string{"$D"}},
		{"exec -l login-shell env is opaque", `exec -l bash -c 'cat "$D/etc/passwd"'`, []string{"$D"}},
		{"exec -c behind a wrapper is opaque", `command exec -c bash -c 'cat "$D/etc/passwd"'`, []string{"$D"}},
		{"sudo env_reset makes every expansion unassessable", `sudo cat "$D/etc/passwd"`, []string{"$D"}},
		{"sudo without a reference escalates via the opaque sentinel", `sudo cat cfg`, []string{opaqueConstructToken}},
		{"set +x option-disable does not rebind positionals", `set +x; awk '{print $1}' log.txt`, nil},
		{"set +o with operand stays quiet", `set +o nounset; awk '{print $1}' log.txt`, nil},
		{"set +x with an operand still rebinds positionals", `set +x y; awk '{print $1}' log.txt`, []string{"$1}"}},
		{"dynamic declare flag converts attributes", `F=n; declare -$F R=D; read -r R < cfg; cat "$D"`, []string{"$D"}},
		{"dynamic local flag converts attributes", `F=n; f() { local -$F R=D; read -r R < cfg; cat "$D"; }; f`, []string{"$D"}},
		{"command-substituted declare flag converts", `declare -$(echo n) R=D; read -r R < cfg; cat "$D"`, []string{"$D"}},
		{"su login env reset is opaque", `su -l root -c 'cat "$D/etc/passwd"'`, []string{"$D"}},
		{"doas env reset is opaque", `doas cat "$D/etc/passwd"`, []string{"$D"}},
		{"ssh remote env is opaque", `ssh host 'cat "$D/etc/passwd"'`, []string{"$D"}},
		{"apostrophe inside double quotes keeps awk quiet", `echo "it's fine" && awk '{print $1}' log.txt`, nil},
		{"double-quoted apostrophe does not mask a positional suffix", `echo "don't"; cat "$1/etc/passwd"`, []string{"$1/etc/passwd"}},
		{"source without a reference escalates via the opaque sentinel", `. ./cfg`, []string{opaqueConstructToken}},
		{"eval of command substitution without a reference escalates", `eval "$(cat cfg)"`, []string{opaqueConstructToken}},
		{"pipe-fed interpreter without a reference escalates", `cat cfg | bash`, []string{opaqueConstructToken}},
		{"non-POSIX dialect without a reference escalates", `zsh -c 'source cfg'`, []string{opaqueConstructToken}},
		{"command -v query stays quiet", `command -v source`, nil},
		{"braced operator with path suffix escalates", `cat ${HOME:-safe}/.ssh/id_rsa`, []string{"${HOME:-safe}"}},
		{"braced slice in absolute-shaped word escalates", `X=etc/passwd; cat "/e${X:1}"`, []string{"${X:1}"}},
		{"self-reference path fragment makes the name dynamic", `D=$D/e; D="$D"tc; cat "$D/passwd"`, []string{"$D"}},
		{"over-cap candidate product escalates the word", `Q1=z; Q1=e; Q2=z; Q2=t; Q3=z; Q3=c; Q4=z; Q4=/passwd; cat "/$Q1$Q2$Q3$Q4"`, []string{`"/$Q1$Q2$Q3$Q4"`}},
		{"input-redirect-fed interpreter is opaque", `bash < cfg`, []string{opaqueConstructToken}},
		{"stdin-flag-fed interpreter is opaque", `bash -s < cfg`, []string{opaqueConstructToken}},
		{"procsub-fed interpreter is opaque", `bash <(cat cfg)`, []string{opaqueConstructToken}},
		{"bash script file argument stays quiet", `bash --verbose cfg.sh`, nil},
		{"wrapped pipe-fed interpreter is opaque", `printf 'read D < cfg; cat "$D"' | timeout 10 bash`, []string{"$D"}},
		{"wrapped redirect-fed interpreter is opaque", `timeout 10 bash < cfg`, []string{opaqueConstructToken}},
		{"wrapped stdin-flag interpreter is opaque", `timeout 10 bash -s < cfg`, []string{opaqueConstructToken}},
		{"wrapped procsub-fed interpreter is opaque", `timeout 5 bash <(cat cfg)`, []string{opaqueConstructToken}},
		{"wrapped heredoc-fed interpreter parses the body", "timeout 10 bash <<'EOF'\nread D < cfg\ncat \"$D\"\nEOF", []string{"$D"}},
		{"BASH_ENV with script argument is opaque", `BASH_ENV=cfg bash script.sh`, []string{opaqueConstructToken}},
		{"env --file channel is opaque", `env --file cfg bash -c 'echo ok'`, []string{opaqueConstructToken}},
		{"plain literal binding stays assessable", `D=/ws/safe; cat "$D/a"`, nil},
		{"env-only reference stays assessable", `cat "$PATH"`, nil},
		{"rebinding without a reference is quiet", `D=safe; read D < cfg`, nil},
		{"indirect form without any rebinding is unassessable", `echo ${!X}`, []string{"${!X}"}},
		{"indirect name listing stays quiet", `echo ${!PREFIX@}`, nil},
		{"arithmetic condition without assignment is quiet", `if (( COUNT > 0 )); then echo "$COUNT"; fi`, nil},
		{"arithmetic reference to a rebound name is unassessable", `read D < cfg; cat "$((D))"`, []string{"$((D))"}},
		{"legacy arith reference to a rebound name is unassessable", `read D < cfg; cat "$[D]"`, []string{"$[D]"}},
		{"composed arithmetic reference to a rebound name", `read D < cfg; cat "$((D + 1))x"`, []string{"$((D + 1))"}},
		{"arithmetic on a literal-bound name stays assessable", `D=ws; cat "$((D))/a"`, nil},
		{"arithmetic on an env-only name stays assessable", `echo "$((COUNT)) items"`, nil},
		{"single-quoted arithmetic is literal text", `read D < cfg; echo '$((D))'`, nil},
		{"plain param-default still handled by operator pass", `cat ${VAR:-/etc/passwd}`, []string{"${VAR:-/etc/passwd}"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := UnresolvablePathTokens(tc.command, ShellBash)
			if len(got) != len(tc.want) {
				t.Fatalf("UnresolvablePathTokens(%q) = %v, want %v", tc.command, got, tc.want)
			}
			for i := range got {
				if got[i] != tc.want[i] {
					t.Fatalf("UnresolvablePathTokens(%q) = %v, want %v", tc.command, got, tc.want)
				}
			}
		})
	}
}

// TestUnresolvablePathTokens_RebindConstructsPoshQuiet pins that the
// rebinding pass is bash-only: PowerShell has no binding walker, so its
// dialect reports nothing for variable references (the documented
// environment-only resolution applies there).
func TestUnresolvablePathTokens_RebindConstructsPoshQuiet(t *testing.T) {
	if got := UnresolvablePathTokens(`Get-Content $env:D\x`, ShellPosh); len(got) != 0 {
		t.Fatalf("posh dialect must report nothing, got %v", got)
	}
}

// TestExtractBashPaths_RebindingConstructsStaySuspicious pins the parser
// layer's half of the invisible-rebinding fix: a decoy literal binding
// followed by a rebinding construct ("read D < cfg") must leave every
// reference to the name unexpandable/suspicious — the walk cannot know the
// runtime value, so the command must never read as statically resolved.
func TestExtractBashPaths_RebindingConstructsStaySuspicious(t *testing.T) {
	t.Setenv("D", "")
	for _, cmd := range []string{
		`D=/ws/safe; read -r D < cfg; cat "$D/x"`,
		`D=/ws/safe; unset D; cat "$D/x"`,
		`D=/ws/safe; printf -v D '%s' /etc; cat "$D/x"`,
		`D=/ws/safe; mapfile -t D < cfg; cat "$D/x"`,
		`for D in /etc/passwd; do cat "$D"; done`,
		`. ./cfg; D=/ws/safe; cat "$D/x"`,
		`eval "D=/etc"; cat "$D/x"`,
	} {
		if _, suspicious := extractBashPaths(cmd, "", ""); !suspicious {
			t.Errorf("expected suspicious for rebinding construct: %q", cmd)
		}
	}
}

// TestExtractBashPaths_ArrayAssignmentMarksDynamic is the regression test
// for the array-ordering hole: "A=(x)" carries its elements in n.Array with
// a nil n.Value, so the naked-declaration guard must not run first — the
// array form must mark the name dynamically assigned, keeping references
// suspicious instead of letting the earlier scalar decoy resolve them.
func TestExtractBashPaths_ArrayAssignmentMarksDynamic(t *testing.T) {
	t.Setenv("A", "")
	cmd := `A=` + bindingFixture("ws", "safe") + `; A=(x y); cat "$A/x"`
	paths, suspicious := extractBashPaths(cmd, "", "")
	if !suspicious {
		t.Fatal("expected suspicious for an array-assigned name (scalar decoy must not resolve it)")
	}
	// The scalar decoy stays visible as a best-effort candidate.
	want := filepath.Clean(osAbsPath("ws", "safe", "x"))
	if !sliceContains(paths, want) {
		t.Fatalf("expected scalar decoy candidate %q in paths, got %v", want, paths)
	}
	if got := UnresolvablePathTokens(cmd, ShellBash); !sliceContains(got, "$A") {
		t.Fatalf("expected $A reported unassessable, got %v", got)
	}
}

// TestExtractBashPaths_NakedExportStillPreservesValue guards the reorder:
// a naked "export D" (no value) still neither binds nor clears — it must NOT
// mark the name dynamic the way the array form does, so a literal binding
// already in the summary keeps resolving cleanly.
func TestExtractBashPaths_NakedExportStillPreservesValue(t *testing.T) {
	bind := bindingFixture("tmp", "build")
	t.Setenv("D", bind)
	// env equals the binding, no empty expansion, no rebinding: clean.
	paths, suspicious := extractBashPaths(`D=`+bind+`; export D; cat "$D/a"`, "", "")
	if suspicious {
		t.Fatalf("expected not suspicious for a binding followed by a naked export with matching env, paths=%v", paths)
	}
	want := filepath.Clean(osAbsPath("tmp", "build", "a"))
	if !sliceContains(paths, want) {
		t.Fatalf("expected %q in paths, got %v", want, paths)
	}
}

// TestPathsOutsideRoots_ReboundVarWithAbsoluteSuffix closes the full
// prompt-injection chain at the escalation layer: "D=safe; read D < cfg"
// rebinds D to attacker-controlled file content, and a reference with an
// absolute path suffix must surface the suffix-alone candidate (the empty
// expansion of a dynamic name is always possible) even when the decoy
// literal itself resolves in-root.
func TestPathsOutsideRoots_ReboundVarWithAbsoluteSuffix(t *testing.T) {
	t.Setenv("D", "")
	ws := t.TempDir()
	ctx := WithWorkspacePath(context.Background(), ws)
	cmd := `D=safe; read -r D < cfg; cat "$D/etc/passwd"`
	if out := PathsOutsideRoots(ctx, cmd, ShellBash, ws); !sliceContains(out, "/etc/passwd") {
		t.Fatalf("expected /etc/passwd reported outside roots, got %v", out)
	}
}
