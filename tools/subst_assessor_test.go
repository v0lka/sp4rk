package tools

import (
	"strings"
	"testing"
	"time"
)

// nestedSubstCommand builds `X=$(echo $(echo … hi …)); cat "$X"` with the
// command substitution nested depth levels deep — the adversarial shape a
// prompt injection can emit to pin the judge's CPU.
func nestedSubstCommand(depth int) string {
	var b strings.Builder
	b.WriteString("X=")
	for range depth {
		b.WriteString("$(echo ")
	}
	b.WriteString("hi")
	for range depth {
		b.WriteString(")")
	}
	b.WriteString(`; cat "$X"`)
	return b.String()
}

// TestCommandSubstitutionAssessment_PathologicalNestingBounded pins the
// availability bound of the nested-substitution assessment: without
// memoization, a depth cap, AND pruning of the walk into an already-assessed
// substitution subtree, every nesting level re-prints (via the AST printer, the
// dominant cost) and re-assesses each nested word — ~quadratic in the input
// size, reaching tens of seconds on a few-KB model-controlled command before
// the confirmation gate. Both judge entry points (UnresolvablePathTokens, the
// symlink gate's extractBashPaths) must complete a DEEP (depth-800) analysis in
// bounded time.
func TestCommandSubstitutionAssessment_PathologicalNestingBounded(t *testing.T) {
	cmd := nestedSubstCommand(800)

	start := time.Now()
	_ = UnresolvablePathTokens(cmd, ShellBash)
	_, _ = extractBashPaths(cmd, t.TempDir(), t.TempDir())
	elapsed := time.Since(start)

	if elapsed > 2*time.Second {
		t.Fatalf("depth-800 nested substitution analysis took %v; the nesting bounds are not effective", elapsed)
	}
}

// TestCommandSubstitutionAssessment_DepthCapFailsClosed pins the semantic of
// the depth cap: nesting beyond maxSubstNestingDepth is rejected (the name
// falls back to dynamic/HARD escalation) while shallower nesting keeps the
// pre-change assessability.
func TestCommandSubstitutionAssessment_DepthCapFailsClosed(t *testing.T) {
	shallow := collectCommandEnvBindings(nestedSubstCommand(maxSubstNestingDepth - 1))
	if shallow == nil {
		t.Fatal("parse failed for within-cap nesting")
	}
	if !shallow.cmdSubst["X"] || shallow.dynamic["X"] {
		t.Errorf("within-cap nesting (depth %d) must keep X assessable: cmdSubst=%v dynamic=%v",
			maxSubstNestingDepth-1, shallow.cmdSubst["X"], shallow.dynamic["X"])
	}
	if toks := UnresolvablePathTokens(nestedSubstCommand(maxSubstNestingDepth-1), ShellBash); len(toks) != 0 {
		t.Errorf("within-cap nested assessable substitution produced unresolvable tokens: %v", toks)
	}

	deep := collectCommandEnvBindings(nestedSubstCommand(maxSubstNestingDepth + 2))
	if deep == nil {
		t.Fatal("parse failed for beyond-cap nesting")
	}
	if deep.cmdSubst["X"] || !deep.dynamic["X"] {
		t.Errorf("beyond-cap nesting (depth %d) must fail closed to dynamic: cmdSubst=%v dynamic=%v",
			maxSubstNestingDepth+2, deep.cmdSubst["X"], deep.dynamic["X"])
	}
	if toks := UnresolvablePathTokens(nestedSubstCommand(maxSubstNestingDepth+2), ShellBash); !sliceContains(toks, "$X") {
		t.Errorf("beyond-cap nesting must keep $X unassessable, got tokens %v", toks)
	}
}
