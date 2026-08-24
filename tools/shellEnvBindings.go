// SPDX-License-Identifier: Apache-2.0

package tools

import (
	"os"
	"strings"

	"mvdan.cc/sh/v3/syntax"
)

// collectCommandEnvBindings parses a bash command string and returns the
// in-command literal variable bindings it declares ("VAR=value",
// "export VAR=value", "VAR=value cmd ..."), or nil when the command cannot be
// parsed (the caller then falls back to environment-only resolution).
func collectCommandEnvBindings(command string) map[string]string {
	parser := syntax.NewParser()
	file, err := parser.Parse(strings.NewReader(command), "")
	if err != nil {
		return nil
	}
	return collectShellEnvBindings(file)
}

// collectShellEnvBindings walks a parsed bash AST and returns a map of
// variable-name → literal-value bindings produced by in-command assignments:
//
//	VAR=value             (a prefix assignment on a simple command)
//	export VAR=value      (an export declaration)
//	readonly VAR=value    (a readonly declaration)
//
// Only purely literal (non-dynamic) right-hand sides are bound. An assignment
// whose value depends on shell expansion — "$VAR", "$(cmd)", backticks,
// arithmetic, or globbing — is treated as statically unknown, so any prior
// literal binding for that name is dropped (fail-closed). A later "$VAR"
// reference then stays flagged as unexpandable rather than resolving to a stale
// value.
//
// Later assignments override earlier ones, mirroring the shell's left-to-right
// single-command-line semantics; the map therefore holds the last literal value
// assigned to each name. Because the walk is a single pre-pass over the whole
// command, bindings declared inside blocks/subshells are collected globally —
// an over-approximation that leans fail-closed (a bound value's resolved path is
// still subject to the normal out-of-root containment check).
func collectShellEnvBindings(file *syntax.File) map[string]string {
	bindings := make(map[string]string)
	syntax.Walk(file, func(node syntax.Node) bool {
		n, ok := node.(*syntax.Assign)
		if !ok {
			return true
		}
		// A naked "export D" (no value) preserves the variable's prior value, so
		// it neither binds nor clears; leave the map untouched.
		if n.Name == nil || n.Value == nil {
			return false
		}
		// Append ("+="), indexed ("A[i]=v") and array ("A=(...)") forms do not
		// cleanly replace the variable with a single literal; drop any prior
		// binding so references remain unexpandable (fail-closed).
		if n.Append || n.Index != nil || n.Array != nil {
			delete(bindings, n.Name.Value)
			return false
		}
		// An RHS requiring shell expansion is statically unknown.
		if wordHasDynamicPart(n.Value) {
			delete(bindings, n.Name.Value)
			return false
		}
		lit := wordLiteral(n.Value)
		if lit == "" {
			delete(bindings, n.Name.Value)
			return false
		}
		bindings[n.Name.Value] = lit
		return false
	})
	return bindings
}

// wordHasDynamicPart reports whether a Word contains any shell construct whose
// value is not known statically — variable expansion, command/process
// substitution, arithmetic expansion, or globbing. It is used to decide whether
// an assignment's RHS is a purely literal value eligible for in-command binding.
func wordHasDynamicPart(w *syntax.Word) bool {
	if w == nil {
		return false
	}
	return wordPartsHaveDynamicPart(w.Parts)
}

func wordPartsHaveDynamicPart(parts []syntax.WordPart) bool {
	for _, part := range parts {
		switch p := part.(type) {
		case *syntax.Lit, *syntax.SglQuoted:
			// Purely literal fragments.
		case *syntax.DblQuoted:
			if wordPartsHaveDynamicPart(p.Parts) {
				return true
			}
		default:
			// *syntax.ParamExp, *syntax.CmdSubst, *syntax.ProcSubst,
			// *syntax.ArithmExp, *syntax.ExtGlob.
			return true
		}
	}
	return false
}

// expandWordWithBindings returns the literal value of a Word after substituting
// in-command variable bindings ("VAR=value") for bound variables, plus whether
// the word still contains unexpandable/dynamic content. Bound variables expand
// to their literal value, but a binding never MASKS the process environment:
// because the binding pre-pass is position- and control-flow-unaware (an
// assignment inside a branch not taken still lands in the map), a name that
// ALSO has a non-empty, differing os.Getenv value is ambiguous — the word is
// resolved to the bound literal AND marked unexpandable so the caller keeps
// the command suspicious (fail-closed over-report). Unbound "$VAR", "$(cmd)",
// backticks, process/command substitution, and arithmetic are left unexpanded
// (the literal fragments are concatenated, matching the shell's empty-expansion
// of an unset variable) and the word is marked unexpandable so the caller can
// flag the command suspicious.
func expandWordWithBindings(w *syntax.Word, bindings map[string]string) (literal string, unexpandable bool) {
	if w == nil {
		return "", false
	}
	return expandWordPartsWithBindings(w.Parts, bindings)
}

func expandWordPartsWithBindings(parts []syntax.WordPart, bindings map[string]string) (string, bool) {
	var lit string
	unexp := false
	for _, part := range parts {
		switch p := part.(type) {
		case *syntax.Lit:
			lit += p.Value
		case *syntax.SglQuoted:
			lit += p.Value
		case *syntax.DblQuoted:
			inner, innerUnexp := expandWordPartsWithBindings(p.Parts, bindings)
			lit += inner
			if innerUnexp {
				unexp = true
			}
		case *syntax.ParamExp:
			if p.Param != nil {
				if val, ok := bindings[p.Param.Value]; ok {
					lit += val
					// Union (fail-closed): the binding pre-pass cannot prove the
					// assignment is in effect where this reference executes, so a
					// differing non-empty environment value remains a possible
					// runtime expansion. Keep the word unexpandable so the
					// command stays suspicious; the caller still resolves the
					// bound literal as a path candidate.
					if envVal := os.Getenv(p.Param.Value); envVal != "" && envVal != val {
						unexp = true
					}
				} else {
					unexp = true
				}
			} else {
				unexp = true
			}
		default:
			// *syntax.CmdSubst, *syntax.ProcSubst, *syntax.ArithmExp,
			// *syntax.ExtGlob.
			unexp = true
		}
	}
	return lit, unexp
}
