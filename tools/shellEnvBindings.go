// SPDX-License-Identifier: Apache-2.0

package tools

import (
	"os"
	"path"
	"strings"

	"mvdan.cc/sh/v3/syntax"
)

// envBindings summarizes the in-command variable bindings and rebinding
// constructs of a bash command for static resolution.
//
// A name is RESOLVABLE only when the whole command assigns it exactly once,
// with a purely literal value ("VAR=value", "export VAR=value",
// "VAR=value cmd ...") and nothing else in the command can change or clear
// it. Anything else — a second assignment of any kind, an append ("+="), an
// indexed/array form, a dynamic right-hand side ("$(cmd)", "$VAR",
// backticks, arithmetic, globbing), or a rebinding construct that never
// produces a plain assignment node ("read D", "printf -v D", "mapfile D",
// "unset D", "getopts", for/select loop iteration, an arithmetic assignment
// "((D=5))", an attribute-changing "declare -a/-n/…", or an opaque
// "source"/"."/"eval"/"let" that can rebind ANY name) — makes every
// reference to that name statically ambiguous: the walk is position- and
// control-flow-unaware (an assignment inside a branch not taken still lands
// in the summary, and a re-binding after a reference must not retroactively
// mask the value in effect at that reference), so ambiguity fails closed —
// references stay flagged unexpandable/suspicious while every distinct
// literal value is still reported as a resolution candidate, mirroring the
// environment-value union. A binding adds candidates, it never masks one —
// including the EMPTY expansion of a possibly-unset or cleared variable.
// Constructs that make the child process run with a DIFFERENT environment
// than the judge's ("env -i", "exec -c"/"exec -l", "sudo" with its default
// env_reset) are likewise opaque: the values observed in this process say
// nothing about the child's expansions. The implicitly-mutating special
// parameters ("$_", "RANDOM", "SECONDS", "LINENO") are NOT modeled as
// rebinding — they are excluded from the binding walk by design; a plain
// "$_" reference is treated like any unbound name (environment value plus
// the empty expansion), which still escalates path-suffixed uses.
type envBindings struct {
	// values maps each assigned name to its DISTINCT literal values in
	// first-assignment order.
	values map[string][]string
	// dynamic marks names with at least one binding whose value cannot be
	// determined statically: a dynamic RHS, an append/index/array form, an
	// empty literal, or any non-Assign rebinding construct. Such names are
	// never resolvable and references to them are unassessable (see
	// [UnresolvablePathTokens]).
	dynamic map[string]bool
	// opaque marks a construct that can rebind ANY variable invisibly to the
	// walk ("source", ".", "eval", "let", or an arithmetic assignment whose
	// target cannot be named). With it set no name is resolvable and every
	// variable reference in the command is unassessable.
	opaque bool
	// positionalRebound marks a construct that rebinds the POSITIONAL
	// parameters ("set x y", "set -- …", "shift"): "$1", "$@" and "$*" then
	// hold statically unknown values (see [UnresolvablePathTokens]).
	positionalRebound bool
}

// collectCommandEnvBindings parses a bash command string and returns the
// in-command variable-binding summary it declares, or nil when the command
// cannot be parsed (the caller then falls back to environment-only
// resolution).
func collectCommandEnvBindings(command string) *envBindings {
	parser := syntax.NewParser()
	file, err := parser.Parse(strings.NewReader(command), "")
	if err != nil {
		return nil
	}
	return collectShellEnvBindings(file)
}

// collectShellEnvBindings walks a parsed bash AST and summarizes the variable
// bindings and rebinding constructs declared in-command. The walk runs in
// two phases: assignments first (plain Assign nodes and declaration
// clauses), then the rebinding constructs — so construct analysis sees the
// command's literal bindings (a prefix assignment like "BASH_ENV=cfg bash -c
// …" is folded before the interpreter handling consults it, which a single
// AST-ordered walk could not guarantee). Only purely literal (non-dynamic)
// plain assignments are recorded as values; every other form — including
// the rebinding builtins that never produce *syntax.Assign nodes — makes
// the affected names (or, for "source"/"eval"-style constructs, every name)
// ambiguous. See the envBindings type comment for the fail-closed contract.
//
// Because each phase is a single pre-pass over the whole command,
// assignments declared inside blocks/subshells/branches are collected too —
// an over-approximation that leans fail-closed (every collected value is
// reported as a resolution candidate, and the empty expansion stays
// possible, so a subshell or dead-branch binding can never mask the unset or
// environment value in effect at a reference).
func collectShellEnvBindings(file *syntax.File) *envBindings {
	b := &envBindings{
		values:  make(map[string][]string),
		dynamic: make(map[string]bool),
	}
	// Phase 1: fold every assignment-shaped node.
	syntax.Walk(file, func(node syntax.Node) bool {
		switch n := node.(type) {
		case *syntax.Assign:
			b.walkAssign(n)
		case *syntax.DeclClause:
			b.walkDeclClause(n)
		}
		return true
	})
	// Phase 2: fold the rebinding constructs.
	syntax.Walk(file, func(node syntax.Node) bool {
		switch n := node.(type) {
		case *syntax.CallExpr:
			b.walkCallExpr(n)
		case *syntax.ForClause:
			b.walkForClause(n)
		case *syntax.ArithmCmd:
			b.walkArithm(n.X)
		case *syntax.ArithmExp:
			// A word-level arithmetic assignment ("echo $((D=5))") rebinds
			// like an arithmetic command does.
			b.walkArithm(n.X)
		case *syntax.LetClause:
			// "let D=5 …" rebinds names through arithmetic whose operands are
			// opaque quoted words; fail closed for the whole command.
			b.opaque = true
		case *syntax.Stmt:
			// A bare interpreter fed its script over stdin (heredoc,
			// here-string) executes in-command text the "-c" handling never
			// sees.
			b.walkStmtStdinScript(n)
		case *syntax.BinaryCmd:
			// A pipeline ending in a bare interpreter executes the piped
			// text as its script.
			b.walkPipeInterpreter(n)
		}
		return true
	})
	return b
}

// walkAssign folds a plain assignment node into the summary. Append, indexed
// and array forms are checked BEFORE the naked-declaration guard: an array
// assignment ("A=(…)") carries its elements in n.Array with a nil n.Value,
// so the naked guard would otherwise silently drop it and leave the name
// looking unassigned (fail-open).
func (b *envBindings) walkAssign(n *syntax.Assign) {
	if n.Name == nil {
		// A flag word inside a DeclClause ("declare -a …") or a malformed
		// node; neither binds anything.
		return
	}
	name := n.Name.Value
	// Append ("+="), indexed ("A[i]=v") and array ("A=(...)") forms do not
	// cleanly replace the variable with a single literal; the name is
	// statically unknown.
	if n.Append || n.Index != nil || n.Array != nil {
		b.dynamic[name] = true
		return
	}
	if n.Value == nil {
		// A naked "export D" (no value) preserves the variable's prior value,
		// so it neither binds nor clears; leave the summary untouched.
		return
	}
	// An RHS requiring shell expansion is statically unknown — unless the
	// expansion is a pure SELF-reference ("PATH=$PATH:/x"): an append-style
	// self-assignment's runtime value is the prior value plus literal
	// fragments, which later references already assess through the
	// environment-value and empty-expansion unions, so the name stays
	// assessable instead of hard-escalating every "$PATH" in the command.
	// A self-reference whose fragment itself carries a PATH component
	// ("D=$D/etc/passwd", "D=$D"tc chains) composes a value no candidate
	// list contains, so only ':'-prefixed pathlist appends keep the relief.
	if wordHasDynamicPart(n.Value) {
		if wordSelfReferenceOnly(n.Value, name) && !selfRefCarriesPathComponent(n.Value) {
			// The name stays assessable, but the composed runtime values
			// (prior candidates ∘ literal fragments) join the union — a
			// composed value no candidate list contains would otherwise be
			// invisible ("D=/et; D=$D\"c\"" must keep "/etc" visible).
			b.composeSelfReferenceCandidates(name, n.Value)
			return
		}
		b.dynamic[name] = true
		return
	}
	lit := wordLiteral(n.Value)
	if lit == "" {
		b.dynamic[name] = true
		return
	}
	if !slicesContains(b.values[name], lit) {
		b.values[name] = append(b.values[name], lit)
	}
}

// declBenignFlagChars lists the declaration-command attribute letters that
// do NOT change a variable's value: "x" export, "g" global, "p" display,
// "r" readonly, "f" function. Any other attribute letter can convert or
// clear the named variables ("-a"/"-A" array, "-i" integer, "-n" nameref,
// "-l"/"-u"/"-c" case conversion, and any future letter) and fails closed.
const declBenignFlagChars = "xgprf"

// declFlagConverts reports whether a declaration-command flag word carries a
// value-CONVERTING attribute. Fail-closed by construction: every attribute
// letter outside the benign set counts as converting.
func declFlagConverts(flag string) bool {
	if !strings.HasPrefix(flag, "-") {
		return false // "+" forms remove attributes; removal cannot convert.
	}
	return strings.TrimLeft(flag[1:], declBenignFlagChars) != ""
}

// walkDeclClause folds a declaration command ("declare", "local", "typeset",
// "export", "readonly", "nameref") into the summary. Its arguments surface
// as Assign nodes, with flag words appearing as Assign nodes carrying a nil
// Name. A clause carrying a value-CONVERTING attribute flag ("-a"/"-A"
// array, "-i" integer, "-n" nameref, "-l"/"-u"/"-c" case conversion, and
// any unrecognized letter — see [declFlagConverts]) can clear or transform
// the named variables — an array conversion resets the value to empty, a
// nameref makes every reference expand ANOTHER variable's value, case
// conversion rewrites the string — so its names are statically unknown.
// Benign attributes ("-x" export, "-g" global, "-p" display, "-r" readonly)
// leave the assigned value intact and do not. A flag-less clause
// ("export VAR=value") binds like a plain assignment and is folded by
// walkAssign through the ordinary Assign walk.
func (b *envBindings) walkDeclClause(n *syntax.DeclClause) {
	converting := n.Variant != nil && n.Variant.Value == "nameref"
	for _, arg := range n.Args {
		if arg.Name == nil {
			// A flag word such as the "-a" in "declare -a A". A DYNAMIC
			// flag word ("declare -$F R=D" — wordLiteral collapses it to
			// "-") can carry any attribute letter at runtime, so it counts
			// as converting: fail closed rather than read the collapsed
			// literal.
			if wordHasDynamicPart(arg.Value) || declFlagConverts(wordLiteral(arg.Value)) {
				converting = true
			}
		}
	}
	if !converting {
		return
	}
	for _, arg := range n.Args {
		if arg.Name == nil {
			continue
		}
		// A nameref clause ("declare -n R=D") makes every assignment THROUGH
		// "R" rebind the target "D" as well, so the target name is just as
		// statically unknown as the reference itself.
		if arg.Value != nil {
			if target := wordLiteral(arg.Value); isShellVarName(target) {
				b.dynamic[target] = true
			}
		}
		b.dynamic[arg.Name.Value] = true
	}
}

// rebindDefaultTargets maps the rebinding builtins that assign a DEFAULT
// target variable when no name argument is given.
var rebindDefaultTargets = map[string]string{
	"read":      "REPLY",
	"mapfile":   "MAPFILE",
	"readarray": "MAPFILE",
}

// markEnvUnsetOperand records the NAME operand of env(1)'s "-u"/"--unset":
// the child runs WITHOUT that variable, so its references expand to values
// the judge's process environment says nothing about (the empty expansion
// is the only child-side certainty) — the name is marked unassessable. It
// reports false when the operand cannot be statically pinned (dynamic, or
// not a plain variable name), in which case the caller fails closed.
func (b *envBindings) markEnvUnsetOperand(w *syntax.Word) bool {
	if w == nil || wordHasDynamicPart(w) {
		return false
	}
	name := wordLiteral(w)
	if !isShellVarName(name) {
		return false
	}
	b.dynamic[name] = true
	return true
}

// unescapeCommandWord strips the backslash escapes a shell preserves inside
// command words, so an obfuscated invocation ("\source", "s\ource", "b\ash")
// resolves to the builtin it names — for an unquoted word each "\x" IS "x".
// Quoted parts need no unescaping: wordLiteral already returns their
// unquoted value. Over-collapsing an unusual literal can only alias an
// unknown word onto a known builtin — an over-report in the fail-closed
// direction.
func unescapeCommandWord(lit string) string {
	return strings.ReplaceAll(lit, `\`, "")
}

// resolveCommandWord returns the effective command word of a CallExpr's
// argument list plus the index of its first operand, unwrapping
// "command"/"builtin" wrapper chains — including nested wrappers and their
// option words ("command -p command source …", "builtin -- source …") — and
// stripping backslash escapes from the resolved word. dynamic reports that
// a command word (or a skipped word) carried shell expansion — such a
// command can be ANY command at runtime. envCleared reports that an
// unwrapped "exec" carried an environment-clearing or login-shell option
// ("-c", "-l") — the child runs with an environment the judge's process
// cannot observe, so every later expansion is unassessable.
func resolveCommandWord(args []*syntax.Word) (cmd string, rest int, dynamic, envCleared bool) {
	if len(args) == 0 {
		return "", 0, false, false
	}
	if wordHasDynamicPart(args[0]) {
		return "", 0, true, false
	}
	cmd = unescapeCommandWord(wordLiteral(args[0]))
	rest = 1
	for cmd == "command" || cmd == "builtin" || cmd == "exec" {
		if rest >= len(args) {
			return "", len(args), false, envCleared
		}
		if wordHasDynamicPart(args[rest]) {
			return "", rest, true, envCleared
		}
		w := unescapeCommandWord(wordLiteral(args[rest]))
		if (cmd == "command" || cmd == "builtin") && (w == "-v" || w == "-V") {
			// A query ("command -v name") reports ABOUT a command; it never
			// invokes one.
			return "", len(args), false, envCleared
		}
		if w == "-a" {
			// exec's "-a NAME" carries an operand.
			rest += 2
			continue
		}
		if cmd == "exec" && (w == "-c" || w == "-l") {
			// "exec -c" clears the child environment entirely; "exec -l"
			// simulates a login shell (env semantics the walker cannot
			// model). Either way the process env the judge observes is no
			// longer the child's.
			envCleared = true
			rest++
			continue
		}
		if w != "-" && strings.HasPrefix(w, "-") {
			rest++ // a wrapper option word ("-p", "--")
			continue
		}
		cmd = w
		rest++
	}
	return cmd, rest, false, envCleared
}

// walkCallExpr folds the rebinding shell builtins that never produce
// *syntax.Assign nodes into the summary. "read", "mapfile" and "readarray"
// assign file or stdin content (attacker-controllable), "printf -v NAME"
// assigns formatted output, "unset" clears, and "getopts" assigns the parsed
// option — the affected names are statically unknown. "source", "." and
// "eval" (however spelled: escaped, quoted, or behind a "command"/"builtin"
// wrapper chain with options) execute arbitrary shell text and can rebind
// ANY name, so the whole command goes opaque. The same applies when the
// command word itself is dynamically built ("$TOOL", "$E"), when an "alias"
// definition or "shopt -s expand_aliases" introduces syntax-level expansion,
// and to a shell interpreter's "-c" script, which a purely literal script
// contributes through a nested parse (see [envBindings.walkShellDash]) and
// any other form turns opaque. "env(1)" "NAME=VALUE" argument words, a
// wrapper- or escape-spelled "declare"/"export"/…, and "set --"/"shift"
// rebind their targets without Assign nodes and are folded per-name (or, for
// positionals, flagged for [UnresolvablePathTokens]).
func (b *envBindings) walkCallExpr(n *syntax.CallExpr) {
	cmd, rest, dynamic, envCleared := resolveCommandWord(n.Args)
	if dynamic || envCleared {
		b.opaque = true
		return
	}
	if cmd == "" {
		// A pure assignment statement ("VAR=value") carries no command word
		// — and path.Base("") would yield ".", colliding with the dot-source
		// builtin.
		return
	}
	// A qualified invocation ("./bash", "/usr/bin/bash") names the same
	// command as its base name.
	cmd = path.Base(cmd)
	args := n.Args
	switch cmd {
	case "source", ".", "eval", "let":
		b.opaque = true
	case "bash", "sh", "dash", "zsh", "ksh", "ksh93", "mksh", "ash":
		b.walkShellDash(args, rest, cmd)
	case "cd", "pushd", "popd":
		// cd/pushd rebind PWD (and OLDPWD) to the new directory; popd pops
		// the stack. A LITERAL target contributes the directory as a PWD
		// value candidate, so a reference surfaces the new working root
		// ("cd /; cat "$PWD/etc/passwd"" reports /etc/passwd through the
		// candidate union) while "cd sub && cat "$PWD/x"" stays quiet (the
		// joined candidate is relative, hence in-root). A non-literal or
		// absent target makes PWD statically unknown. pushd/popd maintain
		// DIRSTACK on top.
		if cmd == "popd" {
			b.dynamic["PWD"] = true
			b.dynamic["OLDPWD"] = true
			b.dynamic["DIRSTACK"] = true
			break
		}
		sawTarget := false
		for _, w := range args[rest:] {
			if wordHasDynamicPart(w) {
				b.dynamic["PWD"] = true
				b.dynamic["OLDPWD"] = true
				sawTarget = true
				continue
			}
			lit := wordLiteral(w)
			if lit == "" || strings.HasPrefix(lit, "-") {
				// Flags ("-L", "-P", "--"); "-" alone means $OLDPWD.
				if lit == "-" {
					b.dynamic["PWD"] = true
					b.dynamic["OLDPWD"] = true
					sawTarget = true
				}
				continue
			}
			if !slicesContains(b.values["PWD"], lit) {
				b.values["PWD"] = append(b.values["PWD"], lit)
			}
			sawTarget = true
		}
		if !sawTarget {
			// Bare "cd" goes to $HOME — statically unknown here.
			b.dynamic["PWD"] = true
			b.dynamic["OLDPWD"] = true
		}
		if cmd == "pushd" {
			b.dynamic["DIRSTACK"] = true
		}
	case "alias":
		// An alias definition rewrites shell SYNTAX — with "shopt -s
		// expand_aliases" any later word can expand to arbitrary shell text
		// the walk never sees.
		if len(args) > rest {
			b.opaque = true
		}
	case "shopt":
		for _, w := range args[rest:] {
			if strings.Contains(wordLiteral(w), "expand_aliases") {
				b.opaque = true
			}
		}
	case "sudo", "su", "doas", "ssh":
		// sudo(8) defaults to env_reset, "su -"/"-l"/"--login" replaces the
		// whole environment and euid, doas(1) sanitizes like sudo, and
		// ssh(1) runs the command in a REMOTE shell whose environment the
		// judge's process cannot observe (custom vars are not forwarded by
		// default). In every case the values judge-observed here (HOME,
		// PATH, custom vars) may be absent or different where the child
		// actually expands them — no expansion inside these command lines
		// is statically assessable.
		b.opaque = true
	case "env":
		// env(1) "NAME=VALUE" argument words inject child-side bindings that
		// never surface as Assign nodes; "-u"/"--unset" REMOVE a name
		// child-side (so the empty expansion becomes possible there) — an
		// unset operand that cannot be statically pinned (dynamic, absent,
		// attached to a flag bundle, or not a plain variable name) makes the
		// whole command opaque, because the removal escapes both the
		// candidate union and the unassessable flags — and "-S" embeds a
		// whole command line inside one quoted word (GNU env).
		for i := rest; i < len(args); i++ {
			if wordHasDynamicPart(args[i]) {
				b.opaque = true
				break
			}
			lit := wordLiteral(args[i])
			if lit == "-u" {
				if i+1 >= len(args) || !b.markEnvUnsetOperand(args[i+1]) {
					// An operand the walker cannot pin — dynamic, absent, or
					// not a plain variable name — removes an UNKNOWN name
					// child-side: the removal escapes both the candidate
					// union and the unassessable flags, so fail closed.
					b.opaque = true
					break
				}
				i++
				continue
			}
			if len(lit) > 2 && strings.HasPrefix(lit, "-u") && !strings.HasPrefix(lit, "--") {
				// "-uNAME" carries the operand attached to the flag; only a
				// plain variable name is pinnable, anything else (an operand
				// with metacharacters, a non-name tail) fails closed.
				if name := lit[2:]; isShellVarName(name) {
					b.dynamic[name] = true
					continue
				}
				b.opaque = true
				break
			}
			if lit == "-i" || lit == "--ignore-environment" {
				// "env -i" runs the child with an EMPTY environment: every
				// expansion inside it resolves to the empty string at
				// runtime while the judge's process env would happily
				// supply a value — the exact masking the empty-expansion
				// union exists to prevent. Fail closed.
				b.opaque = true
				break
			}
			if lit == "-f" || lit == "--file" {
				// GNU env's "-f FILE" injects a hidden child environment
				// (a BASH_ENV channel) whose content is not statically
				// available — fail closed.
				b.opaque = true
				break
			}
			if strings.HasPrefix(lit, "--file=") {
				b.opaque = true
				break
			}
			if name, ok := strings.CutPrefix(lit, "--unset="); ok {
				if !isShellVarName(name) {
					// An operand that cannot be pinned (empty, malformed, or
					// carrying metacharacters) removes an unknown name
					// child-side — fail closed.
					b.opaque = true
					break
				}
				b.dynamic[name] = true
				continue
			}
			if lit == "--unset" {
				if i+1 >= len(args) || !b.markEnvUnsetOperand(args[i+1]) {
					b.opaque = true
					break
				}
				i++
				continue
			}
			if lit == "-S" || lit == "--split-string" {
				if i+1 >= len(args) {
					break
				}
				if wordHasDynamicPart(args[i+1]) {
					b.opaque = true
					break
				}
				inner := collectCommandEnvBindings(wordLiteral(args[i+1]))
				if inner == nil {
					b.opaque = true
					break
				}
				b.merge(inner)
				break
			}
			if strings.HasPrefix(lit, "-") {
				// A short-flag bundle carrying an argument-taking flag
				// ("-iu", "-fu X") hides an unset operand, an env-file
				// operand, or an embedded "-S" command line in the rest of
				// the word or the next argv — none of it is statically
				// pinnable, so fail closed. Unknown short flags ("-v") and
				// long flags stay skipped.
				if !strings.HasPrefix(lit, "--") && strings.ContainsAny(lit, "ufS") {
					b.opaque = true
					break
				}
				continue
			}
			name, _, ok := splitAssignWord(lit)
			if !ok {
				break // first non-assignment word is the child command
			}
			b.dynamic[name] = true
		}
	case "set":
		// "set x y" or "set -- …" rebinds the positional parameters ("--"
		// alone clears them); option-only forms ("set -euo pipefail",
		// "set -o pipefail", "set +x") do not — "-o"/"+o" consume a
		// following option NAME operand, and "+"-prefixed words merely
		// DISABLE options, so they rebind nothing either.
		for i := rest; i < len(args); i++ {
			lit := wordLiteral(args[i])
			if lit == "-o" || lit == "+o" {
				i++ // the option-name operand is not a positional argument.
				continue
			}
			if lit == "--" || (!strings.HasPrefix(lit, "-") && !strings.HasPrefix(lit, "+")) {
				b.positionalRebound = true
				break
			}
		}
	case "shift":
		b.positionalRebound = true
	case "trap":
		// A trap handler executes LATER IN THE CURRENT SHELL (DEBUG/RETURN/
		// EXIT and signal traps alike), so its text can rebind any variable
		// between commands: a purely literal handler is parsed and merged
		// like a nested script, anything else fails closed.
		for i := rest; i < len(args); i++ {
			if wordHasDynamicPart(args[i]) {
				b.opaque = true
				break
			}
			lit := wordLiteral(args[i])
			if lit == "--" && i == rest {
				continue
			}
			if strings.HasPrefix(lit, "-") {
				continue // a trap option ("-l", "-p")
			}
			inner := collectCommandEnvBindings(lit)
			if inner == nil {
				b.opaque = true
				break
			}
			b.merge(inner)
			break
		}
	case "declare", "local", "typeset", "export", "readonly", "nameref":
		// Spelled behind a wrapper or escape ("builtin declare -n R=D",
		// "command export D=x", "\export D=x") these parse as plain CallExpr
		// argument words rather than DeclClause Assign nodes, and the walker
		// cannot see which flags convert — so every name-bearing word is
		// treated as a converting declaration. A dynamically-spelled word
		// names an UNKNOWN target: fail closed for the whole command.
		for _, w := range args[rest:] {
			if wordHasDynamicPart(w) {
				b.opaque = true
				continue
			}
			lit := wordLiteral(w)
			name, value, ok := splitAssignWord(lit)
			if !ok {
				if isShellVarName(lit) {
					b.dynamic[lit] = true
				}
				continue
			}
			b.dynamic[name] = true
			// A nameref target ("declare -n R=D" rebinding through R) is as
			// unknown as the reference; flags are hidden here, so any
			// name-shaped value is marked conservatively.
			if isShellVarName(value) {
				b.dynamic[value] = true
			}
		}
	case "read", "mapfile", "readarray", "unset", "getopts":
		marked := false
		if cmd == "getopts" {
			// getopts assigns its implicit targets even when the option
			// name argument is quoted: OPTARG carries the parsed option
			// argument and OPTIND the index.
			b.dynamic["OPTARG"] = true
			b.dynamic["OPTIND"] = true
			marked = true
		}
		for _, w := range args[rest:] {
			if wordHasDynamicPart(w) {
				// A dynamically-spelled target ("read -r "$N"") rebinds an
				// UNKNOWN name — every reference in the command becomes
				// unassessable.
				b.opaque = true
				continue
			}
			lit := wordLiteral(w)
			// Flags ("-r", "-u"), their arguments ("3", "prompt: ") and any
			// free-form word are skipped; marking an option argument that
			// happens to look like a name merely over-reports (fail-closed).
			// An ARRAY-ELEMENT target ("read 'A[0]'") rebinds the base name.
			if !isShellVarName(lit) {
				if base, ok := arrayElementTarget(lit); ok {
					b.dynamic[base] = true
					marked = true
				}
				continue
			}
			b.dynamic[lit] = true
			marked = true
		}
		if !marked {
			// "read" with no name assigns REPLY; "mapfile"/"readarray" with
			// no array assigns MAPFILE.
			if def := rebindDefaultTargets[cmd]; def != "" {
				b.dynamic[def] = true
			}
		}
	case "printf":
		for i := rest; i+1 < len(args); i++ {
			if wordLiteral(args[i]) != "-v" {
				continue
			}
			if wordHasDynamicPart(args[i+1]) {
				// A dynamically-spelled target ("printf -v "$N"") rebinds
				// an UNKNOWN name — fail closed.
				b.opaque = true
			} else if lit := wordLiteral(args[i+1]); isShellVarName(lit) {
				b.dynamic[lit] = true
			} else if base, ok := arrayElementTarget(lit); ok {
				// An array-element target ("printf -v 'A[0]'") rebinds the
				// base name.
				b.dynamic[base] = true
			}
			break
		}
	}
	// A nested interpreter script can also hide behind wrappers that
	// resolveCommandWord does not unwrap ("timeout 30 bash -c …", "nohup
	// bash -lc …"): scan the argument words for an interpreter and fold it
	// through the full nested-script handling — the "-c" parse/merge as
	// well as the stdin/auto-source fail-closed paths. The child process
	// cannot rebind the caller's variables, but the script's OWN references
	// run inside it.
	if !shellIsInterpreter(cmd) {
		for i := rest; i < len(args); i++ {
			if wordHasDynamicPart(args[i]) {
				if isDashCFlag(wordLiteral(args[min(i+1, len(args)-1)])) {
					b.opaque = true // an expansion-built interpreter word
				}
				continue
			}
			if word := unescapeCommandWord(wordLiteral(args[i])); shellIsInterpreter(word) {
				b.walkShellDash(args, i, word)
			}
		}
	}
}

// arrayElementTarget splits an "A[0]"-shaped target word into its base array
// name, reporting whether lit names an array element of a plain variable.
func arrayElementTarget(lit string) (base string, ok bool) {
	idx := strings.IndexByte(lit, '[')
	if idx <= 0 || !strings.HasSuffix(lit, "]") {
		return "", false
	}
	base = lit[:idx]
	if !isShellVarName(base) {
		return "", false
	}
	return base, true
}

// shellIsInterpreter reports whether cmd is a shell interpreter name whose
// "-c" argument is nested command text. A qualified path ("./bash",
// "/usr/bin/bash") names the same interpreter as its base name.
func shellIsInterpreter(cmd string) bool {
	switch path.Base(cmd) {
	case "bash", "sh", "dash", "zsh", "ksh", "ksh93", "mksh", "ash":
		return true
	}
	return false
}

// walkStmtStdinScript folds a statement whose bare interpreter command is
// fed its SCRIPT over stdin — a heredoc ("bash <<'EOF' … EOF", the body
// lives in the redirect's Hdoc word) or a here-string ("bash <<< 'script'").
// A literal body is parsed and merged like a nested "-c" script; a body
// with dynamic parts (an unquoted heredoc delimiter expands) fails closed.
func (b *envBindings) walkStmtStdinScript(s *syntax.Stmt) {
	call, ok := s.Cmd.(*syntax.CallExpr)
	if !ok || len(call.Args) == 0 {
		return
	}
	// The interpreter may sit behind a wrapper ("timeout 30 bash < cfg") —
	// scan the argument words for it, like the nested "-c" scan does.
	interpIdx := -1
	for i, w := range call.Args {
		if wordHasDynamicPart(w) {
			if i == 0 {
				return
			}
			continue
		}
		if word := unescapeCommandWord(wordLiteral(w)); shellIsInterpreter(word) {
			interpIdx = i
			break
		}
	}
	if interpIdx < 0 {
		return
	}
	interp := unescapeCommandWord(wordLiteral(call.Args[interpIdx]))
	if !posixShellDialect(interp) {
		// A zsh/ksh script carries sigil spellings the bash grammar cannot
		// model — fail closed.
		b.opaque = true
		return
	}
	for _, r := range s.Redirs {
		if r.Op == syntax.RdrIn {
			// An input-redirect-fed interpreter ("bash < cfg") executes
			// file content no static pass can see — fail closed.
			b.opaque = true
			return
		}
	}
	for _, r := range s.Redirs {
		body := r.Hdoc
		if body == nil && r.Op == syntax.WordHdoc {
			body = r.Word
		}
		if body == nil {
			continue
		}
		if wordHasDynamicPart(body) {
			b.opaque = true
			return
		}
		inner := collectCommandEnvBindings(wordLiteral(body))
		if inner == nil {
			b.opaque = true // unparseable stdin script — fail closed
			return
		}
		b.merge(inner)
		return
	}
}

// walkPipeInterpreter folds a pipeline whose LAST command is a bare shell
// interpreter ("printf … | bash"): the interpreter executes the piped text
// as its script — in-command dynamic text the walk cannot see — so the
// command fails closed. Benign self-pipes are rare enough that the
// over-report is the acceptable fail-closed price.
func (b *envBindings) walkPipeInterpreter(n *syntax.BinaryCmd) {
	if n.Op != syntax.Pipe {
		return
	}
	call, ok := n.Y.Cmd.(*syntax.CallExpr)
	if !ok || len(call.Args) == 0 {
		return
	}
	// The interpreter may sit behind a wrapper ("printf … | timeout 10
	// bash") — any argument word naming one makes the piped text its
	// script; so does a dynamically-built command word.
	for i, w := range call.Args {
		if wordHasDynamicPart(w) {
			if i == 0 || i == len(call.Args)-1 {
				b.opaque = true
				return
			}
			continue
		}
		if shellIsInterpreter(unescapeCommandWord(wordLiteral(w))) {
			b.opaque = true
			return
		}
	}
}

// isDashCFlag reports whether a flag word carries the interpreter's "-c"
// option — either exactly ("-c") or inside a bundled short-flag spelling
// ("-lc", "-ec", "-uc") — so the following word is the script.
func isDashCFlag(lit string) bool {
	return strings.HasPrefix(lit, "-") && !strings.HasPrefix(lit, "--") && strings.Contains(lit, "c")
}

// posixShellDialect reports whether cmd speaks a dialect the bash parser can
// approximate safely: bash, POSIX sh, dash and busybox ash. zsh and the ksh
// family carry sigil spellings ("$=D", "${(z)D}") and nameref semantics the
// bash grammar cannot see, so their nested scripts fail closed instead of
// being parsed.
func posixShellDialect(cmd string) bool {
	switch path.Base(cmd) {
	case "bash", "sh", "dash", "ash":
		return true
	}
	return false
}

// walkShellDash folds a shell interpreter invocation carrying a "-c" script:
// the nested command text is invisible to the outer parse, so a purely
// literal script is re-parsed and merged into this summary, while any other
// form (expanded, multi-word, or unparseable) fails closed for the whole
// command. An auto-source channel — a "--rcfile"/"--init-file" option or a
// non-empty BASH_ENV/ENV/ZDOTDIR in the command or the environment — makes
// the interpreter execute an extra file the nested parse never sees, so it
// fails closed too; so does a non-POSIX dialect (zsh, ksh), whose sigil
// spellings the bash grammar cannot model. Without "-c" and without
// auto-sourcing the interpreter runs a CHILD process, which cannot rebind
// the caller's variables — nothing is marked.
func (b *envBindings) walkShellDash(args []*syntax.Word, rest int, cmd string) {
	if !posixShellDialect(cmd) {
		b.opaque = true
		return
	}
	for i := rest; i < len(args); i++ {
		lit := wordLiteral(args[i])
		switch {
		case lit == "--rcfile" || lit == "--init-file":
			b.opaque = true
			return
		case isDashCFlag(lit):
			if i+1 >= len(args) {
				return
			}
			script := args[i+1]
			if wordHasDynamicPart(script) || b.autoSourcesEnv() {
				b.opaque = true
				return
			}
			inner := collectCommandEnvBindings(wordLiteral(script))
			if inner == nil {
				b.opaque = true // unparseable nested script — fail closed
				return
			}
			b.merge(inner)
			// Words after the script become the child's POSITIONAL
			// parameters ("bash -c '…' _ /e tc/passwd"): the script's "$1"
			// references then hold statically unknown values.
			if i+2 < len(args) {
				b.positionalRebound = true
			}
			return
		}
	}
	// No "-c" script: an auto-source channel (BASH_ENV/ENV/ZDOTDIR set, or
	// a "--rcfile" option caught above) still makes the interpreter execute
	// an extra file; a "-s"-bearing flag bundle reads the script from
	// stdin; and a dynamic argument word ("bash <(cat cfg)") feeds it
	// generated script text — all fail closed. A plain "bash script.sh"
	// argument runs a child process on a file and stays out of scope by
	// design.
	if b.autoSourcesEnv() {
		b.opaque = true
		return
	}
	for i := rest; i < len(args); i++ {
		if lit := wordLiteral(args[i]); strings.HasPrefix(lit, "-") && !strings.HasPrefix(lit, "--") && strings.Contains(lit, "s") {
			b.opaque = true
			return
		}
		if wordHasDynamicPart(args[i]) {
			b.opaque = true
			return
		}
	}
}

// shellAutoSourceEnvVars lists the environment variables whose non-empty
// value makes a shell interpreter source an extra file at startup
// ("BASH_ENV" for bash, "ENV" for ksh, "ZDOTDIR" relocating zsh's .zshenv).
var shellAutoSourceEnvVars = []string{"BASH_ENV", "ENV", "ZDOTDIR"}

// autoSourcesEnv reports whether a shell interpreter invocation would
// auto-source a startup file: one of [shellAutoSourceEnvVars] is set to a
// non-empty value in the process environment, by an in-command literal
// assignment, or dynamically (an "env(1)" word or a "read" target — a set
// but unknown value still sources something).
func (b *envBindings) autoSourcesEnv() bool {
	for _, name := range shellAutoSourceEnvVars {
		if os.Getenv(name) != "" {
			return true
		}
		if b != nil {
			if b.dynamic[name] {
				return true
			}
			for _, v := range b.values[name] {
				if v != "" {
					return true
				}
			}
		}
	}
	return false
}

// splitAssignWord splits a "NAME=VALUE" argument word (an env(1) or
// wrapped-declaration binding). ok reports whether lit carries a
// shell-variable name before the "=".
func splitAssignWord(lit string) (name, value string, ok bool) {
	eq := strings.IndexByte(lit, '=')
	if eq <= 0 {
		return "", "", false
	}
	name = lit[:eq]
	if !isShellVarName(name) {
		return "", "", false
	}
	return name, lit[eq+1:], true
}

// merge folds another binding summary (from a nested "bash -c '…'" parse)
// into this one: opaque and dynamic markings union, literal values append
// without duplicates.
func (b *envBindings) merge(o *envBindings) {
	if o.opaque {
		b.opaque = true
	}
	for name := range o.dynamic {
		b.dynamic[name] = true
	}
	for name, vals := range o.values {
		for _, v := range vals {
			if !slicesContains(b.values[name], v) {
				b.values[name] = append(b.values[name], v)
			}
		}
	}
}

// walkForClause folds for/select loops: the loop variable is rebound on each
// iteration to the word-list items — literals, but also globs, expansions or
// command substitutions — so it is statically unknown. A C-style loop
// ("for ((i=0; i<n; i++))") rebinds through arithmetic and is folded like an
// arithmetic command.
func (b *envBindings) walkForClause(n *syntax.ForClause) {
	switch loop := n.Loop.(type) {
	case *syntax.WordIter:
		if loop.Name != nil {
			b.dynamic[loop.Name.Value] = true
		}
	case *syntax.CStyleLoop:
		b.walkArithm(loop.Init)
		b.walkArithm(loop.Cond)
		b.walkArithm(loop.Post)
	}
}

// arithmAssignOps lists the arithmetic binary operators that REBIND their
// left operand ("=", "+=", "-=", …). Pure comparisons and arithmetic leave
// every variable unchanged and must not over-flag.
var arithmAssignOps = map[syntax.BinAritOperator]bool{
	syntax.Assgn:    true,
	syntax.AddAssgn: true,
	syntax.SubAssgn: true,
	syntax.MulAssgn: true,
	syntax.QuoAssgn: true,
	syntax.RemAssgn: true,
	syntax.AndAssgn: true,
	syntax.OrAssgn:  true,
	syntax.XorAssgn: true,
	syntax.ShlAssgn: true,
	syntax.ShrAssgn: true,
}

// walkArithm folds an arithmetic expression's assignments ("((D=5))",
// "((i++))", the clauses of a C-style for loop) into the summary: assigned
// names are rebound to values the walk cannot determine. When a rebinding
// operator targets something the walker cannot name, the whole command goes
// opaque (fail closed).
func (b *envBindings) walkArithm(e syntax.ArithmExpr) {
	if e == nil {
		return
	}
	names, anyAssign, allNamed := arithmAssignedNames(e)
	if anyAssign && !allNamed {
		b.opaque = true
		return
	}
	for _, name := range names {
		b.dynamic[name] = true
	}
}

// arithmAssignedNames collects the names rebound by assignment or
// increment/decrement operators anywhere in an arithmetic expression. It
// also reports whether any rebinding operator was seen at all, and whether
// every rebinding target could be named (an unnamed target means the walk
// cannot tell which variable changed).
func arithmAssignedNames(e syntax.ArithmExpr) (names []string, anyAssign, allNamed bool) {
	allNamed = true
	var visit func(e syntax.ArithmExpr)
	visit = func(e syntax.ArithmExpr) {
		switch x := e.(type) {
		case *syntax.BinaryArithm:
			if arithmAssignOps[x.Op] {
				anyAssign = true
				if n, ok := arithmOperandName(x.X); ok {
					names = append(names, n)
				} else {
					allNamed = false
				}
			}
			visit(x.X)
			visit(x.Y)
		case *syntax.UnaryArithm:
			if x.Op == syntax.Inc || x.Op == syntax.Dec {
				anyAssign = true
				if n, ok := arithmOperandName(x.X); ok {
					names = append(names, n)
				} else {
					allNamed = false
				}
			}
			visit(x.X)
		case *syntax.ParenArithm:
			visit(x.X)
		}
	}
	visit(e)
	return names, anyAssign, allNamed
}

// arithmOperandName extracts the variable name of a plain "$name" arithmetic
// operand — the left side of an assignment or the target of an increment.
func arithmOperandName(e syntax.ArithmExpr) (string, bool) {
	w, ok := e.(*syntax.Word)
	if !ok || len(w.Parts) != 1 {
		return "", false
	}
	pe, ok := w.Parts[0].(*syntax.ParamExp)
	if !ok || pe.Param == nil || paramExpHasModifier(pe) {
		return "", false
	}
	return pe.Param.Value, true
}

// isShellVarName reports whether s is a plain shell identifier — the shape
// of a variable NAME as opposed to a flag, an option argument, or a
// free-form word in a rebinding builtin's argument list.
func isShellVarName(s string) bool {
	if s == "" {
		return false
	}
	for i := 0; i < len(s); i++ {
		c := s[i]
		isName := (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
			(c >= '0' && c <= '9' && i > 0) || c == '_'
		if !isName {
			return false
		}
	}
	return true
}

// slicesContains reports whether s contains v.
func slicesContains(s []string, v string) bool {
	for _, x := range s {
		if x == v {
			return true
		}
	}
	return false
}

// resolvable reports whether name has exactly one assignment in the whole
// command and it is a purely literal value: only then does a reference to it
// statically expand to a known value. A nil receiver has no bindings, so no
// name is resolvable; an opaque command never resolves any name.
func (b *envBindings) resolvable(name string) bool {
	if b == nil || b.opaque {
		return false
	}
	return !b.dynamic[name] && len(b.values[name]) == 1
}

// unassessable reports whether a reference to name cannot be assessed at
// all: the name is rebound by a construct whose runtime value is unknown to
// the walk (dynamic), or the command contains an opaque construct that may
// have rebound anything. Callers escalate such references — see
// [UnresolvablePathTokens].
func (b *envBindings) unassessable(name string) bool {
	if b == nil {
		return false
	}
	return b.opaque || b.dynamic[name]
}

// emptyExpansionPossible reports whether name may expand to the EMPTY string
// at a reference: the variable is unset or empty in the process environment
// (the walk is position- and control-flow-unaware, so it cannot prove an
// in-command assignment is live at the reference — the assignment may sit in
// a branch not taken or after the reference), or the name is rebound by a
// construct that can clear or empty it ("unset", "read" on empty input,
// "source"/"eval", an array conversion, …). The empty expansion is a
// resolution candidate like any other: "$D/etc/passwd" with a possibly-unset
// D must surface "/etc/passwd" exactly like the unbound-variable fallback
// always did — a decoy in-command binding must not mask it.
func (b *envBindings) emptyExpansionPossible(name string) bool {
	if b != nil && (b.opaque || b.dynamic[name]) {
		return true
	}
	return os.Getenv(name) == ""
}

// finalValue returns the literal assigned to name by the command's LAST
// literal assignment (the pre-pass keeps first-seen order of distinct values,
// so the last element is the final distinct literal), or "" when the name was
// never assigned a literal. It is used as the best-effort path candidate for
// ambiguous names, whose references stay unexpandable regardless (see
// expandWordWithBindings).
func (b *envBindings) finalValue(name string) string {
	if b == nil {
		return ""
	}
	vals := b.values[name]
	if len(vals) == 0 {
		return ""
	}
	return vals[len(vals)-1]
}

// valueCandidates returns every possible runtime value of name in
// deterministic order: each distinct in-command literal binding (a re-bound
// or dynamically assigned name may hold any of them at a reference,
// position- and control-flow-unaware as the pre-pass is), the
// process-environment value, and — last — the EMPTY expansion when it is
// statically possible ([envBindings.emptyExpansionPossible]), all
// deduplicated. The empty string is a real candidate, never a skip: a
// possibly-unset "$D" concatenated with a path suffix expands to the suffix
// alone, and [resolveEnvToken] resolves that candidate exactly like the
// historical unbound-variable fallback. The result is therefore empty only
// when every source is absent: no bindings, an unset environment value, and
// an impossible empty expansion.
func (b *envBindings) valueCandidates(name string) []string {
	var vals []string
	add := func(v string) {
		if slicesContains(vals, v) {
			return
		}
		vals = append(vals, v)
	}
	if b != nil {
		for _, v := range b.values[name] {
			add(v)
		}
	}
	add(os.Getenv(name))
	if b.emptyExpansionPossible(name) {
		add("")
	}
	return vals
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

// wordSelfReferenceOnly reports whether every dynamic part of w is a plain
// reference to name itself — an append-style self-assignment
// ("PATH=$PATH:/usr/local/bin") whose runtime value is the prior value plus
// literal fragments. Such an assignment leaves the name assessable through
// the environment-value and empty-expansion unions instead of marking it
// dynamically rebound (which would hard-escalate every later "$PATH").
func wordSelfReferenceOnly(w *syntax.Word, name string) bool {
	self := true
	var check func(parts []syntax.WordPart)
	check = func(parts []syntax.WordPart) {
		for _, part := range parts {
			switch p := part.(type) {
			case *syntax.Lit:
				// Purely literal fragment.
			case *syntax.SglQuoted:
				if p.Dollar {
					self = false
				}
			case *syntax.DblQuoted:
				if p.Dollar {
					self = false
				}
				check(p.Parts)
			case *syntax.ParamExp:
				if paramExpHasModifier(p) || p.Param == nil || p.Param.Value != name {
					self = false
				}
			default:
				self = false
			}
		}
	}
	check(w.Parts)
	return self
}

// selfRefCarriesPathComponent reports whether a self-referencing RHS has a
// literal fragment that composes a PATH component beyond a ":"-separated
// pathlist append ("PATH=$PATH:/usr/local/bin" appends a pathlist entry;
// "D=$D/etc" builds an absolute path fragment no candidate list contains).
func selfRefCarriesPathComponent(w *syntax.Word) bool {
	var frags []string
	var collect func(parts []syntax.WordPart)
	collect = func(parts []syntax.WordPart) {
		for _, part := range parts {
			switch p := part.(type) {
			case *syntax.Lit:
				frags = append(frags, p.Value)
			case *syntax.SglQuoted:
				frags = append(frags, p.Value)
			case *syntax.DblQuoted:
				collect(p.Parts)
			}
		}
	}
	collect(w.Parts)
	for _, frag := range frags {
		if !strings.Contains(frag, "/") {
			continue
		}
		// A leading ':' marks a pathlist append (PATH=$PATH:/usr/local/bin);
		// anything else composes a path component.
		if !strings.HasPrefix(frag, ":") {
			return true
		}
	}
	return false
}

// composeSelfReferenceCandidates appends the values a pure self-reference
// RHS ("D=$D\"c\"") composes from the name's existing candidates and the
// RHS's literal fragments, so the union never loses an assembled runtime
// value: "D=/et; D=$D\"c\"" keeps "/etc" visible as a candidate. A
// ':'-prefixed pathlist append composes candidates too — harmless, since a
// pathlist-bearing reference already soft-escalates through its environment
// candidate.
func (b *envBindings) composeSelfReferenceCandidates(name string, w *syntax.Word) {
	var frags []string
	var collect func(parts []syntax.WordPart)
	collect = func(parts []syntax.WordPart) {
		for _, part := range parts {
			switch p := part.(type) {
			case *syntax.Lit:
				frags = append(frags, p.Value)
			case *syntax.SglQuoted:
				frags = append(frags, p.Value)
			case *syntax.DblQuoted:
				collect(p.Parts)
			}
		}
	}
	collect(w.Parts)
	current := b.valueCandidates(name)
	for _, frag := range frags {
		next := make([]string, 0, len(current))
		for _, v := range current {
			next = append(next, v+frag)
		}
		current = next
	}
	for _, v := range current {
		if v != "" && !slicesContains(b.values[name], v) {
			b.values[name] = append(b.values[name], v)
		}
	}
}

func wordPartsHaveDynamicPart(parts []syntax.WordPart) bool {
	for _, part := range parts {
		switch p := part.(type) {
		case *syntax.Lit:
			// Purely literal fragment.
		case *syntax.SglQuoted:
			// ANSI-C $'…' quoting decodes escape sequences (\xHH, \nnn, \e,
			// …) at RUNTIME, so the parser's raw Value is not the value the
			// shell expands — treat it as statically unknown.
			if p.Dollar {
				return true
			}
		case *syntax.DblQuoted:
			// $"…" is locale-translated at runtime; the plain "…" form is
			// static only if its inner parts are.
			if p.Dollar || wordPartsHaveDynamicPart(p.Parts) {
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
// the word still contains unexpandable/dynamic content. Only a name assigned
// exactly once in the whole command with a literal value — whose empty
// expansion is impossible and with no opaque rebinding in the command —
// resolves cleanly, and even then a binding never MASKS the process
// environment: because the binding pre-pass is position- and
// control-flow-unaware, a name that ALSO has a non-empty, differing os.Getenv
// value is ambiguous — the word is resolved to the bound literal AND marked
// unexpandable so the caller keeps the command suspicious (fail-closed
// over-report). A name whose EMPTY expansion is possible (unset or empty in
// the environment — the binding may sit in a branch not taken or after the
// reference — or dynamically rebound/cleared in-command) is ambiguous the
// same way. A re-bound or dynamically assigned name keeps the final literal
// as a best-effort path candidate while the word stays unexpandable. Unbound
// "$VAR", "$(cmd)", backticks, process/command substitution, and arithmetic
// are left unexpanded (the literal fragments are concatenated, matching the
// shell's empty-expansion of an unset variable) and the word is marked
// unexpandable so the caller can flag the command suspicious.
func expandWordWithBindings(w *syntax.Word, bindings *envBindings) (literal string, unexpandable bool) {
	if w == nil {
		return "", false
	}
	return expandWordPartsWithBindings(w.Parts, bindings)
}

func expandWordPartsWithBindings(parts []syntax.WordPart, bindings *envBindings) (string, bool) {
	var lit string
	unexp := false
	for _, part := range parts {
		switch p := part.(type) {
		case *syntax.Lit:
			lit += p.Value
		case *syntax.SglQuoted:
			// Plain '…' is a literal fragment. ANSI-C $'…' decodes escape
			// sequences at runtime, so the raw value is only a best-effort
			// candidate: keep it for path extraction, but the word stays
			// unexpandable (fail-closed).
			lit += p.Value
			if p.Dollar {
				unexp = true
			}
		case *syntax.DblQuoted:
			// $"…" is locale-translated at runtime — same fail-closed
			// treatment as ANSI-C quoting.
			inner, innerUnexp := expandWordPartsWithBindings(p.Parts, bindings)
			lit += inner
			if p.Dollar || innerUnexp {
				unexp = true
			}
		case *syntax.ParamExp:
			// Only the plain "$VAR"/"${VAR}" forms may resolve to a binding.
			// Every modifier form — length/width ("${#a}"), indirect
			// ("${!a}"), index/slice ("${a[i]}", "${a:x:y}"), replacement
			// ("${a/x/y}"), name listing ("${!prefix*}"), and the expansion
			// operators ("${a:-b}", "${a:+b}", …) — derives a DIFFERENT value
			// from the variable than the variable's own literal (e.g.
			// "${a:+b}" yields "b" when "a" is set, not a's value), so
			// substituting the binding here would resolve the word to a
			// confidently wrong, possibly benign-looking path. Fail closed:
			// contribute nothing to the literal and keep the word
			// unexpandable, mirroring the historical wordLiteral treatment of
			// parameter expansions.
			if paramExpHasModifier(p) {
				unexp = true
				continue
			}
			if p.Param != nil {
				name := p.Param.Value
				val := bindings.finalValue(name)
				if val != "" {
					// Resolve-and-stay-suspicious for ambiguous names: the
					// final literal is a possible runtime value, so keep it as
					// the path candidate.
					lit += val
				}
				if !bindings.resolvable(name) || bindings.emptyExpansionPossible(name) {
					unexp = true
					continue
				}
				// Union (fail-closed): the binding pre-pass cannot prove the
				// assignment is in effect where this reference executes, so a
				// differing non-empty environment value remains a possible
				// runtime expansion. Keep the word unexpandable so the
				// command stays suspicious.
				if envVal := os.Getenv(name); envVal != val {
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

// paramExpHasModifier reports whether a ParamExp is anything other than a
// plain "$VAR"/"${VAR}" reference — a length/width/indirect form, an indexed
// or sliced one, a replacement, a name listing, or an expansion operator. Such
// forms derive a value other than the variable's own value (see
// expandWordPartsWithBindings) and must stay unexpandable.
func paramExpHasModifier(p *syntax.ParamExp) bool {
	return p.Length || p.Width || p.Excl || p.Names != 0 ||
		p.Index != nil || p.Slice != nil || p.Repl != nil ||
		p.Exp != nil
}
