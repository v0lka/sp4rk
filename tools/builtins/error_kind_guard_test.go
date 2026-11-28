package builtins

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/printer"
	"go/token"
	"io/fs"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// TestNoArgumentContractOnPlainErrorResult is the builtins' half of the typed
// error-class contract (c0wrk ADR-137; specs/contracts/tools.md): a builtin's
// OWN argument-contract refusal — a required parameter missing, a cross-field
// rule violated, a value out of the allowed range — must be reported through
// tools.InvalidCallResult (ErrorKind invalid_call), never through a plain
// tools.ErrorResult or a bare ToolResult{IsError: true}, which keep the zero
// kind so the host would render the legacy "the tool ran and failed" glyph and
// never count the call in its invalid-call metrics.
//
// Scope, exactly: a string literal (or a fmt.Sprintf over one) passed to
// ErrorResult, or set as the Content of a ToolResult composite with
// IsError: true, that carries an argument-contract phrase ("required",
// "must be", "must not", "cannot be empty", "invalid argument"). Runtime and
// infrastructure failures ("store not available", "no output found", a
// wrapped %v error) do not match and stay legitimate ErrorResult sites. The
// scan is AST-based over every non-test file of this package and its
// subpackages, so comments and literals that merely mention the phrase are
// ignored. It mirrors c0wrk's core/tools/error_kind_test.go guard; the
// sibling subpackages (websearch, …) are covered by the same walk.
func TestNoArgumentContractOnPlainErrorResult(t *testing.T) {
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed; cannot locate package directory")
	}
	pkgDir := filepath.Dir(thisFile)
	fset := token.NewFileSet()
	scanned := 0
	err := filepath.WalkDir(pkgDir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		name := d.Name()
		if d.IsDir() {
			// Test fixtures / vendored helpers are not the package's tools.
			if name == "testdata" || name == "internal" || strings.HasPrefix(name, ".") {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			return nil
		}
		file, err := parser.ParseFile(fset, path, nil, 0)
		if err != nil {
			return fmt.Errorf("parse %s: %w", path, err)
		}
		scanned++
		ast.Inspect(file, func(n ast.Node) bool {
			switch node := n.(type) {
			case *ast.CallExpr:
				sel, isSel := node.Fun.(*ast.SelectorExpr)
				if isSel && sel.Sel.Name == "ErrorResult" && len(node.Args) > 0 && isArgumentContractLiteral(node.Args[0]) {
					t.Errorf("%s: plain ErrorResult carries an argument-contract refusal (use tools.InvalidCallResult): %s",
						fset.Position(node.Pos()), renderNode(fset, node))
				}
			case *ast.CompositeLit:
				if !isToolResultType(node.Type) {
					return true
				}
				var content string
				isError := false
				for _, elt := range node.Elts {
					kv, ok := elt.(*ast.KeyValueExpr)
					if !ok {
						continue
					}
					key, ok := kv.Key.(*ast.Ident)
					if !ok {
						continue
					}
					switch key.Name {
					case "Content":
						if lit, ok := literalString(kv.Value); ok {
							content = lit
						}
					case "IsError":
						if id, ok := kv.Value.(*ast.Ident); ok && id.Name == "true" {
							isError = true
						}
					}
				}
				if isError && isArgumentContractPhrase(content) {
					t.Errorf("%s: bare ToolResult{IsError: true} carries an argument-contract refusal (use tools.InvalidCallResult): %s",
						fset.Position(node.Pos()), renderNode(fset, node))
				}
			}
			return true
		})
		return nil
	})
	if err != nil {
		t.Fatalf("walk package: %v", err)
	}
	if scanned == 0 {
		t.Fatal("no source files scanned; the guard is not covering the package")
	}
}

// isToolResultType matches `ToolResult{…}` and `tools.ToolResult{…}`.
func isToolResultType(expr ast.Expr) bool {
	switch t := expr.(type) {
	case *ast.Ident:
		return t.Name == "ToolResult"
	case *ast.SelectorExpr:
		return t.Sel.Name == "ToolResult"
	}
	return false
}

// contractPhrases are the shapes a builtin's own argument refusal takes.
// Deliberately narrow: infrastructure prose ("store not available", "no
// output found") never matches.
var contractPhrases = []string{"required", "must be", "must not", "cannot be empty", "invalid argument", "invalid url", "invalid timeout", "only http and https", "is not a directory", "is a directory, use", "is not unique", "not found in file"}

// isArgumentContractPhrase reports whether a refusal message reads as an
// argument-contract failure.
func isArgumentContractPhrase(msg string) bool {
	lower := strings.ToLower(strings.TrimSpace(msg))
	if lower == "" {
		return false
	}
	for _, p := range contractPhrases {
		if strings.Contains(lower, p) {
			return true
		}
	}
	return false
}

// isArgumentContractLiteral reports whether expr is a string literal, or a
// fmt.Sprintf whose format is a string literal, carrying an argument-contract
// refusal.
func isArgumentContractLiteral(expr ast.Expr) bool {
	msg, ok := literalString(expr)
	if !ok {
		return false
	}
	return isArgumentContractPhrase(msg)
}

// literalString extracts the static text of a string literal, of a
// fmt.Sprintf(...)/fmt.Errorf(...) whose first argument is a string literal,
// or of an ALL-STATIC concatenation of such operands (any dynamic operand
// disables the check for that expression, since the refusal text is not
// statically known).
func literalString(expr ast.Expr) (string, bool) {
	switch e := expr.(type) {
	case *ast.BasicLit:
		if e.Kind == token.STRING {
			return strings.Trim(e.Value, "`\""), true
		}
	case *ast.BinaryExpr:
		lx, ok := literalString(e.X)
		if !ok {
			return "", false
		}
		ry, ok := literalString(e.Y)
		if !ok {
			return "", false
		}
		return lx + ry, true
	case *ast.CallExpr:
		sel, ok := e.Fun.(*ast.SelectorExpr)
		if !ok || len(e.Args) == 0 {
			return "", false
		}
		if sel.Sel.Name != "Sprintf" && sel.Sel.Name != "Errorf" {
			return "", false
		}
		if lit, ok := e.Args[0].(*ast.BasicLit); ok && lit.Kind == token.STRING {
			return strings.Trim(lit.Value, "`\""), true
		}
	}
	return "", false
}

// renderNode renders a node for the failure message.
func renderNode(fset *token.FileSet, n ast.Node) string {
	var sb strings.Builder
	_ = printer.Fprint(&sb, fset, n)
	return sb.String()
}
