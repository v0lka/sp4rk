package mcp

// Invariant guard for the origin-pinned redirect policy (GO-2026-6911 — the
// CVE-2026-116685 class).
//
// The library default is unsafe by construction: transport.NewStreamableHTTP
// builds an http.Client with a nil CheckRedirect, so a 3xx Location is followed
// to any host and scheme, and the RoundTripper layer re-attaches the operator's
// custom headers (and, on 307/308, the JSON-RPC body) to the redirected
// request. redirect.go closes that class by installing originPinnedRedirects on
// every client this package hands to an mcp-go transport; redirect_test.go
// proves the resulting behaviour.
//
// This test guards the CONTROL POINT rather than the behaviour: every
// construction site of an mcp-go HTTP transport in this package must live in a
// function that installs the pinned client, so a future connect path cannot
// quietly re-enter the library default. When the construction site count drops
// to zero the test fails as well — the package would have stopped connecting
// over HTTP entirely, and that is a reviewable change, not a silent pass.
//
// The check is deliberately syntactic: it asserts that the enclosing function
// calls originPinnedClient. That the returned client actually reaches the
// constructor is a data flow the behaviour tests pin — they fail when the
// client handed to the transport is unpinned.

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"strings"
	"testing"
)

// pinnedRedirectConstructors are the mcp-go entry points that accept an
// http.Client (directly, or through an option this package supplies) and would
// otherwise fall back to the library's unpinned default. Adding a transport to
// this package means adding its constructor here.
var pinnedRedirectConstructors = map[string]bool{
	"NewStreamableHttpClient":      true,
	"NewSSEMCPClient":              true,
	"NewOAuthStreamableHttpClient": true,
	"NewOAuthSSEClient":            true,
	"NewStreamableHTTP":            true,
	"NewSSE":                       true,
}

// pinningControl is the function that installs the redirect policy. The
// syntactic guard looks for this call in the enclosing function.
const pinningControl = "originPinnedClient"

// TestMCPTransportConstructorsInstallOriginPinnedClient fails when an mcp-go
// HTTP transport is constructed by a function that does not install the
// origin-pinned redirect client.
func TestMCPTransportConstructorsInstallOriginPinnedClient(t *testing.T) {
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatalf("reading package directory = %v", err)
	}

	fset := token.NewFileSet()
	sites := 0
	for _, entry := range entries {
		name := entry.Name()
		if entry.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		file, err := parser.ParseFile(fset, name, nil, 0)
		if err != nil {
			t.Fatalf("parsing %s = %v", name, err)
		}
		all, inFuncs := transportSites(fset, file)
		if len(all) != len(inFuncs) {
			t.Errorf("%s: %d transport construction site(s) but %d inside a function declaration; a construction outside a function cannot be checked for the redirect policy",
				name, len(all), len(inFuncs))
		}
		for _, s := range inFuncs {
			if !s.pinned {
				t.Errorf("%s: %s constructs an mcp-go HTTP transport without installing the origin-pinned redirect client; call %s(client, url) and pass the pinned copy (GO-2026-6911: an unpinned client follows a 3xx Location to any host with the operator's headers attached)",
					s.pos, s.fn, pinningControl)
			}
		}
		sites += len(inFuncs)
	}

	if sites == 0 {
		t.Fatalf("found no mcp-go HTTP transport construction site in package %s; the redirect-policy guard would pass vacuously — review where the HTTP transports are built now", "mcp")
	}
}

// site is one transport construction site: where it is and whether the
// enclosing function installs the redirect policy.
type site struct {
	pos    token.Position
	fn     string
	pinned bool
}

// transportSites returns every mcp-go HTTP transport construction call in the
// file, plus those that could be attributed to an enclosing function
// declaration together with that function's pinning status.
func transportSites(fset *token.FileSet, file *ast.File) ([]token.Position, []site) {
	var all []token.Position
	ast.Inspect(file, func(n ast.Node) bool {
		if call, ok := n.(*ast.CallExpr); ok && isTransportConstructor(call) {
			all = append(all, fset.Position(call.Pos()))
		}
		return true
	})

	var inFuncs []site
	for _, decl := range file.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || fn.Body == nil {
			continue
		}
		pinned := callsFunc(fn.Body, pinningControl)
		ast.Inspect(fn.Body, func(n ast.Node) bool {
			if call, ok := n.(*ast.CallExpr); ok && isTransportConstructor(call) {
				inFuncs = append(inFuncs, site{pos: fset.Position(call.Pos()), fn: funcName(fn), pinned: pinned})
			}
			return true
		})
	}
	return all, inFuncs
}

// isTransportConstructor reports whether the call invokes one of the guarded
// mcp-go constructors, matching on the selector name so an import alias does
// not defeat the check.
func isTransportConstructor(call *ast.CallExpr) bool {
	sel, ok := call.Fun.(*ast.SelectorExpr)
	return ok && pinnedRedirectConstructors[sel.Sel.Name]
}

// callsFunc reports whether the subtree contains a call to a function with the
// given name (again by selector name, so the import alias does not matter).
func callsFunc(node ast.Node, name string) bool {
	found := false
	ast.Inspect(node, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		if sel, ok := call.Fun.(*ast.SelectorExpr); ok && sel.Sel.Name == name {
			found = true
			return false
		}
		if ident, ok := call.Fun.(*ast.Ident); ok && ident.Name == name {
			found = true
			return false
		}
		return true
	})
	return found
}

// funcName renders a function declaration for failure messages: the plain name
// for a function, "(recv.T).name" for a method.
func funcName(fn *ast.FuncDecl) string {
	if fn.Recv == nil || len(fn.Recv.List) == 0 {
		return fn.Name.Name
	}
	recv := fn.Recv.List[0].Type
	if star, ok := recv.(*ast.StarExpr); ok {
		recv = star.X
	}
	ident, ok := recv.(*ast.Ident)
	if !ok {
		return "(*)." + fn.Name.Name
	}
	return ident.Name + "." + fn.Name.Name
}
