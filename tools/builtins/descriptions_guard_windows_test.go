//go:build windows

package builtins

import (
	"strings"
	"testing"

	sdktools "github.com/v0lka/sp4rk/tools"
)

// platformShellTools constructs the Windows shell tool for the description
// guard (posh_exec; excluded from AllBuiltinTools).
func platformShellTools() []sdktools.Tool {
	posh, err := NewPoshExecTool(nil)
	if err != nil {
		panic(err) // nil blacklist cannot fail to compile
	}
	return []sdktools.Tool{posh}
}

func TestPoshDescriptionWithinGuardLimit(t *testing.T) {
	poshTool, err := NewPoshExecTool(nil)
	if err != nil {
		t.Fatalf("NewPoshExecTool: %v", err)
	}
	desc := poshTool.Description()
	if strings.TrimSpace(desc) == "" {
		t.Errorf("tool %s: description is empty", poshTool.Name())
	}
	if len(desc) > maxDescriptionLength {
		t.Errorf("tool %s: description is %d chars, exceeds guard limit %d", poshTool.Name(), len(desc), maxDescriptionLength)
	}
}
