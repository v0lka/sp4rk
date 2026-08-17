//go:build windows

package builtins

import (
	"strings"
	"testing"
)

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
