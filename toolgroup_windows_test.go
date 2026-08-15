//go:build windows

package sp4rk

import (
	"github.com/v0lka/sp4rk/tools"
	"github.com/v0lka/sp4rk/tools/builtins"
)

// shellToolExpectations pins the Windows shell tool's group.
var shellToolExpectations = map[string]tools.ToolGroup{
	"posh_exec": tools.GroupExecute,
}

// platformShellTools constructs the Windows shell tool (parameterized;
// excluded from AllBuiltinTools).
func platformShellTools() []tools.Tool {
	posh, err := builtins.NewPoshExecTool(nil)
	if err != nil {
		panic(err) // nil blacklist cannot fail to compile
	}
	return []tools.Tool{posh}
}
