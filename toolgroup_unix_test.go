//go:build !windows

package sp4rk

import (
	"github.com/v0lka/sp4rk/tools"
	"github.com/v0lka/sp4rk/tools/builtins"
)

// shellToolExpectations pins the Unix shell tool's group.
var shellToolExpectations = map[string]tools.ToolGroup{
	"bash_exec": tools.GroupExecute,
}

// platformShellTools constructs the Unix shell tool (parameterized; excluded
// from AllBuiltinTools).
func platformShellTools() []tools.Tool {
	bash, err := builtins.NewBashExecTool(nil)
	if err != nil {
		panic(err) // nil blacklist cannot fail to compile
	}
	return []tools.Tool{bash}
}
