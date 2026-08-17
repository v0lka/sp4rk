//go:build !windows

package builtins

import (
	sdktools "github.com/v0lka/sp4rk/tools"
)

// platformShellTools constructs the Unix shell tool for the description
// guard (bash_exec; excluded from AllBuiltinTools).
func platformShellTools() []sdktools.Tool {
	bash, err := NewBashExecTool(nil)
	if err != nil {
		panic(err) // nil blacklist cannot fail to compile
	}
	return []sdktools.Tool{bash}
}
