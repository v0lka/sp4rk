// SPDX-License-Identifier: Apache-2.0

package tools

import (
	"testing"

	"github.com/v0lka/flowsh/api"
)

// stubShellTool implements Tool + DeclaredShellKind for dialect resolution
// tests without depending on the builtins package.
type stubShellTool struct {
	*BaseTool
	kind ShellKind
}

func (s *stubShellTool) DeclaredShellKind() ShellKind { return s.kind }

// TestAnalyzeShellCommandForJudgeWithDialect covers the explicit-dialect
// entry point hosts use when the shell-exec tool carries an
// operator-configured invocation override: the declared kind decides the
// dialect, not the tool name.
func TestAnalyzeShellCommandForJudgeWithDialect(t *testing.T) {
	ctx := shellCorpusCtx(t)

	// A PowerShell-syntax command analyzed with the PowerShell dialect
	// (the posh_exec legacy mapping) must produce the same verdict when
	// analyzed through the explicit entry point.
	psInput := shellCorpusInput(t, "Remove-Item -Recurse -Force C:\\Windows\\System32")
	legacy, err := AnalyzeShellCommandForJudge(ctx, "posh_exec", psInput)
	if err != nil {
		t.Fatalf("legacy analysis: %v", err)
	}
	explicit, err := AnalyzeShellCommandForJudgeWithDialect(ctx, api.LangPowerShell, psInput)
	if err != nil {
		t.Fatalf("explicit dialect analysis: %v", err)
	}
	if explicit.Digest.Lang != legacy.Digest.Lang {
		t.Errorf("explicit dialect %q != legacy %q", explicit.Digest.Lang, legacy.Digest.Lang)
	}

	// A zsh-declared override analyzes with the bash dialect — zsh belongs
	// to the bash family even though no tool name maps there natively.
	bashInput := shellCorpusInput(t, "echo hi")
	zsh, err := AnalyzeShellCommandForJudgeWithDialect(ctx, api.LangBash, bashInput)
	if err != nil {
		t.Fatalf("bash dialect analysis: %v", err)
	}
	if zsh.Digest.Lang != "bash" {
		t.Errorf("digest lang = %q, want bash", zsh.Digest.Lang)
	}

	// An unknown dialect is an error — callers fail closed.
	if _, err := AnalyzeShellCommandForJudgeWithDialect(ctx, api.Lang("fish"), bashInput); err == nil {
		t.Error("expected error for unsupported dialect, got nil")
	}
}

// TestShellAnalysisLangForTool covers dialect resolution for a registered
// tool instance: the declared kind wins when the tool exposes one, the
// legacy name mapping applies otherwise, and non-shell tools report false.
func TestShellAnalysisLangForTool(t *testing.T) {
	zshTool := &stubShellTool{BaseTool: &BaseTool{ToolName: "bash_exec"}, kind: ShellKindZsh}
	if lang, ok := ShellAnalysisLangForTool(zshTool, "bash_exec"); !ok || lang != api.LangBash {
		t.Errorf("declared zsh → %q,%v, want bash,true", lang, ok)
	}

	pwshTool := &stubShellTool{BaseTool: &BaseTool{ToolName: "bash_exec"}, kind: ShellKindPwsh}
	if lang, ok := ShellAnalysisLangForTool(pwshTool, "bash_exec"); !ok || lang != api.LangPowerShell {
		t.Errorf("declared pwsh → %q,%v, want powershell,true (declared kind must override the tool-name mapping)", lang, ok)
	}

	plainTool := &BaseTool{ToolName: "bash_exec"} // no DeclaredShellKind
	if lang, ok := ShellAnalysisLangForTool(plainTool, "bash_exec"); !ok || lang != api.LangBash {
		t.Errorf("legacy mapping → %q,%v, want bash,true", lang, ok)
	}

	if _, ok := ShellAnalysisLangForTool(plainTool, "read_file"); ok {
		t.Error("non-shell tool must report ok=false")
	}
}
