// SPDX-License-Identifier: Apache-2.0

package tools

import "testing"

// TestAnalyzeShellCommandForJudge_UnboundedEgressWithoutCradleIsNonCanonical
// pins the deliberate v3 C5/C6 split: an unbounded (⊤/conservative) command
// that carries a concrete network egress but establishes NO download-cradle
// flow no longer fires canonical C5 — it fires non-canonical hard C6, which a
// strict judge may positively clear. This is a behavior change from the
// pre-v3 rule, which treated "unbounded + concrete egress" as canonical C5.
func TestAnalyzeShellCommandForJudge_UnboundedEgressWithoutCradleIsNonCanonical(t *testing.T) {
	ctx := shellCorpusCtx(t)
	got, err := AnalyzeShellCommandForJudge(ctx, "bash_exec", shellCorpusInput(t, `eval "$CODE"; curl https://example.com/report`))
	if err != nil {
		t.Fatalf("AnalyzeShellCommandForJudge: %v", err)
	}
	if got.Outcome.Allow {
		t.Fatalf("unbounded command with egress must still escalate (criteria: %+v)", got.Digest.Criteria)
	}
	if got.Outcome.ReasonCode != ReasonCodeCommandUnboundedAnalysis {
		t.Fatalf("winner = %q, want %q (criteria: %+v)", got.Outcome.ReasonCode, ReasonCodeCommandUnboundedAnalysis, got.Digest.Criteria)
	}
	if got.Outcome.Severity != JudgeSeverityHard || got.Canonical {
		t.Errorf("C6 must be hard and non-canonical, got severity=%v canonical=%v", got.Outcome.Severity, got.Canonical)
	}
	for _, c := range got.Digest.Criteria {
		if c.Fired == ReasonCodeCommandDownloadCradle {
			t.Errorf("no cradle flow was established, yet C5 fired: %+v", got.Digest.Criteria)
		}
	}
}
