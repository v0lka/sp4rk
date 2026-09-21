// SPDX-License-Identifier: Apache-2.0

package tools

import "testing"

// TestIsCanonicalReasonCode pins the canonical/canonicality classification a
// host keys deterministic policy off: canonical codes are the hard fired
// controls (or structurally unassessable inputs) a host must never
// auto-override, whereas the two hard-but-clearable shell-analysis codes and
// the soft scope codes are not canonical.
func TestIsCanonicalReasonCode(t *testing.T) {
	canonical := []JudgeReasonCode{
		ReasonCodeCommandBlacklist,
		ReasonCodeCommandExfilFlow,
		ReasonCodeCommandPrivilegeEscalation,
		ReasonCodeCommandSystemWrite,
		ReasonCodeCommandDestructiveOutsideRoots,
		ReasonCodeCommandDownloadCradle,
		ReasonCodeCommandAnalysisUnavailable,
		ReasonCodeSSRFPrivateAddress,
		ReasonCodeSSRFDegraded,
		ReasonCodeUnassessableURL,
		ReasonCodeUnassessablePath,
		ReasonCodeSymlinkEscape,
		ReasonCodeGitInternal,
	}
	for _, code := range canonical {
		if !IsCanonicalReasonCode(code) {
			t.Errorf("IsCanonicalReasonCode(%q) = false, want true", code)
		}
	}

	notCanonical := []JudgeReasonCode{
		ReasonCodeCommandUnboundedAnalysis,     // hard but clearable (⊤ limitation)
		ReasonCodeCommandExternalContentIngest, // hard but clearable (ingest flow)
		ReasonCodeOutsideSessionRoots,          // soft scope question
		ReasonCodeCredentialAccess,             // soft scope question
		ReasonCodeUnresolvablePathToken,        // retained, no longer fired
		ReasonCodeSymlinkSuspicious,            // retained, no longer fired
		"",
		"unknown_code",
	}
	for _, code := range notCanonical {
		if IsCanonicalReasonCode(code) {
			t.Errorf("IsCanonicalReasonCode(%q) = true, want false", code)
		}
	}
}
