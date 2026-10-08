package builtins

import (
	"net/http"
	"testing"
	"time"
)

// TestWebFetchTool_ZeroLimitsUseDefaults is the regression test for review
// finding 6: NewWebFetchTool(WebFetchLimits{{}}) used to produce
// &http.Client{{Timeout: 0}} — no timeout at all — so an origin that accepted
// the connection and stalled hung the tool run indefinitely. The zero-value
// limits must be repaired to the defaults.
func TestWebFetchTool_ZeroLimitsUseDefaults(t *testing.T) {
	tool := NewWebFetchTool(WebFetchLimits{})

	if tool.limits.Timeout != DefaultWebFetchLimits().Timeout {
		t.Errorf("normalized limits.Timeout = %v, want %v", tool.limits.Timeout, DefaultWebFetchLimits().Timeout)
	}
	if tool.client.Timeout != DefaultWebFetchLimits().Timeout {
		t.Errorf("client.Timeout = %v, want %v (bounded)", tool.client.Timeout, DefaultWebFetchLimits().Timeout)
	}
}

// TestWebFetchTool_ZeroLimitFloorsCallerClient verifies that a caller-supplied
// client with no timeout is floored at the configured timeout on the tool's
// copy, and that the caller's client is not mutated.
func TestWebFetchTool_ZeroLimitFloorsCallerClient(t *testing.T) {
	caller := &http.Client{} // Timeout: 0 — would hang on a stalled origin

	tool := NewWebFetchToolWithClient(WebFetchLimits{}, caller)

	if tool.client == caller {
		t.Fatal("tool must use a copy of the caller's client, not the caller's client itself")
	}
	if tool.client.Timeout != DefaultWebFetchLimits().Timeout {
		t.Errorf("tool client.Timeout = %v, want %v", tool.client.Timeout, DefaultWebFetchLimits().Timeout)
	}
	if caller.Timeout != 0 {
		t.Errorf("caller client was mutated: Timeout = %v, want 0", caller.Timeout)
	}
}

// TestWebFetchTool_ExplicitTimeoutsPreserved pins the non-regression side of
// the repair: positive limits and positive caller timeouts must pass through
// unchanged.
func TestWebFetchTool_ExplicitTimeoutsPreserved(t *testing.T) {
	caller := &http.Client{Timeout: 99 * time.Second}

	tool := NewWebFetchToolWithClient(WebFetchLimits{Timeout: 7 * time.Second}, caller)

	if tool.limits.Timeout != 7*time.Second {
		t.Errorf("limits.Timeout = %v, want 7s (explicit config preserved)", tool.limits.Timeout)
	}
	if tool.client.Timeout != 99*time.Second {
		t.Errorf("tool client.Timeout = %v, want 99s (explicit caller timeout preserved)", tool.client.Timeout)
	}
}

// TestWebFetchTool_NegativeTimeoutUseDefaults covers the "non-positive"
// half of the rule: any Timeout <= 0 is replaced, not just the zero value.
func TestWebFetchTool_NegativeTimeoutUseDefaults(t *testing.T) {
	tool := NewWebFetchTool(WebFetchLimits{Timeout: -1 * time.Second})

	if tool.limits.Timeout != DefaultWebFetchLimits().Timeout {
		t.Errorf("normalized limits.Timeout = %v, want %v", tool.limits.Timeout, DefaultWebFetchLimits().Timeout)
	}
}
