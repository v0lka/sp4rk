package llm

import (
	"context"
	"reflect"
	"testing"
	"time"
)

// newProviderForModelTestRouter builds a real Router (via NewRouter, so the
// composite→provider reverse index is populated) with two providers whose
// model lists deliberately differ from what callers may pin.
func newProviderForModelTestRouter(t *testing.T) *Router {
	t.Helper()
	cfg := RouterConfig{
		Providers: []ProviderEntry{
			{Name: "alpha", ProviderType: "openai", BaseURL: "http://localhost:9999", Models: []string{"shared-model", "alpha-only"}},
			{Name: "beta", ProviderType: "openai", BaseURL: "http://localhost:9998", Models: []string{"shared-model", "beta-only"}},
		},
		MaxRetries:     1,
		InitialBackoff: 10 * time.Millisecond,
		MaxBackoff:     20 * time.Millisecond,
	}
	router, err := NewRouter(context.Background(), cfg, nil)
	if err != nil {
		t.Fatalf("NewRouter: %v", err)
	}
	return router
}

// TestRouter_ProviderForModel_Composite verifies the composite branch: ok
// tracks the provider NAME only — the bare model need not appear in the
// provider's model list, because per-request provider routing (req.Provider)
// pins the provider name only (ADR-012).
func TestRouter_ProviderForModel_Composite(t *testing.T) {
	router := newProviderForModelTestRouter(t)

	tests := []struct {
		id           string
		wantProvider string
		wantOK       bool
	}{
		{"alpha/alpha-only", "alpha", true},
		// Known provider, model NOT in its Models list — still routes: the
		// provider name is what Call validates.
		{"alpha/unlisted-model", "alpha", true},
		{"beta/beta-only", "beta", true},
		// Unknown provider: fail.
		{"ghost/anything", "ghost", false},
	}
	for _, tt := range tests {
		provider, ok := router.ProviderForModel(tt.id)
		if provider != tt.wantProvider || ok != tt.wantOK {
			t.Errorf("ProviderForModel(%q) = (%q, %v), want (%q, %v)",
				tt.id, provider, ok, tt.wantProvider, tt.wantOK)
		}
	}
}

// TestRouter_ProviderForModel_Bare verifies the bare branch resolves like
// SetModel: the first sorted composite ID whose bare model matches; a model
// no provider lists fails.
func TestRouter_ProviderForModel_Bare(t *testing.T) {
	router := newProviderForModelTestRouter(t)

	// Ambiguous bare name: both providers list "shared-model"; the sorted
	// first composite ID is "alpha/shared-model" → provider alpha.
	provider, ok := router.ProviderForModel("shared-model")
	if !ok || provider != "alpha" {
		t.Errorf("ProviderForModel(%q) = (%q, %v), want (alpha, true)", "shared-model", provider, ok)
	}

	// Unambiguous bare name.
	provider, ok = router.ProviderForModel("beta-only")
	if !ok || provider != "beta" {
		t.Errorf("ProviderForModel(%q) = (%q, %v), want (beta, true)", "beta-only", provider, ok)
	}

	// Unknown bare name fails.
	provider, ok = router.ProviderForModel("no-such-model")
	if ok || provider != "" {
		t.Errorf("ProviderForModel(%q) = (%q, %v), want (\"\", false)", "no-such-model", provider, ok)
	}
}

// TestRouter_ProviderForModel_ReadOnly verifies the lookup leaves routing
// state untouched (contrast with SetModel, which switches the active
// provider/model).
func TestRouter_ProviderForModel_ReadOnly(t *testing.T) {
	router := newProviderForModelTestRouter(t)

	before := router.ActiveModel()
	beforeProvider := router.ActiveProviderName()

	_, _ = router.ProviderForModel("beta/beta-only")
	_, _ = router.ProviderForModel("shared-model")

	if got := router.ActiveModel(); got != before {
		t.Errorf("ActiveModel changed: %q → %q (ProviderForModel must be read-only)", before, got)
	}
	if got := router.ActiveProviderName(); got != beforeProvider {
		t.Errorf("ActiveProviderName changed: %q → %q (ProviderForModel must be read-only)", beforeProvider, got)
	}
}

// TestRouter_KnownProviders verifies the sorted configured-provider snapshot.
func TestRouter_KnownProviders(t *testing.T) {
	router := newProviderForModelTestRouter(t)

	got := router.KnownProviders()
	if !reflect.DeepEqual(got, []string{"alpha", "beta"}) {
		t.Errorf("KnownProviders() = %v, want [alpha beta]", got)
	}
}
