package llm

import (
	"testing"
	"time"
)

// TestModelRegistry_SetRuntimeMetadata_PartialRuntimeInheritsCatalogFamily is
// the regression for the runtime tier mirroring
// TestModelRegistry_PartialOverrideInheritsCatalogFamily: a server probe that
// observed only the context window must not shadow the built-in catalog's
// authoritative Family. Pre-fix the setter eagerly filled Family with
// resolveFamily's name-only guess — "default" for both models below, since
// their IDs carry no family token — and that non-empty guess blocked
// enrichPartialWith's inheritance, silently stripping the family-gated
// sampling and reasoning adaptations.
func TestModelRegistry_SetRuntimeMetadata_PartialRuntimeInheritsCatalogFamily(t *testing.T) {
	tests := []struct {
		name       string
		model      string
		observed   ModelMetadata
		wantFamily string
		wantWindow int
		wantLimit  int
	}{
		{
			// "k3" (Kimi Code short id) → catalog Family "kimi"; the ID
			// carries no family token, so DetectFamily would guess "default".
			name:       "k3 keeps catalog family kimi",
			model:      "k3",
			observed:   ModelMetadata{ContextWindow: 262144},
			wantFamily: "kimi",
			wantWindow: 262144,
			wantLimit:  131072,
		},
		{
			// "Bonsai 2 27B" → catalog Family "qwen" (a Qwen3.8-dense
			// checkpoint whose name carries no family token either).
			name:       "Bonsai 2 27B keeps catalog family qwen",
			model:      "Bonsai 2 27B",
			observed:   ModelMetadata{ContextWindow: 32768},
			wantFamily: "qwen",
			wantWindow: 32768,
			wantLimit:  32768,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			registry := NewModelRegistry(nil)
			registry.SetRuntimeMetadata(tt.model, tt.observed)

			meta, ok := registry.ResolveLocal(tt.model)
			if !ok {
				t.Fatalf("ResolveLocal(%q) not found", tt.model)
			}
			if meta.Family != tt.wantFamily {
				t.Errorf("Family = %q, want %q (catalog family must survive a partial runtime entry)", meta.Family, tt.wantFamily)
			}
			if meta.ContextWindow != tt.wantWindow {
				t.Errorf("ContextWindow = %d, want the observed %d", meta.ContextWindow, tt.wantWindow)
			}
			if meta.OutputLimit != tt.wantLimit {
				t.Errorf("OutputLimit = %d, want %d (inherited from the catalog)", meta.OutputLimit, tt.wantLimit)
			}
		})
	}
}

// TestModelRegistry_SetRuntimeMetadata_ExplicitFamilyStaysAuthoritative pins
// the other boundary of the raw-storage contract: an entry that DOES name a
// family keeps it (the runtime tier outranks the catalog), and the observed
// window still wins over the catalog spec.
func TestModelRegistry_SetRuntimeMetadata_ExplicitFamilyStaysAuthoritative(t *testing.T) {
	registry := NewModelRegistry(nil)
	registry.SetRuntimeMetadata("k3", ModelMetadata{ContextWindow: 8192, Family: "custom"})

	meta, ok := registry.ResolveLocal("k3")
	if !ok {
		t.Fatal("ResolveLocal(k3) not found")
	}
	if meta.Family != "custom" {
		t.Errorf("Family = %q, want the explicitly observed %q", meta.Family, "custom")
	}
	if meta.ContextWindow != 8192 {
		t.Errorf("ContextWindow = %d, want the observed 8192", meta.ContextWindow)
	}
}

// TestModelRegistry_SetRuntimeMetadata_RawEntryReportsUnobservedFieldsZero
// checks the RuntimeMetadata side of the contract: the entry is returned AS
// STORED, un-enriched — a probe that never observed a Family must not be
// reported one (pre-fix the setter's eager resolveFamily filled it in).
func TestModelRegistry_SetRuntimeMetadata_RawEntryReportsUnobservedFieldsZero(t *testing.T) {
	registry := NewModelRegistry(nil)
	registry.SetRuntimeMetadata("k3", ModelMetadata{ContextWindow: 262144})

	raw, ok := registry.RuntimeMetadata("k3")
	if !ok {
		t.Fatal("RuntimeMetadata(k3) not found")
	}
	if raw.Family != "" {
		t.Errorf("raw entry Family = %q, want empty (RuntimeMetadata is documented as stored, un-enriched)", raw.Family)
	}
}

// TestModelRegistry_Invalidate_SweepsNormalizedCacheTwins is the regression
// for Invalidate vs the fuzzy tier's resolved twins: ResolveLocal under a
// drifted spelling caches the result under the QUERY spelling
// (cacheResolved), and a model switch must sweep every spelling that
// normalizes to the same model — not just the exact key — or a post-switch
// resolve under the drifted spelling serves the previous serving
// arrangement's window from the stale twin.
func TestModelRegistry_Invalidate_SweepsNormalizedCacheTwins(t *testing.T) {
	registry := NewModelRegistry(nil)

	// A runtime-only model (no built-in catalog entry): the fuzzy tier is the
	// only way the drifted spelling resolves — and it caches the twin.
	registry.SetRuntimeMetadata("my-llama-3", ModelMetadata{ContextWindow: 8192})

	if meta, ok := registry.ResolveLocal("myllama3"); !ok || meta.ContextWindow != 8192 {
		t.Fatalf("pre-condition: drifted-spelling resolve = (%+v, %v), want the runtime window 8192", meta, ok)
	}

	registry.Invalidate("my-llama-3")

	if meta, ok := registry.ResolveLocal("myllama3"); ok {
		t.Errorf("post-invalidate resolve of the drifted spelling served a stale twin (window %d), want ok=false", meta.ContextWindow)
	}
	if _, ok := registry.RuntimeMetadata("my-llama-3"); ok {
		t.Error("runtime entry survived Invalidate")
	}
}

// TestModelRegistry_Invalidate_SweepsNormalizedNegativeCacheTwins checks the
// negative half of the sweep: a failed HuggingFace probe recorded under the
// drifted spelling must not keep suppressing probes for the model after a
// switch under its canonical spelling.
func TestModelRegistry_Invalidate_SweepsNormalizedNegativeCacheTwins(t *testing.T) {
	registry := NewModelRegistry(nil)

	registry.mu.Lock()
	registry.negativeCache["myllama3"] = time.Now()
	registry.mu.Unlock()

	registry.Invalidate("my-llama-3")

	registry.mu.RLock()
	_, left := registry.negativeCache["myllama3"]
	registry.mu.RUnlock()
	if left {
		t.Error("negative-cache twin spelling survived Invalidate")
	}
}
