package sp4rk

import (
	"testing"

	"github.com/v0lka/sp4rk/llm"
)

// TestFramework_Streaming_Fluent verifies the fluent Streaming(true) setter
// propagates to ExecutionConfig.Streaming (and defaults to false).
func TestFramework_Streaming_Fluent(t *testing.T) {
	fw, err := NewF().Provider(dummyProvider()).Streaming(true).Build()
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	t.Cleanup(func() { _ = fw.Shutdown() })
	if !fw.cfg.Execution.Streaming {
		t.Error("ExecutionConfig.Streaming = false, want true after .Streaming(true)")
	}

	def := testFramework(t)
	if def.cfg.Execution.Streaming {
		t.Error("ExecutionConfig.Streaming = true by default, want false")
	}
}

// TestFramework_Streaming_Classic verifies the classic Config path preserves
// ExecutionConfig.Streaming.
func TestFramework_Streaming_Classic(t *testing.T) {
	fw, err := New(Config{
		LLM:       LLMConfig{Providers: []llm.ProviderEntry{dummyProvider()}},
		Execution: ExecutionConfig{Streaming: true},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(func() { _ = fw.Shutdown() })
	if !fw.cfg.Execution.Streaming {
		t.Error("ExecutionConfig.Streaming = false, want true")
	}
}

// TestFramework_Streaming_Option verifies the WithStreaming functional option
// flows through mergeConfig.
func TestFramework_Streaming_Option(t *testing.T) {
	fw, err := NewF().Options(WithProvider(dummyProvider()), WithStreaming(true)).Build()
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	t.Cleanup(func() { _ = fw.Shutdown() })
	if !fw.cfg.Execution.Streaming {
		t.Error("ExecutionConfig.Streaming = false, want true via WithStreaming(true)")
	}
}

// TestFramework_Streaming_ExplicitValueOverridesBaseConfig verifies that an
// explicit WithStreaming(false) can turn OFF a Streaming: true that arrived
// from a base WithConfig (and that leaving the option unset preserves the base
// value) — the explicit option value always wins, in both directions.
func TestFramework_Streaming_ExplicitValueOverridesBaseConfig(t *testing.T) {
	base := Config{
		LLM:       LLMConfig{Providers: []llm.ProviderEntry{dummyProvider()}},
		Execution: ExecutionConfig{Streaming: true},
	}

	off, err := NewF().Options(WithConfig(base), WithStreaming(false)).Build()
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	t.Cleanup(func() { _ = off.Shutdown() })
	if off.cfg.Execution.Streaming {
		t.Error("ExecutionConfig.Streaming = true, want false via WithStreaming(false) over a Streaming:true base")
	}

	preserved, err := NewF().Options(WithConfig(base)).Build()
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	t.Cleanup(func() { _ = preserved.Shutdown() })
	if !preserved.cfg.Execution.Streaming {
		t.Error("ExecutionConfig.Streaming = false, want the base Streaming:true preserved when no option is set")
	}
}
