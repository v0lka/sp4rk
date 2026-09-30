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
