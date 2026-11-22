package sp4rk

import (
	"fmt"
	"strings"
	"testing"

	"github.com/v0lka/sp4rk/agent"
	"github.com/v0lka/sp4rk/llm"
)

// newMutationTestFramework builds a Framework with the given Execution config
// (only the fields the history-mutation tests need are set).
func newMutationTestFramework(t *testing.T, execution ExecutionConfig) *Framework {
	t.Helper()
	fw, err := New(Config{
		LLM:       LLMConfig{Providers: []llm.ProviderEntry{dummyProvider()}},
		Execution: execution,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(func() { _ = fw.Shutdown() })
	return fw
}

// mutationTestStep is a cacheable tool step (CacheHash set), the shape the
// age-based eviction acts on.
func mutationTestStep(i int) agent.Step {
	return agent.Step{
		Thought:     "Thought",
		Action:      llm.ToolCall{ID: fmt.Sprintf("call_%d", i), Name: "read_file", Input: []byte(`{"path":"f.txt"}`)},
		Observation: fmt.Sprintf("Full observation %d", i),
		CacheHash:   fmt.Sprintf("hash_%d", i),
	}
}

// countEvictionRefs counts cache-reference replacements in a build.
func countEvictionRefs(msgs []llm.Message) int {
	n := 0
	for _, m := range msgs {
		if strings.Contains(m.Content, "Result evicted to cache") {
			n++
		}
	}
	return n
}

// testModelMetaForWindow is the model metadata the mutation tests build
// context windows with.
func testModelMetaForWindow() llm.ModelMetadata {
	return llm.ModelMetadata{ContextWindow: 128000, OutputLimit: 4096, TokenizerType: "approximate"}
}

// TestExecutionConfig_MutationKnobsReachContextWindow pins finding-3 wiring:
// the history-mutation knobs are reachable from sp4rk.Config on the
// recommended (Framework/Conductor) path. With ToolResultEvictionStep set,
// the built window evicts old tool results to cache references once fill
// passes the pruning threshold (below it every result is protected); with
// the default cohort (5) the boundary advances in K-sized jumps.
func TestExecutionConfig_MutationKnobsReachContextWindow(t *testing.T) {
	fw := newMutationTestFramework(t, ExecutionConfig{ToolResultEvictionStep: 3})
	cw := fw.buildContextWindow("System", testModelMetaForWindow(), "sliding_window")

	// cohortOf replicates memory.cohortBoundary for the expected-eviction
	// arithmetic (k<=1 or pos<=0 → per-step identity).
	cohortOf := func(pos, k int) int {
		if k <= 1 || pos <= 0 {
			return pos
		}
		return (pos / k) * k
	}

	crossed := false
	for i := 1; i <= 40; i++ {
		step := mutationTestStep(i)
		step.Observation = strings.Repeat("x", 12000) // push fill past 50%
		cw.AddStep(step)
		if cw.FillPercent() < 50 {
			continue
		}
		if !crossed {
			crossed = true
		}
		got := countEvictionRefs(cw.BuildPrompt())
		want := cohortOf(i-3, 5) // eviction boundary at len(steps)=i, default cohort 5
		if got != want {
			t.Fatalf("build at len=%d: evicted %d refs, want %d (boundary cohort-quantized with the default cohort size 5)", i, got, want)
		}
	}
	if !crossed {
		t.Fatal("fill never crossed the pruning threshold — test setup broken")
	}
}

// TestExecutionConfig_MutationEvictionOffByDefault pins the no-regression
// default: without ToolResultEvictionStep, a Framework-built window keeps
// cacheable tool results (eviction disabled) even past the pruning threshold
// — unprotected old results are pruned to placeholders, never to cache
// references, and the cohort default (5) alone must not enable eviction.
func TestExecutionConfig_MutationEvictionOffByDefault(t *testing.T) {
	fw := newMutationTestFramework(t, ExecutionConfig{})
	cw := fw.buildContextWindow("System", testModelMetaForWindow(), "sliding_window")

	crossed := false
	for i := 1; i <= 40; i++ {
		step := mutationTestStep(i)
		step.Observation = strings.Repeat("x", 12000)
		cw.AddStep(step)
		if cw.FillPercent() >= 50 {
			crossed = true
		}
		if n := countEvictionRefs(cw.BuildPrompt()); n != 0 {
			t.Fatalf("build at len=%d evicted %d refs — eviction must be disabled by default", i, n)
		}
	}
	if !crossed {
		t.Fatal("fill never crossed the pruning threshold — test setup broken")
	}
}

// TestExecutionConfig_NegativeCohortRestoresLegacyAdvance pins the negative
// sentinel: MutationCohortSize < 0 restores the legacy per-step boundary
// advance (each post-crossing build crosses exactly one more step).
func TestExecutionConfig_NegativeCohortRestoresLegacyAdvance(t *testing.T) {
	fw := newMutationTestFramework(t, ExecutionConfig{ToolResultEvictionStep: 3, MutationCohortSize: -1})
	cw := fw.buildContextWindow("System", testModelMetaForWindow(), "sliding_window")

	prev := -1
	crossed := false
	for i := 1; i <= 40; i++ {
		step := mutationTestStep(i)
		step.Observation = strings.Repeat("x", 12000)
		cw.AddStep(step)
		if cw.FillPercent() < 50 {
			continue
		}
		crossed = true
		got := countEvictionRefs(cw.BuildPrompt())
		want := i - 3 // legacy boundary: len(steps) - eviction step
		if want < 0 {
			want = 0
		}
		if got != want {
			t.Fatalf("build at len=%d: evicted %d refs, want %d (legacy per-step advance)", i, got, want)
		}
		if prev >= 0 && got != prev+1 {
			t.Fatalf("build at len=%d: evicted %d refs after %d — legacy advance must cross exactly one step per build", i, got, prev)
		}
		prev = got
	}
	if !crossed {
		t.Fatal("fill never crossed the pruning threshold — test setup broken")
	}
}
