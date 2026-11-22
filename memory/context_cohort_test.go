package memory

import (
	"fmt"
	"strconv"
	"strings"
	"testing"
	"time"

	sdkagent "github.com/v0lka/sp4rk/agent"
	"github.com/v0lka/sp4rk/llm"
)

// cohortPromptSignature renders a build's messages as a deterministic string
// so consecutive builds can be compared byte-for-byte.
func cohortPromptSignature(msgs []llm.Message) string {
	var sb strings.Builder
	for _, m := range msgs {
		sb.WriteString(m.Role)
		sb.WriteByte(0)
		sb.WriteString(m.Content)
		sb.WriteByte(0)
		sb.WriteString(m.ReasoningContent)
		sb.WriteByte(0)
		sb.WriteString(m.ToolCallID)
		sb.WriteByte(0)
		sb.WriteString(strconv.Itoa(len(m.ToolCalls)))
		sb.WriteByte(1)
	}
	return sb.String()
}

// newCohortTestWindow returns a ContextWindow with standard test metadata and
// the given history mutation applied.
func newCohortTestWindow(t *testing.T, mutation HistoryMutation, pruning ToolOutputPruning) *ContextWindow {
	t.Helper()
	cw := NewContextWindow(ContextWindowConfig{
		SystemPrompt: "System",
		ModelMeta:    testModelMeta(128000),
		Tracker:      llm.NewContextTokenTracker(llm.NewSimpleTokenCounter()),
		Thresholds:   testThresholds(),
		Pruning:      pruning,
	})
	cw.SetHistoryMutation(mutation)
	return cw
}

// TestHistoryMutation_CohortEvictionPrefixStability is the core regression
// test for cohort-quantized eviction: between cohort boundaries the request
// prefix is byte-stable across builds (only the new step's messages are
// appended); at each boundary a single batch of CohortSize steps crosses the
// eviction threshold together.
//
// With ToolResultEvictionStep=3 and CohortSize=5 the legacy boundary
// (len(steps) - 3) reaches multiples of 5 at len = 8, 13, 18 — exactly K
// builds apart.
func TestHistoryMutation_CohortEvictionPrefixStability(t *testing.T) {
	const (
		evictionStep = 3
		cohortSize   = 5
		totalSteps   = 18
	)
	cw := newCohortTestWindow(t, HistoryMutation{
		ToolResultEvictionStep: evictionStep,
		CohortSize:             cohortSize,
	}, ToolOutputPruning{})

	jumpBuild := map[int]bool{8: true, 13: true, 18: true}

	sigs := make(map[int]string, totalSteps)
	evicted := make(map[int]int, totalSteps)
	for i := 1; i <= totalSteps; i++ {
		cw.AddStep(makeStepWithCacheHash(
			fmt.Sprintf("Thought %d", i),
			fmt.Sprintf("Full observation %d", i),
			"read_file",
			fmt.Sprintf("hash_%d", i),
			i,
		))
		sigs[i] = cohortPromptSignature(cw.BuildPrompt())
		evicted[i] = strings.Count(sigs[i], "Result evicted to cache")
	}

	if evicted[1] != 0 {
		t.Fatalf("first build should evict nothing, got %d", evicted[1])
	}

	for i := 2; i <= totalSteps; i++ {
		prefixStable := strings.HasPrefix(sigs[i], sigs[i-1])
		if jumpBuild[i] {
			// Single batch jump: the prefix changes exactly once and exactly
			// CohortSize more steps cross the threshold together.
			if prefixStable {
				t.Errorf("build %d (cohort boundary): expected a batch jump, prefix unchanged", i)
			}
			if got, want := evicted[i], evicted[i-1]+cohortSize; got != want {
				t.Errorf("build %d (cohort boundary): evicted %d refs, want %d (+%d)", i, got, want, cohortSize)
			}
			// The jump evicts exactly the cohort entering the boundary:
			// steps idx in [boundary-cohortSize, boundary).
			boundary := evicted[i]
			for j := boundary - cohortSize; j < boundary; j++ {
				wantRef := evictionReferenceText(fmt.Sprintf("hash_%d", j+1))
				if !strings.Contains(sigs[i], wantRef) {
					t.Errorf("build %d: cohort step idx %d (hash_%d) should carry its cache reference %q", i, j, j+1, wantRef)
				}
			}
			continue
		}
		// Mid-window: prefix byte-stable — no mutation, only the new step's
		// messages appended after the previous build's full prompt.
		if !prefixStable {
			t.Errorf("build %d: prefix mutated mid-window (K-1 consecutive builds must be byte-stable)", i)
		}
		if got, want := evicted[i], evicted[i-1]; got != want {
			t.Errorf("build %d: evicted ref count changed mid-window: %d -> %d", i, want, got)
		}
	}
}

// TestToolOutputPruning_CohortPrefixStability verifies the pruning window
// advances in CohortSize-sized cohorts too: with age-based eviction disabled
// and pruning always active, the KeepLastN boundary jumps once per cohort
// instead of shifting one tool result per build.
//
// With KeepLastN=4 and CohortSize=5 the legacy window start
// (toolCount - 4) reaches multiples of 5 at toolCount = 9, 14.
func TestToolOutputPruning_CohortPrefixStability(t *testing.T) {
	const (
		keepLastN  = 4
		cohortSize = 5
		totalSteps = 18
	)
	cw := newCohortTestWindow(t, HistoryMutation{CohortSize: cohortSize}, ToolOutputPruning{
		KeepLastN:        keepLastN,
		ThresholdPercent: 0, // bypass the fill floor — pruning always active
	})

	jumpBuild := map[int]bool{9: true, 14: true}

	sigs := make(map[int]string, totalSteps)
	pruned := make(map[int]int, totalSteps)
	for i := 1; i <= totalSteps; i++ {
		cw.AddStep(makeStepWithCacheHash(
			fmt.Sprintf("Thought %d", i),
			fmt.Sprintf("Full observation %d", i),
			"read_file",
			fmt.Sprintf("hash_%d", i),
			i,
		))
		sigs[i] = cohortPromptSignature(cw.BuildPrompt())
		pruned[i] = strings.Count(sigs[i], "Tool output pruned")
	}

	for i := 2; i <= totalSteps; i++ {
		prefixStable := strings.HasPrefix(sigs[i], sigs[i-1])
		if jumpBuild[i] {
			if prefixStable {
				t.Errorf("build %d (cohort boundary): expected a batch jump of the pruning window, prefix unchanged", i)
			}
			if got, want := pruned[i], pruned[i-1]+cohortSize; got != want {
				t.Errorf("build %d (cohort boundary): %d pruned placeholders, want %d (+%d)", i, got, want, cohortSize)
			}
			continue
		}
		if !prefixStable {
			t.Errorf("build %d: prefix mutated mid-window (pruning boundary must hold between cohorts)", i)
		}
		if got, want := pruned[i], pruned[i-1]; got != want {
			t.Errorf("build %d: pruned placeholder count changed mid-window: %d -> %d", i, want, got)
		}
	}

	// The KeepLastN most recent tool results are never pruned at any build.
	final := cw.BuildPrompt()
	lastContent := fmt.Sprintf("Full observation %d", totalSteps)
	found := false
	for _, m := range final {
		if m.Role == "tool" && m.Content == lastContent {
			found = true
		}
	}
	if !found {
		t.Errorf("most recent tool result %q should stay protected from pruning", lastContent)
	}
}

// TestHistoryMutation_CohortDisabled_LegacyPerStepAdvance pins the escape
// hatch: CohortSize unset (0) and CohortSize 1 both keep the legacy per-step
// boundary advance — one more step evicted per build, prefix changing on
// every build after the threshold.
func TestHistoryMutation_CohortDisabled_LegacyPerStepAdvance(t *testing.T) {
	for _, cohort := range []int{0, 1} {
		t.Run(fmt.Sprintf("cohort_%d", cohort), func(t *testing.T) {
			const (
				evictionStep = 2
				totalSteps   = 8
			)
			cw := newCohortTestWindow(t, HistoryMutation{
				ToolResultEvictionStep: evictionStep,
				CohortSize:             cohort,
			}, ToolOutputPruning{})

			sigs := make(map[int]string, totalSteps)
			evicted := make(map[int]int, totalSteps)
			for i := 1; i <= totalSteps; i++ {
				cw.AddStep(makeStepWithCacheHash(
					fmt.Sprintf("Thought %d", i),
					fmt.Sprintf("Full observation %d", i),
					"read_file",
					fmt.Sprintf("hash_%d", i),
					i,
				))
				sigs[i] = cohortPromptSignature(cw.BuildPrompt())
				evicted[i] = strings.Count(sigs[i], "Result evicted to cache")
			}

			// Legacy: build i (>= threshold+1) evicts exactly i-evictionStep
			// results and mutates the prefix on EVERY build.
			for i := evictionStep + 1; i <= totalSteps; i++ {
				if got, want := evicted[i], i-evictionStep; got != want {
					t.Errorf("build %d: evicted %d refs, want %d (legacy per-step advance)", i, got, want)
				}
				if strings.HasPrefix(sigs[i], sigs[i-1]) {
					t.Errorf("build %d: legacy advance should mutate the prefix on every build", i)
				}
			}
		})
	}
}

// TestContextWindowConfig_MutationCohortSizeSeeds verifies that
// ContextWindowConfig.MutationCohortSize seeds the mutation cohort without a
// SetHistoryMutation call. It is proven through the pruning window, which
// shares the cohort knob and works with a zero-value HistoryMutation: with
// KeepLastN=4 the legacy window start (toolCount - 4) reaches multiples of 5
// at toolCount = 9 and 14.
func TestContextWindowConfig_MutationCohortSizeSeeds(t *testing.T) {
	cw := NewContextWindow(ContextWindowConfig{
		SystemPrompt: "System",
		ModelMeta:    testModelMeta(128000),
		Tracker:      llm.NewContextTokenTracker(llm.NewSimpleTokenCounter()),
		Thresholds:   testThresholds(),
		Pruning: ToolOutputPruning{
			KeepLastN:        4,
			ThresholdPercent: 0, // bypass the fill floor — pruning always active
		},
		MutationCohortSize: 5,
	})
	if cw.mutation.CohortSize != 5 {
		t.Fatalf("MutationCohortSize should seed mutation.CohortSize, got %d", cw.mutation.CohortSize)
	}

	pruned := make(map[int]int, 14)
	for i := 1; i <= 14; i++ {
		cw.AddStep(makeStepWithCacheHash(
			fmt.Sprintf("Thought %d", i),
			fmt.Sprintf("Full observation %d", i),
			"read_file",
			fmt.Sprintf("hash_%d", i),
			i,
		))
		pruned[i] = strings.Count(cohortPromptSignature(cw.BuildPrompt()), "Tool output pruned")
	}
	for i := 1; i <= 8; i++ {
		if pruned[i] != 0 {
			t.Errorf("build %d: expected no pruning before the first cohort boundary, got %d", i, pruned[i])
		}
	}
	for i := 9; i <= 13; i++ {
		if pruned[i] != 5 {
			t.Errorf("build %d: expected the first cohort (5 pruned), got %d", i, pruned[i])
		}
	}
	if pruned[14] != 10 {
		t.Errorf("build 14: expected the second cohort (10 pruned), got %d", pruned[14])
	}
}

// TestHistoryMutation_CohortDoesNotDelaySameBuildMechanisms pins that the
// cohort quantization affects only the aging boundary: EvictStepStatus and
// DedupRepeatedReads are same-build semantics and evict immediately even when
// cohorts are enabled.
func TestHistoryMutation_CohortDoesNotDelaySameBuildMechanisms(t *testing.T) {
	cw := newCohortTestWindow(t, HistoryMutation{
		EvictStepStatus:        true,
		DedupRepeatedReads:     true,
		ToolResultEvictionStep: 100, // aging never fires in this test
		CohortSize:             5,
	}, ToolOutputPruning{})

	cw.AddStep(makeStepWithCacheHash("Thought 1", "OK", "update_checklist", "h_check", 1))
	cw.AddStep(makeStepWithCacheHash("Thought 2", "content A", "read_file", "h_dup", 2))
	cw.AddStep(makeStepWithCacheHash("Thought 3", "content A", "read_file", "h_dup", 3))

	messages := cw.BuildPrompt()
	// Layout: system, then assistant+tool per step → tool messages at 2, 4, 6.
	if messages[2].Content != stepStatusEvictedText {
		t.Errorf("update_checklist should be evicted immediately (same-build), got %q", messages[2].Content)
	}
	if messages[4].Content != "content A" {
		t.Errorf("first read should keep full content, got %q", messages[4].Content)
	}
	if !strings.Contains(messages[6].Content, "Result evicted to cache") || !strings.Contains(messages[6].Content, "h_dup") {
		t.Errorf("duplicate read should be replaced with a reference immediately (same-build), got %q", messages[6].Content)
	}
}

// TestHistoryMutation_CohortEvictionRecoversViaCache verifies the information
// preservation guarantee under cohort eviction: the batch-evicted placeholder
// carries the step's cache hash and that hash still resolves to the full
// content via ToolResultCache (tool_result_read recoverability).
func TestHistoryMutation_CohortEvictionRecoversViaCache(t *testing.T) {
	cache := sdkagent.NewToolResultCache(time.Minute)
	content := "alpha\nbeta\ngamma\nlong file body"
	hash := cache.Store("read_file", content, sdkagent.ToolCacheMeta{
		Input: `{"path": "/tmp/example.txt"}`,
	})

	cw := newCohortTestWindow(t, HistoryMutation{
		ToolResultEvictionStep: 2,
		CohortSize:             5,
	}, ToolOutputPruning{})

	// Step 1 carries the cached content; steps 2-7 push it across the first
	// cohort boundary (legacy pos = len-2 reaches 5 at len=7).
	cw.AddStep(makeStepWithCacheHash("Thought 1", content, "read_file", hash, 1))
	for i := 2; i <= 7; i++ {
		cw.AddStep(makeStepWithCacheHash(
			fmt.Sprintf("Thought %d", i),
			fmt.Sprintf("Full observation %d", i),
			"read_file",
			fmt.Sprintf("hash_%d", i),
			i,
		))
	}

	messages := cw.BuildPrompt()
	wantRef := evictionReferenceText(hash)
	found := false
	for _, m := range messages {
		if m.Role == "tool" && m.Content == wantRef {
			found = true
		}
	}
	if !found {
		t.Fatalf("cohort-evicted step should carry the exact cache reference %q", wantRef)
	}

	entry, ok := cache.Get(hash)
	if !ok {
		t.Fatalf("cache hash %q from the eviction reference should resolve", hash)
	}
	if entry.Content != content {
		t.Errorf("recovered content mismatch:\n got %q\nwant %q", entry.Content, content)
	}
}
