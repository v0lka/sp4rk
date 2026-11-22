package memory

import (
	"strings"
	"testing"

	sdkagent "github.com/v0lka/sp4rk/agent"
	"github.com/v0lka/sp4rk/llm"
)

// makeStepWithBlocks returns a tool step carrying signed thinking blocks.
func makeStepWithBlocks(thought, observation, toolName string, toolID int, blocks []llm.ThinkingBlock) sdkagent.Step {
	step := makeStepWithTool(thought, observation, toolName, toolID)
	step.ThinkingBlocks = blocks
	return step
}

// assistantCarriedBlocks collects the ThinkingBlocks carried by each assistant
// message of a build.
func assistantCarriedBlocks(msgs []llm.Message) [][]llm.ThinkingBlock {
	var out [][]llm.ThinkingBlock
	for _, m := range msgs {
		if m.Role == "assistant" {
			out = append(out, m.ThinkingBlocks)
		}
	}
	return out
}

// sigBlocks is a minimal signed thinking block set used across the tests.
func sigBlocks(sig string) []llm.ThinkingBlock {
	return []llm.ThinkingBlock{{Type: "thinking", Thinking: "reasoning " + sig, Signature: sig}}
}

// TestThinkingBlocks_EvictionBoundaryDropsStaleBlocks pins the mutation-epoch
// contract for age-based eviction: between cohort boundaries the prefix is
// byte-stable, so block-carrying steps keep replaying their blocks; at a
// cohort boundary the observations are replaced with cache references, which
// mutates the prefix every block is bound to — those blocks must be stripped
// from the replay (the API rejects a block whose signature was computed over
// a different prefix). Steps materialized AFTER the boundary keep carrying
// valid blocks.
func TestThinkingBlocks_EvictionBoundaryDropsStaleBlocks(t *testing.T) {
	const (
		evictionStep = 3
		cohortSize   = 5
	)
	cw := newCohortTestWindow(t, HistoryMutation{
		ToolResultEvictionStep: evictionStep,
		CohortSize:             cohortSize,
	}, ToolOutputPruning{})

	// Steps 1..4: boundary cohortBoundary(len-3, 5) stays 0 for len 4..7 —
	// nothing evicted, blocks valid. The block-carrying steps must keep
	// riding every build in this range (later steps carry no blocks).
	for i := 1; i <= 4; i++ {
		cw.AddStep(makeStepWithBlocks(
			"Thought", "Full observation", "read_file", i, sigBlocks("SIG_A"),
		))
	}
	for i := 5; i <= 7; i++ {
		cw.AddStep(makeStepWithCacheHash(
			"Thought", "Full observation", "read_file", "hash", i,
		))
		msgs := cw.BuildPrompt()
		carried := assistantCarriedBlocks(msgs)
		if len(carried) == 0 || len(carried[0]) == 0 {
			t.Fatalf("build at len=%d: block-carrying steps must still replay their blocks (boundary stable, nothing evicted): %+v", i, carried)
		}
	}

	// Step 8 crosses the cohort boundary (boundary 0 -> 5): steps 0..4 are
	// evicted, so every block materialized before the jump — including the
	// block-carrying step — is bound to a mutated prefix and must be stripped.
	cw.AddStep(makeStepWithCacheHash(
		"Thought", "Full observation", "read_file", "hash", 8,
	))
	msgs := cw.BuildPrompt()
	carried := assistantCarriedBlocks(msgs)
	for i, blocks := range carried {
		if len(blocks) != 0 {
			t.Errorf("build at cohort boundary: assistant message %d must not carry thinking blocks (prefix mutated), got %+v", i, blocks)
		}
	}
	if got := len(msgs); got == 0 {
		t.Fatal("boundary build produced no messages")
	}

	// A step materialized after the boundary is stamped with the new epoch;
	// with the boundary stable until the next cohort, its blocks are valid.
	cw.AddStep(makeStepWithBlocks(
		"Thought", "Full observation", "read_file", 9, sigBlocks("SIG_B"),
	))
	msgs = cw.BuildPrompt()
	carried = assistantCarriedBlocks(msgs)
	if len(carried) == 0 || len(carried[len(carried)-1]) == 0 {
		t.Fatalf("post-boundary build: last assistant turn must carry its thinking blocks (boundary stable since materialization): %+v", carried)
	}
	if carried[len(carried)-1][0].Signature != "SIG_B" {
		t.Errorf("post-boundary build carried %+v, want SIG_B", carried[len(carried)-1])
	}
}

// TestThinkingBlocks_PruningActivationDropsStaleBlocks pins the contract for
// tool-output pruning: when the fill crosses the pruning threshold, old tool
// results are replaced with placeholders — a prefix mutation. Blocks
// materialized before the crossing must be stripped; the last assistant turn
// going out without blocks lets the provider's fail-safe degrade to a
// thinking-disabled (valid) request instead of a guaranteed HTTP 400.
func TestThinkingBlocks_PruningActivationDropsStaleBlocks(t *testing.T) {
	cw := NewContextWindow(ContextWindowConfig{
		SystemPrompt: "System",
		ModelMeta: llm.ModelMetadata{
			ContextWindow: 8000,
			OutputLimit:   1000,
			TokenizerType: "approximate",
		},
		Tracker:    llm.NewContextTokenTracker(llm.NewSimpleTokenCounter()),
		Thresholds: testThresholds(),
		Pruning:    ToolOutputPruning{KeepLastN: 2, ThresholdPercent: 5},
	})

	// Below the pruning threshold: blocks ride the builds.
	for i := 1; i <= 3; i++ {
		cw.AddStep(makeStepWithBlocks(
			"Thought", "Full observation", "read_file", i, sigBlocks("SIG_A"),
		))
	}
	carried := assistantCarriedBlocks(cw.BuildPrompt())
	if len(carried) == 0 || len(carried[len(carried)-1]) == 0 {
		t.Fatalf("pre-threshold build: last assistant turn must carry its thinking blocks: %+v", carried)
	}

	// Grow the context until fill crosses the threshold: the placeholder
	// substitution mutates the prefix, so pre-crossing blocks must go.
	for i := 4; i <= 60; i++ {
		cw.AddStep(makeStepWithCacheHash(
			"Thought", "Large observation to push the fill estimate past the pruning threshold.", "read_file", "hash", i,
		))
		if cw.FillPercent() < 5 {
			continue
		}
		carried = assistantCarriedBlocks(cw.BuildPrompt())
		for j, blocks := range carried {
			if len(blocks) != 0 {
				t.Errorf("post-threshold build: assistant message %d must not carry thinking blocks (pruning mutated the prefix), got %+v", j, blocks)
			}
		}
		return
	}
	t.Fatal("fill never crossed the pruning threshold — test setup broken")
}

// TestThinkingBlocks_CompactionStripsBlocks pins the contract for compaction:
// the frozen prefix must never carry thinking blocks (compaction rewrites the
// history every surviving block is bound to), live pre-compaction steps stop
// carrying theirs on the next build, and a step materialized after the
// compaction keeps its (valid) blocks.
func TestThinkingBlocks_CompactionStripsBlocks(t *testing.T) {
	cw := NewContextWindow(ContextWindowConfig{
		SystemPrompt: "System",
		ModelMeta:    testModelMeta(128000),
		Tracker:      llm.NewContextTokenTracker(llm.NewSimpleTokenCounter()),
		Thresholds:   testThresholds(),
		Strategy:     NewSlidingWindowStrategy(2, 2),
	})

	for i := 1; i <= 4; i++ {
		cw.AddStep(makeStepWithBlocks(
			"Thought", "Full observation", "read_file", i, sigBlocks("SIG_A"),
		))
	}
	if res := cw.Compact(t.Context()); res == nil {
		t.Fatal("Compact did not run")
	}

	// The frozen prefix carries no thinking blocks.
	for i, m := range cw.compactedMessages {
		if m.Role == "assistant" && len(m.ThinkingBlocks) != 0 {
			t.Errorf("frozen prefix assistant message %d carries thinking blocks %+v — must be stripped after compaction", i, m.ThinkingBlocks)
		}
	}

	// The next build renders the frozen prefix + live steps: the live steps
	// are pre-compaction (stale epoch) and must not carry blocks either.
	carried := assistantCarriedBlocks(cw.BuildPrompt())
	for j, blocks := range carried {
		if len(blocks) != 0 {
			t.Errorf("post-compaction build: assistant message %d must not carry thinking blocks (prefix rewritten), got %+v", j, blocks)
		}
	}

	// A step materialized after compaction keeps its blocks.
	cw.AddStep(makeStepWithBlocks(
		"Thought", "Full observation", "read_file", 5, sigBlocks("SIG_B"),
	))
	carried = assistantCarriedBlocks(cw.BuildPrompt())
	if len(carried) == 0 || len(carried[len(carried)-1]) == 0 {
		t.Fatalf("post-compaction step must carry its thinking blocks: %+v", carried)
	}
	if carried[len(carried)-1][0].Signature != "SIG_B" {
		t.Errorf("post-compaction step carried %+v, want SIG_B", carried[len(carried)-1])
	}
}

// TestThinkingBlocks_GroupedCarriageGatedByEpoch pins the gate on the grouped
// renderer: a response group's assistant turn is built from groupSteps[0],
// whose blocks ride only while the epoch matches — after a mutation the whole
// group renders block-free.
func TestThinkingBlocks_GroupedCarriageGatedByEpoch(t *testing.T) {
	cw := NewContextWindow(ContextWindowConfig{
		SystemPrompt: "System",
		ModelMeta:    testModelMeta(128000),
		Tracker:      llm.NewContextTokenTracker(llm.NewSimpleTokenCounter()),
		Thresholds:   testThresholds(),
		Pruning:      ToolOutputPruning{KeepLastN: 1, ThresholdPercent: 0},
	})

	group := func(id int) sdkagent.Step {
		s := makeStepWithBlocks("Thought", "Full observation", "read_file", id, sigBlocks("SIG_A"))
		s.ResponseGroup = 1
		return s
	}
	cw.AddStep(group(1))
	cw.AddStep(group(2))

	// Group below the KeepLastN window start: nothing pruned, blocks ride.
	carried := assistantCarriedBlocks(cw.BuildPrompt())
	if len(carried) == 0 || len(carried[len(carried)-1]) == 0 {
		t.Fatalf("grouped build must carry groupSteps[0]'s thinking blocks: %+v", carried)
	}
	assistantCount := len(carried)
	if assistantCount != 1 {
		t.Fatalf("group must render exactly one assistant turn, got %d", assistantCount)
	}

	// Push two more groups so the first group's tool results fall out of the
	// KeepLastN window (pruned) — a prefix mutation for everything before it,
	// including the surviving group's blocks.
	for i := 3; i <= 6; i++ {
		s := makeStepWithCacheHash("Thought", "Full observation", "read_file", "hash", i)
		s.ResponseGroup = int64(i / 2)
		cw.AddStep(s)
	}
	carried = assistantCarriedBlocks(cw.BuildPrompt())
	for j, blocks := range carried {
		if len(blocks) != 0 {
			t.Errorf("post-prune grouped build: assistant message %d must not carry thinking blocks, got %+v", j, blocks)
		}
	}
}

// TestThinkingBlocks_StablePrefixKeepsBlocks guards against false bumps: with
// eviction and pruning disabled, repeated builds never mutate the prefix, so
// block-carrying steps keep replaying their blocks on every build.
func TestThinkingBlocks_StablePrefixKeepsBlocks(t *testing.T) {
	cw := newCohortTestWindow(t, HistoryMutation{}, ToolOutputPruning{})

	for i := 1; i <= 5; i++ {
		cw.AddStep(makeStepWithBlocks(
			"Thought", "Full observation", "read_file", i, sigBlocks("SIG_A"),
		))
		carried := assistantCarriedBlocks(cw.BuildPrompt())
		for j, blocks := range carried {
			if len(blocks) == 0 {
				t.Errorf("build %d: assistant message %d lost its thinking blocks without any history mutation", i, j)
			}
		}
	}
}

// TestThinkingBlocks_WarmUpKeepsBlocksUntilFirstCohortBoundary pins the
// warm-up contract for age-based eviction: while len(steps) stays below the
// first cohort crossing, the boundary sits at 0 (during warm-up the legacy
// position len-ToolResultEvictionStep is negative and is clamped to 0), so
// every build is append-only — a byte-stable render must not bump the
// mutation epoch, and blocks materialized in that range must keep replaying
// until a real cohort crossing actually evicts something. With the raw
// negative position, the boundary advanced on every warm-up build and bumped
// the epoch despite the byte-identical render, stripping warm-up blocks and
// degrading the rest of the run to thinking-disabled requests.
func TestThinkingBlocks_WarmUpKeepsBlocksUntilFirstCohortBoundary(t *testing.T) {
	const (
		evictionStep = 5
		cohortSize   = 5
	)
	cw := newCohortTestWindow(t, HistoryMutation{
		ToolResultEvictionStep: evictionStep,
		CohortSize:             cohortSize,
	}, ToolOutputPruning{})

	// Warm-up and first-cohort approach (len 1..evictionStep+cohortSize-1):
	// the clamped boundary stays 0 — every build is append-only, epoch-stable,
	// and the warm-up block-carrying step keeps riding every build.
	prevSig := ""
	for i := 1; i <= evictionStep+cohortSize-1; i++ {
		if i == 1 {
			cw.AddStep(makeStepWithBlocks(
				"Thought", "Full observation", "read_file", i, sigBlocks("SIG_A"),
			))
		} else {
			cw.AddStep(makeStepWithCacheHash(
				"Thought", "Full observation", "read_file", "hash", i,
			))
		}
		epoch := cw.mutationEpoch
		msgs := cw.BuildPrompt()
		if cw.mutationEpoch != epoch {
			t.Fatalf("build at len=%d bumped the mutation epoch (%d -> %d) although the render is append-only (boundary 0, nothing evicted)", i, epoch, cw.mutationEpoch)
		}
		sig := cohortPromptSignature(msgs)
		if prevSig != "" && !strings.HasPrefix(sig, prevSig) {
			t.Errorf("build at len=%d mutated the prefix before the first cohort crossing (warm-up and cohort approach must be append-only)", i)
		}
		prevSig = sig
		carried := assistantCarriedBlocks(msgs)
		if len(carried) == 0 || len(carried[0]) == 0 {
			t.Fatalf("build at len=%d: warm-up block-carrying step must keep replaying its blocks: %+v", i, carried)
		}
	}

	// len = evictionStep+cohortSize crosses the first real cohort boundary
	// (pos = cohortSize → boundary = cohortSize): steps 0..cohortSize-1 are
	// evicted, the prefix mutates, and the warm-up blocks — bound to the old
	// prefix — are stripped. Exactly one epoch bump for exactly one render
	// mutation.
	epoch := cw.mutationEpoch
	if epoch != 0 {
		t.Fatalf("warm-up must leave the mutation epoch untouched, got %d", epoch)
	}
	cw.AddStep(makeStepWithCacheHash(
		"Thought", "Full observation", "read_file", "hash", evictionStep+cohortSize,
	))
	msgs := cw.BuildPrompt()
	if got := cw.mutationEpoch; got != epoch+1 {
		t.Fatalf("first cohort crossing must bump the mutation epoch exactly once (%d -> %d), got %d", epoch, epoch+1, got)
	}
	carried := assistantCarriedBlocks(msgs)
	for j, blocks := range carried {
		if len(blocks) != 0 {
			t.Errorf("first cohort crossing build: assistant message %d must not carry pre-mutation thinking blocks, got %+v", j, blocks)
		}
	}
}
