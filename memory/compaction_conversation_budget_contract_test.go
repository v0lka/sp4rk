package memory

import (
	"context"
	"strings"
	"testing"

	"github.com/v0lka/sp4rk/llm"
)

// This file pins the token-budget contract of CompactConversationHistory for
// ALL THREE strategies using fully deterministic fakes:
// mockTokenCounter{countPerChar: 1} (one token per Content byte) and
// fixedSizeSummarizer (constant-size summary, independent of block text).
// Shares under test: sliding_window 0.8 tail / 0.15 head, summarization 0.7
// tail, hierarchical 0.5 recent zone; trimConversationToBudget is the final
// hard ceiling on every path.
//
// Contract case map (case → tests in this file + pre-existing coverage in
// compaction_conversation_manual_test.go):
//
//	short heavy dialog compacted  → ShortHeavyDialogCompactedAllStrategies
//	                                 (sliding also: TokenModeShortDialogCompacted)
//	long light history no-op      → TokenModeFittingHistoryUnchanged (all 3, pre-existing)
//	result within budget          → asserted by every test here +
//	                                 TokenModeResultWithinBudget (pre-existing)
//	last message preserved        → asserted by every test here +
//	                                 NeverRemovesLastMessage (pre-existing, count mode)
//	tail trimmed by budget        → TailCappedByBudgetBelowCountWindow (all 3)
//	budget=0 legacy count mode    → ZeroBudgetKeepsCountMode (pre-existing) +
//	                                 budget=0 contrast runs inside the test below
//	nil TokenCounter fallback     → NilCounterFallsBackToCountMode (pre-existing) +
//	                                 NilCounterFallbackHierarchicalMatchesZeroBudget
//	trim ceiling (hard clamp)     → TrimCeilingDropsOldestPrefix (all 3)
//	defensive paths reachable     → DefensivePathsReachableAndCorrect (all 3)

// fixedSizeSummarizer is a deterministic Summarize fake: every call returns a
// summary of exactly size tokens under mockTokenCounter{countPerChar: 1},
// regardless of the block text it receives, and bumps *calls when non-nil.
func fixedSizeSummarizer(size int, calls *int) func(context.Context, string) (string, error) {
	return func(_ context.Context, _ string) (string, error) {
		if calls != nil {
			*calls++
		}
		return strings.Repeat("s", size), nil
	}
}

// Case "short heavy dialog compacted": 6 messages × 100 tokens = 600 total
// against a 300-token budget. Every strategy must compact — in particular
// sliding_window, whose count window (3+10 ≥ 6) would no-op — and each
// verbatim zone must be sized by its budget share, not by message counts:
// sliding tail 0.8×300=240 → 2 messages, summarization tail 0.7×300=210 → 2
// (not the count-mode keepLast=5), hierarchical recent 0.5×300=150 → 1 (not
// the ratio-derived 3).
func TestCompactConversationBudgetContract_ShortHeavyDialogCompactedAllStrategies(t *testing.T) {
	msgs := uniformMsgs(6, 100) // 6 × 100 = 600 tokens
	const budget = 300

	cases := []struct {
		strategy  string
		want      []string // "note" | "summary" | verbatim content
		wantCalls int
	}{
		{strategy: "sliding_window", want: []string{"note", msgs[4].Content, msgs[5].Content}},
		{strategy: "summarization", want: []string{"summary", msgs[4].Content, msgs[5].Content}, wantCalls: 1},
		{strategy: "hierarchical", want: []string{"summary", "summary", msgs[5].Content}, wantCalls: 2},
	}
	for _, tc := range cases {
		t.Run(tc.strategy, func(t *testing.T) {
			counter := &mockTokenCounter{countPerChar: 1}
			calls := 0
			deps := CompactionDeps{Summarize: fixedSizeSummarizer(50, &calls), TokenCounter: counter}

			out, err := CompactConversationHistory(context.Background(), msgs, budget, tc.strategy, CompactionConfig{}, deps)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(out) != len(tc.want) {
				t.Fatalf("expected %d messages %v, got %d: %+v", len(tc.want), tc.want, len(out), out)
			}
			if got := counter.CountMessages(out); got > budget {
				t.Errorf("result exceeds budget: %d > %d", got, budget)
			}
			if out[len(out)-1].Content != msgs[len(msgs)-1].Content {
				t.Errorf("last message must survive, got %q", out[len(out)-1].Content)
			}
			if calls != tc.wantCalls {
				t.Errorf("expected %d Summarize calls, got %d", tc.wantCalls, calls)
			}
			for i, want := range tc.want {
				switch want {
				case "note":
					if out[i].Role != "system" || !strings.Contains(out[i].Content, "omitted") {
						t.Errorf("message %d: expected omission note, got %s:%q", i, out[i].Role, out[i].Content)
					}
				case "summary":
					if out[i].Role != "system" || !strings.HasPrefix(out[i].Content, "sss") {
						t.Errorf("message %d: expected fixed-size summary, got %s:%q", i, out[i].Role, out[i].Content)
					}
				default:
					if out[i].Content != want {
						t.Errorf("message %d: expected verbatim %q, got %q", i, want, out[i].Content)
					}
				}
			}
		})
	}
}

// Case "tail trimmed by budget": 8 messages × 100 tokens = 800 total against
// a 350-token budget, with count windows wide enough to keep everything
// (keepLast=10, keepFirst=3). The budget share — not the count window — must
// cap each verbatim zone: sliding tail 0.8×350=280 → 2 messages,
// summarization tail 0.7×350=245 → 2, hierarchical recent 0.5×350=175 → 1
// (ratio-derived zone is 3; the overflow folds into the summarized middle).
// The same config with budget=0 must leave the history alone (legacy count
// mode), proving the budget is the only difference.
func TestCompactConversationBudgetContract_TailCappedByBudgetBelowCountWindow(t *testing.T) {
	msgs := uniformMsgs(8, 100) // 8 × 100 = 800 tokens
	const budget = 350

	cfg := CompactionConfig{}
	cfg.SlidingWindow.KeepFirst = 3
	cfg.SlidingWindow.KeepLast = 10
	cfg.Summarization.KeepLast = 10

	cases := []struct {
		strategy      string
		want          []string
		wantCalls     int
		countModeNoop bool // same cfg with budget=0 returns history verbatim
	}{
		{strategy: "sliding_window", want: []string{"note", msgs[6].Content, msgs[7].Content}, countModeNoop: true},
		{strategy: "summarization", want: []string{"summary", msgs[6].Content, msgs[7].Content}, wantCalls: 1, countModeNoop: true},
		{strategy: "hierarchical", want: []string{"summary", "summary", msgs[7].Content}, wantCalls: 2},
	}
	for _, tc := range cases {
		t.Run(tc.strategy, func(t *testing.T) {
			counter := &mockTokenCounter{countPerChar: 1}
			calls := 0
			deps := CompactionDeps{Summarize: fixedSizeSummarizer(50, &calls), TokenCounter: counter}

			out, err := CompactConversationHistory(context.Background(), msgs, budget, tc.strategy, cfg, deps)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(out) != len(tc.want) {
				t.Fatalf("expected %d messages, got %d: %+v", len(tc.want), len(out), out)
			}
			if got := counter.CountMessages(out); got > budget {
				t.Errorf("result exceeds budget: %d > %d", got, budget)
			}
			if out[len(out)-1].Content != msgs[len(msgs)-1].Content {
				t.Errorf("last message must survive, got %q", out[len(out)-1].Content)
			}
			if calls != tc.wantCalls {
				t.Errorf("expected %d Summarize calls, got %d", tc.wantCalls, calls)
			}
			for i, want := range tc.want {
				switch want {
				case "note":
					if out[i].Role != "system" || !strings.Contains(out[i].Content, "omitted") {
						t.Errorf("message %d: expected omission note, got %s:%q", i, out[i].Role, out[i].Content)
					}
				case "summary":
					if out[i].Role != "system" || !strings.HasPrefix(out[i].Content, "sss") {
						t.Errorf("message %d: expected fixed-size summary, got %s:%q", i, out[i].Role, out[i].Content)
					}
				default:
					if out[i].Content != want {
						t.Errorf("message %d: expected verbatim %q, got %q", i, want, out[i].Content)
					}
				}
			}

			if tc.countModeNoop {
				legacy, err := CompactConversationHistory(context.Background(), msgs, 0, tc.strategy, cfg, deps)
				if err != nil {
					t.Fatalf("budget=0: unexpected error: %v", err)
				}
				if len(legacy) != len(msgs) || legacy[0].Content != msgs[0].Content || legacy[len(legacy)-1].Content != msgs[len(msgs)-1].Content {
					t.Errorf("budget=0: count window (keepFirst=3, keepLast=10) must keep all 8 messages, got %d", len(legacy))
				}
			}
		})
	}
}

// Case "trim ceiling": the budget is a hard clamp applied AFTER the strategy
// builds its result. With a summarizer returning oversized 300-token summaries
// (and sliding's fixed-size omission note), the assembled result overruns the
// budget and trimConversationToBudget must drop the oldest messages — notes
// and summaries first — until the verbatim tail fits. The LLM-backed
// strategies still perform their summarize calls: the ceiling trims the
// output, it does not skip the work.
func TestCompactConversationBudgetContract_TrimCeilingDropsOldestPrefix(t *testing.T) {
	msgs := uniformMsgs(12, 100) // 12 × 100 = 1200 tokens
	wantTail := []string{msgs[10].Content, msgs[11].Content}

	cases := []struct {
		strategy  string
		budget    int
		wantCalls int
	}{
		{strategy: "sliding_window", budget: 250, wantCalls: 0}, // note(~79) + 2×100 = 279 > 250 → note dropped
		{strategy: "summarization", budget: 400, wantCalls: 1},  // 300 + 2×100 = 500 > 400 → summary dropped
		{strategy: "hierarchical", budget: 400, wantCalls: 2},   // 2×300 + 2×100 = 800 > 400 → both summaries dropped
	}
	for _, tc := range cases {
		t.Run(tc.strategy, func(t *testing.T) {
			counter := &mockTokenCounter{countPerChar: 1}
			calls := 0
			deps := CompactionDeps{Summarize: fixedSizeSummarizer(300, &calls), TokenCounter: counter}

			out, err := CompactConversationHistory(context.Background(), msgs, tc.budget, tc.strategy, CompactionConfig{}, deps)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(out) != len(wantTail) {
				t.Fatalf("expected ceiling to clamp result to %d messages, got %d: %+v", len(wantTail), len(out), out)
			}
			for i, want := range wantTail {
				if out[i].Content != want {
					t.Errorf("message %d: expected verbatim %q, got %q", i, want, out[i].Content)
				}
			}
			if got := counter.CountMessages(out); got > tc.budget {
				t.Errorf("result exceeds budget: %d > %d", got, tc.budget)
			}
			if calls != tc.wantCalls {
				t.Errorf("expected %d Summarize calls before the ceiling, got %d", tc.wantCalls, calls)
			}
		})
	}
}

// Case "last message preserved" + the documented exception: when the last
// message alone exceeds the budget, every strategy still returns exactly that
// message (never-remove-last outranks the ceiling) even though the result
// stays over budget. Summaries and notes around it are dropped by the trim.
func TestCompactConversationBudgetContract_OversizedLastMessageExceptionAllStrategies(t *testing.T) {
	msgs := []llm.Message{
		{Role: "user", Content: strings.Repeat("a", 100)},
		{Role: "user", Content: strings.Repeat("b", 100)},
		{Role: "user", Content: strings.Repeat("c", 1000)}, // alone over budget
	}
	const budget = 200

	for _, strategy := range []string{"sliding_window", "summarization", "hierarchical"} {
		t.Run(strategy, func(t *testing.T) {
			counter := &mockTokenCounter{countPerChar: 1}
			calls := 0
			deps := CompactionDeps{Summarize: fixedSizeSummarizer(50, &calls), TokenCounter: counter}

			out, err := CompactConversationHistory(context.Background(), msgs, budget, strategy, CompactionConfig{}, deps)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(out) != 1 {
				t.Fatalf("expected only the oversized last message to remain, got %d: %+v", len(out), out)
			}
			if out[0].Content != msgs[2].Content {
				t.Errorf("last message must be preserved verbatim, got %q", out[0].Content)
			}
			if got := counter.CountMessages(out); got <= budget {
				t.Errorf("sanity: expected the documented over-budget exception, got %d ≤ %d", got, budget)
			}
		})
	}
}

// Case "nil TokenCounter fallback": a positive budget without a counter
// cannot enter token mode — hierarchical must behave exactly as with
// budget=0 (pure count mode): 100 messages split by default ratios into
// distant 40 (one block) + middle 30 (three blocks) + verbatim 30, 4
// summarize calls, 34 output messages, no token trimming.
func TestCompactConversationBudgetContract_NilCounterFallbackHierarchicalMatchesZeroBudget(t *testing.T) {
	msgs := convHistory(50) // 100 messages

	run := func(budget int, calls *int) []llm.Message {
		deps := CompactionDeps{Summarize: fixedSizeSummarizer(50, calls)} // no TokenCounter
		out, err := CompactConversationHistory(context.Background(), msgs, budget, "hierarchical", CompactionConfig{}, deps)
		if err != nil {
			t.Fatalf("budget=%d: unexpected error: %v", budget, err)
		}
		return out
	}

	callsZero, callsPositive := 0, 0
	zero := run(0, &callsZero)
	positive := run(500, &callsPositive)

	if callsZero != 4 || callsPositive != 4 {
		t.Errorf("expected 4 Summarize calls in count mode (1 distant + 3 middle), got %d (budget=0) / %d (budget=500)", callsZero, callsPositive)
	}
	if len(zero) != 34 || len(positive) != 34 {
		t.Fatalf("expected count-based 34 messages (4 summaries + 30 verbatim), got %d (budget=0) / %d (budget=500)", len(zero), len(positive))
	}
	if last := positive[len(positive)-1]; last.Content != "assistant message 49" {
		t.Errorf("last message must survive, got %q", last.Content)
	}
	for i := range zero {
		if zero[i].Role != positive[i].Role || zero[i].Content != positive[i].Content {
			t.Fatalf("nil-counter fallback must match budget=0 byte-for-byte, first difference at %d: %+v != %+v", i, positive[i], zero[i])
		}
	}
}

// Case "defensive paths reachable & correct": the defensive early returns
// inside each strategy ARE reachable — the always-keep-last rule lets a
// verbatim zone exceed its budget share — and must degrade gracefully: no
// zero-count omission note, no wasted Summarize calls, ceiling respected
// except for the documented oversized-last exception. Each subtest pins one
// reachable path:
//
//   - sliding_window, omitted == 0: budget 100 with messages of 15 and 90
//     tokens. The head share (15) covers the first message; the tail share
//     (80) does NOT cover the 90-token last message, but always-keep-last
//     keeps it anyway — head+tail span the whole history with nothing to
//     omit. The ceiling then drops the 15-token head (105 > 100), leaving
//     the 90-token tail.
//   - summarization, tail covers the whole history: a single-message history
//     whose lone message exceeds the budget — always-keep-last makes the
//     tail span everything, so nothing is summarized (zero LLM calls).
//   - hierarchical, distant+middle <= 0: the same single over-budget message
//     — both ratio-derived zones truncate to 0 at n=1 and the always-kept
//     recent zone is that message.
//
// The last two assert the documented never-remove-last exception: the result
// stays over budget because the final message alone exceeds it.
func TestCompactConversationBudgetContract_DefensivePathsReachableAndCorrect(t *testing.T) {
	t.Run("sliding_window omitted==0", func(t *testing.T) {
		msgs := []llm.Message{
			{Role: "user", Content: strings.Repeat("a", 15)}, // fits head share 15
			{Role: "user", Content: strings.Repeat("b", 90)}, // > tail share 80, <= budget
		}
		const budget = 100
		counter := &mockTokenCounter{countPerChar: 1}
		calls := 0
		deps := CompactionDeps{Summarize: fixedSizeSummarizer(50, &calls), TokenCounter: counter}

		out, err := CompactConversationHistory(context.Background(), msgs, budget, "sliding_window", CompactionConfig{}, deps)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(out) != 1 || out[0].Content != msgs[1].Content {
			t.Fatalf("expected the 90-token tail alone after the ceiling, got %d messages: %+v", len(out), out)
		}
		if got := counter.CountMessages(out); got > budget {
			t.Errorf("result exceeds budget: %d > %d", got, budget)
		}
		for _, m := range out {
			if strings.Contains(m.Content, "omitted") {
				t.Errorf("zero-omission must not produce an omission note, got %q", m.Content)
			}
		}
	})

	single := []llm.Message{{Role: "user", Content: strings.Repeat("c", 1000)}} // alone over budget
	for _, strategy := range []string{"summarization", "hierarchical"} {
		t.Run(strategy+" single over-budget message", func(t *testing.T) {
			const budget = 200
			counter := &mockTokenCounter{countPerChar: 1}
			calls := 0
			deps := CompactionDeps{Summarize: fixedSizeSummarizer(50, &calls), TokenCounter: counter}

			out, err := CompactConversationHistory(context.Background(), single, budget, strategy, CompactionConfig{}, deps)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(out) != 1 || out[0].Content != single[0].Content {
				t.Fatalf("expected the lone message verbatim, got %d messages: %+v", len(out), out)
			}
			if calls != 0 {
				t.Errorf("defensive path must not call Summarize, got %d calls", calls)
			}
			if got := counter.CountMessages(out); got <= budget {
				t.Errorf("sanity: expected the documented over-budget exception, got %d <= %d", got, budget)
			}
		})
	}
}
