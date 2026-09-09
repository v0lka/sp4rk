package memory

import (
	"context"
	"strings"
	"testing"

	"github.com/v0lka/sp4rk/llm"
)

// This file pins the STRUCTURAL contract of CompactConversationHistory and the
// prediction contract of PredictCompaction. CompactConversationHistory is now
// purely structural (message-count windows, no token-budget sizing), and its
// no-op decision must AGREE with PredictCompaction.WillCompact for the same
// history and config. PredictCompaction is a pure, no-LLM, no-mutation
// function; the tests below pin its exactness for sliding_window and its
// forecast-only nature for the LLM-backed strategies.

// uniformMsgs builds n user messages whose Content is exactly charsPerMsg
// bytes long, so under mockTokenCounter{countPerChar: 1} each message costs
// exactly charsPerMsg tokens (CountMessages counts Content only).
func uniformMsgs(n, charsPerMsg int) []llm.Message {
	msgs := make([]llm.Message, 0, n)
	for i := 0; i < n; i++ {
		marker := "m" + itoa(i) + " "
		msgs = append(msgs, llm.Message{
			Role:    "user",
			Content: marker + strings.Repeat("x", charsPerMsg-len(marker)),
		})
	}
	return msgs
}

// mockTokenCounter is declared in compaction_test.go; referenced here for
// deterministic one-token-per-char counting.

// A short 6-message dialog must be compacted by sliding_window's count window
// only when len > keepFirst+keepLast; with the default 3+10 window a 6-message
// history is a no-op (structural mode — no token-budget overrides the counts).
func TestCompactConversationStructural_SlidingCountWindow(t *testing.T) {
	msgs := uniformMsgs(6, 100)

	out, err := CompactConversationHistory(context.Background(), msgs, "sliding_window", CompactionConfig{}, CompactionDeps{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(out) != len(msgs) {
		t.Fatalf("6 messages are within the default 3+10 window — must be unchanged, got %d of %d", len(out), len(msgs))
	}

	// 40 messages exceed 3+10 → windowed to 3 + note + 10 = 14.
	long := uniformMsgs(40, 100)
	out, err = CompactConversationHistory(context.Background(), long, "sliding_window", CompactionConfig{}, CompactionDeps{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(out) != 14 {
		t.Fatalf("expected count-based window of 14 messages, got %d", len(out))
	}
	if out[0].Content != long[0].Content || out[len(out)-1].Content != long[39].Content {
		t.Errorf("window endpoints wrong: first %q, last %q", out[0].Content, out[len(out)-1].Content)
	}
}

// Count-based summarization: 40 messages, keepLast=4 → 36 summarized in blocks
// of 10 → 4 summaries + 4 verbatim tail. No token counter anywhere.
func TestCompactConversationStructural_SummarizationCountMode(t *testing.T) {
	msgs := convHistory(20) // 40 messages
	cfg := CompactionConfig{}
	cfg.Summarization.BlockSize = 10
	cfg.Summarization.KeepLast = 4

	calls := 0
	deps := CompactionDeps{Summarize: countingSummarizer(&calls)}
	out, err := CompactConversationHistory(context.Background(), msgs, "summarization", cfg, deps)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if calls != 4 || len(out) != 8 {
		t.Errorf("expected 4 calls and 8 messages (4 summaries + 4 tail), got %d calls, %d messages", calls, len(out))
	}
}

// ---------------------------------------------------------------------------
// PredictCompaction — WillCompact is EXACT and agrees with the structural run
// ---------------------------------------------------------------------------

func TestPredictCompaction_UnknownStrategyFailsClosed(t *testing.T) {
	if _, err := PredictCompaction(convHistory(3), "bogus", CompactionConfig{}, CompactionDeps{}); err == nil {
		t.Fatal("expected error for unknown strategy")
	}
}

func TestPredictCompaction_EmptyHistory(t *testing.T) {
	for _, strategy := range []string{"sliding_window", "summarization", "hierarchical"} {
		p, err := PredictCompaction(nil, strategy, CompactionConfig{}, CompactionDeps{})
		if err != nil {
			t.Fatalf("strategy %s: unexpected error: %v", strategy, err)
		}
		if p.WillCompact {
			t.Errorf("strategy %s: empty history must not predict compaction", strategy)
		}
		if !p.Exact {
			t.Errorf("strategy %s: empty history is trivially exact", strategy)
		}
	}
}

// WillCompact must agree with whether CompactConversationHistory actually
// changes the history, across all three strategies, for a short (no-op) and a
// long (compactable) history.
func TestPredictCompaction_WillCompactAgreesWithRun(t *testing.T) {
	counter := llm.NewSimpleTokenCounter()

	cases := []struct {
		name       string
		msgs       []llm.Message
		cfg        CompactionConfig
		strategies []string
		want       bool
	}{
		{
			name:       "short history under every window",
			msgs:       convHistory(1), // 2 messages: no-op for ALL three (hierarchical zones empty)
			strategies: []string{"sliding_window", "summarization", "hierarchical"},
			want:       false,
		},
		{
			name:       "long history over every window",
			msgs:       convHistory(50), // 100 messages
			strategies: []string{"sliding_window", "summarization", "hierarchical"},
			want:       true,
		},
	}
	for _, tc := range cases {
		for _, strategy := range tc.strategies {
			t.Run(tc.name+"/"+strategy, func(t *testing.T) {
				deps := CompactionDeps{Summarize: countingSummarizer(nil), TokenCounter: counter}
				p, err := PredictCompaction(tc.msgs, strategy, tc.cfg, deps)
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				if p.WillCompact != tc.want {
					t.Fatalf("WillCompact = %v, want %v", p.WillCompact, tc.want)
				}

				out, err := CompactConversationHistory(context.Background(), tc.msgs, strategy, tc.cfg, deps)
				if err != nil {
					t.Fatalf("unexpected run error: %v", err)
				}
				changed := len(out) != len(tc.msgs)
				if !changed {
					for i := range out {
						if out[i].Role != tc.msgs[i].Role || out[i].Content != tc.msgs[i].Content {
							changed = true
							break
						}
					}
				}
				if changed != tc.want {
					t.Fatalf("structural run changed=%v, prediction WillCompact=%v — must agree", changed, p.WillCompact)
				}
			})
		}
	}
}

// sliding_window's prediction is an EXACT dry-run: the predicted AfterTokens
// must equal the token count of the actually compacted result, and Reclaim
// must equal Before-After.
func TestPredictCompaction_SlidingWindowExactDryRun(t *testing.T) {
	counter := llm.NewSimpleTokenCounter()
	msgs := convHistory(50) // 100 messages
	cfg := CompactionConfig{}
	cfg.SlidingWindow.KeepFirst = 2
	cfg.SlidingWindow.KeepLast = 4
	deps := CompactionDeps{TokenCounter: counter}

	p, err := PredictCompaction(msgs, "sliding_window", cfg, deps)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !p.WillCompact {
		t.Fatal("100-message history must predict compaction")
	}
	if !p.Exact {
		t.Fatal("sliding_window prediction must be exact")
	}

	out, err := CompactConversationHistory(context.Background(), msgs, "sliding_window", cfg, CompactionDeps{})
	if err != nil {
		t.Fatalf("unexpected run error: %v", err)
	}
	actualAfter := counter.CountMessages(out)
	if p.AfterTokens != actualAfter {
		t.Errorf("sliding_window dry-run AfterTokens = %d, actual = %d", p.AfterTokens, actualAfter)
	}
	if p.Reclaim != p.BeforeTokens-p.AfterTokens {
		t.Errorf("Reclaim %d != Before %d - After %d", p.Reclaim, p.BeforeTokens, p.AfterTokens)
	}
	if p.Reclaim <= 0 {
		t.Errorf("a compactable history must predict positive reclaim, got %d", p.Reclaim)
	}
}

// The LLM-backed strategies' predictions are forecasts, not dry-runs: WillCompact
// is exact, but AfterTokens is a forecast (Exact=false) driven by the
// CompactionForecast ratios, never affecting WillCompact.
func TestPredictCompaction_LLMStrategiesAreForecasts(t *testing.T) {
	counter := llm.NewSimpleTokenCounter()
	msgs := convHistory(50) // 100 messages

	for _, strategy := range []string{"summarization", "hierarchical"} {
		deps := CompactionDeps{Summarize: countingSummarizer(nil), TokenCounter: counter}
		p, err := PredictCompaction(msgs, strategy, CompactionConfig{}, deps)
		if err != nil {
			t.Fatalf("strategy %s: unexpected error: %v", strategy, err)
		}
		if !p.WillCompact {
			t.Fatalf("strategy %s: 100-message history must predict compaction", strategy)
		}
		if p.Exact {
			t.Errorf("strategy %s: LLM-backed prediction must NOT be exact", strategy)
		}
		if p.Reclaim <= 0 {
			t.Errorf("strategy %s: must predict positive reclaim, got %d", strategy, p.Reclaim)
		}
		// The forecast ratios only scale AfterTokens; they never flip WillCompact.
		forecasted := CompactionDeps{
			Summarize:    countingSummarizer(nil),
			TokenCounter: counter,
			Forecast: CompactionForecast{
				SummarizationRatio:       0.05,
				HierarchicalDistantRatio: 0.05,
				HierarchicalMiddleRatio:  0.05,
			},
		}
		p2, err := PredictCompaction(msgs, strategy, CompactionConfig{}, forecasted)
		if err != nil {
			t.Fatalf("strategy %s: unexpected error: %v", strategy, err)
		}
		if p2.WillCompact != p.WillCompact {
			t.Errorf("strategy %s: forecast ratio must not affect WillCompact", strategy)
		}
		if p2.AfterTokens >= p.AfterTokens {
			t.Errorf("strategy %s: a lower ratio must predict a smaller AfterTokens (%d >= %d)", strategy, p2.AfterTokens, p.AfterTokens)
		}
	}
}

// PredictCompaction is pure: it mutates neither the input history nor any
// shared state, and never calls the LLM.
func TestPredictCompaction_PureAndNoLLM(t *testing.T) {
	msgs := convHistory(20)
	original := make([]llm.Message, len(msgs))
	copy(original, msgs)

	calls := 0
	deps := CompactionDeps{Summarize: countingSummarizer(&calls), TokenCounter: llm.NewSimpleTokenCounter()}
	if _, err := PredictCompaction(msgs, "hierarchical", CompactionConfig{}, deps); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if calls != 0 {
		t.Fatalf("prediction must never call the LLM, got %d calls", calls)
	}
	for i := range msgs {
		if msgs[i].Role != original[i].Role || msgs[i].Content != original[i].Content {
			t.Fatalf("prediction mutated input at index %d", i)
		}
	}
}

// Without a token counter the prediction still reports the exact WillCompact
// verdict; only the token fields carry "unknown" (zero).
func TestPredictCompaction_NilCounterStillExactWillCompact(t *testing.T) {
	p, err := PredictCompaction(convHistory(50), "sliding_window", CompactionConfig{}, CompactionDeps{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !p.WillCompact {
		t.Fatal("100-message history must predict compaction even without a counter")
	}
	if p.BeforeTokens != 0 {
		t.Errorf("nil counter must report 0 tokens, got %d", p.BeforeTokens)
	}
}
