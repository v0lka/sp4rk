package memory

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/v0lka/sp4rk/llm"
)

// convHistory builds an alternating user/assistant history of n exchanges
// (2n messages), message i carrying a recognizable marker.
func convHistory(n int) []llm.Message {
	msgs := make([]llm.Message, 0, n*2)
	for i := 0; i < n; i++ {
		msgs = append(msgs,
			llm.Message{Role: "user", Content: "user message " + itoa(i)},
			llm.Message{Role: "assistant", Content: "assistant message " + itoa(i)},
		)
	}
	return msgs
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	digits := ""
	for i > 0 {
		digits = string(rune('0'+i%10)) + digits
		i /= 10
	}
	return digits
}

func countingSummarizer(calls *int) func(ctx context.Context, text string) (string, error) {
	return func(_ context.Context, text string) (string, error) {
		if calls != nil {
			*calls++
		}
		return "SUMMARY: " + firstLine(text), nil
	}
}

func firstLine(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return s[:i]
	}
	return s
}

// ---------------------------------------------------------------------------
// CompactConversationHistory — dispatch & guardrails
// ---------------------------------------------------------------------------

func TestCompactConversationHistory_Empty(t *testing.T) {
	out, err := CompactConversationHistory(context.Background(), nil, 0, "sliding_window", CompactionConfig{}, CompactionDeps{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(out) != 0 {
		t.Fatalf("expected empty output, got %d messages", len(out))
	}
}

func TestCompactConversationHistory_UnknownStrategyFailsClosed(t *testing.T) {
	if _, err := CompactConversationHistory(context.Background(), convHistory(3), 0, "bogus", CompactionConfig{}, CompactionDeps{}); err == nil {
		t.Fatal("expected error for unknown strategy")
	}
}

func TestCompactConversationHistory_SummarizationRequiresSummarizer(t *testing.T) {
	if _, err := CompactConversationHistory(context.Background(), convHistory(10), 0, "summarization", CompactionConfig{}, CompactionDeps{}); err == nil {
		t.Fatal("expected error when Summarize is nil for summarization")
	}
	if _, err := CompactConversationHistory(context.Background(), convHistory(10), 0, "hierarchical", CompactionConfig{}, CompactionDeps{}); err == nil {
		t.Fatal("expected error when Summarize is nil for hierarchical")
	}
}

func TestCompactConversationHistory_NeverRemovesLastMessage(t *testing.T) {
	msgs := convHistory(30)
	last := msgs[len(msgs)-1]

	for _, strategy := range []string{"sliding_window", "summarization", "hierarchical"} {
		cfg := CompactionConfig{}
		deps := CompactionDeps{Summarize: countingSummarizer(nil), TokenCounter: llm.NewSimpleTokenCounter()}
		out, err := CompactConversationHistory(context.Background(), msgs, 0, strategy, cfg, deps)
		if err != nil {
			t.Fatalf("strategy %s: unexpected error: %v", strategy, err)
		}
		if len(out) == 0 {
			t.Fatalf("strategy %s: expected non-empty output", strategy)
		}
		got := out[len(out)-1]
		if got.Role != last.Role || got.Content != last.Content {
			t.Errorf("strategy %s: last message changed: got %s:%s want %s:%s", strategy, got.Role, got.Content, last.Role, last.Content)
		}
	}
}

func TestCompactConversationHistory_ShortHistoryNoop(t *testing.T) {
	msgs := convHistory(2) // 4 messages — under every default window
	for _, strategy := range []string{"sliding_window", "summarization"} {
		deps := CompactionDeps{Summarize: countingSummarizer(nil)}
		out, err := CompactConversationHistory(context.Background(), msgs, 0, strategy, CompactionConfig{}, deps)
		if err != nil {
			t.Fatalf("strategy %s: unexpected error: %v", strategy, err)
		}
		if len(out) != len(msgs) {
			t.Errorf("strategy %s: short history must be returned unchanged, got %d of %d messages", strategy, len(out), len(msgs))
		}
	}
}

// ---------------------------------------------------------------------------
// sliding_window
// ---------------------------------------------------------------------------

func TestCompactConversationHistory_SlidingWindow(t *testing.T) {
	msgs := convHistory(20) // 40 messages
	cfg := CompactionConfig{}
	cfg.SlidingWindow.KeepFirst = 2
	cfg.SlidingWindow.KeepLast = 4

	out, err := CompactConversationHistory(context.Background(), msgs, 0, "sliding_window", cfg, CompactionDeps{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Expected: 2 head + 1 note + 4 tail = 7 messages.
	if len(out) != 7 {
		t.Fatalf("expected 7 messages, got %d: %+v", len(out), out)
	}
	if out[0].Content != "user message 0" {
		t.Errorf("head[0] should be first message, got %q", out[0].Content)
	}
	if out[1].Content != "assistant message 0" {
		t.Errorf("head[1] should be second message, got %q", out[1].Content)
	}
	if out[2].Role != "system" || !strings.Contains(out[2].Content, "omitted") {
		t.Errorf("expected omission note at index 2, got %s:%q", out[2].Role, out[2].Content)
	}
	if out[3].Content != "user message 18" {
		t.Errorf("tail[0] should be message index 36 (user message 18), got %q", out[3].Content)
	}
	if out[len(out)-1].Content != "assistant message 19" {
		t.Errorf("last message should be preserved, got %q", out[len(out)-1].Content)
	}
}

func TestCompactConversationHistory_SlidingWindowBudgetTrimsOldestFirst(t *testing.T) {
	msgs := convHistory(20)
	cfg := CompactionConfig{}
	cfg.SlidingWindow.KeepFirst = 2
	cfg.SlidingWindow.KeepLast = 4
	deps := CompactionDeps{TokenCounter: llm.NewSimpleTokenCounter()}

	out, err := CompactConversationHistory(context.Background(), msgs, 25, "sliding_window", cfg, deps)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Tiny budget: only the newest messages survive; the very last is kept.
	if len(out) < 1 || len(out) > 3 {
		t.Fatalf("expected aggressive trim to 1-3 messages, got %d", len(out))
	}
	if out[len(out)-1].Content != "assistant message 19" {
		t.Errorf("last message must survive trimming, got %q", out[len(out)-1].Content)
	}
	if deps.TokenCounter.CountMessages(out) > 25 {
		t.Errorf("result exceeds budget: %d > 25", deps.TokenCounter.CountMessages(out))
	}
}

// ---------------------------------------------------------------------------
// summarization
// ---------------------------------------------------------------------------

func TestCompactConversationHistory_Summarization(t *testing.T) {
	msgs := convHistory(20) // 40 messages
	cfg := CompactionConfig{}
	cfg.Summarization.BlockSize = 10
	cfg.Summarization.KeepLast = 4

	calls := 0
	deps := CompactionDeps{Summarize: countingSummarizer(&calls), TokenCounter: llm.NewSimpleTokenCounter()}

	out, err := CompactConversationHistory(context.Background(), msgs, 0, "summarization", cfg, deps)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// 36 messages to summarize in blocks of 10 → 4 summaries + 4 verbatim tail.
	if calls != 4 {
		t.Errorf("expected 4 summarize calls, got %d", calls)
	}
	if len(out) != 8 {
		t.Fatalf("expected 8 messages (4 summaries + 4 tail), got %d", len(out))
	}
	for i := 0; i < 4; i++ {
		if out[i].Role != "system" || !strings.HasPrefix(out[i].Content, "SUMMARY:") {
			t.Errorf("summary message %d malformed: %s:%q", i, out[i].Role, out[i].Content)
		}
	}
	if out[4].Content != "user message 18" {
		t.Errorf("verbatim tail should start at message 36 (user message 18), got %q", out[4].Content)
	}
	if out[len(out)-1].Content != "assistant message 19" {
		t.Errorf("last message must be preserved, got %q", out[len(out)-1].Content)
	}
}

func TestCompactConversationHistory_SummarizationPropagatesError(t *testing.T) {
	msgs := convHistory(20)
	cfg := CompactionConfig{}
	cfg.Summarization.BlockSize = 10
	cfg.Summarization.KeepLast = 4

	deps := CompactionDeps{
		Summarize: func(_ context.Context, _ string) (string, error) {
			return "", errors.New("llm down")
		},
		TokenCounter: llm.NewSimpleTokenCounter(),
	}
	if _, err := CompactConversationHistory(context.Background(), msgs, 0, "summarization", cfg, deps); err == nil {
		t.Fatal("expected summarize error to propagate")
	}
}

// ---------------------------------------------------------------------------
// hierarchical
// ---------------------------------------------------------------------------

func TestCompactConversationHistory_Hierarchical(t *testing.T) {
	msgs := convHistory(50) // 100 messages
	cfg := CompactionConfig{}
	cfg.Hierarchical.DistantRatio = 0.4
	cfg.Hierarchical.MiddleRatio = 0.3
	cfg.Hierarchical.RecentRatio = 0.3
	cfg.Summarization.BlockSize = 10

	calls := 0
	deps := CompactionDeps{Summarize: countingSummarizer(&calls), TokenCounter: llm.NewSimpleTokenCounter()}

	out, err := CompactConversationHistory(context.Background(), msgs, 0, "hierarchical", cfg, deps)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// distant=40 → 1 block; middle=30 → 3 blocks; recent=30 verbatim.
	if calls != 4 {
		t.Errorf("expected 4 summarize calls (1 distant + 3 middle), got %d", calls)
	}
	if len(out) != 34 {
		t.Fatalf("expected 34 messages (4 summaries + 30 verbatim), got %d", len(out))
	}
	if out[len(out)-1].Content != "assistant message 49" {
		t.Errorf("last message must be preserved, got %q", out[len(out)-1].Content)
	}
	if out[4].Content != "user message 35" {
		t.Errorf("verbatim zone should start at message 70 (user message 35), got %q", out[4].Content)
	}
}

func TestCompactConversationHistory_HierarchicalShortHistoryNoop(t *testing.T) {
	// 2 messages: distant=middle=0 → verbatim, no summarize calls.
	msgs := convHistory(1)
	calls := 0
	deps := CompactionDeps{Summarize: countingSummarizer(&calls)}
	out, err := CompactConversationHistory(context.Background(), msgs, 0, "hierarchical", CompactionConfig{}, deps)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(out) != len(msgs) || calls != 0 {
		t.Errorf("2-message history must stay verbatim with no LLM calls, got %d messages / %d calls", len(out), calls)
	}
}

// ---------------------------------------------------------------------------
// input safety
// ---------------------------------------------------------------------------

func TestCompactConversationHistory_DoesNotMutateInput(t *testing.T) {
	msgs := convHistory(20)
	original := make([]llm.Message, len(msgs))
	copy(original, msgs)

	cfg := CompactionConfig{}
	cfg.SlidingWindow.KeepFirst = 2
	cfg.SlidingWindow.KeepLast = 4
	deps := CompactionDeps{Summarize: countingSummarizer(nil), TokenCounter: llm.NewSimpleTokenCounter()}

	if _, err := CompactConversationHistory(context.Background(), msgs, 0, "summarization", cfg, deps); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for i := range msgs {
		if msgs[i].Role != original[i].Role || msgs[i].Content != original[i].Content {
			t.Fatalf("input mutated at index %d: %+v != %+v", i, msgs[i], original[i])
		}
	}
}

func TestCompactConversationHistory_UserContentBlocksFlattened(t *testing.T) {
	msgs := []llm.Message{
		{Role: "user", Content: "look at this", ContentBlocks: []llm.ContentBlock{
			{Type: "text", Text: "look at this"},
			{Type: "image", ImageB64: "aGVsbG8=", MediaType: "image/png"},
		}},
		{Role: "assistant", Content: "noted"},
		{Role: "user", Content: "next"},
		{Role: "assistant", Content: "reply"},
		{Role: "user", Content: "and more"},
		{Role: "assistant", Content: "final"},
	}
	cfg := CompactionConfig{}
	cfg.Summarization.BlockSize = 2
	cfg.Summarization.KeepLast = 2

	var seenFirst string
	deps := CompactionDeps{
		Summarize: func(_ context.Context, text string) (string, error) {
			if seenFirst == "" {
				seenFirst = text
			}
			return "SUMMARY", nil
		},
		TokenCounter: llm.NewSimpleTokenCounter(),
	}
	if _, err := CompactConversationHistory(context.Background(), msgs, 0, "summarization", cfg, deps); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(seenFirst, "[image attached]") {
		t.Errorf("block text should flatten image blocks, got %q", seenFirst)
	}
}
