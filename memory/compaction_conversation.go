package memory

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"

	"github.com/v0lka/sp4rk/llm"
	"github.com/v0lka/sp4rk/strutil"
)

// compactConversationHistory compacts conversation history messages to fit within
// a token budget. If the total token count of messages exceeds the budget, older
// messages are summarised into a single system message while the most recent
// messages are kept verbatim.
//
// Strategy:
//  1. If messages fit within budget, return them unchanged.
//  2. Otherwise, keep the most recent messages that fit within keepRecentBudget
//     (75% of the total budget), and summarise the older messages into a condensed
//     system message that captures the key conversation flow.
//
// The returned messages are intended for planner prompts; the original history
// remains unmodified.
//
// keepRecentRatio must be strictly between 0 and 1 (exclusive). Values outside
// this range return an error.
func compactConversationHistory(messages []llm.Message, budgetTokens int, tokenCounter llm.TokenCounter, keepRecentRatio float64) ([]llm.Message, error) {
	if keepRecentRatio <= 0 || keepRecentRatio >= 1.0 {
		return nil, fmt.Errorf("keepRecentRatio must be in (0,1), got %f", keepRecentRatio)
	}
	if len(messages) == 0 || budgetTokens <= 0 {
		return messages, nil
	}
	if tokenCounter == nil {
		return nil, errors.New("tokenCounter must not be nil")
	}

	totalTokens := tokenCounter.CountMessages(messages)
	if totalTokens <= budgetTokens {
		return messages, nil
	}

	keepRecentBudget := int(float64(budgetTokens) * keepRecentRatio)

	// Build result from the end, accumulating recent messages that fit.
	// Pre-allocate a temporary slice and append in reverse order (messages[i]
	// down to 0), then reverse once at the end. This is O(n) instead of the
	// O(n²) that repeated prepend (append([]T{x}, s...)) would produce.
	temp := make([]llm.Message, 0, len(messages))
	recentTokens := 0
	for i := len(messages) - 1; i >= 0; i-- {
		msgTokens := tokenCounter.CountMessages([]llm.Message{messages[i]})
		if recentTokens+msgTokens > keepRecentBudget && len(temp) > 0 {
			// This message doesn't fit; all older messages will be summarised.
			break
		}
		recentTokens += msgTokens
		temp = append(temp, messages[i])
	}
	slices.Reverse(temp)
	recent := temp

	// Determine which messages got summarised.
	summarisedCount := len(messages) - len(recent)
	if summarisedCount <= 0 {
		return messages, nil
	}

	// Build a summary of the older messages.
	summary := buildConversationSummary(messages[:summarisedCount])

	// Combine: summary system message + recent messages.
	result := make([]llm.Message, 0, 1+len(recent))
	result = append(result, llm.Message{
		Role:    "system",
		Content: summary,
	})
	result = append(result, recent...)

	// Verify the result fits within the token budget. If it doesn't, trim the
	// recent portion iteratively (oldest first) until it fits or recent is empty.
	// If still over budget after removing all recent messages, truncate the
	// summary string itself as a last resort.
	resultTokens := tokenCounter.CountMessages(result)
	for resultTokens > budgetTokens && len(recent) > 1 {
		// Remove the oldest recent message (index 1 — index 0 is the summary).
		recent = recent[1:]
		result = make([]llm.Message, 0, 1+len(recent))
		result = append(result, llm.Message{
			Role:    "system",
			Content: summary,
		})
		result = append(result, recent...)
		resultTokens = tokenCounter.CountMessages(result)
	}

	// If still over budget and recent is empty/1, truncate the summary.
	// Halve the summary iteratively until the budget is met or the summary
	// becomes too short to halve further.
	for resultTokens > budgetTokens {
		half := len(summary) / 2
		if half == 0 {
			break // summary is too short to truncate further
		}
		truncated := strutil.TruncateUTF8(summary, half)
		if truncated == summary {
			break // no further progress possible (byte-halving converged on rune boundary)
		}
		summary = truncated
		result[0].Content = summary
		resultTokens = tokenCounter.CountMessages(result)
	}

	return result, nil
}

// ---------------------------------------------------------------------------
// Exported manual conversation-history compaction
// ---------------------------------------------------------------------------

// CompactConversationHistory compacts a plain conversation history ([]llm.Message
// of user/assistant exchanges) using the named strategy, returning the compacted
// messages. Unlike the step-based strategies (which operate on []agent.Step inside
// a ContextWindow), this API serves callers that hold raw conversation histories —
// e.g. a session orchestrator compacting cross-task dialogue context on demand.
//
// Supported strategies (same names as NewCompactionStrategy):
//
//   - "sliding_window": keep the first cfg.SlidingWindow.KeepFirst and the last
//     cfg.SlidingWindow.KeepLast messages verbatim; the middle is replaced by a
//     short system note. No LLM call.
//   - "summarization": keep the last cfg.Summarization.KeepLast messages verbatim;
//     older messages are grouped into cfg.Summarization.BlockSize blocks, each
//     summarized via deps.Summarize into a system message.
//   - "hierarchical": split by cfg.Hierarchical ratios — the distant zone is
//     aggressively summarized into ONE block, the middle zone is summarized per
//     block (cfg.Summarization.BlockSize), the recent zone is kept verbatim.
//
// Invariants:
//   - The very last message is NEVER removed (it anchors the ongoing exchange).
//   - budgetTokens (> 0) is a hard ceiling: the result is trimmed oldest-first
//     until it fits (a nil deps.TokenCounter skips trimming). budgetTokens <= 0
//     means no budget.
//   - Unknown strategies fail closed with an error.
//   - LLM-backed strategies require a non-nil deps.Summarize and propagate its
//     error (the original messages are the caller's to keep — this function
//     never mutates msgs).
//
// Zero-valued cfg fields fall back to the same defaults as NewCompactionStrategy.
func CompactConversationHistory(ctx context.Context, msgs []llm.Message, budgetTokens int, strategy string, cfg CompactionConfig, deps CompactionDeps) ([]llm.Message, error) {
	if len(msgs) == 0 {
		return msgs, nil
	}
	switch strategy {
	case "sliding_window":
		return compactConversationSliding(msgs, budgetTokens, cfg, deps), nil
	case "summarization":
		return compactConversationSummarizing(ctx, msgs, budgetTokens, cfg, deps)
	case "hierarchical":
		return compactConversationHierarchical(ctx, msgs, budgetTokens, cfg, deps)
	default:
		return nil, fmt.Errorf("memory: unknown conversation compaction strategy %q", strategy)
	}
}

// compactConversationSliding implements the sliding-window strategy for message
// histories: verbatim head + omission note + verbatim tail. No LLM calls.
func compactConversationSliding(msgs []llm.Message, budgetTokens int, cfg CompactionConfig, deps CompactionDeps) []llm.Message {
	keepFirst := cfg.SlidingWindow.KeepFirst
	if keepFirst <= 0 {
		keepFirst = 3
	}
	keepLast := cfg.SlidingWindow.KeepLast
	if keepLast <= 0 {
		keepLast = 10
	}
	if len(msgs) <= keepFirst+keepLast {
		return msgs
	}
	omitted := len(msgs) - keepFirst - keepLast
	result := make([]llm.Message, 0, keepFirst+1+keepLast)
	result = append(result, msgs[:keepFirst]...)
	result = append(result, llm.Message{
		Role: "system",
		Content: fmt.Sprintf("[... %d earlier conversation messages omitted by sliding-window compaction ...]",
			omitted),
	})
	result = append(result, msgs[len(msgs)-keepLast:]...)
	return trimConversationToBudget(result, budgetTokens, deps.TokenCounter)
}

// compactConversationSummarizing implements the summarization strategy for
// message histories: per-block LLM summaries of older messages + verbatim tail.
func compactConversationSummarizing(ctx context.Context, msgs []llm.Message, budgetTokens int, cfg CompactionConfig, deps CompactionDeps) ([]llm.Message, error) {
	blockSize := cfg.Summarization.BlockSize
	if blockSize <= 0 {
		blockSize = 10
	}
	keepLast := cfg.Summarization.KeepLast
	if keepLast <= 0 {
		keepLast = 5
	}
	if deps.Summarize == nil {
		return nil, errors.New("memory: conversation summarization requires a non-nil Summarize dependency")
	}
	if len(msgs) <= keepLast {
		return msgs, nil
	}
	numToSummarize := len(msgs) - keepLast
	blocks, err := summarizeConversationBlocks(ctx, msgs[:numToSummarize], blockSize, cfg, deps)
	if err != nil {
		return nil, err
	}
	result := make([]llm.Message, 0, len(blocks)+keepLast)
	result = append(result, blocks...)
	result = append(result, msgs[numToSummarize:]...)
	return trimConversationToBudget(result, budgetTokens, deps.TokenCounter), nil
}

// compactConversationHierarchical implements the hierarchical strategy for
// message histories: one aggressive summary over the distant zone, per-block
// summaries over the middle zone, verbatim recent zone.
func compactConversationHierarchical(ctx context.Context, msgs []llm.Message, budgetTokens int, cfg CompactionConfig, deps CompactionDeps) ([]llm.Message, error) {
	distantRatio := cfg.Hierarchical.DistantRatio
	if distantRatio <= 0 {
		distantRatio = 0.4
	}
	middleRatio := cfg.Hierarchical.MiddleRatio
	if middleRatio <= 0 {
		middleRatio = 0.3
	}
	if deps.Summarize == nil {
		return nil, errors.New("memory: hierarchical conversation compaction requires a non-nil Summarize dependency")
	}

	n := len(msgs)
	distant := int(float64(n) * distantRatio)
	middle := int(float64(n) * middleRatio)
	// The recent zone must keep at least the final message; shrink the middle
	// (then the distant) zone first when the ratios over-cover a short history.
	if distant+middle >= n {
		middle = max(n-distant-1, 0)
	}
	if distant+middle >= n {
		distant = max(n-1-middle, 0)
	}
	if distant+middle <= 0 {
		return msgs, nil
	}

	var result []llm.Message
	// Distant zone: ONE aggressive summary block covering the whole zone.
	distantBlocks, err := summarizeConversationBlocks(ctx, msgs[:distant], distant, cfg, deps)
	if err != nil {
		return nil, err
	}
	result = append(result, distantBlocks...)

	// Middle zone: per-block summaries.
	blockSize := cfg.Summarization.BlockSize
	if blockSize <= 0 {
		blockSize = 10
	}
	middleBlocks, err := summarizeConversationBlocks(ctx, msgs[distant:distant+middle], blockSize, cfg, deps)
	if err != nil {
		return nil, err
	}
	result = append(result, middleBlocks...)

	// Recent zone: verbatim.
	result = append(result, msgs[distant+middle:]...)
	return trimConversationToBudget(result, budgetTokens, deps.TokenCounter), nil
}

// summarizeConversationBlocks groups msgs into consecutive blocks of blockSize
// and summarizes each block via deps.Summarize, returning one system message
// per block. Block text is bounded per message by ObservationTruncate and as a
// whole by MaxSummarizeTokens (when a token counter is available).
func summarizeConversationBlocks(ctx context.Context, msgs []llm.Message, blockSize int, cfg CompactionConfig, deps CompactionDeps) ([]llm.Message, error) {
	truncateChars := cfg.Summarization.ObservationTruncate
	if truncateChars <= 0 {
		truncateChars = 500
	}
	maxSummarizeTokens := deps.MaxSummarizeTokens
	if maxSummarizeTokens <= 0 {
		maxSummarizeTokens = 16000
	}

	result := make([]llm.Message, 0, len(msgs)/max(blockSize, 1)+1)
	for i := 0; i < len(msgs); i += blockSize {
		end := min(i+blockSize, len(msgs))
		blockText := conversationBlockText(msgs[i:end], truncateChars)
		if deps.TokenCounter != nil {
			if count := deps.TokenCounter.Count(blockText); count > maxSummarizeTokens {
				blockText = truncateToTokenBudget(blockText, maxSummarizeTokens)
			}
		}
		summary, err := deps.Summarize(ctx, blockText)
		if err != nil {
			return nil, fmt.Errorf("memory: conversation summarization failed (block %d-%d): %w", i+1, end, err)
		}
		result = append(result, llm.Message{Role: "system", Content: summary})
	}
	return result, nil
}

// conversationBlockText renders a block of conversation messages as plain text
// for LLM summarization ("User: ..."/"Assistant: ..." lines). Each message's
// text is capped at truncateChars (UTF-8 aware); user messages with content
// blocks are flattened via userMessageText.
func conversationBlockText(msgs []llm.Message, truncateChars int) string {
	var b strings.Builder
	for _, m := range msgs {
		text := m.Content
		if m.Role == "user" {
			text = userMessageText(m)
		}
		if truncateChars > 0 && len(text) > truncateChars {
			text = strutil.TruncateUTF8(text, truncateChars)
		}
		b.WriteString(m.Role)
		b.WriteString(": ")
		b.WriteString(text)
		b.WriteByte('\n')
	}
	return b.String()
}

// trimConversationToBudget enforces the token ceiling by dropping the oldest
// messages first. It always keeps at least the final message. A nil counter or
// a non-positive budget disables trimming.
func trimConversationToBudget(result []llm.Message, budgetTokens int, counter llm.TokenCounter) []llm.Message {
	if budgetTokens <= 0 || counter == nil || len(result) <= 1 {
		return result
	}
	for counter.CountMessages(result) > budgetTokens && len(result) > 1 {
		result = result[1:]
	}
	return result
}

// buildConversationSummary creates a condensed text summary of conversation messages.
// It extracts user requests and key assistant outcomes, formatting them as a
// structured chronological summary. The output is capped at ~1500 chars to prevent
// the summary itself from consuming too many tokens.
func buildConversationSummary(messages []llm.Message) string {
	if len(messages) == 0 {
		return ""
	}

	const maxSummaryChars = 1500

	var b strings.Builder
	b.WriteString("Previous conversation history (summarised):\n")

	exchangeNum := 0
outer:
	for i := 0; i < len(messages); i++ {
		msg := messages[i]
		switch msg.Role {
		case "user":
			exchangeNum++
			summary := truncateSummaryContent(userMessageText(msg), 120)
			line := fmt.Sprintf("%d. User: %s\n", exchangeNum, summary)
			if b.Len()+len(line) > maxSummaryChars {
				break outer
			}
			b.WriteString(line)
		case "assistant":
			if exchangeNum > 0 {
				summary := truncateSummaryContent(msg.Content, 80)
				line := fmt.Sprintf("   Assistant: %s\n", summary)
				if b.Len()+len(line) > maxSummaryChars {
					break outer
				}
				b.WriteString(line)
			}
		}
	}

	if exchangeNum == 0 {
		return "[No previous user messages to summarise]"
	}

	return b.String()
}

// userMessageText extracts the textual representation of a user message for
// inclusion in a conversation summary. When the message carries structured
// content blocks, text blocks are concatenated and image blocks are replaced
// with the placeholder "[image attached]" (image data is not useful in a text
// summary); unknown block types are skipped (matching provider behavior).
// When ContentBlocks is empty, the plain Content string is used as before
// (backward compatible for text-only messages).
func userMessageText(msg llm.Message) string {
	blocks := llm.NormalizeContentBlocks(msg)
	if blocks == nil {
		return msg.Content
	}
	var b strings.Builder
	for _, blk := range blocks {
		switch blk.Type {
		case "text":
			b.WriteString(blk.Text)
		case "image":
			b.WriteString("[image attached]")
		default:
			// Unknown block types are skipped (consistent with providers).
		}
	}
	return b.String()
}

// truncateSummaryContent truncates text to maxChars, appending "…" when truncated.
// Truncation is UTF-8 aware: it never splits a multi-byte rune.
func truncateSummaryContent(text string, maxChars int) string {
	return strutil.TruncateUTF8(text, maxChars)
}
