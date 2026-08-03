package memory

import (
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
