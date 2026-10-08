package memory

import (
	"context"
	"fmt"
	"log/slog"
	"strconv"
	"strings"
	"sync"

	sdkagent "github.com/v0lka/sp4rk/agent"
	"github.com/v0lka/sp4rk/llm"
	"github.com/v0lka/sp4rk/security"
	"github.com/v0lka/sp4rk/strutil"
)

// HierarchicalStrategy divides steps into 3 zones with different compression levels:
// - Distant (oldest): aggressive summarization (large blocks)
// - Middle: moderate summarization (smaller blocks)
// - Recent: kept verbatim
type HierarchicalStrategy struct {
	distantRatio        float64 // fraction of steps in distant zone (default: 0.4)
	middleRatio         float64 // fraction in middle zone (default: 0.3)
	recentRatio         float64 // fraction in recent zone (default: 0.3)
	observationTruncate int     // max chars for observations in summary blocks (default: 500)
	summarizer          func(ctx context.Context, text string) (string, error)
	tokenCounter        llm.TokenCounter
	maxSummarizeTokens  int
	logger              *slog.Logger
	noSummarizerWarn    sync.Once
}

// SetLogger sets the logger for the strategy. If nil, slog.Default() is used.
func (h *HierarchicalStrategy) SetLogger(l *slog.Logger) { h.logger = l }

func (h *HierarchicalStrategy) log() *slog.Logger {
	if h.logger != nil {
		return h.logger
	}
	return slog.New(slog.DiscardHandler)
}

// NewHierarchicalStrategy creates a new HierarchicalStrategy.
// distant, middle, recent are the fractions of steps in each zone.
// They will be normalized to sum to 1.0 if they don't already. Any
// non-positive or non-finite ratio makes the whole set unusable — a negative
// ratio can still sum to exactly 1.0 and survive normalization, yet drive
// negative zone boundaries (slice-bounds panics in Compact) — so the set
// falls back to the defaults (0.4/0.3/0.3), mirroring usableRatio.
// observationTruncate is the max chars for observations in summary blocks (default: 500 if <= 0).
// summarizer is the function to call for LLM summarization; a nil summarizer
// makes Compact keep the step history verbatim (fail-closed) rather than
// discard it — see Compact.
// tokenCounter is optional; if provided, blocks are truncated to maxSummarizeTokens.
// maxSummarizeTokens defaults to 16000 if zero.
func NewHierarchicalStrategy(distant, middle, recent float64, observationTruncate int, summarizer func(ctx context.Context, text string) (string, error), tokenCounter llm.TokenCounter, maxSummarizeTokens int) *HierarchicalStrategy {
	// Normalize ratios, rejecting any set that contains a non-positive or
	// non-finite member (usableRatio): normalization alone cannot repair it,
	// because ratios that already sum to 1.0 are used verbatim.
	if !usableRatio(distant) || !usableRatio(middle) || !usableRatio(recent) {
		// Use defaults
		distant = 0.4
		middle = 0.3
		recent = 0.3
	} else if total := distant + middle + recent; total != 1.0 {
		distant /= total
		middle /= total
		recent /= total
	}

	if observationTruncate <= 0 {
		observationTruncate = 500
	}
	if maxSummarizeTokens <= 0 {
		maxSummarizeTokens = 16000
	}

	return &HierarchicalStrategy{
		distantRatio:        distant,
		middleRatio:         middle,
		recentRatio:         recent,
		observationTruncate: observationTruncate,
		summarizer:          summarizer,
		tokenCounter:        tokenCounter,
		maxSummarizeTokens:  maxSummarizeTokens,
	}
}

// Compact divides steps into 3 zones:
// - Distant (oldest): aggressive summarization (large blocks, ~15 steps per summary)
// - Middle: moderate summarization (smaller blocks, ~5 steps per summary)
// - Recent: kept verbatim
// With a nil summarizer the strategy fails closed: it keeps the whole history
// verbatim instead of replacing the distant/middle zones with bare
// placeholders (the output is frozen as the prompt prefix, so a placeholder
// would permanently discard those steps' thought/action/observation from LLM
// context). The window simply stops shrinking until a summarizer is wired —
// the same contract CompactConversationHistory enforces by returning an error.
func (h *HierarchicalStrategy) Compact(ctx context.Context, steps []sdkagent.Step, budgetTokens int) []llm.Message {
	n := len(steps)

	// For very small step counts, just return all as messages
	if n <= 5 {
		return stepsToMessages(steps)
	}

	// Fail closed when no summarizer is wired (see doc comment). The small
	// history above returns verbatim without needing the LLM.
	if h.summarizer == nil {
		h.noSummarizerWarn.Do(func() {
			h.log().Warn("hierarchical compaction: no summarizer configured; keeping step history verbatim (window will not shrink)")
		})
		return stepsToMessages(steps)
	}

	// Calculate zone boundaries
	distantEnd := int(float64(n) * h.distantRatio)
	middleEnd := int(float64(n) * (h.distantRatio + h.middleRatio))

	// Clamp into [0, n] before the ordering fixes below. The constructor
	// rejects unusable ratios, but a strategy assembled without it (zero-value
	// struct, hand-populated fields) must not produce negative boundaries —
	// the slice expressions below panic on them.
	distantEnd = min(max(distantEnd, 0), n)
	middleEnd = min(max(middleEnd, 0), n)

	// Ensure we have at least something in each zone if there are enough steps
	if distantEnd < 1 && n > 3 {
		distantEnd = 1
	}
	if middleEnd <= distantEnd && n > distantEnd+1 {
		middleEnd = distantEnd + 1
	}
	if middleEnd >= n {
		middleEnd = n - 1
	}
	// Keep boundaries ordered: extreme ratios (e.g. distant=1.0, middle=0,
	// recent=0 passed directly to the constructor) can push distantEnd past
	// the clamped middleEnd, which would panic on the slice expressions below.
	if distantEnd > middleEnd {
		distantEnd = middleEnd
	}

	distantSteps := steps[:distantEnd]
	middleSteps := steps[distantEnd:middleEnd]
	recentSteps := steps[middleEnd:]

	// Estimate messages: ~1 summary per block + recent steps verbatim
	// Each step produces 2 messages (assistant + tool)
	messages := make([]llm.Message, 0, len(steps)*2)

	// Distant zone: aggressive summarization (large blocks of ~15 steps)
	distantBlockSize := 15
	messages = append(messages, h.summarizeZone(ctx, distantSteps, "distant", distantBlockSize)...)

	// Middle zone: moderate summarization (smaller blocks of ~5 steps)
	middleBlockSize := 5
	messages = append(messages, h.summarizeZone(ctx, middleSteps, "middle", middleBlockSize)...)

	// Recent zone: kept verbatim
	messages = append(messages, stepsToMessages(recentSteps)...)

	return messages
}

// summarizeZone summarizes a zone of steps with the given block size.
func (h *HierarchicalStrategy) summarizeZone(ctx context.Context, steps []sdkagent.Step, zoneName string, blockSize int) []llm.Message {
	if len(steps) == 0 {
		return nil
	}

	var messages []llm.Message

	for i := 0; i < len(steps); i += blockSize {
		end := i + blockSize
		if end > len(steps) {
			end = len(steps)
		}
		block := steps[i:end]

		// Build text representation of the block
		blockText := h.buildBlockText(block, zoneName)

		// Truncate to token budget before sending to LLM
		if h.tokenCounter != nil && h.maxSummarizeTokens > 0 {
			tokenCount := h.tokenCounter.Count(blockText)
			if tokenCount > h.maxSummarizeTokens {
				blockText = truncateToTokenBudget(blockText, h.maxSummarizeTokens)
			}
		}

		// Summarize the block (the nil-summarizer case is fail-closed in
		// Compact, so the summarizer is always wired here)
		summary, err := h.summarizer(ctx, blockText)
		if err != nil {
			h.log().Error("hierarchy compaction: summarization failed", "error", err)
			// Fallback to a simple indicator if summarization fails
			summary = fmt.Sprintf("[%s zone: %d steps summarized (error: %v)]", zoneName, len(block), err)
		}

		// Add summary as a system message
		summaryMsg := llm.Message{
			Role:    "system",
			Content: summary,
		}
		messages = append(messages, summaryMsg)
	}

	return messages
}

// buildBlockText creates a text representation of a block of steps for summarization.
func (h *HierarchicalStrategy) buildBlockText(steps []sdkagent.Step, zoneName string) string {
	parts := make([]string, 1, 1+len(steps))
	parts[0] = fmt.Sprintf("Summarize the following %d steps from the %s zone:", len(steps), zoneName)

	for i, step := range steps {
		stepText := "\nStep " + strconv.Itoa(i+1) + ":"
		if step.Thought != "" {
			stepText += "\n  Thought: " + step.Thought
		}
		if step.Action.Name != "" {
			stepText += "\n  Action: " + step.Action.Name
		}
		if step.Observation != "" {
			// Truncate long observations more aggressively for distant zone
			// Distant zone uses 60% of the base truncation value
			maxLen := h.observationTruncate * 6 / 10 // 60% for distant zone
			if zoneName == "middle" {
				maxLen = h.observationTruncate
			}
			obs := step.Observation
			if len(obs) > maxLen {
				obs = strutil.TruncateUTF8(obs, maxLen)
			}
			// Honor the untrusted-content boundary: the summarizer is an LLM
			// context, so untrusted observations must be wrapped before they
			// enter it (mirrors the reflector + ContextWindow defense).
			if step.IsUntrusted {
				obs = security.WrapUntrustedContent(obs, step.Action.Name, nil)
			}
			stepText += "\n  Observation: " + obs
		}
		parts = append(parts, stepText)
	}
	return strings.Join(parts, "")
}
