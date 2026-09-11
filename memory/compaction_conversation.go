package memory

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strings"

	"github.com/v0lka/sp4rk/llm"
	"github.com/v0lka/sp4rk/strutil"
)

// unknownConversationStrategyFmt is the single source of the unknown-strategy
// error text: the upfront validation and the switch's default arm both format
// it with the offending strategy name, so the two can never drift apart.
const unknownConversationStrategyFmt = "memory: unknown conversation compaction strategy %q"

// ---------------------------------------------------------------------------
// Exported manual conversation-history compaction
// ---------------------------------------------------------------------------

// CompactConversationHistory compacts a plain conversation history ([]llm.Message
// of user/assistant exchanges) using the named strategy, returning the compacted
// messages. Unlike the step-based strategies (which operate on []agent.Step inside
// a ContextWindow), this API serves callers that hold raw conversation histories —
// e.g. a session orchestrator compacting cross-task dialogue context on demand.
//
// This is a PURELY STRUCTURAL operation: each strategy applies its configured
// message-count window exactly, with no token-budget sizing and no token trim.
// The no-op decision is therefore made on message counts alone — a history the
// strategy leaves unchanged (too short for its window) is returned verbatim.
//
// Supported strategies (same names as NewCompactionStrategy):
//
//   - "sliding_window": keep the first cfg.SlidingWindow.KeepFirst and the last
//     cfg.SlidingWindow.KeepLast messages verbatim; the middle is replaced by a
//     short system note. No LLM call. Verbatim when
//     len(msgs) <= KeepFirst+KeepLast.
//   - "summarization": keep the last cfg.Summarization.KeepLast messages verbatim;
//     older messages are grouped into cfg.Summarization.BlockSize blocks, each
//     summarized via deps.Summarize into a system message. Verbatim when
//     len(msgs) <= KeepLast.
//   - "hierarchical": split by cfg.Hierarchical ratios — the distant zone is
//     aggressively summarized into ONE block, the middle zone is summarized per
//     block (cfg.Summarization.BlockSize), the recent zone is kept verbatim.
//     Verbatim when both summary zones are empty.
//
// Invariants:
//   - The very last message is NEVER removed (it anchors the ongoing exchange).
//   - The input slice is never mutated.
//   - Unknown strategies fail closed with an error (no silent fallback).
//   - LLM-backed strategies require a non-nil deps.Summarize and propagate its
//     error — but ONLY when there is actually something to summarize: a history
//     the strategy would return verbatim needs no LLM, so a nil Summarize is
//     not an error for it.
//
// Zero-valued cfg fields fall back to the same defaults as NewCompactionStrategy.
func CompactConversationHistory(ctx context.Context, msgs []llm.Message, strategy string, cfg CompactionConfig, deps CompactionDeps) ([]llm.Message, error) {
	if len(msgs) == 0 {
		return msgs, nil
	}
	if strategy != "sliding_window" && strategy != "summarization" && strategy != "hierarchical" {
		return nil, fmt.Errorf(unknownConversationStrategyFmt, strategy)
	}

	switch strategy {
	case "sliding_window":
		return compactConversationSliding(msgs, cfg), nil
	case "summarization":
		return compactConversationSummarizing(ctx, msgs, cfg, deps)
	case "hierarchical":
		return compactConversationHierarchical(ctx, msgs, cfg, deps)
	default:
		// Unreachable (validation above fails closed on unknown strategies);
		// kept so the switch stays exhaustive against future case additions.
		return nil, fmt.Errorf(unknownConversationStrategyFmt, strategy)
	}
}

// compactConversationSliding implements the sliding-window strategy for message
// histories: verbatim head + omission note + verbatim tail. No LLM calls.
func compactConversationSliding(msgs []llm.Message, cfg CompactionConfig) []llm.Message {
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
		Role:    "system",
		Content: fmt.Sprintf("[... %d earlier conversation messages omitted by sliding-window compaction ...]", omitted),
	})
	result = append(result, msgs[len(msgs)-keepLast:]...)
	return result
}

// compactConversationSummarizing implements the summarization strategy for
// message histories: per-block LLM summaries of older messages + verbatim tail.
func compactConversationSummarizing(ctx context.Context, msgs []llm.Message, cfg CompactionConfig, deps CompactionDeps) ([]llm.Message, error) {
	blockSize := cfg.Summarization.BlockSize
	if blockSize <= 0 {
		blockSize = 10
	}
	keepLast := cfg.Summarization.KeepLast
	if keepLast <= 0 {
		keepLast = 5
	}
	// A history the strategy returns verbatim needs no LLM: short-circuit
	// BEFORE the Summarize nil check.
	if len(msgs) <= keepLast {
		return msgs, nil
	}
	if deps.Summarize == nil {
		return nil, errors.New("memory: conversation summarization requires a non-nil Summarize dependency")
	}
	numToSummarize := len(msgs) - keepLast
	blocks, err := summarizeConversationBlocks(ctx, msgs[:numToSummarize], blockSize, cfg, deps)
	if err != nil {
		return nil, err
	}
	result := make([]llm.Message, 0, len(blocks)+keepLast)
	result = append(result, blocks...)
	result = append(result, msgs[numToSummarize:]...)
	return result, nil
}

// compactConversationHierarchical implements the hierarchical strategy for
// message histories: one aggressive summary over the distant zone, per-block
// summaries over the middle zone, verbatim recent zone.
func compactConversationHierarchical(ctx context.Context, msgs []llm.Message, cfg CompactionConfig, deps CompactionDeps) ([]llm.Message, error) {
	n := len(msgs)
	distant, middle := conversationHierarchicalZones(n, cfg)
	if distant+middle <= 0 {
		// Nothing to summarize — both zones are empty, so the strategy is a
		// no-op and no LLM dependency is needed.
		return msgs, nil
	}
	if deps.Summarize == nil {
		return nil, errors.New("memory: hierarchical conversation compaction requires a non-nil Summarize dependency")
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
	return result, nil
}

// conversationHierarchicalZones computes the distant/middle zone sizes for an
// n-message history under the hierarchical strategy, mirroring the zone math of
// HierarchicalStrategy.Compact: the three ratios (distant/middle/recent) are
// normalized to sum to 1.0, the distant zone is int(n·distant), and the middle
// zone spans up to the CUMULATIVE boundary middleEnd = int(n·(distant+middle)),
// leaving the recent zone as the remainder n − middleEnd. Ratios ≤ 0 — or
// non-finite (YAML ".nan"/".inf" parses to NaN/Inf, whose int conversion is
// implementation-defined per the Go spec) — fall back to sp4rk's defaults
// (0.4/0.3/0.3). The clamps shrink the middle (then the distant) zone so the
// recent zone always keeps at least the final message, and a history too short
// to form a distant zone is a no-op (both zones empty).
func conversationHierarchicalZones(n int, cfg CompactionConfig) (distant, middle int) {
	distantRatio := cfg.Hierarchical.DistantRatio
	if !usableRatio(distantRatio) {
		distantRatio = 0.4
	}
	middleRatio := cfg.Hierarchical.MiddleRatio
	if !usableRatio(middleRatio) {
		middleRatio = 0.3
	}
	recentRatio := cfg.Hierarchical.RecentRatio
	if !usableRatio(recentRatio) {
		recentRatio = 0.3
	}
	// Normalize the three ratios to sum to 1.0, mirroring NewHierarchicalStrategy.
	// The recent ratio participates only via the denominator: the recent zone is
	// the remainder after the cumulative middle boundary (n − middleEnd).
	if total := distantRatio + middleRatio + recentRatio; total > 0 && total != 1.0 {
		distantRatio /= total
		middleRatio /= total
	}

	distant = int(float64(n) * distantRatio)
	// Cumulative boundary (mirrors HierarchicalStrategy.Compact): the middle
	// zone spans [distant, middleEnd), NOT an independent int(n·middle) slice —
	// the independent rounding diverges from the step-based path even at the
	// default ratios (e.g. n=6 → {2,1,3} instead of {2,2,2}).
	middleEnd := int(float64(n) * (distantRatio + middleRatio))
	// The recent zone must keep at least the final message; shrink the middle
	// (then the distant) zone first when the ratios over-cover a short history.
	if middleEnd >= n {
		middleEnd = n - 1
	}
	if distant > middleEnd {
		distant = middleEnd
	}
	// No distant zone ⇒ no hierarchy to compress: keep the whole history
	// verbatim. This also stops the cumulative rounding from inventing a
	// single-message middle zone on a two-message history (int(n·(d+m)) rounds
	// up to 1 while int(n·d) and int(n·m) both round to 0).
	if distant == 0 {
		return 0, 0
	}
	middle = middleEnd - distant
	return distant, middle
}

// usableRatio reports whether a ratio can drive the compaction math — either a
// configured hierarchical zone ratio or a forecast compression ratio: it must
// be finite (NaN/Inf — which YAML ".nan"/".inf" values parse to, or a host
// calibration with a zero denominator — convert to int implementation-defined-
// ly per the Go spec) and positive. Anything else falls back to the defaults.
func usableRatio(r float64) bool {
	return r > 0 && !math.IsNaN(r) && !math.IsInf(r, 0)
}

// ConversationHierarchicalZones returns the distant/middle zone sizes for an
// n-message history under the hierarchical strategy, using the same normalized
// cumulative zone math as CompactConversationHistory and PredictCompaction.
// Callers that attribute a compaction's output to its zones (e.g. per-zone
// observed compression ratios for EWMA forecast calibration) can derive the
// zone boundaries without duplicating the math.
func ConversationHierarchicalZones(n int, cfg CompactionConfig) (distant, middle int) {
	return conversationHierarchicalZones(n, cfg)
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

// ---------------------------------------------------------------------------
// Compaction prediction (pure, no LLM, no mutation)
// ---------------------------------------------------------------------------

// Default compression-ratio forecasts for the LLM-backed strategies, applied
// when the corresponding CompactionForecast field is zero. They are
// conservative: a summary is assumed to occupy roughly a third of its input
// (hierarchical's distant zone is more aggressive — one summary over a large
// block). The ratio affects ONLY the predicted AfterTokens/Reclaim, never the
// exact WillCompact verdict.
const (
	defaultSummarizationForecastRatio       = 0.3
	defaultHierarchicalDistantForecastRatio = 0.15
	defaultHierarchicalMiddleForecastRatio  = 0.3
)

// CompactionForecast supplies the compression-ratio forecasts used by
// PredictCompaction to estimate the post-summary token count of the
// LLM-backed strategies. Zero values fall back to the conservative defaults
// above. It is ignored by CompactConversationHistory and the step-based
// strategies — it exists only to feed the prediction.
type CompactionForecast struct {
	// SummarizationRatio is the expected fraction of a summarized block's input
	// tokens that the summary occupies (0 < ratio <= 1). Used for the
	// "summarization" strategy and hierarchical's middle zone.
	SummarizationRatio float64
	// HierarchicalDistantRatio is the expected fraction for hierarchical's
	// distant zone (one aggressive summary over a large block).
	HierarchicalDistantRatio float64
	// HierarchicalMiddleRatio is the expected fraction for hierarchical's
	// middle zone (per-block summaries).
	HierarchicalMiddleRatio float64
}

// CompactionPrediction reports the expected effect of compacting a message
// history with a strategy, WITHOUT running the strategy — no LLM call, no
// mutation, no events.
//
// WillCompact is an EXACT structural verdict: true exactly when applying the
// strategy right now would change the history (a non-empty summarized/omitted
// zone). It never depends on the forecast ratios.
//
// AfterTokens is an exact dry-run for the deterministic sliding_window strategy
// (Exact true, given a token counter); for the LLM-backed strategies it is a
// FORECAST derived from the CompactionForecast ratios (Exact false). The
// forecast therefore only ever influences the predicted number — never whether
// the strategy is available.
type CompactionPrediction struct {
	Strategy string
	// WillCompact is the exact "would this strategy shrink the history right
	// now" verdict (non-empty summarized/omitted zone).
	WillCompact bool
	// BeforeTokens is the exact token count of the input history (0 when no
	// token counter is available).
	BeforeTokens int
	// AfterTokens is the predicted post-compaction token count: an exact dry-run
	// for sliding_window, a forecast for summarization/hierarchical.
	AfterTokens int
	// VerbatimTokens is the exact token count of the messages the strategy keeps
	// verbatim — the calibration anchor for the compression-ratio EWMA
	// (summarizedInput = BeforeTokens - VerbatimTokens).
	VerbatimTokens int
	// Reclaim is BeforeTokens - AfterTokens (0 when !WillCompact).
	Reclaim int
	// Exact is true when AfterTokens is a deterministic dry-run rather than a
	// forecast (sliding_window with a token counter, or any verbatim no-op).
	Exact bool
}

// PredictCompaction computes the expected effect of compacting msgs with the
// named strategy without running it. It is safe to call from any goroutine
// (pure function: reads msgs/cfg/deps, mutates nothing). An unknown strategy
// fails closed with the same error as CompactConversationHistory; an empty
// history yields a zero prediction with WillCompact=false.
func PredictCompaction(msgs []llm.Message, strategy string, cfg CompactionConfig, deps CompactionDeps) (CompactionPrediction, error) {
	p := CompactionPrediction{Strategy: strategy}
	if strategy != "sliding_window" && strategy != "summarization" && strategy != "hierarchical" {
		return p, fmt.Errorf(unknownConversationStrategyFmt, strategy)
	}
	if len(msgs) == 0 {
		p.Exact = true
		return p, nil
	}
	p.BeforeTokens = countMessages(deps.TokenCounter, msgs)

	switch strategy {
	case "sliding_window":
		return predictConversationSliding(msgs, cfg, deps, p), nil
	case "summarization":
		return predictConversationSummarizing(msgs, cfg, deps, p), nil
	case "hierarchical":
		return predictConversationHierarchical(msgs, cfg, deps, p), nil
	}
	return p, nil // unreachable
}

// predictConversationSliding is the exact dry-run for sliding_window: the
// post-compaction token count is computable without any LLM — head + note +
// tail, all verbatim or a fixed-size note. AfterTokens is therefore exact.
func predictConversationSliding(msgs []llm.Message, cfg CompactionConfig, deps CompactionDeps, p CompactionPrediction) CompactionPrediction {
	keepFirst := cfg.SlidingWindow.KeepFirst
	if keepFirst <= 0 {
		keepFirst = 3
	}
	keepLast := cfg.SlidingWindow.KeepLast
	if keepLast <= 0 {
		keepLast = 10
	}
	if len(msgs) <= keepFirst+keepLast {
		p.WillCompact = false
		p.AfterTokens = p.BeforeTokens
		p.VerbatimTokens = p.BeforeTokens
		p.Exact = true
		return p
	}
	p.WillCompact = true
	head := msgs[:keepFirst]
	tail := msgs[len(msgs)-keepLast:]
	verbatim := make([]llm.Message, 0, len(head)+len(tail))
	verbatim = append(verbatim, head...)
	verbatim = append(verbatim, tail...)
	p.VerbatimTokens = countMessages(deps.TokenCounter, verbatim)
	note := llm.Message{
		Role:    "system",
		Content: fmt.Sprintf("[... %d earlier conversation messages omitted by sliding-window compaction ...]", len(msgs)-keepFirst-keepLast),
	}
	p.AfterTokens = p.VerbatimTokens + countMessages(deps.TokenCounter, []llm.Message{note})
	p.Reclaim = max(0, p.BeforeTokens-p.AfterTokens)
	p.Exact = deps.TokenCounter != nil
	return p
}

// predictConversationSummarizing predicts the summarization strategy's effect:
// the verbatim tail is exact, the summarized blocks are forecast at
// SummarizationRatio.
func predictConversationSummarizing(msgs []llm.Message, cfg CompactionConfig, deps CompactionDeps, p CompactionPrediction) CompactionPrediction {
	keepLast := cfg.Summarization.KeepLast
	if keepLast <= 0 {
		keepLast = 5
	}
	if len(msgs) <= keepLast {
		p.WillCompact = false
		p.AfterTokens = p.BeforeTokens
		p.VerbatimTokens = p.BeforeTokens
		p.Exact = true
		return p
	}
	p.WillCompact = true
	tail := msgs[len(msgs)-keepLast:]
	p.VerbatimTokens = countMessages(deps.TokenCounter, tail)
	ratio := deps.Forecast.SummarizationRatio
	if !usableRatio(ratio) {
		ratio = defaultSummarizationForecastRatio
	}
	blockSize := cfg.Summarization.BlockSize
	if blockSize <= 0 {
		blockSize = 10
	}
	older := msgs[:len(msgs)-keepLast]
	maxSummarizeTokens := deps.MaxSummarizeTokens
	if maxSummarizeTokens <= 0 {
		maxSummarizeTokens = 16000
	}
	truncateChars := cfg.Summarization.ObservationTruncate
	if truncateChars <= 0 {
		truncateChars = 500
	}
	summarized := 0
	for i := 0; i < len(older); i += blockSize {
		end := min(i+blockSize, len(older))
		blockTokens := textTokens(deps.TokenCounter, conversationBlockText(older[i:end], truncateChars))
		summarized += forecastSummaryTokens(clampSummarizeTokens(blockTokens, maxSummarizeTokens), ratio)
	}
	p.AfterTokens = p.VerbatimTokens + summarized
	p.Reclaim = max(0, p.BeforeTokens-p.AfterTokens)
	p.Exact = false
	return p
}

// predictConversationHierarchical predicts the hierarchical strategy's effect:
// the recent zone is exact; the distant zone (one aggressive summary) and the
// middle zone (per-block summaries) are forecast at their respective ratios.
func predictConversationHierarchical(msgs []llm.Message, cfg CompactionConfig, deps CompactionDeps, p CompactionPrediction) CompactionPrediction {
	distant, middle := conversationHierarchicalZones(len(msgs), cfg)
	if distant+middle <= 0 {
		p.WillCompact = false
		p.AfterTokens = p.BeforeTokens
		p.VerbatimTokens = p.BeforeTokens
		p.Exact = true
		return p
	}
	p.WillCompact = true
	recent := msgs[distant+middle:]
	p.VerbatimTokens = countMessages(deps.TokenCounter, recent)

	distantRatio := deps.Forecast.HierarchicalDistantRatio
	if !usableRatio(distantRatio) {
		distantRatio = defaultHierarchicalDistantForecastRatio
	}
	middleRatio := deps.Forecast.HierarchicalMiddleRatio
	if !usableRatio(middleRatio) {
		middleRatio = defaultHierarchicalMiddleForecastRatio
	}
	blockSize := cfg.Summarization.BlockSize
	if blockSize <= 0 {
		blockSize = 10
	}
	maxSummarizeTokens := deps.MaxSummarizeTokens
	if maxSummarizeTokens <= 0 {
		maxSummarizeTokens = 16000
	}
	truncateChars := cfg.Summarization.ObservationTruncate
	if truncateChars <= 0 {
		truncateChars = 500
	}

	// The distant zone is ONE block (blockSize = distant in the real path),
	// so its forecast input is that whole zone rendered as one block.
	distantTokens := clampSummarizeTokens(textTokens(deps.TokenCounter, conversationBlockText(msgs[:distant], truncateChars)), maxSummarizeTokens)
	middleTokens := 0
	middleMsgs := msgs[distant : distant+middle]
	for i := 0; i < len(middleMsgs); i += blockSize {
		end := min(i+blockSize, len(middleMsgs))
		blockTokens := textTokens(deps.TokenCounter, conversationBlockText(middleMsgs[i:end], truncateChars))
		middleTokens += clampSummarizeTokens(blockTokens, maxSummarizeTokens)
	}

	p.AfterTokens = p.VerbatimTokens +
		forecastSummaryTokens(distantTokens, distantRatio) +
		forecastSummaryTokens(middleTokens, middleRatio)
	p.Reclaim = max(0, p.BeforeTokens-p.AfterTokens)
	p.Exact = false
	return p
}

// forecastSummaryTokens estimates the token count of a summary produced from
// inputTokens of block text at the given compression ratio. An empty input
// yields an empty summary (0); otherwise the estimate is clamped to at least 1
// so a non-empty summarized block is never predicted to vanish.
func forecastSummaryTokens(inputTokens int, ratio float64) int {
	if inputTokens <= 0 {
		return 0
	}
	return max(1, int(float64(inputTokens)*ratio))
}

// clampSummarizeTokens mirrors summarizeConversationBlocks' per-block token
// truncation: block text beyond maxSummarizeTokens is dropped before the LLM
// sees it, so the forecast must clamp a block's input tokens to the same budget
// rather than letting an over-long block inflate AfterTokens (and deflate
// Reclaim). The input is the token count of the block's RENDERED text (see
// [conversationBlockText]) — the per-message ObservationTruncate cap and role
// prefixes are part of that rendering, so they are modeled here too. A zero or
// negative input is returned unchanged; the caller resolves the 16000 default
// before calling.
func clampSummarizeTokens(inputTokens, maxSummarizeTokens int) int {
	if inputTokens > maxSummarizeTokens {
		return maxSummarizeTokens
	}
	return inputTokens
}

// countMessages is a nil-safe CountMessages: a nil counter yields 0 (the
// prediction's token fields then carry "unknown" rather than a fabricated
// number; WillCompact and Exact still hold).
func countMessages(counter llm.TokenCounter, msgs []llm.Message) int {
	if counter == nil {
		return 0
	}
	return counter.CountMessages(msgs)
}

// textTokens is the plain-text counterpart of [countMessages]: a nil counter
// yields 0, matching the "unknown, not fabricated" contract for prediction
// fields.
func textTokens(counter llm.TokenCounter, text string) int {
	if counter == nil {
		return 0
	}
	return counter.Count(text)
}
