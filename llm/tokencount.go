package llm

import (
	"fmt"
	"strings"
	"sync"

	"github.com/pkoukk/tiktoken-go"
)

// TokenCounter — interface for counting tokens to manage context budget.
type TokenCounter interface {
	// Count returns approximate token count for a text string.
	Count(text string) int

	// CountMessages returns total token count across all messages.
	CountMessages(msgs []Message) int
}

// estimatedTokensPerToolOverhead is a fixed framing allowance per tool
// definition on top of the raw name/description/schema content estimate.
//
// The name, description and input schema contents are already counted above;
// this constant covers the JSON wrapper keys and structural punctuation that
// surround them on the wire — `{"type":"function","function":{"name":"",
// "description":"","parameters":{}}}`. That framing is punctuation-heavy and
// its field keys (`type`, `function`, `name`, `description`, `parameters`) are
// common single-token words, so it tokenizes far denser than the ~4
// chars/token prose heuristic would suggest: the fixed wrapper is ~77 chars
// but only ~20 tokens. Using a flat 4 (the prose heuristic) would under-count
// every tool by ~5x, so we pin a realistic per-tool framing allowance instead.
const estimatedTokensPerToolOverhead = 20

// EstimateToolDefinitions returns a conservative token estimate for the given
// tool definitions using the same ~4 chars/token heuristic as
// SimpleTokenCounter.
//
// Tool schemas are attached to every request but are not counted by the
// conversation tracker (which only estimates message history), so callers
// reserve this overhead out of the context budget to avoid under-estimating
// wire usage against local/self-hosted engines whose KV cache overflows on the
// true request size.
func EstimateToolDefinitions(defs []ToolDefinition) int {
	total := 0
	for _, d := range defs {
		total += (len(d.Name) + estimatedTokensPerChar - 1) / estimatedTokensPerChar
		total += (len(d.Description) + estimatedTokensPerChar - 1) / estimatedTokensPerChar
		if len(d.InputSchema) > 0 {
			total += (len(d.InputSchema) + estimatedTokensPerChar - 1) / estimatedTokensPerChar
		}
		total += estimatedTokensPerToolOverhead
	}
	return total
}

// estimatedTokensPerChar is the approximate ratio of characters to tokens
// used for fast token count estimation.
const estimatedTokensPerChar = 4

// estimatedTokensPerImage is the conservative per-image token estimate used
// when a message carries image content blocks. OpenAI's high-detail image
// processing costs roughly 765 tokens for a typical screenshot, while
// Anthropic charges ~85 tokens per image. The higher (conservative) estimate
// is the default so the context budget is not over-optimistically consumed;
// Anthropic-family counters override it with estimatedAnthropicTokensPerImage
// to avoid ~9× over-counting that would trigger premature compaction.
const estimatedTokensPerImage = 765

// estimatedAnthropicTokensPerImage is the per-image token estimate for
// Anthropic-family models, which charge approximately 85 tokens per image
// regardless of size. Using the OpenAI-oriented 765 estimate for Anthropic
// would over-count images ~9× and trigger premature context compaction on
// image-heavy Anthropic conversations.
const estimatedAnthropicTokensPerImage = 85

// countContentTokens returns the token count for a message's content,
// accounting for structured content blocks when present. When the message has
// non-empty ContentBlocks (after normalization), text blocks are counted via
// count and image blocks are estimated at imageEstimate tokens each; unknown
// block types are skipped (matching provider behavior). When ContentBlocks is
// empty, the legacy path counts msg.Content, preserving backward compatibility
// for text-only messages. imageEstimate is provider-specific (see
// estimatedTokensPerImage / estimatedAnthropicTokensPerImage).
func countContentTokens(count func(string) int, msg Message, imageEstimate int) int {
	blocks := NormalizeContentBlocks(msg)
	if blocks == nil {
		return count(msg.Content)
	}
	total := 0
	for _, blk := range blocks {
		switch blk.Type {
		case "text":
			total += count(blk.Text)
		case "image":
			total += imageEstimate
		default:
			// Unknown block types are skipped, matching provider rendering.
		}
	}
	return total
}

// SimpleTokenCounter — approximate token counter using ~4 chars = 1 token rule.
type SimpleTokenCounter struct {
	imageTokenEstimate int // per-image token estimate (provider-specific; see NewSimpleTokenCounter)
}

// NewSimpleTokenCounter creates a SimpleTokenCounter that estimates tokens
// using the ~4 chars = 1 token heuristic. The per-image token estimate
// defaults to the conservative OpenAI-oriented value (estimatedTokensPerImage);
// NewTokenCounter overrides it to estimatedAnthropicTokensPerImage for
// Anthropic-family models so images are not over-counted ~9×.
func NewSimpleTokenCounter() *SimpleTokenCounter {
	return &SimpleTokenCounter{
		imageTokenEstimate: estimatedTokensPerImage,
	}
}

// Count returns the approximate token count for text using ceiling division
// of its byte length by estimatedTokensPerChar.
func (c *SimpleTokenCounter) Count(text string) int {
	if text == "" {
		return 0
	}
	return (len(text) + estimatedTokensPerChar - 1) / estimatedTokensPerChar // ceiling division
}

// CountMessages returns the total approximate token count across all messages,
// including role, content (or structured content blocks), tool-call
// names/inputs, and per-message framing overhead.
func (c *SimpleTokenCounter) CountMessages(msgs []Message) int {
	total := 0
	for _, msg := range msgs {
		total += c.Count(msg.Role)
		total += countContentTokens(c.Count, msg, c.imageTokenEstimate)
		for _, tc := range msg.ToolCalls {
			total += c.Count(tc.Name)
			total += c.Count(string(tc.Input))
		}
		// Add small overhead per message for framing
		total += 4
	}
	return total
}

// TiktokenCounter — accurate token counter using tiktoken-go for OpenAI models.
// The tiktoken-go library's Encode method is NOT safe for concurrent use
// (it mutates internal caches), hence the exclusive Lock.
type TiktokenCounter struct {
	tkm                *tiktoken.Tiktoken
	imageTokenEstimate int // per-image token estimate (OpenAI-oriented; see NewTiktokenCounter)
	mu                 sync.Mutex
}

// NewTiktokenCounter creates a new TiktokenCounter with the specified encoding.
// Valid encodings include: "o200k_base", "cl100k_base", "p50k_base", etc.
// The per-image token estimate defaults to estimatedTokensPerImage (OpenAI
// high-detail); tiktoken counters are OpenAI-oriented, so no Anthropic
// override is applied.
func NewTiktokenCounter(encoding string) (*TiktokenCounter, error) {
	tkm, err := tiktoken.GetEncoding(encoding)
	if err != nil {
		return nil, err
	}
	return &TiktokenCounter{
		tkm:                tkm,
		imageTokenEstimate: estimatedTokensPerImage,
	}, nil
}

// Count returns the exact token count for text using the tiktoken encoding.
// It is safe for concurrent use.
func (c *TiktokenCounter) Count(text string) int {
	if text == "" {
		return 0
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	tokens := c.tkm.Encode(text, nil, nil)
	return len(tokens)
}

// CountMessages returns the total exact token count across all messages,
// including role, content (or structured content blocks), tool-call
// names/inputs, and per-message framing overhead.
func (c *TiktokenCounter) CountMessages(msgs []Message) int {
	total := 0
	for _, msg := range msgs {
		total += c.Count(msg.Role)
		total += countContentTokens(c.Count, msg, c.imageTokenEstimate)
		for _, tc := range msg.ToolCalls {
			total += c.Count(tc.Name)
			total += c.Count(string(tc.Input))
		}
		// Add small overhead per message for framing
		total += 4
	}
	return total
}

// NewTokenCounter creates a TokenCounter based on the tokenizer type.
// Supported types:
//   - "tiktoken/o200k_base" → TiktokenCounter with o200k_base encoding
//   - "tiktoken/cl100k_base" → TiktokenCounter with cl100k_base encoding
//   - "anthropic-api" → SimpleTokenCounter (rely on API correction)
//   - "approximate" or "" or unknown → SimpleTokenCounter
//
// The returned TokenCounter is always valid (never nil). The error indicates
// that a fallback counter was used instead of the requested type.
func NewTokenCounter(tokenizerType string) (TokenCounter, error) {
	switch {
	case strings.HasPrefix(tokenizerType, "tiktoken/"):
		encoding := strings.TrimPrefix(tokenizerType, "tiktoken/")
		counter, err := NewTiktokenCounter(encoding)
		if err != nil {
			return NewSimpleTokenCounter(), fmt.Errorf("failed to create tiktoken counter for encoding %s: %w", encoding, err)
		}
		return counter, nil
	case tokenizerType == "anthropic-api":
		// For Anthropic models, we rely on API correction rather than local
		// counting. Use the Anthropic-specific per-image estimate (~85 tokens)
		// instead of the conservative OpenAI-oriented default (765) so
		// image-heavy Anthropic conversations do not over-count images ~9×
		// and trigger premature context compaction.
		counter := NewSimpleTokenCounter()
		counter.imageTokenEstimate = estimatedAnthropicTokensPerImage
		return counter, nil
	case tokenizerType == "approximate" || tokenizerType == "":
		return NewSimpleTokenCounter(), nil
	default:
		// Unknown tokenizer type, fallback to simple
		return NewSimpleTokenCounter(), fmt.Errorf("unknown tokenizer type: %s", tokenizerType)
	}
}

// ContextTokenTracker — hybrid A+C coordinator that combines predictive counting
// with API-corrected actuals. Uses predictive counter for estimates between API calls,
// then corrects with actual usage from API responses.
type ContextTokenTracker struct {
	predictive    TokenCounter
	lastKnownUsed int // from API response.usage.input_tokens
	pendingDelta  int // estimated tokens added since last API call
	mu            sync.RWMutex
}

// NewContextTokenTracker creates a new ContextTokenTracker with the given predictive counter.
func NewContextTokenTracker(counter TokenCounter) *ContextTokenTracker {
	return &ContextTokenTracker{
		predictive:    counter,
		lastKnownUsed: 0,
		pendingDelta:  0,
	}
}

// EstimateTotal returns the estimated total token count (lastKnownUsed + pendingDelta).
func (t *ContextTokenTracker) EstimateTotal() int {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.lastKnownUsed + t.pendingDelta
}

// AddDelta adds the token count of the given text to pendingDelta.
func (t *ContextTokenTracker) AddDelta(text string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.pendingDelta += t.predictive.Count(text)
}

// Correct updates lastKnownUsed with the actual API input tokens and resets pendingDelta.
func (t *ContextTokenTracker) Correct(apiInputTokens int) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.lastKnownUsed = apiInputTokens
	t.pendingDelta = 0
}

// Reset resets both lastKnownUsed and pendingDelta to 0.
func (t *ContextTokenTracker) Reset() {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.lastKnownUsed = 0
	t.pendingDelta = 0
}

// EstimateMessages returns the estimated token count for the given messages
// using the predictive counter. This is a read-only operation.
func (t *ContextTokenTracker) EstimateMessages(msgs []Message) int {
	return t.predictive.CountMessages(msgs)
}
