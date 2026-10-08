package oneshot

import (
	"encoding/json"
	"errors"
	"regexp"
	"strings"

	"github.com/v0lka/sp4rk/llm"
)

// The parser toolkit: complementary extraction techniques for structured
// values in one-shot LLM responses. Each helper covers a failure mode the
// others miss, and callers compose them freely:
//
//   - CandidateTexts / Text   — multi-field candidates (Content →
//     ReasoningContent → Reasoning) with fence and preamble stripping;
//   - ParseJSON               — JSON objects embedded in prose or fences
//     (built on llm.ExtractJSON);
//   - Marked / BetweenMarkers — sentinel-marker extraction (### X_START / …
//     / ### X_END);
//   - KeyLineParser / ParseKeyLines — KEY: value lines tolerating markdown
//     decoration (**VERDICT:**, list markers, blockquotes, code fences);
//   - SplitInline             — a second KEY hiding inside a value line
//     ("VERDICT: ALLOW — REASON: safe");
//   - TrimEmphasis / Unquote  — decorative **emphasis** and quotes around
//     values.
//
// The toolkit is deliberately mechanical: it recovers text and key/value
// pairs, and leaves domain semantics (which tokens mean which verdict, what
// counts as a safe default) to the caller.

// CandidateTexts returns the response's text fields in extraction priority
// order — Message.Content, Message.ReasoningContent (DeepSeek-style
// providers), Reasoning (OpenAI Responses API). This covers the failure mode
// where small reasoning models put the actual answer into a reasoning field
// and leave Content empty. Fields that are empty or whitespace-only are
// skipped; a nil response yields nil. The returned strings are the raw field
// values (not trimmed) and must not be mutated.
func CandidateTexts(resp *llm.ChatResponse) []string {
	if resp == nil {
		return nil
	}
	candidates := make([]string, 0, 3)
	for _, field := range []string{resp.Message.Content, resp.Message.ReasoningContent, resp.Reasoning} {
		if strings.TrimSpace(field) == "" {
			continue
		}
		candidates = append(candidates, field)
	}
	return candidates
}

// Text returns the first candidate field that still yields non-empty text
// after stripping a wrapping markdown code fence and a leading conversational
// preamble ("Sure, ", "Here's the commit message: ", "Based on my analysis: ",
// …), in candidate priority order. It returns "" when no field produces
// usable text.
func Text(resp *llm.ChatResponse) string {
	for _, candidate := range CandidateTexts(resp) {
		stripped := strings.TrimSpace(StripReasoningPrefix(StripFence(candidate)))
		if stripped != "" {
			return stripped
		}
	}
	return ""
}

// fenceRe matches a single markdown code fence wrapping the ENTIRE (trimmed)
// string: an optional language tag, the body, and the closing fence. The
// non-greedy body keeps multi-line content (blank line + paragraphs) intact
// even when the closing fence is the last character. Backticks legitimately
// appearing inside a body that is not fully fenced are left untouched.
var fenceRe = regexp.MustCompile("(?s)^```[a-zA-Z0-9+-]*[ \t]*\n(.+?)\n```[ \t]*$")

// StripFence removes a single surrounding markdown code fence from s when the
// entire (trimmed) string is wrapped in one — a defensive net for prompts
// that forbid fencing, which some models ignore. Input without a full-string
// fence is returned trimmed; a fenced body is returned with surrounding
// whitespace removed.
func StripFence(s string) string {
	trimmed := strings.TrimSpace(s)
	if m := fenceRe.FindStringSubmatch(trimmed); m != nil {
		return strings.TrimSpace(m[1])
	}
	return trimmed
}

// reasoningPrefixRe matches the common conversational preambles some models
// — especially small ones — accidentally emit before the requested payload
// ("Sure, ", "Here's the commit message: ", "Based on my analysis: ", "OK, ",
// "Below is the optimized prompt: ", …). The match is case-insensitive and
// anchored at the start; everything up to and including the preamble is
// removed while the payload is preserved verbatim.
var reasoningPrefixRe = regexp.MustCompile(`(?i)^` +
	`(?:` +
	`based on my analysis(?:,| of|:) ` +
	`|here(?:['′]s|s) (?:the )?commit message(?:,|:) ` +
	`|the commit message(?:,| is|:) ` +
	`|here(?:['′]s|s) (?:the )?(?:optimized )?prompt(?:,|:) ` +
	`|the optimized prompt(?:,| is|:) ` +
	`|this is the optimized prompt ` +
	`|sure,? ` +
	`|ok,? ` +
	`|ok sure,? ` +
	`|here(?:['′]s|s) an? ` +
	`|below(?:,| is|:) ` +
	`|according to my analysis ` +
	`|from the diff ` +
	`|from the provided diff ` +
	`|from the staged diff ` +
	`|this commit ` +
	`|this prompt ` +
	`)`)

// StripReasoningPrefix removes one leading conversational preamble (see
// reasoningPrefixRe) from s. Input without a recognized preamble is returned
// unchanged. Only one layer is stripped, mirroring the upstream commit-message
// and prompt-optimizer extractors this generalizes.
func StripReasoningPrefix(s string) string {
	return reasoningPrefixRe.ReplaceAllString(s, "")
}

// emphasisTrimSet lists the characters trimmed from values by TrimEmphasis:
// markdown emphasis and code markers plus the whitespace they trail.
const emphasisTrimSet = "*_` \t"

// TrimEmphasis strips leading/trailing whitespace plus markdown emphasis and
// code characters (*, _, backtick) from a value, so decorative wrapping like
// "**ALLOW**" or a leading "** " (left behind by a bold key whose closing
// marker trails the separator) does not corrupt value matching. Emphasis
// INSIDE a value is preserved ("re-enable" stays intact).
func TrimEmphasis(s string) string {
	return strings.Trim(s, emphasisTrimSet)
}

// Unquote trims emphasis and then removes a single layer of matching
// surrounding quote characters (", ', or backtick) from a value, so a reason
// emitted as “"a quoted reason"” is recovered without its decorative quotes.
// Unbalanced or non-matching quotes are left as-is.
func Unquote(s string) string {
	r := TrimEmphasis(s)
	if len(r) >= 2 {
		first, last := r[0], r[len(r)-1]
		if (first == '"' && last == '"') || (first == '\'' && last == '\'') || (first == '`' && last == '`') {
			r = strings.TrimSpace(r[1 : len(r)-1])
		}
	}
	return r
}

// keyLineListPrefixRe matches a leading markdown list marker ("- ", "* ",
// "+ ", "1. ") so list-formatted lines are still recognized as key/value
// pairs.
var keyLineListPrefixRe = regexp.MustCompile(`^(?:[-*+]|\d+\.)\s+`)

// keyRegionDeEmphasis strips emphasis characters from a line's key region
// (the part before the first ':' or '=' separator), exposing prefixes hidden
// by bold/italic/code formatting like "**VERDICT:**". The separator and the
// free-form value (which may legitimately contain colons, e.g. "12:00") are
// untouched.
var keyRegionDeEmphasis = strings.NewReplacer("*", "", "_", "", "`", "")

// StripLineDecoration removes the markdown decorations that would hide a
// leading KEY: prefix from a single line: fence remnants, blockquote markers,
// list markers, and emphasis in the key region. Apply it before key matching.
func StripLineDecoration(line string) string {
	line = strings.TrimSpace(line)
	line = strings.TrimPrefix(line, "```")
	line = strings.TrimSuffix(line, "```")
	// Leading blockquote markers.
	line = strings.TrimLeft(line, "> \t")
	// Leading list marker ("- ", "* ", "+ ", "12. ").
	line = keyLineListPrefixRe.ReplaceAllString(line, "")
	// Emphasis in the key region only (everything up to the separator).
	if idx := strings.IndexAny(line, ":="); idx >= 0 {
		line = keyRegionDeEmphasis.Replace(line[:idx]) + line[idx:]
	}
	return strings.TrimSpace(line)
}

// KeyValue is one parsed KEY: value line. Key is the matched key lowercased
// (so "VERDICT" and "Verdict" both arrive as "verdict"); Value is the raw
// remainder after the separator, whitespace-trimmed but otherwise untouched —
// apply TrimEmphasis/Unquote as the domain requires.
type KeyValue struct {
	Key   string
	Value string
}

// KeyLineParser extracts KEY: value (or KEY = value) lines from an LLM
// response, tolerating the formatting variations models produce around the
// requested format: markdown bold/italics ("**VERDICT:** ALLOW"), list
// markers ("- VERDICT: …"), blockquotes, code fences, lowercase keys
// ("Verdict:"), and the "KEY = value" spelling.
//
// Key matching is case-insensitive. A line matches when its key equals one of
// the parser's keys OR extends one of them, so a parser built with "reason"
// also matches "REASONING:" (the alias judges accept). Keys must be
// word-like: they are matched anchored at the start of the decorated line.
type KeyLineParser struct {
	re   *regexp.Regexp
	keys []string
}

// NewKeyLineParser compiles a key-line parser for the given keys. With no
// keys, Parse returns nil.
func NewKeyLineParser(keys ...string) *KeyLineParser {
	if len(keys) == 0 {
		return &KeyLineParser{}
	}
	quoted := make([]string, len(keys))
	for i, k := range keys {
		quoted[i] = regexp.QuoteMeta(k)
	}
	// \w* after the alternation implements the "extends" semantics: a line
	// key may be one of the keys or extend one ("reason" matches "REASONING:
	// …" the way the judge aliases accept).
	return &KeyLineParser{
		re:   regexp.MustCompile(`(?i)^(?:` + strings.Join(quoted, "|") + `)\w*\s*[:=]\s*(.*)$`),
		keys: keys,
	}
}

// matches reports whether the lowercased line key matches the parser's key
// set: equal to a key, or a key extended by a suffix (reason → reasoning).
func (p *KeyLineParser) matches(key string) bool {
	if len(p.keys) == 0 {
		return false
	}
	for _, want := range p.keys {
		if strings.HasPrefix(key, want) {
			return true
		}
	}
	return false
}

// Parse splits content into lines, strips markdown decoration per line, and
// returns every recognized KEY: value pair in document order. Lines without a
// matching key (prose, empty lines, decoration-only lines) are skipped.
func (p *KeyLineParser) Parse(content string) []KeyValue {
	if p == nil || p.re == nil {
		return nil
	}
	var out []KeyValue
	for _, raw := range strings.Split(content, "\n") {
		line := StripLineDecoration(raw)
		if line == "" {
			continue
		}
		m := p.re.FindStringSubmatch(line)
		if m == nil {
			continue
		}
		// The key is the matched prefix up to its separator (decoration
		// stripping already removed emphasis around it).
		key := m[0]
		if idx := strings.IndexAny(key, ":="); idx >= 0 {
			key = strings.TrimSpace(key[:idx])
		}
		out = append(out, KeyValue{Key: strings.ToLower(key), Value: strings.TrimSpace(m[1])})
	}
	return out
}

// ParseKeyLines is the one-shot convenience form of KeyLineParser: it parses
// content for the given keys in a single call. See KeyLineParser for the
// matching semantics.
func ParseKeyLines(content string, keys ...string) []KeyValue {
	return NewKeyLineParser(keys...).Parse(content)
}

// inlineKeyReFor builds the regex locating a KEY embedded inside a value
// ("ALLOW — REASON: safe"): word boundary, one of the keys (optionally
// extended, so "reason" also finds "REASONING:"), separator, and the rest of
// the line. Keys are regex-quoted so caller input is safe.
func inlineKeyReFor(keys ...string) *regexp.Regexp {
	quoted := make([]string, len(keys))
	for i, k := range keys {
		quoted[i] = regexp.QuoteMeta(k)
	}
	return regexp.MustCompile(`(?i)\b(?:` + strings.Join(quoted, "|") + `)\w*\s*[:=]\s*(.*)$`)
}

// SplitInline detects a second KEY embedded inside a value — the single-line
// answer form "VERDICT: ALLOW — REASON: safe read-only operation" — and
// splits it into the part before the inline key (the first value), the
// matched key (lowercased), and the inline value. ok is false when no inline
// key is present. The returned parts are raw substrings; apply
// TrimEmphasis/Unquote as needed. Both parts are whitespace-trimmed.
func SplitInline(value string, keys ...string) (head, key, inlineValue string, ok bool) {
	if len(keys) == 0 {
		return value, "", "", false
	}
	loc := inlineKeyReFor(keys...).FindStringSubmatchIndex(value)
	if loc == nil {
		return value, "", "", false
	}
	inlineValue = value[loc[2]:loc[3]]
	head = strings.TrimSpace(value[:loc[0]])
	key = strings.ToLower(strings.TrimSpace(value[loc[0]:loc[1]]))
	// The matched region includes the separator and trailing spacing; keep
	// only the key token itself.
	if idx := strings.IndexAny(key, ":="); idx >= 0 {
		key = strings.TrimSpace(key[:idx])
	}
	return head, key, strings.TrimSpace(inlineValue), true
}

// BetweenMarkers returns the text between the first start-marker occurrence
// and the next end-marker after it. It returns ("", true) when both markers
// are present even if the content between them is empty or whitespace-only
// (the caller decides whether empty content is acceptable), and ("", false)
// when either marker is missing or they appear in the wrong order. The
// extracted text is whitespace-trimmed.
//
// "After it" means after the END of the start marker: the end-marker search
// begins past the start marker, so overlapping or equal markers work as the
// same-delimiter case demands (e.g. start == end for a ``` fence pair). The
// end marker may not overlap the start marker's own text.
func BetweenMarkers(s, start, end string) (string, bool) {
	startIdx := strings.Index(s, start)
	if startIdx < 0 {
		return "", false
	}
	// Search from the end of the start marker, not from its own offset:
	// an end marker equal to (or contained in) the start marker would
	// otherwise match at offset 0 and place the content end before the
	// content start — a slice-bounds panic. This also makes start == end
	// extract the text between two successive occurrences, as documented.
	after := startIdx + len(start)
	endIdx := strings.Index(s[after:], end)
	if endIdx < 0 {
		return "", false
	}
	return strings.TrimSpace(s[after : after+endIdx]), true
}

// Marked scans the response's candidate fields (Content → ReasoningContent →
// Reasoning) for the marker pair and returns the first hit, so outputs whose
// answer landed in a reasoning field are still recovered. ok is false when no
// candidate carries both markers.
func Marked(resp *llm.ChatResponse, start, end string) (string, bool) {
	for _, candidate := range CandidateTexts(resp) {
		if extracted, ok := BetweenMarkers(candidate, start, end); ok {
			return extracted, true
		}
	}
	return "", false
}

// ErrNoJSON reports that a response carried no candidate field at all, so no
// JSON could even be attempted.
var ErrNoJSON = errors.New("oneshot: no candidate text in response")

// ParseJSON extracts a JSON object from the response and decodes it into T.
// Each candidate field (Content → ReasoningContent → Reasoning) is run
// through llm.ExtractJSON — which recovers fenced blocks and the outermost
// brace pair from JSON embedded in prose — and decoded; the first field that
// decodes cleanly wins. It returns ErrNoJSON when there is nothing to decode,
// and an error joining ErrNoJSON with the last decode error when every field
// fails to decode (errors.Is(err, ErrNoJSON) holds in both cases, so callers
// can classify without unwrapping). Models that answer a two-line format with
// {"verdict": "ALLOW", …} despite the format request are recovered by this
// path.
func ParseJSON[T any](resp *llm.ChatResponse) (T, error) {
	var zero T
	candidates := CandidateTexts(resp)
	if len(candidates) == 0 {
		return zero, ErrNoJSON
	}
	var lastErr error
	for _, candidate := range candidates {
		raw := llm.ExtractJSON(candidate)
		if raw == "" {
			continue
		}
		var out T
		if err := json.Unmarshal([]byte(raw), &out); err != nil {
			lastErr = err
			continue
		}
		return out, nil
	}
	if lastErr == nil {
		return zero, ErrNoJSON
	}
	return zero, errors.Join(ErrNoJSON, lastErr)
}
