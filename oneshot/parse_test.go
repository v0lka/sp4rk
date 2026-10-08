package oneshot

import (
	"errors"
	"testing"

	"github.com/v0lka/sp4rk/llm"
)

// respWith builds a ChatResponse with the three candidate fields.
func respWith(content, reasoningContent, reasoning string) *llm.ChatResponse {
	return &llm.ChatResponse{
		Message:   llm.Message{Content: content, ReasoningContent: reasoningContent},
		Reasoning: reasoning,
	}
}

// assertKeyValues compares parsed key/value pairs with the expectation.
func assertKeyValues(t *testing.T, got, want []KeyValue) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("parsed key/values = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("parsed[%d] = %+v, want %+v", i, got[i], want[i])
		}
	}
}

// --- CandidateTexts ---

func TestCandidateTexts_NilResponse(t *testing.T) {
	if got := CandidateTexts(nil); got != nil {
		t.Errorf("CandidateTexts(nil) = %v, want nil", got)
	}
}

func TestCandidateTexts_PriorityOrder(t *testing.T) {
	got := CandidateTexts(respWith("content", "reasoning-content", "reasoning"))
	want := []string{"content", "reasoning-content", "reasoning"}
	if len(got) != len(want) {
		t.Fatalf("CandidateTexts() = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("CandidateTexts()[%d] = %q, want %q", i, got[i], want[i])
		}
	}
}

func TestCandidateTexts_SkipsBlankFields(t *testing.T) {
	got := CandidateTexts(respWith("", "   ", "reasoning"))
	if len(got) != 1 || got[0] != "reasoning" {
		t.Errorf("CandidateTexts() = %v, want [reasoning]", got)
	}
}

// --- Text (regressions from c0wrk extractCommitMessage / optimize heuristic) ---

func TestText_FromContent(t *testing.T) {
	if got := Text(respWith("feat(api): add rate limiting", "", "")); got != "feat(api): add rate limiting" {
		t.Errorf("Text() = %q, want %q", got, "feat(api): add rate limiting")
	}
}

func TestText_FallbackToReasoningContent(t *testing.T) {
	if got := Text(respWith("", "fix(db): resolve connection leak", "")); got != "fix(db): resolve connection leak" {
		t.Errorf("Text() = %q, want %q", got, "fix(db): resolve connection leak")
	}
}

func TestText_FallbackToReasoning(t *testing.T) {
	if got := Text(respWith("", "", "docs(readme): update install instructions")); got != "docs(readme): update install instructions" {
		t.Errorf("Text() = %q, want %q", got, "docs(readme): update install instructions")
	}
}

func TestText_StripsReasoningPrefix(t *testing.T) {
	if got := Text(respWith("Based on my analysis: feat(auth): add token refresh", "", "")); got != "feat(auth): add token refresh" {
		t.Errorf("Text() = %q, want %q", got, "feat(auth): add token refresh")
	}
}

func TestText_StripsMarkdownFence(t *testing.T) {
	if got := Text(respWith("```\nfeat(api): add rate limiting\n```", "", "")); got != "feat(api): add rate limiting" {
		t.Errorf("Text() = %q, want %q", got, "feat(api): add rate limiting")
	}
}

func TestText_PrefersContentOverReasoning(t *testing.T) {
	if got := Text(respWith("feat(ui): update button", "fix(db): resolve leak", "")); got != "feat(ui): update button" {
		t.Errorf("Text() = %q, want %q", got, "feat(ui): update button")
	}
}

func TestText_AllEmpty(t *testing.T) {
	if got := Text(respWith("", "", "")); got != "" {
		t.Errorf("Text() = %q, want empty", got)
	}
}

func TestText_NilResponse(t *testing.T) {
	if got := Text(nil); got != "" {
		t.Errorf("Text(nil) = %q, want empty", got)
	}
}

func TestText_StripsOptimizePreamble(t *testing.T) {
	// Regression from the prompt-optimizer heuristic: "the optimized prompt:"
	// and "Sure, " preambles are stripped too.
	for _, tc := range []struct{ in, want string }{
		{"the optimized prompt: Fix the login bug", "Fix the login bug"},
		{"Sure, Fix the login bug", "Fix the login bug"},
		{"OK, Fix the login bug", "Fix the login bug"},
		{"Below is the rewritten prompt:\nFix the login bug", "the rewritten prompt:\nFix the login bug"},
	} {
		if got := Text(respWith(tc.in, "", "")); got != tc.want {
			t.Errorf("Text(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

func TestText_MidStringPreambleNotStripped(t *testing.T) {
	const in = "feat: allow 'Sure, ok' as an alias"
	if got := Text(respWith(in, "", "")); got != in {
		t.Errorf("Text() = %q, want unchanged %q", got, in)
	}
}

// --- StripFence ---

func TestStripFence(t *testing.T) {
	for _, tc := range []struct{ name, in, want string }{
		{"no fence", "feat: x", "feat: x"},
		{"bare fence", "```\nfeat: x\n```", "feat: x"},
		{"language fence", "```text\nfeat: x\n```", "feat: x"},
		{"multi-line body", "```\nfeat: x\n\nbody paragraph\n```", "feat: x\n\nbody paragraph"},
		{"fence not wrapping", "intro\n```\ncode\n```", "intro\n```\ncode\n```"},
		{"inner backticks preserved", "feat: fix `foo` handling", "feat: fix `foo` handling"},
	} {
		if got := StripFence(tc.in); got != tc.want {
			t.Errorf("%s: StripFence(%q) = %q, want %q", tc.name, tc.in, got, tc.want)
		}
	}
}

// --- TrimEmphasis / Unquote ---

func TestTrimEmphasis(t *testing.T) {
	for _, tc := range []struct{ in, want string }{
		{"**ALLOW**", "ALLOW"},
		{"`x`", "x"},
		{"** ", ""},
		{"_italic_", "italic"},
		{"re-enable", "re-enable"},
		{"ALLOW", "ALLOW"},
	} {
		if got := TrimEmphasis(tc.in); got != tc.want {
			t.Errorf("TrimEmphasis(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

func TestUnquote(t *testing.T) {
	for _, tc := range []struct{ in, want string }{
		{`"a quoted reason"`, "a quoted reason"},
		{`'single quoted'`, "single quoted"},
		{"`backticked`", "backticked"},
		{`"unbalanced`, `"unbalanced`},
		{"plain reason", "plain reason"},
	} {
		if got := Unquote(tc.in); got != tc.want {
			t.Errorf("Unquote(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

// --- StripLineDecoration ---

func TestStripLineDecoration(t *testing.T) {
	for _, tc := range []struct{ name, in, want string }{
		// Emphasis in the KEY region is stripped; the value keeps its
		// remnant ("**") — the caller trims it via TrimEmphasis (the judge's
		// split of responsibilities between decoration stripping and value
		// trimming, which preserves internal value emphasis).
		{"bold key", "**VERDICT:** ALLOW", "VERDICT:** ALLOW"},
		{"list marker", "- VERDICT: ALLOW", "VERDICT: ALLOW"},
		{"numbered list", "1. VERDICT: CONFIRM", "VERDICT: CONFIRM"},
		{"blockquote", "> VERDICT: ALLOW", "VERDICT: ALLOW"},
		{"fence remnant", "```VERDICT: ALLOW```", "VERDICT: ALLOW"},
		{"bold key colon time untouched", "**TIME:** 12:00", "TIME:** 12:00"},
		{"plain", "VERDICT: ALLOW", "VERDICT: ALLOW"},
	} {
		if got := StripLineDecoration(tc.in); got != tc.want {
			t.Errorf("%s: StripLineDecoration(%q) = %q, want %q", tc.name, tc.in, got, tc.want)
		}
	}
}

// --- KeyLineParser / ParseKeyLines (regressions from sp4rk judge tests) ---

func TestParseKeyLines_Canonical(t *testing.T) {
	got := ParseKeyLines("VERDICT: ALLOW\nREASON: Safe file read operation", "verdict", "reason")
	want := []KeyValue{{"verdict", "ALLOW"}, {"reason", "Safe file read operation"}}
	assertKeyValues(t, got, want)
}

func TestParseKeyLines_ReasonBeforeVerdict(t *testing.T) {
	got := ParseKeyLines("REASON: First line explanation\nVERDICT: ALLOW", "verdict", "reason")
	want := []KeyValue{{"reason", "First line explanation"}, {"verdict", "ALLOW"}}
	assertKeyValues(t, got, want)
}

func TestParseKeyLines_LowercaseKeys(t *testing.T) {
	got := ParseKeyLines("Verdict: allow\nReason: lowercase keys", "verdict", "reason")
	want := []KeyValue{{"verdict", "allow"}, {"reason", "lowercase keys"}}
	assertKeyValues(t, got, want)
}

func TestParseKeyLines_ReasoningAlias(t *testing.T) {
	// "reason" extends to "reasoning" — the judge's alias.
	got := ParseKeyLines("VERDICT: CONFIRM\nREASONING: needs approval", "verdict", "reason")
	want := []KeyValue{{"verdict", "CONFIRM"}, {"reasoning", "needs approval"}}
	assertKeyValues(t, got, want)
}

func TestParseKeyLines_MarkdownBoldKeys(t *testing.T) {
	// Values keep the bold-key remnant raw; TrimEmphasis (the judge's
	// matchVerdict/normalizeReason step) recovers the clean value.
	got := ParseKeyLines("**VERDICT:** ALLOW\n**REASON:** Safe read operation", "verdict", "reason")
	want := []KeyValue{{"verdict", "** ALLOW"}, {"reason", "** Safe read operation"}}
	assertKeyValues(t, got, want)
	if v := TrimEmphasis(got[0].Value); v != "ALLOW" {
		t.Errorf("TrimEmphasis(%q) = %q, want ALLOW", got[0].Value, v)
	}
	if v := TrimEmphasis(got[1].Value); v != "Safe read operation" {
		t.Errorf("TrimEmphasis(%q) = %q, want %q", got[1].Value, v, "Safe read operation")
	}
}

func TestParseKeyLines_ListMarkers(t *testing.T) {
	got := ParseKeyLines("- VERDICT: ALLOW\n- REASON: safe\n", "verdict", "reason")
	want := []KeyValue{{"verdict", "ALLOW"}, {"reason", "safe"}}
	assertKeyValues(t, got, want)
}

func TestParseKeyLines_NumberedList(t *testing.T) {
	got := ParseKeyLines("1. VERDICT: CONFIRM\n2. REASON: destructive\n", "verdict", "reason")
	want := []KeyValue{{"verdict", "CONFIRM"}, {"reason", "destructive"}}
	assertKeyValues(t, got, want)
}

func TestParseKeyLines_BlockquoteAndFence(t *testing.T) {
	got := ParseKeyLines("> VERDICT: ALLOW\n```\nREASON: fenced answer\n```", "verdict", "reason")
	want := []KeyValue{{"verdict", "ALLOW"}, {"reason", "fenced answer"}}
	assertKeyValues(t, got, want)
}

func TestParseKeyLines_EqualsForm(t *testing.T) {
	got := ParseKeyLines("VERDICT = ALLOW\nREASON = fine", "verdict", "reason")
	want := []KeyValue{{"verdict", "ALLOW"}, {"reason", "fine"}}
	assertKeyValues(t, got, want)
}

func TestParseKeyLines_ExtraWhitespace(t *testing.T) {
	got := ParseKeyLines("VERDICT:   ALLOW   \nREASON:   Extra spaces   ", "verdict", "reason")
	want := []KeyValue{{"verdict", "ALLOW"}, {"reason", "Extra spaces"}}
	assertKeyValues(t, got, want)
}

func TestParseKeyLines_ValueEmphasisLeftRaw(t *testing.T) {
	// Values are returned raw; callers apply TrimEmphasis/Unquote.
	got := ParseKeyLines("VERDICT: **ALLOW**\nREASON: \"a quoted reason\"", "verdict", "reason")
	want := []KeyValue{{"verdict", "**ALLOW**"}, {"reason", `"a quoted reason"`}}
	assertKeyValues(t, got, want)
	if v := TrimEmphasis(got[0].Value); v != "ALLOW" {
		t.Errorf("TrimEmphasis(%q) = %q, want ALLOW", got[0].Value, v)
	}
	if v := Unquote(got[1].Value); v != "a quoted reason" {
		t.Errorf("Unquote(%q) = %q, want %q", got[1].Value, v, "a quoted reason")
	}
}

func TestParseKeyLines_ProseSkipped(t *testing.T) {
	got := ParseKeyLines("I think the answer is:\nVERDICT: ALLOW\nno key here at all", "verdict", "reason")
	want := []KeyValue{{"verdict", "ALLOW"}}
	assertKeyValues(t, got, want)
}

func TestParseKeyLines_EmptyContent(t *testing.T) {
	if got := ParseKeyLines("", "verdict", "reason"); got != nil {
		t.Errorf("ParseKeyLines(\"\") = %v, want nil", got)
	}
}

func TestParseKeyLines_NoKeys(t *testing.T) {
	if got := ParseKeyLines("VERDICT: ALLOW"); got != nil {
		t.Errorf("ParseKeyLines with no keys = %v, want nil", got)
	}
}

func TestKeyLineParser_Matches(t *testing.T) {
	p := NewKeyLineParser("verdict", "reason")
	if !p.matches("verdict") || !p.matches("reasoning") {
		t.Error("expected verdict and reasoning to match")
	}
	if p.matches("unrelated") {
		t.Error("expected unrelated key not to match")
	}
}

// --- SplitInline (regressions from sp4rk judge inline tests) ---

func TestSplitInline(t *testing.T) {
	for _, tc := range []struct {
		name  string
		value string
		head  string
		key   string
		tail  string
		ok    bool
	}{
		{"em-dash", "ALLOW — REASON: safe read-only operation", "ALLOW —", "reason", "safe read-only operation", true},
		{"pipe", "ALLOW | REASON: safe", "ALLOW |", "reason", "safe", true},
		{"plain value", "ALLOW (read-only)", "ALLOW (read-only)", "", "", false},
		{"reasoning alias", "CONFIRMED — REASONING: needs approval", "CONFIRMED —", "reasoning", "needs approval", true},
	} {
		head, key, tail, ok := SplitInline(tc.value, "reason")
		if ok != tc.ok || head != tc.head || key != tc.key || tail != tc.tail {
			t.Errorf("%s: SplitInline(%q) = (%q, %q, %q, %v), want (%q, %q, %q, %v)",
				tc.name, tc.value, head, key, tail, ok, tc.head, tc.key, tc.tail, tc.ok)
		}
	}
}

func TestSplitInline_NoKeys(t *testing.T) {
	if _, _, _, ok := SplitInline("ALLOW — REASON: x"); ok {
		t.Error("SplitInline with no keys must not match")
	}
}

// --- BetweenMarkers / Marked (regressions from c0wrk optimize tests) ---

const (
	markerStart = "### OPTIMIZED_PROMPT_START"
	markerEnd   = "### OPTIMIZED_PROMPT_END"
)

func TestBetweenMarkers(t *testing.T) {
	for _, tc := range []struct {
		name string
		in   string
		want string
		ok   bool
	}{
		{"simple", "### OPTIMIZED_PROMPT_START\nFix the login bug\n### OPTIMIZED_PROMPT_END", "Fix the login bug", true},
		{
			"multi-line",
			"### OPTIMIZED_PROMPT_START\nFix the login bug\n\nSteps:\n1. Check auth middleware\n2. Fix token validation\n### OPTIMIZED_PROMPT_END",
			"Fix the login bug\n\nSteps:\n1. Check auth middleware\n2. Fix token validation",
			true,
		},
		{"preamble", "Some preamble\n### OPTIMIZED_PROMPT_START\nExtracted prompt\n### OPTIMIZED_PROMPT_END", "Extracted prompt", true},
		{"trailing", "### OPTIMIZED_PROMPT_START\nExtracted prompt\n### OPTIMIZED_PROMPT_END\nSome trailing text", "Extracted prompt", true},
		{"whitespace inside", "### OPTIMIZED_PROMPT_START\n\n  Trimmed content  \n\n### OPTIMIZED_PROMPT_END", "Trimmed content", true},
		{"empty between", "### OPTIMIZED_PROMPT_START\n\n### OPTIMIZED_PROMPT_END", "", true},
		{"only whitespace between", "### OPTIMIZED_PROMPT_START   \n### OPTIMIZED_PROMPT_END", "", true},
		{"only start", "### OPTIMIZED_PROMPT_START\ncontent", "", false},
		{"only end", "content\n### OPTIMIZED_PROMPT_END", "", false},
		{"wrong order", "### OPTIMIZED_PROMPT_END\ncontent\n### OPTIMIZED_PROMPT_START", "", false},
		{"no markers", "just plain text", "", false},
	} {
		got, ok := BetweenMarkers(tc.in, markerStart, markerEnd)
		if ok != tc.ok || got != tc.want {
			t.Errorf("%s: BetweenMarkers() = (%q, %v), want (%q, %v)", tc.name, got, ok, tc.want, tc.ok)
		}
	}
}

func TestMarked_PrefersContentOverReasoning(t *testing.T) {
	resp := respWith(
		markerStart+"\nFrom content\n"+markerEnd,
		markerStart+"\nFrom reasoning\n"+markerEnd,
		"",
	)
	if got, ok := Marked(resp, markerStart, markerEnd); !ok || got != "From content" {
		t.Errorf("Marked() = (%q, %v), want (From content, true)", got, ok)
	}
}

func TestMarked_MarkersInReasoningContent(t *testing.T) {
	// Regression: even when Content has dirty text, markers in
	// ReasoningContent win because Content carries no markers.
	resp := respWith(
		"Some preamble text that should be ignored",
		markerStart+"\nExtracted from reasoning\n"+markerEnd,
		"",
	)
	if got, ok := Marked(resp, markerStart, markerEnd); !ok || got != "Extracted from reasoning" {
		t.Errorf("Marked() = (%q, %v), want (Extracted from reasoning, true)", got, ok)
	}
}

func TestMarked_MarkersInReasoningField(t *testing.T) {
	resp := respWith("", "", markerStart+"\nPrompt from Reasoning field\n"+markerEnd)
	if got, ok := Marked(resp, markerStart, markerEnd); !ok || got != "Prompt from Reasoning field" {
		t.Errorf("Marked() = (%q, %v), want (Prompt from Reasoning field, true)", got, ok)
	}
}

func TestMarked_EmptyContentBetweenMarkers(t *testing.T) {
	// Markers present but empty content — ("", true) is returned so the
	// caller can distinguish "model emitted empty payload" from "no markers".
	resp := respWith(markerStart+"\n\n"+markerEnd, "", "")
	if got, ok := Marked(resp, markerStart, markerEnd); !ok || got != "" {
		t.Errorf("Marked() = (%q, %v), want (\"\", true)", got, ok)
	}
}

func TestMarked_NoMarkers(t *testing.T) {
	if got, ok := Marked(respWith("plain text", "", ""), markerStart, markerEnd); ok {
		t.Errorf("Marked() = (%q, %v), want ok=false", got, ok)
	}
	if _, ok := Marked(nil, markerStart, markerEnd); ok {
		t.Error("Marked(nil) must report ok=false")
	}
}

// --- BetweenMarkers with overlapping markers (regression: slice-bounds panic) ---

func TestBetweenMarkers_SameStartAndEnd(t *testing.T) {
	// Regression: the end marker was searched from the start marker's own
	// offset, so start == end always matched at offset 0 and placed the
	// content end before the content start — a slice-bounds panic. Equal
	// delimiters (a ``` fence, a --- rule) must extract the text between
	// two successive occurrences instead, per the documented contract.
	for _, tc := range []struct {
		name  string
		in    string
		start string
		end   string
		want  string
		ok    bool
	}{
		{"fence pair", "```json\n{\"a\":1}\n```", "```", "```", "json\n{\"a\":1}", true},
		{"rule pair", "---\ntext\n---", "---", "---", "text", true},
		{"empty between adjacent", "XX", "X", "X", "", true},
		{"second marker missing", "---\nonly one", "---", "---", "", false},
		{"first pair wins", "---A---B---", "---", "---", "A", true},
	} {
		got, ok := BetweenMarkers(tc.in, tc.start, tc.end)
		if ok != tc.ok || got != tc.want {
			t.Errorf("%s: BetweenMarkers(%q, %q, %q) = (%q, %v), want (%q, %v)",
				tc.name, tc.in, tc.start, tc.end, got, ok, tc.want, tc.ok)
		}
	}
}

func TestBetweenMarkers_EndInsideStart(t *testing.T) {
	// "AB" occurs inside the start marker "AAB" (offset 1). The old search
	// from startIdx matched it there, making contentEnd (startIdx+1)
	// precede contentStart (startIdx+3) and panicking on the slice. The
	// search now begins after the start marker and extracts "xx".
	got, ok := BetweenMarkers("AABxxABzz", "AAB", "AB")
	if !ok || got != "xx" {
		t.Errorf("BetweenMarkers() = (%q, %v), want (%q, true)", got, ok, "xx")
	}
	// An end marker occurring ONLY inside the start marker has no
	// occurrence after it: ("", false), not a panic.
	if got, ok := BetweenMarkers("AAB", "AAB", "AB"); ok || got != "" {
		t.Errorf("BetweenMarkers() = (%q, %v), want (\"\", false)", got, ok)
	}
}

func TestMarked_FencePair(t *testing.T) {
	// Marked delegates to BetweenMarkers, so the same-delimiter fence pair
	// must work through it too (e.g. extracting fenced output).
	resp := respWith("```json\n{\"verdict\": \"ALLOW\"}\n```", "", "")
	want := "json\n{\"verdict\": \"ALLOW\"}"
	if got, ok := Marked(resp, "```", "```"); !ok || got != want {
		t.Errorf("Marked() = (%q, %v), want (%q, true)", got, ok, want)
	}
}

// --- ParseJSON ---

type extractResult struct {
	Translated string   `json:"translated"`
	Keywords   []string `json:"keywords"`
}

func TestParseJSON_PlainObject(t *testing.T) {
	resp := respWith(`{"translated":"fix the login bug","keywords":["auth","login"]}`, "", "")
	got, err := ParseJSON[extractResult](resp)
	if err != nil {
		t.Fatalf("ParseJSON() error = %v", err)
	}
	if got.Translated != "fix the login bug" || len(got.Keywords) != 2 {
		t.Errorf("ParseJSON() = %+v, want translated+2 keywords", got)
	}
}

func TestParseJSON_FencedObject(t *testing.T) {
	resp := respWith("```json\n{\"translated\":\"x\"}\n```", "", "")
	got, err := ParseJSON[extractResult](resp)
	if err != nil {
		t.Fatalf("ParseJSON() error = %v", err)
	}
	if got.Translated != "x" {
		t.Errorf("ParseJSON() = %+v, want translated=x", got)
	}
}

func TestParseJSON_ObjectInProse(t *testing.T) {
	// Regression from the judge: JSON embedded in prose is recovered by
	// llm.ExtractJSON's backward scan.
	resp := respWith("Here is my assessment:\n{\"verdict\":\"CONFIRM\",\"reasoning\":\"destructive\"}\nDone.", "", "")
	got, err := ParseJSON[map[string]string](resp)
	if err != nil {
		t.Fatalf("ParseJSON() error = %v", err)
	}
	if got["verdict"] != "CONFIRM" || got["reasoning"] != "destructive" {
		t.Errorf("ParseJSON() = %v, want verdict/reasoning keys", got)
	}
}

func TestParseJSON_FallbackToReasoningContent(t *testing.T) {
	resp := respWith("no json here at all", `{"translated":"from reasoning"}`, "")
	got, err := ParseJSON[extractResult](resp)
	if err != nil {
		t.Fatalf("ParseJSON() error = %v", err)
	}
	if got.Translated != "from reasoning" {
		t.Errorf("ParseJSON() = %+v, want translated=from reasoning", got)
	}
}

func TestParseJSON_NoCandidates(t *testing.T) {
	if _, err := ParseJSON[extractResult](respWith("", "", "")); !errors.Is(err, ErrNoJSON) {
		t.Errorf("ParseJSON() error = %v, want ErrNoJSON", err)
	}
	if _, err := ParseJSON[extractResult](nil); !errors.Is(err, ErrNoJSON) {
		t.Errorf("ParseJSON(nil) error = %v, want ErrNoJSON", err)
	}
}

func TestParseJSON_AllFieldsUndecodable(t *testing.T) {
	resp := respWith("prose one", "prose two", "prose three")
	_, err := ParseJSON[extractResult](resp)
	if err == nil {
		t.Fatal("ParseJSON() expected an error for undecodable fields")
	}
	if !errors.Is(err, ErrNoJSON) {
		t.Errorf("ParseJSON() error = %v, want it to wrap ErrNoJSON", err)
	}
}
