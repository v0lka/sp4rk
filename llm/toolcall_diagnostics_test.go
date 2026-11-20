package llm

import (
	"bytes"
	"encoding/json"
	"log/slog"
	"strings"
	"testing"
	"unicode/utf8"
)

func newDebugLogger(buf *bytes.Buffer) *slog.Logger {
	return slog.New(slog.NewTextHandler(buf, &slog.HandlerOptions{Level: slog.LevelDebug}))
}

func TestLogToolCallArguments_WarnsOnMalformedArguments(t *testing.T) {
	var buf bytes.Buffer
	logToolCallArguments(newDebugLogger(&buf), "acme", []ToolCall{
		{ID: "call_1", Name: "store_fact", Input: json.RawMessage(`{"keywords": [git, cache]}`)},
	}, nil)
	s := buf.String()
	for _, want := range []string{"level=WARN", "provider=acme", "tool=store_fact", "valid_json=false",
		"repaired=false", "dispatch_outcome=parse_error"} {
		if !strings.Contains(s, want) {
			t.Errorf("log output missing %q: %s", want, s)
		}
	}
}

func TestLogToolCallArguments_DebugsWellFormedArguments(t *testing.T) {
	var buf bytes.Buffer
	logToolCallArguments(newDebugLogger(&buf), "acme", []ToolCall{
		{ID: "call_2", Name: "store_fact", Input: json.RawMessage(`{"keywords":["a","b","c"]}`)},
	}, nil)
	s := buf.String()
	if !strings.Contains(s, "level=DEBUG") || !strings.Contains(s, "valid_json=true") {
		t.Errorf("well-formed log = %q, want level=DEBUG valid_json=true", s)
	}
	if !strings.Contains(s, "repaired=false") || !strings.Contains(s, "dispatch_outcome=ok") {
		t.Errorf("well-formed log = %q, want repaired=false dispatch_outcome=ok", s)
	}
}

func TestLogToolCallArguments_NoCallsWritesNothing(t *testing.T) {
	var buf bytes.Buffer
	logToolCallArguments(newDebugLogger(&buf), "acme", nil, nil)
	if buf.Len() != 0 {
		t.Errorf("empty calls wrote %q, want nothing", buf.String())
	}
}

func TestLogToolCallArguments_TruncatesLongArguments(t *testing.T) {
	long := json.RawMessage(strings.Repeat("x", maxLoggedToolArgs+50))
	var buf bytes.Buffer
	logToolCallArguments(newDebugLogger(&buf), "acme", []ToolCall{{Name: "t", Input: long}}, nil)
	if !strings.Contains(buf.String(), "...(truncated)") {
		t.Errorf("long arguments were not truncated: %s", buf.String())
	}
}

func TestRepairToolCallInput(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string // expected repaired payload when ok; ignored otherwise
		ok    bool
	}{
		{"unquoted duration in timeout key", `{"timeout": 120s}`, `{"timeout": "120s"}`, true},
		{"no surrounding whitespace", `{"timeout":120s}`, `{"timeout":"120s"}`, true},
		{"fractional duration", `{"timeout": 1.5ms}`, `{"timeout": "1.5ms"}`, true},
		{"microseconds", `{"timeout": 750µs}`, `{"timeout": "750µs"}`, true},
		{"hours with trailing space", `{"timeout": 2h }`, `{"timeout": "2h" }`, true},
		{"trailing comma in object", `{"a": 1,}`, `{"a": 1}`, true},
		{"trailing comma in array", `["a",]`, `["a"]`, true},
		{"duration and trailing comma", `{"timeout": 30s,}`, `{"timeout": "30s"}`, true},
		{"nested known key", `{"opts":{"timeout":5s}}`, `{"opts":{"timeout":"5s"}}`, true},
		{"trailing comma after unitless number", `{"timeout": 120,}`, `{"timeout": 120}`, true},
		// Quirk 3: raw \n, \r, \t bytes inside string values — DeepSeek-shaped
		// multi-line command payloads from the 11-10 burst.
		{"raw newline in string value", "{\"command\": \"echo a\necho b\"}", `{"command": "echo a\necho b"}`, true},
		{"raw CRLF and tab in string value", "{\"command\": \"make test\r\n\tmake lint\"}", `{"command": "make test\r\n\tmake lint"}`, true},
		{"raw newline plus trailing comma", "{\"content\": \"line1\nline2\",}", `{"content": "line1\nline2"}`, true},
		{"raw newline plus unquoted duration", "{\"timeout\": 30s, \"note\": \"see\nbelow\"}", `{"timeout": "30s", "note": "see\nbelow"}`, true},
		{"raw newline in string key", "{\"my\nkey\": 1}", `{"my\nkey": 1}`, true},
		// Fail-closed boundaries for quirk 3.
		{"raw newline outside strings is valid JSON", "{\"a\": 1,\n\"b\": 2}", "", false},
		{"escaped newline untouched", `{"a": "x\ny"}`, "", false},
		{"other raw control char stays unrepaired", "{\"a\": \"x\v y\"}", "", false},
		{"unterminated string with raw newline stays unrepaired", "{\"a\": \"bc\n", "", false},
		{"garbage stays unrepaired", "not json", "", false},
		{"unclosed object stays unrepaired", `{"timeout": 120s`, "", false},
		{"unclosed string stays unrepaired", `{"a": "bc`, "", false},
		{"compound duration stays unrepaired", `{"timeout": 1h30m}`, "", false},
		{"unknown key stays unrepaired", `{"delay": 5s}`, "", false},
		{"suffixed unit stays unrepaired", `{"timeout": 120seconds}`, "", false},
		{"uppercase unit stays unrepaired", `{"timeout": 120S}`, "", false},
		{"split unit stays unrepaired", `{"timeout": 120 s}`, "", false},
		{"negative duration stays unrepaired", `{"timeout": -5s}`, "", false},
		{"already-quoted duration untouched", `{"timeout": "120s"}`, "", false},
		{"unitless number untouched", `{"timeout": 120}`, "", false},
		{"arbitrary valid object untouched", `{"keywords":["a","b"]}`, "", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := repairToolCallInput([]byte(tt.input))
			if ok != tt.ok {
				t.Fatalf("repairToolCallInput(%q) ok = %v, want %v (got %q)", tt.input, ok, tt.ok, got)
			}
			if !ok {
				return
			}
			if string(got) != tt.want {
				t.Errorf("repairToolCallInput(%q) = %q, want %q", tt.input, got, tt.want)
			}
			if !json.Valid(got) {
				t.Errorf("repaired payload %q is not valid JSON", got)
			}
		})
	}
}

func TestSanitizeToolCallArguments(t *testing.T) {
	calls := []ToolCall{
		{ID: "call_1", Name: "bash", Input: json.RawMessage(`{"timeout": 120s}`)},
		{ID: "call_2", Name: "read", Input: json.RawMessage(`{"path":"x"}`)},
		{ID: "call_3", Name: "junk", Input: json.RawMessage(`total garbage`)},
	}
	san := sanitizeToolCallArguments(calls)

	if !san.repaired[0] {
		t.Errorf("call_1 not repaired: %v", san.repaired)
	}
	if string(calls[0].Input) != `{"timeout": "120s"}` {
		t.Errorf("call_1 input = %q, want repaired payload", calls[0].Input)
	}
	if string(san.originals[0]) != `{"timeout": 120s}` {
		t.Errorf("call_1 original = %q, want the pre-repair payload", san.originals[0])
	}
	if san.repaired[1] {
		t.Errorf("call_2 (valid JSON) was flagged repaired")
	}
	if string(calls[1].Input) != `{"path":"x"}` {
		t.Errorf("call_2 input changed: %q", calls[1].Input)
	}
	if san.originals[1] != nil {
		t.Errorf("call_2 original recorded: %q", san.originals[1])
	}
	if san.repaired[2] {
		t.Errorf("call_3 (unrepairable garbage) was flagged repaired")
	}
	if string(calls[2].Input) != `total garbage` {
		t.Errorf("call_3 input changed: %q", calls[2].Input)
	}
}

// TestSanitizeToolCallArguments_EscapesRawNewline pins the quirk-3 repair at
// the sanitation step with a DeepSeek-shaped payload from the 11-10 burst: a
// multi-line shell command emitted with a literal newline inside the string
// value. The repaired payload must replace the raw one in the call, the
// original must be recorded for the log, and a second call with the SAME
// payload already escaped must dispatch untouched.
func TestSanitizeToolCallArguments_EscapesRawNewline(t *testing.T) {
	raw := []byte("{\"command\": \"echo a\necho b\"}")
	calls := []ToolCall{
		{ID: "call_1", Name: "bash", Input: json.RawMessage(append([]byte(nil), raw...))},
		{ID: "call_2", Name: "bash", Input: json.RawMessage(`{"command": "echo a\necho b"}`)},
	}
	san := sanitizeToolCallArguments(calls)

	if !san.repaired[0] {
		t.Fatalf("raw-newline payload must be repaired, repaired=%v", san.repaired)
	}
	if string(calls[0].Input) != `{"command": "echo a\necho b"}` {
		t.Errorf("call_1 input = %q, want the escaped payload", calls[0].Input)
	}
	if string(san.originals[0]) != string(raw) {
		t.Errorf("call_1 original = %q, want the pre-repair payload %q", san.originals[0], raw)
	}
	if !json.Valid(calls[0].Input) {
		t.Errorf("repaired call_1 input %q is not valid JSON", calls[0].Input)
	}
	if san.repaired[1] {
		t.Errorf("already-escaped payload must not be flagged repaired")
	}
	if san.originals[1] != nil {
		t.Errorf("call_2 original recorded: %q", san.originals[1])
	}
}

func TestFinalizeToolCallArguments_RepairedArgsDispatchWithWarn(t *testing.T) {
	var buf bytes.Buffer
	calls := []ToolCall{
		{ID: "call_1", Name: "bash", Input: json.RawMessage(`{"timeout": 120s}`)},
	}
	finalizeToolCallArguments(newDebugLogger(&buf), "acme", calls)

	if string(calls[0].Input) != `{"timeout": "120s"}` {
		t.Fatalf("input = %q, want the repaired payload that reaches dispatch", calls[0].Input)
	}
	if !json.Valid(calls[0].Input) {
		t.Errorf("repaired input %q is not valid JSON", calls[0].Input)
	}
	s := buf.String()
	for _, want := range []string{"level=WARN", "valid_json=true", "repaired=true", "dispatch_outcome=ok", "raw_arguments="} {
		if !strings.Contains(s, want) {
			t.Errorf("log output missing %q: %s", want, s)
		}
	}
}

func TestFinalizeToolCallArguments_SanitizesEvenWhenLoggingDisabled(t *testing.T) {
	var buf bytes.Buffer
	logger := slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelError}))
	calls := []ToolCall{
		{ID: "call_1", Name: "bash", Input: json.RawMessage(`{"timeout": 120s}`)},
	}
	finalizeToolCallArguments(logger, "acme", calls)
	if string(calls[0].Input) != `{"timeout": "120s"}` {
		t.Errorf("input = %q, want repaired payload despite logging being disabled", calls[0].Input)
	}
	if buf.Len() != 0 {
		t.Errorf("error-level logger wrote tool-call diagnostics: %s", buf.String())
	}
}

func TestTruncateArgsForLog_CutsOnRuneBoundary(t *testing.T) {
	// 341 three-byte runes occupy exactly 1023 bytes, so byte offset
	// maxLoggedToolArgs (1024) falls mid-rune in the 342nd: the cut must back
	// off to the rune boundary instead of splitting it.
	s := strings.Repeat("日", 341) + "日" + strings.Repeat("x", 64)
	if len(s) <= maxLoggedToolArgs {
		t.Fatalf("test payload is %d bytes, want more than maxLoggedToolArgs=%d", len(s), maxLoggedToolArgs)
	}
	got := truncateArgsForLog(s)
	if !strings.HasSuffix(got, "...(truncated)") {
		t.Fatalf("truncated payload lost its cut marker: %q", got)
	}
	prefix := strings.TrimSuffix(got, "...(truncated)")
	if !utf8.ValidString(prefix) {
		t.Errorf("truncated prefix is not valid UTF-8 (a multi-byte rune was split at the cut): %q", prefix)
	}
	if want := 3 * 341; len(prefix) != want {
		t.Errorf("truncated prefix is %d bytes, want %d (cut backed off to the rune boundary)", len(prefix), want)
	}
}
