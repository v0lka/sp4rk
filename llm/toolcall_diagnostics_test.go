package llm

import (
	"bytes"
	"encoding/json"
	"log/slog"
	"strings"
	"testing"
)

func newDebugLogger(buf *bytes.Buffer) *slog.Logger {
	return slog.New(slog.NewTextHandler(buf, &slog.HandlerOptions{Level: slog.LevelDebug}))
}

func TestLogToolCallArguments_WarnsOnMalformedArguments(t *testing.T) {
	var buf bytes.Buffer
	logToolCallArguments(newDebugLogger(&buf), "acme", []ToolCall{
		{ID: "call_1", Name: "store_fact", Input: json.RawMessage(`{"keywords": [git, cache]}`)},
	})
	s := buf.String()
	for _, want := range []string{"level=WARN", "provider=acme", "tool=store_fact", "valid_json=false"} {
		if !strings.Contains(s, want) {
			t.Errorf("log output missing %q: %s", want, s)
		}
	}
}

func TestLogToolCallArguments_DebugsWellFormedArguments(t *testing.T) {
	var buf bytes.Buffer
	logToolCallArguments(newDebugLogger(&buf), "acme", []ToolCall{
		{ID: "call_2", Name: "store_fact", Input: json.RawMessage(`{"keywords":["a","b","c"]}`)},
	})
	s := buf.String()
	if !strings.Contains(s, "level=DEBUG") || !strings.Contains(s, "valid_json=true") {
		t.Errorf("well-formed log = %q, want level=DEBUG valid_json=true", s)
	}
}

func TestLogToolCallArguments_NoCallsWritesNothing(t *testing.T) {
	var buf bytes.Buffer
	logToolCallArguments(newDebugLogger(&buf), "acme", nil)
	if buf.Len() != 0 {
		t.Errorf("empty calls wrote %q, want nothing", buf.String())
	}
}

func TestLogToolCallArguments_TruncatesLongArguments(t *testing.T) {
	long := json.RawMessage(strings.Repeat("x", maxLoggedToolArgs+50))
	var buf bytes.Buffer
	logToolCallArguments(newDebugLogger(&buf), "acme", []ToolCall{{Name: "t", Input: long}})
	if !strings.Contains(buf.String(), "...(truncated)") {
		t.Errorf("long arguments were not truncated: %s", buf.String())
	}
}
