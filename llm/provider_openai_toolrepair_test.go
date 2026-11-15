package llm

import (
	"bytes"
	"encoding/json"
	"log/slog"
	"strings"
	"testing"

	oai "github.com/openai/openai-go"
)

// chatCompletionWithToolArgs builds a full chat-completion response whose single
// tool call carries the given raw (unescaped at the wire level) arguments
// string, exactly as a provider would return it.
func chatCompletionWithToolArgs(t *testing.T, args string) oai.ChatCompletion {
	t.Helper()
	fullResponse := `{
		"id":"chatcmpl-test",
		"object":"chat.completion",
		"created":1234567890,
		"model":"test-model",
		"choices":[{
			"index":0,
			"message":{
				"role":"assistant",
				"content":"",
				"tool_calls":[{"id":"call-1","type":"function","function":{"name":"bash","arguments":"` + strings.ReplaceAll(args, `"`, `\"`) + `"}}]
			},
			"finish_reason":"tool_calls"
		}],
		"usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30}
	}`
	var completion oai.ChatCompletion
	if err := json.Unmarshal([]byte(fullResponse), &completion); err != nil {
		t.Fatalf("failed to unmarshal ChatCompletion: %v", err)
	}
	return completion
}

func newToolRepairTestProvider(buf *bytes.Buffer) *OpenAIProvider {
	p, err := NewOpenAIProvider(OpenAIProviderConfig{
		Name:   "acme",
		APIKey: "test-key",
		Logger: slog.New(slog.NewTextHandler(buf, &slog.HandlerOptions{Level: slog.LevelDebug})),
	})
	if err != nil {
		panic(err) // fixture construction; callers run inside tests
	}
	return p
}

// TestOpenAIProvider_RepairedToolCallArgsFlowToDispatch proves the end-to-end
// AC: an unquoted duration scalar (a known provider quirk) in the tool-call
// arguments is repaired at finalization, the repaired payload is what the
// ChatResponse hands to dispatch (where it parses instead of surfacing a
// ParseInputError), and the quirk is logged as a Warn with repaired=true.
func TestOpenAIProvider_RepairedToolCallArgsFlowToDispatch(t *testing.T) {
	var buf bytes.Buffer
	p := newToolRepairTestProvider(&buf)

	completion := chatCompletionWithToolArgs(t, `{"timeout": 120s}`)
	result := p.convertChatResponseMessage(completion.Choices[0].Message)

	if len(result.ToolCalls) != 1 {
		t.Fatalf("got %d tool calls, want 1", len(result.ToolCalls))
	}
	got := string(result.ToolCalls[0].Input)
	if got != `{"timeout": "120s"}` {
		t.Errorf("dispatched input = %q, want the repaired payload", got)
	}
	if !json.Valid(result.ToolCalls[0].Input) {
		t.Errorf("dispatched input %q is not valid JSON", got)
	}
	s := buf.String()
	for _, want := range []string{"level=WARN", "tool=bash", "repaired=true", "dispatch_outcome=ok", "raw_arguments="} {
		if !strings.Contains(s, want) {
			t.Errorf("log output missing %q: %s", want, s)
		}
	}
}

// TestOpenAIProvider_UnrepairableToolCallArgsFlowUnchanged proves the negative
// AC: garbage the repair cannot fix flows byte-identical to dispatch (and
// surfaces downstream as ParseInputError, which the tools layer owns), logged
// as a Warn with repaired=false and dispatch_outcome=parse_error.
func TestOpenAIProvider_UnrepairableToolCallArgsFlowUnchanged(t *testing.T) {
	var buf bytes.Buffer
	p := newToolRepairTestProvider(&buf)

	completion := chatCompletionWithToolArgs(t, `total garbage`)
	result := p.convertChatResponseMessage(completion.Choices[0].Message)

	if len(result.ToolCalls) != 1 {
		t.Fatalf("got %d tool calls, want 1", len(result.ToolCalls))
	}
	if got := string(result.ToolCalls[0].Input); got != `total garbage` {
		t.Errorf("dispatched input = %q, want the provider bytes unchanged", got)
	}
	// The payload a tool would receive still fails to parse — the downstream
	// ParseInputError behavior is unchanged by sanitation.
	var dummy map[string]any
	if err := json.Unmarshal(result.ToolCalls[0].Input, &dummy); err == nil {
		t.Errorf("garbage payload unexpectedly parsed")
	}
	s := buf.String()
	for _, want := range []string{"level=WARN", "tool=bash", "repaired=false", "dispatch_outcome=parse_error"} {
		if !strings.Contains(s, want) {
			t.Errorf("log output missing %q: %s", want, s)
		}
	}
}

// TestOpenAIProvider_ValidToolCallArgsUntouched proves valid JSON flows
// byte-identical and stays at Debug with repaired=false.
func TestOpenAIProvider_ValidToolCallArgsUntouched(t *testing.T) {
	var buf bytes.Buffer
	p := newToolRepairTestProvider(&buf)

	completion := chatCompletionWithToolArgs(t, `{"timeout":"120s","cmd":"ls"}`)
	result := p.convertChatResponseMessage(completion.Choices[0].Message)

	if len(result.ToolCalls) != 1 {
		t.Fatalf("got %d tool calls, want 1", len(result.ToolCalls))
	}
	if got := string(result.ToolCalls[0].Input); got != `{"timeout":"120s","cmd":"ls"}` {
		t.Errorf("dispatched input = %q, want byte-identical valid JSON", got)
	}
	s := buf.String()
	for _, want := range []string{"level=DEBUG", "repaired=false", "dispatch_outcome=ok"} {
		if !strings.Contains(s, want) {
			t.Errorf("log output missing %q: %s", want, s)
		}
	}
	if strings.Contains(s, "level=WARN") {
		t.Errorf("valid arguments logged at Warn: %s", s)
	}
}
