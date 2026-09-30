package agent

import (
	"context"
	"testing"

	"github.com/v0lka/sp4rk/llm"
)

// captureEvents records assistant streaming events with their payloads.
type captureEvents struct {
	NoopEvents
	chunks      []string
	doneContent string
	doneIn      int
	doneOut     int
	doneCount   int
}

func (c *captureEvents) AssistantChunk(content string) { c.chunks = append(c.chunks, content) }

func (c *captureEvents) AssistantDone(content string, in, out int) {
	c.doneContent = content
	c.doneIn = in
	c.doneOut = out
	c.doneCount++
}

// streamCaller is an LLMCaller that optionally streams the configured deltas
// through req.DeltaSink before returning a canned response.
type streamCaller struct {
	deltas  []llm.StreamDelta
	resp    *llm.ChatResponse
	gotSink bool
}

func (s *streamCaller) Call(_ context.Context, req llm.ChatRequest) (*llm.ChatResponse, error) {
	if req.DeltaSink != nil {
		s.gotSink = true
		for _, d := range s.deltas {
			if err := req.DeltaSink(d); err != nil {
				return nil, err
			}
		}
	}
	return s.resp, nil
}

func endTurnResponse(content string) *llm.ChatResponse {
	return &llm.ChatResponse{
		Message:    llm.Message{Role: "assistant", Content: content},
		StopReason: "end_turn",
		Usage:      llm.TokenUsage{InputTokens: 3, OutputTokens: 2},
	}
}

// TestExecutor_Streaming_ForwardsDeltas verifies that with streaming enabled the
// executor forwards each provider delta as an AssistantChunk and emits exactly
// one AssistantDone with the full text (no duplicate full-text chunk).
func TestExecutor_Streaming_ForwardsDeltas(t *testing.T) {
	caller := &streamCaller{
		deltas: []llm.StreamDelta{{Text: "Hello "}, {Text: "world"}},
		resp:   endTurnResponse("Hello world"),
	}
	ev := &captureEvents{}
	exec := NewExecutor(caller, newMockToolExecutor(), 10,
		WithTokenCounter(&mockTokenCounter{}),
		WithEvents(ev),
		WithStreaming(true),
	)

	if _, err := exec.Run(context.Background(), nil, newMockContextManager()); err != nil {
		t.Fatalf("Run: %v", err)
	}

	if !caller.gotSink {
		t.Fatal("expected the caller to receive a delta sink")
	}
	if len(ev.chunks) != 2 || ev.chunks[0] != "Hello " || ev.chunks[1] != "world" {
		t.Errorf("chunks = %q, want [Hello , world]", ev.chunks)
	}
	if ev.doneCount != 1 || ev.doneContent != "Hello world" {
		t.Errorf("done events = %d content=%q, want 1 %q", ev.doneCount, ev.doneContent, "Hello world")
	}
}

// TestExecutor_Streaming_FallsBackWhenProviderIgnoresSink verifies graceful
// degradation: when streaming is enabled but the provider delivers no deltas,
// the executor emits the full text as a single AssistantChunk.
func TestExecutor_Streaming_FallsBackWhenProviderIgnoresSink(t *testing.T) {
	caller := &streamCaller{resp: endTurnResponse("Hello world")}
	ev := &captureEvents{}
	exec := NewExecutor(caller, newMockToolExecutor(), 10,
		WithTokenCounter(&mockTokenCounter{}),
		WithEvents(ev),
		WithStreaming(true),
	)

	if _, err := exec.Run(context.Background(), nil, newMockContextManager()); err != nil {
		t.Fatalf("Run: %v", err)
	}

	if !caller.gotSink {
		t.Fatal("expected the caller to receive a delta sink")
	}
	if len(ev.chunks) != 1 || ev.chunks[0] != "Hello world" {
		t.Errorf("chunks = %q, want single fallback [Hello world]", ev.chunks)
	}
	if ev.doneCount != 1 || ev.doneContent != "Hello world" {
		t.Errorf("done events = %d content=%q, want 1 %q", ev.doneCount, ev.doneContent, "Hello world")
	}
}

// TestExecutor_Streaming_Disabled_BehaviorUnchanged verifies that with streaming
// off the caller receives no delta sink and the full text is emitted as one
// chunk (the pre-existing behavior).
func TestExecutor_Streaming_Disabled_BehaviorUnchanged(t *testing.T) {
	caller := &streamCaller{resp: endTurnResponse("Hello world")}
	ev := &captureEvents{}
	exec := NewExecutor(caller, newMockToolExecutor(), 10,
		WithTokenCounter(&mockTokenCounter{}),
		WithEvents(ev),
	)

	if _, err := exec.Run(context.Background(), nil, newMockContextManager()); err != nil {
		t.Fatalf("Run: %v", err)
	}

	if caller.gotSink {
		t.Error("streaming disabled: caller must not receive a delta sink")
	}
	if len(ev.chunks) != 1 || ev.chunks[0] != "Hello world" {
		t.Errorf("chunks = %q, want single [Hello world]", ev.chunks)
	}
	if ev.doneCount != 1 {
		t.Errorf("done events = %d, want 1", ev.doneCount)
	}
}

// TestExecutor_Streaming_SkippedWhenAssistantEventsSuppressed verifies that
// suppressAssistantEvents wins: no sink is set and no assistant events fire.
func TestExecutor_Streaming_SkippedWhenAssistantEventsSuppressed(t *testing.T) {
	caller := &streamCaller{
		deltas: []llm.StreamDelta{{Text: "x"}},
		resp:   endTurnResponse("x"),
	}
	ev := &captureEvents{}
	exec := NewExecutor(caller, newMockToolExecutor(), 10,
		WithTokenCounter(&mockTokenCounter{}),
		WithEvents(ev),
		WithStreaming(true),
		WithSuppressAssistantEvents(true),
	)

	if _, err := exec.Run(context.Background(), nil, newMockContextManager()); err != nil {
		t.Fatalf("Run: %v", err)
	}

	if caller.gotSink {
		t.Error("assistant events suppressed: caller must not receive a delta sink")
	}
	if len(ev.chunks) != 0 || ev.doneCount != 0 {
		t.Errorf("expected no assistant events, got chunks=%q done=%d", ev.chunks, ev.doneCount)
	}
}
