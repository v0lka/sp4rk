package orchestration

import (
	"context"
	"sync"
	"testing"

	"github.com/v0lka/sp4rk/agent"
	"github.com/v0lka/sp4rk/llm"
)

// condStreamLLM is a minimal agent.LLMCaller that delivers configured deltas
// through req.DeltaSink before returning a text-only end_turn response.
type condStreamLLM struct {
	deltas   []llm.StreamDelta
	sinkSeen bool
}

func (l *condStreamLLM) Call(_ context.Context, req llm.ChatRequest) (*llm.ChatResponse, error) {
	if req.DeltaSink != nil {
		l.sinkSeen = true
		for _, d := range l.deltas {
			if err := req.DeltaSink(d); err != nil {
				return nil, err
			}
		}
	}
	return &llm.ChatResponse{
		Message:    llm.Message{Role: "assistant", Content: "streamed final"},
		StopReason: "end_turn",
		Usage:      llm.TokenUsage{InputTokens: 1, OutputTokens: 2},
	}, nil
}

type condStreamEvents struct {
	*agent.NoopEvents
	mu     sync.Mutex
	chunks []string
	done   int
}

func (e *condStreamEvents) AssistantChunk(c string) {
	e.mu.Lock()
	e.chunks = append(e.chunks, c)
	e.mu.Unlock()
}

func (e *condStreamEvents) AssistantDone(string, int, int) {
	e.mu.Lock()
	e.done++
	e.mu.Unlock()
}

// runConductorStreaming runs a Conductor with the given Streaming flag and a
// text-only streaming caller, returning the caller and events for assertions.
func runConductorStreaming(t *testing.T, streaming bool) (*condStreamLLM, *condStreamEvents) {
	t.Helper()
	caller := &condStreamLLM{deltas: []llm.StreamDelta{{Text: "Hello "}, {Text: "world"}}}
	ev := &condStreamEvents{NoopEvents: &agent.NoopEvents{}}
	cfg := ConductorConfig{
		LLM:   caller,
		Tools: condMockTools{},
		ContextFactory: func(_ string, _ llm.ModelMetadata, _ string, _ ...PruningOverride) agent.ContextManager {
			return &compactFakeCM{seq: &seqRecorder{}}
		},
		SystemPrompt: func(_ context.Context, _ string, _ llm.ModelMetadata) string { return "sys" },
		MaxSteps:     10,
		Streaming:    streaming,
	}
	cond := NewConductor(cfg)
	if _, err := cond.Run(context.Background(), "say hi", NewMapBlackboard(), nil, ev, "sliding_window"); err != nil {
		t.Fatalf("Run: %v", err)
	}
	return caller, ev
}

// TestConductor_Streaming_DeliversDeltas verifies ConductorConfig.Streaming
// wires WithStreaming onto the executor: deltas arrive as AssistantChunk events.
func TestConductor_Streaming_DeliversDeltas(t *testing.T) {
	caller, ev := runConductorStreaming(t, true)

	if !caller.sinkSeen {
		t.Fatal("Streaming=true: executor did not install a delta sink")
	}
	if len(ev.chunks) != 2 || ev.chunks[0] != "Hello " || ev.chunks[1] != "world" {
		t.Errorf("chunks = %q, want [Hello , world]", ev.chunks)
	}
	if ev.done != 1 {
		t.Errorf("AssistantDone count = %d, want 1", ev.done)
	}
}

// TestConductor_Streaming_Disabled verifies the default: no sink, single chunk.
func TestConductor_Streaming_Disabled(t *testing.T) {
	caller, ev := runConductorStreaming(t, false)

	if caller.sinkSeen {
		t.Error("Streaming=false: executor must not install a delta sink")
	}
	if len(ev.chunks) != 1 || ev.chunks[0] != "streamed final" {
		t.Errorf("chunks = %q, want single fallback [streamed final]", ev.chunks)
	}
	if ev.done != 1 {
		t.Errorf("AssistantDone count = %d, want 1", ev.done)
	}
}
