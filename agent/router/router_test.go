package router

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/v0lka/sp4rk/llm"
	"github.com/v0lka/sp4rk/tools"
)

// mockLLMCaller is a minimal mock implementation of agent.LLMCaller for testing.
type mockLLMCaller struct {
	responses []*llm.ChatResponse
	callIdx   int
	calls     []llm.ChatRequest
	callFn    func(ctx context.Context, req llm.ChatRequest) (*llm.ChatResponse, error)
	err       error
}

func (m *mockLLMCaller) Call(ctx context.Context, req llm.ChatRequest) (*llm.ChatResponse, error) {
	m.calls = append(m.calls, req)
	if m.callFn != nil {
		return m.callFn(ctx, req)
	}
	if m.err != nil {
		return nil, m.err
	}
	if len(m.responses) == 0 {
		return &llm.ChatResponse{Message: llm.Message{Role: "assistant", Content: "{}"}}, nil
	}
	resp := m.responses[m.callIdx%len(m.responses)]
	m.callIdx++
	return resp, nil
}

func (m *mockLLMCaller) lastCall() llm.ChatRequest {
	if len(m.calls) == 0 {
		return llm.ChatRequest{}
	}
	return m.calls[len(m.calls)-1]
}

func newTestRouter(mock *mockLLMCaller, historyWindow int) *Router {
	return New(mock, Config{
		SystemPrompt:  "Tools: {{AVAILABLE-TOOLS}}\nSkills: {{AVAILABLE-SKILLS}}",
		HistoryWindow: historyWindow,
	})
}

func TestRoute_ReturnsValidRoutingDecision(t *testing.T) {
	mock := &mockLLMCaller{
		responses: []*llm.ChatResponse{{
			Message: llm.Message{
				Role:    "assistant",
				Content: `{"domain":"code","complexity":2,"needs_clarification":false}`,
			},
		}},
	}

	r := newTestRouter(mock, 5)

	decision, err := r.Route(context.Background(), "read the config file", nil, nil, nil)
	if err != nil {
		t.Fatalf("Route returned error: %v", err)
	}

	if decision.Domain != "code" {
		t.Errorf("expected domain 'code', got '%s'", decision.Domain)
	}
	if decision.Complexity != 2 {
		t.Errorf("expected complexity 2, got %d", decision.Complexity)
	}
	if decision.NeedsClarification {
		t.Errorf("expected needs_clarification false, got true")
	}
}

func TestRoute_PassesToolsInPrompt(t *testing.T) {
	mock := &mockLLMCaller{
		responses: []*llm.ChatResponse{{
			Message: llm.Message{
				Role:    "assistant",
				Content: `{"mode":"react","domain":"code","complexity":2}`,
			},
		}},
	}

	r := newTestRouter(mock, 5)

	availableTools := []tools.ToolDescriptor{
		{Name: "bash_exec", Description: "Execute bash commands"},
		{Name: "file_read", Description: "Read file contents"},
	}

	_, err := r.Route(context.Background(), "run a command", availableTools, nil, nil)
	if err != nil {
		t.Fatalf("Route returned error: %v", err)
	}

	systemMessage := mock.lastCall().Messages[0]
	if systemMessage.Role != "system" {
		t.Fatalf("expected first message to be system, got '%s'", systemMessage.Role)
	}
	if !strings.Contains(systemMessage.Content, "bash_exec") {
		t.Error("system prompt should contain 'bash_exec'")
	}
	if !strings.Contains(systemMessage.Content, "file_read") {
		t.Error("system prompt should contain 'file_read'")
	}
	if !strings.Contains(systemMessage.Content, "Execute bash commands") {
		t.Error("system prompt should contain tool description 'Execute bash commands'")
	}
}

// TestRoute_ToolMatchingPromptIncludesToolNames verifies that with semantic
// tool matching enabled the routing prompt still lists every available tool by
// name (AVAILABLE-TOOLS is substituted unconditionally, independent of the
// toolMatching flag) and carries the matched_tools selection instruction, so
// the router LLM can actually pick from real tool names.
func TestRoute_ToolMatchingPromptIncludesToolNames(t *testing.T) {
	mock := &mockLLMCaller{
		responses: []*llm.ChatResponse{{
			Message: llm.Message{
				Role:    "assistant",
				Content: `{"domain":"code","complexity":2,"needs_clarification":false,"matched_tools":["read_file"]}`,
			},
		}},
	}

	r := New(mock, Config{
		SystemPrompt:  "Tools: {{AVAILABLE-TOOLS}}\nSkills: {{AVAILABLE-SKILLS}}\nMatching: {{TOOL-MATCHING}}\nSchema: {{JSON-OUTPUT-SCHEMA}}",
		HistoryWindow: 5,
	})
	r.SetToolMatching(true)

	availableTools := []tools.ToolDescriptor{
		{Name: "read_file", Description: "Read file contents"},
		{Name: "bash_exec", Description: "Execute bash commands"},
	}

	decision, err := r.Route(context.Background(), "read the config file", availableTools, nil, nil)
	if err != nil {
		t.Fatalf("Route returned error: %v", err)
	}
	if len(decision.MatchedTools) != 1 || decision.MatchedTools[0] != "read_file" {
		t.Errorf("expected matched_tools [read_file], got %v", decision.MatchedTools)
	}

	prompt := mock.lastCall().Messages[0].Content
	for _, want := range []string{"read_file", "bash_exec", "matched_tools"} {
		if !strings.Contains(prompt, want) {
			t.Errorf("routing prompt with tool matching enabled must contain %q; prompt:\n%s", want, prompt)
		}
	}
}

// TestRoute_ParseErrorAfterRepairIsSentinel verifies that when the routing
// decision JSON stays unparseable after the built-in repair retry, Route
// returns an error matching ErrRoutingParse, so callers can detect the
// exhausted repair cycle and degrade gracefully (e.g. fall back to a default
// routing decision) instead of failing the whole task.
func TestRoute_ParseErrorAfterRepairIsSentinel(t *testing.T) {
	mock := &mockLLMCaller{
		responses: []*llm.ChatResponse{{
			Message: llm.Message{Role: "assistant", Content: "not json <<<"},
		}},
	}

	r := newTestRouter(mock, 5)

	_, err := r.Route(context.Background(), "do things", nil, nil, nil)
	if err == nil {
		t.Fatal("expected error for unparseable routing JSON")
	}
	if !errors.Is(err, ErrRoutingParse) {
		t.Errorf("expected ErrRoutingParse in error chain, got: %v", err)
	}
	if mock.callIdx != 2 {
		t.Errorf("expected initial call + one repair retry (2 LLM calls), got %d", mock.callIdx)
	}
}

func TestRoute_PassesHistory(t *testing.T) {
	mock := &mockLLMCaller{
		responses: []*llm.ChatResponse{{
			Message: llm.Message{
				Role:    "assistant",
				Content: `{"mode":"react","domain":"code","complexity":2}`,
			},
		}},
	}

	r := newTestRouter(mock, 3)

	history := []llm.Message{
		{Role: "user", Content: "previous message 1"},
		{Role: "assistant", Content: "previous response 1"},
		{Role: "user", Content: "previous message 2"},
		{Role: "assistant", Content: "previous response 2"},
	}

	_, err := r.Route(context.Background(), "current request", nil, history, nil)
	if err != nil {
		t.Fatalf("Route returned error: %v", err)
	}

	// With historyWindow=3, should include last 3 messages from history
	// Messages should be: system + last 3 history + user request
	if len(mock.lastCall().Messages) != 5 {
		t.Fatalf("expected 5 messages, got %d", len(mock.lastCall().Messages))
	}

	foundPrevMsg2 := false
	foundPrevResp2 := false
	for _, msg := range mock.lastCall().Messages {
		if strings.Contains(msg.Content, "previous message 2") {
			foundPrevMsg2 = true
		}
		if strings.Contains(msg.Content, "previous response 2") {
			foundPrevResp2 = true
		}
	}
	if !foundPrevMsg2 {
		t.Error("history should contain 'previous message 2'")
	}
	if !foundPrevResp2 {
		t.Error("history should contain 'previous response 2'")
	}
}

func TestRoute_PlanExecuteMode(t *testing.T) {
	mock := &mockLLMCaller{
		responses: []*llm.ChatResponse{{
			Message: llm.Message{
				Role:    "assistant",
				Content: `{"domain":"mixed","complexity":5,"needs_clarification":false}`,
			},
		}},
	}

	r := newTestRouter(mock, 5)

	decision, err := r.Route(context.Background(), "refactor the entire codebase", nil, nil, nil)
	if err != nil {
		t.Fatalf("Route returned error: %v", err)
	}

	if decision.Complexity != 5 {
		t.Errorf("expected complexity 5, got %d", decision.Complexity)
	}
}

func TestRoute_HandlesJSONInCodeBlocks(t *testing.T) {
	mock := &mockLLMCaller{
		responses: []*llm.ChatResponse{{
			Message: llm.Message{
				Role: "assistant",
				Content: "```json\n" +
					`{"mode":"direct","domain":"general","complexity":1,"compaction_strategy":"sliding_window","suggested_tools":[],"needs_clarification":false}` +
					"\n```",
			},
		}},
	}

	r := newTestRouter(mock, 5)

	decision, err := r.Route(context.Background(), "what is 2+2?", nil, nil, nil)
	if err != nil {
		t.Fatalf("Route returned error: %v", err)
	}

	if decision.Domain != "general" {
		t.Errorf("expected domain 'general', got '%s'", decision.Domain)
	}
	if decision.Complexity != 1 {
		t.Errorf("expected complexity 1, got %d", decision.Complexity)
	}
}

func TestRoute_UsesRouterRole(t *testing.T) {
	mock := &mockLLMCaller{
		callFn: func(ctx context.Context, req llm.ChatRequest) (*llm.ChatResponse, error) {
			return &llm.ChatResponse{
				Message: llm.Message{Role: "assistant", Content: `{"domain":"code","complexity":2}`},
			}, nil
		},
	}

	r := newTestRouter(mock, 5)
	_, _ = r.Route(context.Background(), "test request", nil, nil, nil)
}

func TestNew_DefaultHistoryWindow(t *testing.T) {
	mock := &mockLLMCaller{}

	// Zero history window should default to 10
	r := New(mock, Config{HistoryWindow: 0})
	if r.historyWindow != 10 {
		t.Errorf("expected historyWindow=10 for 0 input, got %d", r.historyWindow)
	}

	// Negative history window should default to 10
	r = New(mock, Config{HistoryWindow: -5})
	if r.historyWindow != 10 {
		t.Errorf("expected historyWindow=10 for -5 input, got %d", r.historyWindow)
	}

	// Positive should be used as-is
	r = New(mock, Config{HistoryWindow: 20})
	if r.historyWindow != 20 {
		t.Errorf("expected historyWindow=20, got %d", r.historyWindow)
	}
}

func TestValidateRoutingDecision(t *testing.T) {
	tests := []struct {
		name        string
		input       RoutingDecision
		wantDomain  string
		wantComplex int
	}{
		{"valid decision unchanged", RoutingDecision{Domain: "code", Complexity: 3}, "code", 3},
		{"unknown domain defaults to general", RoutingDecision{Domain: "unknown", Complexity: 2}, "general", 2},
		{"empty domain defaults to general", RoutingDecision{Domain: "", Complexity: 2}, "general", 2},
		{"complexity clamped to min 1", RoutingDecision{Domain: "code", Complexity: 0}, "code", 1},
		{"complexity clamped to max 5", RoutingDecision{Domain: "code", Complexity: 10}, "code", 5},
		{"negative complexity clamped", RoutingDecision{Domain: "code", Complexity: -1}, "code", 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			d := tt.input
			validateRoutingDecision(&d)
			if d.Domain != tt.wantDomain || d.Complexity != tt.wantComplex {
				t.Errorf("got domain=%q complexity=%d, want domain=%q complexity=%d", d.Domain, d.Complexity, tt.wantDomain, tt.wantComplex)
			}
		})
	}
}

func TestValidateRoutingDecision_MatchedSkills(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		input      []string
		wantSkills []string
	}{
		{"nil skills unchanged", nil, nil},
		{"empty skills unchanged", []string{}, []string{}},
		{"valid skills preserved", []string{"pdf-processing", "data-analysis"}, []string{"pdf-processing", "data-analysis"}},
		{"duplicate skills deduped", []string{"pdf", "data", "pdf"}, []string{"pdf", "data"}},
		{"empty strings removed", []string{"pdf", "", "data"}, []string{"pdf", "data"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			d := RoutingDecision{Domain: "code", Complexity: 3, MatchedSkills: tt.input}
			validateRoutingDecision(&d)
			if len(d.MatchedSkills) != len(tt.wantSkills) {
				t.Fatalf("got %d skills %v, want %d skills %v", len(d.MatchedSkills), d.MatchedSkills, len(tt.wantSkills), tt.wantSkills)
			}
			for i, got := range d.MatchedSkills {
				if got != tt.wantSkills[i] {
					t.Errorf("skill[%d] = %q, want %q", i, got, tt.wantSkills[i])
				}
			}
		})
	}
}

// TestValidateRoutingDecision_MatchedTools verifies that validateRoutingDecision
// deduplicates and trims MatchedTools exactly like MatchedSkills, and is nil-safe.
func TestValidateRoutingDecision_MatchedTools(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		input     []string
		wantTools []string
	}{
		{"nil tools unchanged", nil, nil},
		{"empty tools unchanged", []string{}, []string{}},
		{"valid tools preserved", []string{"bash_exec", "read_file"}, []string{"bash_exec", "read_file"}},
		{"duplicate tools deduped", []string{"bash_exec", "read_file", "bash_exec"}, []string{"bash_exec", "read_file"}},
		{"empty strings removed", []string{"bash_exec", "", "read_file"}, []string{"bash_exec", "read_file"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			d := RoutingDecision{Domain: "code", Complexity: 3, MatchedTools: tt.input}
			validateRoutingDecision(&d)
			if len(d.MatchedTools) != len(tt.wantTools) {
				t.Fatalf("got %d tools %v, want %d tools %v", len(d.MatchedTools), d.MatchedTools, len(tt.wantTools), tt.wantTools)
			}
			for i, got := range d.MatchedTools {
				if got != tt.wantTools[i] {
					t.Errorf("tool[%d] = %q, want %q", i, got, tt.wantTools[i])
				}
			}
		})
	}
}

func TestRoute_RetriesOnInvalidJSON(t *testing.T) {
	callCount := 0
	mock := &mockLLMCaller{
		callFn: func(ctx context.Context, req llm.ChatRequest) (*llm.ChatResponse, error) {
			callCount++
			if callCount == 1 {
				return &llm.ChatResponse{
					Message: llm.Message{Role: "assistant", Content: "I think this is a code task"},
				}, nil
			}
			return &llm.ChatResponse{
				Message: llm.Message{Role: "assistant", Content: `{"domain":"code","complexity":2,"needs_clarification":false}`},
			}, nil
		},
	}
	r := newTestRouter(mock, 5)
	decision, err := r.Route(context.Background(), "fix the bug", nil, nil, nil)
	if err != nil {
		t.Fatalf("expected successful retry, got error: %v", err)
	}
	if decision.Domain != "code" {
		t.Errorf("expected domain 'code', got '%s'", decision.Domain)
	}
	if callCount != 2 {
		t.Errorf("expected 2 LLM calls (original + retry), got %d", callCount)
	}
}

func TestRoute_SetsReasoningEffort(t *testing.T) {
	mock := &mockLLMCaller{
		callFn: func(ctx context.Context, req llm.ChatRequest) (*llm.ChatResponse, error) {
			return &llm.ChatResponse{
				Message: llm.Message{Role: "assistant", Content: `{"domain":"code","complexity":2}`},
			}, nil
		},
	}

	r := newTestRouter(mock, 5)
	r.SetReasoningEffort("high")

	_, err := r.Route(context.Background(), "test", nil, nil, nil)
	if err != nil {
		t.Fatalf("Route returned error: %v", err)
	}

	got := mock.lastCall().ReasoningEffort
	if got != "high" {
		t.Errorf("expected ReasoningEffort=%q, got %q", "high", got)
	}
}

func TestRoute_NoReasoningEffortWhenEmpty(t *testing.T) {
	mock := &mockLLMCaller{
		callFn: func(ctx context.Context, req llm.ChatRequest) (*llm.ChatResponse, error) {
			return &llm.ChatResponse{
				Message: llm.Message{Role: "assistant", Content: `{"domain":"code","complexity":2}`},
			}, nil
		},
	}

	r := newTestRouter(mock, 5)

	_, err := r.Route(context.Background(), "test", nil, nil, nil)
	if err != nil {
		t.Fatalf("Route returned error: %v", err)
	}

	got := mock.lastCall().ReasoningEffort
	if got != "" {
		t.Errorf("expected empty ReasoningEffort, got %q", got)
	}
}

// TestRoute_ToolMatchingInjectsSection verifies that SetToolMatching(true)
// causes the Route system prompt to include the tool-selection instruction
// section and the matched_tools field in the JSON output schema, and that the
// default (disabled) prompt omits both.
func TestRoute_ToolMatchingInjectsSection(t *testing.T) {
	// Template includes the TOOL-MATCHING and JSON-OUTPUT-SCHEMA placeholders
	// so their (conditionally-resolved) content lands in the rendered prompt.
	const tmpl = "Tools: {{AVAILABLE-TOOLS}}\nSkills: {{AVAILABLE-SKILLS}}\n{{TOOL-MATCHING}}\n{{JSON-OUTPUT-SCHEMA}}"

	newMock := func() *mockLLMCaller {
		return &mockLLMCaller{
			callFn: func(_ context.Context, _ llm.ChatRequest) (*llm.ChatResponse, error) {
				return &llm.ChatResponse{Message: llm.Message{Role: "assistant", Content: `{"domain":"code","complexity":2}`}}, nil
			},
		}
	}

	t.Run("enabled injects tool selection instructions and matched_tools schema", func(t *testing.T) {
		m := newMock()
		r := New(m, Config{SystemPrompt: tmpl, HistoryWindow: 5})
		r.SetToolMatching(true)

		if _, err := r.Route(context.Background(), "test", nil, nil, nil); err != nil {
			t.Fatalf("Route returned error: %v", err)
		}

		sys := m.lastCall().Messages[0].Content
		if !strings.Contains(sys, "Tool Selection") {
			t.Error("system prompt should contain the tool-selection instruction when tool matching is enabled")
		}
		if !strings.Contains(sys, "matched_tools") {
			t.Error("system prompt should include matched_tools in the JSON schema when tool matching is enabled")
		}
	})

	t.Run("disabled omits tool selection instructions and matched_tools", func(t *testing.T) {
		m := newMock()
		r := New(m, Config{SystemPrompt: tmpl, HistoryWindow: 5})
		// SetToolMatching deliberately NOT called → defaults to false.

		if _, err := r.Route(context.Background(), "test", nil, nil, nil); err != nil {
			t.Fatalf("Route returned error: %v", err)
		}

		sys := m.lastCall().Messages[0].Content
		if strings.Contains(sys, "Tool Selection") {
			t.Error("system prompt should NOT contain the tool-selection instruction when tool matching is disabled")
		}
		if strings.Contains(sys, "matched_tools") {
			t.Error("system prompt should NOT contain matched_tools when tool matching is disabled")
		}
	})
}

func TestRoute_AppendContextSections(t *testing.T) {
	mock := &mockLLMCaller{
		callFn: func(ctx context.Context, req llm.ChatRequest) (*llm.ChatResponse, error) {
			return &llm.ChatResponse{
				Message: llm.Message{Role: "assistant", Content: `{"domain":"code","complexity":3,"matched_skills":["go-lint"]}`},
			}, nil
		},
	}

	appendCalled := false
	r := New(mock, Config{
		SystemPrompt:  "Tools: {{AVAILABLE-TOOLS}}\nSkills: {{AVAILABLE-SKILLS}}",
		HistoryWindow: 5,
		AppendContextSections: func(ctx context.Context) string {
			appendCalled = true
			return "\n\n## Project Context\nTech stack: Go 1.26, React 19."
		},
	})

	_, err := r.Route(context.Background(), "fix the bug", nil, nil, nil)
	if err != nil {
		t.Fatalf("Route returned error: %v", err)
	}

	if !appendCalled {
		t.Error("AppendContextSections was not called")
	}

	systemMsg := mock.lastCall().Messages[0]
	if !strings.Contains(systemMsg.Content, "## Project Context") {
		t.Error("system prompt should contain appended context section")
	}
	if !strings.Contains(systemMsg.Content, "Tech stack: Go 1.26, React 19.") {
		t.Error("system prompt should contain appended context content")
	}
}

func TestRoute_AppendContextSections_Nil(t *testing.T) {
	// Verify that nil AppendContextSections is a no-op.
	mock := &mockLLMCaller{
		callFn: func(ctx context.Context, req llm.ChatRequest) (*llm.ChatResponse, error) {
			return &llm.ChatResponse{
				Message: llm.Message{Role: "assistant", Content: `{"domain":"code","complexity":2}`},
			}, nil
		},
	}

	r := newTestRouter(mock, 5) // newTestRouter does NOT set AppendContextSections

	_, err := r.Route(context.Background(), "test", nil, nil, nil)
	if err != nil {
		t.Fatalf("Route returned error: %v", err)
	}

	systemMsg := mock.lastCall().Messages[0]
	if strings.Contains(systemMsg.Content, "AGENTS.md") {
		t.Error("system prompt should NOT contain AGENTS.md when AppendContextSections is nil")
	}
}

func TestRoute_NilResponse(t *testing.T) {
	mock := &mockLLMCaller{
		callFn: func(ctx context.Context, req llm.ChatRequest) (*llm.ChatResponse, error) {
			return nil, nil
		},
	}
	r := newTestRouter(mock, 5)
	_, err := r.Route(context.Background(), "test", nil, nil, nil)
	if err == nil {
		t.Fatal("expected error for nil response")
	}
	if !strings.Contains(err.Error(), "nil response") {
		t.Errorf("expected 'nil response' in error, got: %v", err)
	}
}
