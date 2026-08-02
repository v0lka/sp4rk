// Package router classifies user requests by domain and complexity to drive
// execution strategy selection (direct execution vs. plan-and-execute).
package router

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/v0lka/sp4rk/agent"
	"github.com/v0lka/sp4rk/llm"
	"github.com/v0lka/sp4rk/prompt"
	"github.com/v0lka/sp4rk/tools"
)

// toolMatchingInstruction is the prompt section injected via the TOOL-MATCHING
// placeholder when semantic tool selection is enabled (Router.toolMatching). It
// directs the router LLM to pick the relevant tools from the available set and
// return them in the matched_tools field of its JSON response.
const toolMatchingInstruction = `## Tool Selection

From the available tools listed above, select the ones most relevant to this request and return their exact names in the "matched_tools" array. Only include tools that are clearly needed; omit speculative or unrelated tools. Return an empty array when no tools apply.`

// routingJSONSchemaDefault is the JSON output-format directive injected via the
// JSON-OUTPUT-SCHEMA placeholder when tool matching is disabled. It mirrors the
// RoutingDecision struct minus matched_tools.
const routingJSONSchemaDefault = `Respond with ONLY a JSON object in this exact format:
{"domain":"code","complexity":3,"needs_clarification":false,"matched_skills":[]}`

// routingJSONSchemaWithTools is the JSON output-format directive injected via the
// JSON-OUTPUT-SCHEMA placeholder when tool matching is enabled. It extends the
// default schema with the matched_tools field.
const routingJSONSchemaWithTools = `Respond with ONLY a JSON object in this exact format:
{"domain":"code","complexity":3,"needs_clarification":false,"matched_skills":[],"matched_tools":[]}`

// Config holds the configuration for a Router.
type Config struct {
	// SystemPrompt is the routing system prompt template.
	// It must contain AVAILABLE-TOOLS and AVAILABLE-SKILLS placeholders.
	SystemPrompt string
	// HistoryWindow is the number of recent messages to include. Default: 10.
	HistoryWindow int
	// AppendContextSections, when non-nil, is called with the request context
	// to produce additional prompt sections inserted via the PROJECT-CONTEXT
	// placeholder in the system prompt template. This is the hook for injecting
	// AGENTS.md content so the router can consider project conventions when
	// matching skills.
	AppendContextSections func(ctx context.Context) string
	// ToolMatching enables semantic tool selection: the router is instructed to
	// pick relevant tools and return them in RoutingDecision.MatchedTools.
	// Equivalent to calling Router.SetToolMatching(true). Defaults to false.
	ToolMatching bool
}

// Router classifies user requests by domain and complexity.
type Router struct {
	llm                   agent.LLMCaller
	systemPrompt          string
	historyWindow         int
	modelRegistry         *llm.ModelRegistry
	reasoningEffort       string
	toolMatching          bool
	appendContextSections func(ctx context.Context) string
}

// New creates a new Router with the given caller and config.
func New(caller agent.LLMCaller, cfg Config) *Router {
	hw := cfg.HistoryWindow
	if hw <= 0 {
		hw = 10
	}
	return &Router{
		llm:                   caller,
		systemPrompt:          cfg.SystemPrompt,
		historyWindow:         hw,
		toolMatching:          cfg.ToolMatching,
		appendContextSections: cfg.AppendContextSections,
	}
}

// SetModelRegistry sets the model registry for model metadata resolution.
func (r *Router) SetModelRegistry(registry *llm.ModelRegistry) {
	r.modelRegistry = registry
}

// SetReasoningEffort sets the reasoning effort for the router.
func (r *Router) SetReasoningEffort(effort string) {
	r.reasoningEffort = effort
}

// SetToolMatching enables or disables semantic tool selection. When enabled,
// the Route prompt includes a tool-selection instruction and the matched_tools
// field in the JSON output schema; the router populates
// RoutingDecision.MatchedTools. Defaults to false (disabled).
func (r *Router) SetToolMatching(enabled bool) {
	r.toolMatching = enabled
}

// Route analyzes the user's request and determines the best execution strategy.
func (r *Router) Route(ctx context.Context, userMessage string, availableTools []tools.ToolDescriptor, history []llm.Message, availableSkills []SkillDescriptor) (decision *RoutingDecision, err error) {
	// Build tool list for the prompt (grouped by priority tier)
	toolListStr := agent.BuildGroupedToolList(availableTools)

	// Build skill list for the prompt
	skillListStr := formatSkillList(availableSkills)

	// Build context sections from the request (AGENTS.md, etc.).
	var projectContext string
	if r.appendContextSections != nil {
		projectContext = r.appendContextSections(ctx)
	}

	// Build system prompt.
	// If the template uses PROJECT-CONTEXT, insert the context there
	// (before the JSON-output directive to avoid recency bias).
	// Otherwise fall back to appending at the end for backward compatibility.
	// Tool/skill lists and project context (AGENTS.md etc.) are dynamic,
	// externally-influenced values — substituted via ReplaceData so
	// placeholder names inside them are never expanded.
	//
	// TOOL-MATCHING and JSON-OUTPUT-SCHEMA are trusted template constants
	// (defined in-package), so they use Replace (iterative substitution).
	// When tool matching is off, TOOL-MATCHING resolves to empty and
	// JSON-OUTPUT-SCHEMA resolves to the default schema, leaving behavior
	// identical to before this feature existed.
	toolMatchingSection := ""
	jsonSchema := routingJSONSchemaDefault
	if r.toolMatching {
		toolMatchingSection = toolMatchingInstruction
		jsonSchema = routingJSONSchemaWithTools
	}
	builder := prompt.NewBuilder().
		Core(r.systemPrompt).
		Replace("TOOL-MATCHING", toolMatchingSection).
		Replace("JSON-OUTPUT-SCHEMA", jsonSchema).
		ReplaceData("AVAILABLE-TOOLS", toolListStr).
		ReplaceData("AVAILABLE-SKILLS", skillListStr)
	if strings.Contains(r.systemPrompt, "PROJECT-CONTEXT") {
		builder = builder.ReplaceData("PROJECT-CONTEXT", projectContext)
	}
	systemPrompt := builder.Build()
	if projectContext != "" && !strings.Contains(r.systemPrompt, "PROJECT-CONTEXT") {
		systemPrompt += projectContext
	}

	// Build messages for the request
	messages := make([]llm.Message, 0, len(history)+2)
	messages = append(messages, llm.Message{Role: "system", Content: systemPrompt})

	// Add recent history messages (up to historyWindow)
	historyStart := 0
	if len(history) > r.historyWindow {
		historyStart = len(history) - r.historyWindow
	}
	messages = append(messages, history[historyStart:]...)

	// Add user message with the request to classify
	messages = append(messages, llm.Message{
		Role:    "user",
		Content: "Classify this request: " + userMessage,
	})

	// Create chat request
	req := llm.ChatRequest{
		Messages:        messages,
		ReasoningEffort: r.reasoningEffort,
	}

	// Call LLM
	resp, err := r.llm.Call(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("router LLM call failed: %w", err)
	}
	if resp == nil {
		return nil, errors.New("router LLM call returned nil response")
	}

	// Extract JSON from response (handle markdown code blocks)
	jsonStr := llm.ExtractJSON(resp.Message.Content)

	// Unmarshal into RoutingDecision
	var routingDecision RoutingDecision
	if err := json.Unmarshal([]byte(jsonStr), &routingDecision); err != nil {
		// Retry with repair prompt
		repairMessages := make([]llm.Message, len(messages)+2)
		copy(repairMessages, messages)
		repairMessages[len(messages)] = llm.Message{Role: "assistant", Content: resp.Message.Content, ReasoningContent: resp.Message.ReasoningContent}
		repairMessages[len(messages)+1] = llm.Message{
			Role: "user",
			// The repair prompt echoes the schema the router was asked to produce,
			// including matched_tools when tool matching is enabled.
			Content: "Your previous response was not valid JSON. " + jsonSchema,
		}

		retryResp, retryErr := r.llm.Call(ctx, llm.ChatRequest{Messages: repairMessages, ReasoningEffort: r.reasoningEffort})
		if retryErr != nil {
			return nil, fmt.Errorf("router retry LLM call failed: %w", retryErr)
		}
		if retryResp == nil {
			return nil, errors.New("router LLM retry call returned nil response")
		}

		retryJSON := llm.ExtractJSON(retryResp.Message.Content)
		if retryErr := json.Unmarshal([]byte(retryJSON), &routingDecision); retryErr != nil {
			return nil, fmt.Errorf("failed to parse routing decision after retry: %w", retryErr)
		}
	}

	validateRoutingDecision(&routingDecision)

	return &routingDecision, nil
}

// validateRoutingDecision sanitizes and corrects a routing decision from LLM output.
func validateRoutingDecision(d *RoutingDecision) {
	// Validate domain
	switch d.Domain {
	case DomainCode, DomainResearch, DomainGeneral, DomainMixed:
		// valid
	default:
		d.Domain = DomainGeneral
	}

	// Clamp complexity to [1, 5]
	if d.Complexity < 1 {
		d.Complexity = 1
	}
	if d.Complexity > 5 {
		d.Complexity = 5
	}

	// Deduplicate, trim, and order-preserve matched skills/tools.
	d.MatchedSkills = dedupeAndTrimStrings(d.MatchedSkills)
	d.MatchedTools = dedupeAndTrimStrings(d.MatchedTools)
}

// dedupeAndTrimStrings removes empty strings and collapses duplicates from
// values, preserving first-occurrence order. It reuses the input's backing
// array in place. A nil or empty input is returned unchanged so callers can
// distinguish "not set" from an explicitly empty set.
func dedupeAndTrimStrings(values []string) []string {
	if len(values) == 0 {
		return values
	}
	seen := make(map[string]bool, len(values))
	clean := values[:0]
	for _, v := range values {
		if v != "" && !seen[v] {
			seen[v] = true
			clean = append(clean, v)
		}
	}
	return clean
}

// formatSkillList formats available skill descriptors for the router prompt.
// Returns "None" if no skills are available.
func formatSkillList(availableSkills []SkillDescriptor) string {
	if len(availableSkills) == 0 {
		return "None"
	}
	var sb strings.Builder
	for i, s := range availableSkills {
		if i > 0 {
			sb.WriteString("\n")
		}
		sb.WriteString("- " + s.Name + ": " + s.Description)
	}
	return sb.String()
}
