package builtins

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	"github.com/v0lka/sp4rk/agent"
	"github.com/v0lka/sp4rk/tools"
)

const toolStoreFactDescription = `Purpose: store a durable fact for later retrieval by yourself or other agents.
Use when: immediately after learning cross-step information — API signatures, architectural decisions, file locations, error patterns, intermediate results — before context grows large and earlier tool outputs become unavailable. Retrieve later with search_facts; facts persist across steps and execution cycles.
Inputs: content (the fact itself, self-contained); keywords (3-5 retrieval keywords).
Outputs: confirmation of storage.
Example: content "auth middleware lives in core/middleware.go, enforced via applySecurityPolicies" with keywords [auth, middleware, policy, security].
Anti-example: not for ephemeral scratch that dies with the current turn; do not defer storing to the end of a long investigation — store early, store often.`

const toolSearchFactsDescription = `Purpose: search previously stored facts by keywords.
Use when: at the start of a new step or subtask — recover prior decisions and discoveries before re-reading sources. Results rank by relevance (most keyword matches first).
Inputs: keywords (1-5 search terms).
Outputs: the matching stored facts, best match first; empty when nothing matches.
Example: keywords [auth, config] to recall where auth settings live.
Anti-example: not for discovering new information (search the codebase: ripgrep/semantic_search); if nothing matches, do not reconstruct from memory — re-read the source.`

// ---------------------------------------------------------------------------
// store_fact
// ---------------------------------------------------------------------------

// StoreFactTool stores a keyword-tagged fact in the shared fact memory.
type StoreFactTool struct {
	*tools.BaseTool
}

// NewStoreFactTool creates a new StoreFactTool instance.
func NewStoreFactTool() *StoreFactTool {
	return &StoreFactTool{BaseTool: &tools.BaseTool{
		ToolName:        "store_fact",
		ToolGroup:       tools.GroupSystem,
		ToolDescription: toolStoreFactDescription,
		Schema: json.RawMessage(`{
		"type": "object",
		"properties": {
			"keywords": {
				"type": "array",
				"items": {"type": "string"},
				"minItems": 3,
				"maxItems": 5,
				"description": "3-5 keywords for retrieval"
			},
			"content": {
				"type": "string",
				"description": "The fact to store"
			}
		},
		"required": ["keywords", "content"]
	}`),
		Policy: tools.PolicyAlwaysAllow,
	}}
}

// StoreFactInput represents the input parameters for store_fact.
type StoreFactInput struct {
	Keywords []string `json:"keywords"`
	Content  string   `json:"content"`
}

// Execute stores a fact via the FactStore from context.
func (t *StoreFactTool) Execute(ctx context.Context, input json.RawMessage) (tools.ToolResult, error) {
	var params StoreFactInput
	if err := json.Unmarshal(input, &params); err != nil {
		return tools.ParseInputError(err)
	}

	if len(params.Keywords) < 3 {
		return tools.ToolResult{Content: fmt.Sprintf("validation error: keywords must have at least 3 items, got %d", len(params.Keywords)), IsError: true}, nil
	}
	if len(params.Keywords) > 5 {
		return tools.ToolResult{Content: fmt.Sprintf("validation error: keywords must have at most 5 items, got %d", len(params.Keywords)), IsError: true}, nil
	}
	if strings.TrimSpace(params.Content) == "" {
		return tools.ToolResult{Content: "validation error: content is required", IsError: true}, nil
	}

	// ASI06-R8/R10: refuse to persist content that looks like a secret/credential.
	// The fact store persists across sessions; a leaked API key there is a
	// durable secret-disclosure. This is a heuristic lint, not a guarantee —
	// it catches the common "ANTHROPIC_API_KEY=sk-..." / "token=..." shapes.
	// Both content and keywords are scanned: a careless agent could stash a
	// credential in a keyword tag, bypassing a content-only check.
	if hit := firstSecretHit(params.Content, params.Keywords); hit != "" {
		return tools.ToolResult{
			Content: "refused: input appears to contain a secret (" + hit + "). " +
				"Secrets must not be stored in agent memory. Store only the non-secret fact.",
			IsError: true,
		}, nil
	}

	fs := agent.FactStoreFromContext(ctx)
	if fs == nil {
		return tools.ErrorResult("Fact store not available"), nil
	}

	author := agent.StepIDFromContext(ctx)
	fs.StoreFact(params.Keywords, params.Content, author)

	return tools.ToolResult{Content: "Fact stored with keywords: " + strings.Join(params.Keywords, ", ")}, nil
}

// ---------------------------------------------------------------------------
// search_facts
// ---------------------------------------------------------------------------

// SearchFactsTool searches stored facts by keywords.
type SearchFactsTool struct {
	*tools.BaseTool
}

// NewSearchFactsTool creates a new SearchFactsTool instance.
func NewSearchFactsTool() *SearchFactsTool {
	return &SearchFactsTool{BaseTool: &tools.BaseTool{
		ToolName:        "search_facts",
		ToolGroup:       tools.GroupSystem,
		ToolDescription: toolSearchFactsDescription,
		Schema: json.RawMessage(`{
		"type": "object",
		"properties": {
			"keywords": {
				"type": "array",
				"items": {"type": "string"},
				"minItems": 1,
				"maxItems": 5,
				"description": "Keywords to search for"
			}
		},
		"required": ["keywords"]
	}`),
		Policy: tools.PolicyAlwaysAllow,
	}}
}

// SearchFactsInput represents the input parameters for search_facts.
type SearchFactsInput struct {
	Keywords []string `json:"keywords"`
}

// Execute searches facts via the FactStore from context.
func (t *SearchFactsTool) Execute(ctx context.Context, input json.RawMessage) (tools.ToolResult, error) {
	var params SearchFactsInput
	if err := json.Unmarshal(input, &params); err != nil {
		return tools.ParseInputError(err)
	}

	if len(params.Keywords) == 0 {
		return tools.ToolResult{Content: "validation error: keywords must have at least 1 item", IsError: true}, nil
	}

	fs := agent.FactStoreFromContext(ctx)
	if fs == nil {
		return tools.ErrorResult("Fact store not available"), nil
	}

	entries := fs.SearchFacts(params.Keywords)
	if len(entries) == 0 {
		return tools.ToolResult{Content: "No facts found matching the given keywords"}, nil
	}

	var b strings.Builder
	for i, e := range entries {
		fmt.Fprintf(&b, "%d. [%s] (by %s)\n   %s\n", i+1, strings.Join(e.Keywords, ", "), e.Author, e.Content)
	}

	return tools.ToolResult{Content: b.String()}, nil
}

// secretPatterns are heuristics that catch common credential shapes a careless
// agent might try to persist to long-term memory. They are intentionally broad
// (favor recall) since refusing a borderline fact is a recoverable false
// positive, while persisting a real secret is a durable disclosure (ASI06).
//
// The assignment patterns deliberately avoid the bare "keyword: value" prose
// form: a colon-separated lowercase keyword in a sentence ("the token:
// abcd1234", "see auth: ed25519...") is not a secret. Real assignments use
// either "=" (env/shell) or a quoted value (JSON/YAML config); the quoted-value
// requirement is the discriminator that separates structured secrets from
// prose without sacrificing recall for "OPENAI_API_KEY=sk-..." prefixes.
var secretPatterns = []*regexp.Regexp{
	// "=" assignments: KEY=value (env, shell, .env files). Matches prefixed
	// identifiers like OPENAI_API_KEY or ANTHROPIC_API_KEY.
	regexp.MustCompile(`(?i)(?:api[_-]?key|password|passwd|secret|token|access[_-]?key|auth)\s*=\s*\S{8,}`),
	// ":" assignments with a quoted value (JSON/YAML/TOML config). The value
	// must be quoted so prose like "the token: abcd1234" never matches.
	regexp.MustCompile(`(?i)["']?(?:api[_-]?key|password|passwd|secret|token|access[_-]?key|auth)["']?\s*:\s*["']\S{8,}["']`),
	// OpenAI-style keys.
	regexp.MustCompile(`sk-[A-Za-z0-9]{20,}`),
	// Anthropic-style keys.
	regexp.MustCompile(`sk-ant-[A-Za-z0-9_-]{20,}`),
	// Generic long hex/base64 "bearer" tokens.
	regexp.MustCompile(`(?i)bearer\s+[A-Za-z0-9._-]{20,}`),
	// AWS access keys.
	regexp.MustCompile(`AKIA[0-9A-Z]{16}`),
	// GitHub PATs.
	regexp.MustCompile(`gh[pousr]_[A-Za-z0-9]{36,}`),
}

// matchesSecretPattern returns a human-readable label of the first secret
// pattern matched by content, or "" if none match.
func matchesSecretPattern(content string) string {
	for _, re := range secretPatterns {
		if re.MatchString(content) {
			return re.String()
		}
	}
	return ""
}

// firstSecretHit returns the label of the first secret pattern matched across
// the content and each keyword entry, or "" if none match. Scanning keywords
// closes the gap where a credential could be stashed in a keyword tag.
func firstSecretHit(content string, keywords []string) string {
	if hit := matchesSecretPattern(content); hit != "" {
		return hit
	}
	for _, kw := range keywords {
		if hit := matchesSecretPattern(kw); hit != "" {
			return hit
		}
	}
	return ""
}
