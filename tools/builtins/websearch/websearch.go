package websearch

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"github.com/v0lka/sp4rk/tools"
	"github.com/v0lka/sp4rk/tools/builtins"
)

// maxResponseBodyBytes caps how much of a provider HTTP response body is read,
// protecting against unbounded memory use from a hostile or broken endpoint.
const maxResponseBodyBytes = 4 << 20 // 4 MB

// limitBody wraps a response body reader with a hard size cap.
func limitBody(r io.Reader) io.Reader {
	return io.LimitReader(r, maxResponseBodyBytes)
}

const toolWebsearchDescription = `Purpose: search the web and return up to max_results entries, each with a title, URL and a snippet summarizing the page.
Use when: you need information beyond your training data — current docs, recent releases, library versions, external error reports. Pick a promising URL from the results and read the full page with web_fetch.
Inputs: query (be specific — keywords beat prose); optional max_results (default 5).
Outputs: a list of {title, URL, snippet} results.
Example: query "modernc.org/sqlite pure-Go driver requirements", then web_fetch the top hit.
Anti-example: not for reading a URL you already have (web_fetch directly); avoid vague one-word queries — they return noise; never paste secrets into a search query.`

// SearchResult represents a single provider-agnostic search result.
type SearchResult struct {
	Title   string
	URL     string
	Snippet string
}

// SearchProvider defines the interface for web search providers.
// Built-in implementations include BraveProvider, DuckDuckGoProvider,
// ExaProvider, and TavilyProvider. To add a custom provider, implement this
// interface and pass it to NewTool.
type SearchProvider interface {
	Search(ctx context.Context, query string, maxResults int) ([]SearchResult, error)
	Name() string
}

// Limits holds the per-tool web search limits configuration.
type Limits = builtins.WebSearchLimits

// --- Tool ---

// Tool searches the web using a pluggable SearchProvider.
type Tool struct {
	*tools.BaseTool
	provider SearchProvider
	limits   Limits
}

// NewTool creates a new Tool with the given SearchProvider and specified limits.
func NewTool(provider SearchProvider, limits Limits) *Tool {
	schema := `{
		"type": "object",
		"properties": {
			"query": {
				"type": "string",
				"description": "The search query string. Be specific and use keywords for best results."
			},
			"max_results": {
				"type": "integer",
				"description": "Maximum number of results to return. Default: 5."
			}
		},
		"required": ["query"]
	}`
	return &Tool{
		BaseTool: &tools.BaseTool{
			ToolName:        "web_search",
			ToolGroup:       tools.GroupRemoteRead,
			ToolDescription: toolWebsearchDescription,
			Schema:          json.RawMessage(schema),
			Policy:          tools.PolicyAlwaysAllow,
			Untrusted:       true,
		},
		provider: provider,
		limits:   limits,
	}
}

// Execute performs the web search by parsing the input and delegating to
// the configured provider. The query parameter is required; max_results
// defaults to the configured limit when omitted or non-positive.
func (t *Tool) Execute(ctx context.Context, input json.RawMessage) (tools.ToolResult, error) {
	// Unmarshal once into a raw map so fallback extraction does not re-parse.
	var raw map[string]any
	if err := json.Unmarshal(input, &raw); err != nil {
		return tools.ParseInputError(err)
	}

	query, _ := raw["query"].(string)
	if query == "" {
		query = extractQueryFallback(raw)
	}

	// Validate query parameter
	if query == "" {
		return tools.ToolResult{Content: "query parameter is required", IsError: true}, nil
	}

	// Set default max_results if not provided
	maxResults := intFromRaw(raw, "max_results")
	if maxResults <= 0 {
		maxResults = t.limits.MaxResults
	}

	// Perform the search
	results, err := t.provider.Search(ctx, query, maxResults)
	if err != nil {
		return tools.ToolResult{Content: fmt.Sprintf("search failed: %v", err), IsError: true}, nil
	}

	// Check for empty results
	if len(results) == 0 {
		return tools.ToolResult{Content: "No results found", IsError: false}, nil
	}

	// Format results
	output := formatResults(results)
	return tools.ToolResult{Content: output, IsError: false}, nil
}

// intFromRaw extracts an int value from a raw map, handling float64 coercion
// (all JSON numbers unmarshal to float64 in map[string]any).
func intFromRaw(raw map[string]any, key string) int {
	val, ok := raw[key]
	if !ok {
		return 0
	}
	switch v := val.(type) {
	case float64:
		return int(v)
	case int:
		return v
	case int64:
		return int(v)
	}
	return 0
}

// formatResults formats the search results as a readable string.
func formatResults(results []SearchResult) string {
	var output strings.Builder
	for i, result := range results {
		if i > 0 {
			output.WriteString("\n\n")
		}
		fmt.Fprintf(&output, "%d. **%s**\n   URL: %s\n   Snippet: %s", i+1, result.Title, result.URL, result.Snippet)
	}
	return output.String()
}

// extractQueryFallback attempts to extract a query string from common
// parameter variations that models may produce (e.g. "queries", "search_query").
// raw is the already-unmarshalled input map to avoid re-parsing the JSON.
func extractQueryFallback(raw map[string]any) string {
	for _, key := range []string{"queries", "search_query"} {
		val, ok := raw[key]
		if !ok {
			continue
		}
		switch v := val.(type) {
		case string:
			if v != "" {
				return v
			}
		case []any:
			for _, elem := range v {
				if s, ok := elem.(string); ok && s != "" {
					return s
				}
			}
		}
	}
	return ""
}
