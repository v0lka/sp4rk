# Web Search (web_search)

## Role

The `web_search` built-in tool and its provider abstraction: a pluggable `SearchProvider` interface with five shipped implementations (Brave, Tavily, Exa API providers; a self-hosted SearXNG JSON client; a best-effort DuckDuckGo HTML scraper) behind one hardened tool that returns normalized `{title, URL, snippet}` results.

## Key Files

- `github.com/v0lka/sp4rk/tools/builtins/websearch/websearch.go` - the `web_search` tool (descriptor, schema, input handling, result formatting), the `SearchProvider` interface, `SearchResult`, and the `Limits` alias
- `github.com/v0lka/sp4rk/tools/builtins/websearch/brave.go` - Brave Search API provider (`X-Subscription-Token` header auth)
- `github.com/v0lka/sp4rk/tools/builtins/websearch/tavily.go` - Tavily API provider (key in the POST body)
- `github.com/v0lka/sp4rk/tools/builtins/websearch/exa.go` - Exa API provider (`x-api-key` header auth, highlights extraction)
- `github.com/v0lka/sp4rk/tools/builtins/websearch/searxng.go` - self-hosted SearXNG JSON API provider
- `github.com/v0lka/sp4rk/tools/builtins/websearch/duckduckgo.go` - DuckDuckGo HTML scraper: request pacer, anomaly retry, HTML parsing, redirect-URL cleaning
- `github.com/v0lka/sp4rk/tools/builtins/websearch/doc.go` - package doc

## Behavior

### The tool

`NewTool(provider SearchProvider, limits Limits) *Tool` builds the tool; `Limits` is an alias for `builtins.WebSearchLimits` (`MaxResults int`, `Timeout time.Duration`; `DefaultWebSearchLimits()` = 5 results / 30 s). A non-positive `limits.MaxResults` is normalized to the default at construction, so the omitted-`max_results` path can only ever deliver a positive count to a provider. The tool consumes `MaxResults` only — `Timeout` is a configuration default the host applies when constructing the provider's HTTP client.

Descriptor facts: name `web_search`; capability group `tools.GroupRemoteRead`; default policy `tools.PolicyAlwaysAllow`; `Untrusted: true`. The schema is an object with `query` (string, required) and `max_results` (integer, default 5). The description follows the six-part rubric (Purpose / Use when / Inputs / Outputs / Example / Anti-example) and stays within the 1200-character guard (`descriptions_guard_test.go`).

`Execute` flow:

```
web_search{query, max_results?}
   │ unmarshal → query (fallback keys: "queries", "search_query" — string or first string element)
   │ max_results omitted/non-positive → limits.MaxResults
   ▼
SearchProvider.Search(ctx, query, maxResults)      one provider per tool
   │  every provider: context-carrying request, redirect-refusing client, ≤ 4 MiB body cap
   ▼
[]SearchResult{Title, URL, Snippet}                provider order preserved
   ▼
"N. **Title**\n   URL: …\n   Snippet: …"           numbered, blank line between results
```

### Provider abstraction

```go
type SearchResult struct {
	Title   string
	URL     string
	Snippet string
}

type SearchProvider interface {
	Search(ctx context.Context, query string, maxResults int) ([]SearchResult, error)
	Name() string
}
```

A custom provider implements the interface and is passed to `NewTool` — the tool layer is provider-agnostic. Implementations must never panic on a non-positive `maxResults`: SearXNG clamps it to zero (empty result), the DuckDuckGo parser loop-guard yields an empty result, and Brave/Tavily/Exa forward the count to their remote APIs.

### Provider catalog

| Provider | `Name()` | Constructor | Endpoint | Auth | Snippet source |
| -------- | -------- | ----------- | -------- | ---- | -------------- |
| Brave | `brave` | `NewBraveProviderWithClient(apiKey, timeout, client)` | `GET https://api.search.brave.com/res/v1/web/search?q=&count=` | `X-Subscription-Token` header | `web.results[].description` |
| Tavily | `tavily` | `NewTavilyProviderWithClient(apiKey, timeout, client)` | `POST https://api.tavily.com/search` (`search_depth: "basic"`) | `api_key` in the JSON body | `results[].content` |
| Exa | `exa` | `NewExaProviderWithClient(apiKey, timeout, client)` | `POST https://api.exa.ai/search` (`type: "auto"`, `highlights.maxCharacters: 4000`) | `x-api-key` header | `text`, falling back to `highlights` joined with `\n` |
| SearXNG | `searxng` | `NewSearXNGProviderWithClient(baseURL, timeout, client)` | `GET <baseURL>/search?q=&format=json` | none (self-hosted) | `results[].content` |
| DuckDuckGo | `duckduckgo` | `NewDuckDuckGoProviderWithClient(timeout, client)` | `POST https://html.duckduckgo.com/html/` | none (scrape) | `a.result__snippet` text |

Every constructor accepts an optional `*http.Client`; a nil client gets one with the given timeout. Every provider exposes `SetBaseURL` (used by tests and custom endpoints). API keys are constructor parameters the host supplies — the package reads no environment variables itself; the conventional `BRAVE_API_KEY` / `TAVILY_API_KEY` / `EXA_API_KEY` names appear in `docs/tools.md` and gate the (skipped) integration tests, along with `SEARXNG_TEST_URL` for SearXNG.

Results are normalized per provider into `SearchResult` and rendered in provider order; one provider serves a tool, so result lists are single-source.

### Cross-cutting HTTP hardening

- Redirects are never followed: every provider sets `CheckRedirect` to `http.ErrUseLastResponse`, including on a caller-supplied client (via a shallow copy — the caller's client is never mutated), so credentials in custom headers (Brave, Exa) and the POST body (Tavily) can never be re-sent to a redirect target, and a hijacked endpoint cannot aim a server-side request at an arbitrary host. A 3xx surfaces as an `HTTP 3xx` error.
- Response bodies are read through a 4 MiB cap (`maxResponseBodyBytes`), bounding memory against a hostile or broken endpoint.
- All requests are built with `http.NewRequestWithContext`, so caller deadlines and cancellation are honored.

### DuckDuckGo scraper specifics

No official API exists, so the provider scrapes the HTML endpoint and is best-effort by design:

- **POST form shape:** the query is POSTed as `application/x-www-form-urlencoded` (`q`, `kl=wt-wt` — no region restriction) with a same-origin `Referer` derived from the base URL and browser-like `User-Agent`/`Accept` headers; the POST shape survived roughly twice as many requests as GET before the anti-bot flag.
- **Pacing:** consecutive requests from one provider instance are spaced at least 4 s apart (`ddgMinRequestInterval`) plus jitter in `[0, interval/2)`; a fresh provider's first request never waits.
- **Anomaly challenge:** a body carrying the `"bots use DuckDuckGo"` signature — behind a 202 or a 200 — is returned as an anomaly error, never parsed as (empty) results. Only anomaly responses are retried, with bounded exponential backoff (2 s base, doubling per attempt, at most 3 total attempts); after exhaustion the explicit error names the rate limit and suggests switching providers. Non-anomaly errors surface immediately.
- **Parsing:** `div.result` nodes yield `a.result__a` (title + `href`) and `a.result__snippet` (snippet); a result is kept only when both URL and title are non-empty, up to `maxResults`. DuckDuckGo redirect URLs (`/l/?uddg=…`) are decoded to the real target, and protocol-relative `//…` URLs are normalized to `https://…`.

### SearXNG specifics

The provider targets a self-hosted instance's JSON API — the ToS-clean, rate-limit-free path (SearXNG aggregates upstream engines server-side; public instances often disable JSON output or run bot checks). `DefaultSearxngBaseURL` is `http://127.0.0.1:8888` (the upstream docker-compose default); an empty or whitespace-only base URL selects it, and trailing slashes are trimmed. `maxResults` is honored client-side (the JSON API has no count parameter). A 200 whose body is HTML (JSON output disabled, or an anti-bot page) produces an operator-actionable error naming the `search.formats: [html, json]` setting.

### Wiring

The package is host-wired: the host constructs `websearch.NewTool(provider, limits)` with its chosen provider and registers the result on its `ToolRegistry`. The engine itself contains no production call site — `web_search` is an optional tool that hosts register only when a search backend is configured (see [builtins.md](builtins.md)).

## Error Handling

- Provider errors are wrapped by the tool into an `IsError` `ToolResult` (`"search failed: …"`) with a nil Go error; invalid input JSON goes through `tools.ParseInputError`; a missing/empty query (after fallback keys) yields `IsError` with `"query parameter is required"`. Empty search results are a success (`"No results found"`, `IsError: false`).
- An empty configured API key fails at `Search` time with `"API key is not configured"` (Brave, Tavily, Exa).
- Non-2xx responses (including surfaced redirects) produce `HTTP <code>: <capped body>` errors.
- The DuckDuckGo anomaly error names the rate limit, the HTTP status, and the remedies (slow down; switch to an API provider or self-hosted SearXNG). Interrupted pacing/backoff wraps the context error.
- Timeouts come from the provider's HTTP client (the constructor's `timeout` when the caller passes a nil client); `Limits.Timeout` is a host-side configuration default for constructing providers, enforced by the host rather than the tool.

## Invariants

- The tool always declares group `remote_read`, policy `PolicyAlwaysAllow`, and `Untrusted: true`, so the executor classifies its output via `IsToolUntrusted` and wraps observations in `<untrusted-content>` boundaries.
- Every provider response body is read through the 4 MiB cap.
- Every provider HTTP client — default or caller-supplied — refuses redirects, and a caller-supplied client is never mutated.
- Provider implementations never panic on a non-positive `maxResults`.
- `NewTool` always normalizes a non-positive `MaxResults` to the default (5).
- A DuckDuckGo anomaly challenge always surfaces as an explicit error after at most 3 attempts — never as a silent empty result; consecutive requests from one provider instance are always at least `ddgMinRequestInterval` apart, and the first request never waits.
- All requests carry the caller's context, so cancellation always interrupts pacing, backoff, and in-flight requests.

## Related Specs

- [Built-in Tools](builtins.md) - tool catalog, `web_search` registration, description rubric, untrusted-output invariant
- [Security Model](../../architecture/security-model.md) - tool policies, confirmations, and the untrusted-content wrapping this tool's output receives
