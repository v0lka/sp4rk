package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	anthropic "github.com/liushuangls/go-anthropic/v2"
	oai "github.com/openai/openai-go"
	"github.com/openai/openai-go/responses"
)

// The tests in this file pin each provider's wire spelling for prompt-cache
// token accounting: the OpenAI Chat Completions
// prompt_tokens_details.cached_tokens detail, the DeepSeek
// prompt_cache_hit_tokens extension, the OpenAI Responses API
// input_tokens_details.cached_tokens detail, and the Anthropic
// cache_read_input_tokens / cache_creation_input_tokens pair. The cache
// counters are details of the prompt — every assertion below also pins that
// InputTokens itself keeps its provider-native total semantics (billing
// accounting must not drift).

// TestTokenUsage_JSON pins the serialized shape of TokenUsage: the cache
// fields are omitempty, so a response whose provider reported no cache detail
// serializes exactly as before the fields existed (request/response dumps
// stay byte-identical), and a populated split adds its two keys.
func TestTokenUsage_JSON(t *testing.T) {
	t.Run("no cache detail serializes unchanged", func(t *testing.T) {
		b, err := json.Marshal(TokenUsage{InputTokens: 10, OutputTokens: 5})
		if err != nil {
			t.Fatalf("marshal: %v", err)
		}
		if got, want := string(b), `{"input_tokens":10,"output_tokens":5}`; got != want {
			t.Errorf("json = %s, want %s", got, want)
		}
	})

	t.Run("cache split serializes its keys", func(t *testing.T) {
		b, err := json.Marshal(TokenUsage{
			InputTokens:       12100,
			OutputTokens:      5,
			CachedInputTokens: 7000,
			CacheWriteTokens:  5000,
		})
		if err != nil {
			t.Fatalf("marshal: %v", err)
		}
		want := `{"input_tokens":12100,"output_tokens":5,"cached_input_tokens":7000,"cache_write_tokens":5000}`
		if got := string(b); got != want {
			t.Errorf("json = %s, want %s", got, want)
		}
	})
}

// chatCompletionUsageServer is an httptest server answering /chat/completions
// with the given usage JSON so the provider's synchronous path decodes it
// through the real SDK.
func chatCompletionUsageServer(t *testing.T, usageJSON string) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		body := `{"id":"x","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":` + usageJSON + `}`
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)
	return srv
}

// TestOpenAIProvider_UsageCacheDetail_OpenAISpelling pins the OpenAI Chat
// Completions spelling: prompt_tokens_details.cached_tokens is a SUBSET of
// prompt_tokens, so CachedInputTokens carries the detail while InputTokens
// stays the full prompt count.
func TestOpenAIProvider_UsageCacheDetail_OpenAISpelling(t *testing.T) {
	srv := chatCompletionUsageServer(t, `{"prompt_tokens":700,"completion_tokens":10,"total_tokens":710,"prompt_tokens_details":{"cached_tokens":128}}`)

	p, err := NewOpenAIProvider(OpenAIProviderConfig{Name: "p", APIKey: "k", BaseURL: srv.URL})
	if err != nil {
		t.Fatalf("NewOpenAIProvider: %v", err)
	}

	resp, err := p.ChatCompletion(context.Background(), ChatRequest{
		Model:     "m",
		Messages:  []Message{{Role: "user", Content: "hi"}},
		MaxTokens: 10,
	})
	if err != nil {
		t.Fatalf("ChatCompletion: %v", err)
	}

	if resp.Usage.InputTokens != 700 {
		t.Errorf("InputTokens = %d, want 700 (full prompt, unchanged semantics)", resp.Usage.InputTokens)
	}
	if resp.Usage.OutputTokens != 10 {
		t.Errorf("OutputTokens = %d, want 10", resp.Usage.OutputTokens)
	}
	if resp.Usage.CachedInputTokens != 128 {
		t.Errorf("CachedInputTokens = %d, want 128", resp.Usage.CachedInputTokens)
	}
	if resp.Usage.CacheWriteTokens != 0 {
		t.Errorf("CacheWriteTokens = %d, want 0 (wire format carries no cache-write counter)", resp.Usage.CacheWriteTokens)
	}
}

// TestOpenAIProvider_UsageCacheDetail_DeepSeekSpelling pins the DeepSeek
// spelling: prompt_cache_hit_tokens (prompt_tokens = hit + miss) replaces
// prompt_tokens_details, so the typed SDK detail decodes 0 and the provider
// must recover the count from the usage object's raw JSON.
func TestOpenAIProvider_UsageCacheDetail_DeepSeekSpelling(t *testing.T) {
	srv := chatCompletionUsageServer(t, `{"prompt_tokens":704,"completion_tokens":10,"total_tokens":714,"prompt_cache_hit_tokens":640,"prompt_cache_miss_tokens":64}`)

	p, err := NewOpenAIProvider(OpenAIProviderConfig{Name: "deepseek", APIKey: "k", BaseURL: srv.URL})
	if err != nil {
		t.Fatalf("NewOpenAIProvider: %v", err)
	}

	resp, err := p.ChatCompletion(context.Background(), ChatRequest{
		Model:     "m",
		Messages:  []Message{{Role: "user", Content: "hi"}},
		MaxTokens: 10,
	})
	if err != nil {
		t.Fatalf("ChatCompletion: %v", err)
	}

	if resp.Usage.InputTokens != 704 {
		t.Errorf("InputTokens = %d, want 704 (hit + miss, unchanged semantics)", resp.Usage.InputTokens)
	}
	if resp.Usage.CachedInputTokens != 640 {
		t.Errorf("CachedInputTokens = %d, want 640 (prompt_cache_hit_tokens)", resp.Usage.CachedInputTokens)
	}
	if resp.Usage.CacheWriteTokens != 0 {
		t.Errorf("CacheWriteTokens = %d, want 0", resp.Usage.CacheWriteTokens)
	}
}

// TestChatCompletionUsage_NeitherSpelling pins the no-cache-detail case:
// neither wire spelling present yields zero cache fields with untouched
// totals, and a malformed raw JSON degrades to 0 rather than an error.
func TestChatCompletionUsage_NeitherSpelling(t *testing.T) {
	t.Run("no cache fields", func(t *testing.T) {
		var u oai.CompletionUsage
		if err := json.Unmarshal([]byte(`{"prompt_tokens":704,"completion_tokens":10,"total_tokens":714}`), &u); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		got := chatCompletionUsage(u)
		if got.InputTokens != 704 || got.OutputTokens != 10 || got.CachedInputTokens != 0 || got.CacheWriteTokens != 0 {
			t.Errorf("usage = %+v, want totals 704/10 and zero cache fields", got)
		}
	})

	t.Run("malformed raw json stays zero", func(t *testing.T) {
		u := chatCompletionUsage(oai.CompletionUsage{
			PromptTokens: 5, CompletionTokens: 1, TotalTokens: 6,
		})
		if u.CachedInputTokens != 0 {
			t.Errorf("CachedInputTokens = %d, want 0", u.CachedInputTokens)
		}
	})
}

// TestOpenAIProvider_ChatCompletionStream_UsageCacheDetail pins the streaming
// path: the usage-only terminal chunk reports the same cache detail as the
// synchronous path, in both the OpenAI and DeepSeek spellings.
func TestOpenAIProvider_ChatCompletionStream_UsageCacheDetail(t *testing.T) {
	t.Run("openai spelling", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			sse(t, w, `{"id":"1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":""}]}`)
			sse(t, w, `{"id":"1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`)
			sse(t, w, `{"id":"1","object":"chat.completion.chunk","created":1,"model":"m","choices":[],"usage":{"prompt_tokens":700,"completion_tokens":10,"total_tokens":710,"prompt_tokens_details":{"cached_tokens":128}}}`)
			_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
		}))
		t.Cleanup(srv.Close)

		p, _ := NewOpenAIProvider(OpenAIProviderConfig{Name: "p", APIKey: "k", BaseURL: srv.URL})
		resp, err := p.ChatCompletion(context.Background(), ChatRequest{
			Model:     "m",
			Messages:  []Message{{Role: "user", Content: "hi"}},
			DeltaSink: func(StreamDelta) error { return nil },
		})
		if err != nil {
			t.Fatalf("ChatCompletion(stream): %v", err)
		}
		if resp.Usage.InputTokens != 700 || resp.Usage.OutputTokens != 10 {
			t.Errorf("usage = %+v, want 700/10", resp.Usage)
		}
		if resp.Usage.CachedInputTokens != 128 {
			t.Errorf("CachedInputTokens = %d, want 128", resp.Usage.CachedInputTokens)
		}
	})

	t.Run("deepseek spelling", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			sse(t, w, `{"id":"1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":""}]}`)
			sse(t, w, `{"id":"1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`)
			sse(t, w, `{"id":"1","object":"chat.completion.chunk","created":1,"model":"m","choices":[],"usage":{"prompt_tokens":704,"completion_tokens":10,"total_tokens":714,"prompt_cache_hit_tokens":640,"prompt_cache_miss_tokens":64}}`)
			_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
		}))
		t.Cleanup(srv.Close)

		p, _ := NewOpenAIProvider(OpenAIProviderConfig{Name: "deepseek", APIKey: "k", BaseURL: srv.URL})
		resp, err := p.ChatCompletion(context.Background(), ChatRequest{
			Model:     "m",
			Messages:  []Message{{Role: "user", Content: "hi"}},
			DeltaSink: func(StreamDelta) error { return nil },
		})
		if err != nil {
			t.Fatalf("ChatCompletion(stream): %v", err)
		}
		if resp.Usage.InputTokens != 704 || resp.Usage.OutputTokens != 10 {
			t.Errorf("usage = %+v, want 704/10", resp.Usage)
		}
		if resp.Usage.CachedInputTokens != 640 {
			t.Errorf("CachedInputTokens = %d, want 640 (prompt_cache_hit_tokens)", resp.Usage.CachedInputTokens)
		}
	})
}

// TestResponsesAPI_UsageCacheDetail pins the Responses API spelling:
// input_tokens_details.cached_tokens is a subset of input_tokens and populates
// CachedInputTokens without changing InputTokens.
func TestResponsesAPI_UsageCacheDetail(t *testing.T) {
	resp := &responses.Response{
		Status: responses.ResponseStatusCompleted,
		Output: []responses.ResponseOutputItemUnion{
			{Type: "message"},
		},
		Usage: responses.ResponseUsage{
			InputTokens:  100,
			OutputTokens: 50,
			InputTokensDetails: responses.ResponseUsageInputTokensDetails{
				CachedTokens: 80,
			},
		},
	}
	result, err := convertResponsesResponse(resp, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Usage.InputTokens != 100 {
		t.Errorf("InputTokens = %d, want 100 (unchanged semantics)", result.Usage.InputTokens)
	}
	if result.Usage.OutputTokens != 50 {
		t.Errorf("OutputTokens = %d, want 50", result.Usage.OutputTokens)
	}
	if result.Usage.CachedInputTokens != 80 {
		t.Errorf("CachedInputTokens = %d, want 80", result.Usage.CachedInputTokens)
	}
	if result.Usage.CacheWriteTokens != 0 {
		t.Errorf("CacheWriteTokens = %d, want 0 (wire format carries no cache-write counter)", result.Usage.CacheWriteTokens)
	}
}

// TestAnthropicProvider_ParseResponse_CacheSplit pins the Anthropic split:
// input_tokens excludes the cache counters on the wire, the provider keeps
// summing all three into InputTokens (true prompt size, unchanged billing
// accounting), and the split surfaces as CachedInputTokens (cache reads) and
// CacheWriteTokens (cache creation).
func TestAnthropicProvider_ParseResponse_CacheSplit(t *testing.T) {
	p, _ := NewAnthropicProvider(AnthropicProviderConfig{APIKey: "test-key"})

	resp := anthropic.MessagesResponse{
		Content: []anthropic.MessageContent{
			{Type: anthropic.MessagesContentTypeText, Text: stringPtr("Hello!")},
		},
		StopReason: "end_turn",
		Usage: anthropic.MessagesUsage{
			InputTokens:              100,
			OutputTokens:             5,
			CacheCreationInputTokens: 5000,
			CacheReadInputTokens:     7000,
		},
	}

	result, err := p.parseResponse(resp, nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Usage.InputTokens != 12100 {
		t.Errorf("InputTokens = %d, want 100+5000+7000 = 12100 (sum semantics unchanged)", result.Usage.InputTokens)
	}
	if result.Usage.OutputTokens != 5 {
		t.Errorf("OutputTokens = %d, want 5", result.Usage.OutputTokens)
	}
	if result.Usage.CachedInputTokens != 7000 {
		t.Errorf("CachedInputTokens = %d, want 7000 (cache_read_input_tokens)", result.Usage.CachedInputTokens)
	}
	if result.Usage.CacheWriteTokens != 5000 {
		t.Errorf("CacheWriteTokens = %d, want 5000 (cache_creation_input_tokens)", result.Usage.CacheWriteTokens)
	}
}
