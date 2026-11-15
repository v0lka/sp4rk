package llm

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestResponsesSSEAutoResendClassification(t *testing.T) {
	for _, shape := range []string{"error", "response.failed"} {
		for _, tc := range []struct {
			code      string
			kind      ErrKind
			retryable bool
		}{
			{"rate_limit_exceeded", ErrKindRateLimit, true},
			{"server_busy", ErrKindOverloaded, true},
			{"overloaded_error", ErrKindOverloaded, true},
			{"server_error", "", true}, {"timeout", "", true},
			{"engine_overloaded", "", true}, {"service_unavailable", "", true},
			{"insufficient_quota", "", false}, {"invalid_prompt", "", false}, {"unknown", "", false},
		} {
			t.Run(shape+"/"+tc.code, func(t *testing.T) {
				srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
					w.Header().Set("Content-Type", "text/event-stream")
					event := fmt.Sprintf(`{"type":"error","code":%q,"message":"failure"}`, tc.code)
					if shape == "response.failed" {
						event = fmt.Sprintf(`{"type":"response.failed","response":{"id":"r","object":"response","status":"failed","error":{"code":%q,"message":"failure"}}}`, tc.code)
					}
					respSSE(w, event)
				}))
				t.Cleanup(srv.Close)
				p, err := NewOpenAIProvider(OpenAIProviderConfig{Name: "chatgpt", APIKey: "k", BaseURL: srv.URL, RequireStreaming: true})
				if err != nil {
					t.Fatal(err)
				}
				_, err = p.ChatCompletion(context.Background(), ChatRequest{Model: "gpt-5.6", Protocol: ProtocolResponses, Messages: []Message{{Role: "user", Content: "hi"}}})
				var classified *Error
				if !errors.As(err, &classified) {
					t.Fatalf("ChatCompletion(%s, %s) error = %v, want *Error", shape, tc.code, err)
				}
				if classified.Provider != "chatgpt" || classified.ErrKind != tc.kind || classified.StatusCode != 0 || classified.Retryable != tc.retryable {
					t.Errorf("ChatCompletion(%s, %s) classification = provider:%q kind:%q status:%d retryable:%t, want chatgpt/%q/0/%t", shape, tc.code, classified.Provider, classified.ErrKind, classified.StatusCode, classified.Retryable, tc.kind, tc.retryable)
				}
			})
		}
	}
}
