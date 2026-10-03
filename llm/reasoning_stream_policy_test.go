package llm

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestRequireStreaming_ReasoningFloor(t *testing.T) {
	for _, effort := range []string{"none", "minimal", "low"} {
		t.Run(effort, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var body struct {
					Reasoning struct {
						Effort string `json:"effort"`
					} `json:"reasoning"`
				}
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Errorf("Decode(%q) = %v, want nil", effort, err)
					http.Error(w, "invalid JSON", http.StatusBadRequest)
					return
				}
				if body.Reasoning.Effort != "low" {
					t.Errorf("requireStreaming(%q).effort = %q, want low", effort, body.Reasoning.Effort)
					http.Error(w, "unsupported reasoning effort", http.StatusBadRequest)
					return
				}
				w.Header().Set("Content-Type", "text/event-stream")
				if _, err := io.WriteString(w, "data: "+`{"type":"response.completed","response":{"id":"resp_test","status":"completed","output":[{"type":"message","id":"msg_test","role":"assistant","content":[{"type":"output_text","text":"ok"}]}]}}`+"\n\n"); err != nil {
					t.Errorf("Write SSE response = %v, want nil", err)
				}
			}))
			t.Cleanup(srv.Close)
			p, err := NewOpenAIProvider(OpenAIProviderConfig{Name: "subscription", APIKey: "test", BaseURL: srv.URL, HTTPClient: srv.Client(), RequireStreaming: true})
			if err != nil {
				t.Fatalf("NewOpenAIProvider(subscription) = %v, want nil", err)
			}
			resp, err := p.ChatCompletion(context.Background(), ChatRequest{Model: "gpt-5.6", ReasoningEffort: effort, Messages: []Message{{Role: "user", Content: "title"}}})
			if err != nil {
				t.Fatalf("ChatCompletion(%q) error = %v, want nil", effort, err)
			}
			if resp.Message.Content != "ok" {
				t.Errorf("ChatCompletion(%q).Content = %q, want ok", effort, resp.Message.Content)
			}
		})
	}
}
