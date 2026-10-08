package llm

import (
	"context"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	oai "github.com/openai/openai-go"
)

// ---------------------------------------------------------------------------
// Finding #12: readOpenAIErrorBody must bound the READ at the reader — the
// 4 KiB cap has to limit the allocation, not merely cut the already-buffered
// string. A remote endpoint answering a failed request with a very large or
// endless body must not turn the error path into an unbounded heap
// allocation, on either OpenAI protocol (wrapError and wrapResponsesError
// both funnel through this function).
// ---------------------------------------------------------------------------

// countingReadCloser counts how many body bytes the error path actually
// consumed, so a regression to an unbounded io.ReadAll is observable as a
// byte count far beyond the cap.
type countingReadCloser struct {
	r io.Reader
	n int
}

func (c *countingReadCloser) Read(p []byte) (int, error) {
	n, err := c.r.Read(p)
	c.n += n
	return n, err
}

func (c *countingReadCloser) Close() error { return nil }

// newOAIErrorWithRawBody builds a minimal *oai.Error whose Response.Body is a
// counting reader over body — the shape readOpenAIErrorBody consumes.
func newOAIErrorWithRawBody(statusCode int, body io.Reader) (*oai.Error, *countingReadCloser) {
	cb := &countingReadCloser{r: body}
	return &oai.Error{
		Response: &http.Response{
			StatusCode: statusCode,
			Status:     http.StatusText(statusCode),
			Body:       cb,
			Header:     make(http.Header),
		},
		StatusCode: statusCode,
	}, cb
}

func TestReadOpenAIErrorBody_BoundsReadBeforeCap(t *testing.T) {
	const capBytes = 4096

	t.Run("1 MiB body: read is bounded, string is truncated", func(t *testing.T) {
		apiErr, cb := newOAIErrorWithRawBody(http.StatusBadGateway, strings.NewReader(strings.Repeat("x", 1<<20)))
		got := readOpenAIErrorBody(apiErr)
		if cb.n > capBytes+1 {
			t.Errorf("consumed %d body bytes, want at most %d: the read itself must be capped, not just the returned string", cb.n, capBytes+1)
		}
		want := strings.Repeat("x", capBytes) + "..."
		if got != want {
			t.Errorf("truncated body = %d bytes, want %d bytes (4096 x's + ellipsis)", len(got), len(want))
		}
	})

	t.Run("exactly at the cap: returned whole without ellipsis", func(t *testing.T) {
		apiErr, _ := newOAIErrorWithRawBody(http.StatusBadRequest, strings.NewReader(strings.Repeat("y", capBytes)))
		got := readOpenAIErrorBody(apiErr)
		if strings.HasSuffix(got, "...") {
			t.Errorf("body of exactly %d bytes must not be truncated, got suffix %q", capBytes, got[len(got)-3:])
		}
		if len(got) != capBytes {
			t.Errorf("len = %d, want %d", len(got), capBytes)
		}
	})

	t.Run("whitespace is still trimmed before the cap check", func(t *testing.T) {
		apiErr, _ := newOAIErrorWithRawBody(http.StatusBadRequest, strings.NewReader("\n  upstream exploded  \n"))
		if got := readOpenAIErrorBody(apiErr); got != "upstream exploded" {
			t.Errorf("got %q, want %q", got, "upstream exploded")
		}
	})

	t.Run("unreadable body returns empty without panicking", func(t *testing.T) {
		apiErr, _ := newOAIErrorWithRawBody(http.StatusInternalServerError, io.NopCloser(errReader{}))
		if got := readOpenAIErrorBody(apiErr); got != "" {
			t.Errorf("got %q, want empty string", got)
		}
	})
}

// errReader fails every read, simulating a transport error mid-body.
type errReader struct{}

func (errReader) Read([]byte) (int, error) { return 0, io.ErrUnexpectedEOF }

// TestOpenAIProvider_ErrorBodyCappedOnWire drives the full SDK round-trip: a
// gateway answering a failed request with a > 4 KiB non-standard envelope
// ({"detail":...} — no nested "error" field, so apiErr.Message stays empty and
// the raw body is the only diagnostic source) must surface a truncated body,
// never the whole payload.
func TestOpenAIProvider_ErrorBodyCappedOnWire(t *testing.T) {
	body := `{"detail":"` + strings.Repeat("x", 8192) + `"}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = io.WriteString(w, body)
	}))
	t.Cleanup(srv.Close)

	p, err := NewOpenAIProvider(OpenAIProviderConfig{
		Name: "zen", APIKey: "k", BaseURL: srv.URL,
		Logger: slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	if err != nil {
		t.Fatalf("NewOpenAIProvider: %v", err)
	}

	_, err = p.ChatCompletion(context.Background(), ChatRequest{
		Model:    "deepseek-chat",
		Messages: []Message{{Role: "user", Content: "hi"}},
	})
	if err == nil {
		t.Fatal("expected an error from the 400 response, got nil")
	}
	// The body prefix that survives the cap is `{"detail":"` (11 bytes) plus
	// 4085 x's; 4086 consecutive x's cannot occur in the truncated message.
	if !strings.Contains(err.Error(), strings.Repeat("x", 4085)) {
		t.Errorf("expected the error to carry the 4 KiB body prefix, got: %.200s", err.Error())
	}
	if strings.Contains(err.Error(), strings.Repeat("x", 4086)) {
		t.Errorf("error message exceeds the 4 KiB cap: %d bytes of the payload survived", len(err.Error()))
	}
}

// ---------------------------------------------------------------------------
// Finding #60 (Chat Completions): the assistant refusal channel must be
// surfaced — a refusal or content-filtered answer arrives as HTTP 200 with an
// empty "content" and the model's actual message in "refusal"; dropping it
// ends the run as a silent empty success.
// ---------------------------------------------------------------------------

func TestOpenAIProvider_ChatRefusalSurfaced(t *testing.T) {
	const refusalText = "I cannot help with that."

	t.Run("null content with refusal", func(t *testing.T) {
		body := `{"id":"1","object":"chat.completion","created":1,"model":"gpt-4o","choices":[{"index":0,"message":{"role":"assistant","content":null,"refusal":"` + refusalText + `"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`
		p := newOpenAIProviderForBody(t, body, http.StatusOK)

		resp, err := p.ChatCompletion(context.Background(), ChatRequest{
			Model:    "gpt-4o",
			Messages: []Message{{Role: "user", Content: "hi"}},
		})
		if err != nil {
			t.Fatalf("ChatCompletion: %v", err)
		}
		if resp.Message.Content != refusalText {
			t.Errorf("content = %q, want the refusal %q", resp.Message.Content, refusalText)
		}
		if resp.StopReason != "end_turn" {
			t.Errorf("stop reason = %q, want end_turn", resp.StopReason)
		}
	})

	t.Run("empty content with refusal", func(t *testing.T) {
		body := `{"id":"1","object":"chat.completion","created":1,"model":"gpt-4o","choices":[{"index":0,"message":{"role":"assistant","content":"","refusal":"` + refusalText + `"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`
		p := newOpenAIProviderForBody(t, body, http.StatusOK)

		resp, err := p.ChatCompletion(context.Background(), ChatRequest{
			Model:    "gpt-4o",
			Messages: []Message{{Role: "user", Content: "hi"}},
		})
		if err != nil {
			t.Fatalf("ChatCompletion: %v", err)
		}
		if resp.Message.Content != refusalText {
			t.Errorf("content = %q, want the refusal %q", resp.Message.Content, refusalText)
		}
	})

	t.Run("refusal does not override real content", func(t *testing.T) {
		body := `{"id":"1","object":"chat.completion","created":1,"model":"gpt-4o","choices":[{"index":0,"message":{"role":"assistant","content":"the answer","refusal":"` + refusalText + `"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`
		p := newOpenAIProviderForBody(t, body, http.StatusOK)

		resp, err := p.ChatCompletion(context.Background(), ChatRequest{
			Model:    "gpt-4o",
			Messages: []Message{{Role: "user", Content: "hi"}},
		})
		if err != nil {
			t.Fatalf("ChatCompletion: %v", err)
		}
		if resp.Message.Content != "the answer" {
			t.Errorf("content = %q, want %q (refusal must only fill an empty content)", resp.Message.Content, "the answer")
		}
	})
}

// TestOpenAIProvider_ChatStreamRefusalSurfaced verifies the streaming
// counterpart: refusal fragments arrive in the typed delta.Refusal field, are
// forwarded to the sink as text deltas, and assemble into the message content
// when no content deltas arrived.
func TestOpenAIProvider_ChatStreamRefusalSurfaced(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		sse(t, w, `{"id":"1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","refusal":"I can"},"finish_reason":""}]}`)
		sse(t, w, `{"id":"1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{"refusal":"not do that."},"finish_reason":""}]}`)
		sse(t, w, `{"id":"1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`)
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	t.Cleanup(srv.Close)

	p, err := NewOpenAIProvider(OpenAIProviderConfig{Name: "p", APIKey: "k", BaseURL: srv.URL})
	if err != nil {
		t.Fatalf("NewOpenAIProvider: %v", err)
	}

	var deltas []StreamDelta
	resp, err := p.ChatCompletion(context.Background(), ChatRequest{
		Model:     "gpt-4o",
		Messages:  []Message{{Role: "user", Content: "hi"}},
		DeltaSink: func(d StreamDelta) error { deltas = append(deltas, d); return nil },
	})
	if err != nil {
		t.Fatalf("ChatCompletion(stream): %v", err)
	}

	const want = "I cannot do that."
	if resp.Message.Content != want {
		t.Errorf("content = %q, want %q", resp.Message.Content, want)
	}
	if resp.StopReason != "end_turn" {
		t.Errorf("stop reason = %q, want end_turn", resp.StopReason)
	}
	if len(deltas) != 2 || deltas[0].Text != "I can" || deltas[1].Text != "not do that." {
		t.Errorf("deltas = %+v, want [I can, not do that.]", deltas)
	}
}

// newOpenAIProviderForBody returns a provider backed by a server answering
// every request with body and statusCode — a minimal wire harness for the
// synchronous Chat Completions path.
func newOpenAIProviderForBody(t *testing.T, body string, statusCode int) *OpenAIProvider {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(statusCode)
		_, _ = io.WriteString(w, body)
	}))
	t.Cleanup(srv.Close)

	p, err := NewOpenAIProvider(OpenAIProviderConfig{
		Name: "p", APIKey: "k", BaseURL: srv.URL,
		Logger: slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	if err != nil {
		t.Fatalf("NewOpenAIProvider: %v", err)
	}
	return p
}
