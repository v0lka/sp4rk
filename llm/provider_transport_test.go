package llm

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/url"
	"strings"
	"syscall"
	"testing"
	"time"

	oai "github.com/openai/openai-go"
)

// TestTransportKind pins the diagnostic classification of connection-level
// failures. Observability only: the labels let upstream curators tell a
// timeout from a mid-body reset from a DNS miss when a failure reproduces on
// specific payloads across runs.
func TestTransportKind(t *testing.T) {
	deadlineCtx, cancel := context.WithTimeout(context.Background(), 5*time.Millisecond)
	defer cancel()
	<-deadlineCtx.Done()

	dnsErr := &net.DNSError{Err: "no such host", Name: "api.example.com", IsNotFound: true}
	sentinel := errors.New("wrapped")

	tests := []struct {
		name string
		err  error
		want string
	}{
		{"nil", nil, ""},
		{"context deadline", deadlineCtx.Err(), "timeout"},
		{"dns", dnsErr, "dns"},
		{"dns wrapped", fmt.Errorf("dial: %w", dnsErr), "dns"},
		{"conn reset", fmt.Errorf("read: %w", syscall.ECONNRESET), "reset"},
		{"broken pipe", syscall.EPIPE, "reset"},
		{"refused", syscall.ECONNREFUSED, "connect"},
		{"host unreachable", syscall.EHOSTUNREACH, "connect"},
		{"op error reset", &net.OpError{Op: "read", Err: syscall.ECONNRESET}, "reset"},
		{"op error refused", &net.OpError{Op: "dial", Err: syscall.ECONNREFUSED}, "connect"},
		{"url error inner reset", &url.Error{Op: "Post", URL: "https://x", Err: &net.OpError{Op: "read", Err: syscall.ECONNRESET}}, "reset"},
		{"url error plain", &url.Error{Op: "Post", URL: "https://x", Err: sentinel}, "connect"},
		{"eof", io.EOF, "eof"},
		{"unexpected eof", io.ErrUnexpectedEOF, "eof"},
		{"plain error", sentinel, "other"},
		{"tls text", errors.New("tls: handshake failure"), "tls"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := TransportKind(tt.err); got != tt.want {
				t.Errorf("TransportKind = %q, want %q", got, tt.want)
			}
		})
	}
}

// TestAnnotateChatTransport_ConnectionLevelOnly pins that the transport
// suffix attaches only to HTTP 0 (*Error with StatusCode 0) failures and that
// the appended text stays outside the *Error wrapper so retryability
// classification is untouched.
func TestAnnotateChatTransport_ConnectionLevelOnly(t *testing.T) {
	p := &OpenAIProvider{name: "deepseek"}

	// Connection-level failure: annotated, still retryable via the chain.
	inner := NewError("deepseek", 0, true, fmt.Errorf("openai chat completion: %w",
		&url.Error{Op: "Post", Err: &net.OpError{Op: "read", Err: syscall.ECONNRESET}}))
	got := p.annotateChatTransport(inner, oai.ChatCompletionNewParams{})
	if got == nil {
		t.Fatal("expected an error")
	}
	if !strings.Contains(got.Error(), "(transport=reset, request_bytes=") {
		t.Errorf("annotated error = %q, want transport=reset suffix with a size estimate", got.Error())
	}
	if !IsRetryable(got) {
		t.Error("annotation must not alter retryability")
	}
	var llmErr *Error
	if !errors.As(got, &llmErr) || llmErr.StatusCode != 0 {
		t.Error("*Error must stay unwrap-reachable with StatusCode 0 after annotation")
	}

	// HTTP-carrying failure: untouched.
	httpErr := NewError("deepseek", 429, true, errors.New("rate limited"))
	before := httpErr.Error()
	if after := p.annotateChatTransport(httpErr, oai.ChatCompletionNewParams{}).Error(); after != before {
		t.Errorf("HTTP-carrying failure changed: %q -> %q", before, after)
	}

	// Nil passes through.
	if err := p.annotateChatTransport(nil, oai.ChatCompletionNewParams{}); err != nil {
		t.Errorf("nil error must pass through, got %v", err)
	}

	// Non-llm error (pre-call validation): untouched.
	plain := errors.New("openai: missing media type")
	if after := p.annotateChatTransport(plain, oai.ChatCompletionNewParams{}).Error(); after != plain.Error() {
		t.Errorf("non-llm error changed: %q -> %q", plain.Error(), after)
	}
}

// TestAnnotateChatTransport_RequestSizeEstimate pins the serialized-size
// estimate: the marshaled request body length, so census triage can correlate
// reproducible failures with payload size.
func TestAnnotateChatTransport_RequestSizeEstimate(t *testing.T) {
	p := &OpenAIProvider{name: "deepseek"}
	params := oai.ChatCompletionNewParams{}
	params.Messages = append(params.Messages, oai.UserMessage(strings.Repeat("x", 1000)))
	inner := NewError("deepseek", 0, true, errors.New("post \"https://api.example.com\": connection reset by peer"))
	got := p.annotateChatTransport(inner, params)
	if got == nil {
		t.Fatal("expected an error")
	}
	if !strings.Contains(got.Error(), "request_bytes=") {
		t.Errorf("annotated error = %q, want a request_bytes estimate", got.Error())
	}
	if strings.Contains(got.Error(), "request_bytes=-1") {
		t.Errorf("marshalable params must produce a real size estimate, got %q", got.Error())
	}
}
