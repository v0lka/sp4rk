package llm

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
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

// TestTransportKindErrno pins the errno-to-label mapping behind TransportKind's
// errors.As(syscall.Errno) branch. The POSIX sentinels classify identically on
// every platform — on Windows they are matched for parity even though real
// failures carry raw Winsock codes (see errno_windows_test.go for those).
func TestTransportKindErrno(t *testing.T) {
	tests := []struct {
		name  string
		errno syscall.Errno
		want  string
	}{
		{"ECONNRESET", syscall.ECONNRESET, "reset"},
		{"EPIPE", syscall.EPIPE, "reset"},
		{"ECONNREFUSED", syscall.ECONNREFUSED, "connect"},
		{"EHOSTUNREACH", syscall.EHOSTUNREACH, "connect"},
		{"ENETUNREACH", syscall.ENETUNREACH, "connect"},
		{"ETIMEDOUT", syscall.ETIMEDOUT, "connect"},
		{"ECONNABORTED", syscall.ECONNABORTED, "connect"},
		{"unrelated", syscall.EINVAL, ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := transportKindForErrno(tt.errno); got != tt.want {
				t.Errorf("transportKindForErrno(%s = %d) = %q, want %q", tt.name, tt.errno, got, tt.want)
			}
		})
	}
}

// TestAnnotateChatTransport_LogsRequestedModel pins that the WARN record's
// model key carries the model actually requested (params.Model) — the
// dimension census triage correlates failures by — not the provider name,
// which the provider key already carries.
func TestAnnotateChatTransport_LogsRequestedModel(t *testing.T) {
	var buf bytes.Buffer
	p := &OpenAIProvider{name: "deepseek", logger: slog.New(slog.NewTextHandler(&buf, nil))}

	inner := NewError("deepseek", 0, true, fmt.Errorf("openai chat completion: %w",
		&url.Error{Op: "Post", Err: &net.OpError{Op: "read", Err: syscall.ECONNRESET}}))
	params := oai.ChatCompletionNewParams{Model: "deepseek-chat"}
	if got := p.annotateChatTransport(inner, params); got == nil {
		t.Fatal("expected an error")
	}
	logged := buf.String()
	if !strings.Contains(logged, "model=deepseek-chat") {
		t.Errorf("WARN log must carry the requested model (deepseek-chat), got %q", logged)
	}
	if strings.Contains(logged, "model=deepseek ") || strings.Contains(logged, "model=deepseek\n") {
		t.Errorf("model key must not carry the provider name, got %q", logged)
	}
	if !strings.Contains(logged, "provider=deepseek") {
		t.Errorf("WARN log must keep the provider key, got %q", logged)
	}
}

// TestAnnotateChatTransport_CallerCancellationNotAnnotated pins that
// caller-initiated cancellation is not annotated: a cancellation is a routine
// host decision (shutdown, pause, step budget), not a connection-level
// provider failure, so it passes through with no WARN log and no
// transport=…/request_bytes=… suffix. A genuine deadline expiry must stay
// annotated as a timeout.
func TestAnnotateChatTransport_CallerCancellationNotAnnotated(t *testing.T) {
	var buf bytes.Buffer
	p := &OpenAIProvider{name: "deepseek", logger: slog.New(slog.NewTextHandler(&buf, nil))}

	// Real-shaped cancellation: HTTP 0 *Error wrapping the SDK's *url.Error
	// wrapping context.Canceled — exactly how a mid-call cancel surfaces.
	inner := NewError("deepseek", 0, true, fmt.Errorf("openai chat completion: %w",
		&url.Error{Op: "Post", URL: "https://api.example.com/v1/chat/completions", Err: context.Canceled}))
	got := p.annotateChatTransport(inner, oai.ChatCompletionNewParams{Model: "deepseek-chat"})
	if got == nil {
		t.Fatal("expected an error")
	}
	if !errors.Is(got, context.Canceled) {
		t.Error("cancellation must stay errors.Is-detectable after annotate")
	}
	if msg := got.Error(); strings.Contains(msg, "transport=") || strings.Contains(msg, "request_bytes=") {
		t.Errorf("caller cancellation must not be annotated, got %q", msg)
	}
	if buf.Len() != 0 {
		t.Errorf("caller cancellation must not WARN-log, got %q", buf.String())
	}

	// Bare and plainly wrapped cancellation pass through untouched as well.
	for _, err := range []error{
		context.Canceled,
		fmt.Errorf("openai chat completion: %w", context.Canceled),
	} {
		if got := p.annotateChatTransport(err, oai.ChatCompletionNewParams{}); got.Error() != err.Error() {
			t.Errorf("cancellation %v changed: %q", err, got.Error())
		}
	}
	if buf.Len() != 0 {
		t.Errorf("cancellation must never WARN-log, got %q", buf.String())
	}

	// A genuine deadline expiry is a transport outcome, not cancellation:
	// still annotated as a timeout.
	deadline := NewError("deepseek", 0, true, fmt.Errorf("openai chat completion: %w",
		&url.Error{Op: "Post", URL: "https://api.example.com/v1/chat/completions", Err: context.DeadlineExceeded}))
	annotated := p.annotateChatTransport(deadline, oai.ChatCompletionNewParams{Model: "deepseek-chat"})
	if !strings.Contains(annotated.Error(), "(transport=timeout, request_bytes=") {
		t.Errorf("deadline expiry must stay annotated as timeout, got %q", annotated.Error())
	}
}
