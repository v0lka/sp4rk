package llm

import (
	"bufio"
	"bytes"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"strings"

	"github.com/openai/openai-go/option"
)

// The openai-go v1.12.0 `packages/ssestream` decoder dispatches an event on
// EVERY blank line and then `Stream.Next` unconditionally `json.Unmarshal`s the
// accumulated `data:` payload — WITHOUT checking that the payload is a complete
// JSON value. Two SSE shapes therefore abort the whole stream with
// `unexpected end of JSON input`:
//
//   - a DATA-LESS event — a `:` comment/keep-alive, a `retry:` directive, a
//     stray leading/trailing blank line, or the widespread `event: X` + blank
//     line + `data: …` framing — yields an empty payload; and
//   - an event whose payload is a TRUNCATED JSON prefix (the decoder cuts the
//     event at a blank line before the value is complete) yields an incomplete
//     payload.
//
// Both are upstream openai/openai-go#556 (empty events) and #618 (incomplete
// JSON). They are NOT fixed by the module bump: the v3 line still ships the
// same decoder, because the real fix belongs in the generated `Stream.Next`
// guard, not in the decoder. The official Rust Codex client — the reference
// consumer of the very backend this provider talks to — simply SKIPS any event
// it cannot parse, so it survives exactly the streams that kill the Go SDK.
//
// The ChatGPT subscription endpoint (llm.chatgpt.auth.mode: "oauth", served
// through https://chatgpt.com/backend-api/codex) answers ONLY with SSE and can
// interleave such events at an arbitrary point, so on the pinned v1 module a
// subscription call could die at a random moment with
// `llm [chatgpt] error (HTTP 0, retryable=false): responses API: unexpected end
// of JSON input` — and, because a status-0 error is classified non-retryable,
// the failure was terminal for the task.
//
// ssestream exposes RegisterDecoder, but NewDecoder looks a decoder up by the
// RAW `Content-Type` header value, so a `text/event-stream; charset=utf-8`
// answer matches no plain registration and the SDK would still pick its buggy
// fallback decoder. Rather than depend on the exact header spelling,
// sseSanitizingMiddleware normalizes the RESPONSE BODY itself: it parses the
// SSE framing and re-emits only the events whose payload is a complete JSON
// value, which the SDK's decoder handles correctly. Empty and unparsable
// (truncated) events are dropped — mirroring the Rust client's "skip what you
// cannot parse" behavior — instead of aborting the call.
//
// The normalizer is engaged when the response `Content-Type` names an event
// stream, OR (defensively) when the body's first line is unmistakably an SSE
// field, so a backend or gateway that streams SSE under a wrong/missing media
// type is still covered. Every other body passes through byte-for-byte.

// isEventStream reports whether a Content-Type header names the SSE media
// type, tolerating parameters (`; charset=utf-8`) and casing.
func isEventStream(contentType string) bool {
	return strings.HasPrefix(strings.ToLower(strings.TrimSpace(contentType)), "text/event-stream")
}

// sseFieldPrefixes are the SSE field names (and the comment marker) accepted at
// column 0. Used only to sniff a body whose Content-Type did not identify an
// event stream.
var sseFieldPrefixes = []string{"data:", "event:", "id:", "retry:", ":"}

// isSSEFieldLine reports whether a line (newline already trimmed) is an SSE
// field line. It requires the token at column 0, which JSON and other non-SSE
// bodies never produce, so it cannot misclassify a non-SSE payload.
func isSSEFieldLine(line string) bool {
	if line == "" {
		return false
	}
	for _, p := range sseFieldPrefixes {
		if strings.HasPrefix(line, p) {
			return true
		}
	}
	return false
}

// sseSanitizingMiddleware returns an openai-go request middleware that wraps a
// streaming response body in an sseSanitizingBody so data-less and unparsable
// events cannot abort the call (see the file comment). It is a no-op for every
// non-SSE response — those bodies pass through untouched.
func sseSanitizingMiddleware(logger *slog.Logger) option.Middleware {
	return func(req *http.Request, next option.MiddlewareNext) (*http.Response, error) {
		res, err := next(req)
		if err != nil || res == nil || res.Body == nil {
			return res, err
		}
		trusted := isEventStream(res.Header.Get("Content-Type"))
		res.Body = newSSESanitizingBody(res.Body, trusted, logger)
		// A rewritten stream has a different length (events are dropped), so a
		// declared length is no longer authoritative.
		res.ContentLength = -1
		res.Header.Del("Content-Length")
		return res, nil
	}
}

// newSSESanitizingBody wraps src. When trusted is true the caller has already
// established (from Content-Type) that the body is an event stream; otherwise
// the body sniffs its first line and falls back to a byte-for-byte passthrough
// when that line is not an SSE field. The caller keeps ownership of closing src
// through the returned ReadCloser.
func newSSESanitizingBody(src io.ReadCloser, trusted bool, logger *slog.Logger) io.ReadCloser {
	b := &sseSanitizingBody{src: src, br: bufio.NewReader(src), logger: logger}
	if trusted {
		b.decided = true
	}
	return b
}

// sseSanitizingBody is an io.ReadCloser that re-emits a text/event-stream body
// with data-less and unparsable events removed. It buffers one SSE event at a
// time (SSE is dispatched event-by-event anyway, so this adds no latency),
// emits an event only when its joined `data:` payload is a complete JSON value,
// and always terminates an emitted event with a blank line so the SDK's decoder
// dispatches it. A body that turns out not to be an event stream is copied
// through unchanged.
type sseSanitizingBody struct {
	src    io.ReadCloser
	br     *bufio.Reader
	logger *slog.Logger

	decided bool         // the sniff decision has been made
	raw     bool         // passthrough mode (not an event stream)
	head    bytes.Buffer // bytes consumed while sniffing, replayed in raw mode
	detErr  error        // the error observed while sniffing (replayed in raw mode)

	out     bytes.Buffer    // bytes ready to hand to the caller
	evData  strings.Builder // joined `data:` fields of the event being read
	evEvent string          // the event's `event:` name, if any
	hasData bool            // the current event carried at least one `data:` field
	eof     bool
	err     error
}

func (b *sseSanitizingBody) Read(p []byte) (int, error) {
	for b.out.Len() == 0 && !b.eof && b.err == nil && !b.passthrough() {
		b.step()
	}
	if b.passthrough() {
		// Not an event stream: copy through byte-for-byte.
		if b.head.Len() > 0 {
			return b.head.Read(p)
		}
		if b.detErr != nil {
			err := b.detErr
			b.detErr = nil
			return 0, err
		}
		return b.br.Read(p)
	}
	if b.out.Len() > 0 {
		return b.out.Read(p)
	}
	if b.err != nil {
		return 0, b.err
	}
	return 0, io.EOF
}

func (b *sseSanitizingBody) Close() error {
	return b.src.Close()
}

// passthrough reports whether the body was found not to be an event stream and
// is being copied through byte-for-byte.
func (b *sseSanitizingBody) passthrough() bool { return b.decided && b.raw }

// step makes the sniff decision once, then reads and processes one line.
func (b *sseSanitizingBody) step() {
	if !b.decided {
		b.detect()
		return
	}
	line, err := b.br.ReadString('\n')
	if err != nil {
		if err == io.EOF {
			// A line with no trailing newline is a partial (truncated) line,
			// and an event with no terminating blank line is an unterminated
			// event: the SSE spec discards both at EOF, and so does the SDK's
			// own scanner. Dropping them keeps a connection cut mid-event a
			// clean "no terminal event" instead of a corrupt payload.
			b.eof = true
			b.err = io.EOF
			return
		}
		b.err = err
		return
	}
	b.handleLine(trimSSELine(line))
}

// detect reads the first line and decides whether the body is an event stream.
func (b *sseSanitizingBody) detect() {
	b.decided = true
	line, err := b.br.ReadString('\n')
	if line == "" {
		// Nothing to inspect (empty body, or an error before any byte).
		b.raw = true
		b.detErr = err
		return
	}
	trimmed := trimSSELine(line)
	if err == nil && isSSEFieldLine(trimmed) {
		b.handleLine(trimmed)
		return
	}
	// Not an SSE field line: pass everything through unchanged.
	b.raw = true
	b.head.WriteString(line)
	b.detErr = err
}

// trimSSELine removes the line terminator, tolerating both LF and CRLF.
func trimSSELine(raw string) string {
	return strings.TrimRight(strings.TrimRight(raw, "\n"), "\r")
}

// handleLine folds one raw SSE line into the current event, dispatching it on a
// blank line. Comments, `retry:`/`id:` directives and unknown fields are
// dropped, exactly as a spec-compliant parser would.
func (b *sseSanitizingBody) handleLine(line string) {
	if line == "" {
		b.dispatch()
		return
	}
	name, value, found := strings.Cut(line, ":")
	if found && value != "" && value[0] == ' ' {
		value = value[1:]
	}
	switch name {
	case "data":
		if b.hasData {
			b.evData.WriteByte('\n')
		}
		b.evData.WriteString(value)
		b.hasData = true
	case "event":
		b.evEvent = value
	}
}

// dispatch emits the accumulated event only when it carries a complete JSON
// payload, and resets state.
func (b *sseSanitizingBody) dispatch() {
	has := b.hasData
	data := b.evData.String()
	event := b.evEvent
	b.evData.Reset()
	b.hasData = false
	b.evEvent = ""

	if !has {
		// A data-less event (keep-alive, retry/id directive, `event:` split
		// from its `data:` by a blank line): valid per the SSE spec and must be
		// skipped, not dispatched to the SDK's json.Unmarshal.
		return
	}
	if !ssePayloadIsCompleteJSON(data) {
		// An empty or truncated payload. The SDK would json.Unmarshal it and
		// abort the whole stream (openai/openai-go#618); the reference client
		// skips it, so we do too and let the stream continue.
		if b.logger != nil {
			b.logger.Debug("llm: skipping unparsable SSE event",
				"event", event, "payload_bytes", len(data), "payload_preview", previewSSELine(data, 120))
		}
		return
	}
	b.emit(event, data)
}

// emit writes one SDK-safe SSE event, preserving its name and multi-line data.
func (b *sseSanitizingBody) emit(event, data string) {
	if event != "" {
		b.out.WriteString("event: ")
		b.out.WriteString(event)
		b.out.WriteByte('\n')
	}
	for _, line := range strings.Split(data, "\n") {
		b.out.WriteString("data: ")
		b.out.WriteString(line)
		b.out.WriteByte('\n')
	}
	b.out.WriteByte('\n')
}

// ssePayloadIsCompleteJSON reports whether an event's joined payload is a
// complete, dispatchable value: the `[DONE]` sentinel or a whole JSON
// object/array. A truncated prefix (openai/openai-go#618) and an empty payload
// (#556) both fail, and are skipped by dispatch.
func ssePayloadIsCompleteJSON(data string) bool {
	t := strings.TrimSpace(data)
	if t == "" {
		return false
	}
	if strings.HasPrefix(t, "[DONE]") {
		return true
	}
	if t[0] != '{' && t[0] != '[' {
		return false
	}
	return json.Valid([]byte(t))
}

// previewSSELine returns a bounded, single-line preview for diagnostics.
func previewSSELine(s string, n int) string {
	s = strings.ReplaceAll(s, "\n", "\\n")
	s = strings.ReplaceAll(s, "\r", "\\r")
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
