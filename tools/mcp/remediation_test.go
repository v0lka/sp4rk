package mcp

// Regression tests for the MCP remediation set: finding numbers refer to the
// third-party code review this package was remediated against.
//   - 40: the SSE fallback's blocking GET is bounded (boundedStreamHTTPClient)
//   - 41: the SSE stream is not bound to the caller's context lifetime
//   - 44: a stdio server with an empty Command is rejected, never spawned
//   - 51: tools/list preserves the server's raw inputSchema byte-for-byte
//   - 58: the stdio child's stderr pipe is drained, so a chatty server cannot
//     wedge the connection
//   - 63: non-text MCP tool results are delivered as JSON, not a Go value dump
//   - 64: Close drains in-flight tool calls instead of racing the SSE reader
//     goroutine's send on a closed response channel

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/mark3labs/mcp-go/mcp"
	mcpserver "github.com/mark3labs/mcp-go/server"
)

// newSSETestServer starts an SSE-only MCP server (so the sp4rk connect path
// takes the SSE fallback leg) exposing a single instant "echo" tool, plus —
// when delay > 0 — a "slow" tool that blocks for delay before answering.
func newSSETestServer(t *testing.T, delay time.Duration) *httptest.Server {
	t.Helper()
	mcpServer := mcpserver.NewMCPServer("test-sse", "1.0.0")
	mcpServer.AddTool(mcp.NewTool("echo", mcp.WithDescription("echo tool")),
		func(_ context.Context, _ mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			return mcp.NewToolResultText("ok"), nil
		})
	if delay > 0 {
		mcpServer.AddTool(mcp.NewTool("slow", mcp.WithDescription("slow tool")),
			func(_ context.Context, _ mcp.CallToolRequest) (*mcp.CallToolResult, error) {
				time.Sleep(delay)
				return mcp.NewToolResultText("ok"), nil
			})
	}
	srv := httptest.NewServer(mcpserver.NewSSEServer(mcpServer))
	t.Cleanup(srv.Close)
	return srv
}

// TestServer_Connect_StdioEmptyCommandRejected verifies finding 44: a stdio
// server without a command is rejected with a config error at Connect — never
// handed to mcp-go, whose reader goroutine would panic on the nil stdout
// reader and kill the host process.
func TestServer_Connect_StdioEmptyCommandRejected(t *testing.T) {
	// An all-zero config: Transport empty (defaults to stdio), Command empty.
	s := newServer("empty-cmd")
	err := s.Connect(context.Background(), ServerConfig{})
	if err == nil {
		t.Fatal("expected Connect to reject an empty stdio command")
	}
	if !strings.Contains(err.Error(), "non-empty command") {
		t.Errorf("unexpected error: %v", err)
	}
	if s.IsConnected() {
		t.Error("server must not be connected after a rejected connect")
	}

	// Explicit stdio transport hits the same guard.
	s2 := newServer("empty-cmd-explicit")
	err = s2.Connect(context.Background(), ServerConfig{Transport: "stdio", Command: ""})
	if err == nil || !strings.Contains(err.Error(), "non-empty command") {
		t.Errorf("expected the empty-command rejection for explicit stdio, got: %v", err)
	}

	// A URL-only entry that omits Transport is defaulted to stdio — the trap
	// the review cites — and must fail with the same validation error, not a
	// panic in a transport goroutine.
	s3 := newServer("url-only")
	err = s3.Connect(context.Background(), ServerConfig{URL: "http://127.0.0.1:1/mcp"})
	if err == nil || !strings.Contains(err.Error(), "non-empty command") {
		t.Errorf("expected the empty-command rejection for a URL-only entry, got: %v", err)
	}
}

// TestGateway_Start_EmptyStdioCommandFailsClosed verifies finding 44 at the
// gateway boundary: the misconfigured server is recorded as failed and the
// gateway keeps serving the remaining servers.
func TestGateway_Start_EmptyStdioCommandFailsClosed(t *testing.T) {
	gateway := newGateway()

	err := gateway.Start(context.Background(), map[string]ServerConfig{
		"empty": {Transport: "stdio", Command: ""},
	})
	if err == nil {
		t.Fatal("expected a StartError for an empty-command stdio server")
	}
	var startErr *StartError
	if !errors.As(err, &startErr) {
		t.Errorf("expected *StartError, got %T: %v", err, err)
	}
	if gateway.GetServer("empty") != nil {
		t.Error("fail-closed: the rejected server must not enter the connection map")
	}

	status := gateway.Status()
	if len(status) != 1 {
		t.Fatalf("expected 1 status entry, got %d", len(status))
	}
	if status[0].Name != "empty" || status[0].Connected || status[0].Error == "" {
		t.Errorf("expected a disconnected 'empty' entry with the validation error, got %+v", status[0])
	}
}

// TestServer_Connect_SSEHeaderWaitBounded verifies finding 40: an SSE endpoint
// that accepts the TCP connection cannot hold Connect open indefinitely —
// neither of the SSE leg's two blocking waits may outlive the handshake bound:
// the wait for the response headers (bounded by the transport HTTP client) and
// the wait for the SSE endpoint event (bounded by the Start race in
// initializeClientWithStartCtx, which caps mcp-go's fixed 30s internal
// endpoint timeout).
func TestServer_Connect_SSEHeaderWaitBounded(t *testing.T) {
	t.Run("header-blackhole", func(t *testing.T) {
		const timeout = 300 * time.Millisecond

		release := make(chan struct{})
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			<-release // the blackhole: never write response headers
		}))
		t.Cleanup(func() {
			close(release)
			srv.Close()
		})

		s := newServer("sse-blackhole")
		start := time.Now()
		err := s.Connect(context.Background(), ServerConfig{
			Transport: "http",
			URL:       srv.URL,
			Timeout:   timeout,
		})
		elapsed := time.Since(start)

		if err == nil {
			t.Fatal("expected Connect to fail against a header-blackholing endpoint")
		}
		if elapsed < timeout {
			t.Errorf("Connect returned after %v (err %v), before the %v bound elapsed", elapsed, err, timeout)
		}
		// Both transport attempts are bounded: Streamable HTTP's initialize by the
		// handshake context, the SSE GET by the transport client's response-header
		// timeout — twice the bound in total, plus scheduling slack.
		if elapsed > 2*timeout+10*time.Second {
			t.Errorf("Connect did not honor the SSE header bound (took %v; err %v)", elapsed, err)
		}
		if s.IsConnected() {
			t.Error("server must not be left connected after a failed connect")
		}
	})

	t.Run("headers-then-stall", func(t *testing.T) {
		// The second blocking wait: an endpoint that sends 200 + SSE headers
		// and then goes quiet never reaches the header bound — mcp-go's SSE
		// Start instead blocks waiting for the endpoint event under a FIXED
		// 30s internal timeout. Neither the bounded transport HTTP client
		// (headers already arrived) nor the detached start context (no
		// deadline) tightens that wait, so initializeClientWithStartCtx races
		// Start against the handshake bound: Connect must fail with a
		// TimeoutError within ~2x the bound, NOT ~30s.
		const timeout = 300 * time.Millisecond

		release := make(chan struct{})
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method == http.MethodPost {
				// The Streamable HTTP attempt: reject it immediately so the
				// SSE fallback leg is reached without waiting on the
				// handshake context.
				w.WriteHeader(http.StatusMethodNotAllowed)
				return
			}
			// The SSE GET: deliver the response headers, then stall before
			// the endpoint event.
			w.Header().Set("Content-Type", "text/event-stream")
			w.Header().Set("Cache-Control", "no-cache")
			w.WriteHeader(http.StatusOK)
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
			<-release
		}))
		t.Cleanup(func() {
			close(release)
			srv.Close()
		})

		s := newServer("sse-stall")
		start := time.Now()
		err := s.Connect(context.Background(), ServerConfig{
			Transport: "http",
			URL:       srv.URL,
			Timeout:   timeout,
		})
		elapsed := time.Since(start)

		if err == nil {
			t.Fatal("expected Connect to fail against a headers-then-stall endpoint")
		}
		var timeoutErr *TimeoutError
		if !errors.As(err, &timeoutErr) {
			t.Errorf("expected a *TimeoutError from the Start race, got %T: %v", err, err)
		}
		// Streamable HTTP's 405 rejection is immediate; the SSE endpoint wait
		// is cut by the Start race at the bound — twice the bound plus
		// scheduling slack, and decisively below mcp-go's fixed 30s endpoint
		// timeout, which is the whole point of the race.
		if elapsed > 2*timeout+2*time.Second {
			t.Errorf("Connect did not honor the endpoint-event bound (took %v; err %v)", elapsed, err)
		}
		if s.IsConnected() {
			t.Error("server must not be left connected after a failed connect")
		}
	})
}

// TestServer_Connect_SSEStreamSurvivesCallerContextCancel verifies finding 41:
// the SSE event stream outlives the context passed to Connect — cancelling the
// caller's context right after a successful connect must not tear down the
// live connection.
func TestServer_Connect_SSEStreamSurvivesCallerContextCancel(t *testing.T) {
	srv := newSSETestServer(t, 0)

	s := newServer("sse-lifetime")
	ctx, cancel := context.WithCancel(context.Background())
	cfg := ServerConfig{
		Transport:   "http",
		URL:         srv.URL + "/sse",
		Timeout:     10 * time.Second,
		CallTimeout: 5 * time.Second,
	}
	if err := s.Connect(ctx, cfg); err != nil {
		t.Fatalf("Connect to SSE test server: %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })

	// The host idiom the review cites: the enclosing scope's context is
	// cancelled (or its deadline elapses) right after startup.
	cancel()

	// The connection must still be alive and able to serve calls.
	if err := s.DiscoverTools(context.Background()); err != nil {
		t.Fatalf("tools/list after caller-context cancellation: %v", err)
	}
	res, err := s.CallTool(context.Background(), "echo", nil)
	if err != nil {
		t.Fatalf("tools/call after caller-context cancellation: %v", err)
	}
	if len(res.Content) == 0 || mcp.GetTextFromContent(res.Content[0]) != "ok" {
		t.Errorf("unexpected tool result after cancellation: %+v", res)
	}
}

// TestServer_DiscoverTools_PreservesRawInputSchema verifies finding 51: the
// input schema of a discovered tool is stored byte-for-byte as the server
// served it — top-level keywords the mcp-go ToolInputSchema struct does not
// model (enum, oneOf, $ref, title) must survive discovery instead of being
// dropped by a lossy re-marshal.
func TestServer_DiscoverTools_PreservesRawInputSchema(t *testing.T) {
	s := newServer("raw-schema")
	cfg := stdioHelperServerConfig(t, stdioHelperRawSchema, 10*time.Second, 10*time.Second)
	if err := s.Connect(context.Background(), cfg); err != nil {
		t.Fatalf("Connect to scripted stdio helper: %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })

	if err := s.DiscoverTools(context.Background()); err != nil {
		t.Fatalf("DiscoverTools: %v", err)
	}

	tools := s.Tools()
	if len(tools) != 1 {
		t.Fatalf("expected 1 discovered tool, got %d", len(tools))
	}
	tool := tools[0]
	if tool.Name != "raw_tool" {
		t.Errorf("tool name = %q, want %q", tool.Name, "raw_tool")
	}
	// Byte-for-byte: the exact schema object from rawSchemaToolListJSON.
	const wantSchema = `{"type":"object","title":"Raw","properties":{"q":{"type":"string"}},"required":["q"],"enum":["a","b"],"oneOf":[{"type":"object"},{"type":"string"}],"$ref":"#/definitions/Args"}`
	if string(tool.InputSchema) != wantSchema {
		t.Errorf("input schema not preserved verbatim:\n got: %s\nwant: %s", tool.InputSchema, wantSchema)
	}
}

// TestServer_Connect_StdioChattyStderrDoesNotWedge verifies finding 58: a
// stdio server that floods stderr past the OS pipe buffer before answering
// cannot wedge the handshake, because the transport's stderr pipe is drained.
func TestServer_Connect_StdioChattyStderrDoesNotWedge(t *testing.T) {
	s := newServer("chatty")
	cfg := stdioHelperServerConfig(t, stdioHelperStderrFlood, 10*time.Second, 10*time.Second)
	if err := s.Connect(context.Background(), cfg); err != nil {
		t.Fatalf("Connect wedged on a chatty-stderr server: %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })

	// The connection is fully usable after the flood.
	if err := s.DiscoverTools(context.Background()); err != nil {
		t.Fatalf("DiscoverTools after the stderr flood: %v", err)
	}
	res, err := s.CallTool(context.Background(), "fast", nil)
	if err != nil {
		t.Fatalf("tools/call after the stderr flood: %v", err)
	}
	if len(res.Content) == 0 || mcp.GetTextFromContent(res.Content[0]) != "ok" {
		t.Errorf("unexpected tool result after the stderr flood: %+v", res)
	}
}

// TestServer_Close_DrainsInFlightCall verifies finding 64: Close waits for an
// in-flight tool call to complete (so the call gets its result) instead of
// closing the transport under it — which races the SSE reader goroutine's
// send on the response channel it has already fetched ("send on closed
// channel" panics kill the host).
func TestServer_Close_DrainsInFlightCall(t *testing.T) {
	// 500ms, not a bare minimum: the call is launched in a goroutine and only
	// assumed in-flight after a fixed sleep, so every window below scales with
	// the delay — the 1/3 sleep (~167ms) before Close and the 1/3 minimum
	// Close duration (~167ms) must dwarf scheduler jitter even on a loaded,
	// race-enabled CI. At the old 150ms the 50ms windows flaked when the call
	// goroutine was slow to schedule and Close found nothing to drain.
	const toolDelay = 500 * time.Millisecond
	srv := newSSETestServer(t, toolDelay)

	for i := 0; i < 3; i++ {
		s := newServer("race")
		cfg := ServerConfig{
			Transport:   "http",
			URL:         srv.URL + "/sse",
			Timeout:     10 * time.Second,
			CallTimeout: 10 * time.Second,
		}
		if err := s.Connect(context.Background(), cfg); err != nil {
			t.Fatalf("iteration %d: Connect: %v", i, err)
		}

		type callResult struct {
			res *mcp.CallToolResult
			err error
		}
		callCh := make(chan callResult, 1)
		go func() {
			res, err := s.CallTool(context.Background(), "slow", nil)
			callCh <- callResult{res: res, err: err}
		}()

		time.Sleep(toolDelay / 3) // let the call go in flight

		closeStart := time.Now()
		if err := s.Close(); err != nil {
			t.Fatalf("iteration %d: Close: %v", i, err)
		}
		closeElapsed := time.Since(closeStart)

		cr := <-callCh
		if cr.err != nil {
			t.Fatalf("iteration %d: the in-flight call must complete, got error: %v", i, cr.err)
		}
		if len(cr.res.Content) == 0 || mcp.GetTextFromContent(cr.res.Content[0]) != "ok" {
			t.Errorf("iteration %d: unexpected in-flight result: %+v", i, cr.res)
		}
		// The drain is observable: Close must not return before the in-flight
		// call had a chance to finish (two thirds of the delay remained).
		if closeElapsed < toolDelay/3 {
			t.Errorf("iteration %d: Close returned after %v, before the in-flight call could drain (~%v remaining)",
				i, closeElapsed, 2*toolDelay/3)
		}
	}
}

// TestServer_Close_NoInFlightCallIsPrompt verifies the drain adds no latency
// to the ordinary teardown path (no calls in flight).
func TestServer_Close_NoInFlightCallIsPrompt(t *testing.T) {
	srv := newSSETestServer(t, 0)

	s := newServer("prompt-close")
	cfg := ServerConfig{
		Transport:   "http",
		URL:         srv.URL + "/sse",
		Timeout:     10 * time.Second,
		CallTimeout: 5 * time.Second,
	}
	if err := s.Connect(context.Background(), cfg); err != nil {
		t.Fatalf("Connect: %v", err)
	}

	start := time.Now()
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if elapsed := time.Since(start); elapsed > 2*time.Second {
		t.Errorf("Close with no in-flight calls took %v; expected a prompt close", elapsed)
	}
}

// TestExtractTextFromContent_NonTextIsJSON verifies finding 63: every
// non-text content type is delivered as its JSON encoding — valid, parseable,
// and tagged with its MCP content type — never as a Go value dump, and never
// via the lossy fmt.Sprintf default arm of mcp.GetTextFromContent.
func TestExtractTextFromContent_NonTextIsJSON(t *testing.T) {
	t.Run("image", func(t *testing.T) {
		result := extractTextFromContent(mcp.NewImageContent("base64data", "image/png"))
		if !json.Valid([]byte(result)) {
			t.Fatalf("image content is not valid JSON: %q", result)
		}
		if !strings.Contains(result, `"type":"image"`) {
			t.Errorf("image content JSON lacks its type tag: %s", result)
		}
		if !strings.Contains(result, "base64data") {
			t.Errorf("image content JSON lacks the data: %s", result)
		}
	})
	t.Run("audio", func(t *testing.T) {
		result := extractTextFromContent(mcp.NewAudioContent("audiodata", "audio/mp3"))
		if !json.Valid([]byte(result)) {
			t.Fatalf("audio content is not valid JSON: %q", result)
		}
		if !strings.Contains(result, `"type":"audio"`) {
			t.Errorf("audio content JSON lacks its type tag: %s", result)
		}
	})
	t.Run("convertMCPResult", func(t *testing.T) {
		// End to end through convertMCPResult: an image-only result must not
		// degrade into a Go struct dump.
		result := convertMCPResult(&mcp.CallToolResult{
			Content: []mcp.Content{mcp.NewImageContent("base64data", "image/png")},
		})
		if !json.Valid([]byte(result.Content)) {
			t.Fatalf("converted content is not valid JSON: %q", result.Content)
		}
		if strings.Contains(result.Content, "ImageContent") {
			t.Errorf("converted content looks like a Go value dump: %q", result.Content)
		}
	})
}
