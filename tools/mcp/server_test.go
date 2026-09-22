package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/mark3labs/mcp-go/mcp"
)

func TestNewServer_Initialization(t *testing.T) {
	tests := []struct {
		name       string
		serverName string
	}{
		{"simple name", "test-server"},
		{"empty name", ""},
		{"name with special chars", "server-with-dashes_and_underscores"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := newServer(tt.serverName)
			if s == nil {
				t.Fatal("NewServer returned nil")
			}
			if s.Name() != tt.serverName {
				t.Errorf("Name() = %q, want %q", s.Name(), tt.serverName)
			}
			if s.IsConnected() {
				t.Error("new server should not be connected")
			}
			if len(s.Tools()) != 0 {
				t.Error("new server should have no tools")
			}
		})
	}
}

func TestServer_CallTool_NilClient(t *testing.T) {
	s := newServer("test")

	_, err := s.CallTool(context.Background(), "some_tool", nil)
	if err == nil {
		t.Fatal("expected error when calling tool on disconnected server")
	}

	expected := "mcp server test is not connected"
	if err.Error() != expected {
		t.Errorf("error = %q, want %q", err.Error(), expected)
	}
}

func TestServer_CallTool_NilClientWithArgs(t *testing.T) {
	s := newServer("my-server")

	args := map[string]any{
		"path": "/tmp",
	}

	_, err := s.CallTool(context.Background(), "read_file", args)
	if err == nil {
		t.Fatal("expected error when calling tool on disconnected server")
	}

	if err.Error() != "mcp server my-server is not connected" {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestServer_DiscoverTools_NilClient(t *testing.T) {
	s := newServer("test")

	err := s.DiscoverTools(context.Background())
	if err == nil {
		t.Fatal("expected error when discovering tools on disconnected server")
	}

	expected := "mcp server test is not connected"
	if err.Error() != expected {
		t.Errorf("error = %q, want %q", err.Error(), expected)
	}
}

func TestServer_Close_NilClient_PreservesTools(t *testing.T) {
	s := newServer("test")

	// Closing a server with no client should return nil
	err := s.Close()
	if err != nil {
		t.Errorf("Close() on nil client should return nil, got: %v", err)
	}
}

func TestServer_Close_NilClient_ClearsState(t *testing.T) {
	s := newServer("test")
	// Manually set some tools
	s.tools = []ToolInfo{
		{Name: "tool1", Description: "desc", InputSchema: json.RawMessage(`{}`)},
	}

	err := s.Close()
	if err != nil {
		t.Errorf("Close() error = %v", err)
	}

	// When client is nil, Close returns early — tools are NOT cleared
	if len(s.tools) != 1 {
		t.Errorf("tools should not be cleared when client is nil, got len=%d", len(s.tools))
	}
	if s.IsConnected() {
		t.Error("should not be connected after Close")
	}
}

func TestServer_Close_MultipleTimes(t *testing.T) {
	s := newServer("test")

	// Multiple closes should be safe
	for i := 0; i < 3; i++ {
		err := s.Close()
		if err != nil {
			t.Errorf("Close() call %d error = %v", i, err)
		}
	}
}

func TestServer_Tools_ReturnsCopy(t *testing.T) {
	s := newServer("test")
	s.tools = []ToolInfo{
		{Name: "tool1", Description: "desc1", InputSchema: json.RawMessage(`{}`)},
		{Name: "tool2", Description: "desc2", InputSchema: json.RawMessage(`{}`)},
	}

	tools1 := s.Tools()
	tools2 := s.Tools()

	// Verify it returns correct count
	if len(tools1) != 2 {
		t.Fatalf("expected 2 tools, got %d", len(tools1))
	}

	// Modify the returned slice and verify original is unaffected
	tools1[0].Name = "modified"
	if tools2[0].Name == "modified" {
		t.Error("Tools() should return a copy, not a reference")
	}

	// Original should be unaffected
	original := s.Tools()
	if original[0].Name != "tool1" {
		t.Error("original tools should not be modified")
	}
}

func TestServer_IsConnected(t *testing.T) {
	s := newServer("test")

	if s.IsConnected() {
		t.Error("new server should not be connected")
	}

	// We can't easily set a real client, but we verified the nil path
}

func TestServer_Name(t *testing.T) {
	tests := []struct {
		name     string
		expected string
	}{
		{"simple", "simple"},
		{"empty", ""},
		{"with-special-chars", "with-special-chars"},
		{"unicode-日本語", "unicode-日本語"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := newServer(tt.expected)
			if got := s.Name(); got != tt.expected {
				t.Errorf("Name() = %q, want %q", got, tt.expected)
			}
		})
	}
}

func TestServer_Connect_UnsupportedTransport(t *testing.T) {
	s := newServer("test")

	cfg := ServerConfig{
		Transport: "unsupported",
	}

	err := s.Connect(context.Background(), cfg)
	if err == nil {
		t.Fatal("expected error for unsupported transport")
	}

	expected := `unsupported transport type "unsupported" for MCP server test`
	if err.Error() != expected {
		t.Errorf("error = %q, want %q", err.Error(), expected)
	}
}

func TestServer_Connect_HTTP_MissingURL(t *testing.T) {
	s := newServer("test")

	cfg := ServerConfig{
		Transport: "http",
		URL:       "",
	}

	err := s.Connect(context.Background(), cfg)
	if err == nil {
		t.Fatal("expected error for HTTP transport without URL")
	}

	expected := "http transport requires URL for MCP server test"
	if err.Error() != expected {
		t.Errorf("error = %q, want %q", err.Error(), expected)
	}
}

func TestServer_Connect_HTTP_InvalidURL(t *testing.T) {
	s := newServer("test")

	cfg := ServerConfig{
		Transport: "http",
		URL:       "http://localhost:99999/invalid", // Invalid port
	}

	err := s.Connect(context.Background(), cfg)
	if err == nil {
		t.Fatal("expected error for invalid HTTP URL")
	}

	// Should get an error related to HTTP connection failure
	// The error could be from Streamable HTTP or SSE fallback
	if !containsSubstring(err.Error(), "failed to start MCP client") && !containsSubstring(err.Error(), "failed to create HTTP MCP client") {
		t.Errorf("error should indicate HTTP connection failure, got: %v", err)
	}
}

func TestServerConfig_DefaultsToStdio(t *testing.T) {
	// Verify that empty transport defaults to stdio behavior
	cfg := ServerConfig{
		Transport: "",
		Command:   "/nonexistent/command",
	}

	s := newServer("test")
	err := s.Connect(context.Background(), cfg)

	// Should fail because command doesn't exist, not because of transport type
	if err == nil {
		t.Fatal("expected error for nonexistent command")
	}

	// Error should mention stdio client creation
	if !containsSubstring(err.Error(), "stdio") {
		t.Errorf("error should mention stdio, got: %v", err)
	}
}

func TestServerConfig_TransportStdioExplicit(t *testing.T) {
	// Verify explicit "stdio" transport works
	cfg := ServerConfig{
		Transport: "stdio",
		Command:   "/nonexistent/command",
	}

	s := newServer("test")
	err := s.Connect(context.Background(), cfg)

	// Should fail because command doesn't exist
	if err == nil {
		t.Fatal("expected error for nonexistent command")
	}

	// Error should mention stdio client creation
	if !containsSubstring(err.Error(), "stdio") {
		t.Errorf("error should mention stdio, got: %v", err)
	}
}

// containsSubstring checks if s contains substr (case-insensitive).
func containsSubstring(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || substr == "" ||
		(s != "" && substr != "" && containsFold(s, substr)))
}

func containsFold(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		match := true
		for j := 0; j < len(substr); j++ {
			if toLower(s[i+j]) != toLower(substr[j]) {
				match = false
				break
			}
		}
		if match {
			return true
		}
	}
	return false
}

func toLower(c byte) byte {
	if c >= 'A' && c <= 'Z' {
		return c + ('a' - 'A')
	}
	return c
}

// ---------------------------------------------------------------------------
// stdio environment allowlist (ASI04)
// ---------------------------------------------------------------------------

func TestIsAllowedStdioEnvVar_Allowlist(t *testing.T) {
	allowed := []string{
		"PATH=/usr/bin:/bin",
		"HOME=/home/user",
		"USER=user",
		"SHELL=/bin/bash",
		"LANG=en_US.UTF-8",
		"TERM=xterm-256color",
		"TMPDIR=/tmp",
		"TEMP=/tmp",
		"TMP=/tmp",
		"LC_ALL=en_US.UTF-8",
		"LC_CTYPE=C",
	}
	for _, e := range allowed {
		if !isAllowedStdioEnvVar(e) {
			t.Errorf("expected %q to be allowed", e)
		}
	}
}

func TestIsAllowedStdioEnvVar_RejectsSecrets(t *testing.T) {
	rejected := []string{
		"ANTHROPIC_API_KEY=sk-ant-xxx",
		"OPENAI_API_KEY=sk-xxx",
		"AWS_SECRET_ACCESS_KEY=abcdef",
		"GITHUB_TOKEN=ghp_xxx",
		"DATABASE_URL=postgres://user:pass@host/db",
	}
	for _, e := range rejected {
		if isAllowedStdioEnvVar(e) {
			t.Errorf("expected %q to be REJECTED (not on allowlist)", e)
		}
	}
}

// TestIsAllowedStdioEnvVar_AllowsProxyAndRuntimeVars verifies that network
// proxy configuration (whose credentials are stripped at forwarding time by
// safeStdioEnv), CA trust anchors, and Python venv/conda activation variables
// are allowlisted for stdio MCP servers.
func TestIsAllowedStdioEnvVar_AllowsProxyAndRuntimeVars(t *testing.T) {
	allowed := []string{
		"HTTP_PROXY=http://user:pass@proxy:8080",
		"HTTPS_PROXY=http://proxy:8443",
		"NO_PROXY=localhost,127.0.0.1,.internal",
		"ALL_PROXY=socks5://proxy:1080",
		"http_proxy=http://proxy:8080",
		"https_proxy=http://proxy:8443",
		"no_proxy=localhost",
		"all_proxy=socks5://proxy:1080",
		"SSL_CERT_FILE=/etc/ssl/certs/corp-root.pem",
		"SSL_CERT_DIR=/etc/ssl/certs",
		"REQUESTS_CA_BUNDLE=/etc/ssl/certs/corp-root.pem",
		"CURL_CA_BUNDLE=/etc/ssl/certs/corp-root.pem",
		"NODE_EXTRA_CA_CERTS=/etc/ssl/certs/corp-root.pem",
		"VIRTUAL_ENV=/home/user/.venv",
		"CONDA_PREFIX=/opt/miniconda3/envs/proj",
		"CONDA_DEFAULT_ENV=proj",
		"CONDA_SHLVL=1",
		"CONDA_PREFIX_1=/opt/miniconda3",
		"CONDA_PROMPT_MODIFIER=(proj) ",
		"CONDA_EXE=/opt/miniconda3/bin/conda",
		"CONDA_PYTHON_EXE=/opt/miniconda3/bin/python",
	}
	for _, e := range allowed {
		if !isAllowedStdioEnvVar(e) {
			t.Errorf("expected %q to be allowed", e)
		}
	}
}

func TestSafeStdioEnv_FiltersToAllowlist(t *testing.T) {
	raw := []string{
		"PATH=/usr/bin",
		"HOME=/home/user",
		"TMPDIR=/tmp",
		"ANTHROPIC_API_KEY=sk-ant-secret",
		"OPENAI_API_KEY=sk-secret",
		"LC_ALL=en_US.UTF-8",
		"SOME_RANDOM_VAR=value",
	}
	got := safeStdioEnv(raw)
	if len(got) != 4 { // PATH, HOME, TMPDIR, LC_ALL
		t.Fatalf("expected 4 allowlisted vars, got %d: %v", len(got), got)
	}
	for _, e := range got {
		if strings.Contains(e, "secret") {
			t.Errorf("secret leaked through filter: %q", e)
		}
	}
}

func TestIsAllowedStdioEnvVarForOS_Windows(t *testing.T) {
	allowed := []string{
		// Windows spells PATH as "Path"; the matcher must be case-insensitive.
		"Path=C:\\Windows;C:\\Windows\\System32",
		"SystemRoot=C:\\Windows",
		"ComSpec=C:\\Windows\\system32\\cmd.exe",
		"PATHEXT=.COM;.EXE;.BAT;.CMD",
		"USERPROFILE=C:\\Users\\alice",
		"APPDATA=C:\\Users\\alice\\AppData\\Roaming",
		"LOCALAPPDATA=C:\\Users\\alice\\AppData\\Local",
		"TEMP=C:\\Users\\alice\\AppData\\Local\\Temp",
		"TMP=C:\\Users\\alice\\AppData\\Local\\Temp",
	}
	for _, e := range allowed {
		if !isAllowedStdioEnvVarForOS(e, "windows") {
			t.Errorf("expected %q to be allowed on Windows", e)
		}
	}

	rejected := []string{
		"ANTHROPIC_API_KEY=sk-ant-xxx",
		"AWS_SECRET_ACCESS_KEY=abcdef",
		"GITHUB_TOKEN=ghp_xxx",
	}
	for _, e := range rejected {
		if isAllowedStdioEnvVarForOS(e, "windows") {
			t.Errorf("expected %q to be REJECTED on Windows", e)
		}
	}

	// The Windows case-insensitivity must not leak into POSIX hosts.
	if isAllowedStdioEnvVarForOS("Path=/bin", "linux") {
		t.Errorf("expected mixed-case %q to be REJECTED on POSIX (case-sensitive)", "Path")
	}
	if isAllowedStdioEnvVarForOS("systemroot=C:\\Windows", "linux") {
		t.Errorf("expected mixed-case %q to be REJECTED on POSIX (case-sensitive)", "systemroot")
	}
}

func TestSafeStdioEnvForOS_Windows(t *testing.T) {
	raw := []string{
		"Path=C:\\Windows",
		"SystemRoot=C:\\Windows",
		"ComSpec=C:\\Windows\\system32\\cmd.exe",
		"ANTHROPIC_API_KEY=sk-ant-secret",
	}
	got := safeStdioEnvForOS(raw, "windows")
	if len(got) != 3 {
		t.Fatalf("expected 3 Windows allowlisted vars, got %d: %v", len(got), got)
	}
	for _, e := range got {
		if strings.Contains(e, "secret") {
			t.Errorf("secret leaked through filter: %q", e)
		}
	}
}

func TestSafeStdioEnv_EmptyReturnsNil(t *testing.T) {
	if got := safeStdioEnv(nil); got != nil {
		t.Errorf("expected nil for empty input, got %v", got)
	}
}

func TestSafeStdioEnv_StripsProxyCredentials(t *testing.T) {
	raw := []string{
		"PATH=/usr/bin",
		"HTTP_PROXY=http://alice:s3cret@proxy.example.com:8080",
		"HTTPS_PROXY=https://bob:hunter2@secure-proxy.example.com:8443",
		"ALL_PROXY=socks5://carol:p@ss@proxy.example.com:1080",
		"NO_PROXY=localhost,.internal,10.0.0.0/8",
		"http_proxy=http://dave:pw@proxy.example.com:8080",
	}
	got := safeStdioEnv(raw)

	want := map[string]string{
		"PATH":        "/usr/bin",
		"HTTP_PROXY":  "http://proxy.example.com:8080",
		"HTTPS_PROXY": "https://secure-proxy.example.com:8443",
		"ALL_PROXY":   "socks5://proxy.example.com:1080",
		"NO_PROXY":    "localhost,.internal,10.0.0.0/8",
		"http_proxy":  "http://proxy.example.com:8080",
	}
	if len(got) != len(want) {
		t.Fatalf("expected %d allowlisted vars, got %d: %v", len(want), len(got), got)
	}
	for _, e := range got {
		key, val, _ := strings.Cut(e, "=")
		if want[key] != val {
			t.Errorf("unexpected value for %s: got %q, want %q", key, val, want[key])
		}
		if strings.Contains(e, "@") {
			t.Errorf("userinfo component not stripped: %q", e)
		}
	}
}

func TestSafeStdioEnv_StripsSchemeLessProxyCredentials(t *testing.T) {
	raw := []string{
		"HTTP_PROXY=alice:s3cret@proxy.example.com:8080",
	}
	got := safeStdioEnv(raw)
	if len(got) != 1 {
		t.Fatalf("expected 1 var, got %d: %v", len(got), got)
	}
	if got[0] != "HTTP_PROXY=proxy.example.com:8080" {
		t.Errorf("expected scheme-less credential stripped, got %q", got[0])
	}
}

// ---------------------------------------------------------------------------
// stdio transport: timeout, cancellation, unhealthy tracking
//
// These tests exercise a real subprocess over the stdio transport, driven by a
// scripted MCP server. Rather than shipping a separate helper binary, the test
// binary re-executes itself: when stdioHelperEnvVar is set, TestMain runs
// runStdioHelper instead of the tests. This is the standard "helper process"
// pattern — no extra build step, no binary in testdata, and it works on every
// platform the package is tested on.
// ---------------------------------------------------------------------------

// stdioHelperEnvVar selects the scripted-server behavior when the test binary is
// re-executed as a helper subprocess.
const stdioHelperEnvVar = "SP4RK_MCP_STDIO_HELPER_TEST"

// Scripted-helper modes.
const (
	// stdioHelperNoInit consumes stdin but never answers, so the client's
	// initialize handshake can only end via its own timeout.
	stdioHelperNoInit = "noinit"
	// stdioHelperWedge answers initialize/tools/list so a connection can be
	// established, answers tools/call "fast" immediately, and stays silent on
	// any other tool (notably "wedge") so that call can only end via
	// timeout or cancellation.
	stdioHelperWedge = "wedge"
)

// TestMain lets the same test binary act as a scripted stdio MCP server when it
// is re-executed with stdioHelperEnvVar set.
func TestMain(m *testing.M) {
	if mode := os.Getenv(stdioHelperEnvVar); mode != "" {
		runStdioHelper(mode)
		return
	}
	os.Exit(m.Run())
}

// runStdioHelper implements the scripted MCP server used by the stdio transport
// tests. It speaks newline-delimited JSON-RPC on stdin/stdout (the stdio
// transport framing).
//
// It returns on stdin EOF — which is exactly what Server.Close triggers by
// closing the child's stdin — so the process exits cleanly and teardown never
// hangs on a wedged request. That is why a "wedge" is implemented as silence
// (no reply) rather than a blocked handler: the helper stays responsive to the
// other requests, and still terminates the instant the client goes away.
func runStdioHelper(mode string) {
	if mode == stdioHelperNoInit {
		// Read (and discard) forever without ever answering: initialize can
		// only be ended by the client's timeout.
		_, _ = io.Copy(io.Discard, os.Stdin)
		return
	}

	dec := json.NewDecoder(os.Stdin)
	enc := json.NewEncoder(os.Stdout)
	for {
		var req struct {
			ID     json.RawMessage `json:"id"`
			Method string          `json:"method"`
			Params json.RawMessage `json:"params"`
		}
		if err := dec.Decode(&req); err != nil {
			return // EOF (or broken pipe): the client is gone, exit.
		}
		if len(req.ID) == 0 {
			continue // notification (e.g. notifications/initialized)
		}

		switch req.Method {
		case "initialize":
			writeHelperResult(enc, req.ID, map[string]any{
				"protocolVersion": mcp.LATEST_PROTOCOL_VERSION,
				"capabilities":    map[string]any{},
				"serverInfo": map[string]any{
					"name":    "stdio-helper",
					"version": "1.0.0",
				},
			})
		case "tools/list":
			writeHelperResult(enc, req.ID, map[string]any{"tools": []any{}})
		case "tools/call":
			var p struct {
				Name string `json:"name"`
			}
			_ = json.Unmarshal(req.Params, &p)
			if p.Name == "fast" {
				writeHelperResult(enc, req.ID, map[string]any{
					"content": []any{map[string]any{"type": "text", "text": "ok"}},
					"isError": false,
				})
			}
			// Any other tool (intentionally including "wedge") is left
			// unanswered so the client's call can only end via
			// timeout/cancellation.
		default:
			writeHelperError(enc, req.ID, -32601, "method not found")
		}
	}
}

func writeHelperResult(enc *json.Encoder, id json.RawMessage, result any) {
	_ = enc.Encode(map[string]any{
		"jsonrpc": "2.0",
		"id":      id,
		"result":  result,
	})
}

func writeHelperError(enc *json.Encoder, id json.RawMessage, code int, message string) {
	_ = enc.Encode(map[string]any{
		"jsonrpc": "2.0",
		"id":      id,
		"error":   map[string]any{"code": code, "message": message},
	})
}

// stdioHelperCommand returns the path to the currently running test binary,
// which is re-executed as the scripted MCP server.
func stdioHelperCommand(t *testing.T) string {
	t.Helper()
	exe, err := os.Executable()
	if err != nil {
		t.Fatalf("resolve test binary path: %v", err)
	}
	return exe
}

// stdioHelperServerConfig builds a stdio ServerConfig that launches the test
// binary as the scripted helper in the given mode.
func stdioHelperServerConfig(t *testing.T, mode string, handshakeTimeout, callTimeout time.Duration) ServerConfig {
	t.Helper()
	return ServerConfig{
		Transport:   "stdio",
		Command:     stdioHelperCommand(t),
		Env:         map[string]string{stdioHelperEnvVar: mode},
		Timeout:     handshakeTimeout,
		CallTimeout: callTimeout,
	}
}

// connectStdioHelper connects to the scripted helper in "wedge" mode with the
// given per-call timeout and registers teardown.
func connectStdioHelper(t *testing.T, callTimeout time.Duration) *Server {
	t.Helper()
	s := newServer("stdio-helper")
	cfg := stdioHelperServerConfig(t, stdioHelperWedge, 10*time.Second, callTimeout)
	if err := s.Connect(context.Background(), cfg); err != nil {
		t.Fatalf("Connect to scripted stdio helper: %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })
	return s
}

// TestServer_Connect_StdioHandshakeTimeout verifies that a server which reads
// the handshake but never answers cannot stall Connect: the initialize exchange
// is bounded by the configured timeout and surfaces a *TimeoutError.
func TestServer_Connect_StdioHandshakeTimeout(t *testing.T) {
	const timeout = 200 * time.Millisecond

	s := newServer("noinit")
	start := time.Now()
	err := s.Connect(context.Background(), stdioHelperServerConfig(t, stdioHelperNoInit, timeout, 0))
	elapsed := time.Since(start)

	if err == nil {
		t.Fatal("expected a handshake timeout, got a nil error")
	}
	var timeoutErr *TimeoutError
	if !errors.As(err, &timeoutErr) {
		t.Fatalf("error = %T (%v), want *TimeoutError", err, err)
	}
	if timeoutErr.Op != "initialize" {
		t.Errorf("TimeoutError.Op = %q, want %q", timeoutErr.Op, "initialize")
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("errors.Is(err, context.DeadlineExceeded) = false; err = %v", err)
	}
	if elapsed < timeout {
		t.Errorf("handshake returned after %v, before the %v timeout elapsed", elapsed, timeout)
	}
	if elapsed > timeout+5*time.Second {
		t.Errorf("handshake timeout fired too late (took %v)", elapsed)
	}
	if s.IsConnected() {
		t.Error("server must not be left connected after a failed handshake")
	}
}

// TestServer_CallTool_StdioTimeout verifies that a wedged tools/call returns a
// *TimeoutError quickly instead of hanging, and does not by itself mark the
// server unhealthy.
func TestServer_CallTool_StdioTimeout(t *testing.T) {
	const callTimeout = 200 * time.Millisecond

	s := connectStdioHelper(t, callTimeout)

	start := time.Now()
	_, err := s.CallTool(context.Background(), "wedge", nil)
	elapsed := time.Since(start)

	if err == nil {
		t.Fatal("expected a timeout for the wedged call, got a nil error")
	}
	var timeoutErr *TimeoutError
	if !errors.As(err, &timeoutErr) {
		t.Fatalf("error = %T (%v), want *TimeoutError", err, err)
	}
	if timeoutErr.Op != "call_tool" || timeoutErr.Tool != "wedge" {
		t.Errorf("TimeoutError = {Op:%q Tool:%q}, want {call_tool wedge}", timeoutErr.Op, timeoutErr.Tool)
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("errors.Is(err, context.DeadlineExceeded) = false; err = %v", err)
	}
	if elapsed < callTimeout {
		t.Errorf("call returned after %v, before the %v timeout elapsed", elapsed, callTimeout)
	}
	if elapsed > callTimeout+5*time.Second {
		t.Errorf("call timeout fired too late (took %v)", elapsed)
	}
	if s.Status().Unhealthy {
		t.Error("a single timeout must not mark the server unhealthy")
	}
}

// TestServer_CallTool_StdioContextCancel verifies that cancelling the caller's
// context aborts an in-flight call promptly with context.Canceled, and that the
// cancellation is not misattributed to the server (no *TimeoutError, no
// unhealthy marking, no consecutive-timeout increment).
func TestServer_CallTool_StdioContextCancel(t *testing.T) {
	// A long per-call timeout guarantees the caller's cancellation — not the
	// wire timeout — is what ends the call.
	s := connectStdioHelper(t, 30*time.Second)

	ctx, cancel := context.WithCancel(context.Background())
	timer := time.AfterFunc(100*time.Millisecond, cancel)
	defer timer.Stop()

	start := time.Now()
	_, err := s.CallTool(ctx, "wedge", nil)
	elapsed := time.Since(start)

	if !errors.Is(err, context.Canceled) {
		t.Fatalf("error = %v, want context.Canceled", err)
	}
	var timeoutErr *TimeoutError
	if errors.As(err, &timeoutErr) {
		t.Errorf("caller cancellation must not be reported as a server *TimeoutError (got %v)", timeoutErr)
	}
	if elapsed > 5*time.Second {
		t.Errorf("cancelled call did not abort promptly (took %v)", elapsed)
	}
	if st := s.Status(); st.Unhealthy || st.Error != "" {
		t.Errorf("caller cancellation must not mark the server unhealthy: %+v", st)
	}
	if s.consecutiveTimeouts != 0 {
		t.Errorf("consecutiveTimeouts = %d after a cancellation, want 0", s.consecutiveTimeouts)
	}
}

// TestServer_CallTool_StdioUnhealthyAfterConsecutiveTimeouts verifies the
// back-to-back timeout streak: fewer than the threshold keeps the server
// healthy, reaching it flips Unhealthy (with an error string), and a subsequent
// call that returns in time clears both.
func TestServer_CallTool_StdioUnhealthyAfterConsecutiveTimeouts(t *testing.T) {
	const callTimeout = 150 * time.Millisecond

	s := connectStdioHelper(t, callTimeout)

	for i := 1; i <= unhealthyTimeoutThreshold; i++ {
		if _, err := s.CallTool(context.Background(), "wedge", nil); err == nil {
			t.Fatalf("timeout %d/%d: expected an error, got nil", i, unhealthyTimeoutThreshold)
		}
		if i < unhealthyTimeoutThreshold {
			if s.Status().Unhealthy {
				t.Fatalf("server marked unhealthy after only %d timeout(s)", i)
			}
		}
	}

	st := s.Status()
	if !st.Unhealthy {
		t.Fatalf("server not marked unhealthy after %d consecutive timeouts", unhealthyTimeoutThreshold)
	}
	if st.Error == "" {
		t.Error("an unhealthy server must report a non-empty error")
	}

	// A call that returns in time proves the server is responsive again.
	if _, err := s.CallTool(context.Background(), "fast", nil); err != nil {
		t.Fatalf("recovery call failed: %v", err)
	}

	st = s.Status()
	if st.Unhealthy {
		t.Error("a successful call must clear the unhealthy flag")
	}
	if st.Error != "" {
		t.Errorf("a successful call must clear the error, got %q", st.Error)
	}
	if s.consecutiveTimeouts != 0 {
		t.Errorf("consecutiveTimeouts = %d after recovery, want 0", s.consecutiveTimeouts)
	}
}
