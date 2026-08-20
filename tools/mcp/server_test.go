package mcp

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
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
