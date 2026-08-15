package mcp

import (
	"context"
	"testing"

	sdktools "github.com/v0lka/sp4rk/tools"
)

// newConnectedServer builds a Server in the state a successful Connect leaves
// it in for the given transport. Connect against a live server process is not
// practical in unit tests (see server_test.go: only error paths are covered),
// so tests simulate the post-Connect state exactly.
func newConnectedServer(name, transport string) *Server {
	s := newServer(name)
	s.mu.Lock()
	s.transportType = transport
	s.mu.Unlock()
	return s
}

func newGroupTestTool(s *Server) *Tool {
	return NewTool(s, ToolInfo{
		Name:        "some_tool",
		Description: "test tool",
		InputSchema: []byte(`{"type": "object"}`),
	})
}

// TestTool_Group_StdioTransportIsLocalMCP verifies stdio-served tools are
// tagged local_mcp: the server runs as a local child process.
func TestTool_Group_StdioTransportIsLocalMCP(t *testing.T) {
	server := newConnectedServer("local-server", "stdio")
	if got := newGroupTestTool(server).Group(); got != sdktools.GroupLocalMCP {
		t.Fatalf("stdio tool group = %q, want %q", got, sdktools.GroupLocalMCP)
	}
}

// TestTool_Group_HTTPTransportIsRemoteMCP verifies http-served tools are
// tagged remote_mcp: the server is reached over the network.
func TestTool_Group_HTTPTransportIsRemoteMCP(t *testing.T) {
	server := newConnectedServer("remote-server", "http")
	if got := newGroupTestTool(server).Group(); got != sdktools.GroupRemoteMCP {
		t.Fatalf("http tool group = %q, want %q", got, sdktools.GroupRemoteMCP)
	}
}

// TestTool_Group_UnconnectedServerDefaultsToLocal verifies a server that has
// not completed Connect yet defaults to the stdio/local mapping, mirroring
// Server.Connect's defaulting of an unspecified transport.
func TestTool_Group_UnconnectedServerDefaultsToLocal(t *testing.T) {
	server := newServer("fresh")
	if got := newGroupTestTool(server).Group(); got != sdktools.GroupLocalMCP {
		t.Fatalf("unconnected tool group = %q, want %q", got, sdktools.GroupLocalMCP)
	}
}

// TestTool_Group_PerServerOverrideWins verifies a ServerConfig
// ToolGroupOverride tags the server's tools regardless of transport.
func TestTool_Group_PerServerOverrideWins(t *testing.T) {
	server := newConnectedServer("tunneled", "http")
	server.mu.Lock()
	server.toolGroupOverride = sdktools.GroupRemoteWrite
	server.mu.Unlock()
	if got := newGroupTestTool(server).Group(); got != sdktools.GroupRemoteWrite {
		t.Fatalf("overridden tool group = %q, want %q", got, sdktools.GroupRemoteWrite)
	}
}

// TestTool_Group_InvalidOverrideFallsBackToTransport verifies an override that
// does not declare a valid group is ignored and the transport default applies.
func TestTool_Group_InvalidOverrideFallsBackToTransport(t *testing.T) {
	server := newConnectedServer("misconfigured", "http")
	server.mu.Lock()
	server.toolGroupOverride = sdktools.ToolGroup("unknown")
	server.mu.Unlock()
	if got := newGroupTestTool(server).Group(); got != sdktools.GroupRemoteMCP {
		t.Fatalf("invalid-override tool group = %q, want %q (transport default)", got, sdktools.GroupRemoteMCP)
	}
}

// TestConnect_CapturesToolGroupOverride verifies the ServerConfig → Server
// override plumbing through the real Connect path: the override is captured
// even when the connection attempt itself fails (here: an http config without
// a URL), so the configured tag still applies to any tool listed later.
func TestConnect_CapturesToolGroupOverride(t *testing.T) {
	s := newServer("plumbing")
	err := s.Connect(context.Background(), ServerConfig{
		Transport:         "http",                 // missing URL → connect fails
		ToolGroupOverride: sdktools.GroupLocalMCP, // e.g. http-tunnel to localhost
	})
	if err == nil {
		t.Fatal("expected Connect to fail without a URL")
	}
	if got := s.ToolGroup(); got != sdktools.GroupLocalMCP {
		t.Fatalf("post-Connect ToolGroup() = %q, want override %q", got, sdktools.GroupLocalMCP)
	}
}

// TestTool_Judge_NoConcern verifies the MCP judge stays a zero outcome (no
// tool-specific concern) under the JudgeOutcome shape.
func TestTool_Judge_NoConcern(t *testing.T) {
	outcome := newGroupTestTool(newServer("srv")).Judge(context.Background(), nil)
	if outcome.Allow || outcome.Reason != "" {
		t.Fatalf("expected zero no-concern outcome, got %+v", outcome)
	}
}
