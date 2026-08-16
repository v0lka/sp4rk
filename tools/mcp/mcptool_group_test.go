package mcp

import (
	"bytes"
	"context"
	"log/slog"
	"strings"
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

// TestTool_Group_SystemOverrideIsIgnored verifies the reserved system group is
// never honored as a ToolGroupOverride: "system" declares a valid group, but
// hosts treat system tools as trusted orchestration builtins that bypass
// policy gates — honoring the override would exempt an entire untrusted
// external server from every security check. Like any other invalid override,
// it falls back to the transport-derived group.
func TestTool_Group_SystemOverrideIsIgnored(t *testing.T) {
	server := newConnectedServer("hostile-override", "http")
	server.mu.Lock()
	server.toolGroupOverride = sdktools.GroupSystem
	server.mu.Unlock()
	if got := newGroupTestTool(server).Group(); got != sdktools.GroupRemoteMCP {
		t.Fatalf("system-override tool group = %q, want %q (transport default; system is reserved)", got, sdktools.GroupRemoteMCP)
	}
	if got := server.ToolGroup(); got != sdktools.GroupRemoteMCP {
		t.Fatalf("server ToolGroup() = %q, want %q (transport default; system is reserved)", got, sdktools.GroupRemoteMCP)
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

// TestServerConfigFromEntry_ForwardsToolGroupOverride verifies the
// ServerEntry → ServerConfig plumbing (the path StartGateway and Reconfigure
// both use): the gateway-level configuration can express the per-server group
// override, and an empty override stays empty so the transport default
// applies. Env/URL/header expansion is exercised in the same pass.
func TestServerConfigFromEntry_ForwardsToolGroupOverride(t *testing.T) {
	expand := func(s string) string { return s } // identity; expansion is covered elsewhere
	entry := ServerEntry{
		Transport:         "http",
		URL:               "http://localhost:9000",
		Headers:           map[string]string{"X-Api-Key": "${API_KEY}"},
		ToolGroupOverride: sdktools.GroupLocalMCP, // e.g. http-tunnel to localhost
	}
	cfg := serverConfigFromEntry(entry, "/tmp/wd", nil, expand)
	if cfg.ToolGroupOverride != sdktools.GroupLocalMCP {
		t.Fatalf("ServerConfig.ToolGroupOverride = %q, want forwarded %q", cfg.ToolGroupOverride, sdktools.GroupLocalMCP)
	}
	if cfg.URL != entry.URL {
		t.Fatalf("ServerConfig.URL = %q, want %q", cfg.URL, entry.URL)
	}
	if cfg.WorkDir != "/tmp/wd" {
		t.Fatalf("ServerConfig.WorkDir = %q, want the resolved workDir", cfg.WorkDir)
	}

	empty := serverConfigFromEntry(ServerEntry{Transport: "stdio"}, "", nil, expand)
	if empty.ToolGroupOverride != "" {
		t.Fatalf("empty entry override forwarded as %q, want empty (transport default applies)", empty.ToolGroupOverride)
	}
}

// TestConfigChanged_ToolGroupOverrideTriggersReconnect verifies a change to
// ONLY the group override counts as a config change: tools are re-registered
// with the new group, so treating it as a no-op would leave stale groups in
// the registry.
func TestConfigChanged_ToolGroupOverrideTriggersReconnect(t *testing.T) {
	g := newGateway()
	g.expandedConfigs["srv"] = ServerConfig{
		Transport:         "http",
		URL:               "http://localhost:9000",
		ToolGroupOverride: sdktools.GroupRemoteMCP,
	}
	changed := ServerConfig{
		Transport:         "http",
		URL:               "http://localhost:9000",
		ToolGroupOverride: sdktools.GroupLocalMCP, // only the override differs
	}
	if !g.configChanged("srv", changed) {
		t.Fatal("configChanged = false for override-only change, want true (tools must be re-registered under the new group)")
	}
	same := g.expandedConfigs["srv"]
	if g.configChanged("srv", same) {
		t.Fatal("configChanged = true for identical config, want false")
	}
}

// TestConfigChanged_IgnoredOverrideChangeIsNotAChange verifies that a raw
// override change BETWEEN two ignored values (unknown → reserved "system",
// or an ignored value → empty) does not count as a config change: both
// normalize to the same transport-derived group (effectiveToolGroup, the
// same rule Server.ToolGroup applies), so no registered tool's group would
// change and reconnecting the server process would be pure churn.
func TestConfigChanged_IgnoredOverrideChangeIsNotAChange(t *testing.T) {
	g := newGateway()
	base := func(override sdktools.ToolGroup) ServerConfig {
		return ServerConfig{
			Transport:         "stdio",
			Command:           "cmd",
			ToolGroupOverride: override,
		}
	}
	g.expandedConfigs["srv"] = base(sdktools.ToolGroup("typo_group"))

	if g.configChanged("srv", base(sdktools.GroupSystem)) {
		t.Fatal("configChanged = true for unknown → reserved (both ignored), want false: effective group unchanged")
	}
	if g.configChanged("srv", base("")) {
		t.Fatal("configChanged = true for unknown → empty (both ignored), want false: effective group unchanged")
	}
	// Sanity: an ignored value → a VALID override does change the effective
	// group and must still trigger a reconnect.
	if !g.configChanged("srv", base(sdktools.GroupLocalWrite)) {
		t.Fatal("configChanged = false for ignored → valid override, want true")
	}
}

// TestConnect_WarnsOnIgnoredToolGroupOverride verifies the misconfiguration
// is surfaced: an override that will be ignored (here the reserved "system"
// group, with a connection that fails fast for an unrelated reason) must log
// a warning instead of silently falling back to the transport-derived group.
func TestConnect_WarnsOnIgnoredToolGroupOverride(t *testing.T) {
	var logs bytes.Buffer
	s := newServer("misconfigured-warn")
	s.logger = slog.New(slog.NewJSONHandler(&logs, nil))

	_ = s.Connect(context.Background(), ServerConfig{
		Transport:         "http", // missing URL → connect fails fast
		ToolGroupOverride: sdktools.GroupSystem,
	})

	out := logs.String()
	if !strings.Contains(out, "override ignored") {
		t.Fatalf("expected an 'override ignored' warning for the reserved system override, got logs: %s", out)
	}
	if !strings.Contains(out, string(sdktools.GroupSystem)) {
		t.Fatalf("warning must name the ignored override value, got logs: %s", out)
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
