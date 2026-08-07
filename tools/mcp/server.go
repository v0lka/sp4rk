// Package mcp provides MCP (Model Context Protocol) integration for the agent.
// It manages connections to external MCP servers and exposes their tools through
// the unified Tool interface.
package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/exec"
	"strings"
	"sync"

	mcpclient "github.com/mark3labs/mcp-go/client"
	"github.com/mark3labs/mcp-go/client/transport"
	"github.com/mark3labs/mcp-go/mcp"

	"github.com/v0lka/sp4rk/sysproc"
)

// ServerConfig defines how to launch an MCP server.
// This is a local copy to avoid importing backend/config.
type ServerConfig struct {
	Transport  string            // "stdio" | "http"; default "stdio"
	Command    string            // stdio: command to execute
	Args       []string          // stdio: command arguments
	Env        map[string]string // stdio: environment variables
	URL        string            // http: server URL
	Headers    map[string]string // http: custom headers
	WorkDir    string            // stdio: working directory for the server process
	HTTPClient *http.Client      // http: optional proxy-configured HTTP client
}

// Server represents a connection to an external MCP server process.
type Server struct {
	name          string
	client        *mcpclient.Client
	tools         []ToolInfo
	lastError     string
	transportType string
	logger        *slog.Logger
	mu            sync.RWMutex
}

// ToolInfo holds metadata about a tool discovered from an MCP server.
type ToolInfo struct {
	Name        string
	Description string
	InputSchema json.RawMessage
}

// newServer creates a new Server instance with the given name.
func newServer(name string) *Server {
	return &Server{
		name:  name,
		tools: make([]ToolInfo, 0),
	}
}

func (s *Server) log() *slog.Logger {
	if s.logger != nil {
		return s.logger
	}
	return slog.Default()
}

// Name returns the server's configured name.
func (s *Server) Name() string {
	return s.name
}

// Connect spawns the MCP server process and initializes the connection.
// Supports both stdio and HTTP transports based on cfg.Transport.
func (s *Server) Connect(ctx context.Context, cfg ServerConfig) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Determine transport type (default to stdio when unspecified)
	transportType := cfg.Transport
	if transportType == "" {
		transportType = "stdio"
	}

	var client *mcpclient.Client
	var err error

	switch transportType {
	case "stdio":
		client, err = s.connectStdio(ctx, cfg)
	case "http":
		client, err = s.connectHTTP(ctx, cfg)
	default:
		s.lastError = fmt.Sprintf("unsupported transport type %q", transportType)
		return fmt.Errorf("unsupported transport type %q for MCP server %s", transportType, s.name)
	}

	if err != nil {
		s.lastError = err.Error()
		return err
	}

	s.client = client
	s.transportType = transportType
	s.lastError = ""
	s.log().Debug("MCP server connected", "server", s.name, "transport", transportType)
	return nil
}

// connectStdio creates a stdio MCP client.
func (s *Server) connectStdio(ctx context.Context, cfg ServerConfig) (*mcpclient.Client, error) {
	// Build environment variables slice using an allowlist rather than
	// os.Environ(). MCP servers run arbitrary, potentially third-party
	// commands from config (ASI04 — agentic supply chain); forwarding the
	// full parent environment would leak host secrets (LLM API keys, proxy
	// credentials, etc.) to every stdio MCP child process. Only a minimal
	// safe set required for a process to find its own executables and
	// locale is inherited; anything else the server needs must be declared
	// explicitly in its cfg.Env.
	env := safeStdioEnv(os.Environ())
	for key, value := range cfg.Env {
		env = append(env, fmt.Sprintf("%s=%s", key, value))
	}

	// Always install a custom command factory so the spawned MCP server process
	// runs without a visible console window (CREATE_NO_WINDOW on Windows). The
	// mcp-go transport's default path does not set this flag, which causes a
	// console window to flash or stay open for every stdio MCP server under a
	// GUI-subsystem host application. cmdEnv already carries os.Environ()+cfg.Env,
	// matching the default merge performed by the transport, so behaviour is
	// preserved apart from window suppression.
	workDir := cfg.WorkDir
	opts := []transport.StdioOption{
		transport.WithCommandFunc(
			func(cmdCtx context.Context, command string, cmdEnv []string, args []string) (*exec.Cmd, error) {
				cmd := exec.CommandContext(cmdCtx, command, args...)
				cmd.Env = cmdEnv
				if workDir != "" {
					cmd.Dir = workDir
				}
				sysproc.HideConsole(cmd)
				return cmd, nil
			},
		),
	}

	// Create stdio MCP client
	client, err := mcpclient.NewStdioMCPClientWithOptions(cfg.Command, env, cfg.Args, opts...)
	if err != nil {
		return nil, fmt.Errorf("failed to create stdio MCP client for %s: %w", s.name, err)
	}

	if err := s.initializeClient(ctx, client); err != nil {
		if closeErr := client.Close(); closeErr != nil {
			s.log().Debug("failed to close MCP client after connection failure", "error", closeErr)
		}
		return nil, err
	}

	return client, nil
}

// stdioEnvAllowlist is the set of environment variables that are inherited
// from the host by stdio MCP server processes. These are the minimal vars a
// process needs to locate its own executables (PATH, HOME), identify the user
// (USER, SHELL), select a locale (LANG, LC_*), and place temp files (TMPDIR,
// TEMP/TMP). Everything else must be declared explicitly in the server's
// cfg.Env. Secrets — LLM API keys, proxy credentials, etc. — must never be
// forwarded implicitly (ASI04).
//
// HOME is intentionally included: many Node/Python MCP servers read
// ~/.config, ~/.npmrc or ~/.cache at startup and misbehave or crash without
// it. HOME is not itself a secret. An operator who wants strict isolation
// (e.g. to block ~/.aws/credentials reads) can strip HOME via cfg.Env —
// explicit cfg.Env entries are applied after this filter and win.
var stdioEnvAllowlist = map[string]struct{}{
	"PATH":   {},
	"HOME":   {},
	"USER":   {},
	"SHELL":  {},
	"LANG":   {},
	"TERM":   {},
	"TMPDIR": {},
	"TEMP":   {},
	"TMP":    {},
	// Locale variables share the LC_ prefix.
}

// isAllowedStdioEnvVar reports whether the given env var (NAME=value form)
// is on the allowlist (exact match on NAME, or an LC_* locale variable).
func isAllowedStdioEnvVar(entry string) bool {
	key, _, ok := strings.Cut(entry, "=")
	if !ok {
		return false
	}
	if _, allowed := stdioEnvAllowlist[key]; allowed {
		return true
	}
	return strings.HasPrefix(key, "LC_")
}

// safeStdioEnv filters a raw os.Environ()-style slice down to the allowlisted
// variables only. Explicit server cfg.Env values are applied on top by the
// caller, so they always win and are not subject to this filter.
func safeStdioEnv(raw []string) []string {
	if len(raw) == 0 {
		return nil
	}
	out := make([]string, 0, len(stdioEnvAllowlist)+4)
	for _, e := range raw {
		if isAllowedStdioEnvVar(e) {
			out = append(out, e)
		}
	}
	return out
}

// connectHTTP creates an HTTP MCP client with fallback from Streamable HTTP to SSE.
func (s *Server) connectHTTP(ctx context.Context, cfg ServerConfig) (*mcpclient.Client, error) {
	if cfg.URL == "" {
		return nil, fmt.Errorf("http transport requires URL for MCP server %s", s.name)
	}

	// Prepare headers option
	var opts []transport.StreamableHTTPCOption
	if len(cfg.Headers) > 0 {
		opts = append(opts, transport.WithHTTPHeaders(cfg.Headers))
	}
	if cfg.HTTPClient != nil {
		opts = append(opts, transport.WithHTTPBasicClient(cfg.HTTPClient))
	}

	// Try Streamable HTTP first
	client, err := mcpclient.NewStreamableHttpClient(cfg.URL, opts...)
	if err == nil {
		if initErr := s.initializeClient(ctx, client); initErr == nil {
			return client, nil
		}
		// Initialization failed, close and try SSE fallback
		if closeErr := client.Close(); closeErr != nil {
			s.log().Debug("failed to close MCP client after connection failure", "error", closeErr)
		}
	}

	// Fallback to SSE
	var sseOpts []transport.ClientOption
	if len(cfg.Headers) > 0 {
		sseOpts = append(sseOpts, transport.WithHeaders(cfg.Headers))
	}
	if cfg.HTTPClient != nil {
		sseOpts = append(sseOpts, transport.WithHTTPClient(cfg.HTTPClient))
	}

	client, err = mcpclient.NewSSEMCPClient(cfg.URL, sseOpts...)
	if err != nil {
		return nil, fmt.Errorf("failed to create HTTP MCP client for %s (tried Streamable HTTP and SSE): %w", s.name, err)
	}

	if err := s.initializeClient(ctx, client); err != nil {
		if closeErr := client.Close(); closeErr != nil {
			s.log().Debug("failed to close MCP client after connection failure", "error", closeErr)
		}
		return nil, err
	}

	return client, nil
}

// initializeClient initializes the MCP connection for the given client.
func (s *Server) initializeClient(ctx context.Context, client *mcpclient.Client) error {
	// Start the client transport
	if err := client.Start(ctx); err != nil {
		return fmt.Errorf("failed to start MCP client for %s: %w", s.name, err)
	}

	// Initialize the MCP connection
	initReq := mcp.InitializeRequest{
		Params: mcp.InitializeParams{
			ProtocolVersion: mcp.LATEST_PROTOCOL_VERSION,
			ClientInfo: mcp.Implementation{
				Name:    "agent",
				Version: "1.0.0",
			},
			Capabilities: mcp.ClientCapabilities{},
		},
	}

	_, err := client.Initialize(ctx, initReq)
	if err != nil {
		return fmt.Errorf("failed to initialize MCP server %s: %w", s.name, err)
	}

	return nil
}

// DiscoverTools calls tools/list on the MCP server and stores the discovered tools.
func (s *Server) DiscoverTools(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.client == nil {
		return fmt.Errorf("mcp server %s is not connected", s.name)
	}

	// List all tools from the MCP server
	result, err := s.client.ListTools(ctx, mcp.ListToolsRequest{})
	if err != nil {
		return fmt.Errorf("failed to list tools from MCP server %s: %w", s.name, err)
	}

	// Convert MCP tools to our internal format
	s.tools = make([]ToolInfo, 0, len(result.Tools))
	for _, tool := range result.Tools {
		// Marshal the input schema to json.RawMessage
		schema, err := json.Marshal(tool.InputSchema)
		if err != nil {
			// Fall back to raw schema if structured marshaling fails
			if tool.RawInputSchema != nil {
				schema = tool.RawInputSchema
			} else {
				schema = []byte(`{"type":"object"}`)
			}
		}

		s.tools = append(s.tools, ToolInfo{
			Name:        tool.Name,
			Description: tool.Description,
			InputSchema: schema,
		})
	}

	toolNames := make([]string, len(s.tools))
	for i, t := range s.tools {
		toolNames[i] = t.Name
	}
	s.log().Debug("MCP tools discovered", "server", s.name, "count", len(s.tools), "tools", toolNames)

	return nil
}

// Tools returns the list of discovered tools.
func (s *Server) Tools() []ToolInfo {
	s.mu.RLock()
	defer s.mu.RUnlock()

	// Return a copy to prevent external modification
	tools := make([]ToolInfo, len(s.tools))
	copy(tools, s.tools)
	return tools
}

// CallTool invokes a tool on the MCP server and returns the result.
func (s *Server) CallTool(ctx context.Context, name string, arguments map[string]any) (*mcp.CallToolResult, error) {
	s.mu.RLock()
	client := s.client
	s.mu.RUnlock()

	if client == nil {
		return nil, fmt.Errorf("mcp server %s is not connected", s.name)
	}

	req := mcp.CallToolRequest{
		Params: mcp.CallToolParams{
			Name:      name,
			Arguments: arguments,
		},
	}

	return client.CallTool(ctx, req)
}

// Close shuts down the MCP server connection.
func (s *Server) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.client == nil {
		return nil
	}

	err := s.client.Close()
	s.client = nil
	s.tools = nil
	if err != nil {
		s.lastError = err.Error()
	}
	return err
}

// IsConnected returns whether the server is currently connected.
func (s *Server) IsConnected() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.client != nil
}

// Status returns the current status of the server.
func (s *Server) Status() ServerStatus {
	s.mu.RLock()
	defer s.mu.RUnlock()

	// Collect tool names
	toolNames := make([]string, len(s.tools))
	for i, tool := range s.tools {
		toolNames[i] = tool.Name
	}

	return ServerStatus{
		Name:      s.name,
		Transport: s.transportType,
		Connected: s.client != nil,
		ToolCount: len(s.tools),
		Tools:     toolNames,
		Error:     s.lastError,
	}
}
