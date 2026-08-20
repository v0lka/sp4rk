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
	"net/url"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"sync"

	mcpclient "github.com/mark3labs/mcp-go/client"
	"github.com/mark3labs/mcp-go/client/transport"
	"github.com/mark3labs/mcp-go/mcp"

	"github.com/v0lka/sp4rk/sysproc"
	sdktools "github.com/v0lka/sp4rk/tools"
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
	// ToolGroupOverride optionally tags every tool served by this server with
	// an explicit capability group instead of the transport-derived default
	// (stdio → local_mcp, http → remote_mcp). Use it when the transport
	// default misrepresents the server's trust boundary (e.g. an http-tunnel
	// to a process on the same machine). An empty, undeclared, or reserved
	// value is ignored and the transport default applies — in particular the
	// reserved "system" group is NEVER honored: hosts treat system tools as
	// trusted orchestration builtins that bypass policy gates, so honoring a
	// system override would exempt an entire untrusted external server from
	// every security check.
	ToolGroupOverride sdktools.ToolGroup
}

// Server represents a connection to an external MCP server process.
type Server struct {
	name              string
	client            *mcpclient.Client
	tools             []ToolInfo
	lastError         string
	transportType     string
	toolGroupOverride sdktools.ToolGroup // per-server group override from ServerConfig; empty = derive from transport
	logger            *slog.Logger
	mu                sync.RWMutex
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

	// Capture the per-server group override up front: it reflects operator
	// intent for this server regardless of how the connection attempt ends.
	s.toolGroupOverride = cfg.ToolGroupOverride
	// Surface an ignored override instead of silently falling back to the
	// transport-derived group: a typo'd or reserved value would otherwise be
	// invisible misconfiguration (the operator believes they re-tagged the
	// server while every tool keeps the old group).
	if cfg.ToolGroupOverride != "" && !isValidToolGroupOverride(cfg.ToolGroupOverride) {
		s.log().Warn("MCP server tool_group override ignored: unknown or reserved group; using the transport-derived group",
			"server", s.name, "override", string(cfg.ToolGroupOverride))
	}

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

// ToolGroup returns the capability group applied to every tool served by this
// server: the operator's per-server override (ServerConfig.ToolGroupOverride)
// when it declares a valid, non-reserved group, otherwise the group derived
// from the active transport (stdio → local_mcp, http → remote_mcp). The
// reserved GroupSystem is ignored like an invalid override — an external MCP
// server's tools are never host-trusted orchestration builtins.
func (s *Server) ToolGroup() sdktools.ToolGroup {
	s.mu.RLock()
	override := s.toolGroupOverride
	transportName := s.transportType
	s.mu.RUnlock()

	return effectiveToolGroup(override, transportName)
}

// isValidToolGroupOverride reports whether g is usable as a per-server tool
// group override: a declared, non-reserved group. Empty ("no override"),
// unknown, and the reserved GroupSystem values are not (GroupSystem would
// exempt an entire untrusted external server from every policy gate).
func isValidToolGroupOverride(g sdktools.ToolGroup) bool {
	return g != "" && g != sdktools.GroupSystem && sdktools.IsValidToolGroup(g)
}

// effectiveToolGroup resolves the effective capability group for an MCP
// server: the override when it is a valid, non-reserved group, otherwise the
// transport-derived default. This is the single normalization for the
// override — Server.ToolGroup applies it at tool-tagging time and
// Gateway.configChanged applies it at config-diff time, so a change that
// does not alter the effective group (e.g. between two ignored values) is
// not treated as a reconnect-worthy change.
func effectiveToolGroup(override sdktools.ToolGroup, transportName string) sdktools.ToolGroup {
	if isValidToolGroupOverride(override) {
		return override
	}
	return sdktools.MCPToolGroup(transportName)
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

	// Always install a custom command factory for two reasons:
	//  1. It hides the spawned process's console window (CREATE_NO_WINDOW on
	//     Windows). The mcp-go transport's default path does not set this flag,
	//     so a console window flashes or stays open for every stdio MCP server
	//     under a GUI-subsystem host application.
	//  2. It keeps the env allowlist effective. cmdEnv is the filtered slice
	//     built above (safeStdioEnv(os.Environ()) + cfg.Env), so the child never
	//     receives the transport's default os.Environ() merge. Do not remove this
	//     factory or let cmd.Env fall back to the parent's full environment —
	//     that would forward host secrets to the MCP child process.
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
// from the host by stdio MCP server processes. These are the vars a process
// needs to locate its own executables (PATH), resolve its home and user
// identity (HOME, USER, SHELL on POSIX; USERPROFILE on Windows), select a
// locale (LANG, LC_*), place temp files (TMPDIR, TEMP/TMP), reach the network
// through a configured proxy (HTTP_PROXY and friends), trust a private CA
// (SSL_CERT_FILE and friends), and activate a Python virtualenv or conda
// environment (VIRTUAL_ENV, CONDA_*). Windows-essential variables (APPDATA,
// LOCALAPPDATA, SystemRoot, ComSpec, PATHEXT) are also allowlisted so
// Node/Python MCP servers can start under a GUI host that was launched
// without a console environment. Everything else must be declared explicitly
// in the server's cfg.Env — LLM API keys and application credentials are
// never forwarded implicitly (ASI04).
//
// Variable names are matched case-insensitively on Windows (environment names
// are case-insensitive there) and case-sensitively elsewhere; the keys below
// use the canonical uppercase spelling. The proxy variables are additionally
// allowlisted in lowercase (http_proxy et al.) because toolchains disagree on
// case: Go's net/http uses the uppercase forms while curl and many shells use
// the lowercase ones.
//
// Proxy variables are forwarded so a stdio MCP server can reach the network
// through a configured proxy, but any credentials embedded in their userinfo
// component (user:password@host) are stripped first — an authenticated proxy
// URL must not leak its password into an untrusted third-party child process
// (ASI04). NO_PROXY is a hostname list and never carries credentials, so it is
// forwarded unchanged. Operators who want stricter isolation can override or
// clear any of these via cfg.Env — explicit cfg.Env entries are applied after
// this filter and win (the same escape hatch as HOME).
var stdioEnvAllowlist = map[string]struct{}{
	// Executable and identity resolution.
	"PATH":        {},
	"HOME":        {},
	"USER":        {},
	"SHELL":       {},
	"USERPROFILE": {},

	// Locale and terminal.
	"LANG": {},
	"TERM": {},

	// Temp directories.
	"TMPDIR": {},
	"TEMP":   {},
	"TMP":    {},

	// Windows essentials.
	"APPDATA":      {},
	"LOCALAPPDATA": {},
	"SYSTEMROOT":   {},
	"COMSPEC":      {},
	"PATHEXT":      {},

	// Network proxy configuration (upper- and lower-case spellings).
	"HTTP_PROXY":  {},
	"HTTPS_PROXY": {},
	"NO_PROXY":    {},
	"ALL_PROXY":   {},
	"http_proxy":  {},
	"https_proxy": {},
	"no_proxy":    {},
	"all_proxy":   {},

	// CA certificate trust anchors (private root CAs from corporate MITM
	// proxies).
	"SSL_CERT_FILE":       {},
	"SSL_CERT_DIR":        {},
	"REQUESTS_CA_BUNDLE":  {},
	"CURL_CA_BUNDLE":      {},
	"NODE_EXTRA_CA_CERTS": {},

	// Python virtualenv activation.
	"VIRTUAL_ENV": {},

	// Conda environment activation.
	"CONDA_PREFIX":          {},
	"CONDA_DEFAULT_ENV":     {},
	"CONDA_SHLVL":           {},
	"CONDA_PREFIX_1":        {},
	"CONDA_PROMPT_MODIFIER": {},
	"CONDA_EXE":             {},
	"CONDA_PYTHON_EXE":      {},
}

// isAllowedStdioEnvVar reports whether the given env var (NAME=value form)
// is on the allowlist (exact match on NAME, or an LC_* locale variable) for
// the current host OS.
func isAllowedStdioEnvVar(entry string) bool {
	return isAllowedStdioEnvVarForOS(entry, runtime.GOOS)
}

// isAllowedStdioEnvVarForOS is the OS-parameterized core of
// isAllowedStdioEnvVar. On Windows, environment variable names are
// case-insensitive, so allowlist membership is decided with a case-folded key;
// elsewhere names are compared exactly.
func isAllowedStdioEnvVarForOS(entry, goos string) bool {
	key, _, ok := strings.Cut(entry, "=")
	if !ok {
		return false
	}
	if stdioEnvKeyAllowed(key, goos) {
		return true
	}
	return strings.HasPrefix(key, "LC_")
}

// stdioEnvKeyAllowed reports whether key is on the allowlist for the given
// host OS.
func stdioEnvKeyAllowed(key, goos string) bool {
	if goos == "windows" {
		_, ok := stdioEnvAllowlist[strings.ToUpper(key)]
		return ok
	}
	_, ok := stdioEnvAllowlist[key]
	return ok
}

// safeStdioEnv filters a raw os.Environ()-style slice down to the allowlisted
// variables only. Explicit server cfg.Env values are applied on top by the
// caller, so they always win and are not subject to this filter.
func safeStdioEnv(raw []string) []string {
	return safeStdioEnvForOS(raw, runtime.GOOS)
}

// safeStdioEnvForOS is the OS-parameterized core of safeStdioEnv.
func safeStdioEnvForOS(raw []string, goos string) []string {
	if len(raw) == 0 {
		return nil
	}
	out := make([]string, 0, len(stdioEnvAllowlist)+4)
	for _, e := range raw {
		if !isAllowedStdioEnvVarForOS(e, goos) {
			continue
		}
		out = append(out, sanitizeProxyEntry(e))
	}
	return out
}

// sanitizeProxyEntry strips embedded credentials from allowlisted proxy
// variables before they are forwarded to a stdio MCP child process. Only the
// credential-capable proxy variables (HTTP_PROXY, HTTPS_PROXY, ALL_PROXY in
// either case) are touched; everything else is returned unchanged.
func sanitizeProxyEntry(entry string) string {
	key, value, ok := strings.Cut(entry, "=")
	if !ok {
		return entry
	}
	if !isCredentialedProxyVar(key) {
		return entry
	}
	stripped := stripUserinfo(value)
	if stripped == value {
		return entry
	}
	return key + "=" + stripped
}

// isCredentialedProxyVar reports whether key names a proxy variable whose
// value may embed credentials in its userinfo component.
func isCredentialedProxyVar(key string) bool {
	switch {
	case strings.EqualFold(key, "HTTP_PROXY"),
		strings.EqualFold(key, "HTTPS_PROXY"),
		strings.EqualFold(key, "ALL_PROXY"):
		return true
	}
	return false
}

// stripUserinfo removes the userinfo component (user:password@) from a proxy
// URL value. Values that carry no userinfo, or that cannot be parsed as a URL,
// are returned unchanged. A scheme-less "user:password@host:port" value, which
// url.Parse misreads as an opaque URL, is handled by the trailing fallback.
func stripUserinfo(value string) string {
	u, err := url.Parse(value)
	if err == nil && u.User != nil {
		u.User = nil
		return u.String()
	}
	if i := strings.LastIndex(value, "@"); i >= 0 && !strings.Contains(value[:i], "/") {
		return value[i+1:]
	}
	return value
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
