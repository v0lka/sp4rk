package mcp

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"sort"
	"sync"

	sdktools "github.com/v0lka/sp4rk/tools"
)

// Gateway manages connections to multiple MCP servers and provides
// their tools to the agent through the ToolRegistry.
type Gateway struct {
	servers map[string]*Server
	// expandedConfigs holds the env-expanded ServerConfig for each connected
	// server, keyed by name. configChanged compares against this to avoid
	// false positives when raw ${VAR} placeholders in the GatewayConfig
	// entries were already substituted before connection.
	expandedConfigs map[string]ServerConfig
	// failedServers holds the last known failure for each CONFIGURED server
	// that is not in servers (its Connect or DiscoverTools failed). Status()
	// merges these entries so callers always see every configured server,
	// disconnected ones included. An entry is cleared as soon as the server
	// connects successfully or is removed from the configuration. Guarded by
	// mu; deliberately separate from servers to preserve the clean-state
	// invariant (a failed server never enters the connection map).
	failedServers   map[string]ServerStatus
	defaultWorkDir  string
	schemaSanitizer SchemaSanitizer
	logger          *slog.Logger
	mu              sync.RWMutex
}

// newGateway creates a new Gateway instance.
func newGateway() *Gateway {
	return &Gateway{
		servers:         make(map[string]*Server),
		expandedConfigs: make(map[string]ServerConfig),
		failedServers:   make(map[string]ServerStatus),
	}
}

func (g *Gateway) log() *slog.Logger {
	if g.logger != nil {
		return g.logger
	}
	return slog.Default()
}

// SetDefaultWorkDir updates the default working directory for new stdio server connections.
func (g *Gateway) SetDefaultWorkDir(dir string) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.defaultWorkDir = dir
}

// defaultWorkDirForLocked returns dir if non-empty, otherwise g.defaultWorkDir.
// Caller must hold g.mu.
func (g *Gateway) defaultWorkDirForLocked(dir string) string {
	if dir != "" {
		return dir
	}
	return g.defaultWorkDir
}

// Start connects to all configured MCP servers and discovers their tools.
// It returns an error if any server fails to connect, but continues
// connecting to remaining servers.
func (g *Gateway) Start(ctx context.Context, configs map[string]ServerConfig) error {
	g.mu.Lock()
	defer g.mu.Unlock()

	var errs []error

	for name, cfg := range configs {
		// Apply default working directory if not explicitly set
		cfg.WorkDir = g.defaultWorkDirForLocked(cfg.WorkDir)
		server := newServer(name)
		server.logger = g.logger

		if err := server.Connect(ctx, cfg); err != nil {
			errs = append(errs, fmt.Errorf("server %s: %w", name, err))
			g.recordFailureLocked(name, cfg.Transport, err)
			continue
		}

		if err := server.DiscoverTools(ctx); err != nil {
			if closeErr := server.Close(); closeErr != nil {
				errs = append(errs, fmt.Errorf("server %s: close after discovery failure: %w", name, closeErr))
			}
			errs = append(errs, fmt.Errorf("server %s: failed to discover tools: %w", name, err))
			g.recordFailureLocked(name, cfg.Transport, err)
			continue
		}

		g.servers[name] = server
		g.expandedConfigs[name] = cfg // store the expanded config used to connect
		delete(g.failedServers, name)
		g.log().Debug("MCP server started", "server", name, "tools", len(server.Tools()))
	}

	if len(errs) > 0 {
		return &StartError{Errors: errs}
	}

	return nil
}

// RegisterTools registers all discovered MCP tools into the ToolRegistry.
// Each tool is wrapped as a Tool that implements the tools.Tool interface.
// Tools are registered with the server name as their source for proper unregistration.
func (g *Gateway) RegisterTools(registry *sdktools.ToolRegistry) error {
	g.mu.RLock()
	defer g.mu.RUnlock()
	sanitizer := g.schemaSanitizer

	for name, server := range g.servers {
		for _, toolInfo := range server.Tools() {
			mcpTool := NewTool(server, toolInfo, sanitizer)
			if err := registry.RegisterWithSourceCategory(mcpTool, name, sdktools.SourceCategoryMCP); err != nil {
				g.log().Warn("MCP tool registration rejected", "server", name, "tool", mcpTool.Name(), "error", err)
			}
		}
		toolNames := make([]string, len(server.Tools()))
		for i, t := range server.Tools() {
			toolNames[i] = t.Name
		}
		g.log().Debug("MCP tools registered", "server", name, "count", len(toolNames), "tools", toolNames)
	}

	return nil
}

// Stop gracefully shuts down all MCP server connections.
func (g *Gateway) Stop() error {
	g.mu.Lock()
	defer g.mu.Unlock()

	var errs []error

	for name, server := range g.servers {
		if err := server.Close(); err != nil {
			errs = append(errs, fmt.Errorf("server %s: %w", name, err))
		}
	}

	g.servers = make(map[string]*Server)
	g.expandedConfigs = make(map[string]ServerConfig)
	g.failedServers = make(map[string]ServerStatus)

	if len(errs) > 0 {
		return &StopError{Errors: errs}
	}

	return nil
}

// Reconfigure updates the gateway's server connections based on a new configuration.
// It handles added, removed, and changed servers while preserving unchanged connections.
func (g *Gateway) Reconfigure(ctx context.Context, newConfig GatewayConfig,
	registry *sdktools.ToolRegistry, expandEnv func(string) string) error {
	g.mu.Lock()
	defer g.mu.Unlock()

	var errs []error

	// Build sets for comparison
	currentNames := make(map[string]bool, len(g.servers))
	for name := range g.servers {
		currentNames[name] = true
	}

	newNames := make(map[string]bool, len(newConfig.Servers))
	for name := range newConfig.Servers {
		newNames[name] = true
	}

	// Process removed servers
	for name := range currentNames {
		if newNames[name] {
			continue
		}

		server := g.servers[name]

		// Unregister tools from this server
		registry.UnregisterBySource(name)

		// Disconnect server
		if err := server.Close(); err != nil {
			errs = append(errs, fmt.Errorf("server %s: disconnect: %w", name, err))
		}

		delete(g.servers, name)
		delete(g.expandedConfigs, name)
		g.log().Debug("MCP server removed", "server", name)
	}

	// Failed servers are not in g.servers, so the removal pass above never
	// visits them. This pass is the sole owner of failedServers cleanup in
	// Reconfigure: it drops every failure record whose name left the config
	// so Status() stops reporting them. (A failed server that STAYS in the
	// config is retried below: it is absent from currentNames, so the
	// add/change loop always reconnects it.)
	for name := range g.failedServers {
		if newNames[name] {
			continue
		}
		delete(g.failedServers, name)
	}

	// Apply the new default work directory before the add/change loop so that
	// servers relying on an empty WorkDir (which inherits the default) are
	// compared and reconnected against the NEW default. Updating it after the
	// loop would cause such servers to keep running with the stale directory
	// until the next Reconfigure call (one cycle late).
	if newConfig.DefaultWorkDir != "" {
		g.defaultWorkDir = newConfig.DefaultWorkDir
	}

	// Process added and changed servers
	for name, entry := range newConfig.Servers {
		// Build ServerConfig from ServerEntry. WorkDir is resolved against the
		// NEW default (applied above) so servers relying on an empty WorkDir
		// are compared and reconnected against it, not the stale one.
		newCfg := serverConfigFromEntry(entry, g.defaultWorkDirForLocked(entry.WorkDir), newConfig.HTTPClient, expandEnv)

		if currentNames[name] {
			// Server exists - check if config changed
			if !g.configChanged(name, newCfg) {
				continue // Unchanged, keep alive
			}

			// Config changed - disconnect old, reconnect new
			oldServer := g.servers[name]

			// Unregister old tools
			registry.UnregisterBySource(name)

			// Disconnect old server
			if err := oldServer.Close(); err != nil {
				errs = append(errs, fmt.Errorf("server %s: disconnect: %w", name, err))
			}

			// Drop the stale entries immediately so a failed reconnect leaves
			// the gateway in a clean state (no dead server, no stale config).
			// This mirrors the brand-new-server path, where a connect failure
			// leaves nothing behind. Without it, a changed server whose new
			// connection fails would leave a closed server in g.servers and
			// the old config in g.expandedConfigs, so Status() would report a
			// dead server and the next Reconfigure's diffing would operate on
			// stale data. On success the entries are repopulated below.
			delete(g.servers, name)
			delete(g.expandedConfigs, name)

			g.log().Debug("MCP server config changed, reconnecting", "server", name)
		}

		// Connect new server
		server := newServer(name)
		server.logger = g.logger
		if err := server.Connect(ctx, newCfg); err != nil {
			errs = append(errs, fmt.Errorf("server %s: connect: %w", name, err))
			g.recordFailureLocked(name, newCfg.Transport, err)
			continue
		}

		if err := server.DiscoverTools(ctx); err != nil {
			if closeErr := server.Close(); closeErr != nil {
				errs = append(errs, fmt.Errorf("server %s: close after discovery failure: %w", name, closeErr))
			}
			errs = append(errs, fmt.Errorf("server %s: discover tools: %w", name, err))
			g.recordFailureLocked(name, newCfg.Transport, err)
			continue
		}

		g.servers[name] = server
		// Track the expanded config so the next Reconfigure compares against
		// post-expansion values (W-7).
		g.expandedConfigs[name] = newCfg
		delete(g.failedServers, name)

		// Register new tools
		for _, toolInfo := range server.Tools() {
			mcpTool := NewTool(server, toolInfo, g.schemaSanitizer)
			if err := registry.RegisterWithSourceCategory(mcpTool, name, sdktools.SourceCategoryMCP); err != nil {
				g.log().Warn("MCP tool registration rejected", "server", name, "tool", mcpTool.Name(), "error", err)
			}
		}

		if currentNames[name] {
			g.log().Debug("MCP server reconnected", "server", name, "tools", len(server.Tools()))
		} else {
			g.log().Debug("MCP server added", "server", name, "tools", len(server.Tools()))
		}
	}

	if len(errs) > 0 {
		return &ReconfigureError{Errors: errs}
	}

	return nil
}

// configChanged compares the new (env-expanded) config with the previously-
// expanded config for a given server. Comparing against the raw
// GatewayConfig entries (${VAR} placeholders intact) would produce false
// positives every time Reconfigure runs and bounce all servers. The expanded
// copy is persisted in expandedConfigs at the end of Reconfigure for the next
// comparison.
func (g *Gateway) configChanged(name string, newCfg ServerConfig) bool {
	oldCfg, exists := g.expandedConfigs[name]
	if !exists {
		return true // New server
	}

	// Compare relevant fields
	if oldCfg.Transport != newCfg.Transport ||
		oldCfg.Command != newCfg.Command ||
		oldCfg.URL != newCfg.URL ||
		oldCfg.WorkDir != newCfg.WorkDir {
		return true
	}

	// A changed group override re-tags every tool the server registers, so it
	// requires a reconnect (tools are unregistered and re-registered with the
	// new group). Comparing it here keeps Reconfigure from treating an
	// override-only change as a no-op and leaving stale groups in the
	// registry. The comparison uses the EFFECTIVE group (effectiveToolGroup),
	// the same normalization Server.ToolGroup applies when tagging tools: a
	// raw change between values that are both ignored (empty, unknown, or
	// reserved — e.g. fixing one typo'd override into another) does not alter
	// any registered tool's group, so reconnecting the server process for it
	// would be pure churn.
	if effectiveToolGroup(oldCfg.ToolGroupOverride, oldCfg.Transport) !=
		effectiveToolGroup(newCfg.ToolGroupOverride, newCfg.Transport) {
		return true
	}

	// Compare args
	if len(oldCfg.Args) != len(newCfg.Args) {
		return true
	}
	for i, arg := range oldCfg.Args {
		if arg != newCfg.Args[i] {
			return true
		}
	}

	// Compare env (both sides are post-expansion)
	if len(oldCfg.Env) != len(newCfg.Env) {
		return true
	}
	for k, v := range oldCfg.Env {
		if newCfg.Env[k] != v {
			return true
		}
	}

	// Compare headers (both sides are post-expansion)
	if len(oldCfg.Headers) != len(newCfg.Headers) {
		return true
	}
	for k, v := range oldCfg.Headers {
		if newCfg.Headers[k] != v {
			return true
		}
	}

	return false
}

// GetServer returns a specific MCP server by name, or nil if not found.
func (g *Gateway) GetServer(name string) *Server {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return g.servers[name]
}

// ServerNames returns a list of all connected server names.
func (g *Gateway) ServerNames() []string {
	g.mu.RLock()
	defer g.mu.RUnlock()

	names := make([]string, 0, len(g.servers))
	for name := range g.servers {
		names = append(names, name)
	}
	return names
}

// ToolCount returns the total number of tools across all connected servers.
func (g *Gateway) ToolCount() int {
	g.mu.RLock()
	defer g.mu.RUnlock()

	count := 0
	for _, server := range g.servers {
		count += len(server.Tools())
	}
	return count
}

// Status returns the current status of all MCP server connections.
// Configured servers whose connection failed are included as disconnected
// entries (Connected=false, Error set) so callers can render every configured
// server. The result is sorted by server name for deterministic output.
func (g *Gateway) Status() []ServerStatus {
	g.mu.RLock()
	defer g.mu.RUnlock()

	statuses := make([]ServerStatus, 0, len(g.servers)+len(g.failedServers))
	for _, server := range g.servers {
		statuses = append(statuses, server.Status())
	}

	// Surface configured servers that failed to connect. They are kept out of
	// g.servers (clean-state invariant), so their status lives in
	// failedServers. Success clears the entry, so a name present in both maps
	// would indicate drift; the lookup skip guards against that.
	for name, failed := range g.failedServers {
		if _, ok := g.servers[name]; ok {
			continue
		}
		statuses = append(statuses, failed)
	}

	// Sort by name for deterministic output
	sort.Slice(statuses, func(i, j int) bool {
		return statuses[i].Name < statuses[j].Name
	})

	return statuses
}

// recordFailureLocked stores the last failure for a configured server that is
// not in servers, so Status() can report it as disconnected. transport is
// normalized with the same default Server.Connect applies ("" → "stdio").
// Caller must hold g.mu.
func (g *Gateway) recordFailureLocked(name, transport string, err error) {
	if transport == "" {
		transport = "stdio"
	}
	msg := ""
	if err != nil {
		msg = err.Error()
	}
	g.failedServers[name] = ServerStatus{
		Name:      name,
		Transport: transport,
		Connected: false,
		Tools:     []string{},
		Error:     msg,
	}
}

// StartError represents errors that occurred during gateway startup.
type StartError struct {
	Errors []error
}

// Error summarizes the startup failure: the single underlying error, or the
// number of servers that failed to connect when multiple errors occurred.
func (e *StartError) Error() string {
	if len(e.Errors) == 1 {
		return fmt.Sprintf("MCP gateway start error: %v", e.Errors[0])
	}
	return fmt.Sprintf("MCP gateway start errors: %d servers failed to connect", len(e.Errors))
}

// StopError represents errors that occurred during gateway shutdown.
type StopError struct {
	Errors []error
}

// Error summarizes the shutdown failure: the single underlying error, or the
// number of servers that failed to stop cleanly when multiple errors occurred.
func (e *StopError) Error() string {
	if len(e.Errors) == 1 {
		return fmt.Sprintf("MCP gateway stop error: %v", e.Errors[0])
	}
	return fmt.Sprintf("MCP gateway stop errors: %d servers failed to stop cleanly", len(e.Errors))
}

// ReconfigureError represents errors that occurred during gateway reconfiguration.
type ReconfigureError struct {
	Errors []error
}

// Error summarizes the reconfiguration failure: the single underlying error,
// or the number of failed operations when multiple errors occurred.
func (e *ReconfigureError) Error() string {
	if len(e.Errors) == 1 {
		return fmt.Sprintf("MCP gateway reconfigure error: %v", e.Errors[0])
	}
	return fmt.Sprintf("MCP gateway reconfigure errors: %d operations failed", len(e.Errors))
}

// ServerStatus represents the current status of an MCP server connection.
//
// Starting is a transient state distinct from Connected/Error: it marks an entry
// that is still being initialized (e.g. the gateway placeholder while startup is
// in flight). It is non-omitempty (always serializes, like Connected) so the
// frontend can treat it as a first-class state without falsy-coalescing.
type ServerStatus struct {
	Name      string   `json:"name"`
	Transport string   `json:"transport"`
	Connected bool     `json:"connected"`
	Starting  bool     `json:"starting"`
	ToolCount int      `json:"tool_count"`
	Tools     []string `json:"tools"`
	Error     string   `json:"error,omitempty"`
}

// GatewayConfig holds the raw MCP server entries for gateway initialization.
type GatewayConfig struct {
	Servers        map[string]ServerEntry
	DefaultWorkDir string       // fallback working directory for stdio servers
	HTTPClient     *http.Client // optional proxy-configured HTTP client for HTTP transport servers

	// SchemaSanitizer transforms tool input schemas before registration.
	// When set, it is called for every tool registered from MCP servers.
	// Use this to strip source-specific parameters that should not be visible
	// to the LLM. Nil means no sanitization.
	SchemaSanitizer SchemaSanitizer
}

// ServerEntry describes how to launch a single MCP server.
type ServerEntry struct {
	Transport string            // "stdio" | "http"; default "stdio"
	Command   string            // stdio: command to execute
	Args      []string          // stdio: command arguments
	Env       map[string]string // stdio: environment variables (values may contain ${ENV_VAR} references)
	URL       string            // http: server URL (may contain ${ENV_VAR} references)
	Headers   map[string]string // http: custom headers (values may contain ${ENV_VAR} references)
	WorkDir   string            // stdio: working directory for the server process
	// ToolGroupOverride optionally tags every tool served by this server with
	// an explicit capability group instead of the transport-derived default
	// (stdio → local_mcp, http → remote_mcp). It carries the same semantics
	// as ServerConfig.ToolGroupOverride (empty/undeclared/reserved values are
	// ignored and the transport default applies) and is forwarded by
	// StartGateway and Reconfigure, so the gateway-level configuration path
	// can express the override — not only direct mcp.Server construction.
	ToolGroupOverride sdktools.ToolGroup
}

// serverConfigFromEntry builds the runtime ServerConfig for a ServerEntry:
// ${VAR} references in env values, URL, and header values are expanded, and
// the entry's ToolGroupOverride is forwarded so the gateway-level
// configuration path can express the per-server group override. workDir is
// passed in already resolved: StartGateway resolves the default at connect
// time (Gateway.Start), while Reconfigure resolves it eagerly so the expanded
// config stored for diffing matches what actually connects.
func serverConfigFromEntry(entry ServerEntry, workDir string, httpClient *http.Client, expandEnv func(string) string) ServerConfig {
	env := make(map[string]string, len(entry.Env))
	for ek, ev := range entry.Env {
		env[ek] = expandEnv(ev)
	}

	headers := make(map[string]string, len(entry.Headers))
	for hk, hv := range entry.Headers {
		headers[hk] = expandEnv(hv)
	}

	return ServerConfig{
		Transport:  entry.Transport,
		Command:    entry.Command,
		Args:       entry.Args,
		Env:        env,
		URL:        expandEnv(entry.URL),
		Headers:    headers,
		WorkDir:    workDir,
		HTTPClient: httpClient,

		ToolGroupOverride: entry.ToolGroupOverride,
	}
}

// StartGateway creates, configures, starts, and registers an Gateway.
// expandEnv resolves ${VAR} references in environment values.
// Returns (nil, nil) if no servers are configured.
func StartGateway(ctx context.Context, cfg GatewayConfig, registry *sdktools.ToolRegistry, expandEnv func(string) string, logger *slog.Logger) (*Gateway, error) {
	if len(cfg.Servers) == 0 {
		return nil, nil
	}

	mcpConfigs := make(map[string]ServerConfig, len(cfg.Servers))
	for name, entry := range cfg.Servers {
		mcpConfigs[name] = serverConfigFromEntry(entry, entry.WorkDir, cfg.HTTPClient, expandEnv)
	}

	gateway := newGateway()
	gateway.defaultWorkDir = cfg.DefaultWorkDir
	gateway.schemaSanitizer = cfg.SchemaSanitizer
	gateway.logger = logger

	if err := gateway.Start(ctx, mcpConfigs); err != nil {
		if logger != nil {
			logger.Warn("MCP gateway start errors", "error", err)
		}
	}

	if err := gateway.RegisterTools(registry); err != nil {
		if logger != nil {
			logger.Warn("MCP tool registration errors", "error", err)
		}
	}

	if logger != nil {
		logger.Debug("MCP gateway ready", "servers", len(gateway.servers), "total_tools", gateway.ToolCount())
	}

	return gateway, nil
}
