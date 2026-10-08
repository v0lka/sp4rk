// Package mcp provides MCP (Model Context Protocol) integration for the agent.
// It manages connections to external MCP servers and exposes their tools through
// the unified Tool interface.
package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	mcpclient "github.com/mark3labs/mcp-go/client"
	"github.com/mark3labs/mcp-go/client/transport"
	"github.com/mark3labs/mcp-go/mcp"

	"github.com/v0lka/sp4rk/sysproc"
	sdktools "github.com/v0lka/sp4rk/tools"
)

// defaultMCPTimeout is the timeout applied to an MCP server that leaves its
// handshake bound unset (Timeout zero or negative). It bounds the
// initialization handshake (initialize + tools/list), not the lifetime of the
// server connection — and, because a per-call bound that is left unset inherits
// the handshake bound, it also caps a single tools/call by default (see
// resolveTimeoutBounds).
//
// It is intentionally fixed SDK policy rather than a tunable knob: a host whose
// servers have unusual latency overrides Timeout / CallTimeout per server entry
// instead of re-tuning a package default.
const defaultMCPTimeout = 60 * time.Second

// unhealthyTimeoutThreshold is the number of consecutive tools/call timeouts
// after which a server is flagged unhealthy. A single slow call is tolerated
// (it may be transient); a sustained streak of back-to-back timeouts indicates
// a server that is persistently slow or unresponsive, which callers should be
// able to distinguish from a healthy server via ServerStatus.Unhealthy. Like
// defaultMCPTimeout it is intentionally fixed SDK policy — the sensitivity is a
// property of the advisory flag, not a per-server setting.
const unhealthyTimeoutThreshold = 3

// ServerConfig defines how to launch an MCP server.
// This is a local copy to avoid importing backend/config.
type ServerConfig struct {
	Transport string            // "stdio" | "http"; default "stdio"
	Command   string            // stdio: command to execute
	Args      []string          // stdio: command arguments
	Env       map[string]string // stdio: environment variables
	URL       string            // http: server URL
	Headers   map[string]string // http: custom headers
	WorkDir   string            // stdio: working directory for the server process
	// HTTPClient optionally supplies a pre-configured HTTP client for HTTP
	// transport servers (typically a proxy setup). It REPLACES the SDK's
	// bounded default wholesale: the SDK bounds the SSE leg's connection-level
	// waits (dial, TLS, response headers) only when HTTPClient is nil (see
	// boundedStreamHTTPClient), so a supplied client must carry its own
	// timeout and transport bounds — a zero-bound client lets an unresponsive
	// endpoint stall Connect and tool calls indefinitely. Prefer
	// transport-level bounds over Client.Timeout: the latter also caps the
	// long-lived SSE event stream itself and would cut live sessions short.
	HTTPClient *http.Client
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

	// Timeout bounds this server's initialization handshake (initialize +
	// tools/list). Zero or negative selects defaultMCPTimeout. For the HTTP
	// transport it applies to each transport attempt — the Streamable HTTP
	// attempt and, if that fails, the SSE fallback — so an unresponsive server
	// can spend up to twice the bound before Connect fails.
	Timeout time.Duration
	// CallTimeout bounds a single tools/call invocation against this server.
	// Zero or negative inherits Timeout (which itself defaults to
	// defaultMCPTimeout), so a per-call wire timeout is always in effect —
	// including for a server that sets neither field, where the 60s default
	// caps calls that were previously unbounded. Raise it (or raise Timeout)
	// for a server whose tools legitimately run longer; there is no way to
	// disable the per-call bound.
	CallTimeout time.Duration
}

// Server represents a connection to an external MCP server process.
type Server struct {
	name              string
	client            *mcpclient.Client
	tools             []ToolInfo
	lastError         string
	transportType     string
	toolGroupOverride sdktools.ToolGroup // per-server group override from ServerConfig; empty = derive from transport
	// timeout bounds the initialization handshake; callTimeout bounds a single
	// tools/call. Both are resolved (never zero) from ServerConfig by
	// resolveTimeoutBounds — the same normalization the gateway diffs against —
	// then only read: a changed bound takes effect on the next Connect, which
	// is why configChanged() treats an effective-bounds change as
	// reconnect-worthy. Guarded by mu.
	timeout     time.Duration
	callTimeout time.Duration
	// consecutiveTimeouts counts tools/call invocations that hit callTimeout
	// back to back. Only a successful call resets it; a failure that is not a
	// timeout (a transport/protocol error, or a caller cancellation) neither
	// extends nor clears the streak, so it still reflects genuinely
	// back-to-back timeouts. It lets a caller detect a server that is
	// persistently slow/unresponsive (as opposed to a one-off slow call).
	// Guarded by mu.
	consecutiveTimeouts int
	// unhealthy is set once consecutiveTimeouts reaches
	// unhealthyTimeoutThreshold; it is surfaced through ServerStatus.Unhealthy
	// so callers can deprioritize or warn about a persistently unresponsive
	// server. A subsequent successful call clears it, as do Connect and Close
	// (a closed connection can no longer be slow). Guarded by mu.
	unhealthy bool
	// stdioCmd is the child process handle captured by the stdio command
	// factory during connectStdio (nil for HTTP transports, and cleared once a
	// close has consumed it). Close uses it as the last-resort kill switch for
	// a server that ignores stdin EOF instead of waiting for it indefinitely.
	// Guarded by mu (written during Connect, read during Close — both hold the
	// lock).
	stdioCmd *exec.Cmd
	// closeGrace bounds how long Close waits for a server to exit voluntarily
	// after its stdin is closed before the child is killed. Zero means
	// defaultServerCloseGrace. Test hook — production code leaves it zero.
	// Guarded by mu.
	closeGrace time.Duration
	// inflight counts CallTool invocations that have captured a non-nil client
	// and have not returned yet. Close drains it before closing the transport:
	// the SSE transport's reader goroutine fetches a call's response channel
	// and only then sends on it, while Close closes those same channels — a
	// send on a closed channel panics an unrecoverable goroutine and kills the
	// host. Draining guarantees no response channel is still registered when
	// the transport closes. Add runs under mu.RLock and Wait after mu has been
	// released, so they never overlap (WaitGroup contract); Close resets
	// s.client to nil under mu.Lock before waiting, so no new call can join.
	inflight sync.WaitGroup
	logger   *slog.Logger
	mu       sync.RWMutex
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

	// Resolve the effective bounds once, at connect time, and make them the
	// single source of truth the gateway diffs against — see
	// resolveTimeoutBounds for the one normalization both paths share.
	s.timeout, s.callTimeout = resolveTimeoutBounds(cfg)
	s.consecutiveTimeouts = 0
	s.unhealthy = false

	// Determine transport type (default to stdio when unspecified)
	transportType := cfg.Transport
	if transportType == "" {
		transportType = "stdio"
	}

	// A stdio server without a command cannot be spawned — and mcp-go's stdio
	// transport does not reject the empty command itself: its spawnCommand
	// treats it as "nothing to spawn", skips creating the stdout pipe, and its
	// reader goroutine then panics on the nil reader, killing the whole host.
	// Fail closed here instead (the gateway records the failure and keeps
	// serving the remaining servers).
	if transportType == "stdio" && cfg.Command == "" {
		s.lastError = "stdio transport requires a non-empty command"
		return fmt.Errorf("MCP server %s: stdio transport requires a non-empty command", s.name)
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

// resolveTimeoutBounds normalizes a server's configured timeout bounds into the
// effective bounds Connect captures and applies: a non-positive handshake
// timeout (Timeout) falls back to defaultMCPTimeout, and a non-positive
// per-call timeout (CallTimeout) inherits the resolved handshake bound, so
// neither resolved bound is ever zero. It is the single normalization for the
// bounds — Server.Connect applies it when capturing the live bounds and
// Gateway.configChanged applies it when diffing configs — so an edit that does
// not alter the effective bounds (e.g. Timeout flipping between two
// non-positive sentinels, or a CallTimeout that still resolves to the same
// handshake bound) is not treated as reconnect-worthy.
func resolveTimeoutBounds(cfg ServerConfig) (handshake, call time.Duration) {
	handshake = cfg.Timeout
	if handshake <= 0 {
		handshake = defaultMCPTimeout
	}
	call = cfg.CallTimeout
	if call <= 0 {
		call = handshake
	}
	return handshake, call
}

// TimeoutError reports that an MCP operation exceeded the timeout configured
// for its server. It is a typed error (rather than a bare context error) so
// callers can attribute the timeout to a specific server/operation/tool,
// distinguish a slow server from a caller cancellation, and inspect the bound
// that elapsed. Unwrap reports context.DeadlineExceeded, so
// errors.Is(err, context.DeadlineExceeded) holds for every TimeoutError.
type TimeoutError struct {
	Server  string        // MCP server name the operation ran against
	Op      string        // operation: "initialize" | "list_tools" | "call_tool"
	Tool    string        // tool name for Op == "call_tool"; empty otherwise
	Timeout time.Duration // the bound that elapsed
}

// Error renders the timeout with its attribution. The tool name is included
// only for per-tool operations.
func (e *TimeoutError) Error() string {
	if e.Tool != "" {
		return fmt.Sprintf("MCP server %s: %s %s timed out after %s", e.Server, e.Op, e.Tool, e.Timeout)
	}
	return fmt.Sprintf("MCP server %s: %s timed out after %s", e.Server, e.Op, e.Timeout)
}

// Unwrap exposes context.DeadlineExceeded as the cause, so a TimeoutError
// satisfies errors.Is(err, context.DeadlineExceeded) and interoperates with
// callers that already branch on the context sentinel.
func (e *TimeoutError) Unwrap() error { return context.DeadlineExceeded }

// timeoutErrorFor classifies a failed call whose timeout was applied by
// deriving child from base via context.WithTimeoutCause. Alongside the error to
// surface it reports whether that error is OUR *TimeoutError (a genuine server
// timeout) as opposed to the caller's context error, so a caller can act on a
// genuine timeout without re-inspecting the context. The error is the last
// result, as the error-return convention requires.
//
// It yields isTimeout=true only when OUR timer fired — the child deadline
// elapsed while the caller's context is still live. When the caller's context
// is already done it yields that context's error (context.Canceled for a
// cancel, context.DeadlineExceeded for the caller's own deadline) with
// isTimeout=false, so a caller cancellation is never misattributed to the
// server. When neither context is done it yields (false, nil) — the call failed
// for some other reason, and the caller should surface the underlying error
// unchanged.
//
// base is the caller-supplied context; child is the context actually handed to
// the SDK call; fallback is a pre-built attribution returned when the child
// deadline elapsed but its cause is not a recoverable *TimeoutError. The cause
// is recovered with errors.As rather than a bare type assertion, so a wrapped
// cause is still recognised; base.Err() is checked first, so a genuine caller
// cancellation short-circuits before the cause is consulted.
func timeoutErrorFor(base, child context.Context, fallback *TimeoutError) (bool, error) {
	if err := base.Err(); err != nil {
		// The caller stopped us (cancel or its own deadline): surface the
		// parent's error, never a TimeoutError blamed on the server.
		return false, err
	}
	if child.Err() == context.DeadlineExceeded {
		var cause *TimeoutError
		if errors.As(context.Cause(child), &cause) {
			return true, cause
		}
		return true, fallback
	}
	return false, nil
}

// connectStdio creates a stdio MCP client.
func (s *Server) connectStdio(ctx context.Context, cfg ServerConfig) (*mcpclient.Client, error) {
	// Defense in depth for callers that reach the stdio transport without
	// passing through Connect's transport switch: an empty command must never
	// reach mcp-go, whose spawnCommand skips the stdout pipe for it and whose
	// reader goroutine then panics on the nil reader (finding 44).
	if cfg.Command == "" {
		return nil, fmt.Errorf("MCP server %s: stdio transport requires a non-empty command", s.name)
	}

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
				// Place the child in its own process group so the whole tree
				// it spawns can be reaped on close (see closeClientBounded):
				// `command` is frequently a launcher (npx, pnpm dlx, bunx)
				// whose real server is a grandchild, so killing just the
				// direct child would orphan it.
				sysproc.SetProcessGroup(cmd)
				// Record the child handle: Close needs it to kill a server
				// that ignores stdin EOF instead of waiting for it forever
				// (see closeClientBounded).
				s.stdioCmd = cmd
				return cmd, nil
			},
		),
	}

	// Create stdio MCP client
	client, err := mcpclient.NewStdioMCPClientWithOptions(cfg.Command, env, cfg.Args, opts...)
	if err != nil {
		return nil, fmt.Errorf("failed to create stdio MCP client for %s: %w", s.name, err)
	}

	// Drain the child's stderr for the lifetime of the connection. mcp-go
	// creates the stderr pipe but never reads it (its reader goroutine reads
	// stdout only, and Close merely closes the pipe); a server that logs more
	// than the OS pipe buffer (~64 KiB on Linux) to the MCP-sanctioned logging
	// channel then blocks in write(2) and stops servicing the JSON-RPC
	// protocol — wedging the handshake or every subsequent tools/call. The
	// drain must start before the handshake: a server that floods stderr
	// during startup would otherwise stall Connect itself (finding 58).
	s.drainStdioStderr(client)

	if err := s.initializeClient(ctx, client); err != nil {
		// The child was already spawned by Start, so the handshake-failure
		// cleanup needs the same bounded close as a live connection: a server
		// that ignores stdin EOF must not stall the connect-failure path
		// either.
		if closeErr := s.closeClientLocked(client); closeErr != nil {
			s.log().Debug("failed to close MCP client after connection failure", "error", closeErr)
		}
		return nil, err
	}

	return client, nil
}

// stderrRingBytes bounds how much of a stdio server's drained stderr is kept
// for diagnostics. The drain itself is unbounded (to io.Discard semantics) —
// only the retained tail is capped, so a chatty server cannot grow memory.
const stderrRingBytes = 8 * 1024

// stderrRing is a write-only bounded buffer that keeps the LAST stderrRingBytes
// bytes written to it: older output is dropped, so the retained tail always
// reflects a server's most recent log lines. It satisfies io.Writer.
type stderrRing struct {
	buf []byte
}

// Write appends p to the ring, discarding the oldest bytes when the cap is
// exceeded. It never fails and always reports the full write, so io.Copy's
// drain is never cut short.
func (r *stderrRing) Write(p []byte) (int, error) {
	if len(p) >= stderrRingBytes {
		r.buf = append(r.buf[:0], p[len(p)-stderrRingBytes:]...)
		return len(p), nil
	}
	if len(r.buf)+len(p) > stderrRingBytes {
		keep := len(r.buf) + len(p) - stderrRingBytes
		r.buf = append(r.buf[:0], r.buf[keep:]...)
	}
	r.buf = append(r.buf, p...)
	return len(p), nil
}

// drainStdioStderr starts a goroutine that drains the stdio child's stderr
// pipe for the lifetime of the connection, teeing it into a bounded tail
// buffer that is logged at Debug level when the drain ends (child exit or
// connection close). It must be called as soon as the client exists and
// BEFORE the handshake: a server that floods stderr during startup wedges
// Connect itself when nobody reads the pipe. The goroutine always terminates:
// the child's exit (or Close closing the pipe read end) ends the Copy with
// EOF or a closed-pipe error. This cannot be done inside the command factory
// — setting cmd.Stderr there makes mcp-go's own StderrPipe() call fail.
func (s *Server) drainStdioStderr(client *mcpclient.Client) {
	stderr, ok := mcpclient.GetStderr(client)
	if !ok || stderr == nil {
		return
	}
	go func() {
		ring := &stderrRing{buf: make([]byte, 0, stderrRingBytes)}
		_, _ = io.Copy(ring, stderr)
		if len(ring.buf) > 0 {
			s.log().Debug("MCP stdio server stderr (last bytes)",
				"server", s.name, "stderr", string(ring.buf))
		}
	}()
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

	// The handshake bound (s.timeout) applies to EACH transport attempt below:
	// the Streamable HTTP attempt and, if it fails, the SSE fallback. On the
	// Streamable HTTP leg the transport's requests carry the bounded contexts
	// handed to Initialize/CallTool, so a stalled server is cut off by the
	// handshake bound itself. The SSE leg blocks in TWO waits, each bounded on
	// its own: the wait for the response headers is bounded by the transport
	// HTTP client installed below (when the host supplies none — a
	// host-supplied HTTPClient replaces it and must carry its own bounds), and
	// the wait for the SSE endpoint event is bounded by the Start race in
	// initializeClientWithStartCtx. A stalled endpoint therefore costs a small
	// multiple of the bound, never an unbounded transport-internal wait. Only
	// the Initialize exchange is bounded by the handshake context — the SSE
	// transport's Start deliberately runs on a detached context (see the
	// fallback leg below).

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
	httpClient := cfg.HTTPClient
	if httpClient == nil {
		// The SSE transport's built-in client has NO timeouts at all, and its
		// Start performs the blocking event-stream GET itself with whatever
		// context it is handed — with the default client and a context without
		// a deadline (the framework path), a server that accepts the TCP
		// connection but never sends response headers would hold Connect
		// indefinitely (finding 40). Install a client whose dial, TLS, and
		// response-header waits are bounded by the handshake bound instead;
		// Client.Timeout stays zero because the connected event stream is
		// long-lived and must not be cut mid-session.
		httpClient = boundedStreamHTTPClient(s.timeout)
	}
	sseOpts = append(sseOpts, transport.WithHTTPClient(httpClient))

	client, err = mcpclient.NewSSEMCPClient(cfg.URL, sseOpts...)
	if err != nil {
		return nil, fmt.Errorf("failed to create HTTP MCP client for %s (tried Streamable HTTP and SSE): %w", s.name, err)
	}

	// The SSE transport retains the Start context as the LIFETIME of its event
	// stream: the blocking GET runs on a context derived from it, and only
	// Close cancels it. Handing the caller's context to Start therefore ties a
	// long-lived connection to whatever scope happened to call
	// StartGateway/Reconfigure — a host's `ctx, cancel := context.WithTimeout(...);
	// defer cancel()` would silently kill the live server the moment that
	// scope exits (finding 41). Start on a detached context instead (values
	// are preserved; cancellation and deadlines are dropped), and keep the
	// Initialize exchange below bounded by the caller's context and the
	// handshake timeout as before. The detached stream is still torn down by
	// Server.Close → client.Close, which cancels it explicitly.
	startCtx := context.WithoutCancel(ctx)
	if err := s.initializeClientWithStartCtx(ctx, startCtx, client); err != nil {
		if closeErr := client.Close(); closeErr != nil {
			s.log().Debug("failed to close MCP client after connection failure", "error", closeErr)
		}
		return nil, err
	}

	return client, nil
}

// boundedStreamHTTPClient builds the HTTP client used for the SSE fallback
// transport when the host supplies none: connection-level waits are bounded so
// a stalled endpoint cannot hold Connect open indefinitely, while
// Client.Timeout remains zero — the SSE event stream is a long-lived response
// body that must stay open for the lifetime of the connection
// (ResponseHeaderTimeout bounds only the wait for headers, not the body).
func boundedStreamHTTPClient(bound time.Duration) *http.Client {
	return &http.Client{
		Transport: &http.Transport{
			Proxy:                 http.ProxyFromEnvironment,
			DialContext:           (&net.Dialer{Timeout: bound}).DialContext,
			ForceAttemptHTTP2:     true,
			TLSHandshakeTimeout:   bound,
			ResponseHeaderTimeout: bound,
		},
	}
}

// initializeClient initializes the MCP connection for the given client.
//
// Caller must hold s.mu (Connect holds it), so reading the resolved s.timeout
// without an additional lock is safe.
func (s *Server) initializeClient(ctx context.Context, client *mcpclient.Client) error {
	return s.initializeClientWithStartCtx(ctx, ctx, client)
}

// initializeClientWithStartCtx starts the client transport on startCtx and
// runs the MCP initialize exchange on ctx. The split exists because transports
// disagree about what a Start context means: stdio pre-starts its child with
// context.Background() in the constructor (Start is a no-op), Streamable HTTP
// does not retain it (Start is a no-op without continuous listening), but the
// SSE transport keeps it as the lifetime of its event stream — so the SSE leg
// passes a detached context as startCtx while ctx continues to bound the
// handshake. The Start call itself is raced against the resolved handshake
// bound (see below): the SSE transport's Start blocks until the server's
// endpoint event arrives, under a fixed 30s internal timeout that neither the
// detached start context nor the bounded transport HTTP client can tighten.
//
// Caller must hold s.mu (Connect holds it), so reading the resolved s.timeout
// without an additional lock is safe.
func (s *Server) initializeClientWithStartCtx(ctx, startCtx context.Context, client *mcpclient.Client) error {
	// Start the client transport. Start's CONTEXT is DELIBERATELY NOT derived
	// from the handshake timeout: for the stdio transport the process is
	// already started with context.Background() by the transport constructor,
	// and Start only attaches to it — deriving Start's context from a timeout
	// would couple the child process's lifetime to the handshake deadline, so
	// a server that is merely slow to finish the MCP handshake would have its
	// process killed (and the connection torn down) instead of the handshake
	// being aborted and retried.
	//
	// The Start CALL is instead raced against the resolved handshake bound.
	// For stdio and Streamable HTTP the race is inert — Start returns as soon
	// as the transport is up. The SSE transport's Start, however, BLOCKS until
	// the server's endpoint event arrives, under a fixed 30s internal timeout
	// (mcp-go transport/sse.go): the detached start context carries no
	// deadline to tighten it, and the bounded transport HTTP client installed
	// by connectHTTP covers only the wait for response headers, so an endpoint
	// that sends headers and then stalls would otherwise hold Connect — under
	// the gateway's write lock — for ~30s regardless of the configured
	// Timeout. When the bound fires, the race context is cancelled — which
	// makes the SSE endpoint wait return through its own ctx.Done() arm — and
	// the Start goroutine is reaped BEFORE returning, so the error path's
	// client.Close (Connect's callers close on every failure) can never
	// overlap a still-running Start: mcp-go's transports do not synchronize a
	// concurrent Start against Close (the SSE transport assigns
	// cancelSSEStream racily), and Connect then fails with a TimeoutError
	// inside the configured bound.
	raceCtx, raceCancel := context.WithCancel(startCtx)
	// startCompleted records that Start returned successfully, so the
	// deferred cancel below becomes a no-op on the success path: the SSE
	// transport retains raceCtx as the parent of its event stream, and
	// cancelling it would kill the live connection — the exact lifetime
	// coupling finding 41 removed. On every failure path the cancel still
	// fires, so no context is ever leaked.
	startCompleted := false
	defer func() {
		if !startCompleted {
			raceCancel()
		}
	}()
	startDone := make(chan error, 1) // buffered: the timer branch reaps the goroutine without deadlocking on its send
	go func() {
		startDone <- client.Start(raceCtx)
	}()

	startTimer := time.NewTimer(s.timeout)
	defer startTimer.Stop()

	select {
	case err := <-startDone:
		if err != nil {
			return fmt.Errorf("failed to start MCP client for %s: %w", s.name, err)
		}
		startCompleted = true
	case <-ctx.Done():
		// The host cancelled (or deadline-expired) the Connect context while
		// the endpoint wait was stalled: abort promptly and attribute the
		// failure to the caller instead of misreporting it as a server
		// timeout — the same attribution rule timeoutErrorFor follows.
		// Cancel and reap exactly like the timer branch below, so the error
		// path's client.Close can never overlap a still-running Start. If
		// Start completes concurrently with the cancellation, the race
		// context is cancelled first, so the just-started stream is dead on
		// arrival and Connect's caller closes it on this failure path.
		raceCancel()
		<-startDone
		return ctx.Err()
	case <-startTimer.C:
		// Cancel first (unblocks a stalled Start in milliseconds through the
		// transport's own context handling), then reap the goroutine so the
		// error path's client.Close — Connect's callers close on every
		// failure — can never overlap a still-running Start: mcp-go's
		// transports do not synchronize a concurrent Start against Close (the
		// SSE transport assigns cancelSSEStream racily). Start is guaranteed
		// to observe the cancellation — its endpoint wait selects on
		// ctx.Done() and its HTTP request carries the context — so the reap
		// is prompt; the receive is intentionally unbounded, because a bound
		// would re-open the Start/Close overlap this branch exists to avoid.
		// Tie-break: if Start completes right at the tick, the just-started
		// stream is cancelled by raceCancel and the TimeoutError is still
		// reported — the microscopic window favors honoring the advertised
		// bound; the caller's error path closes the transport and may retry.
		raceCancel()
		<-startDone
		return &TimeoutError{Server: s.name, Op: "initialize", Timeout: s.timeout}
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

	// Bound the handshake exchange by the resolved handshake timeout. The
	// timeout is attached as a context cause (the *TimeoutError we want the
	// caller to see), so we can tell OUR timer firing apart from the caller
	// cancelling ctx — see timeoutErrorFor.
	initCtx := ctx
	if s.timeout > 0 {
		var cancel context.CancelFunc
		initCtx, cancel = context.WithTimeoutCause(ctx, s.timeout,
			&TimeoutError{Server: s.name, Op: "initialize", Timeout: s.timeout})
		defer cancel()
	}

	if _, err := client.Initialize(initCtx, initReq); err != nil {
		if _, timeoutErr := timeoutErrorFor(ctx, initCtx,
			&TimeoutError{Server: s.name, Op: "initialize", Timeout: s.timeout}); timeoutErr != nil {
			return timeoutErr
		}
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

	// Bound tools/list by the resolved handshake timeout. Discovery is part of
	// the connection handshake: a server that answers initialize but then hangs
	// on tools/list must not stall startup (or a reconnect) indefinitely.
	listCtx := ctx
	if s.timeout > 0 {
		var cancel context.CancelFunc
		listCtx, cancel = context.WithTimeoutCause(ctx, s.timeout,
			&TimeoutError{Server: s.name, Op: "list_tools", Timeout: s.timeout})
		defer cancel()
	}

	// Page through tools/list at the transport level and keep every tool's
	// inputSchema as the server sent it. Decoding into mcp.Tool instead would
	// silently drop every top-level schema keyword its ToolInputSchema struct
	// does not model (enum, oneOf, $ref, ... — the struct's marshaller emits
	// only type/$defs/properties/required/additionalProperties, and the
	// RawInputSchema escape hatch is json:"-" with no UnmarshalJSON, so it is
	// always nil on the client decode path): a tool advertised with a top-level
	// enum or $ref would reach the LLM parameterless and unusable (finding 51).
	tools, err := s.listToolsRaw(listCtx)
	if err != nil {
		if _, timeoutErr := timeoutErrorFor(ctx, listCtx,
			&TimeoutError{Server: s.name, Op: "list_tools", Timeout: s.timeout}); timeoutErr != nil {
			return timeoutErr
		}
		return fmt.Errorf("failed to list tools from MCP server %s: %w", s.name, err)
	}

	s.tools = tools

	toolNames := make([]string, len(s.tools))
	for i, t := range s.tools {
		toolNames[i] = t.Name
	}
	s.log().Debug("MCP tools discovered", "server", s.name, "count", len(s.tools), "tools", toolNames)

	return nil
}

// mcpRawRequestID generates request IDs for DiscoverTools' transport-level
// tools/list paging. The mcp-go client numbers its requests sequentially from
// 1 and routes responses by echoed ID through a shared map, so reusing small
// IDs could hijack the response of a concurrent in-flight call on the same
// connection (Server.CallTool drops the server lock before its wire call).
// Seeding the counter far above any sequential run keeps the two ID spaces
// disjoint. The base must stay under 2^53: mcp-go's RequestId re-parses
// response IDs through float64, so larger IDs lose precision on the way back
// and the response would no longer match the map key. DiscoverTools runs once
// per (re)connect under the server write lock, so its own calls never collide
// with each other.
const mcpRawRequestIDBase = int64(1) << 52

var mcpRawRequestID atomic.Int64

// rawTool is the decode target for one entry of a tools/list page. InputSchema
// is captured as raw bytes and stored verbatim — see listToolsRaw.
type rawTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"inputSchema"`
}

// rawListToolsPage is the decode target for one page of a tools/list response.
type rawListToolsPage struct {
	Tools      []rawTool  `json:"tools"`
	NextCursor mcp.Cursor `json:"nextCursor,omitempty"`
}

// listToolsRaw pages through tools/list on the server connection and returns
// the discovered tools with their input schemas preserved byte-for-byte. It
// mirrors what mcp-go's own paginated helper does on the wire (JSON-RPC
// request with a params.cursor, JSON-RPC error surfaced as a Go error) while
// decoding each page into rawTool so no schema keyword is lost.
//
// Caller must hold s.mu (DiscoverTools holds it): the client and the transport
// are read without an additional lock, same as the CallTool path reads the
// captured client.
func (s *Server) listToolsRaw(ctx context.Context) ([]ToolInfo, error) {
	tr := s.client.GetTransport()
	if tr == nil {
		return nil, fmt.Errorf("mcp server %s: client has no transport", s.name)
	}

	var tools []ToolInfo
	cursor := mcp.Cursor("")
	for {
		response, err := tr.SendRequest(ctx, transport.JSONRPCRequest{
			JSONRPC: mcp.JSONRPC_VERSION,
			ID:      mcp.NewRequestId(mcpRawRequestIDBase + mcpRawRequestID.Add(1)),
			Method:  "tools/list",
			Params:  mcp.PaginatedParams{Cursor: cursor},
		})
		if err != nil {
			return nil, err
		}
		if response.Error != nil {
			return nil, response.Error.AsError()
		}

		var page rawListToolsPage
		if err := json.Unmarshal(response.Result, &page); err != nil {
			return nil, fmt.Errorf("failed to unmarshal tools/list response: %w", err)
		}

		for _, tool := range page.Tools {
			schema := tool.InputSchema
			if len(schema) == 0 {
				// A server may omit inputSchema entirely (an MCP tool with no
				// arguments); advertise an empty object schema rather than nil.
				schema = json.RawMessage(`{"type":"object"}`)
			}
			tools = append(tools, ToolInfo{
				Name:        tool.Name,
				Description: tool.Description,
				InputSchema: schema,
			})
		}

		if page.NextCursor == "" {
			return tools, nil
		}
		cursor = page.NextCursor
	}
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
	callTimeout := s.callTimeout
	if client != nil {
		// Register the in-flight call while still holding the lock, so a
		// concurrent Close cannot reach client.Close() while this call's
		// response channel is registered in the transport: the SSE transport
		// closes those channels in Close and its reader goroutine sends on
		// them after dropping its lock, so closing under an in-flight call can
		// panic that goroutine with "send on closed channel" and kill the host
		// (finding 64). Close releases s.mu before waiting, and resets
		// s.client under the lock, so no call can join after the drain starts.
		s.inflight.Add(1)
	}
	s.mu.RUnlock()

	if client == nil {
		return nil, fmt.Errorf("mcp server %s is not connected", s.name)
	}
	defer s.inflight.Done()

	req := mcp.CallToolRequest{
		Params: mcp.CallToolParams{
			Name:      name,
			Arguments: arguments,
		},
	}

	// Bound the call by the resolved per-call timeout. The *TimeoutError is
	// attached as the deadline cause so timeoutErrorFor can tell OUR timer
	// firing apart from the caller cancelling ctx.
	callCtx := ctx
	if callTimeout > 0 {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeoutCause(ctx, callTimeout,
			&TimeoutError{Server: s.name, Op: "call_tool", Tool: name, Timeout: callTimeout})
		defer cancel()
	}

	result, err := client.CallTool(callCtx, req)
	if err != nil {
		isTimeout, timeoutErr := timeoutErrorFor(ctx, callCtx,
			&TimeoutError{Server: s.name, Op: "call_tool", Tool: name, Timeout: callTimeout})
		if timeoutErr != nil {
			// Count only genuine timeouts toward the back-to-back streak; a
			// caller cancellation surfaces the parent error but is not the
			// server's fault.
			if isTimeout {
				s.mu.Lock()
				s.consecutiveTimeouts++
				if s.consecutiveTimeouts >= unhealthyTimeoutThreshold {
					s.unhealthy = true
					s.lastError = fmt.Sprintf("%d consecutive tool-call timeouts (last timeout: %s)",
						s.consecutiveTimeouts, s.callTimeout)
				}
				s.mu.Unlock()
			}
			return nil, timeoutErr
		}
		return nil, err
	}

	// A clean return clears the back-to-back timeout streak and any unhealthy
	// mark: a call that returns in time proves the server is responsive again,
	// so the stale lastError from the timeout streak is dropped too.
	s.mu.Lock()
	s.consecutiveTimeouts = 0
	s.unhealthy = false
	s.lastError = ""
	s.mu.Unlock()

	return result, nil
}

// defaultServerCloseGrace bounds how long Close waits for a server to exit
// voluntarily after its stdin is closed before the child is killed. Freshly
// connected stdio servers exit in milliseconds; a server draining in-flight
// work can take many seconds (observed in the field: ~10s), which is exactly
// the stall this bound exists to cap.
const defaultServerCloseGrace = 2 * time.Second

// effectiveCloseGrace resolves the close grace period. Callers hold mu.
func (s *Server) effectiveCloseGrace() time.Duration {
	if s.closeGrace > 0 {
		return s.closeGrace
	}
	return defaultServerCloseGrace
}

// Close shuts down the MCP server connection.
//
// The underlying stdio close closes the child's stdin and then blocks in
// cmd.Wait until the process exits — without a bound of its own. A server
// that ignores stdin EOF (draining in-flight work, or wedged) would stall
// whatever goroutine closes it — during app shutdown, the main thread — for
// as long as it pleases. Close therefore waits at most closeGrace for a
// voluntary exit and then kills the child: stdin EOF already delivered the
// polite shutdown request, the kill only reaps what refused it.
//
// In-flight tool calls are drained first (bounded by their own per-call
// bound, or closeGrace for a hand-built server that never connected): closing
// the transport while a call is still registered can panic the SSE reader
// goroutine with "send on closed channel" (finding 64), and a drained call
// simply completes instead of being killed mid-flight.
func (s *Server) Close() error {
	// Phase 1 (under the lock): detach the client and reset state. New calls
	// observe client == nil from here on; calls that already captured the
	// client are tracked by the inflight counter.
	s.mu.Lock()
	if s.client == nil {
		s.mu.Unlock()
		return nil
	}

	client := s.client
	cmd := s.stdioCmd
	grace := s.effectiveCloseGrace()
	drainBound := s.callTimeout
	s.client = nil
	s.stdioCmd = nil
	s.tools = nil
	// A closed connection can no longer be slow: drop the advisory unhealthy
	// mark together with the timeout streak and its recorded description
	// (lastError), so Status() does not keep advertising Unhealthy: true — or a
	// stale timeout error — for a server that is simply disconnected.
	s.consecutiveTimeouts = 0
	s.unhealthy = false
	s.lastError = ""
	s.mu.Unlock()

	// Phase 2 (without the lock): drain in-flight calls so the transport is
	// closed with no response channel still registered. The wait is bounded by
	// the calls' own per-call bound — each is wrapped in a call-context
	// deadline — falling back to the close grace for a server whose bounds
	// were never resolved. The lock must not be held while waiting: a drained
	// call takes it to record its outcome.
	if drainBound <= 0 {
		drainBound = grace
	}
	s.drainInflight(drainBound)

	err := closeClientBounded(client, cmd, grace)
	s.mu.Lock()
	if err != nil {
		s.lastError = err.Error()
	}
	s.mu.Unlock()
	return err
}

// drainInflight waits for in-flight CallTool invocations to return, bounded by
// bound. Calls whose bound elapses are allowed to keep running: closing under
// them re-opens the finding 64 race, but an unbounded wait would let a single
// wedged call stall shutdown forever — the residual risk is limited to a call
// that ignores its own context deadline, which no connected server can
// produce (CallTool always derives one from the per-call bound).
func (s *Server) drainInflight(bound time.Duration) {
	done := make(chan struct{})
	go func() {
		s.inflight.Wait()
		close(done)
	}()
	timer := time.NewTimer(bound)
	defer timer.Stop()
	select {
	case <-done:
	case <-timer.C:
		s.log().Debug("MCP server close proceeding with tool calls still in flight",
			"server", s.name, "waited", bound)
	}
}

// closeClientLocked closes the transport client with the bounded-close
// semantics documented on Close, consuming the recorded stdio child handle.
// Callers hold mu.
func (s *Server) closeClientLocked(client *mcpclient.Client) error {
	cmd := s.stdioCmd
	s.stdioCmd = nil
	return closeClientBounded(client, cmd, s.effectiveCloseGrace())
}

// closeClientBounded closes the mcp-go client, killing the recorded stdio
// child if it has not exited voluntarily within grace. The close runs in its
// own goroutine because mcp-go's stdio Close blocks in cmd.Wait without a
// deadline. After a kill that wait unblocks promptly, so the goroutine is
// always drained and the child is reaped. Without a child handle (HTTP
// transports) the close result is abandoned to the goroutine on timeout — the
// Server is discarded by the caller either way, and the HTTP transports carry
// network timeouts of their own.
//
// The kill reaps the whole process tree (sysproc.KillTree, which targets the
// child's process group on Unix and uses taskkill /T on Windows), not just the
// direct child: a stdio server launched through a package runner (npx, pnpm
// dlx, bunx) is a grandchild of the launcher, and killing only the launcher
// would leave the server running as an orphan.
func closeClientBounded(client *mcpclient.Client, cmd *exec.Cmd, grace time.Duration) error {
	closeDone := make(chan error, 1)
	go func() { closeDone <- client.Close() }()

	timer := time.NewTimer(grace)
	defer timer.Stop()
	select {
	case err := <-closeDone:
		return normalizeCloseError(err)
	case <-timer.C:
	}

	if cmd != nil && cmd.Process != nil {
		_ = sysproc.KillTree(cmd)
		// The kill unblocks client.Close's cmd.Wait, so draining the result
		// here is bounded and reaps the child instead of leaving a zombie.
		closeErr := <-closeDone
		if closeErr != nil {
			return fmt.Errorf("server did not exit within %s of stdin close; killed: %w", grace, closeErr)
		}
		return fmt.Errorf("server did not exit within %s of stdin close; killed", grace)
	}
	return fmt.Errorf("server close did not return within %s; abandoned", grace)
}

// normalizeCloseError classifies the result of an MCP client Close. A stdio
// server that exits with a NON-ZERO status after its stdin is closed has still
// disconnected cleanly: closing stdin IS the MCP shutdown request, the child
// was reaped, and the transport is done — the process's own exit code says
// nothing about whether the close succeeded. cmd.Wait surfaces that code as a
// bare *exec.ExitError, and real launchers emit non-zero teardown codes
// routinely (`gh mcp` exits 255; a SIGPIPE-killed child reports "signal: broken
// pipe"). Returning it would turn an ordinary teardown into a spurious error on
// every Reconfigure that reconnects the server (surfaced to the user as a mode
// change failing) and on every shutdown ("N servers failed to stop cleanly").
// Any other close failure — a stdin/stderr close error, or a wait that failed
// for a reason other than the child's exit status — is returned unchanged.
//
// The kill and abandon paths above do NOT funnel through here: they build their
// own "did not exit within ...; killed" / "abandoned" errors, which stay errors
// because it is the server (not its exit code) that is the anomaly there.
func normalizeCloseError(err error) error {
	var exitErr *exec.ExitError
	if errors.As(err, &exitErr) {
		return nil
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
		Unhealthy: s.unhealthy,
		ToolCount: len(s.tools),
		Tools:     toolNames,
		Error:     s.lastError,
	}
}
