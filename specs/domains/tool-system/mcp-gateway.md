# MCP Gateway

## Role

Manages connections to external MCP (Model Context Protocol) servers, discovers their tools at runtime via the MCP `tools/list` call, and proxies tool execution through the `ToolRegistry`. With it, an agent can call tools hosted by databases, APIs, filesystems, browsers, or any process that speaks MCP — without writing custom Go code per server.

## Key Files

- `github.com/v0lka/sp4rk/tools/mcp` — `Gateway`, `GatewayConfig`, `ServerEntry` (`Timeout`, `CallTimeout`), `StartGateway`, `Server`, `ServerConfig` (`Timeout`, `CallTimeout`), `Server.ToolGroup`, `Tool`, `SchemaSanitizer`, `ServerStatus` (`Unhealthy`), the timeout bounds (`defaultMCPTimeout` = 60s, `unhealthyTimeoutThreshold` = 3), error types (`StartError`/`StopError`/`ReconfigureError`, `TimeoutError`)
- `github.com/v0lka/sp4rk/tools` — `ToolRegistry`, `RegisterWithSource`, `RegisterWithSourceCategory`, `UnregisterBySource`, `StripParamsFromSchema`
- `github.com/v0lka/sp4rk/sysproc` — `HideConsole` (applied to the stdio subprocess so a GUI-subsystem host spawns no console window)

## Behavior

The package is organized around three layers: `Gateway` (manages many servers), `Server` (one connection), and `Tool` (wraps a single MCP tool as a `tools.Tool`).

### Lifecycle

```
StartGateway(ctx, cfg, registry, expandEnv, logger)
│
├─ 1. Expand ${VAR} references in every ServerEntry; build ServerConfig values.
├─ 2. Create a Gateway; store config, default work dir, schema sanitizer, logger.
├─ 3. gateway.Start(ctx, configs):
│      for each server:
│        ├─ connect (spawn process or HTTP connect)
│        ├─ DiscoverTools (MCP tools/list)
│        └─ on failure: collect error, continue with remaining servers
├─ 4. gateway.RegisterTools(registry):
│      for each discovered tool:
│        ├─ resolve the server group (valid override, else transport default)
│        ├─ wrap as tools.Tool via NewTool
│        └─ registry.RegisterWithSource(tool, serverName)
└─ returns (nil, nil) when no servers are configured — safe to call unconditionally
```

`StartGateway` returns `(nil, nil)` when no servers are configured. `Gateway.Start` collects per-server errors and continues connecting to the rest; if any server fails it returns a `*StartError` holding all individual errors, but `StartGateway` still returns the gateway (with whatever servers connected) so the agent keeps working.

### Transports

| Transport | Description | Config fields |
| --------- | ----------- | ------------- |
| `stdio` (default) | Spawn a local subprocess; communicate over stdin/stdout | `Command`, `Args`, `Env`, `WorkDir` |
| `http` | Connect to a remote server over HTTP | `URL`, `Headers`, `HTTPClient` |

For HTTP, the client first tries the **Streamable HTTP** transport and falls back to **SSE** (Server-Sent Events) if initialization fails — compatible with both modern and legacy MCP HTTP servers.

### Per-server timeouts

Each server carries two optional bounds, resolved once at connect time and applied via `context.WithTimeoutCause` (never zero at any use site):

- `ServerEntry.Timeout` / `ServerConfig.Timeout` — bounds the **initialization handshake** (`initialize` + `tools/list`). Zero or negative selects `defaultMCPTimeout` (**60s**). Forwarded from the entry by `serverConfigFromEntry`. For the HTTP transport the bound applies to **each transport attempt** — the Streamable HTTP attempt and, if it fails, the SSE fallback — so an unresponsive server can spend up to twice the bound before `Connect` fails.
- `ServerEntry.CallTimeout` / `ServerConfig.CallTimeout` — bounds a single `tools/call`. Zero or negative **inherits `Timeout`** (which itself already includes the default), so a per-call wire timeout is always in effect.

Because `CallTimeout` inherits `Timeout`, a server that sets **neither** field now caps every `tools/call` at **60s**. This is a behavior change for a host that previously ran long-running calls (browser automation, builds, batch jobs) unbounded on the default configuration; such a host must raise `CallTimeout` (or `Timeout`) per server entry. There is no way to disable the per-call bound — the always-bounded guarantee is intentional. The two defaults and the unhealthy threshold are fixed SDK policy, not per-host knobs.

`Server.Connect` resolves both bounds via `resolveTimeoutBounds` into the unexported `timeout` / `callTimeout`, which then become the single source of truth the gateway diffs against: `configChanged` compares the **resolved** bounds — the same normalization `effectiveToolGroup` applies to a group override — and treats any effective change as reconnect-worthy so a new bound re-applies immediately, while an edit that leaves the effective bounds unchanged (e.g. `Timeout: 0` vs `-1`, or a `CallTimeout` equal to `Timeout`) does not churn the connection (the bounds are captured at connect time and never re-read). The `initialize` and `tools/list` exchanges are bounded by the handshake timeout; `tools/call` by the per-call timeout. `client.Start` is **deliberately never wrapped** — for stdio the child process is spawned with `context.Background()` and `Start` only attaches to it, so coupling its context to the handshake deadline would kill a merely-slow server's process instead of aborting and retrying the handshake; `Start` also returns as soon as the transport is up, so it is not the blocking step a timeout needs to bound.

A bounded operation that exceeds its bound returns a typed `*TimeoutError` (server / `Op` / tool attribution; unwraps to `context.DeadlineExceeded`). Only **our** timer firing counts toward unhealthiness — `timeoutErrorFor` reports `isTimeout=false` for a caller cancellation, which surfaces the parent context's error and is never blamed on the server. After `unhealthyTimeoutThreshold` (**3**) consecutive genuine `tools/call` timeouts the server is marked `Unhealthy` (with a streak description recorded as its last error); only a successful call resets the streak and clears the mark (a failure that is not a timeout neither extends nor clears it), and `Connect` / `Close` clear it too. A timeout never closes the connection or kills the process — `Unhealthy` is an advisory status flag.

### Capability-group derivation and override

Every discovered tool receives the server's effective `tools.ToolGroup`. Without an override, `stdio` maps to `local_mcp` and `http` maps to `remote_mcp`. `ServerEntry.ToolGroupOverride` is forwarded through `StartGateway`/`Reconfigure` into `ServerConfig.ToolGroupOverride`; `Server.ToolGroup()` returns a valid, non-reserved override when present, otherwise the transport-derived group.

Empty, unknown, and `system` overrides are ignored and the transport default applies. `Connect` logs a warning for a non-empty ignored value. The reserved `system` group is never available to an external server because it identifies host-trusted orchestration tools that bypass policy gates.

Reconfiguration compares effective groups rather than raw override strings. A change that alters the effective group reconnects and re-registers the server so live descriptors change atomically; two raw values with the same effective group do not cause reconnect churn.

For stdio, the subprocess is always built through a custom command factory that merges the allowlisted host environment (see [Stdio environment allowlist](#stdio-environment-allowlist-supply-chain-defense) below) with the configured `Env` (matching the transport's default merge), applies `WorkDir`/`DefaultWorkDir` when set, and calls `sysproc.HideConsole(cmd)` so a GUI-subsystem host application does not allocate a console window for the long-lived server process (`CREATE_NO_WINDOW` on Windows; no-op elsewhere). The factory is installed unconditionally — even when no `WorkDir` is set — so window suppression applies to every stdio server.

### Stdio environment allowlist (supply-chain defense)

A stdio MCP server runs an arbitrary third-party command declared in config, so it is treated as untrusted by default. The subprocess inherits only an allowlisted subset of the host environment (`safeStdioEnv`): `PATH`, `HOME`, `USER`, `SHELL`, `USERPROFILE`, `LANG`, `TERM`, `TMPDIR`, `TEMP`, `TMP`, `APPDATA`, `LOCALAPPDATA`, `SYSTEMROOT`, `COMSPEC`, `PATHEXT`, the proxy variables (`HTTP_PROXY`/`HTTPS_PROXY`/`NO_PROXY`/`ALL_PROXY` in both cases, with embedded credentials stripped), CA trust anchors (`SSL_CERT_FILE`, `SSL_CERT_DIR`, `REQUESTS_CA_BUNDLE`, `CURL_CA_BUNDLE`, `NODE_EXTRA_CA_CERTS`), Python venv/conda activation (`VIRTUAL_ENV`, `CONDA_*`), and `LC_*` locale variables. Host secrets — LLM API keys, proxy credentials — are never forwarded implicitly. Explicit `cfg.Env` entries are applied on top of the allowlist and always win, so a host that wants to pass a secret to a server declares it explicitly, and one that wants stricter isolation (e.g. to block `~/.aws/credentials` reads via `HOME`) can override `HOME` through `cfg.Env`.

### Environment variable expansion

`${VAR}` references may appear in `Env` values (stdio), `URL` (http), and `Headers` values (http). Expansion is performed by the `expandEnv` function passed to `StartGateway` (typically `os.ExpandEnv`, but a custom function can inject secrets from a vault). Expansion happens **before** the server starts, and the gateway persists the *expanded* config so subsequent reconfigurations compare against post-expansion values (avoiding spurious reconnects).

### Tool discovery & MCP Tool characteristics

After connecting, each server is queried with `tools/list`. Returned tools become `ToolInfo` records wrapped by `NewTool` into a `Tool` implementing `tools.Tool`, and registered with the server name as `Source`. From the executor's perspective, MCP tools are indistinguishable from built-in tools. Every MCP tool wrapper has fixed behaviors:

- `Group()` → the server's effective group: valid non-`system` override, otherwise `local_mcp` for stdio or `remote_mcp` for http.
- `DefaultPolicy()` → `PolicyUserConfirm` (conservative default — remote, opaque tools).
- `IsUntrusted()` → always `true` (output from external servers may be adversarial).
- `Judge()` → always a zero `JudgeOutcome` ("no tool-specific concern"; the gateway cannot inspect remote semantics, so it offers no heuristic of its own).
- `Execute()` → calls the MCP server's `tools/call`, extracts text content (joins it; marshals structured content to JSON when no text is present).

### SchemaSanitizer

`SchemaSanitizer func(source string, schema json.RawMessage) json.RawMessage` transforms an input schema before it is exposed to the LLM (applied in `NewTool`). The typical use is stripping source-specific parameters the model should not be asked to fill (e.g. a `project` scoping field the host supplies itself). `tools.StripParamsFromSchema(schema, paramsToRemove)` is the SDK helper for this; return the input unchanged to pass through.

### Reconfigure

`Gateway.Reconfigure(ctx, newConfig, registry, expandEnv)` updates server connections based on a new config while preserving unchanged connections: removed servers are unregistered (`UnregisterBySource`) and closed; added servers are connected/discovered/registered; changed servers get a fresh connection; unchanged servers are left alive. Config comparison uses the *expanded* config, the effective tool group, and the resolved timeout bounds (`resolveTimeoutBounds`), so an edit that changes neither the effective group nor the effective bounds does not force a reconnect.

### Status & introspection

`Status()` returns per-server `ServerStatus` (name, transport, connected, unhealthy, starting, tool count, tools, error), sorted by name for deterministic output. `starting` is a transient state distinct from `Connected`/`Error`, marking an entry still being initialized; it is non-omitempty (always serialized) so a frontend can treat it as a first-class state. `unhealthy` is likewise non-omitempty — an advisory mark set after 3 consecutive `tools/call` timeouts (see [Per-server timeouts](#per-server-timeouts)); a successful call clears it, as do `Connect` and `Close`. `ServerNames()`, `ToolCount()`, and `GetServer(name)` provide further introspection.

`Status()` also includes configured servers whose `Connect` or `DiscoverTools` failed: their last failure is kept in a separate `failedServers` map (`Connected: false`, last error set, transport defaulted) and merged into the result, so callers can render every configured server. Failed entries are cleared when the server connects successfully, is removed from the config, or the gateway stops. Failed servers never enter the live connection map (`servers`/`expandedConfigs`) — the clean-state invariant for reconnection diffing is unchanged — and `Reconfigure` always retries a configured server that is only in `failedServers`.

## Error Handling

MCP failures are **non-fatal** — a single broken server never prevents the agent from running with the remaining tools.

- `Gateway.Start` collects per-server errors into a `*StartError` and continues.
- `StartGateway` logs start errors as warnings and still returns the gateway.
- `Gateway.Stop` returns a `*StopError` if any server fails to close cleanly.
- `Gateway.Reconfigure` returns a `*ReconfigureError` for failed operations.
- `Gateway.Stop()` always attempts graceful close of all connections.
- A bounded operation that exceeds its timeout returns a typed `*TimeoutError` (attributed to server / `Op` / tool; `errors.Is(err, context.DeadlineExceeded)` holds). A handshake timeout fails that server's connect (non-fatal to the remaining servers, as above); a call timeout is returned to the caller. A timeout is never swallowed and never closes the connection.

## Invariants

- MCP gateway failure is non-fatal (the application runs without MCP tools).
- MCP tools always have source tag `<server_name>` and `SourceCategory == MCP`.
- MCP tools default to `PolicyUserConfirm`, always report `IsUntrusted() == true`, and always carry a declared effective group.
- Stdio servers default to `local_mcp`; HTTP servers default to `remote_mcp`; only a valid non-`system` override replaces that mapping.
- An MCP tool may never shadow an already-registered non-MCP tool; an MCP server re-registering its own tools is allowed.
- `Reconfigure` is additive/preserving: unchanged servers are not reconnected.
- `Stop` always attempts graceful close.
- Every server has a non-zero resolved handshake bound (`defaultMCPTimeout` = 60s when `Timeout` is unset) and a non-zero per-call bound (inherits the handshake bound when `CallTimeout` is unset) — no MCP operation runs unbounded.
- `client.Start` is never bounded by the handshake timeout (a slow handshake must not kill the stdio child process).
- 3 consecutive genuine `tools/call` timeouts set `ServerStatus.Unhealthy`; a successful call (or `Connect`) clears it, and `Close` clears it too (a disconnected server never reports `Unhealthy: true`); a timeout **marks**, never kills — the connection stays open.

## Related Specs

- [README.md](README.md) — tool system overview and policy enforcement
- [builtins.md](builtins.md) — built-in tool catalog
- [../orchestration/executor.md](../orchestration/executor.md) — MCP cache TTL and untrusted wrapping
- [../memory/compaction.md](../memory/compaction.md) — MCP-sourced cache entries expire on TTL
