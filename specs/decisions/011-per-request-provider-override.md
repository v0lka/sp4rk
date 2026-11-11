# ADR-011: Per-request provider override on ChatRequest

## Status

Accepted

## Context

Subagent Profiles (see [../domains/agents.md](../domains/agents.md)) carry a `model:` frontmatter field, and the executor can already force a per-request model: the router fills `req.Model` only when empty, so a wrapper caller (`agent.NewModelOverrideCaller`) pins a model without touching global state. Provider routing had no equivalent per-request escape hatch: every call went to the router's active provider (`r.activeProvider`), switchable only globally via `SetModel`.

Hosts need to route a specific caller — typically one delegated subagent — to a specific provider (e.g. a local OpenAI-compatible server) while the session's active provider stays where it is, and a caller pinned this way must be immune to a mid-task `SetModel` performed by an unrelated part of the session. The request dump (the executor's JSONL LLM log) should also record which provider a call was pinned to.

## Decision

`llm.ChatRequest` gains an additive, JSON-tagged field:

```go
Provider string `json:"provider,omitempty"`
```

- **Routing:** in `Router.Call`, after the active-state snapshot (read lock) and before `prepareRequest` and the retry loop, a non-empty `req.Provider` resolves `r.providers[req.Provider]` and replaces the active provider for this call only.
- **Miss = immediate non-retryable config error:** an unknown name returns an immediate non-retryable error before the retry loop runs — a plain wrapped error carrying no HTTP status, naming the provider and wrapping the `llm.ErrUnknownProvider` sentinel (`errors.Is(err, llm.ErrUnknownProvider)`); zero attempts against any provider, no backoff. This is a configuration mismatch, not a transient transport failure.
- **Legacy path byte-identical:** an empty `Provider` (the zero value, `omitempty`) selects the active provider exactly as before, and the field is absent from serialized requests — existing callers and request dumps are unchanged byte-for-byte.
- **Model pairing is the caller's responsibility:** `prepareRequest`/`validateContextWindow` are unchanged and keep working on the bare `req.Model`. A caller that overrides the provider should also pin a model the named provider serves (otherwise the active bare model string is sent there, exactly as a pre-set `req.Model` is sent to the active provider today).
- **Wrapper:** `agent.NewModelProviderOverrideCaller(inner, provider, model)` stamps `req.Provider` and `req.Model` on every call (each applies independently and only when non-empty; both empty → inner returned unchanged). The legacy `agent.NewModelOverrideCaller(inner, model)` is kept as the model-only form and delegates with an empty provider. A pinned caller is stateless per call, so a mid-task `SetModel` cannot affect it.
- **Read-only validation surface:** `Router.ProviderForModel(model) (provider string, ok bool)` answers "which provider would this identifier route to" without mutating state — a composite resolves to its named provider (ok tracks the provider NAME only, exactly the runtime `Call` miss predicate; the bare model need not appear in the provider's list), a bare name resolves like `SetModel`. `Router.KnownProviders() []string` snapshots the configured provider names, sorted. Hosts pair the two to fail a composite `provider/model` override closed BEFORE launching the caller it pins, with an error naming the model and the known providers.
- **Discovery:** `agents.AgentDescriptor` gains `Model string json:"model,omitempty"` (mirroring `AgentMetadata.Model`), populated by `Descriptor()`, so consumers see a profile's model override at discovery time without loading the full profile.

## Consequences

Positive:

- Per-call provider pinning with no global switching, no new router instance, and no extra concurrency surface; the retry/backoff/sampling/validation machinery is reused unchanged.
- Request dumps record the override (`provider` key) for debugging misrouted calls.
- Fully additive: empty `Provider` is byte-identical, so existing hosts, providers, and dumps are unaffected.

Negative:

- The caller owns provider/model pairing; a mispaired request sends a model name the named provider may not serve. That exposure is the same class as pre-setting `req.Model` today and surfaces as a provider-side error, not silent misrouting.
- The override names a provider by its configured logical name; renaming a provider entry breaks pinned callers — the same exposure composite model IDs (`"provider/model"`) already carry through `SetModel`.
- Providers ignore the field (routing is the router's job), so a Provider implemented outside the router must be told to disregard it — a minor contract note for host-supplied providers.

## Alternatives Considered

- **Composite provider inside `req.Model`** ("provider/model" parsed by `prepareRequest`): rejected — it overloads the model string, breaks the invariant that the bare model name is what is sent to the API (every provider seam and metadata lookup would need to strip the prefix), and is ambiguous with model names that themselves contain `/`.
- **Per-subagent router** (construct a second `Router` pinned to the provider for the subagent): rejected — it multiplies router instances and duplicated retry/backoff/sampling configuration per pinned caller, changes the caller identity so shared wrapping (token tracking, dump callers) must be rebuilt, and still cannot express "this one caller stays pinned" without extra per-caller state.

## Related

- [../contracts/llm-providers.md](../contracts/llm-providers.md) — the executor → Router contract this decision extends
- [../domains/agents.md](../domains/agents.md) — Subagent Profile `model:` frontmatter and the override callers
- [ADR-009](./009-opt-in-request-scoped-streaming.md) — the prior additive `ChatRequest` extension (DeltaSink) this field's additive shape follows
