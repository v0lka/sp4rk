# ADR-009: Opt-in request-scoped streaming channel

## Status

Accepted

## Context

sp4rk's LLM call path was strictly synchronous: `llm.Provider.ChatCompletion` returns a complete `*ChatResponse`, `agent.LLMCaller.Call` wraps it, and every consumer (the ReAct executor, the Conductor, `oneshot` service calls) assumed the full text arrived at once. Hosts that wanted a live-typing UX had no way to observe assistant text before the response completed, even though the underlying providers (OpenAI Chat Completions, Anthropic Messages) expose server-sent-event streaming and the `agent.Events` interface already carried the `AssistantChunk`/`AssistantDone` hooks.

Adding streaming naively risked breaking the public contracts that downstream host applications and third-party or test providers implement:

- Adding a `ChatCompletionStream` method to the `llm.Provider` interface would force every existing implementation — downstream providers, test doubles, mock LLMs — to implement it or stop compiling. It is a breaking change to a frozen contract.
- The same applies to `agent.LLMCaller`: a new streaming method would ripple through every caller wrapper (`agent.LoggingLLMCaller`, `agent.NewDumpCaller`, `agent.NewModelOverrideCaller`, `llm.TrackingCaller`).
- Streaming must stay optional. Most calls have no interest in intermediate text (routing, planning, reflection, judging, subagent service calls), and delivering text before the response is final has correctness implications — a response can be discarded by a nudge, and a naive retry loop would re-issue text the host already rendered.

The engine needed a way to stream that is additive, preserves the synchronous model exactly for everyone who does not opt in, and introduces no new interface method.

## Decision

1. **The streaming channel is an optional field on the request, not a new method.** `llm.ChatRequest` gains `DeltaSink func(StreamDelta) error` (tagged `json:"-"`) and a transport-neutral `llm.StreamDelta{Text, Reasoning}` value. A non-nil sink asks the provider to invoke it synchronously for each incremental text/reasoning delta as it arrives off the wire. The provider still returns the **same fully-assembled `*ChatResponse`** (content, reasoning, tool calls, usage, stop reason) the synchronous path would. A nil sink — the default — selects the unchanged synchronous path. `llm.Provider.ChatCompletion`, `agent.LLMCaller.Call`, `agent.Events`, and `ChatResponse` are all unchanged.
2. **The hook rides inside the request and flows through every wrapper untouched.** Because the sink is a request field, existing decorators (`LoggingLLMCaller`, `NewDumpCaller`, `NewModelOverrideCaller`, `TrackingCaller`) forward it without modification; the `json:"-"` tag keeps request dumps (`NewDumpCaller`) serializable, since a function has no wire representation and carries no request semantics of its own.
3. **Streaming changes delivery timing only.** The assembled response is identical to the synchronous result; consumers parse one response shape whether or not streaming was on.
4. **Retries are streaming-aware.** `Router.Call` wraps the caller's sink in a per-attempt `deltaGuard`. It resets the guard before each attempt and refuses to retry an attempt that already emitted a delta — text the host has rendered cannot be retracted. A retryable failure before the first delta is retried as usual, preserving the router's existing backoff policy.
5. **Provider support is opt-in per path, with graceful degradation.** The OpenAI Chat Completions path (`Chat.Completions.NewStreaming`) and the Anthropic Messages path (`CreateMessagesStream` + `OnContentBlockDelta`, which also covers Claude models routed through the OpenAI provider's `ProtocolAnthropic` delegate) implement streaming. The Google `generateContent` delegate ignores the hook; the executor detects that no delta arrived and falls back to a single full-text chunk, so that provider is unaffected. *(Historical note: at the time of this decision the Responses API path also ignored the hook — Responses streaming was added later, when the subscription Codex backend required streaming on the wire; the Responses path now streams via `client.Responses.NewStreaming` whenever `RequireStreaming` is set or the sink is non-nil. The canonical support matrix lives in [../domains/llm-providers.md](../domains/llm-providers.md#streaming).)* On the OpenAI path a trailing `include_usage` chunk keeps token accounting identical to the synchronous result; servers that reject the `stream_options` request field with an HTTP 400 are retried once without it (usage then reports zeros, which the executor tolerates), so enabling streaming stays safe on older OpenAI-compatible endpoints.
6. **The executor is the opt-in switch and the single point of event emission.** `agent.WithStreaming(bool)` / `Executor.SetStreaming(bool)` (off by default) install the sink on every loop LLM call and forward each non-empty text delta to `Events.AssistantChunk`, recording `runState.assistantStreamed`. On an implicit finish — the response that ends the loop — a single `emitAssistantEvents` helper emits `AssistantDone` exactly once and emits the full-text `AssistantChunk` only when nothing streamed — so the non-streaming path is reproduced exactly and a degraded provider yields one chunk. A response that ends in tool calls emits its streamed deltas without an `AssistantDone` terminator. A Run that exits with deltas still unterminated — a successful explicit `finish` (whose answer travels in the `ExecutorResult`, not an implicit assistant response), an error, a pause or the step limit — is closed by a Run-exit finalizer that emits one closing `AssistantDone` carrying the accumulated text plus the summed usage of the responses that produced it (a response aborted mid-stream contributes no usage, since none completed), so a subsequent Run over the same `Events` consumer starts from a clean accumulator. Streaming is skipped when assistant events are suppressed (`WithSuppressAssistantEvents`).
7. **Configured at every layer through the existing plumbing.** `ExecutionConfig.Streaming` → `ConductorConfig.Streaming` → `executor.SetStreaming(true)`, plus the fluent `FrameworkBuilder.Streaming(bool)` / `sp4rk.WithStreaming(bool)`.

## Consequences

Positive:

- **Zero breaking changes.** No interface gained a method; `Provider`, `LLMCaller`, `Events`, and `ChatResponse` are untouched, so downstream hosts and third-party/test providers keep compiling — the reference consumer builds unchanged.
- **One response shape.** Streaming does not fork the parsing/assembly contract, so tool-call handling, usage accounting, stop-reason mapping, and event emission stay single-sourced.
- **Transparent through the wrapper stack.** The request-scoped hook needs no changes to any of the four caller decorators.
- **Safe by default.** A nil sink is the synchronous path; a provider that ignores the hook degrades to a single chunk rather than erroring. Enabling streaming can never break a host that does not consume deltas.
- **Retry correctness.** The `deltaGuard` prevents the router from replaying already-delivered text while keeping retries for pre-delta transport failures.

Negative / rules to keep:

- **Two provider code paths.** Each streaming provider duplicates assembly logic (usage from the final `include_usage` chunk, tool-call fragments concatenated by index and sorted, `reasoning_content` read from raw JSON). These must stay parity-equivalent with the synchronous path — the per-provider streaming tests exist to hold that line.
- **Delta timing precedes finalization.** Deltas are emitted before the executor knows a response is final; a response later discarded by a syntax/finish nudge will already have been partially streamed. This is a delivery-timing artifact only — the assembled response driving the loop is unchanged. A reactive-compaction retry is no longer such a case: the executor refuses to re-issue a context-exceeded attempt whose deltas were already delivered (the error is returned instead), so partially streamed text can never be followed by a second attempt of the same call.
- **Mid-stream failures are not retried.** Once a delta is delivered, a subsequent retryable error is returned rather than retried, trading a possible recovery for correctness (no duplicated text). Callers see this as a normal `Call` error.
- **Sink abort is immediate on both built-in streaming paths.** A sink error cancels the underlying HTTP request, so the provider stops reading the wire and the endpoint stops generating; the sink error itself (not the resulting transport failure) is surfaced to the caller. On the OpenAI path the SDK stream is closed directly; on the Anthropic path the go-anthropic streaming callback cannot return an error, so the callback records the failure and cancels the per-call request context.
- **Google is not yet streamed.** It degrades to a single full-text chunk; adding it later needs no interface change — only a new provider streaming path. *(The same originally held for the Responses API path; it has since gained streaming — see the historical note in decision point 5.)*

## Alternatives Considered

- **Add a `ChatCompletionStream` method to `llm.Provider` (and mirror it on `agent.LLMCaller`).** Rejected: it is a breaking change to a public interface every downstream and test implementation must satisfy, and it would force edits through all four caller wrappers. The request-scoped field keeps streaming purely additive.
- **Return a `<-chan StreamDelta` from a streaming entry point.** Rejected: it splits the API into two response shapes, complicates error propagation (a channel cannot carry the final error ergonomically), and still has to produce the assembled `ChatResponse`, duplicating the synchronous result.
- **Introduce dedicated streaming request/response types.** Rejected: it would fork prompt/response mapping and let the streamed result drift from the synchronous one — exactly the divergence the shared `ChatResponse` avoids.
- **Put the sink on the provider (a provider-level callback).** Rejected: the sink is per-call, not per-provider; a provider-level callback cannot distinguish concurrent or sequential calls and would leak one call's deltas into another.
- **Make streaming the default whenever a sink-capable provider is used.** Rejected: it would change the timing contract for every existing host and expose the delta-before-finalization artifact to callers who never asked for it. Opt-in preserves the synchronous model exactly.
- **Route parse-repair/retry through the stream.** Rejected: keep retry policy where it is (transport-level in the router, response-level in `oneshot`); the `deltaGuard` is the minimal streaming-specific concession — no retry after partial delivery.

## Related

- [../domains/llm-providers.md](../domains/llm-providers.md#streaming) — the transport hook, `StreamDelta`, `DeltaSink`, and the `deltaGuard`.
- [../contracts/llm-providers.md](../contracts/llm-providers.md) — `ChatRequest`/`StreamDelta` in the provider contract and the rule that a streaming provider MUST honor `DeltaSink`.
- [../domains/orchestration/executor.md](../domains/orchestration/executor.md#streaming) — executor opt-in, `runState.assistantStreamed`, and the `emitAssistantEvents` finalization.
- [../architecture/data-flow.md](../architecture/data-flow.md) — `AssistantChunk`/`AssistantDone` semantics under streaming.
- [008-unified-oneshot-service-client.md](./008-unified-oneshot-service-client.md) — the structured service calls that deliberately leave `DeltaSink` nil and take the synchronous path.
