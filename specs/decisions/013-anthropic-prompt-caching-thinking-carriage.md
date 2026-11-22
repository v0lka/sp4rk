# ADR-013: Anthropic prompt-cache breakpoints and thinking-block carriage

## Status

Accepted

## Context

The Anthropic Messages API supports provider-level prompt caching (`cache_control` breakpoints, ≤4 per request) and extended thinking whose blocks are signed and MUST round-trip. sp4rk used neither: every request re-prefilled the full prompt (the provider armed `cache_control` only on multi-part system prompts — a partial case of one breakpoint — while tools and history were never cached), and `parseResponse` collapsed thinking blocks into `ChatResponse.Reasoning` with the signature dropped, so a thinking-enabled tool-use continuation re-sent an assistant turn that did not START with its `thinking`/`redacted_thinking` block. The API rejects exactly that shape with HTTP 400 ("Expected `thinking` or `redacted_thinking`"), so the provider carried a blanket fail-safe: `historyHasAssistantToolUse` disabled thinking for EVERY request whose history re-sent an assistant tool-use turn — deterministically forfeiting extended thinking on every tool call of a reasoning-enabled run.

Two constraints shaped the fix. First, the history the executor replays is built through `agent.Step` and the `memory` renderers, not from `ChatResponse.Message` directly — carriage must survive that boundary or it is dead code. Second, the request prefix must stay byte-stable across turns for the cache to hit at all (history mutation advances in K-step cohorts, keeping the prefix byte-stable between boundaries), so the history breakpoint is sound only against that prefix-stability guarantee.

## Decision

Two wire-side extensions on the Anthropic provider, both additive:

- **Cache breakpoints.** On cache-armed requests (non-deterministic `CallPurpose` — executor and the undeclared default; routing/compaction/summarization are one-shot calls where a cache write is pure 1.25× premium) the provider places exactly three `cache_control` breakpoints: the end of the stable system (the `CacheBreakMarker` boundary part of `MultiSystem`, second-to-last part when a dynamic suffix follows, the only part otherwise), the LAST tool definition (caching the whole tools prefix — tools precede system in the API's caching hierarchy), and the stable history boundary (the last cacheable block of the last rendered message, so the whole prompt becomes the cached prefix the next request reads back and extends). Three ≤ the API's 4-breakpoint budget.
- **Thinking carriage.** `llm.Message` gains `ThinkingBlocks []ThinkingBlock` (`thinking` text + verification `Signature`, or `redacted_thinking` encrypted `Data`), mirroring `ReasoningItems`. `parseResponse` captures every block verbatim onto the assistant `Message`; `agent.Step` gains the mirrored field; both history renderers re-emit the blocks at the START of the assistant turn, byte-for-byte (the signature covers the exact thinking text — trimming invalidates it). Nudge-only and batch-guard sites gate the carriage to the response's first materialized step, exactly like `ReasoningItems`, so a duplicate assistant turn is never emitted.
- **Narrowed fail-safe.** `historyHasAssistantToolUse` is replaced by `assistantHistoryMissingLeadingThinking`, which disables thinking only when the FINAL assistant turn in the outgoing history carries NO thinking blocks (a foreign provider's turn or a pre-carriage persisted session) — the one shape the API rejects. Continuations whose turns carry the blocks keep thinking enabled. The budget clamp is unchanged: `budget_tokens` starts at 32000, clamps to `max_tokens/2` when `max_tokens` is smaller, and thinking is skipped entirely below 1024.

## Consequences

Positive:

- Anthropic prompt caching works end to end: tools + system + the growing history prefix are written once and read back every turn (0.1× reads vs full-price prefill), with per-turn uncached input reduced to one turn's worth of tokens.
- Extended thinking survives tool continuations — reasoning quality is no longer forfeited on every tool call of a reasoning-enabled run, and the request no longer needs the blanket thinking disable to avoid the documented HTTP 400.
- The carriage is provider-neutral in shape (`ThinkingBlocks` is ignored by the OpenAI/Google providers, exactly as `ReasoningItems` is ignored outside the Responses path), so the message contract grows without cross-provider coupling.

Negative:

- Hosts that rebuild executor history from their own persistence must preserve `ThinkingBlocks` (and `agent.Step.ThinkingBlocks`) or Anthropic continuations fall back to the narrowed fail-safe (thinking disabled on those requests) — degraded, not broken.
- Cache breakpoints spend the write premium on the first request of every loop-shaped session; sessions that never take a second turn pay ~1.25× on the marked prefix for nothing. One-shot deterministic calls are excluded by the purpose gate; short executor turns are not.

## Alternatives Considered

- **Flat-string re-synthesis** (re-emit the thinking text without its signature, or re-sign host-side): rejected — invalid per the Messages API. The signature is produced by Anthropic and verified on return; an altered or missing signature on a replayed thinking block fails the request, and redacted_thinking carries only the opaque encrypted payload, which cannot be re-synthesized at all.
- **Keep the blanket fail-safe, extend it to text-only final assistant turns** (disable thinking whenever any assistant turn exists without thinking blocks): rejected for carrier turns — it would keep forfeiting thinking on every normal tool continuation, the exact defect this ADR removes; the narrowed check retains it only for histories that genuinely cannot satisfy the API contract.
- **Cache the full history unconditionally, including one-shot calls**: rejected — routing/compaction/summarization calls have no reuse (each is unique content), so breakpoints there convert nothing and only add the write premium; the purpose gate keeps their wire shape byte-identical to the legacy behavior.
- **Breakpoint per system part** (the previous nested marking of all-but-last part): rejected — each breakpoint marks the prefix up to itself, so additional breakpoints on earlier parts are redundant and consume the shared 4-per-request budget that the tools and history breakpoints need.

## Related

- [../domains/llm-providers.md](../domains/llm-providers.md) — Anthropic Messages wire section (breakpoints, carriage, fail-safe)
- [ADR-012](./012-per-request-provider-override.md) — the prior additive `ChatRequest`/message-contract extension pattern this follows
- [../domains/memory/compaction.md](../domains/memory/compaction.md) — the cohort-quantized prefix stability the history breakpoint relies on
