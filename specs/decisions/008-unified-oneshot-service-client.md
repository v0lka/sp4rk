# ADR-008: Unified one-shot service client

## Status

Accepted

## Context

Before this decision, every engine component that made a single structured LLM call implemented its own private call/repair loop, and drift had accumulated across the five call sites (router classification, planner, reflector, and the three LLM judges):

- The router retried a parse failure exactly once; the planner, reflector, and judges each hand-rolled their own loop shape. A parse-repair improvement had to be re-implemented per site and could not be tested in one place.
- The judges pinned their own `Provider` + `Model` pair, a private 2-minute timeout, and a fixed temperature. That decoupled them from the session's active model (switching the session model left the judges behind on a stale pinned pair), let a slow judge stall past the host's service timeout, let judge tokens escape session accounting (no `TrackingCaller` unless each site remembered to wrap), and hard-4xx'ed on endpoints that reject the temperature parameter outright.
- Structured calls that asked for reasoning got the family server default — on effort-seeded families (Qwen 3.8+, GLM 5.2+) that is the *strongest* effort, adding latency and tokens to short structured verdicts that do not benefit from it.
- Log hygiene diverged: some sites logged provider diagnostics, which can echo untrusted tool arguments back into logs.

## Decision

1. **Single client.** `oneshot.Do` is the one implementation of the one-shot structured call: on a parse failure, re-ask with a corrective nudge — an assistant echo of the model's own failed output plus a `[System]`-prefixed user message restating the required format. Exactly two nudges (three attempts), then a final refusal resolved by an explicit `OnFailurePolicy` (`error` / `fallback` / typed `fail_safe`). The package is a leaf: it imports only `llm` and the standard library, so every layer can use it.
2. **Single timeout class.** The timeout of a service call is the caller's context — one host-side service timeout. The judges' private 2-minute budgets are deleted; no call site creates its own deadline.
3. **Tiered reasoning.** Structured service calls request reasoning tier **Off**, mapped per family and model version by `llm.ReasoningForCall` to the native disable spelling (`""` = no field sent, fail-closed). Router/planner/reflector resolve the tier from an optional family/model identity; an explicit per-site override wins. The judges resolve family/model from the caller's `ActiveModel()`, degrading to no field when the caller hides it.
4. **Two-nudge policy everywhere.** The unified loop replaces the router's single repair retry and the judges'/planner's/reflector's private loops. Feedback text stays per site: a static `RetryHint` or an error-driven `RetryHintFn` (the planner's two distinct feedback texts, with the underlying error quoted verbatim).
5. **Temperature catalog, not pins.** Service calls pin no temperature: they declare `CallPurposeRouting`, and the Router injects the model-catalog deterministic temperature (`DeterministicTemperature`: `0.0` default, `google`/`kimi` `1.0`, `qwen` `0.6`) and strips all sampling fields for models that authoritatively cannot accept the parameter.
6. **Judges ride the Router.** `NewToolJudge` takes a `Caller` (by construction the session's `llm.Router`) instead of `Provider` + `Model` and follows the active model; a supplied `*llm.UsageTracker` wraps the caller in `llm.TrackingCaller`, so judge token usage is accounted structurally. The `security.judge.model`-style configuration knob is removed together with the fields it fed.

## Consequences

Positive:

- Parse-repair is fixed and extended in one place, with one test suite covering the loop, the nudge shapes, and the failure policies.
- Service calls track the session's active model; a model switch re-targets the judges instead of stranding them on a pinned pair.
- Judge token accounting is structural (`TrackingCaller` wrap), not per-site discipline.
- Structured calls get the latency and token profile they need (reasoning off) and stop 400-ing on sampling-rejecting endpoints.
- Fail-safe call sites are greppable (`OnFailureFailSafe`) and reviewed as security decisions; log hygiene (no model output, no provider diagnostics) is enforced by the shared client.

Negative / rules to keep:

- The client is deliberately metadata-agnostic: every new call site assembles its own profile (purpose, tier, caller, tracker). Nothing in the type system forces this — the unified profile in [../domains/oneshot.md](../domains/oneshot.md) is the checklist.
- Transport-error handling stays per call site: `Do` returns the error as-is, so security-sensitive sites (the judges) must keep mapping it to their fail-safe terminals themselves and must not log it.
- Strict-mode and loop-judge calls may now spend up to three LLM attempts on consecutive parse failures (previously one per invocation), still uncached and without fast paths — the per-invocation security properties are unchanged, only the worst-case call count moved.

## Alternatives Considered

- **Keep per-site loops, align them by review.** Rejected: the sites had already drifted (different attempt counts, different nudge shapes, different logging); alignment by convention does not hold.
- **Parse-repair middleware inside the Router.** Rejected: the Router's retry policy is transport-level (backoff on 429/5xx); parse repair needs response-level context (the failed output echoed as an assistant turn) and would couple the primitive layer to conversation-repair semantics.
- **Keep judge model/temperature pinned in configuration.** Rejected — that was the status quo: it decouples judges from the active model, 400s on sampling-rejecting endpoints, and requires a config knob that can silently disagree with the session.
- **Make `Do` resolve sampling and reasoning itself.** Rejected: it would force `oneshot` to depend on the registry and duplicate `Router.prepareRequest`'s metadata policy; metadata-agnostic keeps `oneshot` a leaf importable from any layer.
