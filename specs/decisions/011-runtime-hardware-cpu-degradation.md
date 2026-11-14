# ADR-011: Runtime hardware→CPU degradation after consecutive inference failures

## Status

Accepted

## Context

The `auto` execution provider already degrades gracefully at construction time: the ladder CUDA → OpenVINO is walked at the session-options stage and again at the session-creation stage, and a request whose hardware rungs all fail continues on the CPU provider with a WARN naming every reason ([domains/embedding.md](../domains/embedding.md)). Once a hardware-backed session is **live**, however, the contract inverts: any later failure is a loud error, forever. The invariant exists for a good reason — the pre-runtime-degradation fallback rebuilt session options and stopped the dedicated ONNX thread, which would strand a live session that still runs on that thread, and a silent CPU slide would disguise a broken hardware setup as a working one.

The cost of keeping the invariant absolute surfaced on real hardware: a CUDA state that starts failing mid-flight (a driver upgrade that leaves it erroring, device lost after a suspend/resume cycle, device memory exhausted by another process) leaves the embedder bricked for the rest of the process. Every `EmbedDocuments`/`EmbedQuery` call fails — including single-text calls on the eager session — while the CPU provider remains perfectly usable. The host's only recovery was restarting the process, and the session-creation-time degradation cannot help because it is gated on *no live session*.

A provider that **hangs** rather than erroring is a different failure mode and is deliberately outside this mechanism: nothing returns, so there is nothing to count and no degradation can fire. `Embedder.Wedged(bound)` — the per-job heartbeat on the dedicated ONNX thread, readable without the embedder mutex — is the designated diagnosis for it, and the caller-side timeout is only the backstop that bounds the wait.

## Decision

For an `auto` request that construction resolved to a hardware provider, the embedder now degrades to the CPU provider at **runtime**, once, when inference itself fails repeatedly:

1. **Consecutive-failure counter.** Every ONNX inference attempt that **returns** an error bumps a counter; any successful inference resets it to zero. The counter is embedder state under the embedder mutex. An attempt that never returns (a wedged thread) is invisible to it by construction — see the hang paragraph in Context and `Embedder.Wedged`.
2. **Threshold.** The degradation triggers when the counter reaches `inferenceDegradeThreshold` (a compile-time constant, `2`). One failure stays a loud error — a transient hiccup must not demote the embedder on its own; the second consecutive failure is evidence the provider stopped working mid-flight.
3. **Degradation.** Under the embedder mutex, on the (quiescent) dedicated ONNX thread, `degradeToCPU` — extended, not duplicated — now destroys every live session as one job on that thread before stopping it, rebuilds the session options on the CPU provider, and commits the new state. The CPU options are still built first: a build failure leaves the embedder exactly as it was. The existing WARN log carries the causing error.
4. **Lazy re-creation and a single retry.** The call that hit the threshold lazily re-creates the session it needs and retries the failed inference exactly once on the CPU provider. Every later call rebuilds what it needs as well, so cross-call state cannot stay degraded: the batch session and the length buckets through `ensureBatchSession`/`ensureBucketSession`, the single-text fast path through `ensureQuerySession` (otherwise a degradation triggered by a multi-text call would silently leave every later query on the much larger batch session — correct results, ~15–25× the per-call compute; a re-create that fails is logged at WARN and only that query falls back to the batch session, so the gate can never fail a query). The caller's invocation therefore either completes on the CPU provider or returns **one loud error** chaining the original hardware failure with whichever degradation or retry step failed. The retry is structurally once: the degradation clears the pending flag, so a later failure can never trigger a second degradation.
5. **Explicit requests unchanged.** An explicit `cuda` or `openvino` request never degrades at runtime — its inference failures keep surfacing as loud errors exactly as before. A closed embedder never degrades.
6. **Host observability.** `EmbedderConfig.OnProviderDegrade(from, to, cause)` fires once per degradation, synchronously under the embedder mutex, next to the WARN log. It covers both post-construction degradation paths (the **lazy** session-creation fallback and the runtime degradation); the initial resolution inside `NewEmbedder` and the in-constructor eager session fallback of fixed mode keep their existing WARN + `requestedExecutionProvider` reporting and do not fire the callback.

## Consequences

Positive:

- A dead hardware provider recovers to the CPU provider inside the running process — indexing and queries continue (slower) instead of failing until restart.
- The degradation is loud and observable: WARN log with the cause, one-shot callback, and the flipped value of `ExecutionProvider()`.
- `degradeToCPU` is the single choke point for post-construction degradation, now covering both the no-session-yet and live-session cases; the session-destruction teardown mirrors `Close`'s single-job shape on the ONNX thread.

Negative / rules to keep:

- The "a later failure with an already-built session is a loud error, never a silent CPU retry" invariant is **amended, not removed**: the runtime degradation is allowed only for `auto` requests, only after `inferenceDegradeThreshold` consecutive failures, only once, and always with the WARN + callback. Session-*creation* failures with a live session on the winning rung keep the loud-error contract (their ladder walk must not rebuild options a live session was built from).
- The counter does not classify errors: a caller-induced failure shape (e.g. an internal invariant inside `Run`) counts toward the threshold like any other. This is accepted — ONNX Runtime errors are opaque strings, classification would be unreliable, and a host that repeatedly trips it gets a visible, logged degradation rather than silent behavior.
- A **hang** is not recoverable by this mechanism: an inference that never returns never bumps the counter, so no degradation fires and `OnProviderDegrade` never runs. `Embedder.Wedged(bound)` is the authoritative verdict for it (see Context); the host decides what to do — fail the operation, restart the process — and a caller-side timeout stays a backstop, never the diagnosis.
- All live sessions are destroyed on degradation, including buckets unrelated to the failing call — a cost accepted because the provider itself is presumed dead.
- The threshold is a compile-time constant, not a config knob; changing it is a code change with a new benchmark/observation behind it, in line with the domain's default-flip policy.

## Alternatives Considered

- **Retry on the same hardware provider first.** Rejected: a dead provider fails the retry too, doubling the latency of every call and never recovering; the CPU retry must be reached in one step.
- **Degrade on the first inference failure.** Rejected: a single transient failure (one off allocation failure under memory pressure) would permanently demote a healthy GPU setup; two consecutive failures keep the demotion evidence-based.
- **Classify provider-caused vs caller-caused errors and count only the former.** Rejected: ONNX Runtime exposes no error taxonomy through the Go wrapper; string matching is brittle and would silently under-count the very failures the mechanism exists for.
- **Make the threshold a config knob.** Rejected for now: no host has asked for a different value, and the domain keeps high-risk inference behavior behind compile-time decisions, not host flags. A knob can be added when a concrete need appears.
- **Re-create every destroyed session eagerly during degradation.** Rejected: eager re-creation serializes several ~2 s model loads under the embedder mutex for sessions the host may never use again; lazy re-creation by the sites that need them is strictly cheaper.

## Related

- [domains/embedding.md](../domains/embedding.md) — Execution providers, invariants, and configuration updated for the runtime degradation
