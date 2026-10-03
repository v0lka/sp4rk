# One-Shot Service Client

## Purpose

The `github.com/v0lka/sp4rk/oneshot` package is the engine's single client for one-shot, structured-output LLM calls: send a request, parse the response into a typed value, and when the response is unparseable re-ask the model with a corrective nudge instead of failing silently. Every engine component that makes a non-executor LLM call — request classification ([orchestration/router.md](orchestration/router.md)), plan generation ([orchestration/planner.md](orchestration/planner.md)), failure analysis ([orchestration/reflector.md](orchestration/reflector.md)), and the LLM judges ([../architecture/security-model.md](../architecture/security-model.md)) — issues its calls through this package, so the parse-repair policy, failure handling, and observability of structured calls are defined and tested in exactly one place.

## Key Files

- `github.com/v0lka/sp4rk/oneshot/oneshot.go` - `Do`, `Caller`, `Parse`, `Options`, `OnFailurePolicy` — the retry loop and failure policies
- `github.com/v0lka/sp4rk/oneshot/parse.go` - the parser toolkit: `CandidateTexts`, `Text`, `StripFence`, `StripReasoningPrefix`, `ParseJSON`, `KeyLineParser`/`ParseKeyLines`, `SplitInline`, `Marked`/`BetweenMarkers`, `Unquote`, `TrimEmphasis`, `StripLineDecoration`

## Core Types

```go
// The minimal LLM surface Do needs. *llm.Router and agent.LLMCaller both
// satisfy it, so hosts pass their existing caller without adaptation.
type Caller interface {
    Call(ctx context.Context, req llm.ChatRequest) (*llm.ChatResponse, error)
}

// A non-nil error marks the response unparseable and triggers the nudge loop.
// The error text is the caller's own parser diagnostics — format feedback,
// safe to log and to quote back to the model.
type Parse[T any] func(*llm.ChatResponse) (T, error)

// What Do returns after the final (third consecutive) parse failure.
type OnFailurePolicy int

const (
    OnFailureError    OnFailurePolicy = iota // default: propagate an error wrapping the last parse error
    OnFailureFallback                        // return Options.FallbackValue with a nil error
    OnFailureFailSafe                        // return Options.FallbackValue with a nil error, logged at Warn
)

type Options[T any] struct {
    Kind          string                      // log/dump label ("routing_decision", "judge_strict", …)
    Logger        *slog.Logger                // nil disables logging
    Dump          io.Writer                   // optional per-attempt JSONL records
    OnFailure     OnFailurePolicy
    FallbackValue T                           // returned when OnFailure is Fallback or FailSafe
    RetryHint     string                      // static format restatement inside the "[System]" nudge
    RetryHintFn   func(parseErr error) string // per-nudge hint rendered from the parse error; wins over RetryHint
}
```

## Flow

### The Do loop

`Do[T]` sends the request, parses the response, and on a parse failure retries with corrective nudges appended to the conversation:

```
attempt 1: [system, user]                                            → parse fail
attempt 2: [system, user, assistant(echo), user("[System] …")]       → parse fail
attempt 3: [system, user, assistant, user, assistant, user]          → final refusal
```

Exactly two nudges are issued (`maxParseRetries = 2`); the third consecutive parse failure is the final refusal, resolved by `Options.OnFailure`. Each nudge pairs an **assistant echo** of the model's own failed output (taken from `CandidateTexts`) with a **user message** starting with `[System]`:

> `[System] Your previous response could not be parsed.` + the retry hint + `Respond again with only the corrected answer, strictly following the required output format.`

The echo goes in as an assistant-role message — the model's own words, not prose quoted inside user/system text — so the retry shows the model what it produced without letting that (possibly untrusted-argument-echoing) text masquerade as instructions. The user nudge restates the required format via `RetryHint`, or, when `RetryHintFn` is set, via a hint rendered from the parse error that triggered it (callers whose corrective feedback depends on WHAT failed — e.g. the planner distinguishing "invalid JSON" from "valid JSON but zero steps"). The appended messages keep roles alternating (`[system, user, assistant, user, …]`); some providers (Gemini) reject consecutive same-role messages. This repeats the executor's nudge convention (see [orchestration/executor.md](orchestration/executor.md)): an operator-side `[System]` directive delivered through the user channel.

Failure semantics:

- **Transport errors** (the `caller.Call` error) are **never retried** — provider-level retry is the Router's policy; `Do` returns the error as-is. A nil response with a nil error is treated as a parse failure (retryable), not a transport error.
- A done context is checked before each attempt and returned as a transport error (no pointless extra call).
- **Final refusal**: `OnFailureError` wraps the last parse error and names the attempt count; `OnFailureFallback`/`OnFailureFailSafe` return `FallbackValue` with a nil error. `OnFailureFailSafe` documents the judge-style contract where the fallback is a typed SAFE value whose selection is a security decision (a CONFIRM verdict deferring to a human, a DENY verdict stopping the run) — the policy exists so such call sites are greppable and reviewed as security decisions, and the substitution is logged at Warn.

`Do` is **metadata-agnostic**: it never touches sampling or reasoning metadata — `Temperature`, `ReasoningEffort`, `CallPurpose`, `MaxTokens` and friends are set by the caller on the request it passes in — it only appends nudge messages, and it never mutates the caller's request.

### Observability

`Do` logs one structured record per outcome (`kind`/`model`/`duration`/`outcome`/`attempt`; outcomes: `ok`, `transport_error`, `fallback`, `fail_safe`, `error`). Parse errors are logged at Debug — they are the caller's own parser diagnostics, not model output. Raw model output and provider error details are **never logged** (both can echo untrusted tool arguments — the judges' security posture). `Options.Dump` is the only full-payload channel: one best-effort JSONL record per attempt shaped like the executor's dump records (`{"ts","direction","attempt","data","error"}`); a host that already dumps at the caller level (`agent.NewDumpCaller`) needs no `Options.Dump`, otherwise it wires `agent.DumpWriterFromContext(ctx)` in.

### The unified service-call profile

Every built-in call site composes `Do` with the same metadata profile (see [ADR-008](../decisions/008-unified-oneshot-service-client.md) for why):

- **Caller** — the session's `llm.Router` (or an `agent.LLMCaller` wrapping it): service calls ride the active model; no site pins its own model.
- **Timeout** — the caller's ctx (one host-side service timeout); no site creates a private deadline.
- **Sampling** — no pinned temperature: sites declare `CallPurposeRouting`, and the Router injects the model-catalog deterministic temperature (`llm.DeterministicTemperature`: `0.0` default, `google`/`kimi` `1.0`, `qwen` `0.6`) and strips all sampling fields for models that authoritatively cannot accept the parameter (see [../contracts/llm-providers.md](../contracts/llm-providers.md)).
- **Reasoning** — tier Off, resolved per call via the model-aware `llm.ReasoningForCall(family, model, ReasoningTierOff)`: use a native disable spelling only when the model offers it, otherwise use its cheapest enabled effort. GLM 5.3/Flash and Codex use `low`; GPT-5 Pro uses `high`, GPT-5.2/5.4/5.5 Pro `medium`. `""` means send no reasoning field (fail-closed). An explicit per-site override wins (`Router.SetReasoningEffort`, `Planner.Cfg.ReasoningEffort`, `Reflector.SetReasoningEffort`); the judges resolve family/model from the caller's `ActiveModel()`.
- **Token accounting** — `llm.TrackingCaller` wrapping where the site owns a tracker (the judges take one at construction).

### Built-in call sites

| Call site | `Kind` | Parse | Retry hint | Final refusal |
| --------- | ------ | ----- | ---------- | ------------- |
| `agent/router.Router.Route` | `routing_decision` | `oneshot.ParseJSON[RoutingDecision]` + `validateRoutingDecision` | the active JSON output schema (incl. `matched_tools`) | `OnFailureError` → routing-decision-unparseable error |
| `planner.Planner` (Plan/Replan/PlanContinuation) | `plan_generation` | JSON → `*orchestration.Plan` + DAG validation | `RetryHintFn`: two feedback texts (invalid JSON/DAG error vs zero steps), underlying error verbatim | `OnFailureError` → plan-parse error |
| `agent/reflector.Reflector.Reflect` | `reflection` | `parseReflectionResponse` + sanitize layer | static `reflectionRetryHint` | `OnFailureError` → reflection-parse error |
| `tools.ToolJudge.Judge` (advisory) | `judge_advisory` | `ParseKeyLines(verdict, reason)` + `SplitInline` + JSON fallback | `advisoryJudgeRetryHint` | `OnFailureFailSafe` → `VerdictConfirm` + unparsed-reason |
| `tools.ToolJudge.JudgeStrict` | `judge_strict` | strict `VERDICT`/`REASON` parser | `strictJudgeRetryHint` | `OnFailureFailSafe` → `VerdictConfirm` |
| `tools.ToolJudge.JudgeStepLimit` (loop judge) | `judge_step_limit` | explicit-token verdict parser | `stepLimitJudgeRetryHint` | `OnFailureFailSafe` → `LoopVerdictDeny` (fail-closed) |

The judges map **transport errors to the same fail-safe terminals** as the final parse refusal, and never log the error details.

### Parser toolkit

The toolkit in `parse.go` covers, composable and mechanical:

- **`CandidateTexts` / `Text`** — response text fields in extraction priority order `Message.Content` → `Message.ReasoningContent` (DeepSeek-style) → `Reasoning` (OpenAI Responses API), covering models that put the answer in a reasoning field and leave `Content` empty; `Text` additionally strips a wrapping code fence (`StripFence`) and one conversational preamble (`StripReasoningPrefix`: "Sure, ", "Here's the commit message: ", …).
- **`ParseJSON[T]`** — runs `llm.ExtractJSON` per candidate field (recovering fenced blocks and JSON embedded in prose) and decodes the first clean field; `ErrNoJSON` when there is nothing to decode.
- **`KeyLineParser` / `ParseKeyLines`** — KEY: value (or `KEY = value`) lines tolerating markdown decoration: `**VERDICT:**`, list markers, blockquotes, fence remnants, lowercase keys; key matching is case-insensitive with "extends" semantics (`reason` also matches `REASONING:`).
- **`SplitInline`** — a second KEY hiding inside a value line ("VERDICT: ALLOW — REASON: safe").
- **`Marked` / `BetweenMarkers`** — sentinel-marker extraction (`### X_START` … `### X_END`) across all candidate fields.
- **`TrimEmphasis` / `Unquote`** — decorative `**emphasis**`/backticks and one layer of matching quotes around a value.

The toolkit deliberately recovers text and key/value pairs only; domain semantics (which tokens mean which verdict, what counts as a safe default) stay with the caller.

## Invariants

- `Do` never retries transport errors; the Router owns provider-level retry policy. A nil response with a nil error is a parse failure (retryable), never a transport error.
- Exactly two corrective nudges are issued; the third consecutive parse failure is the final refusal resolved by `Options.OnFailure`.
- Nudge messages keep roles alternating, and each nudge pairs an assistant echo of the model's own failed output with a `[System]`-prefixed user message.
- `Do` never mutates the caller's request and never sets sampling/reasoning metadata — the request's metadata profile is entirely the call site's.
- Log records carry no raw model output and no provider error details; parse-error diagnostics (caller-authored) are safe to log; `Options.Dump` is the only full-payload channel and is opt-in.
- `OnFailureFailSafe` exists for typed security fallbacks only; every fail-safe substitution is logged at Warn.
- The parser toolkit is mechanical: it recovers text and key/value pairs; verdict semantics and safe defaults stay with the caller.
- Candidate-field extraction order is `Content` → `ReasoningContent` → `Reasoning`, so responses whose answer landed in a reasoning field are still recoverable.

## Configuration

No configuration file — behavior is `Options` per call. The knobs:

- `Options.OnFailure` / `Options.FallbackValue` — final-refusal behavior per call site.
- `Options.RetryHint` / `Options.RetryHintFn` — nudge feedback text (static vs parse-error-driven).
- `Options.Kind` / `Logger` / `Dump` — observability.

The metadata profile (routing purpose, catalog temperature, tier-Off reasoning, active-model caller, tracker wrap) is code, not config: it is the unified profile above, owned by the call sites and the Router's sampling policy.

## Extension Points

- **New engine or host service calls** (commit messages, title generation, prompt optimization): define a `Parse[T]` composed from the toolkit, choose `OnFailure`, pass `Kind`/`Logger`/`Dump`, and follow the unified profile — Router caller, ctx timeout, `CallPurposeRouting`, tier-Off reasoning. Any `*llm.Router` or `agent.LLMCaller` satisfies `Caller`.
- **`RetryHintFn`** — when corrective feedback must depend on what failed (the planner pattern: distinct hints for distinct parse errors, underlying error quoted verbatim).
- **New toolkit helpers** — add them when a new response-decoration failure mode appears; keep them mechanical and leave domain semantics at the call site.

## Related Specs

- [../contracts/llm-providers.md](../contracts/llm-providers.md) - the LLM boundary `Do` consumes: `ChatRequest`/`ChatResponse`, `CallPurpose`, deterministic temperatures, `ReasoningForCall`/`ReasoningTier`, `TrackingCaller`
- [orchestration/router.md](orchestration/router.md) - routing classification call site
- [orchestration/planner.md](orchestration/planner.md) - plan-generation call site
- [orchestration/reflector.md](orchestration/reflector.md) - reflection call site
- [orchestration/executor.md](orchestration/executor.md) - the executor nudge convention `Do` repeats
- [../architecture/security-model.md](../architecture/security-model.md) - the judge call sites and the no-diagnostics logging posture
- [../decisions/008-unified-oneshot-service-client.md](../decisions/008-unified-oneshot-service-client.md) - why the client exists and the decisions behind the unified profile
