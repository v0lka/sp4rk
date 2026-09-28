// Package oneshot performs one-shot, structured-output LLM calls with a
// built-in parse-failure retry loop: send a request, parse the response into
// a typed value, and when the response is unparseable re-ask the model with a
// corrective nudge instead of giving up (or crashing).
//
// The package is a leaf: it imports only github.com/v0lka/sp4rk/llm and the
// standard library, so any layer can use it without pulling in the executor
// or the tool registry. Host code passes its own caller (llm.Router and
// agent.LLMCaller both satisfy the Caller interface).
//
// # The Do loop
//
// Do sends req, runs parse on the response, and returns the parsed value.
// When parse reports an error the response is treated as unparseable and the
// evaluation is retried with a corrective nudge appended to the conversation:
//
//	attempt 1: [system, user]                                  → parse fail
//	attempt 2: [system, user, assistant(echo), user("[System] …")]   → parse fail
//	attempt 3: [system, user, assistant, user, assistant, user]      → final refusal
//
// Exactly two nudges are issued; the third consecutive parse failure is the
// final refusal, resolved by Options.OnFailure (propagate an error, return a
// fallback value, or return a typed fail-safe value such as a CONFIRM/DENY
// verdict). Each nudge pairs an assistant echo of the model's own failed
// output with a user message starting with "[System]". The echo goes in as an
// assistant-role message — the model's own words, not prose quoted inside
// user/system text — so the retry shows the model what it produced without
// letting that (possibly untrusted-argument-echoing) text masquerade as
// instructions. The user nudge restates the format via Options.RetryHint.
// The appended messages keep roles alternating ([system, user, assistant,
// user, …]); some providers (Gemini) reject consecutive same-role messages.
//
// Transport errors (the caller.Call error) are NEVER retried here: the Router
// owns provider-level retry policy, so Do returns them to the caller as-is.
// A nil response with a nil error is treated as a parse failure (retryable),
// not a transport error.
//
// # Metadata-agnostic client
//
// Do never touches sampling or reasoning metadata: Temperature,
// ReasoningEffort, CallPurpose, MaxTokens and friends must be set by the
// caller on the request it passes in (the judge pins its deterministic
// temperature, the commit-message path declares a summarization purpose, and
// so on). Do only appends nudge messages to req.Messages; the caller's
// request is never mutated.
//
// # Observability
//
// Do logs one structured record per outcome (kind/model/duration/outcome/
// attempt) and optionally dumps every attempt's full request and response as
// JSONL to Options.Dump. Following the judge's security posture, log records
// never carry raw model output or provider error details (both can echo
// untrusted tool arguments); the dump is an explicit opt-in debugging
// artifact and is the only place full payloads are written. A host that
// already dumps at the caller level (agent.NewDumpCaller) needs no Options.Dump;
// a host that wants per-attempt dumps wires its dump writer (e.g.
// agent.DumpWriterFromContext(ctx)) into Options.Dump.
package oneshot

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"strings"
	"time"

	"github.com/v0lka/sp4rk/llm"
)

// Caller is the minimal LLM surface Do needs. Any agent.LLMCaller
// (sp4rk's executor caller interface) and any *llm.Router satisfy it, so
// hosts pass their existing caller without adaptation.
type Caller interface {
	Call(ctx context.Context, req llm.ChatRequest) (resp *llm.ChatResponse, err error)
}

// Parse converts a raw LLM response into a typed value. A non-nil error
// marks the response unparseable and triggers Do's nudge retry loop; the
// error text is surfaced in logs and in the final refusal error.
type Parse[T any] func(*llm.ChatResponse) (T, error)

// OnFailurePolicy selects what Do returns after the final (third consecutive)
// parse failure.
type OnFailurePolicy int

const (
	// OnFailureError propagates the parse failure as an error (the default).
	// The returned error wraps the last parse error and names the attempt
	// count, so callers can surface a precise refusal ("the model produced
	// an invalid commit message after multiple attempts").
	OnFailureError OnFailurePolicy = iota
	// OnFailureFallback returns Options.FallbackValue with a nil error: the
	// call "succeeds" with a caller-chosen degraded result (e.g. the prompt
	// optimizer falls back to the original user prompt when extraction JSON
	// cannot be parsed).
	OnFailureFallback
	// OnFailureFailSafe returns Options.FallbackValue with a nil error,
	// logging the substitution at Warn. It documents the judge-style contract
	// where the fallback is a typed SAFE value whose selection is a security
	// decision (a CONFIRM verdict that defers to a human, or a DENY verdict
	// that stops the run — fail-closed). The policy exists so such call sites
	// are greppable and reviewed as security decisions, not silent defaults.
	OnFailureFailSafe
)

// Options configures a Do call. The zero value is valid: no logging, no
// dump, OnFailureError, and a generic retry hint.
type Options[T any] struct {
	// Kind labels the call in logs and dump records (e.g. "commit_message",
	// "judge_strict"). Empty kinds are logged as "call".
	Kind string
	// Logger receives the structured outcome records. Nil disables logging.
	Logger *slog.Logger
	// Dump optionally receives one JSONL record per attempt (the request
	// before the call, the response after it), shaped like the executor's
	// dump records: {"ts","direction","attempt","data","error"}. Best-effort:
	// encode failures are silently ignored. Nil disables dumping.
	Dump io.Writer
	// OnFailure selects the final-refusal behavior (see OnFailurePolicy).
	OnFailure OnFailurePolicy
	// FallbackValue is returned (with a nil error) when OnFailure is
	// OnFailureFallback or OnFailureFailSafe.
	FallbackValue T
	// RetryHint restates the required output format inside the "[System]"
	// nudge (e.g. "Answer strictly as two lines: VERDICT: … / REASON: …").
	// Empty uses a generic "follow the required output format" wording.
	RetryHint string
	// RetryHintFn, when non-nil, renders the retry hint for each nudge from
	// the parse error that triggered it, so callers whose corrective feedback
	// depends on WHAT failed (e.g. a planner distinguishing "invalid JSON"
	// from "valid JSON but zero steps") can restate the exact problem. The
	// error is the caller's own parse error — format diagnostics, safe to
	// quote back. When non-nil it takes precedence over RetryHint; when nil,
	// RetryHint is used as-is.
	RetryHintFn func(parseErr error) string
}

// maxParseRetries is the number of corrective nudges after the first failed
// parse: two nudges mean three attempts total, and the third consecutive
// parse failure is the final refusal.
const maxParseRetries = 2

// kindOrDefault normalizes an empty log label.
func kindOrDefault(kind string) string {
	if kind == "" {
		return "call"
	}
	return kind
}

// Do performs the one-shot call with the built-in parse-failure retry loop.
// See the package documentation for the retry contract and the failure
// policies. Do never mutates req.
func Do[T any](ctx context.Context, caller Caller, req llm.ChatRequest, parse Parse[T], opts Options[T]) (T, error) {
	var zero T
	if caller == nil {
		return zero, errors.New("oneshot: nil caller")
	}
	if parse == nil {
		return zero, errors.New("oneshot: nil parse function")
	}

	log := opts.Logger
	if log == nil {
		log = slog.New(slog.DiscardHandler)
	}
	kind := kindOrDefault(opts.Kind)
	start := time.Now()

	var nudges []llm.Message // accumulated [assistant, user] retry pairs
	var lastErr error
	attempts := 0

	for attempt := 0; ; attempt++ {
		attempts++

		// Bail out before a pointless extra call when the caller's context is
		// already done (e.g. canceled while the previous parse ran). This is
		// a call-path failure, reported like a transport error.
		if err := ctx.Err(); err != nil {
			log.Warn("oneshot: context done before attempt",
				"kind", kind, "model", req.Model, "attempt", attempts,
				"outcome", "transport_error", "duration", time.Since(start))
			return zero, err
		}

		attemptReq := req
		if len(nudges) > 0 {
			msgs := make([]llm.Message, 0, len(req.Messages)+len(nudges))
			msgs = append(msgs, req.Messages...)
			msgs = append(msgs, nudges...)
			attemptReq.Messages = msgs
		}
		dumpAttempt(opts.Dump, "request", attempts, attemptReq, nil)

		resp, err := caller.Call(ctx, attemptReq)
		dumpAttempt(opts.Dump, "response", attempts, resp, err)
		if err != nil {
			// Transport errors are the Router's domain: never retried here.
			// The error itself is NOT logged (provider diagnostics can echo
			// the request and therefore sensitive arguments — the judge's
			// policy); the caller receives it and decides how to report it.
			log.Warn("oneshot: LLM call failed",
				"kind", kind, "model", req.Model, "attempt", attempts,
				"outcome", "transport_error", "duration", time.Since(start))
			return zero, err
		}

		var result T
		if resp == nil {
			// A nil response with a nil error is a provider contract
			// violation, but it is "the model produced nothing parseable",
			// not a transport failure — treat it as a parse failure so the
			// nudge loop and failure policies apply.
			lastErr = errors.New("empty response (nil response with no error)")
		} else {
			result, lastErr = parse(resp)
		}
		if lastErr == nil {
			log.Debug("oneshot: done",
				"kind", kind, "model", req.Model, "attempt", attempts,
				"outcome", "ok", "duration", time.Since(start))
			return result, nil
		}

		if attempt >= maxParseRetries {
			break
		}
		// Parse failures are format failures, not danger signals: re-ask with
		// a nudge. The parse error comes from the caller's own parser (format
		// diagnostics, not model output), so logging it is safe.
		log.Debug("oneshot: response unparseable, retrying with a format nudge",
			"kind", kind, "model", req.Model, "attempt", attempts, "err", lastErr)
		hint := opts.RetryHint
		if opts.RetryHintFn != nil {
			hint = opts.RetryHintFn(lastErr)
		}
		nudges = append(nudges,
			llm.Message{Role: "assistant", Content: echoText(resp)},
			llm.Message{Role: "user", Content: nudgeMessage(hint)})
	}

	// Final refusal: every attempt was unparseable.
	switch opts.OnFailure {
	case OnFailureFallback:
		log.Warn("oneshot: response unparseable after all attempts, returning fallback value",
			"kind", kind, "model", req.Model, "attempt", attempts,
			"outcome", "fallback", "duration", time.Since(start))
		return opts.FallbackValue, nil
	case OnFailureFailSafe:
		log.Warn("oneshot: response unparseable after all attempts, returning typed fail-safe value",
			"kind", kind, "model", req.Model, "attempt", attempts,
			"outcome", "fail_safe", "duration", time.Since(start))
		return opts.FallbackValue, nil
	default:
		log.Warn("oneshot: response unparseable after all attempts",
			"kind", kind, "model", req.Model, "attempt", attempts,
			"outcome", "error", "duration", time.Since(start))
		return zero, fmt.Errorf("oneshot: %s: response unparseable after %d attempts: %w",
			kind, attempts, lastErr)
	}
}

// echoText picks the assistant echo for the retry turn: the model's own
// output that failed to parse, taken from the first non-empty candidate field
// (Content → ReasoningContent → Reasoning). An entirely empty response gets a
// placeholder so the assistant message is never empty.
func echoText(resp *llm.ChatResponse) string {
	if candidates := CandidateTexts(resp); len(candidates) > 0 {
		return candidates[0]
	}
	return "(empty response)"
}

// nudgeMessage renders the corrective user message. It is prefixed with
// "[System]" so a model reading the transcript treats it as an operator-side
// directive delivered through the user channel (system prompts cannot be
// revised mid-conversation without rebuilding the request).
func nudgeMessage(hint string) string {
	var b strings.Builder
	b.WriteString("[System] Your previous response could not be parsed.\n")
	if hint = strings.TrimSpace(hint); hint != "" {
		b.WriteString(hint)
		b.WriteString("\n")
	}
	b.WriteString("Respond again with only the corrected answer, strictly following the required output format.")
	return b.String()
}

// dumpEntry is one JSONL record written to Options.Dump. It mirrors the
// executor's dump-record shape (ts/direction/data/error) and adds the 1-based
// attempt number so multi-attempt calls stay attributable.
type dumpEntry struct {
	Timestamp string          `json:"ts"`
	Direction string          `json:"direction"`
	Attempt   int             `json:"attempt,omitempty"`
	Data      json.RawMessage `json:"data"`
	Error     string          `json:"error,omitempty"`
}

// dumpAttempt writes one best-effort JSONL record. Dumping is a debugging
// aid: marshal/encode failures are ignored rather than failing the call.
func dumpAttempt(w io.Writer, direction string, attempt int, data any, callErr error) {
	if w == nil {
		return
	}
	raw, err := json.Marshal(data)
	if err != nil {
		return
	}
	entry := dumpEntry{
		Timestamp: time.Now().UTC().Format(time.RFC3339Nano),
		Direction: direction,
		Attempt:   attempt,
		Data:      raw,
	}
	if callErr != nil {
		entry.Error = callErr.Error()
	}
	_ = json.NewEncoder(w).Encode(entry)
}
