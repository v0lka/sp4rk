package llm

import (
	"context"
	"encoding/json"
	"log/slog"
	"unicode/utf8"
)

// maxLoggedToolArgs bounds the raw argument payload captured per tool call so a
// single oversized or duplicated arguments string cannot flood the logs.
const maxLoggedToolArgs = 1024

// repairableDurationKeys lists the argument keys whose values are duration
// strings (Go time.ParseDuration spellings). Only these keys qualify for the
// unquoted-duration repair; a duration-like scalar anywhere else is left as-is.
var repairableDurationKeys = map[string]struct{}{
	"timeout":         {},
	"timeout_s":       {},
	"timeout_sec":     {},
	"timeout_secs":    {},
	"timeout_seconds": {},
	"timeout_ms":      {},
}

// argSanitization records the outcome of sanitizeToolCallArguments for logging.
type argSanitization struct {
	// repaired[i] is true when calls[i].Input was rewritten.
	repaired []bool
	// originals[i] carries the pre-repair payload when repaired[i] is true
	// (nil otherwise), so the log can show both what the provider sent and
	// what will dispatch.
	originals []json.RawMessage
}

// finalizeToolCallArguments is the single tool-call argument sanitation step on
// every provider response path: it repairs known provider quirks in place and
// then records the arguments (see logToolCallArguments). Sanitation runs
// unconditionally — before any log-level check — so repaired arguments reach
// dispatch even when logging is fully disabled.
func finalizeToolCallArguments(logger *slog.Logger, provider string, calls []ToolCall) {
	san := sanitizeToolCallArguments(calls)
	logToolCallArguments(logger, provider, calls, san)
}

// sanitizeToolCallArguments applies repairToolCallInput to every call in place
// (the caller's ToolCalls slice shares backing storage with the returned
// ChatResponse, so rewritten arguments flow to dispatch). Payloads that are
// already valid JSON — and payloads no known quirk repair can fix — are left
// byte-identical. The returned record is aligned with calls and may be nil
// when calls is empty.
func sanitizeToolCallArguments(calls []ToolCall) *argSanitization {
	if len(calls) == 0 {
		return nil
	}
	san := &argSanitization{
		repaired:  make([]bool, len(calls)),
		originals: make([]json.RawMessage, len(calls)),
	}
	for i := range calls {
		input := calls[i].Input
		if len(input) == 0 || json.Valid(input) {
			continue // valid (or empty) arguments dispatch untouched
		}
		if fixed, ok := repairToolCallInput(input); ok {
			san.originals[i] = json.RawMessage(append([]byte(nil), input...))
			calls[i].Input = json.RawMessage(fixed)
			san.repaired[i] = true
		}
	}
	return san
}

// repairToolCallInput rewrites known provider quirks in an invalid-JSON
// tool-argument payload:
//
//  1. an unquoted duration-like scalar under a known duration key
//     (`{"timeout": 120s}` → `{"timeout": "120s"}`) — some models emit
//     shell-style durations where the JSON schema declares a string;
//  2. a trailing comma before `}` / `]`;
//  3. a raw \n, \r or \t byte inside a JSON string value (`"a<NL>b"` →
//     `"a\nb"`) — some models emit multi-line string values with literal
//     newlines, which JSON forbids inside strings.
//
// The rewrite is deterministic and narrow: string tokens are copied
// verbatim except for the three raw control bytes above, only the exact
// shapes above fire, and the candidate is accepted only when it parses as
// valid JSON — otherwise ok is false and the caller must dispatch
// the original bytes unchanged. Valid input is returned unrepaired.
func repairToolCallInput(input []byte) ([]byte, bool) {
	if json.Valid(input) {
		return nil, false
	}
	out := make([]byte, 0, len(input)+2)
	i, n := 0, len(input)
	for i < n {
		switch c := input[i]; c {
		case '"':
			start := i // token start in the ORIGINAL input (duration repair reads it)
			// Copy the string token (key or value). Raw \n, \r and \t
			// bytes inside the token are escaped (quirk 3); every other
			// byte and every existing escape sequence is copied verbatim.
			out = append(out, '"')
			i++
			for i < n {
				if input[i] == '\\' {
					if i+1 < n {
						out = append(out, input[i:i+2]...)
						i += 2
						continue
					}
					out = append(out, '\\') // lone trailing backslash: malformed, copy verbatim
					i++
					continue
				}
				if input[i] == '"' {
					out = append(out, '"')
					i++
					break
				}
				switch input[i] {
				case '\n':
					out = append(out, '\\', 'n')
				case '\r':
					out = append(out, '\\', 'r')
				case '\t':
					out = append(out, '\\', 't')
				default:
					out = append(out, input[i])
				}
				i++
			}
			// Duration repair: a known key directly followed by an unquoted
			// duration-like scalar (`"timeout": 120s}`).
			if i > start+2 && isDurationKey(input[start+1:i-1]) {
				j := skipJSONWS(input, i)
				if j < n && input[j] == ':' {
					j = skipJSONWS(input, j+1)
					if end, ok := matchDurationToken(input, j); ok {
						out = append(out, input[i:j]...)
						out = append(out, '"')
						out = append(out, input[j:end]...)
						out = append(out, '"')
						i = end
						continue
					}
				}
			}
		case ',':
			// Trailing-comma repair: drop the comma when only whitespace
			// precedes the closing `}` or `]`.
			j := skipJSONWS(input, i+1)
			if j < n && (input[j] == '}' || input[j] == ']') {
				i++
				continue
			}
			out = append(out, c)
			i++
		default:
			out = append(out, c)
			i++
		}
	}
	if !json.Valid(out) {
		return nil, false
	}
	return out, true
}

// isDurationKey reports whether key (the raw bytes between an argument key's
// quotes) is in repairableDurationKeys.
func isDurationKey(key []byte) bool {
	_, ok := repairableDurationKeys[string(key)]
	return ok
}

// skipJSONWS advances past whitespace at input[i].
func skipJSONWS(input []byte, i int) int {
	for i < len(input) {
		switch input[i] {
		case ' ', '\t', '\n', '\r':
			i++
		default:
			return i
		}
	}
	return i
}

// matchDurationToken reports whether a whole duration-like scalar (`120s`,
// `1.5ms`, `2h`, `500us`, `750µs`) starts at input[i]. It returns the end
// offset only when the token is the complete value: after the unit, nothing
// but whitespace may precede a JSON value terminator (`,`, `}`, `]`) or end of
// input. Compound durations (`1h30m`) and suffixed units (`120seconds`) do not
// match — the repair stays narrow.
func matchDurationToken(input []byte, i int) (int, bool) {
	n := len(input)
	start := i
	for i < n && input[i] >= '0' && input[i] <= '9' {
		i++
	}
	if i == start {
		return 0, false // no integer part
	}
	if i < n && input[i] == '.' {
		i++
		frac := i
		for i < n && input[i] >= '0' && input[i] <= '9' {
			i++
		}
		if i == frac {
			return 0, false // digits required after the decimal point
		}
	}
	// Longest units first so `ms` is not read as `m` followed by a stray `s`.
	for _, unit := range []string{"ns", "us", "µs", "ms", "s", "m", "h"} {
		if i+len(unit) <= n && string(input[i:i+len(unit)]) == unit {
			i += len(unit)
			j := skipJSONWS(input, i)
			if j >= n || input[j] == ',' || input[j] == '}' || input[j] == ']' {
				return i, true
			}
			return 0, false
		}
	}
	return 0, false
}

// logToolCallArguments records each tool call's argument payload as it will
// dispatch. It exists to diagnose tool-call argument parse failures (e.g.
// `invalid character 'g' looking for beginning of value`, which tools surface
// as a ParseInputError): when the model emits malformed or truncated tool
// arguments, this dumps what actually arrived, distinguishing a malformed
// payload from a defect in the tool's own decoding.
//
// Arguments that arrived malformed are logged at Warn so they are visible
// without enabling debug logging — including payloads a repair fixed
// (repaired=true), so the provider-quirk signal stays observable; only calls
// valid exactly as received stay at Debug. san may be nil (no sanitation
// happened); logger may be nil (→ slog.Default()).
func logToolCallArguments(logger *slog.Logger, provider string, calls []ToolCall, san *argSanitization) {
	if len(calls) == 0 {
		return
	}
	if logger == nil {
		logger = slog.Default()
	}
	ctx := context.Background()
	for i, call := range calls {
		wasRepaired := san != nil && i < len(san.repaired) && san.repaired[i]
		valid := json.Valid(call.Input)
		level := slog.LevelDebug
		if !valid || wasRepaired {
			level = slog.LevelWarn
		}
		if !logger.Enabled(ctx, level) {
			continue
		}
		attrs := []slog.Attr{
			slog.String("provider", provider),
			slog.String("tool", call.Name),
			slog.String("tool_call_id", call.ID),
			slog.Int("arg_len", len(call.Input)),
			slog.Bool("valid_json", valid),
			slog.Bool("repaired", wasRepaired),
			// Dispatch projection at this layer: valid arguments dispatch to
			// the tool; invalid ones surface downstream as ParseInputError.
			slog.String("dispatch_outcome", dispatchOutcome(valid)),
			slog.String("arguments", truncateArgsForLog(string(call.Input))),
		}
		if wasRepaired && i < len(san.originals) && san.originals[i] != nil {
			attrs = append(attrs, slog.String("raw_arguments", truncateArgsForLog(string(san.originals[i]))))
		}
		logger.LogAttrs(ctx, level, "tool call arguments received", attrs...)
	}
}

// dispatchOutcome names the projected dispatch result for a tool call:
// dispatchOK when the arguments are valid JSON (they dispatch to the tool),
// dispatchParseError when they are still invalid (the tools layer surfaces a
// ParseInputError downstream).
const (
	dispatchOK         = "ok"
	dispatchParseError = "parse_error"
)

func dispatchOutcome(valid bool) string {
	if valid {
		return dispatchOK
	}
	return dispatchParseError
}

// truncateArgsForLog bounds an arguments payload to maxLoggedToolArgs bytes for
// logging, marking the cut so a truncated value is never mistaken for the whole.
// The cut point is backed off to a UTF-8 rune boundary: tool arguments routinely
// carry non-ASCII text (file paths, file contents), and splitting a multi-byte
// rune would end the logged prefix with invalid UTF-8 that renders as U+FFFD
// mojibake in text handlers.
func truncateArgsForLog(s string) string {
	if len(s) <= maxLoggedToolArgs {
		return s
	}
	end := maxLoggedToolArgs
	for end > 0 && !utf8.RuneStart(s[end]) {
		end--
	}
	return s[:end] + "...(truncated)"
}
