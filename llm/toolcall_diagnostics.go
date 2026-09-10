package llm

import (
	"context"
	"encoding/json"
	"log/slog"
)

// maxLoggedToolArgs bounds the raw argument payload captured per tool call so a
// single oversized or duplicated arguments string cannot flood the logs.
const maxLoggedToolArgs = 1024

// logToolCallArguments records the raw argument JSON of each tool call exactly
// as the provider returned it. It exists to diagnose tool-call argument parse
// failures (e.g. `invalid character 'g' looking for beginning of value`, which
// tools surface as a ParseInputError): when the model emits malformed or
// truncated tool arguments, this dumps what actually arrived, distinguishing a
// malformed payload from a defect in the tool's own decoding.
//
// Malformed (non-JSON) arguments are logged at Warn so they are visible without
// enabling debug logging; well-formed calls stay at Debug. logger may be nil
// (→ slog.Default()).
func logToolCallArguments(logger *slog.Logger, provider string, calls []ToolCall) {
	if len(calls) == 0 {
		return
	}
	if logger == nil {
		logger = slog.Default()
	}
	ctx := context.Background()
	for _, call := range calls {
		valid := json.Valid(call.Input)
		level := slog.LevelDebug
		if !valid {
			level = slog.LevelWarn
		}
		if !logger.Enabled(ctx, level) {
			continue
		}
		logger.LogAttrs(ctx, level, "tool call arguments received",
			slog.String("provider", provider),
			slog.String("tool", call.Name),
			slog.String("tool_call_id", call.ID),
			slog.Int("arg_len", len(call.Input)),
			slog.Bool("valid_json", valid),
			slog.String("arguments", truncateArgsForLog(string(call.Input))),
		)
	}
}

// truncateArgsForLog bounds an arguments payload to maxLoggedToolArgs bytes for
// logging, marking the cut so a truncated value is never mistaken for the whole.
func truncateArgsForLog(s string) string {
	if len(s) <= maxLoggedToolArgs {
		return s
	}
	return s[:maxLoggedToolArgs] + "...(truncated)"
}
