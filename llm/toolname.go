package llm

import (
	"hash/fnv"
	"log/slog"
	"strconv"
	"strings"
)

// Tool names are the one part of a tool definition that the wire protocols
// constrain harder than MCP does. MCP only requires names to be unique
// strings, so an external server is free to expose tools like "ps.list" or
// "app.launch" — but the OpenAI-compatible Chat Completions API rejects any
// tools[].function.name outside ^[a-zA-Z0-9_-]+$ (api.deepseek.com answers
// the WHOLE request with HTTP 400 "Invalid 'tools[3].function.name': string
// does not match pattern", retryable=false — one dotted MCP tool poisons
// every request of the session, before the model is ever asked anything).
// Anthropic Messages documents the same ^[a-zA-Z0-9_-]{1,64}$ pattern; the
// Responses API matches it too (see the historical
// sanitizeResponsesFunctionName), and Google's generateContent set is a
// SUPERSET (dots allowed per the googleThoughtSigSep contract), so mapping
// dots away there is purely cosmetic.
//
// The mapping is therefore done at the PROVIDER layer, per request: internal
// tool names (the registry's, the executor's, the host's) stay verbatim, and
// only the wire spelling is adapted. toolNameMapping is built from the
// request's tool catalog and applied when serializing tool definitions and
// history tool calls, then applied in reverse when parsing the model's tool
// calls — so ToolRegistry dispatch (and every layer above it) keeps seeing
// the real names and needs no changes.

// toolNamePolicy describes the tool-name constraints one wire protocol
// imposes. Every provider path in this package uses defaultToolNamePolicy:
// the strictest common denominator of the four protocols, which keeps the
// wire spelling of a tool identical across protocols (a session that switches
// models mid-run sees one consistent catalog) and can never emit a name any
// protocol would reject. A future protocol with different rules adds its own
// policy value instead of a special case.
type toolNamePolicy struct {
	// maxLen is the wire cap for a tool name, INCLUDING any collision
	// suffix the mapper appends.
	maxLen int
	// allowed reports whether rune r may appear verbatim on the wire.
	allowed func(r rune) bool
	// fallback replaces a name that sanitizes to nothing (empty, or only
	// illegal runes).
	fallback string
}

// defaultToolNamePolicy is ^[a-zA-Z0-9_-]{1,64}$: the pattern the Chat
// Completions and Responses APIs enforce and Anthropic documents, and a
// subset of Google's legal set (Google additionally allows dots).
var defaultToolNamePolicy = toolNamePolicy{
	maxLen:  64,
	allowed: isStrictToolNameRune,
	// "unknown" mirrors the historical sanitizeResponsesFunctionName
	// fallback for an empty name.
	fallback: "unknown",
}

// isStrictToolNameRune reports whether r is legal in a
// ^[a-zA-Z0-9_-]{1,64}$ tool name.
func isStrictToolNameRune(r rune) bool {
	switch {
	case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9':
		return true
	case r == '_', r == '-':
		return true
	default:
		return false
	}
}

// toolNameHashSuffixLen is the length of the "-h<8 hex>" collision suffix
// (see toolNameHashSuffix): a constant, so the sanitizer can reserve room
// for it before appending.
const toolNameHashSuffixLen = len("-h") + 8

// sanitizeToolNameForWire maps one tool name to a policy-legal wire name:
// every illegal rune becomes '_', the result is truncated to the policy cap
// on a rune boundary (a byte cut could split a multi-byte rune and emit
// invalid UTF-8), and a name that sanitizes to nothing becomes the policy
// fallback. Pure and deterministic: the same name always yields the same
// wire name, in every request, for every catalog.
func sanitizeToolNameForWire(name string, p toolNamePolicy) string {
	return sanitizeToolNameBudget(name, p, p.maxLen)
}

// sanitizeToolNameBudget is sanitizeToolNameForWire with an explicit budget
// counted in OUTPUT runes — the mapper uses it to re-sanitize colliding
// names with room reserved for the hash suffix.
func sanitizeToolNameBudget(name string, p toolNamePolicy, budget int) string {
	if budget < 1 {
		budget = 1
	}
	var b strings.Builder
	b.Grow(len(name))
	for _, r := range name {
		// Every rune — legal or mapped — contributes exactly one output
		// rune, so the budget check is one simple counter, not a mix of
		// source-rune widths and written-byte widths.
		if b.Len() >= budget {
			break
		}
		if p.allowed(r) {
			b.WriteRune(r)
		} else {
			b.WriteByte('_')
		}
	}
	if b.Len() == 0 {
		return p.fallback
	}
	return b.String()
}

// toolNameHashSuffix returns "-h" plus the first 8 hex digits of the FNV-1a
// 64-bit hash of name — a short, deterministic, pattern-legal ([0-9a-f], '-')
// disambiguator that is a pure function of the real name, so two requests
// carrying the same catalog always map it identically. round salts the hash
// for the astronomically unlikely case of a hash collision between two
// colliding names (see newToolNameMapping's resolution loop).
func toolNameHashSuffix(name string, round int) string {
	h := fnv.New64a()
	_, _ = h.Write([]byte(name))
	_, _ = h.Write([]byte{'#'})
	_, _ = h.Write([]byte(strconv.Itoa(round)))
	s := strconv.FormatUint(h.Sum64(), 16)
	if len(s) > 8 {
		s = s[:8]
	}
	for len(s) < 8 {
		s = "0" + s
	}
	return "-h" + s
}

// toolNameMapping maps a request's real tool names to wire-legal spellings
// and back. The zero-relevant state is two small maps; instances are built
// per request from req.Tools and never shared across requests or goroutines.
//
// A nil *toolNameMapping is fully functional (see wireFor/realFor): it means
// "no tool catalog in this request" and degrades to sanitize-on-send /
// identity-on-receive, which is byte-identical to the pre-mapping behavior
// for every already-legal name.
type toolNameMapping struct {
	// forward maps real → wire for every catalog name (identity included).
	forward map[string]string
	// reverse maps wire → real for the LOSSY entries only (wire != real);
	// identity entries need no lookup.
	reverse map[string]string
}

// newToolNameMapping builds the mapping for one request's tool catalog under
// the given policy. Duplicate catalog names collapse into one entry. When
// distinct real names sanitize onto the same wire name (e.g. an MCP server's
// "ps.list" next to a legacy "ps_list"), one policy-legal name cannot carry
// both: every member of a colliding group is re-spelled as
// sanitize(real)+toolNameHashSuffix(real, round) — deterministic, catalog-
// local and reversible — and the regrouping repeats with a fresh hash salt
// until the assignment is collision-free (it terminates: each round changes
// every colliding name, and a true hash collision between two members is a
// 2^-64 event that the next round's salt resolves). logger, when non-nil,
// gets one Debug line per lossy rewrite so operators can see exactly how a
// server's names are spelled on the wire.
func newToolNameMapping(defs []ToolDefinition, p toolNamePolicy, logger *slog.Logger) *toolNameMapping {
	// Collect distinct catalog names in declaration order (order is irrelevant
	// to the result — every rewrite is a pure function of the real name — but
	// keeps the resolution loop's behavior easy to reason about).
	names := make([]string, 0, len(defs))
	seen := make(map[string]bool, len(defs))
	for _, d := range defs {
		if d.Name != "" && !seen[d.Name] {
			seen[d.Name] = true
			names = append(names, d.Name)
		}
	}
	if len(names) == 0 {
		return nil
	}

	m := &toolNameMapping{
		forward: make(map[string]string, len(names)),
		reverse: make(map[string]string),
	}
	for _, n := range names {
		m.forward[n] = sanitizeToolNameForWire(n, p)
	}

	// Collision resolution: while any wire name is claimed by two distinct
	// real names, re-spell every member of every colliding group with a hash
	// suffix (salted by the round so a residual collision cannot loop).
	for round := 0; ; round++ {
		groups := make(map[string][]string)
		for _, n := range names {
			w := m.forward[n]
			if len(groups[w]) == 0 || groups[w][len(groups[w])-1] != n {
				groups[w] = append(groups[w], n)
			}
		}
		collided := false
		for _, reals := range groups {
			if len(reals) < 2 {
				continue
			}
			collided = true
			for _, n := range reals {
				suffix := toolNameHashSuffix(n, round)
				m.forward[n] = sanitizeToolNameBudget(n, p, p.maxLen-len(suffix)) + suffix
			}
		}
		if !collided {
			break
		}
	}

	for n, w := range m.forward {
		if w != n {
			m.reverse[w] = n
			if logger != nil {
				logger.Debug("llm: tool name mapped for wire",
					"name", n, "wire", w)
			}
		}
	}
	return m
}

// wireFor maps a real tool name to the spelling to send on this request's
// wire. Catalog names map to EXACTLY the spelling the tool definitions carry
// (identity for already-legal names), so the model's tool calls round-trip.
// A name outside the catalog — a history entry whose tool has since been
// disconnected, or a hallucinated one — is sanitized directly: it stays
// wire-legal (the request cannot be poisoned by stale history) while never
// colliding with a mapped catalog name that a reverse lookup must resolve.
// A nil mapping sanitizes too, which is the identity for every legal name.
func (m *toolNameMapping) wireFor(registered string) string {
	if m != nil {
		if w, ok := m.forward[registered]; ok {
			return w
		}
	}
	return sanitizeToolNameForWire(registered, defaultToolNamePolicy)
}

// realFor maps a wire name from a model response back to the registered tool
// name to dispatch. Names the catalog mapped are reversed; everything else
// (legal catalog names, whose wire spelling IS the real name, and unknown
// names the model invented) passes through unchanged, so the executor's
// not-found tool result with its close-name suggestions stays the single
// handler for a hallucinated call — exactly the pre-mapping behavior. A nil
// mapping is the identity.
func (m *toolNameMapping) realFor(wire string) string {
	if m == nil {
		return wire
	}
	if registered, ok := m.reverse[wire]; ok {
		return registered
	}
	return wire
}
