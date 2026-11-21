package llm

import (
	"encoding/json"
	"regexp"
	"strings"
	"testing"
	"unicode/utf8"
)

// strictToolNameRe is the wire pattern the mapping must never violate.
var strictToolNameRe = regexp.MustCompile(`^[a-zA-Z0-9_-]{1,64}$`)

func TestSanitizeToolNameForWire(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{name: "legal name is identity", input: "bash_exec", want: "bash_exec"},
		{name: "legal name with dash and digits", input: "mcp-server-2", want: "mcp-server-2"},
		{name: "dot becomes underscore", input: "ps.list", want: "ps_list"},
		{name: "multiple dots each mapped", input: "app.launch", want: "app_launch"},
		{name: "space and punctuation mapped", input: "weird name!", want: "weird_name_"},
		{name: "cyrillic runes mapped per rune", input: "инструмент", want: "__________"},
		{name: "empty name falls back", input: "", want: "unknown"},
		{name: "illegal-only name falls back", input: "!!!", want: "___"},
		{
			name:  "long legal name truncated to cap",
			input: strings.Repeat("a", 70),
			want:  strings.Repeat("a", 64),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := sanitizeToolNameForWire(tt.input, defaultToolNamePolicy)
			if got != tt.want {
				t.Errorf("sanitizeToolNameForWire(%q) = %q, want %q", tt.input, got, tt.want)
			}
			if !strictToolNameRe.MatchString(got) {
				t.Errorf("sanitizeToolNameForWire(%q) = %q, want match %v", tt.input, got, strictToolNameRe)
			}
		})
	}
}

func TestSanitizeToolNameForWire_TruncationStaysValidUTF8(t *testing.T) {
	// 70 multi-byte runes exceed the 64-rune cap: truncation must cut on a
	// rune boundary (a byte cut would split the last rune into invalid UTF-8
	// and produce a different wire name after re-encoding).
	input := strings.Repeat("ж", 70)
	got := sanitizeToolNameForWire(input, defaultToolNamePolicy)
	if !utf8.ValidString(got) {
		t.Errorf("sanitizeToolNameForWire(%q) = %q, want valid UTF-8", input, got)
	}
	if runes := utf8.RuneCountInString(got); runes != 64 {
		t.Errorf("sanitizeToolNameForWire(%q) rune count = %d, want 64", input, runes)
	}
}

func TestToolNameHashSuffix(t *testing.T) {
	tests := []struct {
		name  string
		input string
		round int
	}{
		{name: "round zero", input: "ps.list", round: 0},
		{name: "same name next round", input: "ps.list", round: 1},
		{name: "other name", input: "ps_list", round: 0},
	}
	seen := make(map[string]string, len(tests))
	for _, tt := range tests {
		got := toolNameHashSuffix(tt.input, tt.round)
		if len(got) != toolNameHashSuffixLen {
			t.Errorf("toolNameHashSuffix(%q, %d) length = %d, want %d", tt.input, tt.round, len(got), toolNameHashSuffixLen)
		}
		if !strictToolNameRe.MatchString(got) {
			t.Errorf("toolNameHashSuffix(%q, %d) = %q, want pattern-legal suffix", tt.input, tt.round, got)
		}
		if prev, dup := seen[got]; dup {
			t.Errorf("toolNameHashSuffix(%q, %d) = %q collides with earlier (%q) result %q", tt.input, tt.round, got, prev, got)
		}
		seen[got] = tt.input
	}
	// Deterministic: the same inputs always produce the same suffix.
	if got, want := toolNameHashSuffix("ps.list", 0), toolNameHashSuffix("ps.list", 0); got != want {
		t.Errorf("toolNameHashSuffix is not deterministic: %q != %q", got, want)
	}
}

func TestNewToolNameMapping_NilForEmptyCatalog(t *testing.T) {
	if got := newToolNameMapping(nil, defaultToolNamePolicy, nil); got != nil {
		t.Errorf("newToolNameMapping(nil) = %v, want nil", got)
	}
	if got := newToolNameMapping([]ToolDefinition{{Name: "x"}}, defaultToolNamePolicy, nil); got == nil {
		t.Error("newToolNameMapping(one tool) = nil, want non-nil mapping")
	}
}

func TestToolNameMapping_NilMappingSemantics(t *testing.T) {
	var m *toolNameMapping
	if got, want := m.wireFor("ps.list"), "ps_list"; got != want {
		t.Errorf("nil mapping wireFor(%q) = %q, want %q (sanitize fallback)", "ps.list", got, want)
	}
	if got, want := m.wireFor("bash_exec"), "bash_exec"; got != want {
		t.Errorf("nil mapping wireFor(%q) = %q, want %q (identity for legal names)", "bash_exec", got, want)
	}
	if got, want := m.realFor("ps_list"), "ps_list"; got != want {
		t.Errorf("nil mapping realFor(%q) = %q, want %q (identity)", "ps_list", got, want)
	}
}

func TestToolNameMapping_IdentityCatalog(t *testing.T) {
	defs := []ToolDefinition{
		{Name: "bash_exec"},
		{Name: "semantic_search"},
		{Name: "bash_exec"}, // duplicate collapses, still no collision suffix
	}
	m := newToolNameMapping(defs, defaultToolNamePolicy, nil)
	if m == nil {
		t.Fatal("newToolNameMapping(legal names) = nil, want mapping")
	}
	for _, n := range []string{"bash_exec", "semantic_search"} {
		if got := m.wireFor(n); got != n {
			t.Errorf("wireFor(%q) = %q, want %q (identity for legal catalog names)", n, got, n)
		}
		if got := m.realFor(n); got != n {
			t.Errorf("realFor(%q) = %q, want %q", n, got, n)
		}
	}
	if len(m.reverse) != 0 {
		t.Errorf("identity catalog reverse map = %v, want empty", m.reverse)
	}
}

func TestToolNameMapping_DottedNameRoundTrip(t *testing.T) {
	defs := []ToolDefinition{
		{Name: "ps.list"},
		{Name: "app.launch"},
		{Name: "bash_exec"},
	}
	m := newToolNameMapping(defs, defaultToolNamePolicy, nil)
	if m == nil {
		t.Fatal("newToolNameMapping(dotted names) = nil, want mapping")
	}
	if got, want := m.wireFor("ps.list"), "ps_list"; got != want {
		t.Errorf("wireFor(%q) = %q, want %q", "ps.list", got, want)
	}
	if got, want := m.wireFor("bash_exec"), "bash_exec"; got != want {
		t.Errorf("wireFor(%q) = %q, want %q", "bash_exec", got, want)
	}
	if got, want := m.realFor("ps_list"), "ps.list"; got != want {
		t.Errorf("realFor(%q) = %q, want %q", "ps_list", got, want)
	}
	// A name that was never in the catalog does not resolve to a real tool.
	if got, want := m.realFor("totally_unknown"), "totally_unknown"; got != want {
		t.Errorf("realFor(%q) = %q, want %q (unknown names pass through)", "totally_unknown", got, want)
	}
	// The sanitized spelling of an out-of-catalog dotted name must NOT
	// resolve through the reverse map: "other.list" sanitizes to
	// "other_list", which no catalog entry claims.
	if got := m.wireFor("other.list"); got != "other_list" {
		t.Errorf("wireFor(%q) = %q, want %q", "other.list", got, "other_list")
	}
}

func TestToolNameMapping_CollisionGetsDeterministicSuffixes(t *testing.T) {
	defs := []ToolDefinition{
		{Name: "ps.list"}, // → "ps_list"
		{Name: "ps_list"}, // → "ps_list" (identity) — collides with the above
		{Name: "unrelated"},
	}
	m := newToolNameMapping(defs, defaultToolNamePolicy, nil)
	if m == nil {
		t.Fatal("newToolNameMapping(colliding names) = nil, want mapping")
	}

	wire1, wire2 := m.wireFor("ps.list"), m.wireFor("ps_list")
	if wire1 == wire2 {
		t.Fatalf("colliding names mapped to the same wire name %q; want distinct spellings", wire1)
	}
	for _, w := range []string{wire1, wire2} {
		if !strictToolNameRe.MatchString(w) {
			t.Errorf("wire name %q does not match %v", w, strictToolNameRe)
		}
	}
	if got, want := m.wireFor("unrelated"), "unrelated"; got != want {
		t.Errorf("wireFor(%q) = %q, want %q (non-colliding name untouched)", "unrelated", got, want)
	}
	// Both spellings must reverse to their real names.
	if got, want := m.realFor(wire1), "ps.list"; got != want {
		t.Errorf("realFor(%q) = %q, want %q", wire1, got, want)
	}
	if got, want := m.realFor(wire2), "ps_list"; got != want {
		t.Errorf("realFor(%q) = %q, want %q", wire2, got, want)
	}
	// Determinism: a mapping rebuilt from the same catalog assigns identical
	// wire names — required so history tool calls map consistently across
	// requests of one session.
	again := newToolNameMapping(defs, defaultToolNamePolicy, nil)
	for real, want := range m.forward {
		if got := again.wireFor(real); got != want {
			t.Errorf("rebuilt mapping wireFor(%q) = %q, want %q", real, got, want)
		}
	}
}

func TestToolNameMapping_CollisionSuffixRespectsCap(t *testing.T) {
	longA := strings.Repeat("a", 80) + ".tool"
	longB := strings.Repeat("a", 80) + "_tool" // sanitizes onto the same wire name
	defs := []ToolDefinition{{Name: longA}, {Name: longB}}
	m := newToolNameMapping(defs, defaultToolNamePolicy, nil)
	for _, real := range []string{longA, longB} {
		got := m.wireFor(real)
		if !strictToolNameRe.MatchString(got) {
			t.Errorf("wireFor(%q...) = %q, want pattern-legal name within 64 chars", longA[:10], got)
		}
	}
	if got, want := m.wireFor(longA), m.wireFor(longB); got == want {
		t.Errorf("long colliding names mapped to the same wire name %q, want distinct", got)
	}
}

func TestToolNameMapping_WireForIsExactlyTheDefinitionsSpelling(t *testing.T) {
	// The contract the providers rely on: the name wireFor produces for a
	// catalog tool is byte-identical to the name serialized into the tool
	// definitions, so a model echoing a catalog name always reverse-maps.
	defs := []ToolDefinition{
		{Name: "ps.list"},
		{Name: "ps_list"},
		{Name: "notes.v2/draft"},
	}
	m := newToolNameMapping(defs, defaultToolNamePolicy, nil)
	spelling := make(map[string]string, len(defs))
	for _, d := range defs {
		spelling[d.Name] = m.wireFor(d.Name)
	}
	for real, wire := range spelling {
		if got := m.realFor(wire); got != real {
			t.Errorf("realFor(wireFor(%q) = %q) = %q, want %q", real, wire, got, real)
		}
	}
}

func TestToolNameMapping_DefsWithoutNames(t *testing.T) {
	// Tools without a schema or with an empty name must not crash the mapper.
	defs := []ToolDefinition{
		{Name: "real_tool", InputSchema: json.RawMessage(`{"type":"object"}`)},
		{Description: "nameless"},
	}
	m := newToolNameMapping(defs, defaultToolNamePolicy, nil)
	if m == nil {
		t.Fatal("newToolNameMapping(nameless defs) = nil, want mapping for the named entry")
	}
	if got, want := m.wireFor("real_tool"), "real_tool"; got != want {
		t.Errorf("wireFor(%q) = %q, want %q", "real_tool", got, want)
	}
}
