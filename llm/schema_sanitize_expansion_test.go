package llm

import (
	"encoding/json"
	"strconv"
	"strings"
	"testing"
)

// diamondDefsSchema builds a tool schema whose $defs form the classic "JSON
// Schema $ref bomb": Dᵢ carries two properties, each a $ref to Dᵢ₊₁, bottoming
// out at a leaf definition. Each extra level costs ~90 bytes of input and —
// under naive per-site inlining — doubles the materialized output, which is
// how ~1.8 KB of input reached ~57 MB of schema in the review's reproduction.
func diamondDefsSchema(levels int) json.RawMessage {
	var b strings.Builder
	b.WriteString(`{"type":"object","properties":{"root":{"$ref":"#/$defs/D0"}},"$defs":{`)
	b.WriteString(`"D` + strconv.Itoa(levels) + `":{"type":"object","properties":{"value":{"type":"string"}}}`)
	for i := levels - 1; i >= 0; i-- {
		next := strconv.Itoa(i + 1)
		b.WriteString(`,"D` + strconv.Itoa(i) + `":{"type":"object","properties":{` +
			`"left":{"$ref":"#/$defs/D` + next + `"},` +
			`"right":{"$ref":"#/$defs/D` + next + `"}}}`)
	}
	b.WriteString(`}}`)
	return json.RawMessage(b.String())
}

// TestSanitizeSchemaForOpenAINonStrict_DiamondDefsBoundedOutput is the
// regression for the $defs expansion DoS: a small diamond-shaped $defs graph
// from an (untrusted) MCP server must sanitize to a bounded, valid,
// fully-inlined schema instead of expanding exponentially. Pre-fix this exact
// input (20 levels, ~1.8 KB) produced ~57 MB / 1.5M+ nodes of output; the
// memoized, budget-bounded inliner fail-closes it instead.
func TestSanitizeSchemaForOpenAINonStrict_DiamondDefsBoundedOutput(t *testing.T) {
	raw := diamondDefsSchema(20)

	result := SanitizeSchemaForOpenAINonStrict(raw)

	if !json.Valid(result) {
		t.Fatal("sanitizer produced invalid JSON under $defs expansion pressure")
	}
	if strings.Contains(string(result), "$ref") {
		t.Error("output still carries unresolved $ref references")
	}
	const bound = 2 * 1024 * 1024
	if len(result) > bound {
		t.Errorf("diamond expansion not bounded: got %d bytes of output, want under %d", len(result), bound)
	}
}

// TestSanitizeSchemaForAnthropic_DiamondDefsBoundedOutput checks the Anthropic
// sanitizer against the same bomb: both entry points funnel through the same
// inliner, and neither may expand it exponentially.
func TestSanitizeSchemaForAnthropic_DiamondDefsBoundedOutput(t *testing.T) {
	raw := diamondDefsSchema(16)

	result := SanitizeSchemaForAnthropic(raw)

	if !json.Valid(result) {
		t.Fatal("sanitizer produced invalid JSON under $defs expansion pressure")
	}
	if strings.Contains(string(result), "$ref") {
		t.Error("output still carries unresolved $ref references")
	}
	const bound = 2 * 1024 * 1024
	if len(result) > bound {
		t.Errorf("diamond expansion not bounded: got %d bytes of output, want under %d", len(result), bound)
	}
}

// TestResolveRefRecursive_DiamondExpandsLinearly verifies the memoization
// directly: every definition of an acyclic diamond is expanded exactly once
// (one memo entry each) and spliced at its reuse sites, and the resulting
// inlined schema is byte-for-byte what the naive per-site expansion would
// have produced — 2^levels leaf copies for a depth-`levels` diamond.
func TestResolveRefRecursive_DiamondExpandsLinearly(t *testing.T) {
	const levels = 6
	var doc map[string]any
	if err := json.Unmarshal(diamondDefsSchema(levels), &doc); err != nil {
		t.Fatalf("invalid test schema: %v", err)
	}
	defs := extractDefs(doc)

	e := &refExpander{defs: defs}
	got := e.resolveDocument(doc)

	if len(e.memo) != levels+1 {
		t.Errorf("memo holds %d expansions, want %d (one per definition: D0..D%d)", len(e.memo), levels+1, levels)
	}
	for name := range defs {
		if _, ok := e.memo[name]; !ok {
			t.Errorf("definition %q was never memoized", name)
		}
	}

	out, err := json.Marshal(got)
	if err != nil {
		t.Fatalf("marshal inlined schema: %v", err)
	}
	if strings.Contains(string(out), "$ref") {
		t.Error("inlined schema still carries $ref references")
	}
	// The leaf definition carries the only "value" property; a fully inlined
	// depth-6 diamond contains 2^6 = 64 of its copies, exactly as the naive
	// expansion would emit.
	if leafCopies := strings.Count(string(out), `"value"`); leafCopies != 1<<levels {
		t.Errorf("inlined %d leaf copies, want %d", leafCopies, 1<<levels)
	}
}

// TestResolveRefRecursive_ExpansionBudgetFailsClosed verifies the second
// defense line: an expansion whose honest output exceeds
// maxSchemaExpansionNodes (here the depth-20 diamond, ~2^20 leaf copies) is
// cut off — the ref sites beyond the budget fail closed to safeFallbackSchema,
// so work and output stay bounded while the result remains a valid schema.
func TestResolveRefRecursive_ExpansionBudgetFailsClosed(t *testing.T) {
	var doc map[string]any
	if err := json.Unmarshal(diamondDefsSchema(20), &doc); err != nil {
		t.Fatalf("invalid test schema: %v", err)
	}
	defs := extractDefs(doc)

	e := &refExpander{defs: defs}
	got := e.resolveDocument(doc)

	if e.nodes <= maxSchemaExpansionNodes {
		t.Fatalf("expansion weight %d never reached the budget of %d; the fail-closed path was not exercised", e.nodes, maxSchemaExpansionNodes)
	}
	if got == nil {
		t.Fatal("budget overrun produced a nil schema")
	}
	out, err := json.Marshal(got)
	if err != nil {
		t.Fatalf("marshal fail-closed schema: %v", err)
	}
	if !json.Valid(out) {
		t.Error("fail-closed output is not valid JSON")
	}
	if strings.Contains(string(out), "$ref") {
		t.Error("fail-closed output still carries $ref references")
	}
	const bound = 2 * 1024 * 1024
	if len(out) > bound {
		t.Errorf("fail-closed output not bounded: got %d bytes, want under %d", len(out), bound)
	}
}

// TestResolveRefRecursive_CycleStillBreaks pins the cycle semantics the
// memoization must not disturb: a definition chain that loops back onto
// itself terminates with the safe fallback instead of recursing forever.
func TestResolveRefRecursive_CycleStillBreaks(t *testing.T) {
	var doc map[string]any
	schema := `{"type":"object","properties":{"a":{"$ref":"#/$defs/A"}},"$defs":{` +
		`"A":{"type":"object","properties":{"next":{"$ref":"#/$defs/B"}}},` +
		`"B":{"type":"object","properties":{"next":{"$ref":"#/$defs/C"}}},` +
		`"C":{"type":"object","properties":{"next":{"$ref":"#/$defs/A"}}}}}`
	if err := json.Unmarshal([]byte(schema), &doc); err != nil {
		t.Fatalf("invalid test schema: %v", err)
	}
	defs := extractDefs(doc)

	e := &refExpander{defs: defs}
	got := e.resolveDocument(doc)

	out, err := json.Marshal(got)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if strings.Count(string(out), `"next"`) > 3 {
		t.Errorf("cycle did not terminate after three hops: %s", out)
	}
	if len(e.memo) != 3 {
		t.Errorf("memo holds %d expansions, want 3 (A, B, C)", len(e.memo))
	}
}
