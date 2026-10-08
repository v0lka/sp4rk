package llm

import (
	"encoding/json"
	"sort"
)

// getTypes extracts type values as a slice of strings.
func getTypes(typeVal any) []string {
	if typeVal == nil {
		return nil
	}

	switch v := typeVal.(type) {
	case string:
		return []string{v}
	case []any:
		types := make([]string, 0, len(v))
		for _, t := range v {
			if s, ok := t.(string); ok {
				types = append(types, s)
			}
		}
		return types
	default:
		return nil
	}
}

// SanitizeSchemaForOpenAI ensures strict mode compliance for OpenAI.
//
// Use it only for request paths that actually enable OpenAI strict function
// calling (the Responses API sets strict=true). For ordinary Chat Completions
// requests, where strict mode is not enabled, use
// SanitizeSchemaForOpenAINonStrict instead: forcing every property into
// required there makes optional parameters look mandatory and increases the
// chance of malformed argument JSON.
//   - Resolves $ref references against $defs/definitions
//   - Filters out forbidden JSON Schema keywords ($schema, $id, $comment, $defs, definitions, default, examples)
//   - Infers "type": "object" when properties or required are present but type is missing
//   - Adds "additionalProperties": false to all object-type schemas (recursively)
//   - Ensures the required array contains ALL property names from properties
//   - Removes required entries that reference non-existent properties
func SanitizeSchemaForOpenAI(raw json.RawMessage) json.RawMessage {
	return sanitizeSchemaForOpenAI(raw, true)
}

// SanitizeSchemaForOpenAINonStrict normalizes a JSON Schema for OpenAI function
// calling when strict mode is NOT enabled (e.g. Chat Completions, and
// OpenAI-compatible gateways). It applies the same compatibility fixes as
// SanitizeSchemaForOpenAI with one deliberate difference: it does not force
// every property into the required array. Only the required entries declared by
// the caller are kept (filtered to properties that actually exist), so optional
// parameters — such as bash_exec's "timeout" and "working_directory" — remain
// genuinely optional. Keeping them optional means the model is not pressured to
// emit every field on every call, which reduces schema-induced argument errors.
//   - Resolves $ref references against $defs/definitions
//   - Filters out forbidden JSON Schema keywords ($schema, $id, $comment, $defs, definitions, default, examples)
//   - Infers "type": "object" when properties or required are present but type is missing
//   - Adds "additionalProperties": false to all object-type schemas (recursively)
//   - Preserves the caller's required list, dropping only entries that reference non-existent properties
func SanitizeSchemaForOpenAINonStrict(raw json.RawMessage) json.RawMessage {
	return sanitizeSchemaForOpenAI(raw, false)
}

// sanitizeSchemaForOpenAI implements the shared OpenAI schema normalization.
// When strict is true it enforces OpenAI strict-mode constraints (every
// property listed in required); when false it preserves the caller's
// optionality (see SanitizeSchemaForOpenAINonStrict).
func sanitizeSchemaForOpenAI(raw json.RawMessage, strict bool) json.RawMessage {
	if len(raw) == 0 {
		return raw
	}

	var schema map[string]any
	if err := json.Unmarshal(raw, &schema); err != nil {
		return raw
	}

	defs := extractDefs(schema)
	sanitized := sanitizeOpenAISchemaWithDefs(schema, defs, strict)
	return marshalPreservingKeyOrder(raw, sanitized)
}

// extractDefs pulls $defs or definitions from the schema for $ref resolution.
func extractDefs(schema map[string]any) map[string]any {
	if d, ok := schema["$defs"].(map[string]any); ok {
		return d
	}
	if d, ok := schema["definitions"].(map[string]any); ok {
		return d
	}
	return nil
}

// resolveRef replaces a $ref schema with the referenced definition.
// If the $ref target is not found, returns a safe strict-mode fallback.
func resolveRef(schema, defs map[string]any) map[string]any {
	ref, ok := schema["$ref"].(string)
	if !ok || defs == nil {
		return schema
	}

	// Parse JSON Pointer: "#/$defs/Name" or "#/definitions/Name"
	parts := splitRefPath(ref)
	if len(parts) == 0 {
		return safeFallbackSchema()
	}
	name := parts[len(parts)-1]

	defRaw, found := defs[name]
	if !found {
		return safeFallbackSchema()
	}
	defMap, ok := defRaw.(map[string]any)
	if !ok {
		return safeFallbackSchema()
	}

	// Return a copy so we don't mutate the definitions
	cp := make(map[string]any, len(defMap))
	for k, v := range defMap {
		cp[k] = v
	}
	return cp
}

// maxSchemaExpansionNodes caps the expansion weight a single sanitization may
// spend while inlining $ref definitions (see refExpander.charge). A malicious
// or compromised MCP server can hand a tool schema whose $defs graph is a
// deep "diamond" chain — every definition referenced by two sibling
// properties, each sibling inlining it independently. Naive per-site
// expansion turns ~2 KB of input into millions of materialized nodes and
// gigabytes of allocations (OOM or effective hang) on every request build and
// token estimate. Two defenses combine here:
//
//   - Memoization: every definition is fully expanded at most once per
//     document (refExpander.memo) and spliced by reference at reuse sites, so
//     an acyclic diamond DAG costs linear work instead of exponential.
//   - The weight budget: a reuse site is charged the FULL weight of the
//     spliced subtree — shared subtrees still serialize once per occurrence —
//     so the output size of any schema, diamond or not, is bounded.
//     Exceeding the budget fail-closes the offending $ref to
//     safeFallbackSchema.
//
// 50,000 weight units (1 per object node and array element, 1 per 64 bytes
// of string content) sits orders of magnitude above any legitimate tool
// schema while capping the worst case at a couple of megabytes and
// milliseconds.
const maxSchemaExpansionNodes = 50_000

// refExpansion is a fully inlined definition: schema contains no unresolved
// $ref anywhere; names is every definition name transitively inlined inside
// it (including the definition itself); breaks is the subset of those names
// that sat on the ancestor path the expansion was computed with — exactly the
// references that were cycle-broken into the safe fallback; and nodes is the
// expansion's charged weight, splice multiplicity included.
//
// (names, breaks) make a memo entry safe to REUSE across contexts: an
// expansion computed under ancestor path A is reproduced verbatim at a site
// whose path is B precisely when A and B intersect the expansion's reachable
// names in the same set — see refExpander.canSplice.
type refExpansion struct {
	schema map[string]any
	names  map[string]struct{}
	breaks map[string]struct{}
	nodes  int
}

// refExpander inlines $ref references against one document's $defs /
// definitions table. One expander is created per resolveRefRecursive call,
// which the sanitizers invoke once at the document root, so the memo and the
// weight budget cover the whole document.
type refExpander struct {
	defs  map[string]any
	memo  map[string]refExpansion
	names map[string]struct{} // reachable names of the expansion being built
	nodes int                 // weight charged so far, capped by maxSchemaExpansionNodes
}

// resolveRefRecursive replaces $ref with the referenced definition and
// recursively inlines every nested $ref inside the resolved content, breaking
// reference cycles with safeFallbackSchema and bounding the total expansion
// with maxSchemaExpansionNodes (fail-closed). A definition referenced several
// times is expanded once and spliced at every reuse site, so a diamond-shaped
// $defs graph inlines in linear time — producing exactly the fully inlined
// schema the naive per-site expansion would — until the weight budget
// fail-closes the offending references.
func resolveRefRecursive(schema, defs map[string]any) map[string]any {
	e := &refExpander{defs: defs}
	return e.resolveDocument(schema)
}

// resolveDocument inlines a whole document: the definition table is consumed
// through e.defs and must never be inlined into the walk — cloning it would
// re-expand every definition in place (doubling the work and draining the
// budget on unreferenced entries) for keys the sanitizers filter from the
// result anyway. A root that is itself a $ref keeps the table: the reference
// branch replaces the whole map, table included, exactly as the shallow
// resolver always did.
func (e *refExpander) resolveDocument(schema map[string]any) map[string]any {
	if _, ok := schema["$ref"]; !ok {
		for _, key := range []string{"$defs", "definitions"} {
			if _, has := schema[key]; has {
				cp := make(map[string]any, len(schema)-1)
				for k, v := range schema {
					if k == "$defs" || k == "definitions" {
						continue
					}
					cp[k] = v
				}
				schema = cp
				break
			}
		}
	}
	return e.cloneSchema(schema, nil)
}

// cloneSchema inlines schema into the expansion output. A map carrying a
// resolvable "$ref" is not copied but REPLACED with the referenced
// definition's inlined expansion (sibling keys of the $ref site are dropped,
// exactly as the shallow resolveRef always did); anything else is rebuilt key
// by key through cloneValue, so an expansion owns its whole tree and its
// charged weight reflects its true output size.
func (e *refExpander) cloneSchema(schema map[string]any, visited map[string]struct{}) map[string]any {
	if schema == nil {
		return nil
	}
	if ref, ok := schema["$ref"].(string); ok && e.defs != nil {
		parts := splitRefPath(ref)
		if len(parts) == 0 {
			// Unparseable reference path ("#", an external URL): the naive
			// resolver fail-closed to the safe fallback schema.
			return e.fallbackCharged()
		}
		return e.resolveRefName(parts[len(parts)-1], visited)
	}
	cp := make(map[string]any, len(schema))
	e.charge(1)
	for key, val := range schema {
		cp[key] = e.cloneValue(val, visited)
	}
	return cp
}

// cloneValue rebuilds a schema value, charging the budget per node: objects
// count 1, arrays 1 per element plus 1, string content 1 per 64 bytes.
func (e *refExpander) cloneValue(v any, visited map[string]struct{}) any {
	switch t := v.(type) {
	case map[string]any:
		return e.cloneSchema(t, visited)
	case []any:
		e.charge(1 + len(t))
		cp := make([]any, len(t))
		for i, item := range t {
			cp[i] = e.cloneValue(item, visited)
		}
		return cp
	case string:
		e.charge((len(t) + 63) / 64)
		return t
	default:
		e.charge(1)
		return v
	}
}

// resolveRefName inlines the definition addressed by a "$ref" path's final
// segment. A back-reference to a definition on the current ancestor path
// (visited) is a cycle and fails closed; a memoized, path-independent
// expansion is spliced; anything else starts a fresh expansion. The three
// outcomes mirror the naive resolver's — full definition, cycle-broken
// fallback — with the memo in front.
func (e *refExpander) resolveRefName(name string, visited map[string]struct{}) map[string]any {
	if _, seen := visited[name]; seen {
		return e.fallbackCharged()
	}
	if exp, ok := e.memo[name]; ok && e.canSplice(exp, visited) {
		// Memoized splice, provably exact: the expansion cycle-broke its
		// internal references to exactly exp.breaks, and the ancestor path
		// reaching this site would break the same set and nothing else —
		// so the expansion is independent of the path that reached here.
		e.recordNames(exp.names)
		if !e.charge(exp.nodes) {
			return e.fallbackCharged()
		}
		return exp.schema
	}
	if e.nodes >= maxSchemaExpansionNodes {
		// Budget exhausted: fail closed instead of starting another expansion.
		return e.fallbackCharged()
	}
	visited = copyVisitedSet(visited)
	visited[name] = struct{}{}
	return e.expandDefinition(name, visited)
}

// expandDefinition fully inlines the definition `name`, whose target has
// already been added to visited on the caller's private copy. The body is
// rebuilt through cloneSchema/cloneValue, so nested refs recurse via
// resolveRefName and memoized subtrees splice rather than recompute. The
// finished expansion is memoized (first computation wins) together with its
// reachable-name set and the cycle breaks its ancestor path forced: those two
// sets are what let canSplice decide, at any later reuse site, whether the
// cached tree is exactly what a fresh expansion under that site's own path
// would produce. When it is not, the site re-expands under its own context —
// bounded, like everything else, by the weight budget.
func (e *refExpander) expandDefinition(name string, visited map[string]struct{}) map[string]any {
	savedNames := e.names
	savedNodes := e.nodes
	e.names = make(map[string]struct{})
	e.names[name] = struct{}{}

	var resolved map[string]any
	if defMap, ok := e.defs[name].(map[string]any); ok {
		resolved = e.cloneSchema(defMap, visited)
	} else {
		// Missing target or a non-object definition (JSON Schema allows
		// boolean schemas): the naive resolver's safe fallback.
		resolved = e.fallbackCharged()
	}

	names := e.names
	e.names = savedNames
	if savedNames != nil {
		// Merge this expansion's reachable names into the parent's recorder,
		// so a memoized parent lists every name its tree transitively inlines.
		for n := range names {
			savedNames[n] = struct{}{}
		}
	}
	if _, ok := e.memo[name]; !ok {
		// First expansion of this definition: record it as the canonical
		// entry, annotated with the breaks its ancestor path forced.
		breaks := make(map[string]struct{})
		for n := range visited {
			if _, inTree := names[n]; inTree {
				breaks[n] = struct{}{}
			}
		}
		if e.memo == nil {
			e.memo = make(map[string]refExpansion)
		}
		e.memo[name] = refExpansion{
			schema: resolved,
			names:  names,
			breaks: breaks,
			nodes:  e.nodes - savedNodes,
		}
	}
	return resolved
}

// charge adds n to the expansion weight and reports whether the document is
// still within maxSchemaExpansionNodes. The counter keeps growing past the
// budget so later decisions see the overrun; ref-resolution sites consult the
// return value (or the counter directly) to fail closed.
func (e *refExpander) charge(n int) bool {
	e.nodes += n
	return e.nodes <= maxSchemaExpansionNodes
}

// fallbackCharged returns a fresh safeFallbackSchema — the fail-closed
// replacement for a reference that cannot be inlined (cycle, missing target,
// non-object definition, budget exhaustion) — and charges its weight. Fresh
// per call: sharing one instance would alias a single map into every
// fallback site of the output.
func (e *refExpander) fallbackCharged() map[string]any {
	e.charge(4)
	return safeFallbackSchema()
}

// recordNames unions names into the recorder of the expansion currently being
// built, so its memo entry lists every name its tree transitively inlines.
func (e *refExpander) recordNames(names map[string]struct{}) {
	if e.names == nil {
		e.names = make(map[string]struct{}, len(names))
	}
	for n := range names {
		e.names[n] = struct{}{}
	}
}

// canSplice reports whether the memoized expansion exp may be spliced
// verbatim at a site whose ancestor path is visited. The expansion broke its
// internal references to exactly exp.breaks; the splice is exact when the
// site's own path intersects the expansion's reachable names in that same
// set — no name the expansion inlined would now have to break, and no break
// the expansion made would now be skipped. Structural induction over the
// expansion tree: with equal break sets, every subtree recursion sees the
// same context, so the two expansions coincide.
func (e *refExpander) canSplice(exp refExpansion, visited map[string]struct{}) bool {
	matched := 0
	for n := range visited {
		if _, inTree := exp.names[n]; inTree {
			if _, broken := exp.breaks[n]; !broken {
				return false
			}
			matched++
		}
	}
	return matched == len(exp.breaks)
}

// copyVisitedSet returns a shallow copy of a cycle-detection visited set.
func copyVisitedSet(visited map[string]struct{}) map[string]struct{} {
	cp := make(map[string]struct{}, len(visited)+1)
	for k := range visited {
		cp[k] = struct{}{}
	}
	return cp
}

// splitRefPath splits a $ref string like "#/$defs/Foo" into path segments after "#".
func splitRefPath(ref string) []string {
	// Trim leading "#/"
	if len(ref) < 2 || ref[0] != '#' {
		return nil
	}
	trimmed := ref[1:]
	if trimmed != "" && trimmed[0] == '/' {
		trimmed = trimmed[1:]
	}
	if trimmed == "" {
		return nil
	}

	// Split on "/"
	var parts []string
	start := 0
	for i := 0; i < len(trimmed); i++ {
		if trimmed[i] == '/' {
			if i > start {
				parts = append(parts, trimmed[start:i])
			}
			start = i + 1
		}
	}
	if start < len(trimmed) {
		parts = append(parts, trimmed[start:])
	}
	return parts
}

// safeFallbackSchema returns a minimal strict-mode-compatible object schema.
func safeFallbackSchema() map[string]any {
	return map[string]any{
		"type":                 "object",
		"properties":           map[string]any{},
		"required":             []any{},
		"additionalProperties": false,
	}
}

// sanitizeOpenAISchemaWithDefs recursively processes a schema map for OpenAI
// function calling, carrying top-level definitions for $ref resolution. When
// strict is true it enforces strict-mode required semantics; when false it
// preserves the caller's optional parameters.
func sanitizeOpenAISchemaWithDefs(schema, defs map[string]any, strict bool) map[string]any {
	if schema == nil {
		return nil
	}

	// Resolve $ref before processing, including nested $ref chains
	schema = resolveRefRecursive(schema, defs)

	// Copy keys, filtering out forbidden keywords
	result := make(map[string]any)
	for key, value := range schema {
		switch key {
		case "$schema", "$id", "$comment", "$defs", "definitions", "default", "examples":
			continue
		default:
			result[key] = value
		}
	}

	// Infer object type if type is missing/empty but has object indicators
	typeVal := result["type"]
	types := getTypes(typeVal)
	_, hasProperties := result["properties"]
	_, hasRequired := result["required"]

	if (len(types) == 0 || (len(types) == 1 && types[0] == "")) && (hasProperties || hasRequired) {
		result["type"] = "object"
		types = []string{"object"}
	}
	// Clean up empty type string
	if len(types) == 1 && types[0] == "" {
		delete(result, "type")
	}

	isObjectType := len(types) == 1 && types[0] == "object"

	// For object types, enforce strict mode constraints
	if isObjectType {
		// ALWAYS force additionalProperties to false
		result["additionalProperties"] = false

		// Ensure properties exists
		if _, hasProps := result["properties"].(map[string]any); !hasProps {
			result["properties"] = map[string]any{}
		}

		// Normalize the required array. Behavior depends on strict mode:
		//   - strict: the array MUST list every property (OpenAI strict requirement).
		//     Phase 1 filters out required entries that don't exist in properties;
		//     Phase 2 adds every property name missing from required.
		//   - non-strict: preserve only the caller-declared required entries that
		//     reference existing properties. Missing properties are NOT promoted, so
		//     optional parameters stay optional.
		props, propsOK := result["properties"].(map[string]any)
		if strict {
			if propsOK && len(props) > 0 {
				// Phase 1: keep only valid existing required entries
				requiredSet := make(map[string]struct{})
				if required, ok := result["required"].([]any); ok {
					for _, req := range required {
						if reqStr, ok := req.(string); ok {
							if _, exists := props[reqStr]; exists {
								requiredSet[reqStr] = struct{}{}
							}
						}
					}
				}

				// Phase 2: add any missing property names
				for propName := range props {
					requiredSet[propName] = struct{}{}
				}

				// Build sorted required list for deterministic output
				sortedNames := make([]string, 0, len(requiredSet))
				for name := range requiredSet {
					sortedNames = append(sortedNames, name)
				}
				sort.Strings(sortedNames)
				allRequired := make([]any, len(sortedNames))
				for i, name := range sortedNames {
					allRequired[i] = name
				}
				result["required"] = allRequired
			} else {
				// No properties or empty properties: strict mode still requires the array
				result["required"] = []any{}
			}
		} else if required, ok := result["required"].([]any); ok {
			// Keep only required entries that reference existing properties; never
			// promote an optional property to required.
			requiredSet := make(map[string]struct{})
			for _, req := range required {
				if reqStr, ok := req.(string); ok {
					if _, exists := props[reqStr]; exists {
						requiredSet[reqStr] = struct{}{}
					}
				}
			}
			sortedNames := make([]string, 0, len(requiredSet))
			for name := range requiredSet {
				sortedNames = append(sortedNames, name)
			}
			sort.Strings(sortedNames)
			kept := make([]any, len(sortedNames))
			for i, name := range sortedNames {
				kept[i] = name
			}
			result["required"] = kept
		}
	}

	// Process nested objects in properties
	if props, ok := result["properties"].(map[string]any); ok {
		newProps := make(map[string]any)
		for propName, propVal := range props {
			if propMap, ok := propVal.(map[string]any); ok {
				newProps[propName] = sanitizeOpenAISchemaWithDefs(propMap, defs, strict)
			} else {
				newProps[propName] = propVal
			}
		}
		result["properties"] = newProps
	}

	// Process items for array type
	if items, ok := result["items"]; ok {
		switch v := items.(type) {
		case map[string]any:
			result["items"] = sanitizeOpenAISchemaWithDefs(v, defs, strict)
		case []any:
			newItems := make([]any, len(v))
			for i, item := range v {
				if itemMap, ok := item.(map[string]any); ok {
					newItems[i] = sanitizeOpenAISchemaWithDefs(itemMap, defs, strict)
				} else {
					newItems[i] = item
				}
			}
			result["items"] = newItems
		}
	}

	// Process anyOf, oneOf, allOf recursively
	for _, key := range []string{"anyOf", "oneOf", "allOf"} {
		if arr, ok := result[key].([]any); ok {
			newArr := make([]any, len(arr))
			for i, item := range arr {
				if itemMap, ok := item.(map[string]any); ok {
					newArr[i] = sanitizeOpenAISchemaWithDefs(itemMap, defs, strict)
				} else {
					newArr[i] = item
				}
			}
			result[key] = newArr
		}
	}

	return result
}

// SanitizeSchemaForAnthropic normalizes JSON Schema for Anthropic API compatibility.
// Anthropic has specific requirements:
//   - Unsupported keywords ($schema, $id, $comment, patternProperties) are removed
//   - $ref references are resolved inline from $defs/definitions
//   - $defs/definitions are removed after resolution
//   - Top-level allOf single item is unwrapped; multiple items are merged
//   - Top-level oneOf/anyOf picks the first variant
//   - Type "object" is inferred when properties/required present but type missing
//   - Recursively processes properties, items, and additionalProperties
func SanitizeSchemaForAnthropic(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 {
		return raw
	}

	var schema map[string]any
	if err := json.Unmarshal(raw, &schema); err != nil {
		return raw
	}

	defs := extractDefs(schema)
	sanitized := sanitizeAnthropicSchema(schema, defs)
	return marshalPreservingKeyOrder(raw, sanitized)
}

// sanitizeAnthropicSchema recursively processes a schema map for Anthropic compatibility.
func sanitizeAnthropicSchema(schema, defs map[string]any) map[string]any {
	if schema == nil {
		return nil
	}

	// Resolve $ref before processing, including nested $ref chains
	schema = resolveRefRecursive(schema, defs)

	// Flatten top-level composition keywords before filtering
	schema = flattenAnthropicComposition(schema, defs)

	// Copy keys, filtering out unsupported keywords
	result := make(map[string]any)
	for key, value := range schema {
		switch key {
		case "$schema", "$id", "$comment", "$defs", "definitions", "patternProperties", "$ref":
			continue
		default:
			result[key] = value
		}
	}

	// Infer object type if type is missing but has object indicators
	typeVal := result["type"]
	types := getTypes(typeVal)
	_, hasProperties := result["properties"]
	_, hasRequired := result["required"]

	if len(types) == 0 && (hasProperties || hasRequired) {
		result["type"] = "object"
	}

	// Process nested objects in properties
	if props, ok := result["properties"].(map[string]any); ok {
		newProps := make(map[string]any)
		for propName, propVal := range props {
			if propMap, ok := propVal.(map[string]any); ok {
				newProps[propName] = sanitizeAnthropicSchema(propMap, defs)
			} else {
				newProps[propName] = propVal
			}
		}
		result["properties"] = newProps
	}

	// Process items for array type
	if items, ok := result["items"]; ok {
		switch v := items.(type) {
		case map[string]any:
			result["items"] = sanitizeAnthropicSchema(v, defs)
		case []any:
			newItems := make([]any, len(v))
			for i, item := range v {
				if itemMap, ok := item.(map[string]any); ok {
					newItems[i] = sanitizeAnthropicSchema(itemMap, defs)
				} else {
					newItems[i] = item
				}
			}
			result["items"] = newItems
		}
	}

	// Process additionalProperties if it's a schema object
	if addProps, ok := result["additionalProperties"].(map[string]any); ok {
		result["additionalProperties"] = sanitizeAnthropicSchema(addProps, defs)
	}

	// Process anyOf, oneOf, allOf recursively (nested, non-top-level)
	for _, key := range []string{"anyOf", "oneOf", "allOf"} {
		if arr, ok := result[key].([]any); ok {
			newArr := make([]any, len(arr))
			for i, item := range arr {
				if itemMap, ok := item.(map[string]any); ok {
					newArr[i] = sanitizeAnthropicSchema(itemMap, defs)
				} else {
					newArr[i] = item
				}
			}
			result[key] = newArr
		}
	}

	return result
}

// flattenAnthropicComposition flattens top-level allOf/oneOf/anyOf for Anthropic.
//   - allOf single item → unwrap
//   - allOf multiple → merge properties (union)
//   - oneOf/anyOf → pick first variant
func flattenAnthropicComposition(schema, defs map[string]any) map[string]any {
	// Handle allOf
	if allOf, ok := schema["allOf"].([]any); ok && len(allOf) > 0 {
		if len(allOf) == 1 {
			// Single item: unwrap into parent
			if item, ok := allOf[0].(map[string]any); ok {
				merged := mergeSchemas(schema, resolveRef(item, defs))
				delete(merged, "allOf")
				return merged
			}
		} else {
			// Multiple items: merge all into parent
			merged := copySchemaExcluding(schema, "allOf")
			for _, entry := range allOf {
				if item, ok := entry.(map[string]any); ok {
					merged = mergeSchemas(merged, resolveRef(item, defs))
				}
			}
			return merged
		}
	}

	// Handle oneOf: pick first variant
	if oneOf, ok := schema["oneOf"].([]any); ok && len(oneOf) > 0 {
		if item, ok := oneOf[0].(map[string]any); ok {
			merged := mergeSchemas(schema, resolveRef(item, defs))
			delete(merged, "oneOf")
			return merged
		}
	}

	// Handle anyOf: pick first variant
	if anyOf, ok := schema["anyOf"].([]any); ok && len(anyOf) > 0 {
		if item, ok := anyOf[0].(map[string]any); ok {
			merged := mergeSchemas(schema, resolveRef(item, defs))
			delete(merged, "anyOf")
			return merged
		}
	}

	return schema
}

// mergeSchemas merges src into dst. Properties are unioned; other fields from src
// overwrite dst only if not already set (dst takes precedence for scalars).
func mergeSchemas(dst, src map[string]any) map[string]any {
	result := make(map[string]any, len(dst)+len(src))
	for k, v := range dst {
		result[k] = v
	}

	for k, v := range src {
		if k == "properties" {
			// Union properties
			dstProps, _ := result["properties"].(map[string]any)
			srcProps, _ := v.(map[string]any)
			if srcProps != nil {
				if dstProps == nil {
					dstProps = make(map[string]any)
				}
				for propName, propVal := range srcProps {
					if _, exists := dstProps[propName]; !exists {
						dstProps[propName] = propVal
					}
				}
				result["properties"] = dstProps
			}
		} else if k == "required" {
			// Union required arrays
			dstReq := toStringSlice(result["required"])
			srcReq := toStringSlice(v)
			seen := make(map[string]struct{})
			for _, r := range dstReq {
				seen[r] = struct{}{}
			}
			for _, r := range srcReq {
				if _, exists := seen[r]; !exists {
					dstReq = append(dstReq, r)
					seen[r] = struct{}{}
				}
			}
			anySlice := make([]any, len(dstReq))
			for i, s := range dstReq {
				anySlice[i] = s
			}
			result["required"] = anySlice
		} else if _, exists := result[k]; !exists {
			result[k] = v
		}
	}
	return result
}

// copySchemaExcluding copies a schema map excluding a specific key.
func copySchemaExcluding(schema map[string]any, excludeKey string) map[string]any {
	result := make(map[string]any, len(schema))
	for k, v := range schema {
		if k != excludeKey {
			result[k] = v
		}
	}
	return result
}

// toStringSlice extracts a []string from an any that is expected to be []any of strings.
func toStringSlice(val any) []string {
	arr, ok := val.([]any)
	if !ok {
		return nil
	}
	result := make([]string, 0, len(arr))
	for _, item := range arr {
		if s, ok := item.(string); ok {
			result = append(result, s)
		}
	}
	return result
}
