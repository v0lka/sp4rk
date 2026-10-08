package llm

import (
	"encoding/json"
)

// SanitizeSchemaForGoogle normalizes a JSON Schema for the Google
// generateContent `Schema` proto, whose single-valued `type` enum
// (STRING/NUMBER/INTEGER/BOOLEAN/ARRAY/OBJECT/NULL) and limited keyword set
// reject JSON-Schema constructs the other providers tolerate. Without this
// pass a tool declaring a type union (`"type": ["array", "string"]` — e.g. the
// in-tree store_fact/search_facts tools) fails the whole request with
// `400 INVALID_ARGUMENT` ("Proto field is not repeating, cannot start list").
//
// The pass reuses SanitizeSchemaForOpenAINonStrict — Gemini and OpenAI
// non-strict share most constraints ($ref/$defs inlining, $schema/$id/$comment/
// default/examples removal, object-type inference from properties/required) —
// and then applies Google-specific rewrites:
//   - a `type` array collapses to the single type Google's enum accepts
//     (drop "null", prefer "string" among the remaining members, else the
//     first declared non-null member, defaulting to "string"); a dropped
//     "null" member is preserved as `"nullable": true`;
//   - `items` is dropped when the (collapsed) type is a scalar — it only
//     describes array elements;
//   - keywords Google's Schema proto does not model are removed
//     (additionalProperties, strict, oneOf/allOf via flattening below, const,
//     uniqueItems, patternProperties, …);
//   - top-level (and nested) `allOf` is merged into the parent schema and
//     `oneOf` is renamed to `anyOf` (Google models `anyOf` only; renaming is
//     a safe relaxation that keeps every variant).
//
// Sanitization is best-effort: input that cannot be parsed is returned
// unchanged, so a request is never failed by the sanitizer itself.
func SanitizeSchemaForGoogle(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 {
		return raw
	}
	// Shared compatibility pass: $ref/$defs inlining, $schema/$id/$comment/
	// default/examples removal, object-type inference, recursive walk.
	sanitized := SanitizeSchemaForOpenAINonStrict(raw)
	if len(sanitized) == 0 {
		return sanitized
	}
	var schema map[string]any
	if err := json.Unmarshal(sanitized, &schema); err != nil {
		return sanitized
	}
	if schema == nil {
		// A literal "null" schema: nothing to rewrite.
		return sanitized
	}
	return marshalPreservingKeyOrder(raw, sanitizeGoogleSchema(schema))
}

// googleUnsupportedSchemaKeywords lists JSON-Schema keywords Google's
// generateContent Schema proto does not define. Leaving any of them in place
// risks a rejected (or silently misinterpreted) function declaration. Some are
// already removed by the shared OpenAI pass; they are listed so this sanitizer
// stays correct on its own. Composition keywords (allOf/oneOf) are rewritten
// by flattenGoogleComposition before this set is consulted.
var googleUnsupportedSchemaKeywords = map[string]struct{}{
	"$anchor":               {},
	"$comment":              {},
	"$defs":                 {},
	"$dynamicAnchor":        {},
	"$dynamicRef":           {},
	"$id":                   {},
	"$ref":                  {},
	"$schema":               {},
	"$vocabulary":           {},
	"additionalProperties":  {},
	"const":                 {},
	"contains":              {},
	"contentMediaType":      {},
	"contentSchema":         {},
	"default":               {},
	"definitions":           {},
	"dependencies":          {},
	"dependentRequired":     {},
	"dependentSchemas":      {},
	"else":                  {},
	"examples":              {},
	"exclusiveMaximum":      {},
	"exclusiveMinimum":      {},
	"if":                    {},
	"multipleOf":            {},
	"not":                   {},
	"patternProperties":     {},
	"propertyNames":         {},
	"strict":                {},
	"then":                  {},
	"unevaluatedItems":      {},
	"unevaluatedProperties": {},
	"uniqueItems":           {},
}

// googleScalarTypes are the primitive JSON types whose schemas cannot carry an
// "items" keyword (only array schemas can).
var googleScalarTypes = map[string]struct{}{
	"boolean": {},
	"integer": {},
	"number":  {},
	"string":  {},
}

// sanitizeGoogleSchema rewrites one schema level for Google compatibility and
// recurses into properties, items and anyOf variants. The input map is freshly
// built by the shared sanitizer pass, so in-place mutation is safe.
func sanitizeGoogleSchema(schema map[string]any) map[string]any {
	if schema == nil {
		return nil
	}

	// Rewrite composition keywords Google does not model before stripping.
	schema = flattenGoogleComposition(schema)

	for key := range schema {
		if _, unsupported := googleUnsupportedSchemaKeywords[key]; unsupported {
			delete(schema, key)
		}
	}

	collapseGoogleTypeUnion(schema)

	// Recurse into the sub-schemas Google models: properties, items, anyOf.
	if props, ok := schema["properties"].(map[string]any); ok {
		for name, val := range props {
			if sub, ok := val.(map[string]any); ok {
				props[name] = sanitizeGoogleSchema(sub)
			}
		}
	}
	switch items := schema["items"].(type) {
	case map[string]any:
		schema["items"] = sanitizeGoogleSchema(items)
	case []any:
		for i, val := range items {
			if sub, ok := val.(map[string]any); ok {
				items[i] = sanitizeGoogleSchema(sub)
			}
		}
	}
	if variants, ok := schema["anyOf"].([]any); ok {
		for i, val := range variants {
			if sub, ok := val.(map[string]any); ok {
				variants[i] = sanitizeGoogleSchema(sub)
			}
		}
	}
	return schema
}

// flattenGoogleComposition rewrites the composition keywords Google's Schema
// proto does not model into forms it does:
//   - allOf: every item is merged into the parent (properties unioned,
//     required unioned, other members first-wins) — the same merge strategy
//     the Anthropic sanitizer uses;
//   - oneOf: renamed to anyOf. Google models anyOf only; oneOf is the stricter
//     "exactly one" form, so anyOf is a safe relaxation that keeps every
//     variant instead of silently dropping all but one. When both keys are
//     present, anyOf wins and oneOf is dropped.
func flattenGoogleComposition(schema map[string]any) map[string]any {
	if allOf, ok := schema["allOf"].([]any); ok && len(allOf) > 0 {
		merged := copySchemaExcluding(schema, "allOf")
		for _, entry := range allOf {
			if item, ok := entry.(map[string]any); ok {
				merged = mergeSchemas(merged, item)
			}
		}
		schema = merged
	}
	if oneOf, ok := schema["oneOf"].([]any); ok {
		if _, hasAnyOf := schema["anyOf"]; !hasAnyOf {
			schema["anyOf"] = oneOf
		}
		delete(schema, "oneOf")
	}
	return schema
}

// collapseGoogleTypeUnion resolves a JSON-Schema type union (or absent/odd
// type value) to the single type string Google's singular Schema.type enum
// accepts. "null" members are dropped and recorded as `"nullable": true`
// (Google's spelling of a may-be-null field). Among the remaining members
// "string" wins when present (the most broadly coercible shape — a model asked
// for an unrepresentable union degrades to text it can still emit); otherwise
// the first declared non-null member is kept, preserving author intent;
// a union of only "null" (or an empty list) falls back to "string".
func collapseGoogleTypeUnion(schema map[string]any) {
	typeVal, ok := schema["type"]
	if !ok {
		return
	}
	list, ok := typeVal.([]any)
	if !ok {
		return // already a single type (or a non-array oddity): nothing to do
	}
	nonNull := make([]string, 0, len(list))
	nullable := false
	for _, entry := range list {
		member, ok := entry.(string)
		if !ok {
			continue
		}
		if member == "null" {
			nullable = true
			continue
		}
		nonNull = append(nonNull, member)
	}

	chosen := "string"
	switch {
	case len(nonNull) == 1:
		chosen = nonNull[0]
	case len(nonNull) > 1:
		// First declared non-null member by default ("author intent"),
		// overridden by "string" when the union offers it.
		chosen = nonNull[0]
		for _, member := range nonNull {
			if member == "string" {
				chosen = member
				break
			}
		}
	}
	schema["type"] = chosen
	if nullable {
		if existing, ok := schema["nullable"].(bool); !ok || !existing {
			schema["nullable"] = true
		}
	}
	// "items" only describes array elements; a collapsed scalar schema must
	// not carry it (e.g. ["array","string"] → "string" drops the element
	// schema, the array shape degrades to text the tool coerces itself).
	if _, scalar := googleScalarTypes[chosen]; scalar {
		delete(schema, "items")
	}
}
