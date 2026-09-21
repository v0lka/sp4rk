package tools

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
)

// Tool-input validation. Provider-side JSON-schema validation that runs for
// free on native tool definitions (one typed tool call per tool) is absent
// whenever tool arguments travel as a free-form object (meta-tool envelopes,
// replayed calls, queued actions): wrong argument names (e.g.
// read_file{offset, limit} instead of start_line/end_line) are silently
// ignored by json.Unmarshal — the tool executes with defaulted parameters
// and the model receives a bewildering whole-file result. The structural
// validator below catches the failure cheaply BEFORE dispatch and returns an
// actionable error naming the valid parameters.
//
// The validator is deliberately lightweight (no external JSON-schema
// dependency): it checks that the input is an object, the presence of
// required keys, JSON types of declared properties, and unknown keys against
// the declared property set — the failure modes models actually produce —
// recursively: every array element against the "items" schema and every
// nested property object against its own schema, with error paths like
// tasks[2].id. Anything the validator does not model fails OPEN (never
// blocks a legitimate call): schemas without a "properties" object at a
// level, unparseable schemas, $ref subtrees (no resolver by design), and
// values nested deeper than maxValidationDepth.

// maxValidationDepth bounds recursive validation. Values nested deeper than
// this many levels below the input root are skipped fail-open (a defensive
// cap against pathological schemas and inputs).
const maxValidationDepth = 16

// InputValidationError describes a structural violation of a tool input
// object against the tool's input schema. It renders as an actionable
// message for the model — offending parameter name, its path for nested
// values, and the valid parameter names included — so the rendered error
// text is the product.
type InputValidationError struct {
	// Tool is the name of the tool whose input was validated.
	Tool string
	// Path locates the offending value inside the input ("" at the top
	// level), e.g. "tasks[2].id".
	Path string
	// Reason is the human-readable violation, e.g.
	// `missing required parameter "id"`.
	Reason string
	// ValidParams lists the declared property names of the violated schema
	// level (sorted for determinism); empty when that level declares none.
	ValidParams []string
}

// Error renders the actionable validation message for the model.
func (e *InputValidationError) Error() string {
	var msg string
	// Top-level properties and array elements quote the value's own name in
	// the reason ("parameter \"path\" must be…"), so repeating it as an
	// `at <path>` location would duplicate it — emit the location only when
	// the reason does not already carry it.
	if e.Path != "" && !strings.Contains(e.Reason, "\""+e.Path+"\"") {
		msg = fmt.Sprintf("input for %q is invalid at %s: %s", e.Tool, e.Path, e.Reason)
	} else {
		msg = fmt.Sprintf("input for %q is invalid: %s", e.Tool, e.Reason)
	}
	if len(e.ValidParams) > 0 {
		msg += "; valid parameters: " + strings.Join(e.ValidParams, ", ")
	}
	return msg
}

// ValidateToolInput checks raw tool input against a tool's raw JSON input
// schema. It returns nil when the input is structurally compatible (or when
// a schema level declares no closed property set, in which case that level
// is skipped fail-open) and an *InputValidationError describing the first
// problem otherwise. Both arguments are raw JSON documents; input may be
// nil/empty, which is treated as an empty object (so any required parameter
// fails, while schemas without required parameters pass).
//
// A schema level that omits (or nulls) additionalProperties is treated as a
// CLOSED set — hand-written built-in schemas declare every parameter, so an
// unknown key is a model mistake worth rejecting with an actionable message.
// ValidateToolInput is therefore the strict built-in variant; for the
// category-dependent default the registry applies (closed for built-ins, OPEN
// for MCP-proxied tools whose server-supplied schemas frequently omit the
// keyword), call [ValidateToolInputForCategory].
//
// The schema argument comes FIRST and the raw input arguments SECOND — both
// are json.RawMessage, so a silent swap parses the input as the schema and
// disables validation (no top-level properties → early nil). Host Execute
// wrappers call it before dispatch.
func ValidateToolInput(tool string, schema, input json.RawMessage) error {
	return validateToolInput(tool, schema, input, true)
}

// ValidateToolInputForCategory is [ValidateToolInput] with the tool's source
// category made explicit. It applies the same key-admission default the
// registry's Execute uses: a level that omits (or nulls) additionalProperties
// is a CLOSED set for a built-in ([SourceCategoryCore]) tool, but the JSON
// Schema spec default of an OPEN set for an MCP-proxied
// ([SourceCategoryMCP]) tool — an external server schema frequently omits the
// keyword while still accepting extra arguments that json.Unmarshal would
// silently ignore. Hosts that pre-validate before dispatch and route
// MCP-proxied tools should call this rather than ValidateToolInput, which
// always applies the strict built-in (closed) default.
func ValidateToolInputForCategory(tool string, schema, input json.RawMessage, category ToolSourceCategory) error {
	return validateToolInput(tool, schema, input, category != SourceCategoryMCP)
}

// validateToolInput is the implementation behind ValidateToolInput.
// closedByDefault controls how an object level that omits (or nulls)
// additionalProperties is treated: true means a closed set (unknown keys are
// rejected), false means the JSON Schema spec default of an open set. The
// registry passes false for MCP-proxied tools, because an external server
// schema frequently omits additionalProperties while still accepting extra
// arguments that json.Unmarshal would silently ignore.
func validateToolInput(tool string, schema, input json.RawMessage, closedByDefault bool) error {
	var node schemaNode
	if err := json.Unmarshal(schema, &node); err != nil {
		return nil //nolint:nilerr // unparseable schema: fail-open by design
	}
	if node.Ref != "" {
		return nil // $ref schema: fail-open by design
	}
	if len(node.Properties) == 0 {
		return nil //nolint:nilerr // open/undocumented schema: fail-open
	}

	var obj map[string]json.RawMessage
	if len(bytes.TrimSpace(input)) == 0 {
		obj = map[string]json.RawMessage{}
	} else if err := json.Unmarshal(input, &obj); err != nil {
		return &InputValidationError{Tool: tool, Reason: "input must be a JSON object", ValidParams: propertyNames(node.Properties)}
	}
	return validateSchemaObject(tool, "", node, obj, 0, closedByDefault)
}

// schemaNode is the subset of JSON Schema the validator models for one
// schema level (an object schema or an array's "items" schema).
type schemaNode struct {
	Properties           map[string]json.RawMessage `json:"properties"`
	Required             []string                   `json:"required"`
	AdditionalProperties json.RawMessage            `json:"additionalProperties"`
	Items                json.RawMessage            `json:"items"`
	Ref                  string                     `json:"$ref"`
	Type                 json.RawMessage            `json:"type"`
}

// validateSchemaObject checks the contents of one object value against its
// schema level: required-key presence, unknown keys against the closed
// property set, then each declared property's value (type check plus
// recursion) via validateSchemaValue. depth is the nesting depth of the
// object being validated (the input root is 0).
func validateSchemaObject(tool, path string, node schemaNode, obj map[string]json.RawMessage, depth int, closedByDefault bool) error {
	names := propertyNames(node.Properties)
	nameSet := make(map[string]struct{}, len(node.Properties))
	for k := range node.Properties {
		nameSet[k] = struct{}{}
	}

	// Required keys must be present.
	for _, req := range node.Required {
		if _, ok := obj[req]; !ok {
			return &InputValidationError{
				Tool:        tool,
				Path:        path,
				Reason:      fmt.Sprintf("missing required parameter %q", req),
				ValidParams: names,
			}
		}
	}

	// Unknown keys are rejected when the schema level is a closed set.
	additionalAllowed := additionalPropertiesAllowed(node.AdditionalProperties, closedByDefault)
	argNames := make([]string, 0, len(obj))
	for k := range obj {
		argNames = append(argNames, k)
	}
	sort.Strings(argNames)
	for _, k := range argNames {
		if _, ok := nameSet[k]; !ok && !additionalAllowed {
			return &InputValidationError{
				Tool:        tool,
				Path:        path,
				Reason:      fmt.Sprintf("unknown parameter %q (not accepted by this tool; check the tool's parameter list)", k),
				ValidParams: names,
			}
		}
	}

	// Type-check and recurse into declared properties.
	for _, k := range argNames {
		prop, ok := node.Properties[k]
		if !ok {
			continue
		}
		if err := validateSchemaValue(tool, k, joinPath(path, k), prop, obj[k], names, depth+1, closedByDefault); err != nil {
			return err
		}
	}
	return nil
}

// validateSchemaValue checks one value against its property/element schema:
// the declared JSON type first, then structural recursion (nested object
// contents, array elements). name is the display name of the value (the
// property key, or the full path for array elements, which have no key);
// path is its full path; validParams lists the enclosing schema level's
// declared property names for error messages. depth is the value's nesting
// depth below the input root (top-level properties are 1).
func validateSchemaValue(tool, name, path string, schemaRaw, value json.RawMessage, validParams []string, depth int, closedByDefault bool) error {
	if depth > maxValidationDepth {
		return nil // deeper than the cap: subtree skipped fail-open
	}
	var node schemaNode
	if err := json.Unmarshal(schemaRaw, &node); err != nil {
		return nil //nolint:nilerr // unparseable property schema: skip fail-open
	}
	if node.Ref != "" {
		return nil // $ref subtree: skip fail-open (no resolver by design)
	}

	// Type check the declared "type" (a string or an array of strings). A
	// schema without "type" — or with a type form the decoder does not
	// model (explicit null, empty strings, non-string junk) — skips the
	// check fail-open.
	if allowed := decodeSchemaTypes(node.Type); len(allowed) > 0 {
		actual := jsonTypeNameOf(value)
		match := false
		for _, want := range allowed {
			if typeValueCompatible(want, value) {
				match = true
				break
			}
		}
		if !match {
			// A node that declares its own properties is its own best
			// valid-parameter list (e.g. an array's items schema).
			if len(node.Properties) > 0 {
				validParams = propertyNames(node.Properties)
			}
			return &InputValidationError{
				Tool:        tool,
				Path:        path,
				Reason:      fmt.Sprintf("parameter %q must be of type %s, got %s", name, strings.Join(allowed, "|"), actual),
				ValidParams: validParams,
			}
		}
	}

	kind := jsonTypeNameOf(value)
	if kind == "object" && len(node.Properties) > 0 {
		var obj map[string]json.RawMessage
		if err := json.Unmarshal(value, &obj); err != nil {
			return nil //nolint:nilerr // not a parseable object: skip fail-open
		}
		return validateSchemaObject(tool, path, node, obj, depth, closedByDefault)
	}
	if kind == "array" && len(node.Items) > 0 {
		var arr []json.RawMessage
		if err := json.Unmarshal(value, &arr); err != nil {
			return nil //nolint:nilerr // not a parseable array: skip fail-open
		}
		for i, elem := range arr {
			elemPath := fmt.Sprintf("%s[%d]", path, i)
			if err := validateSchemaValue(tool, elemPath, elemPath, node.Items, elem, nil, depth+1, closedByDefault); err != nil {
				return err
			}
		}
	}
	return nil
}

// joinPath appends a property key to a validation path, building dotted
// paths like "tasks[2].id" once the array branch has produced "tasks[2]".
func joinPath(path, key string) string {
	if path == "" {
		return key
	}
	return path + "." + key
}

// additionalPropertiesAllowed reports whether a schema level accepts keys
// beyond its declared properties: an explicit boolean is taken at face value,
// and the object form (a schema for the extra keys) allows them. An absent —
// or explicit JSON null (a null keyword is ignored, i.e. equivalent to
// absent — decoding null into a bool would be a silent no-op, so it is
// checked explicitly) — keyword falls back to the level's default,
// closedByDefault. Built-in schemas use closedByDefault=true (they declare
// every parameter, so an unknown key is a mistake); MCP-proxied server
// schemas use false, the JSON Schema spec default, because they frequently
// omit the keyword while still accepting extra arguments.
func additionalPropertiesAllowed(raw json.RawMessage, closedByDefault bool) bool {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return !closedByDefault
	}
	var asBool bool
	if err := json.Unmarshal(raw, &asBool); err == nil {
		return asBool
	}
	return true // object-form constraint: extra keys allowed
}

// decodeSchemaTypes parses a schema "type" value, which may be a string
// ("integer") or an array of strings (["string","null"]). Forms the decoder
// does not model return nil so the caller skips the type check fail-open:
// an explicit JSON null (some generators emit "type":null — decoding it
// into a string is a silent no-op that would otherwise yield an "" type
// matching nothing), an empty string, an empty array, arrays whose members
// are all empty, and non-string junk.
func decodeSchemaTypes(raw json.RawMessage) []string {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return nil
	}
	var single string
	if err := json.Unmarshal(raw, &single); err == nil {
		if single == "" {
			return nil
		}
		return []string{single}
	}
	var many []string
	if err := json.Unmarshal(raw, &many); err == nil {
		types := make([]string, 0, len(many))
		for _, t := range many {
			if t != "" {
				types = append(types, t)
			}
		}
		if len(types) == 0 {
			return nil
		}
		return types
	}
	return nil
}

// typeValueCompatible maps one declared schema type name onto a raw JSON
// value. JSON Schema "integer" accepts only numbers with a zero fractional
// part (1, 1.0 and 1e2 qualify; 1.5 does not), but encoding/json reports
// every number as a single "number" kind — so the integer check inspects
// the number's textual form, which also avoids float64 precision loss on
// large integer literals.
func typeValueCompatible(want string, value json.RawMessage) bool {
	actual := jsonTypeNameOf(value)
	if want == actual {
		return true
	}
	if want == "integer" && actual == "number" {
		return isIntegralJSONNumber(value)
	}
	return false
}

// isIntegralJSONNumber reports whether a raw JSON number has no fractional
// part: literals an int64 parse covers qualify directly; the rest are
// parsed as float64 and checked for a zero fraction (1.0, 2e3 qualify).
// A number too large to reason about fails open as integral.
func isIntegralJSONNumber(raw json.RawMessage) bool {
	text := string(bytes.TrimSpace(raw))
	if _, err := strconv.ParseInt(text, 10, 64); err == nil {
		return true
	}
	f, err := strconv.ParseFloat(text, 64)
	if err != nil {
		return true // out of float64 range: cannot reason, fail open
	}
	return f == math.Trunc(f)
}

// jsonTypeNameOf reports the JSON type name of a raw value ("null" for an
// explicit null, "null" for absent). It assumes valid JSON.
func jsonTypeNameOf(raw json.RawMessage) string {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 {
		return "null"
	}
	switch trimmed[0] {
	case '"':
		return "string"
	case 't', 'f':
		return "boolean"
	case 'n':
		return "null"
	case '[':
		return "array"
	case '{':
		return "object"
	default:
		return "number"
	}
}

// propertyNames returns the sorted declared property names of a schema's
// properties map (for deterministic error messages).
func propertyNames(props map[string]json.RawMessage) []string {
	names := make([]string, 0, len(props))
	for k := range props {
		names = append(names, k)
	}
	sort.Strings(names)
	return names
}
