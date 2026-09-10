package llm

import (
	"bytes"
	"encoding/json"
	"sort"
)

// marshalPreservingKeyOrder re-marshals vals using the key order of the
// original schema document raw. encoding/json renders a map[string]any with its
// keys sorted alphabetically, so sanitizing a tool schema through a map loses
// the author's key order: "properties" entries come back alphabetized (e.g.
// {content, keywords} instead of {keywords, content}) and "required" is
// re-sorted. Models tend to emit tool arguments in schema-declared order, so
// preserving that order removes a schema-induced cause of argument reordering
// (and, when arguments are truncated mid-stream, shifts which field is cut).
//
// On any inconsistency between vals and raw the function falls back to a plain
// alphabetical marshal, so sanitization never fails because of ordering.
func marshalPreservingKeyOrder(raw json.RawMessage, vals map[string]any) json.RawMessage {
	order, err := parseJSONKeyOrder(raw)
	if err != nil {
		return marshalAlphabetical(vals)
	}
	var buf bytes.Buffer
	if err := writeOrderedJSON(&buf, vals, order); err != nil {
		return marshalAlphabetical(vals)
	}
	return buf.Bytes()
}

// marshalAlphabetical is the fallback marshaller used when the original key
// order cannot be recovered.
func marshalAlphabetical(vals map[string]any) json.RawMessage {
	out, err := json.Marshal(vals)
	if err != nil {
		return nil
	}
	return out
}

// orderNode records the declared structure of a JSON value: object members in
// order, array element nodes, and array scalar values in order. It is built by
// a single streaming pass over the original document.
type orderNode struct {
	isObj   bool
	kv      []orderKV
	isArr   bool
	elems   []*orderNode
	scalars []any
}

// orderKV is one object member: its key and the node describing its value.
type orderKV struct {
	key   string
	child *orderNode
}

// parseJSONKeyOrder streams raw and returns its structural order tree.
func parseJSONKeyOrder(raw json.RawMessage) (*orderNode, error) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	node, _, err := readOrderNode(dec)
	return node, err
}

// readOrderNode consumes one JSON value, returning its order node and, for
// scalars, the scalar value itself.
func readOrderNode(dec *json.Decoder) (*orderNode, any, error) {
	tok, err := dec.Token()
	if err != nil {
		return nil, nil, err
	}
	delim, ok := tok.(json.Delim)
	if !ok {
		return nil, tok, nil // scalar leaf
	}
	switch delim {
	case '{':
		node := &orderNode{isObj: true}
		for dec.More() {
			keyTok, err := dec.Token()
			if err != nil {
				return nil, nil, err
			}
			key, _ := keyTok.(string)
			child, _, err := readOrderNode(dec)
			if err != nil {
				return nil, nil, err
			}
			node.kv = append(node.kv, orderKV{key: key, child: child})
		}
		if _, err := dec.Token(); err != nil { // consume '}'
			return nil, nil, err
		}
		return node, nil, nil
	case '[':
		node := &orderNode{isArr: true}
		for dec.More() {
			child, scalar, err := readOrderNode(dec)
			if err != nil {
				return nil, nil, err
			}
			node.elems = append(node.elems, child)
			node.scalars = append(node.scalars, scalar)
		}
		if _, err := dec.Token(); err != nil { // consume ']'
			return nil, nil, err
		}
		return node, nil, nil
	}
	return nil, nil, nil
}

// writeOrderedJSON renders val to buf, using orig to order object members and
// to restore the declared order of "required" arrays.
func writeOrderedJSON(buf *bytes.Buffer, val any, orig *orderNode) error {
	switch v := val.(type) {
	case map[string]any:
		return writeOrderedObject(buf, v, orig)
	case []any:
		return writeOrderedArray(buf, v, orig)
	default:
		return writeScalar(buf, v)
	}
}

// writeOrderedObject writes m with members ordered as in orig; members not
// present in orig are appended in alphabetical order for determinism.
func writeOrderedObject(buf *bytes.Buffer, m map[string]any, orig *orderNode) error {
	buf.WriteByte('{')
	first := true
	written := make(map[string]struct{}, len(m))
	if orig != nil && orig.isObj {
		for _, kv := range orig.kv {
			v, ok := m[kv.key]
			if !ok {
				continue
			}
			if !first {
				buf.WriteByte(',')
			}
			first = false
			if err := writeMember(buf, kv.key, v, kv.child); err != nil {
				return err
			}
			written[kv.key] = struct{}{}
		}
	}
	extras := make([]string, 0, len(m))
	for k := range m {
		if _, ok := written[k]; !ok {
			extras = append(extras, k)
		}
	}
	sort.Strings(extras)
	for _, k := range extras {
		if !first {
			buf.WriteByte(',')
		}
		first = false
		if err := writeMember(buf, k, m[k], nil); err != nil {
			return err
		}
	}
	buf.WriteByte('}')
	return nil
}

// writeMember writes a single "key": value pair. The "required" array keeps the
// caller's declared order (the sanitizer rebuilds it deterministically, which
// would otherwise move a mandatory field to the end).
func writeMember(buf *bytes.Buffer, key string, val any, child *orderNode) error {
	kb, err := json.Marshal(key)
	if err != nil {
		return err
	}
	buf.Write(kb)
	buf.WriteByte(':')
	if key == "required" {
		if items, ok := val.([]any); ok && child != nil && child.isArr {
			val = reorderScalars(items, child.scalars)
		}
	}
	return writeOrderedJSON(buf, val, child)
}

// writeOrderedArray writes a with element nodes taken from orig by position.
func writeOrderedArray(buf *bytes.Buffer, a []any, orig *orderNode) error {
	buf.WriteByte('[')
	for i, e := range a {
		if i > 0 {
			buf.WriteByte(',')
		}
		var child *orderNode
		if orig != nil && orig.isArr && i < len(orig.elems) {
			child = orig.elems[i]
		}
		if err := writeOrderedJSON(buf, e, child); err != nil {
			return err
		}
	}
	buf.WriteByte(']')
	return nil
}

// writeScalar writes a non-container value as standard JSON.
func writeScalar(buf *bytes.Buffer, v any) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}
	buf.Write(b)
	return nil
}

// reorderScalars returns items reordered to follow the order in which the same
// string elements appeared in the original document's array (declared). Items
// not present in declared keep their relative order and are appended.
func reorderScalars(items, declared []any) []any {
	if len(declared) == 0 || len(items) == 0 {
		return items
	}
	pos := make(map[string]int, len(declared))
	for i, d := range declared {
		s, ok := d.(string)
		if !ok {
			continue
		}
		if _, dup := pos[s]; !dup {
			pos[s] = i
		}
	}
	type ranked struct {
		item any
		idx  int
	}
	rankedItems := make([]ranked, 0, len(items))
	for i, it := range items {
		s, ok := it.(string)
		if !ok {
			return items // unexpected shape; leave as-is
		}
		p, known := pos[s]
		if !known {
			p = len(pos) + i // stable trailing slot
		}
		rankedItems = append(rankedItems, ranked{item: it, idx: p})
	}
	sort.SliceStable(rankedItems, func(a, b int) bool {
		return rankedItems[a].idx < rankedItems[b].idx
	})
	out := make([]any, len(rankedItems))
	for i, r := range rankedItems {
		out[i] = r.item
	}
	return out
}
