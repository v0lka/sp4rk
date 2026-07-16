package agent

import (
	"regexp"
	"strings"
)

// validTodoLineRe matches a strict checklist line: it must start (no leading
// whitespace — nesting is disallowed) with "- [ ] " or "- [x] " followed by
// the item text. Shared between the update_checklist tool (which additionally
// validates and reports errors) and the executor (which diffs checklist
// contents to detect batched updates) so both parse identically.
var validTodoLineRe = regexp.MustCompile(`^- \[([ x])\] (.+)$`)

// ParseTodoLine parses a single Markdown checklist line. It returns the item
// and true when the line matches the strict "- [ ] "/"- [x] " format (after
// right-trimming trailing spaces/tabs/CR). Blank lines and malformed lines
// return false. No validation is performed — callers that must reject invalid
// input (nested lists, non-checkbox text) do so themselves.
func ParseTodoLine(line string) (TodoItem, bool) {
	trimmed := strings.TrimRight(line, " \t\r")
	m := validTodoLineRe.FindStringSubmatch(trimmed)
	if m == nil {
		return TodoItem{}, false
	}
	return TodoItem{
		Text:    m[2],
		Checked: m[1] == "x",
	}, true
}

// ParseTodoItems extracts all valid checklist items from a multi-line Markdown
// todo list string, preserving order. Blank and malformed lines are skipped.
// It is the multi-line convenience wrapper around ParseTodoLine.
func ParseTodoItems(rawTodoList string) []TodoItem {
	items := make([]TodoItem, 0)
	for _, line := range strings.Split(rawTodoList, "\n") {
		if item, ok := ParseTodoLine(line); ok {
			items = append(items, item)
		}
	}
	return items
}
