package skills

import (
	"strings"
	"testing"
)

// maxDescriptionLength mirrors the guard in tools/builtins: descriptions follow
// the purpose -> when-to-use -> inputs -> outputs -> example -> anti-example
// rubric; anything beyond this limit means marketing prose crept back in.
const maxDescriptionLength = 1200

// TestReadSkillResourceDescriptionFollowsRubric guards read_skill_resource — the
// one built-in whose description is not declared in tools/builtins. It must
// carry the same rubric as every other tool so host UIs that render tool
// descriptions (e.g. c0wrk's always-present picker tooltip) format it
// identically instead of falling back to a single unlabelled paragraph.
func TestReadSkillResourceDescriptionFollowsRubric(t *testing.T) {
	desc := NewReadSkillResourceTool(nil).Description()

	if strings.TrimSpace(desc) == "" {
		t.Fatal("read_skill_resource: description is empty")
	}
	for _, section := range []string{"Purpose:", "Use when:", "Inputs:", "Outputs:", "Example:", "Anti-example:"} {
		if !strings.Contains(desc, section) {
			t.Errorf("read_skill_resource: description lacks %q section", section)
		}
	}
	// Every rubric section must begin its own line, otherwise a line-based
	// markdown reformatter cannot split it into bold-labelled paragraphs.
	for _, line := range []string{"Purpose:", "Use when:", "Inputs:", "Outputs:", "Example:", "Anti-example:"} {
		if !strings.Contains(desc, "\n"+line) && !strings.HasPrefix(desc, line) {
			t.Errorf("read_skill_resource: %q does not start a line — the markdown reformatter needs one label per line", line)
		}
	}
	if len(desc) > maxDescriptionLength {
		t.Errorf("read_skill_resource: description is %d chars, exceeds guard limit %d", len(desc), maxDescriptionLength)
	}
}
