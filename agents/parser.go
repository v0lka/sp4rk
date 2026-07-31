package agents

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"

	"gopkg.in/yaml.v3"
)

// agentNamePattern validates the agent name field: lowercase alphanumeric and
// hyphens, no leading/trailing hyphens. Same shape as the skill name pattern.
var agentNamePattern = regexp.MustCompile(`^[a-z0-9]([a-z0-9-]*[a-z0-9])?$`)

// ParseError describes a validation failure in an AGENT.md file.
type ParseError struct {
	Path    string
	Message string
}

// Error returns the validation failure message prefixed with the AGENT.md path.
func (e *ParseError) Error() string {
	return fmt.Sprintf("agent parse error (%s): %s", e.Path, e.Message)
}

// ParseAgent reads and validates an AGENT.md file, returning an Agent.
// agentMDPath is the path to AGENT.md; dirPath is the absolute path to the
// agent directory (used for DirPath and to validate that name == dir name).
func ParseAgent(agentMDPath, dirPath string) (*Agent, error) {
	data, err := os.ReadFile(agentMDPath)
	if err != nil {
		return nil, fmt.Errorf("read AGENT.md: %w", err)
	}

	meta, body, err := parseFrontmatter(string(data))
	if err != nil {
		return nil, &ParseError{Path: agentMDPath, Message: err.Error()}
	}

	agent := &Agent{
		Metadata: *meta,
		Body:     body,
		DirPath:  dirPath,
	}

	if err := validateAgent(agent, dirPath); err != nil {
		return nil, &ParseError{Path: agentMDPath, Message: err.Error()}
	}

	return agent, nil
}

// parseFrontmatter splits an AGENT.md into YAML frontmatter and a markdown
// body. It is a local copy of the skill parser's frontmatter splitter so this
// package stays self-contained (it must not import the skills package). It
// expects the content to start with "---" on its own line.
//
// Unknown frontmatter keys are silently ignored: yaml.Unmarshal populates only
// the fields modeled on AgentMetadata and leaves the rest, so not-yet-supported
// fields do not cause a parse error.
func parseFrontmatter(content string) (*AgentMetadata, string, error) {
	if content == "" {
		return nil, "", errors.New("empty AGENT.md content")
	}

	rest := content
	if rest != "" && rest[0] == '\n' {
		rest = rest[1:]
	}

	// Match opening --- delimiter.
	idx := findFrontmatterDelim(rest)
	if idx < 0 {
		return nil, "", errors.New("missing opening --- frontmatter delimiter")
	}
	rest = rest[idx:]

	// Find closing --- delimiter.
	endIdx := findFrontmatterDelim(rest)
	if endIdx < 0 {
		return nil, "", errors.New("missing closing --- frontmatter delimiter")
	}
	yamlContent := rest[:endIdx]
	body := rest[endIdx:]

	// Trim leading whitespace/newline from the body.
	if body != "" && body[0] == '\n' {
		body = body[1:]
	}

	var meta AgentMetadata
	if err := yaml.Unmarshal([]byte(yamlContent), &meta); err != nil {
		return nil, "", fmt.Errorf("invalid YAML frontmatter: %w", err)
	}

	return &meta, body, nil
}

// findFrontmatterDelim finds the position after the next "---\n" delimiter,
// skipping the delimiter itself. Returns -1 if not found. It requires the "---"
// to be at the start of a line (or start of string) so that "---" inside YAML
// string values is not mistaken for a delimiter.
func findFrontmatterDelim(s string) int {
	for i := 0; i < len(s); i++ {
		if s[i] == '-' {
			if i > 0 && s[i-1] != '\n' {
				continue
			}
			if i+2 < len(s) && s[i+1] == '-' && s[i+2] == '-' {
				// Found ---, skip past it and any trailing whitespace/newline.
				j := i + 3
				for j < len(s) && (s[j] == ' ' || s[j] == '\t' || s[j] == '\r') {
					j++
				}
				if j < len(s) && s[j] == '\n' {
					j++
				}
				return j
			}
		}
	}
	return -1
}

// validateAgent checks the v1 spec constraints on a parsed Agent. Only the
// required fields are validated: name (present, regex, matches directory) and
// description (present). Unknown fields are intentionally not checked.
func validateAgent(agent *Agent, dirPath string) error {
	// Name is required.
	if agent.Metadata.Name == "" {
		return errors.New("name field is required")
	}

	// Name format: lowercase alphanumeric + hyphens, no leading/trailing hyphens.
	if !agentNamePattern.MatchString(agent.Metadata.Name) {
		return fmt.Errorf("name %q must be lowercase alphanumeric with hyphens, no leading/trailing hyphens", agent.Metadata.Name)
	}

	// Name must match the parent directory name.
	dirName := filepath.Base(dirPath)
	if agent.Metadata.Name != dirName {
		return fmt.Errorf("name %q must match parent directory name %q", agent.Metadata.Name, dirName)
	}

	// Description is required.
	if agent.Metadata.Description == "" {
		return errors.New("description field is required")
	}

	return nil
}
