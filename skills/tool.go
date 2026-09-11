package skills

import (
	"context"
	"encoding/json"
	"os"

	sdktools "github.com/v0lka/sp4rk/tools"
)

const toolReadSkillResourceDesc = `Purpose: read a resource file bundled with an activated skill — its reference material, scripts, or other supporting files — by skill name and a path relative to that skill's directory.
Use when: a skill's instructions point you at a bundled file (e.g. "see references/api.md"); the skill must already be active on the current request, otherwise it is not addressable.
Inputs: skill (name of the active skill that contains the resource); path (relative path within the skill directory, e.g. 'references/api.md', 'scripts/setup.sh').
Outputs: the resource file's raw contents as text.
Example: skill="pdf-processing", path="references/forms.md".
Anti-example: not for reading a skill's instruction body (already injected into your prompt) or arbitrary workspace files (use read_file with a path); only active skills resolve, and paths that escape the skill directory are rejected.`

// SkillPathResolver resolves a skill name to its directory path.
// Returns ("", false) if the skill is not found or not active.
// The context carries the per-session activation state (see ActiveSkills in core).
type SkillPathResolver func(ctx context.Context, skillName string) (dirPath string, ok bool)

// ReadSkillResourceTool reads files from activated skill directories.
type ReadSkillResourceTool struct {
	*sdktools.BaseTool
	resolvePath SkillPathResolver
}

// NewReadSkillResourceTool creates a tool that reads resources from skill directories.
func NewReadSkillResourceTool(resolver SkillPathResolver) *ReadSkillResourceTool {
	return &ReadSkillResourceTool{
		BaseTool: &sdktools.BaseTool{
			ToolName:        "read_skill_resource",
			ToolGroup:       sdktools.GroupSystem,
			ToolDescription: toolReadSkillResourceDesc,
			Schema: json.RawMessage(`{
				"type": "object",
				"properties": {
					"skill": {
						"type": "string",
						"description": "Name of the active skill containing the resource."
					},
					"path": {
						"type": "string",
						"description": "Relative path within the skill directory (e.g., 'references/api.md', 'scripts/setup.sh')."
					}
				},
				"required": ["skill", "path"]
			}`),
			Policy: sdktools.PolicyAlwaysAllow, // skill resources are read-only, safe
		},
		resolvePath: resolver,
	}
}

// readSkillResourceInput holds the parsed tool input.
type readSkillResourceInput struct {
	Skill string `json:"skill"`
	Path  string `json:"path"`
}

// Execute reads the requested resource file from the skill directory.
func (t *ReadSkillResourceTool) Execute(ctx context.Context, input json.RawMessage) (sdktools.ToolResult, error) {
	if t.resolvePath == nil {
		return sdktools.ErrorResult("skill resolver is not available"), nil
	}

	var parsed readSkillResourceInput
	if err := json.Unmarshal(input, &parsed); err != nil {
		return sdktools.ParseInputError(err)
	}

	if parsed.Skill == "" {
		return sdktools.ErrorResult("skill name is required"), nil
	}
	if parsed.Path == "" {
		return sdktools.ErrorResult("resource path is required"), nil
	}

	// Resolve the skill's directory path via the manager
	skillDir, ok := t.resolvePath(ctx, parsed.Skill)
	if !ok {
		return sdktools.ErrorResult("skill %q not found or not active", parsed.Skill), nil
	}

	// Resolve the resource path safely (prevent path traversal)
	absPath, err := SafeResolvePath(ctx, skillDir, parsed.Path)
	if err != nil {
		return sdktools.ErrorResult("invalid resource path: %v", err), nil
	}

	// Read the file
	data, err := os.ReadFile(absPath)
	if err != nil {
		if os.IsNotExist(err) {
			return sdktools.ErrorResult("resource %q not found in skill %q", parsed.Path, parsed.Skill), nil
		}
		return sdktools.ErrorResult("failed to read resource: %v", err), nil
	}

	return sdktools.ToolResult{Content: string(data)}, nil
}
