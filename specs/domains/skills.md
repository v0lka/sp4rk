# Skills

## Purpose

Discovery, parsing, and serving of Agent Skills following the open agentskills.io specification. Skills are markdown documents (`SKILL.md`) with YAML frontmatter that bundle instructions and optional resources an agent can activate on demand. `SkillManager` discovers skills from configured directories, parses them, and serves their metadata/bodies; the [router](orchestration/router.md) matches skill descriptors to requests.

## Key Files

- `github.com/v0lka/sp4rk/skills` — `SkillManager`, `NewSkillManager`, `Scan`, `List`, `Get`, `SkillPath`, `SafeResolvePath`
- `github.com/v0lka/sp4rk/skills` (parsing) — `Skill`, `SkillMetadata`, `SkillDescriptor`, `ParseSkill`, `ParseError`
- `github.com/v0lka/sp4rk/skills` — `ReadSkillResourceTool`, `SkillPathResolver`
- `github.com/v0lka/sp4rk/pathutil` — `IsWithinPath`, `ResolveExistingPrefix` (used by `SafeResolvePath`)
- `github.com/v0lka/sp4rk/tools` — `IsWithinRoot`, `CaseInsensitivePathsFrom` (case-aware containment used by `SafeResolvePath`)

## Core Types

```go
type SkillManager struct { /* unexported */ }

type Skill struct {
    Metadata SkillMetadata
    Body     string // markdown body after the YAML frontmatter
    DirPath  string // absolute path to the skill directory
}

type SkillMetadata struct {
    Name          string            `yaml:"name"`
    Description   string            `yaml:"description"`
    License       string            `yaml:"license,omitempty"`
    Compatibility string            `yaml:"compatibility,omitempty"`
    AllowedTools  string            `yaml:"allowed-tools,omitempty"` // space-separated (experimental)
    Extra         map[string]string `yaml:"metadata,omitempty"`
}

// Lightweight discovery-time representation (~100 tokens) — what List() returns
// and what the router matches against.
type SkillDescriptor struct {
    Name        string `json:"name"`
    Description string `json:"description"`
}
```

## Flow

```
NewSkillManager(dirs []string, logger)         // dirs in priority order, highest first
  │
  ├─ Scan()                                    // walks dirs in reverse priority order so
  │      higher-priority entries overwrite lower-priority ones of the same name;
  │      an oversized SKILL.md (safeio.ErrTooLarge) is logged at warn and
  │      skipped, other parse errors are logged at debug and skipped
  │        ├─ follows directory symlinks
  │        └─ ParseSkill each SKILL.md → validate → catalog
  │
  ├─ List()  → []SkillDescriptor               // name + description only
  ├─ Get(name) → (*Skill, bool)                // full body + metadata
  └─ SkillPath(name) → (absDir, bool)
```

`AllowedToolList()` parses the space-separated `allowed-tools` field into a slice (experimental tool restriction; `nil` when empty).

## SKILL.md format & validation

A `SKILL.md` is YAML frontmatter (delimited by `---`) followed by a markdown body. `ParseSkill(skillMDPath, dirPath)` validates and returns a `*Skill` or a `*ParseError`. The file is read via `safeio.ReadFile`, which refuses non-regular files (FIFOs/devices) without blocking and rejects files over the 10 MiB `safeio.DefaultMaxFileSize` cap with `safeio.ErrTooLarge`; a read failure returns a wrapped error (not a `*ParseError`), while frontmatter/validation failures return `*ParseError`.

| Field | Rule |
| ----- | ---- |
| `name` | Required. 1–64 chars, lowercase alphanumeric and hyphens, no leading/trailing hyphens (`^[a-z0-9]([a-z0-9-]*[a-z0-9])?$`); consecutive hyphens permitted. Must match the parent directory name. |
| `description` | Required. At most 1024 characters. |
| `compatibility` | At most 500 characters. |

A skill whose `name` does not match its parent directory name is rejected.

## SafeResolvePath

`SafeResolvePath(ctx, baseDir, relPath) (string, error)` resolves a relative path within a base directory, preventing path traversal. The context carries the session case-sensitivity flag; a nil `ctx` falls back to plain `pathutil` containment. It is used by both `SkillManager` and `ReadSkillResourceTool` to safely access skill-bundled resources:

1. Cleans `baseDir`; if `baseDir` itself is a symlink, resolves it (but does not resolve ancestor symlinks like macOS `/var → /private/var`, preserving textual path consistency).
2. Joins and cleans the relative path.
3. Resolves symlinks in the joined path to their real filesystem paths (through the longest existing prefix via `pathutil.ResolveExistingPrefix`), so partially-existing paths are still symlink-checked — no textual-only fallback.
4. Verifies the resolved path is within `baseDir` via the case-sensitivity-aware `tools.IsWithinRoot(ctx, base, abs)` when `ctx` is non-nil (session flag from `tools.CaseInsensitivePathsFrom`); with a nil `ctx` it falls back to `pathutil.IsWithinPath`. Returns an error if it escapes.

## ReadSkillResourceTool

`read_skill_resource` reads resource files from activated skill directories. It is constructed with a `SkillPathResolver` (`func(ctx, skillName) (dirPath string, ok bool)`) that resolves a skill name to its directory path, carrying per-session activation state. It uses `SafeResolvePath` to prevent traversal, so an agent cannot read files outside a skill's directory. Resource reads go through `safeio.ReadFile` (regular files only, 10 MiB `safeio.DefaultMaxFileSize` cap); an oversized resource returns an explicit `exceeds the N byte read limit` error. Skill resources are read-only and safe, so the tool uses an always-allow policy and declares `ToolGroup: tools.GroupSystem` (the reserved host-trusted group that bypasses policy gates).

## Invariants

- Directories are scanned in **reverse priority order** so higher-priority entries win on name collision.
- `Scan` is idempotent (clears the catalog first); safe to call repeatedly (e.g. after skills are added/removed).
- Invalid `SKILL.md` files are skipped, not fatal; a directory that is not an agent skill is ignored.
- `Scan` follows directory symlinks.
- A skill's `name` must match its parent directory name.
- `List()` returns only lightweight descriptors (~100 tokens); `Get()` returns the full body.
- `SafeResolvePath` never resolves ancestor symlinks and always verifies containment (via the case-sensitivity-aware `tools.IsWithinRoot` when a context is supplied, `pathutil.IsWithinPath` otherwise).

## Configuration

`NewSkillManager(dirs []string, logger)` takes discovery directories in priority order (highest first). The host wires the active skill set per task — the SDK's `SkillManager` only discovers/parses/serves; activation, system-prompt injection of skill bodies, and tool-policy overrides are host concerns (the router matches descriptors, the planner/exectuor consume activated skills via host-provided context functions).

## Extension Points

- **Custom discovery**: implement a directory provider and pass the resolved directory list to `NewSkillManager`.
- **Skill bundling**: a skill may carry resource files read safely via `ReadSkillResourceTool` (or `SafeResolvePath` directly).
- **Activation policy**: the host decides which skills to activate (router-matched + user-specified) and how to render their bodies in the system prompt.

## Related Specs

- [safeio.md](safeio.md) — read hardening (`ReadFile`, size cap, regular-files-only) behind SKILL.md and resource reads
- [orchestration/router.md](orchestration/router.md) — matches `SkillDescriptor`s to requests
- [orchestration/planner.md](orchestration/planner.md) — consumes activated skills via host context functions
- [tool-system/builtins.md](tool-system/builtins.md) — `read_skill_resource` tool
