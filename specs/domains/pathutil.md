# Path Utilities

## Role

Reusable filesystem-path algorithms with no knowledge of any application's directory layout: symlink-aware containment checks, component splitting, filesystem case-sensitivity probing, longest-existing-prefix symlink resolution, and pure duplicate-component repair. Safe for use from any layer.

## Key Files

- `github.com/v0lka/sp4rk/pathutil/pathutil.go` - the entire exported surface: `IsWithinPath`, `IsWithinPathFold`, `SplitPathComponents`, `DetectCaseInsensitive`, `ResolveExistingPrefix`, `CollapseDuplicateComponents`

## Behavior

**Containment (`IsWithinPath` / `IsWithinPathFold`).** Reports whether child equals or descends from parent. Both paths are symlink-resolved through their longest existing prefix (`ResolveExistingPrefix`), so OS-level ancestor symlinks (macOS `/var` → `/private/var`) are handled even for not-yet-existing paths. The child is resolved in OS component order from its raw spelling — never lexically cleaned first — so a `symlink/..` sequence is evaluated the way the OS evaluates it: the link resolves, and `..` climbs from the link TARGET's directory, which correctly rejects `<parent>/<link-outside>/../x`. `IsWithinPathFold` lowercases both resolved sides before the same prefix check, mirroring case-insensitive filesystems; containment decided by an explicit separator-terminated prefix test (never `filepath.Rel`, whose Windows comparison is case-insensitive), with a volume-mismatch error for different Windows drives. Both paths must be absolute; an empty parent fails closed with an error. Case folding never weakens escape prevention — it can only mark an otherwise-mismatched path as within.

**Component splitting (`SplitPathComponents`).** Splits a cleaned absolute path into non-empty components, stripping the volume prefix and root separator. Splitting follows the platform's separator RULES: on Windows both `\` and `/` separate, so forward-slash spellings yield the same components as backslash ones; on unix a backslash is an ordinary filename byte.

**Case-sensitivity probe (`DetectCaseInsensitive`).** Probes whether the filesystem at dir folds letter case, by creating a uniquely-named mixed-case temporary file and checking whether the opposite-cased name resolves to it. Results are memoized per probed directory for the process lifetime (including failures), and the probe runs at most once per directory under a mutex. When dir does not exist or is not writable, the probe climbs to the nearest existing ancestor; if none is usable it returns `false` — the fail-safe default, since a wrong "case-insensitive" answer can turn a non-local path into a "local" one.

**Symlink resolution (`ResolveExistingPrefix`).** Resolves symlinks on the longest existing prefix of path and joins the raw non-existent suffix back, so partially-existing paths are still symlink-checked. The climb cuts raw components without cleaning, so `symlink/..` sequences survive to the next resolution attempt — cleaning there would erase the link before it resolves. The resolved prefix is joined to the raw remainder through `filepath.Join`, whose single Clean is safe because everything in the resolved prefix is a real directory by construction, so it matches what the OS would do with the remaining components. Permission errors and relative paths at their top return the input unchanged.

**Duplicate repair (`CollapseDuplicateComponents`).** A pure, filesystem-free repair of spellings with a duplicated leading component run (`/a/b/a/b/c` → `/a/b/c`) or an immediately repeated component (`/x/x/f.txt` → `/x/f.txt`), in one deterministic pass, longest leading run first. A spelling containing any `.`/`..` component is returned unchanged — traversal components are semantically load-bearing, and removing any component would re-point every later `..` climb. No-duplication spellings come back byte-for-byte unchanged (no separator normalization), relative and volume-relative spellings are untouched, and the volume prefix plus root separator are preserved exactly as written (a forward-slash UNC spelling stays forward-slash). The caller decides when a repair is worth attempting and must verify containment of the result before using it.

## Error Handling

- `IsWithinPath`/`IsWithinPathFold` return an error only for an empty parent (containment cannot be determined) or a volume mismatch between parent and child; every other outcome is a boolean.
- `DetectCaseInsensitive`, `ResolveExistingPrefix`, and `CollapseDuplicateComponents` never fail: the probe degrades to `false`, and resolution/repair return their input unchanged when they cannot improve it.

## Invariants

- Containment is always decided on symlink-resolved, raw-component-order paths: a `symlink/..` escape is evaluated exactly as the OS would evaluate it, and a cleaned-first misjudgment cannot occur.
- The separator-terminated prefix test always rejects a prefix-only sibling whose name merely starts with the parent's last component (`/usr` vs `/us`).
- `CollapseDuplicateComponents` is pure: it never touches the filesystem and never alters a spelling beyond removing duplicated components — separator style, volume prefix, and dot-bearing spellings are always preserved exactly as written.
- `DetectCaseInsensitive` creates its probe file at most once per directory per process, always removes it before returning, and returns `false` whenever it cannot obtain a trustworthy answer.
- The package imports only the standard library; it never imports higher engine layers.

## Related Specs

- [safeio.md](safeio.md) - `MkdirAllRealWithin` resolves its containment boundary through `ResolveExistingPrefix` and `IsWithinPath`
- [Skills](skills.md) - `SafeResolvePath` resolves the joined resource path through `ResolveExistingPrefix` and verifies containment with `IsWithinPath`
- [Architecture Layers](../architecture/layers.md) - pathutil is a leaf utility package; the import rule keeps it free of upward dependencies
