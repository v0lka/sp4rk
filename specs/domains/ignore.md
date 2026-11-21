# Ignore Resolver

## Role

Multi-root `.gitignore`/`.aiignore` resolution: each root is walked once to collect and compile every nested ignore file into doublestar globs, and ignore queries are answered by whichever root contains the path. A pure algorithmic building block — hidden-dotfile and binary-file filtering stay caller-side concerns layered on top.

## Key Files

- `github.com/v0lka/sp4rk/ignore/ignore.go` - the entire package: `Resolver` (`NewResolver`, `NewResolverContext`, `NewResolverWithOptions`, `NewResolverWithOptionsContext`), `Options` (`IgnoreFiles`, `ExtraPatterns`), `Multi` (`NewMulti`, `NewMultiFromResolvers`, `RootFor`), `IgnoreChecker`, pattern compilation (`parsePatternLines`, `compile`, `analyze`), and the walk (`load`, `readIgnoreFile`)

## Behavior

**Per-root resolution.** `NewResolverWithOptionsContext` validates `Options` before any filesystem access, canonicalizes the root to an absolute, symlink-resolved form (longest-existing-prefix resolution, so a not-yet-fully-existing root loads cleanly and the stored root matches the form the tools emit — e.g. `/private/tmp` on macOS), detects the root filesystem's case sensitivity via `pathutil.DetectCaseInsensitive`, compiles `ExtraPatterns` first, and walks the root. The walk is cooperatively cancellable: once the context is done the walk abandons the tree and returns an error wrapping the context error; a cancelled build never publishes a half-collected resolver. `NewResolver`/`NewResolverWithOptions` are the `context.Background` forms.

**Walk and pruning.** `load` collects patterns only from regular files in the honoured name set (`.gitignore`, `.aiignore` by default; `Options.IgnoreFiles` narrows the set — an unknown name fails the build, and an empty non-nil slice honours no ignore file, leaving only `ExtraPatterns`). The walk always skips `.git` and prunes any directory already ignored by the patterns gathered so far — ignore files beneath an ignored directory contribute nothing (a known false negative for origin-scoped queries, safe for OR-merge consumers that already have git's own verdict).

**Pattern compilation.** A rule's anchoring follows gitignore semantics: a leading slash anchors to the file's directory, an internal slash anchors implicitly, a bare name matches at any depth, a trailing slash marks the rule directory-only. Blank lines, comments, and negation lines (`!`) are skipped — negation is intentionally unsupported and silently skipped. Escaped literal trailing spaces are not honoured (trailing spaces are stripped). Rules carry an origin (`.gitignore`, `.aiignore`, or `config` for `ExtraPatterns`).

**Matching.** `Match(relPath, isDir)` walks every ancestor of the root-relative path as a directory — once an ancestor matches, the path is ignored (the "ignored directory carries its contents" rule). `Ignored(absPath, isDir)` canonicalizes the absolute path through `ResolveExistingPrefix`, makes it root-relative, and delegates; paths that cannot be made relative to the root are never ignored. `IgnoredByAIIgnore` restricts matching to `.aiignore`-origin rules (`.gitignore` and `config` rules excluded), for callers that layer `.aiignore` verdicts on top of git's own `.gitignore` answer — which honours negation — instead of overriding it. The root itself, empty, and escaping paths are never ignored.

**Multi-root.** `Multi` holds one `Resolver` per root. `RootFor` returns the resolver whose root contains the path, using `pathutil.IsWithinPathFold` on case-insensitive roots and `pathutil.IsWithinPath` otherwise; a containment-check error skips that resolver. `Ignored`/`IgnoredByAIIgnore` return `false` when no root contains the path — paths outside all roots are never ignored. `NewMulti` fails all-or-nothing; `NewMultiFromResolvers` wraps pre-built resolvers (nil entries filtered, nil result when none survive) for per-root skip semantics. `Resolvers` returns a copy.

## Error Handling

- An unresolvable root, a failed walk, or a failed ignore-file read fails the build with a wrapping error — except a non-regular ignore-named entry, which is skipped and the walk continues: the os.Stat guard skips FIFOs/devices before opening, and `safeio.Open`'s fstat on the already-open descriptor catches a non-regular file swapped in after the guard (the Stat→Open race).
- `Multi.Ignored`/`RootFor` never fail on individual roots: an errored containment check skips the resolver at Debug level.
- Invalid `Options.IgnoreFiles` entries fail before any filesystem access.

## Invariants

- Only regular ignore files are ever opened: a FIFO, socket, or device named `.gitignore`/`.aiignore` is skipped instead of blocking the walk, on both the Stat guard and the open path.
- Ignore files beneath an ignored directory or beneath `.git` never contribute patterns.
- Negation patterns never change a verdict: `!`-prefixed lines are silently skipped.
- Every query honours ancestor-directory semantics: an ignored ancestor always ignores the path.
- Origin-scoped queries (`IgnoredByAIIgnore`) consider only `.aiignore`-sourced patterns; `.gitignore` and `config` patterns are excluded from them.
- A path outside every root is never ignored; the root itself and escaping paths are never ignored.
- The `.git` directory is always pruned from the walk.
- `ignore` never imports `tools`; both sides define `Ignored(absPath, isDir) bool`, so `ignore.Resolver` and `ignore.Multi` satisfy `tools.IgnoreChecker` structurally, and `IgnoreChecker` is the canonical interface name on this side.
- The package imports the standard library, `doublestar`, and the leaf `pathutil`/`safeio` packages; it never imports higher engine layers.

## Related Specs

- [safeio.md](safeio.md) - the non-blocking, regularity-checked opens behind ignore-file reads
- [Path Utilities](pathutil.md) - root canonicalization, case-sensitivity probing, and containment checks
- [Tool system](tool-system/README.md) - the `IgnoreChecker` the glob and ripgrep tools consume (satisfied structurally by `ignore.Resolver`/`ignore.Multi`)
