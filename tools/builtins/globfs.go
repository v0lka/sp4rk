package builtins

import (
	"context"
	"errors"
	"io"
	"io/fs"
)

// Sentinel errors surfaced by boundedFS and the glob walk callback. The walk
// runs with doublestar.WithFailOnIOErrors, so an error returned here from any
// fs.FS method propagates out of GlobWalk unchanged and is translated into a
// clear ToolResult message by GlobTool.Execute.
var (
	// errGlobCanceled is returned once the walk's context is done (canceled or
	// deadline exceeded). It is what makes a runaway walk interruptible.
	errGlobCanceled = errors.New("glob canceled")
	// errGlobEntryBudget is returned once the entries-visited budget is spent.
	// It bounds the total work of a single walk (symlink loops, huge trees).
	errGlobEntryBudget = errors.New("glob entry budget exceeded")
	// errGlobResultsLimited is returned by the walk callback once MaxResults
	// matches have been collected, stopping the walk early.
	errGlobResultsLimited = errors.New("glob results limited")
)

// readDirBatch is the number of directory entries read per fs.ReadDirFile
// ReadDir call while streaming a directory. Reading in bounded batches keeps a
// single huge directory interruptible (the walk context is re-checked on every
// entry) and memory-bounded (the entry budget aborts the read instead of
// materializing the whole directory slice at once).
const readDirBatch = 256

// Compile-time interface checks: boundedFS must satisfy every fs interface
// doublestar consults. It reads directories incrementally via fs.ReadDirFile
// (fs.ReadDirFS) and stats paths via fs.Stat (fs.StatFS), falling back to Open
// otherwise.
var (
	_ fs.FS        = (*boundedFS)(nil)
	_ fs.ReadDirFS = (*boundedFS)(nil)
	_ fs.StatFS    = (*boundedFS)(nil)
)

// boundedFS wraps an inner fs.FS to make a doublestar walk non-following,
// interruptible and bounded. Every filesystem access it performs (Open,
// ReadDir, Stat) checks the walk context first and is then charged against an
// entries-visited budget, so a symlink loop or an enormous directory tree can
// no longer hang or exhaust the walk: the access returns errGlobCanceled
// (context done) or errGlobEntryBudget (budget spent) instead, which surfaces
// through doublestar.WithFailOnIOErrors.
//
// Directory entries are charged one at a time as the directory is streamed
// (ReadDir reads in readDirBatch-sized batches), so the budget counts entries,
// not directory accesses: a single directory holding more entries than the
// budget is aborted mid-read rather than read whole.
//
// Symbolic links are not followed because the walk is run with
// doublestar.WithNoFollow, so a self-referential or escaping symlink is never
// traversed. The double guard is deliberate: WithNoFollow closes the common
// loop, and the budgets bound any other filesystem runaway.
//
// The walk is single-goroutine and strictly sequential, so the plain seen
// counter needs no synchronization.
type boundedFS struct {
	ctx   context.Context
	inner fs.FS
	lim   GlobLimits
	seen  int
}

// newBoundedFS wraps inner with the given limits and walk context.
func newBoundedFS(ctx context.Context, inner fs.FS, lim GlobLimits) *boundedFS {
	return &boundedFS{ctx: ctx, inner: inner, lim: lim}
}

// charge enforces both guards: it returns errGlobCanceled as soon as the
// context is done, then errGlobEntryBudget once more than the configured
// number of entries has been visited (a non-positive MaxEntries disables the
// entry budget). n is the number of entries this access represents.
func (b *boundedFS) charge(n int) error {
	if err := b.ctx.Err(); err != nil {
		return errGlobCanceled
	}
	b.seen += n
	if b.lim.MaxEntries > 0 && b.seen > b.lim.MaxEntries {
		return errGlobEntryBudget
	}
	return nil
}

// Open implements fs.FS.
func (b *boundedFS) Open(name string) (fs.File, error) {
	if err := b.charge(1); err != nil {
		return nil, err
	}
	return b.inner.Open(name)
}

// Stat implements fs.StatFS.
func (b *boundedFS) Stat(name string) (fs.FileInfo, error) {
	if err := b.charge(1); err != nil {
		return nil, err
	}
	return fs.Stat(b.inner, name)
}

// ReadDir implements fs.ReadDirFS. Instead of a one-shot fs.ReadDir it streams
// the directory through fs.ReadDirFile.ReadDir in readDirBatch-sized reads,
// charging each entry against the entry budget, so a single huge directory is
// bounded and interruptible rather than materialized into one uncancellable
// slice.
func (b *boundedFS) ReadDir(name string) ([]fs.DirEntry, error) {
	if err := b.ctx.Err(); err != nil {
		return nil, errGlobCanceled
	}
	f, err := b.inner.Open(name)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()

	rdf, ok := f.(fs.ReadDirFile)
	if !ok {
		// The filesystem cannot stream directory reads; fall back to a
		// one-shot read, still charged against the budget after the fact.
		entries, err := fs.ReadDir(b.inner, name)
		if err != nil {
			return nil, err
		}
		if err := b.charge(len(entries)); err != nil {
			return nil, err
		}
		return entries, nil
	}

	var all []fs.DirEntry
	for {
		batch, err := rdf.ReadDir(readDirBatch)
		for _, entry := range batch {
			if cerr := b.charge(1); cerr != nil {
				return nil, cerr
			}
			all = append(all, entry)
		}
		if err != nil {
			if errors.Is(err, io.EOF) {
				return all, nil
			}
			return nil, err
		}
	}
}
