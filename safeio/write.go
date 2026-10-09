package safeio

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"

	"github.com/v0lka/sp4rk/pathutil"
)

// ErrSymlink marks a symbolic link where a real directory was required and
// none could be resolved: a link in a MkdirAllReal path that dangles or
// loops, or a link swapped into a component MkdirAllReal had set out to
// create. Pre-existing resolvable links are NOT errors — MkdirAllReal
// resolves them (see MkdirAllReal). Callers that must distinguish this from
// a genuine mkdir failure test it with errors.Is(err, ErrSymlink).
var ErrSymlink = errors.New("symlink in path")

// SymlinkError reports that a component of a path is a symbolic link where a
// real directory was required and the link could not be honored: the link
// dangles (its target — and everything under it — does not exist), or the
// link appeared in a component position while MkdirAllReal was creating the
// missing tail of the path. It matches errors.Is(err, ErrSymlink).
type SymlinkError struct {
	Path string
}

func (e *SymlinkError) Error() string { return e.Path + " is a symlink" }

// Is makes errors.Is(err, ErrSymlink) report true for any SymlinkError.
func (e *SymlinkError) Is(target error) bool { return target == ErrSymlink }

// ErrNotDirectory marks an existing path component that is not a directory
// where a real directory was required — a regular file, FIFO, socket, or
// device in a MkdirAllReal component position. It matches
// errors.Is(err, ErrNotDirectory) via the *fs.PathError that carries it.
var ErrNotDirectory = errors.New("not a directory")

// WriteFile writes data to the named file, creating it with the given
// permission bits (subject to the process umask) and truncating an existing
// file. It is the hardened replacement for os.WriteFile on any fixed,
// user- or workspace-controlled path: the open goes through the same
// non-blocking open plus already-open-descriptor regularity check as OpenFile
// (see the package comment), so a FIFO planted at path is refused instead of
// blocking the write-open forever.
//
// On unix the final path component is never followed: a symbolic link at path
// makes the open fail with ELOOP (*fs.PathError wrapping syscall.ELOOP) and
// the link's target is never written through. On Windows no O_NOFOLLOW
// equivalent exists without new dependencies, so a final symlink is still
// resolved and written through (see OpenFileNoFollow for the parity note).
//
// A missing file is created; a missing parent directory is an error (use
// MkdirAllReal first). A non-regular pre-existing target returns a
// *NotRegularError (or, when nothing holds the FIFO open, the fast ENXIO
// *fs.PathError from the non-blocking open — see OpenFile).
func WriteFile(path string, data []byte, perm fs.FileMode) error {
	f, err := OpenFileNoFollow(path, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, perm)
	if err != nil {
		return err
	}
	_, err = f.Write(data)
	if err1 := f.Close(); err == nil {
		err = err1
	}
	return err
}

// WriteFileAtomic writes data to the named file by creating a uniquely named
// temporary file in the target's directory (os.CreateTemp with a
// base+".tmp*" pattern — the random suffix means there is no plantable fixed
// .tmp path), chmod-ing it to perm, writing the data, fsync-ing the file, and
// then renaming it over path. The rename is an atomic replace: on unix
// rename(2) overwrites an existing entry without following it, so a symbolic
// link at path is replaced ITSELF — its target is never written through; on
// Windows os.Rename uses MoveFileEx with MOVEFILE_REPLACE_EXISTING and
// likewise replaces the entry.
//
// Atomicity means a reader of path sees either the complete old contents or
// the complete new contents — never a truncated intermediate state — and a
// crash mid-write leaves the old file intact. On any error the temporary file
// is closed and removed best-effort, leaving the target untouched.
//
// Permission bits: perm is applied with fchmod and is therefore NOT filtered
// through the process umask — unlike WriteFile/os.WriteFile, whose open(2)
// call applies it. Pass the exact bits you want (e.g. 0600 for state that
// must stay private): a caller porting from os.WriteFile(path, data, 0644) on
// a host with a restrictive umask gets a wider file than open(2) would have
// produced, so mask the mode itself if umask filtering is intended. On
// Windows Chmod only toggles the read-only attribute (no umask exists there).
//
// The parent directory of path must exist and be a real directory (use
// MkdirAllReal to create it).
func WriteFileAtomic(path string, data []byte, perm fs.FileMode) error {
	tmp, err := os.CreateTemp(filepath.Dir(path), filepath.Base(path)+".tmp*")
	if err != nil {
		return err
	}
	name := tmp.Name()

	if err := writeAndSync(tmp, data, perm); err != nil {
		_ = tmp.Close()
		_ = os.Remove(name)
		return err
	}
	if err := tmp.Close(); err != nil {
		_ = os.Remove(name)
		return err
	}
	if err := os.Rename(name, path); err != nil {
		_ = os.Remove(name)
		return err
	}
	return nil
}

// writeAndSync chmods f to perm, writes data, and fsyncs the file so the
// contents survive a crash before the rename publishes them.
func writeAndSync(f *os.File, data []byte, perm fs.FileMode) error {
	if err := f.Chmod(perm); err != nil {
		return err
	}
	if _, err := f.Write(data); err != nil {
		return err
	}
	return f.Sync()
}

// MkdirAllReal creates dir along with any missing parents, like os.MkdirAll,
// while guaranteeing that every directory it is asked to create is a REAL
// directory, never a path traversed through a symlink that appeared in a
// component position:
//
//   - Pre-existing symbolic links anywhere in the path are RESOLVED once, to
//     their real targets, before anything is created. Ancestor links are
//     ordinary system- or operator-created filesystem structure — macOS makes
//     /var a link to /private/var (so t.TempDir paths contain one), and an
//     operator is free to symlink the whole agent directory — so refusing
//     them would break every such environment. Each resolved target must be
//     a real directory: a link to a file (or any other pre-existing
//     non-directory — FIFO, socket, device) at a component position returns
//     a *fs.PathError wrapping ErrNotDirectory, and a dangling link returns
//     a *SymlinkError.
//   - The missing tail of the path is created component by component under
//     the resolved prefix. A link swapped into the component BEING created
//     makes its Mkdir fail with EEXIST, and the re-check refuses whatever
//     appeared — including a symlink — instead of following it. This is a
//     per-component guarantee: a link swapped into an ALREADY-created
//     intermediate component between two Mkdir calls can still redirect the
//     remainder of the tail (path-based APIs cannot pin intermediate dirs
//     without openat; pair MkdirAllReal with CheckRealDirsBelow where the
//     subtree is content-controlled and the residual window matters).
//
// os.MkdirAll resolves each missing component with a symlink-following Stat,
// so a link planted at any depth redirects the whole subtree at creation
// time; the per-component EEXIST re-check closes the same-component slice of
// that window, and the one-time resolution of pre-existing links keeps the
// operator's intent while making the resulting tree inspectable.
//
// A genuine mkdir failure passes os.Mkdir's *fs.PathError through. Permission
// bits are subject to the process umask, as with os.MkdirAll.
func MkdirAllReal(path string, perm fs.FileMode) error {
	if path == "" {
		return &fs.PathError{Op: "mkdirall", Path: path, Err: fs.ErrInvalid}
	}
	path = filepath.Clean(path)

	// Root of an absolute path: the volume name (e.g. "C:" or a UNC share on
	// Windows, "" on unix) plus a leading separator. Relative paths walk from
	// their first component with no root to check.
	vol := filepath.VolumeName(path)
	rest := path[len(vol):]
	isAbs := rest != "" && os.IsPathSeparator(rest[0])

	var prefix string
	if isAbs {
		prefix = vol + string(os.PathSeparator)
		rest = rest[1:]
		if err := checkRealDir(prefix); err != nil {
			return err
		}
	} else {
		prefix = vol
	}

	comps := strings.Split(rest, string(os.PathSeparator))
	for i, comp := range comps {
		if comp == "" {
			continue
		}
		next := prefix
		if next != "" && !os.IsPathSeparator(next[len(next)-1]) {
			next += string(os.PathSeparator)
		}
		next += comp

		fi, err := os.Lstat(next)
		switch {
		case err == nil && fi.Mode()&fs.ModeSymlink != 0:
			// Pre-existing link: resolve it once and continue from the
			// target. A dangling link has no target to resolve — refuse it
			// (there is nothing to create through, and creating the target
			// tree under the link's name would silently adopt a path the
			// operator left broken).
			resolved, rerr := filepath.EvalSymlinks(next)
			if rerr != nil {
				// No real directory can be resolved here when the link
				// dangles (ErrNotExist) or LOOPS — filepath.EvalSymlinks
				// walks links with its own counter and reports a loop as
				// the plain "EvalSymlinks: too many links" error, never as
				// a kernel ELOOP, so the loop is matched by message. Both
				// carry the SymlinkError marker; any other resolution
				// failure (permissions, I/O) passes through unchanged.
				if errors.Is(rerr, fs.ErrNotExist) || errors.Is(rerr, syscall.ELOOP) ||
					strings.Contains(rerr.Error(), "too many links") {
					return &SymlinkError{Path: next}
				}
				return rerr
			}
			if err := checkRealDir(resolved); err != nil {
				return err
			}
			prefix = resolved
		case err == nil:
			if err := checkRealDirInfo(next, fi); err != nil {
				return err
			}
			prefix = next
		case errors.Is(err, fs.ErrNotExist):
			// First missing component: everything from here on is created
			// under the RESOLVED prefix, and ensureRealDir refuses a symlink
			// swapped into any created component.
			for _, tail := range comps[i:] {
				if tail == "" {
					continue
				}
				t := prefix
				if t != "" && !os.IsPathSeparator(t[len(t)-1]) {
					t += string(os.PathSeparator)
				}
				t += tail
				if err := ensureRealDir(t, perm); err != nil {
					return err
				}
				prefix = t
			}
			return nil
		default:
			return err
		}
	}
	return nil
}

// ensureRealDir makes sure path is a real directory, creating it (with perm,
// subject to umask) when it does not exist.
func ensureRealDir(path string, perm fs.FileMode) error {
	fi, err := os.Lstat(path)
	if err == nil {
		return checkRealDirInfo(path, fi)
	}
	if !errors.Is(err, fs.ErrNotExist) {
		return err
	}
	// Missing: create it. A component swapped in between this Lstat and the
	// Mkdir makes Mkdir fail with EEXIST, and the re-check below refuses
	// whatever appeared — including a symlink.
	if err := os.Mkdir(path, perm); err == nil {
		return nil
	} else if !errors.Is(err, fs.ErrExist) {
		return err
	}
	fi, err = os.Lstat(path)
	if err != nil {
		return err
	}
	return checkRealDirInfo(path, fi)
}

// checkRealDir verifies that path is a real directory, refusing a symlink
// with a *SymlinkError and any other non-directory with an
// ErrNotDirectory-wrapped *fs.PathError.
func checkRealDir(path string) error {
	fi, err := os.Lstat(path)
	if err != nil {
		return err
	}
	return checkRealDirInfo(path, fi)
}

// checkRealDirInfo is checkRealDir against an already-fetched Lstat result
// (os.Lstat does not follow the path's own symlink, so the mode is authoritative).
// On Windows a directory junction is a name-surrogate reparse point that git
// and the Win32 APIs resolve transparently — it is refused exactly like a
// symlink (os.Lstat reports it ModeIrregular there, never ModeSymlink).
func checkRealDirInfo(path string, fi fs.FileInfo) error {
	if fi.Mode()&fs.ModeSymlink != 0 || (runtime.GOOS == "windows" && fi.Mode()&fs.ModeIrregular != 0) {
		return &SymlinkError{Path: path}
	}
	if !fi.IsDir() {
		return &fs.PathError{Op: "mkdirall", Path: path, Err: ErrNotDirectory}
	}
	return nil
}

// CheckRealDirsBelow verifies that every component of path strictly BELOW
// from — which must be a lexical path-prefix ancestor of path — is a real
// directory: a symbolic link returns a *SymlinkError (errors.Is(err,
// ErrSymlink)) and any other non-directory an ErrNotDirectory-wrapped
// *fs.PathError, in both cases with nothing created or followed. The `from`
// root itself and the ancestors above it are NOT inspected: system- or
// operator-created links there (macOS /var → /private/var, a symlinked home
// directory) are ordinary structure, not content.
//
// Pair it with MkdirAllReal when the subtree below from is repository- or
// content-controlled: MkdirAllReal resolves pre-existing links by contract
// (so ordinary environments work), and this check then refuses a link
// SHIPPED inside the controlled subtree that resolution would otherwise
// honor. Both calls are racing-prone individually; together they shrink the
// window to the space between them, the same guarantee class as
// OpenFileNoFollow's final-component check.
func CheckRealDirsBelow(from, path string) error {
	from = filepath.Clean(from)
	path = filepath.Clean(path)
	rel, err := filepath.Rel(from, path)
	if err != nil {
		return &fs.PathError{Op: "checkrealdirs", Path: path, Err: err}
	}
	if rel == "." {
		return nil
	}
	if rel == ".." || strings.HasPrefix(rel, ".."+string(os.PathSeparator)) {
		return &fs.PathError{Op: "checkrealdirs", Path: path, Err: fs.ErrInvalid}
	}
	cur := from
	for _, comp := range strings.Split(rel, string(os.PathSeparator)) {
		if comp == "" || comp == "." {
			continue
		}
		cur = filepath.Join(cur, comp)
		fi, err := os.Lstat(cur)
		if err != nil {
			return err
		}
		if err := checkRealDirInfo(cur, fi); err != nil {
			return err
		}
	}
	return nil
}

// MkdirAllRealWithin creates dir (with any missing parents) exactly like
// [MkdirAllReal] and additionally enforces that the RESULTING tree stays
// within a containment boundary: every pre-existing symbolic link anywhere
// in dir's path — at any component depth — must resolve to a target inside
// root, or the whole call fails with a *SymlinkError and nothing is created.
//
// root is the app-owned anchor the caller's path is derived from (the agent
// directory, a session directory, ...). It is resolved first, so an operator
// symlink on root itself (or on any ancestor above it — macOS's /var) stays
// "operator intent": the boundary is the FULLY RESOLVED root, and a whole-
// tree symlink therefore keeps working. Links BELOW the boundary are a
// different matter — the agent tree is content-controlled and a link planted
// there redirects every subsequent write out of it (the containment-bypass
// class this function closes), so a link resolving outside root is refused
// even though plain MkdirAllReal would have resolved it as intent.
//
// The check runs BEFORE any creation: dir's longest existing chain is fully
// resolved and the not-yet-existing tail is joined lexically, and the result
// must land inside the boundary — a planted link can therefore not coax a
// single mkdir or file creation out of the boundary. After MkdirAllReal the
// finished dir is resolved once more as defense in depth (catching a link
// swapped into the tail between the two checks; the per-component EEXIST
// re-check already refuses same-component swaps).
//
// Both paths must be absolute (containment across relative paths is not
// defined). A genuine mkdir failure passes os.Mkdir's *fs.PathError through;
// an unresolvable root passes its EvalSymlinks error through — the anchor is
// expected to exist (callers create it before deriving any paths below it).
func MkdirAllRealWithin(root, dir string, perm fs.FileMode) error {
	if root == "" {
		return &fs.PathError{Op: "mkdirallwithin", Path: root, Err: fs.ErrInvalid}
	}
	root = filepath.Clean(root)
	dir = filepath.Clean(dir)

	boundary, err := filepath.EvalSymlinks(root)
	if err != nil {
		return err
	}
	// Pre-check, before any creation: where would dir's real tree land?
	land := pathutil.ResolveExistingPrefix(dir)
	within, err := pathutil.IsWithinPath(boundary, land)
	if err != nil {
		return err
	}
	if !within {
		return &SymlinkError{Path: dir}
	}
	if err := MkdirAllReal(dir, perm); err != nil {
		return err
	}
	// Post-check: resolve the now-existing dir once more. A link swapped
	// into the created tail between the pre-check and the creation (or any
	// silent degradation of ResolveExistingPrefix) must not survive.
	resolved, err := filepath.EvalSymlinks(dir)
	if err != nil {
		return err
	}
	within, err = pathutil.IsWithinPath(boundary, resolved)
	if err != nil {
		return err
	}
	if !within {
		return &SymlinkError{Path: dir}
	}
	return nil
}
