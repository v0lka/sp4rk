package tools

import (
	"context"
	"encoding/json"
)

// CacheMode controls how a tool result is stored in the tool-result cache.
//
// It is consulted by the executor only for read-style tools (read_file) whose
// result can be either streamed from the file on disk (file-backed) or stored
// in memory (content-backed). The default keeps the existing heuristic; a read
// wrapper that returns a transformed view of a file opts into content-backed
// caching via CacheModeContentBacked.
type CacheMode int

const (
	// CacheModeDefault keeps the executor's existing caching heuristic:
	// read_file is file-backed (the file on disk is the cache backing store),
	// everything else is content-backed. This is the zero value, so every
	// tool that does not opt in behaves exactly as before.
	CacheModeDefault CacheMode = iota
	// CacheModeContentBacked forces content-backed caching even for read_file.
	// The full result is stored in memory and tool_result_read paginates it,
	// rather than streaming fragments from the file on disk. This is required
	// by read wrappers that return a transformed representation of a file
	// (e.g. a decoded or format-converted view) where the bytes on disk are
	// not a valid backing store.
	CacheModeContentBacked
)

// ContentBackedReader is an optional interface implemented by read tools whose
// output is a transformed representation of the file (e.g. decoded, decrypted,
// or format-converted) rather than its raw bytes.
//
// When IsContentBacked reports true for a given input, the executor caches the
// result in memory (content-backed) instead of treating the file on disk as the
// backing store (file-backed). This ensures tool_result_read paginates the
// transformed content rather than re-reading raw bytes from disk. The file
// coherence metadata (path + mtime + size) is still attached to the cache entry
// so the executor can detect when the source file changes.
//
// The decision is per-input so the same tool can return raw bytes for some
// files (e.g. plain text) and a transformed view for others (e.g. binary
// documents). Implementations must be cheap and side-effect free: the executor
// calls IsContentBacked on the caching hot path for every read_file result.
type ContentBackedReader interface {
	// IsContentBacked reports whether the read of the file described by input
	// produces a transformed/decoded view that must be cached in memory. The
	// input is the tool's raw JSON arguments (e.g. {"path": "..."}). Returning
	// false (or not implementing the interface) keeps the default file-backed
	// behavior.
	IsContentBacked(ctx context.Context, input json.RawMessage) bool
}
