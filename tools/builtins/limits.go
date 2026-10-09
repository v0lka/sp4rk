package builtins

import (
	"bytes"
	"fmt"
	"time"
)

// maxShellOutputBytes caps how much of a child shell's combined stdout+stderr
// the bash_exec/posh_exec tools buffer in memory. The per-call timeout bounds
// duration, not peak memory: without this cap a command like `yes` or a
// runaway build log grows the buffer until the host is OOM-killed, and the
// executor's central truncation layer only runs after Execute returns. Bytes
// past the cap are still drained (so the child never blocks on a full pipe)
// but discarded; the tool result carries a truncation marker naming the cap.
const maxShellOutputBytes = 10 << 20 // 10 MiB

// boundedOutputBuffer is an io.Writer that stores at most max bytes and
// discards — while continuing to count — anything written past the cap.
// Writes never fail and always report the full length, so a child process
// keeps running and its pipes keep draining after the cap is reached.
//
// bash_exec and posh_exec pass the SAME *boundedOutputBuffer value as both
// cmd.Stdout and cmd.Stderr: os/exec detects the equal writer values and
// serializes all writes through one goroutine, so the buffer needs no
// internal locking.
type boundedOutputBuffer struct {
	buf     bytes.Buffer
	max     int
	dropped int64
}

// newBoundedOutputBuffer returns a buffer storing at most limit bytes.
func newBoundedOutputBuffer(limit int) *boundedOutputBuffer {
	return &boundedOutputBuffer{max: limit}
}

// Write stores p up to the remaining cap and discards the rest, counting the
// discarded bytes. It always consumes the whole slice.
func (b *boundedOutputBuffer) Write(p []byte) (int, error) {
	if room := b.max - b.buf.Len(); room > 0 {
		if len(p) > room {
			b.buf.Write(p[:room])
			b.dropped += int64(len(p) - room)
		} else {
			b.buf.Write(p)
		}
	} else {
		b.dropped += int64(len(p))
	}
	return len(p), nil
}

// Bytes returns the stored output without any truncation marker.
func (b *boundedOutputBuffer) Bytes() []byte { return b.buf.Bytes() }

// marker returns the incompleteness notice for the bytes discarded past the
// cap, or "" when nothing was discarded. The wording mirrors the tools'
// other truncation notices.
func (b *boundedOutputBuffer) marker() string {
	if b.dropped == 0 {
		return ""
	}
	return fmt.Sprintf("\n[output truncated — %d bytes discarded (cap %d bytes)]", b.dropped, b.max)
}

// String returns the stored output, plus the truncation marker when anything
// was discarded past the cap.
func (b *boundedOutputBuffer) String() string {
	return b.buf.String() + b.marker()
}

// FileLimits holds configurable limits for file operation tools.
type FileLimits struct {
	ReadDefaultLines int // max lines per read call when no explicit range is given
	MaxLineBytes     int // per-line byte cap; lines exceeding this are truncated (0 = no cap)
	MaxWindowLines   int // hard cap on lines returned per call even for explicit ranges (0 = no cap)
}

// BashTimeouts holds configurable timeout values for the bash_exec tool.
// A zero BashTimeouts (or one with a non-positive MaxTimeout) is replaced by
// DefaultBashTimeouts() in the NewBashExecTool*/NewPoshExecTool* constructors,
// so a zero-value config cannot derive a zero per-call deadline (which would
// cancel every command before it starts). The bounds are a property of the
// tool, not of every caller.
type BashTimeouts struct {
	MaxTimeout time.Duration // maximum allowed timeout for bash commands
	WaitDelay  time.Duration // grace period for pipe readers after process kill
}

// DefaultBashTimeouts returns the default timeouts for bash_exec.
func DefaultBashTimeouts() BashTimeouts {
	return BashTimeouts{
		MaxTimeout: 120 * time.Second,
		WaitDelay:  5 * time.Second,
	}
}

// normalizeBashTimeouts repairs caller-supplied BashTimeouts so a per-call
// deadline can never be zero: a completely zero struct is replaced by
// DefaultBashTimeouts() (mirroring NewGlobToolWithLimits), and an otherwise
// populated struct with a non-positive MaxTimeout gets the default maximum
// while keeping the caller's WaitDelay. See the BashTimeouts doc comment.
func normalizeBashTimeouts(t BashTimeouts) BashTimeouts {
	switch {
	case t == (BashTimeouts{}):
		return DefaultBashTimeouts()
	case t.MaxTimeout <= 0:
		t.MaxTimeout = DefaultBashTimeouts().MaxTimeout
	}
	return t
}

// DefaultFileLimits returns the default limits for file operation tools.
func DefaultFileLimits() FileLimits {
	return FileLimits{
		ReadDefaultLines: 2000,
		MaxLineBytes:     1 << 20, // 1 MiB
		MaxWindowLines:   50000,
	}
}

// RipgrepLimits holds configurable limits for the ripgrep tool.
// A zero RipgrepLimits (or one with a non-positive Timeout) is replaced by
// DefaultRipgrepLimits() in the NewRipgrepTool* constructors, so a zero-value
// config cannot derive a zero per-call deadline (which would abort every
// search before it starts).
type RipgrepLimits struct {
	Timeout time.Duration // timeout for ripgrep search operations
}

// DefaultRipgrepLimits returns the default limits for ripgrep.
func DefaultRipgrepLimits() RipgrepLimits {
	return RipgrepLimits{
		Timeout: 60 * time.Second,
	}
}

// GlobLimits holds configurable limits for the glob tool. Together they make a
// single glob walk bounded: a filesystem runaway (a symlink loop or an
// enormous directory tree) must neither hang the tool nor exhaust memory.
// A completely zero GlobLimits is replaced by DefaultGlobLimits() in
// NewGlobToolWithLimits, so a zero-value GlobLimits cannot register an
// unbounded walk (the bounds are a property of the tool, not of every caller).
// An individual zero field on an otherwise-populated struct is honored as
// "disabled" (see the field comments below). A host that has already
// distinguished "unset" from "explicitly zero on every knob" and wants the
// all-zero combination honored verbatim registers via
// NewGlobToolWithLimitsOverride, which performs no substitution.
type GlobLimits struct {
	MaxEntries int           // max filesystem entries visited (directory entries read, plus Open/Stat calls) before the walk is aborted (0 = no entry budget)
	MaxResults int           // max matching paths collected before the walk is aborted (0 = unlimited)
	Timeout    time.Duration // wall-clock timeout for a single glob walk (0 = no timeout)
}

// DefaultGlobLimits returns the default limits for glob.
func DefaultGlobLimits() GlobLimits {
	return GlobLimits{
		MaxEntries: 500000,
		MaxResults: 10000,
		Timeout:    30 * time.Second,
	}
}

// WebFetchLimits holds configurable limits for the web_fetch tool.
// A non-positive Timeout is replaced by DefaultWebFetchLimits().Timeout in
// NewWebFetchToolWithClient: unlike a timeout, "no limit" is never a valid
// configuration for web_fetch — it would let a stalled origin hang the tool
// run indefinitely.
type WebFetchLimits struct {
	Timeout time.Duration // timeout for HTTP requests; doubled on each retry attempt
	Retries int           // number of retries after the initial attempt; 0 disables retrying
}

// DefaultWebFetchLimits returns the default limits for web_fetch. The 30s
// timeout matches the tool description and docs/tools.md ("Requests time out
// after 30 seconds by default").
func DefaultWebFetchLimits() WebFetchLimits {
	return WebFetchLimits{
		Timeout: 30 * time.Second,
		Retries: 2,
	}
}

// WebSearchLimits holds configurable limits for the web_search tool.
type WebSearchLimits struct {
	MaxResults int           // max number of search results
	Timeout    time.Duration // timeout for search provider HTTP requests
}

// DefaultWebSearchLimits returns the default limits for web_search.
func DefaultWebSearchLimits() WebSearchLimits {
	return WebSearchLimits{
		MaxResults: 5,
		Timeout:    30 * time.Second,
	}
}
