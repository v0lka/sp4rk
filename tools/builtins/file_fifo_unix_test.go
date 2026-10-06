//go:build unix

package builtins

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"syscall"
	"testing"
	"time"

	"github.com/v0lka/sp4rk/safeio"
)

// fifoWatchdogTimeout bounds how long a FIFO operation may take before the
// test declares a hang. A read-open (O_RDONLY) of a FIFO blocks inside open(2)
// until a writer appears, where no context check or deadline can interrupt it;
// safeio.Open's O_NONBLOCK open plus post-open fstat must instead refuse the
// FIFO promptly. The bound turns a regression into a deterministic failure
// rather than a test that hangs forever.
const fifoWatchdogTimeout = 5 * time.Second

// toolOutcome captures the parts of a tool result the FIFO tests assert on,
// without naming the tools.ToolResult type.
type toolOutcome struct {
	content string
	isError bool
	err     error
}

// mkFifoForTest creates a named pipe (FIFO) in a fresh temp dir and returns its
// path. Nothing ever opens the write end, so any plain read-open would block.
func mkFifoForTest(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "pipe")
	if err := syscall.Mkfifo(path, 0o644); err != nil {
		t.Fatalf("mkfifo: %v", err)
	}
	return path
}

// withFifoWatchdog runs fn in a goroutine and fails the test if it does not
// return within fifoWatchdogTimeout. It returns whatever fn produced.
func withFifoWatchdog[T any](t *testing.T, what string, fn func() T) T {
	t.Helper()
	done := make(chan T, 1)
	go func() { done <- fn() }()
	select {
	case out := <-done:
		return out
	case <-time.After(fifoWatchdogTimeout):
		t.Fatalf("%s hung on a FIFO (no return in %s)", what, fifoWatchdogTimeout)
		var zero T
		return zero
	}
}

// TestReadFileRange_RejectsFifoWithoutBlocking proves the streaming read path
// refuses a FIFO promptly with a clean, classifiable error instead of blocking
// inside open(2).
func TestReadFileRange_RejectsFifoWithoutBlocking(t *testing.T) {
	fifo := mkFifoForTest(t)

	type outcome struct {
		res *FileReadResult
		err error
	}
	out := withFifoWatchdog(t, "ReadFileRange", func() outcome {
		res, err := ReadFileRange(FileReadParams{Path: fifo, DefaultLines: 10})
		return outcome{res: res, err: err}
	})

	if out.err == nil {
		t.Fatalf("ReadFileRange on a FIFO returned a nil error (result %+v)", out.res)
	}
	if !errors.Is(out.err, safeio.ErrNotRegular) {
		t.Fatalf("ReadFileRange error = %v, want errors.Is(err, safeio.ErrNotRegular)", out.err)
	}
}

// TestReadSingleLine_RejectsFifoWithoutBlocking covers the tool_result_read
// escape hatch, which opens the file independently of ReadFileRange.
func TestReadSingleLine_RejectsFifoWithoutBlocking(t *testing.T) {
	fifo := mkFifoForTest(t)

	type outcome struct {
		line  string
		total int
		err   error
	}
	out := withFifoWatchdog(t, "ReadSingleLine", func() outcome {
		line, total, err := ReadSingleLine(fifo, 1)
		return outcome{line: line, total: total, err: err}
	})

	if out.err == nil {
		t.Fatalf("ReadSingleLine on a FIFO returned a nil error (line %q, total %d)", out.line, out.total)
	}
	if !errors.Is(out.err, safeio.ErrNotRegular) {
		t.Fatalf("ReadSingleLine error = %v, want errors.Is(err, safeio.ErrNotRegular)", out.err)
	}
}

// TestReadFileTool_Execute_RejectsFifo proves the read_file tool surfaces a
// clean IsError result (not a hang) when its target is a FIFO.
func TestReadFileTool_Execute_RejectsFifo(t *testing.T) {
	fifo := mkFifoForTest(t)
	tool := NewReadFileTool()
	input, _ := json.Marshal(ReadFileInput{Path: fifo})

	out := withFifoWatchdog(t, "read_file", func() toolOutcome {
		res, err := tool.Execute(context.Background(), input)
		return toolOutcome{content: res.Content, isError: res.IsError, err: err}
	})

	if out.err != nil {
		t.Fatalf("read_file Execute returned a transport error: %v", out.err)
	}
	if !out.isError {
		t.Fatalf("read_file on a FIFO was not flagged as an error: %q", out.content)
	}
}

// TestEditFileTool_Execute_RejectsFifo proves the edit_file tool surfaces a
// clean IsError result (not a hang) when its target is a FIFO — the read leg
// of the read-modify-write is what previously blocked.
func TestEditFileTool_Execute_RejectsFifo(t *testing.T) {
	fifo := mkFifoForTest(t)
	tool := NewEditFileTool()
	input, _ := json.Marshal(EditFileInput{Path: fifo, OldString: "a", NewString: "b"})

	out := withFifoWatchdog(t, "edit_file", func() toolOutcome {
		res, err := tool.Execute(context.Background(), input)
		return toolOutcome{content: res.Content, isError: res.IsError, err: err}
	})

	if out.err != nil {
		t.Fatalf("edit_file Execute returned a transport error: %v", out.err)
	}
	if !out.isError {
		t.Fatalf("edit_file on a FIFO was not flagged as an error: %q", out.content)
	}
}

// TestWriteFileTool_Execute_ReplacesFifoWithoutBlocking validates the documented
// FIFO-safety of atomicWriteFile: the target is never open()ed — a regular temp
// file is created next to it and renamed over it — so writing through a FIFO
// path neither blocks nor fails; it replaces the FIFO with a regular file.
func TestWriteFileTool_Execute_ReplacesFifoWithoutBlocking(t *testing.T) {
	fifo := mkFifoForTest(t)
	tool := NewWriteFileTool()
	input, _ := json.Marshal(WriteFileInput{Path: fifo, Content: "replaced"})

	out := withFifoWatchdog(t, "write_file", func() toolOutcome {
		res, err := tool.Execute(context.Background(), input)
		return toolOutcome{content: res.Content, isError: res.IsError, err: err}
	})

	if out.err != nil {
		t.Fatalf("write_file Execute returned a transport error: %v", out.err)
	}
	if out.isError {
		t.Fatalf("write_file over a FIFO failed: %q", out.content)
	}

	info, err := os.Stat(fifo)
	if err != nil {
		t.Fatalf("stat after write: %v", err)
	}
	if !info.Mode().IsRegular() {
		t.Fatalf("write over a FIFO left a non-regular file: mode %v", info.Mode())
	}
	got, err := os.ReadFile(fifo)
	if err != nil {
		t.Fatalf("read after write: %v", err)
	}
	if string(got) != "replaced" {
		t.Fatalf("content = %q, want %q", got, "replaced")
	}
}
