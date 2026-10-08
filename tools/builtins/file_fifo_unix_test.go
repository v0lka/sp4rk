//go:build unix

package builtins

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
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

// TestWriteFileTool_Execute_RefusesFifoWithoutBlocking validates the fix for
// review finding #56: a FIFO is not a regular file, so write_file must refuse
// to replace it — the atomic rename would otherwise silently overwrite the
// device/FIFO node with a regular file. The refusal must also not block: the
// target is probed with os.Stat (which never opens the FIFO), so the call
// returns promptly instead of hanging inside open(2).
func TestWriteFileTool_Execute_RefusesFifoWithoutBlocking(t *testing.T) {
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
	if !out.isError {
		t.Fatalf("write_file over a FIFO was not refused: %q", out.content)
	}
	if !strings.Contains(out.content, "not a regular file") {
		t.Errorf("expected a not-a-regular-file refusal, got: %q", out.content)
	}

	// The FIFO must survive untouched.
	info, err := os.Stat(fifo)
	if err != nil {
		t.Fatalf("stat after refused write: %v", err)
	}
	if info.Mode()&os.ModeNamedPipe == 0 {
		t.Fatalf("refused write clobbered the FIFO: mode %v", info.Mode())
	}
}

// TestDeleteFileTool_Execute_RefusesFifo validates the delete_file half of the
// non-regular-target refusal (#56): a FIFO must not be unlinked — the tool
// reports an error and the FIFO survives.
func TestDeleteFileTool_Execute_RefusesFifo(t *testing.T) {
	fifo := mkFifoForTest(t)
	tool := NewDeleteFileTool()
	input, _ := json.Marshal(DeleteFileInput{Path: fifo})

	out := withFifoWatchdog(t, "delete_file", func() toolOutcome {
		res, err := tool.Execute(context.Background(), input)
		return toolOutcome{content: res.Content, isError: res.IsError, err: err}
	})

	if out.err != nil {
		t.Fatalf("delete_file Execute returned a transport error: %v", out.err)
	}
	if !out.isError {
		t.Fatalf("delete_file of a FIFO was not refused: %q", out.content)
	}
	if !strings.Contains(out.content, "not a regular file") {
		t.Errorf("expected a not-a-regular-file refusal, got: %q", out.content)
	}
	if _, err := os.Stat(fifo); err != nil {
		t.Fatalf("FIFO did not survive the refused delete: %v", err)
	}
}

// TestWriteFileTool_Execute_RefusesDevNull validates the harmless-device half
// of the non-regular-target refusal (review finding #56): /dev/null (os.DevNull)
// is judge-exempted as a harmless device, but it is a character device, not a
// regular file — write_file must refuse to rename a regular file over it even
// though the judge would auto-approve. POSIX-only: on Windows NUL is a
// reserved name with different semantics, so the case is skipped there.
func TestWriteFileTool_Execute_RefusesDevNull(t *testing.T) {
	tool := NewWriteFileTool()
	input, _ := json.Marshal(WriteFileInput{Path: os.DevNull, Content: "clobber"})

	out := withFifoWatchdog(t, "write_file "+os.DevNull, func() toolOutcome {
		res, err := tool.Execute(context.Background(), input)
		return toolOutcome{content: res.Content, isError: res.IsError, err: err}
	})

	if out.err != nil {
		t.Fatalf("write_file Execute returned a transport error: %v", out.err)
	}
	if !out.isError {
		t.Fatalf("write_file over %s was not refused: %q", os.DevNull, out.content)
	}
	if !strings.Contains(out.content, "not a regular file") {
		t.Errorf("expected a not-a-regular-file refusal, got: %q", out.content)
	}
}

// TestDeleteFileTool_Execute_RefusesDevNull is the delete_file half of the
// same refusal: unlinking the null device would break every process on the
// host that redirects to it, so the tool must refuse.
func TestDeleteFileTool_Execute_RefusesDevNull(t *testing.T) {
	tool := NewDeleteFileTool()
	input, _ := json.Marshal(DeleteFileInput{Path: os.DevNull})

	out := withFifoWatchdog(t, "delete_file "+os.DevNull, func() toolOutcome {
		res, err := tool.Execute(context.Background(), input)
		return toolOutcome{content: res.Content, isError: res.IsError, err: err}
	})

	if out.err != nil {
		t.Fatalf("delete_file Execute returned a transport error: %v", out.err)
	}
	if !out.isError {
		t.Fatalf("delete_file of %s was not refused: %q", os.DevNull, out.content)
	}
	if !strings.Contains(out.content, "not a regular file") {
		t.Errorf("expected a not-a-regular-file refusal, got: %q", out.content)
	}
	if _, err := os.Stat(os.DevNull); err != nil {
		t.Fatalf("%s must survive the refused delete: %v", os.DevNull, err)
	}
}

// TestWriteFileTool_CreateNewAtExemptedPathAllowed proves the refusal is
// scoped to REPLACING an existing non-regular target: a regular file whose
// NAME collides with a Windows reserved-device spelling (a plain file named
// "NUL" on POSIX, where it is an ordinary name) is created/overwritten
// normally. This pins the "creating a new file at an exempted path stays
// allowed" half of finding #56.
func TestWriteFileTool_CreateNewAtExemptedPathAllowed(t *testing.T) {
	path := filepath.Join(t.TempDir(), "NUL") // ordinary file name on POSIX
	tool := NewWriteFileTool()
	input, _ := json.Marshal(WriteFileInput{Path: path, Content: "plain file"})

	res, err := tool.Execute(context.Background(), input)
	if err != nil {
		t.Fatalf("unexpected transport error: %v", err)
	}
	if res.IsError {
		t.Fatalf("creating a new regular file at %q failed: %s", path, res.Content)
	}
}
