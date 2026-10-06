//go:build unix

package safeio

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"
)

// TestReadFile_RefusesFifoWithoutBlocking is the core regression guard: a FIFO
// at a read path must be refused promptly, never awaited. Before the
// O_NONBLOCK+fstat open, a read-open of a FIFO blocks inside open(2) until a
// writer appears — uninterruptible — and hangs the caller forever. A watchdog
// turns a regression into a test failure instead of a hung CI job.
func TestReadFile_RefusesFifoWithoutBlocking(t *testing.T) {
	fifo := filepath.Join(t.TempDir(), "fifo")
	if err := syscall.Mkfifo(fifo, 0o644); err != nil {
		t.Fatalf("mkfifo: %v", err)
	}

	done := make(chan error, 1)
	go func() {
		_, err := ReadFile(fifo)
		done <- err
	}()
	select {
	case err := <-done:
		if err == nil {
			t.Fatal("ReadFile(FIFO) succeeded, want a non-regular refusal")
		}
		if !errors.Is(err, ErrNotRegular) || !strings.Contains(err.Error(), "not a regular file") {
			t.Fatalf("ReadFile(FIFO) err = %v, want ErrNotRegular", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("ReadFile hung on a FIFO (blocking open)")
	}
}

// TestOpen_RefusesCharacterDevice proves the guard also covers a device node:
// the O_NONBLOCK open succeeds, and only the post-open fstat refuses it.
func TestOpen_RefusesCharacterDevice(t *testing.T) {
	if _, err := os.Stat("/dev/null"); err != nil {
		t.Skip("/dev/null unavailable")
	}
	f, err := Open("/dev/null")
	if err == nil {
		_ = f.Close()
		t.Fatal("Open(/dev/null) succeeded, want a non-regular refusal")
	}
	if !errors.Is(err, ErrNotRegular) {
		t.Fatalf("Open(/dev/null) err = %v, want ErrNotRegular", err)
	}
}

// TestIsRegular_FifoDoesNotBlock mirrors the same guarantee through IsRegular.
func TestIsRegular_FifoDoesNotBlock(t *testing.T) {
	fifo := filepath.Join(t.TempDir(), "fifo")
	if err := syscall.Mkfifo(fifo, 0o644); err != nil {
		t.Fatalf("mkfifo: %v", err)
	}

	done := make(chan bool, 1)
	go func() {
		ok, err := IsRegular(fifo)
		done <- ok && err == nil
	}()
	select {
	case regular := <-done:
		if regular {
			t.Fatal("IsRegular(FIFO) = true, want false")
		}
	case <-time.After(10 * time.Second):
		t.Fatal("IsRegular hung on a FIFO (blocking open)")
	}
}

// TestOpenFile_RefusesFifoWithoutBlocking proves a write-open of a FIFO is
// refused promptly even with no reader: a FIFO open(W) with no reader blocks
// inside open(2) until a reader appears, so before the O_NONBLOCK+fstat guard
// this hung the caller forever. With O_NONBLOCK the open fails fast (ENXIO),
// and the watchdog turns any regression into a failure rather than a hung job.
func TestOpenFile_RefusesFifoWithoutBlocking(t *testing.T) {
	fifo := filepath.Join(t.TempDir(), "fifo")
	if err := syscall.Mkfifo(fifo, 0o644); err != nil {
		t.Fatalf("mkfifo: %v", err)
	}

	done := make(chan error, 1)
	go func() {
		f, err := OpenFile(fifo, os.O_WRONLY, 0o644)
		if err == nil {
			_ = f.Close()
		}
		done <- err
	}()
	select {
	case err := <-done:
		if err == nil {
			t.Fatal("OpenFile(FIFO, O_WRONLY) succeeded, want a refusal")
		}
	case <-time.After(10 * time.Second):
		t.Fatal("OpenFile hung on a FIFO (blocking write-open)")
	}
}

// TestOpenFile_RefusesFifoViaFstat forces the post-open guard: with a reader
// already holding the FIFO, the O_NONBLOCK write-open succeeds, so only the
// fstat on the open descriptor can refuse it — the property a plain Stat→Open
// pair cannot provide against a racing swap.
func TestOpenFile_RefusesFifoViaFstat(t *testing.T) {
	fifo := filepath.Join(t.TempDir(), "fifo")
	if err := syscall.Mkfifo(fifo, 0o644); err != nil {
		t.Fatalf("mkfifo: %v", err)
	}

	// Hold the read end without blocking (a blocking read-open of a FIFO waits
	// for a writer), so the subsequent write-open has a reader and returns.
	reader, err := os.OpenFile(fifo, os.O_RDONLY|syscall.O_NONBLOCK, 0)
	if err != nil {
		t.Fatalf("open reader: %v", err)
	}
	defer func() { _ = reader.Close() }()

	f, err := OpenFile(fifo, os.O_WRONLY, 0o644)
	if err == nil {
		_ = f.Close()
		t.Fatal("OpenFile(FIFO, O_WRONLY) with a reader succeeded, want a non-regular refusal")
	}
	if !errors.Is(err, ErrNotRegular) {
		t.Fatalf("OpenFile(FIFO) err = %v, want errors.Is(err, ErrNotRegular)", err)
	}
	if !strings.Contains(err.Error(), "not a regular file") {
		t.Fatalf("OpenFile(FIFO) err = %v, want it to mention a non-regular file", err)
	}
}
