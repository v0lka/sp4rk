package safeio

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestReadFile_ReadsRegularFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "a.txt")
	if err := os.WriteFile(path, []byte("hello\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	got, err := ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	if string(got) != "hello\n" {
		t.Fatalf("ReadFile content = %q, want %q", got, "hello\n")
	}
}

func TestReadFile_MissingIsErrNotExist(t *testing.T) {
	_, err := ReadFile(filepath.Join(t.TempDir(), "missing.txt"))
	if !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("ReadFile(missing) err = %v, want errors.Is(err, fs.ErrNotExist)", err)
	}
}

func TestReadFile_RefusesDirectory(t *testing.T) {
	dir := t.TempDir()
	_, err := ReadFile(dir)
	if err == nil {
		t.Fatal("ReadFile(directory) succeeded, want a non-regular refusal")
	}
	if !errors.Is(err, ErrNotRegular) {
		t.Fatalf("ReadFile(directory) err = %v, want errors.Is(err, ErrNotRegular)", err)
	}
	if !strings.Contains(err.Error(), "not a regular file") {
		t.Fatalf("ReadFile(directory) err = %v, want it to mention a non-regular file", err)
	}
}

func TestIsRegular(t *testing.T) {
	dir := t.TempDir()
	regular := filepath.Join(dir, "f.txt")
	if err := os.WriteFile(regular, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}

	if ok, err := IsRegular(regular); err != nil || !ok {
		t.Fatalf("IsRegular(regular file) = (%v, %v), want (true, nil)", ok, err)
	}
	if ok, err := IsRegular(dir); err != nil || ok {
		t.Fatalf("IsRegular(directory) = (%v, %v), want (false, nil)", ok, err)
	}
	if ok, err := IsRegular(filepath.Join(dir, "missing")); err == nil || ok {
		t.Fatalf("IsRegular(missing) = (%v, %v), want (false, non-nil)", ok, err)
	}
}

// TestOpenFile_WritesRegularFile exercises the happy path of OpenFile — the
// write-open replacement for os.OpenFile — so the new API is covered on every
// platform.
func TestOpenFile_WritesRegularFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "out.txt")

	f, err := OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		t.Fatalf("OpenFile: %v", err)
	}
	if _, err := f.WriteString("hello\n"); err != nil {
		_ = f.Close()
		t.Fatalf("WriteString: %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("os.ReadFile: %v", err)
	}
	if string(got) != "hello\n" {
		t.Fatalf("OpenFile wrote %q, want %q", got, "hello\n")
	}
}

// TestReadFileLimited_AtLimitReadsWholeFile pins the boundary: a file of
// exactly limit bytes is read in full, not refused.
func TestReadFileLimited_AtLimitReadsWholeFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "edge.bin")
	content := strings.Repeat("a", 4096)
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	got, err := ReadFileLimited(path, int64(len(content)))
	if err != nil {
		t.Fatalf("ReadFileLimited(at limit): %v", err)
	}
	if string(got) != content {
		t.Fatalf("ReadFileLimited returned %d bytes, want the full %d", len(got), len(content))
	}
}

// TestReadFileLimited_OverLimitRefused pins the explicit too-large error: an
// oversized file is refused instead of being read into memory.
func TestReadFileLimited_OverLimitRefused(t *testing.T) {
	path := filepath.Join(t.TempDir(), "big.bin")
	if err := os.WriteFile(path, []byte(strings.Repeat("a", 4097)), 0o644); err != nil {
		t.Fatal(err)
	}

	_, err := ReadFileLimited(path, 4096)
	if err == nil {
		t.Fatal("ReadFileLimited(over limit) succeeded, want a too-large refusal")
	}
	if !errors.Is(err, ErrTooLarge) {
		t.Fatalf("ReadFileLimited(over limit) err = %v, want errors.Is(err, ErrTooLarge)", err)
	}
	var tooLarge *TooLargeError
	if !errors.As(err, &tooLarge) {
		t.Fatalf("ReadFileLimited(over limit) err = %v, want a *TooLargeError", err)
	}
	if tooLarge.Path != path {
		t.Errorf("TooLargeError.Path = %q, want %q", tooLarge.Path, path)
	}
	if tooLarge.Limit != 4096 {
		t.Errorf("TooLargeError.Limit = %d, want 4096", tooLarge.Limit)
	}
	if !strings.Contains(err.Error(), "too large") {
		t.Errorf("err = %v, want it to mention the file being too large", err)
	}
}

// TestReadFile_DefaultCapRefusesOversizedFile is the OOM regression: a file
// just over DefaultMaxFileSize bytes is refused with the explicit error
// instead of io.ReadAll slurping it into memory.
func TestReadFile_DefaultCapRefusesOversizedFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "huge.bin")
	if err := os.WriteFile(path, make([]byte, DefaultMaxFileSize+1), 0o644); err != nil {
		t.Fatal(err)
	}

	_, err := ReadFile(path)
	if err == nil {
		t.Fatal("ReadFile(just over default cap) succeeded, want a too-large refusal")
	}
	if !errors.Is(err, ErrTooLarge) {
		t.Fatalf("ReadFile(just over default cap) err = %v, want errors.Is(err, ErrTooLarge)", err)
	}
}

// TestReadFile_DefaultCapAllowsFileAtCap pins the other side of the default
// cap: a file of exactly DefaultMaxFileSize bytes still reads in full.
func TestReadFile_DefaultCapAllowsFileAtCap(t *testing.T) {
	path := filepath.Join(t.TempDir(), "at-cap.bin")
	if err := os.WriteFile(path, make([]byte, DefaultMaxFileSize), 0o644); err != nil {
		t.Fatal(err)
	}

	got, err := ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile(at default cap): %v", err)
	}
	if int64(len(got)) != DefaultMaxFileSize {
		t.Fatalf("ReadFile returned %d bytes, want %d", len(got), DefaultMaxFileSize)
	}
}
