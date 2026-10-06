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
