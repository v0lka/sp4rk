package builtins

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/v0lka/sp4rk/agent"
	"github.com/v0lka/sp4rk/tools"
)

// --- Name tests for individual tools ---

func TestReadFileTool_Name(t *testing.T) {
	tool := NewReadFileTool()
	if tool.Name() != "read_file" {
		t.Errorf("expected name 'read_file', got '%s'", tool.Name())
	}
}

func TestWriteFileTool_Name(t *testing.T) {
	tool := NewWriteFileTool()
	if tool.Name() != "write_file" {
		t.Errorf("expected name 'write_file', got '%s'", tool.Name())
	}
}

func TestEditFileTool_Name(t *testing.T) {
	tool := NewEditFileTool()
	if tool.Name() != "edit_file" {
		t.Errorf("expected name 'edit_file', got '%s'", tool.Name())
	}
}

func TestListDirectoryTool_Name(t *testing.T) {
	tool := NewListDirectoryTool()
	if tool.Name() != "list_directory" {
		t.Errorf("expected name 'list_directory', got '%s'", tool.Name())
	}
}

func TestCreateDirectoryTool_Name(t *testing.T) {
	tool := NewCreateDirectoryTool()
	if tool.Name() != "create_directory" {
		t.Errorf("expected name 'create_directory', got '%s'", tool.Name())
	}
}

func TestDeleteDirectoryTool_Name(t *testing.T) {
	tool := NewDeleteDirectoryTool()
	if tool.Name() != "delete_directory" {
		t.Errorf("expected name 'delete_directory', got '%s'", tool.Name())
	}
}

func TestDeleteFileTool_Name(t *testing.T) {
	tool := NewDeleteFileTool()
	if tool.Name() != "delete_file" {
		t.Errorf("expected name 'delete_file', got '%s'", tool.Name())
	}
}

// --- InputSchema test for a representative tool ---

func TestReadFileTool_InputSchema(t *testing.T) {
	tool := NewReadFileTool()
	schema := tool.InputSchema()
	if len(schema) == 0 {
		t.Error("expected non-empty schema")
	}

	var parsed map[string]any
	if err := json.Unmarshal(schema, &parsed); err != nil {
		t.Errorf("schema is not valid JSON: %v", err)
	}
}

// --- Read tests ---

func TestReadFileTool_ReadFile(t *testing.T) {
	tool := NewReadFileTool()
	ctx := context.Background()
	tmpDir := t.TempDir()

	// Create a test file
	testFile := filepath.Join(tmpDir, "test.txt")
	testContent := "Hello, World!"
	if err := os.WriteFile(testFile, []byte(testContent), 0o644); err != nil {
		t.Fatalf("failed to create test file: %v", err)
	}

	// Read the file
	input, _ := json.Marshal(map[string]string{
		"path": testFile,
	})

	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Errorf("expected success, got error: %s", result.Content)
	}

	// Verify metadata header is present
	if !strings.HasPrefix(result.Content, "[File:") {
		t.Errorf("expected result to start with metadata header '[File:', got: %s", result.Content)
	}
	if !strings.Contains(result.Content, "Lines 1-1 of 1") {
		t.Errorf("expected metadata header to contain 'Lines 1-1 of 1', got: %s", result.Content)
	}

	// Verify original file content is preserved
	if !strings.Contains(result.Content, testContent) {
		t.Errorf("expected content to contain '%s', got: %s", testContent, result.Content)
	}
}

func TestReadFileTool_ReadFile_NonExistent(t *testing.T) {
	tool := NewReadFileTool()
	ctx := context.Background()
	tmpDir := t.TempDir()

	input, _ := json.Marshal(map[string]string{
		"path": filepath.Join(tmpDir, "nonexistent.txt"),
	})

	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Error("expected IsError=true for non-existent file")
	}
	if !strings.Contains(result.Content, "failed to read file") {
		t.Errorf("expected error message about failed read, got: %s", result.Content)
	}
}

// --- Write tests ---

func TestWriteFileTool_WriteFile(t *testing.T) {
	tool := NewWriteFileTool()
	ctx := context.Background()
	tmpDir := t.TempDir()

	testFile := filepath.Join(tmpDir, "output.txt")
	testContent := "Test content to write"

	input, _ := json.Marshal(map[string]string{
		"path":    testFile,
		"content": testContent,
	})

	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Errorf("expected success, got error: %s", result.Content)
	}

	// Verify the file was written
	data, err := os.ReadFile(testFile)
	if err != nil {
		t.Fatalf("failed to read written file: %v", err)
	}
	if string(data) != testContent {
		t.Errorf("expected content '%s', got '%s'", testContent, string(data))
	}
}

func TestWriteFileTool_WriteFile_NestedPath(t *testing.T) {
	tool := NewWriteFileTool()
	ctx := context.Background()
	tmpDir := t.TempDir()

	// Nested path that doesn't exist yet
	testFile := filepath.Join(tmpDir, "deep", "nested", "dir", "file.txt")
	testContent := "Nested content"

	input, _ := json.Marshal(map[string]string{
		"path":    testFile,
		"content": testContent,
	})

	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Errorf("expected success, got error: %s", result.Content)
	}

	// Verify the file was written
	data, err := os.ReadFile(testFile)
	if err != nil {
		t.Fatalf("failed to read written file: %v", err)
	}
	if string(data) != testContent {
		t.Errorf("expected content '%s', got '%s'", testContent, string(data))
	}
}

// TestWriteFileTool_PreservesExistingPermissions is a regression test for the
// atomic write fix. Previously write_file used os.WriteFile with a hardcoded
// 0o644 mode, which silently reset an existing executable (0o755) or otherwise
// restricted file to 0o644 on overwrite. The atomic temp-file+rename helper
// (shared with edit_file) preserves the existing permission bits. It also must
// not leave temp files behind on success.
func TestWriteFileTool_PreservesExistingPermissions(t *testing.T) {
	tool := NewWriteFileTool()
	ctx := context.Background()
	tmpDir := t.TempDir()

	testFile := filepath.Join(tmpDir, "script.sh")
	// Pre-create the file as executable so the regression (0o644 reset) is
	// detectable.
	if err := os.WriteFile(testFile, []byte("old"), 0o755); err != nil {
		t.Fatalf("setup write failed: %v", err)
	}

	input, _ := json.Marshal(map[string]string{"path": testFile, "content": "new"})
	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Fatalf("expected success, got error: %s", result.Content)
	}

	info, err := os.Stat(testFile)
	if err != nil {
		t.Fatalf("failed to stat written file: %v", err)
	}
	// Unix permission bits are not meaningful on Windows — os.Chmod only
	// affects the read-only attribute and os.Stat().Mode().Perm() returns
	// synthetic values (typically 0o666) that don't reflect the original
	// 0o755. Skip the permission-preservation check on Windows.
	if runtime.GOOS != "windows" {
		if perm := info.Mode().Perm(); perm != 0o755 {
			t.Errorf("expected existing perms 0o755 to be preserved, got %v", perm)
		}
	}

	// A successful atomic write must not leave temp files behind.
	entries, err := os.ReadDir(tmpDir)
	if err != nil {
		t.Fatalf("failed to read dir: %v", err)
	}
	if len(entries) != 1 {
		names := make([]string, 0, len(entries))
		for _, e := range entries {
			names = append(names, e.Name())
		}
		t.Errorf("expected exactly 1 file (no temp leftovers), got %d: %v", len(entries), names)
	}
}

// --- Edit tests ---

func TestEditFileTool_EditFile_UniqueMatch(t *testing.T) {
	tool := NewEditFileTool()
	ctx := context.Background()
	tmpDir := t.TempDir()

	testFile := filepath.Join(tmpDir, "edit.txt")
	originalContent := "Hello, World! This is a test."
	if err := os.WriteFile(testFile, []byte(originalContent), 0o644); err != nil {
		t.Fatalf("failed to create test file: %v", err)
	}

	input, _ := json.Marshal(map[string]string{
		"path":       testFile,
		"old_string": "World",
		"new_string": "Universe",
	})

	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Errorf("expected success, got error: %s", result.Content)
	}

	// Verify the replacement
	data, err := os.ReadFile(testFile)
	if err != nil {
		t.Fatalf("failed to read edited file: %v", err)
	}
	expected := "Hello, Universe! This is a test."
	if string(data) != expected {
		t.Errorf("expected content '%s', got '%s'", expected, string(data))
	}
}

func TestEditFileTool_EditFile_NonUniqueMatch(t *testing.T) {
	tool := NewEditFileTool()
	ctx := context.Background()
	tmpDir := t.TempDir()

	testFile := filepath.Join(tmpDir, "edit.txt")
	originalContent := "This is a test. Another test here."
	if err := os.WriteFile(testFile, []byte(originalContent), 0o644); err != nil {
		t.Fatalf("failed to create test file: %v", err)
	}

	input, _ := json.Marshal(map[string]string{
		"path":       testFile,
		"old_string": "test",
		"new_string": "example",
	})

	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Error("expected IsError=true for non-unique match")
	}
	if !strings.Contains(result.Content, "not unique") {
		t.Errorf("expected 'not unique' error message, got: %s", result.Content)
	}
	if !strings.Contains(result.Content, "2 occurrences") {
		t.Errorf("expected '2 occurrences' in error message, got: %s", result.Content)
	}
}

func TestEditFileTool_EditFile_NotFound(t *testing.T) {
	tool := NewEditFileTool()
	ctx := context.Background()
	tmpDir := t.TempDir()

	testFile := filepath.Join(tmpDir, "edit.txt")
	originalContent := "Hello, World!"
	if err := os.WriteFile(testFile, []byte(originalContent), 0o644); err != nil {
		t.Fatalf("failed to create test file: %v", err)
	}

	input, _ := json.Marshal(map[string]string{
		"path":       testFile,
		"old_string": "nonexistent",
		"new_string": "replacement",
	})

	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Error("expected IsError=true for string not found")
	}
	if !strings.Contains(result.Content, "not found") {
		t.Errorf("expected 'not found' error message, got: %s", result.Content)
	}
}

// --- List test ---

func TestListDirectoryTool_ListDirectory(t *testing.T) {
	tool := NewListDirectoryTool()
	ctx := context.Background()
	tmpDir := t.TempDir()

	if err := os.WriteFile(filepath.Join(tmpDir, "file1.txt"), []byte("content1"), 0o644); err != nil {
		t.Fatalf("failed to create file1: %v", err)
	}
	if err := os.WriteFile(filepath.Join(tmpDir, "file2.txt"), []byte("content2"), 0o644); err != nil {
		t.Fatalf("failed to create file2: %v", err)
	}
	if err := os.Mkdir(filepath.Join(tmpDir, "subdir"), 0o755); err != nil {
		t.Fatalf("failed to create subdir: %v", err)
	}

	input, _ := json.Marshal(map[string]string{
		"path": tmpDir,
	})

	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Errorf("expected success, got error: %s", result.Content)
	}

	if !strings.Contains(result.Content, "file1.txt") {
		t.Errorf("expected 'file1.txt' in listing, got: %s", result.Content)
	}
	if !strings.Contains(result.Content, "file2.txt") {
		t.Errorf("expected 'file2.txt' in listing, got: %s", result.Content)
	}
	if !strings.Contains(result.Content, "subdir") {
		t.Errorf("expected 'subdir' in listing, got: %s", result.Content)
	}
	if !strings.Contains(result.Content, "dir") {
		t.Errorf("expected 'dir' type in listing, got: %s", result.Content)
	}
	if !strings.Contains(result.Content, "file") {
		t.Errorf("expected 'file' type in listing, got: %s", result.Content)
	}
}

// --- DefaultPolicy test ---

func TestReadFileTool_DefaultPolicy(t *testing.T) {
	tool := NewReadFileTool()
	if tool.DefaultPolicy() != tools.PolicyAlwaysAllow {
		t.Errorf("expected DefaultPolicy() to return PolicyAlwaysAllow, got %v", tool.DefaultPolicy())
	}
}

// --- Judge tests ---

func TestReadFileTool_Judge_ReadOnlyAction(t *testing.T) {
	tool := NewReadFileTool()

	t.Run("outside workspace requires confirmation", func(t *testing.T) {
		input, _ := json.Marshal(map[string]string{
			"path": "/some/path.txt",
		})

		judgeOut := tool.Judge(context.Background(), input)
		allow, reasoning := judgeOut.Allow, judgeOut.Reason
		if allow {
			t.Error("expected Judge to return allow=false for path outside workspace")
		}
		if reasoning == "" {
			t.Error("expected non-empty reasoning for path outside workspace")
		}
	})

	t.Run("inside workspace auto-allows", func(t *testing.T) {
		tmpDir := t.TempDir()
		ctx := tools.WithWorkspacePath(context.Background(), tmpDir)

		testFile := filepath.Join(tmpDir, "test.txt")
		input, _ := json.Marshal(map[string]string{
			"path": testFile,
		})

		judgeOut := tool.Judge(ctx, input)
		allow, reasoning := judgeOut.Allow, judgeOut.Reason
		if !allow {
			t.Errorf("expected Judge to return allow=true for path inside workspace, got reasoning: %s", reasoning)
		}
	})

	t.Run("inside temp dir auto-allows", func(t *testing.T) {
		tmpDir := t.TempDir()
		ctx := tools.WithTempDir(context.Background(), tmpDir)

		testFile := filepath.Join(tmpDir, "output.txt")
		input, _ := json.Marshal(map[string]string{
			"path": testFile,
		})

		judgeOut := tool.Judge(ctx, input)
		allow, reasoning := judgeOut.Allow, judgeOut.Reason
		if !allow {
			t.Errorf("expected Judge to return allow=true for path inside temp dir, got reasoning: %s", reasoning)
		}
	})
}

func TestWriteFileTool_Judge_WriteActionInsideWorkspace(t *testing.T) {
	tool := NewWriteFileTool()

	tmpDir := t.TempDir()
	ctx := tools.WithWorkspacePath(context.Background(), tmpDir)

	testFile := filepath.Join(tmpDir, "test.txt")
	input, _ := json.Marshal(map[string]string{
		"path": testFile,
	})

	judgeOut := tool.Judge(ctx, input)
	allow, reasoning := judgeOut.Allow, judgeOut.Reason
	if !allow {
		t.Error("expected Judge to return allow=true for write action inside workspace")
	}
	if !strings.Contains(reasoning, "workspace") {
		t.Errorf("expected reasoning to mention 'workspace', got: %s", reasoning)
	}
}

func TestWriteFileTool_Judge_WriteActionOutsideWorkspace(t *testing.T) {
	tool := NewWriteFileTool()

	tmpDir := t.TempDir()
	otherDir := t.TempDir()
	ctx := tools.WithWorkspacePath(context.Background(), tmpDir)

	testFile := filepath.Join(otherDir, "test.txt")
	input, _ := json.Marshal(map[string]string{
		"path": testFile,
	})

	judgeOut := tool.Judge(ctx, input)
	allow, reasoning := judgeOut.Allow, judgeOut.Reason
	if allow {
		t.Error("expected Judge to return allow=false for write action outside workspace")
	}
	if !strings.Contains(reasoning, "outside") {
		t.Errorf("expected reasoning to mention 'outside', got: %s", reasoning)
	}
}

func TestWriteFileTool_Judge_WriteActionNoWorkspace(t *testing.T) {
	tool := NewWriteFileTool()

	ctx := context.Background()

	input, _ := json.Marshal(map[string]string{
		"path": "/some/path.txt",
	})

	judgeOut := tool.Judge(ctx, input)
	allow, reasoning := judgeOut.Allow, judgeOut.Reason
	if allow {
		t.Error("expected Judge to return allow=false when no workspace in context")
	}
	if !strings.Contains(reasoning, "outside") {
		t.Errorf("expected reasoning to mention 'outside' when no workspace, got: %s", reasoning)
	}
}

func TestWriteFileTool_Judge_InvalidJSON(t *testing.T) {
	tool := NewWriteFileTool()

	judgeOut := tool.Judge(context.Background(), json.RawMessage(`{invalid`))
	allow, reasoning := judgeOut.Allow, judgeOut.Reason
	if allow {
		t.Error("expected Judge to return allow=false for invalid JSON")
	}
	if reasoning != "" {
		t.Errorf("expected empty reasoning for invalid JSON, got: %s", reasoning)
	}
}

func TestWriteFileTool_InvalidJSON(t *testing.T) {
	tool := NewWriteFileTool()
	ctx := context.Background()

	result, err := tool.Execute(ctx, json.RawMessage(`{invalid json}`))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Error("expected IsError=true for invalid JSON")
	}
	if !strings.Contains(result.Content, "failed to parse input") {
		t.Errorf("expected 'failed to parse input' error message, got: %s", result.Content)
	}
}

// --- Directory management tests ---

func TestCreateDirectoryTool_CreateDirectory(t *testing.T) {
	tool := NewCreateDirectoryTool()
	ctx := context.Background()
	tmpDir := t.TempDir()

	newDir := filepath.Join(tmpDir, "newdir")
	input, _ := json.Marshal(map[string]string{
		"path": newDir,
	})

	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Errorf("expected success, got error: %s", result.Content)
	}
	if !strings.Contains(result.Content, "successfully created directory") {
		t.Errorf("expected success message, got: %s", result.Content)
	}

	info, err := os.Stat(newDir)
	if err != nil {
		t.Fatalf("directory was not created: %v", err)
	}
	if !info.IsDir() {
		t.Error("expected path to be a directory")
	}
}

func TestCreateDirectoryTool_CreateDirectory_Nested(t *testing.T) {
	tool := NewCreateDirectoryTool()
	ctx := context.Background()
	tmpDir := t.TempDir()

	nestedDir := filepath.Join(tmpDir, "a", "b", "c", "d")
	input, _ := json.Marshal(map[string]string{
		"path": nestedDir,
	})

	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Errorf("expected success, got error: %s", result.Content)
	}

	// Verify all intermediate directories exist
	for _, dir := range []string{
		filepath.Join(tmpDir, "a"),
		filepath.Join(tmpDir, "a", "b"),
		filepath.Join(tmpDir, "a", "b", "c"),
		nestedDir,
	} {
		info, err := os.Stat(dir)
		if err != nil {
			t.Fatalf("intermediate directory %s was not created: %v", dir, err)
		}
		if !info.IsDir() {
			t.Errorf("expected %s to be a directory", dir)
		}
	}
}

func TestDeleteDirectoryTool_DeleteDirectory_Empty(t *testing.T) {
	tool := NewDeleteDirectoryTool()
	ctx := context.Background()
	tmpDir := t.TempDir()

	emptyDir := filepath.Join(tmpDir, "empty")
	if err := os.Mkdir(emptyDir, 0o755); err != nil {
		t.Fatalf("failed to create empty dir: %v", err)
	}

	input, _ := json.Marshal(map[string]any{
		"path":      emptyDir,
		"recursive": false,
	})

	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Errorf("expected success, got error: %s", result.Content)
	}
	if !strings.Contains(result.Content, "successfully deleted directory") {
		t.Errorf("expected success message, got: %s", result.Content)
	}

	if _, err := os.Stat(emptyDir); !os.IsNotExist(err) {
		t.Error("expected directory to be deleted")
	}
}

func TestDeleteDirectoryTool_DeleteDirectory_NonEmpty_NonRecursive(t *testing.T) {
	tool := NewDeleteDirectoryTool()
	ctx := context.Background()
	tmpDir := t.TempDir()

	nonEmptyDir := filepath.Join(tmpDir, "notempty")
	if err := os.Mkdir(nonEmptyDir, 0o755); err != nil {
		t.Fatalf("failed to create dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(nonEmptyDir, "file.txt"), []byte("data"), 0o644); err != nil {
		t.Fatalf("failed to create file: %v", err)
	}

	input, _ := json.Marshal(map[string]any{
		"path":      nonEmptyDir,
		"recursive": false,
	})

	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Error("expected IsError=true for non-empty directory with recursive=false")
	}
	if !strings.Contains(result.Content, "failed to delete directory") {
		t.Errorf("expected error message about failed delete, got: %s", result.Content)
	}
}

func TestDeleteDirectoryTool_DeleteDirectory_Recursive(t *testing.T) {
	tool := NewDeleteDirectoryTool()
	ctx := context.Background()
	tmpDir := t.TempDir()

	dirToDelete := filepath.Join(tmpDir, "todelete")
	subDir := filepath.Join(dirToDelete, "sub")
	if err := os.MkdirAll(subDir, 0o755); err != nil {
		t.Fatalf("failed to create dirs: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dirToDelete, "file1.txt"), []byte("data1"), 0o644); err != nil {
		t.Fatalf("failed to create file: %v", err)
	}
	if err := os.WriteFile(filepath.Join(subDir, "file2.txt"), []byte("data2"), 0o644); err != nil {
		t.Fatalf("failed to create file: %v", err)
	}

	input, _ := json.Marshal(map[string]any{
		"path":      dirToDelete,
		"recursive": true,
	})

	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Errorf("expected success, got error: %s", result.Content)
	}
	if !strings.Contains(result.Content, "successfully deleted directory") {
		t.Errorf("expected success message, got: %s", result.Content)
	}

	if _, err := os.Stat(dirToDelete); !os.IsNotExist(err) {
		t.Error("expected directory to be deleted")
	}
}

func TestDeleteDirectoryTool_DeleteDirectory_NonExistent(t *testing.T) {
	tool := NewDeleteDirectoryTool()
	ctx := context.Background()
	tmpDir := t.TempDir()

	input, _ := json.Marshal(map[string]any{
		"path":      filepath.Join(tmpDir, "nonexistent"),
		"recursive": false,
	})

	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Error("expected IsError=true for non-existent directory")
	}
	if !strings.Contains(result.Content, "failed to stat path") {
		t.Errorf("expected stat error message, got: %s", result.Content)
	}
}

// --- File delete tests ---

func TestDeleteFileTool_DeleteFile(t *testing.T) {
	tool := NewDeleteFileTool()
	ctx := context.Background()
	tmpDir := t.TempDir()

	testFile := filepath.Join(tmpDir, "todelete.txt")
	if err := os.WriteFile(testFile, []byte("delete me"), 0o644); err != nil {
		t.Fatalf("failed to create test file: %v", err)
	}

	input, _ := json.Marshal(map[string]string{
		"path": testFile,
	})

	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Errorf("expected success, got error: %s", result.Content)
	}
	if !strings.Contains(result.Content, "successfully deleted file") {
		t.Errorf("expected success message, got: %s", result.Content)
	}

	if _, err := os.Stat(testFile); !os.IsNotExist(err) {
		t.Error("expected file to be deleted")
	}
}

func TestDeleteFileTool_DeleteFile_NonExistent(t *testing.T) {
	tool := NewDeleteFileTool()
	ctx := context.Background()
	tmpDir := t.TempDir()

	input, _ := json.Marshal(map[string]string{
		"path": filepath.Join(tmpDir, "nonexistent.txt"),
	})

	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Error("expected IsError=true for non-existent file")
	}
	if !strings.Contains(result.Content, "failed to stat path") {
		t.Errorf("expected stat error message, got: %s", result.Content)
	}
}

func TestDeleteFileTool_DeleteFile_IsDirectory(t *testing.T) {
	tool := NewDeleteFileTool()
	ctx := context.Background()
	tmpDir := t.TempDir()

	dirPath := filepath.Join(tmpDir, "adir")
	if err := os.Mkdir(dirPath, 0o755); err != nil {
		t.Fatalf("failed to create dir: %v", err)
	}

	input, _ := json.Marshal(map[string]string{
		"path": dirPath,
	})

	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Error("expected IsError=true when trying to delete a directory with delete_file")
	}
	if !strings.Contains(result.Content, "path is a directory, use delete_directory instead") {
		t.Errorf("expected directory error message, got: %s", result.Content)
	}
}

// --- More Judge tests ---

func TestCreateDirectoryTool_Judge_CreateDirectoryInsideWorkspace(t *testing.T) {
	tool := NewCreateDirectoryTool()

	tmpDir := t.TempDir()
	ctx := tools.WithWorkspacePath(context.Background(), tmpDir)

	newDir := filepath.Join(tmpDir, "newdir")
	input, _ := json.Marshal(map[string]string{
		"path": newDir,
	})

	judgeOut := tool.Judge(ctx, input)
	allow, reasoning := judgeOut.Allow, judgeOut.Reason
	if !allow {
		t.Error("expected Judge to return allow=true for create_directory inside workspace")
	}
	if !strings.Contains(reasoning, "workspace") {
		t.Errorf("expected reasoning to mention 'workspace', got: %s", reasoning)
	}
}

func TestDeleteDirectoryTool_Judge_DeleteDirectoryOutsideWorkspace(t *testing.T) {
	tool := NewDeleteDirectoryTool()

	tmpDir := t.TempDir()
	otherDir := t.TempDir()
	ctx := tools.WithWorkspacePath(context.Background(), tmpDir)

	input, _ := json.Marshal(map[string]any{
		"path":      filepath.Join(otherDir, "somedir"),
		"recursive": true,
	})

	judgeOut := tool.Judge(ctx, input)
	allow, reasoning := judgeOut.Allow, judgeOut.Reason
	if allow {
		t.Error("expected Judge to return allow=false for delete_directory outside workspace")
	}
	if !strings.Contains(reasoning, "outside") {
		t.Errorf("expected reasoning to mention 'outside', got: %s", reasoning)
	}
}

func TestDeleteFileTool_Judge_DeleteFileInsideWorkspace(t *testing.T) {
	tool := NewDeleteFileTool()

	tmpDir := t.TempDir()
	ctx := tools.WithWorkspacePath(context.Background(), tmpDir)

	testFile := filepath.Join(tmpDir, "file.txt")
	input, _ := json.Marshal(map[string]string{
		"path": testFile,
	})

	judgeOut := tool.Judge(ctx, input)
	allow, reasoning := judgeOut.Allow, judgeOut.Reason
	if !allow {
		t.Error("expected Judge to return allow=true for delete_file inside workspace")
	}
	if !strings.Contains(reasoning, "workspace") {
		t.Errorf("expected reasoning to mention 'workspace', got: %s", reasoning)
	}
}

// --- Pagination tests ---

func TestReadFileTool_ReadFile_DefaultPagination(t *testing.T) {
	tool := NewReadFileTool()
	ctx := context.Background()
	tmpDir := t.TempDir()

	// Create a temp file with 3000 lines
	testFile := filepath.Join(tmpDir, "large.txt")
	lines := make([]string, 0, 3000)
	for i := 1; i <= 3000; i++ {
		lines = append(lines, fmt.Sprintf("Line %d", i))
	}
	if err := os.WriteFile(testFile, []byte(strings.Join(lines, "\n")), 0o644); err != nil {
		t.Fatalf("failed to create test file: %v", err)
	}

	// Read with no start_line/end_line — returns default window (2000 lines)
	input, _ := json.Marshal(map[string]string{
		"path": testFile,
	})

	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Errorf("expected success, got error: %s", result.Content)
	}

	// Verify first window is returned (lines 1-2000)
	if !strings.Contains(result.Content, "Line 1\n") {
		t.Errorf("expected 'Line 1' in content")
	}
	if !strings.Contains(result.Content, "Line 2000") {
		t.Errorf("expected 'Line 2000' in content (default window)")
	}
	if strings.Contains(result.Content, "Line 3000") {
		t.Errorf("did not expect 'Line 3000' in default window (should be truncated)")
	}

	// Verify metadata header shows window range
	if !strings.Contains(result.Content, "[File: large.txt | Lines 1-2000 of 3000") {
		t.Errorf("expected metadata header with window range, got: %s", result.Content)
	}

	// Verify continuation hint
	if !strings.Contains(result.Content, "[Use start_line=2001 to continue reading]") {
		t.Errorf("expected continuation hint, got: %s", result.Content)
	}
}

func TestReadFileTool_ReadFile_StreamingWindow(t *testing.T) {
	tool := NewReadFileTool()
	ctx := context.Background()
	tmpDir := t.TempDir()

	// Create a temp file with 10000 lines
	testFile := filepath.Join(tmpDir, "huge.txt")
	var sb strings.Builder
	for i := 1; i <= 10000; i++ {
		if i > 1 {
			sb.WriteByte('\n')
		}
		fmt.Fprintf(&sb, "Line %d", i)
	}
	if err := os.WriteFile(testFile, []byte(sb.String()), 0o644); err != nil {
		t.Fatalf("failed to create test file: %v", err)
	}

	// Read with no range — should return default window
	input, _ := json.Marshal(map[string]string{"path": testFile})
	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Errorf("expected success, got error: %s", result.Content)
	}

	// Verify default window = ReadDefaultLines (2000)
	if !strings.Contains(result.Content, "Lines 1-2000 of 10000") {
		t.Errorf("expected 'Lines 1-2000 of 10000' in header, got: %s", result.Content[:min(80, len(result.Content))])
	}
	if !strings.Contains(result.Content, "[Use start_line=2001 to continue reading]") {
		t.Errorf("expected continuation hint for 10000-line file")
	}

	// Read next window
	input2, _ := json.Marshal(map[string]any{"path": testFile, "start_line": 2001})
	result2, err := tool.Execute(ctx, input2)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(result2.Content, "Lines 2001-4000 of 10000") {
		t.Errorf("expected 'Lines 2001-4000 of 10000' in header, got: %s", result2.Content[:min(80, len(result2.Content))])
	}
}

func TestReadFileTool_ReadFile_ExplicitRange(t *testing.T) {
	tool := NewReadFileTool()
	ctx := context.Background()
	tmpDir := t.TempDir()

	// Create a temp file with 100 lines
	testFile := filepath.Join(tmpDir, "medium.txt")
	lines := make([]string, 0, 100)
	for i := 1; i <= 100; i++ {
		lines = append(lines, fmt.Sprintf("Line %d", i))
	}
	if err := os.WriteFile(testFile, []byte(strings.Join(lines, "\n")), 0o644); err != nil {
		t.Fatalf("failed to create test file: %v", err)
	}

	// Read with explicit range: start_line=10, end_line=20
	input, _ := json.Marshal(map[string]any{
		"path":       testFile,
		"start_line": 10,
		"end_line":   20,
	})

	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Errorf("expected success, got error: %s", result.Content)
	}

	// Verify exactly lines 10-20 returned
	if strings.Contains(result.Content, "Line 9") {
		t.Errorf("did not expect 'Line 9' in content (should start at 10)")
	}
	if !strings.Contains(result.Content, "Line 10") {
		t.Errorf("expected 'Line 10' in content")
	}
	if !strings.Contains(result.Content, "Line 20") {
		t.Errorf("expected 'Line 20' in content")
	}
	if strings.Contains(result.Content, "Line 21") {
		t.Errorf("did not expect 'Line 21' in content (should end at 20)")
	}

	// Verify metadata header shows correct range
	if !strings.Contains(result.Content, "[File: medium.txt | Lines 10-20 of 100") {
		t.Errorf("expected metadata header with lines 10-20, got: %s", result.Content)
	}
}

func TestReadFileTool_ReadFile_LongLinesTruncated(t *testing.T) {
	tool := NewReadFileTool()
	ctx := context.Background()
	tmpDir := t.TempDir()

	// Create a temp file with one 5000-char line
	testFile := filepath.Join(tmpDir, "longline.txt")
	longLine := strings.Repeat("a", 5000)
	if err := os.WriteFile(testFile, []byte(longLine), 0o644); err != nil {
		t.Fatalf("failed to create test file: %v", err)
	}

	input, _ := json.Marshal(map[string]string{
		"path": testFile,
	})

	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Errorf("expected success, got error: %s", result.Content)
	}

	// Verify the full line is returned (no per-tool truncation; central layer handles it)
	if !strings.Contains(result.Content, longLine) {
		t.Errorf("expected full 5000-char line without truncation, got: ...%s", result.Content[len(result.Content)-100:])
	}

	// Verify no truncation notice
	if strings.Contains(result.Content, "...(line truncated") {
		t.Errorf("did not expect line truncation notice")
	}
}

func TestReadFileTool_ReadFile_SmallFile(t *testing.T) {
	tool := NewReadFileTool()
	ctx := context.Background()
	tmpDir := t.TempDir()

	// Create a temp file with 50 lines
	testFile := filepath.Join(tmpDir, "small.txt")
	lines := make([]string, 0, 50)
	for i := 1; i <= 50; i++ {
		lines = append(lines, fmt.Sprintf("Line %d", i))
	}
	if err := os.WriteFile(testFile, []byte(strings.Join(lines, "\n")), 0o644); err != nil {
		t.Fatalf("failed to create test file: %v", err)
	}

	input, _ := json.Marshal(map[string]string{
		"path": testFile,
	})

	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Errorf("expected success, got error: %s", result.Content)
	}

	// Verify all lines returned
	if !strings.Contains(result.Content, "Line 1") {
		t.Errorf("expected 'Line 1' in content")
	}
	if !strings.Contains(result.Content, "Line 50") {
		t.Errorf("expected 'Line 50' in content")
	}

	// Verify NO continuation hint (all lines fit)
	if strings.Contains(result.Content, "[Use start_line=") {
		t.Errorf("did not expect continuation hint for small file, got: %s", result.Content)
	}

	// Verify NO truncation notice
	if strings.Contains(result.Content, "truncated") {
		t.Errorf("did not expect truncation notice, got: %s", result.Content)
	}

	// Verify metadata header shows all lines
	if !strings.Contains(result.Content, "[File: small.txt | Lines 1-50 of 50") {
		t.Errorf("expected metadata header showing all lines, got: %s", result.Content)
	}
}

func TestReadFileTool_ReadFile_OutOfRangeClamp(t *testing.T) {
	tool := NewReadFileTool()
	ctx := context.Background()
	tmpDir := t.TempDir()

	// Create a temp file with 100 lines
	testFile := filepath.Join(tmpDir, "clamp.txt")
	lines := make([]string, 0, 100)
	for i := 1; i <= 100; i++ {
		lines = append(lines, fmt.Sprintf("Line %d", i))
	}
	if err := os.WriteFile(testFile, []byte(strings.Join(lines, "\n")), 0o644); err != nil {
		t.Fatalf("failed to create test file: %v", err)
	}

	// Read with end_line=500 (beyond file size)
	input, _ := json.Marshal(map[string]any{
		"path":       testFile,
		"start_line": 1,
		"end_line":   500,
	})

	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Errorf("expected success, got error: %s", result.Content)
	}

	// Verify all lines returned (clamped to 100)
	if !strings.Contains(result.Content, "Line 1") {
		t.Errorf("expected 'Line 1' in content")
	}
	if !strings.Contains(result.Content, "Line 100") {
		t.Errorf("expected 'Line 100' in content")
	}

	// Verify metadata header shows clamped range
	if !strings.Contains(result.Content, "[File: clamp.txt | Lines 1-100 of 100") {
		t.Errorf("expected metadata header with clamped line range, got: %s", result.Content)
	}
}

// --- Relative path resolution with workspace context ---

func TestFileTools_RelativePathWithWorkspace(t *testing.T) {
	workspace := t.TempDir()
	ctx := tools.WithWorkspacePath(context.Background(), workspace)

	t.Run("read_file", func(t *testing.T) {
		testContent := "hello workspace"
		if err := os.WriteFile(filepath.Join(workspace, "test.txt"), []byte(testContent), 0o644); err != nil {
			t.Fatal(err)
		}
		tool := NewReadFileTool()
		input, _ := json.Marshal(ReadFileInput{Path: "test.txt"})
		result, err := tool.Execute(ctx, input)
		if err != nil {
			t.Fatal(err)
		}
		if result.IsError {
			t.Fatalf("unexpected error: %s", result.Content)
		}
		if !strings.Contains(result.Content, testContent) {
			t.Errorf("expected content %q in result, got: %s", testContent, result.Content)
		}
	})

	t.Run("write_file", func(t *testing.T) {
		tool := NewWriteFileTool()
		input, _ := json.Marshal(WriteFileInput{Path: "written.txt", Content: "workspace write"})
		result, err := tool.Execute(ctx, input)
		if err != nil {
			t.Fatal(err)
		}
		if result.IsError {
			t.Fatalf("unexpected error: %s", result.Content)
		}
		data, err := os.ReadFile(filepath.Join(workspace, "written.txt"))
		if err != nil {
			t.Fatalf("file not created at workspace path: %v", err)
		}
		if string(data) != "workspace write" {
			t.Errorf("expected 'workspace write', got %q", string(data))
		}
	})

	t.Run("edit_file", func(t *testing.T) {
		if err := os.WriteFile(filepath.Join(workspace, "edit.txt"), []byte("old content here"), 0o644); err != nil {
			t.Fatal(err)
		}
		tool := NewEditFileTool()
		input, _ := json.Marshal(EditFileInput{Path: "edit.txt", OldString: "old", NewString: "new"})
		result, err := tool.Execute(ctx, input)
		if err != nil {
			t.Fatal(err)
		}
		if result.IsError {
			t.Fatalf("unexpected error: %s", result.Content)
		}
		data, err := os.ReadFile(filepath.Join(workspace, "edit.txt"))
		if err != nil {
			t.Fatal(err)
		}
		if string(data) != "new content here" {
			t.Errorf("expected 'new content here', got %q", string(data))
		}
	})

	t.Run("list_directory", func(t *testing.T) {
		subDir := filepath.Join(workspace, "mysubdir")
		if err := os.Mkdir(subDir, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(subDir, "a.txt"), []byte("a"), 0o644); err != nil {
			t.Fatal(err)
		}
		tool := NewListDirectoryTool()
		input, _ := json.Marshal(ListDirectoryInput{Path: "mysubdir"})
		result, err := tool.Execute(ctx, input)
		if err != nil {
			t.Fatal(err)
		}
		if result.IsError {
			t.Fatalf("unexpected error: %s", result.Content)
		}
		if !strings.Contains(result.Content, "a.txt") {
			t.Errorf("expected 'a.txt' in listing, got: %s", result.Content)
		}
	})

	t.Run("delete_file", func(t *testing.T) {
		delFile := filepath.Join(workspace, "todelete.txt")
		if err := os.WriteFile(delFile, []byte("bye"), 0o644); err != nil {
			t.Fatal(err)
		}
		tool := NewDeleteFileTool()
		input, _ := json.Marshal(DeleteFileInput{Path: "todelete.txt"})
		result, err := tool.Execute(ctx, input)
		if err != nil {
			t.Fatal(err)
		}
		if result.IsError {
			t.Fatalf("unexpected error: %s", result.Content)
		}
		if _, err := os.Stat(delFile); !os.IsNotExist(err) {
			t.Error("expected file to be deleted")
		}
	})
}

// ── Batch tests ────────────────────────────────────────────────────────────

func TestNewBatchTool(t *testing.T) {
	tool := NewBatchTool()
	if tool.Name() != "batch" {
		t.Errorf("expected name 'batch', got %q", tool.Name())
	}
	if tool.DefaultPolicy() != tools.PolicyAlwaysAllow {
		t.Errorf("expected PolicyAlwaysAllow, got %v", tool.DefaultPolicy())
	}
}

func TestBatchTool_Execute(t *testing.T) {
	tool := NewBatchTool()
	_, err := tool.Execute(context.Background(), nil)
	if err == nil {
		t.Fatal("expected error from batch Execute")
	}
	if !strings.Contains(err.Error(), "batch is handled at the executor level") {
		t.Errorf("expected batch-level error, got: %v", err)
	}
}

// ── Coherence format tests ─────────────────────────────────────────────────

func TestFormatWriteConflict(t *testing.T) {
	now := time.Now()
	c := &tools.CoherenceConflict{
		Path: "/ws/file.go",
		LastReadSig: tools.FileSig{
			ModTime: now.Add(-1 * time.Hour),
			Size:    1024,
		},
		CurrentSig: tools.FileSig{
			ModTime: now,
			Size:    2048,
		},
		ModifiedBy: "other-session",
		ModifiedAt: now,
	}
	msg := formatWriteConflict(c)
	if !strings.Contains(msg, "/ws/file.go") {
		t.Errorf("expected path in message, got: %s", msg)
	}
	if !strings.Contains(msg, "other-session") {
		t.Errorf("expected modified-by in message, got: %s", msg)
	}
	if !strings.Contains(msg, "re-read the file") {
		t.Errorf("expected action required hint, got: %s", msg)
	}
}

func TestFormatReadConflict(t *testing.T) {
	c := &tools.CoherenceConflict{
		Path:       "/ws/file.go",
		ModifiedBy: "external",
	}
	msg := formatReadConflict(c)
	if !strings.Contains(msg, "external") {
		t.Errorf("expected modified-by in message, got: %s", msg)
	}
	if !strings.Contains(msg, "modified") {
		t.Errorf("expected 'modified' in message, got: %s", msg)
	}
}

// ── Limits tests ───────────────────────────────────────────────────────────

func TestDefaultWebSearchLimits(t *testing.T) {
	limits := DefaultWebSearchLimits()
	if limits.MaxResults != 5 {
		t.Errorf("expected MaxResults=5, got %d", limits.MaxResults)
	}
	if limits.Timeout != 30*time.Second {
		t.Errorf("expected Timeout=30s, got %v", limits.Timeout)
	}
}

func TestDefaultFileLimits(t *testing.T) {
	limits := DefaultFileLimits()
	if limits.ReadDefaultLines != 2000 {
		t.Errorf("expected ReadDefaultLines=2000, got %d", limits.ReadDefaultLines)
	}
	if limits.MaxLineBytes != 1<<20 {
		t.Errorf("expected MaxLineBytes=%d, got %d", 1<<20, limits.MaxLineBytes)
	}
	if limits.MaxWindowLines != 50000 {
		t.Errorf("expected MaxWindowLines=50000, got %d", limits.MaxWindowLines)
	}
}

func TestDefaultBashTimeouts(t *testing.T) {
	to := DefaultBashTimeouts()
	if to.MaxTimeout != 120*time.Second {
		t.Errorf("expected MaxTimeout=120s, got %v", to.MaxTimeout)
	}
	if to.WaitDelay != 5*time.Second {
		t.Errorf("expected WaitDelay=5s, got %v", to.WaitDelay)
	}
}

func TestDefaultRipgrepLimits(t *testing.T) {
	limits := DefaultRipgrepLimits()
	if limits.Timeout != 60*time.Second {
		t.Errorf("expected Timeout=60s, got %v", limits.Timeout)
	}
}

// ── Additional Judge tests for tools that lacked them ──────────────────────

func TestEditFileTool_Judge_InsideWorkspace(t *testing.T) {
	tool := NewEditFileTool()
	tmpDir := t.TempDir()
	ctx := tools.WithWorkspacePath(context.Background(), tmpDir)

	testFile := filepath.Join(tmpDir, "edit.txt")
	if err := os.WriteFile(testFile, []byte("hello"), 0o644); err != nil {
		t.Fatal(err)
	}
	input, _ := json.Marshal(EditFileInput{Path: testFile, OldString: "hello", NewString: "world"})

	judgeOut := tool.Judge(ctx, input)
	allow, reasoning := judgeOut.Allow, judgeOut.Reason
	if !allow {
		t.Errorf("expected allow=true for edit inside workspace, got reasoning: %s", reasoning)
	}
	if !strings.Contains(reasoning, "workspace") {
		t.Errorf("expected reasoning to mention workspace, got: %s", reasoning)
	}
}

func TestEditFileTool_Judge_OutsideWorkspace(t *testing.T) {
	tool := NewEditFileTool()
	tmpDir := t.TempDir()
	otherDir := t.TempDir()
	ctx := tools.WithWorkspacePath(context.Background(), tmpDir)

	input, _ := json.Marshal(EditFileInput{Path: filepath.Join(otherDir, "edit.txt"), OldString: "x", NewString: "y"})

	judgeOut := tool.Judge(ctx, input)
	allow, reasoning := judgeOut.Allow, judgeOut.Reason
	if allow {
		t.Error("expected allow=false for edit outside workspace")
	}
	if !strings.Contains(reasoning, "outside") {
		t.Errorf("expected reasoning to mention outside, got: %s", reasoning)
	}
}

func TestEditFileTool_Judge_InvalidJSON(t *testing.T) {
	tool := NewEditFileTool()
	judgeOut := tool.Judge(context.Background(), json.RawMessage(`{bad`))
	allow, reasoning := judgeOut.Allow, judgeOut.Reason
	if allow {
		t.Error("expected allow=false for invalid JSON")
	}
	if reasoning != "" {
		t.Errorf("expected empty reasoning for invalid JSON, got: %s", reasoning)
	}
}

func TestListDirectoryTool_Judge_InsideWorkspace(t *testing.T) {
	tool := NewListDirectoryTool()
	tmpDir := t.TempDir()
	ctx := tools.WithWorkspacePath(context.Background(), tmpDir)

	input, _ := json.Marshal(ListDirectoryInput{Path: tmpDir})

	judgeOut := tool.Judge(ctx, input)
	allow, reasoning := judgeOut.Allow, judgeOut.Reason
	if !allow {
		t.Errorf("expected allow=true for list inside workspace, got reasoning: %s", reasoning)
	}
	if !strings.Contains(reasoning, "workspace") {
		t.Errorf("expected reasoning to mention workspace, got: %s", reasoning)
	}
}

func TestListDirectoryTool_Judge_OutsideWorkspace(t *testing.T) {
	tool := NewListDirectoryTool()
	tmpDir := t.TempDir()
	otherDir := t.TempDir()
	ctx := tools.WithWorkspacePath(context.Background(), tmpDir)

	input, _ := json.Marshal(ListDirectoryInput{Path: otherDir})

	judgeOut := tool.Judge(ctx, input)
	allow, reasoning := judgeOut.Allow, judgeOut.Reason
	if allow {
		t.Error("expected allow=false for list outside workspace")
	}
	if !strings.Contains(reasoning, "outside") {
		t.Errorf("expected reasoning to mention outside, got: %s", reasoning)
	}
}

func TestGlobTool_Judge_InsideWorkspace(t *testing.T) {
	tool := NewGlobTool()
	tmpDir := t.TempDir()
	ctx := tools.WithWorkspacePath(context.Background(), tmpDir)

	input, _ := json.Marshal(GlobInput{Pattern: "*.go", Path: tmpDir})

	judgeOut := tool.Judge(ctx, input)
	allow, reasoning := judgeOut.Allow, judgeOut.Reason
	if !allow {
		t.Errorf("expected allow=true for glob inside workspace, got reasoning: %s", reasoning)
	}
}

func TestGlobTool_Judge_NoWorkspace(t *testing.T) {
	tool := NewGlobTool()
	ctx := context.Background()

	input, _ := json.Marshal(GlobInput{Pattern: "*.go", Path: "/some/path"})

	judgeOut := tool.Judge(ctx, input)
	allow, reasoning := judgeOut.Allow, judgeOut.Reason
	// With no workspace and no temp dir, any path is considered "outside" —
	// the judge returns false with a descriptive reason.
	if allow {
		t.Error("expected allow=false when no workspace or temp dir is configured")
	}
	if !strings.Contains(reasoning, "outside") {
		t.Errorf("expected 'outside' in reasoning, got: %s", reasoning)
	}
}

func TestRipgrepTool_Judge_InsideWorkspace(t *testing.T) {
	tool := NewRipgrepTool()
	tmpDir := t.TempDir()
	ctx := tools.WithWorkspacePath(context.Background(), tmpDir)

	input, _ := json.Marshal(RipgrepInput{Pattern: "hello", Path: tmpDir})

	judgeOut := tool.Judge(ctx, input)
	allow, reasoning := judgeOut.Allow, judgeOut.Reason
	if !allow {
		t.Errorf("expected allow=true for ripgrep inside workspace, got reasoning: %s", reasoning)
	}
}

// ── validateWorkDir tests ──────────────────────────────────────────────────

func TestValidateWorkDir_EmptyWorkspace(t *testing.T) {
	// Empty roots → any directory accepted.
	err := validateWorkDir(context.Background(), "/anywhere", nil)
	if err != nil {
		t.Errorf("expected nil error when no workspace, got: %v", err)
	}
}

func TestValidateWorkDir_InsideWorkspace(t *testing.T) {
	tmpDir := t.TempDir()
	subDir := filepath.Join(tmpDir, "subdir")
	if err := os.Mkdir(subDir, 0o755); err != nil {
		t.Fatal(err)
	}
	err := validateWorkDir(context.Background(), subDir, []string{tmpDir})
	if err != nil {
		t.Errorf("expected nil error for inside workspace, got: %v", err)
	}
}

func TestValidateWorkDir_OutsideWorkspace(t *testing.T) {
	tmpDir := t.TempDir()
	// Use a path that is clearly outside workspace and NOT inside the system temp dir.
	// We use /dev because it's a real directory on Unix that is not under /tmp.
	err := validateWorkDir(context.Background(), "/dev", []string{tmpDir})
	if err == nil {
		t.Error("expected error for outside workspace")
	}
}

func TestValidateWorkDir_InsideTempDir(t *testing.T) {
	tmpDir := t.TempDir()
	subDir := filepath.Join(tmpDir, "cache")
	if err := os.Mkdir(subDir, 0o755); err != nil {
		t.Fatal(err)
	}
	// Temp dir from context (different from workspace) should still allow.
	ws := t.TempDir()
	err := validateWorkDir(context.Background(), subDir, []string{ws, tmpDir})
	if err != nil {
		t.Errorf("expected nil error for inside temp dir, got: %v", err)
	}
}

func TestValidateWorkDir_InsideSystemTemp(t *testing.T) {
	// System temp dir should always be allowed.
	tmpDir := os.TempDir()
	ws := t.TempDir()
	err := validateWorkDir(context.Background(), tmpDir, []string{ws})
	if err != nil {
		t.Errorf("expected nil error for system temp dir, got: %v", err)
	}
}

// ── resolveWorkspaceRoot tests ─────────────────────────────────────────────

func TestResolveWorkspaceRoot_Existing(t *testing.T) {
	tmpDir := t.TempDir()
	resolved, err := resolveWorkspaceRoot(tmpDir)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resolved == "" {
		t.Fatal("expected non-empty resolved path")
	}
	// Clean the result — resolved path should be clean.
	if resolved != filepath.Clean(resolved) {
		t.Errorf("expected clean path, got %q", resolved)
	}
}

func TestResolveWorkspaceRoot_NonExistent(t *testing.T) {
	// A path that doesn't exist should fall back to cleaned path.
	nonExistent := filepath.Join(t.TempDir(), "does-not-exist")
	resolved, err := resolveWorkspaceRoot(nonExistent)
	if err != nil {
		t.Fatalf("unexpected error for non-existent path: %v", err)
	}
	if resolved != filepath.Clean(nonExistent) {
		t.Errorf("expected cleaned path %q, got %q", filepath.Clean(nonExistent), resolved)
	}
}

// ── EditFile tool path validation tests ────────────────────────────────────

func TestEditFileTool_Execute_MissingPath(t *testing.T) {
	tool := NewEditFileTool()
	result, err := tool.Execute(context.Background(), json.RawMessage(`{"path":"","old_string":"x","new_string":"y"}`))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Error("expected IsError=true for missing path")
	}
	if !strings.Contains(result.Content, "path is required") {
		t.Errorf("expected 'path is required', got: %s", result.Content)
	}
}

func TestEditFileTool_Execute_MissingOldString(t *testing.T) {
	tool := NewEditFileTool()
	result, err := tool.Execute(context.Background(), json.RawMessage(`{"path":"/some/file","old_string":"","new_string":"y"}`))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Error("expected IsError=true for missing old_string")
	}
	if !strings.Contains(result.Content, "old_string is required") {
		t.Errorf("expected 'old_string is required', got: %s", result.Content)
	}
}

func TestEditFileTool_Execute_OutsideSessionRoots(t *testing.T) {
	tool := NewEditFileTool()
	ws := t.TempDir()
	otherDir := t.TempDir()
	target := filepath.Join(otherDir, "file.txt")
	if err := os.WriteFile(target, []byte("hello world"), 0o644); err != nil {
		t.Fatal(err)
	}
	ctx := tools.WithWorkspacePath(context.Background(), ws)
	input, _ := json.Marshal(EditFileInput{Path: target, OldString: "hello", NewString: "goodbye"})
	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Errorf("expected Execute to edit file outside session roots (containment check removed), got: %s", result.Content)
	}
}

// ── Path resolution edge cases ─────────────────────────────────────────────

func TestResolvePath_NoWorkspace(t *testing.T) {
	ctx := context.Background()
	result := resolvePath(ctx, "/some/absolute/path")
	if result != "/some/absolute/path" {
		t.Errorf("expected path returned as-is without workspace, got %q", result)
	}
}

func TestResolvePath_AbsoluteInsideWorkspace(t *testing.T) {
	tmpDir := t.TempDir()
	ctx := tools.WithWorkspacePath(context.Background(), tmpDir)
	subFile := filepath.Join(tmpDir, "file.txt")
	if err := os.WriteFile(subFile, []byte("data"), 0o644); err != nil {
		t.Fatal(err)
	}
	result := resolvePath(ctx, subFile)
	if result == "" {
		t.Fatal("expected non-empty path")
	}
}

func TestValidateResolvedPath_EmptyResolved(t *testing.T) {
	err := validateResolvedPath("")
	if err == nil {
		t.Error("expected error for empty resolved path")
	}
	if !strings.Contains(err.Error(), "outside the session workspace") {
		t.Errorf("expected workspace error, got: %v", err)
	}
}

func TestValidateResolvedPath_NonEmpty(t *testing.T) {
	err := validateResolvedPath("/some/absolute/path")
	if err != nil {
		t.Errorf("expected nil error for non-empty resolved path, got: %v", err)
	}
}

func TestValidateResolvedPath_InsideWorkspace(t *testing.T) {
	tmpDir := t.TempDir()
	subFile := filepath.Join(tmpDir, "file.txt")
	if err := os.WriteFile(subFile, []byte("data"), 0o644); err != nil {
		t.Fatal(err)
	}
	err := validateResolvedPath(subFile)
	if err != nil {
		t.Errorf("expected nil error for path inside workspace, got: %v", err)
	}
}

// ── Read file validation edge cases ─────────────────────────────────────────

func TestReadFileTool_Execute_MissingPath(t *testing.T) {
	tool := NewReadFileTool()
	result, err := tool.Execute(context.Background(), json.RawMessage(`{"path":""}`))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Error("expected IsError=true for missing path")
	}
	if !strings.Contains(result.Content, "path is required") {
		t.Errorf("expected 'path is required', got: %s", result.Content)
	}
}

func TestReadFileTool_Execute_InvalidStartLine(t *testing.T) {
	tool := NewReadFileTool()
	result, err := tool.Execute(context.Background(), json.RawMessage(`{"path":"/some/file","start_line":-1}`))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Error("expected IsError=true for negative start_line")
	}
	if !strings.Contains(result.Content, "start_line must be >= 1") {
		t.Errorf("expected start_line validation error, got: %s", result.Content)
	}
}

func TestReadFileTool_Execute_InvalidEndLine(t *testing.T) {
	tool := NewReadFileTool()
	result, err := tool.Execute(context.Background(), json.RawMessage(`{"path":"/some/file","end_line":-5}`))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Error("expected IsError=true for negative end_line")
	}
	if !strings.Contains(result.Content, "end_line must be >= 1") {
		t.Errorf("expected end_line validation error, got: %s", result.Content)
	}
}

func TestReadFileTool_Execute_StartAfterEnd(t *testing.T) {
	tool := NewReadFileTool()
	result, err := tool.Execute(context.Background(), json.RawMessage(`{"path":"/some/file","start_line":10,"end_line":5}`))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Error("expected IsError=true when start_line > end_line")
	}
	if !strings.Contains(result.Content, "must not exceed end_line") {
		t.Errorf("expected start/end validation error, got: %s", result.Content)
	}
}

func TestReadFileTool_Execute_OutsideSessionRoots(t *testing.T) {
	tool := NewReadFileTool()
	ws := t.TempDir()
	otherDir := t.TempDir()
	target := filepath.Join(otherDir, "file.txt")
	if err := os.WriteFile(target, []byte("data"), 0o644); err != nil {
		t.Fatal(err)
	}
	ctx := tools.WithWorkspacePath(context.Background(), ws)
	// Build the JSON with json.Marshal so Windows backslash paths
	// (e.g. C:\Users) are properly escaped — string concatenation would
	// produce invalid JSON escape sequences (\U) on Windows.
	input, _ := json.Marshal(map[string]string{"path": target})
	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Errorf("expected Execute to read file outside session roots (containment check removed), got: %s", result.Content)
	}
}

// ── Delete file validation edge cases ───────────────────────────────────────

func TestDeleteFileTool_Execute_MissingPath(t *testing.T) {
	tool := NewDeleteFileTool()
	result, err := tool.Execute(context.Background(), json.RawMessage(`{"path":""}`))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Error("expected IsError=true for missing path")
	}
	if !strings.Contains(result.Content, "path is required") {
		t.Errorf("expected 'path is required', got: %s", result.Content)
	}
}

func TestDeleteFileTool_Execute_OutsideSessionRoots(t *testing.T) {
	tool := NewDeleteFileTool()
	ws := t.TempDir()
	otherDir := t.TempDir()
	target := filepath.Join(otherDir, "file.txt")
	if err := os.WriteFile(target, []byte("data"), 0o644); err != nil {
		t.Fatal(err)
	}
	ctx := tools.WithWorkspacePath(context.Background(), ws)
	input, _ := json.Marshal(DeleteFileInput{Path: target})
	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Errorf("expected Execute to delete file outside session roots (containment check removed), got: %s", result.Content)
	}
}

// ── Create directory validation edge cases ──────────────────────────────────

func TestCreateDirectoryTool_Execute_MissingPath(t *testing.T) {
	tool := NewCreateDirectoryTool()
	result, err := tool.Execute(context.Background(), json.RawMessage(`{"path":""}`))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Error("expected IsError=true for missing path")
	}
	if !strings.Contains(result.Content, "path is required") {
		t.Errorf("expected 'path is required', got: %s", result.Content)
	}
}

func TestCreateDirectoryTool_Execute_OutsideSessionRoots(t *testing.T) {
	tool := NewCreateDirectoryTool()
	ws := t.TempDir()
	otherDir := t.TempDir()
	target := filepath.Join(otherDir, "newdir")
	ctx := tools.WithWorkspacePath(context.Background(), ws)
	input, _ := json.Marshal(CreateDirectoryInput{Path: target})
	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Errorf("expected Execute to create directory outside session roots (containment check removed), got: %s", result.Content)
	}
}

// ── Delete directory validation edge cases ──────────────────────────────────

func TestDeleteDirectoryTool_Execute_MissingPath(t *testing.T) {
	tool := NewDeleteDirectoryTool()
	result, err := tool.Execute(context.Background(), json.RawMessage(`{"path":"","recursive":false}`))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Error("expected IsError=true for missing path")
	}
	if !strings.Contains(result.Content, "path is required") {
		t.Errorf("expected 'path is required', got: %s", result.Content)
	}
}

func TestDeleteDirectoryTool_Execute_NotADirectory(t *testing.T) {
	tool := NewDeleteDirectoryTool()
	tmpDir := t.TempDir()
	testFile := filepath.Join(tmpDir, "file.txt")
	if err := os.WriteFile(testFile, []byte("data"), 0o644); err != nil {
		t.Fatal(err)
	}
	ctx := tools.WithWorkspacePath(context.Background(), tmpDir)
	input, _ := json.Marshal(DeleteDirectoryInput{Path: testFile, Recursive: false})
	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Error("expected IsError=true when path is not a directory")
	}
	if !strings.Contains(result.Content, "not a directory") {
		t.Errorf("expected 'not a directory' in error, got: %s", result.Content)
	}
}

func TestDeleteDirectoryTool_Execute_OutsideSessionRoots(t *testing.T) {
	tool := NewDeleteDirectoryTool()
	ws := t.TempDir()
	otherDir := t.TempDir()
	target := filepath.Join(otherDir, "somedir")
	if err := os.Mkdir(target, 0o755); err != nil {
		t.Fatal(err)
	}
	ctx := tools.WithWorkspacePath(context.Background(), ws)
	input, _ := json.Marshal(DeleteDirectoryInput{Path: target, Recursive: false})
	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Errorf("expected Execute to delete directory outside session roots (containment check removed), got: %s", result.Content)
	}
}

// ── Write file validation edge cases ────────────────────────────────────────

func TestWriteFileTool_Execute_MissingPath(t *testing.T) {
	tool := NewWriteFileTool()
	result, err := tool.Execute(context.Background(), json.RawMessage(`{"path":"","content":"test"}`))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Error("expected IsError=true for missing path")
	}
	if !strings.Contains(result.Content, "path is required") {
		t.Errorf("expected 'path is required', got: %s", result.Content)
	}
}

func TestWriteFileTool_Execute_OutsideSessionRoots(t *testing.T) {
	tool := NewWriteFileTool()
	ws := t.TempDir()
	otherDir := t.TempDir()
	target := filepath.Join(otherDir, "newfile")
	ctx := tools.WithWorkspacePath(context.Background(), ws)
	input, _ := json.Marshal(WriteFileInput{Path: target, Content: "test"})
	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Errorf("expected Execute to write file outside session roots (containment check removed), got: %s", result.Content)
	}
}

// ── tool_result_read basic tests ────────────────────────────────────────────

func TestNewToolResultReadTool(t *testing.T) {
	tool := NewToolResultReadTool()
	if tool.Name() != "tool_result_read" {
		t.Errorf("expected name 'tool_result_read', got %q", tool.Name())
	}
	if tool.DefaultPolicy() != tools.PolicyAlwaysAllow {
		t.Errorf("expected PolicyAlwaysAllow, got %v", tool.DefaultPolicy())
	}
}

func TestToolResultReadTool_Execute_MissingHash(t *testing.T) {
	tool := NewToolResultReadTool()
	result, err := tool.Execute(context.Background(), json.RawMessage(`{"hash":""}`))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Error("expected IsError=true for missing hash")
	}
	if !strings.Contains(result.Content, "hash is required") {
		t.Errorf("expected 'hash is required', got: %s", result.Content)
	}
}

func TestToolResultReadTool_Execute_InvalidJSON(t *testing.T) {
	tool := NewToolResultReadTool()
	result, err := tool.Execute(context.Background(), json.RawMessage(`{bad`))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Error("expected IsError=true for invalid JSON")
	}
	if !strings.Contains(result.Content, "failed to parse input") {
		t.Errorf("expected parse error, got: %s", result.Content)
	}
}

func TestToolResultReadTool_Execute_NoCache(t *testing.T) {
	tool := NewToolResultReadTool()
	result, err := tool.Execute(context.Background(), json.RawMessage(`{"hash":"abc123"}`))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Error("expected IsError=true when no cache in context")
	}
	if !strings.Contains(result.Content, "Tool result cache not available") {
		t.Errorf("expected cache error, got: %s", result.Content)
	}
}

func TestToolResultReadTool_FileBacked(t *testing.T) {
	tmpDir := t.TempDir()
	testFile := filepath.Join(tmpDir, "cached.go")

	var sb strings.Builder
	for i := 1; i <= 1000; i++ {
		if i > 1 {
			sb.WriteByte('\n')
		}
		fmt.Fprintf(&sb, "Line %d", i)
	}
	if err := os.WriteFile(testFile, []byte(sb.String()), 0o644); err != nil {
		t.Fatal(err)
	}

	info, err := os.Stat(testFile)
	if err != nil {
		t.Fatal(err)
	}

	cache := agent.NewToolResultCache(5 * time.Minute)
	hash := cache.Store("read_file", "", agent.ToolCacheMeta{
		FilePath:   testFile,
		FileMtime:  info.ModTime().UnixNano(),
		FileSize:   info.Size(),
		FileBacked: true,
	})

	ctx := agent.WithToolResultCache(context.Background(), cache)
	tool := NewToolResultReadTool()

	// Read first 10 lines
	input, _ := json.Marshal(map[string]any{
		"hash":       hash,
		"start_line": 1,
		"num_lines":  10,
	})
	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Errorf("expected success, got error: %s", result.Content)
	}
	if !strings.Contains(result.Content, "Lines 1-10 of 1000") {
		t.Errorf("expected 'Lines 1-10 of 1000' in header, got: %s", result.Content[:min(80, len(result.Content))])
	}
	if !strings.Contains(result.Content, "Line 1") {
		t.Errorf("expected 'Line 1' in content")
	}
	if !strings.Contains(result.Content, "Line 10") {
		t.Errorf("expected 'Line 10' in content")
	}
	if !strings.Contains(result.Content, "[Use tool_result_read(") {
		t.Errorf("expected continuation hint")
	}
}

// The `line` parameter is the escape-hatch: a file-backed read whose window
// truncates a pathological line (via MaxLineBytes) must be recoverable in full
// by an explicit tool_result_read(hash, line=N) request. Default reads keep the
// DoS guard; only the explicit line request bypasses it.
func TestToolResultReadTool_FileBacked_LineEscapeHatch(t *testing.T) {
	tmpDir := t.TempDir()
	testFile := filepath.Join(tmpDir, "bundle.txt")

	longLine := strings.Repeat("a", 5000) // exceeds default MaxLineBytes (1 MiB? no — see below)
	// Use a small cap to force truncation deterministically without a 1 MiB line.
	content := "line one\n" + longLine + "\nline three\n"
	if err := os.WriteFile(testFile, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	info, err := os.Stat(testFile)
	if err != nil {
		t.Fatal(err)
	}

	cache := agent.NewToolResultCache(5 * time.Minute)
	hash := cache.Store("read_file", "", agent.ToolCacheMeta{
		FilePath:   testFile,
		FileMtime:  info.ModTime().UnixNano(),
		FileSize:   info.Size(),
		FileBacked: true,
	})

	ctx := agent.WithToolResultCache(context.Background(), cache)
	tool := NewToolResultReadTool()

	// Default window read applies MaxLineBytes (1 MiB default) — the 5000-byte
	// line fits, so no truncation here; this confirms the guard's default size.
	windowInput, _ := json.Marshal(map[string]any{
		"hash":       hash,
		"start_line": 1,
		"num_lines":  3,
	})
	windowRes, err := tool.Execute(ctx, windowInput)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(windowRes.Content, longLine[:100]) {
		t.Errorf("expected the long line in the window read, got: %s", windowRes.Content)
	}

	// Explicit line=N returns the FULL raw line, bypassing any cap.
	lineInput, _ := json.Marshal(map[string]any{
		"hash": hash,
		"line": 2,
	})
	lineRes, err := tool.Execute(ctx, lineInput)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if lineRes.IsError {
		t.Errorf("expected success, got error: %s", lineRes.Content)
	}
	if !strings.Contains(lineRes.Content, "Full line 2 of 3") {
		t.Errorf("expected 'Full line 2 of 3' header, got: %s", lineRes.Content[:min(80, len(lineRes.Content))])
	}
	if !strings.Contains(lineRes.Content, longLine) {
		t.Errorf("escape-hatch must return the full raw line (bypassing MaxLineBytes)")
	}

	// Line past end of file is reported gracefully.
	pastInput, _ := json.Marshal(map[string]any{"hash": hash, "line": 99})
	pastRes, err := tool.Execute(ctx, pastInput)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(pastRes.Content, "not available") {
		t.Errorf("expected 'not available' for line past EOF, got: %s", pastRes.Content)
	}
}

func TestToolResultReadTool_FileBacked_MidFile(t *testing.T) {
	tmpDir := t.TempDir()
	testFile := filepath.Join(tmpDir, "cached.go")

	var sb strings.Builder
	for i := 1; i <= 1000; i++ {
		if i > 1 {
			sb.WriteByte('\n')
		}
		fmt.Fprintf(&sb, "Line %d", i)
	}
	if err := os.WriteFile(testFile, []byte(sb.String()), 0o644); err != nil {
		t.Fatal(err)
	}

	info, err := os.Stat(testFile)
	if err != nil {
		t.Fatal(err)
	}

	cache := agent.NewToolResultCache(5 * time.Minute)
	hash := cache.Store("read_file", "", agent.ToolCacheMeta{
		FilePath:   testFile,
		FileMtime:  info.ModTime().UnixNano(),
		FileSize:   info.Size(),
		FileBacked: true,
	})

	ctx := agent.WithToolResultCache(context.Background(), cache)
	tool := NewToolResultReadTool()

	// Read lines 500-510
	input, _ := json.Marshal(map[string]any{
		"hash":       hash,
		"start_line": 500,
		"num_lines":  10,
	})
	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Errorf("expected success, got error: %s", result.Content)
	}
	if !strings.Contains(result.Content, "Lines 500-509 of 1000") {
		t.Errorf("expected 'Lines 500-509 of 1000' in header, got: %s", result.Content[:min(80, len(result.Content))])
	}
	if !strings.Contains(result.Content, "Line 500") {
		t.Errorf("expected 'Line 500' in content")
	}
	if !strings.Contains(result.Content, "Line 509") {
		t.Errorf("expected 'Line 509' in content")
	}
}

func TestToolResultReadTool_FileBacked_StaleFile(t *testing.T) {
	tmpDir := t.TempDir()
	testFile := filepath.Join(tmpDir, "cached.go")
	if err := os.WriteFile(testFile, []byte("original content"), 0o644); err != nil {
		t.Fatal(err)
	}

	info, err := os.Stat(testFile)
	if err != nil {
		t.Fatal(err)
	}

	cache := agent.NewToolResultCache(5 * time.Minute)
	hash := cache.Store("read_file", "", agent.ToolCacheMeta{
		FilePath:   testFile,
		FileMtime:  info.ModTime().UnixNano(),
		FileSize:   info.Size(),
		FileBacked: true,
	})

	// Modify the file after caching
	if err := os.WriteFile(testFile, []byte("modified content that is longer"), 0o644); err != nil {
		t.Fatal(err)
	}

	ctx := agent.WithToolResultCache(context.Background(), cache)
	tool := NewToolResultReadTool()

	input, _ := json.Marshal(map[string]any{
		"hash":       hash,
		"start_line": 1,
		"num_lines":  10,
	})
	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Error("expected IsError=true for stale file-backed entry")
	}
	if !strings.Contains(result.Content, "stale") {
		t.Errorf("expected 'stale' in error message, got: %s", result.Content)
	}
}

func TestToolResultReadTool_FileBacked_EmptyFile(t *testing.T) {
	tmpDir := t.TempDir()
	testFile := filepath.Join(tmpDir, "empty.go")
	if err := os.WriteFile(testFile, []byte(""), 0o644); err != nil {
		t.Fatal(err)
	}

	info, err := os.Stat(testFile)
	if err != nil {
		t.Fatal(err)
	}

	cache := agent.NewToolResultCache(5 * time.Minute)
	hash := cache.Store("read_file", "", agent.ToolCacheMeta{
		FilePath:   testFile,
		FileMtime:  info.ModTime().UnixNano(),
		FileSize:   info.Size(),
		FileBacked: true,
	})

	ctx := agent.WithToolResultCache(context.Background(), cache)
	tool := NewToolResultReadTool()

	input, _ := json.Marshal(map[string]any{
		"hash":       hash,
		"start_line": 1,
		"num_lines":  10,
	})
	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Errorf("expected success for empty file, got error: %s", result.Content)
	}
	if !strings.Contains(result.Content, "empty file") {
		t.Errorf("expected 'empty file' in content, got: %s", result.Content)
	}
}

// ── Session roots: temp directory equal-peer tests ──────────────────────────
//
// The session workspace and session temp directory are equal peers. File
// operations (read and write) inside either root auto-approve (when
// auto_approve_workspace_writes is enabled for write tools, or always for
// read tools with PolicyAlwaysAllow). Operations outside both roots require
// user confirmation. These tests verify the temp directory is treated as a
// first-class root, not a second-class citizen.

func TestWriteFileTool_Judge_InsideTempDir(t *testing.T) {
	tool := NewWriteFileTool()
	ws := t.TempDir()
	tempDir := t.TempDir()
	ctx := tools.WithWorkspacePath(tools.WithTempDir(context.Background(), tempDir), ws)

	testFile := filepath.Join(tempDir, "output.txt")
	input, _ := json.Marshal(WriteFileInput{Path: testFile, Content: "data"})

	judgeOut := tool.Judge(ctx, input)
	allow, reasoning := judgeOut.Allow, judgeOut.Reason
	if !allow {
		t.Errorf("expected Judge to return allow=true for write inside temp dir, got reasoning: %s", reasoning)
	}
	if !strings.Contains(reasoning, "temp") {
		t.Errorf("expected reasoning to mention 'temp', got: %s", reasoning)
	}
}

func TestEditFileTool_Judge_InsideTempDir(t *testing.T) {
	tool := NewEditFileTool()
	ws := t.TempDir()
	tempDir := t.TempDir()
	ctx := tools.WithWorkspacePath(tools.WithTempDir(context.Background(), tempDir), ws)

	target := filepath.Join(tempDir, "edit.txt")
	if err := os.WriteFile(target, []byte("hello"), 0o644); err != nil {
		t.Fatal(err)
	}
	input, _ := json.Marshal(EditFileInput{Path: target, OldString: "hello", NewString: "world"})

	judgeOut := tool.Judge(ctx, input)
	allow, reasoning := judgeOut.Allow, judgeOut.Reason
	if !allow {
		t.Errorf("expected Judge to return allow=true for edit inside temp dir, got reasoning: %s", reasoning)
	}
}

func TestDeleteFileTool_Judge_InsideTempDir(t *testing.T) {
	tool := NewDeleteFileTool()
	ws := t.TempDir()
	tempDir := t.TempDir()
	ctx := tools.WithWorkspacePath(tools.WithTempDir(context.Background(), tempDir), ws)

	target := filepath.Join(tempDir, "file.txt")
	if err := os.WriteFile(target, []byte("data"), 0o644); err != nil {
		t.Fatal(err)
	}
	input, _ := json.Marshal(DeleteFileInput{Path: target})

	judgeOut := tool.Judge(ctx, input)
	allow, _ := judgeOut.Allow, judgeOut.Reason
	if !allow {
		t.Error("expected Judge to return allow=true for delete_file inside temp dir")
	}
}

func TestCreateDirectoryTool_Judge_InsideTempDir(t *testing.T) {
	tool := NewCreateDirectoryTool()
	ws := t.TempDir()
	tempDir := t.TempDir()
	ctx := tools.WithWorkspacePath(tools.WithTempDir(context.Background(), tempDir), ws)

	target := filepath.Join(tempDir, "newdir")
	input, _ := json.Marshal(CreateDirectoryInput{Path: target})

	judgeOut := tool.Judge(ctx, input)
	allow, _ := judgeOut.Allow, judgeOut.Reason
	if !allow {
		t.Error("expected Judge to return allow=true for create_directory inside temp dir")
	}
}

func TestDeleteDirectoryTool_Judge_InsideTempDir(t *testing.T) {
	tool := NewDeleteDirectoryTool()
	ws := t.TempDir()
	tempDir := t.TempDir()
	ctx := tools.WithWorkspacePath(tools.WithTempDir(context.Background(), tempDir), ws)

	target := filepath.Join(tempDir, "somedir")
	if err := os.Mkdir(target, 0o755); err != nil {
		t.Fatal(err)
	}
	input, _ := json.Marshal(DeleteDirectoryInput{Path: target, Recursive: false})

	judgeOut := tool.Judge(ctx, input)
	allow, _ := judgeOut.Allow, judgeOut.Reason
	if !allow {
		t.Error("expected Judge to return allow=true for delete_directory inside temp dir")
	}
}

func TestReadFileTool_Judge_InsideTempDir_WithWorkspace(t *testing.T) {
	tool := NewReadFileTool()
	ws := t.TempDir()
	tempDir := t.TempDir()
	ctx := tools.WithWorkspacePath(tools.WithTempDir(context.Background(), tempDir), ws)

	target := filepath.Join(tempDir, "file.txt")
	if err := os.WriteFile(target, []byte("data"), 0o644); err != nil {
		t.Fatal(err)
	}
	input, _ := json.Marshal(map[string]string{"path": target})

	judgeOut := tool.Judge(ctx, input)
	allow, _ := judgeOut.Allow, judgeOut.Reason
	if !allow {
		t.Error("expected Judge to return allow=true for read inside temp dir (with workspace set)")
	}
}

func TestListDirectoryTool_Judge_InsideTempDir(t *testing.T) {
	tool := NewListDirectoryTool()
	ws := t.TempDir()
	tempDir := t.TempDir()
	ctx := tools.WithWorkspacePath(tools.WithTempDir(context.Background(), tempDir), ws)

	input, _ := json.Marshal(ListDirectoryInput{Path: tempDir})

	judgeOut := tool.Judge(ctx, input)
	allow, _ := judgeOut.Allow, judgeOut.Reason
	if !allow {
		t.Error("expected Judge to return allow=true for list_directory inside temp dir")
	}
}

// ── Session roots: operations outside both roots require confirmation ───────

func TestWriteFileTool_Judge_OutsideBothRoots(t *testing.T) {
	tool := NewWriteFileTool()
	ws := t.TempDir()
	tempDir := t.TempDir()
	otherDir := t.TempDir()
	ctx := tools.WithWorkspacePath(tools.WithTempDir(context.Background(), tempDir), ws)

	target := filepath.Join(otherDir, "file.txt")
	input, _ := json.Marshal(WriteFileInput{Path: target, Content: "data"})

	judgeOut := tool.Judge(ctx, input)
	allow, reasoning := judgeOut.Allow, judgeOut.Reason
	if allow {
		t.Error("expected Judge to return allow=false for write outside both roots")
	}
	if !strings.Contains(reasoning, "outside") {
		t.Errorf("expected reasoning to mention 'outside', got: %s", reasoning)
	}
}

func TestReadFileTool_Judge_OutsideBothRoots(t *testing.T) {
	tool := NewReadFileTool()
	ws := t.TempDir()
	tempDir := t.TempDir()
	otherDir := t.TempDir()
	ctx := tools.WithWorkspacePath(tools.WithTempDir(context.Background(), tempDir), ws)

	target := filepath.Join(otherDir, "file.txt")
	if err := os.WriteFile(target, []byte("data"), 0o644); err != nil {
		t.Fatal(err)
	}
	input, _ := json.Marshal(map[string]string{"path": target})

	judgeOut := tool.Judge(ctx, input)
	allow, reasoning := judgeOut.Allow, judgeOut.Reason
	if allow {
		t.Error("expected Judge to return allow=false for read outside both roots")
	}
	if !strings.Contains(reasoning, "outside") {
		t.Errorf("expected reasoning to mention 'outside', got: %s", reasoning)
	}
}

// ── isPathInSessionRoots unit tests ──────────────────────────────────────────

func TestIsPathInSessionRoots_Workspace(t *testing.T) {
	ws := t.TempDir()
	ctx := tools.WithWorkspacePath(context.Background(), ws)
	target := filepath.Join(ws, "subdir", "file.txt")
	if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(target, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if !isPathInSessionRoots(ctx, target) {
		t.Error("expected path inside workspace to be in session roots")
	}
}

func TestIsPathInSessionRoots_TempDir(t *testing.T) {
	ws := t.TempDir()
	tempDir := t.TempDir()
	ctx := tools.WithWorkspacePath(tools.WithTempDir(context.Background(), tempDir), ws)
	target := filepath.Join(tempDir, "file.txt")
	if err := os.WriteFile(target, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if !isPathInSessionRoots(ctx, target) {
		t.Error("expected path inside temp dir to be in session roots")
	}
}

func TestIsPathInSessionRoots_Outside(t *testing.T) {
	ws := t.TempDir()
	tempDir := t.TempDir()
	otherDir := t.TempDir()
	ctx := tools.WithWorkspacePath(tools.WithTempDir(context.Background(), tempDir), ws)
	target := filepath.Join(otherDir, "file.txt")
	if isPathInSessionRoots(ctx, target) {
		t.Error("expected path outside both roots to NOT be in session roots")
	}
}

func TestIsPathInSessionRoots_NoRoots(t *testing.T) {
	ctx := context.Background()
	if isPathInSessionRoots(ctx, "/some/path") {
		t.Error("expected false when no roots configured")
	}
}

// TestIsPathInSessionRoots_HarmlessDevice verifies harmless special-device
// paths are treated as local by the file-tool judges, so read_file/write_file
// targeting them do not force a confirmation. The harmless set is host-
// dependent: POSIX exempts /dev/null and /dev/full, Windows exempts the
// reserved NUL device (its /dev/null equivalent) — on Windows /dev/null and
// /dev/full are ordinary out-of-root paths, not devices.
func TestIsPathInSessionRoots_HarmlessDevice(t *testing.T) {
	ws := t.TempDir()
	ctx := tools.WithWorkspacePath(context.Background(), ws)
	harmless := []string{"/dev/null", "/dev/full"}
	if runtime.GOOS == "windows" {
		harmless = []string{"NUL"}
	}
	for _, p := range harmless {
		if !isPathInSessionRoots(ctx, p) {
			t.Errorf("expected harmless device %q to be treated as local", p)
		}
	}
	// Process streams are NOT harmless — their targets are host-defined file
	// descriptors — so they must be treated as out-of-root and gated.
	for _, p := range []string{"/dev/stdin", "/dev/stdout", "/dev/stderr"} {
		if isPathInSessionRoots(ctx, p) {
			t.Errorf("expected process stream %q to NOT be treated as local", p)
		}
	}
	// A genuinely out-of-root path must still be rejected.
	if isPathInSessionRoots(ctx, "/etc/passwd") {
		t.Error("expected /etc/passwd to NOT be in session roots")
	}
}

// TestIsPathInSessionRoots_AllowedRoot proves a path inside an additional
// allowed root (auxiliary working directory) is treated identically to the
// workspace and temp directory.
func TestIsPathInSessionRoots_AllowedRoot(t *testing.T) {
	ws := t.TempDir()
	allowedRoot := t.TempDir()
	ctx := tools.WithWorkspacePath(context.Background(), ws)
	ctx = tools.WithAllowedRoots(ctx, []string{allowedRoot})
	target := filepath.Join(allowedRoot, "subdir", "file.txt")
	if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(target, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if !isPathInSessionRoots(ctx, target) {
		t.Error("expected path inside allowed root to be in session roots")
	}
}

// TestIsPathInSessionRoots_DifferingCase proves path-locality is
// case-insensitive: a target whose path components differ only in letter case
// from a session root is still recognized as local. The decisive case is a
// NOT-YET-EXISTING target under a root whose canonical casing cannot be
// recovered by EvalSymlinks — here the differing component lives entirely in
// the non-existent suffix of an allowed root. This mirrors macOS APFS /
// Windows NTFS semantics, where letter case does not change which file is
// referenced.
func TestIsPathInSessionRoots_DifferingCase(t *testing.T) {
	ws := t.TempDir()
	allowedRoot := t.TempDir()
	ctx := tools.WithWorkspacePath(context.Background(), ws)
	ctx = tools.WithAllowedRoots(ctx, []string{allowedRoot})
	// allowedRoot exists on disk (canonical casing). Build a target whose
	// non-existent suffix uses a casing variant of a NEW component under it.
	// The existing prefix (allowedRoot) resolves to itself, so the case
	// difference survives in the unresolved tail and must be folded.
	target := filepath.Join(allowedRoot, "Project", "newfile.txt") // existing root, new "Project"
	// Probe with the opposite casing in the target's differing component.
	casedTarget := filepath.Join(allowedRoot, "project", "newfile.txt")
	if isPathInSessionRoots(ctx, target) != isPathInSessionRoots(ctx, casedTarget) {
		t.Error("locality verdict must be identical regardless of letter case")
	}
	if !isPathInSessionRoots(ctx, casedTarget) {
		t.Error("expected differing-case target to be in session roots (case-insensitive)")
	}
}

// TestValidateWorkDir_InsideAllowedRoot proves validateWorkDir accepts a
// working directory inside an additional allowed root.
func TestValidateWorkDir_InsideAllowedRoot(t *testing.T) {
	ws := t.TempDir()
	allowedRoot := t.TempDir()
	subDir := filepath.Join(allowedRoot, "build")
	if err := os.Mkdir(subDir, 0o755); err != nil {
		t.Fatal(err)
	}
	roots := []string{ws, allowedRoot}
	if err := validateWorkDir(context.Background(), subDir, roots); err != nil {
		t.Errorf("expected nil error for inside allowed root, got: %v", err)
	}
}

// --- Optional-path read judge (glob, ripgrep) ---

// TestGlobTool_Judge_OptionalPath verifies that glob — whose `path` parameter
// is optional and defaults to the workspace root — auto-approves when `path`
// is omitted (the safest case), instead of escalating with "cannot determine
// target path".
func TestGlobTool_Judge_OptionalPath(t *testing.T) {
	tool := NewGlobTool()

	t.Run("omitted path defaults to workspace and auto-allows", func(t *testing.T) {
		tmpDir := t.TempDir()
		ctx := tools.WithWorkspacePath(context.Background(), tmpDir)

		input, _ := json.Marshal(map[string]string{
			"pattern": "frontend/src/hooks/events/*.test.ts",
			// no "path" — should default to workspace
		})

		judgeOut := tool.Judge(ctx, input)
		allow, reasoning := judgeOut.Allow, judgeOut.Reason
		if !allow {
			t.Errorf("expected allow=true for omitted path (workspace default), got reasoning: %s", reasoning)
		}
	})

	t.Run("explicit path inside workspace auto-allows", func(t *testing.T) {
		tmpDir := t.TempDir()
		subDir := filepath.Join(tmpDir, "src")
		_ = os.MkdirAll(subDir, 0o755)
		ctx := tools.WithWorkspacePath(context.Background(), tmpDir)

		input, _ := json.Marshal(map[string]string{
			"pattern": "**/*.ts",
			"path":    subDir,
		})

		judgeOut := tool.Judge(ctx, input)
		allow, reasoning := judgeOut.Allow, judgeOut.Reason
		if !allow {
			t.Errorf("expected allow=true for path inside workspace, got reasoning: %s", reasoning)
		}
	})

	t.Run("explicit path outside workspace escalates", func(t *testing.T) {
		tmpDir := t.TempDir()
		otherDir := t.TempDir()
		ctx := tools.WithWorkspacePath(context.Background(), tmpDir)

		input, _ := json.Marshal(map[string]string{
			"pattern": "**/*.ts",
			"path":    otherDir,
		})

		judgeOut := tool.Judge(ctx, input)
		allow, _ := judgeOut.Allow, judgeOut.Reason
		if allow {
			t.Error("expected allow=false for path outside workspace")
		}
	})

	t.Run("omitted path with no workspace fails closed", func(t *testing.T) {
		input, _ := json.Marshal(map[string]string{
			"pattern": "**/*.ts",
		})

		judgeOut := tool.Judge(context.Background(), input)
		allow, reasoning := judgeOut.Allow, judgeOut.Reason
		if allow {
			t.Error("expected allow=false when no workspace and no path")
		}
		if reasoning != "cannot determine target path" {
			t.Errorf("expected 'cannot determine target path', got: %s", reasoning)
		}
	})
}

// TestRipgrepTool_Judge_OptionalPath verifies ripgrep — whose `path` parameter
// is optional and defaults to the workspace root — auto-approves when `path`
// is omitted.
func TestRipgrepTool_Judge_OptionalPath(t *testing.T) {
	tool := NewRipgrepTool()

	t.Run("omitted path defaults to workspace and auto-allows", func(t *testing.T) {
		tmpDir := t.TempDir()
		ctx := tools.WithWorkspacePath(context.Background(), tmpDir)

		input, _ := json.Marshal(map[string]string{
			"pattern": "foo",
		})

		judgeOut := tool.Judge(ctx, input)
		allow, reasoning := judgeOut.Allow, judgeOut.Reason
		if !allow {
			t.Errorf("expected allow=true for omitted path (workspace default), got reasoning: %s", reasoning)
		}
	})

	t.Run("explicit path outside workspace escalates", func(t *testing.T) {
		tmpDir := t.TempDir()
		otherDir := t.TempDir()
		ctx := tools.WithWorkspacePath(context.Background(), tmpDir)

		input, _ := json.Marshal(map[string]string{
			"pattern": "foo",
			"path":    otherDir,
		})

		judgeOut := tool.Judge(ctx, input)
		allow, _ := judgeOut.Allow, judgeOut.Reason
		if allow {
			t.Error("expected allow=false for path outside workspace")
		}
	})
}

// TestReadFileTool_Judge_RequiredPathFailsClosed confirms read_file — whose
// `path` is required — still fails closed on an empty path (no behavior
// change from the optional-path fix).
func TestReadFileTool_Judge_RequiredPathFailsClosed(t *testing.T) {
	tool := NewReadFileTool()
	tmpDir := t.TempDir()
	ctx := tools.WithWorkspacePath(context.Background(), tmpDir)

	input, _ := json.Marshal(map[string]string{}) // no path

	judgeOut := tool.Judge(ctx, input)
	allow, reasoning := judgeOut.Allow, judgeOut.Reason
	if allow {
		t.Error("expected allow=false for empty required path")
	}
	if reasoning != "cannot determine target path" {
		t.Errorf("expected 'cannot determine target path', got: %s", reasoning)
	}
}

// TestListDirectoryTool_Judge_RequiredPathFailsClosed confirms list_directory
// — whose `path` is required — still fails closed on an empty path.
func TestListDirectoryTool_Judge_RequiredPathFailsClosed(t *testing.T) {
	tool := NewListDirectoryTool()
	tmpDir := t.TempDir()
	ctx := tools.WithWorkspacePath(context.Background(), tmpDir)

	input, _ := json.Marshal(map[string]string{}) // no path

	judgeOut := tool.Judge(ctx, input)
	allow, reasoning := judgeOut.Allow, judgeOut.Reason
	if allow {
		t.Error("expected allow=false for empty required path")
	}
	if reasoning != "cannot determine target path" {
		t.Errorf("expected 'cannot determine target path', got: %s", reasoning)
	}
}
