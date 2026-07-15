package builtins

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/v0lka/sp4rk/agent"
)

// mockAttachmentStore implements agent.AttachmentStore for testing.
type mockAttachmentStore struct {
	attachments map[string]agent.AttachmentEntry
}

func (m *mockAttachmentStore) GetAttachments() []agent.AttachmentEntry {
	if len(m.attachments) == 0 {
		return nil
	}
	out := make([]agent.AttachmentEntry, 0, len(m.attachments))
	for _, a := range m.attachments {
		out = append(out, a)
	}
	return out
}

func (m *mockAttachmentStore) GetAttachment(id string) (agent.AttachmentEntry, bool) {
	a, ok := m.attachments[id]
	return a, ok
}

func ctxWithAttachmentStore(store agent.AttachmentStore) context.Context {
	return agent.WithAttachmentStore(context.Background(), store)
}

// ---------------------------------------------------------------------------
// read_attachment
// ---------------------------------------------------------------------------

func TestReadAttachmentTool_ReturnsMarkdown(t *testing.T) {
	store := &mockAttachmentStore{
		attachments: map[string]agent.AttachmentEntry{
			"att_1": {
				ID:              "att_1",
				OriginalName:    "report.pdf",
				Format:          "pdf",
				SizeBytes:       4096,
				MarkdownContent: "# Report\n\nSome converted markdown content.",
				AttachedAt:      time.Now(),
			},
		},
	}
	ctx := ctxWithAttachmentStore(store)

	tool := NewReadAttachmentTool()
	input, _ := json.Marshal(ReadAttachmentInput{AttachmentID: "att_1"})

	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Fatalf("unexpected error result: %s", result.Content)
	}
	if !strings.Contains(result.Content, "# Report") {
		t.Errorf("expected markdown content in result, got %q", result.Content)
	}
}

func TestReadAttachmentTool_NotFound(t *testing.T) {
	store := &mockAttachmentStore{
		attachments: map[string]agent.AttachmentEntry{
			"att_1": {ID: "att_1", MarkdownContent: "x"},
		},
	}
	ctx := ctxWithAttachmentStore(store)

	tool := NewReadAttachmentTool()
	input, _ := json.Marshal(ReadAttachmentInput{AttachmentID: "nonexistent"})

	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Fatal("expected error result for unknown attachment id")
	}
	if !strings.Contains(result.Content, "not found") {
		t.Errorf("expected 'not found' message, got %q", result.Content)
	}
	if !strings.Contains(result.Content, "nonexistent") {
		t.Errorf("expected error to mention the requested id, got %q", result.Content)
	}
}

func TestReadAttachmentTool_NoStore(t *testing.T) {
	tool := NewReadAttachmentTool()
	input, _ := json.Marshal(ReadAttachmentInput{AttachmentID: "att_1"})

	result, err := tool.Execute(context.Background(), input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Fatal("expected error result when attachment store not available")
	}
	if !strings.Contains(result.Content, "not available") {
		t.Errorf("expected 'not available' message, got %q", result.Content)
	}
}

func TestReadAttachmentTool_EmptyID(t *testing.T) {
	store := &mockAttachmentStore{}
	ctx := ctxWithAttachmentStore(store)

	tool := NewReadAttachmentTool()
	input, _ := json.Marshal(ReadAttachmentInput{AttachmentID: ""})

	result, err := tool.Execute(ctx, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Fatal("expected error result for empty attachment_id")
	}
	if !strings.Contains(result.Content, "required") {
		t.Errorf("expected 'required' validation message, got %q", result.Content)
	}
}

func TestReadAttachmentTool_InvalidJSON(t *testing.T) {
	store := &mockAttachmentStore{}
	ctx := ctxWithAttachmentStore(store)

	tool := NewReadAttachmentTool()

	result, err := tool.Execute(ctx, []byte(`{invalid`))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Fatal("expected error result for invalid JSON")
	}
}
