package tools

import (
	"context"
	"encoding/json"
	"testing"
)

// contentBackedMockTool is a mock Tool that optionally implements
// ContentBackedReader for testing cache-strategy resolution.
type contentBackedMockTool struct {
	mockTool
	contentBacked bool
}

func (m *contentBackedMockTool) IsContentBacked(_ context.Context, _ json.RawMessage) bool {
	return m.contentBacked
}

func TestCacheStrategy_UnknownTool(t *testing.T) {
	reg := NewToolRegistry()
	mode := reg.CacheStrategy(context.Background(), "nope", json.RawMessage(`{}`))
	if mode != CacheModeDefault {
		t.Errorf("unknown tool: got %v, want CacheModeDefault", mode)
	}
}

func TestCacheStrategy_DefaultTool(t *testing.T) {
	reg := NewToolRegistry()
	reg.Register(newMockTool("plain_read", "plain read"))
	mode := reg.CacheStrategy(context.Background(), "plain_read", json.RawMessage(`{}`))
	if mode != CacheModeDefault {
		t.Errorf("plain tool: got %v, want CacheModeDefault", mode)
	}
}

func TestCacheStrategy_ContentBackedTrue(t *testing.T) {
	reg := NewToolRegistry()
	reg.Register(&contentBackedMockTool{
		mockTool:      mockTool{BaseTool: BaseTool{ToolName: "doc_read", ToolDescription: "doc read", Schema: json.RawMessage(`{"type":"object"}`), Policy: PolicyAlwaysAllow}},
		contentBacked: true,
	})
	mode := reg.CacheStrategy(context.Background(), "doc_read", json.RawMessage(`{"path":"doc.pdf"}`))
	if mode != CacheModeContentBacked {
		t.Errorf("content-backed tool: got %v, want CacheModeContentBacked", mode)
	}
}

func TestCacheStrategy_ContentBackedFalse(t *testing.T) {
	reg := NewToolRegistry()
	reg.Register(&contentBackedMockTool{
		mockTool:      mockTool{BaseTool: BaseTool{ToolName: "txt_read", ToolDescription: "txt read", Schema: json.RawMessage(`{"type":"object"}`), Policy: PolicyAlwaysAllow}},
		contentBacked: false,
	})
	mode := reg.CacheStrategy(context.Background(), "txt_read", json.RawMessage(`{"path":"f.txt"}`))
	if mode != CacheModeDefault {
		t.Errorf("content-backed=false tool: got %v, want CacheModeDefault", mode)
	}
}
