package memory

import (
	"testing"

	"github.com/v0lka/sp4rk/llm"
)

func TestContextWindow_BuildPrompt_WithContentBlocks(t *testing.T) {
	tracker := llm.NewContextTokenTracker(llm.NewSimpleTokenCounter())

	t.Run("SetTaskWithBlocks emits Message with ContentBlocks and Content", func(t *testing.T) {
		cw := NewContextWindow(ContextWindowConfig{
			SystemPrompt: "You are helpful.",
			ModelMeta:    testModelMeta(128000),
			Tracker:      tracker,
			Thresholds:   testThresholds(),
		})
		blocks := []llm.ContentBlock{
			{Type: "text", Text: "Analyze this image"},
			{Type: "image", MediaType: "image/png", ImageB64: "iVBOR"},
		}
		cw.SetTaskWithBlocks("Analyze this image", blocks)

		msgs := cw.BuildPrompt()
		// Find the user message carrying blocks (after system message)
		var userMsg *llm.Message
		for i := range msgs {
			if msgs[i].Role == "user" && len(msgs[i].ContentBlocks) > 0 {
				userMsg = &msgs[i]
				break
			}
		}
		if userMsg == nil {
			t.Fatal("BuildPrompt after SetTaskWithBlocks: expected a user message with ContentBlocks, got none")
		}
		if userMsg.Content != "Analyze this image" {
			t.Errorf("BuildPrompt user message Content = %q, want %q", userMsg.Content, "Analyze this image")
		}
		if len(userMsg.ContentBlocks) != 2 {
			t.Fatalf("BuildPrompt user message ContentBlocks len = %d, want 2", len(userMsg.ContentBlocks))
		}
		if userMsg.ContentBlocks[0].Type != "text" || userMsg.ContentBlocks[0].Text != "Analyze this image" {
			t.Errorf("BuildPrompt user message block 0 = %+v, want text 'Analyze this image'", userMsg.ContentBlocks[0])
		}
		if userMsg.ContentBlocks[1].Type != "image" {
			t.Errorf("BuildPrompt user message block 1 = %+v, want image", userMsg.ContentBlocks[1])
		}
	})

	t.Run("SetTaskWithBlocks with empty slice falls back to text-only path", func(t *testing.T) {
		cw := NewContextWindow(ContextWindowConfig{
			SystemPrompt: "You are helpful.",
			ModelMeta:    testModelMeta(128000),
			Tracker:      tracker,
			Thresholds:   testThresholds(),
		})
		cw.SetTaskWithBlocks("plain task", []llm.ContentBlock{})

		msgs := cw.BuildPrompt()
		var userMsg *llm.Message
		for i := range msgs {
			if msgs[i].Role == "user" {
				userMsg = &msgs[i]
				break
			}
		}
		if userMsg == nil {
			t.Fatal("BuildPrompt: expected a user message, got none")
		}
		if userMsg.Content != "plain task" {
			t.Errorf("BuildPrompt user message Content = %q, want %q", userMsg.Content, "plain task")
		}
		if len(userMsg.ContentBlocks) != 0 {
			t.Errorf("BuildPrompt user message ContentBlocks len = %d, want 0 (empty slice = legacy path)", len(userMsg.ContentBlocks))
		}
	})

	t.Run("SetTaskWithBlocks with nil blocks falls back to text-only path", func(t *testing.T) {
		cw := NewContextWindow(ContextWindowConfig{
			SystemPrompt: "You are helpful.",
			ModelMeta:    testModelMeta(128000),
			Tracker:      tracker,
			Thresholds:   testThresholds(),
		})
		cw.SetTaskWithBlocks("plain task", nil)

		msgs := cw.BuildPrompt()
		var userMsg *llm.Message
		for i := range msgs {
			if msgs[i].Role == "user" {
				userMsg = &msgs[i]
				break
			}
		}
		if userMsg == nil {
			t.Fatal("BuildPrompt: expected a user message, got none")
		}
		if userMsg.Content != "plain task" {
			t.Errorf("BuildPrompt user message Content = %q, want %q", userMsg.Content, "plain task")
		}
		if len(userMsg.ContentBlocks) != 0 {
			t.Errorf("BuildPrompt user message ContentBlocks len = %d, want 0 (nil = legacy path)", len(userMsg.ContentBlocks))
		}
	})

	t.Run("image-only blocks carry Content as fallback text", func(t *testing.T) {
		cw := NewContextWindow(ContextWindowConfig{
			SystemPrompt: "You are helpful.",
			ModelMeta:    testModelMeta(128000),
			Tracker:      tracker,
			Thresholds:   testThresholds(),
		})
		blocks := []llm.ContentBlock{
			{Type: "image", MediaType: "image/png", ImageB64: "iVBOR"},
		}
		cw.SetTaskWithBlocks("Describe this screenshot", blocks)

		msgs := cw.BuildPrompt()
		var userMsg *llm.Message
		for i := range msgs {
			if msgs[i].Role == "user" && len(msgs[i].ContentBlocks) > 0 {
				userMsg = &msgs[i]
				break
			}
		}
		if userMsg == nil {
			t.Fatal("BuildPrompt: expected a user message with ContentBlocks, got none")
		}
		// Content is carried alongside blocks; NormalizeContentBlocks (called by
		// providers) will prepend it as a text block since blocks have no text.
		if userMsg.Content != "Describe this screenshot" {
			t.Errorf("BuildPrompt user message Content = %q, want %q (fallback text)", userMsg.Content, "Describe this screenshot")
		}
	})
}

func TestUserMessageText_ContentBlocks(t *testing.T) {
	tests := []struct {
		name string
		msg  llm.Message
		want string
	}{
		{
			name: "nil blocks uses Content",
			msg:  llm.Message{Role: "user", Content: "hello"},
			want: "hello",
		},
		{
			name: "empty slice uses Content",
			msg:  llm.Message{Role: "user", Content: "hello", ContentBlocks: []llm.ContentBlock{}},
			want: "hello",
		},
		{
			name: "text block uses block text",
			msg: llm.Message{Role: "user", Content: "ignored", ContentBlocks: []llm.ContentBlock{
				{Type: "text", Text: "block text"},
			}},
			want: "block text",
		},
		{
			name: "image block replaced with placeholder",
			msg: llm.Message{Role: "user", Content: "", ContentBlocks: []llm.ContentBlock{
				{Type: "image", MediaType: "image/png", ImageB64: "abc"},
			}},
			want: "[image attached]",
		},
		{
			name: "image-only blocks with Content prepends text fallback",
			msg: llm.Message{Role: "user", Content: "Describe", ContentBlocks: []llm.ContentBlock{
				{Type: "image", MediaType: "image/png", ImageB64: "abc"},
			}},
			want: "Describe[image attached]",
		},
		{
			name: "mixed text+image concatenates",
			msg: llm.Message{Role: "user", Content: "ignored", ContentBlocks: []llm.ContentBlock{
				{Type: "text", Text: "look"},
				{Type: "image", MediaType: "image/png", ImageB64: "abc"},
			}},
			want: "look[image attached]",
		},
		{
			name: "unknown block type skipped",
			msg: llm.Message{Role: "user", Content: "ignored", ContentBlocks: []llm.ContentBlock{
				{Type: "text", Text: "keep"},
				{Type: "audio", Text: "drop"},
			}},
			want: "keep",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := userMessageText(tt.msg)
			if got != tt.want {
				t.Errorf("userMessageText(%+v) = %q, want %q", tt.msg, got, tt.want)
			}
		})
	}
}

// TestContextWindow_SetTask_ClearsContentBlocks verifies that SetTask reverts
// to the text-only prompt path by clearing any content blocks set by a prior
// SetTaskWithBlocks call. Without this, BuildPrompt would keep emitting the
// stale blocks alongside the new task text on a reused ContextWindow.
func TestContextWindow_SetTask_ClearsContentBlocks(t *testing.T) {
	tracker := llm.NewContextTokenTracker(llm.NewSimpleTokenCounter())
	cw := NewContextWindow(ContextWindowConfig{
		SystemPrompt: "You are helpful.",
		ModelMeta:    testModelMeta(128000),
		Tracker:      tracker,
		Thresholds:   testThresholds(),
	})

	// First turn: set task with image blocks.
	blocks := []llm.ContentBlock{
		{Type: "image", MediaType: "image/png", ImageB64: "iVBOR"},
	}
	cw.SetTaskWithBlocks("Analyze this screenshot", blocks)

	msgs := cw.BuildPrompt()
	var hasBlocks bool
	for _, m := range msgs {
		if m.Role == "user" && len(m.ContentBlocks) > 0 {
			hasBlocks = true
		}
	}
	if !hasBlocks {
		t.Fatal("after SetTaskWithBlocks: expected a user message with ContentBlocks")
	}

	// Second turn: revert to text-only via SetTask.
	cw.SetTask("implement variant a")

	msgs = cw.BuildPrompt()
	for _, m := range msgs {
		if m.Role == "user" {
			if len(m.ContentBlocks) > 0 {
				t.Errorf("after SetTask: user message still carries %d ContentBlocks, want 0 (stale blocks not cleared)", len(m.ContentBlocks))
			}
			if m.Content != "implement variant a" {
				t.Errorf("after SetTask: user message Content = %q, want %q", m.Content, "implement variant a")
			}
		}
	}
}
