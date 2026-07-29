package llm

import (
	"testing"

	"github.com/liushuangls/go-anthropic/v2"
)

func TestAnthropicProvider_ConvertMessage_ContentBlocks(t *testing.T) {
	p, _ := NewAnthropicProvider(AnthropicProviderConfig{APIKey: "test-key"})

	tests := []struct {
		name          string
		msg           Message
		wantBlocks    int
		wantFirstText string
		wantImage     bool
	}{
		{
			name: "text-only blocks render one text block",
			msg: Message{
				Role: "user",
				ContentBlocks: []ContentBlock{
					{Type: "text", Text: "hello world"},
				},
			},
			wantBlocks:    1,
			wantFirstText: "hello world",
		},
		{
			name: "image-only blocks with Content prepends text fallback",
			msg: Message{
				Role:    "user",
				Content: "Analyze this screenshot",
				ContentBlocks: []ContentBlock{
					{Type: "image", MediaType: "image/png", ImageB64: "iVBOR"},
				},
			},
			wantBlocks:    2,
			wantFirstText: "Analyze this screenshot",
			wantImage:     true,
		},
		{
			name: "mixed text+image blocks render both",
			msg: Message{
				Role: "user",
				ContentBlocks: []ContentBlock{
					{Type: "text", Text: "describe this"},
					{Type: "image", MediaType: "image/jpeg", ImageB64: "abc123"},
				},
			},
			wantBlocks:    2,
			wantFirstText: "describe this",
			wantImage:     true,
		},
		{
			name: "empty slice falls back to Content path",
			msg: Message{
				Role:          "user",
				Content:       "plain text",
				ContentBlocks: []ContentBlock{},
			},
			wantBlocks:    1,
			wantFirstText: "plain text",
		},
		{
			name: "unknown block type skipped",
			msg: Message{
				Role: "user",
				ContentBlocks: []ContentBlock{
					{Type: "text", Text: "keep me"},
					{Type: "audio", Text: "drop me"},
				},
			},
			wantBlocks:    1,
			wantFirstText: "keep me",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := p.convertMessage(tt.msg)
			if err != nil {
				t.Fatalf("convertMessage(%+v) unexpected error: %v", tt.msg, err)
			}
			if result.Role != anthropic.RoleUser {
				t.Errorf("convertMessage(%+v) role = %q, want %q", tt.msg, result.Role, anthropic.RoleUser)
			}
			if len(result.Content) != tt.wantBlocks {
				t.Fatalf("convertMessage(%+v) got %d content blocks, want %d", tt.msg, len(result.Content), tt.wantBlocks)
			}
			// First block should be text
			if result.Content[0].Type != anthropic.MessagesContentTypeText {
				t.Errorf("convertMessage(%+v) first block type = %q, want text", tt.msg, result.Content[0].Type)
			}
			if result.Content[0].Text == nil || *result.Content[0].Text != tt.wantFirstText {
				t.Errorf("convertMessage(%+v) first block text = %v, want %q", tt.msg, result.Content[0].Text, tt.wantFirstText)
			}
			if tt.wantImage {
				foundImage := false
				for _, c := range result.Content {
					if c.Type == anthropic.MessagesContentTypeImage {
						foundImage = true
						break
					}
				}
				if !foundImage {
					t.Errorf("convertMessage(%+v) expected an image block among %d blocks", tt.msg, len(result.Content))
				}
			}
		})
	}
}

func TestAnthropicProvider_BuildRequest_ValidateImageBlocks(t *testing.T) {
	p, _ := NewAnthropicProvider(AnthropicProviderConfig{APIKey: "test-key"})

	t.Run("missing media_type returns error", func(t *testing.T) {
		req := ChatRequest{
			Model: "claude-3",
			Messages: []Message{
				{Role: "user", Content: "look", ContentBlocks: []ContentBlock{
					{Type: "image", ImageB64: "abc"},
				}},
			},
		}
		_, err := p.buildRequest(req)
		if err == nil {
			t.Error("buildRequest with missing media_type expected error, got nil")
		}
	})

	t.Run("valid image blocks pass", func(t *testing.T) {
		req := ChatRequest{
			Model: "claude-3",
			Messages: []Message{
				{Role: "user", Content: "look", ContentBlocks: []ContentBlock{
					{Type: "image", MediaType: "image/png", ImageB64: "abc"},
				}},
			},
		}
		_, err := p.buildRequest(req)
		if err != nil {
			t.Errorf("buildRequest with valid image blocks unexpected error: %v", err)
		}
	})
}

func TestOpenAIProvider_ConvertRequestMessage_ContentBlocks(t *testing.T) {
	p, _ := NewOpenAIProvider(OpenAIProviderConfig{Name: "openai", APIKey: "k"})

	tests := []struct {
		name      string
		msg       Message
		wantParts int
		wantText  string
		wantImage bool
	}{
		{
			name: "text-only blocks render one text part",
			msg: Message{
				Role: "user",
				ContentBlocks: []ContentBlock{
					{Type: "text", Text: "hello"},
				},
			},
			wantParts: 1,
			wantText:  "hello",
		},
		{
			name: "image-only blocks with Content prepends text fallback",
			msg: Message{
				Role:    "user",
				Content: "Describe this",
				ContentBlocks: []ContentBlock{
					{Type: "image", MediaType: "image/png", ImageB64: "abc"},
				},
			},
			wantParts: 2,
			wantText:  "Describe this",
			wantImage: true,
		},
		{
			name: "mixed text+image blocks render both",
			msg: Message{
				Role: "user",
				ContentBlocks: []ContentBlock{
					{Type: "text", Text: "look here"},
					{Type: "image", MediaType: "image/jpeg", ImageB64: "xyz"},
				},
			},
			wantParts: 2,
			wantText:  "look here",
			wantImage: true,
		},
		{
			name: "empty slice falls back to Content string",
			msg: Message{
				Role:          "user",
				Content:       "plain",
				ContentBlocks: []ContentBlock{},
			},
			wantParts: 0, // OfString path, not parts
			wantText:  "plain",
		},
		{
			name: "unknown block type skipped",
			msg: Message{
				Role: "user",
				ContentBlocks: []ContentBlock{
					{Type: "text", Text: "keep"},
					{Type: "audio", Text: "drop"},
				},
			},
			wantParts: 1,
			wantText:  "keep",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := p.convertRequestMessage(tt.msg)
			if result.OfUser == nil {
				t.Fatalf("convertRequestMessage(%+v) expected user message, got nil", tt.msg)
			}
			if tt.wantParts == 0 {
				// Legacy OfString path
				if result.OfUser.Content.OfString.Value != tt.wantText {
					t.Errorf("convertRequestMessage(%+v) content = %q, want %q", tt.msg, result.OfUser.Content.OfString.Value, tt.wantText)
				}
				return
			}
			parts := result.OfUser.Content.OfArrayOfContentParts
			if len(parts) != tt.wantParts {
				t.Fatalf("convertRequestMessage(%+v) got %d parts, want %d", tt.msg, len(parts), tt.wantParts)
			}
			if parts[0].OfText == nil || parts[0].OfText.Text != tt.wantText {
				t.Errorf("convertRequestMessage(%+v) first part text = %v, want %q", tt.msg, parts[0].OfText, tt.wantText)
			}
			if tt.wantImage {
				foundImage := false
				for _, part := range parts {
					if part.OfImageURL != nil {
						foundImage = true
						break
					}
				}
				if !foundImage {
					t.Errorf("convertRequestMessage(%+v) expected an image part among %d parts", tt.msg, len(parts))
				}
			}
		})
	}
}

func TestConvertToResponsesInput_ContentBlocks(t *testing.T) {
	tests := []struct {
		name           string
		messages       []Message
		wantItemCount  int
		wantContentLen int
		wantFirstText  string
		wantImage      bool
	}{
		{
			name: "text-only blocks render content list with one text part",
			messages: []Message{
				{Role: "user", ContentBlocks: []ContentBlock{
					{Type: "text", Text: "hello"},
				}},
			},
			wantItemCount:  1,
			wantContentLen: 1,
			wantFirstText:  "hello",
		},
		{
			name: "image-only blocks with Content prepends text fallback",
			messages: []Message{
				{Role: "user", Content: "Analyze", ContentBlocks: []ContentBlock{
					{Type: "image", MediaType: "image/png", ImageB64: "abc"},
				}},
			},
			wantItemCount:  1,
			wantContentLen: 2,
			wantFirstText:  "Analyze",
			wantImage:      true,
		},
		{
			name: "mixed text+image blocks render both",
			messages: []Message{
				{Role: "user", ContentBlocks: []ContentBlock{
					{Type: "text", Text: "describe"},
					{Type: "image", MediaType: "image/png", ImageB64: "xyz"},
				}},
			},
			wantItemCount:  1,
			wantContentLen: 2,
			wantFirstText:  "describe",
			wantImage:      true,
		},
		{
			name: "empty slice falls back to OfString path",
			messages: []Message{
				{Role: "user", Content: "plain", ContentBlocks: []ContentBlock{}},
			},
			wantItemCount:  1,
			wantContentLen: 0, // OfString, not content list
			wantFirstText:  "plain",
		},
		{
			name: "unknown block type skipped",
			messages: []Message{
				{Role: "user", ContentBlocks: []ContentBlock{
					{Type: "text", Text: "keep"},
					{Type: "audio", Text: "drop"},
				}},
			},
			wantItemCount:  1,
			wantContentLen: 1,
			wantFirstText:  "keep",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			items := convertToResponsesInput(tt.messages, nil)
			if len(items) != tt.wantItemCount {
				t.Fatalf("convertToResponsesInput got %d items, want %d", len(items), tt.wantItemCount)
			}
			if items[0].OfMessage == nil {
				t.Fatal("expected OfMessage to be set")
			}
			if tt.wantContentLen == 0 {
				// OfString path
				if !items[0].OfMessage.Content.OfString.Valid() {
					t.Fatal("expected OfString to be set")
				}
				if items[0].OfMessage.Content.OfString.Value != tt.wantFirstText {
					t.Errorf("content = %q, want %q", items[0].OfMessage.Content.OfString.Value, tt.wantFirstText)
				}
				return
			}
			contentList := items[0].OfMessage.Content.OfInputItemContentList
			if len(contentList) != tt.wantContentLen {
				t.Fatalf("got %d content parts, want %d", len(contentList), tt.wantContentLen)
			}
			if contentList[0].OfInputText == nil || contentList[0].OfInputText.Text != tt.wantFirstText {
				t.Errorf("first content part text = %v, want %q", contentList[0].OfInputText, tt.wantFirstText)
			}
			if tt.wantImage {
				foundImage := false
				for _, c := range contentList {
					if c.OfInputImage != nil {
						foundImage = true
						break
					}
				}
				if !foundImage {
					t.Errorf("expected an image part among %d content parts", len(contentList))
				}
			}
		})
	}
}
