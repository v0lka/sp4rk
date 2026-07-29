package llm

import (
	"bytes"
	"log/slog"
	"strings"
	"testing"
)

func TestNormalizeContentBlocks(t *testing.T) {
	tests := []struct {
		name string
		msg  Message
		want []ContentBlock
	}{
		{
			name: "nil blocks returns nil",
			msg:  Message{Role: "user", Content: "hello"},
			want: nil,
		},
		{
			name: "empty slice returns nil",
			msg:  Message{Role: "user", Content: "hello", ContentBlocks: []ContentBlock{}},
			want: nil,
		},
		{
			name: "text-only blocks returned as-is",
			msg: Message{Role: "user", Content: "fallback", ContentBlocks: []ContentBlock{
				{Type: "text", Text: "block text"},
			}},
			want: []ContentBlock{
				{Type: "text", Text: "block text"},
			},
		},
		{
			name: "image-only blocks with Content prepends text fallback",
			msg: Message{Role: "user", Content: "Analyze this", ContentBlocks: []ContentBlock{
				{Type: "image", MediaType: "image/png", ImageB64: "abc"},
			}},
			want: []ContentBlock{
				{Type: "text", Text: "Analyze this"},
				{Type: "image", MediaType: "image/png", ImageB64: "abc"},
			},
		},
		{
			name: "image-only blocks with empty Content returns blocks as-is",
			msg: Message{Role: "user", Content: "", ContentBlocks: []ContentBlock{
				{Type: "image", MediaType: "image/png", ImageB64: "abc"},
			}},
			want: []ContentBlock{
				{Type: "image", MediaType: "image/png", ImageB64: "abc"},
			},
		},
		{
			name: "mixed text+image blocks returned as-is (has text)",
			msg: Message{Role: "user", Content: "fallback", ContentBlocks: []ContentBlock{
				{Type: "text", Text: "describe"},
				{Type: "image", MediaType: "image/png", ImageB64: "abc"},
			}},
			want: []ContentBlock{
				{Type: "text", Text: "describe"},
				{Type: "image", MediaType: "image/png", ImageB64: "abc"},
			},
		},
		{
			name: "unknown-type-only blocks with Content prepends text fallback",
			msg: Message{Role: "user", Content: "instruction", ContentBlocks: []ContentBlock{
				{Type: "audio", Text: "ignored"},
			}},
			want: []ContentBlock{
				{Type: "text", Text: "instruction"},
				{Type: "audio", Text: "ignored"},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := NormalizeContentBlocks(tt.msg)
			if len(got) != len(tt.want) {
				t.Fatalf("NormalizeContentBlocks(%+v) returned %d blocks, want %d", tt.msg, len(got), len(tt.want))
			}
			for i := range tt.want {
				if got[i] != tt.want[i] {
					t.Errorf("NormalizeContentBlocks(%+v) block %d = %+v, want %+v", tt.msg, i, got[i], tt.want[i])
				}
			}
		})
	}
}

func TestValidateContentBlocks(t *testing.T) {
	tests := []struct {
		name    string
		blocks  []ContentBlock
		wantErr bool
	}{
		{
			name:    "nil blocks valid",
			blocks:  nil,
			wantErr: false,
		},
		{
			name:    "empty blocks valid",
			blocks:  []ContentBlock{},
			wantErr: false,
		},
		{
			name:    "text-only blocks valid",
			blocks:  []ContentBlock{{Type: "text", Text: "hi"}},
			wantErr: false,
		},
		{
			name:    "image block with media and data valid",
			blocks:  []ContentBlock{{Type: "image", MediaType: "image/png", ImageB64: "abc"}},
			wantErr: false,
		},
		{
			name:    "image block missing media_type invalid",
			blocks:  []ContentBlock{{Type: "image", ImageB64: "abc"}},
			wantErr: true,
		},
		{
			name:    "image block missing image_b64 invalid",
			blocks:  []ContentBlock{{Type: "image", MediaType: "image/png"}},
			wantErr: true,
		},
		{
			name: "mixed valid blocks valid",
			blocks: []ContentBlock{
				{Type: "text", Text: "look"},
				{Type: "image", MediaType: "image/jpeg", ImageB64: "xyz"},
			},
			wantErr: false,
		},
		{
			name: "second image block invalid",
			blocks: []ContentBlock{
				{Type: "image", MediaType: "image/png", ImageB64: "ok"},
				{Type: "image", MediaType: "image/png"},
			},
			wantErr: true,
		},
		{
			name:    "text block with empty text invalid",
			blocks:  []ContentBlock{{Type: "text", Text: ""}},
			wantErr: true,
		},
		{
			name: "mixed valid text and empty text invalid",
			blocks: []ContentBlock{
				{Type: "text", Text: "ok"},
				{Type: "text", Text: ""},
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateContentBlocks(tt.blocks)
			gotErr := err != nil
			if gotErr != tt.wantErr {
				t.Errorf("ValidateContentBlocks(%v) error = %v, want error presence = %t", tt.blocks, err, tt.wantErr)
			}
		})
	}
}

func TestCountContentTokens(t *testing.T) {
	counter := NewSimpleTokenCounter()

	tests := []struct {
		name string
		msg  Message
		want int
	}{
		{
			name: "nil blocks counts Content",
			msg:  Message{Role: "user", Content: "hello"},
			want: counter.Count("hello"),
		},
		{
			name: "empty slice counts Content (not zero)",
			msg:  Message{Role: "user", Content: "hello", ContentBlocks: []ContentBlock{}},
			want: counter.Count("hello"),
		},
		{
			name: "text block counts block text",
			msg: Message{Role: "user", Content: "ignored", ContentBlocks: []ContentBlock{
				{Type: "text", Text: "blocktext"},
			}},
			want: counter.Count("blocktext"),
		},
		{
			name: "image block counts estimatedTokensPerImage",
			msg: Message{Role: "user", Content: "", ContentBlocks: []ContentBlock{
				{Type: "image", MediaType: "image/png", ImageB64: "abc"},
			}},
			want: estimatedTokensPerImage,
		},
		{
			name: "image-only blocks with Content prepends text fallback (counts both)",
			msg: Message{Role: "user", Content: "Analyze this", ContentBlocks: []ContentBlock{
				{Type: "image", MediaType: "image/png", ImageB64: "abc"},
			}},
			want: counter.Count("Analyze this") + estimatedTokensPerImage,
		},
		{
			name: "mixed text+image counts both",
			msg: Message{Role: "user", Content: "ignored", ContentBlocks: []ContentBlock{
				{Type: "text", Text: "describe"},
				{Type: "image", MediaType: "image/png", ImageB64: "abc"},
			}},
			want: counter.Count("describe") + estimatedTokensPerImage,
		},
		{
			name: "unknown block type skipped (not counted as text)",
			msg: Message{Role: "user", Content: "ignored", ContentBlocks: []ContentBlock{
				{Type: "audio", Text: "audiodata"},
			}},
			want: counter.Count("ignored"), // Content prepended as text fallback
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := countContentTokens(counter.Count, tt.msg, counter.imageTokenEstimate)
			if got != tt.want {
				t.Errorf("countContentTokens(%+v) = %d, want %d", tt.msg, got, tt.want)
			}
		})
	}
}

// TestAnthropicProvider_BuildRequest_ImageOnlyNotSkipped verifies that
// buildRequest does not silently drop an image-only user message (empty
// Content, non-empty ContentBlocks). Before the fix, the skip-empty guard
// only checked Content, so such messages were dropped before convertMessage.
func TestAnthropicProvider_BuildRequest_ImageOnlyNotSkipped(t *testing.T) {
	p, _ := NewAnthropicProvider(AnthropicProviderConfig{APIKey: "test-key"})
	req := ChatRequest{
		Model: "claude-3",
		Messages: []Message{
			{Role: "user", Content: "", ContentBlocks: []ContentBlock{
				{Type: "image", MediaType: "image/png", ImageB64: "iVBOR"},
			}},
		},
	}
	anthropicReq, err := p.buildRequest(req)
	if err != nil {
		t.Fatalf("buildRequest unexpected error: %v", err)
	}
	if len(anthropicReq.Messages) != 1 {
		t.Fatalf("buildRequest produced %d messages, want 1 (image-only message must not be skipped)",
			len(anthropicReq.Messages))
	}
	if len(anthropicReq.Messages[0].Content) == 0 {
		t.Error("buildRequest produced an empty-content message for image-only input")
	}
}

// TestNewTokenCounter_AnthropicImageEstimate verifies that an Anthropic-family
// token counter (created via NewTokenCounter("anthropic-api")) estimates image
// blocks at estimatedAnthropicTokensPerImage (85), not the conservative
// OpenAI-oriented default (765), so image-heavy Anthropic conversations do
// not over-count images ~9× and trigger premature context compaction.
func TestNewTokenCounter_AnthropicImageEstimate(t *testing.T) {
	counter, err := NewTokenCounter("anthropic-api")
	if err != nil {
		t.Fatalf("NewTokenCounter(\"anthropic-api\") unexpected error: %v", err)
	}
	msgs := []Message{
		{Role: "user", Content: "", ContentBlocks: []ContentBlock{
			{Type: "image", MediaType: "image/png", ImageB64: "abc"},
		}},
	}
	got := counter.CountMessages(msgs)
	// role "user" (1 token) + image (85) + framing (4) = 90.
	want := counter.Count("user") + estimatedAnthropicTokensPerImage + 4
	if got != want {
		t.Errorf("Anthropic counter CountMessages = %d, want %d (role + %d image + 4 framing)",
			got, want, estimatedAnthropicTokensPerImage)
	}
	// Confirm the default (OpenAI-oriented) counter uses the higher estimate.
	defaultCounter := NewSimpleTokenCounter()
	gotDefault := defaultCounter.CountMessages(msgs)
	wantDefault := defaultCounter.Count("user") + estimatedTokensPerImage + 4
	if gotDefault != wantDefault {
		t.Errorf("default counter CountMessages = %d, want %d (role + %d image + 4 framing)",
			gotDefault, wantDefault, estimatedTokensPerImage)
	}
	if gotDefault <= got {
		t.Errorf("default counter (%d) should exceed Anthropic counter (%d) since 765 > 85",
			gotDefault, got)
	}
}

// TestUnknownBlockType_DebugLog verifies that providers emit a debug-level log
// when an unknown content block type is encountered, so misconfigured callers
// can diagnose silently dropped content. The default slog logger is replaced
// with a capturing text handler for the duration of the test.
func TestUnknownBlockType_DebugLog(t *testing.T) {
	var buf bytes.Buffer
	handler := slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug})
	origLogger := slog.Default()
	slog.SetDefault(slog.New(handler))
	defer slog.SetDefault(origLogger)

	unknownMsg := Message{
		Role: "user",
		ContentBlocks: []ContentBlock{
			{Type: "text", Text: "keep"},
			{Type: "audio", Text: "drop"},
		},
	}

	t.Run("anthropic convertMessage", func(t *testing.T) {
		buf.Reset()
		p, _ := NewAnthropicProvider(AnthropicProviderConfig{APIKey: "k"})
		if _, err := p.convertMessage(unknownMsg); err != nil {
			t.Fatalf("convertMessage unexpected error: %v", err)
		}
		if !strings.Contains(buf.String(), "skipping unknown content block type") {
			t.Errorf("expected debug log for unknown block type, got: %s", buf.String())
		}
		if !strings.Contains(buf.String(), "audio") {
			t.Errorf("expected log to mention block type 'audio', got: %s", buf.String())
		}
	})

	t.Run("openai convertRequestMessage", func(t *testing.T) {
		buf.Reset()
		p, _ := NewOpenAIProvider(OpenAIProviderConfig{Name: "openai", APIKey: "k"})
		_ = p.convertRequestMessage(unknownMsg)
		if !strings.Contains(buf.String(), "skipping unknown content block type") {
			t.Errorf("expected debug log for unknown block type, got: %s", buf.String())
		}
	})

	t.Run("openai responses convertToResponsesInput", func(t *testing.T) {
		buf.Reset()
		_ = convertToResponsesInput([]Message{unknownMsg}, nil)
		if !strings.Contains(buf.String(), "skipping unknown content block type") {
			t.Errorf("expected debug log for unknown block type, got: %s", buf.String())
		}
	})
}
