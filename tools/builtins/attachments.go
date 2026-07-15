package builtins

import (
	"context"
	"encoding/json"
	"strings"

	"github.com/v0lka/sp4rk/agent"
	"github.com/v0lka/sp4rk/tools"
)

const toolReadAttachmentDescription = "Read the markdown content of a user-attached file by its ID. The attachment IDs are provided in the user message attachment list. Use this to inspect the converted markdown content of files the user attached to the conversation (e.g. PDFs, spreadsheets, images)."

// ---------------------------------------------------------------------------
// read_attachment
// ---------------------------------------------------------------------------

// ReadAttachmentTool reads the markdown content of a user-attached file from
// the AttachmentStore in context.
type ReadAttachmentTool struct {
	*tools.BaseTool
}

// NewReadAttachmentTool creates a new ReadAttachmentTool instance.
func NewReadAttachmentTool() *ReadAttachmentTool {
	return &ReadAttachmentTool{BaseTool: &tools.BaseTool{
		ToolName:        "read_attachment",
		ToolDescription: toolReadAttachmentDescription,
		Schema: json.RawMessage(`{
		"type": "object",
		"properties": {
			"attachment_id": {
				"type": "string",
				"description": "The ID of the attachment to read (provided in the user message attachment list)"
			}
		},
		"required": ["attachment_id"]
	}`),
		Policy: tools.PolicyAlwaysAllow,
	}}
}

// ReadAttachmentInput represents the input parameters for read_attachment.
type ReadAttachmentInput struct {
	AttachmentID string `json:"attachment_id"`
}

// Execute reads the attachment markdown content via the AttachmentStore from
// context.
func (t *ReadAttachmentTool) Execute(ctx context.Context, input json.RawMessage) (tools.ToolResult, error) {
	var params ReadAttachmentInput
	if err := json.Unmarshal(input, &params); err != nil {
		return tools.ParseInputError(err)
	}

	if strings.TrimSpace(params.AttachmentID) == "" {
		return tools.ToolResult{Content: "validation error: attachment_id is required", IsError: true}, nil
	}

	store := agent.AttachmentStoreFromContext(ctx)
	if store == nil {
		return tools.ErrorResult("Attachment store not available"), nil
	}

	att, ok := store.GetAttachment(params.AttachmentID)
	if !ok {
		return tools.ToolResult{
			Content: "attachment not found: no attachment with id " + params.AttachmentID + ". Check the attachment IDs provided in the user message.",
			IsError: true,
		}, nil
	}

	return tools.ToolResult{Content: att.MarkdownContent}, nil
}
