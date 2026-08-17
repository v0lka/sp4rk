package builtins

import (
	"context"
	"encoding/json"
	"strings"

	"github.com/v0lka/sp4rk/agent"
	"github.com/v0lka/sp4rk/tools"
)

const toolReadAttachmentDescription = `Purpose: read the markdown content of a file the user attached to the conversation, by its ID.
Use when: the user's message lists attachments (each with an ID) and their content matters for the task — converted documents (e.g. PDFs, spreadsheets) are returned as readable markdown.
Inputs: attachment_id (from the attachment list in the user message).
Outputs: the attachment's converted markdown content.
Example: attachment_id from the user's latest message.
Anti-example: not for workspace files (read_file with a path); the ID is not a file path — guessing one fails.`

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
		ToolGroup:       tools.GroupSystem,
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
		// ASI01: converted markdown (PDFs, docs, spreadsheets) is external
		// content that may carry prompt-injection payloads. Treat as untrusted.
		Untrusted: true,
		Policy:    tools.PolicyAlwaysAllow,
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
