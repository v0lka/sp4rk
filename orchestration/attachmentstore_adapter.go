package orchestration

import "github.com/v0lka/sp4rk/agent"

// blackboardAttachmentStore adapts a Blackboard to the agent.AttachmentStore
// interface.
type blackboardAttachmentStore struct {
	bb Blackboard
}

// NewAttachmentStore wraps a Blackboard as an agent.AttachmentStore.
func NewAttachmentStore(bb Blackboard) agent.AttachmentStore {
	return &blackboardAttachmentStore{bb: bb}
}

// GetAttachments returns all attachments as AttachmentEntry values.
// Delegates to the blackboard, which already returns defensive copies.
func (s *blackboardAttachmentStore) GetAttachments() []agent.AttachmentEntry {
	attachments := s.bb.GetAttachments()
	entries := make([]agent.AttachmentEntry, len(attachments))
	for i, a := range attachments {
		entries[i] = mapAttachmentToEntry(a)
	}
	return entries
}

// GetAttachment returns the attachment with the given ID as an AttachmentEntry.
// Delegates to the blackboard, which already returns a defensive copy.
func (s *blackboardAttachmentStore) GetAttachment(id string) (agent.AttachmentEntry, bool) {
	a, ok := s.bb.GetAttachment(id)
	if !ok {
		return agent.AttachmentEntry{}, false
	}
	return mapAttachmentToEntry(a), true
}

// mapAttachmentToEntry maps an orchestration.Attachment to an
// agent.AttachmentEntry.
func mapAttachmentToEntry(a Attachment) agent.AttachmentEntry {
	return agent.AttachmentEntry{
		ID:              a.ID,
		OriginalName:    a.OriginalName,
		Format:          a.Format,
		SizeBytes:       a.SizeBytes,
		MarkdownContent: a.MarkdownContent,
		AttachedAt:      a.AttachedAt,
	}
}
