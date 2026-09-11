package orchestration

import (
	"errors"
	"sort"
	"strings"

	"github.com/v0lka/sp4rk/agent"
)

// pausedCheckpointOutput is the marker returned for a step whose recorded
// result is a cooperative-pause checkpoint rather than a terminal outcome.
// Such a step has no final output yet; hosts resume it from its checkpointed
// trajectory. Surfacing the marker (instead of hiding the step) keeps the
// step-output tools honest: an invisible step reads as vanished/failed.
const pausedCheckpointOutput = "[paused checkpoint — no final output yet]"

// isPausedStepErr reports whether a step-result error is the cooperative-pause
// sentinel. It matches the in-memory sentinel (errors.Is) plus the
// string-reconstructed forms a persisting host produces on restore: the exact
// sentinel message, or the sentinel as the TRAILING segment of a "%w"-style
// wrap/prefix (e.g. "step 3: executor paused at step boundary") — c0wrk's
// TaskPersistence/LoadTaskState rebuilds the error via errors.New(ErrorText).
// The match is anchored to the END of the message, deliberately NOT an
// arbitrary substring, so a genuinely failed step whose error text merely
// mentions the sentence mid-way is never reclassified as a paused checkpoint.
func isPausedStepErr(err error) bool {
	if err == nil {
		return false
	}
	return errors.Is(err, agent.ErrPaused) ||
		strings.HasSuffix(err.Error(), agent.ErrPaused.Error())
}

// blackboardStepOutputStore adapts a Blackboard to the agent.StepOutputStore interface.
// Steps that completed without error are exposed with their full output; steps
// paused at a cooperative checkpoint are exposed with an explicit paused
// marker; failed steps (any other error) are excluded, matching the
// historical behaviour where outputs were stored only on success.
type blackboardStepOutputStore struct {
	bb Blackboard
}

// NewStepOutputStore wraps a Blackboard as an agent.StepOutputStore.
func NewStepOutputStore(bb Blackboard) agent.StepOutputStore {
	return &blackboardStepOutputStore{bb: bb}
}

func (s *blackboardStepOutputStore) GetStepOutput(stepID string) (string, bool) {
	sr, ok := s.bb.GetStepResult(stepID)
	if !ok || (sr.FullOutput == "" && !isPausedStepErr(sr.Error)) {
		return "", false
	}
	if sr.Error != nil {
		if isPausedStepErr(sr.Error) {
			return pausedCheckpointOutput, true
		}
		return "", false
	}
	return sr.FullOutput, true
}

func (s *blackboardStepOutputStore) ListStepOutputs() []agent.StepOutputEntry {
	all := s.bb.GetAllStepResults()
	entries := make([]agent.StepOutputEntry, 0, len(all))
	for stepID, sr := range all {
		switch {
		case sr.Error == nil && sr.FullOutput != "":
			entries = append(entries, agent.StepOutputEntry{
				StepID:     stepID,
				FullOutput: sr.FullOutput,
			})
		case isPausedStepErr(sr.Error):
			entries = append(entries, agent.StepOutputEntry{
				StepID:     stepID,
				FullOutput: pausedCheckpointOutput,
			})
		}
	}
	// Deterministic order sorted by step ID.
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].StepID < entries[j].StepID
	})
	return entries
}
