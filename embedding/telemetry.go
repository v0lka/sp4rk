package embedding

import (
	"sync"
	"time"
)

// TelemetryStage is a bounded label for an embedding pipeline stage.
type TelemetryStage string

const (
	// StageTokenization measures tokenizer encoding work.
	StageTokenization TelemetryStage = "tokenization"
	// StageONNXInference measures blocking ONNX session runs.
	StageONNXInference TelemetryStage = "onnx_inference"
	// StageSessionCreate measures successful ONNX session construction.
	StageSessionCreate TelemetryStage = "session_create"
)

// StageMetrics contains aggregate timing for one bounded pipeline stage.
type StageMetrics struct {
	Calls    int64
	Duration time.Duration
}

// TelemetrySnapshot is an aggregate, content-free view of embedding work.
// It contains no text, model paths, or caller-provided labels.
type TelemetrySnapshot struct {
	Stages    map[TelemetryStage]StageMetrics
	Texts     int64
	Tokens    int64
	MinTokens int
	MaxTokens int
	// InferenceCount counts ONNX inference ATTEMPTS, including ones that
	// failed — it always equals Stages[StageONNXInference].Calls, so a
	// derived average inference latency (Duration / InferenceCount) stays
	// meaningful after transient failures.
	InferenceCount int64
	SessionCount   int64
	BatchRows      int64
	BatchCapacity  int64
}

// AverageTokens returns the mean non-padding token count per text.
func (s TelemetrySnapshot) AverageTokens() float64 {
	if s.Texts == 0 {
		return 0
	}
	return float64(s.Tokens) / float64(s.Texts)
}

// BatchFill returns the fraction of ONNX batch capacity occupied by real rows.
func (s TelemetrySnapshot) BatchFill() float64 {
	if s.BatchCapacity == 0 {
		return 0
	}
	return float64(s.BatchRows) / float64(s.BatchCapacity)
}

// Telemetry aggregates low-cardinality, content-free embedding metrics. It is
// safe for concurrent use. A nil *Telemetry is intentionally a no-op.
type Telemetry struct {
	mu            sync.Mutex
	stages        map[TelemetryStage]StageMetrics
	texts         int64
	tokens        int64
	minTokens     int
	maxTokens     int
	inferences    int64
	sessions      int64
	batchRows     int64
	batchCapacity int64
}

func (t *Telemetry) observe(stage TelemetryStage, elapsed time.Duration) {
	if t == nil {
		return
	}
	t.mu.Lock()
	if t.stages == nil {
		t.stages = make(map[TelemetryStage]StageMetrics, 3)
	}
	m := t.stages[stage]
	m.Calls++
	m.Duration += elapsed
	t.stages[stage] = m
	t.mu.Unlock()
}

func (t *Telemetry) observeTokens(attentionMask []int64, rows, seqLen int) {
	if t == nil || rows <= 0 || seqLen <= 0 {
		return
	}
	t.mu.Lock()
	for row := 0; row < rows; row++ {
		count := 0
		for _, mask := range attentionMask[row*seqLen : (row+1)*seqLen] {
			if mask != 0 {
				count++
			}
		}
		t.texts++
		t.tokens += int64(count)
		if t.texts == 1 || count < t.minTokens {
			t.minTokens = count
		}
		if count > t.maxTokens {
			t.maxTokens = count
		}
	}
	t.mu.Unlock()
}

func (t *Telemetry) observeInference(rows, capacity int, elapsed time.Duration) {
	if t == nil {
		return
	}
	t.observe(StageONNXInference, elapsed)
	t.mu.Lock()
	t.inferences++
	t.batchRows += int64(rows)
	t.batchCapacity += int64(capacity)
	t.mu.Unlock()
}

func (t *Telemetry) observeSession(elapsed time.Duration) {
	if t == nil {
		return
	}
	t.observe(StageSessionCreate, elapsed)
	t.mu.Lock()
	t.sessions++
	t.mu.Unlock()
}

// Snapshot returns a stable copy of the current aggregate metrics.
func (t *Telemetry) Snapshot() TelemetrySnapshot {
	if t == nil {
		return TelemetrySnapshot{Stages: map[TelemetryStage]StageMetrics{}}
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	stages := make(map[TelemetryStage]StageMetrics, len(t.stages))
	for stage, metrics := range t.stages {
		stages[stage] = metrics
	}
	return TelemetrySnapshot{
		Stages: stages, Texts: t.texts, Tokens: t.tokens,
		MinTokens: t.minTokens, MaxTokens: t.maxTokens,
		InferenceCount: t.inferences, SessionCount: t.sessions,
		BatchRows: t.batchRows, BatchCapacity: t.batchCapacity,
	}
}

// Reset clears all aggregates while preserving the collector instance.
func (t *Telemetry) Reset() {
	if t == nil {
		return
	}
	t.mu.Lock()
	t.stages = nil
	t.texts, t.tokens = 0, 0
	t.minTokens, t.maxTokens = 0, 0
	t.inferences, t.sessions = 0, 0
	t.batchRows, t.batchCapacity = 0, 0
	t.mu.Unlock()
}
