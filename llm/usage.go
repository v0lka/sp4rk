package llm

import (
	"context"
	"sync"
	"time"
)

// UsageObserver is called after each LLM call with per-call and cumulative usage.
type UsageObserver func(usage TokenUsage, totalIn, totalOut int, model, family string)

// TimedUsageObserver is called after each timed LLM call with per-call usage,
// the measured wall-clock duration of the call, and cumulative usage. It is
// notified only by RecordTimed; Record (the duration-less path) never fires it.
type TimedUsageObserver func(usage TokenUsage, duration time.Duration, totalIn, totalOut int, model, family string)

// UsageTracker accumulates token usage across all LLM calls in a session.
// Thread-safe. Upper layers register observers to react to usage changes.
type UsageTracker struct {
	mu             sync.Mutex
	totalIn        int
	totalOut       int
	observers      []UsageObserver
	timedObservers []TimedUsageObserver
}

// NewUsageTracker creates a new UsageTracker.
func NewUsageTracker() *UsageTracker {
	return &UsageTracker{}
}

// Record adds per-call usage to the running totals and notifies all observers.
func (t *UsageTracker) Record(usage TokenUsage, model, family string) {
	t.mu.Lock()
	t.totalIn += usage.InputTokens
	t.totalOut += usage.OutputTokens
	totalIn := t.totalIn
	totalOut := t.totalOut
	// Snapshot observers under lock to avoid races on the slice.
	observers := make([]UsageObserver, len(t.observers))
	copy(observers, t.observers)
	t.mu.Unlock()

	for _, fn := range observers {
		fn(usage, totalIn, totalOut, model, family)
	}
}

// AddObserver registers a callback that is invoked on every Record call.
func (t *UsageTracker) AddObserver(fn UsageObserver) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.observers = append(t.observers, fn)
}

// RecordTimed adds per-call usage to the running totals and notifies all
// observers, passing the measured wall-clock duration to timed observers.
// Plain UsageObservers are notified exactly as by Record; TimedUsageObservers
// are notified only through this method.
func (t *UsageTracker) RecordTimed(usage TokenUsage, duration time.Duration, model, family string) {
	t.mu.Lock()
	t.totalIn += usage.InputTokens
	t.totalOut += usage.OutputTokens
	totalIn := t.totalIn
	totalOut := t.totalOut
	// Snapshot observers under lock to avoid races on the slices.
	observers := make([]UsageObserver, len(t.observers))
	copy(observers, t.observers)
	timedObservers := make([]TimedUsageObserver, len(t.timedObservers))
	copy(timedObservers, t.timedObservers)
	t.mu.Unlock()

	for _, fn := range observers {
		fn(usage, totalIn, totalOut, model, family)
	}
	for _, fn := range timedObservers {
		fn(usage, duration, totalIn, totalOut, model, family)
	}
}

// AddTimedObserver registers a callback that is invoked on every RecordTimed
// call. Record never notifies timed observers.
func (t *UsageTracker) AddTimedObserver(fn TimedUsageObserver) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.timedObservers = append(t.timedObservers, fn)
}

// Totals returns the current accumulated input and output token counts.
func (t *UsageTracker) Totals() (inputTokens, outputTokens int) {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.totalIn, t.totalOut
}

// TrackingCaller wraps a Caller and automatically:
//   - Records usage from Call() responses into UsageTracker
//   - Corrects the active ContextTokenTracker (if set) after each call
type TrackingCaller struct {
	inner      Caller
	session    *UsageTracker
	ctxMu      sync.RWMutex
	ctxTracker *ContextTokenTracker
}

// NewTrackingCaller creates a TrackingCaller that records usage into tracker.
func NewTrackingCaller(inner Caller, tracker *UsageTracker) *TrackingCaller {
	return &TrackingCaller{
		inner:   inner,
		session: tracker,
	}
}

// Call delegates to the inner caller, measuring the wall-clock duration of the
// call, then records usage (with duration) and corrects the context tracker.
func (tc *TrackingCaller) Call(ctx context.Context, req ChatRequest) (*ChatResponse, error) {
	start := time.Now()
	resp, err := tc.inner.Call(ctx, req)
	if err != nil {
		return nil, err
	}

	tc.session.RecordTimed(resp.Usage, time.Since(start), resp.Model, resp.Family)

	tc.ctxMu.RLock()
	ct := tc.ctxTracker
	tc.ctxMu.RUnlock()

	if ct != nil && resp.Usage.InputTokens > 0 {
		ct.Correct(resp.Usage.InputTokens)
	}

	return resp, nil
}

// WithContextTracker returns a new TrackingCaller that shares the same inner
// caller and session-level UsageTracker, but corrects the given per-step
// ContextTokenTracker. Use this to create step-local callers for parallel execution.
func (tc *TrackingCaller) WithContextTracker(t *ContextTokenTracker) *TrackingCaller {
	return &TrackingCaller{
		inner:      tc.inner,
		session:    tc.session,
		ctxTracker: t,
	}
}
