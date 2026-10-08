package orchestration

import (
	"context"
	"sync"
	"testing"
	"time"
)

// --- Fakes ---

// ctxAwareCheckpointer records whether the context handed to SaveCheckpoint
// carries a deadline, and can block on it. Only the first save blocks (until
// its context is cancelled or release is closed); later saves return nil at
// once.
type ctxAwareCheckpointer struct {
	release chan struct{}
	entered chan struct{} // signaled (non-blocking) on every save entry

	mu            sync.Mutex
	saves         int
	sawDeadline   bool
	firstCanceled bool
}

func newCtxAwareCheckpointer() *ctxAwareCheckpointer {
	return &ctxAwareCheckpointer{
		release: make(chan struct{}),
		entered: make(chan struct{}, 1),
	}
}

func (c *ctxAwareCheckpointer) SaveCheckpoint(ctx context.Context, _ string, _ Blackboard) error {
	select {
	case c.entered <- struct{}{}:
	default:
	}
	c.mu.Lock()
	n := c.saves
	c.saves++
	if _, ok := ctx.Deadline(); ok {
		c.sawDeadline = true
	}
	c.mu.Unlock()

	if n == 0 {
		select {
		case <-c.release:
		case <-ctx.Done():
			c.mu.Lock()
			c.firstCanceled = true
			c.mu.Unlock()
			return ctx.Err()
		}
	}
	return nil
}

func (c *ctxAwareCheckpointer) LoadCheckpoint(context.Context, string) (Blackboard, error) {
	return nil, nil
}

func (c *ctxAwareCheckpointer) DeleteCheckpoint(context.Context, string) error { return nil }

func (c *ctxAwareCheckpointer) stats() (saves int, sawDeadline, firstCanceled bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.saves, c.sawDeadline, c.firstCanceled
}

// stuckCheckpointer blocks inside SaveCheckpoint and ignores its context —
// the pathological backend that previously wedged the worker and hung
// Shutdown. With a nil release it blocks forever; with a non-nil release it
// unblocks when that channel is closed.
type stuckCheckpointer struct {
	entered chan struct{}
	release chan struct{} // nil = block forever
}

func newStuckCheckpointer() *stuckCheckpointer {
	return &stuckCheckpointer{entered: make(chan struct{}, 1)}
}

func (s *stuckCheckpointer) SaveCheckpoint(context.Context, string, Blackboard) error {
	select {
	case s.entered <- struct{}{}:
	default:
	}
	if s.release != nil {
		<-s.release
	} else {
		<-make(chan struct{}) // block forever, ignoring ctx
	}
	return nil
}

func (s *stuckCheckpointer) LoadCheckpoint(context.Context, string) (Blackboard, error) {
	return nil, nil
}

func (s *stuckCheckpointer) DeleteCheckpoint(context.Context, string) error { return nil }

// --- Tests ---

// TestCheckpointedBlackboard_OperationBoundedByContext is the regression test
// for the persistence-timeout fix: the per-operation timeout must bound the
// operation itself (the worker passes a context with the deadline to
// SaveCheckpoint), not only the caller's wait. Without it a blocked save
// stalls the single worker forever, the queue grows without bound and
// Shutdown never returns. Here the first save is released only through
// context cancellation — it can only unblock (and let the worker advance to
// the second, fast save) if the operation context actually carries the
// deadline.
func TestCheckpointedBlackboard_OperationBoundedByContext(t *testing.T) {
	cp := newCtxAwareCheckpointer()
	pb := NewCheckpointedBlackboard("op-bounded", cp, nil, 50*time.Millisecond)
	defer pb.Shutdown()

	// The first save blocks; its caller wait times out. The second write is
	// enqueued behind it (its own caller wait times out too) and can only
	// execute once the worker is freed by the first operation's context
	// deadline — so saves>=2 proves the worker advanced.
	pb.SetPlan(&Plan{Steps: []PlanStep{{ID: "s1"}}})
	pb.SetFinalResult("done")

	// Wait for the worker to move past the cancelled op and complete the
	// queued follow-up save.
	deadline := time.Now().Add(5 * time.Second)
	for {
		saves, sawDeadline, firstCanceled := cp.stats()
		if saves >= 2 {
			if !sawDeadline {
				t.Fatal("SaveCheckpoint received a context without a deadline — operation timeout does not bound the operation")
			}
			if !firstCanceled {
				t.Fatal("first save returned without context cancellation — it was not bounded by the operation timeout")
			}
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("worker did not advance past the blocked save: saves=%d sawDeadline=%v firstCanceled=%v",
				saves, sawDeadline, firstCanceled)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

// TestCheckpointedBlackboard_ShutdownReturnsDespiteStuckWorker is the
// regression test for the Shutdown hang: a SaveCheckpoint that blocks forever
// while ignoring its context used to wedge the single worker and make
// Shutdown's wg.Wait() never return, hanging process shutdown. Shutdown must
// now return within the (test-shortened) grace period even though the worker
// stays stuck.
func TestCheckpointedBlackboard_ShutdownReturnsDespiteStuckWorker(t *testing.T) {
	cp := newStuckCheckpointer()
	pb := NewCheckpointedBlackboard("stuck-worker", cp, nil, 20*time.Millisecond)
	pb.shutdownGrace = 100 * time.Millisecond // test-shortened grace
	defer pb.Shutdown()                       // idempotent; second call returns at once

	pb.SetOriginalRequest("task") // caller wait expires; the worker stays stuck
	select {
	case <-cp.entered:
	case <-time.After(5 * time.Second):
		t.Fatal("worker never entered SaveCheckpoint")
	}

	done := make(chan struct{})
	go func() {
		pb.Shutdown()
		close(done)
	}()
	select {
	case <-done:
		// Shutdown returned despite the permanently stuck worker.
	case <-time.After(5 * time.Second):
		t.Fatal("Shutdown did not return within 5s despite a 100ms grace — it is still waiting for the stuck worker")
	}
}

// TestCheckpointedBlackboard_QueueCapDropsWhenWorkerStuck verifies the queue
// cap: with the worker blocked in a save and the caller-side waits expired,
// flooding writes must stop accumulating at maxPersistenceQueue — excess
// operations are dropped (logged) instead of growing the queue without bound.
func TestCheckpointedBlackboard_QueueCapDropsWhenWorkerStuck(t *testing.T) {
	cp := newStuckCheckpointer()
	cp.release = make(chan struct{}) // releasable variant: blocks until closed below
	pb := NewCheckpointedBlackboard("queue-cap", cp, nil, time.Millisecond)
	pb.shutdownGrace = time.Second // never wait long if the drain assertion fails
	defer pb.Shutdown()

	pb.SetOriginalRequest("task") // dequeued by the worker, which then blocks in save
	select {
	case <-cp.entered:
	case <-time.After(5 * time.Second):
		t.Fatal("worker never entered SaveCheckpoint")
	}

	for i := 0; i < maxPersistenceQueue+10; i++ {
		pb.StoreFact(Fact{Keywords: []string{"k"}, Content: "v"}) // caller wait (1ms) expires; op enqueued or dropped
	}

	pb.queueMu.Lock()
	queued := len(pb.queue)
	pb.queueMu.Unlock()
	if queued != maxPersistenceQueue {
		t.Errorf("queued ops = %d, want exactly %d (queue cap enforced)", queued, maxPersistenceQueue)
	}

	// Release the worker; it must drain without issue.
	close(cp.release)
	deadline := time.Now().Add(5 * time.Second)
	for {
		pb.queueMu.Lock()
		queued = len(pb.queue)
		pb.queueMu.Unlock()
		if queued == 0 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("queue did not drain after release: %d ops pending", queued)
		}
		time.Sleep(5 * time.Millisecond)
	}
}
