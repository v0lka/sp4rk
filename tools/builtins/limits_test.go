package builtins

import (
	"strings"
	"testing"
	"time"
)

func TestDefaultWebFetchLimits(t *testing.T) {
	limits := DefaultWebFetchLimits()
	if limits.Timeout != 30*time.Second {
		t.Errorf("DefaultWebFetchLimits().Timeout = %v, want 30s (matches the tool description and docs/tools.md)", limits.Timeout)
	}
	if limits.Retries != 2 {
		t.Errorf("DefaultWebFetchLimits().Retries = %d, want 2", limits.Retries)
	}
}

// TestNormalizeBashTimeouts pins the zero-value repair: a zero (or
// non-positive-MaxTimeout) config must never survive into a constructor,
// because Execute derives the per-call deadline from MaxTimeout and a zero
// deadline cancels every command before it starts (review finding 31).
func TestNormalizeBashTimeouts(t *testing.T) {
	def := DefaultBashTimeouts()

	tests := []struct {
		name             string
		in               BashTimeouts
		wantMaxTimeout   time.Duration
		wantWaitDelay    time.Duration
		wantDefaultDelay bool // expect the default WaitDelay (5s)
	}{
		{
			name:             "zero value replaced entirely",
			in:               BashTimeouts{},
			wantMaxTimeout:   def.MaxTimeout,
			wantDefaultDelay: true,
		},
		{
			name:             "non-positive MaxTimeout with caller WaitDelay",
			in:               BashTimeouts{MaxTimeout: -1 * time.Second, WaitDelay: 9 * time.Second},
			wantMaxTimeout:   def.MaxTimeout,
			wantWaitDelay:    9 * time.Second,
			wantDefaultDelay: false,
		},
		{
			name:             "populated struct unchanged",
			in:               BashTimeouts{MaxTimeout: 42 * time.Second, WaitDelay: time.Second},
			wantMaxTimeout:   42 * time.Second,
			wantWaitDelay:    time.Second,
			wantDefaultDelay: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := normalizeBashTimeouts(tt.in)
			if got.MaxTimeout != tt.wantMaxTimeout {
				t.Errorf("MaxTimeout = %v, want %v", got.MaxTimeout, tt.wantMaxTimeout)
			}
			if tt.wantDefaultDelay {
				if got.WaitDelay != def.WaitDelay {
					t.Errorf("WaitDelay = %v, want the default %v", got.WaitDelay, def.WaitDelay)
				}
			} else if got.WaitDelay != tt.wantWaitDelay {
				t.Errorf("WaitDelay = %v, want %v (caller value preserved)", got.WaitDelay, tt.wantWaitDelay)
			}
		})
	}
}

// TestBoundedOutputBuffer covers the cap-and-drain writer backing the
// bounded shell-output capture (review finding 5).
func TestBoundedOutputBuffer(t *testing.T) {
	t.Run("under cap stores everything", func(t *testing.T) {
		b := newBoundedOutputBuffer(64)
		p := []byte("hello world")
		n, err := b.Write(p)
		if err != nil || n != len(p) {
			t.Fatalf("Write = (%d, %v), want (%d, nil)", n, err, len(p))
		}
		if got := b.String(); got != "hello world" {
			t.Errorf("String() = %q, want %q", got, "hello world")
		}
		if b.marker() != "" {
			t.Errorf("marker() = %q, want empty when nothing was dropped", b.marker())
		}
	})

	t.Run("past cap drains and counts", func(t *testing.T) {
		const capBytes = 16
		b := newBoundedOutputBuffer(capBytes)
		payload := strings.Repeat("x", 40)
		n, err := b.Write([]byte(payload))
		if err != nil || n != 40 {
			t.Fatalf("Write = (%d, %v), want (40, nil) — the writer must consume everything so the child never blocks", n, err)
		}
		if got := b.Bytes(); string(got) != strings.Repeat("x", capBytes) {
			t.Errorf("Bytes() length = %d, want %d (capped)", len(got), capBytes)
		}
		want := "\n[output truncated — 24 bytes discarded (cap 16 bytes)]"
		if got := b.marker(); got != want {
			t.Errorf("marker() = %q, want %q", got, want)
		}
		if s := b.String(); !strings.HasSuffix(s, want) {
			t.Errorf("String() = %q, want suffix %q", s, want)
		}
	})

	t.Run("single write larger than cap", func(t *testing.T) {
		b := newBoundedOutputBuffer(4)
		payload := strings.Repeat("y", 10)
		if n, err := b.Write([]byte(payload)); err != nil || n != 10 {
			t.Fatalf("Write = (%d, %v), want (10, nil)", n, err)
		}
		if got := string(b.Bytes()); got != "yyyy" {
			t.Errorf("Bytes() = %q, want %q", got, "yyyy")
		}
		if !strings.Contains(b.marker(), "6 bytes discarded") {
			t.Errorf("marker() = %q, want it to report 6 discarded bytes", b.marker())
		}
	})
}
