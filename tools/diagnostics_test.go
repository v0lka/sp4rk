package tools

import (
	"context"
	"log/slog"
	"reflect"
	"sync"
	"testing"
)

type expectedDiagnostic struct {
	message string
	attrs   map[string]any
}

type diagnosticRecords struct {
	mu      sync.Mutex
	records []slog.Record
}

type diagnosticHandler struct {
	state *diagnosticRecords
	attrs []slog.Attr
	group string
}

func (h *diagnosticHandler) Enabled(ctx context.Context, level slog.Level) bool {
	return level >= slog.LevelWarn || slog.Default().Handler().Enabled(ctx, level)
}
func (h *diagnosticHandler) Handle(ctx context.Context, r slog.Record) error {
	if r.Level < slog.LevelWarn {
		return slog.Default().Handler().Handle(ctx, r)
	}
	r = r.Clone()
	r.AddAttrs(h.attrs...)
	h.state.mu.Lock()
	defer h.state.mu.Unlock()
	h.state.records = append(h.state.records, r)
	return nil
}
func (h *diagnosticHandler) WithAttrs(attrs []slog.Attr) slog.Handler {
	copyHandler := *h
	copyHandler.attrs = append(append([]slog.Attr(nil), h.attrs...), attrs...)
	return &copyHandler
}
func (h *diagnosticHandler) WithGroup(name string) slog.Handler {
	copyHandler := *h
	copyHandler.group = name
	return &copyHandler
}

func expectedDiagnostics(t *testing.T, want ...expectedDiagnostic) *slog.Logger {
	t.Helper()
	h := &diagnosticHandler{state: &diagnosticRecords{}}
	t.Cleanup(func() {
		h.state.mu.Lock()
		defer h.state.mu.Unlock()
		got := h.state.records
		if len(got) != len(want) {
			t.Errorf("diagnostics(%s) count = %d, want %d; records=%v", t.Name(), len(got), len(want), got)
		}
		for i, record := range got {
			if i >= len(want) {
				break
			}
			expected := want[i]
			if record.Level != slog.LevelWarn || record.Message != expected.message {
				t.Errorf("diagnostics(%s)[%d] = (%s, %q), want (WARN, %q)", t.Name(), i, record.Level, record.Message, expected.message)
			}
			attrs := make(map[string]any)
			record.Attrs(func(a slog.Attr) bool { attrs[a.Key] = a.Value.Resolve().Any(); return true })
			for key, value := range expected.attrs {
				actual, ok := attrs[key]
				if predicate, isPredicate := value.(func(any) bool); isPredicate {
					if !ok || !predicate(actual) {
						t.Errorf("diagnostics(%s)[%d].%s = %v, want matching payload", t.Name(), i, key, actual)
					}
				} else if !ok || !reflect.DeepEqual(actual, value) {
					t.Errorf("diagnostics(%s)[%d].%s = %v, want %v", t.Name(), i, key, actual, value)
				}
			}
		}
	})
	return slog.New(h)
}
