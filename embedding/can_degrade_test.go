// SPDX-License-Identifier: Apache-2.0

package embedding

import "testing"

// TestEmbedder_canDegradeToCPU pins the lazy CUDA->CPU fallback gate: a pending
// auto-on-CUDA request may degrade only while no CUDA-backed session is live.
// A live session runs on the dedicated ONNX thread the fallback stops, so
// degrading mid-flight would strand it — the gate must say no.
func TestEmbedder_canDegradeToCPU(t *testing.T) {
	cases := []struct {
		name string
		e    *Embedder
		want bool
	}{
		{"pending with no live session", &Embedder{cudaFallbackPending: true}, true},
		{"not pending", &Embedder{}, false},
		{"pending but the legacy session is live", &Embedder{cudaFallbackPending: true, sess: &onnxSession{}}, false},
		{"pending but the batch session is live", &Embedder{cudaFallbackPending: true, batchSess: &onnxSession{}}, false},
		{"pending but a bucket session is live", &Embedder{cudaFallbackPending: true, bucketSessions: map[sessionKey]*onnxSession{{}: {}}}, false},
	}
	for _, tc := range cases {
		if got := tc.e.canDegradeToCPU(); got != tc.want {
			t.Errorf("%s: canDegradeToCPU() = %v, want %v", tc.name, got, tc.want)
		}
	}
}

// TestEmbedder_ExecutionProvider_ZeroValue pins the getter's documented
// zero-value default: an uninitialized embedder reports "cpu" (the legacy
// default), never the empty string. The provider field is written at runtime
// (degradeToCPU) under e.mu, so the getter takes the read lock — exercised
// under `go test -race` by the tool-result path that polls it concurrently.
func TestEmbedder_ExecutionProvider_ZeroValue(t *testing.T) {
	var e Embedder
	if got := e.ExecutionProvider(); got != ExecutionProviderCPU {
		t.Errorf("zero-value ExecutionProvider() = %q, want %q", got, ExecutionProviderCPU)
	}
}
