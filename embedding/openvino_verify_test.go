package embedding

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"testing"
	"time"
)

// fakeXpuSMIStats returns a fake xpu-smi shell script that prints a
// `stats -j` payload reporting v percent compute-engine utilization on
// device 0 — the same shape the production probe parses.
func fakeXpuSMIStats(v float64) string {
	return fmt.Sprintf("echo '[{\"device_id\":0,\"compute_engine_utilization\":%s}]'\n",
		strconv.FormatFloat(v, 'f', 1, 64))
}

// TestVerifyByComputeUtilization_ThresholdReached is the positive path: a
// fake xpu-smi reporting a GPU-class compute load verifies the run. The
// per-engine object shape exercises the composite branch of the payload
// parser end to end.
func TestVerifyByComputeUtilization_ThresholdReached(t *testing.T) {
	installFakeTool(t, "xpu-smi",
		"echo '[{\"device_id\":0,\"compute_engine_utilization\":{\"all\":84.32,\"compute0\":81.1}}]'\n")
	inferCalls := 0
	verified, err := verifyByComputeUtilization(context.Background(), "xpu-smi", func(context.Context) error {
		inferCalls++
		return nil
	})
	if err != nil {
		t.Fatalf("verifyByComputeUtilization() error = %v", err)
	}
	if !verified {
		t.Error("verifyByComputeUtilization() = false, want true at 84.32% compute utilization")
	}
	if inferCalls == 0 {
		t.Error("inference closure never ran; verification sampled an idle GPU")
	}
}

// TestVerifyByComputeUtilization_NeverReachesThreshold pins the honest
// negative: a counter that stays at idle proves nothing and reports
// (false, nil) — "not verified", not an error and never "verified".
func TestVerifyByComputeUtilization_NeverReachesThreshold(t *testing.T) {
	installFakeTool(t, "xpu-smi", fakeXpuSMIStats(0))
	verified, err := verifyByComputeUtilization(context.Background(), "xpu-smi", func(context.Context) error {
		return nil
	})
	if err != nil {
		t.Fatalf("verifyByComputeUtilization() error = %v", err)
	}
	if verified {
		t.Error("verifyByComputeUtilization() = true at 0.00% utilization, want false")
	}
}

// TestVerifyByComputeUtilization_ThresholdIsNotNoise pins the threshold
// semantics: readings below openVINOVerifyComputeThreshold never verify, a
// reading at the threshold does.
func TestVerifyByComputeUtilization_ThresholdIsNotNoise(t *testing.T) {
	below := openVINOVerifyComputeThreshold - 5
	installFakeTool(t, "xpu-smi", fakeXpuSMIStats(below))
	verified, err := verifyByComputeUtilization(context.Background(), "xpu-smi", func(context.Context) error { return nil })
	if err != nil {
		t.Fatalf("verifyByComputeUtilization() error = %v", err)
	}
	if verified {
		t.Errorf("verifyByComputeUtilization() = true at %.1f%%, below the %.1f%% threshold", below, openVINOVerifyComputeThreshold)
	}

	at := openVINOVerifyComputeThreshold
	installFakeTool(t, "xpu-smi", fakeXpuSMIStats(at))
	verified, err = verifyByComputeUtilization(context.Background(), "xpu-smi", func(context.Context) error { return nil })
	if err != nil {
		t.Fatalf("verifyByComputeUtilization() error = %v", err)
	}
	if !verified {
		t.Errorf("verifyByComputeUtilization() = false at %.1f%%, want true (threshold is inclusive)", at)
	}
}

// TestVerifyByComputeUtilization_InferenceFailureSurfaces pins the honesty
// contract: a failing inference closure must surface as an error — never as
// a silent unverified or, worse, a verified verdict.
func TestVerifyByComputeUtilization_InferenceFailureSurfaces(t *testing.T) {
	installFakeTool(t, "xpu-smi", fakeXpuSMIStats(84.32))
	verified, err := verifyByComputeUtilization(context.Background(), "xpu-smi", func(context.Context) error {
		return errors.New("inference exploded")
	})
	if verified {
		t.Error("verifyByComputeUtilization() = true despite inference failure, want false")
	}
	if err == nil {
		t.Fatal("verifyByComputeUtilization() error = nil despite inference failure")
	}
	if !strings.Contains(err.Error(), "inference exploded") {
		t.Errorf("error %q does not carry the inference failure cause", err.Error())
	}
}

// TestVerifyByComputeUtilization_ProbeFailureSurfaces: an xpu-smi that
// exists but cannot answer is "unverified with a cause", never a verdict.
func TestVerifyByComputeUtilization_ProbeFailureSurfaces(t *testing.T) {
	installFakeTool(t, "xpu-smi", "exit 7\n")
	verified, err := verifyByComputeUtilization(context.Background(), "xpu-smi", func(context.Context) error { return nil })
	if verified {
		t.Error("verifyByComputeUtilization() = true despite probe failure, want false")
	}
	if err == nil {
		t.Fatal("verifyByComputeUtilization() error = nil despite probe failure")
	}
}

// TestVerifyByComputeUtilization_ExpiredContextSurfaces pins the deadline
// contract: an already-expired context surfaces as an error (unverified),
// never as a threshold read cut short by the deadline reported as verified.
func TestVerifyByComputeUtilization_ExpiredContextSurfaces(t *testing.T) {
	installFakeTool(t, "xpu-smi", fakeXpuSMIStats(84.32))
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	verified, err := verifyByComputeUtilization(ctx, "xpu-smi", func(context.Context) error { return nil })
	if verified {
		t.Error("verifyByComputeUtilization(expired ctx) = true, want false")
	}
	if err == nil {
		t.Fatal("verifyByComputeUtilization(expired ctx) error = nil, want the context error")
	}
	if !errors.Is(err, context.Canceled) {
		t.Errorf("error %v does not wrap context.Canceled", err)
	}
}

// TestVerifyOpenVINOExecution_AbsentXpuSmi pins the GPUInUse-mirroring
// semantics: without xpu-smi the signal is unverifiable, reported as
// (false, nil) — the caller must treat it as unverified, not as failure.
func TestVerifyOpenVINOExecution_AbsentXpuSmi(t *testing.T) {
	t.Setenv("PATH", "")
	verified, err := VerifyOpenVINOExecution(context.Background(), &Embedder{})
	if verified {
		t.Error("VerifyOpenVINOExecution() = true without xpu-smi, want false")
	}
	if err != nil {
		t.Errorf("VerifyOpenVINOExecution() error = %v without xpu-smi, want nil (absent tool is unverifiable, not a failure)", err)
	}
}

// TestVerifyOpenVINOExecution_NilEmbedder: the guard must fire before any
// probe — a nil embedder is a programming error, not "unverified".
func TestVerifyOpenVINOExecution_NilEmbedder(t *testing.T) {
	installFakeTool(t, "xpu-smi", fakeXpuSMIStats(84.32))
	verified, err := VerifyOpenVINOExecution(context.Background(), nil)
	if verified {
		t.Error("VerifyOpenVINOExecution(nil) = true, want false")
	}
	if err == nil {
		t.Fatal("VerifyOpenVINOExecution(nil) error = nil, want the nil-embedder error")
	}
}

// TestVerifyOpenVINOExecution_RealProbe is the runnable-on-any-machine
// smoke run through the exported surface: wherever xpu-smi exists the call
// is bounded by its internal budget; wherever it does not, the absent path
// returns immediately.
func TestVerifyOpenVINOExecution_RealProbe(t *testing.T) {
	start := time.Now()
	// A nil embedder would error before the probe — use the absent-tool
	// fast path instead: without xpu-smi this returns (false, nil)
	// immediately; with xpu-smi it must conclude inside the verify budget.
	t.Setenv("PATH", "")
	verified, err := VerifyOpenVINOExecution(context.Background(), &Embedder{})
	elapsed := time.Since(start)
	if err != nil {
		t.Fatalf("VerifyOpenVINOExecution() error = %v", err)
	}
	if verified {
		t.Error("VerifyOpenVINOExecution() = true without xpu-smi, want false")
	}
	if elapsed > openVINOVerifyTimeout {
		t.Errorf("VerifyOpenVINOExecution took %v, exceeding the %v internal budget", elapsed, openVINOVerifyTimeout)
	}
}

// TestXpuSMIComputeUtilizations unit-tests the `xpu-smi stats -j` payload
// parser: only compute-engine utilization values may come out, never the
// other numeric stats the same payload carries.
func TestXpuSMIComputeUtilizations(t *testing.T) {
	t.Parallel()
	for name, tc := range map[string]struct {
		payload string
		want    []float64
	}{
		"bare number": {
			payload: `[{"device_id":0,"gpu_utilization":77,"compute_engine_utilization":42,
				"media_engine_utilization":9,"gpu_frequency_mhz":2050,"gpu_memory_used_mib":312}]`,
			want: []float64{42},
		},
		"per-engine object": {
			payload: `[{"device_id":0,"compute_engine_utilization":{"all":31.5,"compute0":18.25},
				"copy_engine_utilization":{"all":64}}]`,
			want: []float64{18.25, 31.5},
		},
		"multi-device": {
			payload: `[{"device_id":0,"compute_engine_utilization":7},
				{"device_id":1,"compute_engine_utilization":55}]`,
			want: []float64{7, 55},
		},
		"no compute field": {
			payload: `[{"device_id":0,"gpu_utilization":88,"gpu_memory_used_mib":128}]`,
			want:    nil,
		},
		"not json": {
			payload: "xpu-smi: no devices found\n",
			want:    nil,
		},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			got := xpuSMIComputeUtilizations([]byte(tc.payload))
			if len(got) != len(tc.want) {
				t.Fatalf("xpuSMIComputeUtilizations() = %v, want %v", got, tc.want)
			}
			for i := range got {
				if got[i] != tc.want[i] {
					t.Errorf("xpuSMIComputeUtilizations()[%d] = %v, want %v", i, got[i], tc.want[i])
				}
			}
		})
	}
}
