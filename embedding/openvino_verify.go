package embedding

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os/exec"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/v0lka/sp4rk/sysproc"
)

// The OpenVINO counterpart of GPUInUse's nvidia-smi verification: an
// external, driver-side answer to "did inference really execute on the
// Intel GPU?" that does not depend on the ONNX Runtime (or the plugin)
// reporting which device it picked.
//
// The mechanism is device-level, and deliberately so: xpu-smi's per-process
// attribution (`xpu-smi ps`) returns no process data on integrated Arc
// GPUs, so the per-PID rule ADR-045 uses for CUDA has no Intel equivalent
// on exactly the machines c0wrk targets. The device-level discriminator is
// the compute-engine utilization counter (`compute_engine_utilization` in
// `xpu-smi stats --device 0 -j` output): it reads 0 at idle — the desktop
// compositor keeps the *render*/media engines busy, which pollutes the
// raw gpu/media counters but not the compute counter —
// and jumps above 80% while a real model computes on the iGPU. Sustained
// inference during the sampling window therefore proves the work reached
// the GPU; a counter that never rises means the plugin slid to a CPU device
// inside OpenVINO, which a green inference call alone cannot reveal.
const (
	// openVINOVerifyTimeout bounds the whole verification (inference loop +
	// sampling window). A healthy run concludes in a few seconds: each
	// xpu-smi spawn answers in tens of milliseconds and the sampling
	// cadence comes from the spawns themselves — there is deliberately no
	// sleep between samples. The cap exists so a wedged driver can stall
	// at most the caller's own deadline, never unboundedly.
	openVINOVerifyTimeout = 30 * time.Second

	// openVINOVerifySamples caps the number of xpu-smi readings per
	// verification. Sampling stops early once the threshold is crossed.
	openVINOVerifySamples = 12

	// openVINOVerifyComputeThreshold is the compute-utilization percentage
	// a sample must reach for the run to count as verified. Idle reads 0%;
	// real jina-model inference on the Arrow Lake iGPU sustains
	// 84–87%; 25% keeps a wide margin over idle noise while still requiring
	// genuinely GPU-resident compute.
	openVINOVerifyComputeThreshold = 25.0
)

// VerifyOpenVINOExecution drives sustained inference on emb while sampling
// xpu-smi's device-level compute-utilization counter, and reports whether
// any sample reached openVINOVerifyComputeThreshold — external proof that
// the OpenVINO execution provider really executes on the Intel GPU rather
// than on a CPU device inside the plugin.
//
// The verification is device-level by nature: xpu-smi exposes no per-process
// attribution on integrated Intel GPUs (its `ps` view carries no data
// there), so unlike GPUInUse this probe cannot name the owning PID. A
// rising compute counter on a machine where nothing else computes is still
// a strong external signal; the caller must present the verdict as
// device-level, not process-level.
//
// Result semantics (mirroring GPUInUse):
//   - xpu-smi not found in PATH → (false, nil). The machine has no Intel
//     GPU tooling, so the signal is unverifiable rather than negative; the
//     caller must treat this as "unverified", never as "not on GPU".
//   - xpu-smi found but the invocation fails (driver down, non-zero exit,
//     timeout) or the sustained inference itself fails → (false, err).
//     Callers decide whether to log the diagnostic and MUST treat any
//     error as "unverified".
//   - A sample reached the threshold → (true, nil).
//   - Sampling concluded without ever reaching the threshold → (false, nil):
//     inference demonstrably never drove the GPU's compute engine.
//
// The probe is diagnostics-only by contract: it never blocks or fails
// embedder startup, and callers run it in a background/diagnostic context.
//
// Cancellation: the caller-supplied ctx governs cancellation; the effective
// deadline is min(caller deadline, openVINOVerifyTimeout). An expired
// context surfaces as an error (unverified), never as a false "verified".
func VerifyOpenVINOExecution(ctx context.Context, emb *Embedder) (bool, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if emb == nil {
		return false, errors.New("verifying OpenVINO execution: nil embedder")
	}

	bin, ok := xpuSmiPath()
	if !ok {
		return false, nil
	}

	ctx, cancel := context.WithTimeout(ctx, openVINOVerifyTimeout)
	defer cancel()

	infer := func(ctx context.Context) error {
		_, err := emb.EmbedQuery(ctx, "OpenVINO execution verification")
		return err
	}
	return verifyByComputeUtilization(ctx, bin, infer)
}

// xpuSmiPath resolves the xpu-smi executable in PATH, mirroring
// nvidiaSmiPath: the boolean is false whenever the binary is unavailable,
// which the caller translates into the "unverifiable, not negative" answer
// (false, nil).
func xpuSmiPath() (string, bool) {
	bin, err := exec.LookPath("xpu-smi")
	if err != nil {
		return "", false
	}
	return bin, true
}

// verifyByComputeUtilization is the orchestration core of
// VerifyOpenVINOExecution, factored out for testability: infer is any
// inference closure (production passes EmbedQuery; tests pass stubs). It
// runs infer in a loop on a background goroutine while the calling
// goroutine takes up to openVINOVerifySamples `xpu-smi stats --device 0 -j`
// readings of the device-0 compute-engine utilization counter, stopping
// early once a reading reaches openVINOVerifyComputeThreshold.
//
// Device 0 is the only sampled device: the target machines expose a single
// Intel adapter (the integrated GPU), and xpu-smi's own ordinal space gives
// it id 0. Multi-adapter Intel rigs would need per-device sampling — out of
// scope until such hardware exists to test against.
//
// The verification concludes when sampling stops: the inference goroutine
// is joined (an inference error never leaks past the join) and the outcome
// is (thresholdReached, inferenceErr) — a threshold crossing with a
// concurrent inference failure still reports the error, erring on the
// honest side of "unverified".
func verifyByComputeUtilization(ctx context.Context, xpuSMI string, infer func(context.Context) error) (bool, error) {
	var (
		inferErr error
		wg       sync.WaitGroup
	)
	stop := make(chan struct{})
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			select {
			case <-stop:
				return
			case <-ctx.Done():
				return
			default:
				if err := infer(ctx); err != nil {
					inferErr = fmt.Errorf("sustained verification inference failed: %w", err)
					return
				}
			}
		}
	}()

	var (
		maxPct    float64
		threshold bool
	)
	for range openVINOVerifySamples {
		pct, err := xpuComputeUtilization(ctx, xpuSMI)
		if err != nil {
			if ctx.Err() != nil {
				break
			}
			stopInference(&wg, stop)
			return false, fmt.Errorf("querying xpu-smi compute utilization: %w", err)
		}
		if pct > maxPct {
			maxPct = pct
		}
		if maxPct >= openVINOVerifyComputeThreshold {
			threshold = true
			break
		}
	}
	stopInference(&wg, stop)

	if inferErr != nil {
		return false, inferErr
	}
	if err := ctx.Err(); err != nil {
		// The budget expired before sampling could conclude: surface the
		// expiry as an error so the caller reports "unverified", never a
		// threshold read cut short by the deadline as "verified".
		return false, fmt.Errorf("OpenVINO execution verification context ended: %w", err)
	}
	return threshold, nil
}

// stopInference signals the inference loop to stop and waits for it to
// finish, so no goroutine outlives the verification.
func stopInference(wg *sync.WaitGroup, stop chan struct{}) {
	close(stop)
	wg.Wait()
}

// xpuComputeUtilization takes one bounded xpu-smi reading of the
// compute-engine utilization counter and returns the highest percentage in
// the payload. The invocation is the documented `stats` subcommand with
// JSON output (`xpu-smi stats --device 0 -j`); parsing every numeric leaf
// under a compute-engine-utilization key — rather than max-scanning every
// numeric token the payload also carries (frequencies, temperatures,
// memory sizes, the gpu/media/copy utilization classes) — is delegated to
// xpuSMIComputeUtilizations, the same parser the live GPU-visibility test
// uses. A payload without a recognizable compute-utilization field yields
// (0, nil): the counter idles at 0 and sampling simply continues.
func xpuComputeUtilization(ctx context.Context, xpuSMI string) (float64, error) {
	cmd := exec.CommandContext(ctx, xpuSMI, "stats", "--device", "0", "-j")
	// Suppress the console window a GUI-subsystem host would allocate for
	// the child probe process (CREATE_NO_WINDOW on Windows; no-op
	// elsewhere).
	sysproc.HideConsole(cmd)
	out, err := cmd.Output()
	if err != nil {
		return 0, err
	}
	var maxPct float64
	for _, pct := range xpuSMIComputeUtilizations(out) {
		if pct > maxPct {
			maxPct = pct
		}
	}
	return maxPct, nil
}

// xpuSMIComputeUtilizations extracts the compute-engine utilization
// percentages from one `xpu-smi stats -j` payload. The CLI prints a JSON
// array of per-device objects; a device's compute-engine utilization lives
// under a "compute_engine_utilization" key whose value is either a bare
// number or an object keyed by engine/tile, depending on the xpu-smi
// version. Every numeric leaf below that key is returned in ascending
// order (JSON object keys are walked in Go's random map order, so the
// result is sorted to stay deterministic); all other numeric
// stats — frequencies, temperatures, memory sizes, the gpu/media/copy
// utilization classes — are ignored, none of which measure compute work.
// A payload with no recognizable compute-utilization field yields nil.
func xpuSMIComputeUtilizations(data []byte) []float64 {
	var payload any
	if err := json.Unmarshal(data, &payload); err != nil {
		return nil
	}
	var pcts []float64
	xpuSMIWalkCompute(payload, false, &pcts)
	sort.Float64s(pcts)
	return pcts
}

// xpuSMIWalkCompute walks a decoded JSON value and appends every numeric
// leaf reached through a compute-engine-utilization key to pcts. inCompute
// marks values already reached through such a key, so composites (objects,
// arrays) below it are followed too.
func xpuSMIWalkCompute(node any, inCompute bool, pcts *[]float64) {
	switch v := node.(type) {
	case map[string]any:
		for key, child := range v {
			xpuSMIWalkCompute(child, inCompute || xpuSMIComputeUtilKey(key), pcts)
		}
	case []any:
		for _, child := range v {
			xpuSMIWalkCompute(child, inCompute, pcts)
		}
	case float64:
		if inCompute {
			*pcts = append(*pcts, v)
		}
	}
}

// xpuSMIComputeUtilKey reports whether a JSON key names a compute-engine
// utilization metric (e.g. "compute_engine_utilization"), as opposed to the
// gpu/media/copy utilization classes and the non-percent stats.
func xpuSMIComputeUtilKey(key string) bool {
	k := strings.ToLower(key)
	return strings.Contains(k, "compute") && strings.Contains(k, "util")
}
