package embedding

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"github.com/v0lka/sp4rk/sysproc"
)

// gpuProbeTimeout bounds a single nvidia-smi probe invocation. A healthy
// nvidia-smi answers in tens of milliseconds; the cap exists so that a wedged
// driver or a hung NVML call can never stall a caller that treats GPUInUse as
// a cheap, non-blocking monitoring check. On timeout the context kills the
// process and the probe returns an error — it never hangs. The cap composes
// with a caller-supplied context: the probes derive their working context
// via context.WithTimeout(ctx, gpuProbeTimeout), and WithTimeout honours an
// earlier parent deadline, so the effective budget is min(caller deadline,
// gpuProbeTimeout).
const gpuProbeTimeout = 2 * time.Second

// GPUInUse reports whether the NVIDIA driver currently registers *this*
// process as a CUDA compute application — an external, driver-side answer to
// "is this process actually using the GPU?" that does not depend on the ONNX
// Runtime reporting which execution provider it picked.
//
// The probe shells out to
//
//	nvidia-smi --query-compute-apps=pid --format=csv,noheader
//
// (bounded by gpuProbeTimeout), parses the listed compute-application PIDs
// and compares them with os.Getpid(). An empty list means no process on the
// machine is using the GPU right now; a non-empty list without our PID means
// some other process is, but not us.
//
// Result semantics:
//   - nvidia-smi not found in PATH → (false, nil). The machine has no NVIDIA
//     userspace, so the signal is unverifiable rather than negative; the
//     probe must never turn that into a startup error or a block.
//   - nvidia-smi found but the invocation fails (driver down, non-zero exit,
//     timeout) → (false, err). The tool exists yet could not answer; callers
//     decide whether to log the diagnostic, and MUST treat any error as
//     "unverified", never as "not on GPU".
//   - Our PID is listed → (true, nil).
//   - Our PID is not listed → (false, nil).
//
// A CUDA context is created lazily (first on the CUDA execution provider
// initialization / first inference) and lives until the ONNX Runtime
// environment is destroyed, so the correct moment to probe is after at least
// one real inference, not immediately after session creation.
//
// Known limitation: in PID-namespaced environments (containers, WSL) NVML may
// report host-namespace PIDs that never match the in-namespace os.Getpid();
// the probe would then answer false even for a genuinely GPU-bound process.
// The desktop app runs on the host, where the namespaces coincide.
//
// Cancellation: the caller-supplied ctx governs cancellation. The probe
// derives its working context with context.WithTimeout(ctx,
// gpuProbeTimeout), so the effective deadline is min(caller deadline, 2s) —
// an earlier caller deadline always wins, while the internal cap remains the
// upper bound for callers without one. A cancelled or expired context
// surfaces as an error, but only once nvidia-smi is actually invoked: the
// absent-tool fast path below returns the no-error empty result without
// consulting ctx.
func GPUInUse(ctx context.Context) (bool, error) {
	if ctx == nil {
		ctx = context.Background()
	}

	bin, ok := nvidiaSmiPath()
	if !ok {
		return false, nil
	}

	ctx, cancel := context.WithTimeout(ctx, gpuProbeTimeout)
	defer cancel()

	cmd := exec.CommandContext(ctx, bin,
		"--query-compute-apps=pid", "--format=csv,noheader")
	// Suppress the console window a GUI-subsystem host would allocate for the
	// child probe process (CREATE_NO_WINDOW on Windows; no-op elsewhere).
	sysproc.HideConsole(cmd)
	out, err := cmd.Output()
	if err != nil {
		return false, fmt.Errorf("querying nvidia-smi compute apps: %w", err)
	}

	own := os.Getpid()
	for _, pid := range parseComputeAppPIDs(string(out)) {
		if pid == own {
			return true, nil
		}
	}
	return false, nil
}

// nvidiaSmiPath resolves the nvidia-smi executable in PATH. The boolean is
// false whenever the binary is unavailable for any reason (absent, PATH
// broken). Callers translate that into the probe's "not verified" answer
// (false, nil) rather than an error — which is why the lookup error is
// deliberately absorbed here instead of being surfaced under an `if err != nil`
// block that returns a nil error.
func nvidiaSmiPath() (string, bool) {
	bin, err := exec.LookPath("nvidia-smi")
	if err != nil {
		return "", false
	}
	return bin, true
}

// parseComputeAppPIDs extracts the compute-application PIDs from the stdout
// of `nvidia-smi --query-compute-apps=pid --format=csv,noheader` — normally
// one decimal PID per line.
//
// The parser is deliberately forgiving, because nvidia-smi substitutes
// placeholder prose for numbers in several states ("No running processes
// found", "[N/A]", "[Not Supported]"), Windows drivers emit \r\n line
// endings, and wider queries add comma-separated columns after the pid.
// Rules, in order: take the first comma-separated field of each line, trim
// whitespace, parse it as a decimal integer, keep it only when positive.
// Everything else — empty lines, prose, hex, negatives, overflow — is
// ignored, so garbage output degrades to "no PIDs" instead of an error or a
// spurious match.
func parseComputeAppPIDs(out string) []int {
	lines := strings.Split(out, "\n")
	pids := make([]int, 0, len(lines))
	for _, line := range lines {
		field, _, _ := strings.Cut(line, ",")
		field = strings.TrimSpace(field)
		if field == "" {
			continue
		}
		pid, err := strconv.Atoi(field)
		if err != nil || pid <= 0 {
			continue
		}
		pids = append(pids, pid)
	}
	return pids
}
