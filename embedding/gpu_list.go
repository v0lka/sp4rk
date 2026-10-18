package embedding

import (
	"context"
	"errors"
	"fmt"
	"os/exec"
	"strconv"
	"strings"

	"github.com/v0lka/sp4rk/sysproc"
)

// GPUDevice describes one GPU reported by a driver stack, as offered to the
// vector-index device picker.
type GPUDevice struct {
	// Index is the driver-assigned ordinal of the device as used by
	// CUDA_VISIBLE_DEVICES and other ordinal-keyed tooling ("0" is the
	// primary GPU). The ordinal is only meaningful within its Provider:
	// the caller selects a (provider, index) pair, never a bare index —
	// vector_index.device_id is scoped to vector_index.execution_provider.
	Index int
	// Name is the product name as reported by the driver ("NVIDIA GeForce
	// RTX 4070", "Intel(R) Graphics (ARL)"). It is informational only.
	Name string
	// Provider names the execution provider this device belongs to:
	// ExecutionProviderCUDA for nvidia-smi-enumerated devices,
	// ExecutionProviderOpenVINO for Intel devices enumerated through
	// vulkaninfo. Selecting an Intel device maps to device_id 0 — the
	// plugin-based OpenVINO surface currently exposes no per-device
	// selection (see appendOpenVINOProvider), so 0 is the only valid
	// OpenVINO device index and Intel entries always carry it.
	Provider string
}

// ListGPUDevices returns the GPUs visible on the machine across every
// enumeration source the embedding layer knows, each tagged with the
// execution provider it can be reached through:
//
//   - NVIDIA GPUs via `nvidia-smi --query-gpu=index,name --format=csv,noheader`
//     (Provider "cuda");
//   - Intel GPUs via `vulkaninfo --summary` (Provider "openvino"), filtered
//     to the Intel PCI vendor — the hardware the OpenVINO execution provider
//     runs on. The name travels verbatim from the Vulkan driver ("Intel(R)
//     Graphics (ARL)" on Mesa): vulkaninfo is an external probe under the
//     caller's control, unlike the EP metadata inside the ONNX Runtime
//     process, which registers state once per process and whose wrapper
//     surface exposes no device-name accessor.
//
// Every source runs under the same gpuProbeTimeout budget as GPUInUse, so a
// wedged driver can never stall the caller.
//
// Each source follows the GPUInUse result semantics independently:
//   - source tool not found in PATH → that source contributes no devices and
//     no error. Absence of a driver's userspace means "no GPUs from that
//     vendor", not a failure.
//   - source tool found but the invocation fails (driver down, non-zero
//     exit, timeout) → that source contributes an error and no devices.
//
// Merge semantics: devices from every source that answered are concatenated
// (NVIDIA first, then Intel). One broken driver stack must not hide the GPUs
// another driver still reports, so a partial answer wins: the merged list is
// returned with a nil error as long as at least one source answered. Only
// when no source produced a device AND at least one present source failed
// does the call fail, with the errors joined — the caller then sees
// "hardware present, probe failed" instead of a bare "no GPUs".
//
// Cancellation: the caller-supplied ctx governs cancellation. Every source
// derives its working context with context.WithTimeout(ctx,
// gpuProbeTimeout), so the effective deadline is min(caller deadline, 2s) —
// an earlier caller deadline always wins, while the internal cap remains the
// upper bound for callers without one. A cancelled or expired context
// surfaces as an error from the sources that were actually invoked: the
// absent-tool fast paths return their no-error empty result without
// consulting ctx.
func ListGPUDevices(ctx context.Context) ([]GPUDevice, error) {
	if ctx == nil {
		ctx = context.Background()
	}

	var (
		merged   = make([]GPUDevice, 0, 4)
		failures []error
	)

	nvidia, nvidiaErr := listNvidiaGPUDevices(ctx)
	merged = append(merged, nvidia...)
	if nvidiaErr != nil {
		failures = append(failures, nvidiaErr)
	}

	intel, intelErr := listIntelGPUDevices(ctx)
	merged = append(merged, intel...)
	if intelErr != nil {
		failures = append(failures, intelErr)
	}

	if len(merged) == 0 {
		// No source answered with a device: "no GPUs" stays the nil slice
		// (the documented absent-tools shape) when nothing failed, and the
		// joined "hardware present, probe failed" error otherwise.
		if len(failures) > 0 {
			return nil, errors.Join(failures...)
		}
		return nil, nil
	}
	return merged, nil
}

// listNvidiaGPUDevices enumerates the NVIDIA GPUs through nvidia-smi,
// following the result semantics documented on ListGPUDevices.
func listNvidiaGPUDevices(ctx context.Context) ([]GPUDevice, error) {
	bin, ok := nvidiaSmiPath()
	if !ok {
		return nil, nil
	}

	ctx, cancel := context.WithTimeout(ctx, gpuProbeTimeout)
	defer cancel()

	cmd := exec.CommandContext(ctx, bin,
		"--query-gpu=index,name", "--format=csv,noheader")
	// Suppress the console window a GUI-subsystem host would allocate for the
	// child probe process (CREATE_NO_WINDOW on Windows; no-op elsewhere).
	sysproc.HideConsole(cmd)
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("querying nvidia-smi GPU list: %w", err)
	}

	return parseGPUDevices(string(out)), nil
}

// parseGPUDevices extracts (index, name) pairs from the stdout of
// `nvidia-smi --query-gpu=index,name --format=csv,noheader` — normally one
// device per line as "0, NVIDIA GeForce RTX 4070".
//
// The parser is deliberately forgiving, following parseComputeAppPIDs:
// nvidia-smi substitutes placeholder prose for values in several states
// ("No running processes found", "[N/A]", "[Not Supported]"), Windows
// drivers emit \r\n line endings, and blank lines appear around the output.
// Rules, in order: a line must contain the comma separator at all — a valid
// device line always carries both fields, so a bare number or prose without
// a comma is malformed and dropped; take everything up to the first comma as
// the index field and the remainder as the name (GPU product names may
// themselves contain commas), trim whitespace from both, keep the line only
// when the index parses as a non-negative decimal integer. Everything else
// — empty lines, prose, hex, negatives — is dropped, so garbage output
// degrades to "no GPUs" instead of an error. GPU index 0 is the primary
// device and is valid, unlike a PID of 0. Every parsed device is tagged
// with the CUDA provider.
func parseGPUDevices(out string) []GPUDevice {
	lines := strings.Split(out, "\n")
	devices := make([]GPUDevice, 0, len(lines))
	for _, line := range lines {
		indexField, nameField, found := strings.Cut(line, ",")
		if !found {
			continue
		}
		index, err := strconv.Atoi(strings.TrimSpace(indexField))
		if err != nil || index < 0 {
			continue
		}
		devices = append(devices, GPUDevice{
			Index:    index,
			Name:     strings.TrimSpace(nameField),
			Provider: ExecutionProviderCUDA,
		})
	}
	return devices
}
