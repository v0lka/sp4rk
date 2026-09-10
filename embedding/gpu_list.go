package embedding

import (
	"context"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
)

// GPUDevice describes one GPU reported by the NVIDIA driver.
type GPUDevice struct {
	// Index is the driver-assigned ordinal of the device, as used by
	// CUDA_VISIBLE_DEVICES and other ordinal-keyed tooling ("0" is the
	// primary GPU).
	Index int
	// Name is the product name as reported by nvidia-smi ("NVIDIA GeForce
	// RTX 4070"). It is informational only.
	Name string
}

// ListGPUDevices returns the NVIDIA GPUs visible on the machine.
//
// The call shells out to
//
//	nvidia-smi --query-gpu=index,name --format=csv,noheader
//
// bounded by the same gpuProbeTimeout budget as GPUInUse, so a wedged
// driver can never stall the caller.
//
// Result semantics (mirroring GPUInUse):
//   - nvidia-smi not found in PATH → (nil, nil). The machine has no NVIDIA
//     userspace, hence no GPUs to report; absence of the tool is not an
//     error.
//   - nvidia-smi found but the invocation fails (driver down, non-zero
//     exit, timeout) → (nil, err). The tool exists yet could not answer.
//   - Otherwise a device per parseable output line; empty output (or output
//     consisting entirely of prose such as "No running processes found" or
//     "[N/A]") yields an empty, non-nil slice with a nil error.
func ListGPUDevices() ([]GPUDevice, error) {
	bin, ok := nvidiaSmiPath()
	if !ok {
		return nil, nil
	}

	ctx, cancel := context.WithTimeout(context.Background(), gpuProbeTimeout)
	defer cancel()

	out, err := exec.CommandContext(ctx, bin,
		"--query-gpu=index,name", "--format=csv,noheader").Output()
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
// device and is valid, unlike a PID of 0.
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
			Index: index,
			Name:  strings.TrimSpace(nameField),
		})
	}
	return devices
}
