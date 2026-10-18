package embedding

import (
	"context"
	"fmt"
	"os/exec"
	"strconv"
	"strings"

	"github.com/v0lka/sp4rk/sysproc"
)

// intelPCIVendorID is the PCI SIG vendor ID of Intel Corporation. Vulkan
// reports physical devices with their PCI vendor ID, so this constant is
// the filter that reduces `vulkaninfo --summary` output to the Intel
// hardware the OpenVINO execution provider runs on — excluding NVIDIA
// (0x10de) and AMD (0x1002) devices, Mesa software renderers such as
// llvmpipe (0x10005), and any other Vulkan-capable adapter.
const intelPCIVendorID = 0x8086

// listIntelGPUDevices enumerates the Intel GPUs through
// `vulkaninfo --summary`, following the result semantics documented on
// ListGPUDevices:
//   - vulkaninfo not found in PATH → (nil, nil). Absence of the Vulkan
//     tooling means "no Intel GPUs to offer", not a failure.
//   - vulkaninfo found but the invocation fails (no usable ICD, non-zero
//     exit, timeout) → (nil, err). The tool exists yet could not answer.
//   - Otherwise one entry per distinct Intel physical device.
//
// vulkaninfo is the enumeration vehicle because the in-process alternative
// — the OpenVINO plugin's EP metadata, which carries the marketing device
// name — is only reachable after the once-per-process plugin registration
// and exposes no device-name accessor through the onnxruntime_go wrapper;
// a settings picker must stay a side-effect-free external probe. The
// trade-off is the driver's own naming: Mesa's ANV reports chipset names
// ("Intel(R) Graphics (ARL)") rather than OpenVINO's marketing names
// ("Intel(R) Arc(TM) Graphics (iGPU)") — both describe the same hardware,
// and the caller-presented name is always the driver's own.
func listIntelGPUDevices(ctx context.Context) ([]GPUDevice, error) {
	bin, ok := vulkanInfoPath()
	if !ok {
		return nil, nil
	}

	ctx, cancel := context.WithTimeout(ctx, gpuProbeTimeout)
	defer cancel()

	cmd := exec.CommandContext(ctx, bin, "--summary")
	// Suppress the console window a GUI-subsystem host would allocate for the
	// child probe process (CREATE_NO_WINDOW on Windows; no-op elsewhere).
	sysproc.HideConsole(cmd)
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("querying vulkaninfo GPU list: %w", err)
	}

	return parseIntelVulkanDevices(string(out)), nil
}

// vulkanInfoPath resolves the vulkaninfo executable in PATH, mirroring
// nvidiaSmiPath: the boolean is false whenever the binary is unavailable
// for any reason, which callers translate into "no Intel GPUs" rather than
// an error.
func vulkanInfoPath() (string, bool) {
	bin, err := exec.LookPath("vulkaninfo")
	if err != nil {
		return "", false
	}
	return bin, true
}

// parseIntelVulkanDevices extracts the Intel physical devices from the
// stdout of `vulkaninfo --summary`.
//
// The summary's Devices section lists one block per physical device:
//
//	Devices:
//	========
//	GPU0:
//		apiVersion         = 1.4.354
//		vendorID           = 0x8086
//		deviceID           = 0x7d51
//		deviceType         = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
//		deviceName         = Intel(R) Graphics (ARL)
//		...
//
// The parser walks the blocks: a line matching "GPU<ordinal>:" starts a new
// device (resetting the captured fields), "vendorID = 0x…" and
// "deviceName = …" lines capture their values, and a block is flushed as a
// GPUDevice when it ends with both a valid Intel vendor ID and a non-empty
// name. The parser is deliberately forgiving — vulkaninfo versions differ
// in field padding and surrounding prose, unknown fields are ignored, and
// blocks missing either field are dropped, so unfamiliar output degrades to
// "no Intel GPUs" instead of an error.
//
// Every emitted device carries Index 0 and the OpenVINO provider: the
// plugin-based OpenVINO surface has no per-device selection (see
// appendOpenVINOProvider), so device_id 0 — "the provider's best device" —
// is the only selectable index, and the entry exists to tell the picker
// *that* Intel hardware is present and what it is called. Duplicate names
// (the same adapter surfacing through multiple ICDs) collapse to the first
// occurrence, preserving encounter order.
func parseIntelVulkanDevices(out string) []GPUDevice {
	var (
		devices []GPUDevice
		seen    = make(map[string]bool)
		vendor  uint64
		name    string
		haveID  bool
	)

	flush := func() {
		if haveID && vendor == intelPCIVendorID && name != "" && !seen[name] {
			seen[name] = true
			devices = append(devices, GPUDevice{
				Index:    0,
				Name:     name,
				Provider: ExecutionProviderOpenVINO,
			})
		}
		vendor, name, haveID = 0, "", false
	}

	for _, line := range strings.Split(out, "\n") {
		if isVulkanGPUHeader(line) {
			flush()
			continue
		}
		field, value, found := strings.Cut(line, "=")
		if !found {
			continue
		}
		field = strings.Trim(field, " \t\r")
		value = strings.TrimSpace(value)
		switch field {
		case "vendorID":
			if id, err := strconv.ParseUint(strings.TrimPrefix(value, "0x"), 16, 32); err == nil {
				vendor, haveID = id, true
			}
		case "deviceName":
			name = value
		}
	}
	flush()
	return devices
}

// isVulkanGPUHeader reports whether the line starts a new physical-device
// block in vulkaninfo --summary output: exactly "GPU<ordinal>" followed by
// a colon, optionally indented, e.g. "GPU0:" or "GPU12:". Anything else —
// the "Devices:" section banner, prose, field lines — is not a header.
func isVulkanGPUHeader(line string) bool {
	trimmed := strings.TrimSpace(line)
	body, found := strings.CutSuffix(trimmed, ":")
	if !found {
		return false
	}
	digits, ok := strings.CutPrefix(body, "GPU")
	if !ok || digits == "" {
		return false
	}
	for _, r := range digits {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
}
