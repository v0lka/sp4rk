package embedding

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

// realSummary is the verbatim `vulkaninfo --summary` Devices section from
// the reference Arrow Lake iGPU machine (Intel vendor 0x8086, Mesa ANV).
const realSummary = `Devices:
========
GPU0:
	apiVersion         = 1.4.354
	driverVersion      = 26.2.2
	vendorID           = 0x8086
	deviceID           = 0x7d51
	deviceType         = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
	deviceName         = Intel(R) Graphics (ARL)
	driverID           = DRIVER_ID_INTEL_OPEN_SOURCE_MESA
	driverName         = Intel open-source Mesa driver
	driverInfo         = Mesa 26.2.2-arch1.1
	conformanceVersion = 1.4.0.0
	deviceUUID         = 8680517d-0300-0000-0002-000000000000
	driverUUID         = af44288b-59dd-a79f-c93d-b6e4e1ea7f6f
`

func TestParseIntelVulkanDevices_RealSummary(t *testing.T) {
	got := parseIntelVulkanDevices(realSummary)
	want := []GPUDevice{{Index: 0, Name: "Intel(R) Graphics (ARL)", Provider: ExecutionProviderOpenVINO}}
	if !devicesEqual(got, want) {
		t.Errorf("parseIntelVulkanDevices(real) = %v, want %v", got, want)
	}
}

func TestParseIntelVulkanDevices_FiltersNonIntelVendors(t *testing.T) {
	// A desktop box with an NVIDIA dGPU and a Mesa software renderer: only
	// the Intel device may survive the vendor filter.
	out := `Devices:
========
GPU0:
	vendorID           = 0x10de
	deviceName         = NVIDIA GeForce RTX 4070
GPU1:
	vendorID           = 0x8086
	deviceName         = Intel(R) Arc(TM) B580
GPU2:
	vendorID           = 0x10005
	deviceName         = llvmpipe (LLVM 19.1.7, 256 bits)
GPU3:
	vendorID           = 0x1002
	deviceName         = AMD Radeon RX 7900 XT
`
	got := parseIntelVulkanDevices(out)
	want := []GPUDevice{{Index: 0, Name: "Intel(R) Arc(TM) B580", Provider: ExecutionProviderOpenVINO}}
	if !devicesEqual(got, want) {
		t.Errorf("parseIntelVulkanDevices(multi-vendor) = %v, want %v", got, want)
	}
}

func TestParseIntelVulkanDevices_MultipleIntelDevices(t *testing.T) {
	out := `Devices:
========
GPU0:
	vendorID           = 0x8086
	deviceName         = Intel(R) Arc(TM) Graphics (iGPU)
GPU1:
	vendorID           = 0x8086
	deviceName         = Intel(R) Arc(TM) Pro B60
`
	got := parseIntelVulkanDevices(out)
	if len(got) != 2 {
		t.Fatalf("parseIntelVulkanDevices(two Intel) = %v, want 2 devices", got)
	}
	for i, d := range got {
		if d.Index != 0 {
			t.Errorf("device %d index = %d, want 0 (the OpenVINO surface has no per-device selection)", i, d.Index)
		}
		if d.Provider != ExecutionProviderOpenVINO {
			t.Errorf("device %d provider = %q, want %q", i, d.Provider, ExecutionProviderOpenVINO)
		}
	}
}

func TestParseIntelVulkanDevices_DuplicateNamesCollapse(t *testing.T) {
	// The same adapter surfacing through two ICDs must collapse to the
	// first occurrence.
	out := `Devices:
========
GPU0:
	vendorID           = 0x8086
	deviceName         = Intel(R) Graphics (ARL)
GPU1:
	vendorID           = 0x8086
	deviceName         = Intel(R) Graphics (ARL)
`
	got := parseIntelVulkanDevices(out)
	want := []GPUDevice{{Index: 0, Name: "Intel(R) Graphics (ARL)", Provider: ExecutionProviderOpenVINO}}
	if !devicesEqual(got, want) {
		t.Errorf("parseIntelVulkanDevices(duplicate) = %v, want %v", got, want)
	}
}

func TestParseIntelVulkanDevices_IncompleteBlocksDropped(t *testing.T) {
	// A block missing deviceName, one missing vendorID, and one with an
	// unparseable vendorID: all dropped, nothing malformed leaks through.
	out := `Devices:
========
GPU0:
	vendorID           = 0x8086
	deviceType         = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
GPU1:
	deviceName         = Intel(R) Graphics (no vendor id)
GPU2:
	vendorID           = vendor-str
	deviceName         = Intel(R) Graphics (bad vendor id)
`
	if got := parseIntelVulkanDevices(out); len(got) != 0 {
		t.Errorf("parseIntelVulkanDevices(incomplete) = %v, want empty", got)
	}
}

func TestParseIntelVulkanDevices_NoDevicesSection(t *testing.T) {
	// vulkaninfo prints extension/layer inventories before Devices; none of
	// that may produce a device. An empty output is equally empty.
	prose, _, _ := strings.Cut(realSummary, "GPU0:")
	for _, out := range []string{"", prose} {
		if got := parseIntelVulkanDevices(out); len(got) != 0 {
			t.Errorf("parseIntelVulkanDevices(%q) = %v, want empty", out, got)
		}
	}
}

func TestParseIntelVulkanDevices_CRLFAndPadding(t *testing.T) {
	// Windows vulkaninfo emits \r\n; field padding around "=" varies by
	// version. Both must parse identically.
	out := "GPU0:\r\n  vendorID=0x8086\r\n  deviceName     =  Intel(R) Arc(TM) Graphics (iGPU)  \r\n"
	got := parseIntelVulkanDevices(out)
	want := []GPUDevice{{Index: 0, Name: "Intel(R) Arc(TM) Graphics (iGPU)", Provider: ExecutionProviderOpenVINO}}
	if !devicesEqual(got, want) {
		t.Errorf("parseIntelVulkanDevices(CRLF) = %v, want %v", got, want)
	}
}

func TestIsVulkanGPUHeader(t *testing.T) {
	yes := []string{"GPU0:", "GPU12:", "\tGPU3:", "  GPU7:  ", "GPU0:\r"}
	no := []string{"GPU:", "GPU0x:", "GPU0", "Devices:", "========", "vendorID           = 0x8086", "GPU:", "", "gpu0:"}
	for _, line := range yes {
		if !isVulkanGPUHeader(line) {
			t.Errorf("isVulkanGPUHeader(%q) = false, want true", line)
		}
	}
	for _, line := range no {
		if isVulkanGPUHeader(line) {
			t.Errorf("isVulkanGPUHeader(%q) = true, want false", line)
		}
	}
}

// installFakeTool writes an executable shell script impersonating tool and
// prepends its directory to PATH, so the shim shadows any real tool of the
// same name while previously installed shims (other tools) stay reachable —
// a test may install several. The shim approach requires a POSIX shell: on
// Windows (and any GOOS without /bin/sh) the exec-based tests are skipped —
// parsing and absent-tool semantics are covered without shims; the shimmed
// exec path additionally runs on the Linux and macOS CI legs.
func installFakeTool(t *testing.T, tool, script string) string {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skipf("shell-script shims of %s require a POSIX shell; exec path covered on linux/darwin CI legs", tool)
	}
	dir := t.TempDir()
	path := filepath.Join(dir, tool)
	if err := os.WriteFile(path, []byte("#!/bin/sh\n"+script), 0o755); err != nil {
		t.Fatalf("writing fake %s: %v", tool, err)
	}
	t.Setenv("PATH", dir+string(os.PathListSeparator)+os.Getenv("PATH"))
	return path
}

func TestListIntelGPUDevices_ShimmedVulkaninfo(t *testing.T) {
	installFakeTool(t, "vulkaninfo", "cat <<'EOF'\n"+realSummary+"EOF\n")
	got, err := listIntelGPUDevices(context.Background())
	if err != nil {
		t.Fatalf("listIntelGPUDevices() error = %v", err)
	}
	want := []GPUDevice{{Index: 0, Name: "Intel(R) Graphics (ARL)", Provider: ExecutionProviderOpenVINO}}
	if !devicesEqual(got, want) {
		t.Errorf("listIntelGPUDevices() = %v, want %v", got, want)
	}
}

func TestListIntelGPUDevices_AbsentTool(t *testing.T) {
	t.Setenv("PATH", "")
	devices, err := listIntelGPUDevices(context.Background())
	if devices != nil {
		t.Errorf("listIntelGPUDevices() = %v without vulkaninfo, want nil", devices)
	}
	if err != nil {
		t.Errorf("listIntelGPUDevices() error = %v without vulkaninfo, want nil (absent tool means no GPUs, not a failure)", err)
	}
}

func TestListIntelGPUDevices_FailingTool(t *testing.T) {
	installFakeTool(t, "vulkaninfo", "echo 'Cannot create Vulkan instance' >&2\nexit 1\n")
	devices, err := listIntelGPUDevices(context.Background())
	if devices != nil {
		t.Errorf("listIntelGPUDevices() = %v on probe failure, want nil", devices)
	}
	if err == nil {
		t.Fatal("listIntelGPUDevices() error = nil on probe failure, want the wrapped error")
	}
	if !strings.Contains(err.Error(), "vulkaninfo") {
		t.Errorf("error %q does not name the failing probe", err.Error())
	}
}

func TestListGPUDevices_MergeBothSources(t *testing.T) {
	// A hybrid machine: nvidia-smi reports the dGPU, vulkaninfo the iGPU.
	// Both must appear, NVIDIA first, each tagged with its provider.
	installFakeTool(t, "nvidia-smi", "echo '0, NVIDIA GeForce RTX 4070'\n")
	installFakeTool(t, "vulkaninfo", "cat <<'EOF'\n"+realSummary+"EOF\n")

	got, err := ListGPUDevices(context.Background())
	if err != nil {
		t.Fatalf("ListGPUDevices() error = %v", err)
	}
	want := []GPUDevice{
		{Index: 0, Name: "NVIDIA GeForce RTX 4070", Provider: ExecutionProviderCUDA},
		{Index: 0, Name: "Intel(R) Graphics (ARL)", Provider: ExecutionProviderOpenVINO},
	}
	if !devicesEqual(got, want) {
		t.Errorf("ListGPUDevices() = %v, want %v", got, want)
	}
}

func TestListGPUDevices_PartialFailureKeepsHealthySource(t *testing.T) {
	// nvidia-smi present but broken; vulkaninfo healthy. The working
	// source must not be hidden behind the broken one.
	installFakeTool(t, "nvidia-smi", "exit 1\n")
	installFakeTool(t, "vulkaninfo", "cat <<'EOF'\n"+realSummary+"EOF\n")

	got, err := ListGPUDevices(context.Background())
	if err != nil {
		t.Fatalf("ListGPUDevices() error = %v, want nil (partial answer beats hiding working hardware)", err)
	}
	want := []GPUDevice{{Index: 0, Name: "Intel(R) Graphics (ARL)", Provider: ExecutionProviderOpenVINO}}
	if !devicesEqual(got, want) {
		t.Errorf("ListGPUDevices() = %v, want %v", got, want)
	}
}

func TestListGPUDevices_AllSourcesFailed(t *testing.T) {
	// Both tools present and both failing: "hardware present, probe failed"
	// — the joined error, never a bare empty list.
	installFakeTool(t, "nvidia-smi", "exit 3\n")
	installFakeTool(t, "vulkaninfo", "exit 4\n")
	got, err := ListGPUDevices(context.Background())
	if got != nil {
		t.Errorf("ListGPUDevices() = %v when every source failed, want nil", got)
	}
	if err == nil {
		t.Fatal("ListGPUDevices() error = nil when every present source failed, want the joined error")
	}
	if !strings.Contains(err.Error(), "nvidia-smi") || !strings.Contains(err.Error(), "vulkaninfo") {
		t.Errorf("joined error %q must name every failed source", err.Error())
	}
}

func TestListGPUDevices_AbsentSourcesNoError(t *testing.T) {
	// Neither tool on PATH: "no GPUs, not an error" — the merged contract
	// of both absent sources.
	t.Setenv("PATH", "")
	got, err := ListGPUDevices(context.Background())
	if got != nil {
		t.Errorf("ListGPUDevices() = %v with no sources, want nil", got)
	}
	if err != nil {
		t.Errorf("ListGPUDevices() error = %v with no sources, want nil", err)
	}
}

// TestListGPUDevices_ShimmedMergeUnblocked complements the real-machine
// smoke test: through the shimmed exec path the merged call completes well
// inside the probe budget.
func TestListGPUDevices_ShimmedMergeUnblocked(t *testing.T) {
	installFakeTool(t, "vulkaninfo", "cat <<'EOF'\n"+realSummary+"EOF\n")
	start := time.Now()
	got, err := ListGPUDevices(context.Background())
	elapsed := time.Since(start)
	if err != nil {
		t.Fatalf("ListGPUDevices() error = %v", err)
	}
	if elapsed > gpuProbeTimeout+5*time.Second {
		t.Errorf("ListGPUDevices took %v, exceeding probe timeout %v + slack", elapsed, gpuProbeTimeout)
	}
	if len(got) != 1 || got[0].Provider != ExecutionProviderOpenVINO {
		t.Errorf("ListGPUDevices() = %v, want the single shimmed Intel device", got)
	}
}
