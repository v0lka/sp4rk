package embedding

import (
	"context"
	"strings"
	"testing"
	"time"
)

// =============================================================================
// parseGPUDevices — unit tests (no GPU required)
// =============================================================================

func TestParseGPUDevices_SingleGPU(t *testing.T) {
	got := parseGPUDevices("0, NVIDIA GeForce RTX 4070\n")
	want := []GPUDevice{{Index: 0, Name: "NVIDIA GeForce RTX 4070"}}
	if !devicesEqual(got, want) {
		t.Errorf("parseGPUDevices(single) = %v, want %v", got, want)
	}
}

func TestParseGPUDevices_MultipleGPUs(t *testing.T) {
	out := "0, NVIDIA GeForce RTX 4090\n1, NVIDIA GeForce RTX 4090\n2, Tesla T4\n"
	got := parseGPUDevices(out)
	want := []GPUDevice{
		{Index: 0, Name: "NVIDIA GeForce RTX 4090"},
		{Index: 1, Name: "NVIDIA GeForce RTX 4090"},
		{Index: 2, Name: "Tesla T4"},
	}
	if !devicesEqual(got, want) {
		t.Errorf("parseGPUDevices(multi) = %v, want %v", got, want)
	}
}

func TestParseGPUDevices_IndexZeroIsValid(t *testing.T) {
	// GPU ordinal 0 is the primary device — unlike a PID of 0, it must be
	// kept, otherwise every single-GPU machine would report no GPUs.
	got := parseGPUDevices("0, NVIDIA A100\n")
	if len(got) != 1 || got[0].Index != 0 {
		t.Errorf("parseGPUDevices(%q) = %v, want [{0 NVIDIA A100}]", "0, NVIDIA A100\n", got)
	}
}

func TestParseGPUDevices_EmptyAndBlankLines(t *testing.T) {
	for _, out := range []string{"", "\n", "   \n", "\n\n", " \r\n \n"} {
		if got := parseGPUDevices(out); len(got) != 0 {
			t.Errorf("parseGPUDevices(%q) = %v, want empty", out, got)
		}
	}
}

func TestParseGPUDevices_CRLFLineEndings(t *testing.T) {
	// Windows drivers emit \r\n; the trailing \r must not end up in the name
	// (or break the index parse).
	out := "0, NVIDIA GeForce RTX 4070\r\n1, Tesla T4\r\n"
	got := parseGPUDevices(out)
	want := []GPUDevice{
		{Index: 0, Name: "NVIDIA GeForce RTX 4070"},
		{Index: 1, Name: "Tesla T4"},
	}
	if !devicesEqual(got, want) {
		t.Errorf("parseGPUDevices(CRLF) = %v, want %v", got, want)
	}
}

func TestParseGPUDevices_ProseAndPlaceholders(t *testing.T) {
	// nvidia-smi prints placeholder prose instead of values in several
	// states. None of it may parse as a device.
	for _, out := range []string{
		"No running processes found\n",
		"[N/A], [N/A]\n",
		"[Not Supported]\n",
		"No devices were found\n",
	} {
		if got := parseGPUDevices(out); len(got) != 0 {
			t.Errorf("parseGPUDevices(%q) = %v, want empty", out, got)
		}
	}
}

func TestParseGPUDevices_GarbageMixedWithValid(t *testing.T) {
	// Bracket placeholders, hex, negatives, overflow, a bare integer with no
	// comma, and a name-only line: everything unparseable is dropped, the
	// well-formed lines are kept in order.
	out := strings.Join([]string{
		"[Not Supported]",
		"0, NVIDIA GeForce RTX 4070",
		"0x1A, hex",
		"-1, negative index",
		"9999999999999999999999, overflow",
		"7",
		"NVIDIA without index",
		"1, Tesla T4",
		"",
	}, "\n") + "\n"
	got := parseGPUDevices(out)
	want := []GPUDevice{
		{Index: 0, Name: "NVIDIA GeForce RTX 4070"},
		{Index: 1, Name: "Tesla T4"},
	}
	if !devicesEqual(got, want) {
		t.Errorf("parseGPUDevices(garbage) = %v, want %v", got, want)
	}
}

func TestParseGPUDevices_NameWithComma(t *testing.T) {
	// Some marketing names contain commas; the Cut keeps only the first
	// comma as the field separator, so the rest of the line stays in the
	// name verbatim.
	got := parseGPUDevices("0, NVIDIA Graphics Device, rev a1\n")
	want := []GPUDevice{{Index: 0, Name: "NVIDIA Graphics Device, rev a1"}}
	if !devicesEqual(got, want) {
		t.Errorf("parseGPUDevices(comma in name) = %v, want %v", got, want)
	}
}

func TestParseGPUDevices_SurroundingWhitespace(t *testing.T) {
	got := parseGPUDevices("  0 ,   NVIDIA A100-SXM4-40GB  \n")
	want := []GPUDevice{{Index: 0, Name: "NVIDIA A100-SXM4-40GB"}}
	if !devicesEqual(got, want) {
		t.Errorf("parseGPUDevices(whitespace) = %v, want %v", got, want)
	}
}

// =============================================================================
// ListGPUDevices — environment-dependent behaviour
// =============================================================================

// TestListGPUDevices_AbsentNvidiaSmi verifies the "no GPUs, not an error"
// semantics on a machine without nvidia-smi: (nil, nil), no panic, no
// blocking. PATH is cleared because nvidia-smi is looked up in PATH at call
// time (not cached).
func TestListGPUDevices_AbsentNvidiaSmi(t *testing.T) {
	t.Setenv("PATH", "")

	done := make(chan struct{})
	var (
		devices []GPUDevice
		err     error
	)
	go func() {
		defer close(done)
		devices, err = ListGPUDevices(context.Background())
	}()

	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("ListGPUDevices blocked on a machine without nvidia-smi")
	}

	if devices != nil {
		t.Errorf("ListGPUDevices() = %v on machine without nvidia-smi, want nil", devices)
	}
	if err != nil {
		t.Errorf("ListGPUDevices() error = %v, want nil (absent nvidia-smi means no GPUs, not an error)", err)
	}
}

// TestListGPUDevices_LocalCallRunsUnblocked is the runnable-on-any-machine
// smoke test: wherever nvidia-smi exists the call completes quickly;
// wherever it does not, the absent-path returns immediately. Either way the
// context cap must bound the call.
func TestListGPUDevices_LocalCallRunsUnblocked(t *testing.T) {
	start := time.Now()
	devices, err := ListGPUDevices(t.Context())
	elapsed := time.Since(start)

	if elapsed > gpuProbeTimeout+5*time.Second {
		t.Errorf("ListGPUDevices took %v, exceeding probe timeout %v + slack", elapsed, gpuProbeTimeout)
	}
	if err != nil {
		t.Errorf("ListGPUDevices() error = %v, want nil on a healthy machine or one without nvidia-smi", err)
	}
	for _, d := range devices {
		if d.Index < 0 {
			t.Errorf("parsed negative GPU index %d from real nvidia-smi output", d.Index)
		}
	}
	t.Logf("ListGPUDevices() = (%d devices, %v) in %v", len(devices), err, elapsed)
}

// devicesEqual reports whether two device slices are element-wise equal.
// Test-only helper.
func devicesEqual(a, b []GPUDevice) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestListGPUDevices_CanceledContext pins the cancellation contract: on a
// machine with nvidia-smi, a pre-canceled context must surface as an error —
// never a silent (nil, nil) "no GPUs" answer — and must not hang. Skipped
// where nvidia-smi is absent because the fast path returns before ctx is
// ever consulted, so there is nothing to cancel.
func TestListGPUDevices_CanceledContext(t *testing.T) {
	if _, ok := nvidiaSmiPath(); !ok {
		t.Skip("nvidia-smi not found; cancellation path unreachable (absent-tool fast path ignores ctx)")
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	done := make(chan struct{})
	var (
		devices []GPUDevice
		err     error
	)
	go func() {
		defer close(done)
		devices, err = ListGPUDevices(ctx)
	}()

	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("ListGPUDevices hung on a pre-canceled context")
	}

	if err == nil {
		t.Errorf("ListGPUDevices(canceled ctx) = (%v, nil), want an error; cancellation must not silently report an empty GPU list", devices)
	}
	if devices != nil {
		t.Errorf("ListGPUDevices(canceled ctx) = %v, want nil", devices)
	}
}

// TestListGPUDevices_ShortCallerDeadline verifies budget composition: a
// caller deadline earlier than the internal 2s cap wins. The call must
// complete well inside the internal cap and the expiry must surface as an
// error — min(caller deadline, gpuProbeTimeout) is what bounds the probe.
func TestListGPUDevices_ShortCallerDeadline(t *testing.T) {
	if _, ok := nvidiaSmiPath(); !ok {
		t.Skip("nvidia-smi not found; cancellation path unreachable (absent-tool fast path ignores ctx)")
	}

	ctx, cancel := context.WithTimeout(context.Background(), time.Millisecond)
	defer cancel()

	start := time.Now()
	devices, err := ListGPUDevices(ctx)
	elapsed := time.Since(start)

	if elapsed > time.Second {
		t.Errorf("ListGPUDevices took %v with a 1ms caller deadline; min(caller, 2s) budget not respected", elapsed)
	}
	if err == nil {
		t.Errorf("ListGPUDevices(1ms deadline) = (%v, nil), want an error; an expired context must surface as an error", devices)
	}
	if devices != nil {
		t.Errorf("ListGPUDevices(1ms deadline) = %v, want nil", devices)
	}
}
