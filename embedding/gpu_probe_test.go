package embedding

import (
	"context"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"testing"
	"time"
)

// =============================================================================
// parseComputeAppPIDs — unit tests (no GPU required)
// =============================================================================

func TestParseComputeAppPIDs_CleanSinglePID(t *testing.T) {
	got := parseComputeAppPIDs("1925\n")
	if len(got) != 1 || got[0] != 1925 {
		t.Errorf("parseComputeAppPIDs(%q) = %v, want [1925]", "1925\n", got)
	}
}

func TestParseComputeAppPIDs_MultiplePIDs(t *testing.T) {
	out := "1925\n354075\n42\n"
	got := parseComputeAppPIDs(out)
	want := []int{1925, 354075, 42}
	if len(got) != len(want) {
		t.Fatalf("parseComputeAppPIDs(%q) = %v, want %v", out, got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("parseComputeAppPIDs(%q)[%d] = %d, want %d", out, i, got[i], want[i])
		}
	}
}

func TestParseComputeAppPIDs_Empty(t *testing.T) {
	// The single most common real-world output: zero compute apps.
	for _, out := range []string{"", "\n", "   \n", "\n\n"} {
		if got := parseComputeAppPIDs(out); len(got) != 0 {
			t.Errorf("parseComputeAppPIDs(%q) = %v, want empty", out, got)
		}
	}
}

func TestParseComputeAppPIDs_NoRunningProcessesProse(t *testing.T) {
	// nvidia-smi prints this phrase instead of numbers when no compute app
	// is registered. It must not parse as a PID and must not error.
	out := "No running processes found\n"
	if got := parseComputeAppPIDs(out); len(got) != 0 {
		t.Errorf("parseComputeAppPIDs(%q) = %v, want empty", out, got)
	}
}

func TestParseComputeAppPIDs_CRLFLineEndings(t *testing.T) {
	// Windows drivers emit \r\n; the trailing \r must be trimmed away before
	// Atoi, otherwise "1925\r" fails to parse.
	got := parseComputeAppPIDs("1925\r\n354075\r\n")
	want := []int{1925, 354075}
	if len(got) != len(want) {
		t.Fatalf("parseComputeAppPIDs(CRLF) = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("parseComputeAppPIDs(CRLF)[%d] = %d, want %d", i, got[i], want[i])
		}
	}
}

func TestParseComputeAppPIDs_Garbage(t *testing.T) {
	// Placeholder prose, brackets, hex, negative and overflowing numbers,
	// mixed with valid PIDs: everything unparseable is dropped, the rest kept.
	out := "[Not Supported]\n1925\n0x1A\n-5\n[N/A]\n9999999999999999999999\n7\n"
	got := parseComputeAppPIDs(out)
	want := []int{1925, 7}
	if len(got) != len(want) {
		t.Fatalf("parseComputeAppPIDs(garbage) = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("parseComputeAppPIDs(garbage)[%d] = %d, want %d", i, got[i], want[i])
		}
	}
}

func TestParseComputeAppPIDs_ExtraColumns(t *testing.T) {
	// Wider queries append columns after the pid; only the first field counts.
	out := "1925, /usr/bin/Telegram, 512 MiB\n354075, /opt/c0wrk/c0wrk-desktop, 2048 MiB\n"
	got := parseComputeAppPIDs(out)
	want := []int{1925, 354075}
	if len(got) != len(want) {
		t.Fatalf("parseComputeAppPIDs(extra columns) = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("parseComputeAppPIDs(extra columns)[%d] = %d, want %d", i, got[i], want[i])
		}
	}
}

func TestParseComputeAppPIDs_ZeroPIDRejected(t *testing.T) {
	// PID 0 is never a valid compute app; it must be dropped so a garbage
	// "0" line can never match an (impossible) own pid of 0.
	if got := parseComputeAppPIDs("0\n"); len(got) != 0 {
		t.Errorf("parseComputeAppPIDs(%q) = %v, want empty", "0\n", got)
	}
}

// =============================================================================
// GPUInUse — environment-dependent behaviour
// =============================================================================

// TestGPUInUse_AbsentNvidiaSmi verifies the "not verified" semantics on a
// machine without nvidia-smi: (false, nil), no panic, and no blocking. It
// forces the condition by clearing PATH, since nvidia-smi is looked up in
// PATH at call time (not cached), and a PATH-less exec.LookPath fails on
// every platform.
func TestGPUInUse_AbsentNvidiaSmi(t *testing.T) {
	t.Setenv("PATH", "")

	done := make(chan struct{})
	var (
		inUse bool
		err   error
	)
	go func() {
		defer close(done)
		inUse, err = GPUInUse(context.Background())
	}()

	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("GPUInUse blocked on a machine without nvidia-smi")
	}

	if inUse {
		t.Error("GPUInUse() = true on machine without nvidia-smi, want false")
	}
	if err != nil {
		t.Errorf("GPUInUse() error = %v, want nil (absent nvidia-smi is 'not verified', not an error)", err)
	}
}

// TestGPUInUse_LocalProbeRunsUnblocked is the runnable-on-any-machine smoke
// test: wherever nvidia-smi exists the probe completes quickly; wherever it
// does not, the absent-path returns immediately. Both branches must return
// within a generous multiple of the probe timeout — proving the context cap
// actually bounds the call.
func TestGPUInUse_LocalProbeRunsUnblocked(t *testing.T) {
	start := time.Now()
	inUse, err := GPUInUse(t.Context())
	elapsed := time.Since(start)

	if elapsed > gpuProbeTimeout+5*time.Second {
		t.Errorf("GPUInUse took %v, exceeding probe timeout %v + slack", elapsed, gpuProbeTimeout)
	}
	t.Logf("GPUInUse() = (%v, %v) in %v", inUse, err, elapsed)
}

// TestGPUInUse_FindsOwnPIDWhenListed pins the PID-comparison logic without a
// GPU: the comparison used by GPUInUse is literally parse + os.Getpid
// equality, so a fake stdout containing the own PID must match, and one with
// only foreign PIDs must not.
func TestGPUInUse_FindsOwnPIDWhenListed(t *testing.T) {
	own := os.Getpid()
	foreign := own + 1

	if got := parseComputeAppPIDs(strconv.Itoa(own) + "\n"); !containsPID(got, own) {
		t.Errorf("own PID %d not found in parsed %v", own, got)
	}

	if got := parseComputeAppPIDs(strconv.Itoa(foreign) + "\n"); containsPID(got, own) {
		t.Errorf("foreign PID %d wrongly matched own PID %d in %v", foreign, own, got)
	}
}

// containsPID reports whether pids contains pid. Test-only helper mirroring
// the membership check GPUInUse performs inline.
func containsPID(pids []int, pid int) bool {
	for _, p := range pids {
		if p == pid {
			return true
		}
	}
	return false
}

// TestGPUInUse_CanceledContext pins the cancellation contract: on a machine
// with nvidia-smi, a pre-canceled context must surface as an error — never a
// silent (false, nil) "not on GPU" answer — and must not hang. Skipped where
// nvidia-smi is absent because the fast path returns before ctx is ever
// consulted, so there is nothing to cancel.
func TestGPUInUse_CanceledContext(t *testing.T) {
	if _, ok := nvidiaSmiPath(); !ok {
		t.Skip("nvidia-smi not found; cancellation path unreachable (absent-tool fast path ignores ctx)")
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	done := make(chan struct{})
	var (
		inUse bool
		err   error
	)
	go func() {
		defer close(done)
		inUse, err = GPUInUse(ctx)
	}()

	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("GPUInUse hung on a pre-canceled context")
	}

	if err == nil {
		t.Errorf("GPUInUse(canceled ctx) = (%v, nil), want an error; cancellation must surface as 'unverified', never a silent false", inUse)
	}
	if inUse {
		t.Error("GPUInUse(canceled ctx) = true, want false")
	}
}

// TestGPUInUse_ShortCallerDeadline verifies budget composition: a caller
// deadline earlier than the internal 2s cap wins. The call must complete
// well inside the internal cap and the expiry must surface as an error —
// min(caller deadline, gpuProbeTimeout) is what bounds the probe.
func TestGPUInUse_ShortCallerDeadline(t *testing.T) {
	if _, ok := nvidiaSmiPath(); !ok {
		t.Skip("nvidia-smi not found; cancellation path unreachable (absent-tool fast path ignores ctx)")
	}

	ctx, cancel := context.WithTimeout(context.Background(), time.Millisecond)
	defer cancel()

	start := time.Now()
	inUse, err := GPUInUse(ctx)
	elapsed := time.Since(start)

	if elapsed > time.Second {
		t.Errorf("GPUInUse took %v with a 1ms caller deadline; min(caller, 2s) budget not respected", elapsed)
	}
	if err == nil {
		t.Errorf("GPUInUse(1ms deadline) = (%v, nil), want an error; an expired context must surface as 'unverified'", inUse)
	}
	if inUse {
		t.Error("GPUInUse(1ms deadline) = true, want false")
	}
}

// =============================================================================
// GPUInUse — live nvidia-smi / real-CUDA integration tests
// (skip cleanly where the hardware or assets are absent)
// =============================================================================

// TestGPUInUse_ParseMatchesRealNvidiaSmiOutput guards against drift between
// the parser and a live nvidia-smi on machines that have one: the real tool
// must yield a well-formed PID list the parser accepts, and — because the
// test process itself never creates a CUDA context — must not contain our
// PID.
func TestGPUInUse_ParseMatchesRealNvidiaSmiOutput(t *testing.T) {
	bin, ok := nvidiaSmiPath()
	if !ok {
		t.Skip("nvidia-smi not found; skipping live-output parse test")
	}

	ctx, cancel := context.WithTimeout(context.Background(), gpuProbeTimeout)
	defer cancel()

	out, err := exec.CommandContext(ctx, bin,
		"--query-compute-apps=pid", "--format=csv,noheader").Output()
	if err != nil {
		t.Skipf("nvidia-smi failed on this machine (%v); skipping live-output parse test", err)
	}

	pids := parseComputeAppPIDs(string(out))
	for _, pid := range pids {
		if pid <= 0 {
			t.Errorf("parsed non-positive PID %d from real nvidia-smi output %q", pid, string(out))
		}
	}

	// This test itself never creates a CUDA context — but the shared-process
	// suite may already have driven a CUDA inference in another test, and a
	// CUDA context lives until the process ends. Only assert our own PID is
	// absent while no test in this process has touched CUDA.
	if !processCreatedCUDAContext.Load() && containsPID(pids, os.Getpid()) {
		t.Errorf("own PID %d unexpectedly listed by nvidia-smi without any CUDA inference in this process", os.Getpid())
	}
	t.Logf("live nvidia-smi compute apps: %v (raw %q)", pids, strings.TrimSpace(string(out)))
}

// TestGPUInUse_AfterRealCUDAInference is the end-to-end acceptance test for
// criterion (b): on a machine with nvidia-smi and a CUDA-capable ONNX Runtime
// build, after a real CUDA inference through the embedder the driver must
// list our own PID among its compute applications.
//
// Gated by the same env vars as the other ONNX tests plus
// EMBEDDING_TEST_PROVIDER=cuda:
//
//	EMBEDDING_TEST_MODEL_PATH=/path/jina-v2-small.onnx \
//	EMBEDDING_TEST_TOKENIZER_PATH=/path/tokenizer.json \
//	EMBEDDING_TEST_LIBRARY_PATH=/path/libonnxruntime.so \
//	EMBEDDING_TEST_PROVIDER=cuda \
//	go test -run TestGPUInUse_AfterRealCUDAInference -v ./embedding
//
// The test is skipped on machines where nvidia-smi is absent even when all
// other variables are set: there the signal cannot be produced at all.
//
// The probe runs only after a real inference because a CUDA context is
// created lazily: probing right after session creation could see a PID list
// not yet containing ours and produce a false negative.
func TestGPUInUse_AfterRealCUDAInference(t *testing.T) {
	if os.Getenv("EMBEDDING_TEST_PROVIDER") != ExecutionProviderCUDA {
		t.Skip("EMBEDDING_TEST_PROVIDER != cuda; skipping GPU integration test")
	}
	if _, ok := nvidiaSmiPath(); !ok {
		t.Skip("nvidia-smi not found; GPU signal unverifiable on this machine")
	}

	libPath := testLibraryPath(t)
	tokPath := testTokenizerPath(t)
	modelPath := testModelPath(t)

	emb, err := NewEmbedder(EmbedderConfig{
		ModelPath:         modelPath,
		TokenizerPath:     tokPath,
		LibraryPath:       libPath,
		ExecutionProvider: ExecutionProviderCUDA,
	})
	if err != nil {
		t.Fatalf("NewEmbedder(cuda) error = %v", err)
	}
	// Close the session only; the process-global ONNX environment is owned by
	// TestMain and shared across all tests, so Embedder.Close (which destroys
	// it) must not be used here.
	defer closeSessionOnly(emb)

	// Real CUDA inference: EmbedDocuments drives a session.Run through the
	// CUDA execution provider, which creates the CUDA context and registers
	// the process with the driver.
	if _, err := emb.EmbedDocuments(context.Background(),
		[]string{"gpu probe integration test"}); err != nil {
		t.Fatalf("EmbedDocuments under CUDA failed: %v", err)
	}

	inUse, err := GPUInUse(t.Context())
	if err != nil {
		t.Fatalf("GPUInUse() error = %v", err)
	}
	if !inUse {
		t.Fatal("GPUInUse() = false after real CUDA inference; own PID not found among compute apps")
	}
}
