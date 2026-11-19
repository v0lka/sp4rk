//go:build darwin || linux

package embedding

import (
	"context"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"syscall"
)

func benchmarkPeakRSSBytes() uint64 {
	var usage syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &usage); err != nil {
		return 0
	}
	peak := uint64(usage.Maxrss) //nolint:gosec // Maxrss is non-negative by OS contract.
	if peak == 0 {
		return 0
	}
	// Linux reports KiB; Darwin reports bytes.
	if isLinuxBenchmarkRSS {
		return peak * 1024
	}
	return peak
}

// benchmarkCurrentRSSBytes returns this process's resident set now (0 when
// unavailable). getrusage only exposes the high-water mark, so it asks ps,
// which reports KiB on both Darwin and Linux. It is process-wide: earlier
// benchmarks in the same process leave residue, so compare values only across
// isolated per-case processes.
func benchmarkCurrentRSSBytes() uint64 {
	out, err := exec.CommandContext(context.Background(), "ps", "-o", "rss=", "-p", strconv.Itoa(os.Getpid())).Output()
	if err != nil {
		return 0
	}
	kib, err := strconv.ParseUint(strings.TrimSpace(string(out)), 10, 64)
	if err != nil {
		return 0
	}
	return kib * 1024
}
