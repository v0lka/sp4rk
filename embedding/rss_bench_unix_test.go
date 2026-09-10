//go:build darwin || linux

package embedding

import "syscall"

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
