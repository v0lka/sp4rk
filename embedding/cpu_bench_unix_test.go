//go:build darwin || linux

package embedding

import (
	"syscall"
	"time"
)

func benchmarkCPUTime() time.Duration {
	var usage syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &usage); err != nil {
		return 0
	}
	return timevalDuration(usage.Utime) + timevalDuration(usage.Stime)
}

func timevalDuration(value syscall.Timeval) time.Duration {
	return time.Duration(value.Sec)*time.Second + time.Duration(value.Usec)*time.Microsecond
}
