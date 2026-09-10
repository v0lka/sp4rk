//go:build !darwin && !linux

package embedding

import "time"

func benchmarkCPUTime() time.Duration { return 0 }
