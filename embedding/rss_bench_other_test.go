//go:build !darwin && !linux

package embedding

func benchmarkPeakRSSBytes() uint64 { return 0 }
