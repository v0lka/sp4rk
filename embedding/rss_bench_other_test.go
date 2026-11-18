//go:build !darwin && !linux

package embedding

func benchmarkPeakRSSBytes() uint64 { return 0 }

func benchmarkCurrentRSSBytes() uint64 { return 0 }
