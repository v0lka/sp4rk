package embedding

import (
	"fmt"
	"strings"
	"testing"

	ort "github.com/yalue/onnxruntime_go"
)

// restoreOpenVINOPreference snapshots the package-wide resolved device and
// restores it when the test ends: the value is process-global (like the ORT
// environment it belongs to), so a test that moves it would otherwise leak
// into every later test in the package.
func restoreOpenVINOPreference(t *testing.T) {
	t.Helper()
	previous := openVINOPreferredDevice.Load()
	t.Cleanup(func() { openVINOPreferredDevice.Store(previous) })
}

// TestNarrowOpenVINODevices pins the append-time device selection: an
// unresolved (-1) or out-of-range preference appends the whole list (the
// plugin then decides), while a resolved preference appends exactly that
// device so a later session goes straight to the device that already worked.
func TestNarrowOpenVINODevices(t *testing.T) {
	t.Parallel()
	devices := []ort.EpDevice{{}, {}, {}}
	for _, tc := range []struct {
		name      string
		preferred int
		want      int
	}{
		{name: "unresolved appends all", preferred: -1, want: 3},
		{name: "negative out of range appends all", preferred: -7, want: 3},
		{name: "first device", preferred: 0, want: 1},
		{name: "last device", preferred: 2, want: 1},
		{name: "above the range appends all", preferred: 3, want: 3},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			if got := len(narrowOpenVINODevices(devices, tc.preferred)); got != tc.want {
				t.Errorf("narrowOpenVINODevices(preferred=%d) kept %d devices, want %d", tc.preferred, got, tc.want)
			}
		})
	}
}

// TestOpenVINOAttemptedDevice pins which index counts as "already tried": the
// resolved preference, or 0 while the preference is unresolved (the plugin
// takes the first device of the appended list).
func TestOpenVINOAttemptedDevice(t *testing.T) {
	restoreOpenVINOPreference(t)
	openVINOPreferredDevice.Store(-1)
	if got := openVINOAttemptedDevice(); got != 0 {
		t.Errorf("openVINOAttemptedDevice() with an unresolved preference = %d, want 0", got)
	}
	openVINOPreferredDevice.Store(2)
	if got := openVINOAttemptedDevice(); got != 2 {
		t.Errorf("openVINOAttemptedDevice() = %d, want 2", got)
	}
}

// TestRetryOpenVINODevices pins the device walk behind the OpenVINO rung: the
// device that just failed is skipped, the remaining devices are attempted in
// enumeration order until one creates a session, every failed candidate is
// reported, and the winner becomes the process-wide preference so later
// sessions append only that device.
func TestRetryOpenVINODevices(t *testing.T) {
	restoreOpenVINOPreference(t)

	type call struct {
		attempted []int
		installed []int
		reasons   []string
		warnings  []string
	}

	// run drives retryOpenVINODevices with fakes: sessFails names the device
	// indexes whose session creation must fail, optsFail those whose options
	// must fail.
	run := func(deviceCount, failed int, sessFails, optsFail map[int]bool) (*call, bool) {
		openVINOPreferredDevice.Store(-1)
		var c call
		sess, ok := retryOpenVINODevices(deviceCount, failed,
			func(provider string) (*ort.SessionOptions, error) {
				if provider != ExecutionProviderOpenVINO {
					t.Fatalf("buildOpts(provider=%q), want %q", provider, ExecutionProviderOpenVINO)
				}
				device := openVINOAttemptedDevice()
				c.attempted = append(c.attempted, device)
				if optsFail[device] {
					return nil, fmt.Errorf("device %d: options failed", device)
				}
				return &ort.SessionOptions{}, nil
			},
			func(*ort.SessionOptions) {
				c.installed = append(c.installed, openVINOAttemptedDevice())
			},
			func() (*onnxSession, error) {
				device := openVINOAttemptedDevice()
				if sessFails[device] {
					return nil, fmt.Errorf("device %d: session failed", device)
				}
				return &onnxSession{}, nil
			},
			func(msg, provider, cause string) {
				if provider != ExecutionProviderOpenVINO {
					t.Errorf("warn provider = %q, want %q", provider, ExecutionProviderOpenVINO)
				}
				c.warnings = append(c.warnings, msg+": "+cause)
			},
			&c.reasons,
		)
		if ok && sess == nil {
			t.Error("retryOpenVINODevices() reported success with a nil session")
		}
		return &c, ok
	}

	t.Run("takes the next device when the first cannot compile", func(t *testing.T) {
		c, ok := run(3, 0, map[int]bool{1: true}, nil)
		if !ok {
			t.Fatal("retryOpenVINODevices() ok = false, want true (device 2 works)")
		}
		if want := []int{1, 2}; !equalInts(c.attempted, want) {
			t.Errorf("attempted devices = %v, want %v (device 0 was already tried)", c.attempted, want)
		}
		if want := []int{1, 2}; !equalInts(c.installed, want) {
			t.Errorf("installed options for devices = %v, want %v", c.installed, want)
		}
		if len(c.reasons) != 1 || !strings.Contains(c.reasons[0], "device 1") {
			t.Errorf("reasons = %v, want only the failed device 1 recorded", c.reasons)
		}
		if len(c.warnings) != 1 {
			t.Errorf("warnings = %v, want one line for the failed device", c.warnings)
		}
		if got := int(openVINOPreferredDevice.Load()); got != 2 {
			t.Errorf("preferred device = %d, want 2 (the device that created the session)", got)
		}
	})

	t.Run("walks every remaining device and fails when none works", func(t *testing.T) {
		c, ok := run(3, 1, map[int]bool{0: true, 1: true, 2: true}, nil)
		if ok {
			t.Fatal("retryOpenVINODevices() ok = true, want false (no device can run the model)")
		}
		if want := []int{0, 2}; !equalInts(c.attempted, want) {
			t.Errorf("attempted devices = %v, want %v (the failed device 1 is skipped)", c.attempted, want)
		}
		if len(c.reasons) != len(c.attempted) {
			t.Errorf("reasons = %d entries, want one per attempted device (%d)", len(c.reasons), len(c.attempted))
		}
		if got := int(openVINOPreferredDevice.Load()); got != -1 {
			t.Errorf("preferred device = %d, want -1 (a failed walk must not pin a device)", got)
		}
	})

	t.Run("skips an options-stage failure and keeps walking", func(t *testing.T) {
		c, ok := run(3, 0, nil, map[int]bool{1: true})
		if !ok {
			t.Fatal("retryOpenVINODevices() ok = false, want true (device 2 works)")
		}
		if want := []int{1, 2}; !equalInts(c.attempted, want) {
			t.Errorf("attempted devices = %v, want %v", c.attempted, want)
		}
		if want := []int{2}; !equalInts(c.installed, want) {
			t.Errorf("installed options for devices = %v, want %v (failed options are never installed)", c.installed, want)
		}
		if len(c.reasons) != 1 || !strings.Contains(c.reasons[0], "options failed") {
			t.Errorf("reasons = %v, want the options failure recorded", c.reasons)
		}
	})

	t.Run("a single-device plugin has nothing to walk", func(t *testing.T) {
		c, ok := run(1, 0, map[int]bool{0: true}, nil)
		if ok {
			t.Fatal("retryOpenVINODevices() ok = true with a single failing device")
		}
		if len(c.attempted) != 0 {
			t.Errorf("attempted devices = %v, want none (the only device already failed)", c.attempted)
		}
	})
}

// equalInts reports whether two int slices hold the same values in order.
func equalInts(got, want []int) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}
