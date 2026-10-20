package embedding

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"

	ort "github.com/yalue/onnxruntime_go"
)

// The OpenVINO execution provider is not compiled into libonnxruntime: it is
// a plugin shared library (from Intel's onnxruntime-ep-openvino distribution)
// loaded at runtime through the V2 execution-provider API. Unlike the CUDA
// provider — whose provider libraries the runtime dlopens on its own from the
// main library's directory — the plugin must be registered explicitly with an
// ABSOLUTE path before its devices appear in ort.GetEpDevices, and session
// options then append one or more of those devices via
// AppendExecutionProviderV2 (the string-based V1 append is compiled-in
// providers only and ORT rejects it with the list of V2-capable names).
//
// Packaging contract (mirrors the CUDA flavor): the c0wrk installer
// (`make fetch-onnx-openvino`) places the plugin and the OpenVINO/TBB runtime
// libraries it needs (RUNPATH=$ORIGIN, so no LD_LIBRARY_PATH is involved)
// beside the main ONNX Runtime library, which is where this file looks for
// them. Verified live against the digest-pinned ORT 1.28.1 library and the
// 1.8.0 plugin on an Arrow Lake-P iGPU: registration, enumeration and GPU
// inference (B=8/S=512 in ~32 ms vs ~286 ms on OpenVINO's CPU device).
const (
	// openVINOPluginLibrary is the plugin shared library file the c0wrk
	// openvino flavor installs next to the main ONNX Runtime library. The
	// flavor (and therefore this provider) is Linux-only today; on other
	// platforms the file is simply never found and every openvino request
	// fails loudly naming the missing path.
	openVINOPluginLibrary = "libonnxruntime_providers_openvino_plugin.so"

	// openVINORegistrationName is the name the plugin library is registered
	// under. The name is this process's own registry key (any string works),
	// but it must be stable because a duplicate registration of the same
	// library fails with "library is already registered under ..." — see
	// registerOpenVINOPlugin for how that shapes the once-per-process rule.
	openVINORegistrationName = "openvino_ep"

	// openVINOExecutionProviderName is the EP name the plugin registers its
	// real devices under. The plugin additionally exposes an automatic
	// meta-device under the separate EP name "OpenVINOExecutionProvider.AUTO";
	// exact-name matching below excludes it deliberately (measured live: AUTO
	// gives no speedup over targeting the real devices, and ORT documents the
	// meta-device as incompatible with EP context models).
	openVINOExecutionProviderName = "OpenVINOExecutionProvider"
)

// openVINORegistrationOnce guards the process-global plugin registration the
// same way onnxInitOnce guards the environment: ORT has no unregister-on-
// failure story here (a second registration of the same library is rejected),
// so the first attempt is final for the process lifetime — its outcome is
// cached and returned to every later caller.
var (
	openVINORegistrationOnce sync.Once
	openVINORegistrationErr  error
)

// openVINOPreferredDevice is the index — into the device order returned by
// selectOpenVINODevices — of the plugin device that created a session
// successfully. The plugin applies its own priority to an appended device list
// but then uses only the FIRST device ("OpenVINO EP factory ... is not a meta
// device but N OpenVINO devices were specified. Using first ov_device only"),
// so appending every device hands the decision to whatever the plugin
// enumerated first — on Intel Core Ultra parts that is the NPU, which cannot
// compile the dynamic shapes of a transformer encoder. The wrapper surface
// (EpName/EpVendor only) carries no hardware-device class, so the class cannot
// be filtered before the fact; the device is therefore resolved empirically,
// by session creation (see retryOpenVINODevices).
//
// -1 means "unresolved": the full device list is appended and the plugin picks
// its own first device, which is the legacy behaviour and the first attempt.
// The value is process-wide for the same reason the ORT environment and the
// plugin registration are: ONNX Runtime is a process-global singleton and only
// one Embedder can exist per process (see NewEmbedder's DESIGN NOTE), so a
// resolved device stays valid for every later session and for a re-created
// embedder.
var openVINOPreferredDevice atomic.Int32

func init() { openVINOPreferredDevice.Store(-1) }

// openVINOPluginPath resolves the absolute path of the OpenVINO plugin
// library expected beside the main ONNX Runtime shared library and verifies
// the file exists. The absolute form is mandatory: ORT resolves a relative
// library argument against its own library directory, which produces a
// confusing "Failed to load library <ortdir>/<relpath>" error instead of a
// clear statement about the config. The existence check is equally
// deliberate: it runs before any ORT call, so an "auto" request never even
// attempts a registration on a machine without the plugin (and the
// once-per-process registration below is never burned on a foreseeable
// failure), while an explicit ExecutionProviderOpenVINO request gets the hard
// error ADR-045 requires — naming the missing path, never a silent CPU
// degradation.
func openVINOPluginPath(ortLibraryPath string) (string, error) {
	if ortLibraryPath == "" {
		return "", errors.New("resolving OpenVINO execution provider plugin: the ONNX Runtime library path is empty")
	}
	dir := filepath.Dir(ortLibraryPath)
	pluginPath := filepath.Join(dir, openVINOPluginLibrary)
	absPath, err := filepath.Abs(pluginPath)
	if err != nil {
		return "", fmt.Errorf("resolving OpenVINO execution provider plugin path: %w", err)
	}
	if _, err := os.Stat(absPath); err != nil {
		return "", fmt.Errorf("OpenVINO execution provider plugin not found at %s: %w", absPath, err)
	}
	return absPath, nil
}

// registerOpenVINOPlugin registers the plugin library with the ORT environment
// exactly once per process. The environment itself is a process-global
// singleton (see initONNXRuntime), and registered provider libraries live
// until it is destroyed, so a registration cannot be meaningfully retried,
// scoped, or paired with an UnregisterExecutionProviderLibrary in production:
// the single embedder owner tears the whole environment down at shutdown,
// which unloads the plugin with it. A failed first attempt is cached like a
// failed initONNXRuntime — the plugin not loading is a deployment problem a
// retry inside the same process cannot fix.
func registerOpenVINOPlugin(pluginPath string) error {
	openVINORegistrationOnce.Do(func() {
		if err := ort.RegisterExecutionProviderLibrary(openVINORegistrationName, pluginPath); err != nil {
			openVINORegistrationErr = fmt.Errorf("registering OpenVINO execution provider library %s: %w", pluginPath, err)
		}
	})
	return openVINORegistrationErr
}

// selectOpenVINODevices returns the plugin's real execution-provider devices:
// every ort.GetEpDevices entry whose EP name is exactly
// openVINOExecutionProviderName, which excludes both the runtime's own CPU
// device and the plugin's ".AUTO" meta-device entries. An empty result after a
// successful registration is an error, not a silent CPU slide: the plugin
// loaded but surfaced nothing this process could run on.
func selectOpenVINODevices() ([]ort.EpDevice, error) {
	devices, err := ort.GetEpDevices()
	if err != nil {
		return nil, fmt.Errorf("enumerating execution provider devices: %w", err)
	}
	var openVINO []ort.EpDevice
	for _, device := range devices {
		if device.EpName() == openVINOExecutionProviderName {
			openVINO = append(openVINO, device)
		}
	}
	if len(openVINO) == 0 {
		return nil, errors.New("OpenVINO execution provider registered but reported no devices")
	}
	return openVINO, nil
}

// appendOpenVINOProvider configures opts to run on the OpenVINO execution
// provider, mirroring appendCUDAProvider's role for the plugin-based provider.
//
// Device selection: the wrapper surface (onnxruntime_go v1.29.0) exposes only
// EpName/EpVendor on an EpDevice — neither the hardware-device class
// (OrtHardwareDevice_Type) nor the plugin's "ov_device" metadata that
// distinguishes its NPU/GPU/CPU devices has a Go accessor (in any wrapper
// version to date), and the legacy "device_type"/"device_id" provider options
// are advisory only (the plugin logs them as ignored and still takes the FIRST
// appended OrtEpDevice). The device is therefore resolved empirically: the
// first session attempt appends every real device and lets the plugin choose
// (the legacy behaviour), and when that first device cannot compile the model
// the remaining devices are tried one at a time, the winner being remembered
// in openVINOPreferredDevice (see retryOpenVINODevices). The live target
// measurement that motivated appending the full list still holds for the GPU
// device (B=8/S=512 in ~32 ms; the CPU device alone measures ~286 ms). A
// non-zero deviceID stays a configuration error, not a silently ignored knob:
// the caller asked for a specific device this surface cannot select by number,
// and per ADR-045's loud-configuration rule the request must fail instead of
// pretending. Per-device OpenVINO indexing selected by the caller becomes
// possible only when the wrapper grows metadata accessors.
//
// On failure opts is left untouched for the caller to destroy. The plugin
// registration (when this call gets that far) is process-global and
// permanent; see registerOpenVINOPlugin.
func appendOpenVINOProvider(opts *ort.SessionOptions, ortLibraryPath string, deviceID int) error {
	if deviceID != 0 {
		return fmt.Errorf("OpenVINO execution provider does not support device_id=%d: the onnxruntime_go wrapper exposes no per-device selection for OpenVINO devices; omit device_id (0 lets OpenVINO pick its best device)", deviceID)
	}
	pluginPath, err := openVINOPluginPath(ortLibraryPath)
	if err != nil {
		return err
	}
	if err := registerOpenVINOPlugin(pluginPath); err != nil {
		return err
	}
	devices, err := selectOpenVINODevices()
	if err != nil {
		return err
	}
	if err := opts.AppendExecutionProviderV2(narrowOpenVINODevices(devices, int(openVINOPreferredDevice.Load())), nil); err != nil {
		return fmt.Errorf("appending OpenVINO execution provider (%d devices): %w", len(devices), err)
	}
	return nil
}

// narrowOpenVINODevices returns the devices to append for the preferred device
// index: exactly that device once one has been resolved, or the whole list
// (the plugin's own choice) while the index is still -1 or out of range.
func narrowOpenVINODevices(devices []ort.EpDevice, preferred int) []ort.EpDevice {
	if preferred >= 0 && preferred < len(devices) {
		return devices[preferred : preferred+1]
	}
	return devices
}

// openVINODeviceCount reports how many real OpenVINO devices the plugin
// exposes. It is the size of the candidate space retryOpenVINODevices walks.
func openVINODeviceCount() (int, error) {
	devices, err := selectOpenVINODevices()
	if err != nil {
		return 0, err
	}
	return len(devices), nil
}

// openVINOAttemptedDevice is the device index the current session options were
// built for: the resolved preference, or 0 while it is unresolved (an
// appended-only list is decided by the plugin, and the plugin takes the first
// device).
func openVINOAttemptedDevice() int {
	if preferred := int(openVINOPreferredDevice.Load()); preferred >= 0 {
		return preferred
	}
	return 0
}

// retryOpenVINOSession is the call-site wrapper around retryOpenVINODevices:
// it resolves the candidate count and, when no device can build the session,
// returns firstErr — the failure of the device the caller already tried — so
// the provider-level (ladder or loud-failure) handling is unchanged.
func retryOpenVINOSession(
	buildOpts func(provider string) (*ort.SessionOptions, error),
	replaceOpts func(opts *ort.SessionOptions),
	createSess func() (*onnxSession, error),
	warn func(msg, provider, cause string),
	reasons *[]string,
	firstErr error,
) (*onnxSession, error) {
	deviceCount, err := openVINODeviceCount()
	if err != nil || deviceCount <= 1 {
		// A single device is no candidate space: whatever failed is all there
		// is, and re-running it would only repeat the same compile.
		return nil, firstErr
	}
	if sess, ok := retryOpenVINODevices(deviceCount, openVINOAttemptedDevice(), buildOpts, replaceOpts, createSess, warn, reasons); ok {
		return sess, nil
	}
	return nil, firstErr
}

// retryOpenVINODevices walks the plugin's devices after one failed to compile
// the model, returning the first device that can create a session.
//
// It exists because OpenVINO's plugin-level device selection is all-or-nothing
// at the ORT boundary: the plugin takes the FIRST appended OrtEpDevice and
// ignores the rest, and the onnxruntime_go surface exposes neither the
// hardware-device class (OrtHardwareDevice_Type: CPU/GPU/NPU) nor the "ov_device"
// metadata that would let a caller skip an unsuitable device up front. An
// Intel NPU — enumerated first on Core Ultra hosts — rejects the embedding
// model at compile time ("expected exactly 1 dynamic output bound dimension,
// got 2"), which would otherwise sink the whole provider and force the CPU
// fallback even though the integrated GPU runs the model fine. Trying the
// devices in enumeration order is therefore the only reliable way to land on
// hardware this model can run on; the winner is remembered in
// openVINOPreferredDevice so later sessions go straight to it.
//
// failed is the index already attempted (the caller's failing session); it is
// skipped. Each candidate rebuilds the OpenVINO options for that single device
// (buildOpts), swaps them into the embedder (replaceOpts, which also releases
// the previous handle) and retries session creation (createSess). Reasons for
// failed candidates are appended to reasons for the caller's fallback WARN.
func retryOpenVINODevices(
	deviceCount, failed int,
	buildOpts func(provider string) (*ort.SessionOptions, error),
	replaceOpts func(opts *ort.SessionOptions),
	createSess func() (*onnxSession, error),
	warn func(msg, provider, cause string),
	reasons *[]string,
) (*onnxSession, bool) {
	for i := 0; i < deviceCount; i++ {
		if i == failed {
			continue
		}
		openVINOPreferredDevice.Store(int32(i))
		opts, optsErr := buildOpts(ExecutionProviderOpenVINO)
		if optsErr != nil {
			reason := fmt.Sprintf("openvino device %d: %s", i, optsErr)
			*reasons = append(*reasons, reason)
			warn("openvino device options failed, trying next device", ExecutionProviderOpenVINO, optsErr.Error())
			continue
		}
		replaceOpts(opts)
		sess, sessErr := createSess()
		if sessErr == nil {
			return sess, true
		}
		reason := fmt.Sprintf("openvino device %d: %s", i, sessErr)
		*reasons = append(*reasons, reason)
		warn("openvino device session creation failed, trying next device", ExecutionProviderOpenVINO, sessErr.Error())
	}
	// No device can run this model: forget the preference so the next attempt
	// starts from the plugin's own choice again instead of pinning a device
	// that is known to fail.
	openVINOPreferredDevice.Store(-1)
	return nil, false
}
