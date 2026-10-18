package embedding

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"

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
// EpName/EpVendor on an EpDevice — the "ov_device" metadata that distinguishes
// the plugin's GPU from its CPU device has no Go accessor (in any wrapper
// version to date), and the legacy "device_type" provider option is ignored
// by the plugin EP. The selection is therefore "all of the plugin's real
// devices, letting OpenVINO apply its own device priority" — verified live on
// the target iGPU machine to land on the GPU (B=8/S=512 in ~32 ms; pinning
// the CPU device alone measures ~286 ms). A non-zero deviceID is therefore a
// configuration error, not a silently ignored knob: the caller asked for a
// specific device this surface cannot select, and per ADR-045's
// loud-configuration rule the request must fail instead of pretending.
// Per-device OpenVINO indexing becomes possible only when the wrapper grows
// metadata accessors.
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
	if err := opts.AppendExecutionProviderV2(devices, nil); err != nil {
		return fmt.Errorf("appending OpenVINO execution provider (%d devices): %w", len(devices), err)
	}
	return nil
}
