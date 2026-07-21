package embedding

import (
	"os"
	"testing"
)

// onnxEnvManagedForTests reports whether TestMain has taken ownership of the
// process-global ONNX Runtime environment for this test run. It is true only
// when EMBEDDING_TEST_LIBRARY_PATH is set, in which case the environment is
// initialized once before any test runs and destroyed once after all tests
// finish. When true, no individual test may destroy the environment — doing so
// would kill the shared singleton, which initONNXRuntime (sync.Once-guarded)
// can never reinitialize.
var onnxEnvManagedForTests bool

// TestMain owns the ONNX Runtime environment lifecycle for the whole embedding
// test package.
//
// The ONNX Runtime environment is a process-global singleton: initONNXRuntime
// is guarded by sync.Once, so the first initialization is final and can never
// be repeated — even after destroyONNXRuntime has torn the environment down,
// subsequent init calls return the cached result without reinitializing. This
// makes any test that both initializes and destroys the environment mutually
// order-fragile with every other such test: whichever runs first kills the
// environment for the rest, so a later NewEmbedder or buildSessionOptions call
// fails against a dead environment.
//
// To make the suite order-independent, TestMain initializes the environment
// exactly once up front (when the shared library is available) and tears it
// down exactly once at the end. All ONNX-dependent tests then share a single
// live environment, and none of them destroys it — they release only their own
// session via closeSessionOnly. Tests that require a *missing* environment (the
// Embedder.Close error path) skip themselves via skipIfEnvManaged when
// onnxEnvManagedForTests is true, because that path is unreachable while the
// environment is alive.
func TestMain(m *testing.M) {
	libPath := os.Getenv("EMBEDDING_TEST_LIBRARY_PATH")
	if libPath != "" {
		onnxEnvManagedForTests = true
		// The result is cached by sync.Once; individual tests surface any
		// initialization failure through their own assertions.
		_ = initONNXRuntime(libPath)
	}

	code := m.Run()

	if onnxEnvManagedForTests {
		// Single teardown for the whole suite.
		_ = destroyONNXRuntime()
	}
	os.Exit(code)
}

// closeSessionOnly tears down an Embedder's session and session-options without
// destroying the process-global ONNX Runtime environment. It is the test-only
// counterpart of Embedder.Close for tests that share the environment managed by
// TestMain; production code must use Embedder.Close, which also destroys the
// global environment. Safe to call on a partially constructed Embedder.
func closeSessionOnly(e *Embedder) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.sess != nil {
		e.sess.destroy()
		e.sess = nil
	}
	if e.sessOpts != nil {
		_ = e.sessOpts.Destroy()
		e.sessOpts = nil
	}
	e.tokenizer = nil
}

// skipIfEnvManaged skips tests that require a *missing* ONNX Runtime
// environment (the Embedder.Close error path). Such tests are unreachable while
// TestMain manages a live environment, because calling Close would destroy the
// shared singleton that initONNXRuntime cannot reinitialize.
func skipIfEnvManaged(t *testing.T) {
	t.Helper()
	if onnxEnvManagedForTests {
		t.Skip("ONNX environment is managed by TestMain; close-error path is unreachable while env is live")
	}
}
