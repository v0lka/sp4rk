//go:build !windows

package mcp

import (
	"context"
	"os"
	"path/filepath"
	"strconv"
	"syscall"
	"testing"
	"time"
)

// processAlive reports whether pid names a live (or not-yet-reaped) process.
// signal 0 performs the permission/existence check without delivering a signal.
func processAlive(pid int) bool {
	err := syscall.Kill(pid, 0)
	return err == nil || err == syscall.EPERM
}

// TestServer_Close_KillsLauncherProcessTree is the process-tree regression for
// finding 5: with the server launched through a runner (modeled by the "tree"
// helper, which spawns a long-lived grandchild), Close must reap the WHOLE
// group, not just the direct child. A bare cmd.Process.Kill would terminate
// the launcher and leave the grandchild running as an orphan holding its pipes.
func TestServer_Close_KillsLauncherProcessTree(t *testing.T) {
	pidFile := filepath.Join(t.TempDir(), "grandchild.pid")

	s := newServer("tree")
	s.closeGrace = 300 * time.Millisecond
	cfg := ServerConfig{
		Transport: "stdio",
		Command:   stdioHelperCommand(t),
		Env: map[string]string{
			stdioHelperEnvVar:     stdioHelperTree,
			stdioHelperPidFileEnv: pidFile,
		},
		Timeout:     10 * time.Second,
		CallTimeout: 10 * time.Second,
	}
	if err := s.Connect(context.Background(), cfg); err != nil {
		t.Fatalf("Connect to the tree helper: %v", err)
	}

	// The launcher publishes its grandchild's pid once it has spawned it.
	pid := readGrandchildPid(t, pidFile)
	if !processAlive(pid) {
		t.Fatalf("grandchild %d is not running before Close", pid)
	}

	if err := s.Close(); err == nil {
		t.Fatal("expected the bounded close to report a killed server, got nil")
	}

	// The grandchild was never the client's direct child, so only a
	// process-group kill can reap it. Poll to ESRCH: the killed grandchild is
	// briefly a zombie until its reaper collects it.
	deadline := time.Now().Add(5 * time.Second)
	for processAlive(pid) {
		if time.Now().After(deadline) {
			t.Fatalf("grandchild %d survived Server.Close — the process tree was not reaped", pid)
		}
		time.Sleep(20 * time.Millisecond)
	}
}

// readGrandchildPid blocks until the tree helper has written its grandchild pid
// to pidFile, returning it.
func readGrandchildPid(t *testing.T, pidFile string) int {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for {
		if data, err := os.ReadFile(pidFile); err == nil && len(data) > 0 {
			if pid, perr := strconv.Atoi(string(data)); perr == nil {
				return pid
			}
		}
		if time.Now().After(deadline) {
			t.Fatal("the launcher never published its grandchild pid")
		}
		time.Sleep(20 * time.Millisecond)
	}
}
