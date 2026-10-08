package builtins

import (
	"context"
	"net"
	"testing"
)

// TestPrivateNetworksParse verifies that all baked-in CIDR literals parse
// successfully at package init. Regression guard for W-4 (panic-on-init was
// replaced with a deferred error; this test catches typos that would silently
// disable SSRF protection).
func TestPrivateNetworksParse(t *testing.T) {
	if len(privateNetworks) == 0 {
		t.Fatal("privateNetworks list is empty after init")
	}
}

// TestIsPrivateIP_Smoke covers a handful of representative addresses to confirm
// the parsed CIDR ranges classify correctly.
func TestIsPrivateIP_Smoke(t *testing.T) {
	tests := []struct {
		ip      string
		private bool
	}{
		{"10.0.0.1", true},
		{"172.16.5.5", true},
		{"192.168.1.1", true},
		{"127.0.0.1", true},
		{"169.254.169.254", true}, // EC2 metadata
		{"100.64.0.1", true},      // CGNAT
		{"::1", true},
		{"fe80::1", true},
		{"fc00::1", true},
		{"::", true},              // IPv6 unspecified — review finding 20
		{"0:0:0:0:0:0:0:0", true}, // long form of ::
		{"ff02::1", true},         // link-local multicast
		{"224.0.0.1", true},       // IPv4 multicast
		{"8.8.8.8", false},
		{"1.1.1.1", false},
		{"203.0.114.0", false}, // outside 203.0.113.0/24
	}
	for _, tt := range tests {
		t.Run(tt.ip, func(t *testing.T) {
			ip := net.ParseIP(tt.ip)
			if ip == nil {
				t.Fatalf("invalid test IP: %s", tt.ip)
			}
			if got := isPrivateIP(ip); got != tt.private {
				t.Errorf("isPrivateIP(%s) = %v, want %v", tt.ip, got, tt.private)
			}
		})
	}
}

// TestResolveHostIsPrivate_IPv6Unspecified is the regression test for review
// finding 20: the unspecified address :: is not in any listed network unless
// "::/128" (or the IP-class predicates) covers it, so both SSRF gates used to
// classify http://[::]:PORT/ as public and auto-allow a loopback request.
func TestResolveHostIsPrivate_IPv6Unspecified(t *testing.T) {
	addr, private, err := resolveHostIsPrivate(context.Background(), "http://[::]:8080/")
	if err != nil {
		t.Fatalf("resolveHostIsPrivate returned error: %v", err)
	}
	if !private {
		t.Errorf("resolveHostIsPrivate(http://[::]:8080/) private = false, want true (addr = %q)", addr)
	}
	if addr != "::" {
		t.Errorf("resolved addr = %q, want %q", addr, "::")
	}
}
