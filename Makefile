.PHONY: build test lint vulncheck

# govulncheck version pinned for reproducible vulnerability scans (CI runs the
# same `make vulncheck` command; upgrade deliberately, both repos in lockstep).
GOVULNCHECK_VERSION := v1.7.0

build:
	go build ./...

test:
	go test ./...

lint:
	golangci-lint run

# Go dependency vulnerability gate: fails when a vulnerability from the
# official Go vulnerability database is reachable from this module's code.
# Uses the same pinned govulncheck version as the CI scan step, so local and
# CI results are identical. Requires network access to fetch vuln.go.dev.
# Windows (no make): go run golang.org/x/vuln/cmd/govulncheck@v1.7.0 ./...
vulncheck:
	go run golang.org/x/vuln/cmd/govulncheck@$(GOVULNCHECK_VERSION) ./...
