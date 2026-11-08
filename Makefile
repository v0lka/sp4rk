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
#
# The gate is internal/vulncheck rather than govulncheck directly: govulncheck
# has no ignore flag, so an advisory that is unreachable by construction in this
# build is carried as a REVIEWED, DATED entry in
# internal/vulncheck/allowlist.json. The wrapper is fail-closed in every
# direction — an unmatched finding, an EXPIRED entry and an entry that no longer
# matches any finding all fail the run, so an exception can never become a
# permanent or silent waiver.
# Windows (no make): go run ./internal/vulncheck -govulncheck-version v1.7.0
vulncheck:
	go run ./internal/vulncheck -govulncheck-version $(GOVULNCHECK_VERSION)
