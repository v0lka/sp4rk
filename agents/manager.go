package agents

import (
	"log/slog"
	"os"
	"path/filepath"
	"sync"
)

// agentFileName is the marker file scanned for inside each agent directory.
const agentFileName = "AGENT.md"

// AgentManager discovers, parses, and serves Subagent Profiles from configured
// directories. Directories are scanned in priority order; the first occurrence
// of an agent name wins (higher-priority directories override lower ones). It
// mirrors the skills.SkillManager design.
type AgentManager struct {
	mu     sync.RWMutex
	agents map[string]*Agent // keyed by agent name
	dirs   []string          // discovery directories in priority order (highest first)
	logger *slog.Logger
}

// NewAgentManager creates an AgentManager that will discover agents from the
// given directories (highest priority first). Call Scan() to populate the
// catalog. A nil logger is tolerated and falls back to a default text handler.
func NewAgentManager(dirs []string, logger *slog.Logger) *AgentManager {
	return &AgentManager{
		agents: make(map[string]*Agent),
		dirs:   append([]string(nil), dirs...),
		logger: logger,
	}
}

func (m *AgentManager) log() *slog.Logger {
	if m.logger != nil {
		return m.logger
	}
	return slog.New(slog.NewTextHandler(os.Stderr, nil))
}

// Scan walks all discovery directories and loads valid agents.
//
// Agents with the same name in a higher-priority directory override those in a
// lower-priority one (first occurrence / highest priority wins). The walk is
// performed in reverse priority order so that earlier (higher-priority)
// directories are visited last and overwrite later entries. Invalid AGENT.md
// files are logged at Debug level and skipped — they never abort a scan.
func (m *AgentManager) Scan() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Clear existing agents (allows re-scanning).
	m.agents = make(map[string]*Agent)

	// Walk directories in reverse priority so higher-priority entries overwrite.
	for i := len(m.dirs) - 1; i >= 0; i-- {
		m.scanDir(m.dirs[i])
	}

	m.log().Info("agent scan complete", "count", len(m.agents), "dirs", m.dirs)
	return nil
}

// scanDir reads all subdirectories of dir and attempts to parse each as an
// agent. Symlinks pointing to directories are followed (resolved via os.Stat,
// since os.ReadDir reports symlinks as non-directories even when they point to
// one).
func (m *AgentManager) scanDir(dir string) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		if !os.IsNotExist(err) {
			m.log().Warn("agent dir unreadable", "dir", dir, "error", err)
		}
		return
	}

	for _, entry := range entries {
		if !entry.IsDir() {
			// Follow symlinks: os.ReadDir reports symlinks as non-Dirs even
			// when they point to a directory. Resolve with os.Stat.
			if entry.Type()&os.ModeSymlink == 0 {
				continue
			}
			target := filepath.Join(dir, entry.Name())
			info, err := os.Stat(target)
			if err != nil || !info.IsDir() {
				continue
			}
		}
		agentDir := filepath.Join(dir, entry.Name())
		agentMD := filepath.Join(agentDir, agentFileName)

		agent, err := ParseAgent(agentMD, agentDir)
		if err != nil {
			// Skip invalid agents (e.g. a directory that is not an agent at all).
			m.log().Debug("skipped invalid agent", "dir", agentDir, "error", err)
			continue
		}

		m.agents[agent.Metadata.Name] = agent
		m.log().Debug("loaded agent", "name", agent.Metadata.Name, "dir", agentDir)
	}
}

// List returns lightweight descriptors for all discovered agents (discovery
// phase). Hidden agents are included and carry Hidden=true; the autocomplete
// consumer filters them out as needed.
func (m *AgentManager) List() []AgentDescriptor {
	m.mu.RLock()
	defer m.mu.RUnlock()

	result := make([]AgentDescriptor, 0, len(m.agents))
	for _, a := range m.agents {
		result = append(result, a.Descriptor())
	}
	return result
}

// Get returns the full Agent by name, or (nil, false) if not found. Hidden
// agents are returned here just like any other — hiding only affects discovery
// (List / autocomplete), not invocation.
func (m *AgentManager) Get(name string) (*Agent, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	a, ok := m.agents[name]
	return a, ok
}
