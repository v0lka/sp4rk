package planner

import (
	"context"

	"github.com/v0lka/sp4rk/skills"
)

// DefaultConfig returns a Config with sensible defaults for standalone use,
// suitable for production framework use.
func DefaultConfig() Config {
	return Config{}.withDefaults()
}

// withDefaults returns a copy of c in which every unset injected context
// function is replaced by the no-op default and a non-positive
// MaxExploreSteps is clamped to the exploration budget. NewPlanner applies it
// so a hand-assembled Config cannot reach the plan paths with a nil function
// and panic the host process (the plan paths dereference these functions
// unconditionally).
func (c Config) withDefaults() Config {
	if c.DomainFromContext == nil {
		c.DomainFromContext = func(context.Context) string { return "" }
	}
	if c.ComplexityFromContext == nil {
		c.ComplexityFromContext = func(context.Context) int { return 0 }
	}
	if c.UserSkillsFromContext == nil {
		c.UserSkillsFromContext = func(context.Context) []string { return nil }
	}
	if c.FormatSkillList == nil {
		c.FormatSkillList = func(context.Context, []skills.SkillDescriptor) string { return "None" }
	}
	if c.FormatWorkspacePath == nil {
		c.FormatWorkspacePath = func(context.Context) string { return "" }
	}
	if c.AppendContextSections == nil {
		c.AppendContextSections = func(_ context.Context, base string) string { return base }
	}
	if c.MaxExploreSteps <= 0 {
		c.MaxExploreSteps = defaultMaxExploreSteps
	}
	return c
}
