// Package judge_prompts provides embedded prompt templates for the tool safety judge.
package judge_prompts

import _ "embed"

// JudgeSystem is the embedded system prompt for the advisory tool safety judge.
//
//go:embed judge_system.md
var JudgeSystem string

// JudgeStrictSystem is the embedded system prompt for strict automatic
// evaluation of user-confirmation gates.
//
//go:embed judge_strict_system.md
var JudgeStrictSystem string
