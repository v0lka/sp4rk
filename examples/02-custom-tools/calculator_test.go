package main

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

func TestEvaluateValidExpressions(t *testing.T) {
	cases := []struct {
		expr string
		want float64
	}{
		{"17 * 23 + 100", 491},
		{"2 + 3 * 4", 14},
		{"(2 + 3) * 4", 20},
		{"10 / 4", 2.5},
		{"-5 + 3", -2},
		{"3.5 * 2", 7},
	}
	for _, c := range cases {
		got, err := evaluate(c.expr)
		if err != nil {
			t.Fatalf("evaluate(%q) returned error: %v", c.expr, err)
		}
		if got != c.want {
			t.Fatalf("evaluate(%q) = %g, want %g", c.expr, got, c.want)
		}
	}
}

// TestEvaluateBadInputReturnsError verifies that malformed input produces a
// returned error instead of a panic: the tokenizer's errors are returned
// directly, and the parser's panics are recovered inside evaluate.
func TestEvaluateBadInputReturnsError(t *testing.T) {
	cases := []string{
		"17 * 23 + 100 =", // trailing '=' — common LLM output
		"17 × 23",         // non-ASCII multiplication sign
		"1 + 2 apples",    // stray word
		"1.2.3",           // malformed number
		"1 / 0",           // division by zero
		"(1 + 2",          // unbalanced parenthesis
		"1 + ",            // dangling operator
		"+",               // operator without operands
		"",                // empty expression
		"   ",             // whitespace only
	}
	for _, expr := range cases {
		if _, err := evaluate(expr); err == nil {
			t.Fatalf("evaluate(%q) = nil error, want error", expr)
		}
	}
}

// TestCalculatorToolExecuteBadInputReturnsToolError verifies the tool path:
// bad model-supplied input yields an IsError ToolResult, never a process panic.
func TestCalculatorToolExecuteBadInputReturnsToolError(t *testing.T) {
	tool := NewCalculatorTool()
	cases := []string{
		"1 / 0",
		"17 * 23 + 100 =",
		"1.2.3",
		"17 × 23",
		"",
	}
	for _, expr := range cases {
		input, err := json.Marshal(map[string]string{"expression": expr})
		if err != nil {
			t.Fatalf("json.Marshal(%q): %v", expr, err)
		}
		result, err := tool.Execute(context.Background(), input)
		if err != nil {
			t.Fatalf("Execute(%q) returned error: %v", expr, err)
		}
		if !result.IsError {
			t.Fatalf("Execute(%q).IsError = false, want true (result %q)", expr, result.Content)
		}
		if !strings.Contains(strings.ToLower(result.Content), "error") {
			t.Fatalf("Execute(%q).Content = %q, want an error description", expr, result.Content)
		}
	}
}

func TestCalculatorToolExecuteValidExpression(t *testing.T) {
	tool := NewCalculatorTool()
	input, err := json.Marshal(map[string]string{"expression": "17 * 23 + 100"})
	if err != nil {
		t.Fatalf("json.Marshal: %v", err)
	}
	result, err := tool.Execute(context.Background(), input)
	if err != nil {
		t.Fatalf("Execute returned error: %v", err)
	}
	if result.IsError {
		t.Fatalf("Execute reported IsError for valid input: %q", result.Content)
	}
	if !strings.Contains(result.Content, "491") {
		t.Fatalf("Execute content = %q, want it to contain 491", result.Content)
	}
}
