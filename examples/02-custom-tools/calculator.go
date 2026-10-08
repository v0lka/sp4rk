package main

import (
	"fmt"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
)

// evaluate parses and evaluates a simple arithmetic expression.
// Supports +, -, *, /, parentheses, and integer/float literals.
// This is a minimal recursive-descent evaluator — not a production calculator.
// It never panics on malformed input: tokenizer errors are returned as errors,
// and the recover below converts any parser panic (e.g. "division by zero",
// "expected ')'") into a returned error before it can escape to the caller.
func evaluate(expr string) (val float64, err error) {
	// The recover must be installed here, before tokenize runs, so that a
	// panic anywhere in evaluation becomes a tool error, not a crashed process.
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("%v", r)
		}
	}()
	tokens, err := tokenize(expr)
	if err != nil {
		return 0, err
	}
	p := &parser{tokens: tokens}
	return p.parseExpr()
}

// --- tokenizer ---

type tokenKind int

const (
	tokNumber tokenKind = iota
	tokPlus
	tokMinus
	tokStar
	tokSlash
	tokLParen
	tokRParen
	tokEOF
)

type token struct {
	kind  tokenKind
	value float64
}

func tokenize(s string) ([]token, error) {
	var tokens []token
	for i := 0; i < len(s); {
		ch, size := utf8.DecodeRuneInString(s[i:])
		switch {
		case unicode.IsSpace(ch):
			i += size
		case ch == '+':
			tokens = append(tokens, token{kind: tokPlus})
			i += size
		case ch == '-':
			tokens = append(tokens, token{kind: tokMinus})
			i += size
		case ch == '*':
			tokens = append(tokens, token{kind: tokStar})
			i += size
		case ch == '/':
			tokens = append(tokens, token{kind: tokSlash})
			i += size
		case ch == '(':
			tokens = append(tokens, token{kind: tokLParen})
			i += size
		case ch == ')':
			tokens = append(tokens, token{kind: tokRParen})
			i += size
		case ch >= '0' && ch <= '9' || ch == '.':
			j := i
			for j < len(s) && (s[j] >= '0' && s[j] <= '9' || s[j] == '.') {
				j++
			}
			num, err := strconv.ParseFloat(s[i:j], 64)
			if err != nil {
				return nil, fmt.Errorf("invalid number %q: %w", s[i:j], err)
			}
			tokens = append(tokens, token{kind: tokNumber, value: num})
			i = j
		default:
			return nil, fmt.Errorf("unexpected character %q at position %d", ch, i)
		}
	}
	tokens = append(tokens, token{kind: tokEOF})
	return tokens, nil
}

// --- recursive-descent parser ---

type parser struct {
	tokens []token
	pos    int
}

func (p *parser) peek() token { return p.tokens[p.pos] }
func (p *parser) next() token { t := p.tokens[p.pos]; p.pos++; return t }

// parseExpr parses an addition/subtraction chain. The parser signals
// malformed input ("division by zero", "expected ')'", "unexpected token")
// via panic; evaluate's recover converts those into returned errors.
func (p *parser) parseExpr() (float64, error) {
	return p.parseAddSub(), nil
}

func (p *parser) parseAddSub() float64 {
	left := p.parseMulDiv()
	for {
		t := p.peek()
		switch t.kind {
		case tokPlus:
			p.next()
			left += p.parseMulDiv()
		case tokMinus:
			p.next()
			left -= p.parseMulDiv()
		default:
			return left
		}
	}
}

func (p *parser) parseMulDiv() float64 {
	left := p.parseUnary()
	for {
		t := p.peek()
		switch t.kind {
		case tokStar:
			p.next()
			left *= p.parseUnary()
		case tokSlash:
			p.next()
			div := p.parseUnary()
			if div == 0 {
				panic("division by zero")
			}
			left /= div
		default:
			return left
		}
	}
}

func (p *parser) parseUnary() float64 {
	t := p.peek()
	if t.kind == tokMinus {
		p.next()
		return -p.parseUnary()
	}
	if t.kind == tokPlus {
		p.next()
		return p.parseUnary()
	}
	return p.parsePrimary()
}

func (p *parser) parsePrimary() float64 {
	t := p.next()
	switch t.kind {
	case tokNumber:
		return t.value
	case tokLParen:
		val := p.parseAddSub()
		if p.next().kind != tokRParen {
			panic("expected ')'")
		}
		return val
	default:
		panic("unexpected token: " + tokenKindName(t.kind))
	}
}

func tokenKindName(k tokenKind) string {
	switch k {
	case tokNumber:
		return "number"
	case tokPlus:
		return "+"
	case tokMinus:
		return "-"
	case tokStar:
		return "*"
	case tokSlash:
		return "/"
	case tokLParen:
		return "("
	case tokRParen:
		return ")"
	case tokEOF:
		return "EOF"
	default:
		return strings.ToLower(fmt.Sprintf("token(%d)", k))
	}
}
