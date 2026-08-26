package builtins

import (
	"bytes"
	"unicode"
	"unicode/utf16"
	"unicode/utf8"

	encodingunicode "golang.org/x/text/encoding/unicode"
)

// decodePowerShellOutput turns PowerShell's text output into UTF-8. Windows
// PowerShell can write UTF-16 to redirected stdout/stderr, whereas PowerShell
// Core and normal command output use UTF-8. A BOM is authoritative. Without
// one, UTF-8 is left unchanged and UTF-16 candidates are accepted only when
// their code units form valid, text-like Unicode in one byte order.
func decodePowerShellOutput(output []byte) string {
	// Strip a UTF-8 BOM: a PowerShell session configured with the standard
	// BOM-emitting UTF8 encoding ([Console]::OutputEncoding =
	// [System.Text.Encoding]::UTF8) writes the StreamWriter preamble first.
	output = bytes.TrimPrefix(output, []byte{0xef, 0xbb, 0xbf})
	if len(output) == 0 {
		return ""
	}

	if hasUTF16BOM(output) {
		if decoded, err := encodingunicode.UTF16(encodingunicode.LittleEndian, encodingunicode.UseBOM).NewDecoder().Bytes(output); err == nil {
			return string(decoded)
		}
		return string(output)
	}

	if utf8.Valid(output) && !containsNUL(output) {
		return string(output)
	}

	if order, ok := utf16ByteOrder(output); ok {
		if decoded, err := encodingunicode.UTF16(order, encodingunicode.IgnoreBOM).NewDecoder().Bytes(output); err == nil {
			return string(decoded)
		}
	}

	return string(output)
}

func hasUTF16BOM(output []byte) bool {
	return len(output) >= 2 && ((output[0] == 0xff && output[1] == 0xfe) || (output[0] == 0xfe && output[1] == 0xff))
}

// utf16ByteOrder detects text-like UTF-16 without a BOM. It requires at least
// two code units and evaluates both byte orders so non-ASCII text, including
// surrogate pairs, does not depend on ASCII-specific alternating NUL bytes.
func utf16ByteOrder(output []byte) (encodingunicode.Endianness, bool) {
	if len(output) < 4 || len(output)%2 != 0 {
		return encodingunicode.LittleEndian, false
	}

	littleValid, littleText := validUTF16Text(output, encodingunicode.LittleEndian)
	bigValid, bigText := validUTF16Text(output, encodingunicode.BigEndian)
	switch {
	case littleValid && !bigValid:
		if utf16OrderHasNULSignal(output, encodingunicode.LittleEndian) {
			return encodingunicode.LittleEndian, true
		}
	case bigValid && !littleValid:
		if utf16OrderHasNULSignal(output, encodingunicode.BigEndian) {
			return encodingunicode.BigEndian, true
		}
	case littleValid && bigValid && littleText > bigText:
		if utf16OrderHasNULSignal(output, encodingunicode.LittleEndian) {
			return encodingunicode.LittleEndian, true
		}
	case bigValid && littleValid && bigText > littleText:
		if utf16OrderHasNULSignal(output, encodingunicode.BigEndian) {
			return encodingunicode.BigEndian, true
		}
	case littleValid && bigValid:
		littleNUL, bigNUL := utf16NULAlignment(output)
		if littleNUL > bigNUL {
			return encodingunicode.LittleEndian, true
		}
		if bigNUL > littleNUL {
			return encodingunicode.BigEndian, true
		}
	}
	return encodingunicode.LittleEndian, false
}

func validUTF16Text(output []byte, order encodingunicode.Endianness) (valid bool, textRunes int) {
	units := make([]uint16, 0, len(output)/2)
	for i := 0; i < len(output); i += 2 {
		if order == encodingunicode.LittleEndian {
			units = append(units, uint16(output[i])|uint16(output[i+1])<<8)
		} else {
			units = append(units, uint16(output[i+1])|uint16(output[i])<<8)
		}
	}

	for i := 0; i < len(units); i++ {
		r := rune(units[i])
		switch {
		case utf16.IsSurrogate(r):
			if i+1 >= len(units) || !utf16.IsSurrogate(rune(units[i+1])) {
				return false, 0
			}
			r = utf16.DecodeRune(r, rune(units[i+1]))
			if r == unicode.ReplacementChar {
				return false, 0
			}
			i++
		case unicode.IsControl(r) && r != '\n' && r != '\r' && r != '\t':
			return false, 0
		}
		if unicode.IsGraphic(r) || r == '\n' || r == '\r' || r == '\t' {
			textRunes++
		}
	}

	return textRunes >= 2, textRunes
}

func utf16NULAlignment(output []byte) (little, big int) {
	for i, b := range output {
		if b == 0 {
			if i%2 == 0 {
				big++
			} else {
				little++
			}
		}
	}
	return little, big
}

func utf16OrderHasNULSignal(output []byte, order encodingunicode.Endianness) bool {
	little, big := utf16NULAlignment(output)
	if order == encodingunicode.LittleEndian {
		return little > big && little > 0
	}
	return big > little && big > 0
}

func containsNUL(output []byte) bool {
	for _, b := range output {
		if b == 0 {
			return true
		}
	}
	return false
}
