"""Lexical analyzer for Ekilang language.

Tokenizes Ekilang source code into a stream of tokens for parsing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Token:
    """Represents a single token in the source code."""

    type: str
    value: str
    line: int
    col: int


KEYWORDS = {
    "fn",
    "async",
    "await",
    "use",
    "as",
    "if",
    "elif",
    "else",
    "match",
    "while",
    "return",
    "yield",
    "for",
    "in",
    "true",
    "false",
    "none",
    "and",
    "or",
    "not",
    "is",
    "break",
    "continue",
    "class",
}


class Lexer:
    """Tokenizes Ekilang source code."""

    def __init__(self, source: str) -> None:
        self.source = source

    def reset(self, source: str = "") -> None:
        """Reset the lexer with new source (or empty)."""
        self.source = source

    def tokenize(self) -> List[Token]:
        """Convert source code into a list of tokens."""
        tokens: List[Token] = []
        i = 0
        line_no = 1
        col = 1
        src = self.source

        def emit_nl():
            nonlocal line_no, col, i
            tokens.append(Token("NL", "\n", line_no, col))
            line_no += 1
            col = 1
            i += 1

        while i < len(src):
            ch = src[i]

            # Newlines (support CRLF and LF)
            if ch == "\n":
                emit_nl()
                continue
            if ch == "\r" and i + 1 < len(src) and src[i + 1] == "\n":
                # Windows line ending
                i += 1
                emit_nl()
                continue

            # Comments
            if ch == "#":
                while i < len(src) and src[i] not in "\n\r":
                    i += 1
                    col += 1
                continue

            # Whitespace
            if ch.isspace():
                i += 1
                col += 1
                continue

            # Identifiers (and f/t string prefixes)
            if ch.isalpha() or ch == "_":
                start = i
                start_col = col
                while i < len(src) and (src[i].isalnum() or src[i] == "_"):
                    i += 1
                    col += 1
                ident = src[start:i]

                # f/t prefixed strings (single or triple quoted)
                if ident in {"f", "t"} and i < len(src) and src[i] in ('"', "'"):
                    is_f = ident == "f"
                    # is_t determined by: not is_f
                    quote = src[i]
                    triple = i + 2 < len(src) and src[i : i + 3] == quote * 3
                    # consume quote(s)
                    if triple:
                        i += 3
                        col += 3
                        end_seq = quote * 3
                    else:
                        i += 1
                        col += 1
                        end_seq = quote
                    buf: List[str] = []
                    while i < len(src):
                        if triple and src.startswith(end_seq, i):
                            i += 3
                            col += 3
                            break
                        if not triple and src.startswith(end_seq, i):
                            i += 1
                            col += 1
                            break
                        if src[i] == "\n":
                            buf.append("\n")
                            i += 1
                            line_no += 1
                            col = 1
                            continue
                        if src[i] == "\r" and i + 1 < len(src) and src[i + 1] == "\n":
                            buf.append("\n")
                            i += 2
                            line_no += 1
                            col = 1
                            continue
                        if src[i] == "\\" and i + 1 < len(src):
                            buf.append(src[i + 1])
                            i += 2
                            col += 2
                        else:
                            buf.append(src[i])
                            i += 1
                            col += 1
                    else:
                        kind = "f-string" if is_f else "t-string"
                        raise SyntaxError(
                            f"Unterminated {kind} at {line_no}:{start_col}"
                        )
                    tok_type = "FSTR" if is_f else "TSTR"
                    tokens.append(Token(tok_type, "".join(buf), line_no, start_col))
                    continue

                tok_type = "KW" if ident in KEYWORDS else "ID"
                tokens.append(Token(tok_type, ident, line_no, start_col))
                continue

            # Numbers
            if ch.isdigit():
                start = i
                start_col = col
                while i < len(src) and src[i].isdigit():
                    i += 1
                    col += 1
                if (
                    i < len(src)
                    and src[i] == "."
                    and i + 1 < len(src)
                    and src[i + 1].isdigit()
                ):
                    i += 1
                    col += 1
                    while i < len(src) and src[i].isdigit():
                        i += 1
                        col += 1
                    num = src[start:i]
                    tokens.append(Token("FLOAT", num, line_no, start_col))
                else:
                    num = src[start:i]
                    tokens.append(Token("INT", num, line_no, start_col))
                continue

            # Strings (single or triple quoted)
            if ch in ('"', "'"):
                quote = ch
                start_col = col
                triple = i + 2 < len(src) and src[i : i + 3] == quote * 3
                if triple:
                    i += 3
                    col += 3
                    end_seq = quote * 3
                else:
                    i += 1
                    col += 1
                    end_seq = quote
                buf: List[str] = []
                while i < len(src):
                    if triple and src.startswith(end_seq, i):
                        i += 3
                        col += 3
                        break
                    if not triple and src.startswith(end_seq, i):
                        i += 1
                        col += 1
                        break
                    if src[i] == "\n":
                        buf.append("\n")
                        i += 1
                        line_no += 1
                        col = 1
                        continue
                    if src[i] == "\r" and i + 1 < len(src) and src[i + 1] == "\n":
                        buf.append("\n")
                        i += 2
                        line_no += 1
                        col = 1
                        continue
                    if src[i] == "\\" and i + 1 < len(src):
                        buf.append(src[i + 1])
                        i += 2
                        col += 2
                    else:
                        buf.append(src[i])
                        i += 1
                        col += 1
                else:
                    raise SyntaxError(f"Unterminated string at {line_no}:{start_col}")
                tokens.append(Token("STR", "".join(buf), line_no, start_col))
                continue

            two = src[i : i + 2]
            three = src[i : i + 3]
            if three in {"..=", "<<=", ">>=", "**=", "//="}:
                tokens.append(Token("OP", three, line_no, col))
                i += 3
                col += 3
                continue
            if two in {
                "=>",
                "==",
                "!=",
                ">=",
                "<=",
                "+=",
                "-=",
                "*=",
                "/=",
                "%=",
                "//",
                "..",
                "<<",
                ">>",
                "&=",
                "|=",
                "^=",
                "::",
                "->",
                "|>",
                "<|",
                "??",
                "?.",
                "**",
            }:
                tokens.append(Token("OP", two, line_no, col))
                i += 2
                col += 2
                continue
            if ch in "+-*/%=<>(),{}[]:|&^.@":
                ttype = "OP" if ch in "+-*/%=<>:|&^.@" else ch
                tokens.append(Token(ttype, ch, line_no, col))
                i += 1
                col += 1
                continue

            # Semicolon as statement separator
            if ch == ";":
                tokens.append(Token("NL", ";", line_no, col))
                i += 1
                col += 1
                continue

            raise SyntaxError(f"Unexpected character '{ch}' at {line_no}:{col}")

        tokens.append(Token("EOF", "", line_no, col))
        return tokens
