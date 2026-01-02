"""Lexical analyzer for Ekilang language.

Tokenizes Ekilang source code into a stream of tokens.
"""

from typing import List
from .types import Token
from ._rust_lexer import tokenize


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
    """Tokenizes Ekilang source code using Rust implementation."""

    def __init__(self, source: str) -> None:
        self.source = source

    def reset(self, source: str = "") -> None:
        """Reset the lexer with new source (or empty)."""
        self.source = source

    def tokenize(self) -> List[Token]:
        """Convert source code into a list of tokens using Rust implementation."""
        return tokenize(self.source)
