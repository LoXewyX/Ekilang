"""Type stub for _rust_lexer Rust extension module (PyO3)."""

from typing import List, Any
from .lexer import Token

def tokenize(source: str) -> List[Token]:
    """Tokenize Ekilang source code.

    Args:
        source: The source code to tokenize

    Returns:
        A list of Token objects with type, value, line, col attributes
    """

def apply_binop(left: Any, op: str, right: Any) -> Any:
    """Apply a binary operation efficiently using Rust.

    Args:
        left: Left operand
        op: Operator string ('+', '-', '*', '/', etc.)
        right: Right operand

    Returns:
        The result of the operation
    """

def apply_compare(left: Any, op: str, right: Any) -> bool:
    """Apply a comparison operation efficiently using Rust.

    Args:
        left: Left operand
        op: Comparison operator string ('<', '<=', '>', '>=', '==', '!=')
        right: Right operand

    Returns:
        The boolean result of the comparison
    """
