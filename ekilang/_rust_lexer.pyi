"""Type stub for _rust_lexer Rust extension module (PyO3)."""

from typing import Any, List
from ekilang.lexer import Token

def tokenize(source: str) -> List[Token]:  # pylint: disable=unused-argument
    """Tokenize Ekilang source code.

    Args:
        source: The source code to tokenize

    Returns:
        A list of Token objects with type, value, line, col attributes
    """

def apply_binop(
    left: Any, op: str, right: Any  # pylint: disable=unused-argument
) -> Any:
    """Apply a binary operation efficiently using Rust.

    Args:
        left: Left operand
        op: Operator string ('+', '-', '*', '/', etc.)
        right: Right operand

    Returns:
        The result of the operation
    """

def apply_compare(
    left: Any, op: str, right: Any  # pylint: disable=unused-argument
) -> bool:
    """Apply a comparison operation efficiently using Rust.

    Args:
        left: Left operand
        op: Comparison operator string ('<', '<=', '>', '>=', '==', '!=')
        right: Right operand

    Returns:
        The boolean result of the comparison
    """

# Parser helpers
def get_operator_precedence(op: str) -> int:  # pylint: disable=unused-argument
    """Get operator precedence level (higher = higher precedence)."""

def is_aug_assign_op(op: str) -> bool:  # pylint: disable=unused-argument
    """Check if operator is augmented assignment (+=, -=, etc.)."""

def is_comparison_op(op: str) -> bool:  # pylint: disable=unused-argument
    """Check if operator is comparison (<, >, ==, etc.)."""

def is_binary_op(op: str) -> bool:  # pylint: disable=unused-argument
    """Check if operator is binary arithmetic."""

def is_unary_op(op: str) -> bool:  # pylint: disable=unused-argument
    """Check if operator is unary (-, ~, not)."""

def is_statement_keyword(kw: str) -> bool:  # pylint: disable=unused-argument
    """Check if keyword starts a statement."""

def is_right_associative(op: str) -> bool:  # pylint: disable=unused-argument
    """Check if operator is right-associative."""

def is_valid_token_type(type_str: str) -> bool:  # pylint: disable=unused-argument
    """Validate token type string."""

def canonicalize_operator(op: str) -> str:  # pylint: disable=unused-argument
    """Return canonical form of operator."""

def validate_interpolation_braces(
    content: str,
) -> bool:  # pylint: disable=unused-argument
    """Check if string interpolation braces are balanced."""

def is_valid_id_start(ch: str) -> bool:  # pylint: disable=unused-argument
    """Check if character can start an identifier."""

def is_valid_id_continue(ch: str) -> bool:  # pylint: disable=unused-argument
    """Check if character can continue an identifier."""

def validate_operators(
    operators: List[str],
) -> List[bool]:  # pylint: disable=unused-argument
    """Batch validate multiple operators."""

def classify_operator(
    op: str,
) -> tuple[bool, bool, bool, bool]:  # pylint: disable=unused-argument
    """Classify operator: (is_assign, is_aug_assign, is_comparison, is_binary)."""

def classify_keyword(kw: str) -> int:  # pylint: disable=unused-argument
    """Classify keyword: 0=not_keyword, 1=definition, 2=control_flow, 3=simple_stmt, 4=expression."""
