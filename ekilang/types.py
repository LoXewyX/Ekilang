"""AST node definitions for Ekilang parser."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Token:
    """Represents a single token in the source code."""

    type: str
    value: str
    line: int
    col: int


@dataclass
class Module:
    """Root AST node representing a module."""

    body: List[Statement]


@dataclass
class Let:
    """Variable declaration with optional type annotation."""

    name: str | List[str]  # Single name or list of names for unpacking
    value: ExprNode
    type_annotation: Optional[str] = None


@dataclass
class Fn:
    """Function definition with parameters, types, and decorators."""

    name: str
    params: List[str]
    param_types: List[Optional[str]]
    return_type: Optional[str]
    body: List[Statement]
    defaults: List[ExprNode | None] | None = field(
        default=None
    )  # Default values for parameters
    vararg: Optional[str] = None  # *args parameter name
    vararg_type: Optional[str] = None  # Type annotation for *args
    kwarg: Optional[str] = None  # **kwargs parameter name
    kwarg_type: Optional[str] = None  # Type annotation for **kwargs
    decorators: List[ExprNode] | None = field(default=None)  # Decorator expressions


@dataclass
class Class:
    """Class definition with bases and decorators."""

    name: str
    bases: List[str]  # Base class names
    body: List[Statement]  # Class body (methods, etc.)
    decorators: List[ExprNode] | None = field(default=None)  # Decorator expressions


@dataclass
class If:
    """Conditional statement with test, consequence, and alternative."""

    test: ExprNode
    conseq: List[Statement]
    alt: Optional[List[Statement]]


@dataclass
class Match:
    """Match statement with subject and cases."""

    subject: ExprNode
    cases: List[Case]


@dataclass
class Case:
    """Case in match with pattern, guard, and body."""

    patterns: List[ExprNode]
    guard: Optional[ExprNode]
    body: List[Statement]


@dataclass
class While:
    """While loop with test condition and body."""

    test: ExprNode
    body: List[Statement]


@dataclass
class For:
    """For loop with target, iterator, and body."""

    target: List[Name]
    iter: ExprNode
    body: List[Statement]


@dataclass
class Return:
    """Return statement with optional value."""

    value: Optional[ExprNode] = None


@dataclass
class Break:
    """Break statement to exit loops."""


@dataclass
class Continue:
    """Continue statement to skip to next loop iteration."""


@dataclass
class Yield:
    """Yield statement with optional value (for generators)."""

    value: Optional[ExprNode]


@dataclass
class ExprStmt:
    """Expression statement for standalone expressions."""

    value: ExprNode


@dataclass
class BinOp:
    """Binary operation with left operand, operator, and right operand."""

    left: ExprNode
    op: str
    right: ExprNode


@dataclass
class UnaryOp:
    """Unary operation with operator and operand."""

    op: str
    operand: ExprNode


@dataclass
class Call:
    """Function call with arguments and keyword arguments."""

    func: ExprNode
    args: List[ExprNode | Starred]
    kwargs: List[Tuple[str, ExprNode]] | None = field(
        default=None
    )  # keyword arguments as (name, value) pairs


@dataclass
class Index:
    """Indexing or slicing operation."""

    value: ExprNode
    index: Slice | ExprNode


@dataclass
class Slice:
    """Slice ExprNode with start, stop, and step."""

    start: ExprNode | None
    stop: ExprNode | None
    step: ExprNode | None


@dataclass
class Attr:
    """Attribute access on an ExprNode."""

    value: ExprNode
    attr: str


@dataclass
class Name:
    """Variable name reference."""

    id: str


@dataclass
class Int:
    """Integer literal."""

    value: int


@dataclass
class Float:
    """Floating-point literal."""

    value: float


@dataclass
class Complex:
    """Complex/imaginary number literal."""

    value: complex


@dataclass
class Str:
    """String literal."""

    value: str


@dataclass
class BStr:
    """Byte string literal."""

    value: str


@dataclass
class Starred:
    """Starred expression for unpacking (*args)."""

    value: ExprNode


@dataclass
class Bool:
    """Boolean literal (true/false)."""

    value: bool


@dataclass
class NoneLit:
    """None literal."""


@dataclass
class Assign:
    """Assignment statement."""

    target: Name | ExprNode
    value: ExprNode


@dataclass
class AugAssign:
    """Augmented assignment (+=, -=, etc.)."""

    target: Name | ExprNode
    op: str
    value: ExprNode


@dataclass
class ListLit:
    """List literal."""

    elements: List[ExprNode]


@dataclass
class DictLit:
    """Dictionary literal."""

    pairs: List[Tuple[ExprNode, ExprNode]]


@dataclass
class Lambda:
    """Lambda function (anonymous function)."""

    params: List[str]
    body: List[Statement] | None = field(default=None)
    expr: ExprNode | None = field(default=None)


@dataclass
class FString:
    """F-string with interpolated expressions and format specifiers."""

    parts: List[str]  # String literal parts only
    exprs: List[ExprNode]  # Expression nodes only
    formats: List[str | None]  # Format specifiers for each expression
    debug_exprs: List[str | None]  # Debug expressions for x= syntax


@dataclass
class TString:
    """Template string with interpolated expressions."""

    parts: List[str]  # String literal parts only
    exprs: List[ExprNode]  # Expression nodes only
    formats: List[str | None]  # Format specifiers for each expression
    debug_exprs: List[str | None]  # Debug expressions for x= syntax


@dataclass
class TernaryOp:
    """Ternary conditional expression (test if true else false)."""

    test: ExprNode
    if_true: ExprNode
    if_false: ExprNode


@dataclass
class Range:
    """Range expression (.. or ..=)."""

    start: ExprNode
    end: ExprNode
    inclusive: bool  # True for ..=, False for ..


@dataclass
class ListComp:
    """List comprehension."""

    expr: ExprNode
    target: str
    iter: ExprNode
    condition: Optional[ExprNode]  # Optional if clause


@dataclass
class DictComp:
    """Dictionary comprehension."""

    key: ExprNode
    value: ExprNode
    target: str
    iter: ExprNode
    condition: Optional[ExprNode]  # Optional if clause


@dataclass
class SetComp:
    """Set comprehension."""

    expr: ExprNode
    target: str
    iter: ExprNode
    condition: Optional[ExprNode]  # Optional if clause


@dataclass
class UseItem:
    """Single import item with optional alias."""

    name: str
    alias: Optional[str]


@dataclass
class Use:
    """Import statement (use keyword)."""

    module: List[str]  # module path segments (may be empty for direct imports)
    items: List[UseItem]


@dataclass
class Cast:
    """Type cast expression."""

    value: ExprNode
    target_type: str


@dataclass
class Pipe:
    """Pipe operator (|> or <|)."""

    left: ExprNode
    op: str  # '|>' or '<|'
    right: ExprNode


@dataclass
class TupleLit:
    """Tuple literal."""

    elements: List[ExprNode]


@dataclass
class SetLit:
    """Set literal."""

    elements: List[ExprNode]


@dataclass
class AsyncFn:
    """Async function definition."""

    name: str
    params: List[str]
    param_types: List[Optional[str]]
    return_type: Optional[str]
    body: List[Statement]
    defaults: List[ExprNode | None] | None = field(
        default=None
    )  # Default values for parameters
    vararg: Optional[str] = None  # *args parameter name
    vararg_type: Optional[str] = None  # Type annotation for *args
    kwarg: Optional[str] = None  # **kwargs parameter name
    kwarg_type: Optional[str] = None  # Type annotation for **kwargs
    decorators: List[ExprNode] | None = field(default=None)  # Decorator expressions


@dataclass
class Await:
    """Await expression for async operations."""

    value: ExprNode


@dataclass
class NamedExpr:
    """Named expression with assignment operator (:=) - walrus operator."""

    target: str  # Variable name
    value: ExprNode


@dataclass
class ExceptHandler:
    """Exception handler for try-except blocks."""

    type: Optional[str]  # Exception type name (None for bare except)
    name: Optional[str]  # Variable name for 'as' clause (None if not present)
    body: List[Statement]


@dataclass
class Try:
    """Try-except-finally statement."""

    body: List[Statement]  # try block body
    handlers: List[ExceptHandler]  # except handlers
    orelse: Optional[List[Statement]] = None  # else block (optional)
    finalbody: Optional[List[Statement]] = None  # finally block (optional)


Statement = (
    Class
    | Use
    | For
    | Break
    | Continue
    | Yield
    | Return
    | AsyncFn
    | Fn
    | If
    | Match
    | While
    | Assign
    | AugAssign
    | Let
    | ExprStmt
    | Try
)

Body = List[List[Statement]]

ExprNode = (
    BinOp
    | Call
    | Index
    | Attr
    | Name
    | Int
    | Float
    | Complex
    | Str
    | BStr
    | Bool
    | NoneLit
    | ListLit
    | DictLit
    | Lambda
    | FString
    | TString
    | TernaryOp
    | Range
    | ListComp
    | DictComp
    | SetComp
    | UnaryOp
    | Pipe
    | TupleLit
    | SetLit
    | Cast
    | Await
    | NamedExpr
)