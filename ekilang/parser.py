"""Parser for Ekilang language.

Converts token stream into Abstract Syntax Tree (AST).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from .types import (
    Statement,
    ExprNode,
    Module,
    Let,
    Fn,
    Class,
    If,
    Match,
    Case,
    While,
    For,
    Use,
    UseItem,
    Break,
    Continue,
    Yield,
    Return,
    AsyncFn,
    Assign,
    AugAssign,
    ExprStmt,
    Name,
    BinOp,
    UnaryOp,
    TernaryOp,
    Pipe,
    Cast,
    Range,
    Await,
    Int,
    Float,
    Str,
    Bool,
    NoneLit,
    ListLit,
    DictLit,
    Lambda,
    FString,
    TString,
    Call,
    Index,
    Attr,
    Slice,
    Starred,
    TupleLit,
    SetLit,
    ListComp,
    DictComp,
    SetComp,
)

from .lexer import Token, Lexer

# Operator constants
AUG_ASSIGN_OPS = {
    "+=",
    "-=",
    "*=",
    "**=",
    "/=",
    "//=",
    "%=",
    "&=",
    "|=",
    "^=",
    "<<=",
    ">>=",
}


class Parser:
    """Recursive descent parser for Ekilang language."""

    def __init__(self, tokens: List[Token]) -> None:
        self.tokens: List[Token] = tokens
        self.i: int = 0

    def peek(self) -> Token:
        """Look at current token without consuming it."""
        return self.tokens[self.i]

    def match(self, type_: str, value: Optional[str] = None) -> Token:
        """Consume and return token if it matches, otherwise raise SyntaxError."""
        tok = self.peek()
        if tok.type != type_ or (value is not None and tok.value != value):
            raise SyntaxError(f"Expected {type_} {value or ''} at {tok.line}:{tok.col}")
        self.i += 1
        return tok

    def accept(self, type_: str, value: Optional[str] = None) -> Optional[Token]:
        """Consume and return token if it matches, otherwise return None."""
        tok = self.peek()
        if tok.type == type_ and (value is None or tok.value == value):
            self.i += 1
            return tok
        return None

    def parse(self) -> Module:
        """Parse token stream into Module AST."""
        body: List[Statement] = []
        while self.peek().type != "EOF":
            if self.accept("NL") or self.accept(";"):
                continue
            body.append(self.statement())
        return Module(body)

    def statement(
        self,
    ) -> Statement:
        """Parse a statement (including decorated statements)."""
        decorators = self._parse_decorators()
        tok = self.peek()

        # Handle keyword statements
        if tok.type == "KW":
            if tok.value == "class":
                return self._parse_class(decorators)
            if tok.value == "use":
                return self.use_stmt()
            if tok.value == "for":
                return self.for_stmt()
            if tok.value in {"break", "continue", "yield", "return"}:
                return self._parse_simple_stmt(tok.value)
            if tok.value == "async":
                return self._parse_async_fn(decorators)
            if tok.value == "fn":
                return self.fn_def(decorators)
            if tok.value == "if":
                return self.if_stmt()
            if tok.value == "match":
                return self.match_stmt()
            if tok.value == "while":
                return self.while_stmt()

        # Handle ID-based statements (assignments, aug-assigns, unpacking)
        if tok.type == "ID":
            result = self._parse_id_stmt()
            if result is not None:
                return result

        # Destructuring assignment (unpack targets)
        unpack_targets = self.parse_unpack_target()
        if (
            unpack_targets is not None
            and self.peek().type == "OP"
            and self.peek().value == "="
        ):
            self.match("OP", "=")
            value = self.expr()
            self.accept("NL")
            return Let(unpack_targets, value)

        # Check for assignment or fallback to expression statement
        expr = self.expr()
        return self._parse_assignment_or_expr_stmt(expr)

    def _parse_block_body(self) -> List[Statement]:
        """Parse a `{ ... }` block body and return list of statements."""
        self.match("{")
        body: List[Statement] = []
        while not self.accept("}"):
            if self.accept("NL"):
                continue
            body.append(self.statement())
        self.accept("NL")
        return body

    def _parse_simple_stmt(self, keyword: str) -> Break | Continue | Yield | Return:
        """Parse break, continue, yield, or return statements."""
        self.match("KW", keyword)
        if keyword == "break":
            self.accept("NL")
            return Break()
        if keyword == "continue":
            self.accept("NL")
            return Continue()
        # yield and return have optional expressions
        if self.peek().type in ("NL", "EOF", "}"):
            val = None
        else:
            val = self.expr()
        self.accept("NL")
        if keyword == "yield":
            return Yield(val)
        return Return(val)

    def _parse_id_stmt(self) -> Let | Assign | AugAssign | None:
        """Parse ID-based statements: assignments, aug-assigns, or unpacking.
        Returns the statement node or None if ID should be re-parsed as expression."""
        save_i = self.i
        name_tok = self.match("ID")

        # Check for type annotation: ID : type = value
        if self.accept(":"):
            # After type annotation, skip the type and look for assignment
            _ = self.parse_type()  # Skip the type annotation
            # After type annotation, we expect assignment
            if self.accept("OP", "="):
                value = self.expr()
                self.accept("NL")
                return Let(name_tok.value, value)

            self.i = save_i
            return None

        if self.accept("OP", "="):
            value = self.expr()
            self.accept("NL")
            return Assign(Name(name_tok.value), value)

        for op in AUG_ASSIGN_OPS:
            if self.accept("OP", op):
                value = self.expr()
                self.accept("NL")
                return AugAssign(Name(name_tok.value), op, value)

        # Not an assignment, might be unpacking or expression - backtrack
        if self.accept(","):
            self.i = save_i
            return None
        self.i = save_i
        return None

    def _parse_class(self, decorators: List[ExprNode] | None = None) -> Class:
        """Parse a class definition."""
        self.match("KW", "class")
        class_name = self.match("ID").value
        bases: List[str] = []
        if self.accept("("):
            if not self.accept(")"):
                bases.append(self.match("ID").value)
                while self.accept(","):
                    bases.append(self.match("ID").value)
                self.match(")")
        body = self._parse_block_body()
        return Class(
            name=class_name,
            bases=bases,
            body=body,
            decorators=decorators if decorators else None,
        )

    def _parse_assignment_or_expr_stmt(
        self, expr: ExprNode
    ) -> Let | Assign | AugAssign | ExprStmt:
        """Handle assignment forms and expression statement after an `expr()` parse.

        Supports type annotations on names, regular and augmented assignments, and
        falls back to `ExprStmt` when no assignment operator follows.
        """
        type_annotation: Optional[str] = None
        if (
            self.peek().type == "OP"
            and self.peek().value == ":"
            and isinstance(expr, Name)
        ):
            self.match("OP", ":")
            type_annotation = self.parse_type()

        aug_ops_with_assign = {"="} | AUG_ASSIGN_OPS
        if self.peek().type == "OP" and self.peek().value in aug_ops_with_assign:
            op = self.peek().value
            self.match("OP", op)
            value = self.expr()
            self.accept("NL")
            if op == "=":
                if type_annotation and isinstance(expr, Name):
                    return Let(expr.id, value)
                return Assign(expr, value)
            return AugAssign(expr, op, value)

        self.accept("NL")
        return ExprStmt(expr)

    def _parse_async_fn(self, decorators: List[ExprNode] | None = None) -> AsyncFn:
        """Parse an async function definition and return an AsyncFn node."""
        self.match("KW", "async")
        self.match("KW", "fn")
        name = self.match("ID").value
        self.match("(")
        (
            params,
            param_types,
            defaults,
            vararg,
            vararg_type,
            kwarg,
            kwarg_type,
        ) = self._parse_function_params()
        return_type = None
        if self.accept("OP", "->"):
            return_type = self.parse_type()
        self.match("{")
        body: List[Statement] = []
        while not self.accept("}"):
            if self.accept("NL"):
                continue
            body.append(self.statement())
        self.accept("NL")
        if body and isinstance(body[-1], ExprStmt):
            last = body[-1]
            body[-1] = Return(last.value)
        return AsyncFn(
            name=name,
            params=params,
            param_types=param_types,
            return_type=return_type,
            body=body,
            defaults=defaults if defaults else None,
            vararg=vararg,
            vararg_type=vararg_type,
            kwarg=kwarg,
            kwarg_type=kwarg_type,
            decorators=decorators if decorators else None,
        )

    def _parse_decorators(self) -> List[ExprNode]:
        """Parse leading decorators for a statement and return list of expressions."""
        decorators: List[ExprNode] = []
        while self.peek().type == "@":
            self.match("@")
            decorators.append(self.expr())
            self.accept("NL")
        return decorators

    def fn_def(self, decorators: List[ExprNode] | None = None) -> Fn:
        """Parse a function definition, including params, defaults, types, and body."""
        self.match("KW", "fn")
        name = self.match("ID").value
        self.match("(")

        params, param_types, defaults, vararg, vararg_type, kwarg, kwarg_type = (
            self._parse_function_params()
        )

        return_type = None
        if self.accept("OP", "->"):
            return_type = self.parse_type()
        self.match("{")
        body: List[Statement] = []
        while not self.accept("}"):
            if self.accept("NL"):
                continue
            body.append(self.statement())
        self.accept("NL")
        if body and isinstance(body[-1], ExprStmt):
            last = body[-1]
            body[-1] = Return(last.value)

        return Fn(
            name=name,
            params=params,
            param_types=param_types,
            return_type=return_type,
            body=body,
            defaults=defaults if defaults else None,
            vararg=vararg,
            vararg_type=vararg_type,
            kwarg=kwarg,
            kwarg_type=kwarg_type,
            decorators=decorators,
        )

    def if_stmt(self) -> If:
        """Parse an if statement with optional elif and else blocks."""

        def parse_block() -> List[Statement]:
            self.match("{")
            body: List[Statement] = []
            while not self.accept("}"):
                if not self.accept("NL"):
                    body.append(self.statement())
            return body

        self.match("KW", "if")
        root = If(self.expr(), parse_block(), None)
        current = root

        while self.accept("KW", "elif"):
            elif_node = If(self.expr(), parse_block(), None)
            current.alt = [elif_node]
            current = elif_node

        if self.accept("KW", "else"):
            current.alt = parse_block()

        return root

    def match_stmt(self) -> Match:
        """Parse a match statement with cases."""
        self.match("KW", "match")
        subject = self.expr()
        self.match("{")
        cases: List[Case] = []
        while not self.accept("}"):
            if self.accept("NL"):
                continue
            # Parse pattern
            if self.accept("ID", "_"):
                patterns = []
            else:
                patterns: List[ExprNode] = []
                patterns.append(self.bit_xor_expr())
                while self.accept("OP", "|"):
                    patterns.append(self.bit_xor_expr())
            # Optional guard
            guard = None
            if self.accept("KW", "if"):
                guard = self.expr()
            self.match("OP", "=>")
            # Always treat { ... } after => as a block, not as an expression
            if self.peek().type == "{" or (
                self.peek().type == "OP" and self.peek().value == "{"
            ):
                self.match("{")
                body: List[Statement] = []
                while not self.accept("}"):
                    if self.accept("NL"):
                        continue
                    body.append(self.statement())
            else:
                # Single statement without braces
                body: List[Statement] = [self.statement()]
            cases.append(Case(patterns, guard, body))
        return Match(subject, cases)

    def while_stmt(self) -> While:
        """Parse a while loop with a test expression and body."""
        self.match("KW", "while")
        test = self.expr()
        self.match("{")
        body: List[Statement] = []
        while not self.accept("}"):
            if self.accept("NL"):
                continue
            body.append(self.statement())
        self.accept("NL")
        return While(test, body)

    def for_stmt(self) -> For:
        """Parse a for loop with unpacking target(s), iterable expression, and body."""
        self.match("KW", "for")
        target = self.parse_unpack_target()
        if target is None:
            # fallback to single name for error clarity
            target = [self.match("ID").value]
        self.match("KW", "in")
        iterable = self.expr()
        self.match("{")
        body: List[Statement] = []
        while not self.accept("}"):
            if self.accept("NL"):
                continue
            body.append(self.statement())
        self.accept("NL")
        # If only one name, keep as Name node for compatibility; else, pass list
        return For([Name(name) for name in target], iterable, body)

    def use_stmt(self) -> Use:
        """Parse a `use` import statement, with support for relative paths."""
        self.match("KW", "use")
        path: List[str] = []

        # Handle relative imports (.:: or ..:: or ...::, etc.)
        # Check for leading dots (relative imports)
        if self.peek().type == "." or (
            self.peek().type == "OP" and self.peek().value.startswith(".")
        ):
            relative_prefix = ""
            # Handle multi-dot operators like .. and single .
            tok = self.peek()
            if tok.type == ".":
                # Single dots collected one at a time
                while self.peek().type == ".":
                    relative_prefix += "."
                    self.i += 1
            elif tok.type == "OP" and (tok.value == ".." or tok.value.startswith(".")):
                # Handle .. or other multi-dot operators (like ...)
                relative_prefix = tok.value
                self.i += 1
            path.append(relative_prefix)
            # After dots, we expect ::
            if self.peek().type == "OP" and self.peek().value == "::":
                self.i += 1
                # Now get the module name(s) after ::
                path.append(self.match("ID").value)
            else:
                # If no ::, the next thing must be the module name
                path.append(self.match("ID").value)
        else:
            # Regular absolute import
            path.append(self.match("ID").value)

        brace_pending = False
        while True:
            tok = self.peek()
            if (tok.type == "OP" and tok.value == "::") or tok.type == ".":
                self.i += 1
                if self.peek().type == "{":
                    brace_pending = True
                    break
                path.append(self.match("ID").value)
                continue
            break

        if brace_pending or self.accept("{"):
            items: List[UseItem] = []
            if not brace_pending:
                # We consumed '{' already via accept
                pass
            else:
                self.match("{")
            while True:
                if self.peek().type == "OP" and self.peek().value == "*":
                    self.match("OP", "*")
                    items.append(UseItem("*", None))
                else:
                    name = self.match("ID").value
                    alias = None
                    if self.accept("KW", "as"):
                        alias = self.match("ID").value
                    items.append(UseItem(name, alias))
                if self.accept("}"):
                    break
                self.match(",")
            self.accept("NL")
            return Use(module=path, items=items)

        # Single path import (no braces)
        alias = None
        if self.accept("KW", "as"):
            alias = self.match("ID").value
        self.accept("NL")
        if len(path) == 1:
            module_path: List[str] = []
            item_name = path[0]
        else:
            module_path = path[:-1]
            item_name = path[-1]
        return Use(module=module_path, items=[UseItem(item_name, alias)])

    def expr(self) -> ExprNode:
        """Parse a full expression at the highest precedence level."""
        return self.pipe_expr()

    def pipe_expr(self) -> ExprNode:
        """Parse pipe expressions with `|>` and `<|` associativity rules."""
        # Handle pipeline operators |> and <|
        # |> is left-associative: data |> f |> g means g(f(data))
        # <| is right-associative: f <| g <| data means f(g(data))

        # First, handle the base case
        node = self.cast_expr()

        # For |>, use left-associative parsing
        while (
            self.peek().type == "OP"
            and self.peek().value == "|>"
            or (
                self.accept("NL")
                and self.peek().type == "OP"
                and self.peek().value == "|>"
            )
        ):
            self.match("OP")
            node = Pipe(left=node, op="|>", right=self.cast_expr())

        # For <|, we need right-associative parsing
        # If we see <|, consume it and parse the rest as a pipe_expr recursively
        if self.peek().type == "OP" and self.peek().value == "<|":
            self.match("OP")
            node = Pipe(left=node, op="<|", right=self.pipe_expr())

        return node

    def cast_expr(self) -> ExprNode:
        """Parse `as` type cast expressions."""
        node = self.ternary_expr()
        if self.peek().type == "KW" and self.peek().value == "as":
            self.match("KW", "as")
            target_type = self.parse_type()
            return Cast(value=node, target_type=target_type)
        return node

    def ternary_expr(self) -> ExprNode:
        """Parse ternary conditional expression: `a if cond else b`."""
        # Handle ternary: expr if condition else expr
        node = self.or_expr()
        if self.peek().type == "KW" and self.peek().value == "if":
            self.match("KW", "if")
            test = self.or_expr()
            self.match("KW", "else")
            if_false = self.ternary_expr()
            return TernaryOp(test=test, if_true=node, if_false=if_false)
        return node

    def or_expr(self) -> ExprNode:
        """Parse or expression."""
        node = self.and_expr()
        while self.peek().type == "KW" and self.peek().value == "or":
            self.match("KW", "or")
            node = BinOp(node, "or", self.and_expr())
        return node

    def and_expr(self) -> ExprNode:
        """Parse and expression."""
        node = self.bit_or_expr()
        while self.peek().type == "KW" and self.peek().value == "and":
            self.match("KW", "and")
            node = BinOp(node, "and", self.bit_or_expr())
        return node

    def bit_or_expr(self) -> ExprNode:
        """Parse bitwise or expression."""
        node = self.bit_xor_expr()
        while self.peek().type == "OP" and self.peek().value == "|":
            self.i += 1
            node = BinOp(node, "|", self.bit_xor_expr())
        return node

    def bit_xor_expr(self) -> ExprNode:
        """Parse bitwise xor expression."""
        node = self.bit_and_expr()
        while self.peek().type == "OP" and self.peek().value == "^":
            self.i += 1
            node = BinOp(node, "^", self.bit_and_expr())
        return node

    def bit_and_expr(self) -> ExprNode:
        """Parse bitwise and expression."""
        node = self.cmp_expr()
        while self.peek().type == "OP" and self.peek().value == "&":
            self.i += 1
            node = BinOp(node, "&", self.cmp_expr())
        return node

    def cmp_expr(self) -> ExprNode:
        """Parse comparison expression."""
        node = self.range_expr()
        return self._parse_comparators(node)

    def _parse_comparators(self, left: ExprNode) -> ExprNode:
        """Parse chained comparison operators following an initial left-side node."""
        node: ExprNode = left
        while True:
            t = self.peek()
            if t.type == "OP" and t.value in {"==", "!=", ">", "<", ">=", "<="}:
                op: str = t.value
                self.i += 1
            elif t.type == "KW" and t.value == "is":
                self.i += 1
                op = "is not" if self.accept("KW", "not") else "is"
            elif t.type == "KW" and t.value == "in":
                self.i += 1
                op = "in"
            elif t.type == "KW" and t.value == "not":
                self.i += 1
                if self.accept("KW", "in"):
                    op = "not in"
                else:
                    raise SyntaxError(f"Expected 'in' after 'not' at {t.line}:{t.col}")
            else:
                break
            node = BinOp(node, op, self.range_expr())
        return node

    def range_expr(self) -> ExprNode:
        """Parse range expression (.. or ..=)."""
        node = self.shift_expr()
        if self.peek().type == "OP" and self.peek().value in {"..", "..="}:
            inclusive = self.peek().value == "..="
            self.i += 1
            return Range(start=node, end=self.shift_expr(), inclusive=inclusive)
        return node

    def shift_expr(self) -> ExprNode:
        """Parse bit shift expression (<<, >>)."""
        node: ExprNode = self.add_expr()
        while self.peek().type == "OP" and self.peek().value in {"<<", ">>"}:
            self.i += 1
            node = BinOp(node, self.tokens[self.i - 1].value, self.add_expr())
        return node

    def add_expr(self) -> ExprNode:
        """Parse addition/subtraction expression."""
        node: ExprNode = self.power_expr()
        while self.peek().type == "OP" and self.peek().value in {"+", "-"}:
            self.i += 1
            node = BinOp(node, self.tokens[self.i - 1].value, self.power_expr())
        return node

    def mul_expr(self) -> ExprNode:
        """Parse multiplication/division/modulo expression."""
        node: ExprNode = self.unary_expr()
        while self.peek().type == "OP" and self.peek().value in {"*", "/", "//", "%"}:
            self.i += 1
            node = BinOp(node, self.tokens[self.i - 1].value, self.unary_expr())
        return node

    def power_expr(self) -> ExprNode:
        """Parse power (exponentiation) expression with right-associativity."""
        node = self.mul_expr()
        if self.peek().type == "OP" and self.peek().value == "**":
            self.i += 1
            node = BinOp(node, "**", self.power_expr())
        return node

    def unary_expr(self) -> ExprNode:
        """Parse unary expression (-, not, await)."""
        tok = self.peek()
        if tok.type == "OP" and tok.value == "-":
            self.match("OP", "-")
            return UnaryOp("-", self.unary_expr())
        if tok.type == "KW" and tok.value == "not":
            self.match("KW", "not")
            return UnaryOp("not", self.unary_expr())
        if tok.type == "KW" and tok.value == "await":
            self.match("KW", "await")
            return Await(self.unary_expr())
        return self.atom()

    def _parse_interpolated_string(
        self, content: str, tok: Token, string_type: str
    ) -> Tuple[List[str], List[ExprNode], List[str | None], List[str | None]]:
        """Parse interpolated string (f-string or t-string) content.

        Args:
            content: The string content with {} interpolations
            tok: The token for error reporting
            string_type: Either 'f' or 't' for error messages

        Returns:
            Tuple of (string_parts, exprs, format_specs, debug_exprs)
        """
        string_parts: List[str] = []  # Pure string parts
        exprs: List[ExprNode] = []  # Expression nodes
        formats: List[str | None] = []  # Format specifiers
        debug_exprs: List[str | None] = []  # Debug expression strings
        i = 0
        current_str = ""

        while i < len(content):
            brace_start = content.find("{", i)
            if brace_start == -1:
                # No more expressions, collect remaining string
                current_str += content[i:]
                break

            # Check for escaped braces {{
            if brace_start + 1 < len(content) and content[brace_start + 1] == "{":
                current_str += content[i:brace_start] + "{"
                i = brace_start + 2
                continue

            # Add string before brace to current accumulation
            current_str += content[i:brace_start]

            # Find matching }
            brace_end = content.find("}", brace_start)
            if brace_end == -1:
                raise SyntaxError(
                    f"Unclosed {{ in {string_type}-string at {tok.line}:{tok.col}"
                )

            # Save the string part accumulated so far
            if current_str or not string_parts:
                string_parts.append(current_str)
                current_str = ""

            # Parse the expression inside {}
            expr_str = content[brace_start + 1 : brace_end]
            format_spec, debug_expr_str = self._parse_format_spec(expr_str)

            # Parse the expression
            expr_tokens = Lexer(expr_str).tokenize()
            expr_tokens = [t for t in expr_tokens if t.type not in ("EOF", "NL")]
            expr_tokens.append(Token("EOF", "", tok.line, tok.col))
            expr_parser = Parser(expr_tokens)
            expr = expr_parser.expr()

            exprs.append(expr)
            formats.append(format_spec)
            debug_exprs.append(debug_expr_str)
            i = brace_end + 1

        # Add any remaining string
        if current_str or not string_parts:
            string_parts.append(current_str)

        # Handle }} escapes in final processing
        string_parts = [p.replace("}}", "}") for p in string_parts]

        return string_parts, exprs, formats, debug_exprs

    def _parse_format_spec(self, expr_str: str) -> tuple[str | None, str | None]:
        """Parse format specifier and debug flag from interpolation expression.

        Args:
            expr_str: The expression string inside {}

        Returns:
            Tuple of (format_spec, debug_expr_str)
        """
        format_spec = None
        debug_expr_str = None

        # Check for format specifier (split on : first)
        if ":" in expr_str:
            depth = 0
            colon_pos = -1
            for idx, ch in enumerate(expr_str):
                if ch in "([{":
                    depth += 1
                elif ch in ")]}":
                    depth -= 1
                elif ch == ":" and depth == 0:
                    colon_pos = idx
                    break

            if colon_pos != -1:
                format_spec = expr_str[colon_pos + 1 :]
                expr_str = expr_str[:colon_pos]

        # Check for debug format (x=) in the expr part
        ops = ["==", "!=", "<=", ">=", "+=", "-=", "*=", "**=", "/=", "//=", "%="]
        if "=" in expr_str and not any(op in expr_str for op in ops):
            if expr_str.endswith("="):
                debug_expr_str = expr_str[:-1].strip()
                expr_str = debug_expr_str

        return format_spec, debug_expr_str

    def _parse_function_params(self) -> Tuple[
        List[str],
        List[str | None],
        List[ExprNode | None],
        str | None,
        str | None,
        str | None,
        str | None,
    ]:
        """Parse function parameters including *args and **kwargs.

        Returns:
            Tuple of (params, param_types, defaults, vararg, vararg_type, kwarg, kwarg_type)
        """
        params: List[str] = []
        param_types: List[str | None] = []
        defaults: List[ExprNode | None] = []
        vararg: str | None = None
        vararg_type: str | None = None
        kwarg: str | None = None
        kwarg_type: str | None = None

        if not self.accept(")"):
            while True:
                tok = self.peek()
                if tok.type == "OP":
                    if tok.value == "*":
                        self.match("OP", "*")
                        vararg = self.match("ID").value
                        vararg_type = self.parse_type() if self.accept(":") else None
                        if self.accept(")"):
                            break
                        self.match(",")
                        continue
                    if tok.value == "**":
                        self.match("OP", "**")
                        kwarg = self.match("ID").value
                        kwarg_type = self.parse_type() if self.accept(":") else None
                        if self.accept(")"):
                            break
                        self.match(",")
                        continue

                # Regular parameter
                param_name = self.match("ID").value
                param_type = self.parse_type() if self.accept(":") else None
                params.append(param_name)
                param_types.append(param_type)

                if self.accept("OP", "="):
                    default_val = self.expr()
                    defaults.append(default_val)
                elif defaults:
                    raise SyntaxError(
                        f"Non-default parameter after default at "
                        f"{self.peek().line}:{self.peek().col}"
                    )

                if self.accept(")"):
                    break
                self.match(",")

        return params, param_types, defaults, vararg, vararg_type, kwarg, kwarg_type

    def _parse_list_literal_or_comp(self) -> ListComp | ListLit:
        """Parse list literal or list comprehension."""
        elements: List[ExprNode] = []
        if not self.accept("]"):
            self.accept("NL")
            first_expr: ExprNode = self.or_expr()
            self.accept("NL")

            # Check if this is a list comprehension
            if self.peek().type == "KW" and self.peek().value == "for":
                self.match("KW", "for")
                target: str = self.match("ID").value
                self.match("KW", "in")
                iter_expr: ExprNode = self.or_expr()
                condition: Optional[ExprNode] = None
                if self.peek().type == "KW" and self.peek().value == "if":
                    self.match("KW", "if")
                    condition = self.or_expr()
                self.accept("NL")
                self.match("]")
                return ListComp(
                    expr=first_expr,
                    target=target,
                    iter=iter_expr,
                    condition=condition,
                )

            # Regular list literal
            elements.append(first_expr)
            while True:
                self.accept("NL")
                if self.accept("]"):
                    break
                self.match(",")
                self.accept("NL")
                elements.append(self.expr())
                self.accept("NL")
        return ListLit(elements)

    def _parse_dict_or_set_literal_or_comp(
        self,
    ) -> SetLit | SetComp | DictComp | DictLit:
        """Parse dict/set literal or comprehension."""
        if self.accept("}"):
            return SetLit([])
        self.accept("NL")
        first_elem = self.or_expr()
        self.accept("NL")

        # Check for set comprehension
        if self.peek().type == "KW" and self.peek().value == "for":
            self.match("KW", "for")
            target = self.match("ID").value
            self.match("KW", "in")
            iter_expr = self.or_expr()
            condition = None
            if self.peek().type == "KW" and self.peek().value == "if":
                self.match("KW", "if")
                condition = self.or_expr()
            self.accept("NL")
            self.match("}")
            return SetComp(
                expr=first_elem,
                target=target,
                iter=iter_expr,
                condition=condition,
            )

        # Check if it's a dict (has :) or set (has , or })
        if self.accept(":"):
            first_val = self.or_expr()
            self.accept("NL")

            # Check for dict comprehension
            if self.peek().type == "KW" and self.peek().value == "for":
                self.match("KW", "for")
                target = self.match("ID").value
                self.match("KW", "in")
                iter_expr = self.or_expr()
                condition = None
                if self.peek().type == "KW" and self.peek().value == "if":
                    self.match("KW", "if")
                    condition = self.or_expr()
                self.accept("NL")
                self.match("}")
                return DictComp(
                    key=first_elem,
                    value=first_val,
                    target=target,
                    iter=iter_expr,
                    condition=condition,
                )

            # Dict literal
            pairs: List[Tuple[ExprNode, ExprNode]] = [(first_elem, first_val)]
            if not self.accept("}"):
                while True:
                    self.accept("NL")
                    if not self.accept(","):
                        break
                    self.accept("NL")
                    if self.accept("}"):
                        return DictLit(pairs)
                    key = self.expr()
                    self.match(":")
                    val = self.expr()
                    pairs.append((key, val))
                    self.accept("NL")
                self.match("}")
            return DictLit(pairs)

        # Set literal
        elements = [first_elem]
        if not self.accept("}"):
            while True:
                self.match(",")
                self.accept("NL")
                if self.accept("}"):
                    break
                elements.append(self.expr())
                self.accept("NL")
                if self.accept("}"):
                    break
        return SetLit(elements)

    def _parse_call_args(
        self,
    ) -> Tuple[List[ExprNode | Starred], List[Tuple[str, ExprNode]] | None]:
        """Parse function call arguments including *args and **kwargs."""
        args: List[ExprNode | Starred] = []
        kwargs: List[Tuple[str, ExprNode]] = []

        if not self.accept(")"):
            while True:
                # Handle splat (star) arguments: *expr
                if self.peek().type == "OP" and self.peek().value == "*":
                    self.match("OP", "*")
                    star_expr: ExprNode = self.expr()
                    args.append(Starred(star_expr))
                    if self.accept(")"):
                        break
                    self.match(",")
                    continue

                # Check if this is a keyword argument (name = value)
                if self.peek().type == "ID":
                    save_i = self.i
                    name_tok = self.match("ID")
                    if self.accept("OP", "="):
                        value: ExprNode = self.expr()
                        kwargs.append((name_tok.value, value))
                        if self.accept(")"):
                            break
                        self.match(",")
                        continue
                    # Not a keyword arg, backtrack
                    self.i = save_i

                # Regular positional argument
                args.append(self.expr())
                if self.accept(")"):
                    break
                self.match(",")

        return args, kwargs if kwargs else None

    def atom(self) -> ExprNode:
        """Parse atomic expression (literals, names, etc)."""
        tok: Token = self.peek()
        if self.accept("INT"):
            return Int(int(tok.value))
        if self.accept("FLOAT"):
            return Float(float(tok.value))
        if self.accept("STR"):
            return Str(tok.value)
        if self.accept("FSTR"):
            string_parts, exprs, formats, debug_exprs = self._parse_interpolated_string(
                tok.value, tok, "f"
            )
            return FString(string_parts, exprs, formats, debug_exprs)
        if self.accept("TSTR"):
            string_parts, exprs, formats, debug_exprs = self._parse_interpolated_string(
                tok.value, tok, "t"
            )
            return TString(string_parts, exprs, formats, debug_exprs)
        if tok.type == "KW" and tok.value in {"true", "false"}:
            self.i += 1
            return Bool(tok.value == "true")
        if tok.type == "KW" and tok.value == "none":
            self.i += 1
            return NoneLit()
        if self.accept("["):
            return self._parse_list_literal_or_comp()
        if self.accept("{"):
            return self._parse_dict_or_set_literal_or_comp()
        if self.accept("ID"):
            node: object = Name(tok.value)
            # Handle all postfix operations: calls, indexing, attribute access
            while True:
                if self.accept("("):
                    args, kwargs = self._parse_call_args()
                    node = Call(node, args, kwargs)
                elif self.accept("["):
                    node = self._parse_index_or_slice(node)
                elif self.accept("."):
                    attr_name = self.match("ID").value
                    node = Attr(node, attr_name)
                else:
                    break
            return node
        if self.accept("("):
            return self._parse_paren_or_lambda()
        raise SyntaxError(
            f"Unexpected token {tok.type} {tok.value} at {tok.line}:{tok.col}"
        )

    def _parse_index_or_slice(self, base: Name | Call | Index | Attr) -> Index:
        """Parse `[ ... ]` following a base expression, supporting slices.
        Assumes '[' already consumed. Returns `Index(base, idx_or_slice)`.
        """
        # Leading ':' or '::' -> missing start
        is_leading_colon = self.peek().type == ":" or (
            self.peek().type == "OP" and self.peek().value == "::"
        )
        if is_leading_colon:
            start = None
            if self.peek().type == "OP" and self.peek().value == "::":
                self.match("OP", "::")
                stop = None
                step = None if self.peek().type == "]" else self.expr()
                self.match("]")
                return Index(base, Slice(start, stop, step))
            # Single ':' case
            self.accept(":")
            if (
                self.peek().type == "]"
                or self.peek().type == ":"
                or (self.peek().type == "OP" and self.peek().value == "::")
            ):
                stop = None
            else:
                stop = self.expr()
            step = None
            if self.accept(":"):
                step = None if self.peek().type == "]" else self.expr()
            self.match("]")
            return Index(base, Slice(start, stop, step))

        # Parse start/index
        start = self.expr()
        # Slice variants
        if self.peek().type == "OP" and self.peek().value == "::":
            self.match("OP", "::")
            stop = None
            step = None if self.peek().type == "]" else self.expr()
            self.match("]")
            return Index(base, Slice(start, stop, step))
        if self.accept(":"):
            if (
                self.peek().type == "]"
                or self.peek().type == ":"
                or (self.peek().type == "OP" and self.peek().value == "::")
            ):
                stop = None
            else:
                stop = self.expr()
            step = None
            if self.accept(":"):
                step = None if self.peek().type == "]" else self.expr()
            self.match("]")
            return Index(base, Slice(start, stop, step))
        # Plain index
        self.match("]")
        return Index(base, start)

    def _parse_paren_or_lambda(self):
        """Parse parenthesized expression, tuple, or lambda with params in parens."""
        save_i = self.i
        params: List[str] = []
        is_param_seq = True
        is_empty_or_tuple = False

        if self.peek().type == ")":
            self.match(")")
            return TupleLit([])

        while True:
            if self.peek().type != "ID":
                is_param_seq = False
                break
            params.append(self.match("ID").value)
            if self.accept(")"):
                is_empty_or_tuple = True
                break
            if not self.accept(","):
                is_param_seq = False
                break

        if is_param_seq and is_empty_or_tuple:
            if self.accept("OP", "=>"):
                if self.accept("{"):
                    body: List[Statement] = []
                    while not self.accept("}"):
                        if self.accept("NL"):
                            continue
                        body.append(self.statement())
                    return Lambda(params=params, body=body, expr=None)
                expr = self.expr()
                return Lambda(params=params, body=None, expr=expr)

        # Reset and parse as tuple or parenthesized expression
        self.i = save_i
        elements: List[ExprNode] = []
        first_expr = self.expr()
        elements.append(first_expr)

        if self.accept(","):
            if not self.accept(")"):
                while True:
                    elements.append(self.expr())
                    if self.accept(")"):
                        break
                    self.match(",")
            return TupleLit(elements)
        self.match(")")
        return first_expr

    def parse_type(self) -> str:
        """Parse a type annotation and return as string"""
        type_parts: list[str] = []
        type_parts.append(self.match("ID").value)
        while self.accept("."):
            type_parts.append(self.match("ID").value)
        return ".".join(type_parts)

    def parse_unpack_target(self) -> list[str] | None:
        """Parse an unpacking assignment target like a, b or (a, b) or [a, b].

        Returns a list of variable names if an unpack pattern is present, otherwise None.
        """
        start_i: int = self.i

        def parse_list(end_token: str) -> list[str] | None:
            names: list[str] = []
            # Check if list is empty (immediate closing delimiter)
            if self.peek().type == end_token or (
                self.peek().type == "OP" and self.peek().value == end_token
            ):
                return None
            names.append(self.match("ID").value)
            # Accept commas (can be type "," or type "OP" with value ",")
            while self.accept(",") or self.accept("OP", ","):
                names.append(self.match("ID").value)
            return names

        # Parenthesized
        if self.accept("OP", "("):
            names = parse_list(")")
            if names is None or not self.accept("OP", ")"):
                # rollback on failure
                self.i = start_i
                return None
            return names

        # Bracketed
        if self.accept("OP", "["):
            names = parse_list("]")
            if names is None or not self.accept("OP", "]"):
                self.i = start_i
                return None
            return names

        # Bare comma-separated
        if self.peek().type == "ID":
            names = [self.match("ID").value]
            # Check for comma (can be type "," or type "OP" with value ",")
            if not (self.accept(",") or self.accept("OP", ",")):
                # single name, not an unpack pattern
                self.i = start_i
                return None
            while True:
                names.append(self.match("ID").value)
                if not (self.accept(",") or self.accept("OP", ",")):
                    break
            return names

        self.i = start_i
        return None
