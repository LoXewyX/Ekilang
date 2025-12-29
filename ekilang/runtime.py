"""Runtime execution engine for Ekilang.

Compiles Ekilang AST to Python AST and executes it.
Supports fast executor optimization for performance-critical code.
"""

from __future__ import annotations

import ast
from typing import Dict, Any
import os
from .lexer import Lexer
from .parser import (
    Parser,
    Starred,
    NullSafeAttr,
    NullCoalesce,
    Slice,
    DictComp,
    SetComp,
    TString,
    Class,
    Module,
    Let,
    Assign,
    AugAssign,
    Fn,
    AsyncFn,
    If,
    Match,
    While,
    For,
    Return,
    Break,
    Continue,
    ExprStmt,
    BinOp,
    Call,
    Index,
    Attr,
    Name,
    Int,
    Float,
    Str,
    Bool,
    NoneLit,
    ListLit,
    DictLit,
    TupleLit,
    SetLit,
    Lambda,
    UnaryOp,
    FString,
    TernaryOp,
    Range,
    ListComp,
    Use,
    Await,
    Cast,
    Pipe,
    Yield,
)
from .fast_executor import compile_simple_loop, run_fast
from .builtins import BUILTINS


OP_MAP = {
    "+": ast.Add(),
    "-": ast.Sub(),
    "*": ast.Mult(),
    "**": ast.Pow(),
    "/": ast.Div(),
    "//": ast.FloorDiv(),
    "%": ast.Mod(),
    "&": ast.BitAnd(),
    "|": ast.BitOr(),
    "^": ast.BitXor(),
    "<<": ast.LShift(),
    ">>": ast.RShift(),
    ">": ast.Gt(),
    "<": ast.Lt(),
    ">=": ast.GtE(),
    "<=": ast.LtE(),
    "==": ast.Eq(),
    "!=": ast.NotEq(),
    "is": ast.Is(),
    "is not": ast.IsNot(),
    "in": ast.In(),
    "not in": ast.NotIn(),
    "+=": ast.Add(),
    "-=": ast.Sub(),
    "*=": ast.Mult(),
    "/=": ast.Div(),
    "%=": ast.Mod(),
    "&=": ast.BitAnd(),
    "|=": ast.BitOr(),
    "^=": ast.BitXor(),
    "<<=": ast.LShift(),
    ">>=": ast.RShift(),
    "**=": ast.Pow(),
    "//=": ast.FloorDiv(),
}


class CodeGen:
    """Generates Python AST from Ekilang AST."""

    def __init__(self) -> None:
        self.lambda_counter = 0
        self.lambda_defs = []  # Store lambda function definitions

    def _stmts(self, nodes) -> list:
        """Convert list of Ekilang statement nodes to Python AST statements."""
        out = []
        for n in nodes:
            conv = self.stmt(n)
            if conv is not None:
                out.append(conv)
        return out

    def get_lambda_name(self) -> str:
        """Generate unique name for anonymous lambda function."""
        self.lambda_counter += 1
        return f"__lambda_{self.lambda_counter}"

    def expr(self, node) -> ast.AST:
        """Convert Ekilang expression node to Python AST expression."""
        if isinstance(node, Int):
            return ast.Constant(node.value)
        if isinstance(node, Float):
            return ast.Constant(node.value)
        if isinstance(node, Str):
            return ast.Constant(node.value)
        if isinstance(node, Bool):
            # Convert Bool node to Python's True or False
            return ast.Constant(value=node.value)
        if isinstance(node, NoneLit):
            return ast.Constant(None)
        if isinstance(node, ListLit):
            return ast.List(elts=[self.expr(e) for e in node.elements], ctx=ast.Load())
        if isinstance(node, DictLit):
            return ast.Dict(
                keys=[self.expr(k) for k, _ in node.pairs],
                values=[self.expr(v) for _, v in node.pairs],
            )
        if isinstance(node, Lambda):
            if node.expr is not None:
                # Single-line lambda: (x) => x + 1
                return ast.Lambda(
                    args=ast.arguments(
                        posonlyargs=[],
                        args=[ast.arg(arg=p) for p in node.params],
                        kwonlyargs=[],
                        kw_defaults=[],
                        defaults=[],
                    ),
                    body=self.expr(node.expr),
                )
            # Block-style lambda: (a, b) => { ... }
            # Create a named function and reference it
            fn_name = self.get_lambda_name()
            fn_body = []
            for i, stmt in enumerate(node.body):
                ast_stmt = self.stmt(stmt)
                if ast_stmt is None:
                    continue
                # Convert last ExprStmt to Return
                if i == len(node.body) - 1 and isinstance(stmt, ExprStmt):
                    fn_body.append(ast.Return(value=ast_stmt.value))
                elif i == len(node.body) - 1 and isinstance(stmt, If):
                    self._convert_if_to_return(ast_stmt)
                    fn_body.append(ast_stmt)
                else:
                    fn_body.append(ast_stmt)
            if not fn_body:
                fn_body = [ast.Pass()]
            fn_def = ast.FunctionDef(
                name=fn_name,
                args=ast.arguments(
                    posonlyargs=[],
                    args=[ast.arg(arg=p) for p in node.params],
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[],
                ),
                body=fn_body,
                decorator_list=[],
                returns=None,
            )
            self.lambda_defs.append(fn_def)
            return ast.Name(id=fn_name, ctx=ast.Load())
        if isinstance(node, UnaryOp):
            if node.op == "-":
                return ast.UnaryOp(op=ast.USub(), operand=self.expr(node.operand))
            if node.op == "not":
                return ast.UnaryOp(op=ast.Not(), operand=self.expr(node.operand))
            raise TypeError(f"Unsupported unary op {node.op}")
        if isinstance(node, Name):
            return ast.Name(id=node.id, ctx=ast.Load())
        if isinstance(node, BinOp):
            if node.op in {
                ">",
                "<",
                ">=",
                "<=",
                "==",
                "!=",
                "is",
                "is not",
                "in",
                "not in",
            }:
                return ast.Compare(
                    left=self.expr(node.left),
                    ops=[OP_MAP[node.op]],
                    comparators=[self.expr(node.right)],
                )
            if node.op == "and":
                return ast.BoolOp(
                    op=ast.And(), values=[self.expr(node.left), self.expr(node.right)]
                )
            if node.op == "or":
                return ast.BoolOp(
                    op=ast.Or(), values=[self.expr(node.left), self.expr(node.right)]
                )
            return ast.BinOp(
                left=self.expr(node.left),
                op=OP_MAP[node.op],
                right=self.expr(node.right),
            )
        if isinstance(node, Call):
            # Support Starred (splat) args

            py_args = []
            for a in node.args:
                if isinstance(a, Starred):
                    py_args.append(
                        ast.Starred(value=self.expr(a.value), ctx=ast.Load())
                    )
                else:
                    py_args.append(self.expr(a))
            keywords = []
            if node.kwargs:
                keywords = [
                    ast.keyword(arg=name, value=self.expr(val))
                    for name, val in node.kwargs
                ]
            return ast.Call(
                func=self.expr(node.func),
                args=py_args,
                keywords=keywords,
            )
        if isinstance(node, Attr):
            return ast.Attribute(
                value=self.expr(node.value), attr=node.attr, ctx=ast.Load()
            )
        if isinstance(node, NullSafeAttr):
            # value is None ? None : value.attr
            cond = ast.Compare(
                left=self.expr(node.value),
                ops=[ast.Is()],
                comparators=[ast.Constant(None)],
            )
            return ast.IfExp(
                test=cond,
                body=ast.Constant(None),
                orelse=ast.Attribute(
                    value=self.expr(node.value), attr=node.attr, ctx=ast.Load()
                ),
            )
        if isinstance(node, NullCoalesce):
            # left if left is not None else right
            cond = ast.Compare(
                left=self.expr(node.left),
                ops=[ast.IsNot()],
                comparators=[ast.Constant(None)],
            )
            return ast.IfExp(
                test=cond, body=self.expr(node.left), orelse=self.expr(node.right)
            )
        if isinstance(node, Index):
            # Check if index is a Slice object

            if isinstance(node.index, Slice):
                # Generate ast.Slice for slicing operations
                slice_node = ast.Slice(
                    lower=(
                        self.expr(node.index.start)
                        if node.index.start is not None
                        else None
                    ),
                    upper=(
                        self.expr(node.index.stop)
                        if node.index.stop is not None
                        else None
                    ),
                    step=(
                        self.expr(node.index.step)
                        if node.index.step is not None
                        else None
                    ),
                )
                return ast.Subscript(
                    value=self.expr(node.value), slice=slice_node, ctx=ast.Load()
                )
            return ast.Subscript(
                value=self.expr(node.value),
                slice=self.expr(node.index),
                ctx=ast.Load(),
            )
        if isinstance(node, FString):
            # Build f-string with format specifiers
            result = None
            for idx, part in enumerate(node.parts):
                if isinstance(part, str):
                    # String literal part
                    part_ast = ast.Constant(part)
                else:
                    # Expression part - evaluate and format
                    expr_ast = self.expr(part)
                    format_spec = node.formats[idx] if idx < len(node.formats) else None
                    debug_expr = (
                        node.debug_exprs[idx] if idx < len(node.debug_exprs) else None
                    )

                    if debug_expr:
                        # Debug format: x=5 shows as "x=5"
                        debug_prefix = ast.Constant(debug_expr + "=")
                        if format_spec:
                            # Apply format to the value part
                            formatted_value = ast.Call(
                                func=ast.Name(id="format", ctx=ast.Load()),
                                args=[expr_ast, ast.Constant(format_spec)],
                                keywords=[],
                            )
                            part_ast = ast.BinOp(
                                left=debug_prefix, op=ast.Add(), right=formatted_value
                            )
                        else:
                            # No format, just convert to string
                            str_value = ast.Call(
                                func=ast.Name(id="str", ctx=ast.Load()),
                                args=[expr_ast],
                                keywords=[],
                            )
                            part_ast = ast.BinOp(
                                left=debug_prefix, op=ast.Add(), right=str_value
                            )
                    elif format_spec:
                        # Apply format specifier
                        part_ast = ast.Call(
                            func=ast.Name(id="format", ctx=ast.Load()),
                            args=[expr_ast, ast.Constant(format_spec)],
                            keywords=[],
                        )
                    else:
                        # No format, just convert to string
                        part_ast = ast.Call(
                            func=ast.Name(id="str", ctx=ast.Load()),
                            args=[expr_ast],
                            keywords=[],
                        )

                if result is None:
                    result = part_ast
                else:
                    # Concatenate with +
                    result = ast.BinOp(left=result, op=ast.Add(), right=part_ast)
            return result if result is not None else ast.Constant("")
        if isinstance(node, TString):
            # Build t-string (template string) with format specifiers
            result = None
            for idx, part in enumerate(node.parts):
                if isinstance(part, str):
                    # String literal part
                    part_ast = ast.Constant(part)
                else:
                    # Expression part - evaluate and format
                    expr_ast = self.expr(part)
                    format_spec = node.formats[idx] if idx < len(node.formats) else None
                    debug_expr = (
                        node.debug_exprs[idx] if idx < len(node.debug_exprs) else None
                    )

                    if debug_expr:
                        # Debug format: x=5 shows as "x=5"
                        debug_prefix = ast.Constant(debug_expr + "=")
                        if format_spec:
                            # Apply format to the value part
                            formatted_value = ast.Call(
                                func=ast.Name(id="format", ctx=ast.Load()),
                                args=[expr_ast, ast.Constant(format_spec)],
                                keywords=[],
                            )
                            part_ast = ast.BinOp(
                                left=debug_prefix, op=ast.Add(), right=formatted_value
                            )
                        else:
                            # No format, just convert to string
                            str_value = ast.Call(
                                func=ast.Name(id="str", ctx=ast.Load()),
                                args=[expr_ast],
                                keywords=[],
                            )
                            part_ast = ast.BinOp(
                                left=debug_prefix, op=ast.Add(), right=str_value
                            )
                    elif format_spec:
                        # Apply format specifier
                        part_ast = ast.Call(
                            func=ast.Name(id="format", ctx=ast.Load()),
                            args=[expr_ast, ast.Constant(format_spec)],
                            keywords=[],
                        )
                    else:
                        # No format, just convert to string
                        part_ast = ast.Call(
                            func=ast.Name(id="str", ctx=ast.Load()),
                            args=[expr_ast],
                            keywords=[],
                        )

                if result is None:
                    result = part_ast
                else:
                    # Concatenate with +
                    result = ast.BinOp(left=result, op=ast.Add(), right=part_ast)
            return result if result is not None else ast.Constant("")
        if isinstance(node, TernaryOp):
            # Convert to IfExp: test if true else false
            return ast.IfExp(
                test=self.expr(node.test),
                body=self.expr(node.if_true),
                orelse=self.expr(node.if_false),
            )
        if isinstance(node, Range):
            # Create range object: range(start, end) or range(start, end+1)
            start = self.expr(node.start)
            end = self.expr(node.end)
            if node.inclusive:
                # range(start, end+1)
                end = ast.BinOp(left=end, op=ast.Add(), right=ast.Constant(1))
            return ast.Call(
                func=ast.Name(id="range", ctx=ast.Load()),
                args=[start, end],
                keywords=[],
            )
        if isinstance(node, ListComp):
            # Build list comprehension: [expr for target in iter if condition]
            comprehension = ast.comprehension(
                target=ast.Name(id=node.target, ctx=ast.Store()),
                iter=self.expr(node.iter),
                ifs=[self.expr(node.condition)] if node.condition else [],
                is_async=0,
            )
            return ast.ListComp(elt=self.expr(node.expr), generators=[comprehension])
        if isinstance(node, DictComp):
            # Build dict comprehension: {key: value for target in iter if condition}
            comprehension = ast.comprehension(
                target=ast.Name(id=node.target, ctx=ast.Store()),
                iter=self.expr(node.iter),
                ifs=[self.expr(node.condition)] if node.condition else [],
                is_async=0,
            )
            return ast.DictComp(
                key=self.expr(node.key),
                value=self.expr(node.value),
                generators=[comprehension],
            )
        if isinstance(node, SetComp):
            # Build set comprehension: {expr for target in iter if condition}
            comprehension = ast.comprehension(
                target=ast.Name(id=node.target, ctx=ast.Store()),
                iter=self.expr(node.iter),
                ifs=[self.expr(node.condition)] if node.condition else [],
                is_async=0,
            )
            return ast.SetComp(elt=self.expr(node.expr), generators=[comprehension])
        if isinstance(node, Await):
            return ast.Await(value=self.expr(node.value))
        if isinstance(node, Cast):
            type_map = {
                "int": "int",
                "str": "str",
                "float": "float",
                "bool": "bool",
                "list": "list",
                "tuple": "tuple",
                "set": "set",
                "dict": "dict",
            }
            cast_func = type_map.get(node.target_type, node.target_type)
            return ast.Call(
                func=ast.Name(id=cast_func, ctx=ast.Load()),
                args=[self.expr(node.value)],
                keywords=[],
            )
        if isinstance(node, Pipe):
            # Transform pipe operators into function calls
            # data |> func means func(data)
            # func <| data means func(data)
            if node.op == "|>":
                # Forward pipe: left |> right
                # right should be a callable, left is the argument
                return ast.Call(
                    func=self.expr(node.right), args=[self.expr(node.left)], keywords=[]
                )
            if node.op == "<|":
                # Backward pipe: left <| right
                # left should be a callable, right is the argument
                return ast.Call(
                    func=self.expr(node.left), args=[self.expr(node.right)], keywords=[]
                )
        if isinstance(node, TupleLit):
            return ast.Tuple(elts=[self.expr(e) for e in node.elements], ctx=ast.Load())
        if isinstance(node, SetLit):
            return ast.Set(elts=[self.expr(e) for e in node.elements])
        raise TypeError(f"Unsupported expr node: {type(node)}")

    def _convert_if_to_return(self, if_stmt):
        """Convert last expression statement in if branches to return (recursively)"""
        # Handle the consequence body
        if if_stmt.body:
            last = if_stmt.body[-1]
            if isinstance(last, ast.Expr):
                if_stmt.body[-1] = ast.Return(value=last.value)
            elif isinstance(last, ast.If):
                # Recursively convert nested if statements
                self._convert_if_to_return(last)

        # Handle the else/elif branches
        if if_stmt.orelse:
            last = if_stmt.orelse[-1]
            if isinstance(last, ast.Expr):
                if_stmt.orelse[-1] = ast.Return(value=last.value)
            elif isinstance(last, ast.If):
                # Recursively convert nested if statements
                self._convert_if_to_return(last)

    def stmt(self, node):
        """Convert Ekilang statement node to Python AST statement."""
        if isinstance(node, Class):
            # Generate class definition
            bases = [ast.Name(id=base, ctx=ast.Load()) for base in node.bases]
            class_body = self._stmts(node.body) if node.body else [ast.Pass()]
            decorator_list = (
                [self.expr(d) for d in node.decorators] if node.decorators else []
            )
            return ast.ClassDef(
                name=node.name,
                bases=bases,
                keywords=[],
                body=class_body,
                decorator_list=decorator_list,
            )
        if isinstance(node, Let):
            # Handle both single assignment and unpacking
            if isinstance(node.name, list):
                # Multiple names: unpacking assignment
                # x, y, z = value  ->  x, y, z = value
                targets = [
                    ast.Tuple(
                        elts=[ast.Name(id=name, ctx=ast.Store()) for name in node.name],
                        ctx=ast.Store(),
                    )
                ]
            else:
                # Single name: regular assignment
                targets = [ast.Name(id=node.name, ctx=ast.Store())]

            return ast.Assign(targets=targets, value=self.expr(node.value))
        if isinstance(node, Assign):
            # Handle both Name and Attr targets

            if isinstance(node.target, Name):
                target = ast.Name(id=node.target.id, ctx=ast.Store())
            elif isinstance(node.target, Attr):
                target = ast.Attribute(
                    value=self.expr(node.target.value),
                    attr=node.target.attr,
                    ctx=ast.Store(),
                )
            else:
                # Fallback: try to convert the target expression
                target = self.expr(node.target)
                # Change ctx to Store
                if hasattr(target, "ctx"):
                    target.ctx = ast.Store()
            return ast.Assign(targets=[target], value=self.expr(node.value))
        if isinstance(node, AugAssign):
            # Handle both Name and Attr targets

            if isinstance(node.target, Name):
                target = ast.Name(id=node.target.id, ctx=ast.Store())
            elif isinstance(node.target, Attr):
                target = ast.Attribute(
                    value=self.expr(node.target.value),
                    attr=node.target.attr,
                    ctx=ast.Store(),
                )
            else:
                # Fallback
                target = self.expr(node.target)
                if hasattr(target, "ctx"):
                    target.ctx = ast.Store()
            return ast.AugAssign(
                target=target,
                op=OP_MAP[node.op],
                value=self.expr(node.value),
            )
        if isinstance(node, ExprStmt):
            if isinstance(node.value, (Str, FString, TString)):
                # Discard bare string expressions (acts like Python's no-op/docstring behavior)
                return None
            return ast.Expr(value=self.expr(node.value))
        if isinstance(node, Return):
            return ast.Return(
                value=self.expr(node.value) if node.value is not None else None
            )
        if isinstance(node, Yield):
            # In Python AST, a bare yield in statement position is an Expr(Yield(...))
            return ast.Expr(
                value=ast.Yield(
                    value=self.expr(node.value) if node.value is not None else None
                )
            )
        if isinstance(node, If):
            return ast.If(
                test=self.expr(node.test),
                body=self._stmts(node.conseq),
                orelse=self._stmts(node.alt or []),
            )
        if isinstance(node, Match):
            # Compile match to if-elif chain
            subject = self.expr(node.subject)
            orelse = []
            for case in reversed(node.cases):
                if case.pattern is None:  # _
                    test = ast.Constant(True)
                elif isinstance(case.pattern, list):
                    # x in [p1, p2, p3]
                    test = ast.Compare(
                        left=subject,
                        ops=[ast.In()],
                        comparators=[
                            ast.List(
                                elts=[self.expr(p) for p in case.pattern],
                                ctx=ast.Load(),
                            )
                        ],
                    )
                else:
                    test = ast.Compare(
                        left=subject,
                        ops=[ast.Eq()],
                        comparators=[self.expr(case.pattern)],
                    )
                if case.guard:
                    test = ast.BoolOp(
                        op=ast.And(),
                        values=[test, self.expr(case.guard)],
                    )
                body = self._stmts(case.body)
                orelse = [ast.If(test=test, body=body, orelse=orelse)]
            return orelse[0] if orelse else ast.Pass()
        if isinstance(node, While):
            return ast.While(
                test=self.expr(node.test), body=self._stmts(node.body), orelse=[]
            )
        if isinstance(node, For):
            # Support tuple/list unpacking in for loop targets
            if isinstance(node.target, list):
                target = ast.Tuple(
                    elts=[ast.Name(id=t.id, ctx=ast.Store()) for t in node.target],
                    ctx=ast.Store(),
                )
            else:
                target = ast.Name(id=node.target.id, ctx=ast.Store())
            return ast.For(
                target=target,
                iter=self.expr(node.iter),
                body=self._stmts(node.body),
                orelse=[],
            )
        if isinstance(node, Break):
            return ast.Break()
        if isinstance(node, Continue):
            return ast.Continue()
        if isinstance(node, Fn):
            # Build defaults list
            defaults_list = []
            if node.defaults:
                defaults_list = [self.expr(d) for d in node.defaults]

            fn_args = ast.arguments(
                posonlyargs=[],
                args=[
                    ast.arg(arg=p, annotation=ast.Constant(t) if t else None)
                    for p, t in zip(node.params, node.param_types)
                ],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=defaults_list,
                vararg=(
                    ast.arg(
                        arg=node.vararg,
                        annotation=(
                            ast.Constant(node.vararg_type) if node.vararg_type else None
                        ),
                    )
                    if node.vararg
                    else None
                ),
                kwarg=(
                    ast.arg(
                        arg=node.kwarg,
                        annotation=(
                            ast.Constant(node.kwarg_type) if node.kwarg_type else None
                        ),
                    )
                    if node.kwarg
                    else None
                ),
            )
            decorator_list = (
                [self.expr(d) for d in node.decorators] if node.decorators else []
            )
            fn_def = ast.FunctionDef(
                name=node.name,
                args=fn_args,
                body=self._stmts(node.body) or [ast.Pass()],
                decorator_list=decorator_list,
                returns=ast.Constant(node.return_type) if node.return_type else None,
            )
            return fn_def
        if isinstance(node, AsyncFn):
            # Build defaults list
            defaults_list = []
            if node.defaults:
                defaults_list = [self.expr(d) for d in node.defaults]

            fn_args = ast.arguments(
                posonlyargs=[],
                args=[
                    ast.arg(arg=p, annotation=ast.Constant(t) if t else None)
                    for p, t in zip(node.params, node.param_types)
                ],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=defaults_list,
                vararg=(
                    ast.arg(
                        arg=node.vararg,
                        annotation=(
                            ast.Constant(node.vararg_type) if node.vararg_type else None
                        ),
                    )
                    if node.vararg
                    else None
                ),
                kwarg=(
                    ast.arg(
                        arg=node.kwarg,
                        annotation=(
                            ast.Constant(node.kwarg_type) if node.kwarg_type else None
                        ),
                    )
                    if node.kwarg
                    else None
                ),
            )
            decorator_list = (
                [self.expr(d) for d in node.decorators] if node.decorators else []
            )
            fn_def = ast.AsyncFunctionDef(
                name=node.name,
                args=fn_args,
                body=self._stmts(node.body) or [ast.Pass()],
                decorator_list=decorator_list,
                returns=ast.Constant(node.return_type) if node.return_type else None,
            )
            return fn_def

        if isinstance(node, Use):
            aliases = [ast.alias(name=item.name, asname=item.alias) for item in node.items]
            mod_elems = node.module
            
            if not node.module:
                return ast.Import(names=aliases)

            level = 0
            if mod_elems and all(c == "." for c in mod_elems[0]):
                level = len(mod_elems[0])
                mod_elems = mod_elems[1:]

            module = ".".join(mod_elems) or None

            return ast.ImportFrom(
                module=module,
                names=aliases,
                level=level,
            )

    def module(self, mod: Module) -> ast.Module:
        """Convert Ekilang Module to Python AST Module."""
        body = self.lambda_defs + self._stmts(mod.body)
        return ast.Module(body=body, type_ignores=[])


def compile_module(mod: Module) -> Any:
    """Compile Ekilang Module to Python bytecode."""
    gen = CodeGen()
    m = gen.module(mod)
    ast.fix_missing_locations(m)
    code = compile(m, filename="<ekilang>", mode="exec")
    return code


def execute(
    mod: Module,
    globals_ns: Dict[str, Any] | None = None,
    current_file: str = None,
    code_obj: Any = None,
    use_fast: bool = False,
) -> Dict[str, Any]:
    """Execute Ekilang module with optional fast executor optimization.

    Args:
        mod: Parsed Ekilang Module AST
        globals_ns: Optional global namespace
        current_file: Current file path for relative imports
        code_obj: Pre-compiled code object (skips parsing)
        use_fast: Enable fast executor optimization

    Returns:
        Execution namespace with all defined variables
    """
    # Use is imported at module level

    # If code_obj provided, skip parsing and go straight to execution
    if code_obj is not None:
        ns = globals_ns if globals_ns is not None else {}
        # Only inject builtins once when marker is absent
        if not ns.get("__ekilang_builtins_loaded__"):
            ns.update(BUILTINS)
            ns["__ekilang_builtins_loaded__"] = True
        exec(code_obj, ns)
        return ns

    # Process Use statements first to load .eki modules
    imports_ns: Dict[str, Any] = {}
    filtered_body = []
    has_use = False

    def handle_use_import(stmt):
        # Handle .eki file imports
        # For relative imports like ['.', 'helpers'], construct as ".::helpers"
        if stmt.module and stmt.module[0].startswith("."):
            mod_path = (
                stmt.module[0] + "::" + ".".join(stmt.module[1:])
                if len(stmt.module) > 1
                else stmt.module[0]
            )
        else:
            mod_path = ".".join(stmt.module)
        try:
            ekilang_module = load_ekilang_module(mod_path, current_file=current_file)
            # Import the requested items
            for item in stmt.items:
                if item.name == "*":
                    # Import all public items
                    for name, value in ekilang_module.items():
                        if not name.startswith("_"):
                            imports_ns[name] = value
                else:
                    # Import specific item
                    if item.name in ekilang_module:
                        alias_name = item.alias or item.name
                        imports_ns[alias_name] = ekilang_module[item.name]
                    else:
                        raise ImportError(
                            f"Cannot import '{item.name}' from '{mod_path}'"
                        )
            return True
        except ImportError:
            # If .eki import fails for relative imports, raise the error
            # (don't fall back to Python imports for relative paths)
            if stmt.module and stmt.module[0].startswith("."):
                raise
            # For absolute imports, keep the Use statement for Python module import
            filtered_body.append(stmt)
            return False

    for stmt in mod.body:
        if isinstance(stmt, Use):
            has_use = True
            handle_use_import(stmt)
        else:
            filtered_body.append(stmt)

    # Create a new module with filtered body (without Use statements that were handled)
    mod_filtered = Module(body=filtered_body)

    # Standard execution path
    code = compile_module(mod_filtered)
    ns: Dict[str, Any] = {}
    ns.update(BUILTINS)
    ns.update(imports_ns)  # Add imported items
    if globals_ns:
        ns.update(globals_ns)
    # Set __file__ if current_file is provided
    if current_file is not None:
        import os

        ns["__file__"] = os.path.abspath(current_file)
    # Only set __name__ to '__main__' if not already set (i.e., for main script)
    if "__name__" not in ns:
        ns["__name__"] = "__main__"
    exec(code, ns, ns)
    return ns


# Module cache for imported .eki files
MODULE_CACHE: Dict[str, Dict[str, Any]] = {}


def load_ekilang_module(
    module_path: str, search_paths: list = None, current_file: str = None
) -> Dict[str, Any]:
    """Load and cache a .eki module file.

    Supports:
    - Absolute imports: lib::math_utils
    - Relative imports: .::helper (current dir), ..::parent_helper (parent dir)

    Args:
        module_path: Module path (e.g., 'lib.utils' or '.::helper')
        search_paths: Directories to search for modules
        current_file: Current file path for relative imports

    Returns:
        Module namespace dictionary
    """
    # os, Lexer, Parser imported at module level

    if module_path in MODULE_CACHE:
        return MODULE_CACHE[module_path]

    # Handle relative imports
    if module_path.startswith("."):
        if current_file is None:
            raise ImportError("Relative imports require current file context")

        current_dir = os.path.dirname(os.path.abspath(current_file))

        # Count leading dots for parent traversal
        dot_count = 0
        for char in module_path:
            if char == ".":
                dot_count += 1
            else:
                break

        # Go up dot_count-1 levels (. = current, .. = parent, etc.)
        base_dir = current_dir
        for _ in range(dot_count - 1):
            base_dir = os.path.dirname(base_dir)

        # Get the module name part (after the dots)
        module_name = module_path[dot_count:]
        if module_name.startswith(":"):
            module_name = module_name[2:]  # Remove :: if present

        # Convert dots in module name to path separators
        module_file = module_name.replace(".", os.sep)

        # Build the file path
        candidate = os.path.join(base_dir, module_file)
        if not candidate.endswith(".eki"):
            candidate += ".eki"

        if os.path.isfile(candidate):
            file_path = candidate
        else:
            # Try as directory with __init__.eki
            candidate_init = os.path.join(base_dir, module_file, "__init__.eki")
            if os.path.isfile(candidate_init):
                file_path = candidate_init
            else:
                raise ImportError(f"Cannot find relative module: {module_path}")
    else:
        # Absolute imports
        # Default search paths
        if search_paths is None:
            search_paths = [
                os.getcwd(),
            ]

        # Convert module path: "lib.math_utils" -> "lib/math_utils"
        module_file = module_path.replace(".", os.sep)

        # Try to find the file
        file_path = None
        for search_path in search_paths:
            candidate = os.path.join(search_path, module_file)
            if not candidate.endswith(".eki"):
                candidate += ".eki"
            if os.path.isfile(candidate):
                file_path = candidate
                break

        if not file_path:
            # Try as a directory with __init__.eki
            for search_path in search_paths:
                candidate = os.path.join(search_path, module_file, "__init__.eki")
                if os.path.isfile(candidate):
                    file_path = candidate
                    break

        if not file_path:
            raise ImportError(f"Cannot find module: {module_path}")

    # Read and parse the file
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()

    tokens = Lexer(source).tokenize()
    parsed = Parser(tokens).parse()

    # Determine module name for __name__
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    # Set __name__ for the imported module
    module_ns = execute(
        parsed, globals_ns={"__name__": module_name}, current_file=file_path
    )

    # Cache it
    MODULE_CACHE[module_path] = module_ns

    return module_ns
