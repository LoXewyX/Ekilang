"""Fast bytecode executor for Ekilang.

Provides optimized execution path using custom bytecode VM.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, cast
import operator
from .types import (
    ExprNode,
    Int,
    Float,
    FString,
    Starred,
    Statement,
    Str,
    Bool,
    Name,
    BinOp,
    Call,
    TString,
    NoneLit,
    Let,
    Assign,
    While,
    If,
    Return,
    ExprStmt,
    Fn,
    Module,
)
from ._rust_lexer import apply_binop, apply_compare  # pylint: disable=no-name-in-module

# Minimal opcode set for benchmark patterns
LOAD_CONST = "LOAD_CONST"
LOAD_NAME = "LOAD_NAME"
STORE_NAME = "STORE_NAME"
BINARY_OP = "BINARY_OP"
COMPARE_OP = "COMPARE_OP"
JUMP_IF_FALSE = "JUMP_IF_FALSE"
JUMP = "JUMP"
CALL = "CALL"
RETURN = "RETURN"
POP_TOP = "POP_TOP"
MAKE_FUNCTION = "MAKE_FUNCTION"
HALT = "HALT"

# Operator maps for fast dispatch
BIN_OPS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "%": operator.mod,
}

CMP_OPS: Dict[str, Any] = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}


class FastProgram:
    """Compiled bytecode program for fast execution."""

    def __init__(
        self,
        code: List[Tuple[str, Any]],
        consts: List[Any],
        names: List[str],
        functions: Optional[Dict[str, Tuple["FastProgram", List[str]]]] = None,
    ) -> None:
        self.code = code
        self.consts = consts
        self.names = names
        self.functions: Dict[str, Tuple["FastProgram", List[str]]] = functions or {}

    def is_empty(self) -> bool:
        """Return True if program contains no instructions."""
        return len(self.code) == 0

    def instruction_count(self) -> int:
        """Return the number of bytecode instructions in the program."""
        return len(self.code)


class Compiler:
    """Compiles Ekilang AST to custom bytecode."""

    def __init__(self) -> None:
        self.code: List[Tuple[str, Any]] = []
        self.consts: List[Any] = []
        self.names: List[str] = []
        self.const_map: Dict[Any, int] = {}
        self.name_map: Dict[str, int] = {}
        self.functions: Dict[str, Tuple[FastProgram, List[str]]] = {}

    def add_const(self, value: Any) -> int:
        """Add constant to pool and return its index."""
        if value not in self.const_map:
            idx = len(self.consts)
            self.consts.append(value)
            self.const_map[value] = idx
        return self.const_map[value]

    def add_name(self, name: str) -> int:
        """Add name to pool and return its index."""
        if name not in self.name_map:
            idx = len(self.names)
            self.names.append(name)
            self.name_map[name] = idx
        return self.name_map[name]

    def emit(self, op: str, arg: Any = None) -> int:
        """Emit an opcode instruction and return its address."""
        self.code.append((op, arg))
        return len(self.code) - 1

    def patch(self, addr: int, arg: Any) -> None:
        """Patch an instruction at given address with new argument."""
        op, _ = self.code[addr]
        self.code[addr] = (op, arg)

    def compile_expr(self, node: ExprNode | Starred) -> bool:
        """Compile expression node to bytecode. Returns True if compiled, False if unsupported."""

        ok = True

        # Explicitly reject unsupported expression types
        if isinstance(node, FString):
            # Fast path only supports simple concatenation without format/debug
            if any(node.formats) or any(node.debug_exprs):
                ok = False
            else:
                chunk_count = 0
                total_chunks = len(node.parts) + len(node.exprs)

                for expr_idx in range(len(node.exprs) + 1):
                    if expr_idx < len(node.parts):
                        self.emit(LOAD_CONST, self.add_const(node.parts[expr_idx]))
                        chunk_count += 1
                        if chunk_count > 1:
                            self.emit(BINARY_OP, "+")

                    if expr_idx < len(node.exprs):
                        if not self.compile_expr(node.exprs[expr_idx]):
                            ok = False
                            break
                        self.emit(CALL, (self.add_name("str"), 1))
                        chunk_count += 1
                        if chunk_count > 1:
                            self.emit(BINARY_OP, "+")

                if ok and total_chunks == 0:
                    self.emit(LOAD_CONST, self.add_const(""))

        elif isinstance(node, Int):
            self.emit(LOAD_CONST, self.add_const(node.value))

        elif isinstance(node, Float):
            self.emit(LOAD_CONST, self.add_const(node.value))

        elif isinstance(node, Str):
            self.emit(LOAD_CONST, self.add_const(node.value))

        elif isinstance(node, Bool):
            self.emit(LOAD_CONST, self.add_const(node.value))

        elif isinstance(node, NoneLit):
            self.emit(LOAD_CONST, self.add_const(None))

        elif isinstance(node, Name):
            self.emit(LOAD_NAME, self.add_name(node.id))

        elif isinstance(node, BinOp):
            if node.op in CMP_OPS or node.op in BIN_OPS:
                if not self.compile_expr(node.left):
                    ok = False
                elif not self.compile_expr(node.right):
                    ok = False
                else:
                    opcode = COMPARE_OP if node.op in CMP_OPS else BINARY_OP
                    self.emit(opcode, node.op)
            else:
                ok = False

        elif isinstance(node, Call):
            # Reject keyword args
            if node.kwargs is not None:
                ok = False
            else:
                for arg in node.args:
                    if not self.compile_expr(arg):
                        ok = False
                        break

                if ok:
                    if isinstance(node.func, Name):
                        func_name = node.func.id
                        self.emit(CALL, (self.add_name(func_name), len(node.args)))
                    else:
                        ok = False  # Attr or complex call not supported

        elif isinstance(node, TString):
            if any(node.formats) or any(node.debug_exprs):
                ok = False
            else:
                chunk_count = 0
                total_chunks = len(node.parts) + len(node.exprs)

                for expr_idx in range(len(node.exprs) + 1):
                    if expr_idx < len(node.parts):
                        self.emit(LOAD_CONST, self.add_const(node.parts[expr_idx]))
                        chunk_count += 1
                        if chunk_count > 1:
                            self.emit(BINARY_OP, "+")

                    if expr_idx < len(node.exprs):
                        if not self.compile_expr(node.exprs[expr_idx]):
                            ok = False
                            break
                        self.emit(CALL, (self.add_name("str"), 1))
                        chunk_count += 1
                        if chunk_count > 1:
                            self.emit(BINARY_OP, "+")

                if ok and total_chunks == 0:
                    self.emit(LOAD_CONST, self.add_const(""))

        else:
            ok = False

        return ok

    def compile_stmt(self, node: Statement) -> bool:
        """Compile statement node to bytecode. Returns True if compiled, False if unsupported."""

        ok = True

        if isinstance(node, Let):
            if isinstance(node.name, list):
                ok = False  # Unpacking not supported
            else:
                self.compile_expr(node.value)
                self.emit(STORE_NAME, self.add_name(node.name))

        elif isinstance(node, Assign):
            self.compile_expr(node.value)
            if isinstance(node.target, Name):
                self.emit(STORE_NAME, self.add_name(node.target.id))
            else:
                ok = False

        elif isinstance(node, While):
            loop_start = len(self.code)
            self.compile_expr(node.test)
            jump_end = self.emit(JUMP_IF_FALSE, 0)
            for stmt in node.body:
                if not self.compile_stmt(stmt):
                    ok = False
                    break
            if ok:
                self.emit(JUMP, loop_start)
                self.patch(jump_end, len(self.code))

        elif isinstance(node, If):
            self.compile_expr(node.test)
            jump_else = self.emit(JUMP_IF_FALSE, 0)

            for stmt in node.conseq:
                if not self.compile_stmt(stmt):
                    ok = False
                    break

            if ok and node.alt:
                jump_end = self.emit(JUMP, 0)
                self.patch(jump_else, len(self.code))
                for stmt in node.alt:
                    if not self.compile_stmt(stmt):
                        ok = False
                        break
                if ok:
                    self.patch(jump_end, len(self.code))
            else:
                self.patch(jump_else, len(self.code))

        elif isinstance(node, Return):
            if node.value:
                self.compile_expr(node.value)
            else:
                self.emit(LOAD_CONST, self.add_const(None))
            self.emit(RETURN, None)

        elif isinstance(node, ExprStmt):
            if not self.compile_expr(node.value):
                ok = False
            else:
                self.emit(POP_TOP, None)

        elif isinstance(node, Fn):
            # Unsupported fast-executor features
            if node.defaults or node.vararg or node.kwarg:
                ok = False
            else:
                func_compiler = Compiler()
                for stmt in node.body:
                    if not func_compiler.compile_stmt(stmt):
                        ok = False
                        break

                if ok:
                    # Implicit return None
                    if not func_compiler.code or func_compiler.code[-1][0] != RETURN:
                        func_compiler.emit(LOAD_CONST, func_compiler.add_const(None))
                        func_compiler.emit(RETURN, None)

                    func_prog = FastProgram(
                        func_compiler.code,
                        func_compiler.consts,
                        func_compiler.names,
                    )
                    self.functions[node.name] = (func_prog, node.params)
                    self.emit(MAKE_FUNCTION, self.add_name(node.name))

        else:
            ok = False

        return ok

    def compile(self, mod: Module) -> Optional[FastProgram]:
        """Compile entire module to bytecode program."""

        for stmt in mod.body:
            if not self.compile_stmt(stmt):
                return None  # Fallback if unsupported

        self.emit(HALT, None)
        return FastProgram(self.code, self.consts, self.names, self.functions)


def run_fast(program: FastProgram, globals_ns: Dict[str, Any]) -> Dict[str, Any]:
    """Execute fast bytecode program.

    Args:
        program: Compiled FastProgram bytecode
        globals_ns: Global namespace dictionary

    Returns:
        Updated namespace after execution
    """
    stack: List[Any] = []
    pc = 0
    code = program.code
    consts = program.consts
    names = program.names
    env = globals_ns  # Modify in place

    # Add compiled functions to environment
    for func_name, (func_prog, params) in program.functions.items():

        def make_func(prog: FastProgram, param_names: List[str]) -> Callable[..., Any]:
            def func(*args: Any) -> Any | None:
                local_env = dict(env)
                for i, param in enumerate(param_names):
                    if i < len(args):
                        local_env[param] = args[i]
                run_fast(prog, local_env)
                return local_env.get("__return__", None)

            return func

        env[func_name] = make_func(func_prog, params)

    while pc < len(code):
        op, arg = code[pc]
        pc += 1

        if op == LOAD_CONST:
            stack.append(consts[arg])
        elif op == LOAD_NAME:
            name = names[cast(int, arg)]
            if name in env:
                stack.append(env[name])
            else:
                raise NameError(f"Name '{name}' not defined")
        elif op == STORE_NAME:
            env[names[arg]] = stack.pop()
        elif op == BINARY_OP:
            b = stack.pop()
            a = stack.pop()
            stack.append(apply_binop(float(a), arg, float(b)))
        elif op == COMPARE_OP:
            b = stack.pop()
            a = stack.pop()
            stack.append(apply_compare(float(a), arg, float(b)))
        elif op == JUMP_IF_FALSE:
            cond = stack.pop()
            if not cond:
                pc = arg
        elif op == JUMP:
            pc = arg
        elif op == CALL:
            func_idx, nargs = cast(Tuple[int, int], arg)
            func_name = names[func_idx]
            args = [stack.pop() for _ in range(nargs)]
            args.reverse()
            if func_name in env:
                result = env[func_name](*args)
                stack.append(result if result is not None else None)
            else:
                raise NameError(f"Function '{func_name}' not defined")
        elif op == POP_TOP:
            if stack:
                stack.pop()
        elif op == MAKE_FUNCTION:
            # Function already added to env during setup
            pass
        elif op == RETURN:
            val = stack.pop() if stack else None
            env["__return__"] = val
            return env
        elif op == HALT:
            break
        else:
            raise RuntimeError(f"Unsupported opcode: {op}")

    return env
