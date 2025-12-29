"""Fast bytecode executor for Ekilang.

Provides optimized execution path using custom bytecode VM.
"""

from __future__ import annotations

from typing import Any, List, Dict, Optional
import operator
from ekilang.parser import (
    Int,
    Float,
    FString,
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

CMP_OPS = {
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
        code: List[tuple],
        consts: List[Any],
        names: List[str],
        functions: Dict[str, Any] = None,
    ) -> None:
        self.code = code
        self.consts = consts
        self.names = names
        self.functions = functions or {}

    def is_empty(self) -> bool:
        """Return True if program contains no instructions."""
        return len(self.code) == 0

    def instruction_count(self) -> int:
        """Return the number of bytecode instructions in the program."""
        return len(self.code)


class Compiler:
    """Compiles Ekilang AST to custom bytecode."""

    def __init__(self) -> None:
        self.code = []
        self.consts = []
        self.names = []
        self.const_map = {}
        self.name_map = {}
        self.functions = {}

    def add_const(self, value) -> int:
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

    def compile_expr(self, node) -> bool:
        """Compile expression node to bytecode. Returns True if compiled, False if unsupported."""

        ok = True

        # Explicitly reject unsupported expression types
        if isinstance(node, FString):
            ok = False

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
            # Reject splat or keyword args
            if (hasattr(node, "has_splat") and node.has_splat) or \
            (hasattr(node, "keywords") and node.keywords):
                ok = False
            else:
                for arg in node.args:
                    if not self.compile_expr(arg):
                        ok = False
                        break

                if ok:
                    if hasattr(node.func, "id"):
                        func_name = node.func.id
                        self.emit(CALL, (self.add_name(func_name), len(node.args)))
                    else:
                        ok = False  # Attr or complex call not supported

        elif isinstance(node, TString):
            for i, part in enumerate(node.parts):
                if isinstance(part, str):
                    self.emit(LOAD_CONST, self.add_const(part))
                else:
                    if not self.compile_expr(part):
                        ok = False
                        break
                    self.emit(CALL, (self.add_name("str"), 1))

                if i > 0:
                    self.emit(BINARY_OP, "+")

        else:
            ok = False

        return ok

    def compile_stmt(self, node) -> bool:
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
            if hasattr(node.target, "id"):
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
                        func_compiler.emit(
                            LOAD_CONST, func_compiler.add_const(None)
                        )
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

    def compile(self, mod) -> Optional[FastProgram]:
        """Compile entire module to bytecode program."""
        if not isinstance(mod, Module):
            return None

        for stmt in mod.body:
            if not self.compile_stmt(stmt):
                return None  # Fallback if unsupported

        self.emit(HALT, None)
        return FastProgram(self.code, self.consts, self.names, self.functions)


def compile_simple_loop(mod) -> Optional[FastProgram]:
    """Compile Ekilang module to fast bytecode if supported.

    Returns:
        FastProgram if compilation succeeds, None if unsupported features found
    """
    try:
        compiler = Compiler()
        result = compiler.compile(mod)
        return result
    except (TypeError, ValueError, RuntimeError, NotImplementedError):
        return None  # Fallback on known compilation errors


def run_fast(program: FastProgram, globals_ns: Dict[str, Any]) -> Dict[str, Any]:
    """Execute fast bytecode program.

    Args:
        program: Compiled FastProgram bytecode
        globals_ns: Global namespace dictionary

    Returns:
        Updated namespace after execution
    """
    stack = []
    pc = 0
    code = program.code
    consts = program.consts
    names = program.names
    env = dict(globals_ns)  # Copy to avoid mutation

    # Add compiled functions to environment
    for func_name, (func_prog, params) in program.functions.items():

        def make_func(prog, param_names):
            def func(*args) -> Any | None:
                local_env = dict(env)
                for i, param in enumerate(param_names):
                    if i < len(args):
                        local_env[param] = args[i]
                result_env = run_fast(prog, local_env)
                return result_env.get("__return__", None)

            return func

        env[func_name] = make_func(func_prog, params)

    while pc < len(code):
        op, arg = code[pc]
        pc += 1

        if op == LOAD_CONST:
            stack.append(consts[arg])
        elif op == LOAD_NAME:
            name = names[arg]
            if name in env:
                stack.append(env[name])
            else:
                raise NameError(f"Name '{name}' not defined")
        elif op == STORE_NAME:
            env[names[arg]] = stack.pop()
        elif op == BINARY_OP:
            b = stack.pop()
            a = stack.pop()
            stack.append(BIN_OPS[arg](a, b))
        elif op == COMPARE_OP:
            b = stack.pop()
            a = stack.pop()
            stack.append(CMP_OPS[arg](a, b))
        elif op == JUMP_IF_FALSE:
            cond = stack.pop()
            if not cond:
                pc = arg
        elif op == JUMP:
            pc = arg
        elif op == CALL:
            func_idx, nargs = arg
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
