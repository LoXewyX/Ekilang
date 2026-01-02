"""Command-line interface for Ekilang interpreter.

Provides REPL, file execution, and Python AST dump functionality.
"""

from __future__ import annotations

import argparse
import sys
import ast
import traceback
from typing import Any, Dict, Literal

from .lexer import Lexer
from .parser import Parser
from .runtime import execute, CodeGen


def run_source(
    src: str, dump_py: bool = False, current_file: str | None = None
) -> Dict[str, Any]:
    """Run Ekilang source code.

    Fast executor is automatically tried when possible.

    Args:
        src: Ekilang source code string
        dump_py: If True, print generated Python AST
        current_file: Path to current file for relative imports

    Returns:
        Execution namespace dictionary
    """
    tokens = Lexer(src).tokenize()
    mod = Parser(tokens).parse()

    if dump_py:
        # regenerate python AST and unparse to source
        gen = CodeGen()
        m = gen.module(mod)
        ast.fix_missing_locations(m)
        py_src = ast.unparse(m)
        print(py_src)
        return {}  # Return empty namespace without executing

    # Execute with fast_executor enabled by default
    return execute(mod, current_file=current_file)


def main(argv: list[str] | None = None) -> Literal[0] | Literal[1]:
    """Main entry point for Ekilang CLI.

    Args:
        argv: Command-line arguments (for testing)

    Returns:
        Exit code (0 for success, 1 for error)
    """
    ap = argparse.ArgumentParser("ekilang")
    ap.add_argument("file", nargs="?")
    ap.add_argument("--dump-py", action="store_true")
    args = ap.parse_args(argv)

    if not args.file:
        print("Ekilang REPL. Ctrl+C to exit.")
        while True:
            try:
                line = input(">> ")
            except KeyboardInterrupt:
                print()
                break
            if not line.strip():
                continue
            try:
                run_source(line)
            except (SyntaxError, ValueError, TypeError) as e:
                print(f"Error: {e}")
        return 0

    with open(args.file, "r", encoding="utf-8") as f:
        src = f.read()
    try:
        run_source(src, dump_py=args.dump_py, current_file=args.file)
        return 0
    except (SyntaxError, ValueError, TypeError) as e:
        print("Error running Ekilang:", e)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
