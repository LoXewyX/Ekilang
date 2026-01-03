"""Ekilang Executor Module."""

import os
import gc
from typing import Any, Dict
from .parser import Parser
from .runtime import compile_module
from .lexer import Lexer
from .types import Module, Statement, Use
from .builtins import BUILTINS


def execute(
    mod: Module,
    globals_ns: Dict[str, Any] | None = None,
    current_file: str | None = None,
    code_obj: Any = None,
) -> Dict[str, Any]:
    """Execute Ekilang module.

    Args:
        mod: Parsed Ekilang Module AST
        globals_ns: Optional global namespace
        current_file: Current file path for relative imports
        code_obj: Pre-compiled code object (skips parsing)

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
        exec(code_obj, ns)  # pylint: disable=exec-used
        return ns

    # Process Use statements first to load .eki modules
    imports_ns: Dict[str, Any] = {}
    filtered_body: list[Statement] = []

    def handle_use_import(stmt: Use) -> bool:
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

        ns["__file__"] = os.path.abspath(current_file)
    # Only set __name__ to '__main__' if not already set (i.e., for main script)
    if "__name__" not in ns:
        ns["__name__"] = "__main__"

    exec(code, ns, ns)  # pylint: disable=exec-used
    gc.collect()
    return ns


# Module cache for imported .eki files (can be disabled for low-memory environments)
MODULE_CACHE: Dict[str, Dict[str, Any]] = {}


def load_ekilang_module(
    module_path: str,
    search_paths: list[str] | None = None,
    current_file: str | None = None,
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
            search_paths = [os.getcwd()]

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
