"""Ekilang Executor Module."""

import os
import gc
from collections import OrderedDict
from typing import Any, Dict
from .parser import Parser
from .runtime import compile_module
from .lexer import Lexer
from .types import Module, Statement, Use
from .builtins import BUILTINS

_CODE_CACHE_LIMIT = 32
CODE_CACHE: OrderedDict[str, tuple[float | None, Any]] = OrderedDict()


def _get_mtime(path: str) -> float | None:
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def _store_code(key: str, mtime: float | None, code: Any) -> None:
    if mtime is None:
        return
    CODE_CACHE[key] = (mtime, code)
    CODE_CACHE.move_to_end(key)
    if len(CODE_CACHE) > _CODE_CACHE_LIMIT:
        CODE_CACHE.popitem(last=False)


ModuleCacheEntry = tuple[float | None, Dict[str, Any], str]
MODULE_CACHE: Dict[str, ModuleCacheEntry] = {}


def _resolve_optimize(optimize: int | None) -> int:
    try:
        env_val = int(os.environ.get("EKILANG_OPTIMIZE", "0") or 0)
    except ValueError:
        env_val = 0
    level = optimize if optimize is not None else env_val
    return max(0, min(2, level))


def execute(
    mod: Module,
    globals_ns: Dict[str, Any] | None = None,
    current_file: str | None = None,
    code_obj: Any = None,
    optimize: int | None = None,
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
    opt_level = _resolve_optimize(optimize)
    code: Any | None = None
    cache_key: str | None = None
    mtime: float | None = None
    abs_file: str | None = os.path.abspath(current_file) if current_file else None

    if abs_file and os.path.isfile(abs_file):
        mtime = _get_mtime(abs_file)
        cache_key = f"{abs_file}:{opt_level}"
        cached = CODE_CACHE.get(cache_key)
        if cached and cached[0] == mtime:
            CODE_CACHE.move_to_end(cache_key)
            code = cached[1]
        elif cached:
            CODE_CACHE.pop(cache_key, None)

    if code is None:
        code = compile_module(
            mod_filtered,
            filename=abs_file or "<ekilang>",
            optimize=opt_level,
        )
        if cache_key:
            _store_code(cache_key, mtime, code)
    ns: Dict[str, Any] = {}
    ns.update(BUILTINS)
    ns.update(imports_ns)  # Add imported items
    if globals_ns:
        ns.update(globals_ns)
    # Set __file__ if current_file is provided
    if abs_file is not None:
        ns["__file__"] = abs_file
    # Only set __name__ to '__main__' if not already set (i.e., for main script)
    if "__name__" not in ns:
        ns["__name__"] = "__main__"

    exec(code, ns, ns)  # pylint: disable=exec-used
    gc.collect()
    return ns


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

    cached = MODULE_CACHE.get(module_path)
    if cached:
        cached_mtime, cached_ns, cached_path = cached
        current_mtime = _get_mtime(cached_path)
        if cached_mtime is None or current_mtime == cached_mtime:
            return cached_ns
        MODULE_CACHE.pop(module_path, None)

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

    mtime_store = _get_mtime(file_path)
    MODULE_CACHE[module_path] = (mtime_store, module_ns, file_path)

    return module_ns
