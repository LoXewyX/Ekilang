"""Ekilang built-in functions and objects.

Provides all Python builtins to Ekilang runtime environment.
"""

from __future__ import annotations
import builtins as _py_builtins

# Export all Python builtins dynamically (without hardcoding).
# Filter out private names and modules.
BUILTINS = {
    k: getattr(_py_builtins, k) for k in dir(_py_builtins) if not k.startswith("_")
}
