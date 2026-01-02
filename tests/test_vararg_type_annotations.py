"""Tests for vararg and kwarg type annotations in Ekilang language."""

from pathlib import Path
import sys
from ekilang.lexer import Lexer
from ekilang.parser import Parser
from ekilang.runtime import execute

sys.path.insert(0, str(Path(__file__).parent.parent))


def run(code: str):
    """Helper to run code snippets and return the namespace"""
    tokens = Lexer(code).tokenize()
    mod = Parser(tokens).parse()
    return execute(mod)


def test_vararg_type_annotation():
    """Test type annotations on *args"""
    ns = run("""
fn sum_typed(*numbers: int) -> int {
    total = 0
    for n in numbers {
        total = total + n
    }
    total
}
result = sum_typed(1, 2, 3, 4, 5)
""")
    assert ns["result"] == 15


def test_kwarg_type_annotation():
    """Test type annotations on **kwargs"""
    ns = run("""
fn config(**settings: str) {
    settings
}
result = config(host = "localhost", port = "8080")
""")
    assert ns["result"]["host"] == "localhost"
    assert ns["result"]["port"] == "8080"


def test_vararg_and_kwarg_type_annotations():
    """Test type annotations on both *args and **kwargs"""
    ns = run("""
fn process(name: str, *args: int, **kwargs: str) {
    list_args = len(args)
    list_kwargs = len(kwargs)
    (list_args, list_kwargs)
}
result = process("test", 1, 2, 3, a = "x", b = "y")
""")
    assert ns["result"] == (3, 2)


def test_vararg_type_with_defaults():
    """Test *args type annotation with default parameters"""
    ns = run("""
fn func(x: int, y: int = 10, *rest: int) -> int {
    total = x + y
    for n in rest {
        total = total + n
    }
    total
}
r1 = func(5)
r2 = func(5, 20)
r3 = func(5, 20, 1, 2, 3)
""")
    assert ns["r1"] == 15
    assert ns["r2"] == 25
    assert ns["r3"] == 31


if __name__ == "__main__":
    test_vararg_type_annotation()
    print("✓ test_vararg_type_annotation passed")
    
    test_kwarg_type_annotation()
    print("✓ test_kwarg_type_annotation passed")
    
    test_vararg_and_kwarg_type_annotations()
    print("✓ test_vararg_and_kwarg_type_annotations passed")
    
    test_vararg_type_with_defaults()
    print("✓ test_vararg_type_with_defaults passed")
    
    print("\nAll type annotation tests passed!")
