"""Tests for keyword-only parameters (after bare * separator) in Ekilang."""

from pathlib import Path
import sys
from ekilang.lexer import Lexer
from ekilang.parser import Parser
from ekilang.executor import execute

sys.path.insert(0, str(Path(__file__).parent.parent))


def run(code: str):
    """Helper to run code snippets and return the namespace"""
    tokens = Lexer(code).tokenize()
    mod = Parser(tokens).parse()
    return execute(mod)


def test_kwonly_with_defaults():
    """Test keyword-only parameters with defaults after bare *"""
    ns = run(
        """
fn test_kwonly(a, b, *, c = 3, d = 10) {
    a + b + c + d
}

result1 = test_kwonly(1, 2)
result2 = test_kwonly(1, 2, c = 5)
result3 = test_kwonly(1, 2, d = 20)
result4 = test_kwonly(1, 2, c = 5, d = 20)
        """
    )
    assert ns["result1"] == 16  # 1 + 2 + 3 + 10
    assert ns["result2"] == 18  # 1 + 2 + 5 + 10
    assert ns["result3"] == 26  # 1 + 2 + 3 + 20
    assert ns["result4"] == 28  # 1 + 2 + 5 + 20


def test_kwonly_required():
    """Test keyword-only parameters without defaults (required)"""
    ns = run(
        """
fn test_kwonly(a, b, *, c, d = 10) {
    a + b + c + d
}

result = test_kwonly(1, 2, c = 5)
        """
    )
    assert ns["result"] == 18  # 1 + 2 + 5 + 10


def test_kwonly_after_varargs():
    """Test keyword-only parameters after *args"""
    ns = run(
        """
fn test_kwonly(a, *args, b, c = 10) {
    total = a
    for arg in args {
        total = total + arg
    }
    total + b + c
}

result1 = test_kwonly(1, 2, 3, b = 5)
result2 = test_kwonly(1, b = 5, c = 20)
        """
    )
    assert ns["result1"] == 21  # 1 + 2 + 3 + 5 + 10
    assert ns["result2"] == 26  # 1 + 5 + 20


def test_kwonly_with_types():
    """Test keyword-only parameters with type annotations"""
    ns = run(
        """
fn typed_kwonly(x: int, *, y: int = 10, z: str = "hello") -> str {
    f"{x} {y} {z}"
}

result1 = typed_kwonly(5)
result2 = typed_kwonly(5, y = 20)
result3 = typed_kwonly(5, z = "world")
        """
    )
    assert ns["result1"] == "5 10 hello"
    assert ns["result2"] == "5 20 hello"
    assert ns["result3"] == "5 10 world"


def test_kwonly_with_kwargs():
    """Test keyword-only parameters with **kwargs"""
    ns = run(
        """
fn test_kwonly(a, *, b = 10, **kwargs) {
    {
        "a": a,
        "b": b,
        "kwargs": kwargs
    }
}

result1 = test_kwonly(1)
result2 = test_kwonly(1, b = 5)
result3 = test_kwonly(1, b = 5, x = 10, y = 20)
        """
    )
    assert ns["result1"] == {"a": 1, "b": 10, "kwargs": {}}
    assert ns["result2"] == {"a": 1, "b": 5, "kwargs": {}}
    assert ns["result3"] == {"a": 1, "b": 5, "kwargs": {"x": 10, "y": 20}}


def test_bare_star_only():
    """Test bare * with no varargs"""
    ns = run(
        """
fn bare_star(a, b, *, c) {
    a + b + c
}

result = bare_star(1, 2, c = 3)
        """
    )
    assert ns["result"] == 6


def test_complex_signature():
    """Test complex function signature with all parameter types"""
    ns = run(
        """
fn complex_fn(pos1, pos2 = 10, *args, kw1, kw2 = 20, **kwargs) {
    {
        "pos1": pos1,
        "pos2": pos2,
        "args": list(args),
        "kw1": kw1,
        "kw2": kw2,
        "kwargs": kwargs
    }
}

result1 = complex_fn(1, kw1 = 100)
result2 = complex_fn(1, 2, 3, 4, kw1 = 100, kw2 = 200)
result3 = complex_fn(1, 2, 3, 4, kw1 = 100, kw2 = 200, extra = 999)
        """
    )
    assert ns["result1"] == {
        "pos1": 1,
        "pos2": 10,
        "args": [],
        "kw1": 100,
        "kw2": 20,
        "kwargs": {},
    }
    assert ns["result2"] == {
        "pos1": 1,
        "pos2": 2,
        "args": [3, 4],
        "kw1": 100,
        "kw2": 200,
        "kwargs": {},
    }
    assert ns["result3"] == {
        "pos1": 1,
        "pos2": 2,
        "args": [3, 4],
        "kw1": 100,
        "kw2": 200,
        "kwargs": {"extra": 999},
    }


def test_async_kwonly():
    """Test keyword-only parameters in async functions"""
    ns = run(
        """
use asyncio

async fn async_kwonly(a, *, b = 10) {
    a + b
}

result = asyncio.run(async_kwonly(5, b = 20))
        """
    )
    assert ns["result"] == 25


def test_kwonly_position_enforcement():
    """Test that keyword-only args cannot be passed positionally"""
    try:
        run(
            """
fn test_kwonly(a, *, b = 10) {
    a + b
}

result = test_kwonly(1, 5)  # Should fail - b must be keyword
            """
        )
        assert False, "Should have raised TypeError"
    except TypeError:
        pass  # Expected


def test_defaults_with_positional():
    """Test keyword-only with positional defaults"""
    ns = run(
        """
fn mixed_defaults(a, b = 5, *, c = 10, d = 20) {
    a + b + c + d
}

result1 = mixed_defaults(1)
result2 = mixed_defaults(1, 2)
result3 = mixed_defaults(1, c = 15)
result4 = mixed_defaults(1, 2, c = 15, d = 25)
        """
    )
    assert ns["result1"] == 36  # 1 + 5 + 10 + 20
    assert ns["result2"] == 33  # 1 + 2 + 10 + 20
    assert ns["result3"] == 41  # 1 + 5 + 15 + 20
    assert ns["result4"] == 43  # 1 + 2 + 15 + 25
