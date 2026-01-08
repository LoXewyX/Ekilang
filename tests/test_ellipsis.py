"""Tests for ellipsis literal (...)."""

from contextlib import redirect_stdout
from io import StringIO

from ekilang.cli import run_source


def run(code: str) -> str:
    buff = StringIO()
    with redirect_stdout(buff):
        run_source(code)
    return buff.getvalue().strip()


def test_ellipsis_basic():
    """Test basic ellipsis literal."""
    code = "print(...)"
    assert run(code) == "Ellipsis"


def test_ellipsis_assignment():
    """Test assigning ellipsis to a variable."""
    code = """
x = ...
print(x)
print(type(x).__name__)
"""
    output = run(code)
    assert "Ellipsis" in output
    assert "ellipsis" in output


def test_ellipsis_in_list():
    """Test ellipsis in a list."""
    code = """
items = [1, 2, ..., 3, 4]
print(items)
"""
    output = run(code)
    assert "1" in output
    assert "Ellipsis" in output
    assert "4" in output


def test_ellipsis_in_dict():
    """Test ellipsis as dict key or value."""
    code = """
d = {...: "placeholder", "key": ...}
print(d[...])
print(d["key"])
"""
    output = run(code)
    lines = output.strip().split("\n")
    assert lines[0] == "placeholder"
    assert lines[1] == "Ellipsis"


def test_ellipsis_comparison():
    """Test ellipsis identity comparison."""
    code = """
x = ...
y = ...
print(x is y)
print(x == y)
"""
    output = run(code)
    lines = output.strip().split("\n")
    assert lines[0] == "True"
    assert lines[1] == "True"


def test_ellipsis_in_tuple():
    """Test ellipsis in tuple."""
    code = """
t = (1, ..., 2)
print(t)
print(len(t))
"""
    output = run(code)
    lines = output.strip().split("\n")
    assert "Ellipsis" in lines[0]
    assert lines[1] == "3"


def test_ellipsis_as_placeholder():
    """Test ellipsis as a placeholder in function stub."""
    code = """
fn placeholder() {
    ...
}

print("Function defined with ellipsis")
"""
    output = run(code)
    assert output == "Function defined with ellipsis"


def test_ellipsis_multiple():
    """Test multiple ellipsis literals in expression."""
    code = """
print([..., ..., ...])
"""
    output = run(code)
    assert output.count("Ellipsis") == 3
