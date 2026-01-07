"""Tests for unpacking functionality in Ekilang language."""

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


def test_unpack_tuple():
    """Test unpacking tuple into multiple variables"""
    ns = run(
        """
coords = (10, 20)
x, y = coords
        """
    )
    assert ns["x"] == 10
    assert ns["y"] == 20


def test_unpack_list():
    """Test unpacking list into multiple variables"""
    ns = run(
        """
data = [1, 2, 3]
a, b, c = data
        """
    )
    assert ns["a"] == 1
    assert ns["b"] == 2
    assert ns["c"] == 3


def test_unpack_function_return():
    """Test unpacking function return value"""
    ns = run(
        """
fn getCoords() {
    (100, 200)
}

x, y = getCoords()
        """
    )
    assert ns["x"] == 100
    assert ns["y"] == 200


def test_unpack_inline_tuple():
    """Test unpacking inline tuple literal"""
    ns = run(
        """
a, b = (5, 10)
        """
    )
    assert ns["a"] == 5
    assert ns["b"] == 10


def test_unpack_three_values():
    """Test unpacking three values"""
    ns = run(
        """
x, y, z = (1, 2, 3)
        """
    )
    assert ns["x"] == 1
    assert ns["y"] == 2
    assert ns["z"] == 3


def test_unpack_with_computation():
    """Test unpacking with computed values"""
    ns = run(
        """
fn calculate() {
    sum_val = 5 + 10
    product = 5 * 10
    result = (sum_val, product)
    result
}

s, p = calculate()
        """
    )
    assert ns["s"] == 15
    assert ns["p"] == 50


def test_unpack_nested():
    """Test using unpacked values in expressions"""
    ns = run(
        """
a, b = (3, 4)
sum_val = a + b
product = a * b
        """
    )
    assert ns["a"] == 3
    assert ns["b"] == 4
    assert ns["sum_val"] == 7
    assert ns["product"] == 12


def test_unpack_string():
    """Test unpacking string characters"""
    ns = run(
        """
first, second = "ab"
        """
    )
    assert ns["first"] == "a"
    assert ns["second"] == "b"


def test_unpack_range():
    """Test unpacking range values"""
    ns = run(
        """
a, b, c = [1, 2, 3]
        """
    )
    assert ns["a"] == 1
    assert ns["b"] == 2
    assert ns["c"] == 3


def test_unpack_in_function():
    """Test unpacking inside function"""
    ns = run(
        """
fn process() {
    data = (10, 20, 30)
    x, y, z = data
    x + y + z
}

result = process()
        """
    )
    assert ns["result"] == 60


def test_unpack_dict_items():
    """Test unpacking dict keys"""
    ns = run(
        """
d = {"a": 1, "b": 2}
keys_list = list(d.keys())
first_key, second_key = keys_list
        """
    )
    # Dict order is preserved in Python 3.7+
    assert ns["first_key"] == "a"
    assert ns["second_key"] == "b"


def test_multiple_unpacks():
    """Test multiple unpacking operations"""
    ns = run(
        """
a, b = (1, 2)
c, d = (3, 4)
sum_val = a + b + c + d
        """
    )
    assert ns["a"] == 1
    assert ns["b"] == 2
    assert ns["c"] == 3
    assert ns["d"] == 4
    assert ns["sum_val"] == 10


def test_unpack_swap():
    """Test swapping variables using unpacking"""
    ns = run(
        """
x = 10
y = 20
x, y = (y, x)
        """
    )
    assert ns["x"] == 20
    assert ns["y"] == 10
