"""Tests for string slicing with substring and step support."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ekilang.lexer import Lexer
from ekilang.parser import Parser
from ekilang.runtime import execute


def run(source: str):
    tokens = Lexer(source).tokenize()
    mod = Parser(tokens).parse()
    return execute(mod)


def test_string_basic_index():
    """Test basic string indexing"""
    ns = run("""
s = "Hello"
first = s[0]
last = s[4]
    """)
    assert ns["first"] == "H"
    assert ns["last"] == "o"


def test_string_negative_index():
    """Test negative indexing"""
    ns = run("""
s = "Hello"
last = s[-1]
second_last = s[-2]
    """)
    assert ns["last"] == "o"
    assert ns["second_last"] == "l"


def test_string_slice_basic():
    """Test basic string slicing"""
    ns = run("""
s = "Hello World"
sub = s[0:5]
    """)
    assert ns["sub"] == "Hello"


def test_string_slice_start_only():
    """Test slice with only start"""
    ns = run("""
s = "Hello World"
sub = s[6:]
    """)
    assert ns["sub"] == "World"


def test_string_slice_stop_only():
    """Test slice with only stop"""
    ns = run("""
s = "Hello World"
sub = s[:5]
    """)
    assert ns["sub"] == "Hello"


def test_string_slice_negative():
    """Test slice with negative indices"""
    ns = run("""
s = "Hello World"
sub = s[-5:]
    """)
    assert ns["sub"] == "World"


def test_string_slice_with_step():
    """Test slice with step parameter"""
    ns = run("""
s = "Hello World"
every_other = s[0:11:2]
    """)
    assert ns["every_other"] == "HloWrd"


def test_string_slice_step_only():
    """Test slice with only step (every nth character)"""
    ns = run("""
s = "0123456789"
every_third = s[::3]
    """)
    assert ns["every_third"] == "0369"


def test_string_slice_reverse():
    """Test string reversal with negative step"""
    ns = run("""
s = "Hello"
reversed = s[::-1]
    """)
    assert ns["reversed"] == "olleH"


def test_string_slice_complex():
    """Test complex slicing patterns"""
    ns = run("""
s = "0123456789"
sub1 = s[2:8:2]
sub2 = s[1::2]
sub3 = s[::-2]
    """)
    assert ns["sub1"] == "246"
    assert ns["sub2"] == "13579"
    assert ns["sub3"] == "97531"


def test_slice_unpacking():
    """Test unpacking from slice results"""
    ns = run("""
s = "Hello"
a, b = s[0:2]
    """)
    assert ns["a"] == "H"
    assert ns["b"] == "e"


def test_slice_unpacking_three():
    """Test unpacking three values from slice"""
    ns = run("""
s = "Hello World"
x, y, z = s[0:5:2]
    """)
    assert ns["x"] == "H"
    assert ns["y"] == "l"
    assert ns["z"] == "o"


def test_slice_add():
    """Test adding slices"""
    ns = run("""
s = "Hello World"
result = s[0:5] + " " + s[6:11]
    """)
    assert ns["result"] == "Hello World"


def test_slice_multiply():
    """Test multiplying slices"""
    ns = run("""
s = "Hello"
result = s[0:2] * 3
    """)
    assert ns["result"] == "HeHeHe"


def test_slice_in_function():
    """Test slicing inside function"""
    ns = run("""
fn getFirst(s, n) {
    s[0:n]
}
result = getFirst("Hello World", 5)
    """)
    assert ns["result"] == "Hello"


def test_slice_with_variables():
    """Test slicing with variable indices"""
    ns = run("""
s = "Hello World"
start = 6
end = 11
result = s[start:end]
    """)
    assert ns["result"] == "World"


def test_list_slice():
    """Test list slicing"""
    ns = run("""
lst = [1, 2, 3, 4, 5]
sub = lst[1:4]
    """)
    assert ns["sub"] == [2, 3, 4]


def test_list_slice_step():
    """Test list slicing with step"""
    ns = run("""
lst = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
every_other = lst[::2]
    """)
    assert ns["every_other"] == [0, 2, 4, 6, 8]


def test_list_slice_reverse():
    """Test list reversal"""
    ns = run("""
lst = [1, 2, 3, 4, 5]
reversed = lst[::-1]
    """)
    assert ns["reversed"] == [5, 4, 3, 2, 1]


def test_range_slice():
    """Test slicing on range objects"""
    ns = run("""
r = 0..10
sub = list(r)[2:7]
    """)
    assert ns["sub"] == [2, 3, 4, 5, 6]


def test_slice_chaining():
    """Test chained slicing operations"""
    ns = run("""
s = "Hello World"
result = s[0:5][1:4]
    """)
    assert ns["result"] == "ell"


def test_tuple_slice():
    """Test tuple slicing"""
    ns = run("""
t = (1, 2, 3, 4, 5)
sub = t[1:4]
    """)
    assert ns["sub"] == (2, 3, 4)
