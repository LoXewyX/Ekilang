"""Tests for splat (star) arguments in calls"""

import sys
from pathlib import Path
from ekilang.lexer import Lexer
from ekilang.parser import Parser
from ekilang.executor import execute

sys.path.insert(0, str(Path(__file__).parent.parent))



def run(src: str):
    """Helper to run code snippets and return the namespace"""
    tokens = Lexer(src).tokenize()
    mod = Parser(tokens).parse()
    return execute(mod)


def test_print_splat_string():
    """Test splat operator with string"""
    ns = run(
        """
s = "hello"
print(*s)
        """
    )
    # Python's print will output with spaces; we can't capture stdout easily here
    # Instead ensure no error and Starred expands correctly by building a list
    ns = run(
        """
s = "hello"
fn collect(*args) { args }
out = collect(*s)
        """
    )
    assert ns["out"] == ("h", "e", "l", "l", "o")


def test_splat_list():
    """Test splat operator with list"""
    ns = run(
        """
lst = [1, 2, 3]
fn add3(a, b, c) { a + b + c }
total = add3(*lst)
        """
    )
    assert ns["total"] == 6


def test_mixed_positional_and_splat():
    """Test mixing positional and splat arguments"""
    ns = run(
        """
lst = [2, 3]
fn mul(a, b, c) { a * b * c }
result = mul(5, *lst)
        """
    )
    assert ns["result"] == 30


def test_multiple_splats():
    """Test multiple splat arguments in a single call"""
    ns = run(
        """
a = [1]
b = [2]
c = [3]
fn add3(x, y, z) { x + y + z }
total = add3(*a, *b, *c)
        """
    )
    assert ns["total"] == 6


def test_splat_tuple():
    """Test splat operator with tuple"""
    ns = run(
        """
t = (4, 5)
fn add(a, b) { a + b }
res = add(*t)
        """
    )
    assert ns["res"] == 9
