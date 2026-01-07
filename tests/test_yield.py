"""Tests for generator and yield functionality in Ekilang language."""

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


def test_simple_generator_sum():
    """Test simple generator that yields values and sums them"""
    ns = run(
        """
fn gen() {
  yield 1
  yield 2
  yield 3
}

sumv = 0
for v in gen() {
  sumv += v
}
"""
    )
    assert ns["sumv"] == 6


def test_generator_iteration_and_collect():
    ns = run(
        """
fn gen2(n) {
  i = 0
  while i < n {
    yield i
    i += 1
  }
}

vals = []
for v in gen2(4) {
  vals = vals + [v]
}
"""
    )
    assert ns["vals"] == [0, 1, 2, 3]
