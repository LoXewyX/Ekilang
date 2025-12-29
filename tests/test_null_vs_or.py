from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from ekilang.lexer import Lexer
from ekilang.parser import Parser
from ekilang.runtime import execute


def run(code: str):
    tokens = Lexer(code).tokenize()
    mod = Parser(tokens).parse()
    return execute(mod)


def test_null_coalesce_vs_or_with_zero():
    """?? only checks for None, or checks for truthiness"""
    code = """
    zero = 0
    with_or = zero or 5
    with_null_coalesce = zero ?? 5
    """
    ns = run(code)
    assert ns["with_or"] == 5  # 0 is falsy, so or returns 5
    assert ns["with_null_coalesce"] == 0  # 0 is not None, so ?? returns 0


def test_null_coalesce_vs_or_with_empty_string():
    """Empty string is falsy but not None"""
    code = """
    empty = ""
    with_or = empty or "default"
    with_null_coalesce = empty ?? "default"
    """
    ns = run(code)
    assert ns["with_or"] == "default"  # "" is falsy
    assert ns["with_null_coalesce"] == ""  # "" is not None


def test_null_coalesce_vs_or_with_false():
    """False is falsy but not None"""
    code = """
    flag = false
    with_or = flag or true
    with_null_coalesce = flag ?? true
    """
    ns = run(code)
    assert ns["with_or"] is True  # false is falsy
    assert ns["with_null_coalesce"] is False  # false is not None


def test_null_coalesce_vs_or_with_empty_list():
    """Empty list is falsy but not None"""
    code = """
    empty = []
    with_or = empty or [1, 2, 3]
    with_null_coalesce = empty ?? [1, 2, 3]
    """
    ns = run(code)
    assert ns["with_or"] == [1, 2, 3]  # [] is falsy
    assert ns["with_null_coalesce"] == []  # [] is not None


def test_null_coalesce_vs_or_with_none():
    """Both behave the same when left is None"""
    code = """
    val = none
    with_or = val or "default"
    with_null_coalesce = val ?? "default"
    """
    ns = run(code)
    assert ns["with_or"] == "default"
    assert ns["with_null_coalesce"] == "default"
