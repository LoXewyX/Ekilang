from pathlib import Path
import sys
from ekilang.lexer import Lexer
from ekilang.parser import Parser
from ekilang.runtime import execute

sys.path.insert(0, str(Path(__file__).parent.parent))



def run(code: str):
    tokens = Lexer(code).tokenize()
    mod = Parser(tokens).parse()
    return execute(mod)


def test_set_comp_basic():
    code = """
    nums = [1, 2, 3, 4, 5]
    squares = {x * x for x in nums}
    """
    ns = run(code)
    assert ns["squares"] == {1, 4, 9, 16, 25}


def test_set_comp_with_condition():
    code = """
    nums = [1, 2, 3, 4, 5, 6]
    even_squares = {x * x for x in nums if x % 2 == 0}
    """
    ns = run(code)
    assert ns["even_squares"] == {4, 16, 36}


def test_set_comp_from_range():
    code = """
    result = {x for x in 0..5 if x > 2}
    """
    ns = run(code)
    assert ns["result"] == {3, 4}


def test_dict_comp_basic():
    code = """
    nums = [1, 2, 3]
    squares = {x: x * x for x in nums}
    """
    ns = run(code)
    assert ns["squares"] == {1: 1, 2: 4, 3: 9}


def test_dict_comp_with_condition():
    code = """
    nums = [1, 2, 3, 4, 5]
    even_squares = {x: x * x for x in nums if x % 2 == 0}
    """
    ns = run(code)
    assert ns["even_squares"] == {2: 4, 4: 16}


def test_dict_comp_string_keys():
    code = """
    words = ["hello", "world"]
    lengths = {w: len(w) for w in words}
    """
    ns = run(code)
    assert ns["lengths"] == {"hello": 5, "world": 5}


def test_dict_comp_from_range():
    code = """
    result = {x: x * 2 for x in 1..=3}
    """
    ns = run(code)
    assert ns["result"] == {1: 2, 2: 4, 3: 6}


def test_set_comp_deduplication():
    code = """
    nums = [1, 2, 2, 3, 3, 3]
    unique = {x for x in nums}
    """
    ns = run(code)
    assert ns["unique"] == {1, 2, 3}


def test_dict_comp_transform():
    code = """
    data = [1, 2, 3]
    result = {x: str(x) for x in data}
    """
    ns = run(code)
    assert ns["result"] == {1: "1", 2: "2", 3: "3"}


def test_set_comp_nested_expr():
    code = """
    nums = [1, 2, 3]
    result = {x * 2 + 1 for x in nums}
    """
    ns = run(code)
    assert ns["result"] == {3, 5, 7}


def test_dict_comp_complex_condition():
    code = """
    nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    result = {x: x * x for x in nums if x % 3 == 0 or x == 1}
    """
    ns = run(code)
    assert ns["result"] == {1: 1, 3: 9, 6: 36, 9: 81}
