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


def test_tstring_basic():
    code = """
    name = "Alice"
    greeting = t"Hello, {name}!"
    """
    ns = run(code)
    assert ns["greeting"] == "Hello, Alice!"


def test_tstring_multiple_expressions():
    code = """
    x = 5
    y = 10
    result = t"{x} + {y} = {x + y}"
    """
    ns = run(code)
    assert ns["result"] == "5 + 10 = 15"


def test_tstring_no_interpolation():
    code = """
    plain = t"Just a plain string"
    """
    ns = run(code)
    assert ns["plain"] == "Just a plain string"


def test_tstring_with_numbers():
    code = """
    num = 42
    msg = t"The answer is {num}"
    """
    ns = run(code)
    assert ns["msg"] == "The answer is 42"


def test_tstring_complex_expression():
    code = """
    a = 3
    b = 4
    hyp = t"Hypotenuse: {(a * a + b * b) as float}"
    """
    ns = run(code)
    assert ns["hyp"] == "Hypotenuse: 25.0"


def test_tstring_with_function_call():
    code = """
    data = [1, 2, 3]
    msg = t"Length: {len(data)}"
    """
    ns = run(code)
    assert ns["msg"] == "Length: 3"


def test_tstring_nested():
    code = """
    x = 5
    inner = t"value is {x}"
    outer = t"Message: {inner}"
    """
    ns = run(code)
    assert ns["outer"] == "Message: value is 5"


def test_tstring_single_quotes():
    code = """
    name = "Bob"
    msg = t'Hello, {name}!'
    """
    ns = run(code)
    assert ns["msg"] == "Hello, Bob!"


def test_tstring_empty():
    code = """
    empty = t""
    """
    ns = run(code)
    assert ns["empty"] == ""


def test_tstring_with_boolean():
    code = """
    flag = true
    msg = t"Flag is {flag}"
    """
    ns = run(code)
    assert ns["msg"] == "Flag is True"
