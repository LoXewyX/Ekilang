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


def test_loop_basic():
    """Test basic infinite loop with break"""
    code = """
    counter = 0
    loop {
        counter = counter + 1
        if counter >= 5 {
            break
        }
    }
    """
    ns = run(code)
    assert ns["counter"] == 5


def test_loop_with_continue():
    """Test loop with continue statement"""
    code = """
    counter = 0
    sum = 0
    loop {
        counter = counter + 1
        if counter % 2 == 0 {
            if counter >= 10 {
                break
            }
            continue
        }
        sum = sum + counter
    }
    """
    ns = run(code)
    assert ns["counter"] == 10
    assert ns["sum"] == 25  # 1 + 3 + 5 + 7 + 9


def test_loop_nested():
    """Test nested loop statements"""
    code = """
    i = 0
    result = 0
    loop {
        i = i + 1
        j = 0
        loop {
            j = j + 1
            result = result + 1
            if j >= 3 {
                break
            }
        }
        if i >= 2 {
            break
        }
    }
    """
    ns = run(code)
    assert ns["i"] == 2
    assert ns["result"] == 6  # 2 iterations * 3 inner iterations


def test_loop_with_complex_condition():
    """Test loop with more complex break logic"""
    code = """
    x = 0
    y = 0
    loop {
        x = x + 1
        if x > 5 {
            y = y + x
            if y > 20 {
                break
            }
        }
    }
    """
    ns = run(code)
    assert ns["x"] == 8
    assert ns["y"] == 21  # 6 + 7 + 8


def test_loop_accumulator():
    """Test loop as an accumulator pattern"""
    code = """
    numbers = [1, 2, 3, 4, 5]
    index = 0
    total = 0
    loop {
        if index >= len(numbers) {
            break
        }
        total = total + numbers[index]
        index = index + 1
    }
    """
    ns = run(code)
    assert ns["total"] == 15
    assert ns["index"] == 5
