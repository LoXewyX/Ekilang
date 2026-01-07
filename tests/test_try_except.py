"""Tests for try/except/finally exception handling in Ekilang."""

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


def test_basic_exception_handling():
    """Test basic try/except"""
    ns = run(
        """
caught = false
try {
    result = 10 / 0
} except ZeroDivisionError {
    caught = true
}
"""
    )
    assert ns["caught"] is True


def test_exception_with_as_clause():
    """Test exception with 'as' clause to capture exception"""
    ns = run(
        """
error_msg = ""
try {
    result = 10 / 0
} except ZeroDivisionError as e {
    error_msg = str(e)
}
"""
    )
    assert "division by zero" in ns["error_msg"]


def test_multiple_exception_handlers():
    """Test multiple except clauses"""
    ns = run(
        """
caught_type = ""
try {
    value = int("not a number")
} except ValueError {
    caught_type = "ValueError"
} except TypeError {
    caught_type = "TypeError"
}
"""
    )
    assert ns["caught_type"] == "ValueError"


def test_try_except_else():
    """Test try/except with else clause"""
    ns = run(
        """
result = 0
else_ran = false
try {
    result = 10 / 2
} except ZeroDivisionError {
    result = -1
} else {
    else_ran = true
}
"""
    )
    assert ns["result"] == 5.0
    assert ns["else_ran"] is True


def test_try_except_finally():
    """Test try/except with finally clause"""
    ns = run(
        """
cleanup_done = false
try {
    x = 5
} except Exception {
    x = -1
} finally {
    cleanup_done = true
}
"""
    )
    assert ns["cleanup_done"] is True


def test_try_except_else_finally():
    """Test complete try/except/else/finally structure"""
    ns = run(
        """
result = 0
else_ran = false
finally_ran = false
try {
    result = 10 / 2
} except ZeroDivisionError {
    result = -1
} else {
    else_ran = true
} finally {
    finally_ran = true
}
"""
    )
    assert ns["result"] == 5.0
    assert ns["else_ran"] is True
    assert ns["finally_ran"] is True


def test_nested_try_except():
    """Test nested try/except blocks"""
    ns = run(
        """
outer_caught = false
inner_caught = false
try {
    try {
        x = 1 / 0
    } except ZeroDivisionError {
        inner_caught = true
    }
} except Exception {
    outer_caught = true
}
"""
    )
    assert ns["inner_caught"] is True
    assert ns["outer_caught"] is False


def test_exception_with_list_operations():
    """Test exception handling with list indexing"""
    ns = run(
        """
caught = false
try {
    my_list = [1, 2, 3]
    item = my_list[10]
} except IndexError {
    caught = true
}
"""
    )
    assert ns["caught"] is True


def test_exception_with_dict_operations():
    """Test exception handling with dictionary access"""
    ns = run(
        """
caught = false
try {
    my_dict = {"a": 1}
    value = my_dict["missing"]
} except KeyError {
    caught = true
}
"""
    )
    assert ns["caught"] is True


def test_exception_with_type_error():
    """Test TypeError exception handling"""
    ns = run(
        """
caught = false
try {
    result = "string" + 5
} except TypeError {
    caught = true
}
"""
    )
    assert ns["caught"] is True


def test_exception_in_function():
    """Test exception handling with function calls"""
    ns = run(
        """
caught = false
fn risky() {
    return 1 / 0
}
try {
    risky()
} except ZeroDivisionError {
    caught = true
}
"""
    )
    assert ns["caught"] is True


def test_try_finally_without_except():
    """Test try/finally without except clause"""
    ns = run(
        """
finally_ran = false
result = 0
try {
    result = 5 + 5
} finally {
    finally_ran = true
}
"""
    )
    assert ns["result"] == 10
    assert ns["finally_ran"] is True


def test_exception_name_error():
    """Test NameError exception handling"""
    ns = run(
        """
caught = false
try {
    x = undefined_variable
} except NameError {
    caught = true
}
"""
    )
    assert ns["caught"] is True


def test_exception_else_not_executed():
    """Test that else clause doesn't execute when exception is raised"""
    ns = run(
        """
else_ran = false
except_ran = false
try {
    x = 1 / 0
} except ZeroDivisionError {
    except_ran = true
} else {
    else_ran = true
}
"""
    )
    assert ns["except_ran"] is True
    assert ns["else_ran"] is False


def test_finally_always_executes():
    """Test that finally executes even with exception"""
    ns = run(
        """
finally_ran = false
caught = false
try {
    x = 1 / 0
} except ZeroDivisionError {
    caught = true
} finally {
    finally_ran = true
}
"""
    )
    assert ns["caught"] is True
    assert ns["finally_ran"] is True
