"""Tests for else clause functionality in Ekilang."""

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


def test_if_else_basic():
    """Test basic if-else statement"""
    ns = run(
        """
x = 0
if true {
    x = 1
} else {
    x = 2
}
"""
    )
    assert ns["x"] == 1

    ns = run(
        """
x = 0
if false {
    x = 1
} else {
    x = 2
}
"""
    )
    assert ns["x"] == 2


def test_if_elif_else():
    """Test if-elif-else chain"""
    ns = run(
        """
x = 0
if false {
    x = 1
} elif true {
    x = 2
} else {
    x = 3
}
"""
    )
    assert ns["x"] == 2

    ns = run(
        """
x = 0
if false {
    x = 1
} elif false {
    x = 2
} else {
    x = 3
}
"""
    )
    assert ns["x"] == 3


def test_nested_if_else():
    """Test nested if-else statements"""
    ns = run(
        """
x = 0
if true {
    if false {
        x = 1
    } else {
        x = 2
    }
} else {
    x = 3
}
"""
    )
    assert ns["x"] == 2


def test_else_with_comparison():
    """Test else with comparison operators"""
    ns = run(
        """
num = 5
result = ""
if num < 0 {
    result = "negative"
} elif num == 0 {
    result = "zero"
} else {
    result = "positive"
}
"""
    )
    assert ns["result"] == "positive"


def test_ternary_if_else():
    """Test ternary if-else expression"""
    ns = run(
        """
x = 10
result = "big" if x > 5 else "small"
"""
    )
    assert ns["result"] == "big"

    ns = run(
        """
x = 3
result = "big" if x > 5 else "small"
"""
    )
    assert ns["result"] == "small"


def test_ternary_nested():
    """Test nested ternary expressions"""
    ns = run(
        """
x = 15
result = "small" if x < 10 else "medium" if x < 20 else "large"
"""
    )
    assert ns["result"] == "medium"


def test_for_else():
    """Test for-else clause (else runs when loop completes normally)"""
    ns = run(
        """
result = []
for i in [1, 2, 3] {
    result.append(i)
} else {
    result.append("done")
}
"""
    )
    assert ns["result"] == [1, 2, 3, "done"]


def test_for_else_with_break():
    """Test for-else clause with break (else should not run)"""
    ns = run(
        """
result = []
for i in [1, 2, 3, 4, 5] {
    result.append(i)
    if i == 3 {
        break
    }
} else {
    result.append("done")
}
"""
    )
    assert ns["result"] == [1, 2, 3]


def test_while_else():
    """Test while-else clause (else runs when loop completes normally)"""
    ns = run(
        """
result = []
i = 0
while i < 3 {
    result.append(i)
    i += 1
} else {
    result.append("done")
}
"""
    )
    assert ns["result"] == [0, 1, 2, "done"]


def test_while_else_with_break():
    """Test while-else clause with break (else should not run)"""
    ns = run(
        """
result = []
i = 0
while i < 10 {
    result.append(i)
    i += 1
    if i == 3 {
        break
    }
} else {
    result.append("done")
}
"""
    )
    assert ns["result"] == [0, 1, 2]


def test_else_with_multiple_statements():
    """Test else block with multiple statements"""
    ns = run(
        """
x = 0
y = 0
if false {
    x = 1
} else {
    x = 2
    y = 3
}
"""
    )
    assert ns["x"] == 2
    assert ns["y"] == 3


def test_else_in_function():
    """Test else statement inside a function"""
    ns = run(
        """
fn check_sign(num) {
    if num > 0 {
        return "positive"
    } else {
        return "non-positive"
    }
}
result = check_sign(5)
"""
    )
    assert ns["result"] == "positive"


def test_else_with_logical_operators():
    """Test else with logical AND/OR operators"""
    ns = run(
        """
a = true
b = false
result = 0
if a and b {
    result = 1
} else {
    result = 2
}
"""
    )
    assert ns["result"] == 2

    ns = run(
        """
a = true
b = false
result = 0
if a or b {
    result = 1
} else {
    result = 2
}
"""
    )
    assert ns["result"] == 1


def test_else_empty_block():
    """Test else with empty block"""
    ns = run(
        """
x = 0
if true {
    x = 1
} else {
}
"""
    )
    assert ns["x"] == 1
