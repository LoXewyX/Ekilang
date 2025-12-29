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


def test_default_parameters():
    """Test functions with default parameter values"""
    ns = run(
        """
fn greet(name, greeting = "Hello") {
    f"{greeting}, {name}!"
}

result1 = greet("Alice")
result2 = greet("Bob", "Hi")
        """
    )
    assert ns["result1"] == "Hello, Alice!"
    assert ns["result2"] == "Hi, Bob!"


def test_multiple_defaults():
    """Test multiple parameters with defaults"""
    ns = run(
        """
fn create_user(name, age = 18, country = "USA") {
    {"name": name, "age": age, "country": country}
}

user1 = create_user("Alice")
user2 = create_user("Bob", 25)
user3 = create_user("Charlie", 30, "UK")
        """
    )
    assert ns["user1"] == {"name": "Alice", "age": 18, "country": "USA"}
    assert ns["user2"] == {"name": "Bob", "age": 25, "country": "USA"}
    assert ns["user3"] == {"name": "Charlie", "age": 30, "country": "UK"}


def test_varargs():
    """Test *args variable positional arguments"""
    ns = run(
        """
fn sum_all(*numbers) {
    total = 0
    for n in numbers {
        total = total + n
    }
    total
}

result1 = sum_all(1, 2, 3)
result2 = sum_all(10, 20, 30, 40, 50)
        """
    )
    assert ns["result1"] == 6
    assert ns["result2"] == 150


def test_kwargs():
    """Test **kwargs variable keyword arguments"""
    ns = run(
        """
fn build_dict(**options) {
    options
}

result = build_dict(a = 1, b = 2, c = 3)
        """
    )
    assert ns["result"] == {"a": 1, "b": 2, "c": 3}


def test_params_with_varargs():
    """Test regular params combined with *args"""
    ns = run(
        """
fn make_list(first, second, *rest) {
    [first, second] + list(rest)
}

result1 = make_list(1, 2)
result2 = make_list(1, 2, 3, 4, 5)
        """
    )
    assert ns["result1"] == [1, 2]
    assert ns["result2"] == [1, 2, 3, 4, 5]


def test_defaults_with_varargs():
    """Test default params with *args"""
    ns = run(
        """
fn format_message(prefix = "INFO", *parts) {
    sep = ": "
    message = sep.join([str(p) for p in parts])
    prefix + sep + message
}

result1 = format_message("Error", "File", "not", "found")
result2 = format_message()
        """
    )
    assert ns["result1"] == "Error: File: not: found"
    assert ns["result2"] == "INFO: "


def test_params_defaults_varargs_kwargs():
    """Test all parameter types together"""
    ns = run(
        """
fn complex_fn(required, optional = "default", *args, **kwargs) {
    {
        "required": required,
        "optional": optional,
        "args": list(args),
        "kwargs": kwargs
    }
}

result1 = complex_fn("value")
result2 = complex_fn("value", "custom")
result3 = complex_fn("value", "custom", 1, 2, 3)
result4 = complex_fn("value", "custom", 1, 2, a = 10, b = 20)
        """
    )
    assert ns["result1"] == {
        "required": "value",
        "optional": "default",
        "args": [],
        "kwargs": {},
    }
    assert ns["result2"] == {
        "required": "value",
        "optional": "custom",
        "args": [],
        "kwargs": {},
    }
    assert ns["result3"] == {
        "required": "value",
        "optional": "custom",
        "args": [1, 2, 3],
        "kwargs": {},
    }
    assert ns["result4"] == {
        "required": "value",
        "optional": "custom",
        "args": [1, 2],
        "kwargs": {"a": 10, "b": 20},
    }


def test_typed_defaults():
    """Test default parameters with type annotations"""
    ns = run(
        """
fn add(a: int, b: int = 10) -> int {
    a + b
}

result1 = add(5)
result2 = add(5, 20)
        """
    )
    assert ns["result1"] == 15
    assert ns["result2"] == 25


def test_async_with_defaults():
    """Test async functions with defaults and varargs"""
    import asyncio

    ns = run(
        """
use asyncio

async fn async_sum(initial = 0, *nums) {
    total = initial
    for n in nums {
        total = total + n
    }
    total
}

result = asyncio.run(async_sum(10, 1, 2, 3, 4))
        """
    )
    assert ns["result"] == 20


def test_default_expressions():
    """Test that default values can be expressions"""
    ns = run(
        """
fn calculate(x, multiplier = 2 * 3) {
    x * multiplier
}

result = calculate(5)
        """
    )
    assert ns["result"] == 30


def test_kwargs_only():
    """Test function with only **kwargs"""
    ns = run(
        """
fn collect(**data) {
    len(data)
}

result = collect(a = 1, b = 2, c = 3, d = 4)
        """
    )
    assert ns["result"] == 4


def test_varargs_only():
    """Test function with only *args"""
    ns = run(
        """
fn multiply_all(*nums) {
    result = 1
    for n in nums {
        result = result * n
    }
    result
}

result = multiply_all(2, 3, 4, 5)
        """
    )
    assert ns["result"] == 120
