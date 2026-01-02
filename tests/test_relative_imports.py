"""Test relative imports (.:: and ..::)"""

import os
from ekilang.lexer import Lexer
from ekilang.parser import Parser
from ekilang.runtime import execute, MODULE_CACHE


def test_relative_import_current_dir():
    """Test importing from current directory with .::"""
    MODULE_CACHE.clear()

    src = """use .::utils::helpers { greet, format_number }

result1 = greet("Alice")
result2 = format_number(42)
"""

    tokens = Lexer(src).tokenize()
    mod = Parser(tokens).parse()
    current_file = os.path.join(os.getcwd(), "examples", "helpers.eki")
    ns = execute(mod, current_file=current_file)

    assert ns["result1"] == "Hello, Alice!"
    assert ns["result2"] == "Number: 42"


def test_relative_import_subdirectory():
    """Test importing from subdirectory with .::subdir::module"""
    MODULE_CACHE.clear()

    src = """use .::utils::math_utils { double, triple, square }

x = double(5)
y = triple(5)
z = square(5)
"""

    tokens = Lexer(src).tokenize()
    mod = Parser(tokens).parse()
    current_file = os.path.join(os.getcwd(), "examples", "test.eki")
    ns = execute(mod, current_file=current_file)

    assert ns["x"] == 10
    assert ns["y"] == 15
    assert ns["z"] == 25


def test_relative_import_parent_directory():
    """Test importing from parent directory with ..::"""
    MODULE_CACHE.clear()

    src = """use ..::parent_utils { get_message, add }

msg = get_message()
result = add(10, 20)
"""

    tokens = Lexer(src).tokenize()
    mod = Parser(tokens).parse()
    current_file = os.path.join(
        os.getcwd(), "examples", "subdir", "nested", "test_parent_import.eki"
    )
    ns = execute(mod, current_file=current_file)

    assert ns["msg"] == "Hello from parent directory!"
    assert ns["result"] == 30


def test_relative_import_with_alias():
    """Test relative imports with aliases"""
    MODULE_CACHE.clear()

    src = """use .::utils::helpers { greet as say_hello, format_number as fmt_num }

result1 = say_hello("Bob")
result2 = fmt_num(123)
"""

    tokens = Lexer(src).tokenize()
    mod = Parser(tokens).parse()
    current_file = os.path.join(os.getcwd(), "examples", "test.eki")
    ns = execute(mod, current_file=current_file)

    assert ns["result1"] == "Hello, Bob!"
    assert ns["result2"] == "Number: 123"


def test_mixed_absolute_and_relative_imports():
    """Test using both absolute and relative imports in the same file"""
    MODULE_CACHE.clear()

    src = """use examples::utils::math_utils { square as abs_square }
use .::utils::helpers { greet }

x = abs_square(4)
msg = greet("Charlie")
"""

    tokens = Lexer(src).tokenize()
    mod = Parser(tokens).parse()
    current_file = os.path.join(os.getcwd(), "examples", "test.eki")
    ns = execute(mod, current_file=current_file)

    assert ns["x"] == 16
    assert ns["msg"] == "Hello, Charlie!"
