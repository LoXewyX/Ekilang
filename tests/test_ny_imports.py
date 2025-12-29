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


def test_ny_import_single_function():
    """Test importing a single function from a .eki file"""
    code = """
    use examples::utils::math_utils { square }
    result = square(4)
    """
    ns = run(code)
    assert ns["result"] == 16


def test_ny_import_multiple_functions():
    """Test importing multiple functions"""
    code = """
    use examples::utils::math_utils { square, cube }
    s = square(3)
    c = cube(2)
    """
    ns = run(code)
    assert ns["s"] == 9
    assert ns["c"] == 8


def test_ny_import_class():
    """Test importing a class"""
    code = """
    use examples::utils::math_utils { MathHelper }
    helper = MathHelper()
    fact = helper.factorial(4)
    """
    ns = run(code)
    assert ns["fact"] == 24


def test_ny_import_with_alias():
    """Test importing with alias"""
    code = """
    use examples::utils::math_utils { square as sq }
    result = sq(5)
    """
    ns = run(code)
    assert ns["result"] == 25


def test_ny_import_multiple_modules():
    """Test importing from multiple modules"""
    code = """
    use examples::utils::math_utils { square }
    use examples::utils::string_utils { reverse_string }
    num = square(3)
    text = reverse_string("hello")
    """
    ns = run(code)
    assert ns["num"] == 9
    assert ns["text"] == "olleh"
