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

def test_null_safe_attr_none():
    code = """
    x = none
    result = x?.y
    """
    ns = run(code)
    assert ns["result"] is None

def test_null_safe_attr_value():
    # Create a simple object with attribute for testing
    class Obj:
        def __init__(self):
            self.y = 42
    
    code = """
    result = x?.y
    """
    from ekilang.lexer import Lexer
    from ekilang.parser import Parser
    from ekilang.runtime import execute
    
    tokens = Lexer(code).tokenize()
    mod = Parser(tokens).parse()
    ns = execute(mod, {"x": Obj()})
    assert ns["result"] == 42

def test_null_coalesce_left_none():
    code = """
    a = none
    b = 5
    result = a ?? b
    """
    ns = run(code)
    assert ns["result"] == 5

def test_null_coalesce_left_value():
    code = """
    a = 3
    b = 5
    result = a ?? b
    """
    ns = run(code)
    assert ns["result"] == 3
