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


def test_pow_assign():
    ns = run(
        """
x = 2
y = 3
x **= y
print(x)  # Should print 8
        """
    )
    assert ns["x"] == 8
