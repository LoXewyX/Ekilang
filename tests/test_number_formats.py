"""Tests for number format support in Ekilang."""

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


def test_hexadecimal_numbers():
    """Test hexadecimal number parsing"""
    ns = run("""
hex1 = 0x10
hex2 = 0xFF
hex3 = 0xDEADBEEF
hex4 = 0X1A
""")
    assert ns["hex1"] == 16
    assert ns["hex2"] == 255
    assert ns["hex3"] == 3735928559
    assert ns["hex4"] == 26


def test_binary_numbers():
    """Test binary number parsing"""
    ns = run("""
bin1 = 0b1010
bin2 = 0b11111111
bin3 = 0B1100
""")
    assert ns["bin1"] == 10
    assert ns["bin2"] == 255
    assert ns["bin3"] == 12


def test_octal_numbers():
    """Test octal number parsing"""
    ns = run("""
oct1 = 0o10
oct2 = 0o77
oct3 = 0O755
""")
    assert ns["oct1"] == 8
    assert ns["oct2"] == 63
    assert ns["oct3"] == 493


def test_scientific_notation():
    """Test scientific notation parsing"""
    ns = run("""
sci1 = 1e3
sci2 = 1.5e2
sci3 = 2e-3
sci4 = 1E6
sci5 = 3.14e+10
sci6 = 5E-8
""")
    assert ns["sci1"] == 1000.0
    assert ns["sci2"] == 150.0
    assert abs(ns["sci3"] - 0.002) < 1e-10
    assert ns["sci4"] == 1000000.0
    assert ns["sci5"] == 3.14e10
    assert ns["sci6"] == 5e-8


def test_imaginary_numbers():
    """Test imaginary number parsing"""
    ns = run("""
img1 = 3j
img2 = 4.5j
img3 = 1e2j
""")
    assert ns["img1"] == 3j
    assert ns["img2"] == 4.5j
    assert ns["img3"] == 100j


def test_complex_arithmetic():
    """Test complex number operations"""
    ns = run("""
z1 = 3 + 4j
z2 = 1 - 2j
sum_z = z1 + z2
abs_z1 = abs(z1)
""")
    assert ns["z1"] == (3+4j)
    assert ns["z2"] == (1-2j)
    assert ns["sum_z"] == (4+2j)
    assert ns["abs_z1"] == 5.0


def test_number_separators():
    """Test underscore separators in numbers"""
    ns = run("""
big1 = 1_000_000
big2 = 3.141_592
""")
    assert ns["big1"] == 1000000
    assert abs(ns["big2"] - 3.141592) < 1e-10


def test_mixed_number_operations():
    """Test operations with different number bases"""
    ns = run("""
result1 = 0xFF + 0b1111
result2 = 0o10 * 0x10
result3 = 0b1000 / 0o10
""")
    assert ns["result1"] == 270  # 255 + 15
    assert ns["result2"] == 128  # 8 * 16
    assert ns["result3"] == 1.0  # 8 / 8


def test_hex_in_expressions():
    """Test hexadecimal numbers in expressions"""
    ns = run("""
mask = 0xFF
shifted = mask << 8
combined = 0xAB | 0x12
""")
    assert ns["mask"] == 255
    assert ns["shifted"] == 65280
    assert ns["combined"] == 0xBB


def test_binary_operations():
    """Test binary numbers in bitwise operations"""
    ns = run("""
bits = 0b1010
inverted = ~bits
and_op = 0b1100 & 0b1010
or_op = 0b1010 | 0b0101
xor_op = 0b1100 ^ 0b1010
""")
    assert ns["bits"] == 10
    assert ns["inverted"] == ~10
    assert ns["and_op"] == 0b1000
    assert ns["or_op"] == 0b1111
    assert ns["xor_op"] == 0b0110


def test_octal_compatibility():
    """Test octal numbers work in standard contexts"""
    ns = run("""
perms = 0o755
check = perms > 0o600
""")
    assert ns["perms"] == 493
    assert ns["check"] is True


def test_bitwise_not_operator():
    """Test bitwise NOT operator (~)"""
    ns = run("""
val1 = ~10
val2 = ~0xFF
val3 = ~0b1010
negative = ~(-1)
""")
    assert ns["val1"] == ~10
    assert ns["val2"] == ~0xFF
    assert ns["val3"] == ~0b1010
    assert ns["negative"] == ~(-1)


def test_scientific_notation_in_expressions():
    """Test scientific notation in calculations"""
    ns = run("""
large = 1e6 + 2e6
small = 1e-3 * 2e-3
mixed = 1.5e2 / 3e1
""")
    assert ns["large"] == 3000000.0
    assert abs(ns["small"] - 2e-6) < 1e-15
    assert ns["mixed"] == 5.0
