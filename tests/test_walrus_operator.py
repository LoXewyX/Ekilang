"""Tests for walrus operator (:=) in Ekilang."""

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


def test_walrus_basic_if():
    """Test basic walrus operator in if statement"""
    ns = run("""
if (n := 10) > 5 {
    result = n
}
""")
    assert ns["n"] == 10
    assert ns["result"] == 10


def test_walrus_comparison_chain():
    """Test walrus in comparison chain"""
    ns = run("""
passed = false
if (x := 15) > 10 and x < 20 {
    passed = true
}
""")
    assert ns["x"] == 15
    assert ns["passed"] is True


def test_walrus_while_loop():
    """Test walrus operator in while loop"""
    ns = run("""
count = 0
iterations = []
while (count := count + 1) <= 3 {
    iterations.append(count)
}
""")
    assert ns["count"] == 4
    assert ns["iterations"] == [1, 2, 3]


def test_walrus_with_function_call():
    """Test walrus with function calls"""
    ns = run("""
fn get_value() {
    return 42
}
if (result := get_value()) == 42 {
    success = true
}
""")
    assert ns["result"] == 42
    assert ns["success"] is True


def test_walrus_multiple_in_condition():
    """Test multiple walrus operators in one condition"""
    ns = run("""
if (a := 5) > 0 and (b := 10) > 0 {
    total = a + b
}
""")
    assert ns["a"] == 5
    assert ns["b"] == 10
    assert ns["total"] == 15


def test_walrus_nested_conditions():
    """Test walrus in nested conditions"""
    ns = run("""
inner_val = 0
if (outer := 20) > 10 {
    if (inner := outer * 2) > 30 {
        inner_val = inner
    }
}
""")
    assert ns["outer"] == 20
    assert ns["inner"] == 40
    assert ns["inner_val"] == 40


def test_walrus_with_list_operations():
    """Test walrus with list operations"""
    ns = run("""
numbers = [1, 2, 3, 4, 5]
if (length := len(numbers)) > 3 {
    result = length
}
""")
    assert ns["length"] == 5
    assert ns["result"] == 5


def test_walrus_with_string_operations():
    """Test walrus with string methods"""
    ns = run("""
text = "hello"
if (upper := text.upper()) == "HELLO" {
    success = true
}
""")
    assert ns["upper"] == "HELLO"
    assert ns["success"] is True


def test_walrus_with_dict_get():
    """Test walrus with dictionary get method"""
    ns = run("""
data = {"key": "value"}
found = false
if (val := data.get("key")) is not none {
    found = true
}
""")
    assert ns["val"] == "value"
    assert ns["found"] is True


def test_walrus_with_arithmetic():
    """Test walrus with arithmetic expression"""
    ns = run("""
if (calc := 5 * 8 + 2) > 40 {
    result = calc
}
""")
    assert ns["calc"] == 42
    assert ns["result"] == 42


def test_walrus_reassignment():
    """Test walrus reassigning existing variable"""
    ns = run("""
temp = 5
if (temp := temp * 2) > 8 {
    result = temp
}
""")
    assert ns["temp"] == 10
    assert ns["result"] == 10


def test_walrus_in_list_comprehension():
    """Test walrus in list comprehension"""
    ns = run("""
data = [1, 2, 3, 4, 5]
squared = [(x, y) for x in data if (y := x * x) > 10]
""")
    assert ns["squared"] == [(4, 16), (5, 25)]


def test_walrus_with_boolean_logic():
    """Test walrus with boolean operations"""
    ns = run("""
if (flag := true) and (value := 100) > 50 {
    result = value
}
""")
    assert ns["flag"] is True
    assert ns["value"] == 100
    assert ns["result"] == 100


def test_walrus_in_ternary():
    """Test walrus in ternary expression"""
    ns = run("""
status = "big" if (size := 100) > 50 else "small"
""")
    assert ns["size"] == 100
    assert ns["status"] == "big"


def test_walrus_with_method_chaining():
    """Test walrus capturing method results"""
    ns = run("""
text = "  hello  "
if (cleaned := text.strip()) != "" {
    result = cleaned
}
""")
    assert ns["cleaned"] == "hello"
    assert ns["result"] == "hello"


def test_walrus_with_power():
    """Test walrus with exponentiation"""
    ns = run("""
base = 3
if (power := base ** 3) > 25 {
    result = power
}
""")
    assert ns["power"] == 27
    assert ns["result"] == 27


def test_walrus_scope_persistence():
    """Test that walrus variable persists outside condition"""
    ns = run("""
if (scope_test := 999) > 0 {
    inside = scope_test
}
outside = scope_test
""")
    assert ns["inside"] == 999
    assert ns["outside"] == 999
    assert ns["scope_test"] == 999


def test_walrus_with_hex_numbers():
    """Test walrus with hexadecimal numbers"""
    ns = run("""
if (hex_val := 0xFF) == 255 {
    result = hex_val
}
""")
    assert ns["hex_val"] == 255
    assert ns["result"] == 255


def test_walrus_with_binary_numbers():
    """Test walrus with binary numbers"""
    ns = run("""
if (bin_val := 0b1010) == 10 {
    result = bin_val
}
""")
    assert ns["bin_val"] == 10
    assert ns["result"] == 10


def test_walrus_with_complex_numbers():
    """Test walrus with complex numbers"""
    ns = run("""
if (z := 3 + 4j) {
    magnitude = abs(z)
}
""")
    assert ns["z"] == (3+4j)
    assert ns["magnitude"] == 5.0
