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


def test_number_precision():
    """Test .2f, .1f, etc."""
    ns = run("""
price = 95
result = f"The price is {price:.2f} dollars"
""")
    assert ns["result"] == "The price is 95.00 dollars"


def test_number_precision_float():
    """Test formatting floats with precision"""
    ns = run("""
pi = 3.14159
result = f"Pi is approximately {pi:.2f}"
""")
    assert ns["result"] == "Pi is approximately 3.14"


def test_percentage():
    """Test .1% formatting"""
    ns = run("""
value = 0.85
result = f"Success rate: {value:.1%}"
""")
    assert ns["result"] == "Success rate: 85.0%"


def test_alignment_left():
    """Test <5 left alignment"""
    ns = run("""
num = 42
result = f"{num:<5}end"
""")
    assert ns["result"] == "42   end"


def test_alignment_right():
    """Test >5 right alignment"""
    ns = run("""
num = 42
result = f"{num:>5}end"
""")
    assert ns["result"] == "   42end"


def test_alignment_center():
    """Test ^5 center alignment"""
    ns = run("""
num = 42
result = f"{num:^5}end"
""")
    assert ns["result"] == " 42  end"


def test_thousands_separator():
    """Test , for thousands separator"""
    ns = run("""
price = 1234567.89
result = f"${price:,.2f}"
""")
    assert ns["result"] == "$1,234,567.89"


def test_binary_format():
    """Test :b for binary"""
    ns = run("""
num = 10
result = f"{num:b}"
""")
    assert ns["result"] == "1010"


def test_hex_format():
    """Test :x for hex"""
    ns = run("""
num = 255
result = f"{num:x}"
""")
    assert ns["result"] == "ff"


def test_hex_format_upper():
    """Test :X for uppercase hex"""
    ns = run("""
num = 255
result = f"{num:X}"
""")
    assert ns["result"] == "FF"


def test_scientific_notation():
    """Test :e for scientific notation"""
    ns = run("""
num = 12345.6789
result = f"{num:e}"
""")
    assert "1.234568e+04" in ns["result"]


def test_escape_braces():
    """Test {{ and }} for literal braces"""
    ns = run("""
x = 5
result = f"{{x}} = {x}"
""")
    assert ns["result"] == "{x} = 5"


def test_escape_braces_multiple():
    """Test multiple escaped braces"""
    ns = run("""
result = f"{{ and }}"
""")
    assert ns["result"] == "{ and }"


def test_debug_format():
    """Test x= debug format"""
    ns = run("""
x = 42
result = f"{x=}"
""")
    assert ns["result"] == "x=42"


def test_debug_format_with_spec():
    """Test x= with format specifier"""
    ns = run("""
pi = 3.14159
result = f"{pi=:.2f}"
""")
    assert ns["result"] == "pi=3.14"


def test_expression_in_fstring():
    """Test evaluating expressions"""
    ns = run("""
age = 20
result = f"Status: {'Adult' if age >= 18 else 'Minor'}"
""")
    assert ns["result"] == "Status: Adult"


def test_expression_in_fstring_minor():
    """Test expression evaluating to false branch"""
    ns = run("""
age = 15
result = f"Status: {'Adult' if age >= 18 else 'Minor'}"
""")
    assert ns["result"] == "Status: Minor"


def test_method_call_in_fstring():
    """Test calling methods like .upper()"""
    ns = run("""
text = "hello"
result = f"{text.upper()}"
""")
    assert ns["result"] == "HELLO"


def test_dict_access():
    """Test dictionary access user['name']"""
    ns = run("""
user = {"name": "Alice", "age": 30}
result = f"Name: {user['name']}"
""")
    assert ns["result"] == "Name: Alice"


def test_list_access():
    """Test list access"""
    ns = run("""
items = ["apple", "banana", "cherry"]
result = f"First: {items[0]}"
""")
    assert ns["result"] == "First: apple"


def test_alignment_with_strings():
    """Test alignment with strings"""
    ns = run("""
item = "apple"
result = f"{item:<10}|"
""")
    assert ns["result"] == "apple     |"


def test_combined_formatting():
    """Test combining alignment and precision"""
    ns = run("""
price = 42.5
result = f"${price:>8.2f}"
""")
    assert ns["result"] == "$   42.50"


def test_zero_padding():
    """Test zero padding with numbers"""
    ns = run("""
num = 42
result = f"{num:05}"
""")
    assert ns["result"] == "00042"


def test_multiple_format_specs():
    """Test multiple formatted values in one string"""
    ns = run("""
name = "Alice"
age = 30
score = 0.95
result = f"{name:<10} | Age: {age:>3} | Score: {score:.1%}"
""")
    assert ns["result"] == "Alice      | Age:  30 | Score: 95.0%"


def test_nested_expressions():
    """Test complex nested expressions"""
    ns = run("""
x = 5
y = 10
result = f"Sum: {x + y}, Product: {x * y}"
""")
    assert ns["result"] == "Sum: 15, Product: 50"


def test_fstring_with_computation():
    """Test mathematical expressions"""
    ns = run("""
a = 10
b = 3
result = f"{a} / {b} = {a / b:.2f}"
""")
    assert ns["result"] == "10 / 3 = 3.33"


def test_template_string_formatting():
    """Test that t-strings also support formatting"""
    ns = run("""
price = 19.99
result = t"Price: ${price:.2f}"
""")
    assert ns["result"] == "Price: $19.99"


def test_template_string_debug():
    """Test debug format in t-strings"""
    ns = run("""
x = 100
result = t"{x=}"
""")
    assert ns["result"] == "x=100"


def test_attribute_access():
    """Test accessing attributes"""
    ns = run("""
class Point {
    fn __init__(self, x, y) {
        self.x = x
        self.y = y
    }
}
p = Point(10, 20)
result = f"Point: ({p.x}, {p.y})"
""")
    assert ns["result"] == "Point: (10, 20)"


def test_string_width():
    """Test minimum width specification"""
    ns = run("""
num = 7
result = f"{num:3}"
""")
    assert ns["result"] == "  7"


def test_sign_format():
    """Test + for showing sign"""
    ns = run("""
pos = 42
neg = -42
result = f"{pos:+} and {neg:+}"
""")
    assert ns["result"] == "+42 and -42"
