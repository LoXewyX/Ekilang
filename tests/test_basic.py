"""Basic tests for Ekilang features."""

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


def test_semicolon_separator():
    """Test semicolon as statement separator"""
    ns = run(
        """
x = 1; y = 2; z = x + y
"""
    )
    assert ns["z"] == 3


def test_implicit_return():
    """Test implicit return of last expression in function"""
    ns = run(
        """
fn add(a, b) {
  a + b
}
r = add(2, 3)
print(r)
"""
    )
    assert ns["r"] == 5


def test_assign_augassign_lists_maps_lambda():
    """Test assignment, augmented assignment, lists, maps, and lambda functions"""
    ns = run(
        """
x = 1
x += 2
arr = [x, 4, 5]
m = {"a": x, "b": 10}
f = (a, b) => a + b
y = f(3, 4)
print(len(arr))
print(m["a"])
"""
    )
    assert ns["x"] == 3
    assert ns["y"] == 7


def test_literals_for_loop_and_logic():
    """Test literals, for loop, and logical operators"""
    ns = run(
        """
total = 0
data = [1, 2, 3]
for v in data {
    if v > 1 {
        total += v
    }
}
flag = true and not false
nonev = none
print(total)
print(flag)
print(nonev)
"""
    )
    assert ns["total"] == 5
    assert ns["flag"] is True
    assert ns["nonev"] is None


def test_block_lambda_basic():
    """Test basic block-style lambda"""
    ns = run(
        """
f = (a, b) => {
  a + b
}
r = f(2, 3)
"""
    )
    assert ns["r"] == 5


def test_block_lambda_multiple_statements():
    """Test block lambda with multiple statements"""
    ns = run(
        """
f = (x) => {
  doubled = x * 2
  doubled + 1
}
r = f(5)
"""
    )
    assert ns["r"] == 11


def test_block_lambda_with_control_flow():
    """Test block lambda with if/else statements"""
    ns = run(
        """
max_func = (a, b) => {
  if a > b {
    a
  } else {
    b
  }
}
r1 = max_func(3, 7)
r2 = max_func(10, 2)
"""
    )
    assert ns["r1"] == 7
    assert ns["r2"] == 10


def test_block_lambda_with_loop():
    """Test block lambda with loop"""
    ns = run(
        """
sum_func = (arr) => {
  total = 0
  for v in arr {
    total += v
  }
  total
}
r = sum_func([1, 2, 3, 4])
"""
    )
    assert ns["r"] == 10


def test_block_lambda_with_early_return():
    """Test block lambda with early return"""
    ns = run(
        """
find_func = (arr, target) => {
  for v in arr {
    if v == target {
      return true
    }
  }
  false
}
r1 = find_func([1, 2, 3], 2)
r2 = find_func([1, 2, 3], 5)
"""
    )
    assert ns["r1"] is True
    assert ns["r2"] is False


def test_nested_block_lambdas():
    """Test block lambda returning a value"""
    ns = run(
        """
f = (x) => {
  y = x + 1
  y * 2
}
r = f(5)
"""
    )
    assert ns["r"] == 12


def test_block_lambda_in_list():
    """Test block lambda stored in list and called"""
    ns = run(
        """
funcs = [
  (x) => { x + 1 },
  (x) => { x * 2 },
  (x) => { x * x }
]
r1 = funcs[0](5)
r2 = funcs[1](5)
r3 = funcs[2](5)
"""
    )
    assert ns["r1"] == 6
    assert ns["r2"] == 10
    assert ns["r3"] == 25


def test_block_lambda_higher_order():
    """Test block lambda as higher-order function"""
    ns = run(
        """
apply_twice = (f, x) => {
  f(f(x))
}
inc = (n) => { n + 1 }
r = apply_twice(inc, 5)
"""
    )
    assert ns["r"] == 7


def test_block_lambda_with_multiple_params():
    """Test block lambda with multiple parameters and complex logic"""
    ns = run(
        """
calculate = (a, b, op) => {
  if op == "add" {
    a + b
  } else {
    if op == "mul" {
      a * b
    } else {
      a - b
    }
  }
}
r1 = calculate(5, 3, "add")
r2 = calculate(5, 3, "mul")
r3 = calculate(5, 3, "sub")
"""
    )
    assert ns["r1"] == 8
    assert ns["r2"] == 15
    assert ns["r3"] == 2


def test_f_strings():
    """Test f-string interpolation"""
    ns = run(
        """
x = 5
y = 10
msg = f"x is {x} and y is {y}"
result = f"sum: {x + y}"
"""
    )
    assert ns["msg"] == "x is 5 and y is 10"
    assert ns["result"] == "sum: 15"


def test_f_strings_with_expressions():
    """Test f-strings with complex expressions"""
    ns = run(
        """
name = "Alice"
age = 30
msg = f"{name} is {age + 1} years old next year"
"""
    )
    assert ns["msg"] == "Alice is 31 years old next year"


def test_ternary_if_else():
    """Test Python-style ternary if-else operator"""
    ns = run(
        """
x = 5
y = 10
max_val = x if x > y else y
min_val = x if x < y else y
msg = "big" if x > y else "small"
"""
    )
    assert ns["max_val"] == 10
    assert ns["min_val"] == 5
    assert ns["msg"] == "small"


def test_ternary_nested():
    """Test nested ternary operators"""
    ns = run(
        """
x = 5
result = "negative" if x < 0 else "zero" if x == 0 else "positive"
"""
    )
    assert ns["result"] == "positive"


def test_range_exclusive():
    """Test Rust-style range syntax (exclusive)"""
    ns = run(
        """
r = 0 .. 5
arr = []
for i in r {
  arr = arr + [i]
}
"""
    )
    assert ns["arr"] == [0, 1, 2, 3, 4]


def test_range_inclusive():
    """Test inclusive range syntax"""
    ns = run(
        """
r = 1 ..= 5
arr = []
for i in r {
  arr = arr + [i]
}
"""
    )
    assert ns["arr"] == [1, 2, 3, 4, 5]


def test_list_comprehension_simple():
    """Test simple list comprehension"""
    ns = run(
        """
squares = [x * x for x in [1, 2, 3, 4, 5]]
"""
    )
    assert ns["squares"] == [1, 4, 9, 16, 25]


def test_list_comprehension_with_condition():
    """Test list comprehension with if condition"""
    ns = run(
        """
evens = [x for x in [1, 2, 3, 4, 5, 6] if x % 2 == 0]
"""
    )
    assert ns["evens"] == [2, 4, 6]


def test_list_comprehension_complex():
    """Test list comprehension with complex expression"""
    ns = run(
        """
data = [1, 2, 3, 4, 5]
transformed = [x * 2 + 1 for x in data if x > 2]
"""
    )
    assert ns["transformed"] == [7, 9, 11]


def test_combined_features():
    """Test combining multiple new features"""
    ns = run(
        """
nums = [1, 2, 3, 4, 5]
filtered = [x * 2 for x in nums if x > 2]
msg = f"Filtered and doubled: {filtered}"
last_val = filtered[2] if 2 < 3 else 0
"""
    )
    assert ns["filtered"] == [6, 8, 10]
    assert "Filtered and doubled:" in ns["msg"]
    assert ns["last_val"] == 10


def test_compound_assignment_numeric():
    """Test compound assignment operators with numbers"""
    ns = run(
        """
x = 10
x += 5
y = 20
y -= 3
z = 4
z *= 3
w = 20
w /= 4
v = 17
v %= 5
"""
    )
    assert ns["x"] == 15
    assert ns["y"] == 17
    assert ns["z"] == 12
    assert ns["w"] == 5
    assert ns["v"] == 2


def test_string_concatenation():
    """Test string concatenation with + operator"""
    ns = run(
        """
s1 = "Hello"
s2 = " "
s3 = "World"
result = s1 + s2 + s3
"""
    )
    assert ns["result"] == "Hello World"


def test_string_concatenation_compound():
    """Test string concatenation with += operator"""
    ns = run(
        """
msg = "Hello"
msg += " "
msg += "World"
"""
    )
    assert ns["msg"] == "Hello World"


def test_string_repetition():
    """Test string repetition with * operator"""
    ns = run(
        """
s = "ab"
result = s * 3
result2 = 2 * "x"
"""
    )
    assert ns["result"] == "ababab"
    assert ns["result2"] == "xx"


def test_string_repetition_compound():
    """Test string repetition with *= operator"""
    ns = run(
        """
s = "hi"
s *= 4
"""
    )
    assert ns["s"] == "hihihihi"


def test_list_concatenation():
    """Test list concatenation with + operator"""
    ns = run(
        """
lst1 = [1, 2]
lst2 = [3, 4]
result = lst1 + lst2
"""
    )
    assert ns["result"] == [1, 2, 3, 4]


def test_list_concatenation_compound():
    """Test list concatenation with += operator"""
    ns = run(
        """
lst = [1, 2]
lst += [3, 4]
lst += [5]
"""
    )
    assert ns["lst"] == [1, 2, 3, 4, 5]


def test_pow_assign():
    """Test the **= operator"""
    ns = run(
        """
x = 2
y = 3
x **= y
print(x)  # Should print 8
        """
    )
    assert ns["x"] == 8


def test_list_repetition():
    """Test list repetition with * operator"""
    ns = run(
        """
lst = [1, 2]
result = lst * 3
result2 = 2 * [0]
"""
    )
    assert ns["result"] == [1, 2, 1, 2, 1, 2]
    assert ns["result2"] == [0, 0]


def test_list_repetition_compound():
    """Test list repetition with *= operator"""
    ns = run(
        """
lst = [1]
lst *= 3
"""
    )
    assert ns["lst"] == [1, 1, 1]


def test_combined_string_operations():
    """Test combining string operations in complex expressions"""
    ns = run(
        """
prefix = "Value: "
value = 42
suffix = "!"
msg = prefix + str(value) + suffix
repeated = (msg + " ") * 2
"""
    )
    assert ns["msg"] == "Value: 42!"
    assert ns["repeated"] == "Value: 42! Value: 42! "


def test_bitwise_and_shift_operations():
    """Test bitwise operators and shifts"""
    ns = run(
        """
a = 6 | 3
b = 6 & 3
c = 6 ^ 3
d = 1 << 4
e = 32 >> 3
f = (5 ^ 1) << 1
        """
    )
    assert ns["a"] == 7
    assert ns["b"] == 2
    assert ns["c"] == 5
    assert ns["d"] == 16
    assert ns["e"] == 4
    assert ns["f"] == 8


def test_bitwise_augassign_operations():
    """Test compound assignment for bitwise and shifts"""
    ns = run(
        """
v = 5
v |= 2
v &= 6
v ^= 3
s = 8
s <<= 2
s >>= 3
        """
    )
    assert ns["v"] == 5
    assert ns["s"] == 4


def test_comparisons_membership_identity():
    """Test comparison, membership, and identity operators"""
    ns = run(
        """
x = 5
y = 3
lt = x < y
gt = x > y
lte = x <= 5
gte = y >= 3
lst = [1, 2, 3]
has_two = 2 in lst
missing = 5 not in lst
nonev = none
is_none = nonev is none
not_none = nonev is not true
same = lst is lst
        """
    )
    assert ns["lt"] is False
    assert ns["gt"] is True
    assert ns["lte"] is True
    assert ns["gte"] is True
    assert ns["has_two"] is True
    assert ns["missing"] is True
    assert ns["is_none"] is True
    assert ns["not_none"] is True
    assert ns["same"] is True


def test_async_function():
    """Test async function definition"""
    ns = run(
        """
async fn fetch_data() {
  42
}
coro = fetch_data()
        """
    )
    import asyncio

    result = asyncio.run(ns["coro"])
    assert result == 42


def test_await_expression():
    """Test await in async function"""
    ns = run(
        """
async fn compute() {
  100
}
async fn main() {
  val = await compute()
  val * 2
}
result = main()
        """
    )
    import asyncio

    result = asyncio.run(ns["result"])
    assert result == 200


def test_typed_function():
    """Test function with type annotations"""
    ns = run(
        """
fn add(a: int, b: int) -> int {
  a + b
}
result = add(5, 3)
        """
    )
    assert ns["result"] == 8


def test_typed_variable():
    """Test variable with type annotation"""
    ns = run(
        """
x: int = 42
msg: str = "hello"
        """
    )
    assert ns["x"] == 42
    assert ns["msg"] == "hello"


def test_typed_async_function():
    """Test async function with type annotations"""
    ns = run(
        """
async fn greet(name: str) -> str {
  "Hello, " + name
}
async fn main() {
  greeting = await greet("World")
  greeting
}
result = main()
        """
    )
    import asyncio

    result = asyncio.run(ns["result"])
    assert result == "Hello, World"


def test_tuple_literals():
    """Test tuple creation and access"""
    ns = run(
        """
empty_tuple = ()
single = (1,)
pair = (5, "hello")
triple = (1, 2, 3)
first = pair[0]
second = pair[1]
len_triple = len(triple)
        """
    )
    assert ns["empty_tuple"] == ()
    assert ns["single"] == (1,)
    assert ns["pair"] == (5, "hello")
    assert ns["triple"] == (1, 2, 3)
    assert ns["first"] == 5
    assert ns["second"] == "hello"
    assert ns["len_triple"] == 3


def test_set_literals():
    """Test set creation"""
    ns = run(
        """
empty_set = set()
s1 = {1, 2, 3}
s2 = {1, 1, 2, 2, 3}
has_two = 2 in s1
len_set = len(s1)
        """
    )
    assert ns["empty_set"] == set()
    assert ns["s1"] == {1, 2, 3}
    assert ns["s2"] == {1, 2, 3}
    assert ns["has_two"] is True
    assert ns["len_set"] == 3


def test_casting():
    """Test type casting with 'as' operator"""
    ns = run(
        """
x = 42
s = x as str
y = "123"
n = y as int
f = 3 as float
        """
    )
    assert ns["s"] == "42"
    assert ns["n"] == 123
    assert ns["f"] == 3.0


def test_dict_and_tuple_mixed():
    """Test dictionaries with tuple values"""
    ns = run(
        """
coords = {"a": (1, 2), "b": (3, 4)}
point_a = coords["a"]
x = point_a[0]
y = point_a[1]
        """
    )
    assert ns["coords"] == {"a": (1, 2), "b": (3, 4)}
    assert ns["point_a"] == (1, 2)
    assert ns["x"] == 1
    assert ns["y"] == 2


def test_use_import_module_and_item():
    """Test Rust-style use imports for modules and items"""
    ns = run(
        """
use math
use math::sqrt
a = math.sqrt(16)
b = sqrt(25)
        """
    )
    assert ns["a"] == 4.0
    assert ns["b"] == 5.0


def test_use_brace_and_alias():
    """Test grouped imports with aliasing"""
    ns = run(
        """
use math::{sqrt as s, pi}
val = s(pi)
        """
    )
    assert round(ns["val"], 5) == round(
        __import__("math").sqrt(__import__("math").pi), 5
    )


def test_if_elif_else():
    """Test if-elif-else parsing and execution."""
    ns = run(
        """
    x = 7
    state = none
    if x > 10 {
        state = false
    } elif x > 5 {
        state = true
    } else {
        state = false
    }
        """
    )
    assert ns["state"] is True


def test_match_statement():
    """Test match statement with patterns, guards, and wildcards."""
    ns = run(
        """
x = 5
result = "default"
match x {
    1 => {
        result = "one"
    }
    2 | 3 | 4 => {
        result = "small"
    }
    5 if x > 4 => {
        result = "five"
    }
    6 | 7 | 8 if x < 10 => {
        result = "six_to_eight"
    }
    _ if x > 10 => {
        result = "big"
    }
    _ => {
        result = "other"
    }
}
        """
    )
    assert ns["result"] == "five"

    # Test another value
    ns2 = run(
        """
x = 3
result = "default"
match x {
    1 => {
        result = "one"
    }
    2 | 3 | 4 => {
        result = "small"
    }
    _ => {
        result = "other"
    }
}
        """
    )
    assert ns2["result"] == "small"

    # Test wildcard
    ns3 = run(
        """
x = 100
result = "default"
match x {
    1 => {
        result = "one"
    }
    _ => {
        result = "other"
    }
}
        """
    )
    assert ns3["result"] == "other"
