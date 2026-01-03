"""Test lambda currying and higher-order functions"""

from ekilang.lexer import Lexer
from ekilang.parser import Parser
from ekilang.executor import execute


def test_simple_curry():
    """Test basic currying: add = (a) => (b) => a + b"""
    src = '''
add = (a) => (b) => a + b
result = add(3)(2)
'''
    tokens = Lexer(src).tokenize()
    mod = Parser(tokens).parse()
    ns = execute(mod)
    assert ns["result"] == 5


def test_triple_curry():
    """Test triple currying: sum three numbers"""
    src = '''
sum3 = (a) => (b) => (c) => a + b + c
result = sum3(1)(2)(3)
'''
    tokens = Lexer(src).tokenize()
    mod = Parser(tokens).parse()
    ns = execute(mod)
    assert ns["result"] == 6


def test_curry_with_multiplication():
    """Test currying with multiplication"""
    src = '''
multiply = (a) => (b) => a * b
result = multiply(4)(5)
'''
    tokens = Lexer(src).tokenize()
    mod = Parser(tokens).parse()
    ns = execute(mod)
    assert ns["result"] == 20


def test_curry_partial_application():
    """Test partial application of curried function"""
    src = '''
add = (a) => (b) => a + b
add_five = add(5)
result = add_five(10)
'''
    tokens = Lexer(src).tokenize()
    mod = Parser(tokens).parse()
    ns = execute(mod)
    assert ns["result"] == 15


def test_curry_with_closures():
    """Test curried function with closure over outer variable"""
    src = '''
fn make_multiplier(factor) {
    return (x) => factor * x
}
double = make_multiplier(2)
triple = make_multiplier(3)
result1 = double(5)
result2 = triple(5)
'''
    tokens = Lexer(src).tokenize()
    mod = Parser(tokens).parse()
    ns = execute(mod)
    assert ns["result1"] == 10
    assert ns["result2"] == 15


def test_curry_composition():
    """Test composing curried functions"""
    src = '''
add = (a) => (b) => a + b
multiply = (a) => (b) => a * b
result1 = add(2)(3)
result2 = multiply(add(2)(3))(2)
'''
    tokens = Lexer(src).tokenize()
    mod = Parser(tokens).parse()
    ns = execute(mod)
    assert ns["result1"] == 5
    assert ns["result2"] == 10  # (2+3)*2 = 10


def test_lambda_returning_function():
    """Test lambda that returns a function"""
    src = '''
make_adder = (n) => (x) => n + x
add_ten = make_adder(10)
result = add_ten(5)
'''
    tokens = Lexer(src).tokenize()
    mod = Parser(tokens).parse()
    ns = execute(mod)
    assert ns["result"] == 15


def test_higher_order_function():
    """Test higher-order function with lambda"""
    src = '''
apply_twice = (f) => (x) => f(f(x))
increment = (n) => n + 1
result = apply_twice(increment)(10)
'''
    tokens = Lexer(src).tokenize()
    mod = Parser(tokens).parse()
    ns = execute(mod)
    assert ns["result"] == 12  # increment(increment(10)) = 12


def test_lambda_in_list():
    """Test storing curried lambdas in a list"""
    src = '''
operations = [(a) => (b) => a + b, (a) => (b) => a * b, (a) => (b) => a - b]
result1 = operations[0](3)(2)
result2 = operations[1](3)(2)
result3 = operations[2](3)(2)
'''
    tokens = Lexer(src).tokenize()
    mod = Parser(tokens).parse()
    ns = execute(mod)
    assert ns["result1"] == 5  # 3 + 2
    assert ns["result2"] == 6  # 3 * 2
    assert ns["result3"] == 1  # 3 - 2


def test_curried_reduce_pattern():
    """Test currying with apply pattern"""
    src = '''
apply_twice = (f) => (x) => f(f(x))
add_one = (n) => n + 1
result = apply_twice(add_one)(10)
'''
    tokens = Lexer(src).tokenize()
    mod = Parser(tokens).parse()
    ns = execute(mod)
    assert ns["result"] == 12


def test_curried_power_function():
    """Test currying with composition"""
    src = '''
compose = (f) => (g) => (x) => f(g(x))
double = (x) => x * 2
increment = (x) => x + 1
result = compose(double)(increment)(5)
'''
    tokens = Lexer(src).tokenize()
    mod = Parser(tokens).parse()
    ns = execute(mod)
    assert ns["result"] == 12     # (5+1)*2 = 12


def test_curry_with_string_operations():
    """Test currying with string operations"""
    src = '''
concat = (a) => (b) => f"{a}{b}"
greet = concat("Hello, ")
result = greet("World")
'''
    tokens = Lexer(src).tokenize()
    mod = Parser(tokens).parse()
    ns = execute(mod)
    assert ns["result"] == "Hello, World"


def test_nested_curry():
    """Test deeply nested currying"""
    src = '''
f = (a) => (b) => (c) => (d) => a + b + c + d
result = f(1)(2)(3)(4)
'''
    tokens = Lexer(src).tokenize()
    mod = Parser(tokens).parse()
    ns = execute(mod)
    assert ns["result"] == 10


def test_curry_with_conditional():
    """Test currying with conditionals"""
    src = '''
max_curry = (a) => (b) => a if a > b else b
result1 = max_curry(5)(3)
result2 = max_curry(2)(8)
'''
    tokens = Lexer(src).tokenize()
    mod = Parser(tokens).parse()
    ns = execute(mod)
    assert ns["result1"] == 5
    assert ns["result2"] == 8


def test_curry_with_list_operations():
    """Test currying with simple operations"""
    src = '''
make_multiplier = (factor) => (x) => factor * x
double = make_multiplier(2)
result = double(5)
'''
    tokens = Lexer(src).tokenize()
    mod = Parser(tokens).parse()
    ns = execute(mod)
    assert ns["result"] == 10
